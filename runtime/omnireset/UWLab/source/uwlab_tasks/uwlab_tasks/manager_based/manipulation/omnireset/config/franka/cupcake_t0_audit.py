"""Opt-in per-completed-T0 summaries. No actions/RNG changes, no video claim."""
import hashlib
import json
import os
from pathlib import Path
import torch
from isaaclab.utils.math import quat_apply,quat_apply_inverse
from ...mdp.events import MultiResetManager
from .captured_lift_tracker import CapturedLiftTracker


class CupCakeT0AuditedReset(MultiResetManager):
    def __call__(self,env,env_ids,dataset_dir,reset_types,probs,success=None,
                 state_indices=None,state_index_repeats=1,rigid_object_position_offsets=None):
        super().__call__(env,env_ids,dataset_dir,reset_types,probs,success,
                         state_indices,state_index_repeats,rigid_object_position_offsets)
        audit=getattr(env,'_cupcake_t0_audit',None)
        if audit is not None:
            ids=env.scene._ALL_INDICES if env_ids is None else env_ids
            audit.begin(ids,self)


class T0EpisodeAudit:
    def __init__(self,env,output):
        self.env=env
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.rank=os.environ.get('RANK','0')
        self.path=self.output/f't0_episodes_rank{self.rank}.jsonl'
        if self.path.exists():raise FileExistsError(self.path)
        self.robot=env.scene['robot']
        self.obj=env.scene['insertive_object']
        self.plate=env.scene['receptive_object']
        self.grip=env.action_manager.get_term('gripper')
        self.hand=self.robot.find_bodies('fr3_hand')[0][0]
        self.fingers=self.robot.find_joints(['fr3_finger_joint1','fr3_finger_joint2'],preserve_order=True)[0]
        self.offset=torch.tensor(self.grip.cfg.tcp_offset,device=env.device).view(1,3).expand(env.num_envs,-1)
        n=env.num_envs
        self.hold_release_margin_m=float(getattr(self.grip.cfg,'release_margin_m',0.))
        self.tracker=CapturedLiftTracker(n,env.device,20,self.hold_release_margin_m)
        self.active=torch.zeros(n,dtype=torch.bool,device=env.device)
        self.closed=torch.zeros_like(self.active)
        self.episode=torch.zeros(n,dtype=torch.long,device=env.device)
        self.steps=torch.zeros_like(self.episode)
        self.state=torch.full_like(self.episode,-1)
        self.initial=torch.zeros(n,35,device=env.device)
        self.initial_qvel=torch.zeros(n,9,device=env.device)
        self.max_lift=torch.zeros(n,device=env.device)
        self.source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.rows_written=0

    def begin(self,ids,manager):
        self.episode[ids]+=1
        self.steps[ids]=0
        # Resolve by reset type, never assume task index0 means T0 in a forced diagnostic.
        kinds=self.env.cfg.events.reset_from_reset_states.params['reset_types']
        t0=kinds.index('ObjectAnywhereEEAnywhere') if 'ObjectAnywhereEEAnywhere' in kinds else -1
        self.active[ids]=manager.task_id[ids]==t0
        self.state[ids]=manager.state_id[ids]
        self.closed[ids]=False
        self.max_lift[ids]=0
        self.tracker.reset(ids)
        origins=self.env.scene.env_origins[ids]
        obj=self.obj.data.root_state_w[ids].clone()
        plate=self.plate.data.root_state_w[ids].clone()
        obj[:,:3]-=origins
        plate[:,:3]-=origins
        self.initial[ids]=torch.cat([self.robot.data.joint_pos[ids],obj,plate],-1)
        self.initial_qvel[ids]=self.robot.data.joint_vel[ids]

    def step(self):
        env=self.env
        rot=self.robot.data.body_link_quat_w[:,self.hand]
        tcp=self.robot.data.body_link_pos_w[:,self.hand]+quat_apply(rot,self.offset)
        relative=quat_apply_inverse(rot,self.obj.data.root_pos_w-tcp)
        command=self.grip.processed_actions.sum(-1)
        measured=self.robot.data.joint_pos[:,self.fingers].sum(-1)
        self.tracker.step(self.obj.data.root_pos_w[:,2],tcp[:,2],command,measured,relative)
        self.closed|=command==0
        self.steps+=self.active.long()
        lift=self.obj.data.root_pos_w[:,2]-env.scene.env_origins[:,2]-self.initial[:,11]
        self.max_lift=torch.maximum(self.max_lift,lift)
        ids=torch.where(self.active&env.reset_buf.bool())[0]
        if not len(ids):return
        # Same success source as MultiResetManager, evaluated BEFORE automatic reset.
        success=env.reward_manager.get_term_cfg('progress_context').func.success
        integers=torch.stack([ids,self.episode[ids],self.state[ids],self.steps[ids],
                              success[ids].long(),self.closed[ids].long(),self.tracker.given[ids].long()],-1).cpu().tolist()
        initial=self.initial[ids].cpu().tolist()
        initial_qvel=self.initial_qvel[ids].cpu().tolist()
        final=torch.cat([self.robot.data.joint_pos[ids],relative[ids],measured[ids,None],self.max_lift[ids,None]],-1).cpu().tolist()
        lines=[]
        for ints,start,qvel,end in zip(integers,initial,initial_qvel,final):
            row=dict(version='cupcake_t0_episode_summary_v1',rank=self.rank,env_id=ints[0],episode_id=ints[1],
                     state_id=ints[2],steps=ints[3],task_success=bool(ints[4]),guard_ever_closed=bool(ints[5]),
                     acquisition_referenced_held_lift_2s=bool(ints[6]),initial_q=start[:9],initial_qvel=qvel,
                     initial_object_state_env=start[9:22],initial_plate_state_env=start[22:35],
                     final_q=end[:9],final_object_relative_tcp=end[9:12],final_measured_width=end[12],
                     max_reset_relative_lift_m=end[13],sim_step=int(env.common_step_counter),
                     seed=env.cfg.seed,source_sha256=self.source_sha,
                     scope='completed T0 summary; kinematic hold evidence, NOT contact certificate or full trajectory/video')
            lines.append(json.dumps(row,allow_nan=False)+'\n')
        with self.path.open('a') as stream:
            stream.writelines(lines)
            stream.flush()
        self.rows_written+=len(lines)
        self.active[ids]=False


def install_t0_audit(env,output):
    if hasattr(env,'_cupcake_t0_audit'):raise RuntimeError('T0 audit already installed')
    audit=T0EpisodeAudit(env,output)
    root=Path(env.cfg.events.reset_from_reset_states.params['dataset_dir'])
    manifest=root/'manifest.json'
    if not manifest.exists():manifest=root/'candidate_manifest.json'
    metadata=dict(version='cupcake_t0_episode_summary_v1',rank=audit.rank,
                  dataset_root=str(root.resolve()),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  seed=env.cfg.seed,control_dt_s=env.step_dt,source_sha256=audit.source_sha,
                  hold_release_margin_m=audit.hold_release_margin_m,
                  policy_identity='training weights change; correlate sim_step with run checkpoints, not exact saved per-episode policy',
                  coverage='all completed T0 episodes after installation; excludes unfinished episodes and pre-installation history',
                  stored='initial state, final summary, independent task/hold flags; no full action trajectory or video')
    with (audit.output/f't0_audit_metadata_rank{audit.rank}.json').open('x') as stream:
        json.dump(metadata,stream,indent=2)
    env._cupcake_t0_audit=audit
    compute=env.reward_manager.compute
    def compute_with_audit(dt):
        result=compute(dt)
        audit.step()
        return result
    env.reward_manager.compute=compute_with_audit
    return audit
