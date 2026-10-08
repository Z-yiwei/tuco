"""User-approved success400 phase replacement; canonical control/physics unchanged."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from ...mdp.events import MultiResetManager, sample_from_nested_dict
from .cupcake_t0_audit import install_t0_audit
from .trajectory_reset_weights import reset_weights
from . import cupcake_t0_trajectory as original


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CupCakeSuccess400PhaseReset(MultiResetManager):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        assert self.num_tasks==4 and self._forced_state_sequence is None
        self.state_weights=[torch.full((int(n),),1./int(n),dtype=torch.float64,device=env.device) for n in self.num_states]
        self.source_counts=torch.zeros(4,3,dtype=torch.long,device=env.device)

    def __call__(self,env,env_ids,dataset_dir,reset_types,probs,success=None,
                 state_indices=None,state_index_repeats=1,rigid_object_position_offsets=None):
        assert state_indices is None
        if env_ids is None:env_ids=env.scene._ALL_INDICES
        if success is not None:
            self.success_monitor.success_update(self.task_id[env_ids],eval(success)[env_ids].float())
            rates=self.success_monitor.get_success_rate();env.extras.setdefault('log',{})
            for i in range(4):
                env.extras['log'].update({f'Metrics/task_{i}_success_rate':rates[i].item(),
                    f'Metrics/task_{i}_prob':self.probs[i].item(),f'Metrics/task_{i}_normalized_prob':self.probs[i].item()})
            env.extras['log']['Metrics/mean_episode_length']=env.episode_length_buf[env_ids].float().mean().item()
        kinds=torch.multinomial(self.probs,len(env_ids),replacement=True);self.task_id[env_ids]=kinds
        for i in range(4):
            ids=env_ids[kinds==i]
            if not len(ids):continue
            chosen=torch.multinomial(self.state_weights[i],len(ids),replacement=True)
            self.state_id[ids]=chosen
            values=sample_from_nested_dict(self.datasets[i],chosen)
            self._reset_to(values['initial_state'],env_ids=ids,is_relative=True)
            bucket=torch.where(chosen>=2500,2,0) if i==0 else torch.full_like(chosen,2)
            self.source_counts[i]+=torch.bincount(bucket,minlength=3)
        robot=env.scene['robot'];robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]),env_ids=env_ids)
        audit=getattr(env,'_cupcake_t0_audit',None)
        if audit is not None:audit.begin(env_ids,self)


def verify_phase_startup(env,env_ids):
    from .cupcake_policy_gripper_cfg import RESET_TYPES
    from .cupcake_friction_compensated_action import CupCakeFrictionCompensatedDiffIKAction
    from .cupcake_hysteretic_guard import CupCakeHystereticGuard
    from .cupcake_captured_lift_reward import CupCakeCapturedLiftBonus
    root=Path(os.environ['CUPCAKE_RESET_DATASET_ROOT']).resolve()
    m=json.loads((root/'manifest.json').read_text());c=json.loads((root/'training_contract.json').read_text())
    launch=json.loads(Path(os.environ['CUPCAKE_PHASE_LAUNCH_CONTRACT']).read_text())
    assert launch['dataset_manifest_sha256']==sha(root/'manifest.json')
    for path,digest in launch['code_sha256'].items():assert sha(path)==digest,path
    assert m['version']=='cupcake_success400_phase_replacement_v1' and m['sealed_for_training']
    assert c['dataset_manifest_sha256']==sha(root/'manifest.json') and c['gates_passed']
    for p,digest in c['code_sha256'].items():assert sha(p)==digest,p
    for p,digest in c['evidence_sha256'].items():assert sha(p)==digest,p
    for name,digest in m['reset_sha256'].items():assert sha(root/'Resets/CupCakeHalf__Plate'/name)==digest
    assert sha(m['t0_invariant']['file'])==m['t0_invariant']['sha256']
    assert m['source_coverage']==400 and m['replacement_frames']==70776
    assert m['old_pregrasp_rows_in_new_t123']==m['old_t123_rows_in_new_t123']==0
    assert sha(root/'source_index.json')==m['source_index_sha256']
    assert m['four_path_probabilities']==[.25]*4 and m['pregrasp_fraction_per_path']==[0.,0.,0.,0.]
    reset=env.cfg.events.reset_from_reset_states.params
    assert Path(reset['dataset_dir']).resolve()==root and reset['reset_types']==RESET_TYPES and reset['probs']==[.25]*4
    manager=env.event_manager.get_term_cfg('reset_from_reset_states').func
    assert isinstance(manager,CupCakeSuccess400PhaseReset)
    assert manager.num_states.tolist()==[m['counts'][k] for k in RESET_TYPES]
    for i,w in enumerate(manager.state_weights):
        assert torch.all(w>0) and abs(float(w.sum())-1)<1e-8
        assert torch.equal(w,torch.full_like(w,1./len(w)))
        if i==0:assert torch.equal(w,reset_weights(9390,0,env.device))
    assert env.action_manager.total_action_dim==7 and env.observation_manager.group_obs_dim['policy']==(200,) and env.observation_manager.group_obs_dim['critic']==(168,)
    assert env.max_episode_length==640 and math.isclose(env.step_dt,.1) and not env.curriculum_manager.active_terms
    assert tuple(env.scene['insertive_object'].cfg.spawn.scale)==(.5,)*3 and tuple(env.scene['receptive_object'].cfg.spawn.scale)==(2/3,)*3
    arm=env.action_manager.get_term('arm');grip=env.action_manager.get_term('gripper')
    assert type(arm) is CupCakeFrictionCompensatedDiffIKAction and type(grip) is CupCakeHystereticGuard
    assert tuple(arm.cfg.max_joint_velocity)==(.2,)*7 and arm.cfg.ik_damping==.05
    assert arm.cfg.static_friction_compensation_gain==.9 and arm.cfg.moving_friction_compensation_gain==1.
    assert arm.cfg.friction_after_capture_only and arm.cfg.friction_capture_ramp_s==.5 and not arm.cfg.friction_assist_only_toward_target
    assert tuple(grip.cfg.tcp_offset)==(0.,0.,.1034) and grip.cfg.lateral_thresh_xy==.02 and grip.cfg.vertical_thresh_z==.03
    assert grip.cfg.release_margin_m==.002 and grip.cfg.capture_confirm_steps==5 and not grip._hold_processed_action
    robot=env.scene['robot'];assert torch.all(robot.data.joint_stiffness[:,:7]==80) and torch.all(robot.data.joint_damping[:,:7]==4)
    env._cupcake_repair_expected_physics={k:torch.tensor(v,device=env.device,dtype=getattr(robot.data,k).dtype) for k,v in c['arm_physics_signature'].items()}
    env._cupcake_repair_physics_checks=0
    reward=env.reward_manager.get_term_cfg('grasped_and_lifted');assert isinstance(reward.func,CupCakeCapturedLiftBonus) and reward.weight==2 and reward.params=={'hold_steps':20,'release_margin_m':.002}
    floor=env.cfg.scene.fall_catcher;assert tuple(floor.spawn.size)==(1000.,1000.,.5) and tuple(floor.init_state.pos)==(0.,0.,-1.118)
    command=env.command_manager.get_term('task_command');command.success_position_threshold=.02;command.success_orientation_threshold=math.radians(3)
    install_t0_audit(env,Path(env.cfg.log_dir)/'t0_episode_audit')
    quota=Path(os.environ['CUPCAKE_SUCCESS400_ROOT'])
    marker=json.loads((quota/'COMPLETE.json').read_text())
    assert marker['count']==400 and marker['manifest_sha256']==sha(quota/'manifest.jsonl')
    assert marker['manifest_sha256']==c['completed_quota_manifest_sha256']
    # Quota is full: no full-trajectory writer; per-T0 audit remains active.
    if os.environ.get('CUPCAKE_PHASE_FULL_RESET_AUDIT')=='1':
        from .cupcake_success400_native_audit import audit_all_resets
        audit_all_resets(env,manager,root)
    env._augmented_reset_checks=0
    result=dict(dataset_manifest_sha256=sha(root/'manifest.json'),counts=m['counts'],pregrasp_probabilities=[0.,0.,0.,0.],
        original_t0_metrics_boundary='state_id<2500 original; >=2500 trajectory warm-start; do not merge these SRs',
        completed_quota_root=str(quota),training_actions_physics_rewards_unchanged=True)
    (Path(env.cfg.log_dir)/f'phase_contract_rank{os.environ.get("RANK","0")}.json').write_text(json.dumps(result,indent=2))
    print('[SUCCESS400_PHASE_CONTRACT]',json.dumps(result),flush=True)


def verify_phase_after_reset(env,env_ids):
    from .cupcake_grasp_repair_cfg import verify_grasp_repair_physics_after_reset
    verify_grasp_repair_physics_after_reset(env,env_ids)
    ids=env.scene._ALL_INDICES if env_ids is None else env_ids
    q=env.scene['insertive_object'].data.root_quat_w[ids]
    assert torch.isfinite(q).all() and torch.all((q.norm(dim=-1)-1).abs()<1e-4)
    if env._augmented_reset_checks==0:
        mgr=env.event_manager.get_term_cfg('reset_from_reset_states').func
        print('[SUCCESS400_NATIVE_RESET]',len(ids),'source_counts',mgr.source_counts.tolist(),flush=True)
    env._augmented_reset_checks+=len(ids)
