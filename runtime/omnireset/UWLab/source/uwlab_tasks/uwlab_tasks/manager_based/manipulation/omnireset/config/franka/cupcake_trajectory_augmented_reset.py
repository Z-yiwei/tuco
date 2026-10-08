"""Versioned trajectory augmentation; only reset distribution/lifecycle changes."""
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


class CupCakeTrajectoryReset(MultiResetManager):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        assert self.num_tasks==4 and self._forced_state_sequence is None
        self.state_weights=[reset_weights(int(n),i,env.device) for i,n in enumerate(self.num_states)]
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
            boundary=2500 if i==0 else 5000
            bucket=torch.where(chosen>=boundary,2,torch.where((chosen>=2500)&(i>0),1,0))
            self.source_counts[i]+=torch.bincount(bucket,minlength=3)
        robot=env.scene['robot'];robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel[env_ids]),env_ids=env_ids)
        audit=getattr(env,'_cupcake_t0_audit',None)
        if audit is not None:audit.begin(env_ids,self)


class MixedHeldQuota(original.TrajectoryQuota):
    """Same total200 cap; old held records can seed this new union, unchanged."""
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.contract=dict(version='cupcake_mixed_pose_held_union_v1',max_trajectories=200,
            selection='mixed_pose_t0_held_lift_2s',recorder_sha256=sha(original.__file__),selector_sha256=sha(__file__))
        with (self.root/'quota.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);p=self.root/'recording_contract.json'
            if p.exists():assert json.loads(p.read_text())==self.contract
            else:
                with p.open('x') as f:json.dump(self.contract,f,indent=2)
    def save(self,payload):
        if not payload['metadata']['held_lift_2s']:return False
        return super().save(payload)


def verify_augmented_startup(env,env_ids):
    from .cupcake_policy_gripper_cfg import RESET_TYPES
    from .cupcake_friction_compensated_action import CupCakeFrictionCompensatedDiffIKAction
    from .cupcake_hysteretic_guard import CupCakeHystereticGuard
    from .cupcake_captured_lift_reward import CupCakeCapturedLiftBonus
    root=Path(os.environ['CUPCAKE_RESET_DATASET_ROOT']).resolve()
    m=json.loads((root/'manifest.json').read_text());c=json.loads((root/'training_contract.json').read_text())
    assert m['version']=='cupcake_stride3_appended_pregrasp20_v1' and m['sealed_for_training']
    assert c['dataset_manifest_sha256']==sha(root/'manifest.json') and c['gates_passed']
    for p,digest in c['code_sha256'].items():assert sha(p)==digest,p
    for p,digest in c['evidence_sha256'].items():assert sha(p)==digest,p
    for name,digest in m['reset_sha256'].items():assert sha(root/'Resets/CupCakeHalf__Plate'/name)==digest
    for src in m['preserved_sources']:assert sha(src['path'])==src['sha256']
    assert m['four_path_probabilities']==[.25]*4 and m['pregrasp_fraction_per_path']==[0.,.2,.2,.2]
    reset=env.cfg.events.reset_from_reset_states.params
    assert Path(reset['dataset_dir']).resolve()==root and reset['reset_types']==RESET_TYPES and reset['probs']==[.25]*4
    manager=env.event_manager.get_term_cfg('reset_from_reset_states').func
    assert isinstance(manager,CupCakeTrajectoryReset)
    assert manager.num_states.tolist()==[m['counts'][k] for k in RESET_TYPES]
    for i,w in enumerate(manager.state_weights):
        assert torch.all(w>0) and abs(float(w.sum())-1)<1e-8
        if i:assert abs(float(w[2500:5000].sum())-.2)<1e-8
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
    # Install unchanged recording hooks via the already-complete historical quota,
    # then point them at a separate versioned held-only union before any episode.
    original.install_t0_trajectories(env,Path(os.environ['CUPCAKE_T0_TRAJECTORY_ROOT']))
    recorder=env._cupcake_t0_trajectories;assert not recorder.pending and not recorder.buffers
    recorder.quota=MixedHeldQuota(os.environ['CUPCAKE_MIXED_HELD_ROOT'])
    recorder.metadata.update(selection='mixed_pose_t0_held_lift_2s',trajectory_root=str(recorder.quota.root.resolve()),
        original_t0_id_range=[0,2500],augmented_source_index=str(root/'source_index.json'),
        augmented_source_index_sha256=sha(root/'source_index.json'),reset_type='ObjectAnywhereEEAnywhere')
    env._augmented_reset_checks=0
    result=dict(dataset_manifest_sha256=sha(root/'manifest.json'),counts=m['counts'],pregrasp_probabilities=[0.,.2,.2,.2],
        original_t0_metrics_boundary='state_id<2500 original; >=2500 trajectory warm-start; do not merge these SRs',
        held_union_root=str(recorder.quota.root),training_actions_physics_rewards_unchanged=True)
    (Path(env.cfg.log_dir)/f'augmented_contract_rank{os.environ.get("RANK","0")}.json').write_text(json.dumps(result,indent=2))
    print('[AUGMENTED_CONTRACT]',json.dumps(result),flush=True)


def verify_augmented_after_reset(env,env_ids):
    from .cupcake_grasp_repair_cfg import verify_grasp_repair_physics_after_reset
    verify_grasp_repair_physics_after_reset(env,env_ids)
    ids=env.scene._ALL_INDICES if env_ids is None else env_ids
    q=env.scene['insertive_object'].data.root_quat_w[ids]
    assert torch.isfinite(q).all() and torch.all((q.norm(dim=-1)-1).abs()<1e-4)
    if env._augmented_reset_checks==0:
        mgr=env.event_manager.get_term_cfg('reset_from_reset_states').func
        print('[AUGMENTED_NATIVE_RESET]',len(ids),'source_counts',mgr.source_counts.tolist(),flush=True)
    env._augmented_reset_checks+=len(ids)
