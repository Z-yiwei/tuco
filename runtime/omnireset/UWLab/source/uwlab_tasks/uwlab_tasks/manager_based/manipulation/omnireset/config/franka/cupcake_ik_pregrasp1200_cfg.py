"""T0-only replacement. RL starts at saved pregrasp endpoints, never executes IK."""
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import torch
from isaaclab.utils import configclass
from .cupcake_pregrasp_alignment_cfg import CupCakePregraspAlignmentTrainCfg
from .cupcake_pregrasp_alignment_math import AlignmentParameters
from .cupcake_pregrasp_alignment_reward import CupCakePregraspAlignmentProgress
from .cupcake_success400_phase_reset import CupCakeSuccess400PhaseReset
from .cupcake_policy_gripper_cfg import RESET_TYPES
from .cupcake_t0_audit import install_t0_audit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pregrasp1200_startup(env, env_ids):
    from .cupcake_friction_compensated_action import CupCakeFrictionCompensatedDiffIKAction
    from .cupcake_hysteretic_guard import CupCakeHystereticGuard
    from .cupcake_captured_lift_reward import CupCakeCapturedLiftBonus
    root = Path(os.environ['CUPCAKE_PREGRASP1200_DATASET_ROOT']).resolve()
    launch = json.loads(Path(os.environ['CUPCAKE_PREGRASP1200_LAUNCH_CONTRACT']).read_text())
    m = json.loads((root/'manifest.json').read_text())
    assert m['version'] == 'cupcake_pregrasp1200_fourpath_v1'
    assert launch['dataset_manifest_sha256'] == sha(root/'manifest.json')
    for path, digest in launch['code_sha256'].items(): assert sha(path) == digest, path
    for name, digest in m['reset_sha256'].items(): assert sha(root/'Resets/CupCakeHalf__Plate'/name) == digest
    assert m['counts'] == dict(zip(RESET_TYPES, [1200,13648,904,56224]))
    assert m['four_path_probabilities'] == [.25]*4 and m['ik_approach_executed_in_rl'] is False
    old = Path(m['parent_root'])
    assert sha(old/'manifest.json') == m['parent_manifest_sha256']
    for name, source in m['unchanged_t123'].items():
        assert sha(source['path']) == source['sha256'] == m['reset_sha256'][name]
    physical = json.loads((old/'training_contract.json').read_text())
    assert Path(os.environ['CUPCAKE_RESET_DATASET_ROOT']).resolve() == old
    assert sha(old/'training_contract.json') == m['parent_training_contract_sha256']
    reset = env.cfg.events.reset_from_reset_states.params
    assert Path(reset['dataset_dir']).resolve() == root and reset['reset_types'] == RESET_TYPES and reset['probs'] == [.25]*4
    manager = env.event_manager.get_term_cfg('reset_from_reset_states').func
    assert type(manager) is CupCakeSuccess400PhaseReset
    assert manager.num_states.tolist() == [1200,13648,904,56224]
    for w in manager.state_weights: assert torch.equal(w,torch.full_like(w,1./len(w)))
    assert env.action_manager.total_action_dim == 7
    assert env.observation_manager.group_obs_dim['policy'] == (200,) and env.observation_manager.group_obs_dim['critic'] == (168,)
    assert env.max_episode_length == 640 and math.isclose(env.step_dt,.1) and not env.curriculum_manager.active_terms
    assert tuple(env.scene['insertive_object'].cfg.spawn.scale) == (.5,)*3
    assert tuple(env.scene['receptive_object'].cfg.spawn.scale) == (2/3,)*3
    arm, grip = env.action_manager.get_term('arm'), env.action_manager.get_term('gripper')
    assert type(arm) is CupCakeFrictionCompensatedDiffIKAction and type(grip) is CupCakeHystereticGuard
    assert tuple(arm.cfg.max_joint_velocity) == (.2,)*7 and arm.cfg.ik_damping == .05
    assert arm.cfg.static_friction_compensation_gain == .9 and arm.cfg.moving_friction_compensation_gain == 1.
    assert arm.cfg.friction_after_capture_only and arm.cfg.friction_capture_ramp_s == .5 and not arm.cfg.friction_assist_only_toward_target
    assert tuple(grip.cfg.tcp_offset) == (0.,0.,.1034) and grip.cfg.lateral_thresh_xy == .02 and grip.cfg.vertical_thresh_z == .03
    assert grip.cfg.release_margin_m == .002 and grip.cfg.capture_confirm_steps == 5 and not grip._hold_processed_action
    robot = env.scene['robot']
    assert torch.all(robot.data.joint_stiffness[:,:7] == 80) and torch.all(robot.data.joint_damping[:,:7] == 4)
    env._cupcake_repair_expected_physics = {k:torch.tensor(v,device=env.device,dtype=getattr(robot.data,k).dtype)
                                         for k,v in physical['arm_physics_signature'].items()}
    env._cupcake_repair_physics_checks = 0
    env._augmented_reset_checks = 0
    reward = env.reward_manager.get_term_cfg('grasped_and_lifted')
    assert isinstance(reward.func,CupCakeCapturedLiftBonus) and reward.weight == 2 and reward.params == {'hold_steps':20,'release_margin_m':.002}
    alignment = env.reward_manager.get_term_cfg('pregrasp_alignment_progress')
    assert type(alignment.func) is CupCakePregraspAlignmentProgress and alignment.weight == 1.
    assert alignment.params == asdict(AlignmentParameters())
    floor = env.cfg.scene.fall_catcher
    assert tuple(floor.spawn.size) == (1000.,1000.,.5) and tuple(floor.init_state.pos) == (0.,0.,-1.118)
    command = env.command_manager.get_term('task_command')
    command.success_position_threshold = .02; command.success_orientation_threshold = math.radians(3)
    quota = Path(os.environ['CUPCAKE_SUCCESS400_ROOT'])
    assert sha(quota/'manifest.jsonl') == physical['completed_quota_manifest_sha256']
    # Same per-episode logger; new directory/metadata identify NEW pregrasp T0.
    # Historical state_id<2500/original-T0 tally must not be used for this run.
    install_t0_audit(env,Path(env.cfg.log_dir)/'pregrasp_t0_episode_audit')
    if os.environ.get('CUPCAKE_PREGRASP1200_NATIVE_AUDIT') == '1':
        audit_pregrasp_endpoints(env,manager)
    report = dict(version='cupcake_pregrasp1200_rl_v1',dataset_root=str(root),dataset_manifest_sha256=sha(root/'manifest.json'),
        counts=m['counts'],reset_probabilities=[.25]*4,t0_scope='all T0 are new IK-derived pregrasp endpoints; not original distant T0',
        t123_sha256_unchanged=True,ik_executed=False,code_sha256=launch['code_sha256'],
        rewards_controller_physics_config_unchanged=True,full_ik_trajectories=m['ik_trajectory_root'])
    output = Path(env.cfg.log_dir)/f'pregrasp1200_contract_rank{os.environ.get("RANK","0")}.json'
    with output.open('x') as stream: json.dump(report,stream,indent=2)
    print('[PREGRASP1200_RL_CONTRACT]',json.dumps(report),flush=True)


def audit_pregrasp_endpoints(env,manager):
    from ...mdp.events import sample_from_nested_dict
    from .cupcake_success400_native_audit import flatten, measured
    result = dict(scope='all1200 in real RL scene: direct native reset readback, initial guard-open, one physics step; not grasp SR',
                  count=0,max_readback_error=0.,max_one_step_object_motion_m=0.)
    batch = min(env.num_envs,256)
    for start in range(0,1200,batch):
        chosen = torch.arange(start,min(start+batch,1200),device=env.device)
        ids = torch.arange(len(chosen),device=env.device)
        data = sample_from_nested_dict(manager.datasets[0],chosen)['initial_state']
        expected = flatten(data)
        manager._reset_to(data,env_ids=ids,is_relative=True)
        env.sim.forward(); env.scene.update(env.physics_dt)
        actual = measured(env,ids)
        for k in (21,34,47):
            q, ref = actual[:,k:k+4],expected[:,k:k+4]; q[(q*ref).sum(-1)<0] *= -1
        error = float((actual-expected).abs().max())
        assert torch.isfinite(actual).all() and error < 1e-4
        assert (env.action_manager.get_term('gripper').compute_rule_actions()[ids,0] > 0).all()
        assert (actual[:,7:9].sum(-1) > .078).all()
        env.sim.step(render=False); env.scene.update(env.physics_dt)
        after = measured(env,ids)
        assert torch.isfinite(after).all()
        motion = float((after[:,31:34]-actual[:,31:34]).norm(dim=-1).max())
        assert motion < .01
        result['count'] += len(ids)
        result['max_readback_error'] = max(result['max_readback_error'],error)
        result['max_one_step_object_motion_m'] = max(result['max_one_step_object_motion_m'],motion)
    result['passed'] = True
    with (Path(env.cfg.log_dir)/'pregrasp1200_native_reset_audit.json').open('x') as stream: json.dump(result,stream,indent=2)
    print('[PREGRASP1200_NATIVE_AUDIT]',json.dumps(result),flush=True)


@configclass
class CupCakeIKPregrasp1200TrainCfg(CupCakePregraspAlignmentTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_reset_states.params['dataset_dir'] = os.environ['CUPCAKE_PREGRASP1200_DATASET_ROOT']
        self.events.policy_gripper_contract.func = verify_pregrasp1200_startup
