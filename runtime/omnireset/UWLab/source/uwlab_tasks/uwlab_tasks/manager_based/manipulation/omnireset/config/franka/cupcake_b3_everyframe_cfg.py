"""One-for-one every-frame phase replacement with unchanged fixed B3 control.

RL still begins at the sealed pregrasp endpoints. Their old IK trajectories are
NOT demonstrations of this plant. A future IK collector must use this contract.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import yaml
from isaaclab.utils import configclass

from ...mdp.actions.actions_cfg import RelCartesianDiffIKJointPositionActionCfg
from ...mdp.actions.task_space_actions import RelCartesianDiffIKJointPositionAction
from .cupcake_ik_pregrasp1200_cfg import CupCakeIKPregrasp1200TrainCfg, audit_pregrasp_endpoints
from .cupcake_hysteretic_guard import CupCakeHystereticGuard
from .cupcake_policy_gripper_cfg import RESET_TYPES
from .cupcake_success400_phase_reset import CupCakeSuccess400PhaseReset
from .cupcake_t0_audit import install_t0_audit
from .cupcake_unique_t0_recording import UniqueT0Recorder, initial_matrix
from .rl_state_cfg import _B3_NOMINAL_FIXED_EVENTS

VERSION = 'cupcake_b3_unified_no_extra_effort_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(env, name, value):
    path = Path(env.cfg.log_dir) / f'{name}_rank{os.environ.get("RANK", "0")}.json'
    temp = path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def verify_b3_after_reset(env, env_ids):
    """Read back actual PhysX joint/material/mass data after B3 has been applied."""
    ids = env.scene._ALL_INDICES if env_ids is None else env_ids
    robot = env.scene['robot']
    for key, expected in env._unified_expected_joint.items():
        actual = getattr(robot.data, key)[ids, :7]
        assert torch.allclose(actual, expected.expand_as(actual), atol=1e-6, rtol=1e-6), key
    assert torch.all(robot.data.joint_stiffness[ids, :7] == 80)
    assert torch.all(robot.data.joint_damping[ids, :7] == 4)
    assert torch.all(robot.data.joint_stiffness[ids, 7:9] == 5000)
    assert torch.all(robot.data.joint_damping[ids, 7:9] == 50)
    q = env.scene['insertive_object'].data.root_quat_w[ids]
    assert torch.isfinite(q).all() and torch.all((q.norm(dim=-1) - 1).abs() < 1e-4)
    if not env._unified_reset_checks:
        scene = {}
        for asset_name in ('robot', 'insertive_object', 'receptive_object', 'table'):
            asset = env.scene[asset_name]
            view = asset.root_physx_view
            material = view.get_material_properties().cpu()
            expected = _B3_NOMINAL_FIXED_EVENTS[asset_name + '_material']
            triplet = torch.tensor([expected[k][0] for k in
                                    ('static_friction_range', 'dynamic_friction_range', 'restitution_range')])
            assert torch.allclose(material, triplet.expand_as(material), atol=1e-6), asset_name
            masses = view.get_masses().cpu()
            assert torch.isfinite(masses).all() and (masses > 0).all(), asset_name
            # Same asset in every env, and all scale/abs ranges are degenerate.
            assert torch.allclose(masses, masses[:1].expand_as(masses), atol=1e-6), asset_name
            if asset_name == 'insertive_object':
                assert torch.allclose(masses, torch.full_like(masses, .11), atol=1e-6)
            scene[asset_name] = dict(material=material[0].tolist(), masses_kg=masses[0].tolist())
        env._unified_scene_readback = scene
    env._unified_reset_checks += len(ids)
    if env._unified_reset_checks == len(ids):
        report = dict(passed=True, version=VERSION, reset_env_count=len(ids),
                      joint_readback={k: v.tolist() for k, v in env._unified_expected_joint.items()},
                      scene=env._unified_scene_readback, arm_pd=[80, 4], gripper_pd=[5000, 50],
                      scope='actual post-reset plant; no success-rate claim')
        write_report(env, 'unified_physics', report)
        print('[UNIFIED_B3_PHYSICS]', json.dumps(report), flush=True)


def verify_b3_unified_startup(env, env_ids):
    # Startup events run before the trainer creates its checkpoint directory.
    Path(env.cfg.log_dir).mkdir(parents=True, exist_ok=True)
    launch = json.loads(Path(os.environ['CUPCAKE_UNIFIED_CONTRACT']).read_text())
    assert launch['version'] == VERSION
    for path, digest in launch['code_sha256'].items():
        assert sha(path) == digest, path
    root = Path(os.environ['CUPCAKE_PREGRASP1200_DATASET_ROOT']).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    assert sha(root / 'manifest.json') == launch['dataset_manifest_sha256']
    for name, digest in manifest['reset_sha256'].items():
        assert sha(root / 'Resets/CupCakeHalf__Plate' / name) == digest, name
    assert manifest['counts'] == dict(zip(RESET_TYPES, [1200, 13648, 904, 56224]))
    assert manifest['version'] == 'cupcake_b3_everyframe_phase_replacement_v1'
    assert manifest['stride'] == 1 and manifest['dropped_frames'] == 0
    parent = Path(manifest['parent_root'])
    assert sha(parent/'manifest.json') == manifest['parent_manifest_sha256']
    assert sha(root/'source_index.json') == manifest['source_index_sha256']
    assert manifest['reset_sha256']['resets_ObjectAnywhereEEAnywhere.pt'] == manifest['parent_reset_sha256']['resets_ObjectAnywhereEEAnywhere.pt']
    reset = env.cfg.events.reset_from_reset_states.params
    assert Path(reset['dataset_dir']).resolve() == root
    assert reset['reset_types'] == RESET_TYPES and reset['probs'] == [.25] * 4
    manager = env.event_manager.get_term_cfg('reset_from_reset_states').func
    assert type(manager) is CupCakeSuccess400PhaseReset
    assert manager.num_states.tolist() == [1200, 13648, 904, 56224]
    initial = initial_matrix(manager.datasets[0])
    xy = initial[:, [31, 32, 44, 45]]
    center = torch.tensor([.44, -.126, .426, .171])
    assert torch.all((xy-center).abs() <= .02501), 'T0 outside the original 5x5cm boxes'
    tilt = []
    for start in (34, 47):
        quat = initial[:, start:start+4]
        angle = torch.acos((1-2*(quat[:,1].square()+quat[:,2].square())).clamp(-1,1))
        tilt.append(float(torch.rad2deg(angle).max()))
    assert max(tilt) <= 1.1, ('T0 not upright', tilt)
    initial_bounds = dict(count=1200, xy_side_m=.05, centers_xy=center.tolist(),
                          measured_min=xy.min(0).values.tolist(), measured_max=xy.max(0).values.tolist(),
                          max_cupcake_plate_tilt_deg=tilt, scope='T0 initial states unchanged; T1-T3 matching-class frame replacement')
    for weights in manager.state_weights:
        assert torch.equal(weights, torch.full_like(weights, 1. / len(weights)))
    assert env.action_manager.total_action_dim == 7
    assert env.observation_manager.group_obs_dim['policy'] == (200,)
    assert env.observation_manager.group_obs_dim['critic'] == (168,)
    assert env.max_episode_length == 640 and math.isclose(env.step_dt, .1)
    assert env.cfg.decimation == 12 and math.isclose(env.physics_dt, 1/120)
    assert not env.curriculum_manager.active_terms
    assert tuple(env.scene['insertive_object'].cfg.spawn.scale) == (.5,) * 3
    assert tuple(env.scene['receptive_object'].cfg.spawn.scale) == (2/3,) * 3
    arm, grip = env.action_manager.get_term('arm'), env.action_manager.get_term('gripper')
    assert type(arm) is RelCartesianDiffIKJointPositionAction
    assert type(grip) is CupCakeHystereticGuard
    assert not any('friction_' in k for k in vars(arm.cfg))
    assert tuple(arm.cfg.max_joint_velocity) == (.2,) * 7 and arm.cfg.ik_damping == .05
    assert tuple(grip.cfg.tcp_offset) == (0., 0., .1034)
    assert grip.cfg.lateral_thresh_xy == .02 and grip.cfg.vertical_thresh_z == .03
    assert grip.cfg.release_margin_m == .002 and grip.cfg.capture_confirm_steps == 5
    for name, expected in _B3_NOMINAL_FIXED_EVENTS.items():
        event = getattr(env.cfg.events, name)
        assert event.mode == 'startup'
        for key, value in expected.items():
            assert event.params[key] == value, (name, key)
    sysid = env.cfg.events.randomize_arm_sysid
    assert tuple(sysid.params['scale_range']) == (1., 1.)
    assert tuple(sysid.params['delay_range']) == (0, 0)
    assert sha(sysid.params['sysid_metadata_path']) == launch['sysid_sha256']
    s = yaml.safe_load(Path(sysid.params['sysid_metadata_path']).read_text())['sysid']
    robot = env.scene['robot']
    values = dict(joint_armature=s['armature'], joint_friction_coeff=s['static_friction'],
                  joint_dynamic_friction_coeff=[min(a*b, a) for a,b in zip(s['static_friction'], s['dynamic_ratio'])],
                  joint_viscous_friction_coeff=s['viscous_friction'])
    env._unified_expected_joint = {k: torch.tensor(v, device=env.device, dtype=getattr(robot.data,k).dtype)
                                   for k, v in values.items()}
    env._unified_reset_checks = 0
    command = env.command_manager.get_term('task_command')
    command.success_position_threshold = .02
    command.success_orientation_threshold = math.radians(3)
    # Smoke-only exhaustive reset readback. Apply B3 before the physical tick.
    if os.environ.get('CUPCAKE_UNIFIED_NATIVE_AUDIT') == '1':
        term = env.event_manager.get_term_cfg('randomize_arm_sysid')
        term.func(env, None, **term.params)
        audit_pregrasp_endpoints(env, manager)
        from .cupcake_b3_everyframe_native_audit import audit_replacement_frames
        audit_replacement_frames(env, manager, root)
    install_t0_audit(env, Path(env.cfg.log_dir) / 'pregrasp_t0_episode_audit')
    recorder = UniqueT0Recorder(env, os.environ['CUPCAKE_UNIQUE_T0_ROOT'])
    recorder.metadata.update(physics_contract=VERSION, physics_contract_sha256=sha(os.environ['CUPCAKE_UNIFIED_CONTRACT']),
                             extra_arm_effort='zero throughout IK/RL/DP contract; IK not executed by this trainer',
                             old_ik_prefix_compatible=False)
    process, apply, compute = env.action_manager.process_action, env.action_manager.apply_action, env.reward_manager.compute
    evidence = dict(version=VERSION, passed=True, control_steps=0, physics_substeps=0,
                    max_extra_effort_nm=0., max_target_delta_rad=0., max_target_write_error_rad=0.,
                    scope='online executed command/readback check, not task success')

    def process_checked(actions):
        recorder.pre(actions)
        result = process(actions)
        q = robot.data.joint_pos[:, :7]
        delta = float((arm.joint_position_targets - q).abs().max())
        assert math.isfinite(delta) and delta <= .0201
        evidence['max_target_delta_rad'] = max(evidence['max_target_delta_rad'], delta)
        evidence['control_steps'] += 1
        return result

    def apply_checked():
        result = apply()
        effort = float(robot.data.joint_effort_target[:, :7].abs().max())
        error = float((robot.data.joint_pos_target[:, :7] - arm.joint_position_targets).abs().max())
        assert effort == 0. and error <= 1e-7, (effort, error)
        assert torch.isfinite(robot.data.joint_pos).all()
        evidence['physics_substeps'] += 1
        evidence['max_extra_effort_nm'] = max(evidence['max_extra_effort_nm'], effort)
        evidence['max_target_write_error_rad'] = max(evidence['max_target_write_error_rad'], error)
        recorder.substep()
        if evidence['physics_substeps'] % (32 * 12) == 0:
            write_report(env, 'unified_execution', evidence)
        return result

    def compute_recorded(dt):
        result = compute(dt)
        recorder.post()
        return result

    env.action_manager.process_action = process_checked
    env.action_manager.apply_action = apply_checked
    env.reward_manager.compute = compute_recorded
    env._cupcake_t0_trajectories = recorder
    write_report(env, 'unified_startup', dict(passed=True, version=VERSION, code_sha256=launch['code_sha256'],
                 dataset_manifest_sha256=launch['dataset_manifest_sha256'], sysid_sha256=launch['sysid_sha256'],
                 arm_controller=type(arm).__name__, guard_controller=type(grip).__name__,
                 success_position_m=.02, success_orientation_deg=3, ik_executed=False,
                 slow_gripper_change=False, reset_probabilities=[.25]*4, initial_bounds=initial_bounds,
                 replacement_counts=manifest['replacement_counts'],source_unique_states=manifest['source_unique_states']))
    print('[UNIFIED_B3_STARTUP] contract verified; no compensation controller; passive T0 recorder enabled', flush=True)


@configclass
class CupCakeB3EveryFrameTrainCfg(CupCakeIKPregrasp1200TrainCfg):
    def __post_init__(self):
        super().__post_init__()
        old = self.actions.arm
        arm = RelCartesianDiffIKJointPositionActionCfg()
        for key in vars(arm):
            if key != 'class_type' and hasattr(old, key):
                setattr(arm, key, copy.deepcopy(getattr(old, key)))
        self.actions.arm = arm
        for name, params in _B3_NOMINAL_FIXED_EVENTS.items():
            event = getattr(self.events, name)
            assert event.mode == 'startup'
            event.params.update(copy.deepcopy(params))
        grip = self.events.randomize_gripper_actuator_parameters
        grip.params['stiffness_distribution_params'] = (1., 1.)
        grip.params['damping_distribution_params'] = (1., 1.)
        self.events.randomize_arm_sysid.params.update(scale_range=(1., 1.), delay_range=(0, 0))
        self.sim.physx.enable_enhanced_determinism = True
        self.sim.physx.gpu_max_num_partitions = 1
        self.sim.physx.bounce_threshold_velocity = .5
        self.sim.physx.friction_correlation_distance = .025
        self.sim.physx.max_position_iteration_count = 32
        self.events.policy_gripper_contract.func = verify_b3_unified_startup
        self.events.repair_physics_contract.func = verify_b3_after_reset
