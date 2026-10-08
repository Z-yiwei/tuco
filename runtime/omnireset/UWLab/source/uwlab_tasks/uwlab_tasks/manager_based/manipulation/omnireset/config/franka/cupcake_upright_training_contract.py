"""New upright reset contract; original frozen grasp/floor contracts stay intact."""
import hashlib
import json
import math
import os
from pathlib import Path
import torch


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_upright_startup(env,env_ids):
    from .cupcake_policy_gripper_cfg import RESET_TYPES
    from .cupcake_friction_compensated_action import CupCakeFrictionCompensatedDiffIKAction
    from .cupcake_hysteretic_guard import CupCakeHystereticGuard
    from .cupcake_captured_lift_reward import CupCakeCapturedLiftBonus
    from .cupcake_t0_audit import install_t0_audit
    from .cupcake_t0_trajectory import install_t0_trajectories
    root=Path(os.environ['CUPCAKE_RESET_DATASET_ROOT']).resolve()
    manifest=json.loads((root/'manifest.json').read_text());contract=json.loads((root/'training_contract.json').read_text())
    assert manifest['version']=='cupcake_upright_halfmix_v1' and manifest['sealed_for_training']
    assert contract['dataset_manifest_sha256']==sha(root/'manifest.json') and contract['gates_passed']
    assert manifest['pregrasp_fraction_per_path']==[0.,.5,.5,.5]
    assert manifest['four_path_probabilities']==[.25]*4
    assert manifest['counts']=={k:2500 if i==0 else 5000 for i,k in enumerate(RESET_TYPES)}
    for path,digest in contract['gate_evidence_sha256'].items():assert sha(path)==digest,path
    repository=next(p for p in Path(__file__).resolve().parents if (p/'UWLab/source/uwlab_tasks').is_dir())
    for path,digest in contract['code_sha256'].items():assert sha(repository/path)==digest,path
    for name,digest in manifest['reset_sha256'].items():
        path=root/'Resets/CupCakeHalf__Plate'/name;assert sha(path)==digest
        data=torch.load(path,map_location='cpu',weights_only=False)
        q=torch.stack(data['initial_state']['rigid_object']['insertive_object']['root_pose'])[:,3:]
        assert torch.isfinite(q).all() and not torch.count_nonzero(q[:,1:3])
        assert torch.allclose(q.norm(dim=-1),torch.ones(len(q)),atol=1e-6)
    reset=env.cfg.events.reset_from_reset_states.params
    assert Path(reset['dataset_dir']).resolve()==root and reset['reset_types']==RESET_TYPES and reset['probs']==[.25]*4
    assert reset.get('state_indices') is None
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
    robot=env.scene['robot']
    assert torch.all(robot.data.joint_stiffness[:,:7]==80) and torch.all(robot.data.joint_damping[:,:7]==4)
    env._cupcake_repair_expected_physics={k:torch.tensor(v,device=env.device,dtype=getattr(robot.data,k).dtype) for k,v in contract['arm_physics_signature'].items()}
    env._cupcake_repair_physics_checks=0
    reward=env.reward_manager.get_term_cfg('grasped_and_lifted')
    assert isinstance(reward.func,CupCakeCapturedLiftBonus) and reward.weight==2 and reward.params=={'hold_steps':20,'release_margin_m':.002}
    floor=env.cfg.scene.fall_catcher
    assert tuple(floor.spawn.size)==(1000.,1000.,.5) and tuple(floor.init_state.pos)==(0.,0.,-1.118)
    command=env.command_manager.get_term('task_command');command.success_position_threshold=.02;command.success_orientation_threshold=math.radians(3)
    install_t0_audit(env,Path(env.cfg.log_dir)/'t0_episode_audit')
    quota=Path(os.environ['CUPCAKE_T0_TRAJECTORY_ROOT']);marker=json.loads((quota/'COMPLETE.json').read_text())
    assert marker['count']==200 and marker['manifest_sha256']==sha(quota/'manifest.jsonl')==contract['historical_full200_manifest_sha256']
    install_t0_trajectories(env,quota,diagnostic_all=False)
    env._upright_reset_checks=0
    result=dict(reset_root=str(root),manifest_sha256=sha(root/'manifest.json'),initial_upright_all_rows=True,
        historical200_unchanged=True,pregrasp_fraction_per_path=[0.,.5,.5,.5],reset_probs=[.25]*4)
    (Path(env.cfg.log_dir)/f'upright_contract_rank{os.environ.get("RANK","0")}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('[UPRIGHT_CONTRACT]',json.dumps(result),flush=True)


def verify_upright_after_reset(env,env_ids):
    from .cupcake_grasp_repair_cfg import verify_grasp_repair_physics_after_reset
    verify_grasp_repair_physics_after_reset(env,env_ids)
    ids=env.scene._ALL_INDICES if env_ids is None else env_ids
    q=env.scene['insertive_object'].data.root_quat_w[ids]
    assert torch.isfinite(q).all() and (q[:,1:3].abs()<1e-5).all(),'non-upright actual reset'
    if env._upright_reset_checks==0:print('[UPRIGHT_NATIVE_RESET]',len(ids),'verified',flush=True)
    env._upright_reset_checks+=len(ids)
