"""Versioned CupCake grasp repair task. Refuses unsealed/reset-only candidates."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from .cupcake_half_guarded_gripper_cfg import CupCakeHalfGuardedTrainCfg,CupCakeHalfGuardedEvents
from .cupcake_policy_gripper_cfg import RESET_TYPES
from .cupcake_friction_compensated_action import CupCakeFrictionCompensatedDiffIKAction,CupCakeFrictionCompensatedDiffIKActionCfg
from .cupcake_captured_lift_reward import CupCakeCapturedLiftBonus
from .cupcake_hysteretic_guard import CupCakeHystereticGuard,CupCakeHystereticGuardCfg
from .cupcake_t0_audit import CupCakeT0AuditedReset,install_t0_audit


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_grasp_repair_contract(env,env_ids):
    root=Path(os.environ['CUPCAKE_RESET_DATASET_ROOT']).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    contract=json.loads((root/'training_contract.json').read_text())
    assert manifest['version']=='cupcake_half_nearcontact_halfmix_v1'
    assert contract['version']=='cupcake_grasp_repair_v1' and contract['gates_passed'] is True
    assert contract['dataset_manifest_sha256']==sha(root/'manifest.json')
    assert manifest['counts']=={name:2500 if i==0 else 5000 for i,name in enumerate(RESET_TYPES)}
    assert manifest['pregrasp_fraction_per_path']==[0.,.5,.5,.5]
    for name,digest in manifest['reset_sha256'].items():
        assert sha(root/'Resets/CupCakeHalf__Plate'/name)==digest
    for relative,digest in contract['dataset_artifact_sha256'].items():
        assert sha(root/relative)==digest,relative
    # Locate repo by its existing UWLab directory, not by the caller's cwd.
    repository=next(parent for parent in Path(__file__).resolve().parents if (parent/'UWLab/source/uwlab_tasks').is_dir())
    for relative,digest in contract['code_sha256'].items():
        assert sha(repository/relative)==digest,f'source drift: {relative}'
    reset=env.cfg.events.reset_from_reset_states.params
    assert Path(reset['dataset_dir']).resolve()==root
    assert reset['reset_types']==RESET_TYPES and reset['probs']==[.25]*4
    assert env.action_manager.total_action_dim==7
    assert env.observation_manager.group_obs_dim['policy']==(200,)
    assert env.observation_manager.group_obs_dim['critic']==(168,)
    assert env.max_episode_length==640 and math.isclose(env.step_dt,.1)
    assert not env.curriculum_manager.active_terms
    assert tuple(env.scene['insertive_object'].cfg.spawn.scale)==(.5,)*3
    assert tuple(env.scene['receptive_object'].cfg.spawn.scale)==(2/3,)*3
    arm=env.action_manager.get_term('arm')
    grip=env.action_manager.get_term('gripper')
    assert type(arm) is CupCakeFrictionCompensatedDiffIKAction
    assert type(grip) is CupCakeHystereticGuard
    assert tuple(arm.cfg.max_joint_velocity)==(.2,)*7 and arm.cfg.ik_damping==.05
    assert arm.cfg.static_friction_compensation_gain==.9
    assert arm.cfg.moving_friction_compensation_gain==1.
    assert type(contract['friction_after_capture_only']) is bool
    assert arm.cfg.friction_after_capture_only==contract['friction_after_capture_only']
    assert contract['friction_capture_ramp_s'] in (0.,.5)
    assert arm.cfg.friction_capture_ramp_s==contract['friction_capture_ramp_s']
    assert arm.cfg.friction_capture_ramp_s==0 or arm.cfg.friction_after_capture_only
    assert not arm.cfg.friction_assist_only_toward_target
    assert arm.cfg.friction_position_deadband==1e-4 and arm.cfg.friction_velocity_threshold==.001
    assert tuple(grip.cfg.tcp_offset)==(0.,0.,.1034)
    assert grip.cfg.lateral_thresh_xy==.02 and grip.cfg.vertical_thresh_z==.03
    assert grip.cfg.release_margin_m==.002 and grip.cfg.capture_confirm_steps==5
    assert not grip._hold_processed_action
    robot=env.scene['robot']
    assert torch.allclose(robot.data.joint_stiffness[:,:7],torch.full_like(robot.data.joint_stiffness[:,:7],80))
    assert torch.allclose(robot.data.joint_damping[:,:7],torch.full_like(robot.data.joint_damping[:,:7],4))
    for name,values in contract['arm_physics_signature'].items():
        assert hasattr(robot.data,name),f'missing physics field: {name}'
    # B3 sysid is a RESET event, not startup. Validate its actual output only
    # after that event has written the plant parameters, before any env.step.
    env._cupcake_repair_expected_physics={name:torch.tensor(values,device=env.device,dtype=getattr(robot.data,name).dtype)
                                          for name,values in contract['arm_physics_signature'].items()}
    env._cupcake_repair_physics_checks=0
    reward=env.reward_manager.get_term_cfg('grasped_and_lifted')
    assert isinstance(reward.func,CupCakeCapturedLiftBonus) and reward.weight==2 and reward.params=={'hold_steps':20,'release_margin_m':.002}
    command=env.command_manager.get_term('task_command')
    command.success_position_threshold=.02
    command.success_orientation_threshold=math.radians(3)
    result={'version':'cupcake_grasp_repair_v1','manifest_sha256':sha(root/'manifest.json'),
            'training_contract_sha256':sha(root/'training_contract.json'),'reset_root':str(root),
            'reset_probs':[.25]*4,'pregrasp_fraction_per_path':[0.,.5,.5,.5],
            'arm_controller':type(arm).__name__,'lift_reward':type(reward.func).__name__,
            'friction_after_capture_only':arm.cfg.friction_after_capture_only,
            'friction_capture_ramp_s':arm.cfg.friction_capture_ramp_s,
            'guard_capture_confirm_steps':5,'guard_release_margin_m':.002,'reward_release_margin_m':.002,
            'configuration_validated':True,'physics_validation':'pending_post_sysid_reset'}
    print('[CUPCAKE_GRASP_REPAIR_CONTRACT] '+json.dumps(result),flush=True)
    if env.cfg.log_dir:
        dest=Path(env.cfg.log_dir)
        dest.mkdir(parents=True,exist_ok=True)
        (dest/f'cupcake_grasp_repair_contract_rank{os.environ.get("RANK","0")}.json').write_text(json.dumps(result,indent=2)+'\n')
        install_t0_audit(env,dest/'t0_episode_audit')
    else:
        raise RuntimeError('Repair training requires log_dir for mandatory T0 episode audit')


def verify_grasp_repair_physics_after_reset(env,env_ids):
    assert hasattr(env,'_cupcake_repair_expected_physics'),'startup contract was not checked'
    ids=env.scene._ALL_INDICES if env_ids is None else env_ids
    robot=env.scene['robot']
    for name,expected in env._cupcake_repair_expected_physics.items():
        actual=getattr(robot.data,name)[ids,:7]
        assert torch.allclose(actual,expected.expand_as(actual),atol=1e-6,rtol=1e-6),f'arm physics drift after reset: {name}'
    if env._cupcake_repair_physics_checks==0:
        result={'validated':True,'stage':'after_sysid_reset','env_count':len(ids),
                'fields':sorted(env._cupcake_repair_expected_physics),'original_physics_retained':True}
        print('[CUPCAKE_GRASP_REPAIR_PHYSICS] '+json.dumps(result),flush=True)
        path=Path(env.cfg.log_dir)/f'cupcake_grasp_repair_physics_rank{os.environ.get("RANK","0")}.json'
        path.write_text(json.dumps(result,indent=2)+'\n')
    env._cupcake_repair_physics_checks+=len(ids)


@configclass
class CupCakeGraspRepairEvents(CupCakeHalfGuardedEvents):
    policy_gripper_contract=EventTerm(func=verify_grasp_repair_contract,mode='startup')
    repair_physics_contract=EventTerm(func=verify_grasp_repair_physics_after_reset,mode='reset')


@configclass
class CupCakeGraspRepairTrainCfg(CupCakeHalfGuardedTrainCfg):
    events:CupCakeGraspRepairEvents=CupCakeGraspRepairEvents()

    def __post_init__(self):
        super().__post_init__()
        root=Path(os.environ['CUPCAKE_RESET_DATASET_ROOT'])
        contract=json.loads((root/'training_contract.json').read_text())
        assert contract['version']=='cupcake_grasp_repair_v1' and contract['gates_passed'] is True
        assert type(contract['friction_after_capture_only']) is bool
        new=CupCakeFrictionCompensatedDiffIKActionCfg()
        for key,value in vars(self.actions.arm).items():
            if key!='class_type':setattr(new,key,copy.deepcopy(value))
        new.friction_after_capture_only=contract['friction_after_capture_only']
        new.friction_capture_ramp_s=contract['friction_capture_ramp_s']
        self.actions.arm=new
        grip=CupCakeHystereticGuardCfg()
        for key,value in vars(self.actions.gripper).items():
            if key!='class_type':setattr(grip,key,copy.deepcopy(value))
        grip.release_margin_m=.002
        grip.capture_confirm_steps=5
        self.actions.gripper=grip
        self.rewards.grasped_and_lifted.func=CupCakeCapturedLiftBonus
        self.rewards.grasped_and_lifted.params={'hold_steps':20,'release_margin_m':.002}
        self.events.reset_from_reset_states.func=CupCakeT0AuditedReset
