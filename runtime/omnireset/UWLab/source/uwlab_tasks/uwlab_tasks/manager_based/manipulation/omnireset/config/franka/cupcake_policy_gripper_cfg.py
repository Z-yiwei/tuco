"""CupCake four-path PPO with a policy-controlled gripper and no ADR.

Independent of the historical guarded tasks. The small plate is scaled in the
physics scene, not substituted only in recorded RGB.
"""

import copy
import json
import math
import os
from pathlib import Path

import torch

from isaaclab.envs.mdp.actions import BinaryJointPositionAction, BinaryJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sim import PreviewSurfaceCfg
from isaaclab.utils import configclass

from . import rl_state_cfg as panda_state
from .research3_actions import FR3_RELATIVE_DIFF_IK_JOINT_TARGET
from .research3_cfg import Research3MimicFingertipRlStateSceneCfg, remap_research3_names


RESET_TYPES = [
    "ObjectAnywhereEEAnywhere",
    "ObjectRestingEEGrasped",
    "ObjectAnywhereEEGrasped",
    "ObjectPartiallyAssembledEEGrasped",
]
PLATE_SCALE = 2.0 / 3.0


@configclass
class CupCakePolicyGripperActions:
    arm = FR3_RELATIVE_DIFF_IK_JOINT_TARGET
    gripper = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["fr3_finger.*"],
        open_command_expr={"fr3_finger_.*": 0.04},
        close_command_expr={"fr3_finger_.*": 0.0},
    )


def verify_policy_gripper_contract(env, env_ids, success_position_m=0.020, success_orientation_deg=3.0):
    """Resolve success thresholds and fail before training on a drifted contract."""
    command = env.command_manager.get_term("task_command")
    command.success_position_threshold = success_position_m
    command.success_orientation_threshold = math.radians(success_orientation_deg)
    grip = env.action_manager.get_term("gripper")
    arm = env.action_manager.get_term("arm")
    assert type(grip) is BinaryJointPositionAction, type(grip)
    assert env.action_manager.total_action_dim == 7
    assert tuple(arm.cfg.max_joint_velocity) == (0.25,) * 7
    assert math.isclose(env.step_dt, 0.1)
    assert env.max_episode_length == 640
    assert not env.curriculum_manager.active_terms
    assert tuple(env.scene["receptive_object"].cfg.spawn.scale) == (PLATE_SCALE,) * 3
    expected_insertive_scale = float(os.environ.get("CUPCAKE_EXPECTED_INSERTIVE_SCALE", "1.0"))
    assert tuple(env.scene["insertive_object"].cfg.spawn.scale) == (expected_insertive_scale,) * 3
    assert env.cfg.scene.table.init_state.pos[2] == -0.868
    assert "FrankaFR3/fr3.usd" in env.scene["robot"].cfg.spawn.usd_path
    assert env.observation_manager.group_obs_dim["policy"] == (200,)
    assert env.observation_manager.group_obs_dim["critic"] == (168,)
    reset_cfg = env.cfg.events.reset_from_reset_states.params
    assert reset_cfg["reset_types"] == RESET_TYPES
    assert reset_cfg["probs"] == [0.25] * 4
    assert tuple(env.cfg.events.randomize_arm_sysid.params["scale_range"]) == (1.0, 1.0)

    # Test the actual term, with no geometry-dependent override. Restore its input.
    previous = grip.raw_actions.clone()
    for sign, finger_target in ((-1.0, 0.0), (1.0, 0.04)):
        grip.process_actions(torch.full_like(previous, sign))
        assert torch.allclose(grip.processed_actions, torch.full_like(grip.processed_actions, finger_target))
    grip.process_actions(previous)
    contract = {
        "version": "cupcake_fr3_policy_gripper_fourpath_v025h64_20260905",
        "num_envs": env.num_envs,
        "robot_usd": env.scene["robot"].cfg.spawn.usd_path,
        "robot_visual_profile": os.environ.get("OMNIRESET_FR3_VISUAL_PROFILE", "official"),
        "plate_scale": PLATE_SCALE,
        "cupcake_scale": expected_insertive_scale,
        "plate_collision_scaled": True,
        "table_top_z_m": 0.0,
        "gripper_action_class": type(grip).__name__,
        "gripper_sign_test_passed": True,
        "action_dim": 7,
        "actor_obs_dim": 200,
        "critic_obs_dim": 168,
        "joint_velocity_cap_rad_s": 0.25,
        "control_dt_s": env.step_dt,
        "episode_steps": env.max_episode_length,
        "reset_types": reset_cfg["reset_types"],
        "reset_probabilities": reset_cfg["probs"],
        "adr": False,
        "b3_scale_range": [1.0, 1.0],
        "static_domain_randomization": "inherited OmniReset material/mass/gripper ranges",
        "success_position_m": success_position_m,
        "success_orientation_deg": success_orientation_deg,
        "training_success": "original per-step pose reward, no added success termination",
    }
    print("[POLICY_GRIPPER_CONTRACT] " + json.dumps(contract, sort_keys=True), flush=True)
    if env.cfg.log_dir:
        output = Path(env.cfg.log_dir)
        output.mkdir(parents=True, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        (output / f"policy_gripper_contract_rank{rank}.json").write_text(json.dumps(contract, indent=2) + "\n")


@configclass
class CupCakePolicyGripperEvents(panda_state.JointB3TrainEventCfg):
    policy_gripper_contract = EventTerm(
        func=verify_policy_gripper_contract,
        mode="startup",
        params={"success_position_m": 0.020, "success_orientation_deg": 3.0},
    )


@configclass
class CupCakePolicyGripperTrainCfg(panda_state.FrankaFr3GripperRelCartesianDiffIKJointTargetTrainCfg):
    scene: Research3MimicFingertipRlStateSceneCfg = Research3MimicFingertipRlStateSceneCfg(
        num_envs=32, env_spacing=1.5
    )
    actions: CupCakePolicyGripperActions = CupCakePolicyGripperActions()
    events: CupCakePolicyGripperEvents = CupCakePolicyGripperEvents()
    curriculum = {}

    def __post_init__(self):
        super().__post_init__()
        self.actions.arm.max_joint_velocity = (0.25,) * 7
        self.episode_length_s = 64.0
        # Hydra replaces selected asset variants after __post_init__; set both.
        self.variants = copy.deepcopy(self.variants)
        plate = self.variants["scene.receptive_object"]["plate"]
        plate.spawn.scale = (PLATE_SCALE,) * 3
        plate.spawn.visual_material = PreviewSurfaceCfg(
            diffuse_color=(0.85, 0.85, 0.85), roughness=0.6, metallic=0.0
        )
        self.scene.insertive_object = copy.deepcopy(self.variants["scene.insertive_object"]["cupcake"])
        self.scene.receptive_object = copy.deepcopy(plate)
        remap_research3_names(self)
