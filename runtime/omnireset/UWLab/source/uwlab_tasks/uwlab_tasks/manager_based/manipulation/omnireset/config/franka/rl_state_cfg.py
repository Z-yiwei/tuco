# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import os
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_tasks.manager_based.manipulation.factory_extension.factory_assets_cfg import FRANKA_PANDA_CFG

from ... import mdp as task_mdp
from ..ur5e_robotiq_2f85 import rl_state_cfg as ur5_rl
from .actions import (
    FrankaFr3GripperAbsoluteJointTargetAction,
    FrankaFr3GripperRelativeDiffIKJointTargetAction,
    FrankaFr3GripperRelativeJointTargetAction,
    FrankaFr3GripperRelativeOSCAction,
    FrankaFr3GripperRelativeOSCEvalAction,
)
from .robot_contract import resolve_franka_robot_contract


def _tcp_to_peg_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Euclidean distance from TCP (panda_hand + (0,0,0.1034) offset in hand frame) to peg root."""
    robot = env.scene["robot"]
    peg = env.scene["insertive_object"]
    hand_name = resolve_franka_robot_contract(robot).hand_body_name
    hand_idx = robot.find_bodies(hand_name)[0][0]
    hand_pos = robot.data.body_link_pos_w[:, hand_idx]
    hand_quat = robot.data.body_link_quat_w[:, hand_idx]
    tcp_offset = torch.tensor([0.0, 0.0, 0.1034], device=env.device).expand_as(hand_pos)
    tcp_pos = hand_pos + math_utils.quat_apply(hand_quat, tcp_offset)
    return torch.norm(peg.data.root_pos_w - tcp_pos, dim=-1)


def _avg_finger_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Average position of the two panda fingers (0.04 = open, 0.0 = closed)."""
    robot = env.scene["robot"]
    finger_names = resolve_franka_robot_contract(robot).finger_joint_names
    finger_idx = robot.find_joints(list(finger_names), preserve_order=True)[0]
    return robot.data.joint_pos[:, finger_idx].mean(dim=-1)


def peg_in_gripper_smooth(
    env: ManagerBasedRLEnv,
    dist_std: float = 0.02,
    finger_open_pos: float = 0.04,
) -> torch.Tensor:
    """Smooth reward = proximity(TCP→peg) × finger_closeness."""
    proximity = torch.exp(-_tcp_to_peg_distance(env) / dist_std)
    closure = 1.0 - (_avg_finger_pos(env) / finger_open_pos).clamp(0.0, 1.0)
    return proximity * closure


def diag_proximity(env: ManagerBasedRLEnv, dist_std: float = 0.02) -> torch.Tensor:
    """DIAGNOSTIC (weight=0): exp(-d_tcp_peg / dist_std). Use to debug reward shape."""
    return torch.exp(-_tcp_to_peg_distance(env) / dist_std)


def diag_closure(env: ManagerBasedRLEnv, finger_open_pos: float = 0.04) -> torch.Tensor:
    """DIAGNOSTIC (weight=0): 1 - clamp(avg_finger/finger_open_pos, 0, 1). Use to debug reward shape."""
    return 1.0 - (_avg_finger_pos(env) / finger_open_pos).clamp(0.0, 1.0)


def diag_tcp_peg_dist(env: ManagerBasedRLEnv) -> torch.Tensor:
    """DIAGNOSTIC (weight=0): raw TCP↔peg distance in meters."""
    return _tcp_to_peg_distance(env)


def diag_avg_finger_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """DIAGNOSTIC (weight=0): raw average finger position in meters."""
    return _avg_finger_pos(env)


class FirstLiftBonus(ManagerTermBase):
    """One-shot per-episode bonus: 1.0 the *first* step that (fingers closed AND peg lifted).

    Stateful — tracks per-env whether the bonus has already fired this episode. On env reset,
    the flag clears. Prevents the "lift-drop-lift" farming exploit that would otherwise let
    the policy accumulate unbounded reward by repeatedly lifting and dropping the peg.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._already_given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self._already_given[:] = False
        else:
            self._already_given[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        lift_threshold: float = 0.025,
        finger_close_threshold: float = 0.02,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        peg = env.scene["insertive_object"]
        finger_names = resolve_franka_robot_contract(robot).finger_joint_names
        finger_idx = robot.find_joints(list(finger_names), preserve_order=True)[0]
        finger_pos = robot.data.joint_pos[:, finger_idx]
        avg_finger = finger_pos.mean(dim=-1)
        peg_z = peg.data.root_pos_w[:, 2]
        closed = avg_finger < finger_close_threshold
        lifted = peg_z > lift_threshold
        # Fire only on FIRST timestep this episode where condition holds
        fire = closed & lifted & (~self._already_given)
        self._already_given = self._already_given | fire
        return fire.float()


def _make_local_fall_catcher() -> AssetBaseCfg:
    """Create an offline equivalent of the historical ground collision."""
    return AssetBaseCfg(
        prim_path="/World/FallCatcher",
        collision_group=-1,
        # A 20 mm thick box with its top at the historical ground z=-0.868 m.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.878)),
        spawn=sim_utils.CuboidCfg(
            size=(1000.0, 1000.0, 0.02),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
        ),
    )


@configclass
class RlStateSceneCfg(ur5_rl.RlStateSceneCfg):
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    receptive_object = ur5_rl.make_receptive_object(
        f"{ur5_rl.UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole_big.usd"
    )
    fall_catcher: AssetBaseCfg | None = (
        _make_local_fall_catcher() if os.environ.get("OMNIRESET_LOCAL_FALL_CATCHER", "0") == "1" else None
    )


@configclass
class BaseEventCfg(ur5_rl.BaseEventCfg):
    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class TrainEventCfg(ur5_rl.TrainEventCfg):
    # 4-path reset mix matching UR5e original. Requires all four reset state
    # files in ./Datasets/OmniReset/Resets/<pair>/ — Step 2 (Resting) needs
    # re-recording (current file is empty), Step 4 (PartialAssembled) needs
    # OmniReset-PartialAssemblies-v0 to be run first.
    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": os.environ.get("OMNIRESET_DATASET_DIR", "./Datasets/OmniReset"),
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            # Optional table-height curriculum offset.  Peg/PegHole resets were
            # recorded with the tabletop at -13 mm, so both rigid objects must
            # move with a corrected tabletop to preserve feasible insertion
            # geometry while the robot adapts to the new absolute height.
            "rigid_object_position_offsets": {
                "insertive_object": (0.0, 0.0, 0.0),
                "receptive_object": (0.0, 0.0, 0.0),
            },
        },
    )

    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


@configclass
class TrainEvalEventCfg(ur5_rl.TrainEvalEventCfg):
    # Override reset states to use local Franka data
    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": os.environ.get("OMNIRESET_DATASET_DIR", "./Datasets/OmniReset"),
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "rigid_object_position_offsets": {
                "insertive_object": (0.0, 0.0, 0.0),
                "receptive_object": (0.0, 0.0, 0.0),
            },
        },
    )

    randomize_gripper_actuator_parameters = EventTerm(
        func=task_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"]),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )


# ---------------------------------------------------------------------------
# Finetune (Stage 2): curriculum-ramped sysid + OSC gain randomization.
# Mirrors UR5e FinetuneEventCfg / FinetuneEvalEventCfg / FinetuneCurriculumsCfg.
#
# IMPORTANT — sysid metadata:
#   Franka uses the locally identified B3 metadata below because IsaacLab's
#   stock USD does not ship with sysid data. The arm event and OSC event are
#   both ramped by the same ADR progress.
#
# Joint mapping (UR5e → Franka):
#   ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
#    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]   (6-DOF)
#   →
#   ["panda_joint1", ..., "panda_joint7"]                  (7-DOF)
# ---------------------------------------------------------------------------

FRANKA_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

FRANKA_B3_SYSID_METADATA_PATH = "data/sysid/metadata_franka_b3.yaml"
STACKCUBE_XY5_T0_HOME_JOINT_JITTER_RAD = 0.2
STACKCUBE_XY5_T0_TEAM_HOME = (
    -0.057557623295429294,
    0.00018949155714934572,
    -0.010052990086534465,
    -1.5410414293429622,
    -0.03143259380432439,
    1.5459539128296862,
    -2.462186588172044,
)
STACKCUBE_XY5_T0_INSERTIVE_POSE_RANGES = {
    "x": (0.343470, 0.393394),
    "y": (-0.153059, -0.103080),
    "z": (0.020, 0.020),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-math.pi, math.pi),
}
STACKCUBE_XY5_T0_RECEPTIVE_POSE_RANGES = {
    "x": (0.440422, 0.490394),
    "y": (-0.082498, -0.032517),
    "z": (0.020, 0.020),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-math.pi, math.pi),
}


@configclass
class FinetuneEventCfg(TrainEventCfg):
    """Finetune events: B3 arm sysid + curriculum-ramped OSC gains."""

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": FRANKA_ARM_JOINT_NAMES,
            "sysid_metadata_path": FRANKA_B3_SYSID_METADATA_PATH,
            # ImplicitActuatorCfg does not have a delay buffer; delay_range is a no-op.
            "actuator_name": "panda_arm1",
            "scale_range": (0.8, 1.2),
            "delay_range": (0, 1),
            "initial_scale_progress": 0.0,
        },
    )

    randomize_osc_gains = EventTerm(
        func=task_mdp.randomize_rel_cartesian_osc_gains,
        mode="reset",
        params={
            "action_name": "arm",
            "scale_range": (0.8, 1.2),
            "terminal_kp": (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
            "terminal_damping_ratio": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "initial_scale_progress": 0.0,
        },
    )


@configclass
class FinetuneEvalEventCfg(BaseEventCfg):
    """Eval after Stage 2: fixed OSC gains (scale_progress=1) + 1-path resets."""

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid_fixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": FRANKA_ARM_JOINT_NAMES,
            "sysid_metadata_path": FRANKA_B3_SYSID_METADATA_PATH,
            "actuator_name": "panda_arm1",
            "scale_range": (0.8, 1.2),
            "delay_range": (0, 1),
        },
    )

    randomize_osc_gains = EventTerm(
        func=task_mdp.randomize_rel_cartesian_osc_gains_fixed,
        mode="reset",
        params={
            "action_name": "arm",
            "scale_range": (0.8, 1.2),
        },
    )

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": os.environ.get("OMNIRESET_DATASET_DIR", "./Datasets/OmniReset"),
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "rigid_object_position_offsets": {
                "insertive_object": (0.0, 0.0, 0.0),
                "receptive_object": (0.0, 0.0, 0.0),
            },
        },
    )


@configclass
class FinetuneCurriculumsCfg:
    """Finetune curriculum: ADR over B3 arm/OSC dynamics + action-scale ramp."""

    adr_sysid = CurrTerm(
        func=task_mdp.adr_sysid_curriculum,
        params={
            "event_term_names": ["randomize_osc_gains", "randomize_arm_sysid"],
            "reset_event_name": "reset_from_reset_states",
            "success_threshold_up": 0.92,
            "success_threshold_down": 0.9,
            "delta": 0.01,
            "update_every_n_steps": 200,
            "initial_scale_progress": 0.0,
            "warmup_success_threshold": 0.92,
            "initial_warmed_up": None,
        },
    )

    action_scale = CurrTerm(
        func=task_mdp.action_scale_curriculum,
        params={
            "action_name": "arm",
            "reset_event_name": "reset_from_reset_states",
            "initial_scales": [0.02, 0.02, 0.02, 0.02, 0.02, 0.2],
            "target_scales": [0.01, 0.01, 0.002, 0.02, 0.02, 0.2],
            "success_threshold_up": 0.95,
            "success_threshold_down": 0.9,
            "delta": 0.01,
            "update_every_n_steps": 200,
            "initial_progress": 0.0,
        },
    )


# Joint-position teacher dynamics are identified around B3, rather than around
# the stock Franka USD.  Stage 1 therefore applies the exact B3 nominal on every
# reset.  Joint Stage 2 starts from that same nominal and expands only the arm
# sysid envelope; it deliberately has neither OSC-gain nor action-scale ADR.
@configclass
class JointB3TrainEventCfg(TrainEventCfg):
    """Four-path training resets with exact B3 nominal arm dynamics."""

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid_fixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": FRANKA_ARM_JOINT_NAMES,
            "sysid_metadata_path": FRANKA_B3_SYSID_METADATA_PATH,
            "actuator_name": "panda_arm1",
            "scale_range": (1.0, 1.0),
            "delay_range": (0, 0),
        },
    )


@configclass
class JointB3EvalEventCfg(TrainEvalEventCfg):
    """One-path evaluation resets with exact B3 nominal arm dynamics."""

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid_fixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": FRANKA_ARM_JOINT_NAMES,
            "sysid_metadata_path": FRANKA_B3_SYSID_METADATA_PATH,
            "actuator_name": "panda_arm1",
            "scale_range": (1.0, 1.0),
            "delay_range": (0, 0),
        },
    )


@configclass
class JointB3XY5T0EventCfg(JointB3TrainEventCfg):
    """Online XY5 t0 resets with independent team-home arm-joint jitter."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.StackCubeXY5T0OnlineReset,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "arm_joint_names": FRANKA_ARM_JOINT_NAMES,
            "team_home_arm_joint_positions": STACKCUBE_XY5_T0_TEAM_HOME,
            "arm_joint_position_offset_range": (
                -STACKCUBE_XY5_T0_HOME_JOINT_JITTER_RAD,
                STACKCUBE_XY5_T0_HOME_JOINT_JITTER_RAD,
            ),
            "gripper_joint_names": ["panda_finger_joint1", "panda_finger_joint2"],
            "gripper_joint_positions": (0.04, 0.04),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "insertive_pose_ranges": STACKCUBE_XY5_T0_INSERTIVE_POSE_RANGES,
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "receptive_pose_ranges": STACKCUBE_XY5_T0_RECEPTIVE_POSE_RANGES,
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class JointFinetuneEventCfg(TrainEventCfg):
    """Joint Stage 2: expand B3 arm dynamics from nominal to +/-20 percent."""

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": FRANKA_ARM_JOINT_NAMES,
            "sysid_metadata_path": FRANKA_B3_SYSID_METADATA_PATH,
            "actuator_name": "panda_arm1",
            "scale_range": (0.8, 1.2),
            "delay_range": (0, 0),
            "initial_scale_progress": 0.0,
            # Critical for continuity with JointB3TrainEventCfg: p=0 writes
            # identified B3 nominal values, never the stock USD defaults.
            "interpolate_from_sysid_nominal": True,
        },
    )


@configclass
class JointFinetuneEvalEventCfg(TrainEvalEventCfg):
    """Joint Stage-2 play: sample the fixed, fully expanded B3 +/-20% range."""

    randomize_arm_sysid = EventTerm(
        func=task_mdp.randomize_arm_from_sysid_fixed,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_names": FRANKA_ARM_JOINT_NAMES,
            "sysid_metadata_path": FRANKA_B3_SYSID_METADATA_PATH,
            "actuator_name": "panda_arm1",
            "scale_range": (0.8, 1.2),
            "delay_range": (0, 0),
        },
    )


@configclass
class JointFinetuneCurriculumsCfg:
    """Joint-only ADR over B3 arm sysid; no OSC or action-scale terms."""

    adr_sysid = CurrTerm(
        func=task_mdp.adr_sysid_curriculum,
        params={
            "event_term_names": ["randomize_arm_sysid"],
            "reset_event_name": "reset_from_reset_states",
            "success_threshold_up": 0.92,
            "success_threshold_down": 0.9,
            "delta": 0.01,
            "update_every_n_steps": 200,
            "initial_scale_progress": 0.0,
            "warmup_success_threshold": 0.92,
            "initial_warmed_up": None,
        },
    )


@configclass
class ObservationsCfg(ur5_rl.ObservationsCfg):
    @configclass
    class PolicyCfg(ur5_rl.ObservationsCfg.PolicyCfg):
        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "rotation_repr": "axis_angle",
            },
        )

    @configclass
    class CriticCfg(ur5_rl.ObservationsCfg.CriticCfg):
        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "rotation_repr": "axis_angle",
            },
        )

        end_effector_vel_lin_ang_b = ObsTerm(
            func=task_mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg(ur5_rl.RewardsCfg):
    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
    )

    ee_asset_distance = RewTerm(
        func=task_mdp.ee_asset_distance_tanh,
        weight=0.1,
        params={
            "root_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "target_asset_cfg": SceneEntityCfg("insertive_object"),
            "root_asset_offset_metadata_key": "gripper_offset",
            "root_asset_offset_override": {"pos": [0.0, 0.0, 0.1034], "quat": [1.0, 0.0, 0.0, 0.0]},
            "std": 1.0,
        },
    )

    # Smooth shaping (weak): pulls EE toward peg + rewards finger closure proximity.
    # Kept at small weight so it provides gradient when peg-not-yet-grasped, but cannot
    # dominate or be cheated (the strong signal comes from grasped+lifted below).
    peg_in_gripper = RewTerm(
        func=peg_in_gripper_smooth,
        weight=0.1,
        params={"dist_std": 0.05, "finger_open_pos": 0.04},
    )
    # ONE-SHOT per episode: 1.0 reward the first time (fingers closed AND peg lifted ≥ 1cm).
    # Stateful — flag clears on reset. Prevents lift-drop-lift farming, so success_reward
    # (which fires every step in-success) dominates the episodic return for full insertion.
    grasped_and_lifted = RewTerm(
        func=FirstLiftBonus,
        weight=2.0,
        params={"lift_threshold": 0.025, "finger_close_threshold": 0.02},
    )

    # Diagnostic terms (weight=0 skips computation; flip to 1.0 to inspect components)
    diag_proximity = RewTerm(func=diag_proximity, weight=0.0, params={"dist_std": 0.05})
    diag_closure = RewTerm(func=diag_closure, weight=0.0, params={"finger_open_pos": 0.04})
    diag_tcp_peg_dist = RewTerm(func=diag_tcp_peg_dist, weight=0.0)
    diag_avg_finger_pos = RewTerm(func=diag_avg_finger_pos, weight=0.0)


# CupCake/Plate reset states (branch franka-stackcube-2500) were collected
# against the local asset mirror, and the remote omni.client.stat() checks are
# flaky under HF rate-limiting — train with the same local USDs.
_LOCAL_ASSETS_DIR = os.environ.get(
    "UWLAB_LOCAL_ASSETS_DIR",
    os.path.abspath(os.path.join(os.getcwd(), "Datasets/local_assets")),
)

_franka_variants = {
    **ur5_rl.variants,
    "scene.insertive_object": {
        **ur5_rl.variants["scene.insertive_object"],
        "cupcake": ur5_rl.make_insertive_object(f"{_LOCAL_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
        "cupcake_half": ur5_rl.make_insertive_object(
            f"{_LOCAL_ASSETS_DIR}/Props/Custom/CupCakeHalf/cupcake.usd"
        ),
    },
    "scene.receptive_object": {
        **ur5_rl.variants["scene.receptive_object"],
        "plate": ur5_rl.make_receptive_object(f"{_LOCAL_ASSETS_DIR}/Props/Custom/Plate/plate.usd"),
    },
}

_franka_variants["scene.insertive_object"]["cupcake_half"].spawn.scale = (0.5, 0.5, 0.5)


@configclass
class FrankaFr3GripperRlStateCfg(ur5_rl.Ur5eRobotiq2f85RlStateCfg):
    scene: RlStateSceneCfg = RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: FrankaFr3GripperRelativeOSCAction = FrankaFr3GripperRelativeOSCAction()
    rewards: RewardsCfg = RewardsCfg()
    events: BaseEventCfg = MISSING
    variants = _franka_variants


@configclass
class FrankaFr3GripperRelCartesianOSCTrainCfg(FrankaFr3GripperRlStateCfg):
    events: TrainEventCfg = TrainEventCfg()
    actions: FrankaFr3GripperRelativeOSCAction = FrankaFr3GripperRelativeOSCAction()


@configclass
class FrankaFr3GripperRelCartesianOSCEvalCfg(FrankaFr3GripperRlStateCfg):
    events: TrainEvalEventCfg = TrainEvalEventCfg()
    # Stage-1 eval should match the Stage-1 training controller. The stiff
    # eval action is reserved for the finetuned/deployment task below.
    actions: FrankaFr3GripperRelativeOSCAction = FrankaFr3GripperRelativeOSCAction()


_B3_NOMINAL_FIXED_EVENTS = {
    "robot_material": {
        "static_friction_range": (0.75, 0.75),
        "dynamic_friction_range": (0.60, 0.60),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "insertive_object_material": {
        "static_friction_range": (1.50, 1.50),
        "dynamic_friction_range": (1.40, 1.40),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "receptive_object_material": {
        "static_friction_range": (0.40, 0.40),
        "dynamic_friction_range": (0.325, 0.325),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "table_material": {
        "static_friction_range": (0.45, 0.45),
        "dynamic_friction_range": (0.35, 0.35),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "randomize_robot_mass": {"mass_distribution_params": (1.0, 1.0)},
    "randomize_insertive_object_mass": {"mass_distribution_params": (0.11, 0.11)},
    "randomize_receptive_object_mass": {"mass_distribution_params": (1.0, 1.0)},
    "randomize_table_mass": {"mass_distribution_params": (1.0, 1.0)},
}


@configclass
class FrankaFr3GripperRelativeJointTargetStateCfg(FrankaFr3GripperRlStateCfg):
    """State RL in the fixed B3 joint-position domain used by joint DP."""

    actions: FrankaFr3GripperRelativeJointTargetAction = FrankaFr3GripperRelativeJointTargetAction()

    def __post_init__(self):
        super().__post_init__()
        for actuator_name in ("panda_arm1", "panda_arm2"):
            self.scene.robot.actuators[actuator_name].stiffness = 80.0
            self.scene.robot.actuators[actuator_name].damping = 4.0
        hand = self.scene.robot.actuators["panda_hand"]
        hand.stiffness = 1000.0
        hand.damping = 14.0
        hand.effort_limit_sim = 60.0

        for event_name, params in _B3_NOMINAL_FIXED_EVENTS.items():
            event = getattr(self.events, event_name, None)
            if event is None or event.mode != "startup":
                raise ValueError(f"joint teacher requires startup event {event_name}")
            event.params.update(params)
        gripper_event = getattr(self.events, "randomize_gripper_actuator_parameters", None)
        if gripper_event is not None:
            gripper_event.params["stiffness_distribution_params"] = (1.0, 1.0)
            gripper_event.params["damping_distribution_params"] = (1.0, 1.0)
        arm_sysid_event = getattr(self.events, "randomize_arm_sysid", None)
        if arm_sysid_event is None:
            raise ValueError("joint teacher requires a B3 randomize_arm_sysid reset event")
        if arm_sysid_event.params.get("sysid_metadata_path") != FRANKA_B3_SYSID_METADATA_PATH:
            raise ValueError("joint teacher must use the identified Franka B3 metadata")

        self.sim.physx.enable_enhanced_determinism = True
        self.sim.physx.gpu_max_num_partitions = 1
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.friction_correlation_distance = 0.025
        self.sim.physx.max_position_iteration_count = 32


@configclass
class FrankaFr3GripperRelativeJointTargetTrainCfg(FrankaFr3GripperRelativeJointTargetStateCfg):
    events: JointB3TrainEventCfg = JointB3TrainEventCfg()


@configclass
class FrankaFr3GripperRelativeJointTargetEvalCfg(FrankaFr3GripperRelativeJointTargetStateCfg):
    events: JointB3EvalEventCfg = JointB3EvalEventCfg()


@configclass
class FrankaFr3GripperRelativeJointTargetFinetuneCfg(FrankaFr3GripperRelativeJointTargetStateCfg):
    """Joint Stage 2: B3 nominal to +/-20% arm-only ADR."""

    events: JointFinetuneEventCfg = JointFinetuneEventCfg()
    curriculum: JointFinetuneCurriculumsCfg = JointFinetuneCurriculumsCfg()


@configclass
class FrankaFr3GripperRelativeJointTargetFinetuneEvalCfg(FrankaFr3GripperRelativeJointTargetStateCfg):
    """Joint Stage-2 play with the fully expanded B3 arm range."""

    events: JointFinetuneEvalEventCfg = JointFinetuneEvalEventCfg()


@configclass
class FrankaFr3GripperRelCartesianDiffIKJointTargetStateCfg(FrankaFr3GripperRlStateCfg):
    """State RL with Cartesian policy actions executed by IK and joint PD.

    Unlike the direct relative-joint teacher, this task intentionally preserves
    the canonical CupCake/StackCube OSC policies' observation, terminal action
    scale, guarded gripper, and scene randomization.  The intended method
    difference is limited to the arm plant: DLS IK produces a post-limit
    absolute joint-position target, which is tracked by the K80/D4 implicit
    joint controller.
    """

    actions: FrankaFr3GripperRelativeDiffIKJointTargetAction = (
        FrankaFr3GripperRelativeDiffIKJointTargetAction()
    )

    def __post_init__(self):
        super().__post_init__()
        for actuator_name in ("panda_arm1", "panda_arm2"):
            self.scene.robot.actuators[actuator_name].stiffness = 80.0
            self.scene.robot.actuators[actuator_name].damping = 4.0

        hand = self.scene.robot.actuators["panda_hand"]
        if (hand.stiffness, hand.damping, hand.effort_limit_sim) != (5000.0, 50.0, 400.0):
            raise ValueError("DiffIK task requires the canonical K5000/D50/E400 guarded gripper")

        arm = self.actions.arm
        if tuple(arm.scale_xyz_axisangle) != (0.01, 0.01, 0.002, 0.02, 0.02, 0.2):
            raise ValueError("DiffIK action scale must match the canonical Stage-2 OSC semantics")
        if tuple(arm.max_joint_velocity) != (1.5,) * 7:
            raise ValueError("DiffIK joint targets require the 1.5 rad/s command limit")
        if arm.simulation_jacobian_point != "link_origin":
            raise ValueError("DiffIK must use the link-origin simulation Jacobian")


@configclass
class FrankaFr3GripperRelCartesianDiffIKJointTargetTrainCfg(
    FrankaFr3GripperRelCartesianDiffIKJointTargetStateCfg
):
    """Stage 1: exact identified B3 arm nominal with four-path resets."""

    events: JointB3TrainEventCfg = JointB3TrainEventCfg()


@configclass
class FrankaFr3GripperRelCartesianDiffIKJointTargetEvalCfg(
    FrankaFr3GripperRelCartesianDiffIKJointTargetStateCfg
):
    """Stage-1 play task: B3 nominal arm with the canonical t0 reset."""

    events: JointB3EvalEventCfg = JointB3EvalEventCfg()


@configclass
class FrankaFr3GripperRelCartesianDiffIKJointTargetXY5T0TrainCfg(
    FrankaFr3GripperRelCartesianDiffIKJointTargetStateCfg
):
    """Fixed-B3 XY5 single-path training with team-home +/-0.2 rad arm resets."""

    events: JointB3XY5T0EventCfg = JointB3XY5T0EventCfg()


@configclass
class FrankaFr3GripperRelCartesianDiffIKJointTargetXY5T0EvalCfg(
    FrankaFr3GripperRelCartesianDiffIKJointTargetStateCfg
):
    """Evaluation contract matching the XY5 single-path training distribution."""

    events: JointB3XY5T0EventCfg = JointB3XY5T0EventCfg()


@configclass
class FrankaFr3GripperRelCartesianDiffIKJointTargetFinetuneCfg(
    FrankaFr3GripperRelCartesianDiffIKJointTargetStateCfg
):
    """Stage 2: expand the identified B3 arm domain from nominal to +/-20%."""

    events: JointFinetuneEventCfg = JointFinetuneEventCfg()
    curriculum: JointFinetuneCurriculumsCfg = JointFinetuneCurriculumsCfg()


@configclass
class FrankaFr3GripperRelCartesianDiffIKJointTargetFinetuneEvalCfg(
    FrankaFr3GripperRelCartesianDiffIKJointTargetStateCfg
):
    """Stage-2 play task at the fully expanded B3 arm range."""

    events: JointFinetuneEvalEventCfg = JointFinetuneEvalEventCfg()


@configclass
class FrankaFr3GripperPegRelCartesianDiffIKJointTargetStateCfg(
    FrankaFr3GripperRelCartesianDiffIKJointTargetStateCfg
):
    """Peg table0 expert contract with only the arm plant changed to DiffIK.

    The successful Peg OSC expert differs from the CupCake/StackCube experts
    in its soft gripper and fixed B3 midpoint scene.  Keep those values, the
    corrected table height, and the common +13 mm reset translation while the
    inherited six-dimensional Cartesian policy adapts to K80/D4 joint PD.
    """

    def __post_init__(self):
        super().__post_init__()

        hand = self.scene.robot.actuators["panda_hand"]
        hand.stiffness = 1000.0
        hand.damping = 14.0
        hand.effort_limit_sim = 60.0

        for event_name, params in _B3_NOMINAL_FIXED_EVENTS.items():
            event = getattr(self.events, event_name, None)
            if event is None or event.mode != "startup":
                raise ValueError(f"Peg DiffIK requires fixed startup event {event_name}")
            event.params.update(params)

        gripper_event = getattr(self.events, "randomize_gripper_actuator_parameters", None)
        if gripper_event is None:
            raise ValueError("Peg DiffIK requires the gripper actuator reset event")
        gripper_event.params["stiffness_distribution_params"] = (1.0, 1.0)
        gripper_event.params["damping_distribution_params"] = (1.0, 1.0)

        self.scene.table.init_state.pos = (0.4, 0.0, -0.868)
        reset_event = self.events.reset_from_reset_states
        reset_event.params["rigid_object_position_offsets"] = {
            "insertive_object": (0.0, 0.0, 0.013),
            "receptive_object": (0.0, 0.0, 0.013),
        }


@configclass
class FrankaFr3GripperPegRelCartesianDiffIKJointTargetTrainCfg(
    FrankaFr3GripperPegRelCartesianDiffIKJointTargetStateCfg
):
    """Peg Stage 1: table0, fixed midpoint scene, exact B3 nominal arm."""

    events: JointB3TrainEventCfg = JointB3TrainEventCfg()


@configclass
class FrankaFr3GripperPegRelCartesianDiffIKJointTargetEvalCfg(
    FrankaFr3GripperPegRelCartesianDiffIKJointTargetStateCfg
):
    """Peg Stage-1 play: canonical t0 with the matched table0 contract."""

    events: JointB3EvalEventCfg = JointB3EvalEventCfg()


@configclass
class FrankaFr3GripperPegRelCartesianDiffIKJointTargetFinetuneCfg(
    FrankaFr3GripperPegRelCartesianDiffIKJointTargetStateCfg
):
    """Peg Stage 2: expand only the identified B3 arm domain to +/-20%."""

    events: JointFinetuneEventCfg = JointFinetuneEventCfg()
    curriculum: JointFinetuneCurriculumsCfg = JointFinetuneCurriculumsCfg()


@configclass
class FrankaFr3GripperPegRelCartesianDiffIKJointTargetFinetuneEvalCfg(
    FrankaFr3GripperPegRelCartesianDiffIKJointTargetStateCfg
):
    """Peg Stage-2 play at the fully expanded B3 arm range."""

    events: JointFinetuneEvalEventCfg = JointFinetuneEvalEventCfg()


@configclass
class FrankaFr3GripperAbsoluteJointTargetStateCfg(FrankaFr3GripperRelativeJointTargetStateCfg):
    """State eval in the same fixed B3 joint-position domain, accepting absolute q targets."""

    actions: FrankaFr3GripperAbsoluteJointTargetAction = FrankaFr3GripperAbsoluteJointTargetAction()


@configclass
class FrankaFr3GripperAbsoluteJointTargetEvalCfg(FrankaFr3GripperAbsoluteJointTargetStateCfg):
    events: TrainEvalEventCfg = TrainEvalEventCfg()


# Finetune (Stage 2): same scene as Stage 1; curriculum ramps OSC gains and
# action scale.  UR5e swaps to EXPLICIT actuator here; Franka has no explicit
# config yet, so we keep FRANKA_PANDA_CFG.
@configclass
class FrankaFr3GripperRelCartesianOSCFinetuneCfg(FrankaFr3GripperRlStateCfg):
    events: FinetuneEventCfg = FinetuneEventCfg()
    actions: FrankaFr3GripperRelativeOSCAction = FrankaFr3GripperRelativeOSCAction()
    curriculum: FinetuneCurriculumsCfg = FinetuneCurriculumsCfg()


@configclass
class FrankaFr3GripperRelCartesianOSCFinetuneEvalCfg(FrankaFr3GripperRlStateCfg):
    events: FinetuneEvalEventCfg = FinetuneEvalEventCfg()
    actions: FrankaFr3GripperRelativeOSCEvalAction = FrankaFr3GripperRelativeOSCEvalAction()
