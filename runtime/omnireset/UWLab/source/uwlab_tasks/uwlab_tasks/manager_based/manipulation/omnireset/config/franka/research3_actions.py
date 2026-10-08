# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action contracts for the official Franka Research 3 USD."""

from isaaclab.envs.mdp.actions.actions_cfg import AbsBinaryJointPositionActionCfg
from isaaclab.utils import configclass

from ...mdp.actions.actions_cfg import (
    RateLimitedAbsoluteJointPositionActionCfg,
    RelCartesianDiffIKJointPositionActionCfg,
    RelCartesianOSCActionCfg,
)
from .actions import FRANKA_DEFAULT_JOINT_POS
from .grasp_guarded_action import GraspGuardedBinaryGripperActionCfg


FR3_ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]

FR3_GUARDED_GRIPPER = GraspGuardedBinaryGripperActionCfg(
    asset_name="robot",
    joint_names=["fr3_finger.*"],
    open_command_expr={"fr3_finger_.*": 0.04},
    close_command_expr={"fr3_finger_.*": 0.0},
    ee_body_name="fr3_hand",
    peg_asset_name="insertive_object",
    tcp_offset=(0.0, 0.0, 0.1034),
    lateral_thresh_xy=0.02,
    vertical_thresh_z=0.03,
)

FR3_RELATIVE_DIFF_IK_JOINT_TARGET = RelCartesianDiffIKJointPositionActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    jacobian_source="simulation",
    simulation_jacobian_point="link_origin",
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    nullspace_stiffness=0.0,
    nullspace_damping_ratio=1.0,
    nullspace_default_pos=FRANKA_DEFAULT_JOINT_POS,
    ik_damping=0.05,
    ik_step_scale=1.0,
    max_joint_velocity=(1.50,) * 7,
    joint_limit_margin=0.01,
)

FR3_RELATIVE_OSC_EVAL = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    jacobian_source="simulation",
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    nullspace_stiffness=0.0,
    nullspace_damping_ratio=1.0,
    nullspace_default_pos=FRANKA_DEFAULT_JOINT_POS,
)


@configclass
class FrankaResearch3RelativeDiffIKJointTargetAction:
    arm = FR3_RELATIVE_DIFF_IK_JOINT_TARGET
    gripper = FR3_GUARDED_GRIPPER


@configclass
class FrankaResearch3AbsoluteJointTargetAction:
    arm = RateLimitedAbsoluteJointPositionActionCfg(
        asset_name="robot",
        joint_names=FR3_ARM_JOINT_NAMES,
        scale=1.0,
        offset=0.0,
        preserve_order=True,
        use_default_offset=False,
        max_joint_velocity=(0.30,) * 7,
        joint_limit_margin=0.01,
    )
    gripper = AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["fr3_finger_joint1", "fr3_finger_joint2"],
        open_command_expr={"fr3_finger_.*": 0.04},
        close_command_expr={"fr3_finger_.*": 0.0},
        threshold=0.04,
        positive_threshold=True,
    )


@configclass
class FrankaResearch3RelativeOSCEvalAction:
    arm = FR3_RELATIVE_OSC_EVAL
    gripper = FR3_GUARDED_GRIPPER
