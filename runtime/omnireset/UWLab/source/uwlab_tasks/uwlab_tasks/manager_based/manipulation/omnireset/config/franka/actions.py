# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import AbsBinaryJointPositionActionCfg
from isaaclab.utils import configclass

from uwlab_assets.robots.franka import BINARY_GRIPPER

from ...mdp.actions.actions_cfg import (
    RateLimitedAbsoluteJointPositionActionCfg,
    RateLimitedRelativeJointPositionActionCfg,
    RelCartesianDiffIKJointPositionActionCfg,
    RelCartesianOSCActionCfg,
)
from .grasp_guarded_action import GraspGuardedBinaryGripperActionCfg

# Rule-based gripper: closes when peg root is within capture zone of TCP, else opens.
# Replaces BINARY_GRIPPER (policy-controlled) — policy gripper action is ignored.
GRASP_GUARDED_GRIPPER = GraspGuardedBinaryGripperActionCfg(
    asset_name="robot",
    joint_names=["panda_finger.*"],
    open_command_expr={"panda_finger_.*": 0.04},
    close_command_expr={"panda_finger_.*": 0.0},
    ee_body_name="panda_hand",
    peg_asset_name="insertive_object",
    tcp_offset=(0.0, 0.0, 0.1034),
    lateral_thresh_xy=0.02,
    vertical_thresh_z=0.03,
)

# Franka Panda default joint posture (from factory_assets_cfg.py init_state)
FRANKA_DEFAULT_JOINT_POS = (0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0)
FRANKA_ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

# Pre-train gains (soft initial Kp; curriculum ramps to stiff terminal)
FRANKA_FR3_RELATIVE_OSC = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_hand",
    jacobian_source="simulation",
    # xyz translation scale matched to UR5e original (0.02). Earlier 0.08 setting
    # produced single-step EE displacement RMS ≈ 24 cm, far too large to dwell in
    # the ±2 cm grasp capture zone — policy could not bootstrap task_0.
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
    torque_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    nullspace_stiffness=0.0,
    nullspace_damping_ratio=1.0,
    nullspace_default_pos=FRANKA_DEFAULT_JOINT_POS,
)

# Eval / sim2real gains (high Kp, end-of-curriculum values)
FRANKA_FR3_RELATIVE_OSC_EVAL = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_hand",
    jacobian_source="simulation",
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    nullspace_stiffness=0.0,
    nullspace_damping_ratio=1.0,
    nullspace_default_pos=FRANKA_DEFAULT_JOINT_POS,
)

# Unscaled (for sysid scripts)
FRANKA_FR3_RELATIVE_OSC_UNSCALED = RelCartesianOSCActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_hand",
    jacobian_source="simulation",
    scale_xyz_axisangle=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0),
    nullspace_stiffness=0.0,
    nullspace_damping_ratio=1.0,
    nullspace_default_pos=FRANKA_DEFAULT_JOINT_POS,
)

# Cartesian-policy / joint-position-plant contract used to adapt the successful
# CupCake and StackCube OSC policies into executable absolute-joint-target
# teachers.  Both canonical Stage-2 experts use this terminal OSC scale; only
# the downstream arm plant changes.
FRANKA_FR3_RELATIVE_DIFF_IK_JOINT_TARGET = RelCartesianDiffIKJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_hand",
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


@configclass
class FrankaFr3GripperRelativeOSCAction:
    """Action config using the simulation OSC + Franka binary gripper."""

    arm = FRANKA_FR3_RELATIVE_OSC
    gripper = GRASP_GUARDED_GRIPPER


@configclass
class FrankaFr3GripperRelativeOSCEvalAction:
    """Action config with high Kp gains (end-of-curriculum values) for eval / data-collection."""

    arm = FRANKA_FR3_RELATIVE_OSC_EVAL
    gripper = GRASP_GUARDED_GRIPPER


@configclass
class FrankaFr3GripperSysidOSCAction:
    """Unscaled arm action (Cartesian delta) + binary gripper. For Sysid env / scripts."""

    arm = FRANKA_FR3_RELATIVE_OSC_UNSCALED
    gripper = GRASP_GUARDED_GRIPPER


@configclass
class FrankaFr3GripperRelativeDiffIKJointTargetAction:
    """Cartesian deltas converted to executable absolute arm joint targets."""

    arm = FRANKA_FR3_RELATIVE_DIFF_IK_JOINT_TARGET
    # Preserve the exact policy-facing seventh action slot and guarded-gripper
    # behavior used by the canonical CupCake and StackCube OSC checkpoints.
    gripper = GRASP_GUARDED_GRIPPER


@configclass
class FrankaFr3GripperAbsoluteJointTargetAction:
    """Absolute arm joint targets in radians plus total gripper width in meters.

    The 0.30 rad/s rate is part of the current 300x10 simulation checkpoint
    contract. It has not been validated as the Franka streaming-controller
    limit and must not be copied to hardware without a separate safety review.
    """

    arm = RateLimitedAbsoluteJointPositionActionCfg(
        asset_name="robot",
        joint_names=FRANKA_ARM_JOINT_NAMES,
        scale=1.0,
        offset=0.0,
        preserve_order=True,
        use_default_offset=False,
        max_joint_velocity=(0.30,) * 7,
        joint_limit_margin=0.01,
    )
    gripper = AbsBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger_joint1", "panda_finger_joint2"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
        threshold=0.04,
        positive_threshold=True,
    )


@configclass
class FrankaFr3GripperRelativeJointTargetAction:
    """Policy-controlled joint deltas and gripper, executed by joint PD."""

    arm = RateLimitedRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=FRANKA_ARM_JOINT_NAMES,
        scale=1.0,
        offset=0.0,
        preserve_order=True,
        use_default_offset=False,
        max_joint_velocity=(1.50,) * 7,
        joint_limit_margin=0.01,
    )
    gripper = BINARY_GRIPPER
