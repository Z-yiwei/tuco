# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

from . import joint_position_actions, task_space_actions


@configclass
class RelCartesianOSCActionCfg(ActionTermCfg):
    """Configuration for Relative Cartesian OSC action term.

    Uses the analytical Jacobian from calibrated UR5e kinematics and a simple
    task-space PD controller matching the real robot's OSC implementation:
        tau = J^T @ (Kp * pose_error + Kd * vel_error)

    No inertial dynamics decoupling, no mass matrix. Designed to work with
    the DelayedDCMotor actuator for sim2real alignment.
    """

    class_type: type[ActionTerm] = task_space_actions.RelCartesianOSCAction

    @configclass
    class OffsetCfg:
        """Offset configuration for body or frame offsets."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation offset."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Rotation offset as quaternion (w, x, y, z)."""

    joint_names: list[str] = MISSING
    """Joint names for the arm (regex supported)."""

    body_name: str = MISSING
    """End-effector body name (e.g., 'wrist_3_link')."""

    scale_xyz_axisangle: tuple[float, float, float, float, float, float] = MISSING
    """Per-DOF scaling for [x, y, z, rx, ry, rz] action deltas."""

    input_clip: tuple[float, float] | None = None
    """Optional symmetric clip range for scaled actions."""

    max_linear_velocity: float | None = None
    """Optional linear velocity limit for the moving OSC reference, in m/s.

    This limits the desired Cartesian reference, not the measured end-effector
    velocity. The torque controller remains unchanged apart from tracking the
    rate-limited reference and its feed-forward velocity.
    """

    max_angular_velocity: float | None = None
    """Optional angular velocity limit for the moving OSC reference, in rad/s."""

    motion_stiffness: tuple[float, float, float, float, float, float] = (200.0, 200.0, 200.0, 3.0, 3.0, 3.0)
    """Task-space stiffness Kp for [x, y, z, rx, ry, rz]."""

    motion_damping_ratio: tuple[float, float, float, float, float, float] = (3.0, 3.0, 3.0, 1.0, 1.0, 1.0)
    """Task-space damping ratio. Kd = 2 * sqrt(Kp) * damping_ratio."""

    torque_limit: tuple[float, ...] = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
    """Per-joint torque limits (clamped after J^T multiplication). Length must match number of controlled joints."""

    jacobian_source: str = "analytical"
    """Source of Jacobian used by OSC.

    - ``"analytical"``: UR5 analytical Jacobian implementation.
    - ``"simulation"``: PhysX geometric Jacobian from articulation.
    """

    simulation_jacobian_point: str = "physx_com"
    """Point used by a simulation Jacobian.

    - ``"physx_com"`` preserves the historical controller: PhysX returns the
      body Jacobian at its center of mass.
    - ``"link_origin"`` shifts the linear rows to ``body_link_pos_w`` so the
      Jacobian point matches the pose-error point.
    """

    action_reference_blend: float = 0.0
    """Fraction of the previous target blended into the next delta anchor.

    Zero is the historical measured-EE anchor. One fully accumulates deltas on
    the reset-initialized controller reference. Intermediate values retain a
    small target memory while remaining close to the trained action semantics.
    """

    nullspace_stiffness: float = 0.0
    """Joint-space stiffness for nullspace posture holding. Set > 0 to enable (needed for 7-DOF arms)."""

    nullspace_damping_ratio: float = 1.0
    """Damping ratio for nullspace. Kd_null = 2 * sqrt(Kp_null) * ratio."""

    nullspace_default_pos: tuple[float, ...] | None = None
    """Default joint positions for nullspace target. If None, uses the joint positions at env reset."""


@configclass
class RelCartesianDiffIKJointPositionActionCfg(RelCartesianOSCActionCfg):
    """Convert the existing relative Cartesian teacher command into an absolute joint target.

    This keeps the 6-D action interface expected by an existing OSC teacher, but
    executes the command through differential IK and joint-position control.
    """

    class_type: type[ActionTerm] = task_space_actions.RelCartesianDiffIKJointPositionAction

    ik_damping: float = 0.05
    """Damping coefficient for damped-least-squares differential IK."""

    ik_step_scale: float = 1.0
    """Fraction of the DLS joint increment applied before rate and joint-limit clipping."""

    max_joint_velocity: tuple[float, ...] = MISSING
    """Per-joint command velocity limits in rad/s."""

    joint_limit_margin: float = 0.01
    """Margin in radians retained inside the articulation's soft joint limits."""


@configclass
class RateLimitedAbsoluteJointPositionActionCfg(JointPositionActionCfg):
    """Absolute joint targets with configurable rate and joint-limit filters."""

    class_type: type[ActionTerm] = joint_position_actions.RateLimitedAbsoluteJointPositionAction

    max_joint_velocity: tuple[float, ...] = MISSING
    """Per-joint command velocity limits in rad/s."""

    joint_limit_margin: float = 0.01
    """Margin in radians retained inside the articulation's soft joint limits."""


@configclass
class RateLimitedRelativeJointPositionActionCfg(JointPositionActionCfg):
    """Normalized joint deltas converted once into a held absolute target."""

    class_type: type[ActionTerm] = joint_position_actions.RateLimitedRelativeJointPositionAction

    max_joint_velocity: tuple[float, ...] = MISSING
    """Per-joint command velocity limits in rad/s for a normalized action of one."""

    joint_limit_margin: float = 0.01
    """Margin in radians retained inside the articulation's soft joint limits."""
