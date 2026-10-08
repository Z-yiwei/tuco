# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm

from uwlab_assets.robots.ur5e_robotiq_gripper.kinematics import compute_jacobian_analytical

from . import actions_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class RelCartesianOSCAction(ActionTerm):
    """Relative Cartesian OSC action term with task-space PD control.

    Supports two Jacobian sources:
    - analytical (UR5 analytical model, for sim2real alignment)
    - simulation (PhysX geometric Jacobian, robot-agnostic)

    The control law is:
        tau = J^T @ (Kp * pose_error + Kd * vel_error)

    No inertial dynamics decoupling. Velocity is computed from J @ dq.

    The flow per policy step:
        1. process_actions: scale raw 6-DOF delta, compute desired EE pose
        2. apply_actions (every physics step): compute current state and J,
           PD torques, clamp, and apply as joint effort targets

    Frame convention: EE pose and Jacobian are both represented in the robot root-link frame.
    """

    cfg: actions_cfg.RelCartesianOSCActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: actions_cfg.RelCartesianOSCActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Resolve joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._joint_ids_for_jacobian = list(self._joint_ids)
        self._num_dof = len(self._joint_ids)
        # Avoid slice-vs-list indexing overhead when all joints match
        if self._num_dof == self._asset.num_joints:
            self._joint_ids = slice(None)
            self._joint_ids_for_jacobian = list(range(self._asset.num_joints))

        # Resolve EE body
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for body_name '{self.cfg.body_name}', got {len(body_ids)}: {body_names}"
            )
        self._ee_body_idx = body_ids[0]

        # Jacobian source
        source = self.cfg.jacobian_source.lower()
        if source not in ("analytical", "simulation"):
            raise ValueError(
                f"Invalid jacobian_source '{self.cfg.jacobian_source}'. Expected 'analytical' or 'simulation'."
            )
        self._use_analytical_jacobian = source == "analytical"
        if self._use_analytical_jacobian and self._num_dof != 6:
            raise ValueError(
                "Analytical Jacobian currently supports 6-DoF UR5 arm only. "
                "Use jacobian_source='simulation' for other robots."
            )
        self._simulation_jacobian_point = self.cfg.simulation_jacobian_point.lower()
        if self._simulation_jacobian_point not in ("physx_com", "link_origin"):
            raise ValueError(
                "Invalid simulation_jacobian_point "
                f"'{self.cfg.simulation_jacobian_point}'. Expected 'physx_com' or 'link_origin'."
            )
        self._action_reference_blend = float(cfg.action_reference_blend)
        if not 0.0 <= self._action_reference_blend <= 1.0:
            raise ValueError("action_reference_blend must be in [0, 1]")
        for name, value in (
            ("max_linear_velocity", cfg.max_linear_velocity),
            ("max_angular_velocity", cfg.max_angular_velocity),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when set, got {value}")
        self._max_linear_reference_step = (
            None if cfg.max_linear_velocity is None else float(cfg.max_linear_velocity) * float(env.physics_dt)
        )
        self._max_angular_reference_step = (
            None
            if cfg.max_angular_velocity is None
            else float(cfg.max_angular_velocity) * float(env.physics_dt)
        )
        self._reference_rate_limit_enabled = (
            self._max_linear_reference_step is not None
            or self._max_angular_reference_step is not None
        )
        self._physics_dt = float(env.physics_dt)

        # PhysX jacobian indexing (needed for simulation Jacobian)
        if self._asset.is_fixed_base:
            if self._ee_body_idx <= 0:
                raise ValueError(
                    f"Invalid EE body index {self._ee_body_idx} for fixed-base articulation."
                )
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._joint_ids_for_jacobian
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = [joint_id + 6 for joint_id in self._joint_ids_for_jacobian]

        # Controller gains (per-env for domain randomization): Kd = 2 * sqrt(Kp) * damping_ratio
        kp = torch.tensor(cfg.motion_stiffness, device=self.device, dtype=torch.float32)
        damping_ratio = torch.tensor(cfg.motion_damping_ratio, device=self.device, dtype=torch.float32)
        kd = 2.0 * torch.sqrt(kp) * damping_ratio
        # Store defaults (1D) and expand to per-env (N, 6)
        self._kp_default = kp
        self._kd_default = kd
        self._damping_ratio_default = damping_ratio
        self._kp = kp.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._kd = kd.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._torque_max = torch.tensor(cfg.torque_limit, device=self.device, dtype=torch.float32)

        # Action scaling
        self._scale = torch.tensor(cfg.scale_xyz_axisangle, device=self.device, dtype=torch.float32)
        if cfg.input_clip is not None:
            self._input_clip = torch.tensor(cfg.input_clip, device=self.device, dtype=torch.float32)
        else:
            self._input_clip = None

        # Nullspace (for redundant arms, e.g. 7-DOF Franka)
        self._nullspace_enabled = cfg.nullspace_stiffness > 0.0
        if self._nullspace_enabled:
            self._kp_null = cfg.nullspace_stiffness
            self._kd_null = 2.0 * (cfg.nullspace_stiffness ** 0.5) * cfg.nullspace_damping_ratio
            if cfg.nullspace_default_pos is not None:
                self._q0 = torch.tensor(cfg.nullspace_default_pos, device=self.device, dtype=torch.float32)
            else:
                self._q0 = None  # will be set on first reset

        # Buffers
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_pos_des = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_quat_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._ee_pos_goal = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_quat_goal = torch.zeros(self.num_envs, 4, device=self.device)
        self._ee_reference_velocity = torch.zeros(self.num_envs, 6, device=self.device)
        self._joint_torques = torch.zeros(self.num_envs, self._num_dof, device=self.device)
        self._joint_torque_substeps: list[torch.Tensor] = []
        # Diagnostic-only pre-physics-tick state.  These buffers do not feed
        # the controller; they let sim2sim identify the first dynamics split
        # without resetting either simulator between ticks.
        self._joint_pos_substeps: list[torch.Tensor] = []
        self._joint_vel_substeps: list[torch.Tensor] = []
        self._ee_velocity_substeps: list[torch.Tensor] = []
        self._ee_reference_velocity_substeps: list[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def joint_torques(self) -> torch.Tensor:
        """Most recent joint effort target written by this OSC action term."""
        return self._joint_torques

    @property
    def joint_torque_substeps(self) -> torch.Tensor:
        """Joint effort targets written during the current policy step.

        ``process_actions`` clears this buffer once per policy step, and every
        subsequent ``apply_actions`` call appends the torque actually sent to
        ``set_joint_effort_target``. For the default OmniReset decimation this
        has shape ``(12, num_envs, num_arm_joints)`` after ``env.step`` returns.
        """
        if len(self._joint_torque_substeps) == 0:
            return self._joint_torques.new_zeros((0, self.num_envs, self._num_dof))
        return torch.stack(self._joint_torque_substeps, dim=0)

    @property
    def joint_pos_substeps(self) -> torch.Tensor:
        """Arm joint positions immediately before each physics tick."""
        if len(self._joint_pos_substeps) == 0:
            return self._joint_torques.new_zeros((0, self.num_envs, self._num_dof))
        return torch.stack(self._joint_pos_substeps, dim=0)

    @property
    def joint_vel_substeps(self) -> torch.Tensor:
        """Arm joint velocities immediately before each physics tick."""
        if len(self._joint_vel_substeps) == 0:
            return self._joint_torques.new_zeros((0, self.num_envs, self._num_dof))
        return torch.stack(self._joint_vel_substeps, dim=0)

    @property
    def ee_velocity_substeps(self) -> torch.Tensor:
        """Measured end-effector twist for every physics tick in the policy step."""
        if len(self._ee_velocity_substeps) == 0:
            return self._joint_torques.new_zeros((0, self.num_envs, 6))
        return torch.stack(self._ee_velocity_substeps, dim=0)

    @property
    def ee_reference_velocity_substeps(self) -> torch.Tensor:
        """Rate-limited OSC reference twist for every physics tick in the policy step."""
        if len(self._ee_reference_velocity_substeps) == 0:
            return self._joint_torques.new_zeros((0, self.num_envs, 6))
        return torch.stack(self._ee_reference_velocity_substeps, dim=0)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        """Scale raw 6-DOF deltas and compute desired EE pose for the PD tracker.

        Called once per policy step. The desired pose is held fixed while
        apply_actions recomputes torques at each physics step.
        """
        self._raw_actions[:] = actions
        self._joint_torque_substeps.clear()
        self._joint_pos_substeps.clear()
        self._joint_vel_substeps.clear()
        self._ee_velocity_substeps.clear()
        self._ee_reference_velocity_substeps.clear()
        scaled = actions * self._scale
        if self._input_clip is not None:
            scaled = torch.clamp(scaled, min=self._input_clip[0], max=self._input_clip[1])
        self._processed_actions[:] = scaled

        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()
        blend = self._action_reference_blend
        reference_pos = (1.0 - blend) * ee_pos_b + blend * self._ee_pos_goal
        self._ee_pos_goal[:] = reference_pos + scaled[:, :3]
        # Normalized linear quaternion blend with hemisphere alignment. It is
        # exact at both endpoints and avoids introducing another dependency.
        previous_quat = self._ee_quat_goal
        aligned_previous = torch.where(
            torch.sum(ee_quat_b * previous_quat, dim=-1, keepdim=True) < 0.0,
            -previous_quat,
            previous_quat,
        )
        reference_quat = torch.nn.functional.normalize(
            (1.0 - blend) * ee_quat_b + blend * aligned_previous, dim=-1
        )

        # Desired orientation: axis-angle delta -> quaternion -> compose
        delta_rot = scaled[:, 3:6]
        angle = torch.norm(delta_rot, dim=-1, keepdim=True)
        safe_angle = torch.where(angle > 1e-6, angle, torch.ones_like(angle))
        axis = delta_rot / safe_angle
        axis = torch.where(angle > 1e-6, axis, torch.zeros_like(axis))
        half = angle / 2.0
        delta_quat = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
        self._ee_quat_goal[:] = math_utils.quat_mul(delta_quat, reference_quat)
        if not self._reference_rate_limit_enabled:
            self._ee_pos_des[:] = self._ee_pos_goal
            self._ee_quat_des[:] = self._ee_quat_goal
            self._ee_reference_velocity.zero_()

    def apply_actions(self):
        """Compute PD torques using configured Jacobian and apply as joint efforts.

        Called every physics step (decimation times per policy step).
        """
        self._advance_rate_limited_reference()

        # Current state
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
        self._joint_pos_substeps.append(joint_pos.detach().clone())
        self._joint_vel_substeps.append(joint_vel.detach().clone())

        # Jacobian in root-link frame, matching EE pose frame
        if self._use_analytical_jacobian:
            jacobian = compute_jacobian_analytical(joint_pos, device=str(self.device))
        else:
            jacobian = self._get_jacobian_root_frame()

        # EE velocity from J @ dq (consistent with analytical Jacobian)
        ee_vel = torch.bmm(jacobian, joint_vel.unsqueeze(-1)).squeeze(-1)  # (N, 6)
        self._ee_velocity_substeps.append(ee_vel.detach().clone())
        self._ee_reference_velocity_substeps.append(self._ee_reference_velocity.detach().clone())

        # Pose error
        pos_error = self._ee_pos_des - ee_pos_b
        quat_error = math_utils.quat_mul(self._ee_quat_des, math_utils.quat_inv(ee_quat_b))
        axis_angle_error = math_utils.axis_angle_from_quat(quat_error)
        pose_error = torch.cat([pos_error, axis_angle_error], dim=-1)  # (N, 6)

        # Track both the moving pose reference and its feed-forward Cartesian velocity.
        vel_error = self._ee_reference_velocity - ee_vel
        task_force = self._kp * pose_error + self._kd * vel_error
        joint_torques = torch.bmm(jacobian.transpose(-1, -2), task_force.unsqueeze(-1)).squeeze(-1)

        # Nullspace: simple joint-space spring toward default posture.
        # No projector needed — the weak stiffness ensures it doesn't fight
        # the task-space controller significantly.
        if self._nullspace_enabled:
            q0 = self._q0.unsqueeze(0).expand(self.num_envs, -1) if self._q0.dim() == 1 else self._q0
            tau_null = self._kp_null * (q0 - joint_pos) - self._kd_null * joint_vel
            joint_torques = joint_torques + tau_null

        joint_torques = torch.clamp(joint_torques, -self._torque_max, self._torque_max)
        self._joint_torques[:] = joint_torques
        self._joint_torque_substeps.append(joint_torques.detach().clone())

        self._asset.set_joint_effort_target(joint_torques, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset targets to current EE pose to avoid transients."""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()
        self._ee_pos_des[env_ids] = ee_pos_b[env_ids]
        self._ee_quat_des[env_ids] = ee_quat_b[env_ids]
        self._ee_pos_goal[env_ids] = ee_pos_b[env_ids]
        self._ee_quat_goal[env_ids] = ee_quat_b[env_ids]
        self._ee_reference_velocity[env_ids] = 0.0
        # Capture nullspace default from current joint state if not configured
        if self._nullspace_enabled and self._q0 is None:
            self._q0 = self._asset.data.joint_pos[:, self._joint_ids].clone()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clip_vector_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
        norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        scale = torch.clamp(max_norm / torch.clamp_min(norm, 1.0e-12), max=1.0)
        return vector * scale

    def _advance_rate_limited_reference(self) -> None:
        if not self._reference_rate_limit_enabled:
            self._ee_reference_velocity.zero_()
            return

        if self._max_linear_reference_step is None:
            linear_step = self._ee_pos_goal - self._ee_pos_des
        else:
            linear_step = self._clip_vector_norm(
                self._ee_pos_goal - self._ee_pos_des,
                self._max_linear_reference_step,
            )
        self._ee_pos_des.add_(linear_step)

        angular_error = math_utils.quat_box_minus(self._ee_quat_goal, self._ee_quat_des)
        if self._max_angular_reference_step is None:
            angular_step = angular_error
        else:
            angular_step = self._clip_vector_norm(
                angular_error,
                self._max_angular_reference_step,
            )
        self._ee_quat_des[:] = math_utils.quat_box_plus(self._ee_quat_des, angular_step)

        self._ee_reference_velocity[:, :3] = linear_step / self._physics_dt
        self._ee_reference_velocity[:, 3:] = angular_step / self._physics_dt

    def _get_ee_pose_root_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get EE pose in root-link frame from sim state."""
        ee_pos_w = self._asset.data.body_link_pos_w[:, self._ee_body_idx]
        ee_quat_w = self._asset.data.body_link_quat_w[:, self._ee_body_idx]
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            self._asset.data.root_link_pos_w,
            self._asset.data.root_link_quat_w,
            ee_pos_w,
            ee_quat_w,
        )
        return ee_pos_b, ee_quat_b

    def _get_jacobian_root_frame(self) -> torch.Tensor:
        """Get the selected EE geometric Jacobian in the root-link frame."""
        jacobian_w = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        base_rot = self._asset.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, :3, :] = torch.bmm(base_rot_matrix, jacobian_w[:, :3, :])
        jacobian_b[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian_w[:, 3:, :])
        if self._simulation_jacobian_point == "link_origin":
            # PhysX's body Jacobian is evaluated at the center of mass, while
            # _get_ee_pose_root_frame() uses the authored link origin.  For
            # r = origin->COM, v_COM = v_origin + omega x r, hence
            # Jv_origin = Jv_COM + skew(r) @ Jw.
            link_to_com_w = (
                self._asset.data.body_com_pos_w[:, self._ee_body_idx]
                - self._asset.data.body_link_pos_w[:, self._ee_body_idx]
            )
            link_to_com_b = torch.bmm(base_rot_matrix, link_to_com_w.unsqueeze(-1)).squeeze(-1)
            jacobian_b[:, :3, :] += torch.bmm(
                math_utils.skew_symmetric_matrix(link_to_com_b), jacobian_b[:, 3:, :]
            )
        return jacobian_b


class RelCartesianDiffIKJointPositionAction(RelCartesianOSCAction):
    """Bridge a legacy Cartesian OSC teacher to real absolute joint targets.

    The action presented to the teacher remains the historical 6-D Cartesian
    delta. The command actually sent to the articulation is a rate- and
    joint-limit-clipped 7-D position target.
    """

    cfg: actions_cfg.RelCartesianDiffIKJointPositionActionCfg

    def __init__(self, cfg: actions_cfg.RelCartesianDiffIKJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if len(cfg.max_joint_velocity) != self._num_dof:
            raise ValueError(
                "max_joint_velocity must contain one value per controlled joint: "
                f"expected {self._num_dof}, got {len(cfg.max_joint_velocity)}"
            )
        if cfg.ik_damping <= 0.0:
            raise ValueError(f"ik_damping must be positive, got {cfg.ik_damping}")
        if not 0.0 < cfg.ik_step_scale <= 1.0:
            raise ValueError(f"ik_step_scale must be in (0, 1], got {cfg.ik_step_scale}")
        if cfg.joint_limit_margin < 0.0:
            raise ValueError(f"joint_limit_margin must be non-negative, got {cfg.joint_limit_margin}")

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": cfg.ik_damping},
        )
        self._ik_controller = DifferentialIKController(
            ik_cfg, num_envs=self.num_envs, device=str(self.device)
        )
        self._max_joint_delta = (
            torch.tensor(cfg.max_joint_velocity, device=self.device, dtype=torch.float32) * float(env.step_dt)
        )
        self._ik_step_scale = float(cfg.ik_step_scale)
        self._joint_limit_margin = float(cfg.joint_limit_margin)

        shape = (self.num_envs, self._num_dof)
        self._ik_joint_position_targets = torch.zeros(shape, device=self.device)
        self._joint_position_targets = torch.zeros(shape, device=self.device)
        # These survive an automatic environment reset so a collector can read
        # the exact command associated with the pre-step image and observation.
        self._last_ik_joint_position_targets = torch.zeros(shape, device=self.device)
        self._last_joint_position_targets = torch.zeros(shape, device=self.device)
        self._last_applied_joint_position_targets = torch.zeros(shape, device=self.device)
        self._last_ik_valid = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._needs_held_target_sync = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Diagnostic/evaluation-only stop-go mode.  When enabled, subsequent
        # env steps keep applying the exact q target computed on the preceding
        # policy step instead of re-integrating the same Cartesian delta.
        self._hold_joint_position_target = False

    def set_hold_joint_position_target(self, enabled: bool) -> None:
        """Freeze/unfreeze the currently processed absolute joint target."""
        self._hold_joint_position_target = bool(enabled)

    @property
    def ik_joint_position_targets(self) -> torch.Tensor:
        """Unclipped joint target produced by differential IK for the active step."""
        return self._ik_joint_position_targets

    @property
    def joint_position_targets(self) -> torch.Tensor:
        """Rate- and joint-limit-clipped target currently sent to the articulation."""
        return self._joint_position_targets

    @property
    def last_ik_joint_position_targets(self) -> torch.Tensor:
        """Unclipped IK target produced by the most recently processed teacher action."""
        return self._last_ik_joint_position_targets

    @property
    def last_joint_position_targets(self) -> torch.Tensor:
        """Exact joint target sent for the most recently processed teacher action."""
        return self._last_joint_position_targets

    @property
    def last_applied_joint_position_targets(self) -> torch.Tensor:
        """Exact position target passed to ``set_joint_position_target``."""
        return self._last_applied_joint_position_targets

    @property
    def last_ik_valid(self) -> torch.Tensor:
        """Whether the most recent IK target was finite for each environment."""
        return self._last_ik_valid

    def process_actions(self, actions: torch.Tensor):
        if self._hold_joint_position_target:
            # A vectorized env may auto-reset some rows between global policy
            # decisions.  reset() correctly replaces their held target with
            # the new measured home q; keep the public "last" diagnostics in
            # sync before the next apply so collectors never compare a stale
            # pre-reset command against that safe home hold.
            sync = self._needs_held_target_sync
            if bool(sync.any().item()):
                self._last_ik_joint_position_targets[sync] = self._ik_joint_position_targets[sync]
                self._last_joint_position_targets[sync] = self._joint_position_targets[sync]
                self._last_ik_valid[sync] = True
                self._needs_held_target_sync[sync] = False
            return
        # Reuse the historical scaling, anchoring, and quaternion convention to
        # construct exactly the EE target the OSC teacher intended.
        super().process_actions(actions)

        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()
        jacobian = self._get_jacobian_root_frame()
        pose_command = torch.cat([self._ee_pos_des, self._ee_quat_des], dim=-1)
        self._ik_controller.set_command(pose_command)
        ik_target = self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        ik_valid = torch.isfinite(ik_target).all(dim=-1)
        ik_target = torch.where(ik_valid.unsqueeze(-1), ik_target, joint_pos)
        ik_target = joint_pos + self._ik_step_scale * (ik_target - joint_pos)
        self._ik_joint_position_targets[:] = ik_target
        self._last_ik_joint_position_targets[:] = ik_target
        self._last_ik_valid[:] = ik_valid

        joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        lower = joint_limits[..., 0] + self._joint_limit_margin
        upper = joint_limits[..., 1] - self._joint_limit_margin
        joint_limited_target = torch.minimum(torch.maximum(ik_target, lower), upper)
        delta = torch.clamp(
            joint_limited_target - joint_pos,
            min=-self._max_joint_delta,
            max=self._max_joint_delta,
        )
        self._joint_position_targets[:] = joint_pos + delta
        self._last_joint_position_targets[:] = self._joint_position_targets
        self._needs_held_target_sync[:] = False

    def apply_actions(self):
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids]
        self._joint_pos_substeps.append(joint_pos.detach().clone())
        self._joint_vel_substeps.append(joint_vel.detach().clone())
        self._last_applied_joint_position_targets[:] = self._joint_position_targets
        self._asset.set_joint_position_target(self._joint_position_targets, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(self._joint_position_targets), joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._ik_joint_position_targets[env_ids] = joint_pos[env_ids]
        self._joint_position_targets[env_ids] = joint_pos[env_ids]
        self._needs_held_target_sync[env_ids] = True
        self._ik_controller.reset(env_ids)
