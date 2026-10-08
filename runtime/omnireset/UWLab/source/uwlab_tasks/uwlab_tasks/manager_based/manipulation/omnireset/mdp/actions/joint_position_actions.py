# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import (
        RateLimitedAbsoluteJointPositionActionCfg,
        RateLimitedRelativeJointPositionActionCfg,
    )


class RateLimitedAbsoluteJointPositionAction(JointPositionAction):
    """Apply absolute joint targets after configurable rate and joint-limit filters."""

    cfg: RateLimitedAbsoluteJointPositionActionCfg

    def __init__(self, cfg: RateLimitedAbsoluteJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if len(cfg.max_joint_velocity) != self.action_dim:
            raise ValueError(
                "max_joint_velocity must contain one value per controlled joint: "
                f"expected {self.action_dim}, got {len(cfg.max_joint_velocity)}"
            )
        self._max_joint_delta = (
            torch.tensor(cfg.max_joint_velocity, device=self.device, dtype=torch.float32) * float(env.step_dt)
        )
        self._joint_limit_margin = float(cfg.joint_limit_margin)
        self._requested_actions = torch.zeros_like(self._processed_actions)
        self._last_applied_actions = torch.zeros_like(self._processed_actions)

    @property
    def requested_actions(self) -> torch.Tensor:
        """Absolute target before the common safety filter."""
        return self._requested_actions

    @property
    def last_applied_actions(self) -> torch.Tensor:
        """Exact target most recently passed to the articulation."""
        return self._last_applied_actions

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._requested_actions[:] = self._processed_actions

        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        lower = joint_limits[..., 0] + self._joint_limit_margin
        upper = joint_limits[..., 1] - self._joint_limit_margin
        joint_limited_target = torch.minimum(torch.maximum(self._processed_actions, lower), upper)
        delta = torch.clamp(
            joint_limited_target - joint_pos,
            min=-self._max_joint_delta,
            max=self._max_joint_delta,
        )
        self._processed_actions[:] = joint_pos + delta

    def apply_actions(self):
        self._last_applied_actions[:] = self._processed_actions
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(self._processed_actions), joint_ids=self._joint_ids)


class RateLimitedRelativeJointPositionAction(JointPositionAction):
    """Turn one normalized delta into one absolute target per policy step.

    Isaac Lab calls ``apply_actions`` at every physics substep. Computing the
    relative target there would integrate the same policy action repeatedly,
    so the target is frozen in ``process_actions`` instead.
    """

    cfg: RateLimitedRelativeJointPositionActionCfg

    def __init__(self, cfg: RateLimitedRelativeJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if len(cfg.max_joint_velocity) != self.action_dim:
            raise ValueError(
                "max_joint_velocity must contain one value per controlled joint: "
                f"expected {self.action_dim}, got {len(cfg.max_joint_velocity)}"
            )
        self._max_joint_delta = (
            torch.tensor(cfg.max_joint_velocity, device=self.device, dtype=torch.float32) * float(env.step_dt)
        )
        self._joint_limit_margin = float(cfg.joint_limit_margin)
        self._requested_actions = torch.zeros_like(self._processed_actions)
        self._last_applied_actions = torch.zeros_like(self._processed_actions)

    @property
    def requested_actions(self) -> torch.Tensor:
        """Absolute target before the joint-limit filter."""
        return self._requested_actions

    @property
    def last_applied_actions(self) -> torch.Tensor:
        """Exact absolute target most recently passed to the articulation."""
        return self._last_applied_actions

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        normalized_delta = torch.clamp(self._processed_actions, min=-1.0, max=1.0)
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._requested_actions[:] = joint_pos + normalized_delta * self._max_joint_delta

        joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        lower = joint_limits[..., 0] + self._joint_limit_margin
        upper = joint_limits[..., 1] - self._joint_limit_margin
        self._processed_actions[:] = torch.minimum(torch.maximum(self._requested_actions, lower), upper)

    def apply_actions(self):
        self._last_applied_actions[:] = self._processed_actions
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target(torch.zeros_like(self._processed_actions), joint_ids=self._joint_ids)
