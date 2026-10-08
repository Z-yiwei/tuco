# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Rule-based gripper that closes only when the peg is within a capture zone near the TCP.

The policy still emits a gripper action, but it is *ignored*. The gripper is instead controlled
by a deterministic rule that closes when the peg root is within (lateral_thresh, vertical_thresh)
of the gripper TCP. This sidesteps the open-handed grasp exploration problem in RL — the policy
only has to position the arm correctly, and the rule handles the gripper open/close decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GraspGuardedBinaryGripperAction(BinaryJointPositionAction):
    """Pure rule-based binary gripper: close when peg is in capture zone of TCP, else open.

    Policy gripper action is ignored entirely. Simpler than versions that switch control
    to the policy when lifted — that introduced new failure modes (policy dropping peg
    during transport). The rule's "open when out of zone" naturally handles release:
    when peg is in hole and TCP moves away, peg falls out of zone → open → release.
    """

    cfg: "GraspGuardedBinaryGripperActionCfg"

    def __init__(self, cfg: "GraspGuardedBinaryGripperActionCfg", env: "ManagerBasedEnv") -> None:
        super().__init__(cfg, env)
        body_ids, body_names = self._asset.find_bodies(cfg.ee_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"GraspGuardedBinaryGripperAction: expected exactly 1 body matching "
                f"'{cfg.ee_body_name}', got {len(body_ids)}: {body_names}"
            )
        self._ee_body_idx = body_ids[0]
        self._tcp_offset = torch.tensor(cfg.tcp_offset, device=self.device).view(1, 3)
        self._peg = env.scene[cfg.peg_asset_name]
        self._lateral_thresh = float(cfg.lateral_thresh_xy)
        self._vertical_thresh = float(cfg.vertical_thresh_z)
        self._hold_processed_action = False

    def set_hold_processed_action(self, enabled: bool) -> None:
        """Freeze/unfreeze the current binary finger-position command."""
        self._hold_processed_action = bool(enabled)

    def compute_rule_actions(self) -> torch.Tensor:
        """Return the binary command selected by the grasp guard for the current state."""
        # TCP world pose
        hand_pos = self._asset.data.body_link_pos_w[:, self._ee_body_idx]
        hand_quat = self._asset.data.body_link_quat_w[:, self._ee_body_idx]
        tcp_offset_w = math_utils.quat_apply(hand_quat, self._tcp_offset.expand_as(hand_pos))
        tcp_pos = hand_pos + tcp_offset_w

        # Peg in TCP local frame
        peg_pos = self._peg.data.root_pos_w
        diff_w = peg_pos - tcp_pos
        diff_local = math_utils.quat_apply(math_utils.quat_inv(hand_quat), diff_w)
        lateral = torch.norm(diff_local[:, :2], dim=-1)
        vertical = diff_local[:, 2].abs()
        in_zone = (lateral < self._lateral_thresh) & (vertical < self._vertical_thresh)
        return torch.where(in_zone, -1.0, 1.0).unsqueeze(-1)

    def process_actions(self, actions: torch.Tensor):
        if self._hold_processed_action:
            return
        # Pure rule: close iff peg in capture zone
        self._raw_actions[:] = self.compute_rule_actions()
        in_zone = self._raw_actions < 0.0
        self._processed_actions = torch.where(
            in_zone, self._close_command, self._open_command
        )
        if self.cfg.clip is not None and hasattr(self, "_clip"):
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class GraspGuardedBinaryGripperActionCfg(BinaryJointPositionActionCfg):
    class_type: type = GraspGuardedBinaryGripperAction
    ee_body_name: str = "panda_hand"
    peg_asset_name: str = "insertive_object"
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034)
    # Capture zone half-extents (peg root must be inside this box around TCP for "close")
    lateral_thresh_xy: float = 0.02  # 2 cm lateral (peg within ±2cm of TCP in finger-perpendicular plane)
    vertical_thresh_z: float = 0.03  # 3 cm along finger-length axis (TCP within ±3cm of peg height)
    # (removed lifted_threshold — pure rule now, no policy takeover)
