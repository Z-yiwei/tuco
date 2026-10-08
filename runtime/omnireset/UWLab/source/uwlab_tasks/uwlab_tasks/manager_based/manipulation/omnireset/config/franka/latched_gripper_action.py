"""Policy-triggered, one-way gripper closure, reset independently per env."""

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions import BinaryJointPositionAction, BinaryJointPositionActionCfg
from isaaclab.utils import configclass


class LatchedBinaryGripperAction(BinaryJointPositionAction):
    """A negative policy action closes once; later open requests cannot reopen.

    This term does not inspect object geometry or run a grasp guard. Optional
    pre-grasp initialization uses the reset path, not a distance threshold.
    Raw policy actions are retained for PPO and legacy observation compatibility.
    """

    cfg: "LatchedBinaryGripperActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._latched_closed = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)
        self._last_applied_actions = torch.zeros_like(self._processed_actions)
        self._processed_actions[:] = self._open_command

    @property
    def latched_closed(self):
        return self._latched_closed

    @property
    def last_applied_actions(self):
        """Actual finger targets from the last physics tick; survive auto-reset."""
        return self._last_applied_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        close_requested = actions == 0 if actions.dtype == torch.bool else actions < 0
        self._latched_closed |= close_requested
        self._processed_actions[:] = torch.where(
            self._latched_closed, self._close_command, self._open_command
        )
        if self.cfg.clip is not None:
            self._processed_actions[:] = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        self._last_applied_actions[:] = self._processed_actions
        super().apply_actions()

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(env_ids)
        self._latched_closed[ids] = False
        if self.cfg.close_grasped_resets:
            term = self._env.event_manager.get_term_cfg(self.cfg.reset_event_name)
            names = term.params["reset_types"]
            selected_ids = term.func.task_id[ids]
            known = set(self.cfg.grasped_reset_types) | set(self.cfg.open_reset_types)
            if any(name not in known for name in names):
                raise ValueError(f"Unknown reset path for latched gripper: {names}")
            closed_by_path = torch.tensor(
                [name in self.cfg.grasped_reset_types for name in names],
                device=self.device, dtype=torch.bool,
            )
            self._latched_closed[ids] = closed_by_path[selected_ids].unsqueeze(-1)
        self._processed_actions[ids] = torch.where(
            self._latched_closed[ids], self._close_command, self._open_command
        )


@configclass
class LatchedBinaryGripperActionCfg(BinaryJointPositionActionCfg):
    class_type: type = LatchedBinaryGripperAction
    close_grasped_resets: bool = True
    reset_event_name: str = "reset_from_reset_states"
    grasped_reset_types: tuple[str, ...] = (
        "ObjectRestingEEGrasped",
        "ObjectAnywhereEEGrasped",
        "ObjectPartiallyAssembledEEGrasped",
    )
    open_reset_types: tuple[str, ...] = ("ObjectAnywhereEEAnywhere",)


def previous_actions_with_gripper_latch(env):
    """Expose executed gripper state in the existing previous-action slot."""
    actions = env.action_manager.action.clone()
    grip = env.action_manager.get_term("gripper")
    if not isinstance(grip, LatchedBinaryGripperAction):
        raise TypeError("Executed-latch observation requires a latched gripper")
    actions[:, -1] = torch.where(grip.latched_closed[:, 0], -1.0, 1.0)
    return actions
