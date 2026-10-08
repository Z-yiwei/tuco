"""Candidate: binary guard intent, separately rate-limited physical finger targets.

No object pose/velocity writes. Do not attach the historical binary-only executed
action recorder: actual finger targets are continuous and must be saved as such.
"""
import torch
from isaaclab.utils import configclass
from .cupcake_hysteretic_guard import CupCakeHystereticGuard, CupCakeHystereticGuardCfg


def closing_target(previous, desired, max_step):
    return torch.maximum(desired, previous - max_step)


class CupCakeRateLimitedGuard(CupCakeHystereticGuard):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        assert 0 < cfg.closing_width_speed_m_s <= .1
        self._finger_step = cfg.closing_width_speed_m_s * env.sim.get_physics_dt() / 2
        self._applied_target = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._was_closing = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    @property
    def applied_position_targets(self):
        return self._applied_target

    def apply_actions(self):
        closing = self._raw_actions[:, 0] < 0
        begin = closing & ~self._was_closing
        measured = self._asset.data.joint_pos[:, self._joint_ids]
        # New close attempts start at measured aperture, not a stale zero target.
        self._applied_target[begin] = torch.minimum(
            self._open_command, torch.maximum(self._close_command, measured))[begin]
        ramped = closing_target(self._applied_target, self._processed_actions, self._finger_step)
        self._applied_target[:] = torch.where(closing[:, None], ramped, self._processed_actions)
        self._asset.set_joint_position_target(self._applied_target, joint_ids=self._joint_ids)
        self._was_closing[:] = closing

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, '_was_closing'):
            index = slice(None) if env_ids is None else env_ids
            self._was_closing[index] = False
            self._applied_target[index] = self._asset.data.joint_pos[:, self._joint_ids][index]


@configclass
class CupCakeRateLimitedGuardCfg(CupCakeHystereticGuardCfg):
    class_type: type = CupCakeRateLimitedGuard
    closing_width_speed_m_s: float = .05
