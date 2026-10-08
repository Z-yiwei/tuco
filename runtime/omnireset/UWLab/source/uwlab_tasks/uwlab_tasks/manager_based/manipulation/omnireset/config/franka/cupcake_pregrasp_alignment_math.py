"""Bounded open-jaw alignment progress; pure tensor logic, no simulator writes.

Quality is a geometric guide, not a contact/grasp certificate. The target is the
existing 10--20 mm paper-cup pad-height band. Yaw about the cupcake is free.
Only new episode-best quality earns reward; stationary poses and reopening do
not replenish it. Values returned by the tracker are ACTUAL per-step returns.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AlignmentParameters:
    target_height_m: float = .015
    lateral_std_m: float = .020
    height_std_m: float = .008
    pad_level_std_m: float = .005
    open_half_width_m: float = .040
    coarse_std_m: float = .060
    near_full_m: float = .080
    near_zero_m: float = .120
    coarse_fraction: float = .20
    eligible_width_m: float = .060
    episode_budget: float = .50
    step_cap: float = .050

    def __post_init__(self):
        if not all(math.isfinite(x) for x in vars(self).values()):
            raise ValueError('alignment parameters must be finite')
        positive = ('lateral_std_m', 'height_std_m', 'pad_level_std_m',
                    'open_half_width_m', 'coarse_std_m', 'episode_budget', 'step_cap')
        if any(getattr(self, k) <= 0 for k in positive):
            raise ValueError('alignment scales/budgets must be positive')
        if not (0 <= self.near_full_m < self.near_zero_m and
                0 <= self.coarse_fraction < 1 and
                0 < self.eligible_width_m <= 2*self.open_half_width_m and
                0 < self.step_cap <= self.episode_budget):
            raise ValueError('invalid alignment support/width/budget')


def alignment_quality(tcp_in_object, jaw_axis_in_object, params):
    """Unit-range score of TCP centering, low pad height and horizontal jaw axis.

    The open-width reference is FIXED: changing finger width cannot improve the
    quality. jaw_axis is the actual unit hand-Y axis expressed in object frame.
    """
    if tcp_in_object.ndim != 2 or tcp_in_object.shape[-1] != 3:
        raise ValueError('expected [N,3] TCP coordinates')
    if jaw_axis_in_object.shape != tcp_in_object.shape:
        raise ValueError('jaw axis must match TCP coordinates')
    valid = torch.isfinite(tcp_in_object).all(-1) & torch.isfinite(jaw_axis_in_object).all(-1)
    valid &= (jaw_axis_in_object.norm(dim=-1)-1).abs() < 1e-3
    tcp = torch.where(valid[:, None], tcp_in_object, torch.zeros_like(tcp_in_object))
    jaw = torch.where(valid[:, None], jaw_axis_in_object, torch.zeros_like(jaw_axis_in_object))
    lateral2 = tcp[:, :2].square().sum(-1)
    dz = tcp[:, 2]-params.target_height_m
    distance = (lateral2+dz.square()).sqrt()
    gate = ((params.near_zero_m-distance)/(params.near_zero_m-params.near_full_m)).clamp(0, 1)
    gate = gate.square()*(3-2*gate)
    coarse = torch.exp(-.5*(distance/params.coarse_std_m).square())
    fine = torch.exp(-.5*(lateral2/params.lateral_std_m**2 +
                          (dz/params.height_std_m).square() +
                          (params.open_half_width_m*jaw[:, 2]/params.pad_level_std_m).square()))
    score = gate*(params.coarse_fraction*coarse+(1-params.coarse_fraction)*fine)
    return torch.where(valid, score.clamp(0, 1), torch.zeros_like(score)), valid


class AlignmentProgressTracker:
    def __init__(self, num_envs, device='cpu', params=None):
        self.params = AlignmentParameters() if params is None else params
        self.best = torch.zeros(num_envs, device=device)
        self.paid = torch.zeros_like(self.best)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.retired = torch.zeros_like(self.initialized)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.best[ids] = 0
        self.paid[ids] = 0
        self.initialized[ids] = False
        self.retired[ids] = False

    def step(self, score, valid, measured_width, captured, task_success):
        for value in (score, valid, measured_width, captured, task_success):
            if value.shape != self.best.shape:
                raise ValueError('tracker inputs must have shape [num_envs]')
        valid = valid & torch.isfinite(score) & torch.isfinite(measured_width)
        valid &= (score >= 0) & (score <= 1) & (measured_width >= 0)
        safe_score = torch.where(valid, score, self.best)
        # No reward just for spawning in a good pose. Include CLOSED observations
        # in the high-water mark so reopening cannot claim already-seen quality.
        delta = (safe_score-self.best).clamp_min(0)
        self.retired |= captured.bool() | task_success.bool()
        eligible = valid & self.initialized & ~self.retired
        eligible &= measured_width >= self.params.eligible_width_m
        reward = (self.params.episode_budget*delta).clamp(max=self.params.step_cap)
        reward = torch.minimum(reward, (self.params.episode_budget-self.paid).clamp_min(0))
        reward = torch.where(eligible, reward, torch.zeros_like(reward))
        self.best = torch.maximum(self.best, safe_score)
        self.initialized |= valid
        self.paid += reward
        # Clipped progress is NOT queued: holding still never earns later payout.
        return reward
