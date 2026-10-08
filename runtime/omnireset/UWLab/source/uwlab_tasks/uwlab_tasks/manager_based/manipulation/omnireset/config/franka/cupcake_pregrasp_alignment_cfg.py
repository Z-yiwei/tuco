"""Versioned reward-only successor; the sealed Success400 task is unchanged."""
from dataclasses import asdict

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass

from .cupcake_success400_phase_cfg import CupCakeSuccess400PhaseTrainCfg
from .cupcake_pregrasp_alignment_math import AlignmentParameters
from .cupcake_pregrasp_alignment_reward import CupCakePregraspAlignmentProgress, verify_alignment_startup


@configclass
class CupCakePregraspAlignmentTrainCfg(CupCakeSuccess400PhaseTrainCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.pregrasp_alignment_progress = RewardTermCfg(
            func=CupCakePregraspAlignmentProgress, weight=1., params=asdict(AlignmentParameters()))
        self.events.policy_gripper_contract.func = verify_alignment_startup
