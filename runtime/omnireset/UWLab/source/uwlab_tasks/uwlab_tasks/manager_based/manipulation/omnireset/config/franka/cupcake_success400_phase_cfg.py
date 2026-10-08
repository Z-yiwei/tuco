"""Success400 replacement phase curriculum; identical grasp-repair plant/controller."""
import copy
import json
import os
from pathlib import Path

from isaaclab.utils import configclass
from .cupcake_half_guarded_gripper_cfg import CupCakeHalfGuardedTrainCfg
from .cupcake_grasp_repair_cfg import CupCakeGraspRepairEvents
from .cupcake_friction_compensated_action import CupCakeFrictionCompensatedDiffIKActionCfg
from .cupcake_hysteretic_guard import CupCakeHystereticGuardCfg
from .cupcake_captured_lift_reward import CupCakeCapturedLiftBonus
from .cupcake_success400_phase_reset import (
    CupCakeSuccess400PhaseReset, verify_phase_startup, verify_phase_after_reset,
)


@configclass
class CupCakeSuccess400PhaseTrainCfg(CupCakeHalfGuardedTrainCfg):
    events: CupCakeGraspRepairEvents = CupCakeGraspRepairEvents()

    def __post_init__(self):
        super().__post_init__()
        contract = json.loads((Path(os.environ['CUPCAKE_RESET_DATASET_ROOT']) / 'training_contract.json').read_text())
        assert contract['version'] == 'cupcake_success400_phase_contract_v1' and contract['gates_passed'] is True
        new = CupCakeFrictionCompensatedDiffIKActionCfg()
        for key, value in vars(self.actions.arm).items():
            if key != 'class_type':
                setattr(new, key, copy.deepcopy(value))
        new.friction_after_capture_only = True
        new.friction_capture_ramp_s = .5
        self.actions.arm = new
        grip = CupCakeHystereticGuardCfg()
        for key, value in vars(self.actions.gripper).items():
            if key != 'class_type':
                setattr(grip, key, copy.deepcopy(value))
        grip.release_margin_m = .002
        grip.capture_confirm_steps = 5
        self.actions.gripper = grip
        self.rewards.grasped_and_lifted.func = CupCakeCapturedLiftBonus
        self.rewards.grasped_and_lifted.params = {'hold_steps': 20, 'release_margin_m': .002}
        self.events.reset_from_reset_states.func = CupCakeSuccess400PhaseReset
        self.events.policy_gripper_contract.func = verify_phase_startup
        self.events.repair_physics_contract.func = verify_phase_after_reset

