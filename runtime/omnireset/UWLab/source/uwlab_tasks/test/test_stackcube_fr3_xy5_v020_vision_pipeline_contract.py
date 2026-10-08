# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contract tests for the FR3 XY5 t0 1200x5 Vision DP pipeline."""

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "scripts/franka_kl_distill"
COLLECT = SCRIPTS / "collect_stackcube_fr3_xy5_t0_indexed_1200x5.sh"
COLLECT_ORCHESTRATOR = (
    SCRIPTS / "run_stackcube_fr3_xy5_t0_v020_wrist3cm15deg_1200x5.py"
)
TRAIN = SCRIPTS / "train_stackcube_fr3_xy5_t0_v020_2x2.sh"
EVAL = SCRIPTS / "eval_stackcube_fr3_xy5_t0_v020_dp.sh"
TRAIN_EVAL = SCRIPTS / "run_stackcube_fr3_xy5_t0_v020_train_eval_2x2.py"


class TestStackCubeFR3XY5V020VisionPipelineContract(unittest.TestCase):
    def test_collection_contract_is_stochastic_v020_research3(self):
        source = COLLECT.read_text()
        self.assertIn(
            "OmniReset-FrankaResearch3MimicFingertip-RelCartesianOSC-RGB-Play-v0",
            source,
        )
        self.assertIn("--stochastic_teacher_actions", source)
        self.assertIn("--joint_max_velocity 0.20", source)
        self.assertIn("--joint_episode_length_s 20", source)
        self.assertIn("--successful_repeats_per_state", source)
        self.assertIn("white_arm_base_ring_black_finger_pads", source)
        self.assertIn("OMNIRESET_EXACT_WRIST_VERTICAL_FLIP=0", source)

    def test_collection_orchestrator_freezes_1200x5_and_camera_ranges(self):
        source = COLLECT_ORCHESTRATOR.read_text()
        ast.parse(source, filename=str(COLLECT_ORCHESTRATOR))
        for default in (
            'os.environ.get("SELECTED_STATES", "1200")',
            'os.environ.get("REPEATS", "5")',
            'os.environ.get("ARM_JOINT_OFFSET_RAD", "0.20")',
            'os.environ.get("FRONT_CAMERA_POSITION_JITTER_M", "0.020")',
            'os.environ.get("SIDE_CAMERA_POSITION_JITTER_M", "0.020")',
            'os.environ.get("WRIST_CAMERA_POSITION_JITTER_M", "0.030")',
            'os.environ.get("FRONT_CAMERA_ROTATION_JITTER_DEG", "10.0")',
            'os.environ.get("SIDE_CAMERA_ROTATION_JITTER_DEG", "10.0")',
            'os.environ.get("WRIST_CAMERA_ROTATION_JITTER_DEG", "15.0")',
        ):
            self.assertIn(default, source)
        self.assertIn('os.environ.get("GPUS", "0,1,2,4")', source)
        self.assertIn('f"collecting_visual_{SELECTED_STATES}x{REPEATS}"', source)

    def test_training_matrix_changes_only_roi_and_camera_mask(self):
        source = TRAIN.read_text()
        for setting in ("normal_nomask", "normal_mask", "roi_nomask", "roi_mask"):
            self.assertIn(setting, source)
        self.assertIn("MASK_MODE=none", source)
        self.assertIn("MASK_MODE=one_or_none_uniform", source)
        self.assertIn("joint_action_representation=delta_joint_step_v1", source)
        self.assertIn("joint_max_velocity_rad_s=0.2", source)
        self.assertIn('NUM_EPOCHS="${NUM_EPOCHS:-51}"', source)
        self.assertIn("policy.obs_encoder.share_rgb_model=false", source)

    def test_eval_contract_matches_collection_and_supports_both_preprocessors(self):
        source = EVAL.read_text()
        self.assertIn(
            "OmniReset-FrankaResearch3MimicFingertip-AbsoluteJointTarget-RGB-Play-v0",
            source,
        )
        self.assertIn("--joint_max_velocity 0.2", source)
        self.assertIn("OMNIRESET_REAL_WRIST_CAMERA_POSITION_JITTER_M=0.030", source)
        self.assertIn("OMNIRESET_REAL_WRIST_CAMERA_ROTATION_JITTER_DEG=15.0", source)
        self.assertIn("OMNIRESET_EXACT_WRIST_VERTICAL_FLIP=0", source)
        self.assertIn("--image_preprocess_contract axis_roi_v1_20260825", source)
        self.assertIn("--stackcube_fixed_scene", source)

    def test_watcher_requires_all_four_epoch50_checkpoints_and_both_evals(self):
        source = TRAIN_EVAL.read_text()
        ast.parse(source, filename=str(TRAIN_EVAL))
        for setting in ("normal_nomask", "normal_mask", "roi_nomask", "roi_mask"):
            self.assertIn(f'("{setting}"', source)
        self.assertIn('"epoch=0050-val_loss=*.ckpt"', source)
        self.assertIn('for camera_mode in ("fixed", "wild")', source)
        self.assertIn('"NUM_EPISODES": "64"', source)


if __name__ == "__main__":
    unittest.main()
