# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contracts for the CupCake FR3 XY5 v0.2 four-group Vision DP pipeline."""

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "scripts/franka_kl_distill"
PREPARE = SCRIPTS / "prepare_cupcake_fr3_xy5_t0_v020_1200x5_datasets.sh"
TRAIN = SCRIPTS / "train_cupcake_fr3_xy5_t0_v020_2x2.sh"
EVAL = SCRIPTS / "eval_cupcake_fr3_xy5_t0_v020_dp.sh"
PAIRED = SCRIPTS / "eval_cupcake_fr3_xy5_t0_v020_dp_paired.sh"
ORCHESTRATOR = SCRIPTS / "run_cupcake_fr3_xy5_t0_v020_train_eval_2x2.py"
PROMOTION = SCRIPTS / "watch_cupcake_v020_to_demos.py"
CONFIG = REPO_ROOT / (
    "cupid/configs/image/"
    "omnireset_cupcake_fr3_xy5_t0_v020_3cam_jointtarget/config.yaml"
)


class TestCupCakeFR3XY5V020VisionPipelineContract(unittest.TestCase):
    def test_normal_and_roi_are_derived_from_the_same_frozen_zarr(self):
        source = PREPARE.read_text()
        self.assertIn("cupcake_fr3_xy5_t0_stochastic_v020", source)
        self.assertIn("wrist3cm15deg_whiteplate_blackpads_1200x5.zarr", source)
        self.assertIn("normal_sim6000", source)
        self.assertIn("axis_roi_v1_sim6000", source)
        self.assertIn("--image_size 84", source)
        self.assertIn("--expected_episodes 6000", source)

    def test_four_training_groups_share_the_delta_joint_contract(self):
        source = TRAIN.read_text()
        config = CONFIG.read_text()
        for setting in ("normal_nomask", "normal_mask", "roi_nomask", "roi_mask"):
            self.assertIn(setting, source)
        for expected in (
            "MASK_MODE=none",
            "MASK_MODE=one_or_none_uniform",
            "joint_action_representation=delta_joint_step_v1",
            "joint_control_dt_s=0.1",
            "joint_max_velocity_rad_s=0.2",
            'NUM_EPOCHS="${NUM_EPOCHS:-51}"',
            'MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3022}"',
            "policy.obs_encoder.share_rgb_model=false",
        ):
            self.assertIn(expected, source)
        self.assertIn("front_rgb: {shape: [3, 84, 84], type: rgb}", config)
        self.assertIn("proprio: {shape: [8], type: low_dim}", config)
        self.assertIn("action: {shape: [8]}", config)
        self.assertIn("horizon: 16", config)
        self.assertIn("n_obs_steps: 2", config)
        self.assertIn("n_action_steps: 8", config)

    def test_eval_is_cupcake_fr3_whiteplate_v020_and_no_flip(self):
        source = EVAL.read_text()
        for expected in (
            "OmniReset-FrankaResearch3MimicFingertip-AbsoluteJointTarget-RGB-Play-v0",
            "cupcake_xy5_t0_candidate4096_homejitter02_fr3_20260828",
            "axis_3cam_side16x9_wristrear_handy_mirror_20260826",
            "OMNIRESET_EXACT_WRIST_VERTICAL_FLIP=0",
            "white_arm_base_ring_black_finger_pads",
            "--joint_max_velocity 0.2",
            "--joint_episode_length_s 20",
            "--success_position_override_m 0.020",
            "--success_orientation_override_deg 3.0",
            "--b3_nominal_fixed_scene",
            "--preserve_object_face_materials",
            "env.scene.insertive_object=cupcake",
            "env.scene.receptive_object=plate",
            "env.scene.robot.actuators.panda_hand.stiffness=5000.0",
        ):
            self.assertIn(expected, source)
        self.assertNotIn("--stackcube_fixed_scene", source)
        self.assertNotIn("--rotate_180_camera", source)

    def test_exact_e45_selection_uses_matched_fixed_and_wild_pairs(self):
        paired = PAIRED.read_text()
        orchestrator = ORCHESTRATOR.read_text()
        ast.parse(orchestrator, filename=str(ORCHESTRATOR))
        self.assertIn('POLICY_SAMPLING_SEED="${POLICY_SAMPLING_SEED:-4242}"', paired)
        self.assertIn("EVAL_EPOCH = 45", orchestrator)
        self.assertIn("TRAIN_EPOCH = 50", orchestrator)
        self.assertIn("POLICY_SAMPLING_SEED = 4242", orchestrator)
        self.assertIn('for camera_mode in ("fixed", "wild")', orchestrator)
        self.assertIn('"NUM_EPISODES": "64"', orchestrator)
        self.assertIn('"selection_key": ["fixed_plus_wild", "wild", "fixed"]', orchestrator)
        self.assertIn('os.environ.get("TRAIN_GPUS", "0,2,6,7")', orchestrator)
        self.assertIn('"--expected-episodes",', orchestrator)

    def test_promotion_watcher_runs_vision_dp_only_after_demo_pipeline(self):
        source = PROMOTION.read_text()
        ast.parse(source, filename=str(PROMOTION))
        collect = source.index('status["phase"] = "running_fr3_xy5_demo_pipeline"')
        dp = source.index('status["phase"] = "running_vision_dp_normal_roi_mask_2x2"')
        self.assertLess(collect, dp)
        self.assertIn("run_cupcake_fr3_xy5_t0_v020_train_eval_2x2.py", source)
        self.assertIn('parser.add_argument("--dp_gpus", default="0,2,6,7")', source)
        self.assertIn("resuming_after_teacher_promotion", source)


if __name__ == "__main__":
    unittest.main()
