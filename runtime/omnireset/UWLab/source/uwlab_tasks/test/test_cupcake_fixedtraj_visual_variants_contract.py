#!/usr/bin/env python3
"""Static contract checks for fixed-trajectory CupCake visual augmentation."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "scripts/franka_kl_distill"
RERENDER = SCRIPTS / "run_cupcake_fr3_fixedtraj_state_rerender_1200x5.py"
VALIDATOR = SCRIPTS / "validate_fixed_trajectory_visual_variants.py"
DP_WRAPPER = SCRIPTS / "run_cupcake_fr3_fixedtraj_1200x5_to_dp_mask_nomask.sh"
DP_ORCHESTRATOR = SCRIPTS / "run_cupcake_fr3_xy5_t0_v025_train_eval_2x2.py"
COLLECTOR = SCRIPTS / "collect_vision_kl.py"
COLLECT_WRAPPER = SCRIPTS / "collect_cupcake_fr3_xy5_t0_v025_indexed_1200x5.sh"
EVALUATOR = ROOT / "scripts/eval_dp_cupid_image_random_fs.py"
EVAL_WRAPPER = SCRIPTS / "eval_cupcake_fr3_xy5_t0_v025_dp.sh"
RECOVERY = SCRIPTS / "recover_resume_cupcake_fr3_fixedtraj_1200x5.py"
RECOVERY_WRAPPER = SCRIPTS / "resume_cupcake_fr3_fixedtraj_1200x5_then_dp.sh"


class CupCakeFixedTrajectoryVisualVariantsContractTest(unittest.TestCase):
    def test_rerender_is_fixed_source_state_sequence_x5(self):
        source = RERENDER.read_text(encoding="utf-8")
        self.assertIn('SOURCE_DEMOS = 1200', source)
        self.assertIn('VARIANTS_PER_DEMO = 5', source)
        self.assertIn('"--use_source_trajectory_fields"', source)
        self.assertIn('"--render_source_state_sequence"', source)
        self.assertIn('"trajectory_render_mode": "source_state_sequence"', source)
        self.assertIn('"--preflight"', source)

    def test_camera_and_asset_contract_is_explicit(self):
        source = RERENDER.read_text(encoding="utf-8")
        self.assertIn("axis_3cam_side16x9_wristrear_handy_mirror_20260826", source)
        self.assertIn('"OMNIRESET_REAL_CAMERA_POSITION_JITTER_M": "0.020"', source)
        self.assertIn('"OMNIRESET_REAL_SIDE_CAMERA_POSITION_JITTER_M": "0.020"', source)
        self.assertIn('"OMNIRESET_REAL_WRIST_CAMERA_POSITION_JITTER_M": "0.030"', source)
        self.assertIn('"OMNIRESET_REAL_WRIST_CAMERA_ROTATION_JITTER_DEG": "15.0"', source)
        self.assertIn('"OMNIRESET_EXACT_WRIST_VERTICAL_FLIP": "0"', source)
        self.assertIn("white_arm_base_ring_black_finger_pads", source)
        self.assertIn('str(2.0 / 3.0)', source)
        self.assertIn('"--receptive_object_uniform_scale"', source)
        self.assertIn('"plate_scale_semantics"', source)

    def test_validator_checks_source_labels_and_visual_diversity(self):
        source = VALIDATOR.read_text(encoding="utf-8")
        for token in (
            "np.array_equal(action, expected_action)",
            "np.array_equal(proprio, expected_proprio)",
            "three_camera_variation_verified",
            "static across sampled frames",
            "expected_receptive_object_uniform_scale",
            "receptive_object_spawn_scale_xyz",
            "expected_wrist_vertical_flip",
            "wrist_vertical_flips",
        ):
            self.assertIn(token, source)

    def test_dp_wrapper_launches_only_mask_and_no_mask(self):
        source = DP_WRAPPER.read_text(encoding="utf-8")
        self.assertIn('TRAIN_SETTINGS="normal_nomask,normal_mask"', source)
        self.assertIn('TRAIN_GPUS="${TRAIN_GPUS:-0,7}"', source)
        self.assertIn("upright_dualxy5_plate2over3_t0_fixedtraj", source)
        self.assertIn('PLATE_UNIFORM_SCALE="${PLATE_UNIFORM_SCALE:-0.6666666666666666}"', source)
        self.assertIn('RESET_AUDIT_MODE="teamhome_xy5"', source)

    def test_dp_orchestrator_rejects_mislabeled_wrist_flip(self):
        source = DP_ORCHESTRATOR.read_text(encoding="utf-8")
        self.assertIn('wrist_processing.get("vertical_flip") is not False', source)
        self.assertIn("no-flip training contract requires", source)

    def test_plate_scale_is_applied_after_task_variant_resolution(self):
        collector = COLLECTOR.read_text(encoding="utf-8")
        evaluator = EVALUATOR.read_text(encoding="utf-8")
        collect_wrapper = COLLECT_WRAPPER.read_text(encoding="utf-8")
        eval_wrapper = EVAL_WRAPPER.read_text(encoding="utf-8")

        for source in (collector, evaluator):
            self.assertIn('"--receptive_object_uniform_scale"', source)
            self.assertIn(
                "env_cfg.scene.receptive_object.spawn.scale = (receptive_object_scale,) * 3",
                source,
            )
            self.assertIn("explicitly resolved after", source)
        for source in (collect_wrapper, eval_wrapper):
            self.assertIn(
                '--receptive_object_uniform_scale "${PLATE_UNIFORM_SCALE}"',
                source,
            )
            self.assertNotIn("env.scene.receptive_object.spawn.scale=", source)

    def test_recovery_separates_read_only_history_from_local_writes(self):
        recovery = RECOVERY.read_text(encoding="utf-8")
        wrapper = RECOVERY_WRAPPER.read_text(encoding="utf-8")
        self.assertIn('REFERENCE_ROOT = Path(os.environ.get("REFERENCE_ROOT"', recovery)
        self.assertIn("inspect_recovered_original_shard", recovery)
        self.assertIn("inspect_committed_part", recovery)
        self.assertIn('REFERENCE_RECOVERY_MAX_PART_INDEX="${REFERENCE_RECOVERY_MAX_PART_INDEX:-3}"', wrapper)
        self.assertIn('COLLECTION_ROOT="${COLLECTION_ROOT:-/data2/${DATASET_REL}}"', wrapper)
        self.assertIn(".data03_recovery_ro", wrapper)


if __name__ == "__main__":
    unittest.main()
