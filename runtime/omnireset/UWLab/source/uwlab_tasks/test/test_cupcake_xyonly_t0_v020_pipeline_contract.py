# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts for the CupCake XY-only T0 curriculum-to-Vision-DP pipeline."""

import ast
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "scripts/franka_kl_distill"
WATCHER = SCRIPTS / "run_cupcake_xyonly_t0_v020_to_vision_dp.py"
COLLECTOR = SCRIPTS / "run_cupcake_fr3_xy5_t0_v020_1200x5.py"
DP_PIPELINE = SCRIPTS / "run_cupcake_fr3_xy5_t0_v020_train_eval_2x2.py"
AUDITOR = SCRIPTS / "audit_cupcake_xyonly_selected_resets.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCupCakeXYOnlyT0V020PipelineContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.watcher = load_module("cupcake_xyonly_v020_watcher", WATCHER)

    def test_static_contract_is_exact_xyonly_t0_60_percent_and_1200x5(self):
        contract = self.watcher.validate_static_contract()
        self.assertEqual(self.watcher.GATE, 0.60)
        self.assertEqual(self.watcher.EXPECTED_PREDECESSOR, (0.3, 32.0, 320))
        self.assertEqual(
            self.watcher.EXPECTED_STAGES,
            (
                (0, 0.25, 64.0, 640),
                (1, 0.2, 80.0, 800),
            ),
        )
        predecessor = self.watcher.validate_predecessor_gate_evidence(contract)
        self.assertEqual(predecessor["success_rate"], 0.6067742109298706)
        self.assertEqual(Path(predecessor["checkpoint"]).name, "model_4937.pt")
        self.assertEqual(contract["post_curriculum"]["collection_unique_states"], 1200)
        self.assertEqual(
            contract["post_curriculum"]["collection_successful_variants_per_state"], 5
        )
        self.assertEqual(self.watcher.COLLECTION_EPISODE_LENGTH_S, 80.0)
        self.assertEqual(
            contract["post_curriculum"]["collection_episode_length_s"], 80.0
        )
        self.assertEqual(
            tuple(contract["post_curriculum"]["vision_dp_settings"]),
            self.watcher.VISION_DP_SETTINGS,
        )

    def test_final_gate_requires_strictly_greater_success_and_matching_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_999.pt"
            checkpoint.write_bytes(b"checkpoint")
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            stage = {
                "index": 1,
                "status": "complete",
                "run_dir": str(checkpoint.parent),
                "task0_success_gate": 0.60,
                "joint_max_velocity_rad_s": 0.2,
                "episode_length_s": 80.0,
                "max_episode_steps": 800,
                "output_checkpoint": str(checkpoint),
                "task0_gate_evidence": {
                    "metric": "Metrics/task_0_success_rate",
                    "comparison": ">",
                    "threshold": 0.60,
                    "success_rate": 0.601,
                    "iteration": 999,
                    "checkpoint": str(checkpoint),
                    "checkpoint_size_bytes": checkpoint.stat().st_size,
                    "checkpoint_sha256": digest,
                },
            }
            evidence = self.watcher.validate_gate_evidence(
                stage, self.watcher.EXPECTED_STAGES[-1]
            )
            self.assertEqual(evidence["checkpoint_sha256"], digest)
            stage["task0_gate_evidence"]["success_rate"] = 0.60
            with self.assertRaisesRegex(ValueError, "does not prove T0 > 60%"):
                self.watcher.validate_gate_evidence(
                    stage, self.watcher.EXPECTED_STAGES[-1]
                )

    def test_runtime_log_snapshot_is_scoped_to_latest_attempt(self):
        text = """
launch: RUN_NAME=old
 Learning iteration 10/500
 Metrics/task_0_success_rate: 0.99
launch: RUN_NAME=current
 Learning iteration 20/500
 Episode_Termination/abnormal_robot: 0.004
 Metrics/task_0_success_rate: 0.52
 Metrics/task_0_prob: 1.0
 Metrics/mean_episode_length: 191.0
 Learning iteration 21/500
 Episode_Termination/abnormal_robot: 0.005
 Metrics/task_0_success_rate: 0.56
 Metrics/task_0_prob: 1.0
 Metrics/mean_episode_length: 192.0
"""
        snapshot = self.watcher.parse_curriculum_log_snapshot(text)
        self.assertEqual(snapshot["metric_samples"], 2)
        self.assertEqual(snapshot["latest"]["iteration"], 21)
        self.assertEqual(snapshot["latest"]["task0_success_rate"], 0.56)
        self.assertEqual(snapshot["task0_peak_success_rate"], 0.56)
        self.assertEqual(snapshot["task0_last20_mean_success_rate"], 0.54)
        self.assertFalse(snapshot["task0_gate_observed_in_log"])

    def test_collection_reuses_existing_2500_state_reset_and_fails_closed(self):
        source = COLLECTOR.read_text()
        ast.parse(source, filename=str(COLLECTOR))
        for expected in (
            'RESET_MODE = os.environ.get("RESET_MODE", "build_xy5_homejitter")',
            'elif RESET_MODE != "existing_xy_only":',
            'manifest.get("contract") != "canonical_t0_insertive_xy_only_v1"',
            'manifest.get("untouched_tensor_fields_equal") is not True',
            'CANDIDATE_STATES = int(os.environ.get("CANDIDATE_STATES", "4096"))',
            'SELECTED_STATES = int(os.environ.get("SELECTED_STATES", "1200"))',
            "selector_sr < MIN_SELECTOR_SR",
            "replay_rate + 1.0e-12 < MIN_REPLAY_SR",
        ):
            self.assertIn(expected, source)

        watcher = WATCHER.read_text()
        for expected in (
            '"RESET_MODE": "existing_xy_only"',
            '"CANDIDATE_STATES": "2500"',
            '"SELECTED_STATES": "1200"',
            '"MIN_SELECTOR_SR": f"{GATE:.2f}"',
            '"MIN_REPLAY_SR": "0.95"',
            '"JOINT_EPISODE_LENGTH_S": str(COLLECTION_EPISODE_LENGTH_S)',
            '"GPUS": collection_gpus',
            '"TRAIN_GPUS": dp_gpus',
        ):
            self.assertIn(expected, watcher)

    def test_xyonly_audit_does_not_impose_home_or_plate_randomization(self):
        source = AUDITOR.read_text()
        ast.parse(source, filename=str(AUDITOR))
        self.assertIn('CONTRACT = "canonical_t0_insertive_xy_only_v1"', source)
        self.assertIn("MIN_SELECTED_SPAN_COVERAGE = 0.90", source)
        self.assertIn('"only_cupcake_xy_changed": True', source)
        self.assertNotIn("TEAM_HOME", source)
        self.assertNotIn("selected Plate XY span coverage", source)

    def test_four_dp_groups_receive_new_dataset_and_reset_contracts(self):
        source = DP_PIPELINE.read_text()
        ast.parse(source, filename=str(DP_PIPELINE))
        for expected in (
            'RESET_AUDIT_MODE = os.environ.get("RESET_AUDIT_MODE", "teamhome_xy5")',
            'if RESET_AUDIT_MODE == "canonical_xy_only":',
            '"NORMAL_DATASET": NORMAL_DATASET',
            '"ROI_DATASET": ROI_DATASET',
            '"RESET_TYPE": RESET_TYPE',
            "audit_cupcake_xyonly_selected_resets.py",
        ):
            self.assertIn(expected, source)
        watcher = WATCHER.read_text()
        for setting in ("normal_nomask", "normal_mask", "roi_nomask", "roi_mask"):
            self.assertIn(f'"{setting}"', watcher)
        self.assertIn("normal_sim6000_20260829", watcher)
        self.assertIn("axis_roi_v1_sim6000_20260829", watcher)


if __name__ == "__main__":
    unittest.main()
