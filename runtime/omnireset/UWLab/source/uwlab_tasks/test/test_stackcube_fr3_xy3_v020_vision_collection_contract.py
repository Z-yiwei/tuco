# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contract tests for the FR3 XY3 t0 1200x5 collection."""

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "scripts/franka_kl_distill"
WRAPPER = SCRIPTS / "run_stackcube_fr3_xy3_t0_v020_sidewrist1cm5deg_1200x5.sh"
ORCHESTRATOR = SCRIPTS / "run_stackcube_fr3_xy5_t0_v020_wrist3cm15deg_1200x5.py"
COLLECTOR = SCRIPTS / "collect_stackcube_fr3_xy5_t0_indexed_1200x5.sh"
VISION_COLLECTOR = SCRIPTS / "collect_vision_kl.py"
VALIDATOR = SCRIPTS / "validate_jointtarget_dataset.py"
PROMOTER = SCRIPTS / "watch_stackcube_fr3_xy3_t0_v020_1200x5_pilot_then_collect.sh"
FAST84_PIPELINE = SCRIPTS / "run_stackcube_fr3_policy_fast84_1200x5_to_dp.py"


class TestStackCubeFR3XY3V020VisionCollectionContract(unittest.TestCase):
    def test_wrapper_freezes_requested_quota_geometry_and_camera_ranges(self):
        source = WRAPPER.read_text()
        for contract in (
            'GPUS:-0,2,6',
            'NUM_SHARDS:-3',
            'CANDIDATE_STATES:-4096',
            'SELECTED_STATES:-1200',
            'REPEATS:-5',
            'EXPECTED_DEMOS:-6000',
            'RESET_XY_SIZE_M:-0.03',
            'ARM_JOINT_OFFSET_RAD:-0.20',
            'FRONT_CAMERA_POSITION_JITTER_M:-0.020',
            'FRONT_CAMERA_ROTATION_JITTER_DEG:-10.0',
            'SIDE_CAMERA_POSITION_JITTER_M:-0.010',
            'SIDE_CAMERA_ROTATION_JITTER_DEG:-5.0',
            'WRIST_CAMERA_POSITION_JITTER_M:-0.010',
            'WRIST_CAMERA_ROTATION_JITTER_DEG:-5.0',
            'ZARR_COMPRESSOR:-default',
            'FINAL_LAYOUT:-merged',
            'PILOT_STATES:-12',
            'PILOT_REPEATS:-2',
        ):
            self.assertIn(contract, source)

    def test_collection_preserves_prior_control_asset_and_image_contract(self):
        source = COLLECTOR.read_text()
        for contract in (
            "FrankaResearch3MimicFingertip-RelCartesianOSC-RGB-Play-v0",
            "--joint_target_bridge",
            "--stochastic_teacher_actions",
            "--joint_max_velocity 0.20",
            "--joint_episode_length_s 20",
            "--image_size 224",
            "OMNIRESET_EXACT_CAMERA_INTRINSICS=1",
            "OMNIRESET_EXACT_WRIST_VERTICAL_FLIP=0",
            "white_arm_base_ring_black_finger_pads",
            "--zarr_compressor",
        ):
            self.assertIn(contract, source)

    def test_orchestrator_supports_pilot_waves_disk_gate_and_sharded_manifest(self):
        source = ORCHESTRATOR.read_text()
        ast.parse(source, filename=str(ORCHESTRATOR))
        for contract in (
            "for wave_start in range(0, len(pending), len(GPUS))",
            '"pilot_complete"',
            "shutil.disk_usage(OUTPUT_ROOT).free",
            '"sharded_zarr_v1"',
            '"dataset_manifest.json"',
            '"--expected_zarr_compressor"',
            '"budget_bytes_per_demo_with_25pct_margin"',
            '"merging_final_zarr"',
            '"validating_final_zarr"',
            'merge_write_verification=True',
            'CAMERA_RENDER_PROFILE = os.environ.get("CAMERA_RENDER_PROFILE", "native")',
            'IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))',
            'SELECTED_STATE_IDS_SOURCE',
            '"--expected_camera_render_profile"',
            '"--expected_front_side_native_resolution"',
        ):
            self.assertIn(contract, source)

    def test_validator_can_freeze_fast_render_metadata_and_resolution(self):
        source = VALIDATOR.read_text()
        self.assertIn('"--expected_camera_render_profile"', source)
        self.assertIn('"--expected_front_side_native_resolution"', source)
        self.assertIn('config.get("camera_render_profile")', source)

    def test_fast84_pipeline_freezes_collection_and_two_dp_runs(self):
        source = FAST84_PIPELINE.read_text()
        ast.parse(source, filename=str(FAST84_PIPELINE))
        for contract in (
            '"episodes": 6000',
            '"repeats_per_state": 5',
            '"camera_render_profile": "policy_fast84_v1"',
            '"stored_image_size": 84',
            '"normal_nomask", "normal_mask"',
            '"PREPARE_ROI": "0"',
            '"training_started"',
            'SELECTED_STATE_IDS_SHA256',
            'wait_for_gpus(COLLECTION_GPUS',
        ):
            self.assertIn(contract, source)

    def test_formal_collection_requires_validated_pilot_status(self):
        source = PROMOTER.read_text()
        self.assertIn("pilot_complete)", source)
        self.assertIn("STOP_AFTER_PILOT=0", source)
        self.assertIn("failed)", source)

    def test_zstd_is_explicit_lossless_storage_and_is_validated(self):
        collector = VISION_COLLECTOR.read_text()
        validator = VALIDATOR.read_text()
        self.assertIn('choices=("default", "zstd")', collector)
        self.assertIn('cname="zstd"', collector)
        self.assertIn('choices=("default", "zstd")', validator)
        self.assertIn('"lz4" if args.expected_zarr_compressor == "default" else "zstd"', validator)


if __name__ == "__main__":
    unittest.main()
