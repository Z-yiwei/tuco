# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static and lightweight reset tests for the CupCake FR3 XY5 pipeline."""

import json
import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import zarr


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = REPO_ROOT / "scripts/franka_kl_distill"
ATOMIC_EVAL = SCRIPTS / "eval_cupcake_diffik_jointtarget.sh"
FOUR_PATH_EVAL = SCRIPTS / "eval_cupcake_v020_fourpath_stochastic.py"
RESET_BUILDER = SCRIPTS / "build_cupcake_xy5_t0_resets.py"
COLLECTOR = SCRIPTS / "collect_cupcake_fr3_xy5_t0_indexed_1200x5.sh"
MONITOR = SCRIPTS / "monitor_cupcake_v020_fourpath.py"
ORCHESTRATOR = SCRIPTS / "run_cupcake_fr3_xy5_t0_v020_1200x5.py"
REPLAY = SCRIPTS / "replay_jointtarget_visual_variants.py"
PROMOTION_WATCHER = SCRIPTS / "watch_cupcake_v020_to_demos.py"
GENERIC_COLLECTOR = SCRIPTS / "collect_vision_kl.py"
VALIDATOR = SCRIPTS / "validate_jointtarget_dataset.py"
TEMPLATE = REPO_ROOT / (
    "Datasets/OmniReset/Resets/CupCake__Plate/"
    "resets_ObjectAnywhereEEAnywhere_teamhome_20260814.pt"
)


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCupCakeFR3XY5V020PipelineContract(unittest.TestCase):
    def test_eval_is_stochastic_v020_h20_and_covers_all_four_paths(self):
        atomic = ATOMIC_EVAL.read_text()
        four_path = FOUR_PATH_EVAL.read_text()
        self.assertIn('"env.episode_length_s=${EPISODE_LENGTH_S}"', atomic)
        self.assertIn("--stochastic_actions", atomic)
        for reset_type in (
            "ObjectAnywhereEEAnywhere",
            "ObjectRestingEEGrasped",
            "ObjectAnywhereEEGrasped",
            "ObjectPartiallyAssembledEEGrasped",
        ):
            self.assertIn(reset_type, four_path)
        self.assertIn('"JOINT_MAX_VELOCITY": "0.20"', four_path)
        self.assertIn('"EPISODE_LENGTH_S": "20.0"', four_path)
        self.assertIn('"STOCHASTIC_ACTIONS": "1"', four_path)
        self.assertIn('"SUCCESS_POSITION_M": str(SUCCESS_POSITION_M)', four_path)
        self.assertIn('"SUCCESS_ORIENTATION_DEG": str(SUCCESS_ORIENTATION_DEG)', four_path)

    def test_collector_matches_fr3_white_v020_camera_and_guard_label_contract(self):
        source = COLLECTOR.read_text()
        for expected in (
            "OmniReset-FrankaResearch3MimicFingertip-RelCartesianOSC-RGB-Play-v0",
            "white_arm_base_ring_black_finger_pads",
            "axis_3cam_side16x9_wristrear_handy_mirror_20260826",
            "OMNIRESET_EXACT_WRIST_VERTICAL_FLIP=0",
            "--stochastic_teacher_actions",
            "--joint_ik_damping 0.05",
            "--joint_max_velocity 0.20",
            'JOINT_EPISODE_LENGTH_S="${JOINT_EPISODE_LENGTH_S:-80.0}"',
            '--joint_episode_length_s "${JOINT_EPISODE_LENGTH_S}"',
            "--b3_nominal_fixed_scene",
            "--preserve_object_face_materials",
            "env.scene.insertive_object=cupcake",
            "env.scene.receptive_object=plate",
            "plate=authored_white",
            'TEACHER_SHA256="${TEACHER_SHA256}"',
            'RESET_ARTIFACT_SHA256="${RESET_SHA256}"',
        ):
            self.assertIn(expected, source)
        self.assertNotIn("--stackcube_fixed_scene", source)
        self.assertNotIn("--fixed_open_gripper", source)
        self.assertNotIn("--raw_teacher_gripper", source)

    def test_joint_bridge_records_and_verifies_the_executed_guard_target(self):
        collector = GENERIC_COLLECTOR.read_text()
        validator = VALIDATOR.read_text()
        for expected in (
            '"relative_cartesian_diffik_absolute_joint_target"',
            "gripper_binary_command = gripper_term.raw_actions",
            "joint-target bridge grasp-guard label differs from the executed",
            '"gripper_binary_command": gripper_binary_command.detach()',
            '"gripper_width_target": gripper_width_target.detach()',
            '"teacher_checkpoint_sha256": _TEACHER_CHECKPOINT_SHA256',
            '"reset_artifact_sha256": os.environ.get("RESET_ARTIFACT_SHA256")',
        ):
            self.assertIn(expected, collector)
        self.assertIn(
            'config.get("guard_execution_verified_on_nonterminal_frames") is not True',
            validator,
        )

    def test_monitor_is_read_only_and_uses_checkpoint_aligned_rolling100(self):
        source = MONITOR.read_text()
        self.assertIn("checkpoint_rolling_100_overall_sr", source)
        self.assertIn("ready_for_independent_stochastic_eval", source)
        self.assertNotIn("os.kill", source)
        self.assertNotIn("subprocess", source)

    def test_orchestrator_gates_region_replay_and_exact_1200x5_quota(self):
        source = ORCHESTRATOR.read_text()
        for expected in (
            'CANDIDATE_STATES = int(os.environ.get("CANDIDATE_STATES", "4096"))',
            'SELECTED_STATES = int(os.environ.get("SELECTED_STATES", "1200"))',
            "PILOT_STATES = 32",
            "PILOT_REPEATS = 2",
            "FORMAL_REPEATS = 5",
            'MIN_SELECTOR_SR = float(os.environ.get("MIN_SELECTOR_SR", "0.85"))',
            'MIN_REPLAY_SR = float(os.environ.get("MIN_REPLAY_SR", "0.95"))',
            '(("clean", False), ("calibration", True))',
            'f"cupcake_pilot_10episodes_{overlay_name}.mp4"',
            "--require_guarded_demo_policy_eval",
            "--require_b3_nominal_fixed_scene",
            '"--expected_teacher_sha256"',
            '"--expected_reset_sha256"',
            "validate_selector_outputs(selector_outputs, selector_states)",
            "validate_state_chunks(source, outputs)",
        ):
            self.assertIn(expected, source)

    def test_delivery_manifest_is_fail_closed_and_records_reproduction_contract(self):
        source = ORCHESTRATOR.read_text()
        for expected in (
            "def write_delivery_manifest(",
            'if len(episode_ends) != SELECTED_STATES * FORMAL_REPEATS:',
            'unique_state_ids, repeat_counts = np.unique(reset_state_ids, return_counts=True)',
            'if len(unique_state_ids) != SELECTED_STATES:',
            'if not np.all(repeat_counts == FORMAL_REPEATS):',
            'if root.attrs.get("merge_write_verified") is not True:',
            'if not exact_report.is_file():',
            'selector_report.get("gate_passed") is not True',
            'if replay_rate + 1.0e-12 < MIN_REPLAY_SR:',
            'storage_report.get("local_gate_passed") is not True',
            'for key in ("clean", "calibration"):',
            '"result": "PASS"',
            '"method": "pure_behavior_cloning_demonstrations"',
            '"policy_action_mode": "stochastic_sample"',
            '"selected_state_ids_sha256": sha256(selected_ids)',
            '"action_replay_success_rate": replay_rate',
            '"successful_visual_variants_per_state": FORMAL_REPEATS',
            '"success_hold_steps": 5',
            'output = OUTPUT_ROOT / "delivery_manifest.json"',
        ):
            self.assertIn(expected, source)
        manifest_call = source.index("delivery_manifest = write_delivery_manifest(")
        completion = source.index('write_status(\n        "complete"', manifest_call)
        self.assertLess(manifest_call, completion)

    def test_delivery_manifest_runtime_rejects_uneven_state_quota(self):
        orchestrator = load_script_module("cupcake_delivery_manifest_contract", ORCHESTRATOR)
        with tempfile.TemporaryDirectory() as directory:
            root_path = Path(directory)
            final = root_path / "final.zarr"
            root = zarr.open(str(final), mode="w")
            root.create_dataset("meta/episode_ends", data=np.arange(1, 7, dtype=np.int64))
            state_ids = root.create_dataset(
                "meta/reset_state_ids",
                data=np.repeat(np.arange(3, dtype=np.int64), 2),
            )
            root.attrs["merge_write_verified"] = True
            root.attrs["collection_config"] = {"contract": "test"}

            teacher = root_path / "teacher.pt"
            reset = root_path / "reset.pt"
            reset_manifest = root_path / "reset_manifest.json"
            selected = root_path / "selected.json"
            exact_report = root_path / "exact.json"
            clean_video = root_path / "clean.mp4"
            calibration_video = root_path / "calibration.mp4"
            for path in (teacher, reset, clean_video, calibration_video):
                path.write_bytes(b"contract-test")
            reset_manifest.write_text("{}\n", encoding="utf-8")
            selected.write_text("[0, 1, 2]\n", encoding="utf-8")
            exact_report.write_text(
                json.dumps(
                    {
                        "result": "PASS",
                        "merged": str(final.resolve()),
                        "episodes": 6,
                        "frames": 6,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            orchestrator.OUTPUT_ROOT = root_path
            orchestrator.TEACHER = teacher
            orchestrator.RESET_FILE = reset
            orchestrator.RESET_MANIFEST = reset_manifest
            orchestrator.TEACHER_SHA256 = "a" * 64
            orchestrator.SELECTED_STATES = 3
            orchestrator.FORMAL_REPEATS = 2
            output = orchestrator.write_delivery_manifest(
                final=final,
                exact_report=exact_report,
                selected_ids=selected,
                selector_report={"gate_passed": True, "success_rate": 0.9},
                replay_rate=1.0,
                video_products={
                    "clean": str(clean_video),
                    "calibration": str(calibration_video),
                },
                storage_report={
                    "local_gate_passed": True,
                    "final_gate_passed": True,
                },
            )
            self.assertEqual(json.loads(output.read_text())["result"], "PASS")

            state_ids[:] = np.asarray([0, 0, 0, 1, 1, 2], dtype=np.int64)
            with self.assertRaisesRegex(ValueError, "exactly 2 episodes per reset state"):
                orchestrator.write_delivery_manifest(
                    final=final,
                    exact_report=exact_report,
                    selected_ids=selected,
                    selector_report={"gate_passed": True, "success_rate": 0.9},
                    replay_rate=1.0,
                    video_products={
                        "clean": str(clean_video),
                        "calibration": str(calibration_video),
                    },
                    storage_report={
                        "local_gate_passed": True,
                        "final_gate_passed": True,
                    },
                )

    def test_jointtarget_replay_accepts_task_neutral_b3_and_keeps_strict_default(self):
        source = REPLAY.read_text()
        self.assertIn('"b3_nominal_fixed"', source)
        self.assertIn("configure_b3_nominal_fixed_scene", source)
        self.assertIn('"--minimum_success_rate"', source)
        self.assertIn("default=1.0", source)

    def test_promotion_watcher_cannot_stop_training_before_independent_gate(self):
        source = PROMOTION_WATCHER.read_text()
        gate = source.index('summary.get("collection_gate_passed") is True')
        stop = source.index("stop_training(args.launcher_pid)", gate)
        collect = source.index("run_downstream_pipeline(", stop)
        self.assertLess(gate, stop)
        self.assertLess(stop, collect)
        self.assertIn("training_exited_before_promotion", source)
        self.assertIn("independent_eval_error_retrying_same_checkpoint", source)
        self.assertIn("summary.invalid.", source)
        self.assertIn('"--collection_gpus", default="0,2,6,7"', source)
        self.assertIn("fcntl.LOCK_EX | fcntl.LOCK_NB", source)

    def test_promotion_gate_uses_held_five_success_not_single_frame_strict_hit(self):
        evaluator = load_script_module("cupcake_fourpath_eval_contract", FOUR_PATH_EVAL)
        watcher = load_script_module("cupcake_promotion_watcher_contract", PROMOTION_WATCHER)
        checkpoint = Path("/tmp/cupcake_contract_model.pt").resolve()
        results = []
        for index, reset_type in enumerate(evaluator.RESET_TYPES):
            results.append(
                {
                    "path_index": index,
                    "reset_type": reset_type,
                    "strict_successes": 10,
                    "episodes": 10,
                    "strict_sr": 1.0,
                    "stable_successes": 8,
                    "stable_episodes": 10,
                    "stable_sr": 0.8,
                    "abnormal": 0,
                    "done_episodes": 10,
                    "timeouts": 10,
                    "max_joint_target_delta_rad": 0.02,
                    "success_position_m": 0.020,
                    "success_orientation_deg": 3.0,
                }
            )
        summary = evaluator.build_summary(
            checkpoint=checkpoint,
            checkpoint_sha256="a" * 64,
            seed=42,
            num_envs=32,
            episodes_per_path=10,
            stable_steps=5,
            gate=0.9,
            results=results,
        )
        self.assertEqual(summary["overall_strict_sr"], 1.0)
        self.assertEqual(summary["overall_stable_sr"], 0.8)
        self.assertFalse(summary["collection_gate_passed"])
        watcher.validate_independent_summary(
            summary,
            checkpoint=checkpoint,
            checkpoint_hash="a" * 64,
            seed=42,
            episodes_per_path=10,
        )
        summary["collection_gate_passed"] = True
        with self.assertRaisesRegex(ValueError, "gate boolean"):
            watcher.validate_independent_summary(
                summary,
                checkpoint=checkpoint,
                checkpoint_hash="a" * 64,
                seed=42,
                episodes_per_path=10,
            )

    def test_reset_builder_produces_exact_xy5_upright_teamhome_jitter(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "resets.pt"
            manifest_path = Path(directory) / "manifest.json"
            subprocess.run(
                [
                    str(Path(__import__("sys").executable)),
                    str(RESET_BUILDER),
                    "--template",
                    str(TEMPLATE),
                    "--output",
                    str(output),
                    "--manifest",
                    str(manifest_path),
                    "--insertive_center_xy",
                    "0.3684",
                    "-0.1281",
                    "--receptive_center_xy",
                    "0.4654",
                    "-0.0575",
                    "--num_states",
                    "64",
                    "--seed",
                    "20260828",
                    "--xy_size_m",
                    "0.05",
                    "--minimum_object_distance_m",
                    "0.06",
                    "--arm_joint_offset_rad",
                    "0.2",
                ],
                cwd=REPO_ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["xy_size_m"], 0.05)
            self.assertEqual(manifest["arm_joint_offset_range_rad"], [-0.2, 0.2])
            self.assertEqual(manifest["insertive_z_m"], 0.0)
            self.assertEqual(manifest["receptive_z_m"], 0.0)
            self.assertEqual(
                manifest["objects"]["plate"]["appearance"],
                "authored_white_diffuse_0.8",
            )

            reset = torch.load(output, map_location="cpu", weights_only=False)["initial_state"]
            for name in ("insertive_object", "receptive_object"):
                poses = torch.stack(reset["rigid_object"][name]["root_pose"])
                self.assertTrue(torch.allclose(poses[:, 2], torch.zeros_like(poses[:, 2])))
                self.assertTrue(torch.allclose(poses[:, 4:6], torch.zeros_like(poses[:, 4:6])))
            joints = torch.stack(reset["articulation"]["robot"]["joint_position"])
            self.assertTrue(torch.allclose(joints[:, 7:9], torch.full_like(joints[:, 7:9], 0.04)))


if __name__ == "__main__":
    unittest.main()
