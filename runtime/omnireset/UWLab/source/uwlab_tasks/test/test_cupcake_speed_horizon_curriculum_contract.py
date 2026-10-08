# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contract tests for the CupCake speed/horizon curriculum."""

import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "scripts/franka_kl_distill"
CONTRACT = SCRIPT_DIR / "cupcake_speed_horizon_curriculum_t0gate_v1.json"
XY5_CONTRACT = SCRIPT_DIR / "cupcake_speed_horizon_curriculum_xy5_t0_v1.json"
XY5_T075_CONTRACT = SCRIPT_DIR / "cupcake_speed_horizon_curriculum_xy5_t0_t075_v2.json"
XY5_T075_ENTROPY0_CONTRACT = SCRIPT_DIR / (
    "cupcake_speed_horizon_curriculum_xy5_t0_t075_entropy0_v3.json"
)
XY5_T075_ENTROPY0_LR5E5_CONTRACT = SCRIPT_DIR / (
    "cupcake_speed_horizon_curriculum_xy5_t0_t075_entropy0_lr5e5_v4.json"
)
XY5_T075_ENTROPY0_LR5E6_CONTRACT = SCRIPT_DIR / (
    "cupcake_speed_horizon_curriculum_xy5_t0_t075_entropy0_lr5e6_v5.json"
)
XY5_T060_ENTROPY0_LR5E6_CONTRACT = SCRIPT_DIR / (
    "cupcake_speed_horizon_curriculum_xy5_t0_t060_entropy0_lr5e6_v6.json"
)
XY5_T060_FROM_V5S0_CONTRACT = SCRIPT_DIR / (
    "cupcake_speed_horizon_curriculum_xy5_t0_t060_entropy0_lr5e6_"
    "from_v5s0_v7.json"
)
XY5_T060_V025H64_V020H80_CONTRACT = SCRIPT_DIR / (
    "cupcake_speed_horizon_curriculum_xy5_t0_t060_entropy0_lr5e6_"
    "v025h64_v020h80_from_v7s1_v8.json"
)
RUNNER = SCRIPT_DIR / "run_cupcake_speed_horizon_curriculum.py"
BASE_TRAIN = SCRIPT_DIR / "train_cupcake_diffik_jointtarget.sh"
BASE_EVAL = SCRIPT_DIR / "eval_cupcake_diffik_jointtarget.sh"
TRAIN_PY = REPO_ROOT / "UWLab/scripts/reinforcement_learning/rsl_rl/train.py"
OBSERVATIONS = REPO_ROOT / (
    "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/"
    "omnireset/mdp/observations.py"
)
RL_CFG = REPO_ROOT / (
    "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/"
    "omnireset/config/ur5e_robotiq_2f85/rl_state_cfg.py"
)
FRANKA_RL_CFG = REPO_ROOT / (
    "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/"
    "omnireset/config/franka/rl_state_cfg.py"
)


class TestCupCakeSpeedHorizonCurriculumContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT.read_text())
        cls.xy5_contract = json.loads(XY5_CONTRACT.read_text())
        cls.xy5_t075_contract = json.loads(XY5_T075_CONTRACT.read_text())
        cls.xy5_t075_entropy0_contract = json.loads(
            XY5_T075_ENTROPY0_CONTRACT.read_text()
        )
        cls.xy5_t075_entropy0_lr5e5_contract = json.loads(
            XY5_T075_ENTROPY0_LR5E5_CONTRACT.read_text()
        )
        cls.xy5_t075_entropy0_lr5e6_contract = json.loads(
            XY5_T075_ENTROPY0_LR5E6_CONTRACT.read_text()
        )
        cls.xy5_t060_entropy0_lr5e6_contract = json.loads(
            XY5_T060_ENTROPY0_LR5E6_CONTRACT.read_text()
        )
        cls.xy5_t060_from_v5s0_contract = json.loads(
            XY5_T060_FROM_V5S0_CONTRACT.read_text()
        )
        cls.xy5_t060_v025h64_v020h80_contract = json.loads(
            XY5_T060_V025H64_V020H80_CONTRACT.read_text()
        )

    def test_t060_v8_uses_gated_model4937_and_longer_two_step_transition(self):
        contract = self.xy5_t060_v025h64_v020h80_contract
        self.assertEqual(Path(contract["source_checkpoint"]).name, "model_4937.pt")
        self.assertEqual(
            contract["source_checkpoint_sha256"],
            "824cba20e13624cf0735ddb9b604bd529f32ea24a21bf88b0f501560419c85e2",
        )
        self.assertEqual(contract["task0_success_gate"], 0.60)
        self.assertEqual(contract["joint_travel_budget_rad"], 16.0)
        self.assertEqual(
            [stage["joint_max_velocity_rad_s"] for stage in contract["stages"]],
            [0.25, 0.2],
        )
        self.assertEqual(
            [stage["episode_length_s"] for stage in contract["stages"]],
            [64.0, 80.0],
        )
        self.assertEqual(
            [stage["max_episode_steps"] for stage in contract["stages"]],
            [640, 800],
        )
        predecessor = contract["predecessor_gate_evidence"]
        self.assertEqual(predecessor["joint_max_velocity_rad_s"], 0.3)
        self.assertEqual(predecessor["joint_travel_budget_rad"], 9.6)
        self.assertEqual(predecessor["success_rate"], 0.6067742109298706)
        self.assertGreater(predecessor["success_rate"], predecessor["threshold"])

        spec = importlib.util.spec_from_file_location("cupcake_speed_runner_v8", RUNNER)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(
            module.load_and_validate_contract(
                XY5_T060_V025H64_V020H80_CONTRACT
            )["version"],
            contract["version"],
        )

    def test_t060_v7_continues_from_gated_model3250_at_v04(self):
        contract = self.xy5_t060_from_v5s0_contract
        baseline = self.xy5_t075_entropy0_lr5e6_contract
        self.assertEqual(Path(contract["source_checkpoint"]).name, "model_3250.pt")
        self.assertEqual(
            contract["source_checkpoint_sha256"],
            "7ad24a73f0bb22d39e77c04c9353a1e18de21148cbc5fcdee23db3674c18131f",
        )
        self.assertEqual(contract["task0_success_gate"], 0.60)
        self.assertEqual(
            [stage["joint_max_velocity_rad_s"] for stage in contract["stages"]],
            [0.4, 0.3, 0.2],
        )
        self.assertEqual(contract["target_velocity_start_update"], 1000)
        self.assertEqual(contract["frozen_training"], baseline["frozen_training"])
        self.assertEqual(contract["reset_contract"], baseline["reset_contract"])
        predecessor = contract["predecessor_gate_evidence"]
        self.assertEqual(predecessor["joint_max_velocity_rad_s"], 0.5)
        self.assertEqual(predecessor["success_rate"], 0.6726)
        self.assertGreater(predecessor["success_rate"], predecessor["threshold"])
        self.assertEqual(
            predecessor["checkpoint_sha256"], contract["source_checkpoint_sha256"]
        )

        spec = importlib.util.spec_from_file_location("cupcake_speed_runner_v7", RUNNER)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(
            module.load_and_validate_contract(XY5_T060_FROM_V5S0_CONTRACT)["version"],
            contract["version"],
        )

    def test_t060_revision_resumes_complete_model2950_and_changes_the_gate(self):
        baseline = self.xy5_t075_entropy0_lr5e6_contract
        revision = self.xy5_t060_entropy0_lr5e6_contract
        self.assertEqual(Path(revision["source_checkpoint"]).name, "model_2950.pt")
        self.assertEqual(
            revision["source_checkpoint_sha256"],
            "caf32eb257ab370404d44b939286c09d7fd1ad8de1d58ec12019b6dedcf94cc5",
        )
        self.assertEqual(revision["task0_success_gate"], 0.60)
        self.assertEqual(revision["post_curriculum"]["required_task0_success_gate"], 0.60)
        self.assertEqual(revision["stages"], baseline["stages"])
        self.assertEqual(revision["frozen_training"], baseline["frozen_training"])
        self.assertEqual(revision["reset_contract"], baseline["reset_contract"])

    def test_lr5e6_ablation_changes_only_learning_rate_from_entropy0(self):
        baseline = self.xy5_t075_entropy0_contract
        ablation = self.xy5_t075_entropy0_lr5e6_contract
        self.assertEqual(baseline["frozen_training"]["learning_rate"], 2e-6)
        self.assertEqual(ablation["frozen_training"]["learning_rate"], 5e-6)
        self.assertEqual(ablation["frozen_training"]["entropy_coef"], 0.0)
        baseline_copy = json.loads(json.dumps(baseline))
        ablation_copy = json.loads(json.dumps(ablation))
        for payload in (baseline_copy, ablation_copy):
            payload.pop("version")
            payload.pop("output_root")
            payload.pop("ablation_contract")
        baseline_copy["frozen_training"]["learning_rate"] = 5e-6
        self.assertEqual(ablation_copy, baseline_copy)

    def test_lr5e5_ablation_changes_only_learning_rate_from_entropy0(self):
        baseline = self.xy5_t075_entropy0_contract
        ablation = self.xy5_t075_entropy0_lr5e5_contract
        self.assertEqual(baseline["frozen_training"]["learning_rate"], 2e-6)
        self.assertEqual(ablation["frozen_training"]["learning_rate"], 5e-5)
        self.assertEqual(ablation["frozen_training"]["entropy_coef"], 0.0)
        baseline_copy = json.loads(json.dumps(baseline))
        ablation_copy = json.loads(json.dumps(ablation))
        for payload in (baseline_copy, ablation_copy):
            payload.pop("version")
            payload.pop("output_root")
            payload.pop("ablation_contract")
        baseline_copy["frozen_training"]["learning_rate"] = 5e-5
        self.assertEqual(ablation_copy, baseline_copy)

    def test_entropy0_ablation_changes_entropy_only_and_preserves_global_batch(self):
        baseline = self.xy5_t075_contract
        ablation = self.xy5_t075_entropy0_contract
        self.assertEqual(ablation["source_checkpoint"].rsplit("/", 1)[-1], "model_2725.pt")
        self.assertEqual(ablation["frozen_training"]["entropy_coef"], 0.0)
        self.assertEqual(
            ablation["frozen_training"]["nproc"]
            * ablation["frozen_training"]["num_envs_per_gpu"],
            baseline["frozen_training"]["nproc"]
            * baseline["frozen_training"]["num_envs_per_gpu"],
        )
        for key in (
            "reset_type",
            "reset_file_sha256",
            "updates_per_stage",
            "task0_success_gate",
            "control_dt_s",
            "joint_travel_budget_rad",
            "target_velocity_rad_s",
            "target_velocity_start_update",
            "final_target_hold_updates",
            "stages",
            "post_curriculum",
        ):
            self.assertEqual(ablation[key], baseline[key])

    def test_xy5_t075_contract_resumes_model2600_and_gates_every_speed_at_75_percent(self):
        contract = self.xy5_t075_contract
        self.assertEqual(Path(contract["source_checkpoint"]).name, "model_2600.pt")
        self.assertEqual(
            contract["source_checkpoint_sha256"],
            "b04c00db0efeb13a526471197f930c8d8898ef3fe77d9e24012a92bbc5f98267",
        )
        self.assertEqual(contract["task0_success_gate"], 0.75)
        self.assertEqual(
            [stage["joint_max_velocity_rad_s"] for stage in contract["stages"]],
            [0.5, 0.4, 0.3, 0.2],
        )
        self.assertEqual(
            [stage["episode_length_s"] for stage in contract["stages"]],
            [19.2, 24.0, 32.0, 48.0],
        )
        post = contract["post_curriculum"]
        self.assertEqual(post["collection_unique_states"], 1200)
        self.assertEqual(post["collection_successful_variants_per_state"], 5)
        self.assertEqual(
            post["vision_dp_settings"],
            ["normal_nomask", "normal_mask", "roi_nomask", "roi_mask"],
        )

    def test_xy5_t0_contract_starts_at_v05_from_formally_gated_model2575(self):
        contract = self.xy5_contract
        self.assertEqual(Path(contract["source_checkpoint"]).name, "model_2575.pt")
        self.assertEqual(
            contract["source_checkpoint_sha256"],
            "21fd8de881d44b8102773e3fc01b84a8770ae9a9f62de0a67dca266494eba4af",
        )
        self.assertEqual(
            [stage["joint_max_velocity_rad_s"] for stage in contract["stages"]],
            [0.5, 0.4, 0.3, 0.2],
        )
        self.assertEqual(contract["target_velocity_start_update"], 1500)
        self.assertEqual(contract["task0_success_gate"], 0.7)

    def test_xy5_t0_contract_freezes_single_variable_reset(self):
        contract = self.xy5_contract
        self.assertEqual(
            contract["reset_type"],
            "ObjectAnywhereEEAnywhere_cupcake_xy5_x0350_0400_yneg0200_neg0150_20260829",
        )
        self.assertEqual(
            contract["reset_file_sha256"],
            "9fe1d3a906217072c1837ac4f6d6fd4967b180bdce52fd58576f686ed591c4ad",
        )
        reset_contract = contract["reset_contract"]
        self.assertEqual(reset_contract["cupcake_x_range_m"], [0.35, 0.4])
        self.assertEqual(reset_contract["cupcake_y_range_m"], [-0.2, -0.15])
        self.assertEqual(reset_contract["model_2575_v06_held5_success"], "1985/2500")

    def test_speed_reaches_v020_at_update_2000(self):
        stages = self.contract["stages"]
        self.assertEqual([stage["index"] for stage in stages], list(range(5)))
        self.assertEqual(
            [stage["joint_max_velocity_rad_s"] for stage in stages],
            [0.6, 0.5, 0.4, 0.3, 0.2],
        )
        self.assertEqual([stage["start_update"] for stage in stages], [0, 500, 1000, 1500, 2000])
        self.assertEqual(stages[-1]["end_update_exclusive"], 2500)
        self.assertEqual(self.contract["final_target_hold_updates"], 500)

    def test_source_is_the_frozen_healthy_v06_model2150(self):
        source = Path(self.contract["source_checkpoint"])
        self.assertEqual(source.name, "model_2150.pt")
        self.assertEqual(
            self.contract["source_checkpoint_sha256"],
            "41bf7b63dc1000bd710e018c1a83a96bb0dfc5917c1db955fc020a77225f2547",
        )
        if source.is_file():
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                self.contract["source_checkpoint_sha256"],
            )

    def test_horizon_preserves_travel_budget_and_time_left_denominator(self):
        dt = self.contract["control_dt_s"]
        budget = self.contract["joint_travel_budget_rad"]
        for stage in self.contract["stages"]:
            velocity = stage["joint_max_velocity_rad_s"]
            horizon = stage["episode_length_s"]
            self.assertTrue(math.isclose(velocity * horizon, budget, abs_tol=1e-9))
            self.assertEqual(stage["max_episode_steps"], math.ceil(horizon / dt))

        observations = OBSERVATIONS.read_text()
        rl_cfg = RL_CFG.read_text()
        self.assertIn("env.episode_length_buf.float() / env.max_episode_length", observations)
        self.assertIn("time_left = ObsTerm(func=task_mdp.time_left)", rl_cfg)

    def test_segments_preserve_optimizer_and_fixed_b3_contract(self):
        frozen = self.contract["frozen_training"]
        self.assertEqual(frozen["task_stage"], "stage1")
        self.assertFalse(frozen["reset_optimizer_on_resume"])
        self.assertEqual(frozen["learning_rate_schedule"], "fixed")
        self.assertEqual(frozen["nproc"], 4)
        self.assertEqual(frozen["num_envs_per_gpu"], 8192)
        self.assertTrue(frozen["offline_scene"])
        self.assertTrue(frozen["local_fall_catcher"])
        self.assertTrue(frozen["init_at_random_ep_len"])
        self.assertTrue(frozen["freeze_actor_observation_normalizer_on_resume"])
        self.assertEqual(self.contract["task0_success_gate"], 0.7)

        runner = RUNNER.read_text()
        base = BASE_TRAIN.read_text()
        eval_script = BASE_EVAL.read_text()
        train_py = TRAIN_PY.read_text()
        self.assertIn('"RESET_OPTIMIZER_ON_RESUME": "0"', runner)
        self.assertIn("refusing to start from a checkpoint with an unexpected SHA256", runner)
        self.assertIn('"MAX_ITERATIONS": str(contract["updates_per_stage"])', runner)
        self.assertIn('"JOINT_MAX_VELOCITY":', runner)
        self.assertIn('"EPISODE_LENGTH_S":', runner)
        self.assertIn('"OFFLINE_SCENE": "1" if frozen["offline_scene"] else "0"', runner)
        self.assertIn('"INIT_AT_RANDOM_EP_LEN":', runner)
        self.assertIn('"LOCAL_FALL_CATCHER":', runner)
        self.assertIn('"FREEZE_ACTOR_OBS_NORMALIZER_ON_RESUME":', runner)
        self.assertIn('"STOP_ON_TASK0_SUCCESS_RATE":', runner)
        self.assertIn("_validated_task0_gate", runner)
        self.assertIn('CHECKPOINT_ARGS=(--resume_path "${RESUME_PATH}")', base)
        self.assertIn("--disable_random_episode_length_init", base)
        self.assertIn("FREEZE_ACTOR_OBS_NORMALIZER_ON_RESUME", base)
        self.assertIn("--freeze_actor_observation_normalizer_on_resume", base)
        self.assertIn("--stop_on_task0_success_rate", base)
        self.assertIn("install_task0_success_gate", train_py)
        self.assertIn("success_rate <= threshold", train_py)
        self.assertIn('OMNIRESET_LOCAL_FALL_CATCHER="${LOCAL_FALL_CATCHER}"', base)
        self.assertIn('RESET_TYPE="${RESET_TYPE:-}"', base)
        self.assertIn('env.events.reset_from_reset_states.params.reset_types=[${RESET_TYPE}]', base)
        self.assertIn('env.events.reset_from_reset_states.params.probs=[1.0]', base)
        self.assertIn('"RESET_TYPE": str(contract.get("reset_type", ""))', runner)
        self.assertIn("reset_file_sha256", runner)
        self.assertIn('OMNIRESET_LOCAL_FALL_CATCHER="${LOCAL_FALL_CATCHER}"', eval_script)
        franka_rl_cfg = FRANKA_RL_CFG.read_text()
        self.assertIn('os.environ.get("OMNIRESET_LOCAL_FALL_CATCHER", "0")', franka_rl_cfg)
        self.assertIn("pos=(0.0, 0.0, -0.878)", franka_rl_cfg)
        self.assertIn("size=(1000.0, 1000.0, 0.02)", franka_rl_cfg)
        self.assertIn("collision_group=-1", franka_rl_cfg)
        self.assertIn('"env.actions.arm.max_joint_velocity=${JOINT_MAX_VELOCITY_7}"', base)
        self.assertIn('"env.episode_length_s=${EPISODE_LENGTH_S}"', base)

    def test_gate_evidence_requires_strictly_greater_and_matching_checkpoint(self):
        spec = importlib.util.spec_from_file_location("cupcake_speed_runner", RUNNER)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint = run_dir / "model_42.pt"
            checkpoint.write_bytes(b"checkpoint")
            payload = {
                "version": 1,
                "metric": "Metrics/task_0_success_rate",
                "comparison": ">",
                "threshold": 0.7,
                "success_rate": 0.701,
                "iteration": 42,
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
            (run_dir / "task0_success_gate.json").write_text(json.dumps(payload))
            gated_checkpoint, evidence = module._validated_task0_gate(run_dir, 0.7)
            self.assertEqual(gated_checkpoint, checkpoint)
            self.assertEqual(evidence["success_rate"], 0.701)

            payload["success_rate"] = 0.7
            (run_dir / "task0_success_gate.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "does not prove T0"):
                module._validated_task0_gate(run_dir, 0.7)


if __name__ == "__main__":
    unittest.main()
