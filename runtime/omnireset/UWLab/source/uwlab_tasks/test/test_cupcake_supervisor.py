"""CPU checks for fatal-error cutoffs and exclusion of later finite weights."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "cup_supervisor", ROOT / "scripts/franka_kl_distill/supervise_cupcake_latched_four_gpu.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestSupervisor(unittest.TestCase):
    def test_physics_error_time_is_not_poll_time(self):
        error = MODULE.first_failure(
            "2026-09-06T13:24:59Z PhysX error: failed to allocate memory 1176502272 bytes!", 2_000_000_000
        )
        self.assertEqual(error["cutoff"], 1788701099)

    def test_fatal_and_normal_lines(self):
        for text in ("Scene state is corrupted", "PhysX ABORT error", "RuntimeError: CUDA out of memory"):
            self.assertIsNotNone(MODULE.first_failure(text, 100))
        self.assertIsNone(MODULE.first_failure("Learning iteration 1050/40000\nMean value_function loss: 0.03", 100))
        self.assertIsNone(MODULE.first_failure("WARNING: render interval is smaller than decimation", 100))

    def test_post_failure_checkpoint_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for iteration, stamp in ((1000, 40), (1050, 70), (1100, 110)):
                path = root / f"model_{iteration}.pt"
                path.touch()
                os.utime(path, (stamp, stamp))
            with patch.object(MODULE, "validate_checkpoint", return_value=1050) as validate:
                chosen = MODULE.last_safe_checkpoint(root, 100, Path("fallback.pt"))
            self.assertEqual(chosen.name, "model_1050.pt")
            validate.assert_called_once_with(chosen)

    def test_invalid_candidate_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_1050.pt"
            path.touch()
            os.utime(path, (20, 20))
            with patch.object(MODULE, "validate_checkpoint", side_effect=ValueError("bad checkpoint")):
                self.assertEqual(MODULE.last_safe_checkpoint(Path(directory), 100, Path("fallback.pt")), Path("fallback.pt"))

    def test_real_process_fatal_recovers_only_prefault_checkpoint(self):
        # Exercise the supervisor, log polling, SIGTERM and relaunch without a GPU.
        real_popen = subprocess.Popen
        real_sleep = time.sleep
        calls = []
        children = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "watch"
            source = Path(directory) / "source.pt"
            data = {"iter": 1050, "model_state_dict": {"weight": torch.ones(1)},
                    "optimizer_state_dict": {"state": {0: {"step": torch.tensor(1)}}}}
            torch.save(data, source)

            def launch(command, **kwargs):
                env = kwargs["env"]
                self.assertEqual(env["GPU_IDS"], "1,2,3,4")
                self.assertEqual(env["NUM_ENVS_PER_GPU"], "40960")
                calls.append(env["RESUME_PATH"])
                run = Path(env["LOG_ROOT"]) / "franka_fr3_gripper_omnireset_agent" / (
                    env["RSL_RL_RUN_TIMESTAMP"] + "_" + env["RUN_NAME"])
                run.mkdir(parents=True)
                if len(calls) == 1:
                    good = run / "model_1050.pt"
                    bad = run / "model_1100.pt"
                    torch.save(data, good)
                    torch.save({**data, "iter": 1100}, bad)
                    now = time.time()
                    os.utime(good, (now - 60, now - 60))
                    os.utime(bad, (now + 5, now + 5))
                    script = 'import time; print("Scene state is corrupted. Simulation cannot continue!", flush=True); time.sleep(60)'
                else:
                    self.assertEqual(Path(calls[-1]).name, "model_1050.pt")
                    self.assertNotEqual(calls[-1], str(source))
                    (root / "STOP").touch()
                    script = 'import time; time.sleep(60)'
                process = real_popen([sys.executable, "-u", "-c", script], **kwargs)
                children.append(process)
                return process

            try:
                with patch.object(sys, "argv", ["supervisor", "--root", str(root), "--resume", str(source)]), \
                     patch.object(MODULE, "gpu_occupants", return_value=[]), \
                     patch.object(MODULE.subprocess, "Popen", side_effect=launch), \
                     patch.object(MODULE.time, "sleep", side_effect=lambda _: real_sleep(0.01)):
                    MODULE.main()
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(child.poll() is not None for child in children))
                events = (root / "events.jsonl").read_text()
                self.assertIn('"status": "fatal_error"', events)
                self.assertIn('"status": "recovering"', events)
                self.assertIn('"status": "stopped_by_operator"', events)
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
