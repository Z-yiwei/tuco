"""CPU checks for the independent policy-gripper warm start and launcher."""

import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
PREPARE = ROOT / "scripts/franka_kl_distill/prepare_cupcake_policy_gripper_init.py"
spec = importlib.util.spec_from_file_location("prepare_cupcake_policy_gripper_init", PREPARE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class TestPolicyGripperTransfer(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(42)
        self.source = {
            "model_state_dict": {
                "actor.8.weight": torch.randn(7, 64, generator=generator),
                "actor.8.bias": torch.randn(7, generator=generator),
                "actor.0.weight": torch.randn(512, 200, generator=generator),
                "critic.0.weight": torch.randn(512, 171, generator=generator),
                "log_std": torch.randn(64, 7, generator=generator),
                "actor_obs_normalizer.count": torch.tensor(3858563072),
            },
            "iter": 5050,
            "optimizer_state_dict": {"old": True},
            "infos": {"old_curriculum": True},
        }

    def test_actor_only_gripper_mean_changes_and_critic_is_reinitialized(self):
        result = prepare.prepare_checkpoint(self.source)
        before, after = self.source["model_state_dict"], result["model_state_dict"]
        for key in before:
            if key.startswith("critic."):
                self.assertFalse(torch.equal(before[key], after[key]), key)
            elif key in ("actor.8.weight", "actor.8.bias"):
                self.assertTrue(torch.equal(before[key][:6], after[key][:6]))
                self.assertEqual(int(after[key][6].count_nonzero()), 0)
                self.assertGreater(int(before[key][6].count_nonzero()), 0)
            else:
                self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertEqual(result["iter"], 0)
        self.assertEqual(result["infos"], {})
        self.assertNotIn("optimizer_state_dict", result)
        self.assertEqual(after["critic.0.weight"].shape, (512, 168))
        self.assertEqual(after["critic_obs_normalizer._std"].shape, (1, 168))
        self.assertEqual(after["critic_obs_normalizer.count"].item(), 0)

    def test_reject_wrong_action_contract(self):
        self.source["model_state_dict"]["actor.8.weight"] = torch.ones(8, 64)
        with self.assertRaises(ValueError):
            prepare.prepare_checkpoint(self.source)

    def test_latch_initialization_changes_only_gripper_head_with_fr3_critic(self):
        self.source["model_state_dict"]["critic.0.weight"] = torch.randn(512, 168)
        result = prepare.prepare_checkpoint(self.source, gripper_mean_bias=5.0, keep_critic=True)
        before, after = self.source["model_state_dict"], result["model_state_dict"]
        self.assertEqual(float(after["actor.8.bias"][6]), 5.0)
        self.assertEqual(int(after["actor.8.weight"][6].count_nonzero()), 0)
        for key in before:
            if key in ("actor.8.weight", "actor.8.bias"):
                self.assertTrue(torch.equal(before[key][:6], after[key][:6]))
            else:
                self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertNotIn("optimizer_state_dict", result)

    def test_latch_init_rejects_nonfinite_bias_and_panda_critic(self):
        with self.assertRaises(ValueError):
            prepare.prepare_checkpoint(self.source, gripper_mean_bias=float("nan"))
        with self.assertRaises(ValueError):
            prepare.prepare_checkpoint(self.source, keep_critic=True)

    def test_reject_nonfinite_source(self):
        self.source["model_state_dict"]["critic.0.weight"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            prepare.prepare_checkpoint(self.source)

    def test_launcher_uses_plain_ppo_without_freeze_or_adr_flags(self):
        script = (ROOT / "scripts/franka_kl_distill/train_cupcake_policy_gripper_fourpath.sh").read_text()
        self.assertIn("PolicyGripper-DiffIK-State-v0", script)
        self.assertIn("--init_path", script)
        self.assertNotIn("--freeze", script)
        self.assertNotIn("Distillation", script)
        self.assertIn("agent.algorithm.entropy_coef=0.006", script)


if __name__ == "__main__":
    unittest.main()
