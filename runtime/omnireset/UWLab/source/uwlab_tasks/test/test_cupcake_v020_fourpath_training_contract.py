# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contract tests for the CupCake v0.2 four-path continuation."""

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FRANKA_DIR = REPO_ROOT / (
    "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/"
    "omnireset/config/franka"
)
RL_CFG = FRANKA_DIR / "rl_state_cfg.py"
REGISTRY = FRANKA_DIR / "__init__.py"
BASE_TRAIN = REPO_ROOT / "scripts/franka_kl_distill/train_cupcake_diffik_jointtarget.sh"
V020_TRAIN = REPO_ROOT / (
    "scripts/franka_kl_distill/train_cupcake_v020_fourpath_diffik_jointtarget.sh"
)


def _class_source(path: Path, class_name: str) -> str:
    module = ast.parse(path.read_text(), filename=str(path))
    node = next(
        item
        for item in module.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    return ast.unparse(node)


class TestCupCakeV020FourPathTrainingContract(unittest.TestCase):
    def test_task_keeps_stock_four_path_reset_distribution(self):
        registry = REGISTRY.read_text()
        train_events = _class_source(RL_CFG, "TrainEventCfg")
        joint_events = _class_source(RL_CFG, "JointB3TrainEventCfg")

        self.assertIn(
            'id="OmniReset-FrankaFr3Gripper-RelCartesianDiffIKJointTarget-State-v0"',
            registry,
        )
        for reset_type in (
            "ObjectAnywhereEEAnywhere",
            "ObjectRestingEEGrasped",
            "ObjectAnywhereEEGrasped",
            "ObjectPartiallyAssembledEEGrasped",
        ):
            self.assertIn(reset_type, train_events)
        self.assertIn("[0.25, 0.25, 0.25, 0.25]", train_events)
        self.assertIn("randomize_arm_from_sysid_fixed", joint_events)
        self.assertIn("'scale_range': (1.0, 1.0)", joint_events)

    def test_launcher_changes_only_rate_horizon_and_runtime_settings(self):
        base = BASE_TRAIN.read_text()
        launcher = V020_TRAIN.read_text()

        self.assertIn('EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-16.0}"', base)
        self.assertIn('"env.episode_length_s=${EPISODE_LENGTH_S}"', base)
        for frozen_setting in (
            "STAGE=stage1",
            "RESET_OPTIMIZER_ON_RESUME=0",
            "SEED=42",
            "LEARNING_RATE=2.0e-6",
            "SCHEDULE=fixed",
            "JOINT_MAX_VELOCITY=0.20",
            "EPISODE_LENGTH_S=20.0",
        ):
            self.assertIn(frozen_setting, launcher)

        self.assertIn("NUM_ENVS_PER_GPU", launcher)
        self.assertIn("8192", launcher)
        self.assertIn("NPROC", launcher)
        self.assertIn("model_2075.pt", launcher)
        self.assertNotIn("XY5", launcher)
        self.assertNotIn("TEAM_HOME", launcher)
        self.assertNotIn("HOME_JOINT_JITTER", launcher)


if __name__ == "__main__":
    unittest.main()
