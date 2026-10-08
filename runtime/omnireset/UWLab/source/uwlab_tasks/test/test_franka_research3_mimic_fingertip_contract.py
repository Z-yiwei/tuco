# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contract tests for the collision-only Research 3 compatibility asset."""

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FRANKA_DIR = REPO_ROOT / (
    "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/"
    "omnireset/config/franka"
)
SPAWNER = FRANKA_DIR / "research3_mimic_fingertips.py"
ASSETS = FRANKA_DIR / "research3_assets_cfg.py"
CONFIG = FRANKA_DIR / "research3_cfg.py"
REGISTRY = FRANKA_DIR / "__init__.py"
RL_EVAL = REPO_ROOT / (
    "scripts/franka_kl_distill/"
    "eval_stackcube_research3_mimic_fingertip_diffik_jointtarget.sh"
)
REPLAY = REPO_ROOT / "scripts/franka_kl_distill/replay_stackcube_research3_mimic_fingertip.sh"
COLLECT = REPO_ROOT / "scripts/franka_kl_distill/collect_stackcube_research3_mimic_fingertip.sh"
COLLECTOR = REPO_ROOT / "scripts/franka_kl_distill/collect_vision_kl.py"


class TestResearch3MimicFingertipContract(unittest.TestCase):
    def test_spawner_changes_only_fingertip_collision_composition(self):
        source = SPAWNER.read_text()
        ast.parse(source, filename=str(SPAWNER))
        self.assertIn("spawn_from_usd.__wrapped__", source)
        self.assertIn('official_collision.SetActive(False)', source)
        self.assertIn('"/panda/panda_leftfinger/collisions"', source)
        self.assertIn('"/panda/panda_rightfinger/collisions"', source)
        self.assertIn('approximation != "convexHull"', source)
        self.assertIn('FR3_VISUAL_PROFILE_WHITE_ARM = "white_arm_base_ring"', source)
        self.assertIn(
            'FR3_VISUAL_PROFILE_WHITE_ARM_BLACK_PADS = "white_arm_base_ring_black_finger_pads"',
            source,
        )
        self.assertIn("_BASE_RING_SHADER", source)
        self.assertIn("_WHITE_ARM_DARK_SHADERS", source)
        self.assertIn("_FINGER_PAD_SHADER", source)
        self.assertIn("finger_pad.Set(_BLACK)", source)

    def test_asset_retains_official_fr3_usd(self):
        source = ASSETS.read_text()
        ast.parse(source, filename=str(ASSETS))
        self.assertIn("FRANKA_RESEARCH3_MIMIC_FINGERTIP_CFG = FRANKA_RESEARCH3_CFG.replace", source)
        self.assertIn("spawn_research3_with_mimic_fingertips", source)

    def test_hybrid_tasks_are_explicit_and_separate(self):
        registry = REGISTRY.read_text()
        config = CONFIG.read_text()
        for task_id in (
            "OmniReset-FrankaResearch3MimicFingertip-"
            "RelCartesianDiffIKJointTarget-State-Finetune-Play-v0",
            "OmniReset-FrankaResearch3MimicFingertip-AbsoluteJointTarget-RGB-Play-v0",
            "OmniReset-FrankaResearch3MimicFingertip-RelCartesianOSC-RGB-Play-v0",
        ):
            self.assertIn(f'id="{task_id}"', registry)
        self.assertIn("FrankaResearch3MimicFingertipRGBAbsoluteJointTargetEvalCfg", config)
        self.assertIn("_finalize_mimic_fingertip_config", config)

    def test_launchers_select_only_explicit_hybrid_tasks(self):
        self.assertIn(
            "OmniReset-FrankaResearch3MimicFingertip-"
            "RelCartesianDiffIKJointTarget-State-Finetune-Play-v0",
            RL_EVAL.read_text(),
        )
        self.assertIn(
            "OmniReset-FrankaResearch3MimicFingertip-AbsoluteJointTarget-RGB-Play-v0",
            REPLAY.read_text(),
        )
        self.assertIn(
            "OmniReset-FrankaResearch3MimicFingertip-RelCartesianOSC-RGB-Play-v0",
            COLLECT.read_text(),
        )
        for launcher in (RL_EVAL, REPLAY, COLLECT):
            self.assertIn("OMNIRESET_FR3_VISUAL_PROFILE", launcher.read_text())
        self.assertIn("research3_mimic_fingertip_convex_hull", COLLECTOR.read_text())
        self.assertIn('"robot_visual_profile"', COLLECTOR.read_text())


if __name__ == "__main__":
    unittest.main()
