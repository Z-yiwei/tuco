# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contract tests for the opt-in policy camera render profile."""

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
CAMERA_CFG = (
    REPO_ROOT
    / "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/omnireset"
    / "config/franka/data_collection_rgb_cfg.py"
)
COLLECTOR = REPO_ROOT / "scripts/franka_kl_distill/collect_vision_kl.py"


class TestFrankaPolicyFast84RenderProfile(unittest.TestCase):
    def test_profile_is_opt_in_and_preserves_native_default(self):
        source = CAMERA_CFG.read_text()
        self.assertIn('"OMNIRESET_CAMERA_RENDER_PROFILE", "native"', source)
        self.assertIn('"native": (1.0, 1.0, 1.0)', source)
        self.assertIn('"policy_fast84_v1": (0.5, 0.5, 1.0)', source)

    def test_profile_scales_raster_and_both_intrinsic_models(self):
        source = CAMERA_CFG.read_text()
        self.assertIn("def _scale_camera_raster(", source)
        self.assertIn("target_fx * scale_x", source)
        self.assertIn("render_fx * scale_x", source)
        self.assertIn("requires exact", source)

    def test_collection_metadata_records_profile(self):
        source = COLLECTOR.read_text()
        self.assertIn('"camera_render_profile": os.environ.get(', source)


if __name__ == "__main__":
    unittest.main()
