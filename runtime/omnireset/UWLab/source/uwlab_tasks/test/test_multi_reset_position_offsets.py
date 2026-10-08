# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contract tests for table-relative OmniReset rigid-object offsets."""

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
EVENTS = REPO_ROOT / (
    "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/"
    "omnireset/mdp/events.py"
)
FRANKA_CFG = REPO_ROOT / (
    "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/"
    "omnireset/config/franka/rl_state_cfg.py"
)


class TestMultiResetPositionOffsets(unittest.TestCase):
    def test_multi_reset_validates_and_applies_xyz_offsets(self):
        source = EVENTS.read_text()
        ast.parse(source, filename=str(EVENTS))
        self.assertIn('cfg.params.get("rigid_object_position_offsets", {})', source)
        self.assertIn("offset_tensor.shape != (3,)", source)
        self.assertIn("root_pose[:, :3] += position_offset", source)

    def test_franka_train_and_eval_reset_contracts_expose_offsets(self):
        source = FRANKA_CFG.read_text()
        ast.parse(source, filename=str(FRANKA_CFG))
        self.assertGreaterEqual(source.count('"rigid_object_position_offsets": {'), 3)
        self.assertGreaterEqual(source.count('"insertive_object": (0.0, 0.0, 0.0)'), 3)
        self.assertGreaterEqual(source.count('"receptive_object": (0.0, 0.0, 0.0)'), 3)


if __name__ == "__main__":
    unittest.main()
