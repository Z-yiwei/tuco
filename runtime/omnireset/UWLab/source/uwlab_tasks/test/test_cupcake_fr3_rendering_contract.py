# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static contract checks for the canonical CupCake FR3 renderer."""

import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
FRANKA_DIR = REPO_ROOT / (
    "UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/manipulation/"
    "omnireset/config/franka"
)
CAMERA_CFG = FRANKA_DIR / "data_collection_rgb_cfg.py"
FR3_VISUALS = FRANKA_DIR / "research3_mimic_fingertips.py"
RENDERER = REPO_ROOT / "scripts/franka_kl_distill/render_cameras_debug.py"
WRAPPER = REPO_ROOT / "scripts/franka_kl_distill/render_cupcake_fr3_upright_contract.sh"


class TestCupCakeFR3RenderingContract(unittest.TestCase):
    def test_wrapper_freezes_robot_camera_and_table_contract(self):
        source = WRAPPER.read_text()
        for expected in (
            "OmniReset-FrankaResearch3MimicFingertip-RelCartesianOSC-RGB-Play-v0",
            "axis_3cam_side16x9_wristrear_handy_mirror_20260826",
            "white_arm_base_ring_black_finger_pads",
            "OMNIRESET_EXACT_CAMERA_INTRINSICS=1",
            "OMNIRESET_EXACT_WRIST_VERTICAL_FLIP=0",
            "OMNIRESET_REAL_TABLE_COVER_COLLISION=0",
            "OMNIRESET_REAL_RECEPTIVE_OBJECT_Z_LIFT=0",
            'PLATE_UNIFORM_SCALE="${PLATE_UNIFORM_SCALE:-0.6666666666666666}"',
            '--receptive_object_uniform_scale "${PLATE_UNIFORM_SCALE}"',
            "--home_robot_root_pose 0 0 0 1 0 0 0",
            "--insertive_object_z 0",
            "--receptive_object_z 0",
            "--preserve_object_face_materials",
            "env.scene.insertive_object=cupcake",
            "env.scene.receptive_object=plate",
            "env.scene.table.init_state.pos=[0.4,0.0,-0.868]",
        ):
            self.assertIn(expected, source)
        self.assertNotIn("--rotate_180_camera", source)
        self.assertNotIn("axis_roi", source)

    def test_nominal_and_wild_share_one_wrapper(self):
        source = WRAPPER.read_text()
        self.assertIn("nominal|wild", source)
        self.assertIn("RENDER_MODE_ARGS=(--preserve_object_face_materials)", source)
        self.assertIn("RENDER_MODE_ARGS+=(--drop_appearance)", source)

    def test_renderer_records_the_resolved_visual_contract(self):
        source = RENDERER.read_text()
        ast.parse(source, filename=str(RENDERER))
        self.assertIn('"--preserve_object_face_materials"', source)
        self.assertIn('"--receptive_object_uniform_scale"', source)
        self.assertIn(
            "env_cfg.scene.receptive_object.spawn.scale = (receptive_object_scale,) * 3",
            source,
        )
        self.assertIn('"receptive_object_spawn_scale_xyz": list(resolved_receptive_scale)', source)
        self.assertIn('"visual_contract": {', source)
        self.assertIn('"root_position_world_xyz_m"', source)
        self.assertIn('"vertical_flip": term_cfg.params.get("vertical_flip", False)', source)

    def test_camera_background_and_fr3_material_sources_are_frozen(self):
        camera = CAMERA_CFG.read_text()
        visuals = FR3_VISUALS.read_text()
        ast.parse(camera, filename=str(CAMERA_CFG))
        ast.parse(visuals, filename=str(FR3_VISUALS))
        for expected in (
            "_FRONT_POS = (1.3773505058279372, 0.02255364505239039, 0.7922703611113961)",
            "_SIDE_POS = (0.5099174380302429, 0.6603941917419434, 0.30853092670440674)",
            "_WRIST_POS = (0.158257909, -0.006837764, 0.047477325)",
            "_WRIST_FX, _WRIST_FY = 437.7700500488281, 436.719482421875",
            "_CURTAIN_FLOOR_SIZE = (2.70, 5.00, 0.001)",
            "_TABLE_COVER_COLOR = (0.015, 0.015, 0.014)",
            '"texture_prob": 0.5',
            '"intensity_range": _HDRI_INTENSITY_RANGE',
        ):
            self.assertIn(expected, camera)
        self.assertIn(
            'FR3_VISUAL_PROFILE_WHITE_ARM_BLACK_PADS = "white_arm_base_ring_black_finger_pads"',
            visuals,
        )
        self.assertIn("finger_pad.Set(_BLACK)", visuals)


if __name__ == "__main__":
    unittest.main()
