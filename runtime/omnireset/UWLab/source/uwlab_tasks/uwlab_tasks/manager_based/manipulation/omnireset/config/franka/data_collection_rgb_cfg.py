# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka FR3 / Panda OmniReset RGB data-collection + eval config.

Mirrors ``ur5e_robotiq_2f85/data_collection_rgb_cfg.py`` but with Franka geometry:
  * 3 ``TiledCameraCfg`` (front / side / wrist) — placeholder extrinsics, *uncalibrated*
  * Wrist camera parented to ``panda_hand`` (not ``robotiq_base_link``)
  * Robot-appearance DR globs ``Robot/.*/visuals`` (mesh_names=[]) since Franka
    visual sub-prim names differ from Robotiq's
  * Observations use ``panda_joint.*`` (7-DOF arm) and ``panda_hand`` link
  * Texture / HDRI yaml files reused from UR5e ``resources/`` (single source of truth)
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from .actions import FrankaFr3GripperAbsoluteJointTargetAction, FrankaFr3GripperRelativeOSCEvalAction
from .rl_state_cfg import (
    FinetuneEvalEventCfg,
    FRANKA_ARM_JOINT_NAMES,
    FrankaFr3GripperRlStateCfg,
    STACKCUBE_XY5_T0_HOME_JOINT_JITTER_RAD,
    STACKCUBE_XY5_T0_INSERTIVE_POSE_RANGES,
    STACKCUBE_XY5_T0_RECEPTIVE_POSE_RANGES,
    STACKCUBE_XY5_T0_TEAM_HOME,
    RlStateSceneCfg,
    TrainEvalEventCfg,
)

# Default: real Franka-base camera setup used for new DP/IL data.
# Set OMNIRESET_CAMERA_SETUP=historical only when reproducing existing 3cam DP artifacts.
# The legacy OMNIRESET_REAL_BASE_CAMERAS=1 is still accepted as an alias for real.
_CAMERA_SETUP = os.environ.get("OMNIRESET_CAMERA_SETUP", "real").strip().lower()
_USE_REAL_BASE_CAMERAS = _CAMERA_SETUP != "historical" or os.environ.get("OMNIRESET_REAL_BASE_CAMERAS", "0") == "1"
_REAL_CAMERA_PROFILE = os.environ.get(
    "OMNIRESET_REAL_CAMERA_PROFILE", "vision_dp_20260715"
).strip().lower()
_AXIS_CAMERA_PROFILES = {
    "axis_3cam_20260822",
    "axis_3cam_side16x9_20260823",
    "axis_3cam_side16x9_wristrear_20260823",
    "axis_3cam_side16x9_wristrear_d405vga_20260825",
    "axis_3cam_side16x9_wristrear_handy_mirror_20260826",
}
if _REAL_CAMERA_PROFILE not in {"vision_dp_20260715", *_AXIS_CAMERA_PROFILES}:
    raise ValueError(
        "Invalid OMNIRESET_REAL_CAMERA_PROFILE="
        f"{_REAL_CAMERA_PROFILE!r}; expected vision_dp_20260715, "
        "axis_3cam_20260822, axis_3cam_side16x9_20260823, or "
        "axis_3cam_side16x9_wristrear_20260823, or "
        "axis_3cam_side16x9_wristrear_d405vga_20260825, or "
        "axis_3cam_side16x9_wristrear_handy_mirror_20260826"
    )
_EXACT_CAMERA_INTRINSICS = os.environ.get("OMNIRESET_EXACT_CAMERA_INTRINSICS", "0") == "1"
_EXACT_WRIST_VERTICAL_FLIP_VALUE = os.environ.get("OMNIRESET_EXACT_WRIST_VERTICAL_FLIP")
if _EXACT_CAMERA_INTRINSICS and _EXACT_WRIST_VERTICAL_FLIP_VALUE is None:
    raise ValueError(
        "Exact camera intrinsics require explicit OMNIRESET_EXACT_WRIST_VERTICAL_FLIP=0 or 1. "
        "Use 0 for new no-flip collection; use 1 only for verified flipped training data. "
        "Do not infer this setting from a dataset or checkpoint filename."
    )
_EXACT_WRIST_VERTICAL_FLIP_VALUE = (
    "0" if _EXACT_WRIST_VERTICAL_FLIP_VALUE is None else _EXACT_WRIST_VERTICAL_FLIP_VALUE.strip()
)
if _EXACT_WRIST_VERTICAL_FLIP_VALUE not in {"0", "1"}:
    raise ValueError("OMNIRESET_EXACT_WRIST_VERTICAL_FLIP must be 0 or 1")
_EXACT_WRIST_VERTICAL_FLIP = (
    _EXACT_CAMERA_INTRINSICS and _EXACT_WRIST_VERTICAL_FLIP_VALUE == "1"
)
if _EXACT_CAMERA_INTRINSICS and (
    not _USE_REAL_BASE_CAMERAS or _REAL_CAMERA_PROFILE not in _AXIS_CAMERA_PROFILES
):
    raise ValueError(
        "OMNIRESET_EXACT_CAMERA_INTRINSICS=1 is only valid with the "
        "Axis real-camera profiles"
    )
_CAMERA_RENDER_PROFILE = os.environ.get(
    "OMNIRESET_CAMERA_RENDER_PROFILE", "native"
).strip().lower()
_CAMERA_RENDER_PROFILE_SCALES = {
    "native": (1.0, 1.0, 1.0),
    # Bulk policy collection keeps the D405 at its calibrated VGA raster while
    # reducing the two 1080p fixed cameras by 2x in each dimension.
    "policy_fast84_v1": (0.5, 0.5, 1.0),
}
if _CAMERA_RENDER_PROFILE not in _CAMERA_RENDER_PROFILE_SCALES:
    raise ValueError(
        "Invalid OMNIRESET_CAMERA_RENDER_PROFILE="
        f"{_CAMERA_RENDER_PROFILE!r}; expected native or policy_fast84_v1"
    )
if _CAMERA_RENDER_PROFILE != "native" and (
    not _EXACT_CAMERA_INTRINSICS
    or not _USE_REAL_BASE_CAMERAS
    or _REAL_CAMERA_PROFILE not in _AXIS_CAMERA_PROFILES
):
    raise ValueError(
        "OMNIRESET_CAMERA_RENDER_PROFILE=policy_fast84_v1 requires exact "
        "intrinsics with an Axis real-camera profile"
    )
_REAL_OBJECT_TABLE_Z_OFFSET = float(os.environ.get("OMNIRESET_REAL_OBJECT_TABLE_Z_OFFSET", "0.0"))
_REAL_TABLE_COVER_COLLISION = os.environ.get("OMNIRESET_REAL_TABLE_COVER_COLLISION", "0") == "1"
_REAL_CAMERA_POSITION_JITTER_M = float(
    os.environ.get("OMNIRESET_REAL_CAMERA_POSITION_JITTER_M", "0.01")
)
_REAL_CAMERA_ROTATION_JITTER_DEG = float(
    os.environ.get("OMNIRESET_REAL_CAMERA_ROTATION_JITTER_DEG", "3.0")
)
_FRONT_CAMERA_POSITION_JITTER_M = float(
    os.environ.get(
        "OMNIRESET_REAL_FRONT_CAMERA_POSITION_JITTER_M",
        str(_REAL_CAMERA_POSITION_JITTER_M),
    )
)
_SIDE_CAMERA_POSITION_JITTER_M = float(
    os.environ.get(
        "OMNIRESET_REAL_SIDE_CAMERA_POSITION_JITTER_M",
        str(_REAL_CAMERA_POSITION_JITTER_M),
    )
)
_WRIST_CAMERA_POSITION_JITTER_M = float(
    os.environ.get(
        "OMNIRESET_REAL_WRIST_CAMERA_POSITION_JITTER_M",
        str(_REAL_CAMERA_POSITION_JITTER_M),
    )
)
_FRONT_CAMERA_ROTATION_JITTER_DEG = float(
    os.environ.get(
        "OMNIRESET_REAL_FRONT_CAMERA_ROTATION_JITTER_DEG",
        str(_REAL_CAMERA_ROTATION_JITTER_DEG),
    )
)
_SIDE_CAMERA_ROTATION_JITTER_DEG = float(
    os.environ.get(
        "OMNIRESET_REAL_SIDE_CAMERA_ROTATION_JITTER_DEG",
        str(_REAL_CAMERA_ROTATION_JITTER_DEG),
    )
)
_WRIST_CAMERA_ROTATION_JITTER_DEG = float(
    os.environ.get(
        "OMNIRESET_REAL_WRIST_CAMERA_ROTATION_JITTER_DEG",
        str(_REAL_CAMERA_ROTATION_JITTER_DEG),
    )
)
if min(
    _FRONT_CAMERA_POSITION_JITTER_M,
    _SIDE_CAMERA_POSITION_JITTER_M,
    _WRIST_CAMERA_POSITION_JITTER_M,
    _FRONT_CAMERA_ROTATION_JITTER_DEG,
    _SIDE_CAMERA_ROTATION_JITTER_DEG,
    _WRIST_CAMERA_ROTATION_JITTER_DEG,
) < 0.0:
    raise ValueError("Real-camera position and rotation jitter must be non-negative")

_RGB_EVENT_PARENT_OVERRIDE = os.environ.get("OMNIRESET_RGB_EVENT_PARENT", "").strip().lower()
if _RGB_EVENT_PARENT_OVERRIDE in {"train", "train_eval", "traineval", "stage1", "historical"}:
    _RGB_EVENT_PARENT = TrainEvalEventCfg
elif _RGB_EVENT_PARENT_OVERRIDE in {"finetune", "finetune_eval", "stage2", "b3", "real"}:
    _RGB_EVENT_PARENT = FinetuneEvalEventCfg
elif _RGB_EVENT_PARENT_OVERRIDE:
    raise ValueError(
        "Invalid OMNIRESET_RGB_EVENT_PARENT="
        f"{_RGB_EVENT_PARENT_OVERRIDE!r}; expected train_eval or finetune_eval"
    )
else:
    # Real-camera vision collection uses the B3 Stage-2 teacher (model_11200),
    # so it must match Stage-2 eval dynamics. Existing historical 3cam DP
    # artifacts were collected/evaluated before this switch and need TrainEval.
    _RGB_EVENT_PARENT = FinetuneEvalEventCfg if _USE_REAL_BASE_CAMERAS else TrainEvalEventCfg

# Reuse UR5e texture / HDRI yamls — single source of truth, scene-agnostic content
_UR5E_RESOURCES_DIR = Path(__file__).parent.parent / "ur5e_robotiq_2f85" / "resources"
_TEXTURE_YAML = str(_UR5E_RESOURCES_DIR / "texture_paths.yaml")
_HDRI_YAML = str(_UR5E_RESOURCES_DIR / "hdri_paths.yaml")
_TEXTURE_OOD_YAML = str(_UR5E_RESOURCES_DIR / "texture_paths_ood.yaml")
_HDRI_OOD_YAML = str(_UR5E_RESOURCES_DIR / "hdri_paths_ood.yaml")
_OBJECT_GREY = (142.0 / 255.0, 144.0 / 255.0, 137.0 / 255.0)  # #8E9089
_OBJECT_GREY_COLOR = {
    "r": (_OBJECT_GREY[0], _OBJECT_GREY[0]),
    "g": (_OBJECT_GREY[1], _OBJECT_GREY[1]),
    "b": (_OBJECT_GREY[2], _OBJECT_GREY[2]),
}
_HDRI_INTENSITY_RANGE = (3000.0, 8000.0) if _USE_REAL_BASE_CAMERAS else (1000.0, 4000.0)


# ---------------------------------------------------------------------------
# Camera extrinsics
# ---------------------------------------------------------------------------
#   front_rgb = L515 fixed, side_rgb = D415 fixed, wrist_rgb = D405 attached to panda_hand.
# The real calibration rotations are converted from ROS/OpenCV camera axes
# (+Z forward, -Y up) to Isaac/USD OpenGL axes (-Z forward, +Y up).
_FRONT_PRIM_PATH = "{ENV_REGEX_NS}/Robot/rgb_front_camera"
_SIDE_PRIM_PATH = "{ENV_REGEX_NS}/Robot/rgb_side_camera"
_FRONT_CAMERA_PATH_TEMPLATE = "/World/envs/env_{}/Robot/rgb_front_camera"
_SIDE_CAMERA_PATH_TEMPLATE = "/World/envs/env_{}/Robot/rgb_side_camera"


def _usd_focal_from_intrinsics(fx: float, fy: float, width: int) -> float:
    return ((fx + fy) * 0.5) / float(width)


def _pinhole_from_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
) -> sim_utils.PinholeCameraCfg:
    return sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        width=width,
        height=height,
    )


def _scale_camera_raster(
    width: int,
    height: int,
    target_fx: float,
    target_fy: float,
    target_cx: float,
    target_cy: float,
    render_fx: float,
    render_fy: float,
    scale: float,
) -> tuple[int, int, float, float, float, float, float, float]:
    """Scale a camera raster and both intrinsic models without changing its rays."""
    scaled_width = max(32, int(round(width * scale)))
    scaled_height = max(32, int(round(height * scale)))
    scale_x = scaled_width / float(width)
    scale_y = scaled_height / float(height)
    return (
        scaled_width,
        scaled_height,
        target_fx * scale_x,
        target_fy * scale_y,
        target_cx * scale_x,
        target_cy * scale_y,
        render_fx * scale_x,
        render_fy * scale_y,
    )


if _USE_REAL_BASE_CAMERAS:
    if _REAL_CAMERA_PROFILE in _AXIS_CAMERA_PROFILES:
        # MuJoCo/Isaac alignment candidate, revised 2026-08-23.
        # Fixed-camera poses are T_base_camera in OpenGL wxyz convention.
        _FRONT_POS = (1.3773505058279372, 0.02255364505239039, 0.7922703611113961)
        _FRONT_ROT = (0.643695719032916, 0.36031314308826273, 0.2998016978983759, 0.6049373539084899)
        _FRONT_WIDTH, _FRONT_HEIGHT = 1920, 1080
        _FRONT_FX, _FRONT_FY = 1340.8660125732422, 1341.780080795288
        _FRONT_CX, _FRONT_CY = 957.9659999907017, 527.8999992460012

        if _REAL_CAMERA_PROFILE in {
            "axis_3cam_side16x9_20260823",
            "axis_3cam_side16x9_wristrear_20260823",
            "axis_3cam_side16x9_wristrear_d405vga_20260825",
            "axis_3cam_side16x9_wristrear_handy_mirror_20260826",
        }:
            # Final side-camera pose saved after the last alignment C adjustment.
            _SIDE_POS = (0.5099174380302429, 0.6603941917419434, 0.30853092670440674)
            _SIDE_ROT = (0.007348625513673981, 0.03912293745378553, 0.6594412154341235, 0.7507014565423548)
            _SIDE_WIDTH, _SIDE_HEIGHT = 1920, 1080
            _SIDE_CX, _SIDE_CY = 960.0, 540.0
        else:
            # Original Axis dataset contract: 4:3 side frame and pre-C pose.
            _SIDE_POS = (0.5738480257871044, 0.7831233856235653, 0.30908943576373943)
            _SIDE_ROT = (0.03141209908244427, 0.0618725364597239, 0.6707546753346415, 0.7384261877669254)
            _SIDE_WIDTH, _SIDE_HEIGHT = 1440, 1080
            _SIDE_CX, _SIDE_CY = 720.0, 540.0
        _SIDE_FX, _SIDE_FY = 1381.245460510254, 1381.2455034255981

        if _REAL_CAMERA_PROFILE in {
            "axis_3cam_side16x9_wristrear_20260823",
            "axis_3cam_side16x9_wristrear_d405vga_20260825",
            "axis_3cam_side16x9_wristrear_handy_mirror_20260826",
        }:
            # Mirror the D405 mount from the workspace side to the base side of
            # panda_hand by rotating the complete mount 180 deg about hand +Z.
            _WRIST_POS = (0.158257909, 0.006837764, 0.047477325)
            _WRIST_ROT = (-0.283554846, -0.656110765, -0.639403111, -0.283335444)
        else:
            # Original front-mounted D405 transform retained for old datasets.
            _WRIST_POS = (-0.158257909, -0.006837764, 0.047477325)
            _WRIST_ROT = (-0.283335444, -0.639403111, 0.656110765, 0.283554846)
        if _REAL_CAMERA_PROFILE == "axis_3cam_side16x9_wristrear_handy_mirror_20260826":
            # Mirror the complete wrist-camera mount across the panda_hand local
            # XZ plane while preserving a right-handed camera frame.
            _WRIST_POS = (0.158257909, -0.006837764, 0.047477325)
            _WRIST_ROT = (0.283335444, 0.639403111, 0.656110765, 0.283554846)
        if _REAL_CAMERA_PROFILE in {
            "axis_3cam_side16x9_wristrear_d405vga_20260825",
            "axis_3cam_side16x9_wristrear_handy_mirror_20260826",
        }:
            # Intrinsics reported by D405 serial 260322272725 at 640x480.
            _WRIST_WIDTH, _WRIST_HEIGHT = 640, 480
            _WRIST_FX, _WRIST_FY = 437.7700500488281, 436.719482421875
            _WRIST_CX, _WRIST_CY = 323.0035095214844, 234.05731201171875
        else:
            _WRIST_WIDTH, _WRIST_HEIGHT = 1440, 1080
            _WRIST_FX, _WRIST_FY = 888.7927436828613, 886.6597008705139
            _WRIST_CX, _WRIST_CY = 726.0974999517202, 527.935499958694
    else:
        # Camera geometry used by the existing 2026-07/08 Vision DP datasets.
        _FRONT_POS = (1.252999568, 0.046552280, 1.022142743)  # L515
        _FRONT_ROT = (0.658530788, 0.240652938, 0.263012492, 0.662757718)
        _FRONT_WIDTH, _FRONT_HEIGHT = 960, 540
        _FRONT_FX, _FRONT_FY = 670.43300, 670.89000
        _FRONT_CX, _FRONT_CY = 478.98300, 263.95000

        _SIDE_POS = (0.742531964, -0.607496677, 0.375367773)
        _SIDE_ROT = (0.767350171, 0.636200032, 0.070531898, 0.038058987)
        _SIDE_WIDTH, _SIDE_HEIGHT = 960, 540
        _SIDE_FX, _SIDE_FY = 679.96875, 678.28275
        _SIDE_CX, _SIDE_CY = 476.41500, 263.39775

        _WRIST_POS = (-0.197695705, 0.013046155, 0.022151960)
        _WRIST_ROT = (0.274094481, 0.657434782, -0.629528490, -0.310395882)
        _WRIST_WIDTH, _WRIST_HEIGHT = 640, 480
        _WRIST_FX, _WRIST_FY = 395.01900, 394.07100
        _WRIST_CX, _WRIST_CY = 322.71000, 234.63800

    if _EXACT_CAMERA_INTRINSICS:
        # Isaac forces fx=fy and centers the principal point. Render a slightly
        # wider intermediate image so the exact target rays remain in bounds.
        _FRONT_RENDER_FX = _FRONT_RENDER_FY = 1300.0
        _SIDE_RENDER_FX = _SIDE_RENDER_FY = 1350.0
        _WRIST_RENDER_FX = _WRIST_RENDER_FY = (
            420.0
            if _REAL_CAMERA_PROFILE
            in {
                "axis_3cam_side16x9_wristrear_d405vga_20260825",
                "axis_3cam_side16x9_wristrear_handy_mirror_20260826",
            }
            else 850.0
        )
    else:
        _FRONT_RENDER_FX, _FRONT_RENDER_FY = _FRONT_FX, _FRONT_FY
        _SIDE_RENDER_FX, _SIDE_RENDER_FY = _SIDE_FX, _SIDE_FY
        _WRIST_RENDER_FX, _WRIST_RENDER_FY = _WRIST_FX, _WRIST_FY

    _FRONT_RENDER_SCALE, _SIDE_RENDER_SCALE, _WRIST_RENDER_SCALE = (
        _CAMERA_RENDER_PROFILE_SCALES[_CAMERA_RENDER_PROFILE]
    )
    (
        _FRONT_WIDTH,
        _FRONT_HEIGHT,
        _FRONT_FX,
        _FRONT_FY,
        _FRONT_CX,
        _FRONT_CY,
        _FRONT_RENDER_FX,
        _FRONT_RENDER_FY,
    ) = _scale_camera_raster(
        _FRONT_WIDTH,
        _FRONT_HEIGHT,
        _FRONT_FX,
        _FRONT_FY,
        _FRONT_CX,
        _FRONT_CY,
        _FRONT_RENDER_FX,
        _FRONT_RENDER_FY,
        _FRONT_RENDER_SCALE,
    )
    (
        _SIDE_WIDTH,
        _SIDE_HEIGHT,
        _SIDE_FX,
        _SIDE_FY,
        _SIDE_CX,
        _SIDE_CY,
        _SIDE_RENDER_FX,
        _SIDE_RENDER_FY,
    ) = _scale_camera_raster(
        _SIDE_WIDTH,
        _SIDE_HEIGHT,
        _SIDE_FX,
        _SIDE_FY,
        _SIDE_CX,
        _SIDE_CY,
        _SIDE_RENDER_FX,
        _SIDE_RENDER_FY,
        _SIDE_RENDER_SCALE,
    )
    (
        _WRIST_WIDTH,
        _WRIST_HEIGHT,
        _WRIST_FX,
        _WRIST_FY,
        _WRIST_CX,
        _WRIST_CY,
        _WRIST_RENDER_FX,
        _WRIST_RENDER_FY,
    ) = _scale_camera_raster(
        _WRIST_WIDTH,
        _WRIST_HEIGHT,
        _WRIST_FX,
        _WRIST_FY,
        _WRIST_CX,
        _WRIST_CY,
        _WRIST_RENDER_FX,
        _WRIST_RENDER_FY,
        _WRIST_RENDER_SCALE,
    )

    _FRONT_FOCAL = _usd_focal_from_intrinsics(_FRONT_RENDER_FX, _FRONT_RENDER_FY, _FRONT_WIDTH)
    _FRONT_SPAWN = _pinhole_from_intrinsics(
        fx=_FRONT_RENDER_FX, fy=_FRONT_RENDER_FY,
        cx=_FRONT_WIDTH / 2.0, cy=_FRONT_HEIGHT / 2.0,
        width=_FRONT_WIDTH, height=_FRONT_HEIGHT,
    )
    _FRONT_FOCAL_RANGE = (_FRONT_FOCAL, _FRONT_FOCAL)
    _SIDE_FOCAL = _usd_focal_from_intrinsics(_SIDE_RENDER_FX, _SIDE_RENDER_FY, _SIDE_WIDTH)
    _SIDE_SPAWN = _pinhole_from_intrinsics(
        fx=_SIDE_RENDER_FX, fy=_SIDE_RENDER_FY,
        cx=_SIDE_WIDTH / 2.0, cy=_SIDE_HEIGHT / 2.0,
        width=_SIDE_WIDTH, height=_SIDE_HEIGHT,
    )
    _SIDE_FOCAL_RANGE = (_SIDE_FOCAL, _SIDE_FOCAL)
    _WRIST_FOCAL = _usd_focal_from_intrinsics(_WRIST_RENDER_FX, _WRIST_RENDER_FY, _WRIST_WIDTH)
    _WRIST_SPAWN = _pinhole_from_intrinsics(
        fx=_WRIST_RENDER_FX, fy=_WRIST_RENDER_FY,
        cx=_WRIST_WIDTH / 2.0, cy=_WRIST_HEIGHT / 2.0,
        width=_WRIST_WIDTH, height=_WRIST_HEIGHT,
    )
    _WRIST_FOCAL_RANGE = (_WRIST_FOCAL, _WRIST_FOCAL)
    _WRIST_PRIM_PATH = "{ENV_REGEX_NS}/Robot/panda_hand/rgb_wrist_camera"
    _WRIST_CAMERA_PATH_TEMPLATE = "/World/envs/env_{}/Robot/panda_hand/rgb_wrist_camera"

    # Push the side-wall seams away from the L515/D415 view centers; otherwise
    # the wall edges render as black pillar-like bars in the camera images.
    _CURTAIN_LEFT_POS = (0.45, -1.55, 0.80)
    _CURTAIN_RIGHT_POS = (0.45, 1.55, 0.80)
    _CURTAIN_BACK_POS = (-0.45, 0.0, 0.80)
    _CURTAIN_FRONT_POS = (1.45, 0.0, 0.80)
    _CURTAIN_SIDE_ROT = (1.0, 0.0, 0.0, 0.0)
    _CURTAIN_SIDE_SIZE = (6.00, 0.001, 1.80)
    _CURTAIN_BACK_SIZE = (0.001, 5.00, 1.80)
    _CURTAIN_FRONT_SIZE = (0.001, 3.70, 1.80)
    _CURTAIN_TOP_POS = (0.45, 0.0, 1.46)
    _CURTAIN_TOP_SIZE = (2.20, 2.20, 0.01)
    _CURTAIN_COLOR = (0.86, 0.86, 0.84)
    _CURTAIN_FLOOR_COLOR = (0.80, 0.80, 0.78)
    _CURTAIN_FLOOR_POS = (0.45, 0.0, _REAL_OBJECT_TABLE_Z_OFFSET - 0.010)
    _CURTAIN_FLOOR_SIZE = (2.70, 5.00, 0.001)
    _TABLE_COVER_THICKNESS = 0.004
    _TABLE_COVER_POS = (0.64, 0.0, _REAL_OBJECT_TABLE_Z_OFFSET - 0.5 * _TABLE_COVER_THICKNESS)
    _TABLE_COVER_SIZE = (1.00, 1.60, _TABLE_COVER_THICKNESS)
    _TABLE_COVER_COLOR = (0.015, 0.015, 0.014)
    _FRANKA_TABLE_COVER_POS = (-0.28, 0.0, -0.004)
    _FRANKA_TABLE_COVER_SIZE = (0.84, 0.86, 0.008)
    _FRANKA_TABLE_COVER_COLOR = (0.018, 0.018, 0.017)
    _FRANKA_TABLE_FRONT_EDGE_POS = (0.137, 0.0, 0.5 * _REAL_OBJECT_TABLE_Z_OFFSET)
    _FRANKA_TABLE_FRONT_EDGE_SIZE = (0.014, 0.86, abs(_REAL_OBJECT_TABLE_Z_OFFSET))

    _FRONT_CAMERA_POSITION_DELTAS = {
        axis: (-_FRONT_CAMERA_POSITION_JITTER_M, _FRONT_CAMERA_POSITION_JITTER_M)
        for axis in ("x", "y", "z")
    }
    _SIDE_CAMERA_POSITION_DELTAS = {
        axis: (-_SIDE_CAMERA_POSITION_JITTER_M, _SIDE_CAMERA_POSITION_JITTER_M)
        for axis in ("x", "y", "z")
    }
    _FRONT_CAMERA_EULER_DELTAS = {
        axis: (-_FRONT_CAMERA_ROTATION_JITTER_DEG, _FRONT_CAMERA_ROTATION_JITTER_DEG)
        for axis in ("pitch", "yaw", "roll")
    }
    _SIDE_CAMERA_EULER_DELTAS = {
        axis: (-_SIDE_CAMERA_ROTATION_JITTER_DEG, _SIDE_CAMERA_ROTATION_JITTER_DEG)
        for axis in ("pitch", "yaw", "roll")
    }
    _WRIST_CAMERA_POSITION_DELTAS = {
        axis: (-_WRIST_CAMERA_POSITION_JITTER_M, _WRIST_CAMERA_POSITION_JITTER_M)
        for axis in ("x", "y", "z")
    }
    _WRIST_CAMERA_EULER_DELTAS = {
        axis: (-_WRIST_CAMERA_ROTATION_JITTER_DEG, _WRIST_CAMERA_ROTATION_JITTER_DEG)
        for axis in ("pitch", "yaw", "roll")
    }
    _RGB_ENV_SPACING = 4.0
else:
    # Front / side cams: poses borrowed from UR5e (both robots have base at world
    # origin and workspace in +x); good enough for first-pass rendering. Re-tune
    # via the OmniReset align_cameras workflow once you set up real cameras.
    _FRONT_POS = (1.0770121, -0.1679045, 0.4486344)
    _FRONT_ROT = (0.70564552, 0.46613815, 0.25072644, 0.47107948)
    _FRONT_WIDTH, _FRONT_HEIGHT = 320, 240
    _FRONT_FOCAL = 13.20
    _FRONT_SPAWN = sim_utils.PinholeCameraCfg(focal_length=_FRONT_FOCAL)
    _FRONT_FOCAL_RANGE = (_FRONT_FOCAL - 2.0, _FRONT_FOCAL + 2.0)

    _SIDE_POS = (0.8323904, 0.5877843, 0.2805111)  # was (0.4,...): wrong x -> restored to UR5e value
    _SIDE_ROT = (0.29008842, 0.22122445, 0.51336143, 0.77676798)
    _SIDE_WIDTH, _SIDE_HEIGHT = 320, 240
    _SIDE_FOCAL = 20.10
    _SIDE_SPAWN = sim_utils.PinholeCameraCfg(focal_length=_SIDE_FOCAL)
    _SIDE_FOCAL_RANGE = (_SIDE_FOCAL - 2.0, _SIDE_FOCAL + 2.0)

    # panda_hand frame: Y=finger-closing, Z=palm->fingertip(+z toward TCP@0.1034), X=thin side.
    # Eye-in-hand: offset to the +X side of the hand and look at the TCP/grasp point.
    _WRIST_POS = (0.07, 0.0, 0.02)
    _WRIST_ROT = (0.24189, 0.66445, 0.66445, 0.24189)  # look-at TCP from +X side
    _WRIST_WIDTH, _WRIST_HEIGHT = 320, 240
    _WRIST_FOCAL = 24.55
    _WRIST_SPAWN = sim_utils.PinholeCameraCfg(focal_length=_WRIST_FOCAL)
    _WRIST_FOCAL_RANGE = (_WRIST_FOCAL - 1.0, _WRIST_FOCAL + 1.0)
    _WRIST_PRIM_PATH = "{ENV_REGEX_NS}/Robot/panda_hand/rgb_wrist_camera"
    _WRIST_CAMERA_PATH_TEMPLATE = "/World/envs/env_{}/Robot/panda_hand/rgb_wrist_camera"

    _CURTAIN_LEFT_POS = (0.4, -0.68, 0.519)
    _CURTAIN_RIGHT_POS = (0.4, 0.68, 0.519)
    _CURTAIN_BACK_POS = (-0.15, 0.0, 0.519)
    _CURTAIN_FRONT_POS = (0.95, 0.0, 0.519)
    _CURTAIN_SIDE_ROT = (0.707, 0.0, 0.0, -0.707)
    _CURTAIN_SIDE_SIZE = (0.01, 1.0, 1.125)
    _CURTAIN_BACK_SIZE = (0.01, 1.3, 1.125)
    _CURTAIN_FRONT_SIZE = (0.01, 1.3, 1.125)
    _CURTAIN_TOP_POS = (0.4, 0.0, 1.20)
    _CURTAIN_TOP_SIZE = (1.0, 1.3, 0.01)
    _CURTAIN_COLOR = (0.0, 0.0, 0.0)
    _CURTAIN_FLOOR_COLOR = (0.0, 0.0, 0.0)
    _CURTAIN_FLOOR_POS = (0.45, 0.0, 0.0)
    _CURTAIN_FLOOR_SIZE = (1.4, 1.6, 0.001)
    _TABLE_COVER_POS = (0.45, 0.0, 0.001)
    _TABLE_COVER_SIZE = (1.40, 1.60, 0.004)
    _TABLE_COVER_COLOR = (0.38, 0.44, 0.49)

    _FRONT_CAMERA_POSITION_DELTAS = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}
    _SIDE_CAMERA_POSITION_DELTAS = dict(_FRONT_CAMERA_POSITION_DELTAS)
    _BASE_CAMERA_EULER_DELTAS = {"pitch": (-2.0, 2.0), "yaw": (-2.0, 2.0), "roll": (-2.0, 2.0)}
    _FRONT_CAMERA_EULER_DELTAS = dict(_BASE_CAMERA_EULER_DELTAS)
    _SIDE_CAMERA_EULER_DELTAS = dict(_BASE_CAMERA_EULER_DELTAS)
    _WRIST_CAMERA_POSITION_DELTAS = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)}
    _WRIST_CAMERA_EULER_DELTAS = {"pitch": (-3.0, 3.0), "yaw": (-3.0, 3.0), "roll": (-3.0, 3.0)}
    _RGB_ENV_SPACING = 1.5


@configclass
class DataCollectionRGBObjectSceneCfg(RlStateSceneCfg):
    """Scene: Franka RL state scene + 4 curtains + 3 cameras."""

    curtain_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainLeft",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CURTAIN_LEFT_POS, rot=_CURTAIN_SIDE_ROT),
        spawn=sim_utils.CuboidCfg(
            size=_CURTAIN_SIDE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_CURTAIN_COLOR),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    curtain_back = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainBack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CURTAIN_BACK_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=_CURTAIN_BACK_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_CURTAIN_COLOR),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    curtain_front = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainFront",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CURTAIN_FRONT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=_CURTAIN_FRONT_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_CURTAIN_COLOR),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    curtain_right = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainRight",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_CURTAIN_RIGHT_POS, rot=_CURTAIN_SIDE_ROT),
        spawn=sim_utils.CuboidCfg(
            size=_CURTAIN_SIDE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_CURTAIN_COLOR),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    if _USE_REAL_BASE_CAMERAS:
        curtain_floor = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/CurtainFloor",
            init_state=RigidObjectCfg.InitialStateCfg(pos=_CURTAIN_FLOOR_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=_CURTAIN_FLOOR_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_CURTAIN_FLOOR_COLOR),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
        )

        table_cover = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/TableCover",
            init_state=RigidObjectCfg.InitialStateCfg(pos=_TABLE_COVER_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=_TABLE_COVER_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_TABLE_COVER_COLOR),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=_REAL_TABLE_COVER_COLLISION),
            ),
        )

        franka_table_cover = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/FrankaTableCover",
            init_state=RigidObjectCfg.InitialStateCfg(pos=_FRANKA_TABLE_COVER_POS, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=_FRANKA_TABLE_COVER_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_FRANKA_TABLE_COVER_COLOR),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            ),
        )

        if abs(_REAL_OBJECT_TABLE_Z_OFFSET) > 1e-6:
            franka_table_front_edge = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/FrankaTableFrontEdge",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=_FRANKA_TABLE_FRONT_EDGE_POS, rot=(1.0, 0.0, 0.0, 0.0)
                ),
                spawn=sim_utils.CuboidCfg(
                    size=_FRANKA_TABLE_FRONT_EDGE_SIZE,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_FRANKA_TABLE_COVER_COLOR),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                ),
            )

    front_camera = TiledCameraCfg(
        prim_path=_FRONT_PRIM_PATH,
        update_period=0,
        height=_FRONT_HEIGHT,
        width=_FRONT_WIDTH,
        offset=TiledCameraCfg.OffsetCfg(pos=_FRONT_POS, rot=_FRONT_ROT, convention="opengl"),
        data_types=["rgb"],
        spawn=_FRONT_SPAWN,
    )

    side_camera = TiledCameraCfg(
        prim_path=_SIDE_PRIM_PATH,
        update_period=0,
        height=_SIDE_HEIGHT,
        width=_SIDE_WIDTH,
        offset=TiledCameraCfg.OffsetCfg(pos=_SIDE_POS, rot=_SIDE_ROT, convention="opengl"),
        data_types=["rgb"],
        spawn=_SIDE_SPAWN,
    )

    wrist_camera = TiledCameraCfg(
        prim_path=_WRIST_PRIM_PATH,
        update_period=0,
        height=_WRIST_HEIGHT,
        width=_WRIST_WIDTH,
        offset=TiledCameraCfg.OffsetCfg(pos=_WRIST_POS, rot=_WRIST_ROT, convention="opengl"),
        data_types=["rgb"],
        spawn=_WRIST_SPAWN,
    )


# ---------------------------------------------------------------------------
# Event configs: parent = _RGB_EVENT_PARENT. The current default is
# FinetuneEvalEventCfg for B3 Stage-2 vision data collection; see the top of
# this file before switching it for legacy Stage-1 reproduction.
# ---------------------------------------------------------------------------
@configclass
class BaseRGBEventCfg(_RGB_EVENT_PARENT):
    """RGB events: Franka eval physics + camera pose & focal randomization."""

    randomize_front_camera = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": _FRONT_CAMERA_PATH_TEMPLATE,
            "base_position": _FRONT_POS,
            "base_rotation": _FRONT_ROT,
            "position_deltas": _FRONT_CAMERA_POSITION_DELTAS,
            "euler_deltas": _FRONT_CAMERA_EULER_DELTAS,
        },
    )

    randomize_front_camera_focal_length = EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={
            "camera_path_template": _FRONT_CAMERA_PATH_TEMPLATE,
            "focal_length_range": _FRONT_FOCAL_RANGE,
        },
    )

    randomize_side_camera = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": _SIDE_CAMERA_PATH_TEMPLATE,
            "base_position": _SIDE_POS,
            "base_rotation": _SIDE_ROT,
            "position_deltas": _SIDE_CAMERA_POSITION_DELTAS,
            "euler_deltas": _SIDE_CAMERA_EULER_DELTAS,
        },
    )

    randomize_side_camera_focal_length = EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={
            "camera_path_template": _SIDE_CAMERA_PATH_TEMPLATE,
            "focal_length_range": _SIDE_FOCAL_RANGE,
        },
    )

    randomize_wrist_camera = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": _WRIST_CAMERA_PATH_TEMPLATE,
            "base_position": _WRIST_POS,
            "base_rotation": _WRIST_ROT,
            "position_deltas": _WRIST_CAMERA_POSITION_DELTAS,
            "euler_deltas": _WRIST_CAMERA_EULER_DELTAS,
        },
    )

    randomize_wrist_camera_focal_length = EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={
            "camera_path_template": _WRIST_CAMERA_PATH_TEMPLATE,
            "focal_length_range": _WRIST_FOCAL_RANGE,
        },
    )


@configclass
class RGBEventCfg(BaseRGBEventCfg):
    """Full DR: camera + robot/object/table/curtain appearance + HDRI sky light.

    Mesh-specific events from UR5e (``randomize_wrist_mount_appearance``,
    ``randomize_inner_finger_appearance``) are replaced by a single
    ``randomize_robot_appearance`` with ``mesh_names=[]`` that globs every
    ``Robot/*/visuals`` prim. Avoids hard-coding Franka visual sub-prim names.
    """

    # Keep Franka visually stable for sim-real camera matching. Physical material
    # randomization from the parent event cfg remains independent of this.
    randomize_robot_appearance = None

    randomize_insertive_object_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "event_name": "randomize_insertive_object_event",
            "mesh_names": [],
            "texture_prob": 0.0,
            "colors": _OBJECT_GREY_COLOR,
            "roughness_range": (0.25, 0.85),
            "metallic_range": (0.0, 0.2),
            "specular_range": (0.0, 0.6),
        },
    )

    randomize_receptive_object_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "event_name": "randomize_receptive_object_event",
            "mesh_names": [],
            "texture_prob": 0.0,
            "colors": _OBJECT_GREY_COLOR,
            "roughness_range": (0.25, 0.85),
            "metallic_range": (0.0, 0.2),
            "specular_range": (0.0, 0.6),
        },
    )

    # Keep the table visually stable; curtain/wall appearance DR below remains on.
    randomize_table_appearance = None

    randomize_curtain_left_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("curtain_left"),
            "event_name": "randomize_curtain_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": _TEXTURE_YAML,
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_curtain_back_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("curtain_back"),
            "event_name": "randomize_curtain_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": _TEXTURE_YAML,
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_curtain_front_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("curtain_front"),
            "event_name": "randomize_curtain_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": _TEXTURE_YAML,
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_curtain_right_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("curtain_right"),
            "event_name": "randomize_curtain_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": _TEXTURE_YAML,
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    if _USE_REAL_BASE_CAMERAS:
        randomize_curtain_floor_appearance = EventTerm(
            func=task_mdp.randomize_visual_appearance_multiple_meshes,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("curtain_floor"),
                "event_name": "randomize_curtain_event",
                "mesh_names": [],
                "texture_prob": 0.5,
                "texture_config_path": _TEXTURE_YAML,
                "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
                "texture_scale_range": (0.7, 5.0),
                "roughness_range": (0.0, 1.0),
                "metallic_range": (0.0, 1.0),
                "specular_range": (0.0, 1.0),
            },
        )

    randomize_sky_light = EventTerm(
        func=task_mdp.randomize_hdri,
        mode="startup",
        params={
            "light_path": "/World/skyLight",
            "hdri_config_path": _HDRI_YAML,
            "intensity_range": _HDRI_INTENSITY_RANGE,
            "rotation_range": (0.0, 360.0),
        },
    )

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": "./Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "rigid_object_position_offsets": {
                "insertive_object": (0.0, 0.0, 0.0),
                "receptive_object": (0.0, 0.0, 0.0),
            },
        },
    )


@configclass
class DataCollectionRGBEventCfg(RGBEventCfg):
    """Data collection: sample from the 4-path reset mix (mirrors UR5e)."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": "./Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
            "rigid_object_position_offsets": {
                "insertive_object": (0.0, 0.0, 0.0),
                "receptive_object": (0.0, 0.0, 0.0),
            },
        },
    )


@configclass
class XY5T0RGBEventCfg(RGBEventCfg):
    """RGB randomization with the online XY5 t0 physical reset contract."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.StackCubeXY5T0OnlineReset,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "arm_joint_names": FRANKA_ARM_JOINT_NAMES,
            "team_home_arm_joint_positions": STACKCUBE_XY5_T0_TEAM_HOME,
            "arm_joint_position_offset_range": (
                -STACKCUBE_XY5_T0_HOME_JOINT_JITTER_RAD,
                STACKCUBE_XY5_T0_HOME_JOINT_JITTER_RAD,
            ),
            "gripper_joint_names": ["panda_finger_joint1", "panda_finger_joint2"],
            "gripper_joint_positions": (0.04, 0.04),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "insertive_pose_ranges": STACKCUBE_XY5_T0_INSERTIVE_POSE_RANGES,
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "receptive_pose_ranges": STACKCUBE_XY5_T0_RECEPTIVE_POSE_RANGES,
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


# ---------------------------------------------------------------------------
# Commands / Observations / Terminations
# ---------------------------------------------------------------------------
@configclass
class RGBCommandsCfg:
    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )


@configclass
class RGBObservationsCfg:
    @configclass
    class RGBPolicyCfg(ObsGroup):
        """Policy observations (processed images for eval)."""

        last_gripper_action = ObsTerm(func=task_mdp.last_action, params={"action_name": "gripper"})
        last_arm_action = ObsTerm(func=task_mdp.last_action, params={"action_name": "arm"})

        arm_joint_pos = ObsTerm(
            func=task_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
        )

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224),
                "target_intrinsics": (_FRONT_FX, _FRONT_FY, _FRONT_CX, _FRONT_CY)
                if _EXACT_CAMERA_INTRINSICS else None,
            },
        )

        side_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("side_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224),
                "target_intrinsics": (_SIDE_FX, _SIDE_FY, _SIDE_CX, _SIDE_CY)
                if _EXACT_CAMERA_INTRINSICS else None,
            },
        )

        wrist_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224),
                "target_intrinsics": (_WRIST_FX, _WRIST_FY, _WRIST_CX, _WRIST_CY)
                if _EXACT_CAMERA_INTRINSICS else None,
                "vertical_flip": _EXACT_WRIST_VERTICAL_FLIP,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class RGBDataCollectionCfg(ObsGroup):
        """Data-collection observations (raw images saved as int8)."""

        last_gripper_action = ObsTerm(func=task_mdp.last_action, params={"action_name": "gripper"})
        last_arm_action = ObsTerm(func=task_mdp.last_action, params={"action_name": "arm"})

        arm_joint_pos = ObsTerm(
            func=task_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])},
        )

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "process_image": False,
                "output_size": (224, 224),
                "target_intrinsics": (_FRONT_FX, _FRONT_FY, _FRONT_CX, _FRONT_CY)
                if _EXACT_CAMERA_INTRINSICS else None,
            },
        )

        side_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("side_camera"),
                "data_type": "rgb",
                "process_image": False,
                "output_size": (224, 224),
                "target_intrinsics": (_SIDE_FX, _SIDE_FY, _SIDE_CX, _SIDE_CY)
                if _EXACT_CAMERA_INTRINSICS else None,
            },
        )

        wrist_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "process_image": False,
                "output_size": (224, 224),
                "target_intrinsics": (_WRIST_FX, _WRIST_FY, _WRIST_CX, _WRIST_CY)
                if _EXACT_CAMERA_INTRINSICS else None,
                "vertical_flip": _EXACT_WRIST_VERTICAL_FLIP,
            },
        )

        binary_contact = ObsTerm(
            func=task_mdp.binary_force_contact,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "panda_hand",
                "force_threshold": 25.0,
            },
        )

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: RGBPolicyCfg = RGBPolicyCfg()
    data_collection: RGBDataCollectionCfg = RGBDataCollectionCfg()


@configclass
class DataCollectionRGBTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)
    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)
    corrupted_camera = DoneTerm(
        func=task_mdp.corrupted_camera_detected,
        params={"camera_names": ["front_camera", "side_camera", "wrist_camera"], "std_threshold": 10.0},
    )
    early_success = DoneTerm(
        func=task_mdp.early_success_termination,
        params={"num_consecutive_successes": 5, "min_episode_length": 10},
    )
    success = DoneTerm(
        func=task_mdp.consecutive_success_state_with_min_length,
        params={"num_consecutive_successes": 5, "min_episode_length": 10},
    )


# ---------------------------------------------------------------------------
# Top-level configs (eval / data-collection)
# ---------------------------------------------------------------------------
@configclass
class FrankaFr3GripperRGBRelCartesianOSCEvalCfg(FrankaFr3GripperRlStateCfg):
    """Franka RGB eval base: Franka state cfg + RGB scene/obs/term/render."""

    actions: FrankaFr3GripperRelativeOSCEvalAction = FrankaFr3GripperRelativeOSCEvalAction()
    scene: DataCollectionRGBObjectSceneCfg = DataCollectionRGBObjectSceneCfg(
        num_envs=32, env_spacing=_RGB_ENV_SPACING, replicate_physics=False
    )
    observations: RGBObservationsCfg = RGBObservationsCfg()
    terminations: DataCollectionRGBTerminationsCfg = DataCollectionRGBTerminationsCfg()
    commands: RGBCommandsCfg = RGBCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 32.0

        # Render settings (matches UR5e RGB cfg)
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.antialiasing_mode = "DLAA"

        # Speeds up rendering
        self.sim.render_interval = self.decimation

        # Re-render on reset so DR takes effect immediately
        self.num_rerenders_on_reset = 1


@configclass
class FrankaFr3GripperDataCollectionRGBRelCartesianOSCCfg(FrankaFr3GripperRGBRelCartesianOSCEvalCfg):
    """Data-collection variant: 4-path reset mix."""

    events: DataCollectionRGBEventCfg = DataCollectionRGBEventCfg()


@configclass
class FrankaFr3GripperDataCollectionRGBRelCartesianOSCXY5T0Cfg(
    FrankaFr3GripperRGBRelCartesianOSCEvalCfg
):
    """RGB collection with continuous XY5 object and team-home joint resets."""

    events: XY5T0RGBEventCfg = XY5T0RGBEventCfg()


@configclass
class FrankaFr3GripperEvalRGBRelCartesianOSCCfg(FrankaFr3GripperRGBRelCartesianOSCEvalCfg):
    """Eval variant: 1-path reset (ObjectAnywhereEEAnywhere)."""

    events: RGBEventCfg = RGBEventCfg()


@configclass
class JointTargetRGBEventCfg(RGBEventCfg):
    """B3 RGB events without the OSC-only gain event."""

    randomize_osc_gains = None


@configclass
class FrankaFr3GripperEvalRGBAbsoluteJointTargetCfg(FrankaFr3GripperRGBRelCartesianOSCEvalCfg):
    """RGB evaluation environment accepting absolute 7-axis targets plus gripper width."""

    actions: FrankaFr3GripperAbsoluteJointTargetAction = FrankaFr3GripperAbsoluteJointTargetAction()
    events: JointTargetRGBEventCfg = JointTargetRGBEventCfg()

    def __post_init__(self):
        super().__post_init__()
        for actuator_name in ("panda_arm1", "panda_arm2"):
            self.scene.robot.actuators[actuator_name].stiffness = 80.0
            self.scene.robot.actuators[actuator_name].damping = 4.0


# ---------------------------------------------------------------------------
# OOD variants (used by eval_distilled_policy.py for robustness eval)
# ---------------------------------------------------------------------------
@configclass
class OODRGBEventCfg(BaseRGBEventCfg):
    """Same DR as RGBEventCfg but uses OOD texture/HDRI yaml pools."""

    randomize_robot_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "event_name": "randomize_robot_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": _TEXTURE_OOD_YAML,
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    randomize_insertive_object_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "event_name": "randomize_insertive_object_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": _TEXTURE_OOD_YAML,
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    randomize_receptive_object_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "event_name": "randomize_receptive_object_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": _TEXTURE_OOD_YAML,
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    randomize_table_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "event_name": "randomize_table_event",
            "mesh_names": ["visuals/vention_mat"],
            "texture_prob": 0.5,
            "texture_config_path": _TEXTURE_OOD_YAML,
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    randomize_sky_light = EventTerm(
        func=task_mdp.randomize_hdri,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "light_path": "/World/skyLight",
            "hdri_config_path": _HDRI_OOD_YAML,
            "intensity_range": _HDRI_INTENSITY_RANGE,
            "rotation_range": (0.0, 360.0),
        },
    )

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": "./Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class FrankaFr3GripperEvalRGBRelCartesianOSCOODCfg(FrankaFr3GripperEvalRGBRelCartesianOSCCfg):
    """Eval with OOD textures/HDRIs."""

    events: OODRGBEventCfg = OODRGBEventCfg()
