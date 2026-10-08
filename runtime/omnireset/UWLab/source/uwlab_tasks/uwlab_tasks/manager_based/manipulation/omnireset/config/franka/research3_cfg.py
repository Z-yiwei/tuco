# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent OmniReset configs for the official Franka Research 3 asset."""

from __future__ import annotations

from typing import Any

from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from . import data_collection_rgb_cfg as panda_rgb
from . import rl_state_cfg as panda_state
from .research3_actions import (
    FrankaResearch3AbsoluteJointTargetAction,
    FrankaResearch3RelativeDiffIKJointTargetAction,
    FrankaResearch3RelativeOSCEvalAction,
)
from .research3_assets_cfg import FRANKA_RESEARCH3_CFG, FRANKA_RESEARCH3_MIMIC_FINGERTIP_CFG
from .research3_mimic_fingertips import spawn_research3_with_mimic_fingertips


_NAME_REPLACEMENTS = (
    ("panda_joint", "fr3_joint"),
    ("panda_finger", "fr3_finger"),
    ("panda_hand", "fr3_hand"),
)


def _replace_robot_names(value: Any, seen: set[int] | None = None) -> Any:
    """Remap inherited manager config values while preserving actuator dictionary keys."""
    if isinstance(value, str):
        for source, target in _NAME_REPLACEMENTS:
            value = value.replace(source, target)
        return value
    if value is None or isinstance(value, (int, float, bool, bytes)) or callable(value):
        return value
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return value
    seen.add(value_id)
    if isinstance(value, list):
        value[:] = [_replace_robot_names(item, seen) for item in value]
    elif isinstance(value, tuple):
        value = tuple(_replace_robot_names(item, seen) for item in value)
    elif isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _replace_robot_names(item, seen)
    elif hasattr(value, "__dict__") and not isinstance(value, type):
        for name, item in vars(value).items():
            setattr(value, name, _replace_robot_names(item, seen))
    return value


def _finalize_research3_config(cfg: Any) -> None:
    _replace_robot_names(cfg)
    if "FrankaFR3/fr3.usd" not in cfg.scene.robot.spawn.usd_path:
        raise ValueError("Research3 task must use the official FrankaFR3/fr3.usd asset")
    if cfg.scene.robot.actuators["panda_arm1"].joint_names_expr != ["fr3_joint[1-4]"]:
        raise ValueError("Research3 arm actuator names were not resolved")


def _finalize_mimic_fingertip_config(cfg: Any) -> None:
    _finalize_research3_config(cfg)
    if cfg.scene.robot.spawn.func is not spawn_research3_with_mimic_fingertips:
        raise ValueError("Research3 mimic-fingertip task must use the collision compatibility spawner")


def remap_research3_names(cfg: Any) -> Any:
    """Remap a config fragment injected by a shared script after Hydra parsing."""
    return _replace_robot_names(cfg)


@configclass
class Research3RlStateSceneCfg(panda_state.RlStateSceneCfg):
    robot = FRANKA_RESEARCH3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Research3RGBSceneCfg(panda_rgb.DataCollectionRGBObjectSceneCfg):
    robot = FRANKA_RESEARCH3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_hand/rgb_wrist_camera",
        update_period=0,
        height=panda_rgb._WRIST_HEIGHT,
        width=panda_rgb._WRIST_WIDTH,
        offset=TiledCameraCfg.OffsetCfg(
            pos=panda_rgb._WRIST_POS,
            rot=panda_rgb._WRIST_ROT,
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=panda_rgb._WRIST_SPAWN,
    )


@configclass
class Research3MimicFingertipRlStateSceneCfg(Research3RlStateSceneCfg):
    robot = FRANKA_RESEARCH3_MIMIC_FINGERTIP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Research3MimicFingertipRGBSceneCfg(Research3RGBSceneCfg):
    robot = FRANKA_RESEARCH3_MIMIC_FINGERTIP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class FrankaResearch3DiffIKJointTargetFinetuneEvalCfg(
    panda_state.FrankaFr3GripperRelCartesianDiffIKJointTargetFinetuneEvalCfg
):
    scene: Research3RlStateSceneCfg = Research3RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    actions: FrankaResearch3RelativeDiffIKJointTargetAction = (
        FrankaResearch3RelativeDiffIKJointTargetAction()
    )

    def __post_init__(self):
        super().__post_init__()
        _finalize_research3_config(self)


@configclass
class FrankaResearch3DiffIKJointTargetXY5T0EvalCfg(
    panda_state.FrankaFr3GripperRelCartesianDiffIKJointTargetXY5T0EvalCfg
):
    """Official FR3 state eval on the online XY5 t0 reset distribution."""

    scene: Research3RlStateSceneCfg = Research3RlStateSceneCfg(num_envs=32, env_spacing=1.5)
    actions: FrankaResearch3RelativeDiffIKJointTargetAction = (
        FrankaResearch3RelativeDiffIKJointTargetAction()
    )

    def __post_init__(self):
        super().__post_init__()
        _finalize_research3_config(self)


@configclass
class FrankaResearch3RGBAbsoluteJointTargetEvalCfg(
    panda_rgb.FrankaFr3GripperEvalRGBAbsoluteJointTargetCfg
):
    scene: Research3RGBSceneCfg = Research3RGBSceneCfg(
        num_envs=32, env_spacing=panda_rgb._RGB_ENV_SPACING, replicate_physics=False
    )
    actions: FrankaResearch3AbsoluteJointTargetAction = FrankaResearch3AbsoluteJointTargetAction()

    def __post_init__(self):
        super().__post_init__()
        _finalize_research3_config(self)
        if self.scene.wrist_camera.prim_path != "{ENV_REGEX_NS}/Robot/fr3_hand/rgb_wrist_camera":
            raise ValueError("Research3 wrist camera must be parented to fr3_hand")


@configclass
class FrankaResearch3RGBRelCartesianOSCEvalCfg(
    panda_rgb.FrankaFr3GripperEvalRGBRelCartesianOSCCfg
):
    scene: Research3RGBSceneCfg = Research3RGBSceneCfg(
        num_envs=32, env_spacing=panda_rgb._RGB_ENV_SPACING, replicate_physics=False
    )
    actions: FrankaResearch3RelativeOSCEvalAction = FrankaResearch3RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        _finalize_research3_config(self)


@configclass
class FrankaResearch3MimicFingertipDiffIKJointTargetFinetuneEvalCfg(
    FrankaResearch3DiffIKJointTargetFinetuneEvalCfg
):
    scene: Research3MimicFingertipRlStateSceneCfg = Research3MimicFingertipRlStateSceneCfg(
        num_envs=32, env_spacing=1.5
    )

    def __post_init__(self):
        super().__post_init__()
        _finalize_mimic_fingertip_config(self)


@configclass
class FrankaResearch3MimicFingertipDiffIKJointTargetXY5T0EvalCfg(
    FrankaResearch3DiffIKJointTargetXY5T0EvalCfg
):
    """FR3 visuals/dynamics plus Factory convex-hull fingertips on online XY5 t0."""

    scene: Research3MimicFingertipRlStateSceneCfg = Research3MimicFingertipRlStateSceneCfg(
        num_envs=32, env_spacing=1.5
    )

    def __post_init__(self):
        super().__post_init__()
        _finalize_mimic_fingertip_config(self)


@configclass
class FrankaResearch3MimicFingertipRGBAbsoluteJointTargetEvalCfg(
    FrankaResearch3RGBAbsoluteJointTargetEvalCfg
):
    scene: Research3MimicFingertipRGBSceneCfg = Research3MimicFingertipRGBSceneCfg(
        num_envs=32, env_spacing=panda_rgb._RGB_ENV_SPACING, replicate_physics=False
    )

    def __post_init__(self):
        super().__post_init__()
        _finalize_mimic_fingertip_config(self)


@configclass
class FrankaResearch3MimicFingertipRGBRelCartesianOSCEvalCfg(
    FrankaResearch3RGBRelCartesianOSCEvalCfg
):
    scene: Research3MimicFingertipRGBSceneCfg = Research3MimicFingertipRGBSceneCfg(
        num_envs=32, env_spacing=panda_rgb._RGB_ENV_SPACING, replicate_physics=False
    )

    def __post_init__(self):
        super().__post_init__()
        _finalize_mimic_fingertip_config(self)


@configclass
class FrankaResearch3MimicFingertipRGBRelCartesianOSCXY5T0Cfg(
    panda_rgb.FrankaFr3GripperDataCollectionRGBRelCartesianOSCXY5T0Cfg
):
    """FR3 white-arm-capable RGB collection on continuous XY5 t0 resets."""

    scene: Research3MimicFingertipRGBSceneCfg = Research3MimicFingertipRGBSceneCfg(
        num_envs=32, env_spacing=panda_rgb._RGB_ENV_SPACING, replicate_physics=False
    )
    actions: FrankaResearch3RelativeOSCEvalAction = FrankaResearch3RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        _finalize_mimic_fingertip_config(self)
