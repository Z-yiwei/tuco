# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn official Research 3 with the historical Factory fingertip collisions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pxr import Gf, Sdf, Usd, UsdPhysics, UsdShade

from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from uwlab_assets import resolve_cloud_path

if TYPE_CHECKING:
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


FRANKA_MIMIC_USD_PATH = resolve_cloud_path(f"{ISAACLAB_NUCLEUS_DIR}/Factory/franka_mimic.usd")
FR3_VISUAL_PROFILE_ENV = "OMNIRESET_FR3_VISUAL_PROFILE"
FR3_VISUAL_PROFILE_OFFICIAL = "official"
FR3_VISUAL_PROFILE_WHITE_ARM = "white_arm_base_ring"
FR3_VISUAL_PROFILE_WHITE_ARM_BLACK_PADS = "white_arm_base_ring_black_finger_pads"

_FINGERTIP_COLLISION_REFS = {
    "left": "/panda/panda_leftfinger/collisions",
    "right": "/panda/panda_rightfinger/collisions",
}

# These official FR3 materials cover dark base and arm-link panels. Wrist,
# flange, hand, and finger materials deliberately retain their official colors.
_WHITE_ARM_DARK_SHADERS = (
    "material__01_Base_URDF_color_0_0_0/Shader",
    "material__04_05_ELBOW_URDF_color_64_64_64/Shader",
    "material__06_LOWER_ARM_URDF_color_64_64_64/Shader",
)
_BASE_RING_SHADER = "material__01_Base_URDF_color_64_64_64/Shader"
_FINGER_PAD_SHADER = "material_Part__Feature_007/Shader"
_WHITE = Gf.Vec3f(1.0, 1.0, 1.0)
_BLACK = Gf.Vec3f(0.0, 0.0, 0.0)


def _shader_diffuse_input(stage: Usd.Stage, shader_path: str):
    prim = stage.GetPrimAtPath(shader_path)
    if not prim.IsValid() or not prim.IsA(UsdShade.Shader):
        raise RuntimeError(f"Official Research 3 shader is missing: {shader_path}")
    diffuse = UsdShade.Shader(prim).GetInput("diffuse_color_constant")
    if not diffuse:
        raise RuntimeError(f"Research 3 shader has no diffuse color input: {shader_path}")
    return diffuse


def _apply_visual_profile(stage: Usd.Stage, robot_path: str, profile: str) -> None:
    if profile == FR3_VISUAL_PROFILE_OFFICIAL:
        return
    if profile not in (
        FR3_VISUAL_PROFILE_WHITE_ARM,
        FR3_VISUAL_PROFILE_WHITE_ARM_BLACK_PADS,
    ):
        raise ValueError(
            f"Unsupported {FR3_VISUAL_PROFILE_ENV}={profile!r}; expected "
            f"{FR3_VISUAL_PROFILE_OFFICIAL!r}, {FR3_VISUAL_PROFILE_WHITE_ARM!r}, or "
            f"{FR3_VISUAL_PROFILE_WHITE_ARM_BLACK_PADS!r}"
        )

    looks_path = f"{robot_path}/Looks"
    base_ring = _shader_diffuse_input(stage, f"{looks_path}/{_BASE_RING_SHADER}")
    base_ring_color = base_ring.Get()
    if base_ring_color is None or max(float(value) for value in base_ring_color) >= 0.1:
        raise RuntimeError(f"FR3 base-ring material is no longer black: {base_ring_color}")

    for relative_path in _WHITE_ARM_DARK_SHADERS:
        diffuse = _shader_diffuse_input(stage, f"{looks_path}/{relative_path}")
        if not diffuse.Set(_WHITE):
            raise RuntimeError(f"Failed to whiten FR3 material: {relative_path}")

    if profile == FR3_VISUAL_PROFILE_WHITE_ARM_BLACK_PADS:
        finger_pad = _shader_diffuse_input(stage, f"{looks_path}/{_FINGER_PAD_SHADER}")
        finger_pad_color = finger_pad.Get()
        if finger_pad_color is None or any(
            abs(float(value) - 0.2509804) > 1e-4 for value in finger_pad_color
        ):
            raise RuntimeError(
                f"FR3 finger-pad material is no longer the expected gray: {finger_pad_color}"
            )
        if not finger_pad.Set(_BLACK):
            raise RuntimeError(f"Failed to blacken FR3 finger-pad material: {_FINGER_PAD_SHADER}")


def _replace_fingertip_collisions(stage: Usd.Stage, robot_path: str) -> None:
    for side, source_path in _FINGERTIP_COLLISION_REFS.items():
        official_path = f"{robot_path}/fr3_{side}finger/collisions"
        official_collision = stage.GetPrimAtPath(official_path)
        if not official_collision.IsValid():
            raise RuntimeError(f"Official Research 3 collision root is missing: {official_path}")
        official_collision.SetActive(False)

        mimic_path = f"{robot_path}/fr3_{side}finger/mimic_collisions"
        mimic_collision = stage.DefinePrim(mimic_path)
        mimic_collision.GetReferences().AddReference(
            FRANKA_MIMIC_USD_PATH,
            Sdf.Path(source_path),
        )
        composed = stage.GetPrimAtPath(mimic_path)
        if composed.GetTypeName() != "Mesh" or not composed.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"Mimic fingertip collision did not compose as a collision Mesh: {mimic_path}")
        approximation = composed.GetAttribute("physics:approximation").Get()
        if approximation != "convexHull":
            raise RuntimeError(
                f"Mimic fingertip collision must use convexHull, got {approximation!r}: {mimic_path}"
            )


@clone
def spawn_research3_with_mimic_fingertips(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn one official FR3 source prim and replace only its fingertip collisions."""
    del kwargs
    robot = spawn_from_usd.__wrapped__(prim_path, cfg, translation, orientation)
    stage = get_current_stage()
    _replace_fingertip_collisions(stage, prim_path)
    visual_profile = os.environ.get(FR3_VISUAL_PROFILE_ENV, FR3_VISUAL_PROFILE_OFFICIAL)
    _apply_visual_profile(stage, prim_path, visual_profile)
    robot.SetCustomDataByKey("omnireset_fingertip_collision_model", "franka_mimic_convex_hull")
    robot.SetCustomDataByKey("omnireset_visual_profile", visual_profile)
    return robot
