#!/usr/bin/env python3
"""Render controlled before/after videos for CupCake tilt and hand collision."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import zarr
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_cupcake_mujoco as Replay
import compare_peginsert_continuous_mujoco as B3
import cupcake_mujoco_model as CupCake


ROOT = Path(__file__).resolve().parents[3]
SOURCE_ZARR = ROOT / "datasets/cupcake_sim2sim_t0_20260815/isaac_fit_b3center.zarr"
DEFAULT_OUT = (
    ROOT / "log/active/cupcake_sim2sim_20260815/physics_before_after"
)


def x_rotation(degrees: float) -> np.ndarray:
    half_angle = math.radians(degrees) / 2.0
    return np.array([math.cos(half_angle), math.sin(half_angle), 0.0, 0.0])


def rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion)
    return matrix.reshape(3, 3)


def tilt_degrees(quaternion: np.ndarray) -> float:
    local_z_world = rotation_matrix(quaternion)[:, 2]
    return math.degrees(math.acos(np.clip(local_z_world[2], -1.0, 1.0)))


def source_vertices() -> np.ndarray:
    return np.asarray(
        [
            [float(value) for value in line.split()[1:4]]
            for line in CupCake.CUPCAKE_MESH.read_text(encoding="ascii").splitlines()
            if line.startswith("v ")
        ],
        dtype=np.float64,
    )


@dataclass
class ReleaseScene:
    label: str
    detail: str
    model: mujoco.MjModel
    data: mujoco.MjData
    controller: CupCake.Controller


def build_release_scene(store, raw: np.ndarray, end: int, fixed: bool) -> ReleaseScene:
    profile = B3.b3_center_profile(physics_substeps=4)
    profile.update(
        cupcake_collision="convex_decomposition" if fixed else "sdf",
        hand_collision="source_usd" if fixed else "disabled",
        finger_collision="mimic" if fixed else "menagerie_mesh_only",
        noslip_iterations=0 if fixed else 5,
        plate_collision_enabled=False,
        contact_solref=[0.02, 1.0],
        robot_contact_solref=[0.005, 1.0] if fixed else [0.02, 1.0],
    )
    model = CupCake.build_model(raw, profile_cfg=profile)
    controller = CupCake.Controller(model)
    data = mujoco.MjData(model)
    Replay.configure_episode(
        model,
        controller,
        store,
        0,
        end,
        raw,
        friction_combine="physx_average" if fixed else "legacy_b3",
    )
    model.opt.noslip_iterations = 0 if fixed else 5
    if not fixed:
        for pair in range(model.npair):
            names = {
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    int(model.pair_geom1[pair]),
                ),
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    int(model.pair_geom2[pair]),
                ),
            }
            if "table_top" in names and any(
                name and name.startswith("cupcake_collision") for name in names
            ):
                model.pair_dim[pair] = 6
                model.pair_friction[pair, 2:] = [0.05, 0.01, 0.01]

    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if body in controller.robot_body_ids:
            model.geom_contype[geom] = 0
            model.geom_conaffinity[geom] = 0
            model.geom_rgba[geom, 3] = 0.0
        if name == "floor":
            model.geom_contype[geom] = 0
            model.geom_conaffinity[geom] = 0
            model.geom_rgba[geom, 3] = 0.0
        elif name in {"plate_visual", "plate_collision"}:
            model.geom_rgba[geom, 3] = 0.0
        elif name and name.startswith("table_collision_"):
            model.geom_rgba[geom, 3] = 0.0
    return ReleaseScene(
        label="AFTER" if fixed else "BEFORE",
        detail=(
            f"{len(CupCake.convex_decomposition_piece_records())} convex pieces | "
            "condim=3 | no-slip off"
            if fixed
            else "SDF | condim=6 | no-slip on"
        ),
        model=model,
        data=data,
        controller=controller,
    )


def reset_release(scene: ReleaseScene, vertices: np.ndarray, angle: float) -> None:
    model, data, controller = scene.model, scene.data, scene.controller
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    table = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    table_xy = data.geom_xpos[table, :2].copy()
    table_top = float(data.geom_xpos[table, 2] + model.geom_size[table, 2])
    quaternion = x_rotation(angle)
    mesh_min_z = float((vertices @ rotation_matrix(quaternion).T)[:, 2].min())
    qpos = controller.insertive_qpos
    data.qpos[qpos : qpos + 3] = [
        table_xy[0],
        table_xy[1],
        table_top - mesh_min_z + 0.0005,
    ]
    data.qpos[qpos + 3 : qpos + 7] = quaternion
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def camera_for_release(scene: ReleaseScene) -> mujoco.MjvCamera:
    table = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = scene.data.geom_xpos[table] + [0.0, 0.0, 0.055]
    camera.distance = 0.28
    camera.azimuth = 90.0
    camera.elevation = -4.0
    return camera


def annotate(
    frame: np.ndarray,
    label: str,
    detail: str,
    metric: str,
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 54), fill=(12, 12, 12))
    draw.text((12, 8), label, fill=(255, 255, 255))
    draw.text((12, 29), detail, fill=(205, 205, 205))
    draw.text((image.width - 158, 8), metric, fill=(255, 230, 120))
    return np.asarray(image)


def render_release_comparison(
    store,
    raw: np.ndarray,
    end: int,
    out: Path,
    fps: int,
    width: int,
    height: int,
) -> dict:
    scenes = (
        build_release_scene(store, raw, end, fixed=False),
        build_release_scene(store, raw, end, fixed=True),
    )
    vertices = source_vertices()
    for scene in scenes:
        reset_release(scene, vertices, angle=50.0)
    cameras = [camera_for_release(scene) for scene in scenes]
    renderers = [
        mujoco.Renderer(scene.model, width=width, height=height) for scene in scenes
    ]
    output = out / "tilt_50deg_before_after.mp4"
    writer = imageio.get_writer(output, fps=fps, macro_block_size=1)
    frame_count = 5 * fps + 1
    try:
        for frame_index in range(frame_count):
            target_time = frame_index / fps
            panels = []
            for scene, renderer, camera in zip(
                scenes, renderers, cameras, strict=True
            ):
                while scene.data.time + 0.5 * scene.model.opt.timestep < target_time:
                    mujoco.mj_step(scene.model, scene.data)
                renderer.update_scene(scene.data, camera=camera)
                qpos = scene.controller.insertive_qpos
                tilt = tilt_degrees(scene.data.qpos[qpos + 3 : qpos + 7])
                panel = annotate(
                    renderer.render().copy(),
                    scene.label,
                    scene.detail,
                    f"tilt {tilt:5.1f} deg",
                )
                panels.append(panel)
            combined = np.concatenate(panels, axis=1)
            writer.append_data(combined)
            if frame_index in {0, frame_count - 1}:
                imageio.imwrite(
                    out / f"tilt_50deg_frame_{frame_index:03d}.png", combined
                )
    finally:
        writer.close()
        for renderer in renderers:
            renderer.close()
    return {
        "video": str(output.resolve()),
        "initial_tilt_deg": 50.0,
        "final_tilt_deg": {
            scene.label.lower(): tilt_degrees(
                scene.data.qpos[
                    scene.controller.insertive_qpos
                    + 3 : scene.controller.insertive_qpos
                    + 7
                ]
            )
            for scene in scenes
        },
    }


@dataclass
class HandScene:
    label: str
    detail: str
    model: mujoco.MjModel
    data: mujoco.MjData
    qpos: int
    dof: int


def build_hand_scene(active_collision: bool) -> HandScene:
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.001
    spec.option.gravity = [0.0, 0.0, 0.0]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 150
    spec.add_mesh(name="hand_probe", file=str(CupCake.SOURCE_HAND_MESH))
    spec.add_mesh(name="cupcake_probe_visual", file=str(CupCake.CUPCAKE_MESH))
    pieces = CupCake.convex_decomposition_piece_records()
    for piece in pieces:
        spec.add_mesh(name=piece["name"], file=piece["path"])
    spec.worldbody.add_geom(
        name="hand_probe",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="hand_probe",
        contype=int(active_collision),
        conaffinity=int(active_collision),
        friction=[0.5, 0.01, 0.001],
        condim=3,
        solref=CupCake.ROBOT_OBJECT_CONTACT_SOLREF,
        rgba=[0.65, 0.65, 0.70, 1.0],
    )
    body = spec.worldbody.add_body(name="cupcake_probe", pos=[-0.14, 0.0, -0.03])
    body.add_freejoint(name="cupcake_probe_freejoint")
    body.add_geom(
        name="cupcake_probe_visual",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="cupcake_probe_visual",
        mass=0.1268,
        contype=0,
        conaffinity=0,
        rgba=[0.80, 0.30, 0.12, 1.0],
    )
    for index, piece in enumerate(pieces):
        geom_name = f"cupcake_probe_collision_{index:02d}"
        body.add_geom(
            name=geom_name,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=piece["name"],
            mass=0.0,
            friction=[0.5, 0.01, 0.001],
            condim=3,
            solref=CupCake.ROBOT_OBJECT_CONTACT_SOLREF,
            rgba=[0.0, 0.0, 0.0, 0.0],
        )
        if active_collision:
            spec.add_pair(
                geomname1=geom_name,
                geomname2="hand_probe",
                condim=3,
                friction=[0.5, 0.5, 0.0, 0.0, 0.0],
                solref=CupCake.ROBOT_OBJECT_CONTACT_SOLREF,
            )
    model = spec.compile()
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "cupcake_probe_freejoint"
    )
    qpos = int(model.jnt_qposadr[joint])
    dof = int(model.jnt_dofadr[joint])
    data.qvel[dof] = 0.35
    mujoco.mj_forward(model, data)
    return HandScene(
        label="AFTER" if active_collision else "BEFORE",
        detail="source hand convex hull | 5 ms" if active_collision else "hand collision disabled",
        model=model,
        data=data,
        qpos=qpos,
        dof=dof,
    )


def render_hand_comparison(out: Path, fps: int, width: int, height: int) -> dict:
    scenes = (build_hand_scene(False), build_hand_scene(True))
    renderers = [
        mujoco.Renderer(scene.model, width=width, height=height) for scene in scenes
    ]
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.02]
    camera.distance = 0.45
    camera.azimuth = 90.0
    camera.elevation = 0.0
    output = out / "hand_collision_before_after.mp4"
    writer = imageio.get_writer(output, fps=fps, macro_block_size=1)
    minimum_distances = {scene.label.lower(): math.inf for scene in scenes}
    maximum_x = {scene.label.lower(): float(scene.data.qpos[scene.qpos]) for scene in scenes}
    frame_count = int(round(1.2 * fps)) + 1
    try:
        for frame_index in range(frame_count):
            target_time = frame_index / fps
            panels = []
            for scene, renderer in zip(scenes, renderers, strict=True):
                while scene.data.time + 0.5 * scene.model.opt.timestep < target_time:
                    mujoco.mj_step(scene.model, scene.data)
                    maximum_x[scene.label.lower()] = max(
                        maximum_x[scene.label.lower()],
                        float(scene.data.qpos[scene.qpos]),
                    )
                    if scene.data.ncon:
                        minimum_distances[scene.label.lower()] = min(
                            minimum_distances[scene.label.lower()],
                            min(
                                float(contact.dist)
                                for contact in scene.data.contact[: scene.data.ncon]
                            ),
                        )
                renderer.update_scene(scene.data, camera=camera)
                panel = annotate(
                    renderer.render().copy(),
                    scene.label,
                    scene.detail,
                    f"x {scene.data.qpos[scene.qpos] * 1000:6.0f} mm",
                )
                panels.append(panel)
            combined = np.concatenate(panels, axis=1)
            writer.append_data(combined)
            if frame_index in {0, frame_count - 1}:
                imageio.imwrite(
                    out / f"hand_collision_frame_{frame_index:03d}.png", combined
                )
    finally:
        writer.close()
        for renderer in renderers:
            renderer.close()
    return {
        "video": str(output.resolve()),
        "initial_speed_mps": 0.35,
        "maximum_x_m": maximum_x,
        "maximum_penetration_m": {
            key: max(0.0, -value) if math.isfinite(value) else 0.0
            for key, value in minimum_distances.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, default=SOURCE_ZARR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=540)
    parser.add_argument("--height", type=int, default=420)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(args.zarr), mode="r")
    Replay.configure_timing(store)
    raw = np.asarray(store["data/raw_state"][0], dtype=np.float64)
    end = int(np.asarray(store["meta/episode_ends"])[0])
    report = {
        "definition": "controlled same-state CupCake physics before/after videos",
        "tilt_release": render_release_comparison(
            store, raw, end, args.out, args.fps, args.width, args.height
        ),
        "hand_collision": render_hand_comparison(
            args.out, args.fps, args.width, args.height
        ),
    }
    report_path = args.out / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(report_path.resolve())


if __name__ == "__main__":
    main()
