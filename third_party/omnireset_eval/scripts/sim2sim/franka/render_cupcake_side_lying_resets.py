#!/usr/bin/env python3
"""Render ten deterministic samples from the CupCake side-lying reset contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import closed_loop_eval as CL
import cupcake_mujoco_model as CupCake


ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = ROOT / "scripts/cupcake"
if str(SPEC_DIR) not in sys.path:
    sys.path.insert(0, str(SPEC_DIR))
import cupcake_side_lying_reset_spec as reset_spec
DEFAULT_OUT = (
    ROOT / "log/active/cupcake_sim2sim_20260815/side_lying_front3cm_samples"
)
HOME_Q9 = np.array(
    [0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398, 0.04, 0.04],
    dtype=np.float64,
)


def sample_range(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(low if low == high else rng.uniform(low, high))


def raw_state(cupcake_pose: np.ndarray, cupcake_quat: np.ndarray, plate_pose: np.ndarray):
    raw = np.zeros(57, dtype=np.float64)
    raw[:9] = HOME_Q9
    raw[18:25] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    raw[31:34] = cupcake_pose
    raw[34:38] = cupcake_quat
    raw[44:47] = plate_pose
    raw[47:51] = [1.0, 0.0, 0.0, 0.0]
    return raw


def annotate(frame: np.ndarray, index: int, sample: dict) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 430, 66), fill=(18, 18, 18))
    draw.text((20, 17), f"Sample {index:02d} | side roll +90 deg", fill="white")
    draw.text(
        (20, 41),
        (
            f"cup x/y {sample['cupcake_xyz_m'][0]:.3f}/{sample['cupcake_xyz_m'][1]:+.3f} m"
            f" | yaw {sample['cupcake_rpy_deg'][2]:+.1f} deg"
        ),
        fill=(235, 235, 235),
    )
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count != 10:
        raise ValueError("this audit renderer intentionally produces exactly 10 samples")

    cfg = reset_spec
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    yaw_values = np.linspace(
        cfg.CUPCAKE_POSE_RANGE["yaw"][0],
        cfg.CUPCAKE_POSE_RANGE["yaw"][1],
        args.count,
    )
    samples = []
    sheet_panels = []

    for index, yaw in enumerate(yaw_values):
        cup_rpy = np.array(
            [
                sample_range(rng, cfg.CUPCAKE_POSE_RANGE["roll"]),
                sample_range(rng, cfg.CUPCAKE_POSE_RANGE["pitch"]),
                yaw,
            ]
        )
        cup_pos = np.array(
            [sample_range(rng, cfg.CUPCAKE_POSE_RANGE[axis]) for axis in "xyz"]
        )
        plate_pos = np.array(
            [sample_range(rng, cfg.PLATE_POSE_RANGE[axis]) for axis in "xyz"]
        )
        cup_quat = np.asarray(cfg.quat_from_euler_xyz(*cup_rpy), dtype=np.float64)
        raw = raw_state(cup_pos, cup_quat, plate_pos)
        model = CupCake.build_model(
            raw,
            profile_cfg={
                "gripper": CL.B3_GRIPPER,
                "physics_substeps": 1,
                "cupcake_collision": "convex_decomposition",
                "cupcake_plate_collision": "base_cylinder",
            },
        )
        data = mujoco.MjData(model)
        controller = CupCake.Controller(model)
        CupCake.set_raw_state(model, data, controller, raw)
        renderer = mujoco.Renderer(model, width=640, height=480)
        try:
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = [0.45, 0.0, 0.04]
            camera.distance = 0.85
            camera.azimuth = 135.0
            camera.elevation = -32.0
            renderer.update_scene(data, camera=camera)
            sample = {
                "index": index,
                "cupcake_xyz_m": cup_pos.tolist(),
                "cupcake_rpy_deg": np.rad2deg(cup_rpy).tolist(),
                "cupcake_quat_wxyz": cup_quat.tolist(),
                "plate_xyz_m": plate_pos.tolist(),
                "plate_rpy_deg": [0.0, 0.0, 0.0],
            }
            frame = annotate(renderer.render().copy(), index, sample)
        finally:
            renderer.close()
        image_path = args.out / f"sample_{index:02d}.png"
        imageio.imwrite(image_path, frame)
        sheet_panels.append(np.asarray(Image.fromarray(frame).resize((384, 288))))
        sample["image"] = image_path.name
        samples.append(sample)

    rows = [np.concatenate(sheet_panels[i : i + 5], axis=1) for i in (0, 5)]
    sheet = np.concatenate(rows, axis=0)
    sheet_path = args.out / "contact_sheet_10.png"
    imageio.imwrite(sheet_path, sheet)
    manifest = {
        "reset_type": cfg.RESET_TYPE,
        "seed": args.seed,
        "sampling": "x/y uniform; yaw stratified across the configured uniform interval",
        "cupcake_pose_range": cfg.CUPCAKE_POSE_RANGE,
        "plate_pose_range": cfg.PLATE_POSE_RANGE,
        "samples": samples,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )
    print(sheet_path.resolve())


if __name__ == "__main__":
    main()
