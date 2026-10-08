#!/usr/bin/env python3
"""Render the CupCake source mesh and assembled/exploded CoACD pieces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = ROOT / "scripts/sim2sim/assets_mjcf/cupcake"
SOURCE_MESH = ASSET_DIR / "cupcake_collision.obj"
DEFAULT_PIECE_DIR = ASSET_DIR / "coacd_fine"
COLORS = (
    [0.90, 0.25, 0.18, 1.0],
    [0.18, 0.55, 0.88, 1.0],
    [0.95, 0.68, 0.16, 1.0],
    [0.20, 0.70, 0.42, 1.0],
    [0.64, 0.35, 0.82, 1.0],
    [0.08, 0.72, 0.75, 1.0],
    [0.92, 0.42, 0.68, 1.0],
    [0.58, 0.63, 0.18, 1.0],
)


def build_model(
    mode: str,
    pieces: list[dict],
    piece_dir: Path,
    framebuffer_size: int,
    explode: float = 0.0,
):
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = framebuffer_size
    spec.visual.global_.offheight = framebuffer_size
    spec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    spec.visual.headlight.diffuse = [0.75, 0.75, 0.75]
    if mode == "source":
        spec.add_mesh(name="source", file=str(SOURCE_MESH))
        spec.worldbody.add_geom(
            name="source",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="source",
            rgba=[0.76, 0.31, 0.14, 1.0],
            contype=0,
            conaffinity=0,
        )
    else:
        centers = np.asarray(
            [
                0.5
                * (
                    np.asarray(piece["bounds_min_m"])
                    + np.asarray(piece["bounds_max_m"])
                )
                for piece in pieces
            ]
        )
        center = np.mean(centers, axis=0)
        for index, piece in enumerate(pieces):
            spec.add_mesh(
                name=piece["name"], file=str(piece_dir / piece["file"])
            )
            offset = explode * (centers[index] - center)
            body = spec.worldbody.add_body(
                name=f"piece_body_{index:02d}", pos=offset.tolist()
            )
            body.add_geom(
                name=piece["name"],
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=piece["name"],
                rgba=(
                    [0.72, 0.72, 0.74, 1.0]
                    if piece.get("kind") == "support_pad"
                    else COLORS[index % len(COLORS)]
                ),
                contype=0,
                conaffinity=0,
            )
    return spec.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "log/active/cupcake_sim2sim_20260815/coacd_visualization",
    )
    parser.add_argument("--piece-dir", type=Path, default=DEFAULT_PIECE_DIR)
    parser.add_argument("--size", type=int, default=420)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    piece_dir = args.piece_dir.resolve()
    manifest = json.loads((piece_dir / "manifest.json").read_text(encoding="ascii"))
    pieces = manifest["pieces"]
    coacd_count = sum(piece.get("kind") == "coacd" for piece in pieces)
    models = (
        build_model("source", pieces, piece_dir, args.size),
        build_model("coacd", pieces, piece_dir, args.size),
        build_model("coacd", pieces, piece_dir, args.size, explode=0.8),
    )
    data = [mujoco.MjData(model) for model in models]
    for model, datum in zip(models, data, strict=True):
        mujoco.mj_forward(model, datum)
    renderers = [
        mujoco.Renderer(model, width=args.size, height=args.size) for model in models
    ]
    labels = (
        "Source collision mesh",
        f"{len(pieces)} convex pieces",
        f"Exploded: {coacd_count} CoACD + support",
    )
    video_path = args.out / "cupcake_coacd_fine_turntable.mp4"
    image_path = args.out / "cupcake_coacd_fine_overview.png"
    writer = imageio.get_writer(video_path, fps=args.fps, macro_block_size=1)
    try:
        for frame in range(96):
            panels = []
            azimuth = 360.0 * frame / 96.0
            for model, datum, renderer, label in zip(
                models, data, renderers, labels, strict=True
            ):
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                camera.lookat[:] = [0.0, 0.0, 0.05]
                camera.distance = 0.22
                camera.azimuth = azimuth
                camera.elevation = -12.0
                renderer.update_scene(datum, camera=camera)
                panel = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((8, 8, 190, 34), fill=(15, 15, 15))
                draw.text((16, 14), label, fill=(255, 255, 255))
                panels.append(np.asarray(panel))
            combined = np.concatenate(panels, axis=1)
            writer.append_data(combined)
            if frame == 32:
                imageio.imwrite(image_path, combined)
    finally:
        writer.close()
        for renderer in renderers:
            renderer.close()
    print(image_path.resolve())
    print(video_path.resolve())


if __name__ == "__main__":
    main()
