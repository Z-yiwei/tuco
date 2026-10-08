#!/usr/bin/env python3
"""Split the CupCake into vertical convex sectors without horizontal cut shelves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "scripts/sim2sim/assets_mjcf/cupcake/cupcake_collision.obj"
DEFAULT_OUTPUT = ROOT / "scripts/sim2sim/assets_mjcf/cupcake/convex_radial32"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        staging.write_text(content, encoding="ascii")
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def obj_text(vertices: np.ndarray, faces: np.ndarray, name: str) -> str:
    lines = [f"o {name}"]
    lines.extend(f"v {x:.12g} {y:.12g} {z:.12g}" for x, y, z in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sectors", type=int, default=32)
    parser.add_argument("--surface-max-edge-m", type=float, default=0.0015)
    args = parser.parse_args()
    if args.sectors < 4:
        parser.error("--sectors must be at least 4")
    if args.surface_max_edge_m <= 0.0:
        parser.error("--surface-max-edge-m must be positive")
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {output_dir}")

    source_mesh = trimesh.load_mesh(source, force="mesh", process=True)
    source_watertight = bool(source_mesh.is_watertight)
    if not source_mesh.is_watertight:
        source_mesh.fill_holes()
    if not source_mesh.is_watertight:
        raise ValueError("triangle-hole repair did not make source watertight")
    vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    surface_vertices, _ = trimesh.remesh.subdivide_to_size(
        vertices,
        np.asarray(source_mesh.faces, dtype=np.int64),
        max_edge=args.surface_max_edge_m,
        max_iter=12,
    )
    surface_vertices = np.asarray(surface_vertices, dtype=np.float64)
    xy_center = np.mean(vertices[:, :2], axis=0)
    angles = np.arctan2(
        surface_vertices[:, 1] - xy_center[1],
        surface_vertices[:, 0] - xy_center[0],
    )
    z_min = float(vertices[:, 2].min())
    z_max = float(vertices[:, 2].max())

    output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    all_vertices = []
    half_width = np.pi / args.sectors
    for index in range(args.sectors):
        center_angle = -np.pi + (index + 0.5) * 2.0 * np.pi / args.sectors
        difference = np.angle(np.exp(1j * (angles - center_angle)))
        sector_vertices = surface_vertices[
            np.abs(difference) <= half_width + 1.0e-9
        ]
        sector_vertices = np.vstack(
            [
                sector_vertices,
                [xy_center[0], xy_center[1], z_min],
                [xy_center[0], xy_center[1], z_max],
            ]
        )
        # trimesh orients the hull faces consistently.  scipy.spatial.ConvexHull's
        # simplices are not guaranteed to share one outward winding, which makes
        # signed volume checks incorrect when those faces are exported directly.
        mesh = trimesh.convex.convex_hull(sector_vertices)
        name = f"cupcake_radial_piece_{index:02d}"
        path = output_dir / f"{name}.obj"
        atomic_write(
            path,
            obj_text(
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int64),
                name,
            ),
        )
        all_vertices.append(np.asarray(mesh.vertices, dtype=np.float64))
        records.append(
            {
                "name": name,
                "file": path.name,
                "sha256": sha256(path),
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "volume_m3": float(abs(mesh.volume)),
                "watertight": bool(mesh.is_watertight),
                "bounds_min_m": mesh.bounds[0].tolist(),
                "bounds_max_m": mesh.bounds[1].tolist(),
            }
        )

    rng = np.random.default_rng(0)
    directions = rng.normal(size=(8192, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    source_support = np.max(vertices @ directions.T, axis=0)
    piece_support = np.max(np.concatenate(all_vertices) @ directions.T, axis=0)
    support_difference = piece_support - source_support
    source_volume = float(abs(source_mesh.volume))
    piece_volume = float(sum(record["volume_m3"] for record in records))
    manifest = {
        "schema_version": 2,
        "definition": (
            "dense-surface vertical radial convex sectors; internal cut faces "
            "are vertical"
        ),
        "source": str(source),
        "source_sha256": sha256(source),
        "source_watertight_before_repair": source_watertight,
        "source_watertight_after_repair": bool(source_mesh.is_watertight),
        "sector_count": args.sectors,
        "surface_max_edge_m": args.surface_max_edge_m,
        "surface_vertices": int(len(surface_vertices)),
        "piece_count": len(records),
        "source_volume_m3": source_volume,
        "piece_volume_sum_m3": piece_volume,
        "piece_volume_excess_ratio": piece_volume / source_volume - 1.0,
        "support_error": {
            "sample_count": len(directions),
            "minimum_m": float(support_difference.min()),
            "maximum_m": float(support_difference.max()),
            "rms_m": float(np.sqrt(np.mean(support_difference**2))),
        },
        "pieces": records,
    }
    atomic_write(
        output_dir / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
