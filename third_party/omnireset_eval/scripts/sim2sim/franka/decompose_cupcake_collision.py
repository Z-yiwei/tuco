#!/usr/bin/env python3
"""Create deterministic CoACD convex collision pieces for the CupCake mesh."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import uuid
from pathlib import Path

import coacd
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "scripts/sim2sim/assets_mjcf/cupcake/cupcake_collision.obj"
DEFAULT_OUTPUT = ROOT / "scripts/sim2sim/assets_mjcf/cupcake/coacd_fine"
DEFAULT_PARAMETERS = {
    "threshold_m": 0.003,
    "real_metric": True,
    "max_convex_hull": 8,
    "preprocess_mode": "off",
    "resolution": 1000,
    "mcts_nodes": 10,
    "mcts_iterations": 80,
    "mcts_max_depth": 3,
    "merge": True,
    "decimate": True,
    "max_ch_vertex": 64,
    "seed": 0,
}


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


def support_pad_mesh(
    radius: float, bottom_z: float, height: float, segments: int
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles))
    )
    vertices = np.vstack(
        (
            np.column_stack((ring, np.full(segments, bottom_z))),
            np.column_stack((ring, np.full(segments, bottom_z + height))),
            [0.0, 0.0, bottom_z],
            [0.0, 0.0, bottom_z + height],
        )
    )
    bottom_center = 2 * segments
    top_center = bottom_center + 1
    faces = []
    for index in range(segments):
        following = (index + 1) % segments
        faces.extend(
            (
                [bottom_center, following, index],
                [top_center, segments + index, segments + following],
                [index, following, segments + following],
                [index, segments + following, segments + index],
            )
        )
    return vertices, np.asarray(faces, dtype=np.int32)


def support_error(source_vertices: np.ndarray, piece_vertices: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(4096, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    source_support = np.max(source_vertices @ directions.T, axis=0)
    piece_support = np.max(piece_vertices @ directions.T, axis=0)
    difference = piece_support - source_support
    return {
        "sample_count": len(directions),
        "minimum_m": float(difference.min()),
        "maximum_m": float(difference.max()),
        "rms_m": float(np.sqrt(np.mean(difference**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--piece-prefix", default="cupcake")
    parser.add_argument("--threshold-m", type=float, default=0.001)
    parser.add_argument("--max-convex-hull", type=int, default=64)
    parser.add_argument(
        "--preprocess-mode", choices=("auto", "on", "off"), default="off"
    )
    parser.add_argument("--resolution", type=int, default=2000)
    parser.add_argument("--mcts-nodes", type=int, default=20)
    parser.add_argument("--mcts-iterations", type=int, default=150)
    parser.add_argument("--mcts-max-depth", type=int, default=4)
    parser.add_argument(
        "--merge", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--decimate", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-ch-vertex", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--support-pad-radius-m", type=float, default=0.024)
    parser.add_argument("--support-pad-height-m", type=float, default=0.001)
    parser.add_argument("--support-pad-segments", type=int, default=64)
    args = parser.parse_args()
    parameters = {
        **DEFAULT_PARAMETERS,
        "threshold_m": args.threshold_m,
        "max_convex_hull": args.max_convex_hull,
        "preprocess_mode": args.preprocess_mode,
        "resolution": args.resolution,
        "mcts_nodes": args.mcts_nodes,
        "mcts_iterations": args.mcts_iterations,
        "mcts_max_depth": args.mcts_max_depth,
        "merge": args.merge,
        "decimate": args.decimate,
        "max_ch_vertex": args.max_ch_vertex,
        "seed": args.seed,
        "support_pad_radius_m": args.support_pad_radius_m,
        "support_pad_height_m": args.support_pad_height_m,
        "support_pad_segments": args.support_pad_segments,
    }
    if parameters["threshold_m"] <= 0.0:
        raise ValueError("threshold-m must be positive")
    if not args.piece_prefix or not args.piece_prefix.replace("_", "").isalnum():
        raise ValueError("piece-prefix must contain only letters, digits, and underscores")
    if parameters["max_convex_hull"] == 0:
        raise ValueError("max-convex-hull must be positive or -1 for unlimited")
    if parameters["support_pad_radius_m"] < 0.0:
        raise ValueError("support-pad-radius-m must be non-negative")
    if parameters["support_pad_radius_m"] > 0.0:
        if parameters["support_pad_height_m"] <= 0.0:
            raise ValueError("support-pad-height-m must be positive")
        if parameters["support_pad_segments"] < 8:
            raise ValueError("support-pad-segments must be at least 8")
    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {output_dir}")

    source_mesh = trimesh.load_mesh(source, force="mesh", process=True)
    before = {
        "vertices": int(len(source_mesh.vertices)),
        "faces": int(len(source_mesh.faces)),
        "watertight": bool(source_mesh.is_watertight),
    }
    if source_mesh.is_watertight:
        hole_filled = False
    else:
        hole_filled = bool(source_mesh.fill_holes())
    if not source_mesh.is_watertight:
        raise ValueError("deterministic triangle-hole repair did not make mesh watertight")

    source_vertices = np.asarray(source_mesh.vertices, dtype=np.float64)
    source_faces = np.asarray(source_mesh.faces, dtype=np.int32)
    coacd_parts = coacd.run_coacd(
        coacd.Mesh(source_vertices, source_faces),
        threshold=parameters["threshold_m"],
        real_metric=parameters["real_metric"],
        max_convex_hull=parameters["max_convex_hull"],
        preprocess_mode=parameters["preprocess_mode"],
        resolution=parameters["resolution"],
        mcts_nodes=parameters["mcts_nodes"],
        mcts_iterations=parameters["mcts_iterations"],
        mcts_max_depth=parameters["mcts_max_depth"],
        merge=parameters["merge"],
        decimate=parameters["decimate"],
        max_ch_vertex=parameters["max_ch_vertex"],
        seed=parameters["seed"],
    )
    parts = [("coacd", vertices, faces) for vertices, faces in coacd_parts]
    if parameters["support_pad_radius_m"] > 0.0:
        pad_vertices, pad_faces = support_pad_mesh(
            parameters["support_pad_radius_m"],
            float(source_vertices[:, 2].min()),
            parameters["support_pad_height_m"],
            parameters["support_pad_segments"],
        )
        parts.append(("support_pad", pad_vertices, pad_faces))
    output_dir.mkdir(parents=True, exist_ok=False)
    piece_records = []
    all_piece_vertices = []
    for index, (kind, vertices, faces) in enumerate(parts):
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        name = (
            f"{args.piece_prefix}_support_pad"
            if kind == "support_pad"
            else f"{args.piece_prefix}_coacd_piece_{index:03d}"
        )
        path = output_dir / f"{name}.obj"
        atomic_write(
            path,
            obj_text(
                np.asarray(mesh.vertices, dtype=np.float64),
                np.asarray(mesh.faces, dtype=np.int64),
                name,
            ),
        )
        all_piece_vertices.append(np.asarray(mesh.vertices, dtype=np.float64))
        piece_records.append(
            {
                "name": name,
                "kind": kind,
                "file": path.name,
                "sha256": sha256(path),
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
                "volume_m3": float(abs(mesh.volume)),
                "bounds_min_m": mesh.bounds[0].tolist(),
                "bounds_max_m": mesh.bounds[1].tolist(),
            }
        )

    source_volume = float(abs(source_mesh.volume))
    piece_volume = float(sum(record["volume_m3"] for record in piece_records))
    piece_vertices = np.concatenate(all_piece_vertices, axis=0)
    manifest = {
        "schema_version": 1,
        "definition": (
            "CoACD convex pieces"
            + (
                " plus an inscribed flat support pad"
                if parameters["support_pad_radius_m"] > 0.0
                else ""
            )
        ),
        "piece_prefix": args.piece_prefix,
        "source": str(source),
        "source_sha256": sha256(source),
        "coacd_version": importlib.metadata.version("coacd"),
        "parameters": parameters,
        "source_before_repair": before,
        "source_after_repair": {
            "hole_filled": hole_filled,
            "vertices": int(len(source_mesh.vertices)),
            "faces": int(len(source_mesh.faces)),
            "watertight": bool(source_mesh.is_watertight),
            "volume_m3": source_volume,
            "bounds_min_m": source_mesh.bounds[0].tolist(),
            "bounds_max_m": source_mesh.bounds[1].tolist(),
        },
        "piece_count": len(piece_records),
        "piece_vertex_count": int(
            sum(record["vertices"] for record in piece_records)
        ),
        "piece_face_count": int(sum(record["faces"] for record in piece_records)),
        "piece_volume_sum_m3": piece_volume,
        "piece_volume_excess_ratio": piece_volume / source_volume - 1.0,
        "top_z_error_m": float(
            piece_vertices[:, 2].max() - source_vertices[:, 2].max()
        ),
        "support_error": support_error(source_vertices, piece_vertices),
        "pieces": piece_records,
    }
    atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
