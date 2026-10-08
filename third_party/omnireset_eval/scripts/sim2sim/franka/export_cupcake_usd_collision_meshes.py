#!/usr/bin/env python3
"""Export current CupCake/Plate USD collision prims as meter-scale MuJoCo OBJ meshes."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics




def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        staging.write_text(text, encoding="ascii")
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def export_mesh(source: Path, prim_path: str, output: Path) -> dict:
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {source}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"missing mesh prim {prim_path} in {source}")
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError(f"prim is not marked as collision geometry: {prim_path}")

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    if points is None or len(points) == 0 or len(counts) == 0:
        raise ValueError(f"empty collision mesh: {prim_path}")
    if int(counts.sum()) != len(indices):
        raise ValueError(f"face index count mismatch for {prim_path}")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    vertices = np.asarray(
        [
            tuple(transform.Transform(Gf.Vec3d(point)))
            for point in points
        ],
        dtype=np.float64,
    )
    vertices *= meters_per_unit

    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        if count < 3:
            raise ValueError(f"face with fewer than three vertices in {prim_path}")
        for index in range(1, int(count) - 1):
            triangles.append((int(face[0]), int(face[index]), int(face[index + 1])))
        offset += int(count)

    lines = [
        f"# source_usd {source.resolve()}",
        f"# source_prim {prim_path}",
        "# units meter",
        f"o {Path(prim_path).name}",
    ]
    lines.extend(f"v {x:.12g} {y:.12g} {z:.12g}" for x, y, z in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in triangles)
    atomic_write(output, "\n".join(lines) + "\n")
    return {
        "source_usd": str(source.resolve()),

        "source_prim": prim_path,
        "source_approximation": prim.GetAttribute("physics:approximation").Get(),
        "source_meters_per_unit": meters_per_unit,
        "output_obj": str(output.resolve()),

        "vertices": int(len(vertices)),
        "source_faces": int(len(counts)),
        "triangles": int(len(triangles)),
        "bounds_min_m": vertices.min(axis=0).tolist(),
        "bounds_max_m": vertices.max(axis=0).tolist(),
        "extents_m": np.ptp(vertices, axis=0).tolist(),
        "local_to_world_transform_row_major": np.asarray(transform, dtype=np.float64).tolist(),
    }


def export_visual_mesh(source: Path, prim_path: str, output: Path) -> dict:
    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"failed to open USD: {source}")
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"missing mesh prim {prim_path} in {source}")

    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    normals = mesh.GetNormalsAttr().Get()
    uv_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    uvs = uv_primvar.Get() if uv_primvar else None
    face_vertices = int(counts.sum())
    if points is None or len(points) == 0 or len(counts) == 0:
        raise ValueError(f"empty visual mesh: {prim_path}")
    if len(indices) != face_vertices:
        raise ValueError(f"face index count mismatch for {prim_path}")
    if mesh.GetNormalsInterpolation() != UsdGeom.Tokens.faceVarying:
        raise ValueError(f"expected face-varying normals for {prim_path}")
    if normals is None or len(normals) != face_vertices:
        raise ValueError(f"normal count mismatch for {prim_path}")
    if not uv_primvar or uv_primvar.GetInterpolation() != UsdGeom.Tokens.faceVarying:
        raise ValueError(f"expected face-varying UVs for {prim_path}")
    if uvs is None or len(uvs) != face_vertices or uv_primvar.IsIndexed():
        raise ValueError(f"UV count/indexing mismatch for {prim_path}")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    vertices = np.asarray(
        [tuple(transform.Transform(Gf.Vec3d(point))) for point in points],
        dtype=np.float64,
    )
    vertices *= meters_per_unit
    world_normals = np.asarray(
        [tuple(transform.TransformDir(Gf.Vec3d(normal))) for normal in normals],
        dtype=np.float64,
    )
    world_normals /= np.linalg.norm(world_normals, axis=1, keepdims=True)
    texcoords = np.asarray(uvs, dtype=np.float64)

    triangles: list[
        tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    ] = []
    offset = 0
    for count in counts:
        if count < 3:
            raise ValueError(f"face with fewer than three vertices in {prim_path}")
        for index in range(1, int(count) - 1):
            corners = (0, index, index + 1)
            triangles.append(
                tuple(
                    (int(indices[offset + corner]), offset + corner)
                    for corner in corners
                )
            )
        offset += int(count)

    lines = [
        f"# source_usd {source.resolve()}",
        f"# source_prim {prim_path}",
        "# units meter",
        f"o {Path(prim_path).name}",
    ]
    lines.extend(f"v {x:.12g} {y:.12g} {z:.12g}" for x, y, z in vertices)
    lines.extend(f"vt {u:.12g} {v:.12g}" for u, v in texcoords)
    lines.extend(f"vn {x:.12g} {y:.12g} {z:.12g}" for x, y, z in world_normals)
    lines.extend(
        "f "
        + " ".join(
            f"{vertex + 1}/{corner + 1}/{corner + 1}"
            for vertex, corner in triangle
        )
        for triangle in triangles
    )
    atomic_write(output, "\n".join(lines) + "\n")
    return {
        "source_usd": str(source.resolve()),

        "source_prim": prim_path,
        "source_meters_per_unit": meters_per_unit,
        "output_obj": str(output.resolve()),

        "vertices": int(len(vertices)),
        "source_faces": int(len(counts)),
        "triangles": int(len(triangles)),
        "face_varying_normals": int(len(world_normals)),
        "face_varying_uvs": int(len(texcoords)),
        "bounds_min_m": vertices.min(axis=0).tolist(),
        "bounds_max_m": vertices.max(axis=0).tolist(),
        "extents_m": np.ptp(vertices, axis=0).tolist(),
        "local_to_world_transform_row_major": np.asarray(
            transform, dtype=np.float64
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cupcake-usd", type=Path, required=True)
    parser.add_argument("--plate-usd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cupcake_usd = args.cupcake_usd.resolve()
    plate_usd = args.plate_usd.resolve()
    output_dir = args.output_dir.resolve()
    for path in (cupcake_usd, plate_usd):
        if not path.is_file():
            raise FileNotFoundError(path)
    outputs = [
        output_dir / "cupcake_collision.obj",
        output_dir / "cupcake_visual.obj",
        output_dir / "plate_collision.obj",
        output_dir / "manifest.json",
    ]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite: {existing}")

    assets = {
        "cupcake": export_mesh(
            cupcake_usd, "/cupcake/collisions/cupcake", outputs[0]
        ),
        "cupcake_visual": export_visual_mesh(
            cupcake_usd, "/cupcake/visuals/cupcake", outputs[1]
        ),
        "plate": export_mesh(plate_usd, "/plate/collisions/plate", outputs[2]),
    }
    manifest = {
        "schema_version": 1,
        "definition": (
            "Current USD collision prims exported with authored world transforms; "
            "coordinates are meters and no print-oriented Z translation is applied"
        ),
        "assets": assets,
        "exporter": str(Path(__file__).resolve()),
    }
    atomic_write(outputs[3], json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
