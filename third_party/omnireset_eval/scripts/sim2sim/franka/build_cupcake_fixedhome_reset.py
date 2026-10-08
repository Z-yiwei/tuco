#!/usr/bin/env python3
"""Build a contract-filtered fixed-home CupCake reset tensor."""

from __future__ import annotations

import argparse
import copy
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_HOME_ARM = (
    0.00871,
    -0.10368,
    -0.00794,
    -1.49139,
    -0.00083,
    1.38774,
    0.0,
)
CANONICAL_ROOT_POSE = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def stack_rows(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        result = value.detach().cpu()
    else:
        result = torch.stack([torch.as_tensor(row) for row in value])
    if result.ndim == 3 and result.shape[1] == 1:
        result = result[:, 0]
    return result


def select_rows(value: Any, indices: list[int]) -> Any:
    if isinstance(value, dict):
        return {key: select_rows(child, indices) for key, child in value.items()}
    if torch.is_tensor(value):
        return value[indices].clone()
    if isinstance(value, list):
        return [copy.deepcopy(value[index]) for index in indices]
    raise TypeError(f"unsupported reset leaf type: {type(value)!r}")


def overwrite_rows(value: Any, row: torch.Tensor) -> None:
    if torch.is_tensor(value):
        for index in range(len(value)):
            value[index].copy_(row.to(dtype=value.dtype).reshape_as(value[index]))
        return
    if isinstance(value, list):
        for index, current in enumerate(value):
            current_tensor = torch.as_tensor(current)
            replacement = row.to(dtype=current_tensor.dtype).reshape_as(current_tensor)
            value[index] = replacement.clone()
        return
    raise TypeError(f"unsupported reset leaf type: {type(value)!r}")


def nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            nested_equal(a, b) for a, b in zip(left, right)
        )
    if torch.is_tensor(left) and torch.is_tensor(right):
        return (
            left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    return left == right


def local_z_world_z(quaternion: torch.Tensor) -> torch.Tensor:
    _, x, y, _ = quaternion.unbind(dim=1)
    return 1.0 - 2.0 * (x * x + y * y)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--home-arm", type=float, nargs=7, default=DEFAULT_HOME_ARM)
    parser.add_argument("--finger-position", type=float, default=0.04)
    parser.add_argument("--cupcake-x", type=float, nargs=2, default=(0.29, 0.35))
    parser.add_argument("--cupcake-y", type=float, nargs=2, default=(-0.03, 0.03))
    parser.add_argument("--cupcake-z", type=float, nargs=2, default=(0.028, 0.035))
    parser.add_argument("--cupcake-tilt-deg", type=float, nargs=2, default=(70.0, 78.0))
    parser.add_argument("--max-linear-speed", type=float, default=0.05)
    parser.add_argument("--max-angular-speed", type=float, default=1.0)
    parser.add_argument("--min-count", type=int, default=1)
    args = parser.parse_args()

    source_path = args.input.resolve()
    output_path = args.output.resolve()
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output_path.with_suffix(output_path.suffix + ".provenance.json")
    )
    require(source_path.is_file(), f"missing source reset tensor: {source_path}")
    if output_path.exists() or report_path.exists():
        raise FileExistsError(
            f"refusing to overwrite output/report: {output_path}, {report_path}"
        )

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    require(set(source) == {"initial_state"}, f"unexpected top-level keys: {sorted(source)}")
    initial = source["initial_state"]
    robot = initial["articulation"]["robot"]
    cupcake = initial["rigid_object"]["insertive_object"]

    cupcake_pose = stack_rows(cupcake["root_pose"]).to(torch.float64)
    cupcake_velocity = stack_rows(cupcake["root_velocity"]).to(torch.float64)
    robot_joint = stack_rows(robot["joint_position"])
    count = len(cupcake_pose)
    require(tuple(cupcake_pose.shape) == (count, 7), "CupCake root_pose must be [N,7]")
    require(tuple(cupcake_velocity.shape) == (count, 6), "CupCake root_velocity must be [N,6]")
    require(len(robot_joint) == count, "robot and CupCake reset counts differ")

    xyz = cupcake_pose[:, :3]
    axis_z = local_z_world_z(cupcake_pose[:, 3:7])
    tilt_deg = torch.rad2deg(torch.acos(axis_z.clamp(-1.0, 1.0)))
    linear_speed = cupcake_velocity[:, :3].norm(dim=1)
    angular_speed = cupcake_velocity[:, 3:6].norm(dim=1)
    gates = {
        "x": (xyz[:, 0] >= args.cupcake_x[0]) & (xyz[:, 0] <= args.cupcake_x[1]),
        "y": (xyz[:, 1] >= args.cupcake_y[0]) & (xyz[:, 1] <= args.cupcake_y[1]),
        "z": (xyz[:, 2] >= args.cupcake_z[0]) & (xyz[:, 2] <= args.cupcake_z[1]),
        "tilt": (tilt_deg >= args.cupcake_tilt_deg[0])
        & (tilt_deg <= args.cupcake_tilt_deg[1]),
        "linear_speed": linear_speed <= args.max_linear_speed,
        "angular_speed": angular_speed <= args.max_angular_speed,
    }
    valid = torch.ones(count, dtype=torch.bool)
    for gate in gates.values():
        valid &= gate
    indices = torch.where(valid)[0].tolist()
    require(
        len(indices) >= args.min_count,
        f"only {len(indices)}/{count} rows satisfy the final reset contract",
    )

    output = select_rows(source, indices)
    selected_objects = copy.deepcopy(output["initial_state"]["rigid_object"])
    output_robot = output["initial_state"]["articulation"]["robot"]
    home_joint = torch.tensor(
        (*args.home_arm, args.finger_position, args.finger_position),
        dtype=torch.float64,
    )
    overwrite_rows(output_robot["root_pose"], torch.tensor(CANONICAL_ROOT_POSE))
    overwrite_rows(output_robot["root_velocity"], torch.zeros(6))
    overwrite_rows(output_robot["joint_position"], home_joint)
    overwrite_rows(output_robot["joint_velocity"], torch.zeros(9))
    require(
        nested_equal(selected_objects, output["initial_state"]["rigid_object"]),
        "object fields changed while canonicalizing robot state",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    report_tmp = report_path.with_name(f".{report_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        torch.save(output, output_tmp)
        reloaded = torch.load(output_tmp, map_location="cpu", weights_only=False)
        reloaded_robot = reloaded["initial_state"]["articulation"]["robot"]
        q = stack_rows(reloaded_robot["joint_position"])
        qd = stack_rows(reloaded_robot["joint_velocity"])
        root = stack_rows(reloaded_robot["root_pose"])
        root_velocity = stack_rows(reloaded_robot["root_velocity"])
        expected_q = home_joint.to(dtype=q.dtype).expand(len(indices), -1)
        expected_root = torch.tensor(
            CANONICAL_ROOT_POSE, dtype=root.dtype
        ).expand(len(indices), -1)
        require(torch.equal(q, expected_q), "reloaded robot joints are not exact home")
        require(torch.equal(root, expected_root), "reloaded robot root is not canonical")
        require(torch.count_nonzero(qd).item() == 0, "reloaded joint velocity is nonzero")
        require(
            torch.count_nonzero(root_velocity).item() == 0,
            "reloaded root velocity is nonzero",
        )
        require(
            nested_equal(
                selected_objects, reloaded["initial_state"]["rigid_object"]
            ),
            "object fields changed after save/reload",
        )

        os.replace(output_tmp, output_path)
        selected_tilt = tilt_deg[valid]
        selected_xyz = xyz[valid]
        report = {
            "schema_version": 1,
            "protocol_id": "cupcake-side-lying-front3cm-fixedhome-v1",
            "definition": (
                "filter settled CupCake states by final pose/stability, preserve both "
                "object subtrees exactly, and replace only Franka state with canonical home"
            ),
            "source": {
                "path": str(source_path),
                "count": count,
            },
            "output": {
                "path": str(output_path),
                "count": len(indices),
            },
            "selected_source_indices": indices,
            "gate_pass_counts": {
                name: int(gate.sum().item()) for name, gate in gates.items()
            },
            "contract": {
                "cupcake_x_m": list(args.cupcake_x),
                "cupcake_y_m": list(args.cupcake_y),
                "cupcake_z_m": list(args.cupcake_z),
                "cupcake_tilt_deg": list(args.cupcake_tilt_deg),
                "max_linear_speed_mps": args.max_linear_speed,
                "max_angular_speed_radps": args.max_angular_speed,
                "robot_root_pose_wxyz": list(CANONICAL_ROOT_POSE),
                "robot_joint_position": home_joint.tolist(),
                "robot_velocities": "all zero",
            },
            "verification": {
                "objects_selected_bit_exact": True,
                "robot_root_exact": True,
                "robot_joints_exact": True,
                "robot_velocities_zero": True,
                "cupcake_xyz_min_m": selected_xyz.min(dim=0).values.tolist(),
                "cupcake_xyz_max_m": selected_xyz.max(dim=0).values.tolist(),
                "cupcake_tilt_min_deg": float(selected_tilt.min().item()),
                "cupcake_tilt_max_deg": float(selected_tilt.max().item()),
            },
            "builder": str(Path(__file__).resolve()),
        }
        report_tmp.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
        os.replace(report_tmp, report_path)
    finally:
        if output_tmp.exists():
            output_tmp.unlink()
        if report_tmp.exists():
            report_tmp.unlink()

    print(
        f"[cupcake-fixedhome] selected={len(indices)}/{count} "
        f"output={output_path} report={report_path}"
    )


if __name__ == "__main__":
    main()
