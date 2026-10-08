#!/usr/bin/env python3
"""Freeze disjoint CupCake reset panels for sim2sim development and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)






def stack(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    else:
        result = torch.stack([torch.as_tensor(row) for row in value])
    if result.ndim == 3 and result.shape[1] == 1:
        result = result[:, 0]
    return result


def canonical_raw_state(initial: dict[str, Any]) -> np.ndarray:
    robot = initial["articulation"]["robot"]
    insertive = initial["rigid_object"]["insertive_object"]
    receptive = initial["rigid_object"]["receptive_object"]
    parts = [
        stack(robot["joint_position"]),
        stack(robot["joint_velocity"]),
        stack(robot["root_pose"]),
        stack(robot["root_velocity"]),
        stack(insertive["root_pose"]),
        stack(insertive["root_velocity"]),
        stack(receptive["root_pose"]),
        stack(receptive["root_velocity"]),
    ]
    count = int(parts[0].shape[0])
    require(all(int(part.shape[0]) == count for part in parts), "reset fields have different row counts")
    result = torch.cat([part.reshape(count, -1) for part in parts], dim=1)
    require(tuple(result.shape) == (count, 57), f"unexpected raw-state shape {tuple(result.shape)}")
    require(torch.isfinite(result).all().item(), "reset contains non-finite values")
    return result.numpy().astype(np.float32)


def summarize_panel(
    indices: np.ndarray, raw_state: np.ndarray, reachable: np.ndarray
) -> dict[str, Any]:
    selected = raw_state[indices]
    return {
        "count": int(len(indices)),
        "reachable_count": int(reachable[indices].sum()),
        "indices": indices.tolist(),


        "insertive_xyz_min_m": selected[:, 31:34].min(axis=0).astype(float).tolist(),
        "insertive_xyz_max_m": selected[:, 31:34].max(axis=0).astype(float).tolist(),
        "receptive_xyz_min_m": selected[:, 44:47].min(axis=0).astype(float).tolist(),
        "receptive_xyz_max_m": selected[:, 44:47].max(axis=0).astype(float).tolist(),
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-state", type=Path, required=True)
    parser.add_argument("--reset-type", default="ObjectAnywhereEEAnywhere")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-count", type=int, default=64)
    parser.add_argument("--selector-count", type=int, default=256)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--protocol-eval-count", type=int, default=128)
    parser.add_argument("--reachable-x", type=float, nargs=2, default=(0.2, 0.65))
    parser.add_argument("--reachable-y", type=float, nargs=2, default=(-0.3, 0.3))
    parser.add_argument("--reachable-z", type=float, nargs=2, default=(-0.005, 0.15))
    args = parser.parse_args()

    reset_path = args.reset_state.resolve()
    output_path = args.output.resolve()
    require(reset_path.is_file(), f"missing reset file: {reset_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    payload = torch.load(reset_path, map_location="cpu", weights_only=False)
    require(set(payload) == {"initial_state"}, f"unexpected reset keys: {sorted(payload)}")
    initial = payload["initial_state"]
    require(
        set(initial["articulation"]) == {"robot"},
        f"unexpected articulation assets: {sorted(initial['articulation'])}",
    )
    require(
        set(initial["rigid_object"]) == {"insertive_object", "receptive_object"},
        f"unexpected rigid assets: {sorted(initial['rigid_object'])}",
    )
    raw_state = canonical_raw_state(initial)
    total = len(raw_state)
    selected_count = (
        args.fit_count
        + args.selector_count
        + args.eval_count
        + args.protocol_eval_count
    )
    require(selected_count <= total, f"requested {selected_count} rows from {total}")

    permutation = np.random.default_rng(args.seed).permutation(total).astype(np.int64)
    insertive_xyz = raw_state[:, 31:34]
    reachable = (
        (insertive_xyz[:, 0] >= args.reachable_x[0])
        & (insertive_xyz[:, 0] <= args.reachable_x[1])
        & (insertive_xyz[:, 1] >= args.reachable_y[0])
        & (insertive_xyz[:, 1] <= args.reachable_y[1])
        & (insertive_xyz[:, 2] >= args.reachable_z[0])
        & (insertive_xyz[:, 2] <= args.reachable_z[1])
    )
    protocol_eval = permutation[: args.protocol_eval_count]
    protocol_eval_set = set(protocol_eval.tolist())
    reachable_candidates = np.asarray(
        [index for index in permutation if reachable[index] and index not in protocol_eval_set],
        dtype=np.int64,
    )
    reachable_needed = args.fit_count + args.selector_count + args.eval_count
    require(
        len(reachable_candidates) >= reachable_needed,
        f"only {len(reachable_candidates)} disjoint reachable rows for {reachable_needed} requested",
    )
    fit_end = args.fit_count
    selector_end = fit_end + args.selector_count
    eval_end = selector_end + args.eval_count
    selected_set = set(protocol_eval.tolist())
    selected_set.update(reachable_candidates[:eval_end].tolist())
    panels = {
        "fit": reachable_candidates[:fit_end],
        "selector": reachable_candidates[fit_end:selector_end],
        "eval": reachable_candidates[selector_end:eval_end],
        "protocol_eval": protocol_eval,
        "reserve": np.asarray(
            [index for index in permutation if index not in selected_set], dtype=np.int64
        ),
    }
    concatenated = np.concatenate(
        [panels["fit"], panels["selector"], panels["eval"], panels["protocol_eval"]]
    )
    require(len(np.unique(concatenated)) == len(concatenated), "panel index overlap")
    exact_rows = {row.tobytes() for row in raw_state[concatenated]}
    require(len(exact_rows) == len(concatenated), "duplicate canonical raw states across selected panels")

    report = {
        "schema_version": 1,
        "definition": (
            f"CupCake-on-Plate {args.reset_type} sim2sim panels; fit is the only "
            "physics-tuning panel, selector is for successful-trajectory H5 selection, "
            "eval is reachable held-out, and protocol_eval preserves uniform sampling from "
            "the original reset file"
        ),
        "task": "CupCake__Plate",
        "reset_type": args.reset_type,
        "seed": int(args.seed),
        "reset_state": {
            "path": str(reset_path),

            "count": int(total),

        },
        "success_contract": {
            "position_threshold_m": 0.005,
            "orientation_xy_threshold_rad": 0.025,
            "orientation_metric": "abs(wrapped_roll)+abs(wrapped_pitch); yaw ignored",
            "stable_after_release_reported_separately": True,
        },
        "reachability_audit": {
            "definition": (
                "expanded Franka task workspace used only for fit/selector/eval; "
                "protocol_eval remains unfiltered"
            ),
            "insertive_xyz_bounds_m": {
                "x": list(args.reachable_x),
                "y": list(args.reachable_y),
                "z": list(args.reachable_z),
            },
            "reachable_count": int(reachable.sum()),
            "unreachable_count": int((~reachable).sum()),
        },
        "panels": {
            name: summarize_panel(indices, raw_state, reachable)
            for name, indices in panels.items()
        },
        "overlap": {"index": 0, "exact_raw_state": 0},
        "builder_script": str(Path(__file__).resolve()),
    }
    atomic_write_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
