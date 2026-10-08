#!/usr/bin/env python3
"""Create disjoint training and evaluation reset panels for CupCake."""

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


RESET_TYPE = "CupCakeSideLyingFront3cmFixedHome"


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


def canonical_raw_state(initial: dict[str, Any]) -> np.ndarray:
    robot = initial["articulation"]["robot"]
    insertive = initial["rigid_object"]["insertive_object"]
    receptive = initial["rigid_object"]["receptive_object"]
    parts = [
        stack_rows(robot["joint_position"]),
        stack_rows(robot["joint_velocity"]),
        stack_rows(robot["root_pose"]),
        stack_rows(robot["root_velocity"]),
        stack_rows(insertive["root_pose"]),
        stack_rows(insertive["root_velocity"]),
        stack_rows(receptive["root_pose"]),
        stack_rows(receptive["root_velocity"]),
    ]
    count = int(parts[0].shape[0])
    require(all(int(part.shape[0]) == count for part in parts), "reset fields have different row counts")
    raw = torch.cat([part.reshape(count, -1) for part in parts], dim=1)
    require(tuple(raw.shape) == (count, 57), f"unexpected raw-state shape {tuple(raw.shape)}")
    require(torch.isfinite(raw).all().item(), "reset contains non-finite values")
    return np.ascontiguousarray(raw.numpy().astype(np.float64))


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        torch.save(payload, staging)
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def atomic_npz_save(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with staging.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260819)
    parser.add_argument("--attempt-seed", type=int, default=43)
    parser.add_argument("--train-count", type=int, default=250)
    parser.add_argument("--eval-count", type=int, default=50)
    parser.add_argument("--a-attempt-count", type=int, default=1800)
    args = parser.parse_args()

    source = args.reset_state.resolve()
    output_dir = args.output_dir.resolve()
    train_pt = output_dir / "isaac_train" / "Resets" / "CupCake__Plate" / f"resets_{RESET_TYPE}.pt"
    attempt_npz = output_dir / f"mujoco_A_attempt_pool_n{args.a_attempt_count}_seed{args.attempt_seed}.npz"
    eval_npz = output_dir / f"mujoco_eval_holdout_n{args.eval_count}_seed{args.split_seed}.npz"
    report_path = output_dir / "panel_provenance.json"
    outputs = (train_pt, attempt_npz, eval_npz, report_path)

    require(source.is_file(), f"missing reset state: {source}")
    if any(path.exists() for path in outputs):
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(f"refusing to overwrite panel outputs: {existing}")

    payload = torch.load(source, map_location="cpu", weights_only=False)
    require(set(payload) == {"initial_state"}, f"unexpected reset keys: {sorted(payload)}")
    raw = canonical_raw_state(payload["initial_state"])
    total = len(raw)
    require(args.train_count > 0 and args.eval_count > 0, "train/eval counts must be positive")
    require(args.a_attempt_count > 0, "A attempt count must be positive")
    require(args.train_count + args.eval_count <= total, f"need {args.train_count + args.eval_count} rows, have {total}")

    split_rng = np.random.default_rng(args.split_seed)
    permutation = split_rng.permutation(total).astype(np.int64)
    train_indices = permutation[: args.train_count]
    eval_indices = permutation[args.train_count : args.train_count + args.eval_count]
    require(not set(train_indices.tolist()) & set(eval_indices.tolist()), "train/eval index overlap")

    train_payload = select_rows(payload, train_indices.tolist())
    atomic_torch_save(train_pt, train_payload)

    attempt_rng = np.random.default_rng(args.attempt_seed)
    attempt_positions = attempt_rng.integers(0, len(train_indices), size=args.a_attempt_count)
    attempt_source_indices = train_indices[attempt_positions]
    attempt_raw = raw[attempt_source_indices]
    eval_raw = raw[eval_indices]
    train_row_hashes = {row.tobytes() for row in raw[train_indices]}
    eval_row_hashes = {row.tobytes() for row in eval_raw}
    require(train_row_hashes.isdisjoint(eval_row_hashes), "exact raw-state leakage between train and eval")

    common = {
        "reset_type": np.asarray(RESET_TYPE),
        "source_reset_state": np.asarray(str(source)),
        "split_seed": np.asarray(args.split_seed, dtype=np.int64),
    }
    atomic_npz_save(
        attempt_npz,
        raw_state=attempt_raw,
        fixedhome_source_index=attempt_source_indices,
        panel=np.asarray("train_resampled_for_mujoco_A_collection"),
        attempt_seed=np.asarray(args.attempt_seed, dtype=np.int64),
        **common,
    )
    atomic_npz_save(
        eval_npz,
        raw_state=eval_raw,
        fixedhome_source_index=eval_indices,
        panel=np.asarray("heldout_eval"),
        **common,
    )

    report = {
        "schema_version": 1,
        "protocol_id": "cupcake-side-lying-front3cm-fixedhome-cotrain-v1",
        "source": {"path": str(source), "count": total},
        "split_seed": args.split_seed,
        "attempt_seed": args.attempt_seed,
        "train": {
            "unique_count": len(train_indices),
            "source_indices": train_indices.tolist(),
            "isaac_pt": str(train_pt),
            "mujoco_attempt_count": args.a_attempt_count,
            "mujoco_attempt_npz": str(attempt_npz),
        },
        "eval": {
            "count": len(eval_indices),
            "source_indices": eval_indices.tolist(),
            "mujoco_eval_npz": str(eval_npz),
        },
        "overlap": {"source_index": 0, "exact_raw_state": 0},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = report_path.with_name(f".{report_path.name}.tmp-{uuid.uuid4().hex}")
    try:
        staging.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, report_path)
    finally:
        if staging.exists():
            staging.unlink()

    print(
        f"[fixedhome-panels] source={total} train={len(train_indices)} "
        f"eval={len(eval_indices)} A_attempts={args.a_attempt_count} output={output_dir}"
    )


if __name__ == "__main__":
    main()
