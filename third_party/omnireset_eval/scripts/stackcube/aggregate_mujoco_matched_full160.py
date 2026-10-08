#!/usr/bin/env python3
"""Aggregate successful matched StackCube MuJoCo full-160 shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import numpy as np
import zarr

PROTOCOL_ID = "stackcube-mujoco-matched-full160-v1-controller-events"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parts", action="append", required=True,
        help="directory containing full160 shards; repeat to merge candidate pools",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--demos", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_abs_action",
        type=float,
        default=None,
        help=(
            "optional episode-level integrity gate: reject a successful episode "
            "if any recorded teacher action exceeds this absolute value"
        ),
    )
    parser.add_argument(
        "--reference_selection",
        default=None,
        help=(
            "optional existing aggregate whose selected episode identities are "
            "preserved; identities rejected by the action gate are replaced "
            "deterministically from the remaining eligible pool"
        ),
    )
    args = parser.parse_args()
    paths = sorted(
        path
        for directory in args.parts
        for path in Path(directory).resolve().glob("*.zarr")
    )
    if not paths:
        raise FileNotFoundError(f"no Zarr shards under {args.parts}")
    records, attempted = [], []
    invariant = None
    for path in paths:
        group = zarr.open(str(path), mode="r")
        if group.attrs.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"protocol mismatch: {path}")
        current = tuple(group.attrs.get(key) for key in (
            "teacher_checkpoint_sha256", "success_position_threshold_m",
            "success_orientation_xy_threshold_rad", "post_success_rows", "release_rows",
        ))
        if invariant is None:
            invariant = current
        elif current != invariant:
            raise ValueError(f"shard invariant mismatch: {path}")
        ends = np.asarray(group["meta/episode_ends"], dtype=np.int64)
        starts = np.r_[0, ends[:-1]]
        lengths = ends - starts
        if len(ends) and (not np.all(lengths == 160) or int(ends[-1]) != group["data/state"].shape[0]):
            raise ValueError(f"full160 boundary mismatch: {path}")
        source_eps = np.asarray(group["meta/source_episode_index"], dtype=np.int32)
        reset_ids = np.asarray(group["meta/reset_indices"], dtype=np.int32)
        first_success = np.asarray(group["meta/first_success_step"], dtype=np.int32)
        if not (len(ends) == len(source_eps) == len(reset_ids) == len(first_success)):
            raise ValueError(f"metadata length mismatch: {path}")
        if np.any((first_success < 1) | (first_success > 160)):
            raise ValueError(f"invalid first-success step: {path}")
        fingerprint = str(group.attrs["source_fingerprint_sha256"])
        attempted.extend(
            (fingerprint, int(ep))
            for ep in np.asarray(group["meta/attempted_source_episode_index"], dtype=np.int32)
        )
        for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
            records.append({
                "source_episode": int(source_eps[i]), "reset": int(reset_ids[i]),
                "first_success": int(first_success[i]), "fingerprint": fingerprint,
                "state": np.asarray(group["data/state"][start:end]),
                "action": np.asarray(group["data/action"][start:end]),
                "action_std": np.asarray(group["data/action_std"][start:end]),
            })
    if len(attempted) != len(set(attempted)):
        raise ValueError("attempted source episodes overlap across shards")
    identities = [(row["fingerprint"], row["source_episode"]) for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("successful source episode identity is duplicated")
    records_before_action_gate = len(records)
    if args.max_abs_action is not None:
        if not np.isfinite(args.max_abs_action) or args.max_abs_action <= 0:
            raise ValueError("--max_abs_action must be finite and positive")
        records = [
            row for row in records
            if float(np.max(np.abs(row["action"]))) <= args.max_abs_action
        ]
    rejected_by_action_gate = records_before_action_gate - len(records)
    if len(records) < args.demos:
        raise RuntimeError(f"only {len(records)} successful demos available; need {args.demos}")
    rng = np.random.default_rng(args.seed)
    if args.reference_selection is None:
        chosen = rng.choice(len(records), args.demos, replace=False)
        selected = [records[int(index)] for index in sorted(chosen.tolist())]
    else:
        reference = zarr.open(str(Path(args.reference_selection).resolve()), mode="r")
        reference_fingerprints = np.asarray(
            reference["meta/source_fingerprint_sha256"]
        ).astype(str)
        reference_episodes = np.asarray(
            reference["meta/source_episode_index"], dtype=np.int32
        )
        if len(reference_episodes) != args.demos:
            raise ValueError(
                "reference selection size differs from --demos: "
                f"{len(reference_episodes)} != {args.demos}"
            )
        by_identity = {
            (row["fingerprint"], row["source_episode"]): row for row in records
        }
        selected = []
        selected_identities = set()
        for identity in zip(reference_fingerprints, reference_episodes.tolist(), strict=True):
            row = by_identity.get(identity)
            if row is not None:
                selected.append(row)
                selected_identities.add(identity)
        replacements = [
            row for row in records
            if (row["fingerprint"], row["source_episode"]) not in selected_identities
        ]
        missing = args.demos - len(selected)
        if len(replacements) < missing:
            raise RuntimeError(
                f"only {len(replacements)} replacement demos available; need {missing}"
            )
        if missing:
            replacement_indices = rng.choice(len(replacements), missing, replace=False)
            selected.extend(replacements[int(index)] for index in replacement_indices)
    state = np.concatenate([row["state"] for row in selected]).astype(np.float32, copy=False)
    action = np.concatenate([row["action"] for row in selected]).astype(np.float32, copy=False)
    action_std = np.concatenate([row["action_std"] for row in selected]).astype(np.float32, copy=False)
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        root = zarr.open(str(staging), mode="w")
        root.create_dataset("data/state", data=state, chunks=(1024, 200))
        root.create_dataset("data/action", data=action, chunks=(1024, 7))
        root.create_dataset("data/action_std", data=action_std, chunks=(1024, 7))
        root.create_dataset("meta/episode_ends", data=np.arange(1, args.demos + 1, dtype=np.int64) * 160)
        root.create_dataset("meta/source_episode_index", data=np.asarray([r["source_episode"] for r in selected], dtype=np.int32))
        root.create_dataset("meta/source_fingerprint_sha256", data=np.asarray([r["fingerprint"] for r in selected], dtype="S64"))
        root.create_dataset("meta/reset_indices", data=np.asarray([r["reset"] for r in selected], dtype=np.int32))
        root.create_dataset("meta/first_success_step", data=np.asarray([r["first_success"] for r in selected], dtype=np.int32))
        root.create_dataset("meta/success", data=np.ones(args.demos, dtype=np.bool_))
        root.attrs.update(dict(zarr.open(str(paths[0]), mode="r").attrs))
        root.attrs.update({
            "dataset_kind": "successful_fullreset_full160_mujoco_teacher_demos",
            "selection": f"np.random.default_rng({args.seed}).choice over all successful episodes without replacement",
            "selection_seed": int(args.seed), "episodes": args.demos,
            "frames": len(state), "attempted_source_episodes": len(attempted),
            "available_successes": len(records),
            "available_successes_before_action_gate": records_before_action_gate,
            "max_abs_action_gate": args.max_abs_action,
            "rejected_by_action_gate": rejected_by_action_gate,
            "reference_selection": (
                str(Path(args.reference_selection).resolve())
                if args.reference_selection is not None else None
            ),
            "source_zarr": "multiple runtime sources; see source_parts_json",
            "runtime_source_fingerprints_json": json.dumps(sorted({r["fingerprint"] for r in records})),
            "source_parts_json": json.dumps([str(path) for path in paths]),
            "aggregator_script": str(Path(__file__).resolve()),
            "aggregator_script_sha256": sha256_file(Path(__file__).resolve()),
        })
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        f"[RESULT] attempts={len(attempted)} available_successes={len(records)} "
        f"rejected_by_action_gate={rejected_by_action_gate} selected={args.demos} "
        f"frames={len(state)} output={out}"
    )


if __name__ == "__main__":
    main()
