#!/usr/bin/env python3
"""Strictly merge independently collected IsaacSim demonstration Zarr shards."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import zarr


DATA_KEYS = ("state", "action", "action_std")
CONTRACT_KEYS = (
    "task",
    "reset_type",
    "reset_dataset_dir",
    "teacher_checkpoint_sha256",
    "expected_obs_dim",
    "expected_act_dim",
    "expected_episode_steps",
    "receptive_object_usd",
    "insertive_object_usd",
    "episode_length_s",
    "decimation",
    "sim_dt",
    "dynamics_randomization",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-demos", type=int, required=True)
    parser.add_argument("--delete-shards", action="store_true")
    parser.add_argument("shards", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite final dataset: {args.output}")
    partial = args.output.with_name(args.output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)

    roots = []
    summaries = []
    reference_contract = None
    total_demos = 0
    total_steps = 0
    for path in args.shards:
        if not path.is_dir():
            raise FileNotFoundError(path)
        root = zarr.open(str(path), mode="r")
        if any(f"data/{key}" not in root for key in DATA_KEYS):
            raise KeyError(f"missing data array in {path}")
        ends = np.asarray(root["meta/episode_ends"], dtype=np.int64)
        starts = np.concatenate(([0], ends[:-1]))
        lengths = ends - starts
        contract = dict(root.attrs.get("contract", {}))
        normalized = {key: contract.get(key) for key in CONTRACT_KEYS}
        if reference_contract is None:
            reference_contract = normalized
        elif normalized != reference_contract:
            raise ValueError(
                "alignment contract mismatch across shards:\n"
                + json.dumps({"reference": reference_contract, "current": normalized}, indent=2)
            )
        if len(ends) == 0 or not np.all(lengths == 160):
            raise ValueError(f"non-160-step or empty shard: {path}")
        steps = int(ends[-1])
        if root["data/state"].shape != (steps, 200):
            raise ValueError((path, root["data/state"].shape))
        if root["data/action"].shape != (steps, 7):
            raise ValueError((path, root["data/action"].shape))
        if root["data/action_std"].shape != (steps, 7):
            raise ValueError((path, root["data/action_std"].shape))
        roots.append(root)
        summaries.append(
            {
                "path": str(path.resolve()),
                "demos": int(len(ends)),
                "steps": steps,
                "seed": contract.get("seed"),
            }
        )
        total_demos += len(ends)
        total_steps += steps

    if total_demos != args.expected_demos:
        raise ValueError(f"merged demos {total_demos} != expected {args.expected_demos}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = zarr.open(str(partial), mode="w")
    data = out.create_group("data")
    arrays = {
        "state": data.create_dataset("state", shape=(total_steps, 200), chunks=(1024, 200), dtype="f4"),
        "action": data.create_dataset("action", shape=(total_steps, 7), chunks=(1024, 7), dtype="f4"),
        "action_std": data.create_dataset(
            "action_std", shape=(total_steps, 7), chunks=(1024, 7), dtype="f4"
        ),
    }
    merged_ends = []
    offset = 0
    for root in roots:
        shard_steps = int(root["meta/episode_ends"][-1])
        for start in range(0, shard_steps, 8192):
            stop = min(start + 8192, shard_steps)
            for key in DATA_KEYS:
                block = np.asarray(root[f"data/{key}"][start:stop])
                if not np.isfinite(block).all():
                    raise ValueError(f"non-finite {key} values in shard block {start}:{stop}")
                arrays[key][offset + start : offset + stop] = block
        merged_ends.extend((np.asarray(root["meta/episode_ends"], dtype=np.int64) + offset).tolist())
        offset += shard_steps

    meta = out.create_group("meta")
    meta.create_dataset("episode_ends", data=np.asarray(merged_ends, dtype=np.int64))
    first_attrs = dict(roots[0].attrs)
    merged_contract = dict(first_attrs.get("contract", {}))
    merged_contract["seed_shards"] = [item["seed"] for item in summaries]
    merged_contract["source_shards"] = [item["path"] for item in summaries]
    first_attrs.update(
        {
            "data_role": "IsaacSim B5000 merged from strict matched collection shards",
            "simulator": "IsaacSim",
            "num_demos": int(total_demos),
            "num_steps": int(total_steps),
            "contract": merged_contract,
            "merge_summary": summaries,
        }
    )
    out.attrs.update(first_attrs)

    # Re-open before atomic publication so downstream queues never observe a partial store.
    check = zarr.open(str(partial), mode="r")
    check_ends = np.asarray(check["meta/episode_ends"], dtype=np.int64)
    if check["data/state"].shape != (args.expected_demos * 160, 200):
        raise RuntimeError("post-write state shape check failed")
    if len(check_ends) != args.expected_demos or not np.all(np.diff(np.r_[0, check_ends]) == 160):
        raise RuntimeError("post-write episode boundary check failed")
    os.replace(partial, args.output)
    print(json.dumps({"status": "PASS", "output": str(args.output), "shards": summaries}, indent=2))

    if args.delete_shards:
        for path in args.shards:
            shutil.rmtree(path)
        print("Deleted validated input shards.")


if __name__ == "__main__":
    main()
