#!/usr/bin/env python3
"""Merge disjoint eval_mujoco_matched_runtime.py shards with strict checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected_episodes", type=int, default=100)
    args = parser.parse_args()

    part_paths = sorted(Path(args.parts_dir).resolve().glob("start*"))
    if not part_paths:
        raise FileNotFoundError(f"no start* parts under {args.parts_dir}")
    summaries, rows = [], []
    for part in part_paths:
        with (part / "closed_loop_stackcube_summary.json").open() as file:
            summary = json.load(file)
        summaries.append(summary)
        with (part / "closed_loop_stackcube_episodes.jsonl").open() as file:
            part_rows = [json.loads(line) for line in file if line.strip()]
        if len(part_rows) != len(summary["runtime_episode_indices"]):
            raise ValueError(f"row/summary mismatch in {part}")
        for row, runtime_index in zip(
            part_rows, summary["runtime_episode_indices"], strict=True
        ):
            row["episode"] = int(runtime_index)
            rows.append(row)

    checkpoint_hashes = {s["checkpoint_sha256"] for s in summaries}
    runtime_sources = {s["runtime_source"] for s in summaries}
    if len(checkpoint_hashes) != 1 or len(runtime_sources) != 1:
        raise ValueError("parts do not share one checkpoint/runtime source")
    rows.sort(key=lambda row: row["episode"])
    expected = list(range(args.expected_episodes))
    actual = [int(row["episode"]) for row in rows]
    if actual != expected:
        raise ValueError(f"runtime episode coverage mismatch: {actual} != {expected}")

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "closed_loop_stackcube_episodes.jsonl").open("w") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    stable_values = np.asarray(
        [row["success_stable_after_release"] for row in rows], dtype=np.bool_
    )
    proxy_values = np.asarray(
        [row["first_proxy_step"] >= 0 for row in rows], dtype=np.bool_
    )
    reset_indices = np.asarray([row["reset_index"] for row in rows], dtype=np.int32)
    np.savez_compressed(
        out / "closed_loop_stackcube_eval.npz",
        reset_indices=reset_indices,
        stable_success=stable_values,
        proxy_reached=proxy_values,
    )
    merged = dict(summaries[0])
    merged.update(
        episodes=len(rows),
        stable_after_release_successes=int(stable_values.sum()),
        stable_after_release_sr=float(stable_values.mean()),
        proxy_reached_successes=int(proxy_values.sum()),
        proxy_reached_sr=float(proxy_values.mean()),
        runtime_episode_indices=expected,
        reset_indices=reset_indices.tolist(),
        merged_parts=[str(path) for path in part_paths],
    )
    with (out / "closed_loop_stackcube_summary.json").open("w") as file:
        json.dump(merged, file, indent=2, sort_keys=True)
    print(
        f"[RESULT] stable={stable_values.sum()}/{len(rows)}={stable_values.mean():.4f} "
        f"proxy={proxy_values.sum()}/{len(rows)}={proxy_values.mean():.4f}"
    )


if __name__ == "__main__":
    main()
