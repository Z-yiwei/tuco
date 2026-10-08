#!/usr/bin/env python3
"""Aggregate 50-rollout summaries from the last five training checkpoints."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rollouts_per_checkpoint", type=int, default=50)
    args = parser.parse_args()

    checkpoint_root = Path(args.checkpoint_dir).resolve()
    checkpoints = sorted(
        set(checkpoint_root.glob("step_*.pt"))
        | set(checkpoint_root.glob("mlp_bc_step_*.pt")),
        key=lambda path: int(re.search(r"(\d+)$", path.stem).group(1)),
    )[-5:]
    if len(checkpoints) != 5:
        raise ValueError(f"expected at least 5 checkpoints, found {len(checkpoints)}")
    steps = [int(re.search(r"(\d+)$", path.stem).group(1)) for path in checkpoints]
    expected_steps = list(range(46000, 50001, 1000))
    if steps != expected_steps:
        raise ValueError(f"checkpoint steps are not the last-five window {expected_steps}: {steps}")

    rows, reference_resets = [], None
    for step, checkpoint in zip(steps, checkpoints, strict=True):
        summary_path = (
            Path(args.eval_root).resolve()
            / f"step_{step:06d}"
            / "closed_loop_stackcube_summary.json"
        )
        with summary_path.open() as file:
            summary = json.load(file)
        if summary["episodes"] != args.rollouts_per_checkpoint:
            raise ValueError(f"{summary_path}: episodes={summary['episodes']}")
        resets = np.asarray(summary["reset_indices"], dtype=np.int64)
        if reference_resets is None:
            reference_resets = resets
        elif not np.array_equal(reference_resets, resets):
            raise ValueError(f"{summary_path}: eval reset IDs differ across checkpoints")
        rows.append({
            "step": step,
            "checkpoint": str(checkpoint),
            "successes": int(summary["stable_after_release_successes"]),
            "rollouts": int(summary["episodes"]),
            "sr": float(summary["stable_after_release_sr"]),
            "summary": str(summary_path),
        })

    successes = sum(row["successes"] for row in rows)
    rollouts = sum(row["rollouts"] for row in rows)
    result = {
        "metric": "stable_after_release_success",
        "definition": "50 rollouts from each of the last 5 checkpoints; 250 rollouts total",
        "checkpoint_steps": steps,
        "rollouts_per_checkpoint": args.rollouts_per_checkpoint,
        "num_checkpoints": 5,
        "successes": successes,
        "rollouts": rollouts,
        "sr": successes / rollouts,
        "per_checkpoint": rows,
        "eval_reset_indices_shared_across_checkpoints": reference_resets.tolist(),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    print(f"[LAST5 RESULT] stable={successes}/{rollouts}={successes/rollouts:.4f}")


if __name__ == "__main__":
    main()
