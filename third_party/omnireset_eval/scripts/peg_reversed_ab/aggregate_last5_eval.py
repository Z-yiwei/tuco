#!/usr/bin/env python3
"""Aggregate Peg evaluation: 50 rollouts from each of the last five checkpoints."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with Path(args.raw).open() as file:
        rows = json.load(file)
    if len(rows) != 5:
        raise ValueError(f"expected 5 checkpoint results, found {len(rows)}")

    steps, reference_seeds = [], None
    for row in rows:
        match = re.search(r"(?:step_)(\d+)\.pt$", row["checkpoint"])
        if match is None:
            raise ValueError(f"cannot parse checkpoint step: {row['checkpoint']}")
        steps.append(int(match.group(1)))
        if row["episodes"] != 50 or row["eval_slice"] != [400, 450]:
            raise ValueError(f"invalid eval protocol in {row['checkpoint']}")
        seeds = [item["snapshot_seed"] for item in row["episodes_detail"]]
        if reference_seeds is None:
            reference_seeds = seeds
        elif seeds != reference_seeds:
            raise ValueError("rollout reset seeds differ across checkpoints")

    expected_steps = list(range(46000, 50001, 1000))
    if steps != expected_steps:
        raise ValueError(
            f"checkpoint steps are not the last-five late-training window "
            f"{expected_steps}: {steps}"
        )

    successes = sum(int(row["successes"]) for row in rows)
    rollouts = 5 * 50
    result = {
        "metric": "peg_pose_success",
        "definition": "50 rollouts from each of the last 5 checkpoints; 250 rollouts total",
        "checkpoint_steps": steps,
        "rollouts_per_checkpoint": 50,
        "num_checkpoints": 5,
        "successes": successes,
        "rollouts": rollouts,
        "sr": successes / rollouts,
        "eval_snapshot_seeds_shared_across_checkpoints": reference_seeds,
        "per_checkpoint": rows,
        "raw": str(Path(args.raw).resolve()),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as file:
        json.dump(result, file, indent=2, sort_keys=True)
    print(f"[LAST5 RESULT] success={successes}/{rollouts}={successes/rollouts:.4f}")


if __name__ == "__main__":
    main()
