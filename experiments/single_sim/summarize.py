#!/usr/bin/env python3
"""Summarize the final ten 50-rollout RoboMimic evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-evaluations", type=int, default=10)
    parser.add_argument("--rollouts-per-evaluation", type=int, default=50)
    args = parser.parse_args()
    log_path = args.run_dir.resolve() / "logs.json.txt"
    scores = []
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            if '"test/mean_score"' not in line:
                continue
            record = json.loads(line)
            if "test/mean_score" in record:
                scores.append(float(record["test/mean_score"]))
    if len(scores) < args.num_evaluations:
        raise ValueError(
            f"{log_path}: found {len(scores)} rollout evaluations, "
            f"need {args.num_evaluations}"
        )
    tail = scores[-args.num_evaluations :]
    mean = sum(tail) / len(tail)
    std = math.sqrt(sum((score - mean) ** 2 for score in tail) / len(tail))
    result = {
        "definition": (
            f"last {args.num_evaluations} late evaluations x "
            f"{args.rollouts_per_evaluation} rollouts"
        ),
        "num_evaluations": args.num_evaluations,
        "rollouts_per_evaluation": args.rollouts_per_evaluation,
        "total_rollouts": args.num_evaluations * args.rollouts_per_evaluation,
        "scores": tail,
        "mean_success_rate": mean,
        "population_standard_deviation": std,
        "source_log": str(log_path),
    }
    output = args.output or (args.run_dir.resolve() / "result.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{mean:.6f} ± {std:.6f} -> {output}")


if __name__ == "__main__":
    main()

