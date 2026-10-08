#!/usr/bin/env python3
"""Write a validated curation result in the format consumed by CUPID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from tuco.baseline_artifacts import load_curated_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection-dir",
        type=Path,
        required=True,
        help="Directory written by a full-budget tuco-select invocation.",
    )
    parser.add_argument("--split", choices=("filter", "select"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    selected_ids = load_curated_selection(
        args.selection_dir / "selected_ids.json"
    )
    method = json.loads(
        (args.selection_dir / "metadata.json").read_text(encoding="utf-8")
    )["method"]
    with np.load(args.selection_dir / "selection.npz", allow_pickle=False) as arrays:
        candidate_ids = arrays["candidate_ids"]
        indices = arrays["selected_indices"]
        scores = arrays["scores"] if "scores" in arrays else None
    ranked = candidate_ids[indices]
    if not np.array_equal(ranked, selected_ids):
        parser.error("verified JSON and ranking order disagree")
    if args.split == "filter":
        if len(ranked) == len(candidate_ids):
            ranked = ranked[::-1]
        elif scores is not None:
            removed = np.setdiff1d(
                np.arange(len(candidate_ids)), indices, assume_unique=True
            )
            order = np.lexsort((removed, scores[removed]))
            ranked = candidate_ids[removed[order]]
        else:
            ranked = np.setdiff1d(candidate_ids, ranked)
    payload = {method: {int(args.seed): ranked.astype(int).tolist()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.split} ranking to {args.output}", flush=True)


if __name__ == "__main__":
    main()
