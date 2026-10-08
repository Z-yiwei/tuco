"""Validate a TUCO or comparison-method selection artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from tuco.baseline_artifacts import PAPER_METHODS, load_curated_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-ids", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--num-candidates", type=int, required=True)
    parser.add_argument("--method", choices=PAPER_METHODS)
    args = parser.parse_args()
    selected = load_curated_selection(
        args.selected_ids,
        expected_budget=args.budget,
        expected_candidates=args.num_candidates,
        expected_method=args.method,
    )
    print(f"validated {len(selected)} selected trajectories", flush=True)


if __name__ == "__main__":
    main()
