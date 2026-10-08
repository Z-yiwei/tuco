"""Verify that a selection artifact uses the fixed paper configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from tuco.artifacts import load_selection


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-ids", type=Path, required=True)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--num-candidates", type=int)
    args = parser.parse_args(argv)
    selected = load_selection(
        args.selected_ids,
        expected_budget=args.budget,
        expected_candidates=args.num_candidates,
    )
    print(f"verified {len(selected)} selected candidates", flush=True)


if __name__ == "__main__":
    main()
