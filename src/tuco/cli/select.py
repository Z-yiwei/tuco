"""Select candidates from a rollout-by-demonstration influence artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from tuco.artifacts import write_selection
from tuco.config import TucoConfig, PAPER_AGGREGATION
from tuco.selector import select_tuco


def _returns(arrays: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    values = np.asarray(arrays[key])
    if values.dtype == np.bool_:
        return np.where(values, 1.0, -1.0).astype(np.float32)
    return values.astype(np.float32)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--influence-key", default="influence")
    parser.add_argument("--returns-key", default="returns")
    parser.add_argument("--candidate-ids-key", default="candidate_ids")
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f"missing input: {args.input}")
    if args.output_dir.exists():
        parser.error(f"output already exists: {args.output_dir}")
    with np.load(args.input, allow_pickle=False) as arrays:
        for key in (args.influence_key, args.returns_key, "aggregation"):
            if key not in arrays:
                parser.error(f"input is missing key {key!r}")
        if str(arrays["aggregation"]) != PAPER_AGGREGATION:
            parser.error(
                "input aggregation does not match the paper: "
                f"expected {PAPER_AGGREGATION!r}"
            )
        influence = np.asarray(arrays[args.influence_key], dtype=np.float32)
        returns = _returns(arrays, args.returns_key)
        candidate_ids = (
            np.asarray(arrays[args.candidate_ids_key])
            if args.candidate_ids_key in arrays
            else np.arange(influence.shape[1], dtype=np.int64)
        )
    # Paper-facing runs intentionally have no per-experiment hyperparameter
    # overrides. Ablations must use a separate, explicitly labeled entrypoint.
    config = TucoConfig()
    result = select_tuco(influence, returns, args.budget, config)
    write_selection(
        args.output_dir,
        result,
        candidate_ids,
        config,
    )
    print(
        f"selected={len(result.selected_indices)}/{influence.shape[1]} "
        f"feasible={int(result.feasible_mask.sum())} output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
