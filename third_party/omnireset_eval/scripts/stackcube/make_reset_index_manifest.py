#!/usr/bin/env python3
"""Create a tiny zarr manifest containing a deterministic reset-index schedule."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--reset_total", type=int, required=True)
    parser.add_argument(
        "--exclude_zarr", action="append", default=[],
        help="zarr whose meta/reset_indices must be excluded (repeatable)",
    )
    parser.add_argument(
        "--random_seed", type=int, default=None,
        help="sample count indices from the remaining pool instead of start/stride",
    )
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if args.count <= 0 or args.stride <= 0 or args.reset_total <= 0:
        raise ValueError("count, stride, and reset_total must be positive")
    excluded = set()
    for source_path in args.exclude_zarr:
        source = zarr.open(str(Path(source_path).resolve()), mode="r")
        excluded.update(map(int, np.asarray(source["meta/reset_indices"])))
    if args.random_seed is None:
        indices = (args.start + np.arange(args.count) * args.stride) % args.reset_total
    else:
        pool = np.asarray(
            sorted(set(range(args.reset_total)) - excluded), dtype=np.int64
        )
        if args.count > len(pool):
            raise ValueError(f"count={args.count} exceeds remaining pool={len(pool)}")
        indices = np.random.default_rng(args.random_seed).choice(
            pool, size=args.count, replace=False
        )
    if len(np.unique(indices)) != len(indices):
        raise ValueError("generated reset indices are not unique")
    overlap = sorted(set(map(int, indices)).intersection(excluded))
    if overlap:
        raise ValueError(f"generated reset indices overlap excluded IDs: {overlap}")
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(out), mode="w")
    root.create_dataset("meta/reset_indices", data=indices.astype(np.int32))
    root.attrs.update({
        "definition": "deterministic disjoint StackCube evaluation reset schedule",
        "start": int(args.start),
        "stride": int(args.stride),
        "count": int(args.count),
        "reset_total": int(args.reset_total),
        "random_seed": args.random_seed,
        "excluded_reset_count": len(excluded),
        "exclude_zarr": [str(Path(p).resolve()) for p in args.exclude_zarr],
    })
    print(f"[manifest] {out}: {len(indices)} resets, first={indices[0]}, last={indices[-1]}")


if __name__ == "__main__":
    main()
