#!/usr/bin/env python3
"""Create a deterministic whole-episode subset of a state/action Zarr dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr


def sha256_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sampling",
        choices=("choice", "nested_prefix"),
        default="choice",
        help=(
            "choice preserves the historical independent subset behavior; "
            "nested_prefix takes a prefix of one seeded permutation so larger "
            "budgets contain every episode from smaller budgets."
        ),
    )
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    out_path = Path(args.out).resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite {out_path}")

    source = zarr.open(str(source_path), mode="r")
    ends = np.asarray(source["meta/episode_ends"], dtype=np.int64)
    if not 0 < args.episodes <= len(ends):
        raise ValueError(f"--episodes must be in [1,{len(ends)}]")
    starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
    rng = np.random.default_rng(args.seed)
    if args.sampling == "nested_prefix":
        selected = np.sort(rng.permutation(len(ends))[: args.episodes])
    else:
        selected = np.sort(
            rng.choice(len(ends), size=args.episodes, replace=False)
        )
    selected = selected.astype(np.int64)
    frame_indices = np.concatenate(
        [np.arange(starts[index], ends[index], dtype=np.int64) for index in selected]
    )
    lengths = ends[selected] - starts[selected]
    subset_ends = np.cumsum(lengths, dtype=np.int64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = zarr.open(str(out_path), mode="w")
    for key in ("data/state", "data/prev_state", "data/action"):
        if key not in source:
            continue
        value = np.asarray(source[key])[frame_indices]
        chunks = (min(2048, len(value)),) + value.shape[1:]
        output.create_dataset(key, data=value, chunks=chunks)
    output.create_dataset("meta/episode_ends", data=subset_ends)
    output.create_dataset("meta/source_episode_index", data=selected)

    for key in source["meta"].array_keys():
        full_key = f"meta/{key}"
        if full_key in output or key == "episode_ends":
            continue
        value = np.asarray(source[full_key])
        if value.ndim >= 1 and len(value) == len(ends):
            output.create_dataset(full_key, data=value[selected])

    output.attrs.update(dict(source.attrs))
    output.attrs.update(
        {
            "subset_source": str(source_path),
            "subset_source_episode_count": int(len(ends)),
            "subset_episodes": int(args.episodes),
            "subset_seed": int(args.seed),
            "subset_sampling": args.sampling,
            "subset_source_episode_indices_json": json.dumps(selected.tolist()),
            "subset_source_episode_indices_sha256": sha256_array(selected),
            "episodes": int(args.episodes),
            "frames": int(subset_ends[-1]),
        }
    )
    print(
        f"[subset] {source_path}: {len(ends)} episodes -> "
        f"{args.episodes} episodes / {int(subset_ends[-1])} frames -> {out_path}"
    )


if __name__ == "__main__":
    main()
