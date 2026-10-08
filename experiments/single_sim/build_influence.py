#!/usr/bin/env python3
"""Aggregate CUPID sample-pair TRAK scores using the paper definition."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from tuco.aggregation import aggregate_pairwise_samples
from tuco.config import PAPER_AGGREGATION


def _episodes(ends: np.ndarray) -> list[np.ndarray]:
    ends = np.asarray(ends, dtype=np.int64)
    if ends.ndim != 1 or len(ends) == 0:
        raise ValueError("episode ends must be a non-empty vector")
    if np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise ValueError("episode ends must be strictly increasing and positive")
    starts = np.concatenate([[0], ends[:-1]])
    return [np.arange(start, stop, dtype=np.int64) for start, stop in zip(starts, ends)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise", type=Path, required=True)
    parser.add_argument(
        "--pairwise-layout",
        choices=("target_by_candidate", "candidate_by_target"),
        required=True,
        help="Axis order of the raw sample-pair score matrix.",
    )
    parser.add_argument("--target-episode-ends", type=Path, required=True)
    parser.add_argument("--candidate-episode-ends", type=Path, required=True)
    parser.add_argument("--returns", type=Path, required=True)
    parser.add_argument("--candidate-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    pairwise = np.load(args.pairwise, mmap_mode="r")
    target_ends = np.load(args.target_episode_ends)
    candidate_ends = np.load(args.candidate_episode_ends)
    target_episodes = _episodes(target_ends)
    candidate_episodes = _episodes(candidate_ends)
    if args.pairwise_layout == "candidate_by_target":
        pairwise = pairwise.T
    expected_shape = (int(target_ends[-1]), int(candidate_ends[-1]))
    if pairwise.shape != expected_shape:
        raise ValueError(
            f"pairwise matrix has shape {pairwise.shape}, expected {expected_shape} "
            f"for layout {args.pairwise_layout!r}"
        )
    returns = np.load(args.returns).astype(np.float32)
    candidate_ids = (
        np.load(args.candidate_ids)
        if args.candidate_ids
        else np.arange(len(candidate_ends), dtype=np.int64)
    )
    influence = aggregate_pairwise_samples(
        pairwise, target_episodes, candidate_episodes
    )
    if influence.shape[0] != len(returns):
        raise ValueError("return count does not match target episodes")
    if candidate_ids.shape != (len(candidate_ends),):
        raise ValueError("candidate ID count does not match candidate episodes")
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            influence=influence,
            returns=returns,
            candidate_ids=candidate_ids,
            aggregation=np.asarray(PAPER_AGGREGATION),
        )
    os.replace(temporary, args.output)
    print(f"influence={influence.shape} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
