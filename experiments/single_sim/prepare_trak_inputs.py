#!/usr/bin/env python3
"""Convert a completed CUPID TRAK run into the release's portable arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
import numpy as np
import yaml

from diffusion_policy.common.trak_util import get_policy_from_checkpoint


def candidate_layout(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative sample counts and original demo IDs in loader order."""
    episode_ends = np.asarray(dataset.replay_buffer.episode_ends[:], dtype=np.int64)
    candidate_ids = np.flatnonzero(np.asarray(dataset.train_mask, dtype=bool))
    starts = np.asarray(dataset.sampler.indices[:, 0], dtype=np.int64)
    sample_episode = np.searchsorted(episode_ends, starts, side="right")
    counts = np.asarray(
        [(sample_episode == episode).sum() for episode in candidate_ids], dtype=np.int64
    )
    if np.any(counts <= 0) or int(counts.sum()) != len(dataset):
        raise ValueError("could not reconstruct the candidate loader episode layout")
    return np.cumsum(counts), candidate_ids.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--trak-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("filter", "select"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"refusing to reuse non-empty output: {args.output}")

    _, cfg = get_policy_from_checkpoint(args.checkpoint.resolve(), device=args.device)
    train_set = hydra.utils.instantiate(cfg.task.dataset)
    holdout_set = train_set.get_holdout_dataset()
    train_ends, train_ids = candidate_layout(train_set)
    holdout_ends, holdout_ids = candidate_layout(holdout_set)

    score_path = args.trak_dir.resolve() / "scores" / "all_episodes.mmap"
    scores = np.load(score_path, mmap_mode="r")
    expected_candidates = len(train_set) + len(holdout_set)
    metadata_path = args.eval_dir.resolve() / "episodes" / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    target_lengths = np.asarray(metadata["episode_lengths"], dtype=np.int64)
    target_ends = np.cumsum(target_lengths)
    if scores.shape != (expected_candidates, int(target_ends[-1])):
        raise ValueError(
            f"TRAK score shape {scores.shape}, expected "
            f"({expected_candidates}, {int(target_ends[-1])})"
        )
    if args.split == "filter":
        pairwise = scores[: len(train_set)]
        candidate_ends, candidate_ids = train_ends, train_ids
    else:
        pairwise = scores[len(train_set) :]
        candidate_ends, candidate_ids = holdout_ends, holdout_ids
    successes = np.asarray(metadata["episode_successes"], dtype=bool)
    if successes.shape != target_lengths.shape:
        raise ValueError("rollout successes do not match rollout episode lengths")
    returns = np.where(successes, 1.0, -1.0).astype(np.float32)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "PAIRWISE.npy", np.asarray(pairwise, dtype=np.float32))
    np.save(args.output / "TARGET_ENDS.npy", target_ends)
    np.save(args.output / "CANDIDATE_ENDS.npy", candidate_ends)
    np.save(args.output / "RETURNS.npy", returns)
    np.save(args.output / "CANDIDATE_IDS.npy", candidate_ids)
    record = {
        "split": args.split,
        "pairwise_layout": "candidate_by_target",
        "pairwise_shape": list(pairwise.shape),
        "target_episodes": int(len(target_ends)),
        "candidate_episodes": int(len(candidate_ends)),
        "checkpoint": str(args.checkpoint.resolve()),
        "trak_dir": str(args.trak_dir.resolve()),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
