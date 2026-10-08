#!/usr/bin/env python3
"""Build the paper-defined influence matrix for State-MLP candidates."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from tuco.config import PAPER_AGGREGATION
from tuco.sim2sim.config import InfluenceConfig
from tuco.sim2sim.data import NormStats, episode_bounds, load_arrays, read_success_mask
from tuco.sim2sim.influence import finalize_features, gradient_features
from tuco.sim2sim.model import load_checkpoint


def _episode_sum(features: np.ndarray, ends: np.ndarray) -> np.ndarray:
    starts, stops = episode_bounds(ends)
    values = []
    for start, stop in zip(starts, stops):
        value = features[start:stop].sum(axis=0, dtype=np.float64)
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--projection-dim", type=int, default=InfluenceConfig.proj_dim)
    parser.add_argument("--gradient-batch", type=int, default=InfluenceConfig.grad_batch)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    model, norm_dict = load_checkpoint(str(args.base), device=args.device, use_ema=True)
    norm = NormStats.from_dict(norm_dict)
    (candidate_state, candidate_action), candidate_ends = load_arrays(
        str(args.candidates), ["state", "action"]
    )
    (target_state, target_action), target_ends = load_arrays(
        str(args.rollouts), ["state", "action"]
    )
    success = read_success_mask(str(args.rollouts), target_ends)
    if success is None:
        raise ValueError("target rollout Zarr must provide data/success labels")
    returns = np.where(success, 1.0, -1.0).astype(np.float32)
    cfg = InfluenceConfig(proj_dim=args.projection_dim, grad_batch=args.gradient_batch)
    candidate_gradient, projector = gradient_features(
        model,
        norm.norm_state(candidate_state),
        norm.norm_action(candidate_action),
        candidate_ends,
        np.arange(len(candidate_state)),
        cfg,
        args.device,
    )
    target_gradient, _ = gradient_features(
        model,
        norm.norm_state(target_state),
        norm.norm_action(target_action),
        target_ends,
        np.arange(len(target_state)),
        cfg,
        args.device,
        projector=projector,
    )
    del projector
    candidate_feature = finalize_features(candidate_gradient, cfg, args.device)
    target_sums = _episode_sum(target_gradient, target_ends)
    candidate_sums = _episode_sum(candidate_feature, candidate_ends)
    influence = (target_sums @ candidate_sums.T).astype(np.float32)
    _atomic_npz(
        args.output,
        influence=influence,
        returns=returns,
        candidate_ids=np.arange(len(candidate_ends), dtype=np.int64),
        aggregation=np.asarray(PAPER_AGGREGATION),
        projection_dim=np.int64(cfg.proj_dim),
        projection_seed=np.int64(cfg.proj_seed),
        ridge=np.float64(cfg.lambda_reg),
    )
    print(
        f"influence={influence.shape} aggregation={PAPER_AGGREGATION} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
