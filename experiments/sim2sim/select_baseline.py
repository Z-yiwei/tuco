#!/usr/bin/env python3
"""Select source trajectories with a paper comparison method."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tuco.baseline_artifacts import (
    PAPER_BASELINES,
    ranked_subset,
    write_baseline_selection,
)
from tuco.config import PAPER_AGGREGATION
from tuco.sim2sim.baselines import (
    OMNIRESET_CURRENT_STATE_INDICES,
    OMNIRESET_EE_POSITION_INDICES,
    datamil_metagradient_scores,
    determinant_greedy,
    faktual_signature_features,
    normalized_gram,
    psd_scores,
    qoq_scores,
)
from tuco.sim2sim.data import NormStats, load_arrays, read_success_mask
from tuco.sim2sim.external_baselines import (
    cupid_scores,
    deminf_ksg_scores,
    demoscore_scores,
    success_similarity_scores,
)
from tuco.sim2sim.model import load_checkpoint
from tuco.sim2sim.structured_baselines import (
    sieve_budget_subsets,
    tarot_budget_subsets,
)


RANKED_METHODS = {
    "deminf", "demoscore", "success_similarity", "cupid", "qoq", "psd", "datamil"
}


def _require(parser: argparse.ArgumentParser, value: Path | None, flag: str) -> Path:
    if value is None:
        parser.error(f"{flag} is required for this method")
    return value


def _influence(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        required = {"influence", "returns", "candidate_ids", "aggregation"}
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(f"influence artifact is missing {sorted(missing)}")
        if str(arrays["aggregation"]) != PAPER_AGGREGATION:
            raise ValueError("influence aggregation does not match the paper")
        return (
            np.asarray(arrays["influence"], dtype=np.float32),
            np.asarray(arrays["returns"]),
            np.asarray(arrays["candidate_ids"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=PAPER_BASELINES, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--influence", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error(f"output already exists: {args.output_dir}")
    (pool_state, pool_action), pool_ends = load_arrays(
        str(args.pool), ["state", "action"]
    )
    if not 0 <= args.budget <= len(pool_ends):
        parser.error("budget must be in [0, number of source trajectories]")
    candidate_ids = np.arange(len(pool_ends), dtype=np.int64)
    selected: np.ndarray
    scores: np.ndarray | None = None
    metadata: dict[str, object] = {
        "seed": args.seed,
        "source": str(args.pool.resolve()),
        "selection_unit": "complete_trajectory",
    }

    if args.method == "random":
        selected = np.random.default_rng(args.seed).choice(
            len(pool_ends), args.budget, replace=False
        ).astype(np.int64)
        metadata["sampling"] = "uniform_without_replacement"
    elif args.method == "deminf":
        scores = deminf_ksg_scores(
            pool_state, pool_action, pool_ends,
            k=5, seed=args.seed, workers=args.workers,
        )
        selected = ranked_subset(scores, args.budget)
        metadata.update({
            "encoder": "identity_state_action",
            "estimator": "KSG_Chebyshev",
            "k": 5,
            "trajectory_aggregation": "mean_frames",
        })
    elif args.method in {"demoscore", "success_similarity"}:
        rollouts = _require(parser, args.rollouts, "--rollouts")
        (rollout_state, rollout_action), rollout_ends = load_arrays(
            str(rollouts), ["state", "action"]
        )
        success = read_success_mask(str(rollouts), rollout_ends)
        if success is None:
            raise ValueError("rollouts must contain per-trajectory success labels")
        if args.method == "demoscore":
            if pool_action.shape[1] != 7:
                raise ValueError("paper Demo-SCORE expects a 7-D action")
            scores = demoscore_scores(
                rollout_action, rollout_ends, success,
                pool_action, pool_ends,
                seed=args.seed, epochs=200, batch_size=args.batch_size,
                device=args.device,
            )
            metadata.update({
                "rollouts": str(rollouts.resolve()),
                "classifier": "7-8-8-1",
                "epochs": 200,
                "dropout": 0.3,
                "optimizer": "AdamW",
                "learning_rate": 1e-4,
                "weight_decay": 0.1,
                "validation_fraction": 0.2,
            })
        else:
            scores = success_similarity_scores(
                rollout_state, rollout_ends, success,
                pool_state, pool_ends,
            )
            metadata.update({
                "rollouts": str(rollouts.resolve()),
                "normalization": "joint_minmax",
                "metric": "negative_mean_pairwise_euclidean",
                "target_subset": "successful_rollouts",
            })
        selected = ranked_subset(scores, args.budget)
    elif args.method == "cupid":
        influence_path = _require(parser, args.influence, "--influence")
        influence, returns, artifact_ids = _influence(influence_path)
        if influence.shape[1] != len(pool_ends):
            raise ValueError("influence candidate count does not match source pool")
        if not np.array_equal(artifact_ids, candidate_ids):
            raise ValueError("influence candidate IDs do not match source order")
        scores = cupid_scores(influence, returns)
        selected = ranked_subset(scores, args.budget)
        metadata.update({
            "influence": str(influence_path.resolve()),
            "aggregation": PAPER_AGGREGATION,
            "utility": "successful_influence_minus_failed_influence",
        })
    elif args.method == "psd":
        scores = psd_scores(pool_state, pool_ends, OMNIRESET_EE_POSITION_INDICES)
        selected = ranked_subset(scores, args.budget)
        metadata.update({
            "ee_position_indices": OMNIRESET_EE_POSITION_INDICES.tolist(),
            "transform": "full_DFT_no_detrending_no_DC_removal",
            "score": "negative_total_power",
        })
    elif args.method == "faktual":
        features = faktual_signature_features(
            pool_state, pool_action, pool_ends,
            state_indices=OMNIRESET_CURRENT_STATE_INDICES,
        )
        gram = normalized_gram(features)
        selected = determinant_greedy(
            gram, np.arange(len(pool_ends), dtype=np.int64), args.budget,
            jitter=1e-4,
        )
        metadata.update({
            "signature_level": 2,
            "state_representation": "most_recent_history_terms",
            "objective": "determinant_greedy",
            "logdet_regularization": 1e-4,
        })
    elif args.method == "sieve":
        subsets, _ = sieve_budget_subsets(
            pool_state, pool_action, pool_ends, [args.budget],
            seed=args.seed, persistence=5, pca_dim=256,
            min_avg_cluster_size=1000, num_k_candidates=20,
            subset_ratio=0.1,
        )
        selected = subsets[args.budget]
        metadata.update({
            "segment_persistence": 5,
            "pca_dim_max": 256,
            "primitive_count_candidates": 20,
            "search_subset_fraction": 0.1,
            "minimum_average_cluster_size": 1000,
        })
    else:
        target = _require(parser, args.target, "--target")
        base = _require(parser, args.base, "--base")
        (target_state, target_action), target_ends = load_arrays(
            str(target), ["state", "action"]
        )
        model, norm_dict = load_checkpoint(str(base), device="cpu", use_ema=True)
        norm = NormStats.from_dict(norm_dict)
        metadata.update({
            "target": str(target.resolve()),
            "base_checkpoint": str(base.resolve()),
        })
        if args.method == "qoq":
            scores = qoq_scores(
                model, norm,
                pool_state, pool_action, pool_ends,
                target_state, target_action, target_ends,
                device=args.device, seed=args.seed,
                projection_dim=1024, max_validation_frames=4096,
                batch_size=args.batch_size,
            )
            selected = ranked_subset(scores, args.budget)
            metadata.update({
                "gradient": "normalized_Gaussian_action_head",
                "projection": "OPORP_1024",
                "aggregation": "max_target_frame_then_mean_source_trajectory",
                "max_target_frames": 4096,
            })
        elif args.method == "datamil":
            scores = datamil_metagradient_scores(
                model, norm,
                pool_state, pool_action, pool_ends,
                target_state, target_action, target_ends,
                device=args.device, seed=args.seed,
                frame_stride=4, validation_demos=100,
                inner_steps=200, inner_batch_size=256, inner_lr=1e-3,
                feature_batch_size=args.batch_size,
            )
            selected = ranked_subset(scores, args.budget)
            metadata.update({
                "normalization": "target_domain",
                "frame_stride": 4,
                "validation_trajectories": 100,
                "inner_steps": 200,
                "inner_batch_size": 256,
                "inner_learning_rate": 1e-3,
                "scope": "frozen_trunk_action_head",
            })
        elif args.method == "tarot":
            subsets, _ = tarot_budget_subsets(
                model, norm,
                pool_state, pool_action, pool_ends,
                target_state, target_action, target_ends,
                [args.budget], device=args.device, seed=args.seed,
                projection_dim=2048, covariance_jitter=1e-5,
                sinkhorn_epsilon=0.01, sinkhorn_iterations=200,
                batch_size=args.batch_size,
            )
            selected = subsets[args.budget]
            metadata.update({
                "gradient": "mean_action_head_gradient",
                "projection": "OPORP_2048",
                "covariance_jitter": 1e-5,
                "sinkhorn_epsilon": 0.01,
                "sinkhorn_iterations": 200,
                "data_repetition_weights": False,
            })
        else:
            raise AssertionError(args.method)

    write_baseline_selection(
        args.output_dir,
        method=args.method,
        candidate_ids=candidate_ids,
        selected_indices=selected,
        scores=scores if args.method in RANKED_METHODS else None,
        metadata=metadata,
    )
    print(
        f"method={args.method} selected={len(selected)}/{len(candidate_ids)} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
