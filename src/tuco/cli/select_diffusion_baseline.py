"""Select single-simulator Diffusion-Policy trajectories using baseline methods."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from tuco.baseline_artifacts import (
    PAPER_BASELINES,
    ranked_subset,
    write_baseline_selection,
)
from tuco.baseline_scores import cupid_scores
from tuco.single_sim.baselines import (
    datamil_one_step_scores,
    faktual_select,
    faktual_signature_features,
    normalized_gram,
    oporp_project,
    psd_scores,
    qoq_scores,
    sieve_select,
    tarot_select,
    trajectory_mean_gradients,
)


RANKED = {
    "deminf", "demoscore", "success_similarity", "cupid", "qoq", "psd", "datamil"
}

METHOD_SCORE_FIELDS = {
    "deminf": "deminf_scores",
    "demoscore": "demoscore_scores",
    "success_similarity": "success_similarity_scores",
}


def _need(arrays: np.lib.npyio.NpzFile, *keys: str) -> list[np.ndarray]:
    missing = [key for key in keys if key not in arrays]
    if missing:
        raise ValueError(f"feature artifact is missing {missing}")
    return [np.asarray(arrays[key]) for key in keys]


def _trajectories(values: np.ndarray, ends: np.ndarray) -> list[np.ndarray]:
    ends = np.asarray(ends, dtype=np.int64)
    if ends.ndim != 1 or len(ends) == 0 or np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be a non-empty increasing vector")
    if len(values) != int(ends[-1]):
        raise ValueError("frame array and episode_ends disagree")
    starts = np.concatenate([[0], ends[:-1]])
    return [values[start:stop] for start, stop in zip(starts, ends)]


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=PAPER_BASELINES, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error(f"output already exists: {args.output_dir}")

    with np.load(args.input, allow_pickle=False) as arrays:
        (candidate_ids,) = _need(arrays, "candidate_ids")
        candidate_ids = np.asarray(candidate_ids)
        if candidate_ids.ndim != 1:
            raise ValueError("candidate_ids must be a vector")
        if not 0 <= args.budget <= len(candidate_ids):
            raise ValueError("budget is outside the candidate pool")
        scores: np.ndarray | None = None
        metadata: dict[str, object] = {
            "seed": args.seed,
            "setting": "single-sim",
            "feature_artifact": str(args.input.resolve()),
            "selection_unit": "complete_trajectory",
        }

        if args.method == "random":
            selected = np.random.default_rng(args.seed).choice(
                len(candidate_ids), args.budget, replace=False
            ).astype(np.int64)
            metadata["sampling"] = "uniform_without_replacement"
        elif args.method in METHOD_SCORE_FIELDS:
            (scores,) = _need(arrays, METHOD_SCORE_FIELDS[args.method])
            scores = np.asarray(scores, dtype=np.float32)
            selected = ranked_subset(scores, args.budget)
        elif args.method == "cupid":
            influence, returns = _need(arrays, "influence", "returns")
            scores = cupid_scores(influence, returns)
            selected = ranked_subset(scores, args.budget)
            metadata.update({
                "projection_dim": 4000,
                "diffusion_samples_per_transition": 64,
                "utility": "successful_influence_minus_failed_influence",
            })
        elif args.method == "psd":
            position, ends = _need(arrays, "ee_position", "episode_ends")
            scores = psd_scores(_trajectories(position, ends))
            selected = ranked_subset(scores, args.budget)
            metadata.update({
                "transform": "full_DFT_no_detrending_no_DC_removal",
                "score": "negative_total_power",
            })
        elif args.method == "faktual":
            path, ends = _need(arrays, "state_action_path", "episode_ends")
            features = faktual_signature_features(_trajectories(path, ends))
            selected = faktual_select(
                normalized_gram(features), args.budget, seed=args.seed,
                entropy_fraction=0.5, epsilon=0.1,
                logdet_regularization=1e-8,
            )
            metadata.update({
                "signature_level": 2,
                "entropy_fraction": 0.5,
                "stochastic_greedy_tolerance": 0.1,
                "determinant_regularization": 1e-8,
            })
        elif args.method == "sieve":
            state, action, ends = _need(
                arrays, "state", "action", "episode_ends"
            )
            selected = sieve_select(
                _trajectories(state, ends),
                _trajectories(action, ends),
                args.budget,
                seed=args.seed,
                persistence=5,
                pca_dim=256,
                min_avg_cluster_size=20,
            )
            metadata.update({
                "segment_persistence": 5,
                "pca_dim_max": 256,
                "primitive_count_candidates": 20,
                "search_subset_fraction": 0.1,
                "minimum_average_cluster_size": 20,
            })
        elif args.method == "qoq":
            source, source_ids, target = _need(
                arrays, "source_gradients", "source_episode_ids",
                "target_gradients",
            )
            counts = np.asarray(
                [np.count_nonzero(source_ids == value) for value in candidate_ids]
            )
            if np.any(counts < 1) or np.any(counts > 32):
                raise ValueError(
                    "QoQ requires between one and 32 windows per candidate"
                )
            source, target = oporp_project(source, target, 2048, args.seed)
            scores = qoq_scores(
                source, source_ids, target, candidate_ids,
            )
            selected = ranked_subset(scores, args.budget)
            metadata.update({
                "gradient_scope": "final_denoising_block",
                "projection": "OPORP_2048",
                "max_windows_per_trajectory": 32,
                "aggregation": "max_validation_window_then_mean_trajectory",
            })
        elif args.method == "datamil":
            source, source_ids, target = _need(
                arrays, "source_gradients", "source_episode_ids",
                "target_gradients",
            )
            source_mean = trajectory_mean_gradients(
                source, source_ids, candidate_ids
            )
            scores = datamil_one_step_scores(source_mean, target)
            selected = ranked_subset(scores, args.budget)
            metadata.update({
                "gradient_scope": "final_denoising_block",
                "estimator": "one_step_trajectory_weight_metagradient",
                "objective": "negative_target_validation_loss",
            })
        elif args.method == "tarot":
            source, target = _need(
                arrays, "candidate_trajectory_gradients",
                "target_trajectory_gradients",
            )
            source, target = oporp_project(source, target, 2048, args.seed)
            selected = tarot_select(
                source, target, args.budget,
                covariance_jitter=1e-5,
                sinkhorn_epsilon=0.01,
                sinkhorn_iterations=200,
            )
            metadata.update({
                "gradient_scope": "mean_final_denoising_block",
                "projection": "OPORP_2048",
                "covariance_jitter": 1e-5,
                "sinkhorn_epsilon": 0.01,
                "sinkhorn_iterations": 200,
                "data_repetition_weights": False,
            })
        else:
            raise AssertionError(args.method)

    if len(candidate_ids) != (
        len(scores) if scores is not None else len(candidate_ids)
    ):
        raise ValueError("method scores do not match candidate_ids")
    write_baseline_selection(
        args.output_dir,
        method=args.method,
        candidate_ids=candidate_ids,
        selected_indices=selected,
        scores=scores if args.method in RANKED else None,
        metadata=metadata,
    )
    print(
        f"method={args.method} selected={len(selected)}/{len(candidate_ids)} "
        f"output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
