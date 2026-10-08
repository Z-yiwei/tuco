"""Paper-defined construction of rollout-by-demonstration influence."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def aggregate_feature_sums(
    target_sums: np.ndarray,
    candidate_sums: np.ndarray,
    inverse_hessian: np.ndarray,
) -> np.ndarray:
    """Build ``X`` using summed target and candidate trajectory gradients.

    This implements the current paper and appendix definition
    ``X_ri = sum_h sum_k Psi((r,h),(i,k))``.
    """

    target = np.asarray(target_sums, dtype=np.float64)
    candidate = np.asarray(candidate_sums, dtype=np.float64)
    h_inv = np.asarray(inverse_hessian, dtype=np.float64)
    if target.ndim != 2 or candidate.ndim != 2 or h_inv.ndim != 2:
        raise ValueError("feature arrays and inverse_hessian must be rank-2")
    if target.shape[1] != candidate.shape[1]:
        raise ValueError("target and candidate feature dimensions differ")
    if h_inv.shape != (candidate.shape[1], candidate.shape[1]):
        raise ValueError("inverse_hessian has an incompatible shape")
    return (target @ h_inv @ candidate.T).astype(np.float32)


def aggregate_pairwise_samples(
    pairwise_scores: np.ndarray,
    target_episodes: Sequence[np.ndarray],
    candidate_episodes: Sequence[np.ndarray],
) -> np.ndarray:
    """Aggregate sample-pair influences according to the same paper equation."""

    scores = np.asarray(pairwise_scores)
    if scores.ndim != 2:
        raise ValueError("pairwise_scores must be rank-2")
    result = np.empty(
        (len(target_episodes), len(candidate_episodes)), dtype=np.float32
    )
    for target_index, target_ids in enumerate(target_episodes):
        target_ids = np.asarray(target_ids, dtype=np.int64)
        for candidate_index, candidate_ids in enumerate(candidate_episodes):
            candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
            if len(candidate_ids) == 0 or len(target_ids) == 0:
                raise ValueError("empty trajectories are not supported")
            block = scores[np.ix_(target_ids, candidate_ids)]
            result[target_index, candidate_index] = block.sum(dtype=np.float64)
    return result
