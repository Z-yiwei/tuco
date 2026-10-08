"""Cone-gated, quality-weighted log-determinant subset selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import TucoConfig


@dataclass(frozen=True)
class TucoResult:
    """Selection output and diagnostics."""

    selected_indices: np.ndarray
    greedy_order: np.ndarray
    overflow_indices: np.ndarray
    feasible_mask: np.ndarray
    utility: np.ndarray
    cosine_alignment: np.ndarray
    quality_weights: np.ndarray
    normalized_utility: np.ndarray
    marginal_coverage: np.ndarray
    normalized_marginal_coverage: np.ndarray
    objective_score: np.ndarray


def _zscore(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    if values64.size == 0:
        return np.empty(0, dtype=np.float32)
    scale = float(np.max(np.abs(values64)))
    if not np.isfinite(scale) or scale == 0.0:
        return np.zeros(values64.shape, dtype=np.float32)
    scaled = values64 / scale
    std = float(scaled.std())
    if not np.isfinite(std) or std <= np.finfo(np.float64).eps:
        return np.zeros(values64.shape, dtype=np.float32)
    return ((scaled - scaled.mean()) / std).astype(np.float32)


def _rank_desc(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Rank by descending value and then ascending candidate index."""

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values)
    return indices[np.lexsort((indices, -values[indices]))]


def select_tuco(
    influence: np.ndarray,
    returns: np.ndarray,
    budget: int,
    config: Optional[TucoConfig] = None,
) -> TucoResult:
    """Select an exact-size candidate subset from the paper's influence matrix.

    Args:
        influence: Rollout-by-candidate matrix ``X`` with shape ``(M, N)``.
        returns: Target rollout returns ``y`` with shape ``(M,)``.
        budget: Number of candidates to return.
        config: Shared paper configuration.

    The feasible prefix is selected greedily.  If the cone contains fewer
    candidates than requested, remaining slots are filled by raw performance
    utility, exactly as specified in the paper.
    """

    cfg = config or TucoConfig()
    cfg.validate()
    x = np.asarray(influence, dtype=np.float32)
    y = np.asarray(returns, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 1:
        raise ValueError("influence must be rank-2 and returns rank-1")
    if x.shape[0] != len(y):
        raise ValueError("rollout dimension does not match returns")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("influence and returns must be finite")
    num_rollouts, num_candidates = x.shape
    if not 0 <= budget <= num_candidates:
        raise ValueError("budget must be in [0, num_candidates]")
    if num_rollouts == 0:
        raise ValueError("at least one target rollout is required")
    y_norm = float(np.linalg.norm(y))
    if y_norm <= cfg.eps:
        raise ValueError("target returns have zero norm")

    profiles = x.T
    profile_norm = np.linalg.norm(profiles, axis=1)
    utility = profiles @ y
    cosine = np.zeros(num_candidates, dtype=np.float32)
    nonzero = profile_norm > cfg.eps
    cosine[nonzero] = utility[nonzero] / (profile_norm[nonzero] * y_norm)
    feasible = nonzero & (utility > 0.0) & (cosine >= cfg.tau)

    y_hat = y / y_norm
    residual = profiles - np.outer(profiles @ y_hat, y_hat)
    residual_norm = np.linalg.norm(residual, axis=1)
    directions = np.zeros_like(residual, dtype=np.float32)
    nonzero_residual = residual_norm > cfg.eps
    directions[nonzero_residual] = (
        residual[nonzero_residual] / residual_norm[nonzero_residual, None]
    )

    weights = np.zeros(num_candidates, dtype=np.float32)
    if np.any(feasible):
        weights[feasible] = utility[feasible] / float(
            utility[feasible].mean(dtype=np.float64)
        )

    normalized_utility = np.zeros(num_candidates, dtype=np.float32)
    feasible_indices = np.flatnonzero(feasible)
    normalized_utility[feasible_indices] = _zscore(utility[feasible_indices])
    marginal = np.zeros(num_candidates, dtype=np.float32)
    normalized_marginal = np.zeros(num_candidates, dtype=np.float32)
    objective = np.full(num_candidates, -np.inf, dtype=np.float32)

    greedy: list[int] = []
    remaining = feasible_indices.tolist()
    inverse_coverage = np.eye(num_rollouts, dtype=np.float32)
    target_greedy_size = min(budget, len(remaining))
    while len(greedy) < target_greedy_size:
        active = np.asarray(remaining, dtype=np.int64)
        vectors = directions[active]
        quadratic = np.einsum(
            "ij,ij->i", vectors @ inverse_coverage, vectors, optimize=True
        )
        quadratic = np.maximum(quadratic, 0.0)
        raw_gain = np.log1p(cfg.rho * weights[active] * quadratic).astype(np.float32)
        normalized_gain = _zscore(raw_gain)
        gains = normalized_utility[active] + cfg.lambda_cov * normalized_gain
        best_position = int(np.lexsort((active, -gains))[0])
        best = int(active[best_position])

        greedy.append(best)
        marginal[best] = raw_gain[best_position]
        normalized_marginal[best] = normalized_gain[best_position]
        objective[best] = gains[best_position]

        beta = cfg.rho * float(weights[best])
        vector = directions[best]
        inverse_vector = inverse_coverage @ vector
        denominator = 1.0 + beta * float(vector @ inverse_vector)
        if denominator > cfg.eps:
            inverse_coverage -= (beta / denominator) * np.outer(
                inverse_vector, inverse_vector
            )
        remaining.remove(best)

    overflow_count = budget - len(greedy)
    if overflow_count:
        available = np.setdiff1d(
            np.arange(num_candidates, dtype=np.int64),
            np.asarray(greedy, dtype=np.int64),
            assume_unique=True,
        )
        overflow = _rank_desc(utility, available)[:overflow_count]
    else:
        overflow = np.empty(0, dtype=np.int64)
    selected = np.concatenate(
        [np.asarray(greedy, dtype=np.int64), overflow.astype(np.int64)]
    )
    if len(selected) != budget or len(np.unique(selected)) != budget:
        raise RuntimeError("failed to construct an exact-size unique subset")

    return TucoResult(
        selected_indices=selected,
        greedy_order=np.asarray(greedy, dtype=np.int64),
        overflow_indices=overflow.astype(np.int64),
        feasible_mask=feasible,
        utility=utility.astype(np.float32),
        cosine_alignment=cosine,
        quality_weights=weights,
        normalized_utility=normalized_utility,
        marginal_coverage=marginal,
        normalized_marginal_coverage=normalized_marginal,
        objective_score=objective,
    )
