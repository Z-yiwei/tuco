"""Paper-aligned SIEVE and TAROT selectors for State-MLP trajectories."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
import torch
from scipy.special import logsumexp

from .baselines import (
    OMNIRESET_CURRENT_STATE_INDICES,
    OPORPProjector,
    _sample_episode_frames,
    observation_stacks,
)
from .data import NormStats, episode_bounds
from .model import MLPBCPolicy


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((values - mean) / std).astype(np.float32)


def _confirmed_flip_boundaries(binary: np.ndarray, persistence: int) -> list[int]:
    """Return phase starts after a new gripper state persists for N frames."""
    binary = np.asarray(binary, dtype=bool)
    if len(binary) == 0:
        return [0]
    persistence = max(1, int(persistence))
    boundaries = [0]
    current = bool(binary[0])
    i = 1
    while i < len(binary):
        if bool(binary[i]) == current:
            i += 1
            continue
        candidate = bool(binary[i])
        stop = min(len(binary), i + persistence)
        if stop - i == persistence and np.all(binary[i:stop] == candidate):
            boundaries.append(i)
            current = candidate
            i = stop
        else:
            i += 1
    return boundaries


def sieve_segment_features(
    state: np.ndarray,
    action: np.ndarray,
    ends: np.ndarray,
    *,
    state_indices: Sequence[int] = OMNIRESET_CURRENT_STATE_INDICES,
    gripper_action_index: int = 6,
    gripper_threshold: float = 0.5,
    persistence: int = 5,
    pca_dim: int = 256,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], dict[str, np.ndarray | float | int]]:
    """Build state-action analogues of SIEVE's start/middle/end V-JEPA features."""
    state_indices = np.asarray(state_indices, dtype=np.int64)
    if np.any(state_indices >= state.shape[1]):
        raise ValueError("SIEVE state indices exceed the observation dimension")
    if not 0 <= int(gripper_action_index) < action.shape[1]:
        raise ValueError("SIEVE gripper action index exceeds the action dimension")
    frame_repr = np.concatenate([state[:, state_indices], action], axis=1)
    frame_repr = _zscore(frame_repr)
    starts, ends = episode_bounds(ends)
    raw_segments: list[np.ndarray] = []
    segment_episode_ids: list[int] = []
    episode_segment_rows: list[list[int]] = []
    segment_bounds: list[tuple[int, int]] = []
    for episode_id, (start, end) in enumerate(zip(starts, ends)):
        # Match the released SIEVE implementation: gripper command is the last
        # action dimension for a 7-D robot action and open iff command > 0.5.
        # In OmniReset this teacher scalar is not applied directly to the
        # fingers, but it is the only stored command-level phase signal and is
        # therefore the closest paper-faithful segmentation control.
        is_open = action[start:end, int(gripper_action_index)] > float(gripper_threshold)
        local_starts = _confirmed_flip_boundaries(is_open, persistence)
        rows: list[int] = []
        for local_id, local_start in enumerate(local_starts):
            local_end = local_starts[local_id + 1] if local_id + 1 < len(local_starts) else end - start
            if local_end <= local_start:
                continue
            absolute_start = int(start + local_start)
            absolute_end = int(start + local_end)
            middle = absolute_start + (absolute_end - absolute_start - 1) // 2
            sample_rows = np.asarray([absolute_start, middle, absolute_end - 1], dtype=np.int64)
            raw_segments.append(frame_repr[sample_rows].reshape(-1))
            segment_episode_ids.append(episode_id)
            segment_bounds.append((absolute_start, absolute_end))
            rows.append(len(raw_segments) - 1)
        if not rows:
            raise RuntimeError(f"SIEVE produced no segment for episode {episode_id}")
        episode_segment_rows.append(rows)
    raw = np.stack(raw_segments).astype(np.float32)
    centered = raw - raw.mean(axis=0, keepdims=True)
    max_rank = min(centered.shape[0] - 1, centered.shape[1], int(pca_dim))
    if max_rank < 1:
        reduced = np.zeros((len(centered), 1), dtype=np.float32)
    elif max_rank == centered.shape[1]:
        reduced = centered
    else:
        # Feature dimension is only 159 for the released state/action contract;
        # the covariance eigendecomposition is deterministic and inexpensive.
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
        basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:max_rank]]
        reduced = (centered @ basis).astype(np.float32)
    reduced = _zscore(reduced)
    metadata: dict[str, np.ndarray | float | int] = {
        "state_indices": state_indices,
        "gripper_action_index": int(gripper_action_index),
        "gripper_threshold": float(gripper_threshold),
        "persistence": int(persistence),
        "requested_pca_dim": int(pca_dim),
        "effective_pca_dim": int(reduced.shape[1]),
        "num_segments": int(len(reduced)),
        "segment_bounds": np.asarray(segment_bounds, dtype=np.int64),
    }
    return reduced, np.asarray(segment_episode_ids, dtype=np.int64), episode_segment_rows, metadata


def _k_candidates(total_segments: int, min_k: int, min_avg_cluster_size: int,
                  num_candidates: int) -> list[int]:
    max_k = max(int(min_k), total_segments // max(1, int(min_avg_cluster_size)))
    max_k = min(max_k, total_segments - 1)
    if max_k < 2:
        return [1]
    values = np.linspace(min_k, max_k, max(2, num_candidates))
    return sorted({min(max_k, max(min_k, int(round(value)))) for value in values})


def _fit_kmeans(values: np.ndarray, k: int, seed: int, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    if k == 1:
        center = values.mean(axis=0, keepdims=True)
        return center.astype(np.float32), np.zeros(len(values), dtype=np.int64)
    values64 = values.astype(np.float64)
    if k > len(values64):
        raise ValueError(f"cannot fit k={k} to {len(values64)} segment rows")
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(values64)))]
    closest_sq = ((values64 - values64[chosen[0]]) ** 2).sum(axis=1)
    while len(chosen) < k:
        total = float(closest_sq.sum())
        if total <= 1e-20:
            remaining = np.setdiff1d(np.arange(len(values64)), np.asarray(chosen))
            chosen.append(int(remaining[0]))
        else:
            chosen.append(int(rng.choice(len(values64), p=closest_sq / total)))
        new_distance = ((values64 - values64[chosen[-1]]) ** 2).sum(axis=1)
        closest_sq = np.minimum(closest_sq, new_distance)
    centers = values64[np.asarray(chosen)].copy()
    labels = np.zeros(len(values64), dtype=np.int64)
    for _ in range(max(1, int(iterations))):
        distance = ((values64[:, None] - centers[None]) ** 2).sum(axis=2)
        new_labels = distance.argmin(axis=1).astype(np.int64)
        new_centers = centers.copy()
        nearest_distance = distance[np.arange(len(values64)), new_labels]
        for cluster_id in range(k):
            members = values64[new_labels == cluster_id]
            if len(members):
                new_centers[cluster_id] = members.mean(axis=0)
            else:
                # Deterministic farthest-point recovery prevents empty-cluster
                # failures on nearly identical one-phase robot trajectories.
                farthest = int(np.argmax(nearest_distance))
                new_centers[cluster_id] = values64[farthest]
                new_labels[farthest] = cluster_id
                nearest_distance[farthest] = -1.0
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            labels, centers = new_labels, new_centers
            break
        labels, centers = new_labels, new_centers
    return centers.astype(np.float32), labels


def _jaccard_reuse_score(labels: np.ndarray, episode_ids: np.ndarray,
                         selected_episodes: np.ndarray, k: int) -> tuple[float, float, float]:
    sets = [set(labels[episode_ids == episode_id].tolist()) for episode_id in selected_episodes]
    if len(sets) < 2:
        jaccard = 0.0
    else:
        per_episode: list[float] = []
        for i, left in enumerate(sets):
            vals = []
            for j, right in enumerate(sets):
                if i == j:
                    continue
                union = left | right
                vals.append(len(left & right) / len(union) if union else 0.0)
            per_episode.append(float(np.mean(vals)))
        jaccard = float(np.median(per_episode))
    reuse = np.zeros(k, dtype=np.int64)
    for values in sets:
        for value in values:
            reuse[value] += 1
    median_reuse = float(np.median(reuse))
    return (1.0 - jaccard) * math.log1p(median_reuse), jaccard, median_reuse


def _sieve_patterns(labels: np.ndarray, episode_segment_rows: list[list[int]]) -> tuple[list[tuple[int, ...]], dict[tuple[int, ...], list[int]]]:
    patterns: list[tuple[int, ...]] = []
    buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for episode_id, rows in enumerate(episode_segment_rows):
        pattern = tuple(int(labels[row]) for row in rows)
        patterns.append(pattern)
        buckets[pattern].append(episode_id)
    return patterns, dict(buckets)


def _transitions(pattern: tuple[int, ...]) -> list[tuple[int, int]]:
    if len(pattern) == 1:
        return [(pattern[0], -1)]
    return list(zip(pattern[:-1], pattern[1:]))


def _sieve_allocate(buckets: dict[tuple[int, ...], list[int]], budget: int) -> dict[tuple[int, ...], int]:
    patterns = sorted(buckets, key=lambda pattern: (len(pattern), pattern))
    primitive_reuse: Counter[int] = Counter()
    transition_reuse: Counter[tuple[int, int]] = Counter()
    for pattern in patterns:
        primitive_reuse.update(set(pattern))
        transition_reuse.update(set(_transitions(pattern)))
    primitive_mean = np.mean(list(primitive_reuse.values())) if primitive_reuse else 1.0
    transition_mean = np.mean(list(transition_reuse.values())) if transition_reuse else 1.0
    primitive_weights = {key: value / primitive_mean for key, value in primitive_reuse.items()}
    transition_weights = {key: value / transition_mean for key, value in transition_reuse.items()}
    primitive_occ = {pattern: Counter(pattern) for pattern in patterns}
    transition_occ = {pattern: Counter(_transitions(pattern)) for pattern in patterns}
    selected = {pattern: 0 for pattern in patterns}
    primitive_counts: defaultdict[int, int] = defaultdict(int)
    transition_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for _ in range(min(int(budget), sum(map(len, buckets.values())))):
        winner: tuple[int, ...] | None = None
        winner_key = (-np.inf, -1)
        for pattern in patterns:
            remaining = len(buckets[pattern]) - selected[pattern]
            if remaining <= 0:
                continue
            gain = 0.0
            for primitive, occurrence in primitive_occ[pattern].items():
                current = primitive_counts[primitive]
                gain += primitive_weights[primitive] * (
                    math.log1p(current + occurrence) - math.log1p(current)
                )
            for transition, occurrence in transition_occ[pattern].items():
                current = transition_counts[transition]
                gain += transition_weights[transition] * (
                    math.log1p(current + occurrence) - math.log1p(current)
                )
            key = (gain, remaining)
            if key > winner_key:
                winner, winner_key = pattern, key
        if winner is None:
            break
        selected[winner] += 1
        primitive_counts.update(primitive_occ[winner])
        transition_counts.update(transition_occ[winner])
    return {pattern: count for pattern, count in selected.items() if count}


def _sieve_central_orders(
    segment_features: np.ndarray,
    episode_segment_rows: list[list[int]],
    buckets: dict[tuple[int, ...], list[int]],
) -> dict[tuple[int, ...], np.ndarray]:
    orders: dict[tuple[int, ...], np.ndarray] = {}
    for pattern, episode_ids in buckets.items():
        vectors = np.stack([
            np.concatenate([segment_features[row] for row in episode_segment_rows[episode_id]])
            for episode_id in episode_ids
        ]).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit = vectors / np.maximum(norms, 1e-12)
        similarity_sums = np.zeros(len(unit), dtype=np.float64)
        for start in range(0, len(unit), 512):
            similarity_sums[start:start + 512] = (unit[start:start + 512] @ unit.T).sum(axis=1)
        medoid = unit[int(np.argmax(similarity_sums))]
        distance = 1.0 - unit @ medoid
        local_order = np.lexsort((np.asarray(episode_ids), distance))
        orders[pattern] = np.asarray(episode_ids, dtype=np.int64)[local_order]
    return orders


def sieve_budget_subsets(
    state: np.ndarray,
    action: np.ndarray,
    ends: np.ndarray,
    budgets: Sequence[int],
    *,
    state_indices: Sequence[int] = OMNIRESET_CURRENT_STATE_INDICES,
    seed: int = 42,
    gripper_action_index: int = 6,
    gripper_threshold: float = 0.5,
    persistence: int = 5,
    pca_dim: int = 256,
    min_k: int = 2,
    min_avg_cluster_size: int = 1000,
    num_k_candidates: int = 20,
    subset_ratio: float = 0.1,
    kmeans_iterations: int = 100,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray | float | int]]:
    features, episode_ids, episode_rows, metadata = sieve_segment_features(
        state, action, ends, state_indices=state_indices,
        gripper_action_index=gripper_action_index,
        gripper_threshold=gripper_threshold,
        persistence=persistence, pca_dim=pca_dim,
    )
    rng = np.random.default_rng(seed)
    count = min(len(ends), max(2, int(round(len(ends) * float(subset_ratio)))))
    sampled_episodes = np.sort(rng.choice(len(ends), count, replace=False))
    subset_mask = np.isin(episode_ids, sampled_episodes)
    subset_features = features[subset_mask]
    subset_episode_ids = episode_ids[subset_mask]
    candidates = [
        value for value in _k_candidates(
            len(features), min_k, min_avg_cluster_size, num_k_candidates
        )
        if value < len(subset_features)
    ]
    if not candidates:
        candidates = [1]
    best: tuple[float, int, float, float] | None = None
    for k in candidates:
        centers, _ = _fit_kmeans(subset_features, k, seed, kmeans_iterations)
        distances = ((subset_features[:, None] - centers[None]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        score, jaccard, reuse = _jaccard_reuse_score(
            labels, subset_episode_ids, sampled_episodes, k,
        )
        candidate = (score, -k, jaccard, reuse)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    best_k = -best[1]
    centers, labels = _fit_kmeans(features, best_k, seed, kmeans_iterations)
    patterns, buckets = _sieve_patterns(labels, episode_rows)
    central_orders = _sieve_central_orders(features, episode_rows, buckets)
    subsets: dict[int, np.ndarray] = {}
    for budget in sorted({int(value) for value in budgets}):
        allocation = _sieve_allocate(buckets, budget)
        selected = np.concatenate([
            central_orders[pattern][:count] for pattern, count in allocation.items()
        ]).astype(np.int64)
        if len(selected) != budget or len(np.unique(selected)) != budget:
            raise RuntimeError(f"SIEVE returned invalid budget {len(selected)} for {budget}")
        subsets[budget] = np.sort(selected)
    metadata.update({
        "best_k": int(best_k),
        "k_candidates": np.asarray(candidates, dtype=np.int64),
        "k_search_subset_ratio": float(subset_ratio),
        "k_search_episode_count": int(count),
        "min_avg_cluster_size": int(min_avg_cluster_size),
        "num_patterns": int(len(buckets)),
        "cluster_centers": centers,
        "episode_patterns": np.asarray([",".join(map(str, p)) for p in patterns]),
    })
    return subsets, metadata


@torch.inference_mode()
def _trajectory_head_gradients(
    model: MLPBCPolicy,
    norm: NormStats,
    state: np.ndarray,
    action: np.ndarray,
    ends: np.ndarray,
    *,
    device: str,
    frame_stride: int,
    batch_size: int,
) -> np.ndarray:
    """Mean per-frame BC action-head gradient for every full trajectory."""
    frames, episode_ids = _sample_episode_frames(ends, frame_stride)
    state_n = norm.norm_state(state).astype(np.float32)
    action_n = norm.norm_action(action).astype(np.float32)
    dimension = model.act_dim * model.hidden + model.act_dim
    sums = np.zeros((len(ends), dimension), dtype=np.float32)
    counts = np.zeros(len(ends), dtype=np.int64)
    model = model.to(device).eval()
    for start in range(0, len(frames), batch_size):
        idx = frames[start:start + batch_size]
        episode = episode_ids[start:start + batch_size]
        observations = observation_stacks(state_n, ends, idx, model.n_obs_steps)
        hidden = model.features(torch.from_numpy(observations).to(device))
        prediction = model.head(hidden)
        residual = torch.from_numpy(action_n[idx]).to(device) - prediction
        gradient = torch.cat(
            [(residual[:, :, None] * hidden[:, None]).flatten(1), residual], dim=1
        ).cpu().numpy().astype(np.float32)
        np.add.at(sums, episode, gradient)
        np.add.at(counts, episode, 1)
    if np.any(counts == 0):
        raise RuntimeError("TAROT found a trajectory without sampled frames")
    return sums / counts[:, None]


def _sinkhorn_source_potentials(cost: np.ndarray, epsilon: float,
                                iterations: int) -> np.ndarray:
    """Uniform entropic-OT source duals used only at TAROT's boundary layer."""
    cost = np.asarray(cost, dtype=np.float64)
    n_source, n_target = cost.shape
    log_a = -math.log(n_source)
    log_b = -math.log(n_target)
    f = np.zeros(n_source, dtype=np.float64)
    g = np.zeros(n_target, dtype=np.float64)
    epsilon = float(epsilon)
    for _ in range(int(iterations)):
        f = epsilon * (log_a - logsumexp((g[None] - cost) / epsilon, axis=1))
        g = epsilon * (log_b - logsumexp((f[:, None] - cost) / epsilon, axis=0))
    # Potentials are defined up to a constant; centering makes stored values
    # comparable without changing the official largest-potential ordering.
    return (f - f.mean()).astype(np.float32)


def _tarot_fixed_size(
    similarity: np.ndarray,
    candidate_features: np.ndarray,
    target_features: np.ndarray,
    budget: int,
    *,
    sinkhorn_epsilon: float,
    sinkhorn_iterations: int,
) -> np.ndarray:
    sorted_indices = np.argsort(-similarity, axis=0, kind="stable")
    selected: set[int] = set()
    for layer in range(sorted_indices.shape[0]):
        this_layer = np.unique(sorted_indices[layer])
        updated = selected.union(int(value) for value in this_layer)
        if len(updated) <= budget:
            selected = updated
            if len(selected) == budget:
                break
            continue
        remaining = budget - len(selected)
        boundary = np.setdiff1d(this_layer, np.fromiter(selected, dtype=np.int64))
        if remaining > 0:
            cosine = np.clip(candidate_features[boundary] @ target_features.T, -1.0, 1.0)
            cost = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine))
            potential = _sinkhorn_source_potentials(
                cost, sinkhorn_epsilon, sinkhorn_iterations,
            )
            order = np.lexsort((boundary, -potential))
            selected.update(int(value) for value in boundary[order[:remaining]])
        break
    result = np.asarray(sorted(selected), dtype=np.int64)
    if len(result) != budget:
        raise RuntimeError(f"TAROT returned {len(result)} candidates for budget={budget}")
    return result


def tarot_budget_subsets(
    model: MLPBCPolicy,
    norm: NormStats,
    pool_state: np.ndarray,
    pool_action: np.ndarray,
    pool_ends: np.ndarray,
    target_state: np.ndarray,
    target_action: np.ndarray,
    target_ends: np.ndarray,
    budgets: Sequence[int],
    *,
    device: str,
    seed: int = 42,
    projection_dim: int = 2048,
    frame_stride: int = 1,
    batch_size: int = 2048,
    covariance_jitter: float = 1e-5,
    sinkhorn_epsilon: float = 0.01,
    sinkhorn_iterations: int = 200,
) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray | float | int]]:
    pool_gradients = _trajectory_head_gradients(
        model, norm, pool_state, pool_action, pool_ends, device=device,
        frame_stride=frame_stride, batch_size=batch_size,
    )
    target_gradients = _trajectory_head_gradients(
        model, norm, target_state, target_action, target_ends, device=device,
        frame_stride=frame_stride, batch_size=batch_size,
    )
    projector = OPORPProjector.create(pool_gradients.shape[1], projection_dim, seed)
    pool_projected = projector.project(pool_gradients)
    target_projected = projector.project(target_gradients)
    all_features = np.concatenate([pool_projected, target_projected]).astype(np.float32)
    all_features -= all_features.mean(axis=0, keepdims=True)
    tensor = torch.from_numpy(all_features).to(device)
    covariance = tensor.T @ tensor / len(tensor)
    covariance.diagonal().add_(float(covariance_jitter))
    cholesky = torch.linalg.cholesky(covariance)
    # Row-vector form of L^-1 x is x @ L^-T.
    whitened = torch.linalg.solve_triangular(
        cholesky, tensor.T, upper=False,
    ).T
    whitened /= whitened.norm(dim=1, keepdim=True).clamp_min(1e-12)
    pool_features = whitened[:len(pool_ends)].cpu().numpy().astype(np.float32)
    target_features = whitened[len(pool_ends):].cpu().numpy().astype(np.float32)
    similarity = pool_features @ target_features.T
    subsets = {
        int(budget): _tarot_fixed_size(
            similarity, pool_features, target_features, int(budget),
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
        )
        for budget in sorted({int(value) for value in budgets})
    }
    metadata: dict[str, np.ndarray | float | int] = {
        "projection": "OPORP_one_permutation_contiguous_bins",
        "projection_dim": int(pool_features.shape[1]),
        "gradient_scope": "action_head_weight_and_bias",
        "gradient_checkpoint_count": 1,
        "trajectory_gradient_aggregation": "mean_over_all_sampled_frames",
        "frame_stride": int(frame_stride),
        "covariance_jitter": float(covariance_jitter),
        "whitening": "joint_candidate_target_cholesky",
        "unit_l2_normalization": True,
        "selection": "fixed_size_target_neighbor_layers",
        "boundary_ot": "entropic_sinkhorn_source_dual_largest_first",
        "sinkhorn_epsilon": float(sinkhorn_epsilon),
        "sinkhorn_iterations": int(sinkhorn_iterations),
        "data_weighting": False,
    }
    return subsets, metadata
