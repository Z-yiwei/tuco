"""Additional trajectory-level data-curation baselines for Robomimic."""

import math
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


def episode_bounds(lengths: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.ndim != 1 or np.any(lengths <= 0):
        raise ValueError("episode lengths must be a positive 1-D array")
    ends = np.cumsum(lengths)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
    return starts, ends


def stable_descending(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    return np.lexsort((np.arange(len(scores), dtype=np.int64), -scores))


# ---------------------------------------------------------------------------
# PSD
# ---------------------------------------------------------------------------
def psd_scores(trajectories_xyz: Sequence[np.ndarray]) -> np.ndarray:
    """Return negative total EE-position spectral power (larger is better)."""
    result = []
    for xyz in trajectories_xyz:
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("PSD expects one T x 3 end-effector path per demo")
        result.append(-float(np.square(np.abs(np.fft.fft(xyz, axis=0))).sum()))
    return np.asarray(result, dtype=np.float32)


# ---------------------------------------------------------------------------
# FAKTUAL
# ---------------------------------------------------------------------------
def _level2_signature(path: np.ndarray) -> np.ndarray:
    increments = np.diff(np.asarray(path, dtype=np.float64), axis=0)
    width = path.shape[1]
    first = np.zeros(width, dtype=np.float64)
    second = np.zeros((width, width), dtype=np.float64)
    for delta in increments:
        second += np.outer(first, delta) + 0.5 * np.outer(delta, delta)
        first += delta
    return np.concatenate([first, second.reshape(-1)])


def faktual_signature_features(
    trajectories: Sequence[np.ndarray], frame_stride: int = 1
) -> np.ndarray:
    """Exact normalized level-2 signatures of state-action paths."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    joined = np.concatenate([np.asarray(x, dtype=np.float64) for x in trajectories])
    mean = joined.mean(axis=0)
    std = joined.std(axis=0)
    std[std < 1e-8] = 1.0
    features = []
    for trajectory in trajectories:
        path = (np.asarray(trajectory, dtype=np.float64) - mean) / std
        rows = np.arange(0, len(path), frame_stride, dtype=np.int64)
        if rows[-1] != len(path) - 1:
            rows = np.append(rows, len(path) - 1)
        path = path[rows]
        path = np.concatenate(
            [np.linspace(0.0, 1.0, len(path))[:, None], path], axis=1
        )
        path = np.concatenate([np.zeros((1, path.shape[1])), path], axis=0)
        features.append(_level2_signature(path))
    result = np.stack(features)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result.astype(np.float32)


def normalized_gram(features: np.ndarray) -> np.ndarray:
    gram = np.asarray(features, dtype=np.float64) @ np.asarray(features, dtype=np.float64).T
    diagonal = np.sqrt(np.maximum(np.diag(gram), 1e-12))
    gram /= diagonal[:, None] * diagonal[None]
    np.clip(gram, -1.0, 1.0, out=gram)
    np.fill_diagonal(gram, 1.0)
    return gram


def _signature_entropy(subset: np.ndarray) -> float:
    if len(subset) <= 1:
        return 0.0
    values = np.linalg.eigvalsh(subset / float(len(subset)))
    values = values[values > 1e-12]
    return float(-(values * np.log(values)).sum())


def _entropy_greedy(
    gram: np.ndarray, budget: int, seed: int, epsilon: float
) -> np.ndarray:
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = []
    available = np.ones(len(gram), dtype=bool)
    sample_size = int(math.ceil((len(gram) / budget) * math.log(1.0 / epsilon)))
    for _ in range(budget):
        candidates = np.flatnonzero(available)
        if len(candidates) > sample_size:
            candidates = rng.choice(candidates, size=sample_size, replace=False)
        best_id, best_value = None, -np.inf
        for candidate in np.sort(candidates):
            ids = np.asarray(selected + [int(candidate)], dtype=np.int64)
            value = _signature_entropy(gram[np.ix_(ids, ids)])
            if value > best_value + 1e-14:
                best_id, best_value = int(candidate), value
        selected.append(best_id)
        available[best_id] = False
    return np.asarray(selected, dtype=np.int64)


def _logdet_greedy(
    gram: np.ndarray, candidates: np.ndarray, budget: int, jitter: float
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.int64)
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    local = gram[np.ix_(candidates, candidates)].astype(np.float64, copy=True)
    residual = np.diag(local).copy() + jitter
    factors = np.zeros((len(candidates), budget), dtype=np.float64)
    active = np.ones(len(candidates), dtype=bool)
    picked = []
    for column in range(budget):
        pivot = int(np.argmax(np.where(active, residual, -np.inf)))
        picked.append(pivot)
        active[pivot] = False
        correction = 0.0 if column == 0 else factors[:, :column] @ factors[pivot, :column]
        update = (local[:, pivot] - correction) / math.sqrt(max(residual[pivot], jitter))
        factors[:, column] = update
        residual = np.maximum(residual - update * update, 0.0)
    return candidates[np.asarray(picked, dtype=np.int64)]


def faktual_select(
    gram: np.ndarray,
    budget: int,
    seed: int = 0,
    entropy_fraction: float = 0.5,
    epsilon: float = 0.1,
    logdet_regularization: float = 1e-8,
) -> np.ndarray:
    budget = min(max(int(budget), 0), len(gram))
    entropy_budget = int(round(budget * float(entropy_fraction)))
    prefix = _entropy_greedy(gram, entropy_budget, seed, epsilon)
    remaining = np.setdiff1d(np.arange(len(gram)), prefix, assume_unique=True)
    suffix = _logdet_greedy(
        gram, remaining, budget - len(prefix), float(logdet_regularization)
    )
    selected = np.concatenate([prefix, suffix])
    if len(np.unique(selected)) != budget:
        raise RuntimeError("FAKTUAL returned duplicate or missing demos")
    return np.sort(selected)


# ---------------------------------------------------------------------------
# QoQ and one-step DataMIL adaptation from cached diffusion gradients
# ---------------------------------------------------------------------------
def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def oporp_project(
    source: np.ndarray,
    target: np.ndarray,
    output_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one shared OPORP map to source and target gradient features."""
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target gradients must be compatible matrices")
    output_dim = min(max(1, int(output_dim)), source.shape[1])
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(source.shape[1])
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32),
                       size=source.shape[1])
    boundaries = np.linspace(
        0, source.shape[1], output_dim + 1, dtype=np.int64
    )

    def project(values: np.ndarray) -> np.ndarray:
        permuted = values[:, permutation] * signs[None]
        return np.add.reduceat(permuted, boundaries[:-1], axis=1).astype(
            np.float32, copy=False
        )

    return project(source), project(target)


def qoq_scores(
    source_gradients: np.ndarray,
    source_episode_ids: np.ndarray,
    target_gradients: np.ndarray,
    candidate_episode_ids: Sequence[int],
    similarity_batch_size: int = 2048,
) -> np.ndarray:
    """QoQ max-validation gradient similarity, averaged within each demo."""
    source = _unit_rows(source_gradients)
    target = _unit_rows(target_gradients)
    row_scores = np.full(len(source), -np.inf, dtype=np.float32)
    for start in range(0, len(target), similarity_batch_size):
        similarity = source @ target[start:start + similarity_batch_size].T
        row_scores = np.maximum(row_scores, similarity.max(axis=1))
    scores = []
    source_episode_ids = np.asarray(source_episode_ids, dtype=np.int64)
    for episode_id in candidate_episode_ids:
        mask = source_episode_ids == int(episode_id)
        if not np.any(mask):
            raise ValueError("gradient cache has no rows for demo_%d" % int(episode_id))
        scores.append(float(row_scores[mask].mean()))
    return np.asarray(scores, dtype=np.float32)


def trajectory_mean_gradients(
    gradients: np.ndarray, episode_ids: np.ndarray, requested_ids: Sequence[int]
) -> np.ndarray:
    gradients = np.asarray(gradients, dtype=np.float32)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    result = []
    for episode_id in requested_ids:
        mask = episode_ids == int(episode_id)
        if not np.any(mask):
            raise ValueError("gradient cache has no rows for demo_%d" % int(episode_id))
        result.append(gradients[mask].mean(axis=0))
    return np.stack(result).astype(np.float32)


def datamil_one_step_scores(
    source_trajectory_gradients: np.ndarray, target_gradients: np.ndarray
) -> np.ndarray:
    """One-step trajectory-weight metagradient of negative validation loss."""
    target_mean = np.asarray(target_gradients, dtype=np.float32).mean(axis=0)
    return (np.asarray(source_trajectory_gradients) @ target_mean).astype(np.float32)


# ---------------------------------------------------------------------------
# SIEVE
# ---------------------------------------------------------------------------
def _confirmed_flip_boundaries(binary: np.ndarray, persistence: int) -> List[int]:
    boundaries = [0]
    current = bool(binary[0])
    i = 1
    while i < len(binary):
        if bool(binary[i]) == current:
            i += 1
            continue
        stop = min(len(binary), i + persistence)
        if stop - i == persistence and np.all(binary[i:stop] == bool(binary[i])):
            boundaries.append(i)
            current = bool(binary[i])
            i = stop
        else:
            i += 1
    return boundaries


def _kmeans(values: np.ndarray, k: int, seed: int, iterations: int = 100):
    values = np.asarray(values, dtype=np.float64)
    if k == 1:
        return values.mean(axis=0, keepdims=True), np.zeros(len(values), dtype=np.int64)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(values)))]
    closest = np.square(values - values[chosen[0]]).sum(axis=1)
    while len(chosen) < k:
        total = closest.sum()
        remaining = np.setdiff1d(np.arange(len(values)), np.asarray(chosen))
        pick = int(remaining[0]) if total <= 1e-20 else int(rng.choice(len(values), p=closest / total))
        chosen.append(pick)
        closest = np.minimum(closest, np.square(values - values[pick]).sum(axis=1))
    centers = values[chosen].copy()
    labels = np.zeros(len(values), dtype=np.int64)
    for _ in range(iterations):
        distance = np.square(values[:, None] - centers[None]).sum(axis=2)
        new_labels = distance.argmin(axis=1)
        new_centers = centers.copy()
        for cluster in range(k):
            members = values[new_labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(centers, new_centers):
            break
        labels, centers = new_labels, new_centers
    return centers.astype(np.float32), labels.astype(np.int64)


def _sieve_segment_features(
    trajectories: Sequence[np.ndarray], actions: Sequence[np.ndarray], persistence: int, pca_dim: int
):
    joined = np.concatenate([np.concatenate([x, a], axis=1) for x, a in zip(trajectories, actions)])
    mean, std = joined.mean(axis=0), joined.std(axis=0)
    std[std < 1e-6] = 1.0
    raw, rows_by_episode = [], []
    for trajectory, action in zip(trajectories, actions):
        representation = (np.concatenate([trajectory, action], axis=1) - mean) / std
        boundaries = _confirmed_flip_boundaries(action[:, -1] > 0.5, persistence)
        rows = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(action)
            middle = start + (end - start - 1) // 2
            raw.append(representation[[start, middle, end - 1]].reshape(-1))
            rows.append(len(raw) - 1)
        rows_by_episode.append(rows)
    raw = np.stack(raw).astype(np.float32)
    centered = raw - raw.mean(axis=0, keepdims=True)
    rank = min(max(1, len(centered) - 1), centered.shape[1], pca_dim)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    reduced = centered @ vh[:rank].T
    reduced /= np.maximum(reduced.std(axis=0, keepdims=True), 1e-6)
    return reduced.astype(np.float32), rows_by_episode


def _sieve_allocate(buckets: Dict[Tuple[int, ...], List[int]], budget: int):
    patterns = sorted(buckets, key=lambda value: (len(value), value))
    primitive_reuse, transition_reuse = Counter(), Counter()
    transitions = lambda p: list(zip(p[:-1], p[1:])) if len(p) > 1 else [(p[0], -1)]
    for pattern in patterns:
        primitive_reuse.update(set(pattern)); transition_reuse.update(set(transitions(pattern)))
    pmean = np.mean(list(primitive_reuse.values())) if primitive_reuse else 1.0
    tmean = np.mean(list(transition_reuse.values())) if transition_reuse else 1.0
    chosen = {pattern: 0 for pattern in patterns}
    pc, tc = defaultdict(int), defaultdict(int)
    for _ in range(budget):
        winner, winner_key = None, (-np.inf, -1)
        for pattern in patterns:
            remaining = len(buckets[pattern]) - chosen[pattern]
            if remaining <= 0:
                continue
            gain = sum((primitive_reuse[x] / pmean) * (math.log1p(pc[x] + n) - math.log1p(pc[x])) for x, n in Counter(pattern).items())
            gain += sum((transition_reuse[x] / tmean) * (math.log1p(tc[x] + n) - math.log1p(tc[x])) for x, n in Counter(transitions(pattern)).items())
            if (gain, remaining) > winner_key:
                winner, winner_key = pattern, (gain, remaining)
        chosen[winner] += 1
        pc.update(Counter(winner)); tc.update(Counter(transitions(winner)))
    return {key: value for key, value in chosen.items() if value}


def sieve_select(
    trajectories: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    budget: int,
    seed: int = 0,
    persistence: int = 5,
    pca_dim: int = 256,
    min_avg_cluster_size: int = 20,
) -> np.ndarray:
    features, episode_rows = _sieve_segment_features(trajectories, actions, persistence, pca_dim)
    rng = np.random.default_rng(seed)
    sampled = np.sort(rng.choice(len(trajectories), max(2, round(0.1 * len(trajectories))), replace=False))
    subset_rows = np.concatenate([episode_rows[i] for i in sampled])
    max_k = max(1, min(
        len(features) - 1,
        len(subset_rows) - 1,
        len(features) // max(1, min_avg_cluster_size),
    ))
    candidates = sorted(set(np.linspace(1, max_k, min(20, max_k), dtype=int).tolist()))
    best = None
    for k in candidates:
        centers, _ = _kmeans(features[subset_rows], k, seed)
        labels = np.square(features[:, None] - centers[None]).sum(axis=2).argmin(axis=1)
        sets = [set(labels[episode_rows[i]].tolist()) for i in sampled]
        pairwise = []
        for i, left in enumerate(sets):
            pairwise.append(np.mean([len(left & right) / max(1, len(left | right)) for j, right in enumerate(sets) if i != j]))
        reuse = np.zeros(k)
        for value in sets:
            reuse[list(value)] += 1
        score = (1.0 - float(np.median(pairwise))) * math.log1p(float(np.median(reuse)))
        if best is None or (score, -k) > best[0]:
            best = ((score, -k), labels)
    labels = best[1]
    buckets = defaultdict(list)
    for episode_id, rows in enumerate(episode_rows):
        buckets[tuple(int(labels[row]) for row in rows)].append(episode_id)
    allocation = _sieve_allocate(dict(buckets), min(budget, len(trajectories)))
    selected = []
    for pattern, count in allocation.items():
        ids = buckets[pattern]
        vectors = np.stack([np.concatenate([features[row] for row in episode_rows[i]]) for i in ids])
        unit = _unit_rows(vectors)
        medoid = unit[int(np.argmax((unit @ unit.T).sum(axis=1)))]
        order = np.lexsort((np.asarray(ids), 1.0 - unit @ medoid))
        selected.extend(np.asarray(ids)[order[:count]].tolist())
    result = np.sort(np.asarray(selected, dtype=np.int64))
    if len(result) != budget or len(np.unique(result)) != budget:
        raise RuntimeError("SIEVE returned an invalid subset")
    return result


# ---------------------------------------------------------------------------
# TAROT
# ---------------------------------------------------------------------------
def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)) + maximum
    return np.squeeze(result, axis=axis)


def _sinkhorn_source_potentials(cost: np.ndarray, epsilon: float, iterations: int):
    n_source, n_target = cost.shape
    f, g = np.zeros(n_source), np.zeros(n_target)
    for _ in range(iterations):
        f = epsilon * (-math.log(n_source) - _logsumexp((g[None] - cost) / epsilon, axis=1))
        g = epsilon * (-math.log(n_target) - _logsumexp((f[:, None] - cost) / epsilon, axis=0))
    return f - f.mean()


def tarot_select(
    candidate_gradients: np.ndarray,
    target_gradients: np.ndarray,
    budget: int,
    covariance_jitter: float = 1e-5,
    sinkhorn_epsilon: float = 0.01,
    sinkhorn_iterations: int = 200,
) -> np.ndarray:
    all_features = np.concatenate([candidate_gradients, target_gradients]).astype(np.float64)
    all_features -= all_features.mean(axis=0, keepdims=True)
    covariance = all_features.T @ all_features / len(all_features)
    covariance.flat[:: covariance.shape[0] + 1] += covariance_jitter
    cholesky = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(cholesky, all_features.T).T
    whitened = _unit_rows(whitened)
    candidate = whitened[:len(candidate_gradients)]
    target = whitened[len(candidate_gradients):]
    similarity = candidate @ target.T
    sorted_ids = np.argsort(-similarity, axis=0, kind="stable")
    selected = set()
    for layer in range(len(candidate)):
        this_layer = np.unique(sorted_ids[layer])
        updated = selected.union(int(x) for x in this_layer)
        if len(updated) <= budget:
            selected = updated
            if len(selected) == budget:
                break
            continue
        remaining = budget - len(selected)
        boundary = np.setdiff1d(this_layer, np.fromiter(selected, dtype=np.int64))
        cosine = np.clip(candidate[boundary] @ target.T, -1.0, 1.0)
        cost = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * cosine))
        potential = _sinkhorn_source_potentials(cost, sinkhorn_epsilon, sinkhorn_iterations)
        order = np.lexsort((boundary, -potential))
        selected.update(int(x) for x in boundary[order[:remaining]])
        break
    result = np.sort(np.asarray(list(selected), dtype=np.int64))
    if len(result) != budget:
        raise RuntimeError("TAROT returned an invalid subset")
    return result
