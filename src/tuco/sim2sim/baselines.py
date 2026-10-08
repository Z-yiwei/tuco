"""Trajectory-level implementations of four additional curation baselines."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .data import NormStats, episode_bounds
from .model import MLPBCPolicy


# The 200-D OmniReset observation is term-major with five history entries.
# Retaining the newest entry of each term removes duplicated temporal history
# before a trajectory signature is formed.  See scripts/sim2sim/franka/
# obs_reconstruct.py.
OMNIRESET_CURRENT_STATE_INDICES = np.asarray(
    list(range(24, 30))       # pose A, newest
    + list(range(58, 65))     # previous action, newest
    + list(range(101, 110))   # joint position, newest
    + list(range(134, 140))   # pose B / end-effector, newest
    + list(range(164, 170))   # pose C, newest
    + list(range(194, 200)),  # pose D, newest
    dtype=np.int64,
)
OMNIRESET_EE_POSITION_INDICES = np.asarray([134, 135, 136], dtype=np.int64)


def _sample_episode_frames(ends: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if stride < 1:
        raise ValueError("frame stride must be positive")
    starts, ends = episode_bounds(ends)
    frames: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    for episode_id, (start, end) in enumerate(zip(starts, ends)):
        idx = np.arange(start, end, stride, dtype=np.int64)
        if idx.size == 0 or idx[-1] != end - 1:
            idx = np.append(idx, end - 1)
        frames.append(idx)
        episode_ids.append(np.full(len(idx), episode_id, dtype=np.int64))
    return np.concatenate(frames), np.concatenate(episode_ids)


def _episode_means(values: np.ndarray, episode_ids: np.ndarray, n_episodes: int) -> np.ndarray:
    sums = np.bincount(episode_ids, weights=np.asarray(values, np.float64), minlength=n_episodes)
    counts = np.bincount(episode_ids, minlength=n_episodes)
    if np.any(counts == 0):
        raise ValueError("an episode has no scored frames")
    return (sums / counts).astype(np.float32)


def _frame_episode_starts(ends: np.ndarray) -> np.ndarray:
    starts, ends = episode_bounds(ends)
    frame_starts = np.empty(int(ends[-1]), dtype=np.int64)
    for start, end in zip(starts, ends):
        frame_starts[start:end] = start
    return frame_starts


def observation_stacks(
    state_n: np.ndarray,
    ends: np.ndarray,
    frame_indices: np.ndarray,
    n_obs_steps: int,
) -> np.ndarray:
    """Build the exact left-clamped observation history used by MLP-BC."""
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    starts = _frame_episode_starts(ends)
    offsets = np.arange(n_obs_steps - 1, -1, -1, dtype=np.int64)
    indices = frame_indices[:, None] - offsets[None]
    indices = np.maximum(indices, starts[frame_indices, None])
    return np.ascontiguousarray(state_n[indices], dtype=np.float32)


# ---------------------------------------------------------------------------
# PSD: Sojib & Begum, arXiv:2605.01544
# ---------------------------------------------------------------------------
def psd_scores(
    state: np.ndarray,
    ends: np.ndarray,
    ee_position_indices: Sequence[int] = OMNIRESET_EE_POSITION_INDICES,
) -> np.ndarray:
    """Return negative total EE-position spectral power (higher is better).

    The paper ranks demonstrations by ascending W=sum_f,sum_xyz |FFT(x)|^2.
    We negate W so every fixed-score artifact in this repository can use the
    same descending Top-k selector.
    """
    indices = np.asarray(ee_position_indices, dtype=np.int64)
    if indices.shape != (3,):
        raise ValueError("PSD requires exactly three end-effector position indices")
    if np.any(indices < 0) or np.any(indices >= state.shape[1]):
        raise ValueError(f"invalid EE indices {indices.tolist()} for state dim {state.shape[1]}")
    starts, ends = episode_bounds(ends)
    total_power = np.empty(len(ends), dtype=np.float64)
    for i, (start, end) in enumerate(zip(starts, ends)):
        xyz = np.asarray(state[start:end, indices], dtype=np.float64)
        spectrum = np.fft.fft(xyz, axis=0)
        total_power[i] = np.square(np.abs(spectrum)).sum(dtype=np.float64)
    return (-total_power).astype(np.float32)


# ---------------------------------------------------------------------------
# FAKTUAL: Sirigiri et al., arXiv:2603.11634
# ---------------------------------------------------------------------------
def _level2_signature(path: np.ndarray) -> np.ndarray:
    """Exact level-1/2 signature of a piecewise-linear path.

    Chen's identity gives the streaming recurrence below.  It avoids an extra
    dependency and is exact for the truncated linear signature used here.
    """
    increments = np.diff(np.asarray(path, dtype=np.float64), axis=0)
    width = path.shape[1]
    first = np.zeros(width, dtype=np.float64)
    second = np.zeros((width, width), dtype=np.float64)
    for delta in increments:
        second += np.outer(first, delta) + 0.5 * np.outer(delta, delta)
        first += delta
    return np.concatenate([first, second.reshape(-1)])


def faktual_signature_features(
    state: np.ndarray,
    action: np.ndarray,
    ends: np.ndarray,
    *,
    state_indices: Sequence[int] = OMNIRESET_CURRENT_STATE_INDICES,
    frame_stride: int = 1,
    append_time: bool = True,
    prepend_basepoint: bool = True,
) -> np.ndarray:
    """Create normalized level-2 signature features from state-action paths."""
    state_indices = np.asarray(state_indices, dtype=np.int64)
    if np.any(state_indices < 0) or np.any(state_indices >= state.shape[1]):
        raise ValueError("FAKTUAL state indices are outside the observation")
    path_all = np.concatenate([state[:, state_indices], action], axis=1).astype(np.float64)
    mean = path_all.mean(axis=0)
    std = path_all.std(axis=0)
    std[std < 1e-8] = 1.0
    path_all = (path_all - mean) / std
    starts, ends = episode_bounds(ends)
    features: list[np.ndarray] = []
    for start, end in zip(starts, ends):
        frame_ids = np.arange(start, end, frame_stride, dtype=np.int64)
        if frame_ids[-1] != end - 1:
            frame_ids = np.append(frame_ids, end - 1)
        path = path_all[frame_ids]
        if append_time:
            time = np.linspace(0.0, 1.0, len(path), dtype=np.float64)[:, None]
            path = np.concatenate([time, path], axis=1)
        if prepend_basepoint:
            path = np.concatenate([np.zeros((1, path.shape[1])), path], axis=0)
        features.append(_level2_signature(path))
    result = np.stack(features)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return (result / norms).astype(np.float32)


def normalized_gram(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    gram = (features @ features.T).astype(np.float64)
    diag = np.sqrt(np.maximum(np.diag(gram), 1e-12))
    gram /= diag[:, None] * diag[None, :]
    np.clip(gram, -1.0, 1.0, out=gram)
    np.fill_diagonal(gram, 1.0)
    return gram


def signature_entropy(gram_subset: np.ndarray) -> float:
    n = len(gram_subset)
    if n <= 1:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(gram_subset / float(n))
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(-(eigenvalues * np.log(eigenvalues)).sum())


def stochastic_entropy_greedy(
    gram: np.ndarray,
    budget: int,
    *,
    seed: int,
    epsilon: float = 0.1,
) -> np.ndarray:
    """FAKTUAL Appendix-E stochastic greedy on exact signature entropy."""
    n = len(gram)
    budget = min(max(int(budget), 0), n)
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be in (0, 1)")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    remaining = np.ones(n, dtype=bool)
    sample_size = int(math.ceil((n / budget) * math.log(1.0 / epsilon)))
    for _ in range(budget):
        available = np.flatnonzero(remaining)
        candidates = (
            available
            if len(available) <= sample_size
            else rng.choice(available, size=sample_size, replace=False)
        )
        best_id, best_value = None, -np.inf
        for candidate in np.sort(candidates):
            ids = np.asarray(selected + [int(candidate)], dtype=np.int64)
            value = signature_entropy(gram[np.ix_(ids, ids)])
            if value > best_value + 1e-14:
                best_id, best_value = int(candidate), value
        assert best_id is not None
        selected.append(best_id)
        remaining[best_id] = False
    return np.asarray(selected, dtype=np.int64)


def determinant_greedy(
    gram: np.ndarray,
    candidates: np.ndarray,
    budget: int,
    *,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Greedy log-determinant selection via pivoted Cholesky updates."""
    candidates = np.asarray(candidates, dtype=np.int64)
    budget = min(max(int(budget), 0), len(candidates))
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    local = gram[np.ix_(candidates, candidates)].astype(np.float64, copy=True)
    residual = np.diag(local).copy() + jitter
    factors = np.zeros((len(candidates), budget), dtype=np.float64)
    active = np.ones(len(candidates), dtype=bool)
    picked_local: list[int] = []
    for column in range(budget):
        masked = np.where(active, residual, -np.inf)
        pivot = int(np.argmax(masked))
        if not np.isfinite(masked[pivot]):
            raise FloatingPointError(
                "regularized determinant greedy has no finite active pivot"
            )
        picked_local.append(pivot)
        active[pivot] = False
        pivot_value = max(residual[pivot], jitter)
        if column == 0:
            update = local[:, pivot] / math.sqrt(pivot_value)
        else:
            correction = factors[:, :column] @ factors[pivot, :column]
            update = (local[:, pivot] - correction) / math.sqrt(pivot_value)
        if not np.all(np.isfinite(update)):
            raise FloatingPointError(
                "regularized determinant greedy produced a non-finite update; "
                f"increase log-det regularization above {jitter:g}"
            )
        factors[:, column] = update
        residual = np.maximum(residual - update * update, 0.0)
    return candidates[np.asarray(picked_local, dtype=np.int64)]


def faktual_select(
    gram: np.ndarray,
    budget: int,
    *,
    entropy_fraction: float = 0.5,
    seed: int = 0,
    epsilon: float = 0.1,
    logdet_regularization: float = 1e-8,
) -> np.ndarray:
    """Paper's entropy-prefix + determinant-from-remainder FAKTUAL subset."""
    if not 0.0 <= entropy_fraction <= 1.0:
        raise ValueError("entropy_fraction must be in [0, 1]")
    budget = min(max(int(budget), 0), len(gram))
    entropy_budget = int(round(budget * entropy_fraction))
    entropy_ids = stochastic_entropy_greedy(
        gram, entropy_budget, seed=seed, epsilon=epsilon
    )
    remaining = np.setdiff1d(
        np.arange(len(gram), dtype=np.int64), entropy_ids, assume_unique=True
    )
    determinant_ids = determinant_greedy(
        gram,
        remaining,
        budget - len(entropy_ids),
        jitter=logdet_regularization,
    )
    selected = np.concatenate([entropy_ids, determinant_ids])
    if len(np.unique(selected)) != budget:
        raise RuntimeError("FAKTUAL returned duplicate or missing trajectory IDs")
    return np.sort(selected)


# ---------------------------------------------------------------------------
# QoQ: Lee et al., arXiv:2603.09056
# ---------------------------------------------------------------------------
@dataclass
class OPORPProjector:
    permutation: np.ndarray
    signs: np.ndarray
    output_dim: int

    @classmethod
    def create(cls, input_dim: int, output_dim: int, seed: int) -> "OPORPProjector":
        output_dim = min(max(1, int(output_dim)), input_dim)
        rng = np.random.default_rng(seed)
        return cls(
            permutation=rng.permutation(input_dim).astype(np.int64),
            signs=rng.choice(np.asarray([-1.0, 1.0], np.float32), size=input_dim),
            output_dim=output_dim,
        )

    def project(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        permuted = values[:, self.permutation] * self.signs[None]
        # One permutation followed by contiguous random-projection bins.
        boundaries = np.linspace(0, permuted.shape[1], self.output_dim + 1, dtype=np.int64)
        return np.add.reduceat(permuted, boundaries[:-1], axis=1).astype(
            np.float32, copy=False
        )


@torch.inference_mode()
def _head_gradient_features(
    model: MLPBCPolicy,
    state_n: np.ndarray,
    action_n: np.ndarray,
    ends: np.ndarray,
    frame_indices: np.ndarray,
    *,
    device: str,
    batch_size: int,
    projector: OPORPProjector,
) -> np.ndarray:
    """Normalized Gaussian-log-likelihood gradients for the action head."""
    all_features: list[np.ndarray] = []
    for start in range(0, len(frame_indices), batch_size):
        idx = frame_indices[start:start + batch_size]
        obs = observation_stacks(state_n, ends, idx, model.n_obs_steps)
        obs_t = torch.from_numpy(obs).to(device)
        target_t = torch.from_numpy(action_n[idx]).to(device)
        hidden = model.features(obs_t)
        prediction = model.head(hidden)
        residual = target_t - prediction
        weight_grad = residual[:, :, None] * hidden[:, None, :]
        gradient = torch.cat(
            [weight_grad.flatten(1), residual], dim=1
        ).cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(gradient, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        projected = projector.project(gradient / norm)
        projected_norm = np.linalg.norm(projected, axis=1, keepdims=True)
        projected_norm[projected_norm < 1e-12] = 1.0
        all_features.append(projected / projected_norm)
    return np.concatenate(all_features)


def qoq_scores(
    model: MLPBCPolicy,
    norm: NormStats,
    pool_state: np.ndarray,
    pool_action: np.ndarray,
    pool_ends: np.ndarray,
    target_state: np.ndarray,
    target_action: np.ndarray,
    target_ends: np.ndarray,
    *,
    device: str,
    seed: int = 0,
    projection_dim: int = 1024,
    source_frame_stride: int = 1,
    validation_frame_stride: int = 1,
    max_validation_frames: int = 4096,
    batch_size: int = 2048,
) -> np.ndarray:
    """Maximum validation-gradient similarity, averaged per trajectory."""
    pool_frames, pool_episode_ids = _sample_episode_frames(pool_ends, source_frame_stride)
    target_frames, _ = _sample_episode_frames(target_ends, validation_frame_stride)
    if len(target_frames) > max_validation_frames:
        rng = np.random.default_rng(seed)
        target_frames = np.sort(
            rng.choice(target_frames, size=max_validation_frames, replace=False)
        )
    pool_state_n = norm.norm_state(pool_state).astype(np.float32)
    pool_action_n = norm.norm_action(pool_action).astype(np.float32)
    target_state_n = norm.norm_state(target_state).astype(np.float32)
    target_action_n = norm.norm_action(target_action).astype(np.float32)
    input_dim = model.act_dim * model.hidden + model.act_dim
    projector = OPORPProjector.create(input_dim, projection_dim, seed)
    model = model.to(device).eval()
    target_features = _head_gradient_features(
        model, target_state_n, target_action_n, target_ends, target_frames,
        device=device, batch_size=batch_size, projector=projector,
    )
    frame_scores = np.empty(len(pool_frames), dtype=np.float32)
    source_batch = max(1, min(batch_size, 4096))
    for start in range(0, len(pool_frames), source_batch):
        idx = pool_frames[start:start + source_batch]
        source_features = _head_gradient_features(
            model, pool_state_n, pool_action_n, pool_ends, idx,
            device=device, batch_size=batch_size, projector=projector,
        )
        maxima = np.full(len(idx), -np.inf, dtype=np.float32)
        for val_start in range(0, len(target_features), batch_size):
            values = source_features @ target_features[val_start:val_start + batch_size].T
            maxima = np.maximum(maxima, values.max(axis=1))
        frame_scores[start:start + len(idx)] = maxima
    return _episode_means(frame_scores, pool_episode_ids, len(pool_ends))


# ---------------------------------------------------------------------------
# DataMIL: Dass et al., arXiv:2505.09603 (metagradient estimator)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _trunk_features(
    model: MLPBCPolicy,
    state_n: np.ndarray,
    ends: np.ndarray,
    frames: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    model = model.to(device).eval()
    for start in range(0, len(frames), batch_size):
        idx = frames[start:start + batch_size]
        obs = observation_stacks(state_n, ends, idx, model.n_obs_steps)
        pieces.append(model.features(torch.from_numpy(obs).to(device)).cpu())
    return torch.cat(pieces, dim=0)


def datamil_metagradient_scores(
    model: MLPBCPolicy,
    norm: NormStats,
    pool_state: np.ndarray,
    pool_action: np.ndarray,
    pool_ends: np.ndarray,
    target_state: np.ndarray,
    target_action: np.ndarray,
    target_ends: np.ndarray,
    *,
    device: str,
    seed: int = 0,
    frame_stride: int = 4,
    validation_demos: int = 100,
    inner_steps: int = 200,
    inner_batch_size: int = 256,
    inner_lr: float = 1e-3,
    feature_batch_size: int = 2048,
) -> np.ndarray:
    """Trajectory-weight metagradients against target-demonstration BC loss.

    This is a memory-bounded port of DataMIL's metagradient estimator.  The
    MLP trunk is frozen and the action head is differentiably unrolled; the
    final coefficients are -d(target validation loss)/d(source trajectory
    weight), so larger scores are preferred.
    """
    if len(target_ends) < 2:
        raise ValueError("DataMIL requires at least two target trajectories")
    validation_demos = min(max(1, validation_demos), len(target_ends) - 1)
    split = len(target_ends) - validation_demos
    target_frames, target_episode_ids = _sample_episode_frames(target_ends, frame_stride)
    target_train_mask = target_episode_ids < split
    target_val_mask = ~target_train_mask
    pool_frames, pool_episode_ids = _sample_episode_frames(pool_ends, frame_stride)

    pool_state_n = norm.norm_state(pool_state).astype(np.float32)
    pool_action_n = norm.norm_action(pool_action).astype(np.float32)
    target_state_n = norm.norm_state(target_state).astype(np.float32)
    target_action_n = norm.norm_action(target_action).astype(np.float32)
    pool_features = _trunk_features(
        model, pool_state_n, pool_ends, pool_frames,
        device=device, batch_size=feature_batch_size,
    ).to(device)
    target_features = _trunk_features(
        model, target_state_n, target_ends, target_frames,
        device=device, batch_size=feature_batch_size,
    ).to(device)
    pool_actions = torch.from_numpy(pool_action_n[pool_frames]).to(device)
    target_actions = torch.from_numpy(target_action_n[target_frames]).to(device)
    pool_episode_ids_t = torch.from_numpy(pool_episode_ids).to(device)
    train_ids = torch.from_numpy(np.flatnonzero(target_train_mask)).to(device)
    val_ids = torch.from_numpy(np.flatnonzero(target_val_mask)).to(device)

    source_weights = torch.ones(len(pool_ends), device=device, requires_grad=True)
    weight = model.head.weight.detach().to(device).clone().requires_grad_(True)
    bias = model.head.bias.detach().to(device).clone().requires_grad_(True)
    generator = torch.Generator(device=device).manual_seed(seed)
    half = max(1, inner_batch_size // 2)
    for _ in range(inner_steps):
        a_pick = train_ids[torch.randint(len(train_ids), (half,), generator=generator, device=device)]
        b_pick = torch.randint(len(pool_frames), (inner_batch_size - half,), generator=generator, device=device)
        a_prediction = target_features[a_pick] @ weight.T + bias
        b_prediction = pool_features[b_pick] @ weight.T + bias
        a_loss = torch.square(a_prediction - target_actions[a_pick]).mean(dim=1).mean()
        b_loss_each = torch.square(b_prediction - pool_actions[b_pick]).mean(dim=1)
        b_loss = (b_loss_each * source_weights[pool_episode_ids_t[b_pick]]).mean()
        inner_loss = 0.5 * (a_loss + b_loss)
        grad_weight, grad_bias = torch.autograd.grad(
            inner_loss, (weight, bias), create_graph=True
        )
        weight = weight - inner_lr * grad_weight
        bias = bias - inner_lr * grad_bias

    val_prediction = target_features[val_ids] @ weight.T + bias
    val_loss = torch.square(val_prediction - target_actions[val_ids]).mean()
    (weight_gradient,) = torch.autograd.grad(val_loss, source_weights)
    return (-weight_gradient.detach().cpu().numpy()).astype(np.float32)
