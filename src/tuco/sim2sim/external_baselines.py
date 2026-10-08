"""Paper-aligned scalar comparison methods for State-MLP data."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import episode_bounds


class StepwiseMLPClassifier(nn.Module):
    """Demo-SCORE's 7-to-8-to-8-to-1 stepwise success classifier."""

    def __init__(self, input_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(values)).squeeze(-1)


def _episode_means(values: np.ndarray, ends: np.ndarray) -> np.ndarray:
    starts, stops = episode_bounds(ends)
    return np.asarray(
        [values[start:stop].mean(dtype=np.float64) for start, stop in zip(starts, stops)],
        dtype=np.float32,
    )


def _standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    std[std < 1e-8] = 1.0
    normalized = (values - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def deminf_ksg_scores(
    state: np.ndarray,
    action: np.ndarray,
    ends: np.ndarray,
    *,
    k: int = 5,
    seed: int = 0,
    workers: int = 1,
) -> np.ndarray:
    """Local KSG state-action dependence averaged within each trajectory."""
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    if not 1 <= k < len(state):
        raise ValueError(f"KSG k={k} is invalid for {len(state)} frames")
    state_n, _, _ = _standardize(np.asarray(state, dtype=np.float32))
    action_n, _, _ = _standardize(np.asarray(action, dtype=np.float32))
    rng = np.random.default_rng(seed)
    state_n += rng.normal(0.0, 1e-10, state_n.shape).astype(np.float32)
    action_n += rng.normal(0.0, 1e-10, action_n.shape).astype(np.float32)
    joint = np.concatenate([state_n, action_n], axis=1)
    radius = cKDTree(joint).query(
        joint, k=k + 1, p=np.inf, workers=workers
    )[0][:, k]
    radius = np.maximum(radius - 1e-12, 0.0)
    state_count = cKDTree(state_n).query_ball_point(
        state_n, radius, p=np.inf, return_length=True, workers=workers
    )
    action_count = cKDTree(action_n).query_ball_point(
        action_n, radius, p=np.inf, return_length=True, workers=workers
    )
    state_count = np.maximum(np.asarray(state_count, dtype=np.int64) - 1, 1)
    action_count = np.maximum(np.asarray(action_count, dtype=np.int64) - 1, 1)
    local = (
        digamma(k)
        + digamma(len(state_n))
        - digamma(state_count + 1)
        - digamma(action_count + 1)
    )
    return _episode_means(local.astype(np.float32), ends)


def demoscore_scores(
    rollout_action: np.ndarray,
    rollout_ends: np.ndarray,
    rollout_success: np.ndarray,
    candidate_action: np.ndarray,
    candidate_ends: np.ndarray,
    *,
    seed: int = 0,
    epochs: int = 200,
    batch_size: int = 2048,
    device: str = "cuda:0",
) -> np.ndarray:
    """Train Demo-SCORE and score candidates by mean success probability."""
    success = np.asarray(rollout_success, dtype=bool)
    if success.shape != (len(rollout_ends),):
        raise ValueError("one success label is required per rollout")
    if success.sum() == 0 or success.sum() == len(success):
        raise ValueError("Demo-SCORE needs successful and failed rollouts")
    action_n, mean, std = _standardize(np.asarray(rollout_action, dtype=np.float32))
    candidate_n = ((candidate_action - mean) / std).astype(np.float32)
    starts, stops = episode_bounds(rollout_ends)
    labels = np.empty(len(action_n), dtype=np.float32)
    for start, stop, value in zip(starts, stops, success):
        labels[start:stop] = float(value)

    rng = np.random.default_rng(seed)
    train_episodes: list[int] = []
    validation_episodes: list[int] = []
    for label in (False, True):
        indices = rng.permutation(np.flatnonzero(success == label))
        validation_count = max(1, int(round(0.2 * len(indices))))
        validation_episodes.extend(indices[:validation_count].tolist())
        train_episodes.extend(indices[validation_count:].tolist())
    train_mask = np.zeros(len(action_n), dtype=bool)
    validation_mask = np.zeros(len(action_n), dtype=bool)
    for index in train_episodes:
        train_mask[starts[index]:stops[index]] = True
    for index in validation_episodes:
        validation_mask[starts[index]:stops[index]] = True
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Demo-SCORE needs at least two rollouts per class")

    torch.manual_seed(seed)
    target_device = torch.device(device)
    model = StepwiseMLPClassifier(action_n.shape[1]).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(action_n[train_mask]),
            torch.from_numpy(labels[train_mask]),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_x = torch.from_numpy(action_n[validation_mask]).to(target_device)
    validation_y = torch.from_numpy(labels[validation_mask]).to(target_device)
    best_loss = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        for features, target in loader:
            features, target = features.to(target_device), target.to(target_device)
            prediction = model(features)
            weight = target + 1.0 / 3.0
            loss = nn.functional.binary_cross_entropy(
                prediction, target, weight=weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            loss = nn.functional.binary_cross_entropy(
                model(validation_x), validation_y
            ).item()
        if loss < best_loss:
            best_loss = loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Demo-SCORE did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(candidate_n), batch_size):
            batch = torch.from_numpy(candidate_n[start:start + batch_size])
            predictions.append(model(batch.to(target_device)).cpu().numpy())
    return _episode_means(np.concatenate(predictions), candidate_ends)


def success_similarity_scores(
    target_state: np.ndarray,
    target_ends: np.ndarray,
    target_success: np.ndarray,
    candidate_state: np.ndarray,
    candidate_ends: np.ndarray,
) -> np.ndarray:
    """CUPID Success Similarity using exact mean pairwise state distance."""
    success = np.asarray(target_success, dtype=bool)
    if success.shape != (len(target_ends),) or not success.any():
        raise ValueError("successful target rollouts are required")
    target = np.asarray(target_state, dtype=np.float32)
    candidate = np.asarray(candidate_state, dtype=np.float32)
    if target.shape[1:] != candidate.shape[1:]:
        raise ValueError("target and candidate state dimensions differ")
    lower = np.minimum(target.min(axis=0), candidate.min(axis=0))
    upper = np.maximum(target.max(axis=0), candidate.max(axis=0))
    scale = np.maximum(upper - lower, 1e-12)
    target = (target - lower) / scale
    candidate = (candidate - lower) / scale
    target_starts, target_stops = episode_bounds(target_ends)
    candidate_starts, candidate_stops = episode_bounds(candidate_ends)
    successful = [
        target[start:stop]
        for start, stop, keep in zip(target_starts, target_stops, success)
        if keep
    ]
    scores = np.empty(len(candidate_ends), dtype=np.float32)
    for index, (start, stop) in enumerate(zip(candidate_starts, candidate_stops)):
        trajectory = candidate[start:stop]
        values = []
        for rollout in successful:
            squared = (
                np.square(rollout).sum(axis=1)[:, None]
                + np.square(trajectory).sum(axis=1)[None, :]
                - 2.0 * rollout @ trajectory.T
            )
            values.append(-float(np.sqrt(np.maximum(squared, 0.0)).mean()))
        scores[index] = float(np.mean(values))
    return scores
