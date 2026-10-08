"""Zarr data access, normalization, episode bookkeeping, and windowed sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import zarr


# --------------------------------------------------------------------------- #
# Raw zarr access + episode bookkeeping
# --------------------------------------------------------------------------- #
def load_arrays(path: str, keys: Sequence[str]) -> Tuple[List[np.ndarray], np.ndarray]:
    """Load data/<key> arrays and meta/episode_ends from a zarr store."""
    z = zarr.open(path, mode="r")
    arrays = [np.asarray(z[f"data/{k}"], dtype=np.float32) for k in keys]
    ends = np.asarray(z["meta/episode_ends"], dtype=np.int64)
    if ends.ndim != 1 or len(ends) == 0:
        raise ValueError(f"{path}: episode_ends must be a non-empty vector")
    if np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise ValueError(f"{path}: episode_ends must be strictly increasing")
    if any(len(array) != int(ends[-1]) for array in arrays):
        lengths = [len(array) for array in arrays]
        raise ValueError(
            f"{path}: data lengths {lengths} do not end at {int(ends[-1])}"
        )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"{path}: state/action data contain non-finite values")
    return arrays, ends


def episode_bounds(ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (starts, ends) per episode."""
    starts = np.concatenate([[0], ends[:-1]]).astype(np.int64)
    return starts, ends.astype(np.int64)


def episode_frame_indices(ends: np.ndarray) -> List[np.ndarray]:
    """Return a list of per-episode frame-index arrays (global indices)."""
    starts, ends = episode_bounds(ends)
    return [np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)]


# --------------------------------------------------------------------------- #
# Normalization (matches train_mlp_bc)
# --------------------------------------------------------------------------- #
@dataclass
class NormStats:
    s_mean: np.ndarray
    s_std: np.ndarray
    a_center: np.ndarray
    a_scale: np.ndarray

    @classmethod
    def fit(cls, state: np.ndarray, action: np.ndarray) -> "NormStats":
        # Match the reference trainer's reduction precision and Bessel correction.
        state_tensor = torch.as_tensor(state, dtype=torch.float32)
        s_mean = state_tensor.mean(dim=0).numpy()
        s_std = (state_tensor.std(dim=0) + 1e-6).numpy()
        a_max, a_min = action.max(axis=0), action.min(axis=0)
        a_center = (a_max + a_min) / 2.0
        a_scale = (a_max - a_min) / 2.0 + 1e-6
        return cls(s_mean.astype(np.float32), s_std.astype(np.float32),
                   a_center.astype(np.float32), a_scale.astype(np.float32))

    def norm_state(self, s: np.ndarray) -> np.ndarray:
        return (s - self.s_mean) / self.s_std

    def norm_action(self, a: np.ndarray) -> np.ndarray:
        return (a - self.a_center) / self.a_scale

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {"s_mean": self.s_mean, "s_std": self.s_std,
                "a_center": self.a_center, "a_scale": self.a_scale}

    @classmethod
    def from_dict(cls, d: Dict) -> "NormStats":
        def get(key: str) -> np.ndarray:
            value = d[key]
            return (
                value.cpu().numpy()
                if torch.is_tensor(value)
                else np.asarray(value, np.float32)
            )

        return cls(get("s_mean"), get("s_std"), get("a_center"), get("a_scale"))


# --------------------------------------------------------------------------- #
# Windowed (n_obs) sample dataset
# --------------------------------------------------------------------------- #
class WindowDataset(torch.utils.data.Dataset):
    """Emit (obs_stack[n_obs, obs_dim], action[act_dim]) normalized samples.

    Given a concatenation of (already normalized) state/action and episode_ends,
    each sample stacks the last `n_obs` frames within the episode (left-padded by
    clamping to the episode start), identical to train_mlp_bc.
    """

    def __init__(self, state_n: np.ndarray, action_n: np.ndarray, ends: np.ndarray,
                 n_obs: int = 2, target_idxs: Optional[np.ndarray] = None,
                 prev_state_n: Optional[np.ndarray] = None):
        self.state_n = np.ascontiguousarray(state_n, dtype=np.float32)
        self.action_n = np.ascontiguousarray(action_n, dtype=np.float32)
        self.prev_state_n = (
            None if prev_state_n is None
            else np.ascontiguousarray(prev_state_n, dtype=np.float32)
        )
        if self.prev_state_n is not None and self.prev_state_n.shape != self.state_n.shape:
            raise ValueError(
                "prev_state_n must have the same shape as state_n: "
                f"{self.prev_state_n.shape} != {self.state_n.shape}"
            )
        self.n_obs = n_obs
        starts, ends = episode_bounds(ends)
        self.ep_start = np.zeros(len(state_n), np.int64)
        for s, e in zip(starts, ends):
            self.ep_start[s:e] = s
        self.targets = (np.arange(len(state_n), dtype=np.int64)
                        if target_idxs is None else np.asarray(target_idxs, np.int64))

    def __len__(self) -> int:
        return len(self.targets)

    def obs_indices(self, g: int) -> List[int]:
        st = int(self.ep_start[g])
        return [max(st, g - j) for j in range(self.n_obs - 1, -1, -1)]

    def __getitem__(self, k: int):
        g = int(self.targets[k])
        if self.prev_state_n is not None and self.n_obs == 2:
            start = int(self.ep_start[g])
            previous = self.prev_state_n[g] if g == start else self.state_n[g - 1]
            obs = np.stack([previous, self.state_n[g]], axis=0)
        else:
            obs = self.state_n[self.obs_indices(g)]
        return torch.from_numpy(obs), torch.from_numpy(self.action_n[g])


def read_success_mask(path: str, ends: np.ndarray) -> Optional[np.ndarray]:
    """Return a per-episode success bool mask if data/success exists, else None."""
    z = zarr.open(path, mode="r")
    if "data/success" not in z:
        return None
    raw = np.asarray(z["data/success"])
    if len(raw) == len(ends):                       # already per-episode
        return raw.astype(bool)
    if len(raw) != int(ends[-1]):
        raise ValueError(
            f"{path}: success labels have length {len(raw)}, expected "
            f"{len(ends)} episodes or {int(ends[-1])} frames"
        )
    starts, ends_ = episode_bounds(ends)            # per-frame -> per-episode (any)
    return np.array([bool(raw[s:e].max()) for s, e in zip(starts, ends_)])
