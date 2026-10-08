"""TRAK featurize/finalize/score pipeline for deterministic MLP-BC.

The policy-specific part of TRAK is the scalar model output.  For MLP-BC we
use the per-example mean squared training loss and differentiate it with
respect to every policy parameter. Everything after that follows the source
CUPID experiment:

1. featurize: per-example gradient followed by a shared Rademacher JL sketch;
2. finalize: ``G (G.T G + lambda I)^-1`` over candidate-pool gradients;
3. score: target projected gradients against finalized pool features.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import InfluenceConfig
from .data import episode_bounds
from .model import MLPBCPolicy


def build_windows(state_n: np.ndarray, ends: np.ndarray, idxs: np.ndarray, n_obs: int
                  ) -> np.ndarray:
    """Stack the last ``n_obs`` frames, clamped to each episode start."""
    starts, ends_ = episode_bounds(ends)
    ep_start = np.zeros(len(state_n), np.int64)
    for start, end in zip(starts, ends_):
        ep_start[start:end] = start
    idxs = np.asarray(idxs, np.int64)
    windows = np.empty((len(idxs), n_obs, state_n.shape[1]), dtype=np.float32)
    for row, frame_idx in enumerate(idxs):
        start = int(ep_start[frame_idx])
        windows[row] = state_n[
            [max(start, frame_idx - offset) for offset in range(n_obs - 1, -1, -1)]
        ]
    return windows


class _Projector:
    """Fixed dense JL projector shared by pool and target featurization.

    This intentionally matches the State-MLP experiments: one unscaled
    Rademacher matrix is sampled once, then every projected gradient is divided
    by ``sqrt(num_parameters)``.  The paper projection dimension is not a
    multiple of 512, so the optional TRAK CUDA kernel cannot represent this
    protocol exactly.
    """

    def __init__(self, grad_dim: int, proj_dim: int, seed: int, device: str):
        if proj_dim <= 0:
            raise ValueError("proj_dim must be positive when constructing a projector")
        self.grad_dim = int(grad_dim)
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)
        self.device = device
        self.normalize_factor = math.sqrt(float(grad_dim))
        generator = torch.Generator(device=device).manual_seed(self.seed)
        self.matrix = torch.empty(
            self.grad_dim, self.proj_dim, dtype=torch.float32, device=device
        )
        self.matrix.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)

    def __call__(self, gradients: torch.Tensor) -> torch.Tensor:
        return (gradients @ self.matrix) / self.normalize_factor


def _model_output(
    model: MLPBCPolicy,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    obs: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    prediction = torch.func.functional_call(
        model, (params, buffers), (obs.unsqueeze(0),)
    ).squeeze(0)
    return F.mse_loss(prediction, action, reduction="mean")


def _full_model_grad_features(
    model: MLPBCPolicy,
    windows: np.ndarray,
    actions: np.ndarray,
    cfg: InfluenceConfig,
    device: str,
    projector: Optional[_Projector],
) -> tuple[np.ndarray, Optional[_Projector]]:
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    grad_dim = sum(parameter.numel() for parameter in params.values())
    if projector is None and cfg.proj_dim > 0:
        matrix_gib = grad_dim * cfg.proj_dim * 4 / (1024 ** 3)
        print(
            "[TRAK featurize] shared dense Rademacher projector "
            f"{grad_dim}x{cfg.proj_dim} (dense equivalent {matrix_gib:.1f} GiB)",
            flush=True,
        )
        projector = _Projector(
            grad_dim, cfg.proj_dim, cfg.proj_seed, device
        )

    grad_fn = torch.func.grad(_model_output, argnums=1)
    features = []
    starts = range(0, len(windows), cfg.grad_batch)
    report_every = max(1, math.ceil(len(starts) / 10))
    print(
        f"[TRAK featurize] scope=full samples={len(windows)} "
        f"batches={len(starts)} batch_size={cfg.grad_batch}",
        flush=True,
    )
    for batch_index, start in enumerate(starts):
        obs = torch.from_numpy(windows[start:start + cfg.grad_batch]).to(device)
        action = torch.from_numpy(actions[start:start + cfg.grad_batch]).to(device)
        per_parameter = torch.func.vmap(
            grad_fn,
            in_dims=(None, None, None, 0, 0),
            randomness="different",
        )(model, params, buffers, obs, action)
        gradients = torch.cat(
            [gradient.reshape(obs.shape[0], -1) for gradient in per_parameter.values()],
            dim=1,
        )
        if projector is not None:
            gradients = projector(gradients)
        else:
            gradients = gradients / math.sqrt(float(grad_dim))
        features.append(gradients.detach().cpu().numpy().astype(np.float32))
        if (batch_index + 1) % report_every == 0 or batch_index + 1 == len(starts):
            print(
                f"[TRAK featurize] {batch_index + 1}/{len(starts)} batches",
                flush=True,
            )
    return np.concatenate(features, axis=0), projector


def gradient_features(
    model: MLPBCPolicy,
    state_n: np.ndarray,
    action_n: np.ndarray,
    ends: np.ndarray,
    idxs: np.ndarray,
    cfg: InfluenceConfig,
    device: str = "cpu",
    projector: Optional[_Projector] = None,
) -> tuple[np.ndarray, Optional[_Projector]]:
    """TRAK ``featurize`` equivalent for an MLP frame set."""
    model.eval().to(device)
    idxs = np.asarray(idxs, np.int64)
    windows = build_windows(state_n, ends, idxs, model.n_obs_steps)
    actions = np.ascontiguousarray(action_n[idxs], dtype=np.float32)
    return _full_model_grad_features(
        model, windows, actions, cfg, device, projector
    )


def finalize_features(
    train_gradients: np.ndarray,
    cfg: InfluenceConfig,
    device: str = "cpu",
) -> np.ndarray:
    """TRAK ``finalize_features``: return ``G(G.T G + lambda I)^-1``."""
    gradients = torch.as_tensor(train_gradients, dtype=torch.float32, device=device)
    xtx = gradients.T @ gradients
    if cfg.lambda_reg:
        xtx = xtx + cfg.lambda_reg * torch.eye(
            xtx.shape[0], dtype=xtx.dtype, device=xtx.device
        )
    try:
        xtx_inv = torch.linalg.inv(xtx.float())
    except RuntimeError as error:
        raise ValueError(
            "projected source-gradient Gram matrix is singular under the "
            "paper-facing ridge eta=0"
        ) from error
    if not torch.isfinite(xtx_inv).all():
        raise ValueError(
            "TRAK feature finalization produced a non-finite inverse under "
            "the paper-facing ridge eta=0"
        )
    # Matches BasicScoreComputer in the source TRAK checkout.
    inverse_scale = xtx_inv.abs().mean()
    if not torch.isfinite(inverse_scale) or inverse_scale <= 0:
        raise ValueError("TRAK inverse normalization is degenerate")
    xtx_inv = xtx_inv / inverse_scale
    finalized = gradients @ xtx_inv
    return finalized.detach().cpu().numpy().astype(np.float32)


def influence_matrix(target_gradients: np.ndarray,
                     train_features: np.ndarray) -> np.ndarray:
    """TRAK ``score`` for one checkpoint with the source experiment's Q=1."""
    return target_gradients @ train_features.T
