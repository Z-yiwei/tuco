"""MLP-BC policy, EMA, and checkpoint IO.

Network is byte-compatible with co-curation/train_mlp_bc.py (`MLPBCPolicy`) and
sim2sim_cotrain/mlp_util.py (`MLP`): trunk = [obs*n_obs -> 1024 -> hidden ->
hidden -> hidden] with a deterministic linear head named `action_head`.
Checkpoints written here load in either of those codebases and vice-versa.
"""
from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import ModelConfig


class MLPBCPolicy(nn.Module):
    """Stacked state history -> deterministic action."""

    def __init__(self, obs_dim: int, act_dim: int = 7, n_obs_steps: int = 2,
                 hidden: int = 512, head_key: str = "action_head"):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_obs_steps = n_obs_steps
        self.hidden = hidden
        self.head_key = head_key
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim * n_obs_steps, 1024), nn.ReLU(),
            nn.Linear(1024, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        setattr(self, head_key, nn.Linear(hidden, act_dim))

    @property
    def head(self) -> nn.Linear:
        return getattr(self, self.head_key)

    def features(self, obs_stack: torch.Tensor) -> torch.Tensor:
        """Trunk output (pre-head feature) used by last-layer influence."""
        return self.trunk(obs_stack.reshape(obs_stack.shape[0], -1))

    def forward(self, obs_stack: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(obs_stack))

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> "MLPBCPolicy":
        return cls(cfg.obs_dim, cfg.act_dim, cfg.n_obs_steps, cfg.hidden, cfg.head_key)


class EMA:
    """Exponential moving average matching train_mlp_bc (power=0.75)."""

    def __init__(self, model: nn.Module, power: float = 0.75, max_value: float = 0.9999,
                 min_value: float = 0.0, inv_gamma: float = 1.0, update_after_step: int = 0):
        self.power, self.inv_gamma = power, inv_gamma
        self.min_value, self.max_value = min_value, max_value
        self.update_after_step = update_after_step
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _decay(self, step: int) -> float:
        step = max(0, step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return float(min(self.max_value, max(self.min_value, value)))

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = self._decay(step)
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                s.copy_(v)


def _norm_to_tensor(norm: Dict) -> Dict[str, torch.Tensor]:
    out = {}
    for k in ("s_mean", "s_std", "a_center", "a_scale"):
        v = norm[k]
        out[k] = v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v), dtype=torch.float32)
    return out


def save_checkpoint(path: str, model: MLPBCPolicy, norm: Dict,
                    ema: Optional[EMA] = None, extra: Optional[Dict] = None) -> None:
    ckpt = {
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema.shadow if ema is not None else copy.deepcopy(model.state_dict()),
        "norm_stats": _norm_to_tensor(norm),
        "obs_dim": model.obs_dim, "act_dim": model.act_dim,
        "n_obs_steps": model.n_obs_steps, "hidden": model.hidden,
        "head_key": model.head_key,
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def load_checkpoint(path: str, device: str = "cpu", use_ema: bool = True):
    """Return (model, norm_dict[np.ndarray]). Robust to action_head/mean_head."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["ema_state_dict"] if (use_ema and "ema_state_dict" in ck) else ck["model_state_dict"]
    head_key = "action_head" if "action_head.weight" in sd else "mean_head"
    model = MLPBCPolicy(ck["obs_dim"], ck["act_dim"], ck["n_obs_steps"],
                        ck["hidden"], head_key).to(device)
    missing, _ = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.startswith("log_std")]
    if missing:
        raise ValueError(f"checkpoint is missing critical keys: {missing}")
    model.eval()
    ns = ck["norm_stats"]
    norm = {k: (ns[k].cpu().numpy() if torch.is_tensor(ns[k]) else np.asarray(ns[k]))
            for k in ("s_mean", "s_std", "a_center", "a_scale")}
    return model, norm
