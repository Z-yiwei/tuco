"""State-MLP and TRAK configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    obs_dim: int = 200
    act_dim: int = 7
    n_obs_steps: int = 2
    hidden: int = 512
    head_key: str = "action_head"


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 50_000
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup: int = 500
    betas: tuple[float, float] = (0.95, 0.999)
    grad_clip: float = 1.0
    loss: str = "mse"
    ema_power: float = 0.75
    ema_max: float = 0.9999
    seed: int = 42
    log_every: int = 200
    num_workers: int = 4


@dataclass(frozen=True)
class InfluenceConfig:
    proj_dim: int = 4_000
    proj_seed: int = 0
    grad_batch: int = 32
    lambda_reg: float = 0.0
