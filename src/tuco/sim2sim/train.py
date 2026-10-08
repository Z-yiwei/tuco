"""Training loop shared by base training and co-training (train-from-scratch)."""
from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import ModelConfig, TrainConfig
from .data import WindowDataset
from .model import EMA, MLPBCPolicy


def _make_scheduler(optim, warmup: int, steps: int):
    def lr_lambda(step: int):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)


def train_policy(
    state_n: np.ndarray,
    action_n: np.ndarray,
    ends: np.ndarray,
    target_idxs: Optional[np.ndarray],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: str = "cuda",
    steps: Optional[int] = None,
    log: Callable[[str], None] = print,
    prev_state_n: Optional[np.ndarray] = None,
    checkpoint_callback: Optional[Callable[[int, MLPBCPolicy, EMA], None]] = None,
    checkpoint_every: int = 0,
) -> Tuple[MLPBCPolicy, EMA]:
    """Train an MLP-BC on the given (already normalized) frames.

    target_idxs selects which frames act as regression targets (None = all).
    prev_state_n optionally supplies the observation immediately before each
    episode boundary for a two-observation policy.
    For co-training pass the concatenation of A frames and the selected B frames.
    """
    steps = int(steps if steps is not None else train_cfg.steps)
    if checkpoint_every < 0:
        raise ValueError("checkpoint_every must be non-negative")
    if checkpoint_callback is not None and checkpoint_every == 0:
        raise ValueError("checkpoint_callback requires checkpoint_every > 0")
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    ds = WindowDataset(state_n, action_n, ends, n_obs=model_cfg.n_obs_steps,
                       target_idxs=target_idxs, prev_state_n=prev_state_n)
    gen = torch.Generator().manual_seed(train_cfg.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, drop_last=True,
        persistent_workers=train_cfg.num_workers > 0, generator=gen)

    model = MLPBCPolicy.from_config(model_cfg).to(device)
    ema = EMA(model, power=train_cfg.ema_power, max_value=train_cfg.ema_max)
    optim = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, betas=train_cfg.betas,
                              eps=1e-8, weight_decay=train_cfg.weight_decay)
    sched = _make_scheduler(optim, train_cfg.warmup, steps)
    loss_fn = nn.functional.mse_loss if train_cfg.loss == "mse" else nn.functional.l1_loss

    log(f"[train] {len(ds)} samples, {steps} steps, batch={train_cfg.batch_size}")
    model.train()
    it, step, running = iter(loader), 0, 0.0
    while step < steps:
        try:
            obs, act = next(it)
        except StopIteration:
            it = iter(loader)
            obs, act = next(it)
        obs, act = obs.to(device), act.to(device)
        loss = loss_fn(model(obs), act)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optim.step()
        sched.step()
        ema.update(model, step)
        running += loss.item()
        step += 1
        if step % train_cfg.log_every == 0:
            log(f"[train] step {step}/{steps} loss={running/train_cfg.log_every:.6f} "
                f"lr={sched.get_last_lr()[0]:.2e}")
            running = 0.0
        if checkpoint_callback is not None and step % checkpoint_every == 0:
            checkpoint_callback(step, model, ema)
    return model, ema
