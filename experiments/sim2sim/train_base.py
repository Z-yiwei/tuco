#!/usr/bin/env python3
"""Train the target-only State-MLP and save its late checkpoints."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from tuco.sim2sim.config import ModelConfig, TrainConfig
from tuco.sim2sim.data import NormStats, load_arrays
from tuco.sim2sim.model import save_checkpoint
from tuco.sim2sim.train import train_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=TrainConfig.steps)
    parser.add_argument("--checkpoint-every", type=int, default=1_000)
    parser.add_argument("--keep-last", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.steps % args.checkpoint_every:
        parser.error("steps must be divisible by checkpoint-every")
    if not 1 <= args.keep_last <= args.steps // args.checkpoint_every:
        parser.error("keep-last is outside the checkpoint range")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    (state, action), ends = load_arrays(str(args.target), ["state", "action"])
    norm = NormStats.fit(state, action)
    model_config = ModelConfig(obs_dim=state.shape[1], act_dim=action.shape[1])
    train_config = TrainConfig(
        steps=args.steps,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    args.output.mkdir(parents=True)
    first_saved = args.steps - (args.keep_last - 1) * args.checkpoint_every

    def checkpoint(step, model, ema) -> None:
        if step >= first_saved:
            save_checkpoint(
                str(args.output / f"step_{step:06d}.pt"),
                model,
                norm.as_dict(),
                ema=ema,
                extra={"step": step, "seed": args.seed, "role": "target_only"},
            )

    model, ema = train_policy(
        norm.norm_state(state).astype(np.float32),
        norm.norm_action(action).astype(np.float32),
        ends,
        target_idxs=None,
        model_cfg=model_config,
        train_cfg=train_config,
        device=args.device,
        checkpoint_callback=checkpoint,
        checkpoint_every=args.checkpoint_every,
    )
    save_checkpoint(
        str(args.output / "final.pt"),
        model,
        norm.as_dict(),
        ema=ema,
        extra={"step": args.steps, "seed": args.seed, "role": "target_only"},
    )


if __name__ == "__main__":
    main()
