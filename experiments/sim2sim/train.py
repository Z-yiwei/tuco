#!/usr/bin/env python3
"""Train a target-only or target-plus-selected-source State-MLP from scratch."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from tuco.baseline_artifacts import load_curated_selection
from tuco.sim2sim.config import ModelConfig, TrainConfig
from tuco.sim2sim.data import NormStats, episode_frame_indices, load_arrays
from tuco.sim2sim.model import load_checkpoint, save_checkpoint
from tuco.sim2sim.train import train_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Target-only checkpoint; architecture and normalization only",
    )
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--selection", type=Path, help="selected_ids.json from tuco-select")
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
    if (args.source is None) != (args.selection is None):
        parser.error("--source and --selection must be supplied together")
    if args.steps % args.checkpoint_every:
        parser.error("steps must be divisible by checkpoint-every")
    if not 1 <= args.keep_last <= args.steps // args.checkpoint_every:
        parser.error("keep-last is outside the checkpoint range")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    base, norm_dict = load_checkpoint(str(args.base), device="cpu", use_ema=True)
    norm = NormStats.from_dict(norm_dict)
    (target_state, target_action), target_ends = load_arrays(
        str(args.target), ["state", "action"]
    )
    if (
        target_state.shape[1] != base.obs_dim
        or target_action.shape[1] != base.act_dim
    ):
        raise ValueError("target dimensions do not match the base checkpoint")
    target_norm = NormStats.fit(target_state, target_action)
    for key, expected in target_norm.as_dict().items():
        if not np.allclose(norm.as_dict()[key], expected, rtol=1e-5, atol=1e-6):
            raise ValueError(
                f"base normalization does not match the target dataset for {key}"
            )
    states = [target_state]
    actions = [target_action]
    ends = [target_ends]
    selected_ids = np.empty(0, dtype=np.int64)
    if args.source is not None and args.selection is not None:
        (source_state, source_action), source_ends = load_arrays(
            str(args.source), ["state", "action"]
        )
        selected_ids = load_curated_selection(
            args.selection, expected_candidates=len(source_ends)
        ).astype(np.int64)
        if (
            source_state.shape[1:] != target_state.shape[1:]
            or source_action.shape[1:] != target_action.shape[1:]
        ):
            raise ValueError("source and target observation/action shapes differ")
        if len(np.unique(selected_ids)) != len(selected_ids):
            raise ValueError("selection contains duplicate demonstration IDs")
        if np.any((selected_ids < 0) | (selected_ids >= len(source_ends))):
            raise ValueError("selection contains out-of-range demonstration IDs")
        # Dataset order must not leak a method's internal ranking into SGD.
        selected_ids = np.sort(selected_ids)
        frames = episode_frame_indices(source_ends)
        picked_frames = np.concatenate([frames[index] for index in selected_ids])
        states.append(source_state[picked_frames])
        actions.append(source_action[picked_frames])
        lengths = np.diff(np.concatenate([[0], source_ends]))[selected_ids]
        ends.append(np.cumsum(lengths) + len(target_state))
    state = np.concatenate(states).astype(np.float32)
    action = np.concatenate(actions).astype(np.float32)
    episode_ends = np.concatenate(ends).astype(np.int64)
    model_config = ModelConfig(
        obs_dim=base.obs_dim,
        act_dim=base.act_dim,
        n_obs_steps=base.n_obs_steps,
        hidden=base.hidden,
        head_key=base.head_key,
    )
    train_config = TrainConfig(
        steps=args.steps,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    args.output.mkdir(parents=True)

    first_saved = args.steps - (args.keep_last - 1) * args.checkpoint_every

    def checkpoint(step, model, ema) -> None:
        if step < first_saved:
            return
        save_checkpoint(
            str(args.output / f"step_{step:06d}.pt"),
            model,
            norm.as_dict(),
            ema=ema,
            extra={"step": step, "seed": args.seed, "selected_ids": selected_ids.tolist()},
        )

    model, ema = train_policy(
        norm.norm_state(state),
        norm.norm_action(action),
        episode_ends,
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
        extra={"step": args.steps, "seed": args.seed, "selected_ids": selected_ids.tolist()},
    )


if __name__ == "__main__":
    main()
