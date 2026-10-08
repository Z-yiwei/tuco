#!/usr/bin/env python3
"""Collect full successful CupCake MuJoCo trajectories from the Isaac teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr

ROOT = Path(__file__).resolve().parents[2]
FRANKA_DIR = ROOT / "scripts/sim2sim/franka"
if str(FRANKA_DIR) not in sys.path:
    sys.path.insert(0, str(FRANKA_DIR))

import cupcake_mujoco_model as CupCake  # noqa: E402
from cupcake_mujoco_rl_env import (  # noqa: E402
    STABLE_SUCCESS_STEPS,
    MujocoCupCakeVecEnv,
    RewardConfig,
    configure_physics_profile,
)
from cupcake_side_lying_resets import load_npz as load_reset_npz  # noqa: E402
from franka_policy import FrankaPolicy  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--reset-pool-npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-demos", type=int, default=300)
    parser.add_argument("--attempt-limit", type=int, default=None)
    parser.add_argument("--max-abs-action", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--policy-microbatch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_dataset(
    output: Path,
    states: list[np.ndarray],
    actions: list[np.ndarray],
    reset_ids: list[int],
    runtime_episodes: list[int],
    metadata: dict,
) -> None:
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {partial}")
    partial.parent.mkdir(parents=True, exist_ok=True)
    state = np.concatenate(states, axis=0).astype(np.float32, copy=False)
    action = np.concatenate(actions, axis=0).astype(np.float32, copy=False)
    prev_state = np.concatenate(
        [np.concatenate([episode[:1], episode[:-1]], axis=0) for episode in states],
        axis=0,
    ).astype(np.float32, copy=False)
    episode_ends = np.cumsum([len(episode) for episode in states], dtype=np.int64)
    root = zarr.open_group(str(partial), mode="w")
    root.create_group("data")
    root.create_group("meta")
    chunk = min(4096, len(state))
    root.create_dataset("data/state", data=state, chunks=(chunk, state.shape[1]))
    root.create_dataset("data/prev_state", data=prev_state, chunks=(chunk, state.shape[1]))
    root.create_dataset("data/action", data=action, chunks=(chunk, action.shape[1]))
    root.create_dataset("meta/episode_ends", data=episode_ends)
    root.create_dataset("meta/reset_ids", data=np.asarray(reset_ids, dtype=np.int64))
    root.create_dataset("meta/runtime_episodes", data=np.asarray(runtime_episodes, dtype=np.int64))
    root.create_dataset("meta/success_seen", data=np.ones(len(states), dtype=np.bool_))
    root.attrs.update(metadata)
    os.replace(partial, output)


def main() -> None:
    args = parse_args()
    if min(args.num_demos, args.batch_size, args.policy_microbatch_size, args.steps) <= 0:
        raise ValueError("counts and steps must be positive")
    if args.attempt_limit is not None and args.attempt_limit <= 0:
        raise ValueError("attempt-limit must be positive")
    if args.max_abs_action <= 0:
        raise ValueError("max-abs-action must be positive")
    if args.steps != 160:
        raise ValueError("CupCake BC collection requires full 160-step trajectories")

    source = args.source_zarr.resolve()
    reset_path = args.reset_pool_npz.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    for path in (source, reset_path, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    source_store = zarr.open(str(source), mode="r")
    source_episode_count = len(source_store["meta/episode_ends"])
    reset_pool = load_reset_npz(reset_path)
    with np.load(reset_path) as reset_payload:
        reset_type = str(np.asarray(reset_payload["reset_type"]).item())
    attempt_count = min(
        len(reset_pool),
        args.attempt_limit if args.attempt_limit is not None else len(reset_pool),
    )
    policy = FrankaPolicy.load_from_checkpoint(str(checkpoint), device=args.device)
    configure_physics_profile(
        physics_substeps=16,
        friction_combine="physx_average",
        cupcake_collision="convex_decomposition",
        cupcake_plate_collision="base_cylinder",
        hand_collision="source_usd",
        finger_collision="mimic",
    )

    kept_states: list[np.ndarray] = []
    kept_actions: list[np.ndarray] = []
    kept_reset_ids: list[int] = []
    kept_runtime_episodes: list[int] = []
    attempted = abnormal_total = action_filtered_total = 0
    for batch_start in range(0, attempt_count, args.batch_size):
        if len(kept_states) >= args.num_demos:
            break
        reset_ids = list(range(batch_start, min(batch_start + args.batch_size, attempt_count)))
        runtime_episodes = [reset_id % source_episode_count for reset_id in reset_ids]
        assignments = [(str(source), episode) for episode in runtime_episodes]
        env = MujocoCupCakeVecEnv(
            reset_pool,
            assignments,
            runtime_catalog_entries=assignments,
            runtime_mode="fixed_per_worker",
            seed=args.seed + batch_start,
            device="cpu",
            max_episode_length=args.steps,
            reward=RewardConfig(),
            startup_timeout_s=600.0,
        )
        try:
            observations = env.reset(reset_ids)
            count = len(reset_ids)
            batch_states = np.empty((count, args.steps, 200), dtype=np.float32)
            batch_actions = np.empty((count, args.steps, 7), dtype=np.float32)
            abnormal = np.zeros(count, dtype=np.bool_)
            stable = np.zeros(count, dtype=np.bool_)
            for step in range(args.steps):
                batch_states[:, step] = observations["policy"].cpu().numpy()
                with torch.inference_mode():
                    policy_obs = observations["policy"].to(args.device)
                    action_tensor = torch.cat(
                        [policy(policy_obs[i : i + args.policy_microbatch_size])
                         for i in range(0, count, args.policy_microbatch_size)],
                        dim=0,
                    )
                batch_actions[:, step] = action_tensor.cpu().numpy()
                observations, _, _, extras = env.step(action_tensor)
                task = {key: value.cpu().numpy() for key, value in extras["task"].items()}
                abnormal |= task["abnormal"] > 0.5
                stable |= task["stable_success"] > 0.5
            for local, reset_id in enumerate(reset_ids):
                action_in_range = bool(np.max(np.abs(batch_actions[local])) <= args.max_abs_action)
                if stable[local] and not abnormal[local] and action_in_range:
                    kept_states.append(batch_states[local].copy())
                    kept_actions.append(batch_actions[local].copy())
                    kept_reset_ids.append(reset_id)
                    kept_runtime_episodes.append(runtime_episodes[local])
                    if len(kept_states) >= args.num_demos:
                        break
                elif stable[local] and not abnormal[local] and not action_in_range:
                    action_filtered_total += 1
            attempted += count
            abnormal_total += int(abnormal.sum())
            print(
                f"[collect] attempted={attempted}/{attempt_count} "
                f"kept={len(kept_states)}/{args.num_demos} "
                f"batch_stable={int(stable.sum())}/{count} "
                f"batch_abnormal={int(abnormal.sum())}",
                flush=True,
            )
        finally:
            env.close()

    if len(kept_states) < args.num_demos:
        raise RuntimeError(f"reset pool exhausted: collected {len(kept_states)}/{args.num_demos} demos")
    metadata = {
        "definition": "full successful MuJoCo CupCake MLP-BC demonstrations",
        "task_pair": "CupCake__Plate",
        "reset_type": reset_type,
        "source_zarr": str(source),
        "source_panel": source_store.attrs.get("panel"),
        "reset_pool_npz": str(reset_path),
        "reset_pool_sha256": sha256(reset_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "teacher_checkpoint_iteration": policy.ckpt_iter,
        "num_demos": args.num_demos,
        "episode_steps": args.steps,
        "attempted_resets": attempted,
        "attempt_limit": attempt_count,
        "abnormal_attempts": abnormal_total,
        "max_abs_action": args.max_abs_action,
        "action_filtered_attempts": action_filtered_total,
        "success_filtering": True,
        "observation_dim": 200,
        "action_dim": 7,
        "gripper_contract": "teacher action[6] recorded; grasp guard controls execution",
        "physics_profile": json.dumps({
            "physics_substeps": 16,
            "friction_combine": "physx_average",
            "cupcake_collision": "convex_decomposition",
            "cupcake_plate_collision": "base_cylinder",
            "hand_collision": "source_usd",
            "finger_collision": "mimic",
        }, sort_keys=True),
        "seed": args.seed,
        "command": " ".join(sys.argv),
        "strict_position_threshold_m": CupCake.SUCCESS_POSITION_M,
        "strict_orientation_xy_threshold_rad": CupCake.SUCCESS_ORIENTATION_XY_RAD,
        "stable_success_steps": STABLE_SUCCESS_STEPS,
    }
    write_dataset(output, kept_states, kept_actions, kept_reset_ids, kept_runtime_episodes, metadata)
    print(json.dumps({
        "output": str(output),
        "demos": len(kept_states),
        "frames": sum(map(len, kept_states)),
        "attempted": attempted,
        "collection_sr": len(kept_states) / attempted,
        "action_filtered": action_filtered_total,
    }, indent=2))


if __name__ == "__main__":
    main()
