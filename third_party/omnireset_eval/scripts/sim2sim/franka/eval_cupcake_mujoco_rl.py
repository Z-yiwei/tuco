#!/usr/bin/env python3
"""Parallel deterministic evaluation of a CupCake policy in MuJoCo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

import compare_cupcake_mujoco as Replay
import cupcake_mujoco_model as CupCake
from cupcake_side_lying_resets import load_npz as load_side_lying_reset_npz
from cupcake_mujoco_rl_env import (
    STABLE_SUCCESS_STEPS,
    MujocoCupCakeVecEnv,
    RewardConfig,
    configure_physics_profile,
    load_zarr_reset_raw_pool,
)
from franka_policy import FrankaPolicy


ROOT = Path(__file__).resolve().parents[3]
CUPCAKE_SCRIPT_DIR = ROOT / "scripts/cupcake"
if str(CUPCAKE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(CUPCAKE_SCRIPT_DIR))
from train_mlp_bc_cupcake import MLPBCPolicy  # noqa: E402


DEFAULT_SOURCE = Path(
    "datasets/cupcake_sim2sim_t0_20260815/isaac_fit_b3center.zarr"
)


class MLPBCInference:
    """Normalized two-frame MLP-BC inference with per-worker history."""

    def __init__(self, checkpoint: Path, device: str):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = torch.device(device)
        self.model = MLPBCPolicy(
            obs_dim=int(payload["obs_dim"]),
            act_dim=int(payload["act_dim"]),
            n_obs_steps=int(payload["n_obs_steps"]),
            hidden=int(payload["hidden"]),
        ).to(self.device)
        self.model.load_state_dict(payload["ema_state_dict"], strict=True)
        self.model.eval()
        stats = payload["norm_stats"]
        self.s_mean = stats["s_mean"].to(self.device)
        self.s_std = stats["s_std"].to(self.device)
        self.a_center = stats["a_center"].to(self.device)
        self.a_scale = stats["a_scale"].to(self.device)
        self.previous: torch.Tensor | None = None
        self.ckpt_iter = int(payload.get("step", -1))

    def reset(self, observation: torch.Tensor) -> None:
        self.previous = observation.to(self.device).clone()

    @torch.inference_mode()
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        current = observation.to(self.device)
        if self.previous is None or self.previous.shape != current.shape:
            self.reset(current)
        assert self.previous is not None
        stacked = torch.stack([self.previous, current], dim=1)
        normalized = (stacked - self.s_mean) / self.s_std
        action = self.model(normalized) * self.a_scale + self.a_center
        self.previous = current.clone()
        return action


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zarr", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--reset-pool-npz",
        type=Path,
        default=None,
        help=(
            "Optional independent CupCakeSideLyingFront3cm raw-state pool. "
            "The source Zarr still supplies aligned runtime physics."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--policy-type", choices=("rsl_rl", "mlp_bc"), default="rsl_rl"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--save-rollouts",
        type=Path,
        default=None,
        help=(
            "Optionally export policy state/action trajectories and the strict "
            "stable success label as a provenance-bound Zarr for offline "
            "selection-method scoring."
        ),
    )
    parser.add_argument("--episodes", default="all")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--policy-microbatch-size",
        type=int,
        default=1,
        help=(
            "Inference batch size inside each parallel physics batch. The default "
            "keeps each episode's deterministic actor numerics independent of the "
            "number of MuJoCo workers."
        ),
    )
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--physics-substeps", type=int, default=16)
    parser.add_argument(
        "--friction-combine",
        choices=("legacy_b3", "physx_average"),
        default="physx_average",
    )
    parser.add_argument(
        "--cupcake-collision",
        choices=(
            "convex_decomposition",
            "radial32",
            "coacd",
            "sdf",
            "convex_mesh",
            "compound",
        ),
        default="convex_decomposition",
    )
    parser.add_argument(
        "--hand-collision",
        choices=("menagerie", "source_usd", "disabled"),
        default="source_usd",
    )
    parser.add_argument(
        "--cupcake-plate-collision",
        choices=("convex_hull", "radial16", "base_cylinder"),
        default="base_cylinder",
    )
    parser.add_argument(
        "--finger-collision",
        choices=("mimic", "menagerie", "menagerie_mesh_only"),
        default="mimic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or args.policy_microbatch_size <= 0
        or args.steps <= 0
        or args.physics_substeps <= 0
    ):
        raise ValueError(
            "batch-size, policy-microbatch-size, steps, and physics-substeps "
            "must be positive"
        )
    source = args.source_zarr.resolve()
    checkpoint = args.checkpoint.resolve()
    out = args.out.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    store = zarr.open(str(source), mode="r")
    source_episode_count = len(store["meta/episode_ends"])
    if args.reset_pool_npz is None:
        reset_pool = load_zarr_reset_raw_pool(source, required_panel=None)
        episode_count = source_episode_count
        reset_pool_path = None
    else:
        reset_pool_path = args.reset_pool_npz.resolve()
        reset_pool = load_side_lying_reset_npz(reset_pool_path)
        episode_count = len(reset_pool)
    episodes = Replay.parse_episodes(args.episodes, episode_count)
    configure_physics_profile(
        physics_substeps=args.physics_substeps,
        friction_combine=args.friction_combine,
        cupcake_collision=args.cupcake_collision,
        cupcake_plate_collision=args.cupcake_plate_collision,
        hand_collision=args.hand_collision,
        finger_collision=args.finger_collision,
    )
    if args.policy_type == "rsl_rl":
        policy = FrankaPolicy.load_from_checkpoint(str(checkpoint), device=args.device)
    else:
        policy = MLPBCInference(checkpoint, device=args.device)

    rows: list[dict[str, Any]] = []
    rollout_states: list[np.ndarray] = []
    rollout_actions: list[np.ndarray] = []
    rollout_successes: list[bool] = []
    rollout_reset_indices: list[int] = []
    for batch_start in range(0, len(episodes), args.batch_size):
        batch = episodes[batch_start : batch_start + args.batch_size]
        runtime_episodes = [episode % source_episode_count for episode in batch]
        assignments = [
            (str(source), runtime_episode) for runtime_episode in runtime_episodes
        ]
        print(
            f"[cupcake-eval] batch={batch_start // args.batch_size + 1} "
            f"episodes={batch[0]}..{batch[-1]} profile="
            f"{args.cupcake_collision}/sub{args.physics_substeps}",
            flush=True,
        )
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
            observations = env.reset(batch)
            if args.policy_type == "mlp_bc":
                policy.reset(observations["policy"])
            count = len(batch)
            active = np.ones(count, dtype=np.bool_)
            any_success = np.zeros(count, dtype=np.bool_)
            stable_success = np.zeros(count, dtype=np.bool_)
            any_relaxed_success = np.zeros(count, dtype=np.bool_)
            stable_relaxed_success = np.zeros(count, dtype=np.bool_)
            abnormal = np.zeros(count, dtype=np.bool_)
            success_streak = np.zeros(count, dtype=np.int32)
            max_success_streak = np.zeros(count, dtype=np.int32)
            relaxed_success_streak = np.zeros(count, dtype=np.int32)
            max_relaxed_success_streak = np.zeros(count, dtype=np.int32)
            best_position = np.full(count, np.inf, dtype=np.float64)
            best_orientation = np.full(count, np.inf, dtype=np.float64)
            final_position = np.full(count, np.inf, dtype=np.float64)
            final_orientation = np.full(count, np.inf, dtype=np.float64)
            completed_steps = np.zeros(count, dtype=np.int32)
            batch_states: list[list[np.ndarray]] = [[] for _ in range(count)]
            batch_actions: list[list[np.ndarray]] = [[] for _ in range(count)]

            for step in range(args.steps):
                state_before = observations["policy"].detach().cpu().numpy()
                active_before = active.copy()
                with torch.inference_mode():
                    policy_observations = observations["policy"].to(args.device)
                    if args.policy_type == "mlp_bc":
                        actions = policy.act(policy_observations)
                    else:
                        actions = torch.cat(
                            [
                                policy(
                                    policy_observations[
                                        index : index + args.policy_microbatch_size
                                    ]
                                )
                                for index in range(
                                    0, count, args.policy_microbatch_size
                                )
                            ],
                            dim=0,
                        )
                action_before = actions.detach().cpu().numpy()
                if args.save_rollouts is not None:
                    for local in np.flatnonzero(active_before):
                        batch_states[local].append(
                            np.asarray(state_before[local], dtype=np.float32)
                        )
                        batch_actions[local].append(
                            np.asarray(action_before[local], dtype=np.float32)
                        )
                observations, _, dones, extras = env.step(actions)
                task = {
                    key: value.detach().cpu().numpy() for key, value in extras["task"].items()
                }
                pose = task["pose_success"] > 0.5
                relaxed_pose = task["relaxed_pose_success"] > 0.5
                step_abnormal = task["abnormal"] > 0.5
                valid = active & ~step_abnormal
                any_success[valid] |= pose[valid]
                success_streak[valid] = np.where(
                    pose[valid], success_streak[valid] + 1, 0
                )
                max_success_streak = np.maximum(max_success_streak, success_streak)
                stable_success |= max_success_streak >= STABLE_SUCCESS_STEPS
                any_relaxed_success[valid] |= relaxed_pose[valid]
                relaxed_success_streak[valid] = np.where(
                    relaxed_pose[valid], relaxed_success_streak[valid] + 1, 0
                )
                max_relaxed_success_streak = np.maximum(
                    max_relaxed_success_streak, relaxed_success_streak
                )
                stable_relaxed_success |= (
                    max_relaxed_success_streak >= STABLE_SUCCESS_STEPS
                )
                best_position[valid] = np.minimum(
                    best_position[valid], task["position_error_m"][valid]
                )
                best_orientation[valid] = np.minimum(
                    best_orientation[valid], task["orientation_error_rad"][valid]
                )
                final_position[valid] = task["position_error_m"][valid]
                final_orientation[valid] = task["orientation_error_rad"][valid]
                completed_steps[valid] = step + 1
                abnormal |= active & step_abnormal
                active &= ~step_abnormal
                if not active.any():
                    break

            if reset_pool_path is None:
                source_success = np.asarray(store["meta/success_seen"])[batch]
                reset_indices = np.asarray(store["meta/reset_indices"])[batch]
            else:
                source_success = np.full(len(batch), -1, dtype=np.int64)
                reset_indices = np.asarray(batch, dtype=np.int64)
            for local, episode in enumerate(batch):
                rows.append(
                    {
                        "episode": episode,
                        "reset_index": int(reset_indices[local]),
                        "source_success_seen": int(source_success[local]),
                        "any_pose_success": int(any_success[local]),
                        "stable_success": int(stable_success[local]),
                        "max_success_streak": int(max_success_streak[local]),
                        "any_relaxed_pose_success": int(any_relaxed_success[local]),
                        "stable_relaxed_success": int(stable_relaxed_success[local]),
                        "max_relaxed_success_streak": int(
                            max_relaxed_success_streak[local]
                        ),
                        "abnormal": int(abnormal[local]),
                        "steps_completed": int(completed_steps[local]),
                        "best_position_error_m": float(best_position[local]),
                        "best_orientation_error_rad": float(best_orientation[local]),
                        "final_position_error_m": float(final_position[local]),
                        "final_orientation_error_rad": float(final_orientation[local]),
                    }
                )
                if args.save_rollouts is not None:
                    if not batch_states[local]:
                        raise RuntimeError(
                            f"rollout episode {episode} produced no state/action frames"
                        )
                    rollout_states.append(np.stack(batch_states[local]))
                    rollout_actions.append(np.stack(batch_actions[local]))
                    rollout_successes.append(bool(stable_success[local]))
                    rollout_reset_indices.append(int(reset_indices[local]))
        finally:
            env.close()

    write_csv(out / "episodes.csv", rows)
    summary = {
        "definition": "parallel deterministic CupCake MuJoCo closed-loop evaluation",
        "source_zarr": str(source),
        "source_panel": store.attrs.get("panel"),
        "reset_pool_npz": str(reset_pool_path) if reset_pool_path else None,
        "source_success_available": reset_pool_path is None,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_iteration": policy.ckpt_iter,
        "policy_type": args.policy_type,
        "episodes": episodes,
        "steps": args.steps,
        "policy_microbatch_size": args.policy_microbatch_size,
        "profile": {
            "physics_substeps": args.physics_substeps,
            "friction_combine": args.friction_combine,
            "cupcake_collision": args.cupcake_collision,
            "cupcake_plate_collision": args.cupcake_plate_collision,
            "hand_collision": args.hand_collision,
            "finger_collision": args.finger_collision,
        },
        "episodes_evaluated": len(rows),
        "success_definitions": {
            "strict": {
                "position_error_m_lt": CupCake.SUCCESS_POSITION_M,
                "abs_roll_plus_abs_pitch_rad_lt": (
                    CupCake.SUCCESS_ORIENTATION_XY_RAD
                ),
                "stable_steps_gte": STABLE_SUCCESS_STEPS,
            },
            "relaxed": {
                "position_error_m_lt": CupCake.RELAXED_SUCCESS_POSITION_M,
                "abs_roll_plus_abs_pitch_rad_lt": (
                    CupCake.RELAXED_SUCCESS_ORIENTATION_XY_RAD
                ),
                "stable_steps_gte": STABLE_SUCCESS_STEPS,
            },
        },
        "source_successes": (
            sum(row["source_success_seen"] for row in rows)
            if reset_pool_path is None
            else None
        ),
        "any_pose_successes": sum(row["any_pose_success"] for row in rows),
        "stable_successes": sum(row["stable_success"] for row in rows),
        "any_relaxed_pose_successes": sum(
            row["any_relaxed_pose_success"] for row in rows
        ),
        "stable_relaxed_successes": sum(
            row["stable_relaxed_success"] for row in rows
        ),
        "abnormal_episodes": sum(row["abnormal"] for row in rows),
        "results": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.save_rollouts is not None:
        rollout_path = args.save_rollouts.resolve()
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        rollout = zarr.open(str(rollout_path), mode="w")
        state = np.concatenate(rollout_states).astype(np.float32, copy=False)
        action = np.concatenate(rollout_actions).astype(np.float32, copy=False)
        episode_ends = np.cumsum(
            [len(value) for value in rollout_states], dtype=np.int64
        )
        rollout.create_dataset(
            "data/state", data=state, chunks=(min(1024, len(state)), state.shape[1])
        )
        rollout.create_dataset(
            "data/action", data=action, chunks=(min(1024, len(action)), action.shape[1])
        )
        rollout.create_dataset(
            "data/success", data=np.asarray(rollout_successes, dtype=np.bool_)
        )
        rollout.create_dataset("meta/episode_ends", data=episode_ends)
        rollout.create_dataset(
            "meta/reset_indices",
            data=np.asarray(rollout_reset_indices, dtype=np.int64),
        )
        rollout.attrs.update(
            {
                "task": "CupCake-MuJoCo-FixedHome-State-v1",
                "reset_type": "CupCakeSideLyingFront3cmFixedHome",
                "seed": int(args.seed),
                "num_envs": int(args.batch_size),
                "num_episodes": len(rollout_states),
                "success_pos_threshold": float(CupCake.SUCCESS_POSITION_M),
                "success_ori_threshold": float(
                    CupCake.SUCCESS_ORIENTATION_XY_RAD
                ),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": file_sha256(checkpoint),
                "eval_contract_id": "mujoco-cupcake-fixedhome-stable-v1",
                "simulator": "MuJoCo",
                "metric": "strict_stable_success",
                "reset_pool_npz": str(reset_pool_path) if reset_pool_path else "",
                "definition": (
                    "base-policy closed-loop MuJoCo A rollouts for offline data "
                    "selection scoring; sourced from the training reset panel and "
                    "disjoint from the final held-out evaluation reset panel"
                ),
            }
        )
        print(
            f"[rollouts] saved {rollout_path}: episodes={len(rollout_states)} "
            f"frames={len(state)} successes={sum(rollout_successes)}",
            flush=True,
        )
    print(
        json.dumps(
            {
                "episodes": len(rows),
                "any_pose_successes": summary["any_pose_successes"],
                "stable_successes": summary["stable_successes"],
                "stable_relaxed_successes": summary[
                    "stable_relaxed_successes"
                ],
                "abnormal_episodes": summary["abnormal_episodes"],
                "out": str(out),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
