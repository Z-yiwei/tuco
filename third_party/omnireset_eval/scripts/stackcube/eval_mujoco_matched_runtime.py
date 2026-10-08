#!/usr/bin/env python3
"""Evaluate StackCube MLP-BC in the exact matched-runtime MuJoCo A environment.

The rich Isaac runtime source supplies only reset/controller/scene parameters.
Every policy observation and transition is generated live by MuJoCo using the
same model builder, controller, and physics settings as A-demo collection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "")

import mujoco
import numpy as np
import torch
import zarr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sim_eval"))
sys.path.insert(0, str(HERE / "sim2sim_franka"))

import closed_loop_stackcube_eval as Stack
import diagnose_stackcube_top2_chunk_replay as Diagnose
import stackcube_offline_error_cotrain as Offline


PHYSICS_SUBSTEPS = 16
JACOBIAN_POINT = "physx_com"
FINGER_VELOCITY_LIMITS = (0.05, 0.04)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def validate_runtime(path: Path):
    source = zarr.open(str(path), mode="r")
    starts, ends, reset_ids = Diagnose.validate_source(source)
    if np.any(ends - starts != 160):
        raise ValueError("runtime source must contain fixed 160-row episodes")
    if source.attrs.get("preserve_controller_events") is not True:
        raise ValueError("runtime source must preserve controller events")
    required = (
        "data/raw_state", "data/arm_kp", "data/arm_kd", "data/arm_scale",
        "data/gripper_joint_stiffness", "data/gripper_joint_damping",
        "data/gripper_joint_effort_limit", "data/robot_body_masses",
        "data/robot_material_properties", "data/insertive_object_body_masses",
        "data/insertive_object_material_properties",
        "data/receptive_object_body_masses",
        "data/receptive_object_material_properties", "data/table_body_masses",
        "data/table_material_properties",
    )
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"runtime source lacks fields: {missing}")
    return source, starts, ends, reset_ids


def run_episode(source, start: int, end: int, reset_id: int, policy, args):
    raw = np.asarray(source["data/raw_state"][start], dtype=np.float64)
    model, ctrl, effective = Offline.build_episode_model(
        source, start, end, raw,
        physics_substeps=PHYSICS_SUBSTEPS,
        jacobian_point=JACOBIAN_POINT,
        finger_velocity_limits=FINGER_VELOCITY_LIMITS,
    )
    data = mujoco.MjData(model)
    Offline.set_boundary_state(model, data, ctrl, raw)
    builder = Stack.StackCubeObsBuilder(ctrl)
    builder.reset(data)
    policy.reset()
    previous_action = np.zeros(7, dtype=np.float32)
    first_proxy_step = -1
    best_proxy_dist = float("inf")
    policy_steps = 0
    rollout_states, rollout_actions = [], []

    for step in range(args.horizon):
        observation = builder.step(data, previous_action)
        with torch.inference_mode():
            action = policy(torch.from_numpy(observation).unsqueeze(0)).cpu().numpy()[0]
        if args.save_rollouts:
            rollout_states.append(np.asarray(observation, dtype=np.float32))
            rollout_actions.append(np.asarray(action, dtype=np.float32))
        Offline.execute_action(
            model, data, ctrl, action, None,
            jacobian_point=JACOBIAN_POINT,
            finger_velocity_limits=FINGER_VELOCITY_LIMITS,
        )
        previous_action = action.astype(np.float32)
        policy_steps = step + 1
        metric = Stack.stack_metrics(model, data, ctrl)
        best_proxy_dist = min(best_proxy_dist, metric["align_pos_dist_m"])
        if metric["proxy_pose"]:
            first_proxy_step = policy_steps
            break

    # Release-and-settle is a stricter confirmation of a policy success, not a
    # second chance for an unsuccessful rollout.  Previously we released after
    # every rollout, including those that never reached the proxy pose.  A cube
    # could then fall into place during those 60 zero-action steps and be
    # incorrectly counted as a stable policy success.
    stable_flags = []
    if first_proxy_step >= 0:
        for _ in range(args.release_steps):
            Offline.execute_action(
                model, data, ctrl, np.zeros(7, dtype=np.float32), False,
                jacobian_point=JACOBIAN_POINT,
                finger_velocity_limits=FINGER_VELOCITY_LIMITS,
            )
            metric = Stack.stack_metrics(model, data, ctrl)
            stable_flags.append(
                metric["strict_pose"]
                and metric["receptive_contact"]
                and metric["no_robot_contact"]
                and metric["lin_vel_mps"] < args.stable_lin_vel
                and metric["ang_vel_radps"] < args.stable_ang_vel
            )
    final = Stack.stack_metrics(model, data, ctrl)
    stable = len(stable_flags) >= args.stable_window and all(stable_flags[-args.stable_window:])
    result = {
        "reset_index": int(reset_id),
        "policy_steps_executed": int(policy_steps),
        "first_proxy_step": int(first_proxy_step),
        "best_proxy_align_pos_dist_m": float(best_proxy_dist),
        "success_stable_after_release": bool(stable),
        "success_proxy_final": bool(final["proxy_pose"]),
        "success_strict_final": bool(final["strict_pose"]),
        "final": final,
        "effective_config": effective,
    }
    if args.save_rollouts:
        result["rollout_state"] = np.stack(rollout_states)
        result["rollout_action"] = np.stack(rollout_actions)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime_source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--episode_indices", default="",
        help="optional comma-separated runtime episode indices (overrides prefix selection)",
    )
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--release_steps", type=int, default=60)
    parser.add_argument("--stable_window", type=int, default=10)
    parser.add_argument("--stable_lin_vel", type=float, default=0.01)
    parser.add_argument("--stable_ang_vel", type=float, default=0.2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--save_rollouts", default="",
        help=(
            "optional zarr output containing the policy's closed-loop state/action "
            "trajectory and one stable-success label per episode"
        ),
    )
    args = parser.parse_args()
    runtime_path = Path(args.runtime_source).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    source, starts, ends, reset_ids = validate_runtime(runtime_path)
    if args.episode_indices:
        selected = [int(value) for value in args.episode_indices.split(",")]
        if len(selected) != len(set(selected)) or any(i < 0 or i >= len(ends) for i in selected):
            raise ValueError(f"invalid episode_indices={selected} for runtime episodes={len(ends)}")
    else:
        if args.episodes <= 0 or args.episodes > len(ends):
            raise ValueError(f"episodes={args.episodes}, runtime episodes={len(ends)}")
        selected = list(range(args.episodes))
    out.mkdir(parents=True, exist_ok=True)
    policy = Stack.MLPBCPolicyRunner(str(checkpoint_path), use_ema=True, device=args.device)
    results = []
    for output_ep, ep in enumerate(selected):
        row = run_episode(
            source, int(starts[ep]), int(ends[ep]), int(reset_ids[ep]), policy, args
        )
        row["episode"] = output_ep
        row["runtime_episode_index"] = ep
        results.append(row)
        print(
            f"[matched-eval] ep={output_ep:03d} source_ep={ep:03d} reset={row['reset_index']:04d} "
            f"proxy_step={row['first_proxy_step']:3d} "
            f"stable={int(row['success_stable_after_release'])}", flush=True
        )

    compact = []
    for row in results:
        item = {
            key: value for key, value in row.items()
            if key not in {"final", "effective_config", "rollout_state", "rollout_action"}
        }
        item.update({f"final_{key}": value for key, value in row["final"].items()})
        compact.append(item)
    jsonl = out / "closed_loop_stackcube_episodes.jsonl"
    with jsonl.open("w", encoding="utf-8") as file:
        for row in compact:
            file.write(json.dumps(row, sort_keys=True) + "\n")
    np.savez_compressed(
        out / "closed_loop_stackcube_eval.npz",
        reset_indices=np.asarray([r["reset_index"] for r in results], dtype=np.int32),
        stable_success=np.asarray([r["success_stable_after_release"] for r in results], dtype=np.bool_),
        proxy_reached=np.asarray([r["first_proxy_step"] >= 0 for r in results], dtype=np.bool_),
    )
    n = len(results)
    stable = sum(r["success_stable_after_release"] for r in results)
    proxy_reached = sum(r["first_proxy_step"] >= 0 for r in results)
    summary = {
        "definition": "live MuJoCo A eval using the exact A-collector matched runtime/model/controller; release at first proxy pose",
        "checkpoint": str(checkpoint_path),
        "runtime_source": str(runtime_path),
        "runtime_source_protocol": source.attrs.get("protocol_id", ""),
        "episodes": n,
        "horizon": args.horizon,
        "release_steps": args.release_steps,
        "stable_window": args.stable_window,
        "stable_after_release_successes": int(stable),
        "stable_after_release_sr": stable / n,
        "proxy_reached_successes": int(proxy_reached),
        "proxy_reached_sr": proxy_reached / n,
        "runtime_episode_indices": selected,
        "reset_indices": [int(reset_ids[i]) for i in selected],
        "mujoco_profile": results[0]["effective_config"] if results else {},
    }
    with (out / "closed_loop_stackcube_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True, default=json_default)
    if args.save_rollouts:
        rollout_path = Path(args.save_rollouts).resolve()
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        rollout = zarr.open(str(rollout_path), mode="w")
        states = np.concatenate([row["rollout_state"] for row in results]).astype(np.float32)
        actions = np.concatenate([row["rollout_action"] for row in results]).astype(np.float32)
        episode_ends = np.cumsum(
            [len(row["rollout_state"]) for row in results], dtype=np.int64
        )
        rollout.create_dataset("data/state", data=states, chunks=(1024, states.shape[1]))
        rollout.create_dataset("data/action", data=actions, chunks=(1024, actions.shape[1]))
        rollout.create_dataset(
            "data/success",
            data=np.asarray(
                [row["success_stable_after_release"] for row in results], dtype=np.bool_
            ),
        )
        rollout.create_dataset("meta/episode_ends", data=episode_ends)
        rollout.create_dataset(
            "meta/reset_indices",
            data=np.asarray([row["reset_index"] for row in results], dtype=np.int64),
        )
        rollout.attrs.update({
            "task": "StackCube-MuJoCo-MatchedRuntime-State-v1",
            "reset_type": "ObjectAnywhereEEAnywhere",
            "seed": int(source.attrs.get("seed", 44)),
            "num_envs": 1,
            "num_episodes": n,
            "success_pos_threshold": float(Stack.SUCCESS_POS_THRESH),
            "success_ori_threshold": float(Stack.SUCCESS_ORI_XY_THRESH),
            "checkpoint": str(checkpoint_path),
            "eval_contract_id": "mujoco-stackcube-matched-runtime-stable-release-v1",
            "simulator": "MuJoCo",
            "metric": "stable_after_release_success",
            "runtime_source": str(runtime_path),
            "definition": (
                "base-policy closed-loop MuJoCo A rollouts for data selection; "
                "disjoint from the final evaluation reset set"
            ),
        })
        print(
            f"[ROLLOUTS] saved {rollout_path}: episodes={n} frames={len(states)} "
            f"successes={stable}"
        )
    print(f"[RESULT] stable={stable}/{n}={stable/n:.4f} proxy={proxy_reached}/{n}={proxy_reached/n:.4f}")


if __name__ == "__main__":
    main()
