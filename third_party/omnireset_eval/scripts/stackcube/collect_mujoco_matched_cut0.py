#!/usr/bin/env python3
"""Collect StackCube MuJoCo teacher trajectories with Isaac-10k cut0 semantics.

Each candidate carries an Isaac reset and its per-episode dynamics/controller
parameters.  MuJoCo is restored once, the same model_31000 deterministic mean
teacher runs closed-loop on live 200-D MuJoCo observations, and collection
stops immediately after the first official pose success.  Thus the final
stored action is exactly the action that creates the first successful state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
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
from franka_policy import FrankaPolicy


POLICY_STEPS = 160
PHYSICS_SUBSTEPS = 16
JACOBIAN_POINT = "physx_com"
FINGER_VELOCITY_LIMITS = (0.05, 0.04)
PROTOCOL_ID = "stackcube-mujoco-matched-isaac10k-cut0-v2-controller-events"
FULL160_PROTOCOL_ID = "stackcube-mujoco-matched-full160-v1-controller-events"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate(source_path: Path, checkpoint_path: Path):
    if not source_path.is_dir():
        raise FileNotFoundError(source_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    source = zarr.open(str(source_path), mode="r")
    starts, ends, reset_ids = Diagnose.validate_source(source)
    lengths = ends - starts
    if len(ends) == 0 or np.any(lengths != POLICY_STEPS):
        raise ValueError("runtime source must contain fixed 160-row rich episodes")
    if len(np.unique(reset_ids)) != len(reset_ids):
        raise ValueError("runtime source reset IDs are not unique")
    required = (
        "data/raw_state", "data/state", "data/action",
        "data/arm_kp", "data/arm_kd", "data/arm_scale",
        "data/gripper_joint_stiffness", "data/gripper_joint_damping",
        "data/gripper_joint_effort_limit",
        "data/robot_body_masses", "data/robot_material_properties",
        "data/insertive_object_body_masses",
        "data/insertive_object_material_properties",
        "data/receptive_object_body_masses",
        "data/receptive_object_material_properties",
        "data/table_body_masses", "data/table_material_properties",
    )
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"runtime source lacks matched dynamics fields: {missing}")
    if source.attrs.get("preserve_controller_events") is not True:
        raise ValueError(
            "strict Isaac-10k matching requires preserve_controller_events=True"
        )
    declared = Path(str(source.attrs.get("checkpoint", ""))).resolve()
    declared_hash = source.attrs.get("checkpoint_sha256")
    if declared_hash is None and declared.is_file():
        declared_hash = sha256_file(declared)
    requested_hash = sha256_file(checkpoint_path)
    if declared_hash is not None:
        if str(declared_hash) != requested_hash:
            raise ValueError(
                f"teacher hash mismatch: source={declared_hash} requested={requested_hash}"
            )
    elif declared != checkpoint_path:
        raise ValueError(f"teacher mismatch without hash evidence: source={declared} requested={checkpoint_path}")
    fingerprint, logical = Diagnose.source_fingerprint(source)
    return source, starts, ends, reset_ids, fingerprint, logical


def run_one(source, policy, episode: int, start: int, end: int, device: str,
            full_horizon: bool = False):
    raw = np.asarray(source["data/raw_state"][start], dtype=np.float64)
    model, controller, effective = Offline.build_episode_model(
        source, start, end, raw,
        physics_substeps=PHYSICS_SUBSTEPS,
        jacobian_point=JACOBIAN_POINT,
        finger_velocity_limits=FINGER_VELOCITY_LIMITS,
    )
    data = mujoco.MjData(model)
    Offline.set_boundary_state(model, data, controller, raw)
    builder = Stack.StackCubeObsBuilder(controller)
    builder.reset(data)
    previous_action = np.zeros(7, dtype=np.float32)
    states, actions, action_stds = [], [], []
    success_metric = None

    if hasattr(policy, "reset"):
        policy.reset()
    for step in range(POLICY_STEPS):
        observation = builder.step(data, previous_action).astype(np.float32)
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
        with torch.inference_mode():
            action, action_std = policy.gsde_mean_std(obs_tensor)
        action_np = action.cpu().numpy()[0].astype(np.float32)
        std_np = action_std.cpu().numpy()[0].astype(np.float32)
        if observation.shape != (200,) or action_np.shape != (7,):
            raise ValueError(f"episode {episode}: bad shapes {observation.shape}/{action_np.shape}")
        if not (np.isfinite(observation).all() and np.isfinite(action_np).all() and np.isfinite(std_np).all()):
            raise ValueError(f"episode {episode}: non-finite policy row")
        states.append(observation)
        actions.append(action_np)
        action_stds.append(std_np)
        Offline.execute_action(
            model, data, controller, action_np, None,
            jacobian_point=JACOBIAN_POINT,
            finger_velocity_limits=FINGER_VELOCITY_LIMITS,
        )
        previous_action = action_np
        metric = Stack.stack_metrics(model, data, controller)
        if metric["proxy_pose"] and success_metric is None:
            success_metric = metric
            first_success_step = len(states)
            if not full_horizon:
                break

    return {
        "success": success_metric is not None,
        "state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "action_std": np.asarray(action_stds, dtype=np.float32),
        "first_success_step": first_success_step if success_metric is not None else -1,
        "success_metric": success_metric,
        "effective_config": effective,
    }


def write_atomic(out: Path, rows: list[dict], attempted: list[int], reset_ids: np.ndarray,
                 source_path: Path, checkpoint_path: Path, fingerprint: str, logical: dict,
                 full_horizon: bool = False):
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / f".{out.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    successful = [(ep, row) for ep, row in zip(attempted, rows, strict=True) if row["success"]]
    try:
        root = zarr.open(str(staging), mode="w")
        if successful:
            state = np.concatenate([row["state"] for _, row in successful])
            action = np.concatenate([row["action"] for _, row in successful])
            action_std = np.concatenate([row["action_std"] for _, row in successful])
            lengths = np.asarray([len(row["state"]) for _, row in successful], dtype=np.int64)
        else:
            state = np.empty((0, 200), dtype=np.float32)
            action = action_std = np.empty((0, 7), dtype=np.float32)
            lengths = np.empty(0, dtype=np.int64)
        root.create_dataset("data/state", data=state, chunks=(max(1, min(1024, len(state))), 200))
        root.create_dataset("data/action", data=action, chunks=(max(1, min(1024, len(action))), 7))
        root.create_dataset("data/action_std", data=action_std, chunks=(max(1, min(1024, len(action_std))), 7))
        root.create_dataset("meta/episode_ends", data=np.cumsum(lengths, dtype=np.int64))
        kept_eps = np.asarray([ep for ep, _ in successful], dtype=np.int32)
        root.create_dataset("meta/source_episode_index", data=kept_eps)
        root.create_dataset("meta/reset_indices", data=reset_ids[kept_eps].astype(np.int32))
        root.create_dataset(
            "meta/first_success_step",
            data=np.asarray([row["first_success_step"] for _, row in successful], dtype=np.int32),
        )
        root.create_dataset("meta/success", data=np.ones(len(successful), dtype=np.bool_))
        root.create_dataset("meta/attempted_source_episode_index", data=np.asarray(attempted, dtype=np.int32))
        root.attrs.update({
            "schema_version": 1,
            "protocol_id": FULL160_PROTOCOL_ID if full_horizon else PROTOCOL_ID,
            "task": "StackCube",
            "simulator": "MuJoCo",
            "definition": (
                "full-reset live model_31000; Isaac-matched reset/controller/scene runtime; "
                "official pose success; retain the complete 160-step rollout"
                if full_horizon else
                "full-reset live model_31000; Isaac-matched reset/controller/scene runtime; "
                "official pose success; stop after first success-causing action"
            ),
            "source_zarr": str(source_path),
            "source_fingerprint_sha256": fingerprint,
            "source_required_array_hashes_json": json.dumps(logical, sort_keys=True),
            "teacher_checkpoint": str(checkpoint_path),
            "teacher_checkpoint_sha256": sha256_file(checkpoint_path),
            "observation_dim": 200,
            "action_dim": 7,
            "policy_horizon": POLICY_STEPS,
            "policy_dt_s": 0.1,
            "physics_substeps_per_policy_step": PHYSICS_SUBSTEPS,
            "success_position_threshold_m": Stack.SUCCESS_POS_THRESH,
            "success_orientation_xy_threshold_rad": Stack.SUCCESS_ORI_XY_THRESH,
            "success_semantics": (
                "keep a complete 160-step rollout iff official proxy pose is reached at least once"
                if full_horizon else
                "first post-action official proxy pose; retain pre-action rows through causal action; post_success_rows=0"
            ),
            "post_success_rows": "through_step_160" if full_horizon else 0,
            "release_rows": 0,
            "episodes": len(successful),
            "frames": len(state),
            "attempts": len(attempted),
            "collector_script": str(Path(__file__).resolve()),
            "collector_script_sha256": sha256_file(Path(__file__).resolve()),
        })
        os.replace(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--full_horizon", action="store_true",
        help="always execute/store all 160 rows; success only filters which demos are retained",
    )
    args = parser.parse_args()
    source_path, checkpoint_path = Path(args.source).resolve(), Path(args.checkpoint).resolve()
    source, starts, ends, reset_ids, fingerprint, logical = validate(source_path, checkpoint_path)
    stop = args.start + args.count
    if args.start < 0 or args.count <= 0 or stop > len(ends):
        raise ValueError(f"invalid range [{args.start},{stop}) for {len(ends)} source episodes")
    policy = FrankaPolicy.load_from_checkpoint(str(checkpoint_path), device=args.device)
    if policy.gsde_log_std.numel() == 0:
        raise RuntimeError("teacher checkpoint lacks gSDE log_std required for action_std parity")
    attempted, rows = [], []
    for episode in range(args.start, stop):
        row = run_one(
            source, policy, episode, int(starts[episode]), int(ends[episode]),
            args.device, full_horizon=args.full_horizon,
        )
        attempted.append(episode)
        rows.append(row)
        print(f"[matched-cut0] ep={episode:04d} reset={int(reset_ids[episode]):04d} success={int(row['success'])} steps={len(row['state'])}", flush=True)
    write_atomic(Path(args.out).resolve(), rows, attempted, reset_ids, source_path,
                 checkpoint_path, fingerprint, logical, full_horizon=args.full_horizon)
    print(f"[RESULT] attempts={len(rows)} successes={sum(r['success'] for r in rows)} output={Path(args.out).resolve()}", flush=True)


if __name__ == "__main__":
    main()
