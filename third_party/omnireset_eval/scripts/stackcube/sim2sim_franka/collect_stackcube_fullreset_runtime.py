#!/usr/bin/env python3
"""Collect exact full-reset StackCube MuJoCo model_31000 rollouts.

This is deliberately a different protocol from selected-chunk continuation.
Each source episode contributes only its reset identity and its rich runtime
parameters.  MuJoCo is initialized once from ``data/raw_state[episode_start]``;
the live RSL expert then produces 160 actions from live 200-D observations.
Afterward, 60 forced-open steps qualify success using the final-ten strict
stable predicate.  There are no perturbations or retries.

The output is an atomic shard Zarr containing every requested rollout,
including failures.  The companion aggregator validates exact coverage of all
272 source episodes and publishes a deterministic random sample of 100
successful full trajectories.
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
from typing import Any

os.environ.setdefault("MUJOCO_GL", "")

import mujoco
import numpy as np
import torch
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop_stackcube_eval as Stack
import diagnose_stackcube_top2_chunk_replay as Diagnose
import stackcube_offline_error_cotrain as Offline
from franka_policy import FrankaPolicy


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "datasets/stackcube_fresh_probe_20260725"
    / "isaac_a272_rich_exact.zarr"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT.parent
    / "co-curation/logs/rsl_rl/franka_fr3_gripper_omnireset_agent"
    / "2026-07-11_19-39-00_stackcube_stage2_b3_2500_4gpu_from7800"
    / "model_31000.pt"
)

SCHEMA_VERSION = 1
PROTOCOL_ID = "stackcube-a272-fullreset-model31000-strict-release10-v1"
EXPECTED_SOURCE_FINGERPRINT = (
    "e08879c38748196041b1b5b15b7434158826ffde44e41794d59d60f31ce9ef92"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "7ad91043afe2e57e834b6b67ac66e79cb0741342adeaf83b83611202b4202b63"
)
EXPECTED_EPISODES = 272
POLICY_STEPS = 160
RELEASE_STEPS = 60
STABLE_WINDOW = 10
STABLE_LINEAR_VELOCITY_MPS = 0.01
STABLE_ANGULAR_VELOCITY_RADPS = 0.2
PHYSICS_SUBSTEPS = 16
BASE_SEED = 42
JACOBIAN_POINT = "physx_com"
FINGER_VELOCITY_LIMITS = (0.05, 0.04)

RELEASE_METRICS = {
    "strict_pose": np.dtype("bool"),
    "center_lateral_m": np.dtype("float64"),
    "center_z_err_m": np.dtype("float64"),
    "align_euler_xy_rad": np.dtype("float64"),
    "no_robot_contact": np.dtype("bool"),
    "receptive_contact": np.dtype("bool"),
    "lin_vel_mps": np.dtype("float64"),
    "ang_vel_radps": np.dtype("float64"),
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        Offline.to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def raw_reset_sha256(value: np.ndarray) -> str:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (57,):
        raise ValueError(f"raw reset must have shape (57,), got {value.shape}")
    return array_sha256(value)


def controller_profile() -> dict[str, Any]:
    return Offline.make_controller_profile(
        JACOBIAN_POINT, FINGER_VELOCITY_LIMITS
    )


def success_definition() -> dict[str, Any]:
    return {
        "window": f"all final {STABLE_WINDOW} release policy samples",
        "strict_pose": {
            "center_lateral_m_lt": Stack.SUCCESS_POS_THRESH,
            "center_z_error_m_lt": Stack.SUCCESS_POS_THRESH,
            "abs_roll_plus_abs_pitch_rad_lt": Stack.SUCCESS_ORI_XY_THRESH,
        },
        "receptive_contact": True,
        "no_robot_contact": True,
        "upper_cube_linear_velocity_mps_lt": STABLE_LINEAR_VELOCITY_MPS,
        "upper_cube_angular_velocity_radps_lt": STABLE_ANGULAR_VELOCITY_RADPS,
    }


def validate_inputs(
    source_path: Path, checkpoint_path: Path
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not source_path.is_dir():
        raise FileNotFoundError(source_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA-256 mismatch: {checkpoint_hash} != "
            f"{EXPECTED_CHECKPOINT_SHA256}"
        )
    source = zarr.open(str(source_path), mode="r")
    starts, ends, reset_indices = Diagnose.validate_source(source)
    fingerprint, logical = Diagnose.source_fingerprint(source)
    if fingerprint != EXPECTED_SOURCE_FINGERPRINT:
        raise ValueError(
            f"source fingerprint mismatch: {fingerprint} != "
            f"{EXPECTED_SOURCE_FINGERPRINT}"
        )
    if (
        len(ends) != EXPECTED_EPISODES
        or not np.array_equal(
            ends - starts,
            np.full(EXPECTED_EPISODES, POLICY_STEPS, dtype=np.int64),
        )
        or int(ends[-1]) != EXPECTED_EPISODES * POLICY_STEPS
    ):
        raise ValueError("source must contain exactly 272 episodes of 160 rows")
    if len(np.unique(reset_indices)) != EXPECTED_EPISODES:
        raise ValueError("source reset IDs are not unique")
    declared_checkpoint = Path(str(source.attrs["checkpoint"])).resolve()
    if declared_checkpoint != checkpoint_path:
        raise ValueError(
            f"source checkpoint differs from requested checkpoint: "
            f"{declared_checkpoint} != {checkpoint_path}"
        )
    profile = controller_profile()
    provenance = {
        "source_zarr": str(source_path),
        "source_metadata_and_required_arrays_sha256": fingerprint,
        "source_required_array_logical_sha256": logical,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "controller_profile": profile,
        "controller_profile_sha256": canonical_json_sha256(profile),
    }
    return source, starts, ends, reset_indices, provenance


def stable_release(metric: dict[str, Any]) -> bool:
    return bool(
        metric["strict_pose"]
        and metric["no_robot_contact"]
        and metric["receptive_contact"]
        and metric["lin_vel_mps"] < STABLE_LINEAR_VELOCITY_MPS
        and metric["ang_vel_radps"] < STABLE_ANGULAR_VELOCITY_RADPS
    )


def policy_action(policy, observation: np.ndarray, device: str) -> np.ndarray:
    with torch.inference_mode():
        return (
            policy(
                torch.from_numpy(observation)
                .float()
                .unsqueeze(0)
                .to(device)
            )
            .cpu()
            .numpy()[0]
            .astype(np.float32)
        )


def run_episode(
    source,
    policy,
    *,
    episode: int,
    start: int,
    end: int,
    reset_index: int,
    device: str,
) -> dict[str, Any]:
    if end - start != POLICY_STEPS:
        raise ValueError(f"episode {episode} is not exactly 160 rows")
    policy_seed = BASE_SEED + int(reset_index)
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)
    if hasattr(policy, "reset"):
        policy.reset()

    raw = np.asarray(source["data/raw_state"][start], dtype=np.float64)
    model, controller, effective = Offline.build_episode_model(
        source,
        start,
        end,
        raw,
        physics_substeps=PHYSICS_SUBSTEPS,
        jacobian_point=JACOBIAN_POINT,
        finger_velocity_limits=FINGER_VELOCITY_LIMITS,
    )
    if effective["controller_profile"] != controller_profile():
        raise ValueError("episode model did not use the frozen controller profile")
    data = mujoco.MjData(model)
    Offline.set_boundary_state(model, data, controller, raw)
    builder = Stack.StackCubeObsBuilder(controller)
    builder.reset(data)
    previous_action = np.zeros(7, dtype=np.float32)
    observations = np.empty((POLICY_STEPS, 200), dtype=np.float32)
    actions = np.empty((POLICY_STEPS, 7), dtype=np.float32)
    grasp_close = np.empty(POLICY_STEPS, dtype=np.bool_)

    for step in range(POLICY_STEPS):
        observation = builder.step(data, previous_action)
        action = policy_action(policy, observation, device)
        if observation.shape != (200,) or action.shape != (7,):
            raise ValueError(
                f"episode {episode} invalid obs/action shape "
                f"{observation.shape}/{action.shape}"
            )
        if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(action)):
            raise ValueError(f"episode {episode} produced non-finite policy data")
        observations[step] = observation
        actions[step] = action
        grasp_close[step] = Offline.execute_action(
            model,
            data,
            controller,
            action,
            None,
            jacobian_point=JACOBIAN_POINT,
            finger_velocity_limits=FINGER_VELOCITY_LIMITS,
        )
        previous_action = action

    release = {
        name: np.empty(RELEASE_STEPS, dtype=dtype)
        for name, dtype in RELEASE_METRICS.items()
    }
    release_stable = np.empty(RELEASE_STEPS, dtype=np.bool_)
    zero = np.zeros(7, dtype=np.float64)
    for step in range(RELEASE_STEPS):
        Offline.execute_action(
            model,
            data,
            controller,
            zero,
            False,
            jacobian_point=JACOBIAN_POINT,
            finger_velocity_limits=FINGER_VELOCITY_LIMITS,
        )
        metric = Stack.stack_metrics(model, data, controller)
        for name in RELEASE_METRICS:
            release[name][step] = metric[name]
        release_stable[step] = stable_release(metric)
    success = bool(np.all(release_stable[-STABLE_WINDOW:]))
    return {
        "state": observations,
        "action": actions,
        "grasp_guard_close": grasp_close,
        "release": release,
        "release_stable": release_stable,
        "success": success,
        "policy_seed": policy_seed,
        "raw_reset_sha256": raw_reset_sha256(
            np.asarray(source["data/raw_state"][start], dtype=np.float32)
        ),
        "effective_config": effective,
    }


def write_shard_atomic(
    output: Path,
    *,
    source,
    starts: np.ndarray,
    reset_indices: np.ndarray,
    episode_indices: list[int],
    rows: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / (
        f".{output.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        root = zarr.open(str(staging), mode="w")
        state = np.concatenate([row["state"] for row in rows], axis=0)
        action = np.concatenate([row["action"] for row in rows], axis=0)
        close = np.concatenate([row["grasp_guard_close"] for row in rows], axis=0)
        root.create_dataset("data/state", data=state, chunks=(160, 200))
        root.create_dataset("data/action", data=action, chunks=(160, 7))
        root.create_dataset("data/grasp_guard_close", data=close, chunks=(160,))
        count = len(rows)
        root.create_dataset(
            "meta/episode_ends",
            data=np.arange(1, count + 1, dtype=np.int64) * POLICY_STEPS,
        )
        root.create_dataset(
            "meta/source_episode_index",
            data=np.asarray(episode_indices, dtype=np.int32),
        )
        root.create_dataset(
            "meta/source_reset_index",
            data=np.asarray(reset_indices[episode_indices], dtype=np.int32),
        )
        root.create_dataset(
            "meta/source_episode_start_row",
            data=np.asarray(starts[episode_indices], dtype=np.int64),
        )
        root.create_dataset(
            "meta/success_stable_after_release",
            data=np.asarray([row["success"] for row in rows], dtype=np.bool_),
        )
        root.create_dataset(
            "meta/attempt_index",
            data=np.zeros(count, dtype=np.int8),
        )
        root.create_dataset(
            "meta/attempts_executed",
            data=np.ones(count, dtype=np.int8),
        )
        root.create_dataset(
            "meta/exact_initial_reset",
            data=np.ones(count, dtype=np.bool_),
        )
        root.create_dataset(
            "meta/policy_seed",
            data=np.asarray([row["policy_seed"] for row in rows], dtype=np.int64),
        )
        root.create_dataset(
            "meta/raw_reset_sha256",
            data=np.asarray([row["raw_reset_sha256"] for row in rows], dtype="S64"),
        )
        root.create_dataset(
            "meta/effective_config_sha256",
            data=np.asarray(
                [
                    canonical_json_sha256(row["effective_config"])
                    for row in rows
                ],
                dtype="S64",
            ),
        )
        for name, dtype in RELEASE_METRICS.items():
            root.create_dataset(
                f"qualification/{name}",
                data=np.stack([row["release"][name] for row in rows]).astype(
                    dtype, copy=False
                ),
                chunks=(1, RELEASE_STEPS),
            )
        root.create_dataset(
            "qualification/stable",
            data=np.stack([row["release_stable"] for row in rows]),
            chunks=(1, RELEASE_STEPS),
        )
        support = {}
        for name, module in (
            ("closed_loop_stackcube_eval", Stack),
            ("diagnose_stackcube_top2_chunk_replay", Diagnose),
            ("stackcube_offline_error_cotrain", Offline),
        ):
            path = Path(module.__file__).resolve()
            support[name] = {"path": str(path), "sha256": file_sha256(path)}
        root.attrs.update(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "protocol_id": PROTOCOL_ID,
                "definition": (
                    "true full-reset MuJoCo model_31000 StackCube rollouts; "
                    "one raw_state[episode_start] restore, 160 live expert "
                    "steps, then release60 qualification"
                ),
                "source_zarr": provenance["source_zarr"],
                "source_metadata_and_required_arrays_sha256": provenance[
                    "source_metadata_and_required_arrays_sha256"
                ],
                "source_required_array_logical_sha256_json": canonical_json(
                    provenance["source_required_array_logical_sha256"]
                ),
                "checkpoint": provenance["checkpoint"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "collector_script": str(Path(__file__).resolve()),
                "collector_script_sha256": file_sha256(__file__),
                "support_json": canonical_json(support),
                "controller_profile_json": canonical_json(
                    provenance["controller_profile"]
                ),
                "controller_profile_sha256": provenance[
                    "controller_profile_sha256"
                ],
                "physics_substeps_per_isaac_tick": PHYSICS_SUBSTEPS,
                "policy_steps": POLICY_STEPS,
                "release_steps": RELEASE_STEPS,
                "stable_window": STABLE_WINDOW,
                "success_definition_json": canonical_json(success_definition()),
                "attempt_limit": 1,
                "perturbations": False,
                "retries": False,
                "initial_state_restore_count_per_episode": 1,
                "per_step_state_restore": False,
                "live_policy_inference": True,
                "pre_action_observation_capture": True,
                "release_rows_excluded_from_training": True,
                "base_seed": BASE_SEED,
                "episodes": count,
                "frames": int(len(state)),
                "source_episode_begin": min(episode_indices),
                "source_episode_end_exclusive": max(episode_indices) + 1,
                "state_sha256": array_sha256(state),
                "action_sha256": array_sha256(action),
                "grasp_guard_close_sha256": array_sha256(close),
                "effective_configs_json": canonical_json(
                    [row["effective_config"] for row in rows]
                ),
                "successes": int(sum(row["success"] for row in rows)),
                "versions_json": canonical_json(
                    {
                        "numpy": np.__version__,
                        "torch": torch.__version__,
                        "mujoco": mujoco.__version__,
                        "zarr": zarr.__version__,
                    }
                ),
            }
        )
        os.rename(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.out).resolve()
    source, starts, ends, reset_indices, provenance = validate_inputs(
        source_path, checkpoint_path
    )
    if args.start < 0 or args.count <= 0:
        raise ValueError("--start must be nonnegative and --count must be positive")
    stop = args.start + args.count
    if stop > EXPECTED_EPISODES:
        raise ValueError(
            f"requested episode range [{args.start},{stop}) exceeds "
            f"{EXPECTED_EPISODES}"
        )
    episodes = list(range(args.start, stop))
    policy = FrankaPolicy.load_from_checkpoint(
        str(checkpoint_path), device=args.device
    )
    rows = []
    for local, episode in enumerate(episodes):
        row = run_episode(
            source,
            policy,
            episode=episode,
            start=int(starts[episode]),
            end=int(ends[episode]),
            reset_index=int(reset_indices[episode]),
            device=args.device,
        )
        rows.append(row)
        print(
            f"[fullreset-expert] {local + 1}/{len(episodes)} "
            f"episode={episode:03d} reset={int(reset_indices[episode]):04d} "
            f"stable={int(row['success'])}",
            flush=True,
        )
    write_shard_atomic(
        output,
        source=source,
        starts=starts,
        reset_indices=reset_indices,
        episode_indices=episodes,
        rows=rows,
        provenance=provenance,
    )
    print(
        f"[RESULT] episodes={len(rows)} successes="
        f"{sum(row['success'] for row in rows)} frames={len(rows) * POLICY_STEPS} "
        f"output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
