#!/usr/bin/env python3
"""Rank StackCube sim2sim replay error with the Peg top2-chunk protocol.

The input is a rich episode-major IsaacSim Zarr.  For every processed episode:

1. enumerate full, non-overlapping 20-step windows;
2. restore MuJoCo exactly once at each window's recorded ``raw_state[t0]``;
3. replay only the recorded arm actions and processed gripper commands;
4. rank endpoint error as upper-cube position/0.01 m plus relative
   ``|roll| + |pitch|``/0.1 rad;
5. fully refine the two worst coarse windows into aligned 5-step children; and
6. retain the two worst fine children globally for that episode.

This script deliberately contains no policy inference and no expert correction.
Every replay candidate uses one boundary restore and never injects state during
the replay.  The output Zarr and JSON both retain the complete candidate audit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_peginsert_continuous_mujoco as Replay
import quat_utils as Q
import stackcube_offline_error_cotrain as Offline


SCHEMA_VERSION = 1
PROTOCOL_ID = "stackcube-rich-top2-coarse20-fine5-replay-v1"
COARSE_STEPS = 20
FINE_STEPS = 5
TOP_K = 2
POSITION_SCORE_THRESHOLD_M = 0.01
ROLL_PITCH_SCORE_THRESHOLD_RAD = 0.1
DEFAULT_PHYSICS_SUBSTEPS = 16

REQUIRED_ARRAYS = (
    "data/state",
    "data/action",
    "data/raw_state",
    "data/gripper_processed_action",
    "meta/episode_ends",
    "meta/reset_indices",
) + tuple(
    f"data/{key}" for key in Offline.RUNTIME_FIELDS + Offline.SCENE_FIELDS
)
REQUIRED_ATTRS = (
    "task",
    "checkpoint",
    "physics_dt_s",
    "decimation",
    "policy_dt_s",
    "robot_body_names",
)




def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value








def validate_source(store) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    missing_arrays = sorted(path for path in REQUIRED_ARRAYS if path not in store)
    if missing_arrays:
        raise KeyError(f"rich source Zarr is missing arrays: {missing_arrays}")
    missing_attrs = sorted(key for key in REQUIRED_ATTRS if key not in store.attrs)
    if missing_attrs:
        raise KeyError(f"rich source Zarr is missing attrs: {missing_attrs}")

    expected_shapes = {
        "data/state": 200,
        "data/action": 7,
        "data/raw_state": 57,
        "data/gripper_processed_action": 2,
    }
    for path, width in expected_shapes.items():
        array = store[path]
        if array.ndim != 2 or int(array.shape[1]) != width:
            raise ValueError(f"{path} must have shape [N,{width}], got {array.shape}")

    frame_count = int(store["data/raw_state"].shape[0])
    for path in REQUIRED_ARRAYS:
        if path.startswith("data/") and int(store[path].shape[0]) != frame_count:
            raise ValueError(
                f"{path} has {store[path].shape[0]} rows, expected {frame_count}"
            )

    starts, ends = Offline.episode_ranges(store)
    reset_indices = np.asarray(store["meta/reset_indices"], dtype=np.int64)
    if reset_indices.shape != ends.shape:
        raise ValueError(
            "meta/reset_indices must contain exactly one ID per episode: "
            f"{reset_indices.shape} != {ends.shape}"
        )
    if int(ends[-1]) != frame_count:
        raise ValueError(
            f"final episode end {int(ends[-1])} != frame count {frame_count}"
        )
    if len(np.unique(reset_indices)) != len(reset_indices):
        raise ValueError("meta/reset_indices contains duplicate reset IDs")

    physics_dt = float(store.attrs["physics_dt_s"])
    decimation = int(store.attrs["decimation"])
    policy_dt = float(store.attrs["policy_dt_s"])
    if physics_dt <= 0.0 or decimation <= 0 or policy_dt <= 0.0:
        raise ValueError("physics_dt_s, decimation, and policy_dt_s must be positive")
    if abs(physics_dt * decimation - policy_dt) > 1.0e-9:
        raise ValueError("source physics_dt_s * decimation != policy_dt_s")
    robot_body_names = list(store.attrs["robot_body_names"])
    if not robot_body_names:
        raise ValueError("robot_body_names must not be empty")
    if not str(store.attrs["task"]):
        raise ValueError("source task attr must not be empty")
    if not str(store.attrs["checkpoint"]):
        raise ValueError("source checkpoint attr must not be empty")
    return starts, ends, reset_indices


def relative_roll_pitch_error(
    predicted_pose: np.ndarray, reference_pose: np.ndarray
) -> float:
    _, relative_quat = Q.subtract_frame_transforms(
        reference_pose[:3],
        reference_pose[3:7],
        predicted_pose[:3],
        predicted_pose[3:7],
    )
    return Replay.roll_pitch_error(relative_quat)


def replay_candidate(
    store,
    model,
    ctrl,
    *,
    episode_index: int,
    reset_index: int,
    episode_start: int,
    episode_end: int,
    source_step: int,
    horizon: int,
    level: str,
    parent_rank: int = -1,
    parent_source_step: int = -1,
) -> dict[str, Any]:
    source_row = episode_start + source_step
    endpoint_row = source_row + horizon
    if horizon <= 0:
        raise ValueError("candidate horizon must be positive")
    if endpoint_row >= episode_end:
        raise ValueError(
            "candidate endpoint raw state does not exist: "
            f"episode={episode_index} t0={source_step} h={horizon} "
            f"endpoint_row={endpoint_row} episode_end={episode_end}"
        )

    data = mujoco.MjData(model)
    # This is the candidate's only state restore.
    Offline.set_boundary_state(
        model, data, ctrl, np.asarray(store["data/raw_state"][source_row])
    )
    close_steps = 0
    for offset in range(horizon):
        close = Offline.source_gripper_close(store, source_row + offset)
        close_steps += int(close)
        Offline.execute_action(
            model,
            data,
            ctrl,
            np.asarray(store["data/action"][source_row + offset]),
            close,
        )

    predicted = Offline.simulated_pose(data, ctrl).astype(np.float64)
    reference = Offline.reference_pose(
        np.asarray(store["data/raw_state"][endpoint_row])
    ).astype(np.float64)
    position_error = float(np.linalg.norm(predicted[:3] - reference[:3]))
    roll_pitch_error = relative_roll_pitch_error(predicted, reference)
    arm_joint_l2 = float(
        np.linalg.norm(
            data.qpos[:7]
            - np.asarray(store["data/raw_state"][endpoint_row, :7], dtype=np.float64)
        )
    )
    q9_joint_l2 = float(
        np.linalg.norm(
            data.qpos[:9]
            - np.asarray(store["data/raw_state"][endpoint_row, :9], dtype=np.float64)
        )
    )
    score = (
        position_error / POSITION_SCORE_THRESHOLD_M
        + roll_pitch_error / ROLL_PITCH_SCORE_THRESHOLD_RAD
    )
    values = np.concatenate(
        [
            predicted,
            reference,
            np.asarray(
                [position_error, roll_pitch_error, arm_joint_l2, q9_joint_l2, score]
            ),
        ]
    )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            f"non-finite replay result at episode={episode_index}, t0={source_step}"
        )
    return {
        "level": level,
        "episode_index": int(episode_index),
        "reset_index": int(reset_index),
        "episode_start_row": int(episode_start),
        "episode_end_row_exclusive": int(episode_end),
        "source_row": int(source_row),
        "source_step": int(source_step),
        "endpoint_source_row": int(endpoint_row),
        "horizon_steps": int(horizon),
        "processed_gripper_close_steps": int(close_steps),
        "state_restore_count": 1,
        "per_step_state_restore": False,
        "policy_inference_calls": 0,
        "upper_cube_position_error_m": position_error,
        "upper_cube_roll_pitch_error_rad": roll_pitch_error,
        "arm_joint_position_l2_rad": arm_joint_l2,
        "arm_and_finger_position_l2_mixed_units": q9_joint_l2,
        "score": float(score),
        "predicted_upper_cube_pose_robot_root": predicted.astype(np.float32),
        "reference_upper_cube_pose_robot_root": reference.astype(np.float32),
        "parent_critical_coarse_rank": int(parent_rank),
        "parent_coarse_source_step": int(parent_source_step),
    }


def ranking_key(record: dict[str, Any]) -> tuple[float, int, int]:
    return (
        -float(record["score"]),
        int(record["source_step"]),
        int(record["horizon_steps"]),
    )


def diagnose_episode(
    store,
    *,
    episode_index: int,
    reset_index: int,
    episode_start: int,
    episode_end: int,
    physics_substeps: int,
    jacobian_point: str,
    finger_velocity_limits: tuple[float, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode_rows = episode_end - episode_start
    available_transitions = episode_rows - 1
    coarse_starts = list(
        range(0, available_transitions - COARSE_STEPS + 1, COARSE_STEPS)
    )
    if len(coarse_starts) < TOP_K:
        raise ValueError(
            f"episode {episode_index} has {episode_rows} rows, but at least "
            f"{TOP_K * COARSE_STEPS + 1} are required for two full H20 windows"
        )

    model, ctrl, effective = Offline.build_episode_model(
        store,
        episode_start,
        episode_end,
        np.asarray(store["data/raw_state"][episode_start]),
        physics_substeps=physics_substeps,
        jacobian_point=jacobian_point,
        finger_velocity_limits=finger_velocity_limits,
    )
    coarse = [
        replay_candidate(
            store,
            model,
            ctrl,
            episode_index=episode_index,
            reset_index=reset_index,
            episode_start=episode_start,
            episode_end=episode_end,
            source_step=t0,
            horizon=COARSE_STEPS,
            level="coarse",
        )
        for t0 in coarse_starts
    ]
    critical_coarse = sorted(coarse, key=ranking_key)[:TOP_K]

    fine: list[dict[str, Any]] = []
    for parent_rank, parent in enumerate(critical_coarse):
        parent_t0 = int(parent["source_step"])
        for t0 in range(parent_t0, parent_t0 + COARSE_STEPS, FINE_STEPS):
            fine.append(
                replay_candidate(
                    store,
                    model,
                    ctrl,
                    episode_index=episode_index,
                    reset_index=reset_index,
                    episode_start=episode_start,
                    episode_end=episode_end,
                    source_step=t0,
                    horizon=FINE_STEPS,
                    level="fine",
                    parent_rank=parent_rank,
                    parent_source_step=parent_t0,
                )
            )
    fine = sorted(fine, key=ranking_key)
    selected = fine[:TOP_K]
    if len(selected) != TOP_K or len(
        {int(record["source_step"]) for record in selected}
    ) != TOP_K:
        raise AssertionError(
            f"episode {episode_index} did not produce two unique selected chunks"
        )
    selected_records = []
    for rank, record in enumerate(selected):
        copied = dict(record)
        copied["selected_rank"] = rank
        selected_records.append(copied)

    audit = {
        "episode_index": int(episode_index),
        "reset_index": int(reset_index),
        "episode_start_row": int(episode_start),
        "episode_end_row_exclusive": int(episode_end),
        "episode_rows": int(episode_rows),
        "available_transitions_with_endpoint": int(available_transitions),
        "coarse": coarse,
        "critical_coarse": critical_coarse,
        "fine_candidates": fine,
        "selected_top2": selected_records,
        "state_restore_count": len(coarse) + len(fine),
        "per_step_state_restore": False,
        "policy_inference_calls": 0,
    }
    return audit, effective


def support_provenance() -> dict[str, dict[str, Any]]:
    modules = {
        "diagnostic_script": Path(__file__).resolve(),
        "stackcube_offline_error_cotrain": Path(Offline.__file__).resolve(),
        "compare_peginsert_continuous_mujoco": Path(Replay.__file__).resolve(),
        "quat_utils": Path(Q.__file__).resolve(),
        "closed_loop_stackcube_eval": Path(Offline.Stack.__file__).resolve(),
        "compare_stackcube_onestep_mujoco": Path(
            Offline.StackModel.__file__
        ).resolve(),
    }
    return {
        name: {"path": str(path)}
        for name, path in modules.items()
    }


def external_file_provenance(store, attr_name: str) -> dict[str, Any]:
    raw_path = str(store.attrs.get(attr_name, ""))
    record: dict[str, Any] = {"path": raw_path}
    if raw_path:
        path = Path(raw_path).expanduser()
        record["exists"] = path.is_file()
        if path.is_file():
            record["size_bytes"] = path.stat().st_size
    return record


SCALAR_FIELDS = (
    "episode_index",
    "reset_index",
    "episode_start_row",
    "episode_end_row_exclusive",
    "source_row",
    "source_step",
    "endpoint_source_row",
    "horizon_steps",
    "processed_gripper_close_steps",
    "state_restore_count",
    "policy_inference_calls",
    "upper_cube_position_error_m",
    "upper_cube_roll_pitch_error_rad",
    "arm_joint_position_l2_rad",
    "arm_and_finger_position_l2_mixed_units",
    "score",
    "parent_critical_coarse_rank",
    "parent_coarse_source_step",
)


def flatten_episode_records(
    audits: Iterable[dict[str, Any]], key: str
) -> tuple[list[dict[str, Any]], np.ndarray]:
    records: list[dict[str, Any]] = []
    ends = []
    for audit in audits:
        records.extend(audit[key])
        ends.append(len(records))
    return records, np.asarray(ends, dtype=np.int64)


def write_record_group(group, records: list[dict[str, Any]], episode_ends) -> None:
    if not records:
        raise ValueError("cannot write an empty candidate record group")
    integer_fields = {
        "episode_index",
        "reset_index",
        "episode_start_row",
        "episode_end_row_exclusive",
        "source_row",
        "source_step",
        "endpoint_source_row",
        "horizon_steps",
        "processed_gripper_close_steps",
        "state_restore_count",
        "policy_inference_calls",
        "parent_critical_coarse_rank",
        "parent_coarse_source_step",
    }
    for field in SCALAR_FIELDS:
        dtype = np.int64 if field in integer_fields else np.float64
        group.create_dataset(
            field,
            data=np.asarray([record[field] for record in records], dtype=dtype),
        )
    if "selected_rank" in records[0]:
        group.create_dataset(
            "selected_rank",
            data=np.asarray([record["selected_rank"] for record in records], np.int64),
        )
    group.create_dataset(
        "predicted_upper_cube_pose_robot_root",
        data=np.stack(
            [record["predicted_upper_cube_pose_robot_root"] for record in records]
        ).astype(np.float32),
    )
    group.create_dataset(
        "reference_upper_cube_pose_robot_root",
        data=np.stack(
            [record["reference_upper_cube_pose_robot_root"] for record in records]
        ).astype(np.float32),
    )
    group.create_dataset("episode_ends", data=episode_ends)


def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def output_json_path(output_zarr: Path) -> Path:
    if output_zarr.suffix == ".zarr":
        return output_zarr.with_suffix(".json")
    return output_zarr.with_name(f"{output_zarr.name}.json")


def write_outputs(
    args,
    store,
    audits: list[dict[str, Any]],
    effective_configs: list[dict[str, Any]],
) -> tuple[Path, Path, dict[str, Any]]:
    output = Path(args.out).resolve()
    manifest_path = output_json_path(output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output}")
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output manifest exists (pass --overwrite): {manifest_path}"
        )
    if output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    if manifest_path.exists():
        manifest_path.unlink()
    output.parent.mkdir(parents=True, exist_ok=True)

    coarse, coarse_ends = flatten_episode_records(audits, "coarse")
    fine, fine_ends = flatten_episode_records(audits, "fine_candidates")
    selected, selected_ends = flatten_episode_records(audits, "selected_top2")
    total_restores = sum(int(audit["state_restore_count"]) for audit in audits)
    expected_restores = len(coarse) + len(fine)
    if total_restores != expected_restores:
        raise AssertionError(
            f"restore audit mismatch: {total_restores} != {expected_restores}"
        )

    source_path = Path(args.source).resolve()
    controller_profile = Offline.make_controller_profile(
        args.jacobian_point, args.finger_velocity_limits
    )
    source_provenance = {
        "path": str(source_path),


        "root_attrs": dict(store.attrs),
        "checkpoint": external_file_provenance(store, "checkpoint"),
        "reset_state": external_file_provenance(store, "reset_state"),
    }
    protocol = {
        "protocol_id": PROTOCOL_ID,
        "coarse_horizon_steps": COARSE_STEPS,
        "coarse_stride_steps": COARSE_STEPS,
        "coarse_windows": (
            "all full non-overlapping H20 windows whose t0+h raw endpoint exists"
        ),
        "critical_coarse_top_k_per_episode": TOP_K,
        "fine_horizon_steps": FINE_STEPS,
        "fine_refinement": (
            "all four aligned H5 children inside each of the two critical "
            "coarse H20 parents"
        ),
        "selected_fine_top_k_per_episode": TOP_K,
        "ranking": (
            "descending endpoint upper-cube position error/0.01m + relative "
            "official |roll|+|pitch| error/0.1rad; ties by ascending t0, length"
        ),
        "position_score_threshold_m": POSITION_SCORE_THRESHOLD_M,
        "roll_pitch_score_threshold_rad": ROLL_PITCH_SCORE_THRESHOLD_RAD,
        "recorded_replay": (
            "recorded Isaac arm action plus recorded processed gripper command"
        ),
        "state_restore_per_candidate": 1,
        "per_step_state_restore": False,
        "policy_inference_calls": 0,
        "correction_or_expert_collection": False,
        "mujoco_physics_substeps_per_isaac_tick": int(args.physics_substeps),
        "controller_profile": controller_profile,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "smoke_only": bool(
            args.max_episodes > 0 or store.attrs.get("smoke_only", False)
        ),
        "source": source_provenance,
        "support_code": support_provenance(),
        "protocol": protocol,
        "controller_profile": controller_profile,
        "selection": {
            "start_episode": args.start_episode,
            "max_episodes": args.max_episodes,
            "episode_indices": [audit["episode_index"] for audit in audits],
            "reset_indices": [audit["reset_index"] for audit in audits],
        },
        "counts": {
            "episodes": len(audits),
            "coarse_candidates": len(coarse),
            "fine_candidates": len(fine),
            "selected_chunks": len(selected),
            "state_restores": total_restores,
            "expected_state_restores": expected_restores,
            "policy_inference_calls": 0,
        },
        "score": {
            "coarse_mean": float(np.mean([record["score"] for record in coarse])),
            "coarse_max": float(np.max([record["score"] for record in coarse])),
            "fine_mean": float(np.mean([record["score"] for record in fine])),
            "fine_max": float(np.max([record["score"] for record in fine])),
            "selected_mean": float(
                np.mean([record["score"] for record in selected])
            ),
            "selected_max": float(np.max([record["score"] for record in selected])),
        },
        "effective_config_per_episode": effective_configs,
        "episodes": audits,
        "output_zarr": str(output),
        "output_json": str(manifest_path),
    }

    staging = output.with_name(
        f".{output.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    try:
        root = zarr.open(str(staging), mode="w")
        root.attrs.update(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "protocol_id": PROTOCOL_ID,
                "source_zarr": str(source_path),

                "source_task": str(store.attrs["task"]),
                "source_checkpoint": str(store.attrs["checkpoint"]),
                "start_episode": int(args.start_episode),
                "max_episodes": int(args.max_episodes),
                "processed_episode_count": len(audits),
                "position_score_threshold_m": POSITION_SCORE_THRESHOLD_M,
                "roll_pitch_score_threshold_rad": (
                    ROLL_PITCH_SCORE_THRESHOLD_RAD
                ),
                "coarse_horizon_steps": COARSE_STEPS,
                "fine_horizon_steps": FINE_STEPS,
                "top_k_per_episode": TOP_K,
                "mujoco_physics_substeps_per_isaac_tick": int(
                    args.physics_substeps
                ),
                "jacobian_point": controller_profile["jacobian_point"],
                "finger_velocity_limits_mps": controller_profile[
                    "finger_velocity_limits_mps"
                ],
                "controller_profile_json": json.dumps(
                    controller_profile, sort_keys=True
                ),
                "recorded_replay_gripper": (
                    "data/gripper_processed_action; no live MuJoCo guard"
                ),
                "replay_state_restore_count": total_restores,
                "replay_state_restore_per_candidate": 1,
                "replay_per_step_state_restore": False,
                "replay_policy_inference_calls": 0,
                "smoke_only": summary["smoke_only"],
                "protocol_json": json.dumps(protocol, sort_keys=True),
                "source_provenance_json": json.dumps(
                    jsonable(source_provenance), sort_keys=True
                ),
                "support_code_json": json.dumps(
                    jsonable(summary["support_code"]), sort_keys=True
                ),
            }
        )
        meta = root.create_group("meta")
        meta.create_dataset(
            "processed_episode_indices",
            data=np.asarray(
                [audit["episode_index"] for audit in audits], dtype=np.int64
            ),
        )
        meta.create_dataset(
            "reset_indices",
            data=np.asarray([audit["reset_index"] for audit in audits], np.int64),
        )
        meta.create_dataset(
            "source_episode_starts",
            data=np.asarray(
                [audit["episode_start_row"] for audit in audits], dtype=np.int64
            ),
        )
        meta.create_dataset(
            "source_episode_ends",
            data=np.asarray(
                [audit["episode_end_row_exclusive"] for audit in audits],
                dtype=np.int64,
            ),
        )
        write_record_group(root.create_group("coarse"), coarse, coarse_ends)
        write_record_group(root.create_group("fine"), fine, fine_ends)
        write_record_group(
            root.create_group("selected_top2"), selected, selected_ends
        )
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    atomic_json_dump(manifest_path, summary)
    return output, manifest_path, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Peg-style top2 H20->H5 replay diagnosis for rich StackCube Zarr"
    )
    parser.add_argument("--source", required=True, help="rich Isaac StackCube Zarr")
    parser.add_argument("--out", required=True, help="output diagnostic Zarr")
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="zero-based source episode index to begin at",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="process at most this many episodes; 0 means all remaining episodes",
    )
    parser.add_argument(
        "--physics_substeps",
        type=int,
        default=DEFAULT_PHYSICS_SUBSTEPS,
        help=(
            "MuJoCo integration substeps per 1/120 s Isaac physics tick "
            f"(default: {DEFAULT_PHYSICS_SUBSTEPS})"
        ),
    )
    parser.add_argument(
        "--jacobian_point",
        choices=("link_origin", "physx_com"),
        default=Offline.DEFAULT_JACOBIAN_POINT,
        help="EE Jacobian point used by every replay action",
    )
    parser.add_argument(
        "--finger_velocity_limits",
        type=float,
        nargs=2,
        metavar=("FINGER1_MPS", "FINGER2_MPS"),
        default=Offline.DEFAULT_FINGER_VELOCITY_LIMITS,
        help="per-finger actuator velocity limits in m/s",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_episode < 0:
        raise ValueError("--start_episode must be non-negative")
    if args.max_episodes < 0:
        raise ValueError("--max_episodes must be non-negative")
    if args.physics_substeps <= 0:
        raise ValueError("--physics_substeps must be positive")
    controller_profile = Offline.make_controller_profile(
        args.jacobian_point, args.finger_velocity_limits
    )
    source_path = Path(args.source).resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"source Zarr does not exist: {source_path}")
    store = zarr.open(str(source_path), mode="r")
    starts, ends, reset_indices = validate_source(store)
    if args.start_episode >= len(ends):
        raise ValueError(
            f"--start_episode {args.start_episode} is outside {len(ends)} episodes"
        )
    stop_episode = len(ends)
    if args.max_episodes:
        stop_episode = min(stop_episode, args.start_episode + args.max_episodes)
    episode_indices = range(args.start_episode, stop_episode)

    print(
        f"[validate] source={source_path} episodes={len(ends)} "
        f"processing=[{args.start_episode},{stop_episode}) "
        f"controller={json.dumps(controller_profile, sort_keys=True)}",
        flush=True,
    )

    audits: list[dict[str, Any]] = []
    effective_configs: list[dict[str, Any]] = []
    for progress, episode_index in enumerate(episode_indices, start=1):
        audit, effective = diagnose_episode(
            store,
            episode_index=episode_index,
            reset_index=int(reset_indices[episode_index]),
            episode_start=int(starts[episode_index]),
            episode_end=int(ends[episode_index]),
            physics_substeps=args.physics_substeps,
            jacobian_point=args.jacobian_point,
            finger_velocity_limits=tuple(args.finger_velocity_limits),
        )
        audits.append(audit)
        effective_configs.append(jsonable(effective))
        selected = audit["selected_top2"]
        print(
            f"[diagnose] {progress}/{stop_episode - args.start_episode} "
            f"episode={episode_index} reset={int(reset_indices[episode_index])} "
            f"selected=t{selected[0]['source_step']}:{selected[0]['score']:.6f},"
            f"t{selected[1]['source_step']}:{selected[1]['score']:.6f} "
            f"restores={audit['state_restore_count']}",
            flush=True,
        )

    output, manifest, summary = write_outputs(
        args,
        store,
        audits,
        effective_configs,
    )
    result = {
        "output_zarr": str(output),
        "output_json": str(manifest),
        "episodes": summary["counts"]["episodes"],
        "coarse_candidates": summary["counts"]["coarse_candidates"],
        "fine_candidates": summary["counts"]["fine_candidates"],
        "selected_chunks": summary["counts"]["selected_chunks"],
        "state_restores": summary["counts"]["state_restores"],
        "policy_inference_calls": 0,
        "selected_score_max": summary["score"]["selected_max"],
        "smoke_only": summary["smoke_only"],
    }
    print("[RESULT] " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
