"""Reset-at-chunk action replay for the formal fresh B3 paired source.

Each chunk is an independent diagnostic:

1. restore MuJoCo from the recorded Isaac state at the chunk boundary;
2. replay only that chunk's recorded raw policy actions;
3. compare the MuJoCo terminal state with the recorded Isaac state at the
   same boundary.

The physical model and action executor are imported from
``closed_loop_b3_clean_eval.py``.  Apart from the intended chunk reset and
recorded-action replay, this script uses the exact formal MuJoCo evaluator
settings.  It does not modify the source zarr or any training dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "")

import mujoco
import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop_b3_clean_eval as B3
import closed_loop_eval as CL
import compare_peginsert_continuous_mujoco as Replay
import quat_utils as Q


DEFAULT_ZARR = (
    "log/active/peg_b3_direct_sim2sim_20260717/"
    "isaac_b3_center_fresh_unfiltered_seed42_n192_env16_20260724.zarr"
)
DEFAULT_OUT = (
    "log/active/peg_b3_direct_sim2sim_20260717/"
    "chunk_replay_fresh_b3_20260724/smoke_ep1_t0_h20.json"
)

# These are the settings used by the completed formal fresh MuJoCo run.  They
# are intentionally not CLI ablations in this parity harness.
PHYSICS_SUBSTEPS = 16
HOLE_COLLISION = "box_ring_big"
CONTROLLER_SOURCE = "b3_center"
FINGER_VELOCITY_LIMITS = np.array([0.04, 0.04], dtype=np.float64)
EFFORT_SCALE = np.ones(7, dtype=np.float64)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_endpoint(raw: np.ndarray) -> dict:
    peg_pos, peg_quat, hole_pos, hole_quat = Replay.pose_in_robot_root(raw)
    peg_in_hole = Q.pose_in_root(hole_pos, hole_quat, peg_pos, peg_quat)
    _, relative_quat = Q.subtract_frame_transforms(
        hole_pos, hole_quat, peg_pos, peg_quat
    )
    return {
        "assembly_pos_m": float(
            np.linalg.norm(
                peg_in_hole[:3] - np.array([0.0, 0.0, CL.ASSEMBLED_Z])
            )
        ),
        "assembly_roll_pitch_rad": Replay.roll_pitch_error(relative_quat),
        "peg_in_hole_xyz_m": peg_in_hole[:3].tolist(),
    }


def peg_roll_pitch_error(data, ctrl, raw_ref: np.ndarray) -> float:
    _, peg_quat_ref, _, _ = Replay.pose_in_robot_root(raw_ref)
    peg_quat = data.qpos[ctrl.peg_qpos + 3 : ctrl.peg_qpos + 7]
    relative = Q.quat_mul(Q.quat_inv(peg_quat_ref), peg_quat)
    return Replay.roll_pitch_error(relative)


def episode_identity(store, episode: int) -> dict:
    identity = {"episode": episode}
    for key in (
        "episode_env_ids",
        "episode_env_generations",
        "episode_completion_steps",
        "reset_dataset_pool_indices",
    ):
        if key in store.attrs:
            values = store.attrs[key]
            identity[key] = values[episode]
    return identity


def validate_formal_profile(
    store, *, allow_success_selected_full: bool = False
) -> None:
    mode = store.attrs.get("mode")
    selected_full = mode == "isaac_success_selected_full160"
    if selected_full:
        if not allow_success_selected_full:
            raise ValueError(
                "success-selected full source requires explicit "
                "allow_success_selected_full=True"
            )
        selected_expected = {
            "source_collection_mode": "isaac_closed_loop_fresh_unfiltered",
            "success_filtering": True,
            "full_horizon_rows_preserved": True,
            "preserved_full_episode_horizon": 160,
            "failed_source_episodes_in_training": 0,
        }
        selected_mismatches = {
            key: {"observed": store.attrs.get(key), "expected": expected}
            for key, expected in selected_expected.items()
            if store.attrs.get(key) != expected
        }
        if selected_mismatches:
            raise ValueError(
                "selected-full source provenance is invalid: "
                f"{selected_mismatches}"
            )
    expected_attrs = {
        "mode": (
            "isaac_success_selected_full160"
            if selected_full
            else "isaac_closed_loop_fresh_unfiltered"
        ),
        "task_id": B3.FORMAL_TASK_ID,
        "controller_profile": "b3_center",
        "simulation_jacobian_point": "link_origin",
        "physics_dt_s": 1.0 / 480.0,
        "decimation": 48,
        "policy_dt_s": 0.1,
        "gripper_stiffness": 1000.0,
        "gripper_damping": 14.0,
        "gripper_effort_limit": 60.0,
    }
    mismatches = {}
    for key, expected in expected_attrs.items():
        observed = store.attrs.get(key)
        if isinstance(expected, float):
            match = observed is not None and abs(float(observed) - expected) <= 1.0e-12
        else:
            match = observed == expected
        if not match:
            mismatches[key] = {"observed": observed, "expected": expected}
    if mismatches:
        raise ValueError(f"source is not the formal fresh B3 profile: {mismatches}")


def build_chunk_specs(
    episode_ids: list[int],
    starts: np.ndarray,
    ends: np.ndarray,
    chunk_len: int,
    t0: int | None,
    length: int | None,
    max_chunks: int,
) -> list[tuple[int, int, int]]:
    if chunk_len <= 0:
        raise ValueError("--chunk_len must be positive")
    if max_chunks < 0:
        raise ValueError("--max_chunks must be non-negative")

    specs = []
    if t0 is not None:
        if len(episode_ids) != 1:
            raise ValueError("--t0 requires exactly one --episode_id")
        actual_length = chunk_len if length is None else length
        if t0 < 0 or actual_length <= 0:
            raise ValueError("--t0 must be non-negative and --length positive")
        episode = episode_ids[0]
        episode_len = int(ends[episode] - starts[episode])
        if t0 + actual_length >= episode_len:
            raise ValueError(
                "the source stores pre-action states only, so a comparable "
                f"endpoint requires t0+length < {episode_len}; got "
                f"{t0}+{actual_length}"
            )
        specs.append((episode, t0, actual_length))
    else:
        if length is not None:
            raise ValueError("--length is only valid with --t0")
        for episode in episode_ids:
            episode_len = int(ends[episode] - starts[episode])
            # raw_state[t] is the state before action[t].  Therefore the last
            # source row is a valid endpoint but its own outgoing action has no
            # recorded successor state and is deliberately excluded.
            comparable_actions = episode_len - 1
            for local_t0 in range(0, comparable_actions, chunk_len):
                actual_length = min(chunk_len, comparable_actions - local_t0)
                specs.append((episode, local_t0, actual_length))

    return specs[:max_chunks] if max_chunks else specs


def build_episode_context(
    store, episode_start: int, episode_end: int
) -> dict:
    """Compile one immutable model/controller pair for one source episode."""
    raw0 = np.asarray(store["data/raw_state"][episode_start], dtype=np.float64)
    model, ctrl, hole_pos, hole_quat, effective = B3._episode_model(
        store,
        episode_start,
        episode_end,
        raw0,
        PHYSICS_SUBSTEPS,
        HOLE_COLLISION,
        None,
        CONTROLLER_SOURCE,
    )
    return {
        "model": model,
        "ctrl": ctrl,
        "hole_pos": hole_pos,
        "hole_quat": hole_quat,
        "effective": effective,
    }


def top_per_episode(rows: list[dict], top_k: int) -> list[dict]:
    selected = []
    episodes = sorted({row["episode"] for row in rows})
    for episode in episodes:
        episode_rows = [row for row in rows if row["episode"] == episode]
        selected.extend(
            sorted(
                episode_rows,
                key=lambda row: row["endpoint"][
                    "normalized_divergence_score"
                ],
                reverse=True,
            )[:top_k]
        )
    return selected


def split_specs(parents: list[dict], child_length: int) -> list[tuple[int, int, int]]:
    specs = []
    seen = set()
    for parent in parents:
        begin, end = parent["t0"], parent["t1"]
        for t0 in range(begin, end, child_length):
            length = min(child_length, end - t0)
            spec = (parent["episode"], t0, length)
            if length > 0 and spec not in seen:
                seen.add(spec)
                specs.append(spec)
    return specs


def replay_chunk(
    store,
    context: dict,
    episode: int,
    episode_start: int,
    local_t0: int,
    length: int,
    gripper_command: str,
) -> tuple[dict, dict]:
    global_t0 = episode_start + local_t0
    global_t1 = global_t0 + length
    raw_start = np.asarray(store["data/raw_state"][global_t0], dtype=np.float64)
    raw_target = np.asarray(store["data/raw_state"][global_t1], dtype=np.float64)
    actions = np.asarray(
        store["data/action"][global_t0:global_t1], dtype=np.float64
    )
    recorded_close = (
        np.asarray(
            store["data/gripper_processed_action"][global_t0:global_t1],
            dtype=np.float64,
        )[:, 0]
        < 0.02
    )

    model = context["model"]
    ctrl = context["ctrl"]
    hole_pos = context["hole_pos"]
    hole_quat = context["hole_quat"]
    data = mujoco.MjData(model)
    Replay.set_raw_state(model, data, ctrl, raw_start)
    # step_action uses a state-relative target when action_reference_blend=0.
    # Setting these explicitly makes the absence of cross-chunk controller
    # state visible even though the values are overwritten on the first step.
    ctrl.action_reference_pos, ctrl.action_reference_quat = ctrl.ee_root(data)
    fk = Replay.R.FrankaFK()
    restore_metrics = Replay.measure(data, ctrl, raw_start, fk, False)

    live_guard_close = []
    applied_close = []
    peg_position_errors_m = []
    for step, action in enumerate(actions):
        live_guard_close.append(bool(ctrl.grasp_close(data)))
        applied_close.append(
            Replay.step_action(
                model,
                data,
                ctrl,
                action,
                finger_velocity_limits=FINGER_VELOCITY_LIMITS,
                independent_gripper=True,
                gripper_close_override=(
                    bool(recorded_close[step])
                    if gripper_command == "recorded"
                    else None
                ),
                jacobian_point="link_origin",
                nullspace_stiffness=0.0,
                nullspace_damping_ratio=1.0,
                action_reference_blend=0.0,
                bias_compensation_scale=0.0,
                effort_scale=EFFORT_SCALE,
            )
        )
        raw_step_target = np.asarray(
            store["data/raw_state"][global_t0 + step + 1],
            dtype=np.float64,
        )
        peg_position_target, _, _, _ = Replay.pose_in_robot_root(
            raw_step_target
        )
        peg_position_errors_m.append(
            float(
                np.linalg.norm(
                    data.qpos[ctrl.peg_qpos : ctrl.peg_qpos + 3]
                    - peg_position_target
                )
            )
        )

    endpoint = Replay.measure(data, ctrl, raw_target, fk, applied_close[-1])
    endpoint["peg_roll_pitch_error_rad"] = peg_roll_pitch_error(
        data, ctrl, raw_target
    )
    (
        assembly_pos,
        assembly_roll_pitch,
        assembly_square_yaw,
        official_pose,
        square_yaw_pose,
        peg_in_hole,
    ) = Replay.assembly_metrics(data, ctrl, hole_pos, hole_quat)
    source_task = source_endpoint(raw_target)
    endpoint.update(
        {
            "mujoco_assembly_pos_m": assembly_pos,
            "mujoco_assembly_roll_pitch_rad": assembly_roll_pitch,
            "mujoco_assembly_square_yaw_rot_rad": assembly_square_yaw,
            "mujoco_official_pose": official_pose,
            "mujoco_square_yaw_pose": square_yaw_pose,
            "mujoco_peg_in_hole_xyz_m": peg_in_hole.tolist(),
            "source_assembly_pos_m": source_task["assembly_pos_m"],
            "source_assembly_roll_pitch_rad": (
                source_task["assembly_roll_pitch_rad"]
            ),
            "source_peg_in_hole_xyz_m": source_task["peg_in_hole_xyz_m"],
            "assembly_pos_abs_delta_m": abs(
                assembly_pos - source_task["assembly_pos_m"]
            ),
            "assembly_roll_pitch_abs_delta_rad": abs(
                assembly_roll_pitch - source_task["assembly_roll_pitch_rad"]
            ),
        }
    )
    # Dimensionless ranking under the formal B3 task thresholds.
    endpoint["normalized_divergence_score"] = (
        endpoint["peg_pos_m"] / CL.SUC_POS
        + endpoint["peg_roll_pitch_error_rad"] / CL.SUC_ROT
    )
    endpoint["peg_position_error_sum_m"] = float(
        np.sum(peg_position_errors_m)
    )
    endpoint["peg_position_error_mean_m"] = float(
        np.mean(peg_position_errors_m)
    )
    endpoint["peg_position_errors_m"] = peg_position_errors_m

    live_guard_close = np.asarray(live_guard_close, dtype=bool)
    applied_close = np.asarray(applied_close, dtype=bool)
    if gripper_command == "recorded" and not np.array_equal(
        applied_close, recorded_close
    ):
        raise AssertionError(
            "recorded gripper replay did not apply the source command exactly"
        )
    row = {
        **episode_identity(store, episode),
        "t0": local_t0,
        "t1": local_t0 + length,
        "length": length,
        "source_indices": [global_t0, global_t1],
        "start_restore": restore_metrics,
        "endpoint": endpoint,
        "gripper": {
            "semantics": (
                "recorded Isaac gripper_processed_action replay"
                if gripper_command == "recorded"
                else (
                    "formal target evaluator live privileged grasp guard; "
                    "policy scalar ignored"
                )
            ),
            "command_source": gripper_command,
            "source_recorded_close_steps": int(recorded_close.sum()),
            "mujoco_live_guard_close_steps": int(live_guard_close.sum()),
            "applied_close_steps": int(applied_close.sum()),
            "source_vs_live_guard_mismatch_steps": int(
                np.count_nonzero(recorded_close != live_guard_close)
            ),
            "source_recorded_close": recorded_close.astype(int).tolist(),
            "mujoco_live_guard_close": live_guard_close.astype(int).tolist(),
            "applied_close": applied_close.astype(int).tolist(),
        },
    }
    return row


def run(args) -> None:
    source_path = str(Path(args.zarr).resolve())
    store = zarr.open(source_path, mode="r")
    validate_formal_profile(store)
    starts, ends, source_audit = B3._audit_source_dataset(
        store,
        source_path,
        expected_episode_steps=160,
        allow_incomplete_source_episodes=False,
    )

    episode_ids = args.episode_ids or [1]
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("--episode_ids contains duplicates")
    for episode in episode_ids:
        if episode < 0 or episode >= len(ends):
            raise ValueError(f"episode {episode} outside [0, {len(ends) - 1}]")

    initial_specs = build_chunk_specs(
        episode_ids,
        starts,
        ends,
        args.chunk_len,
        args.t0,
        args.length,
        args.max_chunks,
    )
    if not initial_specs:
        raise ValueError("no chunks selected")
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    previous_length = args.chunk_len
    for fine_length in args.fine_lengths:
        if fine_length <= 0 or fine_length >= previous_length:
            raise ValueError(
                "--fine_lengths must be positive and strictly descending "
                f"below --chunk_len; got {args.fine_lengths}"
            )
        previous_length = fine_length

    CL.SIM_DT = float(store.attrs["physics_dt_s"])
    CL.DECIM = int(store.attrs["decimation"])

    contexts = {}
    rows = []
    stages = []

    def execute_specs(stage_name, specs):
        stage_rows = []
        for index, (episode, t0, length) in enumerate(specs):
            if episode not in contexts:
                contexts[episode] = build_episode_context(
                    store, int(starts[episode]), int(ends[episode])
                )
            row = replay_chunk(
                store,
                contexts[episode],
                episode,
                int(starts[episode]),
                t0,
                length,
                args.gripper_command,
            )
            row["stage"] = stage_name
            stage_rows.append(row)
            rows.append(row)
            endpoint = row["endpoint"]
            print(
                f"[b3-chunk-replay] {stage_name} "
                f"{index + 1}/{len(specs)} ep={episode} "
                f"t={t0}:{t0 + length} "
                f"peg={endpoint['peg_pos_m'] * 1000:.3f}mm "
                f"rp={endpoint['peg_roll_pitch_error_rad']:.4f}rad "
                f"joint={endpoint['joint_pos_l2_rad']:.4f}rad "
                "grip_source_vs_live="
                f"{row['gripper']['source_vs_live_guard_mismatch_steps']}/{length}",
                flush=True,
            )
        return stage_rows

    stage_name = "explicit" if args.t0 is not None else f"coarse_{args.chunk_len}"
    current_rows = execute_specs(stage_name, initial_specs)
    stages.append(
        {
            "name": stage_name,
            "nominal_length": args.length or args.chunk_len,
            "chunk_count": len(current_rows),
            "specs": [
                {
                    "episode": row["episode"],
                    "t0": row["t0"],
                    "t1": row["t1"],
                    "length": row["length"],
                }
                for row in current_rows
            ],
            "selected_for_refinement": [],
        }
    )

    # Explicit single-chunk smoke runs deliberately stop after one chunk.
    if args.t0 is None:
        for fine_length in args.fine_lengths:
            parents = top_per_episode(current_rows, args.top_k)
            stages[-1]["selected_for_refinement"] = [
                {
                    "episode": row["episode"],
                    "t0": row["t0"],
                    "t1": row["t1"],
                    "length": row["length"],
                    "normalized_divergence_score": row["endpoint"][
                        "normalized_divergence_score"
                    ],
                }
                for row in parents
            ]
            fine_specs = split_specs(parents, fine_length)
            stage_name = f"fine_{fine_length}"
            current_rows = execute_specs(stage_name, fine_specs)
            stages.append(
                {
                    "name": stage_name,
                    "nominal_length": fine_length,
                    "chunk_count": len(current_rows),
                    "specs": [
                        {
                            "episode": row["episode"],
                            "t0": row["t0"],
                            "t1": row["t1"],
                            "length": row["length"],
                        }
                        for row in current_rows
                    ],
                    "selected_for_refinement": [],
                }
            )

    ranked = sorted(
        current_rows,
        key=lambda row: row["endpoint"]["normalized_divergence_score"],
        reverse=True,
    )
    ranked_all = sorted(
        rows,
        key=lambda row: row["endpoint"]["normalized_divergence_score"],
        reverse=True,
    )
    first_effective = contexts[episode_ids[0]]["effective"]
    report = {
        "definition": (
            "at each chunk boundary restore the recorded Isaac pre-action "
            "state in MuJoCo, replay only that chunk's recorded raw arm policy "
            "actions and "
            + (
                "recorded processed gripper commands"
                if args.gripper_command == "recorded"
                else "target-side live grasp-guard commands"
            )
            + " through the formal B3 target evaluator action executor, and "
            "compare the same-time terminal state"
        ),
        "intended_differences_from_formal_target_eval": [
            "state is restored independently at every chunk boundary",
            "recorded Isaac policy actions replace live MuJoCo policy inference",
        ]
        + (
            [
                "recorded Isaac processed gripper commands replace the "
                "target-side live grasp guard to make this a canonical "
                "source-action replay"
            ]
            if args.gripper_command == "recorded"
            else []
        ),
        "must_match_invariants": {
            "source": "same formal fresh unfiltered paired zarr",
            "controller": "B3 center Kp1000/50, zeta1, source torque limits",
            "jacobian_point": "link_origin",
            "timing": "1/480 s source physics tick x 48; 16 MuJoCo integration substeps",
            "gripper_actuator": (
                "independent 1000/14/60 PD, 0.04 m/s contact-aware "
                "velocity limits"
            ),
            "hole_collision": HOLE_COLLISION,
            "runtime_properties": (
                "per-episode source armature/friction/body inertia/materials"
            ),
        },
        "source": {
            "path": source_path,
            "zarr_signature": CL.zarr_signature(store),
            "checkpoint_sha256": store.attrs.get("checkpoint_sha256"),
            "initial_state_sha256": store.attrs.get("initial_state_sha256"),
            "collector_script_sha256": store.attrs.get("collector_script_sha256"),
            "source_audit": source_audit,
        },
        "harness": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": file_sha256(__file__),
            "formal_evaluator_script": str(Path(B3.__file__).resolve()),
            "formal_evaluator_script_sha256": file_sha256(B3.__file__),
            "action_executor_script": str(Path(Replay.__file__).resolve()),
            "action_executor_script_sha256": file_sha256(Replay.__file__),
        },
        "protocol": {
            "episode_ids": episode_ids,
            "chunk_len": args.chunk_len,
            "explicit_t0": args.t0,
            "explicit_length": args.length,
            "max_chunks": args.max_chunks,
            "initial_chunks": len(initial_specs),
            "total_replayed_chunks": len(rows),
            "top_k_per_episode_per_stage": args.top_k,
            "fine_lengths": args.fine_lengths if args.t0 is None else [],
            "terminal_state_boundary": (
                "raw_state[t] is pre-action; action rows with no recorded "
                "successor state are excluded; for each 160-row fresh episode "
                "automatic chunks end at t1<=159 and action[159] is not replayed"
            ),
            "max_comparable_t1_by_episode": {
                str(episode): int(ends[episode] - starts[episode] - 1)
                for episode in episode_ids
            },
            "physics_substeps_per_source_tick": PHYSICS_SUBSTEPS,
            "hole_collision": HOLE_COLLISION,
            "controller_source": CONTROLLER_SOURCE,
            "gripper_command_source": args.gripper_command,
            "model_reuse": (
                "one model/controller compiled per episode; each chunk uses a "
                "fresh MjData plus exact raw-state restore"
            ),
            "controller_state_isolation": (
                "action_reference_blend=0 makes OSC targets state-relative; "
                "action reference is also explicitly reset at each boundary; "
                "grasp guard is stateless"
            ),
        },
        "effective_config_note": (
            "the effective_config payload is the reused formal target "
            "episode model. Its gripper.command field describes that "
            "evaluator's live-guard default; protocol.gripper_command_source "
            "is the command actually applied by this replay harness"
        ),
        "effective_config_first_chunk": first_effective,
        "effective_config_by_episode": {
            str(episode): context["effective"]
            for episode, context in contexts.items()
        },
        "stages": stages,
        "chunks": rows,
        "ranked_finest_chunks": [
            {
                "episode": row["episode"],
                "t0": row["t0"],
                "t1": row["t1"],
                "normalized_divergence_score": row["endpoint"][
                    "normalized_divergence_score"
                ],
                "peg_pos_m": row["endpoint"]["peg_pos_m"],
                "peg_roll_pitch_error_rad": row["endpoint"][
                    "peg_roll_pitch_error_rad"
                ],
                "joint_pos_l2_rad": row["endpoint"]["joint_pos_l2_rad"],
                "source_vs_live_guard_mismatch_steps": row["gripper"][
                    "source_vs_live_guard_mismatch_steps"
                ],
            }
            for row in ranked
        ],
        "ranked_all_chunks": [
            {
                "stage": row["stage"],
                "episode": row["episode"],
                "t0": row["t0"],
                "t1": row["t1"],
                "normalized_divergence_score": row["endpoint"][
                    "normalized_divergence_score"
                ],
                "peg_pos_m": row["endpoint"]["peg_pos_m"],
                "peg_roll_pitch_error_rad": row["endpoint"][
                    "peg_roll_pitch_error_rad"
                ],
            }
            for row in ranked_all
        ],
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(CL.to_jsonable(report), handle, indent=2)
    print(f"[b3-chunk-replay] report -> {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default=DEFAULT_ZARR)
    parser.add_argument("--episode_ids", type=int, nargs="*")
    parser.add_argument("--chunk_len", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument(
        "--fine_lengths",
        type=int,
        nargs="*",
        default=[10, 5],
        help=(
            "automatic hierarchy for non-explicit runs: select each episode's "
            "top-k parent chunks, then split at each listed length"
        ),
    )
    parser.add_argument(
        "--t0",
        type=int,
        default=None,
        help="single-chunk start; requires exactly one episode ID",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="single-chunk length; valid only with --t0",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=0,
        help="diagnostic cap after selection; 0 processes all selected chunks",
    )
    parser.add_argument(
        "--gripper_command",
        choices=("recorded", "live_guard"),
        default="recorded",
        help=(
            "canonical source-action replay uses recorded; "
            "live_guard reproduces the formal target evaluator's "
            "state-dependent gripper command"
        ),
    )
    parser.add_argument("--out", default=DEFAULT_OUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
