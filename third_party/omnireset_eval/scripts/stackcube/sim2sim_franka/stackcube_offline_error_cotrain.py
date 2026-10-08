"""Offline StackCube sim2sim diagnosis and local expert correction.

This is intentionally not DAgger.  A recorded IsaacSim source trajectory is
the immutable diagnostic input.  For each candidate boundary ``s_t`` MuJoCo
is reset once to the recorded Isaac state and executes exactly H recorded
actions.  Only after errors have been ranked is MuJoCo reset again to that same
``s_t`` and the frozen expert is queried for H live, closed-loop actions.

Formal inputs are rich zarr files from ``export_franka_valset.py``.  They must
contain raw state, policy observation, raw action, processed gripper command,
and the runtime B3 dynamics/material arrays used by IsaacSim.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop_stackcube_eval as Stack
import compare_peginsert_continuous_mujoco as B3
import compare_stackcube_onestep_mujoco as StackModel
from franka_policy import FrankaPolicy


RUNTIME_FIELDS = (
    "arm_scale",
    "arm_kp",
    "arm_kd",
    "arm_torque_max",
    "arm_joint_armature",
    "arm_joint_friction_dynamic",
    "arm_joint_friction_viscous",
)
SCENE_FIELDS = (
    "robot_material_properties",
    "robot_body_masses",
    "robot_body_coms",
    "robot_body_inertias",
    "insertive_object_material_properties",
    "insertive_object_body_masses",
    "insertive_object_body_coms",
    "insertive_object_body_inertias",
    "receptive_object_material_properties",
    "receptive_object_body_masses",
    "table_material_properties",
    "table_body_masses",
)
DEFAULT_JACOBIAN_POINT = "link_origin"
DEFAULT_FINGER_VELOCITY_LIMITS = (0.04, 0.04)


def make_controller_profile(
    jacobian_point=DEFAULT_JACOBIAN_POINT,
    finger_velocity_limits=DEFAULT_FINGER_VELOCITY_LIMITS,
    hand_stiffness=1000.0,
    hand_damping=14.0,
    hand_effort_limit=60.0,
):
    if jacobian_point not in ("physx_com", "link_origin"):
        raise ValueError(f"unknown Jacobian point: {jacobian_point}")
    limits = np.asarray(finger_velocity_limits, dtype=np.float64)
    if (
        limits.shape != (2,)
        or not np.all(np.isfinite(limits))
        or np.any(limits <= 0.0)
    ):
        raise ValueError(
            "finger_velocity_limits must contain two finite positive values"
        )
    return {
        "schema_version": 1,
        "jacobian_point": str(jacobian_point),
        "hand_com_local_m": np.asarray(
            B3.ISAAC_HAND_COM_LOCAL, dtype=np.float64
        ).tolist(),
        "finger_joint_names": [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ],
        "finger_velocity_limits_mps": limits.tolist(),
        "finger_velocity_limit_enforcement": (
            "project independent-PD actuator drive at every MuJoCo integration "
            "substep; contact may exceed the limit"
        ),
        "hand_stiffness": np.broadcast_to(
            np.asarray(hand_stiffness, dtype=np.float64), (2,)
        ).tolist(),
        "hand_damping": np.broadcast_to(
            np.asarray(hand_damping, dtype=np.float64), (2,)
        ).tolist(),
        "hand_effort_limit": np.broadcast_to(
            np.asarray(hand_effort_limit, dtype=np.float64), (2,)
        ).tolist(),
        "nullspace_stiffness": 0.0,
        "nullspace_damping_ratio": 1.0,
        "action_reference_blend": 0.0,
        "bias_compensation_scale": 0.0,
        "arm_effort_scale": [1.0] * 7,
    }




def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def episode_ranges(store):
    ends = np.asarray(store["meta/episode_ends"], dtype=np.int64)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
    if len(ends) == 0 or np.any(ends <= starts):
        raise ValueError("source has empty or invalid episodes")
    return starts, ends


def constant_field(store, key, start, end):
    values = np.asarray(store[f"data/{key}"][start:end], dtype=np.float64)
    drift = float(np.max(np.abs(values - values[0])))
    if drift > 1.0e-6:
        raise ValueError(f"runtime field {key} drifts within episode: {drift}")
    return values[0]


def require_source_schema(store, source_interface, horizon, allow_unmarked_chunks):
    required = {"data/state", "data/action", "data/raw_state", "meta/episode_ends"}
    if "data/gripper_processed_action" not in store:
        required.add("data/gripper_processed_action")
    required.update(f"data/{key}" for key in RUNTIME_FIELDS + SCENE_FIELDS)
    missing = sorted(path for path in required if path not in store)
    if missing:
        raise KeyError(f"rich source zarr is missing: {missing}")
    if store["data/raw_state"].shape[1] != 57:
        raise ValueError("data/raw_state must use the canonical 57-D layout")
    if store["data/state"].shape[1] != 200 or store["data/action"].shape[1] != 7:
        raise ValueError("expected state[*,200] and action[*,7]")
    expected_h = {"action_step": 1, "action_chunk8": 8}.get(source_interface)
    if expected_h is not None and horizon != expected_h:
        raise ValueError(f"{source_interface} requires --horizon {expected_h}")
    if source_interface == "action_chunk8" and "data/action_chunk_start" not in store:
        if not allow_unmarked_chunks:
            raise KeyError(
                "formal chunk8 source requires data/action_chunk_start; "
                "use --allow_unmarked_chunks only for a smoke test"
            )


def configure_timing_and_controller(store, runtime, physics_substeps=1):
    physics_dt = float(store.attrs["physics_dt_s"])
    decimation = int(store.attrs["decimation"])
    policy_dt = float(store.attrs["policy_dt_s"])
    physics_substeps = int(physics_substeps)
    if physics_substeps <= 0:
        raise ValueError("physics_substeps must be positive")
    if abs(physics_dt * decimation - policy_dt) > 1.0e-9:
        raise ValueError("source physics_dt * decimation != policy_dt")
    StackModel.SIM_DT = physics_dt / physics_substeps
    B3.CL.SIM_DT = physics_dt
    B3.CL.DECIM = decimation
    B3.CL.SCALE = runtime["arm_scale"].copy()
    B3.CL.KP = runtime["arm_kp"].copy()
    B3.CL.KD = runtime["arm_kd"].copy()
    B3.CL.TAU_MAX = runtime["arm_torque_max"].copy()
    return physics_dt, decimation, policy_dt


def configure_stack_controller(model):
    ctrl = Stack.configure_controller(model)
    # Reuse the audited B3 state restore and executor.  These aliases only map
    # names; they do not add a second controller implementation.
    ctrl.peg = ctrl.obj
    ctrl.peg_qpos = int(ctrl.obj_qadr)
    ctrl.peg_dof = int(ctrl.obj_dadr)
    ctrl.l0 = ctrl.root
    return ctrl


def apply_stack_materials(model, scene):
    applied = B3.apply_runtime_contact_materials(model, scene)
    mapping = {
        "insertive_cube_geom": "insertive_object_material_properties",
        "receptive_cube_geom": "receptive_object_material_properties",
        "table_top": "table_material_properties",
    }
    for geom_name, source_key in mapping.items():
        geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom >= 0:
            coefficient = float(np.asarray(scene[source_key]).reshape(-1, 3)[0, 1])
            model.geom_friction[geom, 0] = coefficient
            applied[geom_name] = coefficient
    return applied


def build_episode_model(
    store,
    start,
    end,
    raw_at_boundary,
    physics_substeps=1,
    *,
    jacobian_point=DEFAULT_JACOBIAN_POINT,
    finger_velocity_limits=DEFAULT_FINGER_VELOCITY_LIMITS,
):
    gripper_runtime = {}
    for key, default in (
        ("gripper_joint_stiffness", 1000.0),
        ("gripper_joint_damping", 14.0),
        ("gripper_joint_effort_limit", 60.0),
    ):
        gripper_runtime[key] = (
            constant_field(store, key, start, end)
            if f"data/{key}" in store
            else default
        )
    controller_profile = make_controller_profile(
        jacobian_point,
        finger_velocity_limits,
        gripper_runtime["gripper_joint_stiffness"],
        gripper_runtime["gripper_joint_damping"],
        gripper_runtime["gripper_joint_effort_limit"],
    )
    runtime = {key: constant_field(store, key, start, end) for key in RUNTIME_FIELDS}
    scene = {key: constant_field(store, key, start, end) for key in SCENE_FIELDS}
    scene["robot_body_names"] = list(store.attrs["robot_body_names"])
    physics_dt, decimation, policy_dt = configure_timing_and_controller(
        store, runtime, physics_substeps
    )
    _, _, receptive_pos, receptive_quat = B3.pose_in_robot_root(raw_at_boundary)
    model = StackModel.build_model(np.concatenate([receptive_pos, receptive_quat]))
    ctrl = configure_stack_controller(model)
    ctrl.sim2sim_controller_profile = controller_profile

    # The Isaac pat_vention collision mesh_0 top is local z=0.868 m and the
    # asset root is at z=-0.881 m: its support surface is therefore z=-0.013 m
    # in robot-root coordinates.  This is fixed source-asset geometry, not a
    # trajectory-conditioned correction.
    table_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    model.body_pos[table_body, 2] = -0.033

    model.dof_armature[:7] = runtime["arm_joint_armature"]
    model.dof_damping[:7] = runtime["arm_joint_friction_viscous"]
    model.dof_frictionloss[:7] = runtime["arm_joint_friction_dynamic"]
    masses = B3.apply_runtime_body_masses(model, ctrl, scene, include_robot=True)
    materials = apply_stack_materials(model, scene)
    B3.disable_tendon_gripper(model)
    mujoco.mj_setConst(model, mujoco.MjData(model))
    effective = {
        "physics_dt_s": physics_dt,
        "mujoco_physics_substeps_per_isaac_tick": int(physics_substeps),
        "mujoco_integrator_timestep_s": float(model.opt.timestep),
        "decimation": decimation,
        "policy_dt_s": policy_dt,
        "runtime": runtime,
        "masses_applied_kg": masses,
        "contact_materials_applied": materials,
        "integrator": "implicitfast",
        "table_surface_z_m": -0.013,
        "controller_profile": controller_profile,
        "jacobian_point": controller_profile["jacobian_point"],
        "finger_velocity_limits_mps": controller_profile[
            "finger_velocity_limits_mps"
        ],
        "gripper": (
            "independent per-episode Isaac-matched position-PD, "
            f"{controller_profile['finger_velocity_limits_mps']}m/s drive projection"
        ),
    }
    return model, ctrl, effective


def set_boundary_state(model, data, ctrl, raw):
    B3.set_raw_state(model, data, ctrl, np.asarray(raw, dtype=np.float64))
    ctrl.action_reference_pos, ctrl.action_reference_quat = ctrl.ee_root(data)


def source_gripper_close(store, row):
    target = np.asarray(store["data/gripper_processed_action"][row], dtype=np.float64)
    return bool(float(np.mean(target)) < 0.02)


def execute_action(
    model,
    data,
    ctrl,
    action,
    gripper_close_override,
    *,
    jacobian_point=None,
    finger_velocity_limits=None,
):
    stored_profile = getattr(ctrl, "sim2sim_controller_profile", None)
    if stored_profile is None:
        stored_profile = make_controller_profile()
    requested_profile = make_controller_profile(
        stored_profile["jacobian_point"]
        if jacobian_point is None
        else jacobian_point,
        stored_profile["finger_velocity_limits_mps"]
        if finger_velocity_limits is None
        else finger_velocity_limits,
        stored_profile["hand_stiffness"],
        stored_profile["hand_damping"],
        stored_profile["hand_effort_limit"],
    )
    if requested_profile != stored_profile:
        raise ValueError(
            "execute_action controller profile differs from build_episode_model: "
            f"{requested_profile} != {stored_profile}"
        )
    return B3.step_action(
        model,
        data,
        ctrl,
        np.asarray(action, dtype=np.float64),
        finger_velocity_limits=np.asarray(
            requested_profile["finger_velocity_limits_mps"], dtype=np.float64
        ),
        independent_gripper=True,
        gripper_close_override=gripper_close_override,
        jacobian_point=requested_profile["jacobian_point"],
        nullspace_stiffness=0.0,
        nullspace_damping_ratio=1.0,
        action_reference_blend=0.0,
        bias_compensation_scale=0.0,
        effort_scale=np.ones(7, dtype=np.float64),
        gripper_stiffness=np.asarray(
            requested_profile["hand_stiffness"], dtype=np.float64
        ),
        gripper_damping=np.asarray(
            requested_profile["hand_damping"], dtype=np.float64
        ),
        gripper_effort_limit=np.asarray(
            requested_profile["hand_effort_limit"], dtype=np.float64
        ),
    )


def reference_pose(raw):
    peg_pos, peg_quat, _, _ = B3.pose_in_robot_root(np.asarray(raw, dtype=np.float64))
    return np.concatenate([peg_pos, peg_quat])


def simulated_pose(data, ctrl):
    return np.concatenate(
        [
            data.qpos[ctrl.obj_qadr : ctrl.obj_qadr + 3],
            data.qpos[ctrl.obj_qadr + 3 : ctrl.obj_qadr + 7],
        ]
    ).copy()


def candidate_rows(store, source_interface, horizon, stride, max_candidates):
    starts, ends = episode_ranges(store)
    marked = (
        np.asarray(store["data/action_chunk_start"], dtype=bool)
        if "data/action_chunk_start" in store
        else None
    )
    rows = []
    for episode, (start, end) in enumerate(zip(starts, ends)):
        episode_stride = horizon if source_interface == "action_chunk8" else stride
        for row in range(int(start), int(end) - horizon, max(episode_stride, 1)):
            if source_interface == "action_chunk8" and marked is not None and not marked[row]:
                continue
            rows.append((episode, row, row - int(start), int(start), int(end)))
            if max_candidates > 0 and len(rows) >= max_candidates:
                return rows
    return rows


def run_diagnostics(store, candidates, horizon):
    raw = store["data/raw_state"]
    action = store["data/action"]
    records = []
    effective_first = None
    cached_episode = None
    cached_model = cached_ctrl = None
    for index, (episode, row, source_step, ep_start, ep_end) in enumerate(candidates):
        if episode != cached_episode:
            cached_model, cached_ctrl, effective = build_episode_model(
                store, ep_start, ep_end, raw[row]
            )
            cached_episode = episode
        model, ctrl = cached_model, cached_ctrl
        effective_first = effective_first or effective
        data = mujoco.MjData(model)
        set_boundary_state(model, data, ctrl, raw[row])
        for offset in range(horizon):
            execute_action(
                model,
                data,
                ctrl,
                action[row + offset],
                source_gripper_close(store, row + offset),
            )
        predicted = simulated_pose(data, ctrl)
        target = reference_pose(raw[row + horizon])
        records.append(
            {
                "episode": episode,
                "row": row,
                "source_step": source_step,
                "episode_start": ep_start,
                "episode_end": ep_end,
                "object_pos_error_m": float(np.linalg.norm(predicted[:3] - target[:3])),
                "object_rot_error_rad": Stack.quat_angle(predicted[3:7], target[3:7]),
                "joint_l2_error_rad": float(
                    np.linalg.norm(data.qpos[:9] - np.asarray(raw[row + horizon, :9]))
                ),
                "predicted_object_pose": predicted.astype(np.float32),
                "reference_object_pose": target.astype(np.float32),
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(candidates):
            print(f"[diagnose] {index + 1}/{len(candidates)}", flush=True)
    return records, effective_first


def seed_observation_history(builder, observation):
    observation = np.asarray(observation, dtype=np.float64)
    for name, (start, width) in Stack.BLOCKS.items():
        builder.hist[name].clear()
        for history_index in range(Stack.HIST):
            lo = start + history_index * width
            builder.hist[name].append(observation[lo : lo + width].copy())


def select_records(records, high_count, random_count, seed):
    order = np.argsort([-record["object_pos_error_m"] for record in records])
    high_indices = list(map(int, order[: min(high_count, len(order))]))
    remainder = np.asarray([i for i in range(len(records)) if i not in set(high_indices)])
    rng = np.random.default_rng(seed)
    if len(remainder) and random_count:
        random_indices = rng.choice(
            remainder, size=min(random_count, len(remainder)), replace=False
        ).astype(int).tolist()
    else:
        random_indices = []
    return [(index, "high_error") for index in high_indices] + [
        (index, "random") for index in random_indices
    ]


def collect_corrections(store, records, selected, horizon, expert):
    raw = store["data/raw_state"]
    source_state = store["data/state"]
    states = []
    prev_states = []
    actions = []
    source_actions = []
    raw_starts = []
    kinds = []
    diagnostic_indices = []
    ends = []
    policy_calls = 0
    state_restores = 0
    model_cache = {}
    precomputed_actions = None
    if horizon == 1:
        selected_rows = [records[index]["row"] for index, _ in selected]
        action_batches = []
        for batch_start in range(0, len(selected_rows), 1024):
            batch_rows = selected_rows[batch_start : batch_start + 1024]
            obs_batch = torch.from_numpy(np.asarray(source_state[batch_rows])).float()
            with torch.no_grad():
                action_batches.append(expert(obs_batch).cpu().numpy().astype(np.float32))
        precomputed_actions = np.concatenate(action_batches, axis=0)
    for selection_index, (diagnostic_index, kind) in enumerate(selected):
        record = records[diagnostic_index]
        row = record["row"]
        episode = int(record["episode"])
        if episode not in model_cache:
            model_cache[episode] = build_episode_model(
                store, record["episode_start"], record["episode_end"], raw[row]
            )
        model, ctrl, effective = model_cache[episode]
        # Controller arrays are module globals in the audited B3 executor.
        # Restore the cached episode's values before using its model.
        configure_timing_and_controller(store, effective["runtime"])
        data = mujoco.MjData(model)
        set_boundary_state(model, data, ctrl, raw[row])
        state_restores += 1
        builder = Stack.StackCubeObsBuilder(ctrl) if horizon > 1 else None
        if builder is not None:
            seed_observation_history(builder, source_state[row])
        if hasattr(expert, "reset"):
            expert.reset()
        obs = np.asarray(source_state[row], dtype=np.float32).copy()
        raw_starts.append(np.asarray(raw[row], dtype=np.float32))
        source_actions.append(np.asarray(store["data/action"][row : row + horizon], dtype=np.float32))
        for offset in range(horizon):
            prev_states.append(
                np.asarray(
                    source_state[max(row - 1, record["episode_start"])]
                    if offset == 0
                    else states[-1],
                    dtype=np.float32,
                ).copy()
            )
            states.append(obs.copy())
            if precomputed_actions is not None:
                expert_action = precomputed_actions[selection_index]
            else:
                with torch.no_grad():
                    expert_action = (
                        expert(torch.from_numpy(obs).float().unsqueeze(0)).cpu().numpy()[0]
                    )
            policy_calls += 1
            actions.append(expert_action.astype(np.float32))
            # Expert correction uses the task's native privileged grasp guard,
            # exactly as model_31000 evaluation; it does not replay source grip.
            execute_action(model, data, ctrl, expert_action, None)
            if offset + 1 < horizon:
                obs = builder.step(data, expert_action).astype(np.float32)
        kinds.append(1 if kind == "high_error" else 0)
        diagnostic_indices.append(diagnostic_index)
        ends.append(len(states))
        print(
            f"[correct] {selection_index + 1}/{len(selected)} kind={kind} "
            f"row={row} expert_calls={horizon}",
            flush=True,
        )
    return {
        "state": np.asarray(states, dtype=np.float32),
        "prev_state": np.asarray(prev_states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "source_action": np.asarray(source_actions, dtype=np.float32),
        "raw_start": np.asarray(raw_starts, dtype=np.float32),
        "kind_high_error": np.asarray(kinds, dtype=np.uint8),
        "diagnostic_index": np.asarray(diagnostic_indices, dtype=np.int64),
        "episode_ends": np.asarray(ends, dtype=np.int64),
        "policy_calls": policy_calls,
        "state_restores": state_restores,
    }


def write_output(args, store, records, corrections, effective):
    out = Path(args.out).resolve()
    if out.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists (pass --overwrite): {out}")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(out), mode="w")
    diagnostic = root.create_group("diagnostic")
    scalar_fields = (
        "episode",
        "row",
        "source_step",
        "object_pos_error_m",
        "object_rot_error_rad",
        "joint_l2_error_rad",
    )
    for key in scalar_fields:
        diagnostic.create_dataset(key, data=np.asarray([record[key] for record in records]))
    diagnostic.create_dataset(
        "predicted_object_pose",
        data=np.stack([record["predicted_object_pose"] for record in records]),
    )
    diagnostic.create_dataset(
        "reference_object_pose",
        data=np.stack([record["reference_object_pose"] for record in records]),
    )
    correction = root.create_group("correction")
    for key in ("state", "prev_state", "action", "source_action", "raw_start"):
        value = corrections[key]
        correction.create_dataset(key, data=value, chunks=(min(1024, len(value)), *value.shape[1:]))
    meta = root.create_group("meta")
    for key in ("kind_high_error", "diagnostic_index", "episode_ends"):
        meta.create_dataset(key, data=corrections[key])
    checkpoint = str(Path(args.expert).resolve())
    effective_stride = args.horizon if args.source_interface == "action_chunk8" else args.stride
    source_smoke = bool(
        store.attrs.get("smoke_only", False)
        or "smoke" in str(args.source).lower()
        or args.allow_unmarked_chunks
    )
    attrs = {
        "definition": (
            "offline diagnose-then-correct: reset MuJoCo to Isaac s_t once; replay exactly H "
            "recorded source actions with zero policy calls; rank terminal cube-position error; "
            "reset again to the same Isaac s_t; query frozen model_31000 live for H actions"
        ),
        "not_dagger": True,
        "source_zarr": str(Path(args.source).resolve()),
        "source_interface": args.source_interface,
        "horizon": args.horizon,
        "candidate_stride": effective_stride,
        "selector_primary": "terminal cube position L2 in robot-root frame",
        "expert_checkpoint": checkpoint,

        "replay_policy_inference_calls": 0,
        "replay_state_restore_count": len(records),
        "replay_per_step_state_restore": False,
        "correction_policy_inference_calls": corrections["policy_calls"],
        "correction_state_restore_count": corrections["state_restores"],
        "correction_reset_source": "the identical recorded Isaac s_t used for diagnosis",
        "runtime_profile": "B3 exact per-episode runtime arrays from source zarr",
        "effective_config_json": json.dumps(to_jsonable(effective), sort_keys=True),
        "allow_unmarked_chunks": bool(args.allow_unmarked_chunks),
        "smoke_only": source_smoke,
    }
    root.attrs.update(attrs)
    summary = {
        **attrs,
        "candidates": len(records),
        "correction_windows": len(corrections["episode_ends"]),
        "correction_frames": len(corrections["state"]),
        "error_m": {
            "mean": float(np.mean([r["object_pos_error_m"] for r in records])),
            "median": float(np.median([r["object_pos_error_m"] for r in records])),
            "max": float(np.max([r["object_pos_error_m"] for r in records])),
        },
        "out": str(out),
    }
    with (out.parent / f"{out.name}_summary.json").open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(summary), file, indent=2, sort_keys=True)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expert", default=Stack.DEFAULT_CKPT)
    parser.add_argument(
        "--source_interface", choices=("action_step", "action_chunk8", "teacher_smoke"), required=True
    )
    parser.add_argument("--horizon", type=int, choices=(1, 8), required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--high_count", type=int, default=10)
    parser.add_argument("--random_count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--allow_unmarked_chunks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.high_count + args.random_count <= 0:
        raise ValueError("at least one correction window is required")

    store = zarr.open(args.source, mode="r")
    require_source_schema(
        store, args.source_interface, args.horizon, args.allow_unmarked_chunks
    )
    candidates = candidate_rows(
        store, args.source_interface, args.horizon, args.stride, args.max_candidates
    )
    if not candidates:
        raise ValueError("no valid diagnostic candidates")
    records, effective = run_diagnostics(store, candidates, args.horizon)
    selected = select_records(records, args.high_count, args.random_count, args.seed)
    expert = FrankaPolicy.load_from_checkpoint(str(Path(args.expert).resolve()))
    corrections = collect_corrections(store, records, selected, args.horizon, expert)
    write_output(args, store, records, corrections, effective)


if __name__ == "__main__":
    main()
