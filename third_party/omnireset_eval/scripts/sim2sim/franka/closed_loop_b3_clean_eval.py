"""Clean B3 closed-loop PegInsert evaluation in MuJoCo.

This harness deliberately reuses the audited B3 action executor from
``compare_peginsert_continuous_mujoco.py`` while replacing recorded actions
with deterministic policy inference from the live MuJoCo observation.

Episode state is restored once at reset.  After reset, the rollout only writes
controls/generalized forces; it never restores or clips qpos/qvel.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import mujoco
import numpy as np
import torch
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop_eval as CL
import compare_peginsert_continuous_mujoco as Replay
from franka_policy import FrankaPolicy


DEFAULT_ZARR = (
    "sim2sim_cotrain/log/active/peg_b3_direct_sim2sim_20260717/"
    "isaac_b3_480hz_linkorigin_kp500_zeta2_seed53_batch32.zarr"
)
DEFAULT_CHECKPOINT = (
    "co-curation/logs/rsl_rl/franka_fr3_gripper_omnireset_agent/"
    "2026-07-13_20-39-51_peginsert_stage2_b3_4gpu_from4550_b3compat/"
    "model_11200.pt"
)
DEFAULT_OUT = (
    "sim2sim_cotrain/log/active/peg_b3_direct_sim2sim_20260717/"
    "b3_clean_closed_loop_eval_20260718"
)
B3_CENTER_SCALE = np.array([0.01, 0.01, 0.002, 0.02, 0.02, 0.2])
B3_CENTER_KP = np.array([1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0])
B3_CENTER_ZETA = np.ones(6)
B3_CENTER_TORQUE_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
FORMAL_TASK_ID = (
    "OmniReset-FrankaFr3Gripper-RelCartesianOSC-State-Finetune-Play-v0"
)
SOURCE_ROW_PATHS = (
    "data/raw_state",
    "data/arm_scale",
    "data/arm_kp",
    "data/arm_kd",
    "data/arm_torque_max",
    "data/arm_joint_armature",
    "data/arm_joint_friction_static",
    "data/arm_joint_friction_dynamic",
    "data/arm_joint_friction_viscous",
    "data/robot_material_properties",
    "data/robot_body_masses",
    "data/robot_body_coms",
    "data/robot_body_inertias",
    "data/insertive_object_material_properties",
    "data/insertive_object_body_masses",
    "data/insertive_object_body_coms",
    "data/insertive_object_body_inertias",
    "data/receptive_object_material_properties",
    "data/receptive_object_body_masses",
    "data/table_material_properties",
    "data/table_body_masses",
)
SOURCE_REQUIRED_ATTRS = (
    "physics_dt_s",
    "decimation",
    "policy_dt_s",
    "robot_body_names",
)




def _audit_source_dataset(
    store,
    source_path,
    expected_episode_steps,
    allow_incomplete_source_episodes,
):
    """Validate episode boundaries and the exact schema consumed by this harness."""
    missing_paths = [
        path
        for path in ("meta/episode_ends", *SOURCE_ROW_PATHS)
        if path not in store
    ]
    if missing_paths:
        raise KeyError(f"source zarr is missing required paths: {missing_paths}")
    missing_attrs = [key for key in SOURCE_REQUIRED_ATTRS if key not in store.attrs]
    if missing_attrs:
        raise KeyError(f"source zarr is missing required attrs: {missing_attrs}")

    raw_ends = np.asarray(store["meta/episode_ends"])
    if raw_ends.ndim != 1 or not np.issubdtype(raw_ends.dtype, np.integer):
        raise ValueError("meta/episode_ends must be a one-dimensional integer array")
    ends = raw_ends.astype(np.int64, copy=False)
    if len(ends) == 0:
        raise ValueError("source zarr contains no episodes")
    starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
    lengths = ends - starts
    if np.any(lengths <= 0):
        bad = np.flatnonzero(lengths <= 0).tolist()
        raise ValueError(f"source zarr has empty or non-monotonic episodes: {bad}")
    episode_count_attrs = {
        key: int(store.attrs[key])
        for key in ("requested_episodes", "saved_episodes")
        if key in store.attrs
    }
    if (
        "saved_episodes" in episode_count_attrs
        and episode_count_attrs["saved_episodes"] != len(ends)
    ):
        raise ValueError(
            "source attr saved_episodes disagrees with meta/episode_ends: "
            f"{episode_count_attrs['saved_episodes']} != {len(ends)}"
        )
    source_episode_list_attrs = {}
    for key in ("source_episode_ids", "episode_ids", "reset_ids"):
        if key not in store.attrs:
            continue
        values = list(store.attrs[key])
        if len(values) != len(ends):
            raise ValueError(
                f"source attr {key} must contain {len(ends)} entries, "
                f"got {len(values)}"
            )
        source_episode_list_attrs[key] = values

    total_rows = int(ends[-1])
    bad_row_counts = {
        path: int(store[path].shape[0])
        for path in SOURCE_ROW_PATHS
        if not store[path].shape or int(store[path].shape[0]) != total_rows
    }
    if bad_row_counts:
        raise ValueError(
            f"source row arrays must all have {total_rows} rows: {bad_row_counts}"
        )
    if tuple(store["data/raw_state"].shape[1:]) != (57,):
        raise ValueError(
            f"data/raw_state must have shape (T, 57), got "
            f"{store['data/raw_state'].shape}"
        )

    init_state_max_abs = None
    if "data/init_state" in store:
        init_state = np.asarray(store["data/init_state"], dtype=np.float32)
        if init_state.shape != (len(ends), 57):
            raise ValueError(
                f"data/init_state must have shape ({len(ends)}, 57), "
                f"got {init_state.shape}"
            )
        raw_initial = np.asarray(store["data/raw_state"], dtype=np.float32)[starts]
        init_state_max_abs = float(np.max(np.abs(init_state - raw_initial)))
        if init_state_max_abs > 2.0e-6:
            raise ValueError(
                "data/init_state disagrees with data/raw_state at episode starts: "
                f"max abs {init_state_max_abs:.3e}"
            )

    if expected_episode_steps < 0:
        raise ValueError("--expected_episode_steps must be non-negative")
    source_mode = store.attrs.get("mode")
    paired_fresh_unfiltered = source_mode == "isaac_closed_loop_fresh_unfiltered"
    if paired_fresh_unfiltered:
        if allow_incomplete_source_episodes:
            raise ValueError(
                "--allow_incomplete_source_episodes is not permitted for a "
                "formal fresh paired source"
            )
        if "paired_eval_horizon_steps" not in store.attrs:
            raise ValueError(
                "fresh unfiltered source is missing paired_eval_horizon_steps"
            )
        if store.attrs.get("success_filtering") is not False:
            raise ValueError(
                "fresh unfiltered paired source must explicitly set "
                "success_filtering=false"
            )
        formal_attr_expected = {
            "task_id": FORMAL_TASK_ID,
            "controller_profile": "b3_center",
            "decimation": 48,
            "simulation_jacobian_point": "link_origin",
        }
        for key, expected in formal_attr_expected.items():
            observed = store.attrs.get(key)
            if observed != expected:
                raise ValueError(
                    f"fresh paired source attr {key}={observed!r}, "
                    f"expected {expected!r}"
                )
        for key, expected in (
            ("physics_dt_s", 1.0 / 480.0),
            ("policy_dt_s", 0.1),
        ):
            observed = float(store.attrs[key])
            if abs(observed - expected) > 1.0e-12:
                raise ValueError(
                    f"fresh paired source attr {key}={observed}, "
                    f"expected {expected}"
                )
        if (
            "requested_episodes" in episode_count_attrs
            and episode_count_attrs["requested_episodes"] != len(ends)
        ):
            raise ValueError(
                "fresh unfiltered source is incomplete: requested_episodes="
                f"{episode_count_attrs['requested_episodes']} but saved "
                f"{len(ends)} episode boundaries"
            )
        paired_eval_horizon_steps = int(store.attrs["paired_eval_horizon_steps"])
        if paired_eval_horizon_steps <= 0:
            raise ValueError("paired_eval_horizon_steps must be positive")
        if (
            expected_episode_steps
            and expected_episode_steps != paired_eval_horizon_steps
        ):
            raise ValueError(
                "--expected_episode_steps disagrees with source "
                f"paired_eval_horizon_steps={paired_eval_horizon_steps}"
            )
    else:
        paired_eval_horizon_steps = None

    formal_runtime_start_max_abs = {}
    if paired_fresh_unfiltered:
        expected_kd = 2.0 * np.sqrt(B3_CENTER_KP) * B3_CENTER_ZETA
        for path, expected in (
            ("data/arm_scale", B3_CENTER_SCALE),
            ("data/arm_kp", B3_CENTER_KP),
            ("data/arm_kd", expected_kd),
            ("data/arm_torque_max", B3_CENTER_TORQUE_MAX),
        ):
            observed = np.asarray(store[path], dtype=np.float64)[starts]
            max_abs = float(np.max(np.abs(observed - expected)))
            formal_runtime_start_max_abs[path] = max_abs
            if max_abs > 2.0e-6:
                raise ValueError(
                    f"fresh paired source {path} is not B3 center at episode "
                    f"starts: max abs {max_abs:.3e}"
                )

    horizon_attrs = {
        key: int(store.attrs[key])
        for key in (
            "paired_eval_horizon_steps",
            "requested_rollout_steps",
            "episode_steps",
            "source_horizon_steps",
        )
        if key in store.attrs
    }
    if paired_fresh_unfiltered:
        expected_steps = paired_eval_horizon_steps
        expected_steps_source = "source attr paired_eval_horizon_steps"
    elif expected_episode_steps:
        expected_steps = int(expected_episode_steps)
        expected_steps_source = "CLI --expected_episode_steps"
    else:
        expected_steps = None
        expected_steps_source = None
        for key in (
            "requested_rollout_steps",
            "episode_steps",
            "source_horizon_steps",
        ):
            if key in horizon_attrs:
                expected_steps = horizon_attrs[key]
                expected_steps_source = f"source attr {key}"
                break
        if expected_steps is None:
            expected_steps = int(lengths.max())
            expected_steps_source = "maximum source episode length (legacy strict inference)"
    if expected_steps is not None and expected_steps <= 0:
        raise ValueError(
            f"{expected_steps_source} must be positive, got {expected_steps}"
        )

    short_episodes = (
        np.flatnonzero(lengths < expected_steps).tolist()
        if expected_steps is not None
        else []
    )
    oversize_episodes = (
        np.flatnonzero(lengths > expected_steps).tolist()
        if expected_steps is not None
        else []
    )
    incomplete_episodes = sorted(set(short_episodes + oversize_episodes))
    done_steps = None
    early_termination_episodes = []
    invalid_done_step_episodes = []
    done_step_length_mismatch_episodes = []
    unterminated_short_episode_ids = []
    if "done_steps" in store.attrs:
        done_steps = np.asarray(store.attrs["done_steps"], dtype=np.int64)
        if done_steps.shape != (len(ends),):
            raise ValueError(
                f"source attr done_steps must have shape ({len(ends)},), "
                f"got {done_steps.shape}"
            )
        invalid_done = (done_steps < -1) | (done_steps == 0)
        if expected_steps is not None:
            invalid_done |= done_steps > expected_steps
            early_done = (done_steps > 0) & (done_steps < expected_steps)
            early_termination_episodes = np.flatnonzero(early_done).tolist()
        done_step_length_mismatch_episodes = np.flatnonzero(
            (done_steps > 0) & (done_steps != lengths)
        ).tolist()
        if expected_steps is not None:
            unterminated_short_episode_ids = np.flatnonzero(
                (lengths < expected_steps) & (done_steps == -1)
            ).tolist()
        invalid_done_step_episodes = np.flatnonzero(invalid_done).tolist()
    elif paired_fresh_unfiltered:
        raise ValueError(
            "fresh unfiltered paired source must record per-episode done_steps"
        )

    strict_incomplete_episodes = (
        oversize_episodes if paired_fresh_unfiltered else incomplete_episodes
    )
    strict_early_termination_episodes = (
        [] if paired_fresh_unfiltered else early_termination_episodes
    )
    source_anomaly_episodes = sorted(
        set(
            strict_incomplete_episodes
            + strict_early_termination_episodes
            + invalid_done_step_episodes
            + done_step_length_mismatch_episodes
            + unterminated_short_episode_ids
        )
    )
    if source_anomaly_episodes and not allow_incomplete_source_episodes:
        preview = source_anomaly_episodes[:20]
        raise ValueError(
            "source zarr contains invalid episode boundaries/terminations "
            f"{preview}{'...' if len(source_anomaly_episodes) > len(preview) else ''}; "
            f"expected {expected_steps} steps from {expected_steps_source}. "
            "Regenerate the formal source or pass "
            "--allow_incomplete_source_episodes only for diagnostics."
        )

    unique_lengths, length_counts = np.unique(lengths, return_counts=True)
    audit = {
        "path": str(Path(source_path).resolve()),
        "attrs": CL.to_jsonable(dict(store.attrs)),
        "required_row_paths": list(SOURCE_ROW_PATHS),
        "episodes": int(len(ends)),
        "episode_count_attrs": episode_count_attrs,
        "source_episode_list_attrs": source_episode_list_attrs,
        "total_rows": total_rows,
        "episode_length_counts": {
            str(int(length)): int(count)
            for length, count in zip(unique_lengths, length_counts)
        },
        "horizon_attrs": horizon_attrs,
        "source_mode": source_mode,
        "paired_fresh_unfiltered": paired_fresh_unfiltered,
        "paired_eval_horizon_steps": paired_eval_horizon_steps,
        "formal_runtime_start_max_abs": formal_runtime_start_max_abs,
        "expected_episode_steps": expected_steps,
        "expected_episode_steps_source": expected_steps_source,
        "short_episode_ids": short_episodes,
        "oversize_episode_ids": oversize_episodes,
        "incomplete_episode_ids": incomplete_episodes,
        "done_steps": done_steps.tolist() if done_steps is not None else None,
        "early_termination_episode_ids": early_termination_episodes,
        "invalid_done_step_episode_ids": invalid_done_step_episodes,
        "done_step_length_mismatch_episode_ids": (
            done_step_length_mismatch_episodes
        ),
        "unterminated_short_episode_ids": unterminated_short_episode_ids,
        "source_anomaly_episode_ids": source_anomaly_episodes,
        "allow_incomplete_source_episodes": allow_incomplete_source_episodes,
        "init_state_present": "data/init_state" in store,
        "init_state_vs_raw_start_max_abs": init_state_max_abs,

    }
    return starts, ends, audit


def _rollout_step_count(source_episode_length, source_dataset, max_steps):
    """Decouple formal paired MuJoCo horizon from native Isaac termination."""
    if max_steps < 0:
        raise ValueError("--max_steps must be non-negative")
    rollout_horizon = (
        source_dataset["paired_eval_horizon_steps"]
        if source_dataset["paired_fresh_unfiltered"]
        else source_episode_length
    )
    steps = min(
        rollout_horizon,
        max_steps if max_steps > 0 else rollout_horizon,
    )
    return rollout_horizon, steps


def _render_frame(renderer, data, camera_ids, label):
    images = []
    for camera_id in camera_ids:
        renderer.update_scene(data, camera=camera_id)
        images.append(renderer.render())
    gap = np.zeros((images[0].shape[0], 6, 3), dtype=images[0].dtype)
    frame = np.concatenate([images[0], gap, images[1]], axis=1)
    try:
        from PIL import Image, ImageDraw

        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil)
        draw.rectangle([0, 0, pil.width, 24], fill=(0, 0, 0))
        draw.text((8, 5), label, fill=(255, 255, 255))
        return np.asarray(pil)
    except Exception:
        return frame


def _configure_controller(runtime, controller_source):
    """Select OSC gains without changing the zarr-recorded torque limits."""
    if controller_source == "zarr":
        scale = runtime["arm_scale"]
        kp = runtime["arm_kp"]
        zeta = runtime["arm_kd"] / (2.0 * np.sqrt(kp))
        gain_source = "per-episode zarr runtime"
    elif controller_source == "b3_center":
        scale = B3_CENTER_SCALE.copy()
        kp = B3_CENTER_KP.copy()
        zeta = B3_CENTER_ZETA.copy()
        gain_source = "fixed B3 training-distribution center"
    else:
        raise ValueError(f"unknown controller source: {controller_source}")

    kd = 2.0 * np.sqrt(kp) * zeta
    torque_max = runtime["arm_torque_max"].copy()
    CL.set_controller_gains(scale, kp, zeta)
    CL.TAU_MAX = torque_max
    return {
        "source": controller_source,
        "gain_source": gain_source,
        "scale": np.asarray(scale, dtype=np.float64).copy(),
        "kp": np.asarray(kp, dtype=np.float64).copy(),
        "zeta": np.asarray(zeta, dtype=np.float64).copy(),
        "kd": np.asarray(kd, dtype=np.float64).copy(),
        "torque_max": torque_max,
        "torque_max_source": "per-episode zarr runtime arm_torque_max",
    }


def _episode_model(
    store,
    start,
    _end,
    raw0,
    physics_substeps,
    hole_collision,
    peg_mass_kg,
    controller_source,
):
    # A paired fresh source may terminate before the evaluation horizon.  Only
    # its initial physical realization is needed to start the independent
    # MuJoCo rollout, so never couple controller/scene loading to source length.
    runtime = Replay.load_isaac_runtime(store, start, start + 1)
    scene = Replay.load_runtime_scene_properties(store, start, start + 1)

    controller = _configure_controller(runtime, controller_source)

    _, _, hole_pos, hole_quat = Replay.pose_in_robot_root(raw0)
    profile = Replay.b3_center_profile(
        physics_substeps=physics_substeps,
        hole_collision=hole_collision,
        gripper_contact_friction=[1.2, 1.2, 0.0, 0.0, 0.0],
    )
    model = CL.build_model(hole_pos, hole_quat, profile)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    ctrl = CL.Controller(model)
    peg_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "peg")
    ctrl.peg_qpos = int(model.jnt_qposadr[peg_joint])

    # The runtime arrays recorded inside the source rollout are authoritative.
    model.dof_armature[:7] = runtime["arm_joint_armature"]
    model.dof_damping[:7] = runtime["arm_joint_friction_viscous"]
    model.dof_frictionloss[:7] = runtime["arm_joint_friction_dynamic"]
    masses = Replay.apply_runtime_body_masses(model, ctrl, scene, include_robot=True)
    runtime_peg_mass_kg = float(model.body_mass[ctrl.peg])
    if peg_mass_kg is not None:
        if not np.isfinite(peg_mass_kg) or peg_mass_kg <= 0.0:
            raise ValueError("--peg_mass_kg must be a finite positive number")
        inertia_scale = float(peg_mass_kg / runtime_peg_mass_kg)
        model.body_inertia[ctrl.peg] *= inertia_scale
        model.body_mass[ctrl.peg] = peg_mass_kg
        mujoco.mj_setConst(model, mujoco.MjData(model))
    else:
        inertia_scale = 1.0
    materials = Replay.apply_runtime_contact_materials(model, scene)
    Replay.disable_tendon_gripper(model)

    effective = {
        "profile": CL.to_jsonable(profile),
        "runtime": CL.to_jsonable(runtime),
        "controller": CL.to_jsonable(controller),
        "runtime_masses_applied_kg": CL.to_jsonable(masses),
        "peg_mass": {
            "runtime_kg": runtime_peg_mass_kg,
            "override_kg": peg_mass_kg,
            "effective_kg": float(model.body_mass[ctrl.peg]),
            "inertia_scale_from_runtime": inertia_scale,
        },
        "runtime_contact_materials_applied": CL.to_jsonable(materials),
        "compiled": CL.to_jsonable(Replay.effective_model_config(model, ctrl)),
        "integrator": "implicitfast",
        "jacobian_point": "link_origin",
        "arm_friction_mode": "runtime dynamic Coulomb + runtime viscous damping",
        "gripper": {
            "command": "live privileged grasp guard; policy scalar ignored",
            "actuator": "independent 1000*(target-q)-14*dq, clipped to +/-60 N",
            "velocity_limits_mps": [0.04, 0.04],
            "velocity_limit_enforcement": (
                "contact-aware actuator-force projection; no qpos/qvel writes"
            ),
        },
    }
    return model, ctrl, hole_pos, hole_quat, effective


def run_episode(
    store,
    policy,
    episode,
    start,
    end,
    steps,
    physics_substeps,
    out_dir,
    render,
    width,
    height,
    fps,
    hole_collision,
    peg_mass_kg,
    controller_source,
):
    raw0 = np.asarray(store["data/raw_state"][start], dtype=np.float64)
    model, ctrl, hole_pos, hole_quat, effective = _episode_model(
        store,
        start,
        end,
        raw0,
        physics_substeps,
        hole_collision,
        peg_mass_kg,
        controller_source,
    )
    data = mujoco.MjData(model)

    # This is the only rollout-state restore in the entire episode.
    Replay.set_raw_state(model, data, ctrl, raw0)
    obs_builder = CL.ObsBuilder(ctrl)
    obs_builder.reset(data)
    prev_action = np.zeros(7, dtype=np.float32)

    writer = None
    renderer = None
    if render:
        renderer = mujoco.Renderer(model, height, width)
        camera_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in ("cam_front", "cam_side")
        ]
        writer = imageio.get_writer(
            out_dir / f"closedloop_ep{episode}.mp4", fps=fps, macro_block_size=1
        )

    rows = []
    ever_closed = False
    stable_release_streak = 0
    try:
        for step in range(steps):
            obs = obs_builder.step(data, prev_action)
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).cpu().numpy()[0]
            prev_action = action.astype(np.float32)

            close = Replay.step_action(
                model,
                data,
                ctrl,
                action,
                finger_velocity_limits=np.array([0.04, 0.04], dtype=np.float64),
                independent_gripper=True,
                gripper_close_override=None,
                jacobian_point="link_origin",
                nullspace_stiffness=0.0,
                nullspace_damping_ratio=1.0,
                action_reference_blend=0.0,
                bias_compensation_scale=0.0,
                effort_scale=np.ones(7, dtype=np.float64),
            )
            ever_closed = ever_closed or close
            (
                position_error,
                orientation_error,
                square_yaw_error,
                official_pose,
                square_yaw_pose,
                peg_in_hole,
            ) = Replay.assembly_metrics(data, ctrl, hole_pos, hole_quat)
            robot_contact = ctrl.peg_robot_contact(data)
            if ever_closed and not close and official_pose and not robot_contact:
                stable_release_streak += 1
            else:
                stable_release_streak = 0

            row = {
                "episode": episode,
                "step": step + 1,
                "assembly_pos_m": position_error,
                "assembly_roll_pitch_rad": orientation_error,
                "assembly_square_yaw_rot_rad": square_yaw_error,
                "peg_in_hole_x_m": float(peg_in_hole[0]),
                "peg_in_hole_y_m": float(peg_in_hole[1]),
                "peg_in_hole_z_m": float(peg_in_hole[2]),
                "official_pose": int(official_pose),
                "square_yaw_pose": int(square_yaw_pose),
                "close_command": int(close),
                "robot_contact": int(robot_contact),
                "stable_release_streak": stable_release_streak,
                **{f"action_{i}": float(action[i]) for i in range(7)},
            }
            rows.append(row)

            if writer is not None:
                label = (
                    f"B3 clean CLOSED LOOP | ep={episode} step={step + 1}/{steps} | "
                    f"pos={position_error * 1000:.1f}mm "
                    f"rp={orientation_error:.3f} close={int(close)}"
                )
                writer.append_data(_render_frame(renderer, data, camera_ids, label))
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    trace_path = out_dir / f"closedloop_ep{episode}.csv"
    with trace_path.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    task = Replay.summarize_task(rows)
    result = {
        "episode": episode,
        "source_indices": [start, end],
        "steps": steps,
        "official_pose_steps": task["official_pose_steps"],
        "first_official_pose_step": task["first_official_pose_step"],
        "final_position_error_m": task["final_position_error_m"],
        "final_roll_pitch_error_rad": task["final_official_orientation_error_rad"],
        "final_official_pose": task["final_official_pose"],
        "best_position_error_m": task["best_position_error_m"],
        "max_stable_release_steps": task["max_stable_release_steps"],
        "stable_release_success": task["stable_release_success"],
        "final_gripper_close": task["final_gripper_close"],
        "final_robot_contact": task["final_robot_contact"],
        "trace": str(trace_path.resolve()),
        "video": str((out_dir / f"closedloop_ep{episode}.mp4").resolve()) if render else None,
        "effective_config": effective,
    }
    return result


def run(args):
    if args.torch_threads < 0 or args.torch_interop_threads < 0:
        raise ValueError("Torch thread counts must be non-negative")
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    if args.torch_interop_threads:
        try:
            torch.set_num_interop_threads(args.torch_interop_threads)
        except RuntimeError:
            if torch.get_num_interop_threads() != args.torch_interop_threads:
                raise
    out_dir = Path(args.out).resolve()
    source_path = str(Path(args.zarr).resolve())
    store = zarr.open(source_path, mode="r")
    starts, ends, source_dataset = _audit_source_dataset(
        store,
        source_path,
        args.expected_episode_steps,
        args.allow_incomplete_source_episodes,
    )
    checkpoint = str(Path(args.checkpoint).resolve())
    if (
        source_dataset["paired_fresh_unfiltered"]
        and args.controller_source != "b3_center"
    ):
        raise ValueError(
            "fresh paired source requires --controller_source b3_center"
        )
    policy = FrankaPolicy.load_from_checkpoint(checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_physics_dt = float(store.attrs["physics_dt_s"])
    source_decimation = int(store.attrs["decimation"])
    source_policy_dt = float(store.attrs["policy_dt_s"])
    if abs(source_physics_dt * source_decimation - source_policy_dt) > 1.0e-12:
        raise ValueError("inconsistent source timing metadata")
    CL.SIM_DT = source_physics_dt
    CL.DECIM = source_decimation

    if args.episode_ids is not None:
        if not args.episode_ids:
            raise ValueError("--episode_ids was provided without any episode IDs")
        if len(set(args.episode_ids)) != len(args.episode_ids):
            raise ValueError("--episode_ids contains duplicates")
        episode_ids = args.episode_ids
    else:
        episode_ids = list(range(min(args.episodes, len(ends))))
    for episode in episode_ids:
        if episode < 0 or episode >= len(ends):
            raise ValueError(f"episode {episode} outside [0, {len(ends) - 1}]")

    results = []
    for episode in episode_ids:
        start, end = int(starts[episode]), int(ends[episode])
        source_episode_length = end - start
        rollout_horizon, steps = _rollout_step_count(
            source_episode_length,
            source_dataset,
            args.max_steps,
        )
        print(
            f"[b3-closed-loop] episode={episode} source_steps={source_episode_length} "
            f"rollout_steps={steps} "
            f"render={episode == args.render_episode}",
            flush=True,
        )
        result = run_episode(
            store,
            policy,
            episode,
            start,
            end,
            steps,
            args.physics_substeps,
            out_dir,
            episode == args.render_episode,
            args.width,
            args.height,
            args.fps,
            args.hole_collision,
            args.peg_mass_kg,
            args.controller_source,
        )
        source_done_step = (
            source_dataset["done_steps"][episode]
            if source_dataset["done_steps"] is not None
            else None
        )
        result["source_episode_length"] = source_episode_length
        result["source_done_step"] = source_done_step
        result["source_ids"] = {
            key: values[episode]
            for key, values in source_dataset["source_episode_list_attrs"].items()
        }
        result["rollout_horizon_steps"] = rollout_horizon
        result["paired_eval_horizon_steps"] = (
            source_dataset["paired_eval_horizon_steps"]
        )
        results.append(result)
        print(
            f"[b3-closed-loop] ep={episode} pose_steps={result['official_pose_steps']} "
            f"stable={result['stable_release_success']} "
            f"final={result['final_position_error_m'] * 1000:.3f}mm",
            flush=True,
        )

    summary_rows = [
        {key: value for key, value in result.items() if key not in ("effective_config",)}
        for result in results
    ]
    with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    summary = {
        "definition": (
            "true closed-loop deterministic B3 policy evaluation: live MuJoCo state -> "
            f"200-D observation/history -> {Path(checkpoint).name} "
            f"(iteration {policy.ckpt_iter}) action every policy step"
        ),
        "not_replay": (
            "data/action and gripper_processed_action are never consumed by this harness"
        ),
        "state_mutation_audit": (
            "one raw-state restore at episode initialization; no subsequent qpos/qvel writes, "
            "clipping, teleport, weld, reset, or recorded torque playback"
        ),
        "zarr": str(Path(args.zarr).resolve()),
        "zarr_signature": CL.zarr_signature(store),
        "checkpoint": checkpoint,


        "checkpoint_iteration": policy.ckpt_iter,
        "evaluator_script": str(Path(__file__).resolve()),

        "torch_config": {
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        },
        "source_dataset": source_dataset,
        "selection": {
            "episode_ids": episode_ids,
            "max_steps_override": args.max_steps,
            "rollout_steps_by_episode": [
                {"episode": result["episode"], "steps": result["steps"]}
                for result in results
            ],
        },
        "controller": {
            "source": args.controller_source,
            "effective_by_episode": [
                {
                    "episode": result["episode"],
                    **result["effective_config"]["controller"],
                }
                for result in results
            ],
        },
        "ablation": {
            "hole_collision": args.hole_collision,
            "peg_mass_kg": args.peg_mass_kg,
            "peg_mass_source": (
                "fixed override with inertia scaled from runtime"
                if args.peg_mass_kg is not None
                else "per-episode zarr runtime mass and inertia"
            ),
        },
        "timing": {
            "source_physics_dt_s": source_physics_dt,
            "source_decimation": source_decimation,
            "source_policy_dt_s": source_policy_dt,
            "mujoco_integration_substeps_per_source_tick": args.physics_substeps,
        },
        "success_definition": {
            "position_m_lt": CL.SUC_POS,
            "abs_roll_plus_abs_pitch_rad_lt": CL.SUC_ROT,
            "yaw": "ignored, matching UWLab ProgressContext",
            "stable_release_steps_gte": CL.STABLE_RELEASE_STEPS,
        },
        "episodes_evaluated": len(results),
        "episodes_with_any_official_pose": int(
            sum(result["official_pose_steps"] > 0 for result in results)
        ),
        "stable_release_successes": int(
            sum(result["stable_release_success"] for result in results)
        ),
        "results": results,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(CL.to_jsonable(summary), file, indent=2)
    with (out_dir / "effective_config.json").open("w", encoding="utf-8") as file:
        json.dump(CL.to_jsonable(results[0]["effective_config"]), file, indent=2)
    print(json.dumps(CL.to_jsonable({
        "episodes_evaluated": summary["episodes_evaluated"],
        "episodes_with_any_official_pose": summary["episodes_with_any_official_pose"],
        "stable_release_successes": summary["stable_release_successes"],
        "out": str(out_dir),
    }), indent=2))


def aggregate_parts(args):
    """Merge independently evaluated episode directories without rerunning physics."""
    parts_dir = Path(args.aggregate_parts).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.episode_ids is not None:
        if not args.episode_ids:
            raise ValueError("--episode_ids was provided without any episode IDs")
        if len(set(args.episode_ids)) != len(args.episode_ids):
            raise ValueError("--episode_ids contains duplicates")
        episode_ids = args.episode_ids
    else:
        episode_ids = list(range(args.episodes))
    part_summaries = []
    results = []
    for episode in episode_ids:
        path = parts_dir / f"ep{episode}" / "summary.json"
        if not path.exists():
            raise FileNotFoundError(f"missing episode result: {path}")
        with path.open(encoding="utf-8") as file:
            part = json.load(file)
        if len(part["results"]) != 1 or part["results"][0]["episode"] != episode:
            raise ValueError(f"unexpected result payload in {path}")
        part_summaries.append(part)
        result = dict(part["results"][0])
        result.pop("effective_config", None)
        results.append(result)

    first = part_summaries[0]
    invariant_keys = (
        "definition",
        "not_replay",
        "state_mutation_audit",
        "zarr",
        "zarr_signature",
        "checkpoint",

        "checkpoint_iteration",
        "ablation",
        "timing",
        "success_definition",
    )
    optional_invariant_keys = (

        "evaluator_script",

        "source_dataset",
        "torch_config",
    )
    for key in (*invariant_keys, *optional_invariant_keys):
        present = [key in part for part in part_summaries]
        if any(present) and not all(present):
            raise ValueError(f"only some part summaries contain invariant {key}")
        if all(present):
            expected = json.dumps(
                first[key], sort_keys=True, separators=(",", ":")
            )
            for episode, part in zip(episode_ids[1:], part_summaries[1:]):
                observed = json.dumps(
                    part[key], sort_keys=True, separators=(",", ":")
                )
                if observed != expected:
                    raise ValueError(
                        f"part ep{episode} differs in aggregate invariant {key}"
                    )

    summary = {
        key: first[key]
        for key in invariant_keys
    }
    for key in optional_invariant_keys:
        if key in first:
            summary[key] = first[key]

    selection_payloads = [part.get("selection") for part in part_summaries]
    if any(payload is not None for payload in selection_payloads):
        if not all(payload is not None for payload in selection_payloads):
            raise ValueError("only some part summaries record episode selection")
        for episode, payload in zip(episode_ids, selection_payloads):
            if payload["episode_ids"] != [episode]:
                raise ValueError(
                    f"part ep{episode} selection is {payload['episode_ids']}, "
                    "expected exactly its directory episode"
                )
        max_steps_overrides = {
            payload["max_steps_override"] for payload in selection_payloads
        }
        if len(max_steps_overrides) != 1:
            raise ValueError("part summaries use inconsistent max_steps overrides")
        summary["selection"] = {
            "episode_ids": [result["episode"] for result in results],
            "max_steps_override": first["selection"]["max_steps_override"],
            "rollout_steps_by_episode": [
                rollout
                for payload in selection_payloads
                for rollout in payload["rollout_steps_by_episode"]
            ],
        }

    controller_payloads = [part.get("controller") for part in part_summaries]
    if any(payload is not None for payload in controller_payloads):
        if not all(payload is not None for payload in controller_payloads):
            raise ValueError("only some part summaries record controller configuration")
        controller_sources = {payload["source"] for payload in controller_payloads}
        if len(controller_sources) != 1:
            raise ValueError("part summaries use inconsistent controller sources")
        for episode, payload in zip(episode_ids, controller_payloads):
            effective_episodes = [
                effective["episode"]
                for effective in payload["effective_by_episode"]
            ]
            if effective_episodes != [episode]:
                raise ValueError(
                    f"part ep{episode} controller payload is for "
                    f"episodes {effective_episodes}"
                )
        summary["controller"] = {
            "source": first["controller"]["source"],
            "effective_by_episode": [
                effective
                for part in part_summaries
                for effective in part["controller"]["effective_by_episode"]
            ],
        }
    summary.update({
        "parts_dir": str(parts_dir),
        "episodes_evaluated": len(results),
        "episodes_with_any_official_pose": int(
            sum(result["official_pose_steps"] > 0 for result in results)
        ),
        "stable_release_successes": int(
            sum(result["stable_release_success"] for result in results)
        ),
        "official_pose_rate": float(
            sum(result["official_pose_steps"] > 0 for result in results) / len(results)
        ),
        "stable_release_success_rate": float(
            sum(result["stable_release_success"] for result in results) / len(results)
        ),
        "stable_release_success_episodes": [
            result["episode"] for result in results if result["stable_release_success"]
        ],
        "official_pose_but_unstable_episodes": [
            result["episode"]
            for result in results
            if result["official_pose_steps"] > 0 and not result["stable_release_success"]
        ],
        "effective_config": part_summaries[0]["results"][0]["effective_config"],
        "effective_config_representative_episode": episode_ids[0],
        "effective_config_by_episode": [
            {
                "episode": episode,
                "effective_config": part["results"][0]["effective_config"],
            }
            for episode, part in zip(episode_ids, part_summaries)
        ],
        "results": results,
    })
    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(CL.to_jsonable(summary), file, indent=2)
    with (out_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (out_dir / "effective_config.json").open("w", encoding="utf-8") as file:
        json.dump(CL.to_jsonable(summary["effective_config"]), file, indent=2)
    print(json.dumps({
        "episodes_evaluated": summary["episodes_evaluated"],
        "episodes_with_any_official_pose": summary["episodes_with_any_official_pose"],
        "official_pose_rate": summary["official_pose_rate"],
        "stable_release_successes": summary["stable_release_successes"],
        "stable_release_success_rate": summary["stable_release_success_rate"],
        "out": str(out_dir),
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default=DEFAULT_ZARR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--episodes", type=int, default=62)
    parser.add_argument("--episode_ids", type=int, nargs="*")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument(
        "--expected_episode_steps",
        type=int,
        default=0,
        help=(
            "paired MuJoCo horizon and strict horizon for legacy sources; "
            "0 infers paired/requested/episode/source horizon attrs"
        ),
    )
    parser.add_argument(
        "--allow_incomplete_source_episodes",
        action="store_true",
        help=(
            "diagnostic only: permit variable-length or early-terminated source "
            "episodes instead of failing the dataset preflight"
        ),
    )
    parser.add_argument("--physics_substeps", type=int, default=16)
    parser.add_argument(
        "--torch_threads",
        type=int,
        default=0,
        help="if positive, explicitly fix Torch intra-op threads",
    )
    parser.add_argument(
        "--torch_interop_threads",
        type=int,
        default=0,
        help="if positive, explicitly fix Torch inter-op threads",
    )
    parser.add_argument(
        "--controller_source",
        choices=["zarr", "b3_center"],
        default="zarr",
        help=(
            "OSC gain/action-scale source; b3_center fixes scale/Kp/zeta at the "
            "B3 center while torque limits still come from each zarr episode"
        ),
    )
    parser.add_argument(
        "--hole_collision",
        choices=["box_ring_big", "box_ring", "mesh_sdf_big", "mesh_sdf"],
        default="box_ring_big",
    )
    parser.add_argument(
        "--peg_mass_kg",
        type=float,
        default=None,
        help="fixed peg mass; default keeps per-episode zarr runtime mass/inertia",
    )
    parser.add_argument("--render_episode", type=int, default=-1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--aggregate_parts",
        default=None,
        help="merge epN/summary.json directories into --out without rerunning physics",
    )
    args = parser.parse_args()
    if args.aggregate_parts:
        aggregate_parts(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
