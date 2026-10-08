"""Evaluate CupCake policies with live MuJoCo observations and frozen resets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import closed_loop_eval as CL
import compare_cupcake_mujoco as Replay
import compare_peginsert_continuous_mujoco as B3
import cupcake_mujoco_model as CupCake
import quat_utils as Q
from franka_policy import FrankaPolicy


DEFAULT_ZARR = "datasets/cupcake_sim2sim_t0_20260815/isaac_fit_b3center.zarr"
DEFAULT_CHECKPOINT = (
    str(Path(__file__).resolve().parents[5] / "artifacts/experts/sim2sim/cupcake/model_20400.pt")
)
DEFAULT_OUT = "log/active/cupcake_sim2sim_20260815/closed_loop_smoke"
STABLE_SUCCESS_STEPS = 5


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_source(
    store, checkpoint: Path
) -> tuple[np.ndarray, np.ndarray, dict]:
    required = (
        "data/raw_state",
        "data/state",
        "meta/episode_ends",
        "meta/reset_indices",
        "meta/success_seen",
    )
    missing = [path for path in required if path not in store]
    if missing:
        raise KeyError(f"source Zarr is missing required paths: {missing}")
    if store["data/raw_state"].shape[1:] != (57,):
        raise ValueError("data/raw_state must be 57-D")
    if store["data/state"].shape[1:] != (200,):
        raise ValueError("data/state must be 200-D")

    ends = np.asarray(store["meta/episode_ends"], dtype=np.int64)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
    lengths = ends - starts
    if len(ends) == 0 or np.any(lengths <= 0):
        raise ValueError("source episode boundaries are empty or non-monotonic")
    if int(ends[-1]) != int(store["data/raw_state"].shape[0]):
        raise ValueError("source episode boundaries do not cover raw_state")

    physics_dt = float(store.attrs["physics_dt_s"])
    decimation = int(store.attrs["decimation"])
    policy_dt = float(store.attrs["policy_dt_s"])
    if abs(physics_dt * decimation - policy_dt) > 1.0e-12:
        raise ValueError("source physics_dt * decimation != policy_dt")

    return starts, ends, {
        "episode_lengths": lengths.tolist(),
        "checkpoint": str(checkpoint),
        "timing": {
            "physics_dt_s": physics_dt,
            "decimation": decimation,
            "policy_dt_s": policy_dt,
        },
    }


def render_frame(renderer, data, cameras: tuple[str, ...]) -> np.ndarray:
    images = []
    for camera in cameras:
        renderer.update_scene(data, camera=camera)
        images.append(renderer.render().copy())
    return np.concatenate(images, axis=1)


def add_body_axes(renderer, data, body_id: int, length: float = 0.13) -> None:
    # Draw the triad beside the object so the opaque visual mesh cannot hide it.
    origin = data.xpos[body_id] + np.array([-0.11, 0.0, 0.0])
    rotation = data.xmat[body_id].reshape(3, 3)
    colors = (
        np.array([1.0, 0.1, 0.1, 1.0], dtype=np.float32),
        np.array([0.1, 1.0, 0.1, 1.0], dtype=np.float32),
        np.array([0.1, 0.3, 1.0, 1.0], dtype=np.float32),
    )
    for axis, color in enumerate(colors):
        geom = renderer.scene.geoms[renderer.scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).ravel(),
            color,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            0.006,
            origin,
            origin + length * rotation[:, axis],
        )
        renderer.scene.ngeom += 1


def closeup_camera(data, body_id: int) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.xpos[body_id] + np.array([0.0, 0.0, 0.04])
    camera.distance = 0.42
    camera.azimuth = 135.0
    camera.elevation = -18.0
    return camera


def quaternion_yaw(quaternion: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


def render_without_robot(
    renderer, model, data, camera, axis_body_id: int | None = None
) -> np.ndarray:
    robot_body_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in [
            *(f"link{index}" for index in range(8)),
            "hand",
            "left_finger",
            "right_finger",
        ]
    }
    robot_geom_ids = np.asarray(
        [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in robot_body_ids
        ],
        dtype=np.int64,
    )
    original_rgba = model.geom_rgba[robot_geom_ids].copy()
    try:
        model.geom_rgba[robot_geom_ids, 3] = 0.0
        renderer.update_scene(data, camera=camera)
        if axis_body_id is not None:
            add_body_axes(renderer, data, axis_body_id)
        return renderer.render().copy()
    finally:
        model.geom_rgba[robot_geom_ids] = original_rgba


def run_episode(
    store,
    policy: FrankaPolicy,
    episode: int,
    start: int,
    end: int,
    steps: int,
    physics_substeps: int,
    out: Path,
    render: bool,
    width: int,
    height: int,
    fps: int,
    diagnostic_render: bool,
    render_cameras: tuple[str, ...],
    friction_combine: str,
    cupcake_collision: str,
    cupcake_plate_collision: str,
    hand_collision: str,
    finger_collision: str,
) -> dict:
    raw0 = np.asarray(store["data/raw_state"][start], dtype=np.float64)
    profile = B3.b3_center_profile(physics_substeps=physics_substeps)
    profile["cupcake_collision"] = cupcake_collision
    profile["cupcake_plate_collision"] = cupcake_plate_collision
    profile["hand_collision"] = hand_collision
    profile["finger_collision"] = finger_collision
    model = CupCake.build_model(raw0, profile_cfg=profile)
    ctrl = CupCake.Controller(model)
    effective = Replay.configure_episode(
        model,
        ctrl,
        store,
        start,
        end,
        raw0,
        friction_combine=friction_combine,
    )
    data = mujoco.MjData(model)
    geometry = Replay.reset_geometry_audit(model, data, ctrl, raw0, None)

    obs_builder = CL.ObsBuilder(ctrl)
    obs_builder.reset(data)
    prev_action = np.zeros(7, dtype=np.float32)
    rows = []
    success_streak = 0
    max_success_streak = 0
    relaxed_success_streak = 0
    max_relaxed_success_streak = 0
    finite = True
    cupcake_visual_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "cupcake_visual"
    )

    renderer = None
    video = None
    writer = None
    diagnostic_video = None
    diagnostic_writer = None
    axes_video = None
    axes_writer = None
    if render:
        renderer = mujoco.Renderer(model, height=height, width=width)
        video = out / f"closedloop_ep{episode:03d}.mp4"
        writer = imageio.get_writer(video, fps=fps, macro_block_size=1)
        if diagnostic_render:
            diagnostic_video = out / f"closedloop_ep{episode:03d}_no_robot.mp4"
            diagnostic_writer = imageio.get_writer(
                diagnostic_video, fps=fps, macro_block_size=1
            )
            axes_video = out / f"closedloop_ep{episode:03d}_no_robot_axes.mp4"
            axes_writer = imageio.get_writer(
                axes_video, fps=fps, macro_block_size=1
            )

    try:
        for step in range(steps):
            obs = obs_builder.step(data, prev_action)
            action = (
                policy(
                    torch.from_numpy(obs)
                    .unsqueeze(0)
                    .to(policy.obs_mean.device)
                )
                .cpu()
                .numpy()[0]
            )
            prev_action = action.astype(np.float32)
            close = B3.step_action(
                model,
                data,
                ctrl,
                action,
                finger_velocity_limits=None,
                independent_gripper=False,
                gripper_close_override=None,
                jacobian_point="physx_com",
                nullspace_stiffness=0.0,
                nullspace_damping_ratio=1.0,
                action_reference_blend=0.0,
                bias_compensation_scale=0.0,
                effort_scale=np.ones(7, dtype=np.float64),
            )
            metrics = CupCake.success_metrics(data, ctrl)
            success_streak = success_streak + 1 if metrics["strict_success"] else 0
            max_success_streak = max(max_success_streak, success_streak)
            relaxed_success_streak = (
                relaxed_success_streak + 1 if metrics["relaxed_success"] else 0
            )
            max_relaxed_success_streak = max(
                max_relaxed_success_streak, relaxed_success_streak
            )
            contact_bodies = sorted(ctrl.insertive_robot_contact_bodies(data))
            robot_contact_distances = []
            for contact in data.contact[: data.ncon]:
                if contact.geom1 in ctrl.insertive_geom_ids:
                    other_geom = contact.geom2
                elif contact.geom2 in ctrl.insertive_geom_ids:
                    other_geom = contact.geom1
                else:
                    continue
                if int(model.geom_bodyid[other_geom]) in ctrl.robot_body_ids:
                    robot_contact_distances.append(float(contact.dist))
            hand_pos = data.xpos[ctrl.hand]
            hand_quat = data.xquat[ctrl.hand]
            cupcake_in_hand = Q.quat_apply(
                Q.quat_inv(hand_quat),
                data.xpos[ctrl.insertive] - hand_pos,
            )
            _, cupcake_in_plate_quat = Q.subtract_frame_transforms(
                data.xpos[ctrl.receptive],
                data.xquat[ctrl.receptive],
                data.xpos[ctrl.insertive],
                data.xquat[ctrl.insertive],
            )
            cupcake_up_in_plate = Q.quat_apply(
                cupcake_in_plate_quat, np.array([0.0, 0.0, 1.0])
            )
            plate_rotation = data.xmat[ctrl.receptive].reshape(3, 3)
            cupcake_visual_up_in_plate = (
                plate_rotation.T
                @ data.geom_xmat[cupcake_visual_geom].reshape(3, 3)[:, 2]
            )
            step_finite = bool(
                np.all(np.isfinite(data.qpos))
                and np.all(np.isfinite(data.qvel))
                and np.all(np.isfinite(action))
            )
            finite = finite and step_finite
            row = {
                "episode": episode,
                "step": step + 1,
                "position_error_m": metrics["position_error_m"],
                "orientation_xy_error_rad": metrics["orientation_xy_error_rad"],
                "strict_success": int(metrics["strict_success"]),
                "relaxed_success": int(metrics["relaxed_success"]),
                "success_streak": success_streak,
                "relaxed_success_streak": relaxed_success_streak,
                "close_command": int(close),
                "robot_contact": int(bool(contact_bodies)),
                "robot_contact_bodies": "+".join(contact_bodies),
                "robot_contact_min_distance_m": (
                    min(robot_contact_distances)
                    if robot_contact_distances
                    else float("nan")
                ),
                "contact_count": int(data.ncon),
                "finite": int(step_finite),
                **{f"action_{index}": float(value) for index, value in enumerate(action)},
                **{f"arm_q_{index}": float(value) for index, value in enumerate(data.qpos[:7])},
                "cupcake_x_m": float(data.xpos[ctrl.insertive, 0]),
                "cupcake_y_m": float(data.xpos[ctrl.insertive, 1]),
                "cupcake_z_m": float(data.xpos[ctrl.insertive, 2]),
                **{
                    f"cupcake_quat_{name}": float(value)
                    for name, value in zip(
                        "wxyz", data.xquat[ctrl.insertive], strict=True
                    )
                },
                "cupcake_in_hand_x_m": float(cupcake_in_hand[0]),
                "cupcake_in_hand_y_m": float(cupcake_in_hand[1]),
                "cupcake_in_hand_z_m": float(cupcake_in_hand[2]),
                "cupcake_up_in_plate_x": float(cupcake_up_in_plate[0]),
                "cupcake_up_in_plate_y": float(cupcake_up_in_plate[1]),
                "cupcake_up_in_plate_z": float(cupcake_up_in_plate[2]),
                "cupcake_visual_up_in_plate_x": float(
                    cupcake_visual_up_in_plate[0]
                ),
                "cupcake_visual_up_in_plate_y": float(
                    cupcake_visual_up_in_plate[1]
                ),
                "cupcake_visual_up_in_plate_z": float(
                    cupcake_visual_up_in_plate[2]
                ),
                "cupcake_yaw_in_plate_rad": quaternion_yaw(
                    cupcake_in_plate_quat
                ),
                "left_finger_q_m": float(data.qpos[7]),
                "right_finger_q_m": float(data.qpos[8]),
            }
            rows.append(row)
            if writer is not None:
                writer.append_data(render_frame(renderer, data, render_cameras))
                if diagnostic_writer is not None:
                    diagnostic_writer.append_data(
                        render_without_robot(
                            renderer, model, data, "cam_isaac_observer"
                        )
                    )
                if axes_writer is not None:
                    axes_writer.append_data(
                        render_without_robot(
                            renderer,
                            model,
                            data,
                            closeup_camera(data, ctrl.receptive),
                            axis_body_id=ctrl.insertive,
                        )
                    )
            if not step_finite:
                break
    finally:
        if axes_writer is not None:
            axes_writer.close()
        if diagnostic_writer is not None:
            diagnostic_writer.close()
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    trace = out / f"closedloop_ep{episode:03d}.csv"
    write_csv(trace, rows)
    successes = [row["step"] for row in rows if row["strict_success"]]
    relaxed_successes = [row["step"] for row in rows if row["relaxed_success"]]
    final = rows[-1]
    return {
        "episode": episode,
        "source_indices": [start, end],
        "source_reset_index": int(store["meta/reset_indices"][episode]),
        "source_success_seen": bool(store["meta/success_seen"][episode]),
        "source_first_done_step": (
            int(store["meta/first_done_step"][episode])
            if "meta/first_done_step" in store
            else None
        ),
        "steps_requested": steps,
        "steps_completed": len(rows),
        "finite": finite,
        "strict_success_steps": len(successes),
        "first_strict_success_step": successes[0] if successes else None,
        "stable_success": max_success_streak >= STABLE_SUCCESS_STEPS,
        "max_success_streak": max_success_streak,
        "relaxed_success_steps": len(relaxed_successes),
        "first_relaxed_success_step": (
            relaxed_successes[0] if relaxed_successes else None
        ),
        "stable_relaxed_success": (
            max_relaxed_success_streak >= STABLE_SUCCESS_STEPS
        ),
        "max_relaxed_success_streak": max_relaxed_success_streak,
        "best_position_error_m": min(row["position_error_m"] for row in rows),
        "best_orientation_xy_error_rad": min(
            row["orientation_xy_error_rad"] for row in rows
        ),
        "final_position_error_m": final["position_error_m"],
        "final_orientation_xy_error_rad": final["orientation_xy_error_rad"],
        "final_strict_success": bool(final["strict_success"]),
        "final_relaxed_success": bool(final["relaxed_success"]),
        "close_command_steps": sum(row["close_command"] for row in rows),
        "robot_contact_steps": sum(row["robot_contact"] for row in rows),
        "geometry_reset_audit": geometry,
        "effective_runtime": effective,
        "trace": str(trace),
        "video": str(video) if video is not None else None,
        "diagnostic_video": (
            str(diagnostic_video) if diagnostic_video is not None else None
        ),
        "axes_video": str(axes_video) if axes_video is not None else None,
    }


def run(args: argparse.Namespace) -> None:
    source = Path(args.zarr).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(source), mode="r")
    starts, ends, source_audit = audit_source(store, checkpoint)
    episodes = Replay.parse_episodes(args.episodes, len(ends))
    render_episodes = set(
        Replay.parse_episodes(args.render_episodes, len(ends))
        if args.render_episodes
        else []
    )
    if args.render_episode >= 0:
        render_episodes.add(args.render_episode)
    render_cameras = tuple(
        camera.strip() for camera in args.render_cameras.split(",") if camera.strip()
    )
    if not render_cameras:
        render_cameras = (
            ("cam_isaac_observer",)
            if args.diagnostic_render
            else ("cam_front", "cam_task", "cam_top")
        )

    CupCake.CUPCAKE_CONTACT_SOLREF = [args.contact_timeconstant, 1.0]
    CupCake.ROBOT_OBJECT_CONTACT_SOLREF = [
        args.robot_contact_timeconstant,
        1.0,
    ]

    timing = source_audit["timing"]
    CL.SIM_DT = timing["physics_dt_s"]
    CL.DECIM = timing["decimation"]
    policy = FrankaPolicy.load_from_checkpoint(str(checkpoint), device=args.device)

    results = []
    for episode in episodes:
        start, end = int(starts[episode]), int(ends[episode])
        steps = end - start if args.max_steps == 0 else min(args.max_steps, end - start)
        print(
            f"[cupcake-closed-loop] episode={episode} steps={steps} "
            f"render={episode in render_episodes}",
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
            out,
            episode in render_episodes,
            args.width,
            args.height,
            args.fps,
            args.diagnostic_render,
            render_cameras,
            args.friction_combine,
            args.cupcake_collision,
            args.cupcake_plate_collision,
            args.hand_collision,
            args.finger_collision,
        )
        results.append(result)
        print(
            f"[cupcake-closed-loop] episode={episode} "
            f"success_steps={result['strict_success_steps']} "
            f"stable={result['stable_success']} "
            f"best={result['best_position_error_m'] * 1000.0:.2f}mm "
            f"final={result['final_position_error_m'] * 1000.0:.2f}mm",
            flush=True,
        )

    episode_rows = [
        {
            key: value
            for key, value in result.items()
            if key not in {"geometry_reset_audit", "effective_runtime"}
        }
        for result in results
    ]
    write_csv(out / "episodes.csv", episode_rows)
    summary = {
        "definition": (
            "true deterministic closed loop: live MuJoCo state -> 200-D H5 "
            "observation -> checkpoint mean action"
        ),
        "not_replay": "source data/action and next_raw_state are never consumed",
        "state_mutation_audit": (
            "one raw-state restore at episode initialization; no later qpos/qvel writes"
        ),
        "source_zarr": str(source),
        "source_signature": CL.zarr_signature(store),
        "source_audit": source_audit,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": policy.ckpt_iter,
        "policy_device": args.device,
        "episodes": episodes,
        "timing": {
            **timing,
            "mujoco_substeps_per_source_tick": args.physics_substeps,
        },
        "controller_contract": {
            "runtime_source": "per-episode rich Isaac Zarr",
            "jacobian_point": "PhysX panda_hand COM",
            "arm_friction": "runtime dynamic Coulomb + runtime viscous",
            "gripper": "live privileged grasp guard through B3 tendon actuator",
            "policy_gripper_scalar": "ignored, matching source action term",
            "nullspace_stiffness": 0.0,
            "action_reference_blend": 0.0,
        },
        "contact_mapping": {
            "cupcake_collision": args.cupcake_collision,
            "cupcake_plate_collision": args.cupcake_plate_collision,
            "hand_collision": args.hand_collision,
            "finger_collision": args.finger_collision,
            "object_contact_timeconstant_s": CupCake.CUPCAKE_CONTACT_SOLREF[0],
            "robot_contact_timeconstant_s": (
                CupCake.ROBOT_OBJECT_CONTACT_SOLREF[0]
            ),
            "dampratio": CupCake.CUPCAKE_CONTACT_SOLREF[1],
            "solimp": CupCake.CUPCAKE_CONTACT_SOLIMP,
            "friction_combine": args.friction_combine,
            "source_contact_offsets_m": {
                "cupcake_plate_table": CupCake.CUPCAKE_CONTACT_OFFSET,
                "robot": CupCake.ROBOT_CONTACT_OFFSET,
                "robot_cupcake_pair_margin": CupCake.ROBOT_OBJECT_CONTACT_MARGIN,
                "object_object_pair_margin": CupCake.OBJECT_OBJECT_CONTACT_MARGIN,
            },
        },
        "success_definition": {
            "position_error_m_lt": CupCake.SUCCESS_POSITION_M,
            "abs_roll_plus_abs_pitch_rad_lt": CupCake.SUCCESS_ORIENTATION_XY_RAD,
            "stable_success_steps_gte": STABLE_SUCCESS_STEPS,
        },
        "relaxed_success_definition": {
            "position_error_m_lt": CupCake.RELAXED_SUCCESS_POSITION_M,
            "abs_roll_plus_abs_pitch_rad_lt": (
                CupCake.RELAXED_SUCCESS_ORIENTATION_XY_RAD
            ),
            "stable_success_steps_gte": STABLE_SUCCESS_STEPS,
        },
        "episodes_evaluated": len(results),
        "episodes_with_any_strict_success": sum(
            result["strict_success_steps"] > 0 for result in results
        ),
        "stable_successes": sum(result["stable_success"] for result in results),
        "stable_relaxed_successes": sum(
            result["stable_relaxed_success"] for result in results
        ),
        "finite_episodes": sum(result["finite"] for result in results),
        "results": results,
    }
    with (out / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(CL.to_jsonable(summary), file, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "episodes_evaluated": summary["episodes_evaluated"],
                "episodes_with_any_strict_success": summary[
                    "episodes_with_any_strict_success"
                ],
                "stable_successes": summary["stable_successes"],
                "stable_relaxed_successes": summary[
                    "stable_relaxed_successes"
                ],
                "out": str(out),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default=DEFAULT_ZARR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--episodes", default="0")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--physics_substeps", type=int, default=16)
    parser.add_argument("--contact_timeconstant", type=float, default=0.02)
    parser.add_argument("--robot_contact_timeconstant", type=float, default=0.005)
    parser.add_argument(
        "--friction_combine",
        choices=("legacy_b3", "physx_average"),
        default="physx_average",
    )
    parser.add_argument(
        "--cupcake_collision",
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
        "--hand_collision",
        choices=("menagerie", "source_usd", "disabled"),
        default="source_usd",
    )
    parser.add_argument(
        "--cupcake_plate_collision",
        choices=("convex_hull", "radial16", "base_cylinder"),
        default="base_cylinder",
    )
    parser.add_argument(
        "--finger_collision",
        choices=("mimic", "menagerie", "menagerie_mesh_only"),
        default="mimic",
    )
    parser.add_argument("--render_episode", type=int, default=-1)
    parser.add_argument("--render_episodes", default="")
    parser.add_argument("--render_cameras", default="")
    parser.add_argument("--diagnostic_render", action="store_true")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if (
        args.physics_substeps <= 0
        or args.max_steps < 0
        or args.contact_timeconstant <= 0.0
        or args.robot_contact_timeconstant <= 0.0
    ):
        parser.error(
            "physics substeps and contact timeconstants must be positive; "
            "--max_steps must be non-negative"
        )
    run(args)


if __name__ == "__main__":
    main()
