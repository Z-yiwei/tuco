#!/usr/bin/env python3
"""RSL-RL vector environment for native MuJoCo CupCake fine-tuning.

The actor and critic contracts match the released CupCake Isaac checkpoint:
200-D five-frame actor observations, 171-D privileged critic observations,
and seven actions. Each worker owns one audited CupCake MuJoCo runtime.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MUJOCO_GL", "")

import mujoco
import numpy as np
import zarr

import closed_loop_eval as CL
import compare_cupcake_mujoco as Replay
import compare_peginsert_continuous_mujoco as B3
import cupcake_mujoco_model as CupCake
import quat_utils as Q
from peg_mujoco_rl_env import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    CRITIC_OBS_DIM,
    CRITIC_SCENE_FIELDS,
    EFFORT_SCALE,
    OMNIRESET_POLICY_DT_S,
    MujocoPegVecEnv,
    RewardConfig,
    WorkerConfig,
)


STABLE_SUCCESS_STEPS = 5
PHYSICS_SUBSTEPS = 16
FRICTION_COMBINE = "physx_average"
CUPCAKE_COLLISION = "convex_decomposition"
CUPCAKE_PLATE_COLLISION = "base_cylinder"
HAND_COLLISION = "source_usd"
FINGER_COLLISION = "mimic"


def configure_physics_profile(
    *,
    physics_substeps: int = 16,
    friction_combine: str = "physx_average",
    cupcake_collision: str = "convex_decomposition",
    cupcake_plate_collision: str = "base_cylinder",
    hand_collision: str = "source_usd",
    finger_collision: str = "mimic",
) -> None:
    """Set the process-local profile inherited by subsequently forked workers."""
    if physics_substeps <= 0:
        raise ValueError("physics_substeps must be positive")
    if friction_combine not in {"legacy_b3", "physx_average"}:
        raise ValueError(f"unknown friction combine mode: {friction_combine}")
    if cupcake_collision not in {
        "convex_decomposition",
        "radial32",
        "coacd",
        "sdf",
        "convex_mesh",
        "compound",
    }:
        raise ValueError(f"unknown CupCake collision mode: {cupcake_collision}")
    if cupcake_plate_collision not in {
        "convex_hull",
        "radial16",
        "base_cylinder",
    }:
        raise ValueError(
            f"unknown CupCake-Plate collision mode: {cupcake_plate_collision}"
        )
    if hand_collision not in {"menagerie", "source_usd", "disabled"}:
        raise ValueError(f"unknown hand collision mode: {hand_collision}")
    if finger_collision not in {"mimic", "menagerie", "menagerie_mesh_only"}:
        raise ValueError(f"unknown finger collision mode: {finger_collision}")
    global PHYSICS_SUBSTEPS, FRICTION_COMBINE, CUPCAKE_COLLISION
    global CUPCAKE_PLATE_COLLISION, HAND_COLLISION, FINGER_COLLISION
    PHYSICS_SUBSTEPS = int(physics_substeps)
    FRICTION_COMBINE = friction_combine
    CUPCAKE_COLLISION = cupcake_collision
    CUPCAKE_PLATE_COLLISION = cupcake_plate_collision
    HAND_COLLISION = hand_collision
    FINGER_COLLISION = finger_collision


def _episode_bounds(store: Any) -> tuple[np.ndarray, np.ndarray]:
    ends = np.asarray(store["meta/episode_ends"], dtype=np.int64)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
    if len(ends) == 0 or np.any(ends <= starts):
        raise ValueError("runtime source has empty or non-monotonic episodes")
    return starts, ends


def load_zarr_reset_raw_pool(
    path: str | Path,
    required_panel: str | None = "fit",
    mode: str = "home",
    home_fraction: float = 0.5,
) -> np.ndarray:
    """Load home resets or a fit-only source-trajectory reset curriculum."""
    path = Path(path).resolve()
    store = zarr.open(str(path), mode="r")
    starts, ends = _episode_bounds(store)
    if int(ends[-1]) != int(store["data/raw_state"].shape[0]):
        raise ValueError(f"episode boundaries do not cover data/raw_state: {path}")
    home = np.asarray(store["data/raw_state"][starts], dtype=np.float64)
    if home.shape != (len(starts), 57):
        raise ValueError(f"reset pool must map to (N, 57), got {home.shape}")
    if mode == "home":
        raw = home
    elif mode in {
        "source_trajectory",
        "successful_source_trajectory",
        "successful_episode_trajectory",
        "mixed",
        "mixed_successful",
        "mixed_successful_episode",
    }:
        trajectory = np.asarray(store["data/raw_state"], dtype=np.float64)
        if mode in {"successful_source_trajectory", "mixed_successful"}:
            trajectory = trajectory[CupCake.raw_pose_success_mask(trajectory)]
            if len(trajectory) == 0:
                raise ValueError(f"source has no official pose-success rows: {path}")
        if mode in {"successful_episode_trajectory", "mixed_successful_episode"}:
            success_seen = np.asarray(store["meta/success_seen"], dtype=np.bool_)
            if success_seen.shape != (len(starts),):
                raise ValueError(f"meta/success_seen has invalid shape: {path}")
            row_mask = np.repeat(success_seen, ends - starts)
            trajectory = trajectory[row_mask]
            if len(trajectory) == 0:
                raise ValueError(f"source has no successful episode rows: {path}")
        if mode in {
            "source_trajectory",
            "successful_source_trajectory",
            "successful_episode_trajectory",
        }:
            raw = trajectory
        else:
            if not 0.0 < home_fraction < 1.0:
                raise ValueError("home_fraction must be in (0, 1) for mixed modes")
            home_count = int(
                round(len(trajectory) * home_fraction / (1.0 - home_fraction))
            )
            repeats = int(np.ceil(home_count / len(home)))
            home_balanced = np.tile(home, (repeats, 1))[:home_count]
            raw = np.concatenate([home_balanced, trajectory], axis=0)
    else:
        raise ValueError(f"unknown reset-pool mode: {mode}")
    if not np.isfinite(raw).all():
        raise ValueError(f"reset pool contains non-finite values: {path}")
    if required_panel is not None and store.attrs.get("panel") != required_panel:
        raise ValueError(
            f"source panel must be {required_panel!r}, got "
            f"{store.attrs.get('panel')!r}: {path}"
        )
    return np.ascontiguousarray(raw)


def runtime_catalog(paths: Sequence[str | Path]) -> list[tuple[str, int]]:
    """Return every exported CupCake runtime without success filtering."""
    result: list[tuple[str, int]] = []
    for value in paths:
        path = str(Path(value).resolve())
        store = zarr.open(path, mode="r")
        if store.attrs.get("success_filtering") is True:
            raise ValueError(f"runtime source is success-filtered: {path}")
        if store.attrs.get("panel") != "fit":
            raise ValueError(f"runtime source must be the fit panel: {path}")
        starts, _ = _episode_bounds(store)
        result.extend((path, episode) for episode in range(len(starts)))
    if not result:
        raise ValueError("no CupCake runtime episodes found")
    return result


class _CupCakeEpisode:
    """One native MuJoCo CupCake environment inside one physics worker."""

    def __init__(self, cfg: WorkerConfig, reset_pool: np.ndarray):
        self.cfg = cfg
        self.reset_pool = reset_pool
        self.rng = np.random.default_rng(cfg.seed)
        self.runtime_rng = np.random.default_rng(cfg.seed + 1_000_003)
        self._store_cache: dict[str, Any] = {}
        self.current_runtime_path = ""
        self.current_runtime_episode = -1
        self.runtime_load_count = 0
        self._load_runtime(cfg.runtime_path, cfg.runtime_episode)

        self.prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.episode_step = 0
        self.episode_return = 0.0
        self.first_lift_given = False
        self.ever_success = False
        self.success_streak = 0
        self.max_success_streak = 0
        self.current_reset_id = -1
        self.reset_count = 0
        self.current_actor_obs = np.zeros(ACTOR_OBS_DIM, dtype=np.float32)
        self.current_critic_obs = np.zeros(CRITIC_OBS_DIM, dtype=np.float32)
        self.reset()

    def _load_runtime(self, runtime_path: str, runtime_episode: int) -> None:
        store = self._store_cache.get(runtime_path)
        if store is None:
            store = zarr.open(runtime_path, mode="r")
            self._store_cache[runtime_path] = store
        starts, ends = _episode_bounds(store)
        if runtime_episode < 0 or runtime_episode >= len(ends):
            raise IndexError(
                f"runtime episode {runtime_episode} outside [0, {len(ends)})"
            )
        start = int(starts[runtime_episode])
        end = int(ends[runtime_episode])

        _, _, policy_dt = Replay.configure_timing(store)
        self.reward_dt_s = float(policy_dt)
        if not np.isclose(
            self.reward_dt_s, OMNIRESET_POLICY_DT_S, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "runtime policy dt does not match the released CupCake parent: "
                f"{self.reward_dt_s} != {OMNIRESET_POLICY_DT_S}"
            )

        raw0 = np.asarray(store["data/raw_state"][start], dtype=np.float64)
        profile = B3.b3_center_profile(physics_substeps=PHYSICS_SUBSTEPS)
        profile["cupcake_collision"] = CUPCAKE_COLLISION
        profile["cupcake_plate_collision"] = CUPCAKE_PLATE_COLLISION
        profile["hand_collision"] = HAND_COLLISION
        profile["finger_collision"] = FINGER_COLLISION
        self.model = CupCake.build_model(raw0, profile_cfg=profile)
        self.ctrl = CupCake.Controller(self.model)
        self.effective = Replay.configure_episode(
            self.model,
            self.ctrl,
            store,
            start,
            end,
            raw0,
            friction_combine=FRICTION_COMBINE,
        )
        self.torque_max = np.asarray(
            self.effective["arm"]["arm_torque_max"], dtype=np.float64
        )
        self.data = mujoco.MjData(self.model)
        self.obs_builder = CL.ObsBuilder(self.ctrl)
        self.critic_static = self._build_critic_static(store, start)
        self.current_runtime_path = runtime_path
        self.current_runtime_episode = int(runtime_episode)
        self.runtime_load_count += 1

    def _build_critic_static(self, store: Any, row: int) -> np.ndarray:
        scene = [
            np.asarray(store[f"data/{field}"][row], dtype=np.float32).reshape(-1)
            for field in CRITIC_SCENE_FIELDS
        ]
        joint_friction = np.concatenate(
            [
                np.asarray(
                    store["data/arm_joint_friction_static"][row], dtype=np.float32
                ).reshape(7),
                np.asarray(
                    store["data/gripper_joint_friction_static"][row],
                    dtype=np.float32,
                ).reshape(2),
            ]
        )
        joint_armature = np.concatenate(
            [
                np.asarray(
                    store["data/arm_joint_armature"][row], dtype=np.float32
                ).reshape(7),
                np.asarray(
                    store["data/gripper_joint_armature"][row], dtype=np.float32
                ).reshape(2),
            ]
        )
        joint_stiffness = np.concatenate(
            [
                np.zeros(7, dtype=np.float32),
                np.asarray(
                    store["data/gripper_joint_stiffness"][row], dtype=np.float32
                ).reshape(2),
            ]
        )
        joint_damping = np.concatenate(
            [
                np.zeros(7, dtype=np.float32),
                np.asarray(
                    store["data/gripper_joint_damping"][row], dtype=np.float32
                ).reshape(2),
            ]
        )
        # The OSC arm is effort controlled, so its articulation stiffness and
        # damping are zero. The two gripper joints retain their exported PD.
        result = np.concatenate(
            [
                *scene,
                joint_friction,
                joint_armature,
                joint_stiffness,
                joint_damping,
            ]
        ).astype(np.float32)
        if result.shape != (115,):
            raise ValueError(f"critic static layout is {result.shape}, expected (115,)")
        return result

    def _actor_obs(self) -> np.ndarray:
        return self.obs_builder.step(self.data, self.prev_action).astype(
            np.float32, copy=False
        )

    def _critic_obs(self) -> np.ndarray:
        pose, _ = self.obs_builder._terms(self.data)
        ee_velocity = self.ctrl.jac_arm(self.data) @ self.data.qvel[:7]
        # The released source observation applies subtract_frame_transforms to
        # velocity vectors as if they were positions. Reproduce its root-position
        # subtraction because the CupCake resets have a nonzero robot root pose.
        ee_velocity -= np.tile(self.source_root_position_in_root, 2)
        current = np.concatenate(
            [
                pose[CL.MAPPING["poseA"]],
                self.prev_action,
                self.data.qpos[:9],
                pose[CL.MAPPING["poseB"]],
                pose[CL.MAPPING["poseC"]],
                pose[CL.MAPPING["poseD"]],
                np.array(
                    [1.0 - self.episode_step / self.cfg.max_episode_length],
                    dtype=np.float64,
                ),
                self.data.qvel[:9],
                ee_velocity,
                self.critic_static,
            ]
        ).astype(np.float32)
        if current.shape != (CRITIC_OBS_DIM,):
            raise RuntimeError(
                f"critic observation is {current.shape}, expected ({CRITIC_OBS_DIM},)"
            )
        return current

    def reset(self, reset_id: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        if self.cfg.runtime_mode == "resample_on_reset" and self.reset_count > 0:
            runtime_index = int(
                self.runtime_rng.integers(0, len(self.cfg.runtime_catalog))
            )
            self._load_runtime(*self.cfg.runtime_catalog[runtime_index])
        if reset_id is None:
            reset_id = int(self.rng.integers(0, len(self.reset_pool)))
        if reset_id < 0 or reset_id >= len(self.reset_pool):
            raise IndexError(f"reset id {reset_id} outside [0, {len(self.reset_pool)})")

        raw = self.reset_pool[reset_id]
        self.source_root_position_in_root = Q.quat_apply(
            Q.quat_inv(raw[21:25]), raw[18:21]
        )
        Replay.set_fixed_scene_poses(self.model, raw)
        CupCake.set_raw_state(self.model, self.data, self.ctrl, raw)
        self.obs_builder.reset(self.data)

        self.prev_action.fill(0.0)
        self.episode_step = 0
        self.episode_return = 0.0
        self.first_lift_given = False
        self.ever_success = False
        self.success_streak = 0
        self.max_success_streak = 0
        self.current_reset_id = int(reset_id)
        self.reset_count += 1
        self.current_actor_obs = self._actor_obs()
        self.current_critic_obs = self._critic_obs()
        return self.current_actor_obs, self.current_critic_obs

    def _reward_and_task(
        self, action: np.ndarray, close: bool
    ) -> tuple[float, bool, dict[str, float]]:
        reward_cfg = self.cfg.reward
        metrics = CupCake.success_metrics(self.data, self.ctrl)
        pose_success = bool(metrics["strict_success"])
        relaxed_pose_success = bool(metrics["relaxed_success"])
        self.ever_success = self.ever_success or pose_success
        self.success_streak = self.success_streak + 1 if pose_success else 0
        self.max_success_streak = max(self.max_success_streak, self.success_streak)

        hand_pos = self.data.xpos[self.ctrl.hand]
        hand_quat = self.data.xquat[self.ctrl.hand]
        tcp_pos = hand_pos + Q.quat_apply(hand_quat, CL.TCP_OFF)
        tcp_asset_distance = float(
            np.linalg.norm(self.data.xpos[self.ctrl.insertive] - tcp_pos)
        )
        avg_finger = float(np.mean(self.data.qpos[7:9]))
        closure = 1.0 - float(np.clip(avg_finger / 0.04, 0.0, 1.0))
        asset_in_gripper = np.exp(-tcp_asset_distance / 0.05) * closure
        first_lift = (
            avg_finger < 0.02
            and float(self.data.xpos[self.ctrl.insertive, 2]) > 0.025
            and not self.first_lift_given
        )
        self.first_lift_given = self.first_lift_given or first_lift

        action_l2 = min(float(np.sum(np.square(action))), 1.0e4)
        action_rate_l2 = min(
            float(np.sum(np.square(action - self.prev_action))), 1.0e4
        )
        joint_velocity_l2 = min(
            float(np.sum(np.square(self.data.qvel[:7]))), 1.0e4
        )
        ee_asset_distance = 1.0 - np.tanh(tcp_asset_distance / 1.0)
        dense_success = 0.5 * (
            np.exp(-float(metrics["orientation_xy_error_rad"]) / 1.0)
            + np.exp(-float(metrics["position_error_m"]) / 1.0)
        )
        finite = bool(
            np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        )
        abnormal = bool(
            not finite or np.any(np.abs(self.data.qvel[:7]) > 2.0 * CL.VEL_MAX)
        )
        components = {
            "action_magnitude": reward_cfg.action_magnitude * action_l2,
            "action_rate": reward_cfg.action_rate * action_rate_l2,
            "joint_vel": reward_cfg.joint_vel * joint_velocity_l2,
            "abnormal_robot": reward_cfg.abnormal_robot * float(abnormal),
            "progress_context": reward_cfg.progress_context * 0.0,
            "ee_asset_distance": reward_cfg.ee_asset_distance * ee_asset_distance,
            "dense_success_reward": reward_cfg.dense_success_reward * dense_success,
            "success_reward": reward_cfg.success_reward * float(pose_success),
            "peg_in_gripper": reward_cfg.peg_in_gripper * asset_in_gripper,
            "grasped_and_lifted": reward_cfg.grasped_and_lifted * float(first_lift),
        }
        components = {
            name: float(value * self.reward_dt_s)
            for name, value in components.items()
        }
        diagnostics = {
            **components,
            "reward_dt_s": self.reward_dt_s,
            "position_error_m": float(metrics["position_error_m"]),
            "orientation_error_rad": float(metrics["orientation_xy_error_rad"]),
            "official_pose": float(pose_success),
            "relaxed_pose": float(relaxed_pose_success),
            "close": float(close),
            "robot_contact": float(self.ctrl.insertive_robot_contact(self.data)),
            "success_streak": float(self.success_streak),
            "tcp_asset_distance_m": tcp_asset_distance,
        }
        return float(sum(components.values())), abnormal, diagnostics

    def step(self, action: np.ndarray) -> dict[str, Any]:
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        source_reset_id = self.current_reset_id
        if not np.isfinite(action).all():
            reward = self.cfg.reward.abnormal_robot * self.reward_dt_s
            self.episode_step += 1
            self.episode_return += reward
            episode = self._episode_summary(source_reset_id, abnormal=True)
            actor_obs, critic_obs = self.reset()
            return {
                "actor_obs": actor_obs,
                "critic_obs": critic_obs,
                "reward": reward,
                "done": True,
                "timeout": False,
                "episode": episode,
                "task": {
                    "pose_success": 0.0,
                    "relaxed_pose_success": 0.0,
                    "stable_success": 0.0,
                    "position_error_m": float("inf"),
                    "orientation_error_rad": float("inf"),
                    "abnormal": 1.0,
                },
            }

        CL.TAU_MAX = self.torque_max
        close = B3.step_action(
            self.model,
            self.data,
            self.ctrl,
            action,
            finger_velocity_limits=None,
            independent_gripper=False,
            gripper_close_override=None,
            jacobian_point="physx_com",
            nullspace_stiffness=0.0,
            nullspace_damping_ratio=1.0,
            action_reference_blend=0.0,
            bias_compensation_scale=0.0,
            effort_scale=EFFORT_SCALE,
        )
        self.episode_step += 1
        reward, abnormal, diagnostics = self._reward_and_task(action, close)
        self.episode_return += reward
        self.prev_action = action.copy()

        timeout = self.episode_step >= self.cfg.max_episode_length
        done = bool(timeout or abnormal)
        episode = None
        if done:
            episode = self._episode_summary(
                source_reset_id, abnormal=abnormal, diagnostics=diagnostics
            )
            actor_obs, critic_obs = self.reset()
        else:
            actor_obs = self._actor_obs()
            critic_obs = self._critic_obs()
            self.current_actor_obs = actor_obs
            self.current_critic_obs = critic_obs
        return {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
            "reward": reward,
            "done": done,
            "timeout": bool(timeout),
            "episode": episode,
            "task": {
                "pose_success": diagnostics["official_pose"],
                "relaxed_pose_success": diagnostics["relaxed_pose"],
                "stable_success": float(
                    self.max_success_streak >= STABLE_SUCCESS_STEPS
                ),
                "position_error_m": diagnostics["position_error_m"],
                "orientation_error_rad": diagnostics["orientation_error_rad"],
                "abnormal": float(abnormal),
            },
        }

    def _episode_summary(
        self,
        reset_id: int,
        abnormal: bool,
        diagnostics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        diagnostics = diagnostics or {}
        return {
            "reward": float(self.episode_return),
            "length": float(self.episode_step),
            "strict_success": float(self.ever_success),
            "stable_success": float(
                self.max_success_streak >= STABLE_SUCCESS_STEPS
            ),
            "abnormal": float(abnormal),
            "max_success_streak": float(self.max_success_streak),
            "final_pose_success": float(diagnostics.get("official_pose", 0.0)),
            "reset_id": float(reset_id),
            "runtime_episode": float(self.current_runtime_episode),
            "runtime_load_count": float(self.runtime_load_count),
        }


def _worker_main(
    connection: Any, cfg: WorkerConfig, reset_pool: np.ndarray
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        episode = _CupCakeEpisode(cfg, reset_pool)
        connection.send(
            (
                "ready",
                episode.current_actor_obs,
                episode.current_critic_obs,
                {
                    "worker_id": cfg.worker_id,
                    "runtime_path": cfg.runtime_path,
                    "runtime_episode": cfg.runtime_episode,
                    "runtime_mode": cfg.runtime_mode,
                    "runtime_catalog_rows": len(cfg.runtime_catalog),
                    "initial_reset_id": episode.current_reset_id,
                    "physics_dt_s": float(CL.SIM_DT),
                    "decimation": int(CL.DECIM),
                    "physics_substeps": PHYSICS_SUBSTEPS,
                    "reward_policy_dt_s": episode.reward_dt_s,
                    "model_timestep_s": float(episode.model.opt.timestep),
                    "friction_combine": FRICTION_COMBINE,
                    "cupcake_collision": CUPCAKE_COLLISION,
                    "cupcake_plate_collision": CUPCAKE_PLATE_COLLISION,
                    "hand_collision": HAND_COLLISION,
                    "finger_collision": FINGER_COLLISION,
                },
            )
        )
        while True:
            message = connection.recv()
            command = message[0]
            if command == "step":
                connection.send(("step", episode.step(message[1])))
            elif command == "reset":
                actor_obs, critic_obs = episode.reset(message[1])
                connection.send(("reset", actor_obs, critic_obs))
            elif command == "close":
                connection.send(("closed",))
                break
            else:
                raise ValueError(f"unknown worker command: {command}")
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except BaseException:
            pass
    finally:
        connection.close()


class MujocoCupCakeVecEnv(MujocoPegVecEnv):
    """CupCake task adapter over the shared process-parallel VecEnv facade."""

    _worker_target = staticmethod(_worker_main)
    _process_name_prefix = "mujoco-cupcake"

    def __init__(
        self,
        reset_pool: np.ndarray,
        runtime_assignments: Sequence[tuple[str, int]],
        **kwargs: Any,
    ) -> None:
        super().__init__(reset_pool, runtime_assignments, **kwargs)
        self.cfg.update(
            {
                "task": "CupCake__Plate",
                "physics_substeps": PHYSICS_SUBSTEPS,
                "friction_combine": FRICTION_COMBINE,
                "cupcake_collision": CUPCAKE_COLLISION,
                "cupcake_plate_collision": CUPCAKE_PLATE_COLLISION,
                "hand_collision": HAND_COLLISION,
                "finger_collision": FINGER_COLLISION,
                "success": {
                    "position_threshold_m": CupCake.SUCCESS_POSITION_M,
                    "orientation_xy_threshold_rad": (
                        CupCake.SUCCESS_ORIENTATION_XY_RAD
                    ),
                    "stable_steps": STABLE_SUCCESS_STEPS,
                    "relaxed_position_threshold_m": (
                        CupCake.RELAXED_SUCCESS_POSITION_M
                    ),
                    "relaxed_orientation_xy_threshold_rad": (
                        CupCake.RELAXED_SUCCESS_ORIENTATION_XY_RAD
                    ),
                },
            }
        )


def audit_observation_parity(path: str | Path) -> dict[str, float]:
    """Compare one exact Isaac reset against the MuJoCo actor/critic contract."""
    source = str(Path(path).resolve())
    store = zarr.open(source, mode="r")
    if "data/critic_state" not in store:
        raise KeyError(f"parity source has no data/critic_state: {source}")
    reset_pool = load_zarr_reset_raw_pool(source)
    cfg = WorkerConfig(
        worker_id=0,
        seed=42,
        runtime_path=source,
        runtime_episode=0,
        runtime_catalog=((source, 0),),
        runtime_mode="fixed_per_worker",
        max_episode_length=160,
        reward=RewardConfig(),
    )
    episode = _CupCakeEpisode(cfg, reset_pool)
    actor, critic = episode.reset(0)
    actor_reference = np.asarray(store["data/state"][0], dtype=np.float32)
    critic_reference = np.asarray(store["data/critic_state"][0], dtype=np.float32)
    actor_error = float(np.max(np.abs(actor - actor_reference)))
    critic_error = float(np.max(np.abs(critic - critic_reference)))
    if actor_error > 2.0e-6 or critic_error > 1.0e-5:
        raise RuntimeError(
            "CupCake observation parity failed: "
            f"actor={actor_error:.3e}, critic={critic_error:.3e}"
        )
    return {
        "actor_max_abs": actor_error,
        "critic_max_abs": critic_error,
        "actor_threshold": 2.0e-6,
        "critic_threshold": 1.0e-5,
    }
