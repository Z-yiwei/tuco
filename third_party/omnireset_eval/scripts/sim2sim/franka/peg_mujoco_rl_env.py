#!/usr/bin/env python3
"""RSL-RL vector environment for native MuJoCo PegInsert fine-tuning.

The policy-facing contract is intentionally identical to the audited B3
Isaac/MuJoCo evaluator:

* 200-D, five-frame actor observation;
* 7-D action (six relative OSC coordinates plus the ignored gripper scalar);
* B3-center action scale and OSC gains;
* guarded binary gripper and the audited MuJoCo physics mapping.

Each subprocess owns one MuJoCo model/data pair.  State resets are sampled
from the complete fixed-home 3 cm training pool.  Physical parameters come
from unfiltered Isaac rollout episodes and are fixed per worker, matching the
collector's ``startup_scene_runtime_reused_per_vector_env`` behavior.
"""

from __future__ import annotations

import collections
import multiprocessing as mp
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MUJOCO_GL", "")

import mujoco
import numpy as np
import torch
import zarr
from tensordict import TensorDict

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import audit_b3_fresh_chunk_replay as Audit
import closed_loop_eval as CL
import compare_peginsert_continuous_mujoco as Replay
import quat_utils as Q


ACTOR_OBS_DIM = 200
CRITIC_OBS_DIM = 171
ACTION_DIM = 7
FINGER_VELOCITY_LIMITS = np.array([0.04, 0.04], dtype=np.float64)
EFFORT_SCALE = np.ones(7, dtype=np.float64)
OMNIRESET_POLICY_DT_S = 0.1

# These are the non-history terms in the original 171-D privileged critic.
CRITIC_SCENE_FIELDS = (
    "robot_material_properties",
    "insertive_object_material_properties",
    "receptive_object_material_properties",
    "table_material_properties",
    "robot_body_masses",
    "insertive_object_body_masses",
    "receptive_object_body_masses",
    "table_body_masses",
)


def _rows(value: Any) -> torch.Tensor:
    if isinstance(value, list):
        value = torch.stack([torch.as_tensor(row) for row in value])
    tensor = torch.as_tensor(value).detach().cpu().float()
    if tensor.ndim == 3 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    return tensor


def load_reset_raw_pool(path: str | Path) -> np.ndarray:
    """Convert an OmniReset reset file to the audited 57-D raw-state layout."""
    path = Path(path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if set(payload) != {"initial_state"}:
        raise ValueError(f"unexpected reset payload keys in {path}: {set(payload)}")
    initial = payload["initial_state"]
    robot = initial["articulation"]["robot"]
    peg = initial["rigid_object"]["insertive_object"]
    hole = initial["rigid_object"]["receptive_object"]
    parts = (
        _rows(robot["joint_position"]),
        _rows(robot["joint_velocity"]),
        _rows(robot["root_pose"]),
        _rows(robot["root_velocity"]),
        _rows(peg["root_pose"]),
        _rows(peg["root_velocity"]),
        _rows(hole["root_pose"]),
        _rows(hole["root_velocity"]),
    )
    count = max(int(part.shape[0]) if part.ndim > 1 else 1 for part in parts)
    normalized = [
        part.reshape(1, -1).expand(count, -1)
        if part.ndim == 1
        else part.reshape(count, -1)
        for part in parts
    ]
    raw = torch.cat(normalized, dim=1).numpy().astype(np.float64)
    if raw.shape != (count, 57):
        raise ValueError(f"reset pool must map to (N, 57), got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"reset pool contains non-finite values: {path}")
    return np.ascontiguousarray(raw)


def runtime_catalog(paths: Sequence[str | Path]) -> list[tuple[str, int]]:
    """Return every (unfiltered source, episode) available for worker physics."""
    result: list[tuple[str, int]] = []
    for value in paths:
        path = str(Path(value).resolve())
        store = zarr.open(path, mode="r")
        if store.attrs.get("success_filtering") is not False:
            raise ValueError(f"runtime source is success-filtered: {path}")
        ends = np.asarray(store["meta/episode_ends"], dtype=np.int64)
        if len(ends) == 0:
            raise ValueError(f"runtime source contains no episodes: {path}")
        result.extend((path, episode) for episode in range(len(ends)))
    if not result:
        raise ValueError("no runtime episodes found")
    return result


@dataclass(frozen=True)
class RewardConfig:
    """Weights from the saved OmniReset parent run.

    IsaacLab's reward manager multiplies every weighted term by the policy
    time step.  :meth:`_PegEpisode._reward_and_task` applies that same factor;
    these values are therefore the unscaled weights from ``params/env.yaml``.
    """

    action_magnitude: float = -1.0e-4
    action_rate: float = -1.0e-3
    joint_vel: float = -1.0e-2
    abnormal_robot: float = -100.0
    progress_context: float = 0.1
    ee_asset_distance: float = 0.1
    dense_success_reward: float = 0.1
    success_reward: float = 1.0
    peg_in_gripper: float = 0.1
    grasped_and_lifted: float = 2.0


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: int
    seed: int
    runtime_path: str
    runtime_episode: int
    runtime_catalog: tuple[tuple[str, int], ...]
    runtime_mode: str
    max_episode_length: int
    reward: RewardConfig


class _PegEpisode:
    """One native MuJoCo environment; instantiated only inside a worker."""

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
        self.ever_closed = False
        self.first_lift_given = False
        self.stable_release_streak = 0
        self.max_stable_release_streak = 0
        self.current_reset_id = -1
        self.reset_count = 0
        self.current_actor_obs = np.zeros(ACTOR_OBS_DIM, dtype=np.float32)
        self.current_critic_obs = np.zeros(CRITIC_OBS_DIM, dtype=np.float32)
        self.reset()

    def _load_runtime(self, runtime_path: str, runtime_episode: int) -> None:
        """Replace the MuJoCo model with one exported training runtime."""
        store = self._store_cache.get(runtime_path)
        if store is None:
            store = zarr.open(runtime_path, mode="r")
            self._store_cache[runtime_path] = store
        ends = np.asarray(store["meta/episode_ends"], dtype=np.int64)
        starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
        if runtime_episode < 0 or runtime_episode >= len(ends):
            raise IndexError(
                f"runtime episode {runtime_episode} outside [0, {len(ends)})"
            )
        start = int(starts[runtime_episode])
        end = int(ends[runtime_episode])

        # The globals are process-local: each worker owns one Python process.
        CL.SIM_DT = float(store.attrs["physics_dt_s"])
        CL.DECIM = int(store.attrs["decimation"])
        self.reward_dt_s = float(CL.SIM_DT * CL.DECIM)
        if not np.isclose(
            self.reward_dt_s, OMNIRESET_POLICY_DT_S, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "runtime policy dt does not match the saved OmniReset parent: "
                f"{self.reward_dt_s} != {OMNIRESET_POLICY_DT_S}"
            )
        context = Audit.build_episode_context(store, start, end)
        self.model = context["model"]
        self.ctrl = context["ctrl"]
        self.hole_pos_reference = context["hole_pos"]
        self.hole_quat_reference = context["hole_quat"]
        self.effective = context["effective"]
        self.torque_max = np.asarray(
            self.effective["controller"]["torque_max"], dtype=np.float64
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
        # The exporter records arm properties; append the known B3 finger
        # parameters so the layout remains exactly the old 171-D layout.
        joint_friction = np.concatenate(
            [
                np.asarray(
                    store["data/arm_joint_friction_static"][row],
                    dtype=np.float32,
                ).reshape(7),
                np.full(2, 0.5, dtype=np.float32),
            ]
        )
        joint_armature = np.concatenate(
            [
                np.asarray(
                    store["data/arm_joint_armature"][row], dtype=np.float32
                ).reshape(7),
                np.zeros(2, dtype=np.float32),
            ]
        )
        joint_stiffness = np.concatenate(
            [
                np.asarray(
                    store["data/arm_joint_stiffness"][row], dtype=np.float32
                ).reshape(7),
                np.full(2, 1000.0, dtype=np.float32),
            ]
        )
        joint_damping = np.concatenate(
            [
                np.asarray(
                    store["data/arm_joint_damping"][row], dtype=np.float32
                ).reshape(7),
                np.full(2, 14.0, dtype=np.float32),
            ]
        )
        result = np.concatenate(
            [
                *scene,
                joint_friction,
                joint_armature,
                joint_stiffness,
                joint_damping,
            ]
        ).astype(np.float32)
        # 171 total - 56 dynamic = 115 privileged physical values.
        if result.shape != (115,):
            raise ValueError(f"critic static layout is {result.shape}, expected (115,)")
        return result

    def _set_hole_pose(self, raw: np.ndarray) -> None:
        _, _, hole_pos, hole_quat = Replay.pose_in_robot_root(raw)
        hole_quat = np.asarray(hole_quat, dtype=np.float64)
        hole_quat /= np.linalg.norm(hole_quat)
        if int(self.model.body_parentid[self.ctrl.hole]) != 0:
            raise RuntimeError("PegHole body is no longer a direct child of world")
        self.model.body_pos[self.ctrl.hole] = hole_pos
        self.model.body_quat[self.ctrl.hole] = hole_quat

    def _actor_obs(self) -> np.ndarray:
        return self.obs_builder.step(self.data, self.prev_action).astype(
            np.float32, copy=False
        )

    def _critic_obs(self) -> np.ndarray:
        pose, _ = self.obs_builder._terms(self.data)
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
                self.ctrl.jac_arm(self.data) @ self.data.qvel[:7],
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
        self._set_hole_pose(raw)
        Replay.set_raw_state(self.model, self.data, self.ctrl, raw)
        self.obs_builder.reset(self.data)

        self.prev_action.fill(0.0)
        self.episode_step = 0
        self.episode_return = 0.0
        self.ever_closed = False
        self.first_lift_given = False
        self.stable_release_streak = 0
        self.max_stable_release_streak = 0
        self.current_reset_id = int(reset_id)
        self.reset_count += 1
        self.current_actor_obs = self._actor_obs()
        self.current_critic_obs = self._critic_obs()
        return self.current_actor_obs, self.current_critic_obs

    def _reward_and_task(
        self, action: np.ndarray, close: bool
    ) -> tuple[float, bool, bool, dict[str, float]]:
        reward_cfg = self.cfg.reward
        (
            position_error,
            orientation_error,
            _,
            official_pose,
            _,
            _,
        ) = Replay.assembly_metrics(
            self.data,
            self.ctrl,
            self.data.xpos[self.ctrl.hole].copy(),
            self.data.xquat[self.ctrl.hole].copy(),
        )

        hand_pos = self.data.xpos[self.ctrl.hand]
        hand_quat = self.data.xquat[self.ctrl.hand]
        tcp_pos = hand_pos + Q.quat_apply(hand_quat, CL.TCP_OFF)
        tcp_peg_distance = float(
            np.linalg.norm(self.data.xpos[self.ctrl.peg] - tcp_pos)
        )
        avg_finger = float(np.mean(self.data.qpos[7:9]))
        closure = 1.0 - float(np.clip(avg_finger / 0.04, 0.0, 1.0))
        peg_in_gripper = np.exp(-tcp_peg_distance / 0.05) * closure
        first_lift = (
            avg_finger < 0.02
            and float(self.data.xpos[self.ctrl.peg, 2]) > 0.025
            and not self.first_lift_given
        )
        self.first_lift_given = self.first_lift_given or first_lift

        robot_contact = self.ctrl.peg_robot_contact(self.data)
        self.ever_closed = self.ever_closed or close
        if self.ever_closed and not close and official_pose and not robot_contact:
            self.stable_release_streak += 1
        else:
            self.stable_release_streak = 0
        self.max_stable_release_streak = max(
            self.max_stable_release_streak, self.stable_release_streak
        )
        stable_release = self.stable_release_streak >= CL.STABLE_RELEASE_STEPS

        action_l2 = min(float(np.sum(np.square(action))), 1.0e4)
        action_rate_l2 = min(
            float(np.sum(np.square(action - self.prev_action))), 1.0e4
        )
        joint_velocity_l2 = min(
            float(np.sum(np.square(self.data.qvel[:7]))), 1.0e4
        )
        ee_asset_distance = 1.0 - np.tanh(tcp_peg_distance / 1.0)
        dense_success = 0.5 * (
            np.exp(-orientation_error / 1.0) + np.exp(-position_error / 1.0)
        )

        finite = bool(
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
        )
        abnormal = bool(
            not finite
            or np.any(np.abs(self.data.qvel[:7]) > 2.0 * CL.VEL_MAX)
        )
        # IsaacLab RewardManager.compute(dt=self.step_dt) multiplies every
        # configured weighted term by the policy step.  The audited MuJoCo
        # runtime is 1/480 s x 48 substeps = 0.1 s, matching the parent run's
        # 1/120 s x 12 substeps = 0.1 s.
        reward_dt_s = self.reward_dt_s
        components = {
            "action_magnitude": reward_cfg.action_magnitude * action_l2,
            "action_rate": reward_cfg.action_rate * action_rate_l2,
            "joint_vel": reward_cfg.joint_vel * joint_velocity_l2,
            "abnormal_robot": reward_cfg.abnormal_robot * float(abnormal),
            # ProgressContext updates task state but returns exactly zero.
            "progress_context": reward_cfg.progress_context * 0.0,
            "ee_asset_distance": reward_cfg.ee_asset_distance
            * ee_asset_distance,
            "dense_success_reward": reward_cfg.dense_success_reward
            * dense_success,
            "success_reward": reward_cfg.success_reward * float(official_pose),
            "peg_in_gripper": reward_cfg.peg_in_gripper * peg_in_gripper,
            "grasped_and_lifted": reward_cfg.grasped_and_lifted
            * float(first_lift),
        }
        components = {
            name: float(value * reward_dt_s) for name, value in components.items()
        }
        reward = float(sum(components.values()))
        diagnostics = {
            **components,
            "reward_dt_s": reward_dt_s,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "official_pose": float(official_pose),
            "close": float(close),
            "robot_contact": float(robot_contact),
            # Stable release is the external evaluation metric, not a reward.
            "stable_release": float(stable_release),
            "stable_release_streak": float(self.stable_release_streak),
            "tcp_peg_distance_m": tcp_peg_distance,
        }
        return reward, bool(stable_release), abnormal, diagnostics

    def step(self, action: np.ndarray) -> dict[str, Any]:
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        source_reset_id = self.current_reset_id
        if not np.isfinite(action).all():
            reward = self.cfg.reward.abnormal_robot * self.reward_dt_s
            self.episode_step += 1
            self.episode_return += reward
            episode = self._episode_summary(
                source_reset_id, strict_success=False, abnormal=True
            )
            actor_obs, critic_obs = self.reset()
            return {
                "actor_obs": actor_obs,
                "critic_obs": critic_obs,
                "reward": reward,
                "done": True,
                "timeout": False,
                "episode": episode,
            }

        # CL constants are process-local, but restore the recorded per-worker
        # effort limit explicitly so future profiles cannot silently drift it.
        CL.TAU_MAX = self.torque_max
        close = Replay.step_action(
            self.model,
            self.data,
            self.ctrl,
            action,
            finger_velocity_limits=FINGER_VELOCITY_LIMITS,
            independent_gripper=True,
            gripper_close_override=None,
            jacobian_point="link_origin",
            nullspace_stiffness=0.0,
            nullspace_damping_ratio=1.0,
            action_reference_blend=0.0,
            bias_compensation_scale=0.0,
            effort_scale=EFFORT_SCALE,
        )
        self.episode_step += 1
        reward, strict_success, abnormal, diagnostics = self._reward_and_task(
            action, close
        )
        self.episode_return += reward
        self.prev_action = action.copy()

        timeout = self.episode_step >= self.cfg.max_episode_length
        # The parent OmniReset task terminates only on timeout or an abnormal
        # robot state.  Pose success and stable release never end an episode.
        done = bool(timeout or abnormal)
        episode = None
        if done:
            episode = self._episode_summary(
                source_reset_id,
                strict_success=strict_success,
                abnormal=abnormal,
                diagnostics=diagnostics,
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
        }

    def _episode_summary(
        self,
        reset_id: int,
        strict_success: bool,
        abnormal: bool,
        diagnostics: dict[str, float] | None = None,
    ) -> dict[str, float]:
        diagnostics = diagnostics or {}
        return {
            "reward": float(self.episode_return),
            "length": float(self.episode_step),
            "strict_success": float(
                self.max_stable_release_streak >= CL.STABLE_RELEASE_STEPS
            ),
            "abnormal": float(abnormal),
            "max_stable_release_steps": float(self.max_stable_release_streak),
            "final_pose_success": float(diagnostics.get("official_pose", 0.0)),
            "reset_id": float(reset_id),
            "runtime_episode": float(self.current_runtime_episode),
            "runtime_load_count": float(self.runtime_load_count),
        }


def _worker_main(
    connection: Any, cfg: WorkerConfig, reset_pool: np.ndarray
) -> None:
    # Prevent every physics worker from creating its own BLAS thread team.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        episode = _PegEpisode(cfg, reset_pool)
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
                    "reward_policy_dt_s": episode.reward_dt_s,
                    "model_timestep_s": float(episode.model.opt.timestep),
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


class MujocoPegVecEnv:
    """Small RSL-RL VecEnv facade over process-parallel MuJoCo workers."""

    _worker_target = staticmethod(_worker_main)
    _process_name_prefix = "mujoco-peg"

    def __init__(
        self,
        reset_pool: np.ndarray,
        runtime_assignments: Sequence[tuple[str, int]],
        *,
        runtime_catalog_entries: Sequence[tuple[str, int]] | None = None,
        runtime_mode: str = "fixed_per_worker",
        seed: int,
        device: str = "cpu",
        max_episode_length: int = 160,
        reward: RewardConfig | None = None,
        startup_timeout_s: float = 180.0,
    ) -> None:
        if len(runtime_assignments) <= 0:
            raise ValueError("at least one runtime assignment is required")
        if runtime_mode not in {"fixed_per_worker", "resample_on_reset"}:
            raise ValueError(f"unsupported runtime mode: {runtime_mode}")
        catalog = tuple(runtime_catalog_entries or runtime_assignments)
        if runtime_mode == "resample_on_reset" and not catalog:
            raise ValueError("resample_on_reset requires a non-empty runtime catalog")
        if reset_pool.ndim != 2 or reset_pool.shape[1] != 57:
            raise ValueError(f"reset_pool must have shape (N, 57), got {reset_pool.shape}")
        self.num_envs = len(runtime_assignments)
        self.num_actions = ACTION_DIM
        self.max_episode_length = int(max_episode_length)
        self.device = torch.device(device)
        self.episode_length_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.cfg = {
            "engine": "mujoco",
            "actor_obs_dim": ACTOR_OBS_DIM,
            "critic_obs_dim": CRITIC_OBS_DIM,
            "action_dim": ACTION_DIM,
            "max_episode_length": self.max_episode_length,
            "reset_pool_rows": int(len(reset_pool)),
            "runtime_mode": runtime_mode,
            "runtime_catalog_rows": len(catalog),
            "termination_terms": ["timeout", "abnormal_robot"],
            "reward": vars(reward or RewardConfig()),
        }
        self._closed = False
        self._ctx = mp.get_context("fork")
        self._connections: list[Any] = []
        self._processes: list[mp.Process] = []
        self.worker_metadata: list[dict[str, Any]] = []
        worker_reward = reward or RewardConfig()

        for worker_id, (runtime_path, runtime_episode) in enumerate(
            runtime_assignments
        ):
            parent, child = self._ctx.Pipe()
            cfg = WorkerConfig(
                worker_id=worker_id,
                seed=int(seed + 1009 * worker_id),
                runtime_path=str(runtime_path),
                runtime_episode=int(runtime_episode),
                runtime_catalog=catalog,
                runtime_mode=runtime_mode,
                max_episode_length=self.max_episode_length,
                reward=worker_reward,
            )
            process = self._ctx.Process(
                target=self._worker_target,
                args=(child, cfg, reset_pool),
                name=f"{self._process_name_prefix}-{worker_id:02d}",
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

        actor_obs = []
        critic_obs = []
        try:
            for worker_id, connection in enumerate(self._connections):
                if not connection.poll(startup_timeout_s):
                    raise TimeoutError(
                        f"MuJoCo worker {worker_id} did not start within "
                        f"{startup_timeout_s:.1f}s"
                    )
                message = connection.recv()
                self._raise_worker_error(message, worker_id)
                if message[0] != "ready":
                    raise RuntimeError(
                        f"worker {worker_id} sent unexpected startup message {message[0]}"
                    )
                actor_obs.append(message[1])
                critic_obs.append(message[2])
                self.worker_metadata.append(message[3])
        except BaseException:
            self.close(force=True)
            raise
        self._observations = self._make_tensordict(actor_obs, critic_obs)

    def _raise_worker_error(self, message: Any, worker_id: int) -> None:
        if not message:
            raise RuntimeError(f"worker {worker_id} returned an empty message")
        if message[0] == "error":
            raise RuntimeError(f"MuJoCo worker {worker_id} failed:\n{message[1]}")

    def _make_tensordict(
        self, actor_obs: Sequence[np.ndarray], critic_obs: Sequence[np.ndarray]
    ) -> TensorDict:
        actor = torch.as_tensor(
            np.stack(actor_obs), dtype=torch.float32, device=self.device
        )
        critic = torch.as_tensor(
            np.stack(critic_obs), dtype=torch.float32, device=self.device
        )
        return TensorDict(
            {"policy": actor, "critic": critic},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def get_observations(self) -> TensorDict:
        return self._observations

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self._closed:
            raise RuntimeError("cannot step a closed MuJoCo vector environment")
        action_np = actions.detach().cpu().numpy().astype(np.float32, copy=False)
        if action_np.shape != (self.num_envs, ACTION_DIM):
            raise ValueError(
                f"actions must have shape ({self.num_envs}, {ACTION_DIM}), "
                f"got {action_np.shape}"
            )
        for connection, action in zip(self._connections, action_np, strict=True):
            connection.send(("step", action))

        actor_obs: list[np.ndarray] = []
        critic_obs: list[np.ndarray] = []
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.empty(self.num_envs, dtype=np.bool_)
        timeouts = np.empty(self.num_envs, dtype=np.bool_)
        completed: list[dict[str, float]] = []
        task_rows: list[dict[str, float]] = []
        for worker_id, connection in enumerate(self._connections):
            message = connection.recv()
            self._raise_worker_error(message, worker_id)
            if message[0] != "step":
                raise RuntimeError(
                    f"worker {worker_id} sent unexpected step message {message[0]}"
                )
            result = message[1]
            actor_obs.append(result["actor_obs"])
            critic_obs.append(result["critic_obs"])
            rewards[worker_id] = result["reward"]
            dones[worker_id] = result["done"]
            timeouts[worker_id] = result["timeout"]
            if result["episode"] is not None:
                completed.append(result["episode"])
            if result.get("task") is not None:
                task_rows.append(result["task"])

        self._observations = self._make_tensordict(actor_obs, critic_obs)
        reward_tensor = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        done_tensor = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        timeout_tensor = torch.as_tensor(
            timeouts, dtype=torch.bool, device=self.device
        )
        self.episode_length_buf += 1
        self.episode_length_buf[done_tensor] = 0
        extras: dict[str, Any] = {"time_outs": timeout_tensor}
        if completed:
            extras["log"] = {
                key: torch.tensor(
                    [episode[key] for episode in completed],
                    dtype=torch.float32,
                    device=self.device,
                )
                for key in completed[0]
            }
        if len(task_rows) == self.num_envs:
            extras["task"] = {
                key: torch.tensor(
                    [row[key] for row in task_rows],
                    dtype=torch.float32,
                    device=self.device,
                )
                for key in task_rows[0]
            }
        return self._observations, reward_tensor, done_tensor, extras

    def reset(self, reset_ids: Sequence[int] | None = None) -> TensorDict:
        """Reset all workers, optionally to explicit reset-pool rows."""
        if self._closed:
            raise RuntimeError("cannot reset a closed MuJoCo vector environment")
        if reset_ids is None:
            values: list[int | None] = [None] * self.num_envs
        else:
            if len(reset_ids) != self.num_envs:
                raise ValueError(
                    f"reset_ids must have {self.num_envs} rows, got {len(reset_ids)}"
                )
            values = [int(value) for value in reset_ids]
        for connection, reset_id in zip(
            self._connections, values, strict=True
        ):
            connection.send(("reset", reset_id))
        actor_obs: list[np.ndarray] = []
        critic_obs: list[np.ndarray] = []
        for worker_id, connection in enumerate(self._connections):
            message = connection.recv()
            self._raise_worker_error(message, worker_id)
            if message[0] != "reset":
                raise RuntimeError(
                    f"worker {worker_id} sent unexpected reset message {message[0]}"
                )
            actor_obs.append(message[1])
            critic_obs.append(message[2])
        self.episode_length_buf.zero_()
        self._observations = self._make_tensordict(actor_obs, critic_obs)
        return self._observations

    def close(self, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if not force:
            for connection, process in zip(
                self._connections, self._processes, strict=True
            ):
                if process.is_alive():
                    try:
                        connection.send(("close",))
                    except (BrokenPipeError, EOFError, OSError):
                        pass
            for connection, process in zip(
                self._connections, self._processes, strict=True
            ):
                if process.is_alive() and connection.poll(5.0):
                    try:
                        connection.recv()
                    except (EOFError, OSError):
                        pass
        for process in self._processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for connection in self._connections:
            try:
                connection.close()
            except OSError:
                pass

    def __enter__(self) -> "MujocoPegVecEnv":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
