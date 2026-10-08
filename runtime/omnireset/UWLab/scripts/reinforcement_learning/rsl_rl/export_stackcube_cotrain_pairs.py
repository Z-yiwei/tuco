# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Export exact-reset IsaacSim teacher trajectories for StackCube co-training.

Reset IDs are read from a MuJoCo expert zarr.  The official reset state is
injected once before rollout, then the RSL-RL teacher runs closed-loop on live
IsaacSim observations.  Output is episode-major and frame-aligned with B147.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Export paired StackCube IsaacSim rollouts.")
parser.add_argument(
    "--task",
    default="OmniReset-FrankaFr3Gripper-RelCartesianOSC-State-Finetune-Play-v0",
)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--reset_state", required=True)
parser.add_argument("--reset_indices_zarr", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--episode_steps", type=int, default=160)
parser.add_argument("--max_resets", type=int, default=0)
parser.add_argument("--preserve_controller_events", action="store_true")
parser.add_argument(
    "--success_position_threshold",
    type=float,
    default=None,
    help=(
        "optional success-position override in meters; must be supplied together "
        "with --success_orientation_threshold"
    ),
)
parser.add_argument(
    "--success_orientation_threshold",
    type=float,
    default=None,
    help=(
        "optional success-orientation override in radians; must be supplied together "
        "with --success_position_threshold"
    ),
)
parser.add_argument(
    "--fixed_scene_dynamics",
    action="store_true",
    help=(
        "pin the eight B3 startup material/mass events to one midpoint/nominal "
        "scene profile; effective values are still recorded in data/*"
    ),
)
parser.add_argument(
    "--source_policy_type", choices=("rsl_rl", "mlp_step", "dp_chunk8"), default="rsl_rl"
)
parser.add_argument("--action_horizon", type=int, default=8)
parser.add_argument("--ddpm_eval_steps", type=int, default=50)
parser.add_argument("--scheduler", choices=("ddpm", "ddim"), default="ddpm")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--overwrite", action="store_true")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if (args_cli.success_position_threshold is None) != (
    args_cli.success_orientation_threshold is None
):
    parser.error(
        "--success_position_threshold and --success_orientation_threshold "
        "must be supplied together"
    )
for name in ("success_position_threshold", "success_orientation_threshold"):
    value = getattr(args_cli, name)
    if value is not None and value <= 0:
        parser.error(f"--{name} must be positive")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zarr  # noqa: E402
from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: E402
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # noqa: E402

from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import uwlab_tasks  # noqa: F401, E402
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import (  # noqa: E402
    sample_state_data_set,
)

RELEASE_ROOT = Path(__file__).resolve().parents[6]
# The public export path uses rsl_rl; optional legacy adapters are not imported.
if args_cli.source_policy_type == "mlp_step":
    sys.path.insert(0, str(RELEASE_ROOT / "third_party/omnireset_eval/scripts/sim2sim/franka"))
    from mlp_util import load_mlp
elif args_cli.source_policy_type == "dp_chunk8":
    raise ValueError("dp_chunk8 export is not part of the TUCO expert-data profile")


FIXED_SCENE_PROFILE = {
    "robot_material": {
        "static_friction_range": (0.75, 0.75),
        "dynamic_friction_range": (0.60, 0.60),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "insertive_object_material": {
        "static_friction_range": (1.50, 1.50),
        "dynamic_friction_range": (1.40, 1.40),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "receptive_object_material": {
        "static_friction_range": (0.40, 0.40),
        "dynamic_friction_range": (0.325, 0.325),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "table_material": {
        "static_friction_range": (0.45, 0.45),
        "dynamic_friction_range": (0.35, 0.35),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 1,
    },
    "randomize_robot_mass": {"mass_distribution_params": (1.0, 1.0)},
    "randomize_insertive_object_mass": {"mass_distribution_params": (0.11, 0.11)},
    "randomize_receptive_object_mass": {"mass_distribution_params": (1.0, 1.0)},
    "randomize_table_mass": {"mass_distribution_params": (1.0, 1.0)},
}


class MLPSourcePolicy:
    def __init__(self, checkpoint, device, num_envs):
        self.model, norm = load_mlp(checkpoint, device=device, use_ema=True)
        self.device = torch.device(device)
        self.s_mean = torch.as_tensor(norm["s_mean"], device=self.device)
        self.s_std = torch.as_tensor(norm["s_std"], device=self.device)
        self.a_center = torch.as_tensor(norm["a_center"], device=self.device)
        self.a_scale = torch.as_tensor(norm["a_scale"], device=self.device)
        self.n_obs = int(norm["n_obs"])
        self.num_envs = num_envs
        self.history = None
        self._history_reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.last_chunk_start = np.ones(num_envs, dtype=np.uint8)

    def __call__(self, obs):
        obs = policy_obs(obs).to(self.device)
        if self.history is None:
            self.history = obs[:, None, :].repeat(1, self.n_obs, 1)
            self._history_reset_mask[:] = False
        else:
            self.history = torch.cat([self.history[:, 1:], obs[:, None, :]], dim=1)
            if self._history_reset_mask.any():
                self.history[self._history_reset_mask] = obs[self._history_reset_mask, None, :].repeat(
                    1, self.n_obs, 1
                )
                self._history_reset_mask[:] = False
        normalized = (self.history - self.s_mean) / self.s_std
        return self.model(normalized) * self.a_scale + self.a_center

    def reset(self, dones):
        self._history_reset_mask |= dones.to(device=self.device, dtype=torch.bool)


class DPChunkSourcePolicy:
    def __init__(self, checkpoint, device, num_envs, action_horizon, eval_steps, scheduler_name):
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.obs_horizon = int(ckpt["obs_horizon"])
        self.pred_horizon = int(ckpt["pred_horizon"])
        self.action_dim = int(ckpt["action_dim"])
        self.action_horizon = int(action_horizon)
        if self.action_horizon != 8:
            raise ValueError("dp_chunk8 requires --action_horizon 8")
        self.model = DPModel(
            obs_dim=int(ckpt["obs_dim"]),
            obs_horizon=self.obs_horizon,
            action_dim=self.action_dim,
            cond_hidden=int(ckpt.get("cond_hidden", 256)),
            unet_hidden_dims=(int(ckpt.get("unet_h1", 256)), int(ckpt.get("unet_h2", 512))),
        ).to(self.device)
        self.model.load_state_dict(ckpt.get("ema_state_dict", ckpt["model_state_dict"]))
        self.model.eval()
        stats = ckpt["norm_stats"]
        self.s_mean, self.s_std = stats["s_mean"].to(self.device), stats["s_std"].to(self.device)
        self.a_mean, self.a_std = stats["a_mean"].to(self.device), stats["a_std"].to(self.device)
        scheduler_cls = DDIMScheduler if scheduler_name == "ddim" else DDPMScheduler
        scheduler_kwargs = dict(
            num_train_timesteps=int(ckpt["ddpm_steps"]),
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
            clip_sample=False,
        )
        if scheduler_name == "ddim":
            scheduler_kwargs.update(set_alpha_to_one=True, steps_offset=0)
        self.scheduler = scheduler_cls(**scheduler_kwargs)
        self.scheduler.set_timesteps(eval_steps)
        self.history = None
        self._history_reset_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.plan = None
        self.plan_index = self.action_horizon
        self.last_chunk_start = np.zeros(num_envs, dtype=np.uint8)

    def __call__(self, obs):
        obs = policy_obs(obs).to(self.device)
        if self.history is None:
            self.history = obs[:, None, :].repeat(1, self.obs_horizon, 1)
            self._history_reset_mask[:] = False
        else:
            self.history = torch.cat([self.history[:, 1:], obs[:, None, :]], dim=1)
            if self._history_reset_mask.any():
                self.history[self._history_reset_mask] = obs[self._history_reset_mask, None, :].repeat(
                    1, self.obs_horizon, 1
                )
                self._history_reset_mask[:] = False
        replan = self.plan_index >= self.action_horizon
        self.last_chunk_start.fill(int(replan))
        if replan:
            normalized = (self.history - self.s_mean) / self.s_std
            sample = torch.randn(
                self.num_envs, self.pred_horizon, self.action_dim, device=self.device
            )
            for timestep in self.scheduler.timesteps:
                batch_t = torch.full(
                    (self.num_envs,), int(timestep.item()), device=self.device, dtype=torch.long
                )
                prediction = self.model(normalized, sample, batch_t)
                sample = self.scheduler.step(prediction, timestep, sample).prev_sample
            self.plan = (sample * self.a_std + self.a_mean)[:, : self.action_horizon].clone()
            self.plan_index = 0
        action = self.plan[:, self.plan_index]
        self.plan_index += 1
        return action

    def reset(self, dones):
        self._history_reset_mask |= dones.to(device=self.device, dtype=torch.bool)


def md5(path: str | Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_obs(obs):
    if isinstance(obs, tuple):
        obs = obs[0]
    if hasattr(obs, "keys"):
        obs = obs["policy"] if "policy" in obs else next(iter(obs.values()))
    return obs


def capture_raw_states(env) -> torch.Tensor:
    """Canonical 57-D env-local state used by the MuJoCo replay tools."""
    scene = env.unwrapped.scene
    robot = scene["robot"]
    insertive = scene["insertive_object"]
    receptive = scene["receptive_object"]
    origins = scene.env_origins

    def local_root(asset):
        state = asset.data.root_state_w.clone()
        state[:, :3] -= origins
        return state

    robot_root = local_root(robot)
    insertive_root = local_root(insertive)
    receptive_root = local_root(receptive)
    return torch.cat(
        [
            robot.data.joint_pos,
            robot.data.joint_vel,
            robot_root[:, :7],
            robot_root[:, 7:13],
            insertive_root[:, :7],
            insertive_root[:, 7:13],
            receptive_root[:, :7],
            receptive_root[:, 7:13],
        ],
        dim=-1,
    )


def capture_runtime(env, action):
    """Read the effective B3 controller and scene properties, not config intent."""
    base = env.unwrapped
    arm = base.action_manager.get_term("arm")
    gripper = base.action_manager.get_term("gripper")
    # process_actions is deterministic and env.step processes the same command
    # again on the canonical path.  This exposes the actual gripper target.
    gripper.process_actions(action[:, arm.action_dim : arm.action_dim + gripper.action_dim])
    robot = base.scene["robot"]
    gripper_joint_ids = gripper._joint_ids
    out = {
        "gripper_processed_action": gripper.processed_actions,
        "gripper_joint_stiffness": robot.data.joint_stiffness[:, gripper_joint_ids],
        "gripper_joint_damping": robot.data.joint_damping[:, gripper_joint_ids],
        "gripper_joint_effort_limit": robot.root_physx_view.get_dof_max_forces()[
            :, gripper_joint_ids
        ],
        "arm_scale": arm._scale,
        "arm_kp": arm._kp,
        "arm_kd": arm._kd,
        "arm_torque_max": arm._torque_max,
        "arm_joint_armature": robot.data.joint_armature[:, arm._joint_ids],
        "arm_joint_friction_dynamic": robot.data.joint_dynamic_friction_coeff[:, arm._joint_ids],
        "arm_joint_friction_viscous": robot.data.joint_viscous_friction_coeff[:, arm._joint_ids],
    }
    for asset_name in ("robot", "insertive_object", "receptive_object", "table"):
        asset = base.scene[asset_name]
        for property_name, getter in (
            ("material_properties", asset.root_physx_view.get_material_properties),
            ("body_masses", asset.root_physx_view.get_masses),
            ("body_coms", asset.root_physx_view.get_coms),
            ("body_inertias", asset.root_physx_view.get_inertias),
        ):
            out[f"{asset_name}_{property_name}"] = getter().reshape(base.num_envs, -1)
    result = {}
    for key, value in out.items():
        value = value.detach()
        if value.ndim == 1:
            value = value.unsqueeze(0).expand(base.num_envs, -1)
        result[key] = value.cpu().numpy().astype(np.float32)
    return result


def restore_partial_state(env, state, env_ids):
    """Restore the reset file's robot/task entries without a bootstrap dataset lookup."""
    scene = env.unwrapped.scene
    for asset_name, asset_state in state["articulation"].items():
        articulation = scene[asset_name]
        root_pose = asset_state["root_pose"].clone()
        root_pose[:, :3] += scene.env_origins[env_ids]
        articulation.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        articulation.write_root_velocity_to_sim(asset_state["root_velocity"], env_ids=env_ids)
        articulation.write_joint_state_to_sim(
            asset_state["joint_position"], asset_state["joint_velocity"], env_ids=env_ids
        )
        articulation.set_joint_position_target(asset_state["joint_position"], env_ids=env_ids)
        articulation.set_joint_velocity_target(asset_state["joint_velocity"], env_ids=env_ids)
    for asset_name, asset_state in state["rigid_object"].items():
        rigid_object = scene[asset_name]
        root_pose = asset_state["root_pose"].clone()
        root_pose[:, :3] += scene.env_origins[env_ids]
        rigid_object.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(asset_state["root_velocity"], env_ids=env_ids)


def pin_runtime(
    env_cfg: ManagerBasedRLEnvCfg,
    preserve_controller_events: bool = False,
    fixed_scene_dynamics: bool = False,
) -> None:
    env_cfg.observations.policy.enable_corruption = False
    for name in (
        "randomize_gripper_actuator_parameters",
        "randomize_osc_gains",
        "randomize_arm_sysid",
    ):
        term = getattr(env_cfg.events, name, None)
        if term is None:
            continue
        if term.mode == "reset":
            term.mode = "startup"
        if preserve_controller_events:
            continue
        for key in (
            "scale_range",
            "stiffness_distribution_params",
            "damping_distribution_params",
        ):
            if key in term.params:
                term.params[key] = (1.0, 1.0)
        if "delay_range" in term.params:
            term.params["delay_range"] = (0, 0)

    if fixed_scene_dynamics:
        missing = []
        for name, overrides in FIXED_SCENE_PROFILE.items():
            term = getattr(env_cfg.events, name, None)
            if term is None:
                missing.append(name)
                continue
            if term.mode != "startup":
                raise ValueError(
                    f"fixed-scene event {name} must run at startup, got mode={term.mode!r}"
                )
            term.params.update(overrides)
        if missing:
            raise ValueError(f"fixed-scene events missing from task config: {missing}")

    # Same determinism settings used by the existing offline state collector.
    env_cfg.sim.physx.enable_enhanced_determinism = True
    env_cfg.sim.physx.gpu_max_num_partitions = 1
    env_cfg.sim.physx.bounce_threshold_velocity = 0.5
    env_cfg.sim.physx.friction_correlation_distance = 0.025
    env_cfg.sim.physx.max_position_iteration_count = 32


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    if not args_cli.checkpoint:
        raise ValueError("--checkpoint is required")

    source_b = zarr.open(args_cli.reset_indices_zarr, mode="r")
    reset_indices = np.asarray(source_b["meta/reset_indices"], dtype=np.int32)
    if args_cli.max_resets > 0:
        reset_indices = reset_indices[: args_cli.max_resets]
    if len(reset_indices) == 0 or len(reset_indices) != len(np.unique(reset_indices)):
        raise ValueError("reset indices must be non-empty and unique")

    # The wrapper performs one bootstrap reset before the exact reset IDs are
    # injected below.  Point that bootstrap at the same explicitly supplied
    # StackCube dataset; otherwise a cwd-relative dataset can silently resolve
    # to a different robot/reset family (for example a 12-DoF reset).
    # RslRlVecEnvWrapper performs a bootstrap env.reset() before we inject the
    # requested IDs.  Disable the dataset term for that bootstrap so its asset
    # auto-detection cannot select an unrelated 12-DoF reset family.
    env_cfg.events.reset_from_reset_states = None

    pin_runtime(
        env_cfg,
        args_cli.preserve_controller_events,
        args_cli.fixed_scene_dynamics,
    )
    env_cfg.scene.num_envs = len(reset_indices)
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = cli_args.sanitize_rsl_rl_cfg(agent_cfg)
    resume_path = retrieve_file_path(args_cli.checkpoint)
    env_cfg.log_dir = str(Path(resume_path).parent)

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)
    insertive_object_usd = str(
        base_env.unwrapped.scene["insertive_object"].cfg.spawn.usd_path
    ).replace("\\", "/")
    receptive_object_usd = str(
        base_env.unwrapped.scene["receptive_object"].cfg.spawn.usd_path
    ).replace("\\", "/")
    if not insertive_object_usd.endswith(
        "/Props/Custom/InsertiveCube/insertive_cube.usd"
    ) or not receptive_object_usd.endswith(
        "/Props/Custom/ReceptiveCube/receptive_cube.usd"
    ):
        raise ValueError(
            "export_stackcube_cotrain_pairs.py requires the StackCube asset pair; "
            "pass the Hydra overrides "
            "'env.scene.insertive_object=cube env.scene.receptive_object=cube'. "
            f"Resolved insertive={insertive_object_usd!r}, "
            f"receptive={receptive_object_usd!r}"
        )
    env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
    device = env.unwrapped.device

    task_command = env.unwrapped.command_manager.get_term("task_command")
    metadata_success_position_threshold = float(task_command.success_position_threshold)
    metadata_success_orientation_threshold = float(task_command.success_orientation_threshold)
    if args_cli.success_position_threshold is not None:
        task_command.success_position_threshold = float(args_cli.success_position_threshold)
        task_command.success_orientation_threshold = float(
            args_cli.success_orientation_threshold
        )
        success_threshold_source = "cli_override"
    else:
        success_threshold_source = "receptive_object_metadata"
    success_position_threshold = float(task_command.success_position_threshold)
    success_orientation_threshold = float(task_command.success_orientation_threshold)
    print(
        "[stackcube-Isaac-pairs] success thresholds: "
        f"position={success_position_threshold:g} m, "
        f"orientation={success_orientation_threshold:g} rad "
        f"(source={success_threshold_source}; "
        f"metadata={metadata_success_position_threshold:g} m/"
        f"{metadata_success_orientation_threshold:g} rad)",
        flush=True,
    )

    policy_nn = None
    if args_cli.source_policy_type == "rsl_rl":
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=device)
        policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))
    elif args_cli.source_policy_type == "mlp_step":
        policy = MLPSourcePolicy(resume_path, device, len(reset_indices))
    else:
        policy = DPChunkSourcePolicy(
            resume_path,
            device,
            len(reset_indices),
            args_cli.action_horizon,
            args_cli.ddpm_eval_steps,
            args_cli.scheduler,
        )

    reset_dataset = torch.load(args_cli.reset_state, map_location="cpu", weights_only=False)
    reset_ids_t = torch.as_tensor(reset_indices, dtype=torch.long, device=device)
    selected = sample_state_data_set(reset_dataset, reset_ids_t, device)
    env_ids = torch.arange(len(reset_indices), dtype=torch.long, device=device)
    # Reset files intentionally contain only the robot and task objects, not
    # static scene objects such as the table.
    restore_partial_state(env, selected["initial_state"], env_ids)
    env.unwrapped.scene.write_data_to_sim()
    env.unwrapped.sim.forward()
    env.unwrapped.observation_manager.reset(env_ids)
    env.unwrapped.action_manager.reset(env_ids)
    env.unwrapped.reward_manager.reset(env_ids)
    env.unwrapped.termination_manager.reset(env_ids)
    env.unwrapped.episode_length_buf[env_ids] = 0
    env.unwrapped.obs_buf = env.unwrapped.observation_manager.compute(update_history=True)
    obs = env.get_observations()

    robot = env.unwrapped.scene["robot"]
    insertive = env.unwrapped.scene["insertive_object"]
    success_term = env.unwrapped.reward_manager.get_term_cfg("progress_context").func
    # ``ManagerBasedRLEnv.step`` auto-resets done environments before returning.
    # Keep the historical any-auto-reset signal for audit, but only count success
    # while the explicitly injected original episode is still active.
    success_seen_legacy = torch.zeros(len(reset_indices), dtype=torch.bool, device=device)
    success_before_done = torch.zeros(len(reset_indices), dtype=torch.bool, device=device)
    active = torch.ones(len(reset_indices), dtype=torch.bool, device=device)
    first_success_step = torch.full(
        (len(reset_indices),), -1, dtype=torch.int32, device=device
    )
    early_done_seen = torch.zeros(len(reset_indices), dtype=torch.bool, device=device)
    first_done_step = torch.full(
        (len(reset_indices),), -1, dtype=torch.int32, device=device
    )

    obs_rows = []
    action_rows = []
    cube_pos_rows = []
    raw_rows = []
    runtime_rows = {}
    chunk_start_rows = []
    success_by_step_rows = []
    done_by_step_rows = []
    active_before_step_rows = []
    for step in range(args_cli.episode_steps):
        with torch.inference_mode():
            action = policy(obs)
        obs_tensor = policy_obs(obs)
        cube_pos_root, _ = math_utils.subtract_frame_transforms(
            robot.data.root_link_pos_w,
            robot.data.root_link_quat_w,
            insertive.data.root_pos_w,
            insertive.data.root_quat_w,
        )
        obs_rows.append(obs_tensor.detach().cpu().numpy().astype(np.float32))
        action_rows.append(action.detach().cpu().numpy().astype(np.float32))
        chunk_start_rows.append(
            np.asarray(
                getattr(policy, "last_chunk_start", np.ones(len(reset_indices), dtype=np.uint8)),
                dtype=np.uint8,
            ).copy()
        )
        cube_pos_rows.append(cube_pos_root.detach().cpu().numpy().astype(np.float32))
        raw_rows.append(capture_raw_states(env).detach().cpu().numpy().astype(np.float32))
        runtime = capture_runtime(env, action)
        for key, value in runtime.items():
            runtime_rows.setdefault(key, []).append(value)

        obs_new, _, dones, _ = env.step(action)
        obs = obs_new
        step_success = success_term.success.bool().clone()
        done_mask = dones.bool()
        active_before_step = active.clone()
        accounted_success = step_success & active_before_step
        newly_successful = accounted_success & (first_success_step < 0)
        first_success_step[newly_successful] = step + 1
        success_before_done |= accounted_success
        success_seen_legacy |= step_success
        success_by_step_rows.append(step_success.detach().cpu().numpy())
        done_by_step_rows.append(done_mask.detach().cpu().numpy())
        active_before_step_rows.append(active_before_step.detach().cpu().numpy())
        newly_done = done_mask & (first_done_step < 0)
        first_done_step[newly_done] = step + 1
        if step + 1 < args_cli.episode_steps:
            early_done_seen |= done_mask
        active &= ~done_mask
        if isinstance(policy, (MLPSourcePolicy, DPChunkSourcePolicy)):
            policy.reset(dones)
        if policy_nn is not None:
            policy_nn.reset(dones)
        if (step + 1) % 20 == 0:
            print(
                f"[stackcube-Isaac-pairs] step={step+1}/{args_cli.episode_steps} "
                f"safe_success={int(success_before_done.sum())}/{len(reset_indices)} "
                f"legacy_any_reset_success={int(success_seen_legacy.sum())}/{len(reset_indices)} "
                f"early_done={int(early_done_seen.sum())}",
                flush=True,
            )

    # Convert [T, N, D] to episode-major [N*T, D].
    def episode_major(rows):
        arr = np.stack(rows, axis=1)
        return arr.reshape(len(reset_indices) * args_cli.episode_steps, *arr.shape[2:])

    state = episode_major(obs_rows)
    action = episode_major(action_rows)
    cube_position = episode_major(cube_pos_rows)
    raw_state = episode_major(raw_rows)
    action_chunk_start = episode_major(chunk_start_rows)
    runtime_data = {key: episode_major(rows) for key, rows in runtime_rows.items()}
    success_by_step = np.stack(success_by_step_rows, axis=1)
    done_by_step = np.stack(done_by_step_rows, axis=1)
    active_before_step = np.stack(active_before_step_rows, axis=1)
    episode_ends = np.arange(
        args_cli.episode_steps,
        (len(reset_indices) + 1) * args_cli.episode_steps,
        args_cli.episode_steps,
        dtype=np.int64,
    )

    out = Path(args_cli.out).resolve()
    if out.exists():
        if not args_cli.overwrite:
            raise FileExistsError(f"output exists (pass --overwrite): {out}")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(out), mode="w")
    root.create_dataset("data/state", data=state, chunks=(2048, state.shape[1]))
    root.create_dataset("data/action", data=action, chunks=(2048, action.shape[1]))
    root.create_dataset("data/cube_position_isaac", data=cube_position, chunks=(2048, 3))
    root.create_dataset("data/raw_state", data=raw_state, chunks=(2048, raw_state.shape[1]))
    root.create_dataset("data/action_chunk_start", data=action_chunk_start)
    for key, value in runtime_data.items():
        root.create_dataset(
            f"data/{key}", data=value, chunks=(min(2048, len(value)), *value.shape[1:])
        )
    root.create_dataset("meta/episode_ends", data=episode_ends)
    root.create_dataset("meta/reset_indices", data=reset_indices)
    # ``success_seen`` is now the safe, original-episode-only compatibility alias.
    root.create_dataset("meta/success_seen", data=success_before_done.detach().cpu().numpy())
    root.create_dataset(
        "meta/success_before_done", data=success_before_done.detach().cpu().numpy()
    )
    root.create_dataset(
        "meta/success_seen_legacy_any_autoreset",
        data=success_seen_legacy.detach().cpu().numpy(),
    )
    root.create_dataset(
        "meta/first_success_step", data=first_success_step.detach().cpu().numpy()
    )
    root.create_dataset("meta/valid", data=(~early_done_seen).detach().cpu().numpy())
    root.create_dataset("meta/first_done_step", data=first_done_step.detach().cpu().numpy())
    root.create_dataset("meta/success_by_step", data=success_by_step)
    root.create_dataset("meta/done_by_step", data=done_by_step)
    root.create_dataset("meta/active_before_step", data=active_before_step)
    for key in ("source_reset_indices", "face_labels", "face_label_ids"):
        source_key = f"meta/{key}"
        if source_key not in source_b:
            continue
        values = np.asarray(source_b[source_key])
        if len(values) < len(reset_indices):
            raise ValueError(
                f"{source_key} has {len(values)} rows for {len(reset_indices)} reset indices"
            )
        root.create_dataset(source_key, data=values[: len(reset_indices)])
    root.attrs.update(
        {
            "task": args_cli.task,
            "definition": (
                "exact official reset once; closed-loop IsaacSim source policy; rich B3 runtime; "
                "episode-major"
            ),
            "checkpoint": str(Path(resume_path).resolve()),
            "checkpoint_md5": md5(resume_path),
            "reset_state": str(Path(args_cli.reset_state).resolve()),
            "reset_state_md5": md5(args_cli.reset_state),
            "source_b": str(Path(args_cli.reset_indices_zarr).resolve()),
            "episode_steps": int(args_cli.episode_steps),
            "physics_dt_s": float(env.unwrapped.physics_dt),
            "policy_dt_s": float(env.unwrapped.step_dt),
            "decimation": int(round(env.unwrapped.step_dt / env.unwrapped.physics_dt)),
            "robot_body_names": list(robot.body_names),
            "insertive_object_usd": insertive_object_usd,
            "receptive_object_usd": receptive_object_usd,
            "seed": int(args_cli.seed),
            "controller_profile": "stage2_deploy",
            "source_policy_type": args_cli.source_policy_type,
            "source_policy_interface": (
                "action_chunk8" if args_cli.source_policy_type == "dp_chunk8" else "action_step"
            ),
            "success_position_threshold": success_position_threshold,
            "success_orientation_threshold": success_orientation_threshold,
            "success_threshold_source": success_threshold_source,
            "metadata_success_position_threshold": metadata_success_position_threshold,
            "metadata_success_orientation_threshold": metadata_success_orientation_threshold,
            "success_accounting": (
                "meta/success_seen == meta/success_before_done; success is counted on the "
                "first-done transition but never after IsaacLab auto-reset; "
                "meta/success_seen_legacy_any_autoreset preserves the historical signal"
            ),
            "action_horizon": 8 if args_cli.source_policy_type == "dp_chunk8" else 1,
            "smoke_only": bool(args_cli.max_resets > 0 or args_cli.episode_steps < 160),
            "diffusion_scheduler": args_cli.scheduler if args_cli.source_policy_type == "dp_chunk8" else "none",
            "diffusion_eval_steps": int(args_cli.ddpm_eval_steps) if args_cli.source_policy_type == "dp_chunk8" else 0,
            "preserve_controller_events": bool(args_cli.preserve_controller_events),
            "fixed_scene_dynamics": bool(args_cli.fixed_scene_dynamics),
            "fixed_scene_profile_json": json.dumps(
                FIXED_SCENE_PROFILE if args_cli.fixed_scene_dynamics else {},
                sort_keys=True,
            ),
            "reset_manifest_attrs_json": json.dumps(
                dict(source_b.attrs), sort_keys=True, default=str
            ),
            "runtime_pin_json": json.dumps(
                (
                    {
                        "observation_corruption": False,
                        "controller_event_distributions": "preserved",
                        "controller_event_sampling": "once at startup per env",
                        "effective_values": "recorded per frame in data/*",
                        "enhanced_determinism": True,
                        "gpu_max_num_partitions": 1,
                    }
                    if args_cli.preserve_controller_events
                    else {
                        "observation_corruption": False,
                        "actuator_ranges": [1.0, 1.0],
                        "arm_delay": [0, 0],
                        "enhanced_determinism": True,
                        "gpu_max_num_partitions": 1,
                    }
                ),
                sort_keys=True,
            ),
        }
    )
    print(
        f"[stackcube-Isaac-pairs] wrote {out}: {len(reset_indices)} demos, "
        f"{len(state)} frames, Isaac safe_success={int(success_before_done.sum())}/{len(reset_indices)}, "
        f"valid_safe_success={int((success_before_done & ~early_done_seen).sum())}/{len(reset_indices)}, "
        f"legacy_any_reset_success={int(success_seen_legacy.sum())}/{len(reset_indices)}, "
        f"valid={int((~early_done_seen).sum())}/{len(reset_indices)}",
        flush=True,
    )
    if early_done_seen.any():
        bad = reset_indices[early_done_seen.detach().cpu().numpy()].tolist()
        print(f"[stackcube-Isaac-pairs] INVALID early-done reset IDs: {bad}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
