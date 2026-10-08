"""Collect Franka task_0 state demos WITH teacher distribution (mean + std) for KL distillation.

This is a copy of ``scripts/collect_state_demos.py`` with one addition: alongside the
deterministic policy mean (stored as ``data/action``, identical convention to the DDPM
pipeline) it also records the rsl_rl Gaussian actor's per-step std as ``data/action_std``.
Those two fields are what the MLP-Gaussian student needs to fit ``KL(teacher || student)``
the way the original OmniReset distillation does.

Notes
-----
* ``policy = runner.get_inference_policy()`` returns ``act_inference`` (the deterministic
  MEAN), so the action we step with — and store under ``data/action`` — IS the teacher mean.
* Standard rsl_rl PPO uses a STATE-INDEPENDENT std (an ``nn.Parameter`` of shape ``[act_dim]``),
  so we grab ``actor_critic.std`` once and broadcast it to every step. We print it at startup
  as a sanity check. If a future run uses a state-dependent std (gSDE), switch to per-step
  ``actor_critic.action_std`` — see the ``--per_step_std`` flag below.

Example:
    CUDA_VISIBLE_DEVICES=3 python scripts/franka_kl_distill/collect_demos_kl.py \\
        --checkpoint logs/.../model_4574.pt \\
        --num_envs 256 --num_demos 10000 \\
        --output datasets/franka_task0_kl_10k.zarr \\
        --headless \\
        env.scene.robot.actuators.panda_hand.stiffness=1000.0 \\
        env.scene.robot.actuators.panda_hand.damping=14.0 \\
        env.scene.robot.actuators.panda_hand.effort_limit_sim=60.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect Franka task_0 demos with teacher (mean, std).")
parser.add_argument("--task", default="OmniReset-FrankaFr3Gripper-RelCartesianOSC-State-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--num_demos", type=int, default=10000)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--output", type=str, default="./datasets/franka_task0_kl_10k.zarr")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--four_path", action="store_true",
                    help="Collect over the 4-path reset mix [0.25 each] instead of task_0 "
                         "(ObjectAnywhereEEAnywhere) only. The Stage-1 expert was trained on "
                         "this mix, so it is safe (unlike pairing it with Stage-2 gain DR).")
parser.add_argument("--max_steps", type=int, default=200_000, help="Hard cap on env steps to prevent infinite loops.")
parser.add_argument("--policy_gripper", action="store_true", default=False, help="policy-controlled gripper")
parser.add_argument(
    "--keep_dynamics_dr",
    action="store_true",
    help="Keep configured dynamics randomization. Default freezes all randomize_* events for matched A/B collection.",
)
parser.add_argument("--reset_type", default=None, help="Override single reset_type (e.g. ObjectAnywhereEEAnywhere_upright_yaw_3cm). Takes precedence over --four_path.")
parser.add_argument(
    "--dataset_dir",
    default=None,
    help="Optional OmniReset dataset root containing Resets/<pair>/resets_<reset_type>.pt.",
)
parser.add_argument("--expected_obs_dim", type=int, default=200)
parser.add_argument("--expected_act_dim", type=int, default=7)
parser.add_argument("--expected_receptive_usd_basename", default="peg_hole_big.usd")
parser.add_argument(
    "--expected_episode_steps",
    type=int,
    default=160,
    help="Hard alignment gate; accepted successful episodes must have this many policy rows.",
)
# Load OmniReset's rsl_rl helper without assuming that this release repository is
# nested inside an OmniReset checkout.
OMNIRESET_ROOT = os.environ.get("OMNIRESET_ROOT")
if not OMNIRESET_ROOT:
    raise RuntimeError("set OMNIRESET_ROOT to the OmniReset checkout")
sys.path.insert(
    0,
    os.path.join(
        os.path.abspath(OMNIRESET_ROOT),
        "UWLab",
        "scripts",
        "reinforcement_learning",
        "rsl_rl",
    ),
)
import cli_args as _rsl_cli_args  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

ORIGINAL_ARGV = list(sys.argv)
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- everything below runs after Isaac Sim is up ---
import gymnasium as gym
import numpy as np
import torch
import zarr

import isaaclab_tasks  # noqa: F401
import uwlab_tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from uwlab_tasks.utils.hydra import hydra_task_config  # noqa: E402


from _gain_check import check_franka_gains

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    check_franka_gains(env_cfg, args_cli.task)
    agent_cfg = _rsl_cli_args.sanitize_rsl_rl_cfg(agent_cfg)

    disabled_events = []
    if not args_cli.keep_dynamics_dr:
        for event_name in list(vars(env_cfg.events)):
            if event_name.startswith("randomize_") and getattr(env_cfg.events, event_name, None) is not None:
                setattr(env_cfg.events, event_name, None)
                disabled_events.append(event_name)
        print(f"[collect] nominal dynamics: disabled events {disabled_events}")

    # Reset distribution: explicit override > 4-path mix > default single
    if args_cli.reset_type is not None:
        env_cfg.events.reset_from_reset_states.params["reset_types"] = [args_cli.reset_type]
        env_cfg.events.reset_from_reset_states.params["probs"] = [1.0]
        print(f"[INFO] reset distribution: {args_cli.reset_type}")
    elif args_cli.four_path:
        env_cfg.events.reset_from_reset_states.params["reset_types"] = [
            "ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped",
            "ObjectAnywhereEEGrasped", "ObjectPartiallyAssembledEEGrasped",
        ]
        env_cfg.events.reset_from_reset_states.params["probs"] = [0.25, 0.25, 0.25, 0.25]
        print("[INFO] reset distribution: 4-path mix [0.25 each]")
    else:
        env_cfg.events.reset_from_reset_states.params["reset_types"] = ["ObjectAnywhereEEAnywhere"]
        env_cfg.events.reset_from_reset_states.params["probs"] = [1.0]
        print("[INFO] reset distribution: task_0 only (ObjectAnywhereEEAnywhere)")

    if args_cli.dataset_dir is not None:
        env_cfg.events.reset_from_reset_states.params["dataset_dir"] = os.path.abspath(
            args_cli.dataset_dir
        )
        print(f"[INFO] reset dataset root: {env_cfg.events.reset_from_reset_states.params['dataset_dir']}")

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.policy_gripper:
        from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
        env_cfg.actions.gripper = BinaryJointPositionActionCfg(
            asset_name="robot", joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0})
        print("[collect] gripper -> POLICY-CONTROLLED")

    actual_receptive_usd = str(env_cfg.scene.receptive_object.spawn.usd_path)
    if os.path.basename(actual_receptive_usd) != args_cli.expected_receptive_usd_basename:
        raise RuntimeError(
            "receptive-object contract mismatch: "
            f"{actual_receptive_usd!r} does not end in "
            f"{args_cli.expected_receptive_usd_basename!r}"
        )

    effective_contract = {
        "task": args_cli.task,
        "reset_type": args_cli.reset_type,
        "reset_dataset_dir": env_cfg.events.reset_from_reset_states.params.get("dataset_dir"),
        "seed": int(args_cli.seed),
        "teacher_checkpoint": os.path.abspath(args_cli.checkpoint),
        "teacher_checkpoint_sha256": _sha256(args_cli.checkpoint),
        "expected_obs_dim": int(args_cli.expected_obs_dim),
        "expected_act_dim": int(args_cli.expected_act_dim),
        "expected_episode_steps": int(args_cli.expected_episode_steps),
        "receptive_object_usd": str(env_cfg.scene.receptive_object.spawn.usd_path),
        "insertive_object_usd": str(env_cfg.scene.insertive_object.spawn.usd_path),
        "episode_length_s": float(env_cfg.episode_length_s),
        "decimation": int(env_cfg.decimation),
        "sim_dt": float(env_cfg.sim.dt),
        "argv": ORIGINAL_ARGV,
        "dynamics_randomization": "kept" if args_cli.keep_dynamics_dr else "disabled",
        "disabled_events": disabled_events,
    }
    print("[ALIGNMENT CONTRACT] " + json.dumps(effective_contract, sort_keys=True))

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    actor = runner.alg.policy  # rsl_rl ActorCritic (PPO stores it as `.policy`, not `.actor_critic`)

    device = env.unwrapped.device

    # Teacher std: read per-step from the populated distribution
    # (``actor.action_std`` == ``distribution.stddev``). Correct for ALL noise
    # types — scalar / log / gsde — whereas ``actor.std`` only exists for "scalar".
    noise_type = getattr(actor, "noise_std_type", "unknown")
    print(f"[INFO] teacher noise_std_type = {noise_type} (recording per-step action_std)")

    num_envs = env.unwrapped.num_envs
    print(f"[INFO] num_envs={num_envs}, target demos={args_cli.num_demos}")

    buf_obs = [[] for _ in range(num_envs)]
    buf_act = [[] for _ in range(num_envs)]
    buf_std = [[] for _ in range(num_envs)]
    buf_succ_seen = [False] * num_envs
    completed_demos = []  # list of {"state","action","action_std"}
    rejected_episode_lengths = {}

    def progress_term():
        return env.unwrapped.reward_manager.get_term_cfg("progress_context").func

    def _record_obs_tensor(o):
        if isinstance(o, tuple):
            o = o[0]
        if hasattr(o, "keys"):
            try:
                o = o["policy"]
            except (KeyError, TypeError):
                o = next(iter(o.values()))
        if hasattr(o, "keys"):
            o = torch.cat([v.reshape(v.shape[0], -1) for v in o.values()], dim=-1)
        return o

    obs = env.get_observations()

    step_count = 0
    while len(completed_demos) < args_cli.num_demos and step_count < args_cli.max_steps:
        with torch.inference_mode():
            actions = policy(obs)  # deterministic mean — this is the teacher mean
            # Populate the distribution at this state to read its stddev. act()
            # samples internally; that sample is discarded — we step with `actions`
            # (the mean). Works for scalar/log/gsde noise alike.
            actor.act(obs)
            std_step = actor.action_std.detach()  # (N, act_dim)

        obs_flat = _record_obs_tensor(obs)
        obs_np = obs_flat.detach().cpu().numpy()
        act_np = actions.detach().cpu().numpy()
        std_np = std_step.detach().cpu().numpy()
        if obs_np.shape != (num_envs, args_cli.expected_obs_dim):
            raise RuntimeError(
                f"observation contract mismatch: {obs_np.shape} != "
                f"({num_envs}, {args_cli.expected_obs_dim})"
            )
        if act_np.shape != (num_envs, args_cli.expected_act_dim):
            raise RuntimeError(
                f"action contract mismatch: {act_np.shape} != "
                f"({num_envs}, {args_cli.expected_act_dim})"
            )
        for i in range(num_envs):
            buf_obs[i].append(obs_np[i])
            buf_act[i].append(act_np[i])
            buf_std[i].append(std_np[i])

        obs, _, dones, _ = env.step(actions)

        success = progress_term().success  # (N,) bool
        for i in range(num_envs):
            if bool(success[i].item()):
                buf_succ_seen[i] = True

        for i in range(num_envs):
            if bool(dones[i].item()):
                T = len(buf_obs[i])
                if (
                    buf_succ_seen[i]
                    and T > 0
                    and len(completed_demos) < args_cli.num_demos
                ):
                    if T != args_cli.expected_episode_steps:
                        rejected_episode_lengths[T] = rejected_episode_lengths.get(T, 0) + 1
                        print(
                            f"[WARN] rejected successful episode in env {i}: "
                            f"T={T}, expected={args_cli.expected_episode_steps}; "
                            f"length_rejections={sum(rejected_episode_lengths.values())}"
                        )
                    else:
                        completed_demos.append({
                            "state": np.stack(buf_obs[i], axis=0).astype(np.float32),
                            "action": np.stack(buf_act[i], axis=0).astype(np.float32),
                            "action_std": np.stack(buf_std[i], axis=0).astype(np.float32),
                        })
                        if len(completed_demos) % 25 == 0 or len(completed_demos) == 1:
                            print(f"[INFO] collected {len(completed_demos)}/{args_cli.num_demos} demos "
                                  f"(latest T={T}, step={step_count})")
                buf_obs[i] = []
                buf_act[i] = []
                buf_std[i] = []
                buf_succ_seen[i] = False

        step_count += 1

    print(f"[INFO] Done. Collected {len(completed_demos)} successful demos over {step_count} env steps.")
    print(f"[INFO] Rejected episode lengths: {rejected_episode_lengths}")
    if len(completed_demos) != args_cli.num_demos:
        raise RuntimeError(
            f"collection incomplete: {len(completed_demos)} != {args_cli.num_demos}"
        )
    save_zarr(completed_demos, args_cli.output, effective_contract)
    env.close()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_zarr(demos, output_path, effective_contract):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    all_state = np.concatenate([d["state"] for d in demos], axis=0)
    all_action = np.concatenate([d["action"] for d in demos], axis=0)        # teacher MEAN
    all_action_std = np.concatenate([d["action_std"] for d in demos], axis=0)  # teacher STD
    ends = np.cumsum([len(d["state"]) for d in demos]).astype(np.int64)

    print(f"[INFO] Saving {len(demos)} demos -> {output_path}")
    print(f"[INFO]   state:      shape={all_state.shape}  dtype={all_state.dtype}")
    print(f"[INFO]   action:     shape={all_action.shape} (teacher mean)")
    print(f"[INFO]   action_std: shape={all_action_std.shape} (teacher std)")
    print(f"[INFO]   total steps={all_state.shape[0]}, avg ep len={all_state.shape[0]/len(demos):.1f}")

    root = zarr.open(output_path, mode="w")
    data = root.create_group("data")
    sc = (min(1024, all_state.shape[0]), all_state.shape[1])
    ac = (min(1024, all_action.shape[0]), all_action.shape[1])
    data.create_dataset("state", data=all_state, chunks=sc)
    data.create_dataset("action", data=all_action, chunks=ac)
    data.create_dataset("action_std", data=all_action_std, chunks=ac)
    meta = root.create_group("meta")
    meta.create_dataset("episode_ends", data=ends)
    root.attrs.update(
        {
            "data_role": "IsaacSim B pool collected under the explicit alignment contract",
            "simulator": "IsaacSim",
            "collector": os.path.abspath(__file__),
            "num_demos": int(len(demos)),
            "num_steps": int(all_state.shape[0]),
            "contract": effective_contract,
        }
    )
    print("[INFO] Saved.")


if __name__ == "__main__":
    main()
    simulation_app.close()
