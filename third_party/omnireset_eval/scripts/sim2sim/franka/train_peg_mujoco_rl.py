#!/usr/bin/env python3
"""Continue PPO on the audited native MuJoCo PegInsert environment.

By default the saved Isaac checkpoint is resumed in full: actor, critic,
observation normalizers, gSDE state, and optimizer.  ``--init-mode actor_only``
retains the earlier transfer ablation with a fresh critic and optimizer.  A
checkpoint produced by this script can also be resumed with ``--resume``.

Example smoke test (use the OmniReset Python environment)::

    python train_peg_mujoco_rl.py --smoke

Example training run::

    python train_peg_mujoco_rl.py --device cuda:0 --num-envs 32 \
        --num-steps-per-env 4096 --iterations 4 --save-interval 1
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    from rsl_rl.runners import OnPolicyRunner
except ModuleNotFoundError as error:
    raise SystemExit(
        "rsl_rl is unavailable in this Python environment. Run with "
        "the release Isaac environment with vendored RSL-RL on PYTHONPATH"
    ) from error

from franka_policy import FrankaPolicy
from peg_mujoco_rl_env import (
    MujocoPegVecEnv,
    RewardConfig,
    load_reset_raw_pool,
    runtime_catalog,
)


DEFAULT_PARENT_CHECKPOINT = WORKSPACE_ROOT / (
    "co-curation/logs/rsl_rl/franka_fr3_gripper_omnireset_agent/"
    "2026-07-13_20-39-51_peginsert_stage2_b3_4gpu_from4550_b3compat/"
    "model_11200.pt"
)
DEFAULT_PARENT_RUN_DIR = DEFAULT_PARENT_CHECKPOINT.parent
DEFAULT_PARENT_ENV_CONFIG = DEFAULT_PARENT_RUN_DIR / "params/env.yaml"
DEFAULT_REWARD_MANAGER_SOURCE = WORKSPACE_ROOT / (
    "co-curation/UWLab/_isaaclab/IsaacLab/source/isaaclab/isaaclab/"
    "managers/reward_manager.py"
)
DEFAULT_TASK_REWARD_SOURCE = WORKSPACE_ROOT / (
    "co-curation/UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/"
    "manipulation/omnireset/mdp/rewards.py"
)
DEFAULT_FRANKA_REWARD_SOURCE = WORKSPACE_ROOT / (
    "co-curation/UWLab/source/uwlab_tasks/uwlab_tasks/manager_based/"
    "manipulation/omnireset/config/franka/rl_state_cfg.py"
)
DEFAULT_TRAIN_RESETS = WORKSPACE_ROOT / (
    "co-curation/Datasets/OmniReset/Resets/Peg__PegHole/"
    "resets_ObjectAnywhereEEAnywhere_upright_yaw_3cm_"
    "frontcenter_fixedhome_train3200_20260801.pt"
)
DEFAULT_RUNTIME_GLOB = str(
    WORKSPACE_ROOT
    / "sim2sim_cotrain/log/active/peg_fixedhome_a2000_20260801/"
    "isaac_shards/shard*_unfiltered.zarr"
)
DEFAULT_LOG_ROOT = WORKSPACE_ROOT / (
    "sim2sim_cotrain/log/active/peg_mujoco_rl_20260801"
)




def make_train_cfg(num_steps_per_env: int, save_interval: int) -> dict[str, Any]:
    """PPO/MLP configuration copied from the parent OmniReset run."""
    return {
        "seed": 43,
        "num_steps_per_env": int(num_steps_per_env),
        "save_interval": int(save_interval),
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "logger": "tensorboard",
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.006,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "noise_std_type": "gsde",
            "state_dependent_std": False,
            "actor_obs_normalization": True,
            "critic_obs_normalization": True,
            "actor_hidden_dims": [512, 256, 128, 64],
            "critic_hidden_dims": [512, 256, 128, 64],
            "activation": "elu",
        },
    }


def transfer_parent_actor(runner: OnPolicyRunner, checkpoint: Path) -> dict[str, Any]:
    """Copy only policy-compatible state; leave MuJoCo critic/optimizer fresh."""
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = loaded["model_state_dict"]
    destination = runner.alg.policy.state_dict()
    transfer_prefixes = ("actor.", "actor_obs_normalizer.")
    transfer_keys = [
        key
        for key in source
        if key == "log_std" or key.startswith(transfer_prefixes)
    ]
    missing = [key for key in transfer_keys if key not in destination]
    mismatched = {
        key: {
            "source": list(source[key].shape),
            "destination": list(destination[key].shape),
        }
        for key in transfer_keys
        if key in destination and source[key].shape != destination[key].shape
    }
    if missing or mismatched:
        raise ValueError(
            f"parent actor is incompatible; missing={missing}, mismatched={mismatched}"
        )
    merged = {key: value.clone() for key, value in destination.items()}
    for key in transfer_keys:
        merged[key] = source[key].to(dtype=destination[key].dtype).clone()
    runner.alg.policy.load_state_dict(merged, strict=True)
    return {
        "mode": "actor_transfer_not_resume",
        "parent_iter": int(loaded.get("iter", -1)),
        "transferred_keys": transfer_keys,
        "fresh_prefixes": ["critic.", "critic_obs_normalizer."],
        "optimizer": "fresh",
    }


@torch.inference_mode()
def verify_actor_parity(
    runner: OnPolicyRunner, env: MujocoPegVecEnv, checkpoint: Path, device: str
) -> float:
    """Prove that transfer did not alter the deterministic parent actor."""
    observations = env.get_observations()
    rsl_actions = runner.alg.policy.act_inference(observations.to(device)).cpu()
    standalone = FrankaPolicy.load_from_checkpoint(str(checkpoint), device=device)
    reference = standalone(observations["policy"].to(device)).cpu()
    error = float(torch.max(torch.abs(rsl_actions - reference)).item())
    if error > 2.0e-6:
        raise RuntimeError(
            f"transferred actor differs from parent deterministic mean: {error:.3e}"
        )
    return error


def all_model_values_finite(checkpoint: Path) -> bool:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return all(
        bool(torch.isfinite(value).all())
        for value in loaded["model_state_dict"].values()
        if isinstance(value, torch.Tensor)
    )


def choose_runtime_assignments(
    catalog: list[tuple[str, int]], num_envs: int, seed: int
) -> list[tuple[str, int]]:
    if num_envs > len(catalog):
        raise ValueError(f"requested {num_envs} runtimes from catalog of {len(catalog)}")
    rng = np.random.default_rng(seed)
    # A permutation makes worker banks nested: for a fixed seed, an N-worker
    # run is the exact prefix of every larger run.  This keeps batch-size
    # ablations from silently changing all shared worker runtimes.
    indices = rng.permutation(len(catalog))[:num_envs]
    return [catalog[int(index)] for index in indices]


def resolve_device(value: str, smoke: bool) -> str:
    if value != "auto":
        return value
    if smoke:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PARENT_CHECKPOINT)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a checkpoint produced by this MuJoCo trainer.",
    )
    parser.add_argument(
        "--init-mode",
        choices=("full_parent", "actor_only"),
        default="full_parent",
        help=(
            "How to initialize from --checkpoint when --resume is absent. "
            "full_parent restores actor, critic, normalizers, and optimizer; "
            "actor_only is the earlier ablation with a fresh critic/optimizer."
        ),
    )
    parser.add_argument("--reset-file", type=Path, default=DEFAULT_TRAIN_RESETS)
    parser.add_argument("--runtime-glob", default=DEFAULT_RUNTIME_GLOB)
    parser.add_argument(
        "--runtime-mode",
        choices=("fixed_per_worker", "resample_on_reset"),
        default="fixed_per_worker",
        help=(
            "Keep one exported physics runtime per worker, or draw a new "
            "training runtime whenever an episode resets."
        ),
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps-per-env", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--max-episode-length", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--startup-timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two workers, two rollout steps, one PPO update and save/reload check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0 or args.num_steps_per_env <= 0 or args.iterations <= 0:
        raise ValueError("num-envs, num-steps-per-env, and iterations must be positive")
    checkpoint = args.checkpoint.resolve()
    reset_file = args.reset_file.resolve()
    resume = args.resume.resolve() if args.resume else None
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not reset_file.is_file():
        raise FileNotFoundError(reset_file)
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(resume)
    reward_sources = (
        DEFAULT_PARENT_ENV_CONFIG,
        DEFAULT_REWARD_MANAGER_SOURCE,
        DEFAULT_TASK_REWARD_SOURCE,
        DEFAULT_FRANKA_REWARD_SOURCE,
    )
    for reward_source in reward_sources:
        if not reward_source.is_file():
            raise FileNotFoundError(reward_source)

    if args.smoke:
        num_envs = 2
        num_steps_per_env = 2
        iterations = 1
        save_interval = 1
    else:
        num_envs = args.num_envs
        num_steps_per_env = args.num_steps_per_env
        iterations = args.iterations
        save_interval = args.save_interval
    device = resolve_device(args.device, args.smoke)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    runtime_paths = sorted(glob.glob(args.runtime_glob))
    if not runtime_paths:
        raise FileNotFoundError(
            f"runtime glob did not match unfiltered zarr sources: {args.runtime_glob}"
        )
    catalog = runtime_catalog(runtime_paths)
    assignments = choose_runtime_assignments(catalog, num_envs, args.seed)
    reset_pool = load_reset_raw_pool(reset_file)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir is None:
        suffix = "smoke" if args.smoke else f"seed{args.seed}_n{num_envs}"
        run_dir = DEFAULT_LOG_ROOT / f"{timestamp}_{suffix}"
    else:
        run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    reward = RewardConfig()
    train_cfg = make_train_cfg(num_steps_per_env, save_interval)
    train_cfg["seed"] = int(args.seed)
    provenance = {
        "schema_version": 2,
        "definition": "native MuJoCo PPO continued from an Isaac checkpoint",
        "created_at": datetime.now().isoformat(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "parent_checkpoint": str(checkpoint),

        "reward_reference_sources": [
            {"path": str(path)}
            for path in reward_sources
        ],
        "resume_checkpoint": str(resume) if resume else None,
        "requested_initialization": (
            "mujoco_resume" if resume is not None else args.init_mode
        ),
        "reset_file": str(reset_file),

        "reset_rows": int(len(reset_pool)),
        "reset_selection": "uniform over complete fixed-home 3 cm train3200 pool",
        "runtime_sources": runtime_paths,
        "runtime_source_filtering": "all sources require success_filtering=false",
        "runtime_catalog_episodes": len(catalog),
        "runtime_mode": args.runtime_mode,
        "runtime_assignments": [
            {"path": path, "episode": episode} for path, episode in assignments
        ],
        "num_envs": num_envs,
        "num_steps_per_env": num_steps_per_env,
        "iterations": iterations,
        "max_episode_length": args.max_episode_length,
        "device": device,
        "seed": args.seed,
        "reward_profile": "saved_omnireset_parent_exact",
        "reward": vars(reward),
        "reward_formula": "term_value * saved_weight * policy_dt_s",
        "reward_policy_dt_s": 0.1,
        "reward_excluded_metrics": ["stable_release"],
        "termination_terms": ["timeout", "abnormal_robot"],
        "train_cfg": train_cfg,
        "smoke": args.smoke,
    }
    (run_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[mujoco-rl] run_dir={run_dir} envs={num_envs} "
        f"rollout={num_steps_per_env} iterations={iterations} device={device}",
        flush=True,
    )
    print(
        f"[mujoco-rl] resets={len(reset_pool)} full-train states; "
        f"runtime_catalog={len(catalog)} unfiltered episodes",
        flush=True,
    )

    env: MujocoPegVecEnv | None = None
    try:
        env = MujocoPegVecEnv(
            reset_pool,
            assignments,
            runtime_catalog_entries=catalog,
            runtime_mode=args.runtime_mode,
            seed=args.seed,
            device="cpu",
            max_episode_length=args.max_episode_length,
            reward=reward,
            startup_timeout_s=args.startup_timeout_s,
        )
        provenance["worker_metadata"] = env.worker_metadata
        (run_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "[mujoco-rl] all physics workers ready: "
            f"dt={env.worker_metadata[0]['physics_dt_s']} "
            f"decimation={env.worker_metadata[0]['decimation']} "
            f"integrator_dt={env.worker_metadata[0]['model_timestep_s']}",
            flush=True,
        )

        runner_cfg = copy.deepcopy(train_cfg)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(run_dir), device=device)
        if resume is not None:
            runner.load(str(resume), load_optimizer=True, map_location=device)
            transfer = {
                "mode": "full_mujoco_rl_resume",
                "resume": str(resume),

            }
            parity_error = None
        elif args.init_mode == "full_parent":
            runner.load(str(checkpoint), load_optimizer=True, map_location=device)
            transfer = {
                "mode": "full_parent_checkpoint_resume",
                "parent_iter": int(runner.current_learning_iteration),
                "restored": [
                    "actor",
                    "critic",
                    "actor_obs_normalizer",
                    "critic_obs_normalizer",
                    "optimizer",
                ],
            }
            parity_error = verify_actor_parity(runner, env, checkpoint, device)
            print(
                f"[mujoco-rl] full parent checkpoint restored; "
                f"actor parity max_abs={parity_error:.3e}; "
                f"parent_iter={runner.current_learning_iteration}",
                flush=True,
            )
        else:
            transfer = transfer_parent_actor(runner, checkpoint)
            parity_error = verify_actor_parity(runner, env, checkpoint, device)
            print(
                f"[mujoco-rl] parent actor parity max_abs={parity_error:.3e}; "
                "critic and optimizer are fresh",
                flush=True,
            )
        provenance["initialization"] = transfer
        provenance["parent_actor_parity_max_abs"] = parity_error
        (run_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        start = time.perf_counter()
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
        elapsed = time.perf_counter() - start
        final_checkpoint = run_dir / f"model_{runner.current_learning_iteration}.pt"
        if not final_checkpoint.is_file():
            raise FileNotFoundError(
                f"runner did not write expected final checkpoint: {final_checkpoint}"
            )
        if not all_model_values_finite(final_checkpoint):
            raise RuntimeError(f"non-finite model value in {final_checkpoint}")
        # Exercise the full checkpoint/optimizer reload path.  This is a
        # structural check, not a performance metric.
        # RSL-RL updates EmpiricalNormalization._std inside its inference-mode
        # rollout.  Reloading into that same live runner must therefore also
        # happen in inference mode; a fresh runner does not need this wrapper.
        with torch.inference_mode():
            runner.load(
                str(final_checkpoint), load_optimizer=True, map_location=device
            )
        verification = {
            "completed": True,
            "elapsed_s": elapsed,
            "final_checkpoint": str(final_checkpoint.resolve()),

            "final_iteration": int(runner.current_learning_iteration),
            "model_values_finite": True,
            "full_save_reload_passed": True,
            "reportable_success_rate": None,
            "note": (
                "Training/smoke rewards are not an evaluation. Run the frozen "
                "checkpoint through the disjoint eval384 harness."
            ),
        }
        (run_dir / "verification.json").write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"[mujoco-rl] finished in {elapsed:.1f}s; "
            f"checkpoint={final_checkpoint}; save/reload=PASS",
            flush=True,
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
