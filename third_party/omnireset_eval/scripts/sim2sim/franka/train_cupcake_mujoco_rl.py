#!/usr/bin/env python3
"""Continue the released CupCake PPO policy in the audited MuJoCo task."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from cupcake_mujoco_rl_env import (
    MujocoCupCakeVecEnv,
    RewardConfig,
    audit_observation_parity,
    configure_physics_profile,
    load_zarr_reset_raw_pool,
    runtime_catalog,
)
from train_peg_mujoco_rl import (
    OnPolicyRunner,
    all_model_values_finite,
    choose_runtime_assignments,
    file_sha256,
    make_train_cfg,
    resolve_device,
    transfer_parent_actor,
    verify_actor_parity,
)


HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parents[3]
DEFAULT_PARENT_CHECKPOINT = WORKSPACE_ROOT / (
    "co-curation-cupcake-mlp/checkpoints/cupcake_plate_rl/stage2/model_20400.pt"
)
DEFAULT_PARENT_ENV_CONFIG = WORKSPACE_ROOT / (
    "co-curation-cupcake-mlp/checkpoints/cupcake_plate_rl/provenance/"
    "stage2_adr92_env.yaml"
)
DEFAULT_REWARD_MANAGER_SOURCE = WORKSPACE_ROOT / (
    "co-curation/UWLab/_isaaclab/IsaacLab/source/isaaclab/"
    "isaaclab/managers/reward_manager.py"
)
DEFAULT_TASK_REWARD_SOURCE = WORKSPACE_ROOT / (
    "co-curation-cupcake-mlp/UWLab/source/uwlab_tasks/uwlab_tasks/"
    "manager_based/manipulation/omnireset/mdp/rewards.py"
)
DEFAULT_FRANKA_REWARD_SOURCE = WORKSPACE_ROOT / (
    "co-curation-cupcake-mlp/UWLab/source/uwlab_tasks/uwlab_tasks/"
    "manager_based/manipulation/omnireset/config/franka/rl_state_cfg.py"
)
DEFAULT_SOURCE_ZARR = WORKSPACE_ROOT / (
    "sim2sim_cotrain/datasets/cupcake_sim2sim_t0_20260815/"
    "isaac_fit_b3center.zarr"
)
DEFAULT_SPLIT_MANIFEST = WORKSPACE_ROOT / (
    "sim2sim_cotrain/datasets/cupcake_sim2sim_t0_20260815/reset_split.json"
)
DEFAULT_PARITY_ZARR = WORKSPACE_ROOT / (
    "sim2sim_cotrain/datasets/cupcake_sim2sim_t0_20260815/"
    "isaac_critic_parity_smoke1x1.zarr"
)
DEFAULT_LOG_ROOT = WORKSPACE_ROOT / (
    "sim2sim_cotrain/log/active/cupcake_mujoco_rl_20260815"
)


def audit_fit_source(
    source: Path, split_manifest: Path, checkpoint: Path
) -> dict[str, Any]:
    store = zarr.open(str(source), mode="r")
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    reset_indices = np.asarray(store["meta/reset_indices"], dtype=np.int64)
    fit_indices = np.asarray(split["panels"]["fit"]["indices"], dtype=np.int64)
    if store.attrs.get("panel") != "fit":
        raise ValueError(f"source Zarr is not the fit panel: {source}")
    if not np.array_equal(reset_indices, fit_indices):
        raise ValueError("source reset indices do not exactly match split fit panel")
    split_hash = file_sha256(split_manifest)
    if store.attrs.get("split_manifest_sha256") != split_hash:
        raise ValueError("source split-manifest hash does not match frozen manifest")
    checkpoint_hash = file_sha256(checkpoint)
    if store.attrs.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("source checkpoint hash does not match training parent")
    success_seen = np.asarray(store["meta/success_seen"], dtype=np.bool_)
    return {
        "panel": "fit",
        "episodes": int(len(reset_indices)),
        "reset_indices": reset_indices.tolist(),
        "source_success_seen": int(success_seen.sum()),
        "source_failure_seen": int((~success_seen).sum()),
        "split_manifest": str(split_manifest),
        "split_manifest_sha256": split_hash,
        "checkpoint_matches": True,
        "timing": {
            "physics_dt_s": float(store.attrs["physics_dt_s"]),
            "decimation": int(store.attrs["decimation"]),
            "policy_dt_s": float(store.attrs["policy_dt_s"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PARENT_CHECKPOINT)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-mode", choices=("full_parent", "actor_only"), default="full_parent"
    )
    parser.add_argument("--source-zarr", type=Path, default=DEFAULT_SOURCE_ZARR)
    parser.add_argument(
        "--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST
    )
    parser.add_argument("--parity-zarr", type=Path, default=DEFAULT_PARITY_ZARR)
    parser.add_argument(
        "--runtime-mode",
        choices=("fixed_per_worker", "resample_on_reset"),
        default="fixed_per_worker",
    )
    parser.add_argument(
        "--reset-mode",
        choices=(
            "home",
            "source_trajectory",
            "successful_source_trajectory",
            "successful_episode_trajectory",
            "mixed",
            "mixed_successful",
            "mixed_successful_episode",
        ),
        default="home",
    )
    parser.add_argument(
        "--home-reset-fraction",
        type=float,
        default=0.5,
        help="Home-reset share for mixed reset modes.",
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--num-steps-per-env", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--max-episode-length", type=int, default=160)
    parser.add_argument("--physics-substeps", type=int, default=16)
    parser.add_argument(
        "--friction-combine",
        choices=("legacy_b3", "physx_average"),
        default="physx_average",
    )
    parser.add_argument(
        "--cupcake-collision",
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
        "--hand-collision",
        choices=("menagerie", "source_usd", "disabled"),
        default="source_usd",
    )
    parser.add_argument(
        "--cupcake-plate-collision",
        choices=("convex_hull", "radial16", "base_cylinder"),
        default="base_cylinder",
    )
    parser.add_argument(
        "--finger-collision",
        choices=("mimic", "menagerie", "menagerie_mesh_only"),
        default="mimic",
    )
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--exploration-std-scale",
        type=float,
        default=1.0,
        help="Multiply the restored gSDE standard deviation before learning.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--startup-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two workers, two rollout steps, and one PPO update on CPU.",
    )
    return parser.parse_args()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0 or args.num_steps_per_env <= 0 or args.iterations <= 0:
        raise ValueError("num-envs, num-steps-per-env, and iterations must be positive")
    if args.exploration_std_scale <= 0.0:
        raise ValueError("exploration-std-scale must be positive")
    if args.reset_mode.startswith("mixed") and not 0.0 < args.home_reset_fraction < 1.0:
        raise ValueError("home-reset-fraction must be in (0, 1) for mixed modes")
    checkpoint = args.checkpoint.resolve()
    source = args.source_zarr.resolve()
    split_manifest = args.split_manifest.resolve()
    parity_source = args.parity_zarr.resolve()
    resume = args.resume.resolve() if args.resume else None
    required_files = (
        checkpoint,
        source,
        split_manifest,
        parity_source,
        DEFAULT_PARENT_ENV_CONFIG,
        DEFAULT_REWARD_MANAGER_SOURCE,
        DEFAULT_TASK_REWARD_SOURCE,
        DEFAULT_FRANKA_REWARD_SOURCE,
    )
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)
    if resume is not None and not resume.is_file():
        raise FileNotFoundError(resume)

    configure_physics_profile(
        physics_substeps=args.physics_substeps,
        friction_combine=args.friction_combine,
        cupcake_collision=args.cupcake_collision,
        cupcake_plate_collision=args.cupcake_plate_collision,
        hand_collision=args.hand_collision,
        finger_collision=args.finger_collision,
    )
    source_audit = audit_fit_source(source, split_manifest, checkpoint)
    observation_parity = audit_observation_parity(parity_source)
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

    catalog = runtime_catalog([source])
    assignments = choose_runtime_assignments(catalog, num_envs, args.seed)
    reset_pool = load_zarr_reset_raw_pool(
        source, mode=args.reset_mode, home_fraction=args.home_reset_fraction
    )
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
    reward_sources = (
        DEFAULT_PARENT_ENV_CONFIG,
        DEFAULT_REWARD_MANAGER_SOURCE,
        DEFAULT_TASK_REWARD_SOURCE,
        DEFAULT_FRANKA_REWARD_SOURCE,
    )
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "definition": "native MuJoCo CupCake PPO continued from released Isaac checkpoint",
        "created_at": datetime.now().isoformat(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": file_sha256(checkpoint),
        "reward_reference_sources": [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in reward_sources
        ],
        "resume_checkpoint": str(resume) if resume else None,
        "requested_initialization": (
            "mujoco_resume" if resume is not None else args.init_mode
        ),
        "source_zarr": str(source),
        "source_zattrs_sha256": file_sha256(source / ".zattrs"),
        "source_audit": source_audit,
        "observation_parity": {
            **observation_parity,
            "source": str(parity_source),
            "source_zattrs_sha256": file_sha256(parity_source / ".zattrs"),
        },
        "reset_rows": int(len(reset_pool)),
        "reset_mode": args.reset_mode,
        "home_reset_fraction": (
            args.home_reset_fraction if args.reset_mode.startswith("mixed") else None
        ),
        "reset_selection": (
            "uniform over a pool derived only from frozen fit64; mixed-mode home "
            "share is set by home_reset_fraction; mixed_successful applies the "
            "official raw pose threshold, while mixed_successful_episode keeps "
            "all states from source-successful fit episodes"
        ),
        "runtime_sources": [str(source)],
        "runtime_source_filtering": "all fit episodes retained regardless of outcome",
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
        "exploration_std_scale": args.exploration_std_scale,
        "physics_mapping": {
            "physics_substeps": args.physics_substeps,
            "friction_combine": args.friction_combine,
            "cupcake_collision": args.cupcake_collision,
            "cupcake_plate_collision": args.cupcake_plate_collision,
            "hand_collision": args.hand_collision,
            "finger_collision": args.finger_collision,
        },
        "reward_profile": "released_cupcake_parent_exact",
        "reward": vars(reward),
        "reward_formula": "term_value * saved_weight * policy_dt_s",
        "reward_policy_dt_s": 0.1,
        "termination_terms": ["timeout", "abnormal_robot"],
        "success_metric": "pose success seen; stable pose for 5 steps reported separately",
        "train_cfg": train_cfg,
        "smoke": args.smoke,
    }
    write_json(run_dir / "provenance.json", provenance)
    print(
        f"[cupcake-mujoco-rl] run_dir={run_dir} envs={num_envs} "
        f"rollout={num_steps_per_env} iterations={iterations} device={device}",
        flush=True,
    )
    print(
        f"[cupcake-mujoco-rl] fit_resets={len(reset_pool)} "
        f"runtime_catalog={len(catalog)}",
        flush=True,
    )

    env: MujocoCupCakeVecEnv | None = None
    try:
        env = MujocoCupCakeVecEnv(
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
        write_json(run_dir / "provenance.json", provenance)
        print(
            "[cupcake-mujoco-rl] all physics workers ready: "
            f"dt={env.worker_metadata[0]['physics_dt_s']} "
            f"decimation={env.worker_metadata[0]['decimation']} "
            f"integrator_dt={env.worker_metadata[0]['model_timestep_s']}",
            flush=True,
        )

        runner_cfg = copy.deepcopy(train_cfg)
        runner = OnPolicyRunner(env, runner_cfg, log_dir=str(run_dir), device=device)
        if resume is not None:
            runner.load(str(resume), load_optimizer=True, map_location=device)
            initialization = {
                "mode": "full_mujoco_rl_resume",
                "resume": str(resume),
                "resume_sha256": file_sha256(resume),
            }
            parity_error = None
        elif args.init_mode == "full_parent":
            runner.load(str(checkpoint), load_optimizer=True, map_location=device)
            initialization = {
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
        else:
            initialization = transfer_parent_actor(runner, checkpoint)
            parity_error = verify_actor_parity(runner, env, checkpoint, device)
        log_std_offset = float(np.log(args.exploration_std_scale))
        if log_std_offset != 0.0:
            with torch.no_grad():
                runner.alg.policy.log_std.add_(log_std_offset)
        initialization["exploration_std_scale"] = args.exploration_std_scale
        initialization["log_std_offset"] = log_std_offset
        provenance["initialization"] = initialization
        provenance["parent_actor_parity_max_abs"] = parity_error
        write_json(run_dir / "provenance.json", provenance)
        if parity_error is not None:
            print(
                f"[cupcake-mujoco-rl] parent actor parity max_abs={parity_error:.3e}; "
                f"parent_iter={runner.current_learning_iteration}",
                flush=True,
            )
        if log_std_offset != 0.0:
            print(
                "[cupcake-mujoco-rl] scaled restored gSDE std by "
                f"{args.exploration_std_scale:g}",
                flush=True,
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
        with torch.inference_mode():
            runner.load(str(final_checkpoint), load_optimizer=True, map_location=device)
        verification = {
            "completed": True,
            "elapsed_s": elapsed,
            "final_checkpoint": str(final_checkpoint.resolve()),
            "final_checkpoint_sha256": file_sha256(final_checkpoint),
            "final_iteration": int(runner.current_learning_iteration),
            "model_values_finite": True,
            "full_save_reload_passed": True,
            "reportable_success_rate": None,
            "note": "Training rewards are not held-out evaluation metrics.",
        }
        write_json(run_dir / "verification.json", verification)
        print(
            f"[cupcake-mujoco-rl] finished in {elapsed:.1f}s; "
            f"checkpoint={final_checkpoint}; save/reload=PASS",
            flush=True,
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
