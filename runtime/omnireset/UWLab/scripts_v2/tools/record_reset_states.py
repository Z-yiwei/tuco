# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record reset states using IsaacLab framework."""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import copy
import os
import torch
from tqdm import tqdm
from typing import cast

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record reset states for object pairs.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--task", type=str, default="OmniReset-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0", help="Name of the task."
)
parser.add_argument(
    "--dataset_dir", type=str, default="./Datasets/OmniReset/", help="Root Datasets/OmniReset/ directory."
)
parser.add_argument(
    "--reset_type",
    type=str,
    default=None,
    help="Reset type name (e.g. ObjectAnywhereEEAnywhere). Auto-inferred from --task if omitted.",
)
parser.add_argument(
    "--num_reset_states", type=int, default=10000, help="Number of reset states to record. Set to 0 for infinite."
)
parser.add_argument(
    "--keep_shards",
    action="store_true",
    default=False,
    help="Keep per-rank shard files after merge when running with torchrun multi-GPU sharding.",
)
parser.add_argument(
    "--shard_wait_timeout",
    type=int,
    default=7200,
    help="Timeout in seconds for rank 0 waiting on all shard files before merge.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()


def _get_dist_info() -> tuple[int, int, int]:
    """Read distributed rank metadata from environment variables."""
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


DIST_RANK, DIST_WORLD_SIZE, DIST_LOCAL_RANK = _get_dist_info()
if DIST_WORLD_SIZE > 1:
    if args_cli.num_reset_states == 0:
        raise ValueError(
            "--num_reset_states 0 (infinite sampling) is not supported in multi-GPU sharded mode."
        )
    if args_cli.device is not None and "cpu" in str(args_cli.device).lower():
        raise ValueError("Multi-GPU sharded mode requires CUDA device, not CPU.")
    # Route each process to its local GPU when launched via torchrun.
    args_cli.device = f"cuda:{DIST_LOCAL_RANK}"

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import time

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import DatasetExportMode

from uwlab.utils.datasets.torch_dataset_file_handler import TorchDatasetFileHandler

import uwlab_tasks  # noqa: F401
import uwlab_tasks.manager_based.manipulation.omnireset.mdp as task_mdp
from uwlab_tasks.utils.hydra import hydra_task_compose

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _compute_local_target(total: int, rank: int, world_size: int) -> int:
    """Split a global sample target across ranks as evenly as possible."""
    base = total // world_size
    remainder = total % world_size
    return base + (1 if rank < remainder else 0)


def _merge_nested_data_inplace(dst: dict, src: dict, key_path: str = "") -> None:
    """Merge nested torch handler payloads (dicts with list leaves) in place."""
    for key, value in src.items():
        current_path = f"{key_path}.{key}" if key_path else key
        if key not in dst:
            dst[key] = copy.deepcopy(value)
            continue

        if isinstance(value, dict):
            if not isinstance(dst[key], dict):
                raise TypeError(f"Type mismatch at '{current_path}': expected dict.")
            _merge_nested_data_inplace(dst[key], value, current_path)
        elif isinstance(value, list):
            if not isinstance(dst[key], list):
                raise TypeError(f"Type mismatch at '{current_path}': expected list.")
            dst[key].extend(value)
        elif torch.is_tensor(value):
            if not torch.is_tensor(dst[key]):
                raise TypeError(f"Type mismatch at '{current_path}': expected tensor.")
            dst[key] = torch.cat([dst[key], value], dim=0)
        else:
            if dst[key] != value:
                raise ValueError(
                    f"Conflicting scalar values at '{current_path}': {dst[key]} vs {value}."
                )


def _count_samples(data):
    """Infer sample count from first list/tensor leaf in nested payload."""
    if isinstance(data, dict):
        for value in data.values():
            count = _count_samples(value)
            if count is not None:
                return count
        return 0
    if isinstance(data, list):
        return len(data)
    if torch.is_tensor(data):
        return int(data.shape[0]) if data.ndim > 0 else 1
    return None


def _slice_nested_samples(data, target_count: int):
    """Slice nested payload to target_count along sample dimension."""
    if isinstance(data, dict):
        return {key: _slice_nested_samples(value, target_count) for key, value in data.items()}
    if isinstance(data, list):
        return data[:target_count]
    if torch.is_tensor(data):
        if data.ndim == 0:
            return data
        return data[:target_count]
    return data


def _enforce_exact_sample_count(file_path: str, target_count: int, label: str) -> int:
    """Ensure dataset has exactly target_count samples by truncating overflow."""
    if target_count <= 0:
        return 0
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[{label}] dataset file not found: {file_path}")

    data = torch.load(file_path, map_location="cpu")
    current_count = _count_samples(data)
    if current_count is None:
        raise ValueError(f"[{label}] failed to infer sample count from dataset: {file_path}")

    if current_count < target_count:
        raise RuntimeError(f"[{label}] samples {current_count} < target {target_count} in {file_path}")

    if current_count > target_count:
        data = _slice_nested_samples(data, target_count)
        torch.save(data, file_path)
        print(f"[{label}] Truncated dataset from {current_count} to {target_count}: {file_path}")
        current_count = target_count
    else:
        print(f"[{label}] Dataset already matches target {target_count}: {file_path}")

    return current_count


def _wait_for_shards(paths: list[str], timeout_s: int) -> None:
    """Wait until all expected shard files exist on disk."""
    start_t = time.time()
    while True:
        missing = [path for path in paths if not os.path.exists(path)]
        if not missing:
            return
        if time.time() - start_t > timeout_s:
            raise TimeoutError(f"Timed out waiting for shard files: {missing}")
        time.sleep(2.0)


def _merge_shards(paths: list[str]) -> dict:
    """Load and merge shard files into one nested payload."""
    merged = {}
    for path in paths:
        shard_data = torch.load(path, map_location="cpu")
        if shard_data:
            _merge_nested_data_inplace(merged, shard_data)
    return merged


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg, agent_cfg) -> None:
    """Main function to record reset states."""
    # create directory if it does not exist
    if not os.path.exists(args_cli.dataset_dir):
        os.makedirs(args_cli.dataset_dir, exist_ok=True)

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # make sure environment is non-deterministic for diverse pose discovery
    env_cfg.seed = None

    # Derive pair directory and reset type for output path
    insertive_usd_path = env_cfg.scene.insertive_object.spawn.usd_path
    receptive_usd_path = env_cfg.scene.receptive_object.spawn.usd_path
    pair = task_mdp.utils.compute_pair_dir(insertive_usd_path, receptive_usd_path)

    # Auto-infer reset_type from task name if not provided
    reset_type = args_cli.reset_type
    if reset_type is None:
        for candidate in [
            "ObjectAnywhereEEAnywhere",
            "ObjectRestingEEGrasped",
            "ObjectAnywhereEEGrasped",
            "ObjectPartiallyAssembledEEGrasped",
        ]:
            if candidate in args_cli.task:
                reset_type = candidate
                break
        if reset_type is None:
            raise ValueError(f"Could not infer reset_type from task '{args_cli.task}'. Pass --reset_type explicitly.")

    print(f"Recording reset states for: {pair} / {reset_type}")
    print(f"Insertive: {insertive_usd_path}")
    print(f"Receptive: {receptive_usd_path}")

    # Setup recording configuration
    output_dir = os.path.join(args_cli.dataset_dir, "Resets", pair)
    os.makedirs(output_dir, exist_ok=True)
    output_file_name = f"resets_{reset_type}.pt"
    output_file_path = os.path.join(output_dir, output_file_name)

    rank, world_size, local_rank = _get_dist_info()
    use_sharded_multi_gpu = world_size > 1
    local_target = (
        _compute_local_target(args_cli.num_reset_states, rank, world_size)
        if use_sharded_multi_gpu
        else args_cli.num_reset_states
    )

    shard_dir = None
    shard_file_path = None
    if use_sharded_multi_gpu:
        shard_dir = os.path.join(output_dir, "_shards", reset_type)
        os.makedirs(shard_dir, exist_ok=True)
        shard_filename = f"resets_{reset_type}.rank{rank:03d}.pt"
        shard_file_path = os.path.join(shard_dir, shard_filename)
        print(
            f"[Shard mode] rank={rank}/{world_size} local_rank={local_rank} "
            f"device={env_cfg.sim.device} local_target={local_target}"
        )
        print(f"[Shard mode] Writing shard file: {shard_file_path}")
    else:
        print(f"[Single GPU mode] target={local_target} device={env_cfg.sim.device}")

    env_cfg.recorders = task_mdp.StableStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = shard_dir if use_sharded_multi_gpu else output_dir
    env_cfg.recorders.dataset_filename = os.path.basename(shard_file_path) if use_sharded_multi_gpu else output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = TorchDatasetFileHandler

    # Ensure all reset events read datasets from the requested root directory.
    # Several task cfgs default to cloud paths; for embodiment-specific datasets
    # (e.g., Franka grasps/resets) we must route to args_cli.dataset_dir.
    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is not None:
        for term_name in dir(events_cfg):
            if term_name.startswith("_"):
                continue
            term_cfg = getattr(events_cfg, term_name)
            params = getattr(term_cfg, "params", None)
            if isinstance(params, dict) and "dataset_dir" in params:
                params["dataset_dir"] = args_cli.dataset_dir

    num_reset_conditions_evaluated = 0
    final_successful_reset_conditions = 0
    start_time = time.time()

    if local_target > 0:
        # create environment
        env = cast(ManagerBasedRLEnv, gym.make(args_cli.task, cfg=env_cfg)).unwrapped
        env.reset()

        # Run reset state sampling
        current_successful_reset_conditions = 0
        actions = torch.zeros(env.action_space.shape, device=env.device, dtype=torch.float32)
        if "EEGrasped" in args_cli.task:
            actions[:, -1] = -1.0
        else:
            actions[:, -1] = (
                torch.randint(0, 2, (env.num_envs,), device=env.device, dtype=torch.float32) * 2 - 1
            )  # Randomly choose between -1 and 1

        # Create progress bar
        pbar = tqdm(total=local_target, desc="Successful reset states", unit="reset states")

        while current_successful_reset_conditions < local_target:
            # Step environment (this evaluates reset conditions in parallel across environments)
            _, _, terminated, truncated, _ = env.step(actions)
            dones = terminated | truncated
            done_idx = torch.where(dones)[0]

            # Reset actions for environments that are done
            if done_idx.numel() > 0 and "EEGrasped" not in args_cli.task:
                actions[done_idx, -1] = (
                    torch.randint(0, 2, (done_idx.numel(),), device=env.device, dtype=torch.float32) * 2 - 1
                )

            # Update progress based on successful reset conditions
            new_successful_count = env.recorder_manager.exported_successful_episode_count
            if new_successful_count > current_successful_reset_conditions:
                increment = new_successful_count - current_successful_reset_conditions
                current_successful_reset_conditions = new_successful_count
                pbar.update(increment)

            # Count total reset conditions evaluated (sum across all environments)
            num_reset_conditions_evaluated += dones.sum().item()

            if env.sim.is_stopped():
                break

        pbar.close()

        # Get final statistics
        final_successful_reset_conditions = env.recorder_manager.exported_successful_episode_count
        env.close()
    else:
        print("[Shard mode] local_target is 0 for this rank. Skipping simulation loop.")

    print("Reset state recording complete!")
    print(f"Total reset conditions evaluated: {num_reset_conditions_evaluated}")
    print(f"Successful reset conditions: {final_successful_reset_conditions}")
    if num_reset_conditions_evaluated > 0:
        print(f"Success rate: {final_successful_reset_conditions / num_reset_conditions_evaluated:.2%}")
        print(f"Time taken: {(time.time() - start_time) / 60:.2f} minutes")

    # Ensure each rank leaves a shard file for deterministic merge.
    if use_sharded_multi_gpu and shard_file_path is not None and not os.path.exists(shard_file_path):
        print(f"[WARN] No shard file written for rank {rank}; creating empty shard at {shard_file_path}.")
        torch.save({}, shard_file_path)

    # Enforce exact per-rank shard target before merge to avoid overshoot.
    if use_sharded_multi_gpu and shard_file_path is not None and local_target > 0:
        _enforce_exact_sample_count(
            shard_file_path,
            local_target,
            label=f"rank{rank}_shard",
        )

    # Enforce exact sample count in single-GPU mode as well.
    if not use_sharded_multi_gpu and args_cli.num_reset_states > 0:
        exact_count = _enforce_exact_sample_count(
            output_file_path,
            args_cli.num_reset_states,
            label="single_gpu",
        )
        print(f"[single_gpu] Final exact sample count: {exact_count}")

    # Merge shard files on rank 0.
    if use_sharded_multi_gpu and shard_dir is not None:
        expected_shards = [
            os.path.join(shard_dir, f"resets_{reset_type}.rank{i:03d}.pt") for i in range(world_size)
        ]
        if rank == 0:
            print(f"[Shard mode] Waiting for {len(expected_shards)} shard files before merge...")
            _wait_for_shards(expected_shards, timeout_s=args_cli.shard_wait_timeout)
            merged_data = _merge_shards(expected_shards)
            torch.save(merged_data, output_file_path)
            merged_count = _count_samples(merged_data)
            print(f"[Shard mode] Merged dataset saved to: {output_file_path}")
            print(f"[Shard mode] Merged sample count: {merged_count}")
            final_count = _enforce_exact_sample_count(
                output_file_path,
                args_cli.num_reset_states,
                label="merged",
            )
            print(f"[Shard mode] Final exact merged sample count: {final_count}")
            if not args_cli.keep_shards:
                for shard_path in expected_shards:
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(shard_path)
                with contextlib.suppress(OSError):
                    os.rmdir(shard_dir)
                with contextlib.suppress(OSError):
                    os.rmdir(os.path.dirname(shard_dir))
                print("[Shard mode] Deleted shard files after merge.")
        else:
            print(f"[Shard mode] Rank {rank} shard ready: {shard_file_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
