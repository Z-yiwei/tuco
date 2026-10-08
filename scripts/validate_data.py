#!/usr/bin/env python3
"""Validate release datasets before launching a long experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tuco.protocol import load_protocol


def validate_zarr(path: Path, *, episodes: int | None, obs_dim: int, act_dim: int,
                  success: bool, fixed_length: int | None) -> dict[str, object]:
    import zarr

    root = zarr.open(str(path), mode="r")
    for key in ("data/state", "data/action", "meta/episode_ends"):
        if key not in root:
            raise KeyError(f"{path}: missing {key}")
    state = root["data/state"]
    action = root["data/action"]
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if state.ndim != 2 or state.shape[1] != obs_dim:
        raise ValueError(f"{path}: state shape {state.shape}, expected (*, {obs_dim})")
    if action.ndim != 2 or action.shape != (state.shape[0], act_dim):
        raise ValueError(f"{path}: action shape {action.shape}, expected ({state.shape[0]}, {act_dim})")
    if len(ends) == 0 or np.any(np.diff(np.concatenate([[0], ends])) <= 0):
        raise ValueError(f"{path}: invalid episode_ends")
    if int(ends[-1]) != state.shape[0]:
        raise ValueError(f"{path}: final episode end does not equal frame count")
    if episodes is not None and len(ends) != episodes:
        raise ValueError(f"{path}: {len(ends)} episodes, expected {episodes}")
    lengths = np.diff(np.concatenate([[0], ends]))
    if fixed_length is not None and np.any(lengths != fixed_length):
        raise ValueError(f"{path}: trajectories are not all length {fixed_length}")
    if success and "data/success" not in root:
        raise KeyError(f"{path}: rollout dataset lacks data/success")
    return {
        "path": str(path.resolve()),
        "episodes": int(len(ends)),
        "frames": int(state.shape[0]),
        "state_dim": int(state.shape[1]),
        "action_dim": int(action.shape[1]),
        "minimum_length": int(lengths.min()),
        "maximum_length": int(lengths.max()),
    }


def validate_single_sim(root: Path) -> dict[str, object]:
    import h5py

    files = {}
    for task in ("lift", "square", "transport"):
        path = root / "robomimic" / "datasets" / task / "mh" / "low_dim_abs.hdf5"
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path, "r") as handle:
            demos = handle["data"]
            if len(demos) != 300:
                raise ValueError(f"{path}: expected 300 demonstrations, got {len(demos)}")
            first = demos[sorted(demos, key=lambda name: int(name.split("_")[-1]))[0]]
            if first["actions"].shape[1] not in (7, 14):
                raise ValueError(f"{path}: unexpected action shape {first['actions'].shape}")
        files[task] = str(path.resolve())
    return {"setting": "single-sim", "datasets": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="setting", required=True)
    single = subparsers.add_parser("single-sim")
    single.add_argument("--root", type=Path, required=True)
    sim = subparsers.add_parser("sim2sim")
    sim.add_argument("--task", choices=("peg", "stackcube", "cupcake"), required=True)
    sim.add_argument("--target", type=Path, required=True)
    sim.add_argument("--source", type=Path, required=True)
    sim.add_argument("--rollouts", type=Path)
    sim.add_argument(
        "--allow-subset",
        action="store_true",
        help="Validate structure but allow fewer episodes than the formal protocol",
    )
    args = parser.parse_args()

    if args.setting == "single-sim":
        result = validate_single_sim(args.root.resolve())
    else:
        config = load_protocol("sim2sim", args.task)
        data = config["data"]
        length = data.get("trajectory_length", data.get("maximum_trajectory_length"))
        result = {
            "setting": "sim2sim",
            "task": args.task,
            "target": validate_zarr(
                args.target.resolve(), episodes=None if args.allow_subset else data["target_demonstrations"],
                obs_dim=data["observation_dim"], act_dim=data["action_dim"],
                success=False, fixed_length=length if args.task != "stackcube" else None,
            ),
            "source": validate_zarr(
                args.source.resolve(), episodes=None if args.allow_subset else data["source_demonstrations"],
                obs_dim=data["observation_dim"], act_dim=data["action_dim"],
                success=False, fixed_length=length if args.task != "stackcube" else None,
            ),
        }
        if args.rollouts is not None:
            result["rollouts"] = validate_zarr(
                args.rollouts.resolve(), episodes=None if args.allow_subset else config["selection"]["rollouts"],
                obs_dim=data["observation_dim"], act_dim=data["action_dim"],
                success=True, fixed_length=None,
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
