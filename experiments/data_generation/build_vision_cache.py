#!/usr/bin/env python3
"""Create the replay cache consumed by sim-to-real attribution and training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import hydra
import zarr
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
CUPID = ROOT / "third_party" / "cupid"
CONFIGS = {
    "peg": "configs/image/omnireset_peg_image_3cam_jointtarget_teamhome_300x10/config.yaml",
    "stackcube": "configs/image/omnireset_stackcube_image_3cam_jointtarget_model1900_axis_v06_lift2cm_400x10/config.yaml",
    "cupcake": "configs/image/omnireset_cupcake_fr3_xy5_t0_v020_3cam_jointtarget/config.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(CONFIGS), required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=6000)
    args = parser.parse_args()

    source = args.hdf5.resolve()
    if not source.is_file():
        parser.error(f"missing HDF5: {source}")
    with h5py.File(source, "r") as handle:
        if len(handle["data"]) != args.episodes:
            parser.error(
                f"expected {args.episodes} episodes, found {len(handle['data'])}"
            )

    dataset_name = source.parent.name
    relative = Path("data/omnireset/datasets") / dataset_name / source.name
    link = CUPID / relative
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve() != source:
            parser.error(f"dataset link points elsewhere: {link}")
    else:
        link.symlink_to(source)

    config_path = CUPID / CONFIGS[args.task]
    cfg = OmegaConf.load(config_path)
    cache = args.cache.resolve()
    cache.parent.mkdir(parents=True, exist_ok=True)
    updates: dict[str, object] = {
        "dataset_path": str(relative),
        "use_cache": True,
        "cache_path": str(cache),
        "normalizer_episode_limit": args.episodes,
        "normalizer_train_split_only": True,
        "val_ratio": 0.04,
    }
    if args.task in {"peg", "stackcube"}:
        updates.update(
            joint_action_representation="delta_joint_step_v1",
            joint_control_dt_s=0.1,
            joint_max_velocity_rad_s=0.2,
            joint_gripper_max_width_m=0.08,
        )
    for key, value in updates.items():
        OmegaConf.update(cfg.task.dataset, key, value, force_add=True)

    previous = Path.cwd()
    sys.path.insert(0, str(CUPID))
    try:
        os.chdir(CUPID)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
    finally:
        os.chdir(previous)
    if dataset.replay_buffer.n_episodes != args.episodes:
        raise RuntimeError("cache contains the wrong episode count")
    with zarr.ZipStore(str(cache), mode="r") as store:
        root = zarr.group(store=store)
        if root["meta/episode_ends"].shape != (args.episodes,):
            raise RuntimeError("cache contains the wrong episode index")
    print(f"cache ready: task={args.task} episodes={args.episodes} path={cache}")


if __name__ == "__main__":
    main()
