#!/usr/bin/env python3
"""Convert OmniReset VISION zarr (front/side/wrist RGB + state/proprio + action) to
CUPID-compatible robomimic IMAGE HDF5 format.

Mirrors convert_omnireset_to_cupid.py but writes image obs keys. Images are
stored HWC uint8 (N,H,W,3) — the robomimic image dataset does moveaxis(-1,1)/255.
State/proprio are stored too so the config can opt in/out via shape_meta without re-converting.

Usage:
    python scripts/tools/convert_omnireset_image_to_cupid.py \
        --zarr_path ../datasets/franka_vision_1k.zarr \
        --hdf5_path data/omnireset/datasets/peg_image/image.hdf5 \
        --val_ratio 0.04 [--max_demos N]
"""

import argparse
import glob
import json
import os

import cv2
import h5py
import numpy as np
import zarr

CAM_KEYS = ["front_rgb", "side_rgb", "wrist_rgb"]


def resize_hwc_uint8(frames: np.ndarray, image_size: int) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.uint8)
    if image_size <= 0 or frames.shape[1:3] == (image_size, image_size):
        return frames
    return np.stack(
        [
            cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
            for frame in frames
        ],
        axis=0,
    ).astype(np.uint8, copy=False)


def convert(
    zarr_path: str | list[str],
    hdf5_path: str,
    val_ratio: float = 0.04,
    max_demos: int = 0,
    env_name: str = "OmniReset-Peg-Image",
    image_compression: str = "gzip",
    image_size: int = 0,
):
    if image_size < 0:
        raise ValueError(f"image_size must be non-negative, got {image_size}")
    patterns = [zarr_path] if isinstance(zarr_path, str) else zarr_path
    zarr_paths = sorted({
        matched_path
        for pattern in patterns
        for matched_path in glob.glob(pattern)
    })
    if not zarr_paths:
        raise ValueError(f"No zarr datasets matched: {patterns}")

    episode_refs = []
    collection_configs = []
    reference_signature = None
    present_cams = []
    has_state = has_proprio = False
    for path in zarr_paths:
        print(f"Loading zarr: {path}")
        root = zarr.open(path, mode="r")
        data = root["data"]
        episode_ends = np.asarray(root["meta/episode_ends"], dtype=np.int64)
        starts = np.concatenate([[0], episode_ends[:-1]])
        cams = [key for key in CAM_KEYS if key in data]
        signature = {
            "cams": cams,
            "camera_shapes": {key: tuple(data[key].shape[1:]) for key in cams},
            "action_shape": tuple(data["action"].shape[1:]),
            "state_shape": tuple(data["state"].shape[1:]) if "state" in data else None,
            "proprio_shape": tuple(data["proprio"].shape[1:]) if "proprio" in data else None,
        }
        if reference_signature is None:
            reference_signature = signature
            present_cams = cams
            has_state = "state" in data
            has_proprio = "proprio" in data
        elif signature != reference_signature:
            raise ValueError(
                f"Incompatible zarr schema in {path}: {signature} != {reference_signature}"
            )
        collection_configs.append(dict(root.attrs.get("collection_config", {})))
        episode_refs.extend(
            (data, int(start), int(end))
            for start, end in zip(starts, episode_ends)
        )
        print(f"  episodes: {len(episode_ends)}")
        for key in cams:
            print(f"  {key}: {data[key].shape} {data[key].dtype}")

    if max_demos and max_demos < len(episode_refs):
        episode_refs = episode_refs[:max_demos]
        print(f"[smoke] limiting to first {len(episode_refs)} demos")
    num_episodes = len(episode_refs)
    collection_config = {
        **collection_configs[0],
        "converter_source_zarrs": [os.path.abspath(path) for path in zarr_paths],
        "converter_image_processing": {
            "source_shapes_hwc": reference_signature["camera_shapes"],
            "target_size_hw": [image_size, image_size] if image_size else None,
            "resize_interpolation": "cv2.INTER_AREA" if image_size else None,
        },
    }
    print(
        f"cameras: {present_cams} | state: {has_state} | proprio: {has_proprio} "
        f"| sources: {len(zarr_paths)} | episodes: {num_episodes}"
    )

    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    print(f"Writing HDF5: {hdf5_path}")
    with h5py.File(hdf5_path, "w") as f:
        f.attrs["source_zarrs"] = json.dumps(
            [os.path.abspath(path) for path in zarr_paths]
        )
        if len(zarr_paths) == 1:
            f.attrs["source_zarr"] = os.path.abspath(zarr_paths[0])
        f.attrs["collection_config"] = json.dumps(collection_config, sort_keys=True)
        data_grp = f.create_group("data")
        demo_names = []
        total_frames = 0

        for i, (data, s, e) in enumerate(episode_refs):
            ep_len = e - s
            if ep_len < 2:
                continue
            demo_name = f"demo_{len(demo_names)}"
            demo_names.append(demo_name)
            g = data_grp.create_group(demo_name)

            g.create_dataset("actions", data=data["action"][s:e].astype(np.float64))
            obs = g.create_group("obs")
            for k in present_cams:
                # HWC uint8 — robomimic image dataset moveaxis(-1,1)/255 at load
                images = resize_hwc_uint8(data[k][s:e], image_size)
                image_kwargs = {"chunks": (1,) + images.shape[1:]}
                if image_compression == "gzip":
                    image_kwargs.update(compression="gzip", compression_opts=1)
                obs.create_dataset(
                    k,
                    data=images,
                    **image_kwargs,
                )
            if has_state:
                obs.create_dataset("state", data=data["state"][s:e].astype(np.float64))
            if has_proprio:
                obs.create_dataset("proprio", data=data["proprio"][s:e].astype(np.float64))

            g.create_dataset("rewards", data=np.zeros(ep_len, dtype=np.float64))
            g.create_dataset("dones", data=np.zeros(ep_len, dtype=np.bool_))
            g.attrs["num_samples"] = ep_len
            total_frames += ep_len
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{num_episodes} demos, {total_frames} frames")

        total = len(demo_names)
        data_grp.attrs["total"] = total
        print(f"Wrote {total} demos, {total_frames} frames")

        mask_grp = f.create_group("mask")
        idx = np.arange(total)
        np.random.default_rng(seed=42).shuffle(idx)
        val_count = max(1, int(total * val_ratio))
        val_idx = sorted(idx[:val_count]); train_idx = sorted(idx[val_count:])
        mask_grp.create_dataset("train", data=[f"demo_{i}".encode() for i in train_idx])
        mask_grp.create_dataset("valid", data=[f"demo_{i}".encode() for i in val_idx])
        print(f"Split: {len(train_idx)} train, {len(val_idx)} val")

        data_grp.attrs["env_args"] = json.dumps(
            {"env_name": env_name, "type": 1, "env_kwargs": {}})

    print("Done!")
    with h5py.File(hdf5_path, "r") as f:
        d0 = f["data/demo_0"]
        print(f"\nSample demo_0: actions {d0['actions'].shape}")
        for k in d0["obs"].keys():
            print(f"  obs/{k}: {d0[f'obs/{k}'].shape} {d0[f'obs/{k}'].dtype}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--zarr_path",
        nargs="+",
        required=True,
        help="One or more zarr paths or glob patterns.",
    )
    ap.add_argument("--hdf5_path", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.04)
    ap.add_argument("--max_demos", type=int, default=0, help="0 = all; else smoke on first N")
    ap.add_argument("--env_name", default="OmniReset-Peg-Image")
    ap.add_argument(
        "--image_compression",
        choices=("gzip", "none"),
        default="gzip",
        help="HDF5 image compression. Use none for faster local conversion and training.",
    )
    ap.add_argument(
        "--image_size",
        type=int,
        default=0,
        help="Optional square output size. Zero preserves source images; use 84 for Vision DP.",
    )
    args = ap.parse_args()
    convert(
        args.zarr_path,
        args.hdf5_path,
        args.val_ratio,
        args.max_demos,
        args.env_name,
        args.image_compression,
        args.image_size,
    )
