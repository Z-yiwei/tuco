#!/usr/bin/env python3
"""Validate the ten-rollout, decision-aligned real-data contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def _validate_npz_metadata(paths: list[Path]) -> None:
    for path in paths:
        metadata_path = path.parent / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        velocity = metadata.get(
            "joint_max_velocity_rad_s",
            metadata.get("joint_velocity_limit_rad_s"),
        )
        if metadata.get("success") is not True:
            raise ValueError(f"{path}: rollout is not success-marked")
        if metadata.get("action_label_type") != "sent_delta":
            raise ValueError(f"{path}: action labels are not sent_delta")
        if not np.isclose(float(metadata.get("control_dt_s", -1)), 0.1):
            raise ValueError(f"{path}: control_dt_s is not 0.1")
        if velocity is None or not np.isclose(float(velocity), 0.2):
            raise ValueError(f"{path}: joint velocity limit is not 0.2 rad/s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--action-steps", type=int, choices=(1, 2), required=True)
    parser.add_argument("--cupid-root", type=Path, default=ROOT / "third_party/cupid")
    args = parser.parse_args()

    real_root = args.real_root.resolve()
    npz_paths = sorted(real_root.glob("demo_*/trajectory.npz"))
    h5_paths = sorted(real_root.glob("*/rollout_sync.h5"))
    if npz_paths and h5_paths:
        raise ValueError("real root mixes exported NPZ and raw H5 rollouts")
    paths = npz_paths or h5_paths
    if len(paths) != 10:
        raise ValueError(f"paper protocol requires 10 real rollouts, found {len(paths)}")
    if npz_paths:
        _validate_npz_metadata(npz_paths)

    sys.path.insert(0, str(args.cupid_root.resolve()))
    from diffusion_policy.dataset import (  # noqa: E402
        decision_aligned_domain_balanced_image_dataset as real_data,
    )

    _, proprio, actions, source_names, source_files = real_data.load_real_decisions(
        real_root,
        action_steps=args.action_steps,
        allow_measured_motion_proxy=False,
    )
    if len(source_files) != 10 or len(set(source_names)) != 10:
        raise ValueError("real decisions do not map to exactly ten unique rollouts")
    print(
        f"validated real10 decisions={len(actions)} action_steps={args.action_steps} "
        f"proprio={proprio.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
