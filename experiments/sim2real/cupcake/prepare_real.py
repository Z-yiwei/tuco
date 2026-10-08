#!/usr/bin/env python3
"""Export recorded CupCake policy decisions as absolute-joint training pairs."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

CAMERAS = ("front_rgb", "side_rgb", "wrist_rgb")


def extract(path):
    with h5py.File(path, "r") as handle:
        if not bool(handle.attrs.get("success", False)):
            raise ValueError(f"rollout is not success-marked: {path}")
        if int(handle.attrs["action_steps_per_chunk"]) != 8:
            raise ValueError(f"expected eight actions per decision: {path}")
        if not np.isclose(float(handle.attrs["trajectory_time_step_s"]), 0.1):
            raise ValueError(f"expected 0.1-second control intervals: {path}")
        plans = handle["training_capture/plans"]
        observations = handle["training_capture/observations"]
        updates = plans["update"][:]
        observation_updates = observations["update"][:]
        if len(np.unique(updates)) != len(updates) or len(
            np.unique(observation_updates)
        ) != len(observation_updates):
            raise ValueError(f"duplicate decision identifiers: {path}")
        index = {int(update): row for row, update in enumerate(observation_updates)}
        rows = np.asarray([index[int(update)] for update in updates])
        sent = plans["sent_delta"][:]
        targets = plans["absolute_q_target"][:]
        start = plans["start_q"][:]
        count = len(updates)
        if count == 0 or sent.shape != (count, 8, 8) or targets.shape != (count, 8, 7):
            raise ValueError(f"invalid decision action arrays: {path}")
        if not np.all(
            plans["completion_monotonic_ns"][:] > plans["response_monotonic_ns"][:]
        ):
            raise ValueError(f"recording contains incomplete plans: {path}")
        if not np.all(np.isfinite(sent)) or not np.all(np.isfinite(targets)):
            raise ValueError(f"non-finite action values: {path}")
        if np.max(np.abs(sent[..., :7])) > 0.02000001:
            raise ValueError(
                f"joint commands exceed the recorded velocity limit: {path}"
            )
        np.testing.assert_allclose(
            targets,
            start[:, None, :] + np.cumsum(sent[..., :7], axis=1),
            atol=1e-7,
            rtol=0,
        )
        widths = sent[..., 7:8]
        if not np.all(np.isclose(widths, 0.0) | np.isclose(widths, 0.08)):
            raise ValueError(f"expected binary gripper commands: {path}")
        arrays = {camera: observations[camera][:][rows] for camera in CAMERAS}
        for camera, image in arrays.items():
            if image.shape != (count, 2, 84, 84, 3) or image.dtype != np.uint8:
                raise ValueError(f"invalid RGB observations for {camera}: {path}")
        proprio = observations["proprio"][:][rows]
        if proprio.shape != (count, 2, 8) or not np.all(np.isfinite(proprio)):
            raise ValueError(f"invalid proprioception: {path}")
        # Labels are submitted commands, never measured displacement or closure time.
        arrays.update(
            proprio=proprio.astype(np.float32),
            action=np.concatenate([targets, widths], axis=-1).astype(np.float32),
        )
        return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for path in sorted(set(args.source.rglob("rollout_*.h5"))):
        with h5py.File(path, "r") as handle:
            if (
                bool(handle.attrs.get("success", False))
                and handle.attrs.get("outcome") == "operator_success"
            ):
                paths.append(path)
    if len(paths) != 10:
        parser.error(f"expected ten successful rollout HDF5 files, found {len(paths)}")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    extracted = [(path, extract(path)) for path in paths]
    args.output.mkdir(parents=True)
    demos = []
    for trial, (path, arrays) in enumerate(extracted):
        name = f"trial_{trial:03d}.npz"
        np.savez_compressed(args.output / name, **arrays)
        demos.append(
            dict(
                trial=trial,
                file=name,
                decisions=len(arrays["action"]),
                source_h5=str(path.resolve()),
            )
        )
    manifest = dict(
        complete=True,
        schema="cupcake_absq_decision_io_v1",
        representation="absolute_joint_target_binary_width_v1",
        execution_semantics="decision_io_timed8_final_width_deferred_close_v1",
        action_steps=8,
        observation_steps=2,
        control_dt_s=0.1,
        color_order="RGB",
        wrist_transform_applied="none",
        gripper_supervised=True,
        physical_time_relabeling=False,
        wait_frames_inserted=False,
        demos=demos,
        decisions=sum(demo["decisions"] for demo in demos),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prepared ten real rollouts: {args.output}")


if __name__ == "__main__":
    main()
