"""Mix dense sim episodes with decision-aligned successful real rollouts."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.dataset.joint_action_representation import DELTA_JOINT_STEP_V1
from diffusion_policy.dataset.robomimic_replay_image_dataset import (
    RobomimicReplayImageDataset,
)


CAMERA_KEYS = ("front_rgb", "side_rgb", "wrist_rgb")


def _validate_decisions(
    source: Path,
    images: dict[str, np.ndarray],
    proprio: np.ndarray,
    actions: np.ndarray,
    action_steps: int = 2,
) -> None:
    count = len(actions)
    if count == 0:
        raise ValueError(f"{source}: no completed policy decisions")
    if proprio.shape != (count, 2, 8):
        raise ValueError(f"{source}: proprio must be [N,2,8], got {proprio.shape}")
    if actions.shape != (count, action_steps, 8):
        raise ValueError(
            f"{source}: executed_action must be [N,{action_steps},8], got {actions.shape}"
        )
    if not np.all(np.isfinite(proprio)) or not np.all(np.isfinite(actions)):
        raise ValueError(f"{source}: proprio/action contains non-finite values")
    if float(np.max(np.abs(actions[:, :, :7]))) > 0.02001:
        raise ValueError(f"{source}: executed Delta-Q exceeds 0.02 rad")
    widths = actions[:, :, 7]
    if not np.all(np.isclose(widths, 0.0) | np.isclose(widths, 0.08)):
        raise ValueError(f"{source}: gripper labels must be 0/0.08 m")
    for key in CAMERA_KEYS:
        value = images[key]
        if value.shape != (count, 2, 84, 84, 3) or value.dtype != np.uint8:
            raise ValueError(
                f"{source}: {key} must be RGB uint8 [N,2,84,84,3], got "
                f"{value.shape} {value.dtype}"
            )


def _load_npz_demo(path: Path, action_steps: int = 2):
    with np.load(path, allow_pickle=False) as arrays:
        images = {key: np.asarray(arrays[key], dtype=np.uint8) for key in CAMERA_KEYS}
        proprio = np.asarray(arrays["proprio"], dtype=np.float32)
        actions = np.asarray(arrays["executed_action"], dtype=np.float32)
    _validate_decisions(path, images, proprio, actions, action_steps)
    return images, proprio, actions


def _load_rollout_h5(path: Path):
    if not (path.parent / ".operator_success").is_file():
        raise ValueError(f"{path}: rollout is not marked with .operator_success")
    with h5py.File(path, "r") as file:
        if "training_capture" not in file:
            raise ValueError(f"{path}: missing training_capture")
        capture = file["training_capture"]
        observations = capture["observations"]
        plans = capture["plans"]
        completed = np.flatnonzero(plans["completion_monotonic_ns"][:] > 0)
        input_indices = np.asarray(
            plans["input_sample_indices"][completed], dtype=np.int64
        )
        if input_indices.ndim != 2 or input_indices.shape[1] != 2:
            raise ValueError(
                f"{path}: input_sample_indices must be [N,2], got {input_indices.shape}"
            )
        observation_count = len(observations["monotonic_ns"])
        if np.any(input_indices < 0) or np.any(input_indices >= observation_count):
            raise ValueError(f"{path}: policy input index is outside observation stream")
        proprio_stream = np.asarray(observations["proprio"][:], dtype=np.float32)
        proprio = proprio_stream[input_indices]
        images = {}
        for key in CAMERA_KEYS:
            image_stream = np.asarray(observations[key][:], dtype=np.uint8)
            images[key] = image_stream[input_indices]
        actions = np.asarray(plans["sent_delta"][completed], dtype=np.float32)
    _validate_decisions(path, images, proprio, actions)
    return images, proprio, actions


def load_real_decisions(
    root: Path,
    action_steps: int = 2,
    allow_measured_motion_proxy: bool = False,
):
    if action_steps not in (1, 2):
        raise ValueError("real action_steps must be 1 or 2")
    manifest_path = root / "dataset_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("action_label_type") == "measured_motion_proxy"
            and not allow_measured_motion_proxy
        ):
            raise ValueError("measured-motion proxy labels require explicit opt-in")
    npz_paths = sorted(root.glob("demo_*/trajectory.npz"))
    h5_paths = sorted(root.glob("*/rollout_sync.h5"))
    if npz_paths and h5_paths:
        raise ValueError(f"{root}: contains both accepted NPZ demos and raw rollout H5 files")
    paths = npz_paths or h5_paths
    if not paths:
        raise ValueError(f"{root}: no decision-aligned real rollouts found")

    image_parts = {key: [] for key in CAMERA_KEYS}
    proprio_parts = []
    action_parts = []
    source_names = []
    for path in paths:
        if path.suffix == ".npz":
            images, proprio, actions = _load_npz_demo(path, action_steps)
        else:
            if action_steps != 2:
                raise ValueError("raw training_capture H5 requires two executed actions")
            images, proprio, actions = _load_rollout_h5(path)
        for key in CAMERA_KEYS:
            image_parts[key].append(images[key])
        proprio_parts.append(proprio)
        action_parts.append(actions)
        source_names.extend([path.parent.name] * len(actions))

    return (
        {key: np.concatenate(parts, axis=0) for key, parts in image_parts.items()},
        np.concatenate(proprio_parts, axis=0),
        np.concatenate(action_parts, axis=0),
        tuple(source_names),
        tuple(str(path.resolve()) for path in paths),
    )


class DecisionAlignedDomainBalancedImageDataset(RobomimicReplayImageDataset):
    """Use sim sequences plus exact real policy decisions at a fixed sample ratio."""

    def __init__(
        self,
        *args,
        real_decision_root: str,
        real_sampling_ratio: float,
        domain_sampling_seed: int = 42,
        real_action_steps: int = 2,
        allow_measured_motion_proxy: bool = False,
        **kwargs,
    ):
        if not 0.0 <= real_sampling_ratio < 1.0:
            raise ValueError("real_sampling_ratio must be in [0,1)")
        if kwargs.get("joint_action_representation") != DELTA_JOINT_STEP_V1:
            raise ValueError("decision-aligned real rollout requires delta_joint_step_v1")
        super().__init__(*args, **kwargs)
        if self.n_obs_steps != 2:
            raise ValueError(f"decision-aligned real rollout requires n_obs_steps=2")
        action_start = self.n_obs_steps - 1
        if action_start + real_action_steps > self.horizon:
            raise ValueError("horizon is too short for real action ticks")
        if tuple(self.rgb_keys) != CAMERA_KEYS:
            raise ValueError(f"camera keys must be {CAMERA_KEYS}, got {self.rgb_keys}")
        if self.lowdim_keys != ["proprio"]:
            raise ValueError(f"low-dim keys must be ['proprio'], got {self.lowdim_keys}")

        real_root = Path(real_decision_root).expanduser().resolve()
        (
            self._real_images,
            self._real_proprio,
            self._real_actions,
            self.real_decision_sources,
            self.real_source_files,
        ) = load_real_decisions(
            real_root,
            real_action_steps,
            allow_measured_motion_proxy,
        )
        self.real_action_steps = int(real_action_steps)
        self.allow_measured_motion_proxy = bool(allow_measured_motion_proxy)
        self.real_decision_root = str(real_root)
        self.real_unique_decision_count = int(len(self._real_actions))
        self._sim_sequence_count = int(len(self.sampler))
        if real_sampling_ratio == 0.0:
            self._sampled_real_indices = np.empty(0, dtype=np.int64)
        else:
            target_real_count = round(
                self._sim_sequence_count * real_sampling_ratio / (1.0 - real_sampling_ratio)
            )
            rng = np.random.default_rng(domain_sampling_seed)
            self._sampled_real_indices = rng.choice(
                self.real_unique_decision_count,
                size=target_real_count,
                replace=True,
            )
        self.real_sampling_ratio = float(real_sampling_ratio)
        self.real_sequence_count = int(len(self._sampled_real_indices))
        self.sim_sequence_count = self._sim_sequence_count
        self._include_real = True
        effective_ratio = self.real_sequence_count / max(1, len(self))
        print(
            "DECISION_DOMAIN_SAMPLING "
            f"target_real_ratio={real_sampling_ratio:.6f} "
            f"effective_real_ratio={effective_ratio:.6f} "
            f"sim_sequences={self.sim_sequence_count} "
            f"real_sequences={self.real_sequence_count} "
            f"unique_real_decisions={self.real_unique_decision_count} "
            f"real_sources={len(self.real_source_files)} seed={domain_sampling_seed} "
            f"real_action_steps={self.real_action_steps} "
            f"measured_motion_proxy={self.allow_measured_motion_proxy}"
        )

    def get_validation_dataset(self):
        validation = super().get_validation_dataset()
        validation._include_real = False
        return validation

    def get_holdout_dataset(self):
        holdout = super().get_holdout_dataset()
        holdout._include_real = False
        return holdout

    def __len__(self):
        if getattr(self, "_include_real", False):
            return self._sim_sequence_count + len(self._sampled_real_indices)
        return len(self.sampler)

    def _real_item(self, real_index: int):
        threadpool_limits(1)
        obs = {
            key: torch.from_numpy(
                np.moveaxis(self._real_images[key][real_index], -1, 1).astype(np.float32)
                / 255.0
            )
            for key in CAMERA_KEYS
        }
        obs["proprio"] = torch.from_numpy(
            self._real_proprio[real_index].astype(np.float32)
        )

        action = np.zeros((self.horizon, 8), dtype=np.float32)
        action_valid_mask = np.zeros(self.horizon, dtype=np.bool_)
        start = self.n_obs_steps - 1
        action_steps = int(self._real_actions.shape[1])
        action[start : start + action_steps] = self._real_actions[real_index]
        action_valid_mask[start : start + action_steps] = True
        return {
            "obs": obs,
            "action": torch.from_numpy(action),
            "action_valid_mask": torch.from_numpy(action_valid_mask),
            "is_real": torch.tensor(True),
        }

    def __getitem__(self, index: int):
        if not self._include_real or index < self._sim_sequence_count:
            item = super().__getitem__(index)
            item["action_valid_mask"] = torch.ones(self.horizon, dtype=torch.bool)
            item["is_real"] = torch.tensor(False)
            return item
        sampled_index = index - self._sim_sequence_count
        real_index = int(self._sampled_real_indices[sampled_index])
        return self._real_item(real_index)
