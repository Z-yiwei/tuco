"""Train Vision-DP only on exact, decision-aligned real robot samples."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.decision_aligned_domain_balanced_image_dataset import (
    CAMERA_KEYS,
    load_real_decisions,
)
from diffusion_policy.dataset.joint_action_representation import (
    DELTA_JOINT_STEP_V1,
    delta_joint_step_normalizer,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer


class DecisionAlignedRealOnlyImageDataset(BaseImageDataset):
    """Repeat exact real policy decisions to a fixed optimizer-step budget.

    Each source sample contains the two observations consumed by the policy and
    the one or two Delta-Q ticks that the controller actually accepted.  Only
    those action positions contribute to the diffusion loss.  No simulation
    episode, cache, sample, or normalization statistic is loaded by this dataset.

    The validation view intentionally contains all unique real decisions.  It
    is a training diagnostic only, not an unbiased performance estimate; final
    model comparison must use closed-loop robot rollouts.
    """

    def __init__(
        self,
        shape_meta: dict,
        real_decision_root: str,
        target_sequence_count: int,
        sequence_sampling_seed: int = 43,
        expected_real_sources: int | None = None,
        real_action_steps: int = 2,
        horizon: int = 16,
        pad_before: int = 1,
        pad_after: int = 7,
        n_obs_steps: int = 2,
        abs_action: bool = False,
        joint_action_representation: str = DELTA_JOINT_STEP_V1,
        joint_control_dt_s: float = 0.1,
        joint_max_velocity_rad_s: float = 0.2,
        joint_gripper_max_width_m: float = 0.08,
        **unused_kwargs,
    ):
        del unused_kwargs
        if abs_action:
            raise ValueError("real-only decision data require abs_action=false")
        if joint_action_representation != DELTA_JOINT_STEP_V1:
            raise ValueError("real-only decision data require delta_joint_step_v1")
        if int(n_obs_steps) != 2:
            raise ValueError("real-only decision data require n_obs_steps=2")
        if int(real_action_steps) not in (1, 2):
            raise ValueError("real_action_steps must be 1 or 2")
        if int(horizon) < int(n_obs_steps) + int(real_action_steps):
            raise ValueError("horizon is too short for the executed real actions")
        if int(pad_before) != 1 or int(pad_after) != 7:
            raise ValueError("expected the aligned Vision-DP padding contract 1/7")
        if not np.isclose(float(joint_control_dt_s), 0.1):
            raise ValueError("real-only decision data require control_dt_s=0.1")
        if not np.isclose(float(joint_max_velocity_rad_s), 0.2):
            raise ValueError("real-only decision data require max velocity=0.2 rad/s")
        if not np.isclose(float(joint_gripper_max_width_m), 0.08):
            raise ValueError("real-only decision data require gripper width=0.08 m")

        obs_meta = shape_meta["obs"]
        rgb_keys = tuple(
            key for key, value in obs_meta.items() if value.get("type") == "rgb"
        )
        lowdim_keys = tuple(
            key for key, value in obs_meta.items() if value.get("type") != "rgb"
        )
        if rgb_keys != CAMERA_KEYS:
            raise ValueError(f"camera keys must be {CAMERA_KEYS}, got {rgb_keys}")
        if lowdim_keys != ("proprio",):
            raise ValueError(f"low-dim keys must be ('proprio',), got {lowdim_keys}")

        (
            self._real_images,
            self._real_proprio,
            self._real_actions,
            self.real_decision_sources,
            self.real_source_files,
        ) = load_real_decisions(
            Path(real_decision_root).expanduser().resolve(),
            action_steps=int(real_action_steps),
            allow_measured_motion_proxy=False,
        )

        self.real_decision_root = str(Path(real_decision_root).expanduser().resolve())
        self.real_unique_decision_count = int(len(self._real_actions))
        self.real_action_steps = int(real_action_steps)
        if expected_real_sources is not None and len(self.real_source_files) != int(
            expected_real_sources
        ):
            raise ValueError(
                f"expected {expected_real_sources} real sources, got "
                f"{len(self.real_source_files)}"
            )

        target_sequence_count = int(target_sequence_count)
        if target_sequence_count < 1:
            raise ValueError("target_sequence_count must be positive")
        rng = np.random.default_rng(int(sequence_sampling_seed))
        self._sampled_real_indices = rng.choice(
            self.real_unique_decision_count,
            size=target_sequence_count,
            replace=target_sequence_count > self.real_unique_decision_count,
        ).astype(np.int64)

        self.shape_meta = shape_meta
        self.rgb_keys = list(CAMERA_KEYS)
        self.lowdim_keys = ["proprio"]
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.joint_action_representation = joint_action_representation
        self.joint_control_dt_s = float(joint_control_dt_s)
        self.joint_max_velocity_rad_s = float(joint_max_velocity_rad_s)
        self.joint_gripper_max_width_m = float(joint_gripper_max_width_m)
        self.target_sequence_count = target_sequence_count
        self.sequence_sampling_seed = int(sequence_sampling_seed)
        self._validation_view = False

        print(
            "DECISION_REAL_ONLY_SAMPLING "
            f"real_sequences={self.target_sequence_count} "
            f"unique_real_decisions={self.real_unique_decision_count} "
            f"real_sources={len(self.real_source_files)} "
            f"real_action_steps={self.real_action_steps} "
            f"seed={self.sequence_sampling_seed} sim_sequences=0",
            flush=True,
        )

    def get_validation_dataset(self):
        validation = copy.copy(self)
        validation._sampled_real_indices = np.arange(
            self.real_unique_decision_count, dtype=np.int64
        )
        validation._validation_view = True
        return validation

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        del kwargs
        normalizer = LinearNormalizer()
        action_values = self._real_actions.reshape(-1, 8)
        normalizer["action"] = delta_joint_step_normalizer(
            array_to_stats(action_values),
            control_dt_s=self.joint_control_dt_s,
            max_velocity_rad_s=self.joint_max_velocity_rad_s,
            gripper_max_width_m=self.joint_gripper_max_width_m,
        )
        proprio_values = self._real_proprio.reshape(-1, 8)
        normalizer["proprio"] = get_range_normalizer_from_stat(
            array_to_stats(proprio_values)
        )
        for key in CAMERA_KEYS:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._real_actions.reshape(-1, 8))

    def __len__(self) -> int:
        return int(len(self._sampled_real_indices))

    def __getitem__(self, index: int):
        threadpool_limits(1)
        real_index = int(self._sampled_real_indices[index])
        obs = {
            key: torch.from_numpy(
                np.moveaxis(self._real_images[key][real_index], -1, 1).astype(
                    np.float32
                )
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
