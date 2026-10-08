"""Physical contracts for joint-target behavior-cloning actions."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer


DATASET_ACTION = "dataset_action"
DELTA_JOINT_STEP_V1 = "delta_joint_step_v1"
SUPPORTED_JOINT_ACTION_REPRESENTATIONS = {
    DATASET_ACTION,
    DELTA_JOINT_STEP_V1,
}


def validate_joint_action_representation(value: str) -> str:
    value = str(value)
    if value not in SUPPORTED_JOINT_ACTION_REPRESENTATIONS:
        supported = ", ".join(sorted(SUPPORTED_JOINT_ACTION_REPRESENTATIONS))
        raise ValueError(
            f"unsupported joint_action_representation={value!r}; expected one of {supported}"
        )
    return value


def absolute_joint_target_to_delta_step(
    actions: np.ndarray,
    proprio: np.ndarray,
    *,
    control_dt_s: float,
    max_velocity_rad_s: float,
    gripper_max_width_m: float = 0.08,
    range_tolerance_rad: float = 1.0e-4,
    reconstruction_tolerance_rad: float = 1.0e-6,
) -> np.ndarray:
    """Convert ``[q_target, width]`` to ``[q_target - q_measured, width]``.

    The conversion intentionally rejects contract violations instead of clipping
    them. Clipping would hide a temporal alignment or collection error.
    """
    actions = np.asarray(actions, dtype=np.float32)
    proprio = np.asarray(proprio, dtype=np.float32)
    if actions.ndim < 2 or actions.shape[-1] != 8:
        raise ValueError(f"joint actions must end in 8 values, got {actions.shape}")
    if proprio.shape[:-1] != actions.shape[:-1] or proprio.shape[-1] < 7:
        raise ValueError(
            f"proprio shape {proprio.shape} is incompatible with action shape {actions.shape}"
        )
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(proprio[..., :7])):
        raise ValueError("joint action conversion received non-finite values")
    if control_dt_s <= 0.0 or max_velocity_rad_s <= 0.0:
        raise ValueError("control_dt_s and max_velocity_rad_s must be positive")
    if gripper_max_width_m <= 0.0:
        raise ValueError("gripper_max_width_m must be positive")

    q_measured = proprio[..., :7]
    delta = actions[..., :7] - q_measured
    max_step_rad = float(control_dt_s) * float(max_velocity_rad_s)
    max_abs_delta = float(np.max(np.abs(delta)))
    if max_abs_delta > max_step_rad + float(range_tolerance_rad):
        raise ValueError(
            "delta joint step exceeds the physical contract: "
            f"max={max_abs_delta:.9f} rad, allowed={max_step_rad:.9f} rad "
            f"(+{range_tolerance_rad:g} tolerance)"
        )

    reconstruction_error = float(
        np.max(np.abs((q_measured + delta) - actions[..., :7]))
    )
    if reconstruction_error >= float(reconstruction_tolerance_rad):
        raise ValueError(
            "delta joint step reconstruction failed: "
            f"max error={reconstruction_error:.9g} rad"
        )

    widths = actions[..., 7]
    binary_distance = np.minimum(
        np.abs(widths), np.abs(widths - float(gripper_max_width_m))
    )
    if float(np.max(binary_distance)) > 1.0e-5:
        raise ValueError(
            "gripper targets must be binary total widths 0 or "
            f"{gripper_max_width_m:g} m"
        )

    converted = np.empty_like(actions, dtype=np.float32)
    converted[..., :7] = delta
    converted[..., 7] = widths
    return converted


def delta_joint_step_normalizer(
    stats: Mapping[str, np.ndarray],
    *,
    control_dt_s: float,
    max_velocity_rad_s: float,
    gripper_max_width_m: float = 0.08,
) -> SingleFieldLinearNormalizer:
    """Build the fixed physical normalizer for ``delta_joint_step_v1``."""
    max_step_rad = float(control_dt_s) * float(max_velocity_rad_s)
    if max_step_rad <= 0.0 or gripper_max_width_m <= 0.0:
        raise ValueError("joint-step and gripper ranges must be positive")

    dtype = np.asarray(stats["min"]).dtype
    if np.asarray(stats["min"]).shape != (8,):
        raise ValueError(f"delta joint action stats must have shape (8,), got {stats['min'].shape}")
    scale = np.asarray(
        [1.0 / max_step_rad] * 7 + [2.0 / float(gripper_max_width_m)],
        dtype=dtype,
    )
    offset = np.asarray([0.0] * 7 + [-1.0], dtype=dtype)
    normalized_stats = {key: np.asarray(value, dtype=dtype) for key, value in stats.items()}
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=normalized_stats,
    )
