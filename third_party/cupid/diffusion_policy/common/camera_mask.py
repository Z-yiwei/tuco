"""Deterministic train-time camera masking for multi-view policies."""

from __future__ import annotations

from collections.abc import Sequence

import torch


CAMERA_MASK_NONE = "none"
CAMERA_MASK_ONE_OR_NONE_UNIFORM = "one_or_none_uniform"


def apply_training_camera_mask(
    batch: dict,
    *,
    mode: str,
    camera_keys: Sequence[str],
    seed: int,
    global_step: int,
    rank: int = 0,
) -> tuple[dict, torch.Tensor]:
    """Mask no camera or exactly one camera with equal probability per sample."""
    keys = tuple(camera_keys)
    if mode == CAMERA_MASK_NONE:
        first = batch["obs"][keys[0]]
        counts = torch.zeros(len(keys) + 1, dtype=torch.int64, device=first.device)
        counts[0] = first.shape[0]
        return batch, counts
    if mode != CAMERA_MASK_ONE_OR_NONE_UNIFORM:
        raise ValueError(f"unsupported camera mask mode: {mode!r}")
    if not keys:
        raise ValueError("camera_keys must not be empty")

    obs = batch.get("obs")
    if not isinstance(obs, dict):
        raise TypeError("batch['obs'] must be a dict")
    missing = [key for key in keys if key not in obs]
    if missing:
        raise KeyError(f"camera mask keys are missing from observations: {missing}")

    first = obs[keys[0]]
    batch_size = first.shape[0]
    for key in keys:
        if obs[key].shape[0] != batch_size:
            raise ValueError("all camera tensors must have the same batch size")

    generator = torch.Generator(device=first.device)
    mask_seed = int(seed) + 1_000_003 * int(rank) + 97_003 * int(global_step)
    generator.manual_seed(mask_seed)
    categories = torch.randint(
        0,
        len(keys) + 1,
        (batch_size,),
        generator=generator,
        device=first.device,
    )

    masked_batch = dict(batch)
    masked_obs = dict(obs)
    for category, key in enumerate(keys, start=1):
        sample_mask = categories == category
        if torch.any(sample_mask):
            tensor = obs[key].clone()
            tensor[sample_mask] = 0.0
            masked_obs[key] = tensor
    masked_batch["obs"] = masked_obs
    counts = torch.bincount(categories, minlength=len(keys) + 1)
    return masked_batch, counts
