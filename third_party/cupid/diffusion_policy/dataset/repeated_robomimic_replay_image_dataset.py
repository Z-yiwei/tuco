"""Robomimic image dataset with deterministic train-sequence resampling."""

from __future__ import annotations

import numpy as np

from diffusion_policy.dataset.robomimic_replay_image_dataset import (
    RobomimicReplayImageDataset,
)


class RepeatedRobomimicReplayImageDataset(RobomimicReplayImageDataset):
    """Resize the training sequence index to a fixed optimizer-step budget.

    Validation is unchanged.  This is useful for small real-only datasets where
    a normal epoch would contain far fewer batches than an aligned sim baseline.
    """

    def __init__(
        self,
        *args,
        target_sequence_count: int,
        sequence_sampling_seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        target_sequence_count = int(target_sequence_count)
        if target_sequence_count < 1:
            raise ValueError("target_sequence_count must be positive")

        base_indices = np.asarray(self.sampler.indices)
        base_sequence_count = len(base_indices)
        if base_sequence_count < 1:
            raise ValueError("no training sequences were found")

        rng = np.random.default_rng(int(sequence_sampling_seed))
        selected_positions = rng.choice(
            base_sequence_count,
            size=target_sequence_count,
            replace=target_sequence_count > base_sequence_count,
        )
        self.sampler.indices = base_indices[selected_positions]
        self.base_sequence_count = int(base_sequence_count)
        self.target_sequence_count = target_sequence_count
        self.sequence_sampling_seed = int(sequence_sampling_seed)
        self.unique_resampled_sequence_count = int(
            len(np.unique(selected_positions))
        )
        print(
            "SEQUENCE_RESAMPLING "
            f"base_sequences={self.base_sequence_count} "
            f"target_sequences={self.target_sequence_count} "
            f"unique_sequences={self.unique_resampled_sequence_count} "
            f"replacement={target_sequence_count > base_sequence_count} "
            f"seed={self.sequence_sampling_seed}"
        )
