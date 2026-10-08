"""Robomimic image dataset with deterministic real-domain sequence oversampling."""

from __future__ import annotations

import h5py
import numpy as np

from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset


class DomainBalancedRobomimicReplayImageDataset(RobomimicReplayImageDataset):
    def __init__(self, *args, real_sampling_ratio: float, domain_sampling_seed: int = 42, **kwargs):
        dataset_path = kwargs.get("dataset_path")
        if dataset_path is None:
            raise ValueError("dataset_path is required")
        if not 0.0 <= real_sampling_ratio < 1.0:
            raise ValueError("real_sampling_ratio must be in [0,1)")

        super().__init__(*args, **kwargs)
        with h5py.File(dataset_path, "r") as file:
            if "real_episode_start" not in file.attrs:
                raise ValueError(f"{dataset_path}: missing real_episode_start attribute")
            real_episode_start = int(file.attrs["real_episode_start"])
            real_episode_count = int(file.attrs["real_episode_count"])

        episode_count = self.replay_buffer.n_episodes
        if real_episode_start + real_episode_count != episode_count:
            raise ValueError(
                f"real episode range [{real_episode_start}, "
                f"{real_episode_start + real_episode_count}) does not end at {episode_count}"
            )
        real_episode_mask = np.arange(episode_count) >= real_episode_start
        if np.any(self.val_mask & real_episode_mask) or np.any(self.holdout_mask & real_episode_mask):
            raise ValueError("all real episodes must be in the training split")

        episode_ends = np.asarray(self.replay_buffer.episode_ends[:], dtype=np.int64)
        sequence_episode = np.searchsorted(
            episode_ends, self.sampler.indices[:, 0], side="right"
        )
        real_sequence_mask = sequence_episode >= real_episode_start
        sim_indices = self.sampler.indices[~real_sequence_mask]
        real_indices = self.sampler.indices[real_sequence_mask]
        if len(real_indices) == 0:
            raise ValueError("no real training sequences were found")

        if real_sampling_ratio == 0.0:
            sampled_real = real_indices[:0]
        else:
            target_real_count = round(
                len(sim_indices) * real_sampling_ratio / (1.0 - real_sampling_ratio)
            )
            rng = np.random.default_rng(domain_sampling_seed)
            selection = rng.choice(len(real_indices), size=target_real_count, replace=True)
            sampled_real = real_indices[selection]
        self.sampler.indices = np.concatenate((sim_indices, sampled_real), axis=0)
        self.real_sampling_ratio = float(real_sampling_ratio)
        self.real_sequence_count = int(len(sampled_real))
        self.sim_sequence_count = int(len(sim_indices))
        effective_ratio = len(sampled_real) / max(1, len(self.sampler.indices))
        print(
            "DOMAIN_SAMPLING "
            f"target_real_ratio={real_sampling_ratio:.6f} "
            f"effective_real_ratio={effective_ratio:.6f} "
            f"sim_sequences={len(sim_indices)} real_sequences={len(sampled_real)} "
            f"unique_real_sequences={len(real_indices)} seed={domain_sampling_seed}"
        )
