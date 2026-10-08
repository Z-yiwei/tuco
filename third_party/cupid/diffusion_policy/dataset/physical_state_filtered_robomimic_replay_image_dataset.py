"""A zero-copy physical-state subset view over an OmniReset image dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.robomimic_replay_image_dataset import (
    RobomimicReplayImageDataset,
)


class PhysicalStateFilteredRobomimicReplayImageDataset(RobomimicReplayImageDataset):
    """Reuse the full replay cache while sampling only selected physical states."""

    def __init__(
        self,
        *args,
        source_zarr: str,
        selected_state_ids_path: str,
        expected_selected_states: int = 600,
        expected_repeats_per_state: int = 5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        source = zarr.open(str(Path(source_zarr).resolve()), mode="r")
        reset_ids = None
        for key in ("meta/reset_state_indices", "meta/reset_state_ids"):
            if key in source:
                reset_ids = np.asarray(source[key], dtype=np.int64)
                break
        if reset_ids is None:
            raise KeyError(f"no reset-state IDs in {source_zarr}")
        if reset_ids.shape != (self.replay_buffer.n_episodes,):
            raise ValueError(
                f"reset-state IDs {reset_ids.shape} do not match "
                f"{self.replay_buffer.n_episodes} replay episodes"
            )
        selected_path = Path(selected_state_ids_path).resolve()
        selected = np.asarray(
            json.loads(selected_path.read_text(encoding="utf-8")), dtype=np.int64
        )
        if selected.shape != (int(expected_selected_states),):
            raise ValueError(
                f"expected {expected_selected_states} selected states, got {selected.shape}"
            )
        if len(np.unique(selected)) != len(selected):
            raise ValueError("selected physical states are not unique")
        selected_mask = np.isin(reset_ids, selected)
        counts = {
            int(state_id): int(np.count_nonzero(reset_ids == state_id))
            for state_id in selected
        }
        bad = {
            state_id: count
            for state_id, count in counts.items()
            if count != int(expected_repeats_per_state)
        }
        if bad:
            raise ValueError(f"selected states have unexpected repetition counts: {bad}")
        expected_episodes = int(expected_selected_states) * int(expected_repeats_per_state)
        if int(selected_mask.sum()) != expected_episodes:
            raise ValueError(
                f"expected {expected_episodes} selected episodes, got {selected_mask.sum()}"
            )

        # Preserve the full-data split assignment, then restrict every split
        # to selected physical states. Normalization is computed later from
        # this restricted training split, never from validation or real data.
        self.train_mask = np.asarray(self.train_mask, dtype=bool) & selected_mask
        self.val_mask = np.asarray(self.val_mask, dtype=bool) & selected_mask
        self.holdout_mask = np.asarray(self.holdout_mask, dtype=bool) & selected_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.train_mask,
            key_first_k=self.key_first_k,
        )
        self.selected_state_ids = selected
        self.selected_state_ids_path = str(selected_path)
        print(
            "PHYSICAL_STATE_FILTER "
            f"states={len(selected)} episodes={int(selected_mask.sum())} "
            f"train_episodes={int(self.train_mask.sum())} "
            f"val_episodes={int(self.val_mask.sum())} "
            f"repeats={expected_repeats_per_state} manifest={selected_path}",
            flush=True,
        )
