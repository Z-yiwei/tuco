"""Replay-gated CupCake image dataset with absolute joint-target actions."""

import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


CAMERAS = ("front_rgb", "side_rgb", "wrist_rgb")
REP = "absolute_joint_target_binary_width_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def grouped_masks(state_ids, seed=42, val_ratio=0.04):
    ids = np.asarray(state_ids, dtype=np.int64)
    unique = np.unique(ids)
    np.random.default_rng(seed).shuffle(unique)
    count = min(max(1, int(len(unique) * val_ratio)), len(unique) - 1)
    validation = np.isin(ids, unique[:count])
    return ~validation, validation


class CupCakeAbsoluteGroupedImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta,
        dataset_path,
        cache_path,
        horizon=16,
        pad_before=1,
        pad_after=7,
        n_obs_steps=2,
        seed=42,
        val_ratio=0.04,
        joint_action_representation=REP,
        allow_diagnostic=False,
    ):
        _require(not allow_diagnostic, "diagnostic CupCake data are not trainable")
        dataset_path = Path(dataset_path)
        cache_path = Path(cache_path)
        _require(dataset_path.is_file(), f"missing CupCake HDF5: {dataset_path}")
        _require(cache_path.is_file(), f"missing CupCake cache: {cache_path}")
        _require(
            joint_action_representation == REP,
            "CupCake action representation mismatch",
        )
        _require(
            (horizon, pad_before, pad_after, n_obs_steps) == (16, 1, 7, 2),
            "CupCake requires horizon/padding/observations 16/1/7/2",
        )
        _require(seed == 42 and val_ratio == 0.04, "CupCake split must be seed 42 / 0.04")
        _require(
            set(shape_meta["obs"]) == {*CAMERAS, "proprio"},
            "CupCake observation keys mismatch",
        )
        _require(list(shape_meta["action"]["shape"]) == [8], "CupCake action dim is not 8")
        self.cache_path = str(cache_path.resolve())
        self.zip_store = zarr.ZipStore(self.cache_path, mode="r")
        self.owner_pid = os.getpid()
        root = zarr.open_group(self.zip_store, mode="r")
        _require(
            root.attrs["complete"] and root.attrs["joint_action_representation"] == REP,
            "CupCake cache is incomplete or has the wrong action representation",
        )
        _require(
            bool(root.attrs["absolute_action"]) and root.attrs["wrist_vertical_flip"] == 0,
            "CupCake cache must be absolute-Q with no wrist flip",
        )
        ids = np.asarray(root["meta/reset_state_ids"])
        variants = np.asarray(root["meta/visual_variant_ids"])
        train = np.asarray(root["meta/train_mask"], dtype=bool)
        validation = np.asarray(root["meta/val_mask"], dtype=bool)
        _require(
            ids.shape == variants.shape == train.shape == validation.shape == (6000,),
            "CupCake cache metadata shapes are invalid",
        )
        _require(
            np.all(train ^ validation) and not set(ids[train]) & set(ids[validation]),
            "CupCake grouped train/validation split is invalid",
        )
        expected_train, expected_validation = grouped_masks(ids, seed=seed, val_ratio=val_ratio)
        _require(
            np.array_equal(train, expected_train)
            and np.array_equal(validation, expected_validation),
            "CupCake cache split does not match the declared seed",
        )
        _require(
            set(zip(ids.tolist(), variants.tolist()))
            == {(state_id, variant) for state_id in range(1200) for variant in range(5)},
            "CupCake cache does not contain every state/variant pair exactly once",
        )
        self.replay_buffer = ReplayBuffer(root)
        _require(self.replay_buffer.n_episodes == 6000, "CupCake episode count mismatch")
        _require(
            set(root["data"].array_keys()) == {*CAMERAS, "action", "proprio"},
            "CupCake cache data keys mismatch",
        )
        episode_ends = np.asarray(root["meta/episode_ends"], dtype=np.int64)
        _require(
            episode_ends.shape == (6000,) and np.all(np.diff(episode_ends) > 0),
            "CupCake episode boundaries are invalid",
        )
        frame_count = int(episode_ends[-1])
        _require(
            all(
                root["data"][key].shape[0] == frame_count
                for key in (*CAMERAS, "action", "proprio")
            ),
            "CupCake arrays do not match the episode boundaries",
        )
        self.reset_state_ids = ids
        self.visual_variant_ids = variants
        self.train_mask, self.val_mask = train, validation
        self.holdout_mask = np.zeros_like(train)
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.key_first_k = {key: n_obs_steps for key in (*CAMERAS, "proprio")}
        self.sampler = self._sampler(train)
        self.normalizer_train_mask = train.copy()
        split_summary = {
            "train_episodes": int(train.sum()),
            "val_episodes": int(validation.sum()),
            "train_states": len(np.unique(ids[train])),
            "val_states": len(np.unique(ids[validation])),
            "joint_action_representation": REP,
        }
        print(
            "CUPCAKE_ABSOLUTE_GROUP_SPLIT",
            json.dumps(split_summary),
            flush=True,
        )

    def _sampler(self, mask):
        return SequenceSampler(
            self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=mask,
            key_first_k=self.key_first_k,
        )

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.sampler = self._sampler(self.val_mask)
        result.train_mask = self.val_mask
        return result

    def get_normalizer(self, **kwargs):
        ends = self.replay_buffer.episode_ends[:]
        lengths = np.diff(np.r_[0, ends])
        frame_mask = np.repeat(self.normalizer_train_mask, lengths)
        normalizer = LinearNormalizer()
        normalizer["action"] = get_range_normalizer_from_stat(
            array_to_stats(self.replay_buffer["action"][:][frame_mask])
        )
        normalizer["proprio"] = get_range_normalizer_from_stat(
            array_to_stats(self.replay_buffer["proprio"][:][frame_mask])
        )
        for key in CAMERAS:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self):
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        if self.owner_pid != os.getpid():
            self.zip_store.close()
            self.zip_store = zarr.ZipStore(self.cache_path, mode="r")
            self.replay_buffer = ReplayBuffer(zarr.open_group(self.zip_store, mode="r"))
            self.sampler = self._sampler(self.train_mask)
            self.owner_pid = os.getpid()
        threadpool_limits(1)
        data = self.sampler.sample_sequence(index)
        obs = {
            key: torch.from_numpy(
                np.moveaxis(data[key][: self.n_obs_steps], -1, 1).astype(np.float32)
                / 255.0
            )
            for key in CAMERAS
        }
        obs["proprio"] = torch.from_numpy(
            data["proprio"][: self.n_obs_steps].astype(np.float32)
        )
        return dict(obs=obs, action=torch.from_numpy(data["action"].astype(np.float32)))
