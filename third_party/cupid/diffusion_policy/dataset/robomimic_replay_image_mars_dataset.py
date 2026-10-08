import copy
from typing import Dict, Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.common.sampler import SequenceSampler


class RobomimicReplayImageMarsDataset(RobomimicReplayImageDataset):
    """Image dataset wrapper that adds non-leaking past actions and KNN fields."""

    _STATE_HIST = 5
    _STATE_BLOCKS = {
        # Franka OmniReset policy obs is term-major with 5-step history:
        # poseA[0:30], prev_actions[30:65], joint_pos[65:110], poseB/C/D.
        # The newest/current sub-block is the last history slot.
        "joint_pos": (65, 9),
    }

    def __init__(
        self,
        *args,
        source_action_steps=8,
        knn_k=20,
        knn_key_source="resnet",
        knn_rgb_keys: Optional[list] = None,
        knn_resnet_weights="IMAGENET1K_V1",
        knn_resnet_batch_size=256,
        knn_search_batch_size=1024,
        knn_proprio_key="state",
        knn_proprio_weight=1.0,
        knn_weight_temp=None,
        knn_debug_samples=3,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.source_action_steps = int(source_action_steps)
        if self.source_action_steps <= 0:
            raise ValueError(f"source_action_steps must be positive, got {source_action_steps}.")
        if knn_key_source != "resnet":
            raise ValueError("This Robomimic MARS* dataset currently supports knn_key_source='resnet'.")
        self.knn_k = int(knn_k)
        self.knn_key_source = knn_key_source
        self.knn_rgb_keys = list(knn_rgb_keys) if knn_rgb_keys is not None else list(self.rgb_keys)
        self.knn_resnet_weights = knn_resnet_weights
        self.knn_resnet_batch_size = int(knn_resnet_batch_size)
        self.knn_search_batch_size = int(knn_search_batch_size)
        self.knn_proprio_key = knn_proprio_key
        self.knn_proprio_weight = float(knn_proprio_weight)
        self.knn_weight_temp = knn_weight_temp
        self.knn_debug_samples = int(knn_debug_samples)
        self._precompute_knn()

    @staticmethod
    def _sample_pos_to_buffer_idx(
        buffer_start_idx: int,
        sample_start_idx: int,
        sample_end_idx: int,
        sample_pos: int,
    ) -> int:
        if sample_pos < sample_start_idx:
            return int(buffer_start_idx)
        real_len = int(sample_end_idx - sample_start_idx)
        if sample_pos >= sample_end_idx:
            return int(buffer_start_idx + real_len - 1)
        return int(buffer_start_idx + (sample_pos - sample_start_idx))

    def _episode_bounds(self, buffer_idx: int):
        episode_ends = self.replay_buffer.episode_ends[:]
        ep_idx = int(np.searchsorted(episode_ends, buffer_idx, side="right"))
        ep_start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        ep_end = int(episode_ends[ep_idx])
        return ep_start, ep_end

    def _past_action_for_idx(self, idx: int) -> np.ndarray:
        return np.asarray(self.replay_buffer["action"][self._past_action_indices_for_idx(idx)], dtype=np.float32)

    def _past_action_indices_for_idx(self, idx: int) -> np.ndarray:
        buffer_start_idx, _buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
        target_sample_pos = max(int(self.n_obs_steps or 1) - 1, 0)
        target_buffer_idx = self._sample_pos_to_buffer_idx(
            int(buffer_start_idx),
            int(sample_start_idx),
            int(sample_end_idx),
            target_sample_pos,
        )
        ep_start, ep_end = self._episode_bounds(target_buffer_idx)
        past_end = max(target_buffer_idx - 1, ep_start)
        past_indices = np.arange(
            past_end - self.source_action_steps + 1,
            past_end + 1,
            dtype=np.int64,
        )
        return np.clip(past_indices, ep_start, ep_end - 1)

    def _sequence_action_indices_for_idx(self, idx: int, start_pos: int, length: int) -> np.ndarray:
        buffer_start_idx, _buffer_end_idx, sample_start_idx, sample_end_idx = self.sampler.indices[idx]
        positions = np.arange(start_pos, start_pos + length, dtype=np.int64)
        return np.asarray(
            [
                self._sample_pos_to_buffer_idx(
                    int(buffer_start_idx),
                    int(sample_start_idx),
                    int(sample_end_idx),
                    int(pos),
                )
                for pos in positions
            ],
            dtype=np.int64,
        )

    def _current_frame_indices(self) -> np.ndarray:
        cur_pos = max(int(self.n_obs_steps or 1) - 1, 0)
        cur = np.empty(len(self.sampler), dtype=np.int64)
        for idx in range(len(self.sampler)):
            cur[idx] = self._sequence_action_indices_for_idx(idx, cur_pos, 1)[0]
        return cur

    def _load_hdf5_obs_array(self, key: str) -> np.ndarray:
        chunks = []
        with h5py.File(self._dataset_path, "r") as file:
            demos = file["data"]
            for demo_idx in range(len(demos)):
                obs = demos[f"demo_{demo_idx}"]["obs"]
                if key not in obs:
                    raise KeyError(
                        f"MARS* KNN proprio key obs/{key!r} not found in {self._dataset_path}; "
                        f"available obs keys in demo_{demo_idx}: {list(obs.keys())}"
                    )
                chunks.append(obs[key][:].astype(np.float32))
        arr = np.concatenate(chunks, axis=0)
        n_steps = int(self.replay_buffer["action"].shape[0])
        if arr.shape[0] != n_steps:
            raise ValueError(
                f"obs/{key} has {arr.shape[0]} rows but replay action buffer has {n_steps}; "
                "KNN proprio must align frame-for-frame with actions."
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"obs/{key} contains non-finite values; refusing to build KNN keys.")
        return arr

    def _load_knn_proprio_array(self, key: str):
        key = str(key)
        if key in ("state_joint_pos_current", "state:joint_pos_current", "state/joint_pos_current"):
            state = self._load_hdf5_obs_array("state")
            if state.shape[1] != 200:
                raise ValueError(
                    f"{key} expects Franka OmniReset obs/state dim 200, got {state.shape[1]}."
                )
            start, width = self._STATE_BLOCKS["joint_pos"]
            current_start = start + (self._STATE_HIST - 1) * width
            current_end = current_start + width
            return (
                state[:, current_start:current_end],
                f"obs/state[joint_pos_current:{current_start}:{current_end}]",
            )

        prefix = "state_slice:"
        if key.startswith(prefix):
            try:
                start_s, end_s = key[len(prefix):].split(":", 1)
                start, end = int(start_s), int(end_s)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid knn_proprio_key={key!r}; expected state_slice:<start>:<end>."
                ) from exc
            state = self._load_hdf5_obs_array("state")
            if not (0 <= start < end <= state.shape[1]):
                raise ValueError(
                    f"Invalid {key!r} for obs/state dim {state.shape[1]}."
                )
            return state[:, start:end], f"obs/state[{start}:{end}]"

        return self._load_hdf5_obs_array(key), f"obs/{key}"

    @staticmethod
    def _range_normalize_np(arr: np.ndarray, range_eps: float = 1e-7) -> np.ndarray:
        arr = arr.astype(np.float32, copy=False)
        input_min = arr.min(axis=0)
        input_max = arr.max(axis=0)
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range = input_range.copy()
        input_range[ignore_dim] = 2.0
        scale = 2.0 / input_range
        offset = -1.0 - scale * input_min
        offset[ignore_dim] = -input_min[ignore_dim]
        return (arr * scale + offset).astype(np.float32)

    def _normalize_actions_np(self) -> np.ndarray:
        normalizer = self.get_normalizer()
        actions = np.asarray(self.replay_buffer["action"], dtype=np.float32)
        return normalizer["action"].normalize(actions).detach().cpu().numpy().astype(np.float32)

    def _build_resnet_image_keys(self, current_indices: np.ndarray) -> np.ndarray:
        from diffusion_policy.model.vision.model_getter import get_resnet

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = get_resnet("resnet18", weights=self.knn_resnet_weights).to(device).eval()
        for param in net.parameters():
            param.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        all_cam_feats = []
        batch_size = max(1, self.knn_resnet_batch_size)

        with torch.no_grad():
            for key in self.knn_rgb_keys:
                feats = []
                arr = self.replay_buffer[key]
                for start in range(0, len(current_indices), batch_size):
                    idx = current_indices[start : start + batch_size]
                    imgs = np.asarray(arr[idx], dtype=np.float32) / 255.0  # B,H,W,C
                    tensor = torch.from_numpy(np.moveaxis(imgs, -1, 1)).to(device)
                    tensor = (tensor - mean) / std
                    feat = F.normalize(net(tensor), dim=1)
                    feats.append(feat.cpu().numpy())
                all_cam_feats.append(np.concatenate(feats, axis=0).astype(np.float32))

        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

        keys = np.concatenate(all_cam_feats, axis=1).astype(np.float32)
        norm = np.linalg.norm(keys, axis=1, keepdims=True)
        keys = keys / np.maximum(norm, 1e-6)
        print(
            f"[MARS-KNN] image block=resnet rgb_keys={self.knn_rgb_keys} "
            f"weights={self.knn_resnet_weights} dim={keys.shape[1]} unit_norm=True"
        )
        return keys.astype(np.float32)

    def _build_proprio_keys(self, current_indices: np.ndarray) -> Optional[np.ndarray]:
        key = self.knn_proprio_key
        if key is None or str(key).lower() in ("", "none", "null") or self.knn_proprio_weight <= 0:
            print("[MARS-KNN] proprio block disabled")
            return None

        proprio, desc = self._load_knn_proprio_array(str(key))
        proprio_norm = self._range_normalize_np(proprio)
        cur_prop = proprio_norm[current_indices].astype(np.float32)
        proprio_dim = cur_prop.shape[1]
        cur_prop = cur_prop / np.sqrt(max(proprio_dim, 1)) * self.knn_proprio_weight
        self._knn_proprio_desc = desc
        print(
            f"[MARS-KNN] proprio block={desc} dim={proprio_dim} "
            f"weight={self.knn_proprio_weight:g} scaled_norm_mean="
            f"{np.linalg.norm(cur_prop, axis=1).mean():.4f}"
        )
        return cur_prop.astype(np.float32)

    def _build_knn_keys(self, current_indices: np.ndarray) -> np.ndarray:
        image_keys = self._build_resnet_image_keys(current_indices)
        proprio_keys = self._build_proprio_keys(current_indices)
        if proprio_keys is None:
            keys = image_keys
            key_desc = "resnet_image"
        else:
            keys = np.concatenate([image_keys, proprio_keys], axis=1).astype(np.float32)
            key_desc = f"resnet_image+proprio[{getattr(self, '_knn_proprio_desc', self.knn_proprio_key)}]"
        self._knn_key_desc = key_desc
        print(f"[MARS-KNN] key source={key_desc} dim={keys.shape[1]} metric=euclidean")
        return keys

    def _precompute_knn(self):
        n_samples = len(self.sampler)
        if n_samples <= 1:
            raise ValueError("Need at least two samples to compute MARS* KNN diversity fields.")

        actions_norm = self._normalize_actions_np()
        action_dim = actions_norm.shape[-1]
        n_action_steps = int(self.source_action_steps)
        future_start = max(int(self.n_obs_steps or 1) - 1, 0)

        history_actions_norm = np.empty((n_samples, n_action_steps, action_dim), dtype=np.float32)
        future_chunks = np.empty((n_samples, n_action_steps, action_dim), dtype=np.float32)
        for idx in range(n_samples):
            history_actions_norm[idx] = actions_norm[self._past_action_indices_for_idx(idx)]
            future_chunks[idx] = actions_norm[
                self._sequence_action_indices_for_idx(idx, future_start, n_action_steps)
            ]

        current_indices = self._current_frame_indices()
        self._knn_current_indices = current_indices
        knn_keys = self._build_knn_keys(current_indices)
        k = min(self.knn_k, n_samples - 1)
        print(f"[MARS-KNN] Exact torch Euclidean topk KNN on {n_samples} samples, k={k} ...", flush=True)
        search_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keys_t = torch.from_numpy(knn_keys).to(search_device)
        key_norm2 = (keys_t * keys_t).sum(dim=1).view(1, -1)
        neighbor_ids = np.empty((n_samples, k), dtype=np.int64)
        nn_dists = np.empty((n_samples, k), dtype=np.float32)
        query_bs = max(1, self.knn_search_batch_size)
        with torch.no_grad():
            all_ids = torch.arange(n_samples, device=search_device)
            for start in range(0, n_samples, query_bs):
                end = min(start + query_bs, n_samples)
                query = keys_t[start:end]
                dist2 = (query * query).sum(dim=1).view(-1, 1) + key_norm2 - 2.0 * (query @ keys_t.T)
                dist2 = torch.clamp(dist2, min=0.0)
                dist2[torch.arange(end - start, device=search_device), all_ids[start:end]] = float("inf")
                vals, ids = torch.topk(dist2, k=k, dim=1, largest=False, sorted=False)
                dists = torch.sqrt(vals)
                neighbor_ids[start:end] = ids.cpu().numpy()
                nn_dists[start:end] = dists.cpu().numpy()
                if start == 0 or end == n_samples or (end // query_bs) % 10 == 0:
                    print(f"[MARS-KNN] searched {end}/{n_samples}", flush=True)
                del query, dist2, vals, ids, dists
        del keys_t, key_norm2
        if search_device.type == "cuda":
            torch.cuda.empty_cache()

        self.per_sample_neighbor_ids = neighbor_ids
        self.per_sample_knn_distances = nn_dists
        if self.knn_weight_temp is None:
            nn_weights = np.full((n_samples, k), 1.0 / max(k, 1), dtype=np.float32)
        else:
            row_scale = nn_dists.mean(axis=1, keepdims=True) + 1e-9
            logits = -nn_dists / (float(self.knn_weight_temp) * row_scale)
            logits -= logits.max(axis=1, keepdims=True)
            nn_weights = np.exp(logits)
            nn_weights /= nn_weights.sum(axis=1, keepdims=True)
            nn_weights = nn_weights.astype(np.float32)

        target_spread = np.zeros((n_samples, action_dim), dtype=np.float32)
        for idx in range(n_samples):
            diff = np.abs(future_chunks[idx][None, :, :] - future_chunks[neighbor_ids[idx]])
            per_neighbor = diff.mean(axis=1)
            target_spread[idx] = (nn_weights[idx, :, None] * per_neighbor).sum(axis=0)

        self.per_sample_target_spread = target_spread
        self.per_sample_nn_histories = history_actions_norm[neighbor_ids]
        self.per_sample_nn_weights = nn_weights
        start_w0_spread = np.abs(
            history_actions_norm[:, None, :, :] - self.per_sample_nn_histories
        ).mean(axis=2)
        start_w0_spread = (nn_weights[:, :, None] * start_w0_spread).sum(axis=1)
        w0_gap = target_spread - start_w0_spread
        w0_relu = np.maximum(w0_gap, 0.0)
        print(f"[MARS-KNN] target_spread mean per dim: {np.round(target_spread.mean(axis=0), 4).tolist()}")
        print(
            f"[MARS-KNN] target_spread overall mean={target_spread.mean():.4f}, "
            f"max={target_spread.max():.4f}"
        )
        print(
            f"[MARS-KNN] w=0 diversity lower-bound relu_mean={w0_relu.mean():.4f}, "
            f"active_dim_frac={(w0_gap > 0).mean():.4f}, "
            f"active_sample_frac={(w0_gap.max(axis=1) > 0).mean():.4f}"
        )
        if self.knn_debug_samples > 0:
            probe_count = min(self.knn_debug_samples, n_samples)
            print(self.debug_alignment_summary(tuple(range(probe_count))))

    def debug_alignment_summary(self, sample_indices=(0, 1, 2)) -> str:
        lines = [
            f"[MARS-KNN] alignment probe: samples={list(sample_indices)} "
            f"n_obs_steps={self.n_obs_steps} source_action_steps={self.source_action_steps}"
        ]
        future_start = max(int(self.n_obs_steps or 1) - 1, 0)
        for idx in sample_indices:
            idx = int(idx)
            past_idx = self._past_action_indices_for_idx(idx)
            future_idx = self._sequence_action_indices_for_idx(idx, future_start, self.source_action_steps)
            current_idx = self._sequence_action_indices_for_idx(idx, future_start, 1)[0]
            nn_ids = getattr(self, "per_sample_neighbor_ids", None)
            nn_preview = [] if nn_ids is None else nn_ids[idx, : min(5, nn_ids.shape[1])].tolist()
            lines.append(
                f"  sample={idx} sampler={self.sampler.indices[idx].tolist()} current={int(current_idx)} "
                f"past={past_idx.tolist()} future={future_idx.tolist()} "
                f"target_spread_mean={float(self.per_sample_target_spread[idx].mean()):.4f} "
                f"nn_head={nn_preview}"
            )
        return "\n".join(lines)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.train_mask = self.val_mask
        val_set._precompute_knn()
        return val_set

    def get_holdout_dataset(self):
        holdout_set = copy.copy(self)
        holdout_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.holdout_mask,
        )
        holdout_set.train_mask = self.holdout_mask
        holdout_set._precompute_knn()
        return holdout_set

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = super().__getitem__(idx)
        data["past_action"] = torch.from_numpy(self._past_action_for_idx(idx))
        data["target_spread"] = torch.from_numpy(self.per_sample_target_spread[idx])
        data["nn_histories"] = torch.from_numpy(self.per_sample_nn_histories[idx])
        data["nn_weights"] = torch.from_numpy(self.per_sample_nn_weights[idx])
        return data
