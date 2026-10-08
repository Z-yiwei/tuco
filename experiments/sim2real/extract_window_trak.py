#!/usr/bin/env python3
"""Extract window-level projected gradients for Vision-DP TUCO.

The candidate side uses one fixed visual repeat for each of the 1,200
physical reset states. The other repeats are excluded from attribution and
remain available only for final policy training. Every valid Diffusion-Policy
window in the chosen repeat contributes to the TRAK covariance and its
physical state's summed candidate gradient.

The target side consists of decision-aligned, on-policy real robot rollouts.
Every completed policy decision contributes only actions that were actually
sent. Finalization sums both target and candidate windows, following the
current main paper and appendix definitions.
"""

from __future__ import annotations

import argparse
import dill
import json
import os
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import hydra
import h5py
import numpy as np
import torch
import zarr
from torch.utils.data import default_collate

from tuco.config import PAPER_AGGREGATION


ROOT = Path(__file__).resolve().parents[2]
CUPID = Path(os.environ.get("CUPID_SOURCE", ROOT / "third_party/cupid")).resolve()
TRAK_SOURCE = Path(
    os.environ.get("TRAK_SOURCE", CUPID / "third_party/trak")
).resolve()
sys.path.insert(0, str(CUPID))
sys.path.insert(0, str(TRAK_SOURCE))
sys.path.insert(0, str(TRAK_SOURCE / "fast_jl"))

from diffusion_policy.common.pytorch_util import dict_apply  # noqa: E402
from diffusion_policy.common.sampler import SequenceSampler  # noqa: E402
from diffusion_policy.policy.diffusion_unet_image_policy import (  # noqa: E402
    DiffusionUnetImagePolicy,
)
from trak.projectors import CudaProjector, ProjectionType  # noqa: E402


CAMERA_KEYS = ("front_rgb", "side_rgb", "wrist_rgb")
VISION_PROJECTION_DIM = 4096
DIFFUSION_TIMESTEP_SAMPLES = 64


def _extraction_config(args) -> str:
    value = {
        "aggregation": PAPER_AGGREGATION,
        "state_order": "auto" if args.state_order == "AUTO_SORTED" else "explicit",
        "real_contract": args.real_contract,
        "candidate_contract": args.candidate_contract,
        "expected_rollouts": int(args.expected_rollouts),
        "expected_action_steps": int(args.expected_action_steps),
        "repeat_index": int(args.repeat_index),
        "projection_dim": int(args.projection_dim),
        "diffusion_timestep_samples": int(args.num_timesteps),
        "seed": int(args.seed),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class SafeBlockedOutputCudaProjector:
    """A 4096-D JL projection assembled from safe 512-column CUDA blocks.

    The vendored fast-JL extension is stable for the full 140.8M-wide DP
    gradient with 512 output columns, but its larger-output kernels and its
    input-chunk accumulation path fault after a functorch backward on this
    host.  Independent 512-column Rademacher blocks concatenated along the
    output axis have exactly the same JL distribution as one wider matrix.
    """

    def __init__(
        self,
        grad_dim: int,
        proj_dim: int,
        seed: int,
        device: torch.device,
        max_batch_size: int,
    ):
        if proj_dim % 512:
            raise ValueError("blocked-output projection requires a multiple of 512")
        block_dims = [512] * (proj_dim // 512)
        self.projectors = [
            CudaProjector(
                grad_dim=grad_dim,
                proj_dim=width,
                seed=int(seed + index),
                proj_type=ProjectionType.rademacher,
                device=device,
                max_batch_size=max_batch_size,
            )
            for index, width in enumerate(block_dims)
        ]
        self.block_dims = tuple(block_dims)
        self.proj_dim = int(proj_dim)
        self.device = device

    def project(self, gradients: torch.Tensor, model_id: int) -> torch.Tensor:
        gradients = gradients.contiguous()
        # A monolithic isfinite() materializes a >1 GiB boolean tensor for the
        # canonical 8 x 140.8M full-gradient batch.  Check the same values in
        # bounded chunks so this guard does not become the peak-memory owner.
        flat_values = gradients.view(-1)
        finite_check_chunk = 32 * 1024 * 1024
        for start in range(0, flat_values.numel(), finite_check_chunk):
            if not bool(
                torch.isfinite(
                    flat_values[start : start + finite_check_chunk]
                ).all()
            ):
                raise FloatingPointError(
                    "non-finite full DP gradient before projection"
                )
        del flat_values
        actual_batch = len(gradients)
        if actual_batch != 8:
            raise ValueError("fast-JL input must be pre-padded to exactly 8 rows")
        outputs = []
        for index, projector in enumerate(self.projectors):
            try:
                projected = projector.project(gradients, model_id=model_id)
                torch.cuda.synchronize(self.device)
            except RuntimeError as error:
                raise RuntimeError(
                    f"fast-JL failed in output block {index}/{len(self.block_dims)}"
                ) from error
            outputs.append(projected)
        return torch.cat(outputs, dim=1)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _collate_eight(samples: list[dict]) -> tuple[dict, int]:
    """Collate a real batch and duplicate its tail before a short fast-JL call."""
    actual = len(samples)
    if not 1 <= actual <= 8:
        raise ValueError(f"expected 1..8 samples, got {actual}")
    if actual < 8:
        samples = samples + [samples[-1]] * (8 - actual)
    return default_collate(samples), actual


def _load_policy(checkpoint: Path, device: torch.device):
    """Load the checkpoint workspace without importing unrelated hybrid policies."""
    with checkpoint.open("rb") as stream:
        payload = torch.load(stream, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace_class = hydra.utils.get_class(cfg._target_)
    workspace = workspace_class(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if getattr(cfg.training, "use_ema", False) else workspace.model
    policy.to(device).eval()
    return policy, cfg


def _parameter_names(model: torch.nn.Module) -> list[str]:
    return sorted(
        name
        for name in dict(model.named_parameters())
        if (name.startswith("obs_encoder.") or name.startswith("model."))
        and "dummy" not in name
    )


def _load_cupcake_absolute_decisions(root: Path):
    """Load the prepared nine-rollout CupCake absolute-Q dataset."""

    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "complete": True,
        "schema": "cupcake_absq_decision_io_v1",
        "representation": "absolute_joint_target_binary_width_v1",
        "action_steps": 8,
        "observation_steps": 2,
        "gripper_supervised": True,
        "physical_time_relabeling": False,
        "wait_frames_inserted": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"{manifest_path}: invalid {key!r} contract")
    demos = manifest.get("demos", [])
    if len(demos) != 9:
        raise ValueError(f"CupCake requires exactly 9 real rollouts, got {len(demos)}")

    image_parts = {key: [] for key in CAMERA_KEYS}
    proprio_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    source_names: list[str] = []
    source_files: list[str] = []
    for demo in demos:
        path = (root / demo["file"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as source:
            images = {
                key: np.asarray(source[key], dtype=np.uint8)
                for key in CAMERA_KEYS
            }
            proprio = np.asarray(source["proprio"], dtype=np.float32)
            actions = np.asarray(source["action"], dtype=np.float32)
        count = int(demo["decisions"])
        if actions.shape != (count, 8, 8):
            raise ValueError(f"invalid absolute-Q action shape in {path}")
        if proprio.shape != (count, 2, 8):
            raise ValueError(f"invalid proprio shape in {path}")
        if not np.all(np.isfinite(proprio)) or not np.all(np.isfinite(actions)):
            raise ValueError(f"non-finite real proprio/action in {path}")
        if float(np.max(np.abs(actions[..., :7]))) <= 0.2:
            raise ValueError(f"{path}: actions look like deltas, not absolute Q")
        widths = actions[..., 7]
        if not np.all(np.isclose(widths, 0.0) | np.isclose(widths, 0.08)):
            raise ValueError(f"invalid gripper labels in {path}")
        for key in CAMERA_KEYS:
            if images[key].shape != (count, 2, 84, 84, 3):
                raise ValueError(f"invalid {key} shape in {path}")
            image_parts[key].append(images[key])
        proprio_parts.append(proprio)
        action_parts.append(actions)
        rollout_name = f"trial_{int(demo['trial']):03d}"
        source_names.extend([rollout_name] * count)
        source_files.append(str(path))
    if int(manifest.get("decisions", -1)) != 220:
        raise ValueError("CupCake requires the canonical 220 real decisions")
    if sum(len(part) for part in action_parts) != int(manifest["decisions"]):
        raise ValueError("CupCake manifest decision count is inconsistent")
    return (
        {key: np.concatenate(value) for key, value in image_parts.items()},
        np.concatenate(proprio_parts),
        np.concatenate(action_parts),
        tuple(source_names),
        tuple(source_files),
    )


def _load_delta_q_decisions(root: Path):
    """Load exact Delta-Q pairs produced by completed real policy calls."""
    npz_paths = sorted(root.glob("demo_*/trajectory.npz"))
    h5_paths = sorted(root.glob("*/rollout_sync.h5"))
    if npz_paths and h5_paths:
        raise ValueError(f"{root}: contains both exported NPZ and raw H5 rollouts")
    paths = npz_paths or h5_paths
    if not paths:
        raise FileNotFoundError(f"no decision-aligned real rollouts under {root}")
    image_parts = {key: [] for key in CAMERA_KEYS}
    proprio_parts, action_parts, source_names = [], [], []
    action_steps = None
    for path in paths:
        if path.suffix == ".npz":
            metadata_path = path.parent / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            velocity = metadata.get(
                "joint_max_velocity_rad_s",
                metadata.get("joint_velocity_limit_rad_s"),
            )
            if (
                metadata.get("success") is not True
                or metadata.get("action_label_type") != "sent_delta"
                or not np.isclose(float(metadata.get("control_dt_s", -1)), 0.1)
                or not np.isclose(float(velocity), 0.2)
            ):
                raise ValueError(
                    f"{path}: require successful sent_delta data at 0.1 s / 0.2 rad/s"
                )
            with np.load(path, allow_pickle=False) as source:
                images = {
                    key: np.asarray(source[key], dtype=np.uint8)
                    for key in CAMERA_KEYS
                }
                proprio = np.asarray(source["proprio"], dtype=np.float32)
                actions = np.asarray(source["executed_action"], dtype=np.float32)
        else:
            if not (path.parent / ".operator_success").is_file():
                raise ValueError(f"target rollout is not success-marked: {path}")
            with h5py.File(path, "r") as source:
                capture = source["training_capture"]
                observations = capture["observations"]
                plans = capture["plans"]
                completed = np.flatnonzero(plans["completion_monotonic_ns"][:] > 0)
                input_indices = np.asarray(
                    plans["input_sample_indices"][completed], dtype=np.int64
                )
                proprio = np.asarray(observations["proprio"][:], dtype=np.float32)[
                    input_indices
                ]
                actions = np.asarray(plans["sent_delta"][completed], dtype=np.float32)
                images = {
                    key: np.asarray(observations[key][:], dtype=np.uint8)[input_indices]
                    for key in CAMERA_KEYS
                }
        count = len(actions)
        if actions.ndim != 3 or actions.shape[0] != count or actions.shape[2] != 8:
            raise ValueError(f"invalid action shape in {path}: {actions.shape}")
        if actions.shape[1] not in (1, 2):
            raise ValueError(f"real actions must contain one or two sent steps: {path}")
        if action_steps is None:
            action_steps = int(actions.shape[1])
        elif action_steps != int(actions.shape[1]):
            raise ValueError("all real rollouts must use the same action-step count")
        if proprio.shape != (count, 2, 8):
            raise ValueError(f"invalid decision-aligned shape in {path}")
        if not np.all(np.isfinite(proprio)) or not np.all(np.isfinite(actions)):
            raise ValueError(f"non-finite real proprio/action in {path}")
        if float(np.max(np.abs(actions[..., :7]))) > 0.02001:
            raise ValueError(f"real Delta-Q exceeds 0.02 rad in {path}")
        widths = actions[..., 7]
        if not np.all(np.isclose(widths, 0.0) | np.isclose(widths, 0.08)):
            raise ValueError(f"real gripper labels are not 0/0.08 m in {path}")
        for key in CAMERA_KEYS:
            if images[key].shape != (count, 2, 84, 84, 3):
                raise ValueError(f"invalid {key} shape in {path}: {images[key].shape}")
            image_parts[key].append(images[key])
        proprio_parts.append(proprio)
        action_parts.append(actions)
        source_names.extend([path.parent.name] * count)
    return (
        {key: np.concatenate(value) for key, value in image_parts.items()},
        np.concatenate(proprio_parts),
        np.concatenate(action_parts),
        tuple(source_names),
        tuple(str(path.resolve()) for path in paths),
    )


def _load_real_decisions(root: Path, contract: str):
    if contract == "cupcake_absolute_q":
        return _load_cupcake_absolute_decisions(root)
    if contract == "delta_q":
        return _load_delta_q_decisions(root)
    raise ValueError(f"unknown real-data contract: {contract}")


def _state_ids(path: Path) -> np.ndarray:
    root = zarr.open(str(path), mode="r")
    for key in ("meta/reset_state_indices", "meta/reset_state_ids"):
        if key in root:
            value = np.asarray(root[key], dtype=np.int64)
            if value.shape != (6000,):
                raise ValueError(f"expected 6000 episode state IDs, got {value.shape}")
            return value
    raise KeyError(f"no reset-state IDs in {path}")


def _single_repeat_episodes(
    reset_ids: np.ndarray,
    state_order_path: Path,
    repeat_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if str(state_order_path) == "AUTO_SORTED":
        state_order = np.sort(np.unique(reset_ids)).astype(np.int64)
    else:
        state_order = np.asarray(
            json.loads(state_order_path.read_text(encoding="utf-8")), dtype=np.int64
        )
    if state_order.shape != (1200,) or len(np.unique(state_order)) != 1200:
        raise ValueError("state-order manifest must contain 1200 unique IDs")
    grouped: dict[int, list[int]] = defaultdict(list)
    for episode, state_id in enumerate(reset_ids.tolist()):
        grouped[int(state_id)].append(episode)
    if set(grouped) != set(state_order.tolist()) or any(len(v) != 5 for v in grouped.values()):
        raise ValueError("source data is not the expected 1200 physical states x 5 repeats")
    if not 0 <= repeat_index < 5:
        raise ValueError("repeat-index must be in [0,4]")
    episodes = np.asarray(
        [grouped[int(state_id)][repeat_index] for state_id in state_order],
        dtype=np.int64,
    )
    return state_order, episodes


def _load_candidate_dataset(args, cfg, episode_ids: np.ndarray):
    # Reuse the frozen full-data cache but reconstruct the sampler so it exposes
    # exactly one fixed visual repeat for every requested physical state.
    # This repository's dataset splitter intentionally dispatches on a relative
    # path such as ``data/omnireset/...``.  Passing the equivalent absolute path
    # makes it misread ``data2`` as the dataset family.  Keep the canonical
    # repository-relative spelling and ensure that its symlink resolves to the
    # requested HDF5 before instantiating from the repository working directory.
    source_hdf5 = Path(args.source_hdf5).resolve()
    if args.candidate_contract == "cupcake_absolute_q":
        cfg.task.dataset.dataset_path = str(source_hdf5)
        cfg.task.dataset.cache_path = str(Path(args.source_cache).resolve())
    else:
        relative_hdf5 = (
            Path("data/omnireset/datasets")
            / source_hdf5.parent.name
            / source_hdf5.name
        )
        repository_hdf5 = CUPID / relative_hdf5
        if not repository_hdf5.exists() or repository_hdf5.resolve() != source_hdf5:
            raise FileNotFoundError(
                f"canonical source link {repository_hdf5} does not resolve "
                f"to {source_hdf5}"
            )
        cfg.task.dataset.dataset_path = str(relative_hdf5)
        cfg.task.dataset.cache_path = str(Path(args.source_cache).resolve())
    # Keep the checkpoint's valid temporary split while constructing the
    # dataset.  It is replaced in full by ``episode_ids`` immediately below;
    # setting val_ratio=0 trips this fork's "at least one val episode" logic.
    previous_cwd = Path.cwd()
    try:
        os.chdir(CUPID)
        dataset = hydra.utils.instantiate(cfg.task.dataset)
    finally:
        os.chdir(previous_cwd)
    mask = np.zeros(dataset.replay_buffer.n_episodes, dtype=bool)
    mask[episode_ids] = True
    dataset.sampler = SequenceSampler(
        replay_buffer=dataset.replay_buffer,
        sequence_length=dataset.horizon,
        pad_before=dataset.pad_before,
        pad_after=dataset.pad_after,
        episode_mask=mask,
        key_first_k=dataset.key_first_k,
    )
    dataset.train_mask = mask
    return dataset


class RealDecisionDataset:
    def __init__(
        self,
        root: Path,
        horizon: int,
        n_obs_steps: int,
        expected_action_steps: int,
        expected_rollouts: int,
        real_contract: str,
    ):
        images, proprio, actions, source_names, source_files = _load_real_decisions(
            root, real_contract
        )
        if n_obs_steps != 2:
            raise ValueError("decision-aligned real rollouts require n_obs_steps=2")
        self.images = images
        self.proprio = proprio
        self.actions = actions
        self.source_names = tuple(source_names)
        self.source_files = tuple(source_files)
        self.rollout_names = tuple(sorted(set(self.source_names)))
        if len(self.rollout_names) != len(self.source_files):
            raise ValueError("real rollout source-name/file count mismatch")
        if len(self.rollout_names) != expected_rollouts:
            raise ValueError(
                f"expected {expected_rollouts} real rollouts, "
                f"got {len(self.rollout_names)}"
            )
        if self.actions.shape[1] != expected_action_steps:
            raise ValueError(
                f"expected {expected_action_steps} real action steps per decision, "
                f"got {self.actions.shape[1]}"
            )
        name_to_index = {name: i for i, name in enumerate(self.rollout_names)}
        self.rollout_indices = np.asarray(
            [name_to_index[name] for name in self.source_names], dtype=np.int64
        )
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.real_contract = real_contract

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index: int):
        obs = {
            key: torch.from_numpy(
                # ``moveaxis`` alone leaves channel stride=1.  functorch/vmap
                # plus cuDNN GroupNorm is not safe on that view in this torch
                # build, whereas the training dataset returns contiguous CHW.
                np.ascontiguousarray(
                    np.moveaxis(self.images[key][index], -1, 1),
                    dtype=np.float32,
                )
                / 255.0
            )
            for key in CAMERA_KEYS
        }
        obs["proprio"] = torch.from_numpy(self.proprio[index].astype(np.float32))
        action = np.zeros((self.horizon, 8), dtype=np.float32)
        valid = np.zeros(self.horizon, dtype=np.bool_)
        start = self.n_obs_steps - 1
        action_steps = int(self.actions.shape[1])
        action[start : start + action_steps] = self.actions[index]
        valid[start : start + action_steps] = True
        if self.real_contract == "cupcake_absolute_q":
            action[:start] = self.actions[index, :1]
            action[start + action_steps :] = self.actions[index, -1:]
        return {
            "obs": obs,
            "action": torch.from_numpy(action),
            "action_valid_mask": torch.from_numpy(valid),
        }


class WindowGradientComputer:
    """Functional DP gradient matching CUPID's square model-output protocol."""

    def __init__(self, policy, obs_keys, parameter_names, device):
        self.policy = policy
        self.obs_keys = tuple(obs_keys)
        self.parameters = OrderedDict(
            (name, dict(policy.named_parameters())[name]) for name in parameter_names
        )
        self.buffers = dict(policy.named_buffers())
        self.device = device
        self.grad_dim = sum(value.numel() for value in self.parameters.values())
        self.gradient = torch.func.grad(self._model_output, argnums=0)

    def _model_output(self, weights, buffers, timestep, action, valid_mask, *obs):
        encoder_weights = {
            key[len("obs_encoder.") :]: value
            for key, value in weights.items()
            if key.startswith("obs_encoder.")
        }
        encoder_buffers = {
            key[len("obs_encoder.") :]: value
            for key, value in buffers.items()
            if key.startswith("obs_encoder.")
        }
        model_weights = {
            key[len("model.") :]: value
            for key, value in weights.items()
            if key.startswith("model.")
        }
        model_buffers = {
            key[len("model.") :]: value
            for key, value in buffers.items()
            if key.startswith("model.")
        }
        batch_obs = {key: value.unsqueeze(0) for key, value in zip(self.obs_keys, obs)}
        batch_action = action.unsqueeze(0)
        nobs = self.policy.normalizer.normalize(batch_obs)
        trajectory = self.policy.normalizer["action"].normalize(batch_action)
        batch_size, horizon = trajectory.shape[:2]
        if not self.policy.obs_as_global_cond:
            raise NotImplementedError("this task requires obs_as_global_cond=True")
        encoder_obs = dict_apply(
            nobs,
            lambda value: value[:, : self.policy.n_obs_steps].reshape(
                -1, *value.shape[2:]
            ),
        )
        encoded = torch.func.functional_call(
            self.policy.obs_encoder,
            (encoder_weights, encoder_buffers),
            encoder_obs,
            strict=False,
        )
        global_cond = encoded.reshape(batch_size, -1)
        condition_mask = self.policy.mask_generator(trajectory.shape)
        noise = torch.randn_like(trajectory)
        timestep = timestep.long()
        noisy = self.policy.noise_scheduler.add_noise(trajectory, noise, timestep)
        noisy[condition_mask] = trajectory[condition_mask]
        prediction = torch.func.functional_call(
            self.policy.model,
            (model_weights, model_buffers),
            (noisy, timestep),
            {"local_cond": None, "global_cond": global_cond},
            strict=False,
        )
        # CUPID's image setting uses loss_fn=square.  The real target contains
        # only actions that were actually sent, so exclude padded action slots
        # from both numerator and denominator.
        loss_mask = (~condition_mask) & valid_mask[None, :, None].expand_as(condition_mask)
        squared = prediction.square() * loss_mask.to(prediction.dtype)
        count = loss_mask.sum().clamp_min(1).to(prediction.dtype)
        return squared.sum() / count

    def projected(self, batch, num_timesteps, projector, seed):
        action = batch["action"].to(self.device)
        valid = batch.get("action_valid_mask")
        if valid is None:
            valid = torch.ones(action.shape[:2], dtype=torch.bool)
        valid = valid.to(self.device)
        obs = [batch["obs"][key].to(self.device) for key in self.obs_keys]
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        timesteps = torch.randint(
            self.policy.noise_scheduler.config.num_train_timesteps,
            (len(action), num_timesteps),
            generator=generator,
            device=self.device,
        )
        flat = torch.zeros(
            (len(action), self.grad_dim), dtype=action.dtype, device=self.device
        )
        for timestep_index in range(num_timesteps):
            gradients = torch.func.vmap(
                self.gradient,
                in_dims=(None, None, 0, 0, 0, *([0] * len(obs))),
                randomness="different",
            )(
                self.parameters,
                self.buffers,
                timesteps[:, timestep_index : timestep_index + 1],
                action,
                valid,
                *obs,
            )
            pointer = 0
            for value in gradients.values():
                width = value[0].numel()
                flat[:, pointer : pointer + width] += (
                    value.flatten(start_dim=1).detach() / num_timesteps
                )
                pointer += width
            # Drop the loop variable as well: Python otherwise keeps the final
            # per-parameter gradient alive after the dictionary is deleted.
            del gradients, value
        # fast-JL is a custom extension and shares the CUDA allocator poorly
        # with the large functorch backward graph.  Complete backward work and
        # release cached graph blocks before invoking the projector kernel.
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        projected = projector.project(flat.contiguous(), model_id=0).float()
        torch.cuda.synchronize(self.device)
        del flat
        return projected


def _candidate_mode(args, policy, cfg, computer, projector, output: Path):
    reset_ids = _state_ids(Path(args.source_zarr))
    state_ids, single_repeat = _single_repeat_episodes(
        reset_ids, Path(args.state_order), args.repeat_index
    )
    state_positions = np.array_split(np.arange(1200), args.num_shards)[args.shard]
    episode_ids = single_repeat[state_positions]
    dataset = _load_candidate_dataset(args, cfg, episode_ids)
    episode_ends = np.asarray(dataset.replay_buffer.episode_ends, dtype=np.int64)
    episode_to_state = np.full(len(episode_ends), -1, dtype=np.int64)
    episode_to_state[single_repeat] = np.arange(1200, dtype=np.int64)
    sample_episodes = np.searchsorted(
        episode_ends, dataset.sampler.indices[:, 0], side="right"
    )
    sample_states = episode_to_state[sample_episodes]
    if len(dataset) != len(sample_states) or np.any(sample_states < 0):
        raise RuntimeError("candidate sequence-to-physical-state mapping failed")

    episode_to_slot = np.full(len(episode_ends), -1, dtype=np.int64)
    episode_to_slot[episode_ids] = np.arange(len(episode_ids), dtype=np.int64)
    sample_episode_slots = episode_to_slot[sample_episodes]
    if np.any(sample_episode_slots < 0):
        raise RuntimeError("candidate sequence-to-episode mapping failed")
    range_start = 0
    range_stop = len(dataset)

    covariance = np.zeros((args.projection_dim, args.projection_dim), np.float32)
    demo_sums = np.zeros((1200, args.projection_dim), dtype=np.float32)
    demo_counts = np.zeros(1200, dtype=np.int64)
    episode_counts = np.zeros(len(episode_ids), dtype=np.int64)

    next_index = range_start
    if output.is_file() and args.resume:
        old = np.load(output, allow_pickle=False)
        if (
            "extraction_config" not in old
            or str(old["extraction_config"]) != args.extraction_config
        ):
            raise ValueError("resume shard uses a different extraction configuration")
        if int(old["shard"]) != args.shard or int(old["num_shards"]) != args.num_shards:
            raise ValueError("resume shard has a different sharding contract")
        if bool(old["complete"]):
            print(f"[skip] complete shard={args.shard} output={output}", flush=True)
            return
        if "range_start" in old and int(old["range_start"]) != range_start:
            raise ValueError("resume shard has a different range_start")
        if "range_stop" in old and int(old["range_stop"]) != range_stop:
            raise ValueError("resume shard has a different range_stop")
        covariance = old["covariance"].astype(np.float32)
        if (
            "aggregation" not in old
            or str(old["aggregation"]) != PAPER_AGGREGATION
            or "episode_counts" not in old
        ):
            raise ValueError(
                "legacy partial shard cannot resume under paper aggregation"
            )
        if not np.array_equal(old["episode_ids"], episode_ids):
            raise ValueError("resume shard uses different candidate episodes")
        demo_sums = old["demo_sums"].astype(np.float32)
        demo_counts = old["demo_counts"].astype(np.int64)
        episode_counts = old["episode_counts"].astype(np.int64)
        next_index = int(old["next_index"])
        if not range_start <= next_index <= range_stop:
            raise ValueError(
                f"resume next_index={next_index} is outside [{range_start},{range_stop}]"
            )
        print(f"[resume] shard={args.shard} next_index={next_index}", flush=True)

    stop = (
        range_stop
        if args.max_samples <= 0
        else min(range_stop, range_start + args.max_samples)
    )
    print(
        f"[candidate-range] shard={args.shard}/{args.num_shards} "
        f"range=[{range_start},{range_stop}) next={next_index}",
        flush=True,
    )
    for start in range(next_index, stop, args.batch_size):
        end = min(start + args.batch_size, stop)
        batch, actual = _collate_eight(
            [dataset[index] for index in range(start, end)]
        )
        projected = computer.projected(
            batch,
            args.num_timesteps,
            projector,
            seed=args.seed + args.shard * 1_000_000 + start,
        )
        projected = projected[:actual]
        values = projected.detach().cpu().numpy()
        # Accumulate the same float32 outer product on CPU so the custom
        # projection kernel is not followed by another CUDA allocation.
        covariance += values.T @ values
        for offset, (state_position, episode_slot) in enumerate(
            zip(sample_states[start:end], sample_episode_slots[start:end])
        ):
            demo_sums[state_position] += values[offset]
            demo_counts[state_position] += 1
            episode_counts[episode_slot] += 1
        if end % args.log_every < args.batch_size or end == stop:
            print(
                f"[candidate] shard={args.shard}/{args.num_shards} "
                f"windows={end}/{len(dataset)} demos_seen={np.count_nonzero(demo_counts)}",
                flush=True,
            )
        if end % args.resume_every < args.batch_size and end < stop:
            _atomic_npz(
                output,
                covariance=covariance,
                demo_sums=demo_sums,
                demo_counts=demo_counts,
                episode_counts=episode_counts,
                state_ids=state_ids,
                episode_ids=episode_ids,
                next_index=np.int64(end),
                total_windows=np.int64(len(dataset)),
                complete=np.bool_(False),
                range_start=np.int64(range_start),
                range_stop=np.int64(range_stop),
                aggregation=np.asarray(PAPER_AGGREGATION),
                extraction_config=np.asarray(args.extraction_config),
                shard=np.int64(args.shard),
                num_shards=np.int64(args.num_shards),
                projection_block_dims=np.asarray(projector.block_dims, np.int64),
            )
    complete = stop == range_stop
    if complete and np.any(episode_counts <= 0):
        raise RuntimeError("completed shard contains an empty candidate trajectory")
    _atomic_npz(
        output,
        covariance=covariance,
        demo_sums=demo_sums,
        demo_counts=demo_counts,
        episode_counts=episode_counts,
        state_ids=state_ids,
        episode_ids=episode_ids,
        next_index=np.int64(stop),
        total_windows=np.int64(len(dataset)),
        complete=np.bool_(complete),
        range_start=np.int64(range_start),
        range_stop=np.int64(range_stop),
        aggregation=np.asarray(PAPER_AGGREGATION),
        extraction_config=np.asarray(args.extraction_config),
        shard=np.int64(args.shard),
        num_shards=np.int64(args.num_shards),
        projection_block_dims=np.asarray(projector.block_dims, np.int64),
    )
    print(f"[candidate-complete] shard={args.shard} complete={int(complete)}", flush=True)


def _target_mode(args, policy, cfg, computer, projector, output: Path):
    dataset = RealDecisionDataset(
        Path(args.real_root),
        horizon=int(cfg.horizon),
        n_obs_steps=int(cfg.n_obs_steps),
        expected_action_steps=args.expected_action_steps,
        expected_rollouts=args.expected_rollouts,
        real_contract=args.real_contract,
    )
    rollout_sums = np.zeros(
        (len(dataset.rollout_names), args.projection_dim), np.float32
    )
    rollout_counts = np.zeros(len(dataset.rollout_names), np.int64)
    stop = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    for start in range(0, stop, args.batch_size):
        end = min(start + args.batch_size, stop)
        batch, actual = _collate_eight(
            [dataset[index] for index in range(start, end)]
        )
        projected = computer.projected(
            batch,
            args.num_timesteps,
            projector,
            seed=args.seed + 9_000_000 + start,
        )[:actual].detach().cpu().numpy()
        for offset, rollout_index in enumerate(dataset.rollout_indices[start:end]):
            rollout_sums[rollout_index] += projected[offset]
            rollout_counts[rollout_index] += 1
        if end % args.log_every < args.batch_size or end == stop:
            print(f"[target] decisions={end}/{len(dataset)}", flush=True)
    complete = stop == len(dataset)
    if complete and np.any(rollout_counts <= 0):
        raise RuntimeError("at least one target rollout has no completed decision")
    _atomic_npz(
        output,
        rollout_sums=rollout_sums,
        rollout_counts=rollout_counts,
        rollout_names=np.asarray(dataset.rollout_names),
        source_files=np.asarray(dataset.source_files),
        success=np.ones(len(dataset.rollout_names), dtype=np.bool_),
        complete=np.bool_(complete),
        aggregation=np.asarray(PAPER_AGGREGATION),
        extraction_config=np.asarray(args.extraction_config),
        projection_block_dims=np.asarray(projector.block_dims, np.int64),
    )
    print(
        f"[target-complete] complete={int(complete)} "
        f"rollouts_seen={int(np.count_nonzero(rollout_counts))}/"
        f"{len(dataset.rollout_names)} decisions={int(rollout_counts.sum())} "
        f"output={output}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("candidate", "target"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-hdf5", required=True)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--source-zarr", required=True)
    parser.add_argument("--state-order", required=True)
    parser.add_argument("--real-root", required=True)
    parser.add_argument(
        "--real-contract",
        choices=("delta_q", "cupcake_absolute_q"),
        required=True,
    )
    parser.add_argument(
        "--candidate-contract",
        choices=("repository_relative", "cupcake_absolute_q"),
        required=True,
    )
    parser.add_argument("--expected-rollouts", type=int, choices=(9, 10), required=True)
    parser.add_argument("--expected-action-steps", type=int, choices=(1, 2, 8), required=True)
    parser.add_argument("--repeat-index", type=int, choices=range(5), default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    args.projection_dim = VISION_PROJECTION_DIM
    args.num_timesteps = DIFFUSION_TIMESTEP_SAMPLES
    if not 0 <= args.shard < args.num_shards:
        parser.error("shard must be in [0,num-shards)")
    if args.batch_size != 8:
        parser.error("batch-size must be 8 for this host's fast-JL kernel")
    if args.real_contract == "cupcake_absolute_q":
        if (
            args.candidate_contract != "cupcake_absolute_q"
            or args.expected_rollouts != 9
            or args.expected_action_steps != 8
        ):
            parser.error(
                "CupCake requires cupcake_absolute_q candidates, 9 rollouts, "
                "and 8 action steps"
            )
    elif (
        args.candidate_contract != "repository_relative"
        or args.expected_rollouts != 10
        or args.expected_action_steps not in (1, 2)
    ):
        parser.error(
            "Peg/StackCube require repository-relative candidates, 10 rollouts, "
            "and 1 or 2 action steps"
        )
    args.extraction_config = _extraction_config(args)

    device = torch.device(args.device)
    torch.manual_seed(args.seed + args.shard)
    np.random.seed(args.seed + args.shard)
    policy, cfg = _load_policy(Path(args.checkpoint).resolve(), device)
    if not isinstance(policy, DiffusionUnetImagePolicy):
        raise TypeError(f"expected DiffusionUnetImagePolicy, got {type(policy)}")
    parameter_names = _parameter_names(policy)
    obs_keys = list(cfg.shape_meta.obs.keys())
    computer = WindowGradientComputer(policy, obs_keys, parameter_names, device)
    expected_gradient_dim = 140_790_856
    if computer.grad_dim != expected_gradient_dim:
        raise ValueError(
            f"unexpected gradient dimension {computer.grad_dim} != {expected_gradient_dim}"
        )
    projector = SafeBlockedOutputCudaProjector(
        grad_dim=computer.grad_dim,
        proj_dim=args.projection_dim,
        seed=args.seed,
        device=device,
        max_batch_size=8,
    )
    print(
        f"[projection] dim={args.projection_dim} output_blocks={projector.block_dims}",
        flush=True,
    )
    output = Path(args.output).resolve()
    if args.mode == "candidate":
        _candidate_mode(args, policy, cfg, computer, projector, output)
    else:
        _target_mode(args, policy, cfg, computer, projector, output)


if __name__ == "__main__":
    main()
