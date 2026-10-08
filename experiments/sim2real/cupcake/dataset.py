"""Curated CupCake simulation data mixed with nine real rollouts."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from tuco.baseline_artifacts import load_curated_selection

from .canonical_absolute_dataset import (
    CAMERAS,
    REP,
    CupCakeAbsoluteGroupedImageDataset,
)


EXPECTED_REAL_ROLLOUTS = 9
EXPECTED_REAL_DECISIONS = 220


def _load_real_rollouts(root: Path) -> dict[str, np.ndarray]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "complete": True,
        "schema": "cupcake_absq_decision_io_v1",
        "representation": REP,
        "action_steps": 8,
        "observation_steps": 2,
        "gripper_supervised": True,
        "physical_time_relabeling": False,
        "wait_frames_inserted": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"CupCake real manifest has invalid {key!r}")
    demos = manifest.get("demos", [])
    if len(demos) != EXPECTED_REAL_ROLLOUTS:
        raise ValueError("CupCake requires exactly nine successful real rollouts")
    if manifest.get("decisions") != EXPECTED_REAL_DECISIONS:
        raise ValueError("CupCake requires the canonical 220 real decisions")

    parts: list[dict[str, np.ndarray]] = []
    total = 0
    for demo in demos:
        path = root / demo["file"]
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as arrays:
            required = (*CAMERAS, "proprio", "action")
            missing = set(required).difference(arrays.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            part = {key: arrays[key].copy() for key in required}
        count = int(demo["decisions"])
        expected_shapes = {
            **{key: (count, 2, 84, 84, 3) for key in CAMERAS},
            "proprio": (count, 2, 8),
            "action": (count, 8, 8),
        }
        for key, shape in expected_shapes.items():
            if part[key].shape != shape:
                raise ValueError(f"{path}:{key} has shape {part[key].shape}, expected {shape}")
        if not np.all(np.isfinite(part["proprio"])) or not np.all(
            np.isfinite(part["action"])
        ):
            raise ValueError(f"{path} contains non-finite proprioception or actions")
        if float(np.max(np.abs(part["action"][..., :7]))) <= 0.2:
            raise ValueError(f"{path} actions look like deltas rather than absolute Q")
        gripper = part["action"][..., 7]
        if not np.all(np.isclose(gripper, 0.0) | np.isclose(gripper, 0.08)):
            raise ValueError(f"{path} contains a non-binary gripper target")
        parts.append(part)
        total += count
    if total != EXPECTED_REAL_DECISIONS:
        raise ValueError("CupCake per-rollout decision counts do not sum to 220")
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def _selected_group_masks(
    state_ids: np.ndarray,
    selected_ids: np.ndarray,
    *,
    seed: int,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.sort(np.asarray(selected_ids, dtype=np.int64))
    if selected.shape != (600,) or len(np.unique(selected)) != 600:
        raise ValueError("CupCake selection must contain 600 unique physical states")
    shuffled = selected.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    validation_count = min(max(1, int(len(shuffled) * val_ratio)), len(shuffled) - 1)
    validation_states = shuffled[:validation_count]
    validation = np.isin(state_ids, validation_states)
    training = np.isin(state_ids, shuffled[validation_count:])
    if np.any(training & validation):
        raise RuntimeError("CupCake grouped train/validation split overlaps")
    return training, validation


class CuratedCupCakeMixedDataset(CupCakeAbsoluteGroupedImageDataset):
    """Selected 600-state simulation set with 20% real-sequence oversampling.

    Selection is over physical reset states. Final training uses all five visual
    variants of each selected state. As in Peg and StackCube, normalization is
    fitted only on the selected simulation training split.
    """

    def __init__(
        self,
        *args,
        selected_state_ids_path: str,
        real_root: str,
        real_ratio: float = 0.2,
        domain_seed: int = 42,
        **kwargs,
    ) -> None:
        if not 0.0 < real_ratio < 1.0:
            raise ValueError("real_ratio must be in (0, 1)")
        split_seed = int(kwargs.get("seed", 42))
        split_ratio = float(kwargs.get("val_ratio", 0.04))
        super().__init__(*args, **kwargs)

        selected = load_curated_selection(
            Path(selected_state_ids_path),
            expected_budget=600,
            expected_candidates=1200,
        ).astype(np.int64)
        state_ids = np.asarray(self.reset_state_ids, dtype=np.int64)
        if state_ids.shape != (6000,):
            raise ValueError("CupCake simulation cache must contain 6,000 episodes")
        counts = np.unique(state_ids, return_counts=True)[1]
        if len(counts) != 1200 or not np.all(counts == 5):
            raise ValueError("CupCake simulation cache must be 1,200 states x 5 variants")

        self.train_mask, self.val_mask = _selected_group_masks(
            state_ids, selected, seed=split_seed, val_ratio=split_ratio
        )
        if int(self.train_mask.sum() + self.val_mask.sum()) != 3000:
            raise RuntimeError("selected CupCake data must contain 600 states x 5 variants")
        self.normalizer_train_mask = self.train_mask.copy()
        self.sampler = self._sampler(self.train_mask)

        self.real = _load_real_rollouts(Path(real_root))
        self.sim_count = len(self.sampler)
        real_count = round(self.sim_count * real_ratio / (1.0 - real_ratio))
        self.real_indices = np.random.default_rng(domain_seed).choice(
            len(self.real["action"]), size=real_count, replace=True
        )
        self.include_real = True

    def __len__(self) -> int:
        if not self.include_real:
            return len(self.sampler)
        return self.sim_count + len(self.real_indices)

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.sampler = result._sampler(result.val_mask)
        result.train_mask = result.val_mask
        result.include_real = False
        return result

    def _real_item(self, index: int) -> dict[str, object]:
        actions = self.real["action"][index]
        padded = np.concatenate(
            [
                actions[:1],
                actions,
                np.repeat(actions[-1:], self.horizon - 9, axis=0),
            ]
        )
        valid = np.zeros(self.horizon, dtype=bool)
        valid[1:9] = True
        observations = {
            key: torch.from_numpy(
                np.moveaxis(self.real[key][index], -1, 1).astype(np.float32) / 255.0
            )
            for key in CAMERAS
        }
        observations["proprio"] = torch.from_numpy(self.real["proprio"][index])
        return {
            "obs": observations,
            "action": torch.from_numpy(padded.astype(np.float32)),
            "action_valid_mask": torch.from_numpy(valid),
            "is_real": torch.tensor(True),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        if not self.include_real or index < self.sim_count:
            item = super().__getitem__(index)
            item["action_valid_mask"] = torch.ones(self.horizon, dtype=torch.bool)
            item["is_real"] = torch.tensor(False)
            return item
        real_index = int(self.real_indices[index - self.sim_count])
        return self._real_item(real_index)


# Backward-compatible import for existing resolved experiment configs.
TucoCupCakeMixedDataset = CuratedCupCakeMixedDataset
