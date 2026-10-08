#!/usr/bin/env python3
"""Train the paper-facing CupCake TUCO Sim-to-Real policy from scratch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from tuco.baseline_artifacts import load_curated_selection

from .dataset import EXPECTED_REAL_DECISIONS, _load_real_rollouts
from .workspace import CupCakeCotrainWorkspace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physical-states", type=int, required=True)
    parser.add_argument("--selected-states", type=int, required=True)
    parser.add_argument("--visual-repeats", type=int, required=True)
    parser.add_argument("--real-ratio", type=float, required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--terminal-epoch", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-train-steps", type=int, required=True)
    args = parser.parse_args()

    if args.run_dir.exists():
        parser.error(f"refusing to overwrite an existing run: {args.run_dir}")
    if not torch.cuda.is_available():
        parser.error("formal training requires CUDA")
    load_curated_selection(
        args.selection,
        expected_budget=args.selected_states,
        expected_candidates=args.physical_states,
    )
    if (args.physical_states, args.selected_states, args.visual_repeats) != (
        1200, 600, 5
    ):
        parser.error("released CupCake dataset contract is 1200 states, 600 selected, 5 repeats")
    if args.num_epochs != args.terminal_epoch + 1:
        parser.error("num-epochs must equal terminal-epoch + 1")
    method = json.loads(
        (args.selection.parent / "metadata.json").read_text(encoding="utf-8")
    )["method"]
    real = _load_real_rollouts(args.real_root)
    if len(real["action"]) != EXPECTED_REAL_DECISIONS:
        raise RuntimeError("CupCake real-data validation failed")

    cfg = OmegaConf.load(args.base_config)
    if not cfg.task.abs_action:
        raise ValueError("CupCake base config must use absolute joint targets")
    if cfg.task.dataset.joint_action_representation != (
        "absolute_joint_target_binary_width_v1"
    ):
        raise ValueError("CupCake base config has the wrong action representation")
    if (cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps) != (16, 2, 8):
        raise ValueError("CupCake requires horizon/observation/action steps 16/2/8")
    # Exclude machine-specific data-production metadata from the resolved
    # experiment config.
    for key in ("dataset_contract", "run_contract"):
        if key in cfg:
            del cfg[key]

    # Keep only the public dataset interface from the supplied base config.
    dataset_keys = (
        "shape_meta",
        "dataset_path",
        "cache_path",
        "horizon",
        "pad_before",
        "pad_after",
        "n_obs_steps",
        "seed",
        "val_ratio",
        "joint_action_representation",
        "allow_diagnostic",
    )
    dataset_config = {
        key: cfg.task.dataset[key] for key in dataset_keys if key in cfg.task.dataset
    }
    cfg.task.dataset = OmegaConf.create(dataset_config)
    cfg.task.dataset._target_ = "cupcake.dataset.CuratedCupCakeMixedDataset"
    cfg.task.dataset.selected_state_ids_path = str(args.selection.resolve())
    cfg.task.dataset.real_root = str(args.real_root.resolve())
    cfg.task.dataset.real_ratio = args.real_ratio
    cfg.task.dataset.domain_seed = args.seed
    cfg.name = cfg.task_name = cfg.exp_name = f"cupcake_{method}_real9"
    cfg.task.name = cfg.logging.name = cfg.name
    cfg.logging.tags = [
        "cupcake",
        method,
        "real9",
        "absolute_q",
        "selected600",
        "ratio0p20",
        "nomask",
        "scratch",
    ]
    cfg.curation_method = method
    # The upstream workspace indexes epochs from zero and stops before
    # ``num_epochs``. Use 151 iterations so the released terminal checkpoint
    # is genuinely the model after epoch 150, matching Peg and StackCube.
    cfg.training.num_epochs = args.num_epochs
    cfg.training.resume = False
    cfg.training.device = args.device
    cfg.training.seed = args.seed
    cfg.training.debug = False
    cfg.training.max_train_steps = args.max_train_steps
    cfg.dataloader.batch_size = args.batch_size
    cfg.val_dataloader.batch_size = args.batch_size
    cfg.dataloader.num_workers = 4
    cfg.val_dataloader.num_workers = 2
    cfg.checkpoint.topk.k = 0
    cfg.multi_run.run_dir = str(args.run_dir.resolve())
    cfg._target_ = "cupcake.workspace.CupCakeCotrainWorkspace"

    # Resolve and validate the complete mixed dataset before creating a run
    # directory. A malformed input therefore cannot leave a partial run that
    # looks launchable to a later job.
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    expected_episodes = args.selected_states * args.visual_repeats
    if int(dataset.train_mask.sum() + dataset.val_mask.sum()) != expected_episodes:
        raise RuntimeError(
            f"resolved CupCake dataset is not {args.selected_states} states x "
            f"{args.visual_repeats} variants"
        )
    del dataset

    args.run_dir.mkdir(parents=True, exist_ok=False)
    hydra_dir = args.run_dir / ".hydra"
    hydra_dir.mkdir()
    OmegaConf.save(cfg, hydra_dir / "config.yaml", resolve=True)
    workspace = CupCakeCotrainWorkspace(cfg, output_dir=str(args.run_dir.resolve()))
    workspace.run()


if __name__ == "__main__":
    main()
