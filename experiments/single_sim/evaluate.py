#!/usr/bin/env python3
"""Evaluate the final RoboMimic checkpoints without recording videos."""

import argparse
import json
import re
from pathlib import Path

import dill
import hydra
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-checkpoints", type=int, default=10)
    parser.add_argument("--rollouts", type=int, default=50)
    args = parser.parse_args()
    if args.checkpoint:
        checkpoints = [args.checkpoint.resolve()]
    else:
        numbered = []
        for path in (args.run_dir / "checkpoints").glob("*.ckpt"):
            match = re.search(r"epoch[=_](\d+)", path.name)
            if match:
                numbered.append((int(match.group(1)), path.resolve()))
        checkpoints = [p for _, p in sorted(numbered)[-args.num_checkpoints :]]
        if len(checkpoints) != args.num_checkpoints:
            raise ValueError(
                f"Expected {args.num_checkpoints} epoch checkpoints; found {len(checkpoints)}"
            )
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for checkpoint in checkpoints:
        destination = args.output / checkpoint.stem
        destination.mkdir(parents=True, exist_ok=True)
        with checkpoint.open("rb") as stream:
            payload = torch.load(stream, pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        runner_cfg = cfg.task.env_runner
        runner_cfg.n_train = runner_cfg.n_train_vis = runner_cfg.n_test_vis = 0
        runner_cfg.n_test = args.rollouts
        workspace = hydra.utils.get_class(cfg._target_)(
            cfg, output_dir=str(destination)
        )
        workspace.load_payload(payload)
        policy = (
            workspace.ema_model
            if cfg.training.get("use_ema", False)
            else workspace.model
        )
        policy.to(args.device).eval()
        runner = hydra.utils.instantiate(runner_cfg, output_dir=str(destination))
        try:
            result = runner.run(policy)
        finally:
            if hasattr(runner, "env"):
                runner.env.close()
        rows.append(
            {
                "checkpoint": str(checkpoint),
                "sr": float(result["test/mean_score"]),
                "rollouts": args.rollouts,
            }
        )
        (destination / "result.json").write_text(json.dumps(rows[-1], indent=2) + "\n")
        print(rows[-1], flush=True)
        del runner, policy, workspace, payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    scores = [r["sr"] for r in rows]
    summary = {
        "checkpoints": rows,
        "mean_success_rate": float(np.mean(scores)),
        "population_standard_deviation": float(np.std(scores)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
