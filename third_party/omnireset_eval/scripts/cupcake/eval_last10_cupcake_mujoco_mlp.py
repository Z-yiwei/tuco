#!/usr/bin/env python3
"""Evaluate the last 10 late MLP checkpoints on 50 fixed held-out resets each."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = ROOT / "scripts/sim2sim/franka/eval_cupcake_mujoco_rl.py"


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"(?:mlp_bc_)?step_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--num-checkpoints", type=int, default=10)
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--reset-pool-npz", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episode-start", type=int, default=384)
    parser.add_argument("--rollouts-per-checkpoint", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoints = sorted(
        list(args.checkpoint_dir.resolve().glob("mlp_bc_step_*.pt"))
        + list(args.checkpoint_dir.resolve().glob("step_*.pt")),
        key=checkpoint_step,
    )
    if args.num_checkpoints <= 0:
        raise ValueError("num-checkpoints must be positive")
    if len(checkpoints) < args.num_checkpoints:
        raise RuntimeError(
            f"need at least {args.num_checkpoints} checkpoints, found {len(checkpoints)}"
        )
    checkpoints = checkpoints[-args.num_checkpoints:]
    episodes = list(
        range(args.episode_start, args.episode_start + args.rollouts_per_checkpoint)
    )
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    franka_dir = str(ROOT / "scripts/sim2sim/franka")
    env["PYTHONPATH"] = franka_dir + os.pathsep + env.get("PYTHONPATH", "")

    rows = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        run_dir = out / f"step_{step}"
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            command = [
                sys.executable,
                "-u",
                str(EVAL_SCRIPT),
                "--policy-type",
                "mlp_bc",
                "--source-zarr",
                str(args.source_zarr.resolve()),
                "--reset-pool-npz",
                str(args.reset_pool_npz.resolve()),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(run_dir),
                "--episodes",
                ",".join(map(str, episodes)),
                "--batch-size",
                str(args.batch_size),
                "--steps",
                "160",
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--physics-substeps",
                "16",
                "--friction-combine",
                "physx_average",
                "--cupcake-collision",
                "convex_decomposition",
                "--hand-collision",
                "source_usd",
                "--cupcake-plate-collision",
                "base_cylinder",
                "--finger-collision",
                "mimic",
            ]
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "step": step,
            "checkpoint": str(checkpoint),
            "rollouts": int(summary["episodes_evaluated"]),
            "stable_successes": int(summary["stable_successes"]),
            "stable_relaxed_successes": int(summary["stable_relaxed_successes"]),
            "abnormal_episodes": int(summary["abnormal_episodes"]),
        }
        rows.append(row)
        print(
            f"[last10] step={step} stable={row['stable_successes']}/"
            f"{row['rollouts']} abnormal={row['abnormal_episodes']}",
            flush=True,
        )

    total_rollouts = sum(row["rollouts"] for row in rows)
    total_successes = sum(row["stable_successes"] for row in rows)
    aggregate = {
        "definition": (
            f"{args.rollouts_per_checkpoint} fixed held-out rollouts from each "
            f"of the last {args.num_checkpoints} late checkpoints"
        ),
        "episodes_per_checkpoint": episodes,
        "checkpoints": rows,
        "total_rollouts": total_rollouts,
        "total_stable_successes": total_successes,
        "stable_success_rate": total_successes / total_rollouts,
        "total_stable_relaxed_successes": sum(
            row["stable_relaxed_successes"] for row in rows
        ),
        "total_abnormal_episodes": sum(row["abnormal_episodes"] for row in rows),
    }
    (out / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == "__main__":
    main()
