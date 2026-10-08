#!/usr/bin/env python3
"""Run the formal late-checkpoint MuJoCo evaluation for an OmniReset task."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from tuco.protocol import load_protocol


ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "third_party" / "omnireset_eval"


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"(?:mlp_bc_)?step_(\d+)\.pt", path.name)
    if match is None:
        raise ValueError(f"cannot parse checkpoint step: {path}")
    return int(match.group(1))


def run(command: list[str], env: dict[str, str]) -> None:
    print("[eval] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=VENDOR, env=env, check=True)


def evaluate_peg(args: argparse.Namespace, checkpoints: list[Path], env: dict[str, str]
                 ) -> list[dict[str, Any]]:
    if args.peg_reset_pt is None:
        raise ValueError("Peg evaluation requires --peg-reset-pt")
    raw = args.output / "raw.json"
    command = [
        sys.executable, "-u", str(VENDOR / "scripts/peg_reversed_ab/mujoco_active_big.py"),
        "--mode", "eval", "--reset_pt", str(args.peg_reset_pt.resolve()),
        "--checkpoints", *map(str, checkpoints),
        "--eval_start", str(args.episode_start),
        "--eval_stop", str(args.episode_start + args.rollouts_per_checkpoint),
        "--num_episodes", str(args.rollouts_per_checkpoint),
        "--workers", str(args.workers), "--result_json", str(raw),
    ]
    run(command, env)
    payloads = json.loads(raw.read_text(encoding="utf-8"))
    rows = []
    for payload in payloads:
        checkpoint = Path(payload["checkpoint"])
        rows.append({
            **payload,
            "reset_ids": [
                int(episode["snapshot_seed"])
                for episode in payload["episodes_detail"]
            ],
            "detail": str(raw),
        })
    return rows


def evaluate_stackcube(args: argparse.Namespace, checkpoints: list[Path],
                       env: dict[str, str]) -> list[dict[str, Any]]:
    if args.stackcube_runtime_zarr is None:
        raise ValueError("StackCube evaluation requires --stackcube-runtime-zarr")
    rows = []
    indices = ",".join(
        str(index) for index in range(
            args.episode_start, args.episode_start + args.rollouts_per_checkpoint
        )
    )
    script = VENDOR / "scripts/stackcube/eval_mujoco_matched_runtime.py"
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        output = args.output / f"step_{step:06d}"
        run([
            sys.executable, "-u", str(script),
            "--runtime_source", str(args.stackcube_runtime_zarr.resolve()),
            "--checkpoint", str(checkpoint), "--out", str(output),
            "--episode_indices", indices, "--horizon", "160",
            "--release_steps", "60", "--stable_window", "10",
            "--device", args.device,
        ], env)
        summary = json.loads(
            (output / "closed_loop_stackcube_summary.json").read_text(encoding="utf-8")
        )
        rows.append({
            "checkpoint": str(checkpoint),
            "successes": int(summary["stable_after_release_successes"]),
            "episodes": int(summary["episodes"]),
            "sr": float(summary["stable_after_release_sr"]),
            "reset_ids": summary["reset_indices"],
            "detail": str(output / "closed_loop_stackcube_summary.json"),
        })
    return rows


def evaluate_cupcake(args: argparse.Namespace, checkpoints: list[Path],
                      env: dict[str, str]) -> list[dict[str, Any]]:
    if args.cupcake_runtime_zarr is None or args.cupcake_reset_pool_npz is None:
        raise ValueError(
            "CupCake evaluation requires --cupcake-runtime-zarr and --cupcake-reset-pool-npz"
        )
    episode_ids = ",".join(
        str(index) for index in range(
            args.episode_start, args.episode_start + args.rollouts_per_checkpoint
        )
    )
    script = VENDOR / "scripts/sim2sim/franka/eval_cupcake_mujoco_rl.py"
    rows = []
    for checkpoint in checkpoints:
        step = checkpoint_step(checkpoint)
        output = args.output / f"step_{step:06d}"
        run([
            sys.executable, "-u", str(script), "--policy-type", "mlp_bc",
            "--source-zarr", str(args.cupcake_runtime_zarr.resolve()),
            "--reset-pool-npz", str(args.cupcake_reset_pool_npz.resolve()),
            "--checkpoint", str(checkpoint), "--out", str(output),
            "--episodes", episode_ids, "--batch-size", str(args.workers),
            "--policy-microbatch-size", "1", "--steps", "160",
            "--device", args.device, "--physics-substeps", "16",
            "--friction-combine", "physx_average",
            "--cupcake-collision", "convex_decomposition",
            "--hand-collision", "source_usd",
            "--cupcake-plate-collision", "base_cylinder",
            "--finger-collision", "mimic",
        ], env)
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        successes = int(summary["stable_successes"])
        episodes = int(summary["episodes_evaluated"])
        rows.append({
            "checkpoint": str(checkpoint),
            "successes": successes,
            "episodes": episodes,
            "sr": successes / episodes,
            "reset_ids": [int(index) for index in summary["episodes"]],
            "detail": str(output / "summary.json"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("peg", "stackcube", "cupcake"), required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-checkpoints", type=int)
    parser.add_argument("--rollouts-per-checkpoint", type=int)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--peg-reset-pt", type=Path)
    parser.add_argument("--stackcube-runtime-zarr", type=Path)
    parser.add_argument("--cupcake-runtime-zarr", type=Path)
    parser.add_argument("--cupcake-reset-pool-npz", type=Path)
    parser.add_argument(
        "--allow-nonformal-window", action="store_true",
        help="Permit an arbitrary checkpoint count/step window for smoke tests",
    )
    args = parser.parse_args()
    protocol = load_protocol("sim2sim", args.task)
    num_checkpoints = args.num_checkpoints or protocol["evaluation"]["late_checkpoints"]
    rollouts = args.rollouts_per_checkpoint or protocol["evaluation"]["rollouts_per_checkpoint"]
    args.rollouts_per_checkpoint = rollouts
    if num_checkpoints <= 0 or rollouts <= 0:
        parser.error("checkpoint and rollout counts must be positive")
    checkpoints = sorted(
        set(args.checkpoint_dir.resolve().glob("step_*.pt"))
        | set(args.checkpoint_dir.resolve().glob("mlp_bc_step_*.pt")),
        key=checkpoint_step,
    )[-num_checkpoints:]
    if len(checkpoints) != num_checkpoints:
        parser.error(f"found {len(checkpoints)} checkpoints, expected {num_checkpoints}")
    steps = [checkpoint_step(path) for path in checkpoints]
    terminal_step = int(protocol["training"]["steps"])
    checkpoint_interval = int(protocol["training"]["checkpoint_every"])
    formal_steps = [
        terminal_step - checkpoint_interval * offset
        for offset in reversed(range(num_checkpoints))
    ]
    if not args.allow_nonformal_window and steps != formal_steps:
        parser.error(f"formal checkpoint window is {formal_steps}, found {steps}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"refusing to reuse non-empty output: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    python_paths = [
        str(VENDOR / "sim_eval"),
        str(VENDOR / "scripts/sim2sim/franka"),
        str(VENDOR / "scripts/stackcube/sim2sim_franka"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = os.pathsep.join(path for path in python_paths if path)
    env.setdefault("MUJOCO_GL", "egl")

    evaluators = {
        "peg": evaluate_peg,
        "stackcube": evaluate_stackcube,
        "cupcake": evaluate_cupcake,
    }
    rows = evaluators[args.task](args, checkpoints, env)
    if len(rows) != num_checkpoints:
        raise RuntimeError(f"evaluator returned {len(rows)} rows for {num_checkpoints} checkpoints")
    reference = rows[0]["reset_ids"]
    for row in rows:
        if row["episodes"] != rollouts:
            raise RuntimeError(f"{row['checkpoint']}: expected {rollouts} rollouts")
        if row["reset_ids"] != reference:
            raise RuntimeError("evaluation reset IDs differ between checkpoints")
    successes = sum(row["successes"] for row in rows)
    total = sum(row["episodes"] for row in rows)
    summary = {
        "task": args.task,
        "eval_contract_id": protocol["evaluation"]["contract"],
        "definition": f"last {num_checkpoints} checkpoints x {rollouts} fixed MuJoCo rollouts",
        "checkpoint_steps": steps,
        "num_checkpoints": num_checkpoints,
        "rollouts_per_checkpoint": rollouts,
        "successes": successes,
        "rollouts": total,
        "sr": successes / total,
        "per_checkpoint": rows,
    }
    path = args.output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[result] {successes}/{total} = {successes / total:.4f} -> {path}")


if __name__ == "__main__":
    main()
