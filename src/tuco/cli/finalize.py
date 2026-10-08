"""Finalize projected-gradient summaries into paper-aligned influence scores."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np

from tuco.config import PAPER_AGGREGATION, PAPER_CURVATURE_RIDGE


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _solve(
    covariance: np.ndarray,
    candidate_sums: np.ndarray,
    target_sums: np.ndarray,
    ridge: float,
    device: str,
) -> np.ndarray:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("tuco-finalize requires PyTorch") from error
    target_device = torch.device(device)
    matrix = torch.as_tensor(covariance, dtype=torch.float32, device=target_device)
    if ridge:
        matrix.diagonal().add_(ridge)
    candidates = torch.as_tensor(
        candidate_sums.T, dtype=torch.float32, device=target_device
    )
    targets = torch.as_tensor(target_sums, dtype=torch.float32, device=target_device)
    solution = torch.linalg.solve(matrix, candidates)
    return (targets @ solution).cpu().numpy().astype(np.float32)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-shards", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    ridge = PAPER_CURVATURE_RIDGE

    shards = [np.load(path, allow_pickle=False) for path in args.candidate_shards]
    try:
        if not all(bool(shard["complete"]) for shard in shards):
            raise ValueError("at least one candidate shard is incomplete")
        if any(
            "aggregation" not in shard
            or str(shard["aggregation"]) != PAPER_AGGREGATION
            for shard in shards
        ):
            raise ValueError("candidate shard aggregation does not match the paper")
        extraction_configs = [str(shard["extraction_config"]) for shard in shards]
        if len(set(extraction_configs)) != 1:
            raise ValueError("candidate shards use different extraction configurations")
        num_shards = {int(shard["num_shards"]) for shard in shards}
        shard_ids = sorted(int(shard["shard"]) for shard in shards)
        if num_shards != {len(shards)} or shard_ids != list(range(len(shards))):
            raise ValueError("candidate shards are not one complete disjoint shard set")
        candidate_ids = np.asarray(shards[0]["state_ids"])
        if any(not np.array_equal(shard["state_ids"], candidate_ids) for shard in shards):
            raise ValueError("candidate shards use different state orderings")
        covariance = np.sum(
            [shard["covariance"].astype(np.float64) for shard in shards], axis=0
        )
        candidate_sums = np.sum(
            [shard["demo_sums"].astype(np.float64) for shard in shards], axis=0
        )
        candidate_counts = np.sum(
            [shard["demo_counts"].astype(np.int64) for shard in shards], axis=0
        )
    finally:
        for shard in shards:
            shard.close()
    if np.any(candidate_counts <= 0):
        raise ValueError("every candidate must contain at least one valid window")

    with np.load(args.target, allow_pickle=False) as target:
        if not bool(target["complete"]):
            raise ValueError("target extraction is incomplete")
        if (
            "aggregation" not in target
            or str(target["aggregation"]) != PAPER_AGGREGATION
        ):
            raise ValueError("target aggregation does not match the paper")
        if (
            "extraction_config" not in target
            or str(target["extraction_config"]) != extraction_configs[0]
        ):
            raise ValueError("candidate and target extraction configurations differ")
        target_sums = target["rollout_sums"].astype(np.float64)
        target_counts = target["rollout_counts"].astype(np.int64)
        if "returns" in target:
            returns = target["returns"].astype(np.float32)
        elif "success" in target:
            returns = np.where(target["success"], 1.0, -1.0).astype(np.float32)
        else:
            raise ValueError("target needs returns or success")
    influence = _solve(
        covariance, candidate_sums, target_sums, ridge, args.device
    )
    _atomic_npz(
        args.output,
        influence=influence,
        returns=returns,
        candidate_ids=candidate_ids,
        candidate_counts=candidate_counts,
        target_counts=target_counts,
        aggregation=np.asarray(PAPER_AGGREGATION),
        ridge=np.float64(ridge),
        extraction_config=np.asarray(extraction_configs[0]),
    )
    print(
        f"influence={influence.shape} aggregation={PAPER_AGGREGATION} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
