"""Build deterministic MuJoCo resets for ``CupCakeSideLyingFront3cm``.

Robot/root states are sampled from an aligned Isaac t0 panel so the existing
EE-anywhere reset distribution is retained.  Only the CupCake and Plate poses
are replaced by the new reset contract; CupCake velocity is reset to zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zarr


ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = ROOT / "scripts/cupcake"
if str(SPEC_DIR) not in sys.path:
    sys.path.insert(0, str(SPEC_DIR))

from cupcake_side_lying_reset_spec import (  # noqa: E402
    CUPCAKE_POSE_RANGE,
    PLATE_POSE_RANGE,
    RESET_TYPE,
    quat_from_euler_xyz,
)


def _sample(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return float(low if low == high else rng.uniform(low, high))


def build_reset_pool(
    source_zarr: str | Path,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_state[N,57], source_template_episode[N])``."""
    if count <= 0:
        raise ValueError("count must be positive")
    source_zarr = Path(source_zarr).resolve()
    store = zarr.open(str(source_zarr), mode="r")
    ends = np.asarray(store["meta/episode_ends"], dtype=np.int64)
    starts = np.concatenate([np.zeros(1, dtype=np.int64), ends[:-1]])
    templates = np.asarray(store["data/raw_state"][starts], dtype=np.float64)
    if templates.ndim != 2 or templates.shape[1] != 57:
        raise ValueError(f"expected aligned 57-D raw states, got {templates.shape}")

    rng = np.random.default_rng(seed)
    template_ids = rng.integers(0, len(templates), size=count, endpoint=False)
    pool = templates[template_ids].copy()
    for raw in pool:
        cup_rpy = [_sample(rng, CUPCAKE_POSE_RANGE[key]) for key in ("roll", "pitch", "yaw")]
        raw[31:34] = [_sample(rng, CUPCAKE_POSE_RANGE[key]) for key in "xyz"]
        raw[34:38] = quat_from_euler_xyz(*cup_rpy)
        raw[38:44] = 0.0
        raw[44:47] = [_sample(rng, PLATE_POSE_RANGE[key]) for key in "xyz"]
        raw[47:51] = quat_from_euler_xyz(
            *[_sample(rng, PLATE_POSE_RANGE[key]) for key in ("roll", "pitch", "yaw")]
        )
    return np.ascontiguousarray(pool), template_ids.astype(np.int64)


def save_npz(
    output: str | Path,
    pool: np.ndarray,
    template_ids: np.ndarray,
    source_zarr: str | Path,
    seed: int,
) -> Path:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        raw_state=np.asarray(pool, dtype=np.float64),
        source_template_episode=np.asarray(template_ids, dtype=np.int64),
        source_zarr=np.asarray(str(Path(source_zarr).resolve())),
        reset_type=np.asarray(RESET_TYPE),
        seed=np.asarray(seed, dtype=np.int64),
    )
    return output


def load_npz(path: str | Path) -> np.ndarray:
    with np.load(Path(path).resolve()) as payload:
        pool = np.asarray(payload["raw_state"], dtype=np.float64)
        reset_type = str(np.asarray(payload["reset_type"]).item())
    accepted_reset_types = {RESET_TYPE, f"{RESET_TYPE}FixedHome"}
    if reset_type not in accepted_reset_types:
        raise ValueError(
            f"reset type is {reset_type!r}, expected one of {sorted(accepted_reset_types)!r}"
        )
    if pool.ndim != 2 or pool.shape[1] != 57 or not np.isfinite(pool).all():
        raise ValueError(f"invalid reset pool shape/content: {pool.shape}")
    return np.ascontiguousarray(pool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zarr", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pool, template_ids = build_reset_pool(
        args.source_zarr, count=args.count, seed=args.seed
    )
    output = save_npz(
        args.output, pool, template_ids, args.source_zarr, args.seed
    )
    print(f"saved {len(pool)} {RESET_TYPE} resets to {output}")


if __name__ == "__main__":
    main()
