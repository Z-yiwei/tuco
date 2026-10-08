"""Small method scores shared across experiment backends."""

from __future__ import annotations

import numpy as np


def cupid_scores(influence: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """CUPID utility: successful influence minus failed influence."""
    matrix = np.asarray(influence, dtype=np.float32)
    outcome = np.asarray(returns)
    if outcome.dtype == np.bool_:
        outcome = np.where(outcome, 1.0, -1.0)
    outcome = outcome.astype(np.float32)
    if matrix.ndim != 2 or outcome.shape != (matrix.shape[0],):
        raise ValueError("influence and returns have incompatible shapes")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(outcome)):
        raise ValueError("influence and returns must be finite")
    return (matrix.T @ outcome).astype(np.float32)
