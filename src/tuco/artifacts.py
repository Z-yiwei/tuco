"""Read and write TUCO selections and diagnostics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import TucoConfig, PAPER_AGGREGATION
from .selector import TucoResult


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def write_selection(
    output_dir: Path,
    result: TucoResult,
    candidate_ids: np.ndarray,
    config: TucoConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    ids = np.asarray(candidate_ids)
    if ids.ndim != 1 or len(ids) != len(result.utility):
        raise ValueError("candidate_ids do not match the selection result")
    if ids.dtype == object:
        raise ValueError("candidate_ids must use a non-object NumPy dtype")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("candidate_ids must be unique")
    selected_ids = ids[result.selected_indices]
    metadata: dict[str, Any] = {
        "method": "tuco",
        "aggregation": PAPER_AGGREGATION,
        "budget": int(len(result.selected_indices)),
        "num_candidates": int(len(ids)),
        "num_feasible": int(result.feasible_mask.sum()),
        "num_overflow": int(len(result.overflow_indices)),
        "hyperparameters": config.to_dict(),
        "tie_break": "ascending_candidate_index",
    }
    serializable_ids = [
        value.item() if hasattr(value, "item") else value for value in selected_ids
    ]
    _atomic_text(
        output_dir / "selected_ids.json",
        json.dumps(serializable_ids, indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_text(
        output_dir / "metadata.json",
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _atomic_npz(
        output_dir / "selection.npz",
        candidate_ids=ids,
        selected_indices=result.selected_indices,
        selected_ids=selected_ids,
        greedy_order=result.greedy_order,
        overflow_indices=result.overflow_indices,
        feasible_mask=result.feasible_mask,
        utility=result.utility,
        cosine_alignment=result.cosine_alignment,
        quality_weights=result.quality_weights,
        normalized_utility=result.normalized_utility,
        marginal_coverage=result.marginal_coverage,
        normalized_marginal_coverage=result.normalized_marginal_coverage,
        objective_score=result.objective_score,
    )


def load_selection(
    selected_ids_path: Path,
    *,
    expected_budget: Optional[int] = None,
    expected_candidates: Optional[int] = None,
) -> np.ndarray:
    """Load selected IDs and validate their method and cardinality."""
    selected_ids_path = Path(selected_ids_path)
    output_dir = selected_ids_path.parent
    metadata_path = output_dir / "metadata.json"
    arrays_path = output_dir / "selection.npz"
    for path in (selected_ids_path, metadata_path, arrays_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("method") != "tuco":
        raise ValueError("selection was not produced by TUCO")
    if metadata.get("aggregation") != PAPER_AGGREGATION:
        raise ValueError("selection aggregation does not match the paper")
    if metadata.get("hyperparameters") != TucoConfig().to_dict():
        raise ValueError("selection hyperparameters do not match the paper")
    budget = metadata.get("budget")
    num_candidates = metadata.get("num_candidates")
    if not isinstance(budget, int) or not isinstance(num_candidates, int):
        raise ValueError("selection cardinalities are missing")
    if not 0 <= budget <= num_candidates:
        raise ValueError("selection cardinalities are invalid")
    if expected_budget is not None and budget != expected_budget:
        raise ValueError("selection budget does not match the experiment")
    if (
        expected_candidates is not None
        and num_candidates != expected_candidates
    ):
        raise ValueError("candidate count does not match the experiment")

    values = json.loads(selected_ids_path.read_text(encoding="utf-8"))
    selected_ids = np.asarray(values)
    if selected_ids.ndim != 1 or len(selected_ids) != budget:
        raise ValueError("selected_ids.json has an invalid length")
    if len(np.unique(selected_ids)) != len(selected_ids):
        raise ValueError("selected_ids.json contains duplicates")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        required = {
            "candidate_ids",
            "selected_indices",
            "selected_ids",
            "utility",
            "feasible_mask",
        }
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(f"selection.npz is missing keys: {sorted(missing)}")
        candidate_ids = arrays["candidate_ids"]
        indices = arrays["selected_indices"]
        if candidate_ids.shape != (num_candidates,):
            raise ValueError("selection.npz candidate count is inconsistent")
        if len(np.unique(candidate_ids)) != num_candidates:
            raise ValueError("selection.npz candidate IDs are not unique")
        if indices.shape != (budget,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("selection.npz selected indices are invalid")
        if np.any((indices < 0) | (indices >= num_candidates)):
            raise ValueError("selection.npz selected index is out of range")
        if len(np.unique(indices)) != budget:
            raise ValueError("selection.npz selected indices contain duplicates")
        if not np.array_equal(candidate_ids[indices], arrays["selected_ids"]):
            raise ValueError("selection.npz IDs do not match selected indices")
        if not np.array_equal(selected_ids, arrays["selected_ids"]):
            raise ValueError("JSON and NPZ selected IDs disagree")
    return selected_ids
