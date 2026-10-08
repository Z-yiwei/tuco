"""Validated selection artifacts shared by all comparison methods."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


PAPER_BASELINES = (
    "random",
    "deminf",
    "demoscore",
    "success_similarity",
    "cupid",
    "qoq",
    "psd",
    "datamil",
    "faktual",
    "sieve",
    "tarot",
)
PAPER_METHODS = ("tuco",) + PAPER_BASELINES


def stable_descending(scores: np.ndarray) -> np.ndarray:
    """Rank larger scores first and break ties by candidate index."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("scores must be a finite vector")
    return np.lexsort((np.arange(len(values), dtype=np.int64), -values))


def ranked_subset(scores: np.ndarray, budget: int) -> np.ndarray:
    """Return the highest-scoring exact-size subset."""
    values = np.asarray(scores)
    if not 0 <= int(budget) <= len(values):
        raise ValueError("budget must be in [0, num_candidates]")
    return stable_descending(values)[: int(budget)].astype(np.int64)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_baseline_selection(
    output_dir: Path,
    *,
    method: str,
    candidate_ids: np.ndarray,
    selected_indices: np.ndarray,
    scores: Optional[np.ndarray] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write one exact-cardinality baseline selection.

    selected_indices always addresses rows of candidate_ids. For scalar
    methods, scores stores the fixed ranking used at every budget. Set-valued
    methods omit it and record their budget-specific subset.
    """
    method = str(method).lower()
    if method not in PAPER_BASELINES:
        raise ValueError(f"unknown paper baseline: {method}")
    ids = np.asarray(candidate_ids)
    selected = np.asarray(selected_indices, dtype=np.int64)
    if ids.ndim != 1 or ids.dtype == object or len(np.unique(ids)) != len(ids):
        raise ValueError("candidate_ids must be a unique non-object vector")
    if selected.ndim != 1 or len(np.unique(selected)) != len(selected):
        raise ValueError("selected_indices must be a unique vector")
    if np.any((selected < 0) | (selected >= len(ids))):
        raise ValueError("selected_indices contain an out-of-range index")
    arrays: dict[str, np.ndarray] = {
        "candidate_ids": ids,
        "selected_indices": selected,
        "selected_ids": ids[selected],
    }
    if scores is not None:
        values = np.asarray(scores, dtype=np.float32)
        if values.shape != (len(ids),) or not np.all(np.isfinite(values)):
            raise ValueError("scores must be finite and match candidate_ids")
        arrays["scores"] = values

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = {
        "method": method,
        "budget": int(len(selected)),
        "num_candidates": int(len(ids)),
        "selection_kind": "ranked" if scores is not None else "budget_specific",
        "tie_break": "ascending_candidate_index",
    }
    for key, value in (metadata or {}).items():
        if key in record:
            raise ValueError(f"metadata cannot replace reserved key {key!r}")
        record[key] = _json_value(value)
    _atomic_text(
        output_dir / "selected_ids.json",
        json.dumps([_json_value(x) for x in arrays["selected_ids"]],
                   indent=2, ensure_ascii=False) + "\n",
    )
    _atomic_text(
        output_dir / "metadata.json",
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _atomic_npz(output_dir / "selection.npz", **arrays)


def load_curated_selection(
    selected_ids_path: Path,
    *,
    expected_budget: Optional[int] = None,
    expected_candidates: Optional[int] = None,
    expected_method: Optional[str] = None,
) -> np.ndarray:
    """Load either TUCO or a paper-baseline artifact with full validation."""
    selected_ids_path = Path(selected_ids_path)
    root = selected_ids_path.parent
    metadata_path = root / "metadata.json"
    arrays_path = root / "selection.npz"
    for path in (selected_ids_path, metadata_path, arrays_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    method = metadata.get("method")
    if method not in PAPER_METHODS:
        raise ValueError(f"unsupported selection method: {method!r}")
    if expected_method is not None and method != expected_method:
        raise ValueError("selection method does not match the experiment")
    if method == "tuco":
        # Preserve TUCO's stronger aggregation and hyperparameter checks.
        from .artifacts import load_selection

        return load_selection(
            selected_ids_path,
            expected_budget=expected_budget,
            expected_candidates=expected_candidates,
        )
    budget = metadata.get("budget")
    count = metadata.get("num_candidates")
    if not isinstance(budget, int) or not isinstance(count, int):
        raise ValueError("selection cardinalities are missing")
    if not 0 <= budget <= count:
        raise ValueError("selection cardinalities are invalid")
    if expected_budget is not None and budget != expected_budget:
        raise ValueError("selection budget does not match the experiment")
    if expected_candidates is not None and count != expected_candidates:
        raise ValueError("candidate count does not match the experiment")

    values = np.asarray(json.loads(selected_ids_path.read_text(encoding="utf-8")))
    if values.ndim != 1 or len(values) != budget or len(np.unique(values)) != budget:
        raise ValueError("selected_ids.json is not an exact-size unique vector")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        required = {"candidate_ids", "selected_indices", "selected_ids"}
        missing = required.difference(arrays.files)
        if missing:
            raise ValueError(f"selection.npz is missing keys: {sorted(missing)}")
        candidate_ids = arrays["candidate_ids"]
        indices = arrays["selected_indices"]
        selected_ids = arrays["selected_ids"]
        if candidate_ids.shape != (count,) or len(np.unique(candidate_ids)) != count:
            raise ValueError("selection.npz candidate IDs are invalid")
        if indices.shape != (budget,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("selection.npz selected indices are invalid")
        if np.any((indices < 0) | (indices >= count)) or len(np.unique(indices)) != budget:
            raise ValueError("selection.npz selected indices are invalid")
        if not np.array_equal(candidate_ids[indices], selected_ids):
            raise ValueError("selection.npz IDs do not match selected indices")
        if not np.array_equal(values, selected_ids):
            raise ValueError("JSON and NPZ selected IDs disagree")
    return values
