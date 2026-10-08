import json

import numpy as np
import pytest

from tuco.baseline_artifacts import (
    load_curated_selection,
    ranked_subset,
    write_baseline_selection,
)


def test_ranked_subset_has_deterministic_ties():
    scores = np.asarray([0.5, 1.0, 1.0, -2.0], dtype=np.float32)
    assert ranked_subset(scores, 3).tolist() == [1, 2, 0]


def test_baseline_artifact_roundtrip(tmp_path):
    output = tmp_path / "selection"
    ids = np.asarray([10, 20, 30, 40], dtype=np.int64)
    write_baseline_selection(
        output,
        method="psd",
        candidate_ids=ids,
        selected_indices=np.asarray([2, 0]),
        scores=np.asarray([2.0, 0.0, 3.0, -1.0]),
        metadata={"transform": "DFT"},
    )
    loaded = load_curated_selection(
        output / "selected_ids.json",
        expected_budget=2,
        expected_candidates=4,
        expected_method="psd",
    )
    assert loaded.tolist() == [30, 10]
    assert json.loads((output / "metadata.json").read_text())["method"] == "psd"


def test_artifact_rejects_wrong_method(tmp_path):
    output = tmp_path / "selection"
    write_baseline_selection(
        output,
        method="random",
        candidate_ids=np.arange(3),
        selected_indices=np.asarray([1]),
    )
    with pytest.raises(ValueError, match="method"):
        load_curated_selection(
            output / "selected_ids.json", expected_method="cupid"
        )
