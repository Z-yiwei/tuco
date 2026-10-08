"""Test artifacts behavior."""

import json

import numpy as np
import pytest

from tuco.artifacts import load_selection, write_selection
from tuco.config import TucoConfig
from tuco.selector import select_tuco


def _artifact(tmp_path):
    result = select_tuco(
        np.eye(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
        budget=2,
    )
    output = tmp_path / "selection"
    write_selection(
        output,
        result,
        np.arange(3, dtype=np.int64),
        TucoConfig(),
    )
    return output


def test_round_trip_verifies_fixed_paper_contract(tmp_path):
    output = _artifact(tmp_path)
    selected = load_selection(
        output / "selected_ids.json",
        expected_budget=2,
        expected_candidates=3,
    )
    assert selected.shape == (2,)


def test_historical_aggregation_is_rejected(tmp_path):
    output = _artifact(tmp_path)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["aggregation"] = "target_mean_candidate_mean"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="aggregation"):
        load_selection(output / "selected_ids.json")


def test_tampered_selected_ids_are_rejected(tmp_path):
    output = _artifact(tmp_path)
    (output / "selected_ids.json").write_text("[2, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        load_selection(output / "selected_ids.json")
