import json

import numpy as np
import pytest

from tuco.sim2sim.baselines import psd_scores
from tuco.sim2sim.external_baselines import (
    cupid_scores,
    success_similarity_scores,
)
from tuco.cli.select_diffusion_baseline import main as select_diffusion_baseline
from tuco.single_sim.baselines import (
    faktual_select,
    normalized_gram,
    oporp_project,
    tarot_select,
)


def test_cupid_utility_is_signed_rollout_influence():
    influence = np.asarray([[1.0, 3.0], [2.0, -1.0]], dtype=np.float32)
    scores = cupid_scores(influence, np.asarray([True, False]))
    np.testing.assert_allclose(scores, [-1.0, 4.0])


def test_success_similarity_prefers_matching_trajectory():
    target = np.asarray([[0.0], [1.0]], dtype=np.float32)
    candidates = np.asarray([[0.0], [1.0], [8.0], [9.0]], dtype=np.float32)
    scores = success_similarity_scores(
        target,
        np.asarray([2]),
        np.asarray([True]),
        candidates,
        np.asarray([2, 4]),
    )
    assert scores[0] > scores[1]


def test_psd_prefers_lower_total_power():
    state = np.zeros((8, 6), dtype=np.float32)
    state[4:, :3] = np.asarray(
        [[0, 0, 0], [1, -1, 1], [0, 0, 0], [-1, 1, -1]],
        dtype=np.float32,
    )
    scores = psd_scores(state, np.asarray([4, 8]), np.asarray([0, 1, 2]))
    assert scores[0] > scores[1]


def test_faktual_and_tarot_return_exact_subsets():
    features = np.eye(5, dtype=np.float32)
    selected = faktual_select(
        normalized_gram(features), 3, seed=2,
        entropy_fraction=0.5, epsilon=0.1,
    )
    assert len(selected) == len(np.unique(selected)) == 3

    candidate = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    target = np.asarray([[1.0, 0.1], [0.1, 1.0]], dtype=np.float32)
    picked = tarot_select(candidate, target, 2)
    assert len(picked) == len(np.unique(picked)) == 2


def test_oporp_uses_one_shared_deterministic_map():
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    target = source[:2].copy()
    left, right = oporp_project(source, target, output_dim=3, seed=7)
    np.testing.assert_allclose(left[:2], right)
    assert left.shape == (4, 3)


def test_diffusion_baselines_use_method_specific_scores(tmp_path):
    artifact = tmp_path / "features.npz"
    np.savez_compressed(
        artifact,
        candidate_ids=np.asarray([10, 20, 30, 40], dtype=np.int64),
        deminf_scores=np.asarray([4.0, 3.0, 2.0, 1.0]),
        demoscore_scores=np.asarray([1.0, 4.0, 3.0, 2.0]),
        success_similarity_scores=np.asarray([2.0, 1.0, 4.0, 3.0]),
    )
    expected = {
        "deminf": [10, 20],
        "demoscore": [20, 30],
        "success_similarity": [30, 40],
    }

    for method, selected_ids in expected.items():
        output = tmp_path / method
        select_diffusion_baseline([
            "--method", method,
            "--input", str(artifact),
            "--output-dir", str(output),
            "--budget", "2",
        ])
        with (output / "selected_ids.json").open() as handle:
            result = json.load(handle)
        assert result == selected_ids


@pytest.mark.parametrize(
    ("method", "required_field"),
    [
        ("deminf", "deminf_scores"),
        ("demoscore", "demoscore_scores"),
        ("success_similarity", "success_similarity_scores"),
    ],
)
def test_diffusion_baselines_reject_ambiguous_generic_scores(
    tmp_path, method, required_field
):
    artifact = tmp_path / f"{method}.npz"
    np.savez_compressed(
        artifact,
        candidate_ids=np.asarray([0, 1], dtype=np.int64),
        scores=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match=required_field):
        select_diffusion_baseline([
            "--method", method,
            "--input", str(artifact),
            "--output-dir", str(tmp_path / f"output-{method}"),
            "--budget", "1",
        ])
