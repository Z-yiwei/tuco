"""Test aggregation behavior."""

import numpy as np

from tuco.aggregation import aggregate_feature_sums, aggregate_pairwise_samples


def test_pairwise_uses_target_sum_candidate_sum():
    pairwise = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    target = [np.array([0, 1]), np.array([2])]
    candidates = [np.array([0]), np.array([1, 2, 3])]
    result = aggregate_pairwise_samples(pairwise, target, candidates)
    expected = np.array(
        [
            [pairwise[:2, :1].sum(), pairwise[:2, 1:].sum()],
            [pairwise[2:, :1].sum(), pairwise[2:, 1:].sum()],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(result, expected)


def test_feature_aggregation_matches_paper_equation():
    target_sums = np.array([[2.0, 1.0]], dtype=np.float32)
    candidate_sums = np.array([[4.0, 2.0], [2.0, 4.0]], dtype=np.float32)
    result = aggregate_feature_sums(target_sums, candidate_sums, np.eye(2))
    np.testing.assert_allclose(result, [[10.0, 8.0]])
