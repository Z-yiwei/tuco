"""Test selector behavior."""

import numpy as np

from tuco.config import TucoConfig
from tuco.selector import select_tuco


def test_tuco_prefers_a_complementary_profile():
    influence = np.array(
        [
            [2.0, 2.0, 0.5],
            [0.5, 0.5, 2.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    result = select_tuco(influence, np.ones(3), budget=2)
    assert result.selected_indices.tolist() == [0, 2]


def test_selection_is_invariant_to_positive_global_scaling():
    generator = np.random.default_rng(7)
    influence = generator.normal(size=(12, 30)).astype(np.float32)
    returns = np.where(np.arange(12) % 3, 1.0, -1.0).astype(np.float32)
    config = TucoConfig(tau=0.01, lambda_cov=0.2, rho=0.1)
    first = select_tuco(influence, returns, 10, config)
    second = select_tuco(influence * 1e-5, returns, 10, config)
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices)


def test_empty_cone_falls_back_to_raw_utility():
    influence = np.array(
        [[1.0, -1.0, 2.0], [-1.0, 1.0, -2.0]], dtype=np.float32
    )
    returns = np.array([1.0, 1.0], dtype=np.float32)
    result = select_tuco(
        influence, returns, 2, TucoConfig(tau=0.99)
    )
    expected = np.lexsort((np.arange(3), -(influence.T @ returns)))[:2]
    np.testing.assert_array_equal(result.selected_indices, expected)
    assert len(result.greedy_order) == 0


def test_budget_is_exact_and_unique():
    generator = np.random.default_rng(11)
    influence = generator.normal(size=(8, 25)).astype(np.float32)
    returns = np.ones(8, dtype=np.float32)
    result = select_tuco(influence, returns, 20)
    assert len(result.selected_indices) == 20
    assert len(np.unique(result.selected_indices)) == 20


def test_large_budget_exhausts_cone_then_uses_raw_utility():
    influence = np.array(
        [[2.0, 1.0, -1.0, 0.5], [2.0, 1.0, 1.0, -2.0]], dtype=np.float32
    )
    returns = np.ones(2, dtype=np.float32)
    result = select_tuco(
        influence, returns, budget=4, config=TucoConfig(tau=0.9)
    )
    feasible = np.flatnonzero(result.feasible_mask)
    assert set(result.greedy_order.tolist()) == set(feasible.tolist())
    remaining = np.setdiff1d(np.arange(4), feasible)
    expected_overflow = remaining[
        np.lexsort((remaining, -result.utility[remaining]))
    ]
    np.testing.assert_array_equal(result.overflow_indices, expected_overflow)
