"""
test_agg.py
===========
pytest test suite for agg.py under the structural model.

Covers:
    - AggregationMath.run()  (Lp invariants + numerical example), on the
      (N_f, n_edges, n_timesteps) tensor shape.
    - AggregationResult  (properties, defaults, shape consistency).

The original tests were written against the pre-refactor 2-D ``(N_f, N_r)``
layout. The structural refactor moved aggregation to the 3-D
``(N_f, n_edges, n_timesteps)`` tensor. To keep test bodies compact we
collapse the timestep axis to length 1 by lifting 2-D fixtures via
``[..., np.newaxis]`` and view the results' first column for assertions.

Run with:
    pytest tests/test_agg.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from pysuto.agg import AggregationMath, AggregationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(F, Sigma, feed_ids=None, p: float = 1.3):
    """
    Run aggregation on a single-timestep grid.

    Accepts 2-D ``(N_f, N_r)`` fixtures and lifts them to
    ``(N_f, N_r, 1)`` before delegating. Returns the AggregationResult
    object as-is (2-D ``a_agg``/``sigma_agg`` of shape ``(N_r, 1)``).
    """
    F     = np.asarray(F, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if F.ndim == 2:
        F     = F[..., np.newaxis]
        Sigma = Sigma[..., np.newaxis]
    if feed_ids is None:
        feed_ids = [f"feed_{k}" for k in range(F.shape[0])]
    return AggregationMath.run(F, Sigma, feed_ids, p=p)


def _lp_sigma_agg(Omega: np.ndarray, Sigma: np.ndarray, p: float) -> np.ndarray:
    """Reference Lp σ on a 3-D (N_f, n_edges, n_timesteps) tensor."""
    S = np.where(np.isfinite(Sigma), Sigma, 0.0)
    sums = (np.abs(Omega * S) ** p).sum(axis=0)
    return np.power(np.maximum(sums, 0.0), 1.0 / p)


def _col(arr: np.ndarray) -> np.ndarray:
    """View the first timestep column of a 2-D ``(n_edges, n_timesteps)`` array."""
    return arr[:, 0]


# ===========================================================================
# 1.  AggregationMath
# ===========================================================================

class TestAggregationMath:

    def test_single_feed_passthrough(self):
        F     = np.array([[10.0, 20.0, np.nan]])
        Sigma = np.array([[2.0,  4.0,  np.nan]])
        res   = _run(F, Sigma)

        np.testing.assert_allclose(_col(res.a_agg)[:2],     [10.0, 20.0], rtol=1e-12)
        np.testing.assert_allclose(_col(res.sigma_agg)[:2], [2.0,  4.0],  rtol=1e-12)
        assert np.isnan(res.a_agg[2, 0])
        assert np.isnan(res.sigma_agg[2, 0])

    def test_two_feeds_equal_sigma_averages_f(self):
        F     = np.array([[10.0, 30.0],
                          [20.0, 10.0]])
        Sigma = np.array([[5.0, 5.0],
                          [5.0, 5.0]])
        res   = _run(F, Sigma)

        np.testing.assert_allclose(_col(res.a_agg), [15.0, 20.0], rtol=1e-12)
        np.testing.assert_allclose(res.Omega[..., 0], 0.5 * np.ones((2, 2)), rtol=1e-12)

    def test_lower_sigma_gets_higher_weight(self):
        F     = np.array([[100.0],
                          [200.0]])
        Sigma = np.array([[1.0],
                          [10.0]])
        res   = _run(F, Sigma)

        assert res.Omega[0, 0, 0] > res.Omega[1, 0, 0]
        assert abs(res.a_agg[0, 0] - 100.0) < abs(res.a_agg[0, 0] - 200.0)

    def test_omega_sums_to_one_per_covered_column(self):
        rng = np.random.default_rng(42)
        N_f, N_r = 5, 20
        F     = rng.random((N_f, N_r)) * 100
        Sigma = rng.random((N_f, N_r)) + 0.1   # all covered, no NaN
        res   = _run(F, Sigma)

        col_sums = res.Omega.sum(axis=0)             # (N_r, 1)
        np.testing.assert_allclose(col_sums[res.coverage], 1.0, atol=1e-12)

    def test_sigma_agg_bounded_by_max_per_feed_sigma(self):
        F     = np.array([[10.0, 20.0],
                          [12.0, 22.0]])
        Sigma = np.array([[3.0, 4.0],
                          [5.0, 6.0]])
        res   = _run(F, Sigma)

        max_sigma = np.maximum(Sigma[0], Sigma[1])[:, np.newaxis]
        assert np.all(res.sigma_agg <= max_sigma + 1e-12)

    def test_sigma_agg_p2_recovers_min_invariant(self):
        F     = np.array([[10.0, 20.0], [12.0, 22.0]])
        Sigma = np.array([[3.0, 4.0],   [5.0, 6.0]])
        res   = _run(F, Sigma, p=2.0)

        min_sigma = np.minimum(Sigma[0], Sigma[1])[:, np.newaxis]
        assert np.all(res.sigma_agg <= min_sigma + 1e-12)

    def test_coverage_is_union_of_feeds(self):
        F     = np.array([[10.0, np.nan, np.nan],
                          [np.nan, 20.0, np.nan],
                          [np.nan, np.nan, np.nan]])
        Sigma = np.array([[2.0, np.nan, np.nan],
                          [np.nan, 3.0, np.nan],
                          [np.nan, np.nan, np.nan]])
        res   = _run(F, Sigma)

        np.testing.assert_array_equal(_col(res.coverage), [True, True, False])

    def test_uncovered_edges_are_nan(self):
        F     = np.array([[np.nan], [np.nan]])
        Sigma = np.array([[np.nan], [np.nan]])
        res   = _run(F, Sigma)

        assert np.isnan(res.a_agg[0, 0])
        assert np.isnan(res.sigma_agg[0, 0])
        assert not res.coverage[0, 0]

    def test_full_numerical_example_p2(self):
        F     = np.array([[10.0, 20.0, np.nan],
                          [12.0, np.nan, 30.0]])
        Sigma = np.array([[2.0,  4.0,  np.nan],
                          [3.0,  np.nan, 5.0]])
        res   = _run(F, Sigma, ["a", "b"], p=2.0)

        expected_f     = np.array([138 / 13, 20.0, 30.0])
        expected_sigma = np.array([6 / np.sqrt(13), 4.0, 5.0])
        expected_omega = np.array([[9 / 13, 1.0, 0.0],
                                   [4 / 13, 0.0, 1.0]])

        np.testing.assert_allclose(_col(res.a_agg),     expected_f,     rtol=1e-10)
        np.testing.assert_allclose(_col(res.sigma_agg), expected_sigma, rtol=1e-10)
        np.testing.assert_allclose(res.Omega[..., 0],   expected_omega, atol=1e-12)
        np.testing.assert_array_equal(_col(res.coverage), [True, True, True])
        assert res.feed_ids == ["a", "b"]

    @pytest.mark.parametrize("p", [1.1, 1.3, 1.5, 2.0])
    def test_sigma_agg_matches_lp_reference(self, p):
        F     = np.array([[10.0, 20.0, np.nan],
                          [12.0, np.nan, 30.0]])
        Sigma = np.array([[2.0,  4.0,  np.nan],
                          [3.0,  np.nan, 5.0]])
        res = _run(F, Sigma, p=p)
        expected = _lp_sigma_agg(res.Omega, res.Sigma, p)
        np.testing.assert_allclose(
            res.sigma_agg[res.coverage], expected[res.coverage], rtol=1e-10,
        )
        assert np.all(np.isnan(res.sigma_agg[~res.coverage]))

    def test_single_feed_passthrough_under_lp(self):
        F = np.array([[10.0, np.nan]])
        Sigma = np.array([[2.0, np.nan]])
        for p in [1.1, 1.3, 1.5, 2.0]:
            res = _run(F, Sigma, p=p)
            assert res.sigma_agg[0, 0] == pytest.approx(2.0)


# ===========================================================================
# 2.  AggregationResult properties
# ===========================================================================

class TestAggregationResult:

    @pytest.fixture
    def result(self):
        F     = np.array([[1.0, np.nan, 3.0],
                          [np.nan, 2.0, 4.0]])
        Sigma = np.array([[0.5, np.nan, 1.0],
                          [np.nan, 0.8, 1.2]])
        return _run(F, Sigma, ["x", "y"])

    def test_n_feeds(self, result):
        assert result.n_feeds == 2

    def test_n_edges(self, result):
        assert result.n_edges == 3

    def test_n_covered(self, result):
        # All 3 cells covered (one timestep) → n_covered = 3.
        assert result.n_covered == 3

    def test_n_missing(self, result):
        # n_covered (3) + n_missing must equal total cell count = 3 × 1 = 3.
        assert result.n_missing == 0

    def test_n_covered_plus_missing_equals_total_cells(self, result):
        total = result.n_edges * result.n_timeps
        assert result.n_covered + result.n_missing == total

    def test_imputation_log_defaults_none(self, result):
        assert result.imputation_log is None

    def test_shapes_consistent(self, result):
        N_f, n_edges, n_t = result.n_feeds, result.n_edges, result.n_timeps
        assert result.a_agg.shape     == (n_edges, n_t)
        assert result.sigma_agg.shape == (n_edges, n_t)
        assert result.coverage.shape  == (n_edges, n_t)
        assert result.Omega.shape     == (N_f, n_edges, n_t)
        assert result.F.shape         == (N_f, n_edges, n_t)
        assert result.Sigma.shape     == (N_f, n_edges, n_t)

    def test_partial_coverage(self):
        F     = np.array([[1.0, np.nan]])
        Sigma = np.array([[1.0, np.nan]])
        res   = _run(F, Sigma)
        assert res.n_covered == 1
        # 2 edges × 1 timestep − 1 covered cell = 1 missing cell.
        assert res.n_missing == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
