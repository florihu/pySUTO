"""
test_agg.py
===========
pytest test suite for agg.py

Covers:
    - AggregationMath.run()  (all vectorised invariants + numerical example)
    - AggregationResult  (properties, defaults, shape consistency)

Run with:
    pytest test/test_agg.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from pysuto.agg import AggregationMath, AggregationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(F, Sigma, feed_ids=None, p: float = 1.3):
    F     = np.asarray(F, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    if feed_ids is None:
        feed_ids = [f"feed_{k}" for k in range(F.shape[0])]
    return AggregationMath.run(F, Sigma, feed_ids, p=p)


def _lp_sigma_agg(Omega: np.ndarray, Sigma: np.ndarray, p: float) -> np.ndarray:
    """Reference: σ_agg[j] = ( Σ_k |Ω_kj σ_kj|^p )^(1/p), ignoring NaN σ."""
    S = np.where(np.isfinite(Sigma), Sigma, 0.0)
    sums = (np.abs(Omega * S) ** p).sum(axis=0)
    return np.power(np.maximum(sums, 0.0), 1.0 / p)


# ===========================================================================
# 1.  AggregationMath
# ===========================================================================

class TestAggregationMath:

    def test_single_feed_passthrough(self):
        """One feed: a_agg == F[0] and sigma_agg == Sigma[0] where covered."""
        F     = np.array([[10.0, 20.0, np.nan]])
        Sigma = np.array([[2.0,  4.0,  np.nan]])
        res   = _run(F, Sigma)

        np.testing.assert_allclose(res.a_agg[:2],     [10.0, 20.0], rtol=1e-12)
        np.testing.assert_allclose(res.sigma_agg[:2], [2.0,  4.0],  rtol=1e-12)
        assert np.isnan(res.a_agg[2])
        assert np.isnan(res.sigma_agg[2])

    def test_two_feeds_equal_sigma_averages_f(self):
        """Equal sigmas → each feed gets weight 0.5 → a_agg is arithmetic mean."""
        F     = np.array([[10.0, 30.0],
                          [20.0, 10.0]])
        Sigma = np.array([[5.0, 5.0],
                          [5.0, 5.0]])
        res   = _run(F, Sigma)

        np.testing.assert_allclose(res.a_agg, [15.0, 20.0], rtol=1e-12)
        np.testing.assert_allclose(res.Omega, 0.5 * np.ones((2, 2)), rtol=1e-12)

    def test_lower_sigma_gets_higher_weight(self):
        """Feed with σ=1 should dominate over feed with σ=10."""
        F     = np.array([[100.0],
                          [200.0]])
        Sigma = np.array([[1.0],
                          [10.0]])
        res   = _run(F, Sigma)

        # P = [1, 0.01]  →  Omega = [1/1.01, 0.01/1.01]
        assert res.Omega[0, 0] > res.Omega[1, 0]
        # a_agg should be closer to 100 than to 200
        assert abs(res.a_agg[0] - 100.0) < abs(res.a_agg[0] - 200.0)

    def test_omega_sums_to_one_per_covered_column(self):
        """Key invariant: Omega columns sum to 1 wherever coverage is True."""
        rng   = np.random.default_rng(42)
        N_f, N_r = 5, 20
        F     = rng.random((N_f, N_r)) * 100
        Sigma = rng.random((N_f, N_r)) + 0.1   # all covered, no NaN
        res   = _run(F, Sigma)

        col_sums = res.Omega.sum(axis=0)
        np.testing.assert_allclose(col_sums[res.coverage], 1.0, atol=1e-12)

    def test_sigma_agg_bounded_by_max_per_feed_sigma(self):
        """Lp invariant: σ_agg ≤ max_k σ_k (Lp norm of weighted σ never exceeds the largest)."""
        F     = np.array([[10.0, 20.0],
                          [12.0, 22.0]])
        Sigma = np.array([[3.0, 4.0],
                          [5.0, 6.0]])
        res   = _run(F, Sigma)

        max_sigma = np.maximum(Sigma[0], Sigma[1])
        assert np.all(res.sigma_agg <= max_sigma + 1e-12)

    def test_sigma_agg_p2_recovers_min_invariant(self):
        """At p=2 the legacy 'σ_agg ≤ min_k σ_k' invariant must hold exactly."""
        F     = np.array([[10.0, 20.0], [12.0, 22.0]])
        Sigma = np.array([[3.0, 4.0],   [5.0, 6.0]])
        res   = _run(F, Sigma, p=2.0)

        min_sigma = np.minimum(Sigma[0], Sigma[1])
        assert np.all(res.sigma_agg <= min_sigma + 1e-12)

    def test_coverage_is_union_of_feeds(self):
        """An edge covered by any feed must appear in coverage."""
        F     = np.array([[10.0, np.nan, np.nan],
                          [np.nan, 20.0, np.nan],
                          [np.nan, np.nan, np.nan]])
        Sigma = np.array([[2.0, np.nan, np.nan],
                          [np.nan, 3.0, np.nan],
                          [np.nan, np.nan, np.nan]])
        res   = _run(F, Sigma)

        np.testing.assert_array_equal(res.coverage, [True, True, False])

    def test_uncovered_edges_are_nan(self):
        """Edges with no coverage → a_agg NaN and sigma_agg NaN."""
        F     = np.array([[np.nan], [np.nan]])
        Sigma = np.array([[np.nan], [np.nan]])
        res   = _run(F, Sigma)

        assert np.isnan(res.a_agg[0])
        assert np.isnan(res.sigma_agg[0])
        assert not res.coverage[0]

    def test_full_numerical_example_p2(self):
        """
        Hand-calculated reference at p=2 (legacy quadrature):

        Feed 0: F=[10, 20, NaN], Sigma=[2, 4, NaN]
        Feed 1: F=[12, NaN, 30], Sigma=[3, NaN, 5]

        a_agg / Omega: unchanged from inverse-variance BLUE.
        Edge 0: P=[1/4, 1/9] → Omega=[9/13, 4/13]; a_agg = 138/13;
                σ_agg(p=2) = 1/sqrt(P_sum) = 6/sqrt(13)
        Edge 1 (feed 0 only): a_agg=20, σ_agg=4
        Edge 2 (feed 1 only): a_agg=30, σ_agg=5
        """
        F     = np.array([[10.0, 20.0, np.nan],
                          [12.0, np.nan, 30.0]])
        Sigma = np.array([[2.0,  4.0,  np.nan],
                          [3.0,  np.nan, 5.0]])
        res   = _run(F, Sigma, ["a", "b"], p=2.0)

        expected_f     = np.array([138 / 13, 20.0, 30.0])
        expected_sigma = np.array([6 / np.sqrt(13), 4.0, 5.0])
        expected_omega = np.array([[9 / 13, 1.0, 0.0],
                                   [4 / 13, 0.0, 1.0]])

        np.testing.assert_allclose(res.a_agg,     expected_f,     rtol=1e-10)
        np.testing.assert_allclose(res.sigma_agg, expected_sigma, rtol=1e-10)
        np.testing.assert_allclose(res.Omega,     expected_omega, atol=1e-12)
        np.testing.assert_array_equal(res.coverage, [True, True, True])
        assert res.feed_ids == ["a", "b"]

    @pytest.mark.parametrize("p", [1.1, 1.3, 1.5, 2.0])
    def test_sigma_agg_matches_lp_reference(self, p):
        """σ_agg matches the Lp reference formula across the supported p range."""
        F     = np.array([[10.0, 20.0, np.nan],
                          [12.0, np.nan, 30.0]])
        Sigma = np.array([[2.0,  4.0,  np.nan],
                          [3.0,  np.nan, 5.0]])
        res = _run(F, Sigma, p=p)
        expected = _lp_sigma_agg(res.Omega, Sigma, p)
        # Where uncovered, both should be NaN. Where covered, equal.
        np.testing.assert_allclose(res.sigma_agg[res.coverage],
                                   expected[res.coverage], rtol=1e-10)
        assert np.all(np.isnan(res.sigma_agg[~res.coverage]))

    def test_single_feed_passthrough_under_lp(self):
        """When only one feed covers an edge, σ_agg = that feed's σ for any p."""
        F = np.array([[10.0, np.nan]])
        Sigma = np.array([[2.0, np.nan]])
        for p in [1.1, 1.3, 1.5, 2.0]:
            res = _run(F, Sigma, p=p)
            assert res.sigma_agg[0] == pytest.approx(2.0)


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
        # edges 0, 1, 2 are all covered (each by at least one feed)
        assert result.n_covered == 3

    def test_n_missing(self, result):
        assert result.n_missing == 0

    def test_n_covered_plus_missing_equals_n_edges(self, result):
        assert result.n_covered + result.n_missing == result.n_edges

    def test_imputation_log_defaults_none(self, result):
        assert result.imputation_log is None

    def test_shapes_consistent(self, result):
        N_f, N_r = result.n_feeds, result.n_edges
        assert result.a_agg.shape     == (N_r,)
        assert result.sigma_agg.shape == (N_r,)
        assert result.coverage.shape  == (N_r,)
        assert result.Omega.shape     == (N_f, N_r)
        assert result.F.shape         == (N_f, N_r)
        assert result.Sigma.shape     == (N_f, N_r)

    def test_partial_coverage(self):
        """When one edge is completely uncovered, n_missing == 1."""
        F     = np.array([[1.0, np.nan]])
        Sigma = np.array([[1.0, np.nan]])
        res   = _run(F, Sigma)
        assert res.n_covered == 1
        assert res.n_missing == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
