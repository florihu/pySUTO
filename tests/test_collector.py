"""
test_collector.py
=================
pytest test suite for collector.py

Covers:
    - FeedCollector.select()       (validation, chaining, tuple form)
    - FeedCollector._load_arrays() (sigma=0→NaN convention, shapes, values)
    - FeedCollector.aggregate()    (error path, return type, shape)
    - Combination tests            (FeedCollector → AggregationMath end-to-end)

Run with:
    pytest test/test_collector.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from pysuto.collector import FeedCollector
from pysuto.agg import AggregationResult


# ---------------------------------------------------------------------------
# Lightweight mocks (no real registry files or AccountingStructure needed)
# ---------------------------------------------------------------------------

class _MockRegistry:
    """Dict-backed registry with the interface FeedCollector uses."""

    def __init__(self, feeds: dict):
        # feeds = {feed_id: {"F_root": np.ndarray, "sigma_root": np.ndarray}}
        self._feeds = feeds

    def __contains__(self, fid):
        return fid in self._feeds

    def ids(self):
        return list(self._feeds.keys())

    def load_result(self, fid):
        return self._feeds[fid]


class _MockAccStruct:
    """Minimal accounting structure: ``edge_index`` for length + ``time_range``
    for the time axis. The structural collector lifts 1-D feed F_root to a
    ``(n_edges, n_timesteps)`` grid, so tests use a single timestep here for
    compactness."""

    def __init__(self, n_edges: int, n_timeps: int = 1):
        self.edge_index = list(range(n_edges))
        self.time_range = [str(2000 + i) for i in range(n_timeps)]


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_R = 4   # root edges used across most tests

F_A     = np.array([10.0, 0.0,  0.0, 20.0])   # feed_a covers edges 0 and 3
SIG_A   = np.array([ 2.0, 0.0,  0.0,  4.0])   # 0 means "not covered"

F_B     = np.array([ 0.0, 15.0, 5.0,  0.0])   # feed_b covers edges 1 and 2
SIG_B   = np.array([ 0.0,  3.0, 1.0,  0.0])

F_AB    = np.array([12.0, 18.0, 6.0, 22.0])   # feed_ab covers all four edges
SIG_AB  = np.array([ 3.0,  5.0, 2.0,  6.0])


@pytest.fixture
def registry():
    return _MockRegistry({
        "feed_a":  {"F_root": F_A.copy(),  "sigma_root": SIG_A.copy()},
        "feed_b":  {"F_root": F_B.copy(),  "sigma_root": SIG_B.copy()},
        "feed_ab": {"F_root": F_AB.copy(), "sigma_root": SIG_AB.copy()},
    })


@pytest.fixture
def acc():
    return _MockAccStruct(N_R)


@pytest.fixture
def collector(registry, acc):
    return FeedCollector(registry=registry, acc_struct=acc)


# ===========================================================================
# 1.  select()
# ===========================================================================

class TestFeedCollectorSelect:

    def test_select_single_valid_feed(self, collector):
        collector.select(["feed_a"])
        assert collector._feed_ids == ["feed_a"]

    def test_select_multiple_valid_feeds(self, collector):
        collector.select(["feed_a", "feed_b"])
        assert collector._feed_ids == ["feed_a", "feed_b"]

    def test_select_unknown_feed_raises_key_error(self, collector):
        with pytest.raises(KeyError, match="not found"):
            collector.select(["no_such_feed"])

    def test_select_with_run_id_tuple_resolves_to_feed_id(self, collector):
        # When the run-id is None (no pinning) the bare feed_id is used.
        collector.select([("feed_a", None)])
        assert collector._feed_ids == ["feed_a"]

    def test_select_returns_self_for_chaining(self, collector):
        result = collector.select(["feed_a"])
        assert result is collector

    def test_select_mixed_str_and_tuple(self, collector):
        collector.select([("feed_a", None), "feed_b"])
        assert collector._feed_ids == ["feed_a", "feed_b"]


# ===========================================================================
# 2.  _load_arrays()   (sigma=0 → NaN convention)
# ===========================================================================

class TestFeedCollectorLoadArrays:

    def test_sigma_zero_converted_to_nan(self, collector):
        """sigma_root=0 means 'not covered' and must become NaN in the tensor."""
        collector.select(["feed_a"])
        F, Sigma = collector._load_arrays()

        # edges 1 and 2 have sigma=0 in feed_a → must be NaN at every t
        assert np.isnan(Sigma[0, 1, 0])
        assert np.isnan(Sigma[0, 2, 0])

    def test_f_also_nan_where_not_covered(self, collector):
        """F entry must be NaN wherever sigma was 0."""
        collector.select(["feed_a"])
        F, Sigma = collector._load_arrays()

        assert np.isnan(F[0, 1, 0])
        assert np.isnan(F[0, 2, 0])

    def test_covered_values_loaded_correctly(self, collector):
        """Covered entries should match the stored F_root / sigma_root."""
        collector.select(["feed_a"])
        F, Sigma = collector._load_arrays()

        np.testing.assert_allclose(F[0, 0, 0],     F_A[0],   rtol=1e-12)
        np.testing.assert_allclose(Sigma[0, 0, 0], SIG_A[0], rtol=1e-12)
        np.testing.assert_allclose(F[0, 3, 0],     F_A[3],   rtol=1e-12)
        np.testing.assert_allclose(Sigma[0, 3, 0], SIG_A[3], rtol=1e-12)

    def test_shapes_are_3d(self, collector):
        collector.select(["feed_a", "feed_b"])
        F, Sigma = collector._load_arrays()
        assert F.shape     == (2, N_R, 1)
        assert Sigma.shape == (2, N_R, 1)

    def test_single_feed_shape(self, collector):
        collector.select(["feed_ab"])
        F, Sigma = collector._load_arrays()
        assert F.shape == (1, N_R, 1)


# ===========================================================================
# 3.  aggregate()
# ===========================================================================

class TestFeedCollectorAggregate:

    def test_aggregate_without_select_raises_runtime_error(self, collector):
        with pytest.raises(RuntimeError, match="No feeds selected"):
            collector.aggregate()

    def test_aggregate_returns_aggregation_result(self, collector):
        result = collector.select(["feed_a"]).aggregate()
        assert isinstance(result, AggregationResult)

    def test_aggregate_f_agg_has_correct_shape(self, collector):
        result = collector.select(["feed_a"]).aggregate()
        # 2-D (n_edges, n_timesteps); MockAccStruct has 1 timestep by default.
        assert result.a_agg.shape == (N_R, 1)

    def test_aggregate_feed_ids_stored(self, collector):
        result = collector.select(["feed_a", "feed_b"]).aggregate()
        assert result.feed_ids == ["feed_a", "feed_b"]


# ===========================================================================
# 4.  Combination tests  (FeedCollector → AggregationMath → AggregationResult)
# ===========================================================================

class TestFeedCollectorCombination:

    def test_single_feed_end_to_end(self, collector):
        """With one fully-covering feed the result should pass through unchanged."""
        result = collector.select(["feed_ab"]).aggregate()

        # Every cell is covered (4 edges × 1 timestep).
        assert result.n_covered == N_R
        assert result.n_missing == 0

        np.testing.assert_allclose(result.a_agg[:, 0],     F_AB,   rtol=1e-12)
        np.testing.assert_allclose(result.sigma_agg[:, 0], SIG_AB, rtol=1e-12)

    def test_all_zero_sigma_gives_no_coverage(self, collector):
        zero_registry = _MockRegistry({
            "zero_feed": {
                "F_root":     np.array([99.0, 99.0]),
                "sigma_root": np.array([ 0.0,  0.0]),
            }
        })
        fc = FeedCollector(registry=zero_registry, acc_struct=_MockAccStruct(2))
        result = fc.select(["zero_feed"]).aggregate()

        assert result.n_covered == 0
        assert np.all(np.isnan(result.a_agg))

    def test_non_overlapping_feeds_give_full_coverage(self, collector):
        result = collector.select(["feed_a", "feed_b"]).aggregate()
        assert result.n_covered == N_R

    def test_non_overlapping_feeds_passthrough_values(self, collector):
        result = collector.select(["feed_a", "feed_b"]).aggregate()

        np.testing.assert_allclose(result.a_agg[0, 0], F_A[0], rtol=1e-12)
        np.testing.assert_allclose(result.a_agg[3, 0], F_A[3], rtol=1e-12)
        np.testing.assert_allclose(result.a_agg[1, 0], F_B[1], rtol=1e-12)
        np.testing.assert_allclose(result.a_agg[2, 0], F_B[2], rtol=1e-12)

    def test_overlapping_feeds_reduce_sigma(self, collector):
        result = collector.select(["feed_a", "feed_ab"]).aggregate()

        for j in [0, 3]:
            min_sigma = min(SIG_A[j], SIG_AB[j])
            assert result.sigma_agg[j, 0] < min_sigma, (
                f"Edge {j}: sigma_agg={result.sigma_agg[j, 0]:.4f} not < min_sigma={min_sigma:.4f}"
            )

    def test_omega_sums_to_one_on_covered_edges(self, collector):
        result = collector.select(["feed_a", "feed_b"]).aggregate()
        col_sums = result.Omega.sum(axis=0)            # (n_edges, n_timesteps)
        np.testing.assert_allclose(col_sums[result.coverage], 1.0, atol=1e-12)

    def test_n_feeds_in_result_matches_selection(self, collector):
        result = collector.select(["feed_a", "feed_b", "feed_ab"]).aggregate()
        assert result.n_feeds == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
