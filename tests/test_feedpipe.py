"""
test_feedpipe.py
================
pytest test suite for feedpipe.py

Covers:
    - Step D  (apply_with_uncertainty via _MockWeightMatrix / UncertaintyMixin)
    - DataFeedMetadata construction and pedigree conversion
    - PropagationResult fields and numerical correctness
    - DataFeedRegistry  register / save / reload round-trip
    - run_feed_pipeline convenience wrapper (steps A–D + register)
    - Edge cases: zero flows, single-edge identity matrix, perfect-quality scores

Run with:
    pytest test_feedpipe.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make sure the project root is importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from pysuto.meta import DataFeedMetadata
from pysuto.unc import UncertaintyMixin, PropagationResult
from pysuto.registry import DataFeedRegistry


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================

def _make_metadata(**kwargs) -> DataFeedMetadata:
    """Return a DataFeedMetadata with sensible defaults, overridable via kwargs."""
    defaults = dict(
        feed_name                = "test_feed",
        iedc_type                = "1_F_",
        reliability              = 1,
        completeness             = 1,
        temporal_correlation     = 1,
        geographical_correlation = 1,
        other_correlation        = 1,
        version                  = 0,
        date_of_assessment       = "2025-01-01",
        person                   = "Tester",
        dataset_link             = "https://example.com",
    )
    defaults.update(kwargs)
    return DataFeedMetadata(**defaults)


class _MockWeightMatrix(UncertaintyMixin):
    """
    Minimal stand-in for WeightMatrix that exercises UncertaintyMixin
    without requiring sparse matrices or AccountingStructure.
    """

    def __init__(self, W_dense: np.ndarray, feed_id: str = "mock", acc_id: str = "acc_test"):
        self.id            = feed_id
        self.acc_used      = acc_id
        self.odir          = ""
        self.weight_matrix = np.asarray(W_dense, dtype=float)

    def save(self, **kwargs):
        import pickle
        path = kwargs.get("path", f"{self.id}.pkl")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"id": self.id, "W": self.weight_matrix}, fh)

    def apply_with_uncertainty(self, F_src=None, pedigree=None, p: float = 1.3):
        F_src = np.asarray(F_src if F_src is not None else self.F_src, dtype=float)
        pg    = pedigree or self.pedigree
        cv    = pg.total_cv()
        W     = self.weight_matrix
        F_root  = W.T @ F_src
        # Lp σ propagation: σ_root[j] = CV · ( Σ_i (|W_ij|·|f_i|)^p )^(1/p)
        Wp      = np.abs(W) ** p
        sum_p   = Wp.T @ (np.abs(F_src) ** p)
        sigma   = cv * np.power(np.maximum(sum_p, 0.0), 1.0 / p)
        with np.errstate(invalid="ignore", divide="ignore"):
            cv_root = np.where(F_root != 0.0, sigma / np.abs(F_root), np.nan)
        return PropagationResult(
            F_root=F_root, sigma_root=sigma, cv_root=cv_root, cv_src=cv
        )


# ---------------------------------------------------------------------------
# Standard 4-src → 3-root weight matrix used across many tests
# ---------------------------------------------------------------------------
W_STD = np.array([
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.3, 0.7],
])
F_SRC_STD = np.array([100.0, 200.0, 150.0, 80.0])


@pytest.fixture
def meta_perfect():
    return _make_metadata()          # all scores = 1  →  lowest CV


@pytest.fixture
def meta_worst():
    return _make_metadata(
        reliability=4, completeness=4,
        temporal_correlation=4, geographical_correlation=4,
        other_correlation=4,
    )


@pytest.fixture
def wm_std():
    return _MockWeightMatrix(W_STD, feed_id="test_feed")


# ===========================================================================
# 1.  DataFeedMetadata
# ===========================================================================

class TestDataFeedMetadata:

    def test_construction_defaults(self, meta_perfect):
        assert meta_perfect.feed_name == "test_feed"
        assert meta_perfect.reliability == 1

    def test_to_pedigree_scores_returns_object(self, meta_perfect):
        pg = meta_perfect.to_pedigree_scores()
        assert pg is not None

    def test_pedigree_total_cv_is_positive_float(self, meta_perfect):
        cv = meta_perfect.to_pedigree_scores().total_cv()
        assert isinstance(float(cv), float)
        assert cv >= 0.0

    def test_higher_scores_give_higher_cv(self, meta_perfect, meta_worst):
        cv_low  = meta_perfect.to_pedigree_scores().total_cv()
        cv_high = meta_worst.to_pedigree_scores().total_cv()
        assert cv_high > cv_low, (
            f"Worst-quality CV ({cv_high}) should exceed best-quality CV ({cv_low})"
        )

    def test_cv_total_stored_on_metadata(self, meta_perfect):
        """cv_total is populated by fill_computed(), not by to_pedigree_scores()."""
        wm = _MockWeightMatrix(W_STD, "cv_check")
        wm.set_pedigree(meta_perfect.to_pedigree_scores())
        wm.set_F_src(F_SRC_STD)
        result = wm.apply_with_uncertainty()
        meta_perfect.fill_computed(result)
        assert meta_perfect.cv_total is not None
        assert meta_perfect.cv_total >= 0.0


# ===========================================================================
# 2.  UncertaintyMixin / _MockWeightMatrix  (Step D internals)
# ===========================================================================

class TestUncertaintyMixin:

    def test_set_pedigree_and_F_src(self, wm_std, meta_perfect):
        pg = meta_perfect.to_pedigree_scores()
        wm_std.set_pedigree(pg)
        wm_std.set_F_src(F_SRC_STD)
        assert hasattr(wm_std, "pedigree")
        assert hasattr(wm_std, "F_src")
        np.testing.assert_array_equal(wm_std.F_src, F_SRC_STD)

    def test_apply_with_uncertainty_output_shapes(self, wm_std, meta_perfect):
        pg = meta_perfect.to_pedigree_scores()
        wm_std.set_pedigree(pg)
        wm_std.set_F_src(F_SRC_STD)
        result = wm_std.apply_with_uncertainty()
        n_root = W_STD.shape[1]
        assert result.F_root.shape    == (n_root,)
        assert result.sigma_root.shape == (n_root,)
        assert result.cv_root.shape    == (n_root,)

    def test_F_root_numerical_correctness(self, wm_std, meta_perfect):
        """F_root must equal W.T @ F_src."""
        pg = meta_perfect.to_pedigree_scores()
        wm_std.set_pedigree(pg)
        wm_std.set_F_src(F_SRC_STD)
        result = wm_std.apply_with_uncertainty()
        expected = W_STD.T @ F_SRC_STD
        np.testing.assert_allclose(result.F_root, expected, rtol=1e-10)

    def test_sigma_root_non_negative(self, wm_std, meta_perfect):
        pg = meta_perfect.to_pedigree_scores()
        wm_std.set_pedigree(pg)
        wm_std.set_F_src(F_SRC_STD)
        result = wm_std.apply_with_uncertainty()
        assert np.all(result.sigma_root >= 0.0)

    def test_cv_root_consistent_with_sigma_and_F(self, wm_std, meta_perfect):
        """cv_root[i] == sigma_root[i] / |F_root[i]| for non-zero flows."""
        pg = meta_perfect.to_pedigree_scores()
        wm_std.set_pedigree(pg)
        wm_std.set_F_src(F_SRC_STD)
        result = wm_std.apply_with_uncertainty()
        nonzero = result.F_root != 0.0
        np.testing.assert_allclose(
            result.cv_root[nonzero],
            result.sigma_root[nonzero] / np.abs(result.F_root[nonzero]),
            rtol=1e-10,
        )

    def test_cv_src_scalar(self, wm_std, meta_perfect):
        pg = meta_perfect.to_pedigree_scores()
        wm_std.set_pedigree(pg)
        wm_std.set_F_src(F_SRC_STD)
        result = wm_std.apply_with_uncertainty()
        assert np.isscalar(result.cv_src) or result.cv_src.ndim == 0

    def test_worse_pedigree_yields_larger_sigma(self, meta_perfect, meta_worst):
        """Higher pedigree scores → larger sigma_root for the same flows."""
        wm_good = _MockWeightMatrix(W_STD, "good")
        wm_bad  = _MockWeightMatrix(W_STD, "bad")
        for wm, meta in [(wm_good, meta_perfect), (wm_bad, meta_worst)]:
            wm.set_pedigree(meta.to_pedigree_scores())
            wm.set_F_src(F_SRC_STD)
        r_good = wm_good.apply_with_uncertainty()
        r_bad  = wm_bad.apply_with_uncertainty()
        assert np.all(r_bad.sigma_root >= r_good.sigma_root)


# ===========================================================================
# 3.  Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_identity_matrix_preserves_flows(self, meta_perfect):
        """1×1 identity: F_root should equal F_src exactly."""
        W  = np.array([[1.0]])
        fs = np.array([42.0])
        wm = _MockWeightMatrix(W, "identity")
        wm.set_pedigree(meta_perfect.to_pedigree_scores())
        wm.set_F_src(fs)
        result = wm.apply_with_uncertainty()
        np.testing.assert_allclose(result.F_root, fs, rtol=1e-12)

    def test_zero_flow_vector(self, meta_perfect):
        """All-zero F_src should give zero F_root and zero sigma_root."""
        wm = _MockWeightMatrix(W_STD, "zero_flow")
        wm.set_pedigree(meta_perfect.to_pedigree_scores())
        wm.set_F_src(np.zeros(4))
        result = wm.apply_with_uncertainty()
        np.testing.assert_array_equal(result.F_root,    np.zeros(3))
        np.testing.assert_array_equal(result.sigma_root, np.zeros(3))

    def test_single_src_to_multiple_root(self, meta_perfect):
        """One source split evenly across two root nodes."""
        W  = np.array([[0.6, 0.4]])   # 1 src, 2 root
        fs = np.array([200.0])
        wm = _MockWeightMatrix(W, "split")
        wm.set_pedigree(meta_perfect.to_pedigree_scores())
        wm.set_F_src(fs)
        result = wm.apply_with_uncertainty()
        np.testing.assert_allclose(result.F_root, [120.0, 80.0], rtol=1e-10)

    def test_large_random_matrix(self, meta_perfect):
        """Smoke test with a random 20×15 weight matrix – no crash, shapes correct."""
        rng = np.random.default_rng(0)
        W   = rng.random((20, 15))
        fs  = rng.random(20) * 1000
        wm  = _MockWeightMatrix(W, "random")
        wm.set_pedigree(meta_perfect.to_pedigree_scores())
        wm.set_F_src(fs)
        result = wm.apply_with_uncertainty()
        assert result.F_root.shape    == (15,)
        assert result.sigma_root.shape == (15,)
        assert np.all(result.sigma_root >= 0.0)


# ===========================================================================
# 4.  PropagationResult
# ===========================================================================

class TestPropagationResult:

    def test_fields_present(self, wm_std, meta_perfect):
        wm_std.set_pedigree(meta_perfect.to_pedigree_scores())
        wm_std.set_F_src(F_SRC_STD)
        result = wm_std.apply_with_uncertainty()
        for attr in ("F_root", "sigma_root", "cv_root", "cv_src"):
            assert hasattr(result, attr), f"PropagationResult missing attribute '{attr}'"

    def test_cv_root_percent_reasonable(self, wm_std, meta_worst):
        """With worst pedigree scores, CV% should be nonzero but not absurd."""
        wm_std.set_pedigree(meta_worst.to_pedigree_scores())
        wm_std.set_F_src(F_SRC_STD)
        result = wm_std.apply_with_uncertainty()
        nonzero = ~np.isnan(result.cv_root)
        assert np.all(result.cv_root[nonzero] > 0.0)
        assert np.all(result.cv_root[nonzero] < 100.0), (
            "CV% > 10 000% – likely a unit error"
        )


# ===========================================================================
# 5.  DataFeedRegistry
# ===========================================================================

class TestDataFeedRegistry:

    @staticmethod
    def _parents(feed_id: str) -> dict:
        return {
            "inputs": {
                "acc":  {"id": "acc_test", "version": 1},
                "feed": {"id": feed_id,    "version": 1},
            },
            "upstream": {},
        }

    def _run_and_register(self, tmpdir, feed_id="test_feed", meta_kwargs=None):
        meta_kwargs = meta_kwargs or {}
        meta   = _make_metadata(feed_name=feed_id, **meta_kwargs)
        wm     = _MockWeightMatrix(W_STD, feed_id=feed_id)
        wm.set_pedigree(meta.to_pedigree_scores())
        wm.set_F_src(F_SRC_STD)
        result = wm.apply_with_uncertainty()

        registry = DataFeedRegistry(odir=tmpdir)
        entry    = registry.register(
            wm, meta, parents=self._parents(feed_id), result=result,
        )
        registry.save()
        return registry, entry, meta, result

    # ------------------------------------------------------------------

    def test_register_returns_entry(self, tmp_path):
        _, entry, _, _ = self._run_and_register(str(tmp_path))
        assert entry is not None
        assert entry.run_id.startswith("feed_")
        assert entry.stage == "feed"

    def test_entry_index_dict_has_four_fields(self, tmp_path):
        _, entry, _, _ = self._run_and_register(str(tmp_path))
        d = entry.to_index_dict()
        assert set(d.keys()) == {"run_id", "stage", "label", "created_at"}
        assert d["label"] == "test_feed"

    def test_registry_json_created(self, tmp_path):
        self._run_and_register(str(tmp_path))
        assert (tmp_path / "registry.json").exists(), "registry.json not found"

    def test_load_index_after_save(self, tmp_path):
        self._run_and_register(str(tmp_path))
        registry2 = DataFeedRegistry(odir=str(tmp_path))
        registry2.load_index()
        df = registry2.summary_df()
        assert len(df) >= 1

    def test_reload_metadata_round_trip(self, tmp_path):
        _, _, meta, _ = self._run_and_register(str(tmp_path))
        registry2 = DataFeedRegistry(odir=str(tmp_path))
        registry2.load_index()
        meta2 = registry2.load_metadata("test_feed")
        assert meta2.feed_name == meta.feed_name

    def test_reload_result_round_trip(self, tmp_path):
        _, _, _, result = self._run_and_register(str(tmp_path))
        registry2 = DataFeedRegistry(odir=str(tmp_path))
        registry2.load_index()
        r2 = registry2.load_result("test_feed")
        np.testing.assert_allclose(r2["F_root"],     result.F_root,     rtol=1e-12)
        np.testing.assert_allclose(r2["sigma_root"], result.sigma_root, rtol=1e-12)

    def test_re_registering_same_feed_id_creates_new_run(self, tmp_path):
        registry, e1, _, _ = self._run_and_register(str(tmp_path))
        meta2 = _make_metadata(feed_name="test_feed", reliability=2)
        wm2   = _MockWeightMatrix(W_STD, "test_feed")
        wm2.set_pedigree(meta2.to_pedigree_scores())
        wm2.set_F_src(F_SRC_STD)
        result2 = wm2.apply_with_uncertainty()
        e2 = registry.register(
            wm2, meta2, parents=self._parents("test_feed"), result=result2,
        )
        registry.save()
        assert e1.run_id != e2.run_id
        assert len(registry) == 2

    def test_multiple_feeds_in_registry(self, tmp_path):
        registry = DataFeedRegistry(odir=str(tmp_path))
        for fid in ("feed_a", "feed_b", "feed_c"):
            meta   = _make_metadata(feed_name=fid)
            wm     = _MockWeightMatrix(W_STD, feed_id=fid)
            wm.set_pedigree(meta.to_pedigree_scores())
            wm.set_F_src(F_SRC_STD)
            result = wm.apply_with_uncertainty()
            registry.register(wm, meta, parents=self._parents(fid), result=result)
        registry.save()

        registry2 = DataFeedRegistry(odir=str(tmp_path))
        registry2.load_index()
        df = registry2.summary_df()
        assert len(df) >= 3

    def test_cv_total_persisted_correctly(self, tmp_path):
        _, _, meta, _ = self._run_and_register(str(tmp_path))
        registry2 = DataFeedRegistry(odir=str(tmp_path))
        registry2.load_index()
        meta2 = registry2.load_metadata("test_feed")
        if hasattr(meta, "cv_total") and hasattr(meta2, "cv_total"):
            assert abs(meta2.cv_total - meta.cv_total) < 1e-12


# ===========================================================================
# 6.  Full pipeline (step_D_apply_and_propagate as convenience)
# ===========================================================================

class TestStepD:
    """
    Tests that mirror the smoke-test in feedpipe.__main__ but as proper
    pytest assertions so failures are visible in CI.
    """

    def test_step_D_result_matches_manual_calculation(self, meta_perfect):
        wm = _MockWeightMatrix(W_STD, "step_d_test")

        from pysuto.feedpipe import step_D_apply_and_propagate
        result = step_D_apply_and_propagate(wm, F_SRC_STD, meta_perfect)

        expected_F_root = W_STD.T @ F_SRC_STD
        np.testing.assert_allclose(result.F_root, expected_F_root, rtol=1e-10)

    def test_step_D_propagation_result_type(self, meta_perfect):
        from pysuto.feedpipe import step_D_apply_and_propagate
        wm     = _MockWeightMatrix(W_STD, "step_d_type")
        result = step_D_apply_and_propagate(wm, F_SRC_STD, meta_perfect)
        assert isinstance(result, PropagationResult)

    def test_step_D_sigma_scales_with_cv(self):
        """Doubling the CV should approximately double sigma_root."""
        meta_low  = _make_metadata(reliability=1, completeness=1,
                                   temporal_correlation=1,
                                   geographical_correlation=1,
                                   other_correlation=1)
        meta_high = _make_metadata(reliability=4, completeness=4,
                                   temporal_correlation=4,
                                   geographical_correlation=4,
                                   other_correlation=4)
        from pysuto.feedpipe import step_D_apply_and_propagate
        wm_low  = _MockWeightMatrix(W_STD, "low")
        wm_high = _MockWeightMatrix(W_STD, "high")
        r_low  = step_D_apply_and_propagate(wm_low,  F_SRC_STD, meta_low)
        r_high = step_D_apply_and_propagate(wm_high, F_SRC_STD, meta_high)
        # sigma_root is proportional to cv_src; so ratio should be cv_high/cv_low
        ratio_sigma = r_high.sigma_root / r_low.sigma_root
        ratio_cv    = r_high.cv_src     / r_low.cv_src
        np.testing.assert_allclose(ratio_sigma, ratio_cv, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])