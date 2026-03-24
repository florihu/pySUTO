"""
test_unc.py
===========
Unit tests for pysuto.unc:
  - PedigreeScores  (construction, validation, individual CVs, total CV)
  - PropagationResult  (dataclass contract)
  - UncertaintyMixin  (apply_with_uncertainty, set_pedigree*, set_F_src)

All numerical assertions use independently computed ground-truth values
derived directly from Laner et al. (2016) Equations 1–3 and Gaussian
error propagation.

Ground-truth reference values (computed analytically, B = 1.105):
  cv_rel(1)  = 0.75 * exp(1.105*1)  / 100  = 0.022644183...
  cv_rel(2)  = 0.75 * exp(1.105*2)  / 100  = 0.068367873...
  cv_rel(3)  = 0.75 * exp(1.105*3)  / 100  = 0.206417955...
  cv_rel(4)  = 0.75 * exp(1.105*4)  / 100  = 0.623222140...
  cv_other(2, medium) = 0.75 * exp(1.105*1) / 100  = 0.022644183...
  cv_other(3, medium) = 0.75 * exp(1.105*2) / 100  = 0.068367873...
  cv_other(2, not_sensitive)   = 0.375 * exp(1.105*1) / 100 = 0.011322092...
  cv_other(2, highly_sensitive)= 1.500 * exp(1.105*1) / 100 = 0.045288367...

Propagation (W 4×3, F_src=[100,200,150,80], all scores=1):
  cv_tot     = 0.022644183...
  F_root     = [200, 274, 56]
  sigma_root = [3.20237114, 4.11825427, 1.26807428]
  cv_root    = [0.01601186, 0.01503013, 0.02264418]
"""

from __future__ import annotations

import numpy as np
import pytest

from pysuto.unc import (
    PedigreeScores,
    PropagationResult,
    UncertaintyMixin,
    B_UNIVERSAL,
    SENSITIVITY_PARAMS,
    VALID_SCORES,
)

# ── tolerance for floating-point assertions ──────────────────────────────────
RTOL = 1e-8
ATOL = 1e-12


# =============================================================================
# HELPERS  (mirror the formulas independently — no import from unc.py)
# =============================================================================

def _cv_rel(s: int) -> float:
    return 0.750 * np.exp(B_UNIVERSAL * s) / 100.0


def _cv_other(s: int, sens: str) -> float:
    if s == 1:
        return 0.0
    return SENSITIVITY_PARAMS[sens] * np.exp(B_UNIVERSAL * (s - 1)) / 100.0


def _total_cv(r, comp, temp, geo, oth,
              cs="medium_sensitive", ts="medium_sensitive",
              gs="medium_sensitive", os_="medium_sensitive") -> float:
    return float(np.sqrt(
        _cv_rel(r)**2 + _cv_other(comp, cs)**2 + _cv_other(temp, ts)**2
        + _cv_other(geo, gs)**2 + _cv_other(oth, os_)**2
    ))


# ── Minimal UncertaintyMixin carrier (dense numpy W, no sparse dep) ──────────

class _MockFeed(UncertaintyMixin):
    """Stand-in for WeightMatrix: owns a dense weight matrix."""

    def __init__(self, W: np.ndarray):
        # Store as numpy; apply_with_uncertainty accepts anything with .T @ vec
        self.weight_matrix = W


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def all_ones_pedigree():
    """Best-quality pedigree: all scores = 1."""
    return PedigreeScores()


@pytest.fixture
def all_twos_pedigree():
    return PedigreeScores(
        reliability=2, completeness=2, temporal_correlation=2,
        geographical_correlation=2, other_correlation=2,
    )


@pytest.fixture
def all_fours_pedigree():
    return PedigreeScores(
        reliability=4, completeness=4, temporal_correlation=4,
        geographical_correlation=4, other_correlation=4,
    )


@pytest.fixture
def mixed_pedigree():
    """
    reliability=2, completeness=3, temporal_correlation=1,
    geographical_correlation=4, other_correlation=2,
    comp: not_sensitive, geo: highly_sensitive, oth: not_sensitive
    total_cv ≈ 0.42000522
    """
    return PedigreeScores(
        reliability=2,
        completeness=3,
        temporal_correlation=1,
        geographical_correlation=4,
        other_correlation=2,
        completeness_sensitivity="not_sensitive",
        geographical_sensitivity="highly_sensitive",
        other_sensitivity="not_sensitive",
    )


@pytest.fixture
def W_4x3():
    """4 src edges → 3 root edges. Rows sum to 1 (valid concordance)."""
    return np.array([
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.3, 0.7],
    ], dtype=float)


@pytest.fixture
def F_src_4():
    return np.array([100.0, 200.0, 150.0, 80.0])


@pytest.fixture
def feed_4x3(W_4x3):
    return _MockFeed(W_4x3)


# =============================================================================
# 1. MODULE-LEVEL CONSTANTS
# =============================================================================

class TestModuleConstants:

    def test_b_universal_value(self):
        assert B_UNIVERSAL == pytest.approx(1.105)

    def test_valid_scores(self):
        assert VALID_SCORES == frozenset({1, 2, 3, 4})

    def test_sensitivity_params_keys(self):
        assert set(SENSITIVITY_PARAMS) == {
            "not_sensitive", "medium_sensitive", "highly_sensitive"
        }

    def test_sensitivity_params_ordering(self):
        """not_sensitive < medium_sensitive < highly_sensitive."""
        ns = SENSITIVITY_PARAMS["not_sensitive"]
        ms = SENSITIVITY_PARAMS["medium_sensitive"]
        hs = SENSITIVITY_PARAMS["highly_sensitive"]
        assert ns < ms < hs

    def test_sensitivity_ratio_2x(self):
        """Each level doubles the previous (0.375, 0.75, 1.5)."""
        ns = SENSITIVITY_PARAMS["not_sensitive"]
        ms = SENSITIVITY_PARAMS["medium_sensitive"]
        hs = SENSITIVITY_PARAMS["highly_sensitive"]
        assert ms == pytest.approx(2 * ns)
        assert hs == pytest.approx(2 * ms)


# =============================================================================
# 2. PEDIGREE SCORES — CONSTRUCTION & VALIDATION
# =============================================================================

class TestPedigreeScoresConstruction:

    def test_default_scores_are_all_one(self, all_ones_pedigree):
        pg = all_ones_pedigree
        assert pg.reliability == 1
        assert pg.completeness == 1
        assert pg.temporal_correlation == 1
        assert pg.geographical_correlation == 1
        assert pg.other_correlation == 1

    def test_default_sensitivities_are_medium(self, all_ones_pedigree):
        pg = all_ones_pedigree
        assert pg.completeness_sensitivity == "medium_sensitive"
        assert pg.temporal_sensitivity == "medium_sensitive"
        assert pg.geographical_sensitivity == "medium_sensitive"
        assert pg.other_sensitivity == "medium_sensitive"

    def test_all_valid_scores_accepted(self):
        for s in [1, 2, 3, 4]:
            PedigreeScores(
                reliability=s, completeness=s,
                temporal_correlation=s, geographical_correlation=s,
                other_correlation=s,
            )

    def test_all_sensitivity_levels_accepted(self):
        for sens in ["not_sensitive", "medium_sensitive", "highly_sensitive"]:
            PedigreeScores(
                completeness_sensitivity=sens,
                temporal_sensitivity=sens,
                geographical_sensitivity=sens,
                other_sensitivity=sens,
            )

    def test_score_zero_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            PedigreeScores(reliability=0)

    def test_score_five_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            PedigreeScores(completeness=5)

    def test_score_negative_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            PedigreeScores(temporal_correlation=-1)

    def test_score_float_raises(self):
        with pytest.raises((ValueError, TypeError)):
            PedigreeScores(geographical_correlation=1.5)

    def test_invalid_sensitivity_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            PedigreeScores(completeness_sensitivity="very_sensitive")

    def test_invalid_sensitivity_typo_raises(self):
        with pytest.raises(ValueError, match="must be one of"):
            PedigreeScores(temporal_sensitivity="Medium_Sensitive")

    @pytest.mark.parametrize("attr", [
        "reliability", "completeness", "temporal_correlation",
        "geographical_correlation", "other_correlation"
    ])
    def test_each_score_attribute_validated(self, attr):
        with pytest.raises(ValueError, match="must be in"):
            PedigreeScores(**{attr: 99})

    @pytest.mark.parametrize("attr", [
        "completeness_sensitivity", "temporal_sensitivity",
        "geographical_sensitivity", "other_sensitivity"
    ])
    def test_each_sensitivity_attribute_validated(self, attr):
        with pytest.raises(ValueError, match="must be one of"):
            PedigreeScores(**{attr: "wrong"})


# =============================================================================
# 3. INDIVIDUAL CV FORMULAS
# =============================================================================

class TestCvReliability:

    def test_score_1_matches_formula(self):
        pg = PedigreeScores(reliability=1)
        assert pg._cv_reliability(1) == pytest.approx(_cv_rel(1), rel=RTOL)

    def test_score_2_matches_formula(self):
        assert PedigreeScores._cv_reliability(2) == pytest.approx(_cv_rel(2), rel=RTOL)

    def test_score_3_matches_formula(self):
        assert PedigreeScores._cv_reliability(3) == pytest.approx(_cv_rel(3), rel=RTOL)

    def test_score_4_matches_formula(self):
        assert PedigreeScores._cv_reliability(4) == pytest.approx(_cv_rel(4), rel=RTOL)

    def test_score_1_exact_value(self):
        expected = 0.75 * np.exp(1.105) / 100.0
        assert PedigreeScores._cv_reliability(1) == pytest.approx(expected, rel=RTOL)

    def test_score_1_is_nonzero(self):
        """Reliability always contributes uncertainty even at best score."""
        assert PedigreeScores._cv_reliability(1) > 0.0

    def test_strictly_increasing(self):
        cvs = [PedigreeScores._cv_reliability(s) for s in [1, 2, 3, 4]]
        assert all(cvs[i] < cvs[i + 1] for i in range(3))

    def test_returns_fraction_not_percent(self):
        """CV at score=1 must be ~2.26%, i.e. 0.0226 as a fraction."""
        cv = PedigreeScores._cv_reliability(1)
        assert 0.01 < cv < 0.10

    def test_static_method_callable_without_instance(self):
        result = PedigreeScores._cv_reliability(2)
        assert isinstance(result, float)


class TestCvOther:

    def test_score_1_is_zero_for_all_sensitivities(self):
        for sens in SENSITIVITY_PARAMS:
            assert PedigreeScores._cv_other(1, sens) == 0.0

    def test_score_2_medium_matches_formula(self):
        expected = _cv_other(2, "medium_sensitive")
        assert PedigreeScores._cv_other(2, "medium_sensitive") == pytest.approx(
            expected, rel=RTOL
        )

    def test_score_3_medium_matches_formula(self):
        assert PedigreeScores._cv_other(3, "medium_sensitive") == pytest.approx(
            _cv_other(3, "medium_sensitive"), rel=RTOL
        )

    def test_score_4_medium_matches_formula(self):
        assert PedigreeScores._cv_other(4, "medium_sensitive") == pytest.approx(
            _cv_other(4, "medium_sensitive"), rel=RTOL
        )

    def test_score_2_not_sensitive_exact(self):
        expected = 0.375 * np.exp(1.105) / 100.0
        assert PedigreeScores._cv_other(2, "not_sensitive") == pytest.approx(
            expected, rel=RTOL
        )

    def test_score_2_highly_sensitive_exact(self):
        expected = 1.500 * np.exp(1.105) / 100.0
        assert PedigreeScores._cv_other(2, "highly_sensitive") == pytest.approx(
            expected, rel=RTOL
        )

    def test_sensitivity_ordering_at_score_2(self):
        ns = PedigreeScores._cv_other(2, "not_sensitive")
        ms = PedigreeScores._cv_other(2, "medium_sensitive")
        hs = PedigreeScores._cv_other(2, "highly_sensitive")
        assert ns < ms < hs

    def test_sensitivity_ordering_at_score_3(self):
        ns = PedigreeScores._cv_other(3, "not_sensitive")
        ms = PedigreeScores._cv_other(3, "medium_sensitive")
        hs = PedigreeScores._cv_other(3, "highly_sensitive")
        assert ns < ms < hs

    def test_sensitivity_ordering_at_score_4(self):
        ns = PedigreeScores._cv_other(4, "not_sensitive")
        ms = PedigreeScores._cv_other(4, "medium_sensitive")
        hs = PedigreeScores._cv_other(4, "highly_sensitive")
        assert ns < ms < hs

    def test_non_decreasing_with_score_medium(self):
        cvs = [PedigreeScores._cv_other(s, "medium_sensitive") for s in [1, 2, 3, 4]]
        assert all(cvs[i] <= cvs[i + 1] for i in range(3))

    def test_non_decreasing_with_score_not_sensitive(self):
        cvs = [PedigreeScores._cv_other(s, "not_sensitive") for s in [1, 2, 3, 4]]
        assert all(cvs[i] <= cvs[i + 1] for i in range(3))

    def test_ratio_medium_to_not_sensitive_is_2(self):
        """medium_sensitive 'a' is exactly 2× not_sensitive 'a'."""
        for s in [2, 3, 4]:
            ns = PedigreeScores._cv_other(s, "not_sensitive")
            ms = PedigreeScores._cv_other(s, "medium_sensitive")
            assert ms == pytest.approx(2 * ns, rel=RTOL)

    def test_ratio_highly_to_medium_is_2(self):
        for s in [2, 3, 4]:
            ms = PedigreeScores._cv_other(s, "medium_sensitive")
            hs = PedigreeScores._cv_other(s, "highly_sensitive")
            assert hs == pytest.approx(2 * ms, rel=RTOL)

    def test_returns_fraction_not_percent(self):
        cv = PedigreeScores._cv_other(4, "highly_sensitive")
        assert 0.0 < cv < 1.0


# =============================================================================
# 4. TOTAL CV
# =============================================================================

class TestTotalCv:

    def test_all_scores_1_equals_cv_reliability_only(self, all_ones_pedigree):
        """When all indicator scores=1, only reliability contributes."""
        expected = _cv_rel(1)
        assert all_ones_pedigree.total_cv() == pytest.approx(expected, rel=RTOL)

    def test_all_scores_1_exact_value(self, all_ones_pedigree):
        assert all_ones_pedigree.total_cv() == pytest.approx(0.022644183, rel=1e-6)

    def test_all_scores_2_medium_exact(self, all_twos_pedigree):
        assert all_twos_pedigree.total_cv() == pytest.approx(0.082007330, rel=1e-6)

    def test_reliability_only_at_score_3(self):
        pg = PedigreeScores(reliability=3)   # other scores default to 1
        expected = _cv_rel(3)
        assert pg.total_cv() == pytest.approx(expected, rel=RTOL)

    def test_mixed_scores_and_sensitivities(self, mixed_pedigree):
        expected = _total_cv(
            r=2, comp=3, temp=1, geo=4, oth=2,
            cs="not_sensitive", ts="medium_sensitive",
            gs="highly_sensitive", os_="not_sensitive",
        )
        assert mixed_pedigree.total_cv() == pytest.approx(expected, rel=RTOL)

    def test_mixed_scores_exact_value(self, mixed_pedigree):
        assert mixed_pedigree.total_cv() == pytest.approx(0.420005221, rel=1e-6)

    def test_total_cv_returns_float(self, all_ones_pedigree):
        assert isinstance(all_ones_pedigree.total_cv(), float)

    def test_total_cv_is_positive(self, all_ones_pedigree):
        assert all_ones_pedigree.total_cv() > 0.0

    def test_total_cv_increases_with_score(self):
        """Increasing all scores must increase total CV."""
        cv1 = PedigreeScores(reliability=1, completeness=1, temporal_correlation=1,
                             geographical_correlation=1, other_correlation=1).total_cv()
        cv4 = PedigreeScores(reliability=4, completeness=4, temporal_correlation=4,
                             geographical_correlation=4, other_correlation=4).total_cv()
        assert cv4 > cv1

    def test_total_cv_is_sum_in_quadrature(self, mixed_pedigree):
        """total_cv² must equal sum of individual CV²."""
        s = mixed_pedigree.summary()
        sum_sq = (
            s["cv_reliability"]**2
            + s["cv_completeness"]**2
            + s["cv_temporal_correlation"]**2
            + s["cv_geographical_correlation"]**2
            + s["cv_other_correlation"]**2
        )
        assert mixed_pedigree.total_cv()**2 == pytest.approx(sum_sq, rel=RTOL)

    def test_individual_cv_never_exceeds_total(self, mixed_pedigree):
        s = mixed_pedigree.summary()
        total = s["cv_total"]
        for key, val in s.items():
            if key != "cv_total":
                assert val <= total + ATOL, f"{key} = {val} > total = {total}"


# =============================================================================
# 5. SUMMARY DICT
# =============================================================================

class TestSummaryDict:

    def test_summary_returns_dict(self, all_ones_pedigree):
        assert isinstance(all_ones_pedigree.summary(), dict)

    def test_summary_keys(self, all_ones_pedigree):
        expected_keys = {
            "cv_reliability", "cv_completeness", "cv_temporal_correlation",
            "cv_geographical_correlation", "cv_other_correlation", "cv_total",
        }
        assert set(all_ones_pedigree.summary().keys()) == expected_keys

    def test_summary_values_are_floats(self, all_ones_pedigree):
        for v in all_ones_pedigree.summary().values():
            assert isinstance(v, float)

    def test_summary_cv_total_matches_total_cv_method(self, mixed_pedigree):
        assert mixed_pedigree.summary()["cv_total"] == pytest.approx(
            mixed_pedigree.total_cv(), rel=RTOL
        )

    def test_summary_reliability_matches_static_method(self, all_twos_pedigree):
        assert all_twos_pedigree.summary()["cv_reliability"] == pytest.approx(
            PedigreeScores._cv_reliability(2), rel=RTOL
        )

    def test_summary_completeness_matches_static_method(self, all_twos_pedigree):
        assert all_twos_pedigree.summary()["cv_completeness"] == pytest.approx(
            PedigreeScores._cv_other(2, "medium_sensitive"), rel=RTOL
        )

    def test_summary_all_values_non_negative(self, all_fours_pedigree):
        for v in all_fours_pedigree.summary().values():
            assert v >= 0.0

    def test_summary_all_ones_non_reliability_are_zero(self, all_ones_pedigree):
        s = all_ones_pedigree.summary()
        for key in ["cv_completeness", "cv_temporal_correlation",
                    "cv_geographical_correlation", "cv_other_correlation"]:
            assert s[key] == 0.0

    def test_summary_all_ones_reliability_nonzero(self, all_ones_pedigree):
        assert all_ones_pedigree.summary()["cv_reliability"] > 0.0


# =============================================================================
# 6. PROPAGATION RESULT DATACLASS
# =============================================================================

class TestPropagationResult:

    @pytest.fixture
    def sample_result(self):
        return PropagationResult(
            F_root=np.array([1.0, 2.0, 3.0]),
            sigma_root=np.array([0.1, 0.2, 0.3]),
            cv_root=np.array([0.1, 0.1, 0.1]),
            cv_src=0.1,
        )

    def test_F_root_stored(self, sample_result):
        np.testing.assert_array_equal(sample_result.F_root, [1.0, 2.0, 3.0])

    def test_sigma_root_stored(self, sample_result):
        np.testing.assert_array_equal(sample_result.sigma_root, [0.1, 0.2, 0.3])

    def test_cv_root_stored(self, sample_result):
        np.testing.assert_array_equal(sample_result.cv_root, [0.1, 0.1, 0.1])

    def test_cv_src_stored(self, sample_result):
        assert sample_result.cv_src == pytest.approx(0.1)

    def test_repr_contains_shape(self, sample_result):
        r = repr(sample_result)
        assert "F_root.shape" in r
        assert "(3,)" in r

    def test_repr_contains_cv_src(self, sample_result):
        assert "cv_src" in repr(sample_result)

    def test_repr_contains_mean_cv_root(self, sample_result):
        assert "mean_cv_root" in repr(sample_result)

    def test_is_dataclass(self, sample_result):
        import dataclasses
        assert dataclasses.is_dataclass(sample_result)

    def test_accepts_nan_cv_root(self):
        """cv_root may contain NaN where F_root == 0."""
        r = PropagationResult(
            F_root=np.array([0.0, 1.0]),
            sigma_root=np.array([0.0, 0.1]),
            cv_root=np.array([np.nan, 0.1]),
            cv_src=0.05,
        )
        assert np.isnan(r.cv_root[0])


# =============================================================================
# 7. UNCERTAINTY MIXIN — SETUP METHODS
# =============================================================================

class TestUncertaintyMixinSetup:

    def test_set_pedigree_stores_object(self, feed_4x3, all_ones_pedigree):
        feed_4x3.set_pedigree(all_ones_pedigree)
        assert feed_4x3.pedigree is all_ones_pedigree

    def test_set_pedigree_from_dict_creates_pedigree_scores(self, feed_4x3):
        feed_4x3.set_pedigree_from_dict({
            "reliability": 2,
            "completeness": 3,
        })
        assert isinstance(feed_4x3.pedigree, PedigreeScores)
        assert feed_4x3.pedigree.reliability == 2
        assert feed_4x3.pedigree.completeness == 3

    def test_set_pedigree_from_dict_ignores_unknown_keys(self, feed_4x3):
        """Extra keys not in PedigreeScores.__dataclass_fields__ must be ignored."""
        feed_4x3.set_pedigree_from_dict({"reliability": 1, "unknown_key": 99})
        assert feed_4x3.pedigree.reliability == 1

    def test_set_pedigree_from_dict_uses_defaults_for_missing_keys(self, feed_4x3):
        feed_4x3.set_pedigree_from_dict({})
        assert feed_4x3.pedigree.reliability == 1
        assert feed_4x3.pedigree.completeness == 1

    def test_set_F_src_stores_array(self, feed_4x3):
        feed_4x3.set_F_src(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_array_equal(feed_4x3.F_src, [1.0, 2.0, 3.0, 4.0])

    def test_set_F_src_converts_list_to_float_array(self, feed_4x3):
        feed_4x3.set_F_src([1, 2, 3, 4])
        assert feed_4x3.F_src.dtype == np.float64

    def test_set_F_src_converts_int_array(self, feed_4x3):
        feed_4x3.set_F_src(np.array([1, 2, 3, 4], dtype=int))
        assert feed_4x3.F_src.dtype == np.float64


# =============================================================================
# 8. APPLY WITH UNCERTAINTY — POINT ESTIMATE
# =============================================================================

class TestApplyPointEstimate:
    """F_root = W^T @ F_src must be exact regardless of uncertainty."""

    def test_F_root_shape(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        assert result.F_root.shape == (3,)

    def test_F_root_values_exact(self, feed_4x3, all_ones_pedigree, F_src_4, W_4x3):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        expected = W_4x3.T @ F_src_4
        np.testing.assert_allclose(result.F_root, expected, rtol=RTOL, atol=ATOL)

    def test_F_root_values_concrete(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        np.testing.assert_allclose(result.F_root, [200.0, 274.0, 56.0],
                                   rtol=RTOL, atol=ATOL)

    def test_identity_W_preserves_F_src(self, all_ones_pedigree):
        W = np.eye(3)
        F = np.array([10.0, 20.0, 30.0])
        feed = _MockFeed(W)
        feed.set_pedigree(all_ones_pedigree)
        result = feed.apply_with_uncertainty(F)
        np.testing.assert_allclose(result.F_root, F, rtol=RTOL)

    def test_uniform_W_averages_F_src(self, all_ones_pedigree):
        """A single root edge fed by two equal-weight source edges."""
        W = np.array([[0.5], [0.5]])
        F = np.array([100.0, 200.0])
        feed = _MockFeed(W)
        feed.set_pedigree(all_ones_pedigree)
        result = feed.apply_with_uncertainty(F)
        np.testing.assert_allclose(result.F_root, [150.0], rtol=RTOL)

    def test_uses_stored_F_src_when_not_passed(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        feed_4x3.set_F_src(F_src_4)
        result = feed_4x3.apply_with_uncertainty()   # no F_src argument
        np.testing.assert_allclose(result.F_root, [200.0, 274.0, 56.0],
                                   rtol=RTOL, atol=ATOL)

    def test_passed_F_src_overrides_stored(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        feed_4x3.set_F_src(np.zeros(4))              # stored: all zero
        result = feed_4x3.apply_with_uncertainty(F_src_4)  # passed: non-zero
        np.testing.assert_allclose(result.F_root, [200.0, 274.0, 56.0],
                                   rtol=RTOL, atol=ATOL)


# =============================================================================
# 9. APPLY WITH UNCERTAINTY — SIGMA AND CV
# =============================================================================

class TestApplySigma:

    def test_sigma_root_shape(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        assert result.sigma_root.shape == (3,)

    def test_sigma_root_non_negative(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        assert np.all(result.sigma_root >= 0.0)

    def test_sigma_root_concrete_values(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        expected = np.array([3.20237114, 4.11825427, 1.26807428])
        np.testing.assert_allclose(result.sigma_root, expected, rtol=1e-6)

    def test_sigma_root_formula(self, feed_4x3, all_ones_pedigree, F_src_4, W_4x3):
        """sigma_root = cv_tot * sqrt( (W⊙W)^T @ F_src^2 )."""
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        cv = all_ones_pedigree.total_cv()
        expected_var = cv**2 * ((W_4x3**2).T @ F_src_4**2)
        expected_sigma = np.sqrt(expected_var)
        np.testing.assert_allclose(result.sigma_root, expected_sigma, rtol=RTOL)

    def test_cv_root_shape(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        assert result.cv_root.shape == (3,)

    def test_cv_root_concrete_values(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        expected = np.array([0.01601186, 0.01503013, 0.02264418])
        np.testing.assert_allclose(result.cv_root, expected, rtol=1e-5)

    def test_cv_root_equals_sigma_over_F_root(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        expected = result.sigma_root / np.abs(result.F_root)
        np.testing.assert_allclose(result.cv_root, expected, rtol=RTOL)

    def test_cv_src_stored_in_result(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        assert result.cv_src == pytest.approx(all_ones_pedigree.total_cv(), rel=RTOL)

    def test_identity_W_sigma_equals_cv_times_F_src(self, all_twos_pedigree):
        """With W=I: sigma_root_j = CV_tot * F_src_j (no mixing)."""
        W = np.eye(3)
        F = np.array([10.0, 20.0, 30.0])
        feed = _MockFeed(W)
        feed.set_pedigree(all_twos_pedigree)
        result = feed.apply_with_uncertainty(F)
        expected_sigma = all_twos_pedigree.total_cv() * F
        np.testing.assert_allclose(result.sigma_root, expected_sigma, rtol=RTOL)

    def test_sigma_scales_linearly_with_F_src(self, feed_4x3, all_ones_pedigree, F_src_4):
        """Doubling F_src must double sigma_root (linear propagation)."""
        feed_4x3.set_pedigree(all_ones_pedigree)
        r1 = feed_4x3.apply_with_uncertainty(F_src_4)
        r2 = feed_4x3.apply_with_uncertainty(2 * F_src_4)
        np.testing.assert_allclose(r2.sigma_root, 2 * r1.sigma_root, rtol=RTOL)

    def test_sigma_scales_linearly_with_cv(self, feed_4x3, F_src_4):
        """Doubling CV (via higher score) roughly doubles sigma_root."""
        pg_low  = PedigreeScores(reliability=1)
        pg_high = PedigreeScores(reliability=2)
        feed_4x3.set_pedigree(pg_low)
        r_low = feed_4x3.apply_with_uncertainty(F_src_4)
        feed_4x3.set_pedigree(pg_high)
        r_high = feed_4x3.apply_with_uncertainty(F_src_4)
        ratio = r_high.sigma_root / r_low.sigma_root
        expected_ratio = pg_high.total_cv() / pg_low.total_cv()
        np.testing.assert_allclose(ratio, expected_ratio, rtol=RTOL)


# =============================================================================
# 10. DIVERSIFICATION INVARIANT  cv_root[j] ≤ cv_src
# =============================================================================

class TestDiversificationInvariant:
    """
    Aggregation through W can only reduce or preserve proportional
    uncertainty — it can never amplify it.
    Reference: REQUIREMENTS.md §4, requirement U8.
    """

    def test_invariant_all_scores_1(self, feed_4x3, all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        assert np.all(result.cv_root <= result.cv_src + ATOL)

    def test_invariant_all_scores_4(self, feed_4x3, all_fours_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_fours_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        assert np.all(result.cv_root <= result.cv_src + ATOL)

    def test_invariant_mixed_pedigree(self, feed_4x3, mixed_pedigree, F_src_4):
        feed_4x3.set_pedigree(mixed_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        assert np.all(result.cv_root <= result.cv_src + ATOL)

    def test_invariant_identity_W_cv_root_equals_cv_src(self, all_twos_pedigree):
        """With identity W, every root edge has exactly one source → CV preserved."""
        W = np.eye(3)
        F = np.array([10.0, 20.0, 30.0])
        feed = _MockFeed(W)
        feed.set_pedigree(all_twos_pedigree)
        result = feed.apply_with_uncertainty(F)
        cv_expected = all_twos_pedigree.total_cv()
        np.testing.assert_allclose(result.cv_root, cv_expected, rtol=RTOL)

    def test_invariant_many_sources_reduces_cv(self, all_ones_pedigree):
        """N equal-weight sources: cv_root = cv_src / sqrt(N)."""
        N = 4
        W = np.full((N, 1), 1.0 / N)
        F = np.ones(N) * 100.0
        feed = _MockFeed(W)
        feed.set_pedigree(all_ones_pedigree)
        result = feed.apply_with_uncertainty(F)
        cv_expected = all_ones_pedigree.total_cv() / np.sqrt(N)
        np.testing.assert_allclose(result.cv_root[0], cv_expected, rtol=RTOL)
        assert result.cv_root[0] < all_ones_pedigree.total_cv()

    def test_invariant_holds_for_heterogeneous_F_src(self, all_ones_pedigree):
        W = np.array([
            [0.2, 0.8],
            [0.6, 0.4],
            [0.9, 0.1],
        ])
        F = np.array([500.0, 1.0, 300.0])
        feed = _MockFeed(W)
        feed.set_pedigree(all_ones_pedigree)
        result = feed.apply_with_uncertainty(F)
        assert np.all(result.cv_root <= result.cv_src + ATOL)


# =============================================================================
# 11. ZERO AND EDGE-CASE INPUTS
# =============================================================================

class TestEdgeCases:

    def test_zero_F_src_gives_zero_F_root(self, feed_4x3, all_ones_pedigree):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(np.zeros(4))
        np.testing.assert_array_equal(result.F_root, [0.0, 0.0, 0.0])

    def test_zero_F_src_gives_zero_sigma(self, feed_4x3, all_ones_pedigree):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(np.zeros(4))
        np.testing.assert_array_equal(result.sigma_root, [0.0, 0.0, 0.0])

    def test_zero_F_src_gives_nan_cv_root(self, feed_4x3, all_ones_pedigree):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(np.zeros(4))
        assert np.all(np.isnan(result.cv_root))

    def test_single_edge_single_source(self, all_ones_pedigree):
        W = np.array([[1.0]])
        F = np.array([42.0])
        feed = _MockFeed(W)
        feed.set_pedigree(all_ones_pedigree)
        result = feed.apply_with_uncertainty(F)
        assert result.F_root[0] == pytest.approx(42.0)
        expected_sigma = all_ones_pedigree.total_cv() * 42.0
        assert result.sigma_root[0] == pytest.approx(expected_sigma, rel=RTOL)

    def test_very_small_F_src_no_numerical_error(self, feed_4x3, all_ones_pedigree):
        F_tiny = np.array([1e-15, 1e-15, 1e-15, 1e-15])
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_tiny)
        assert np.all(result.sigma_root >= 0.0)

    def test_very_large_F_src_no_overflow(self, feed_4x3, all_ones_pedigree):
        F_large = np.array([1e12, 1e12, 1e12, 1e12])
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_large)
        assert np.all(np.isfinite(result.sigma_root))

    def test_list_input_accepted(self, feed_4x3, all_ones_pedigree):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty([100.0, 200.0, 150.0, 80.0])
        assert result.F_root.shape == (3,)

    def test_pedigree_kwarg_overrides_instance_pedigree(self, feed_4x3,
                                                         all_ones_pedigree,
                                                         all_fours_pedigree,
                                                         F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result_override = feed_4x3.apply_with_uncertainty(F_src_4,
                                                          pedigree=all_fours_pedigree)
        assert result_override.cv_src == pytest.approx(all_fours_pedigree.total_cv(),
                                                        rel=RTOL)


# =============================================================================
# 12. ERROR HANDLING
# =============================================================================

class TestErrorHandling:

    def test_no_F_src_and_none_stored_raises(self, feed_4x3, all_ones_pedigree):
        feed_4x3.set_pedigree(all_ones_pedigree)
        with pytest.raises(ValueError, match="No F_src"):
            feed_4x3.apply_with_uncertainty()

    def test_no_pedigree_raises(self, feed_4x3, F_src_4):
        with pytest.raises(ValueError, match="No PedigreeScores"):
            feed_4x3.apply_with_uncertainty(F_src_4)

    def test_pedigree_error_message_mentions_set_pedigree(self, feed_4x3, F_src_4):
        with pytest.raises(ValueError, match="set_pedigree"):
            feed_4x3.apply_with_uncertainty(F_src_4)

    def test_result_is_propagation_result_instance(self, feed_4x3,
                                                    all_ones_pedigree, F_src_4):
        feed_4x3.set_pedigree(all_ones_pedigree)
        result = feed_4x3.apply_with_uncertainty(F_src_4)
        assert isinstance(result, PropagationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
