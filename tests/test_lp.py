"""
test_lp.py
==========
Unit tests for pysuto.lp — the pure-math power-mean (Lp) σ combiner.
"""
from __future__ import annotations

import numpy as np
import pytest

from pysuto.lp import lp_combine


class TestLpCombine:
    def test_p2_matches_quadrature(self):
        assert lp_combine([3.0, 4.0], p=2.0) == pytest.approx(5.0)

    def test_p1_matches_l1(self):
        assert lp_combine([-2.0, 1.5, 0.5], p=1.0) == pytest.approx(4.0)

    def test_smaller_p_inflates_equal_terms(self):
        # ||1_n||_p = n^(1/p): smaller p → larger value
        x = np.ones(10)
        assert lp_combine(x, p=1.3) > lp_combine(x, p=2.0)

    def test_value_at_the_shipped_exponent(self):
        # ||(1, 1)||_p = 2^(1/p); the shipped p is 1.3
        assert lp_combine([1.0, 1.0], p=1.3) == pytest.approx(2 ** (1 / 1.3), rel=1e-12)

    def test_empty_returns_zero(self):
        assert lp_combine([], p=1.3) == 0.0

    def test_negative_uses_abs(self):
        assert lp_combine([-3.0, -4.0], p=2.0) == pytest.approx(5.0)


class TestLpCombineAxis:
    """``lp_combine_axis`` replaces the inline sums of agg, the σ regularisation and the
    scale-total σ_c; each replacement must keep every float (``array_equal``)."""

    def test_p_default_is_the_documented_exponent(self):
        from pysuto.lp import P_DEFAULT
        assert P_DEFAULT == 1.3

    def test_axis_matches_the_scalar_combiner_per_column(self):
        from pysuto.lp import lp_combine_axis
        x = np.random.default_rng(0).normal(size=(7, 5))
        out = lp_combine_axis(x, 1.3, axis=0)
        for j in range(5):
            assert np.allclose(out[j], lp_combine(x[:, j], p=1.3), rtol=1e-12)

    def test_axis_none_combines_the_whole_array(self):
        from pysuto.lp import lp_combine_axis
        assert lp_combine_axis([3.0, 4.0], 2.0) == pytest.approx(5.0)

    def test_bit_identical_to_the_agg_inline_form(self):
        from pysuto.lp import lp_combine_axis
        rng = np.random.default_rng(1)
        for p in (1.1, 1.3, 1.5):    # the pySUTO range; p = 2 takes numpy's sqrt path
            x = rng.lognormal(size=(9, 400)) * (rng.random((9, 400)) > 0.3)
            old = np.power(np.maximum((np.abs(x) ** p).sum(axis=0), 0.0), 1.0 / p)
            np.testing.assert_array_equal(lp_combine_axis(x, p, axis=0), old)

    def test_bit_identical_to_the_regularisation_inline_form(self):
        from pysuto.constraints.engine import _lp_regularize
        rng = np.random.default_rng(2)
        sigma  = rng.lognormal(size=1000) * (rng.random(1000) > 0.2)
        centre = rng.normal(size=1000) * 10
        for p in (1.1, 1.3, 1.5):
            cv_term = 0.05 * np.abs(centre)
            old = (np.abs(sigma) ** p + np.abs(cv_term) ** p
                   + np.abs(1e-3) ** p) ** (1.0 / p)
            new, _, _ = _lp_regularize(sigma, centre, 0.05, 1e-3, p)
            np.testing.assert_array_equal(new, old)

    def test_bit_identical_to_the_scale_total_nansum_form(self):
        from pysuto.lp import lp_combine_axis
        rng = np.random.default_rng(3)
        for n in (1, 3, 40, 500):
            s = rng.lognormal(size=n)
            s[rng.random(n) > 0.7] = np.nan
            old = float(np.nansum(s ** 1.3) ** (1.0 / 1.3))
            new = float(lp_combine_axis(np.where(np.isnan(s), 0.0, s), 1.3))
            assert new == old
