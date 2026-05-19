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

    def test_empty_returns_zero(self):
        assert lp_combine([], p=1.3) == 0.0

    def test_negative_uses_abs(self):
        assert lp_combine([-3.0, -4.0], p=2.0) == pytest.approx(5.0)
