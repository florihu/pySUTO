"""test_conditioning.py
=====================
The σ conditioning rules of :mod:`pysuto._conditioning` (U4-S1-1, #284): the σ_a
floor and ceiling, the σ_c floor and hard mask, and the refusal of a NaN.
"""
from __future__ import annotations

import numpy as np
import pytest

from pysuto._conditioning import condition_prior, condition_sigma_c


class TestConditionPrior:

    def test_the_floor_is_the_larger_of_the_absolute_and_the_relative_one(self):
        a_0 = np.array([0.0, 1e6, 10.0])
        sig = np.array([1e-12, 1e-12, 4.0])
        a, s = condition_prior(a_0, sig, floor_abs=1e-8, floor_rel=1e-5)
        np.testing.assert_array_equal(a, a_0)
        # |a_0| = 0 -> abs floor; |a_0| = 1e6 -> 1e-5 * 1e6 = 10; a σ above its floor stays
        assert np.allclose(s, [1e-8, 10.0, 4.0], rtol=1e-12)

    def test_the_ceiling_is_the_factor_times_the_largest_finite_positive_sigma(self):
        sig = np.array([2.0, 5.0, 1e40, 0.0])        # 1e40 and 0 do not set the ceiling
        _, s = condition_prior(np.zeros(4), sig, ceiling_factor=10.0)
        assert s[0] == 2.0 and s[1] == 5.0
        assert s[2] == pytest.approx(50.0, rel=1e-12)    # capped at 10 * 5
        assert s[3] == pytest.approx(1e-8, rel=1e-12)    # lifted to the floor

    def test_inputs_are_not_mutated(self):
        a_0, sig = np.array([1.0, 2.0]), np.array([1e-12, 3.0])
        condition_prior(a_0, sig)
        np.testing.assert_array_equal(a_0, [1.0, 2.0])
        np.testing.assert_array_equal(sig, [1e-12, 3.0])

    @pytest.mark.parametrize("which", ["a_0", "sigma_a"])
    def test_a_nan_raises_and_names_its_position(self, which):
        a_0, sig = np.ones(4), np.ones(4)
        {"a_0": a_0, "sigma_a": sig}[which][2] = np.nan
        with pytest.raises(ValueError, match=rf"NaN entries in {which}.*\[2\]"):
            condition_prior(a_0, sig)


class TestConditionSigmaC:

    def test_values_below_the_floor_are_lifted_to_it(self):
        s = condition_sigma_c(np.array([0.0, 1e-12, 0.5]), floor=1e-9)
        assert np.allclose(s, [1e-9, 1e-9, 0.5], rtol=1e-12)

    def test_the_hard_mask_zeroes_hard_rows_after_flooring(self):
        s = condition_sigma_c(np.array([0.0, 0.5]), hard_mask=np.array([True, False]))
        np.testing.assert_array_equal(s, [0.0, 0.5])

    def test_a_nan_raises_also_under_a_hard_mask(self):
        with pytest.raises(ValueError, match="NaN entries in sigma_c"):
            condition_sigma_c(np.array([np.nan, 0.5]), hard_mask=np.array([True, False]))
