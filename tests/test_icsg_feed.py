"""Unit tests for the ICSG feed's observed zeros (#76)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from icsg_feed import observed_zero_sigma  # noqa: E402


def _legs(*pairs):
    return pd.DataFrame(pairs, columns=["src_id", "edge_id"])


def test_a_small_fan_prices_its_zero_at_the_floor():
    sigma = observed_zero_sigma(_legs((0, 1)), np.array([10.0]), np.zeros(3))
    assert sigma[1] == pytest.approx(0.3)
    assert np.isnan(sigma[[0, 2]]).all()


def test_a_large_fan_prices_its_zero_at_one_percent_of_its_usage():
    sigma = observed_zero_sigma(_legs((0, 1)), np.array([100.0]), np.zeros(3))
    assert sigma[1] == pytest.approx(1.0)


def test_a_leg_another_fan_feeds_is_not_an_observed_zero():
    sigma = observed_zero_sigma(_legs((0, 1)), np.array([100.0]), np.array([0.0, 2.5, 0.0]))
    assert np.isnan(sigma).all()


def test_a_leg_several_fans_weigh_at_zero_is_priced_on_their_summed_usage():
    # an aggregate region (ROW) fans each member country's usage onto the same legs
    sigma = observed_zero_sigma(_legs((0, 1), (1, 1)), np.array([50.0, 400.0]), np.zeros(2))
    assert sigma[1] == pytest.approx(4.5)
