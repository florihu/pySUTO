"""test_sweep_stats_guards.py — two guards of ``diag.stats.sweep`` (#279, U8-G3-14)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.diag.stats.sweep import Outcome, _measure, sensitivity_table


def test_the_total_measure_is_nan_over_a_zero_base_not_infinite():
    o = Outcome("d_total", "total")
    idx = np.array([0, 1])
    p, b = np.array([3.0, 1.0]), np.array([0.0, 0.0])
    assert np.isnan(_measure(o, idx, p, p, b, b))
    assert _measure(o, idx, np.array([3.0, 1.0]), p, np.array([2.0, 2.0]), b) == \
        pytest.approx(0.0, abs=1e-12)


def test_a_non_numeric_outcome_column_raises_instead_of_turning_nan():
    frame = pd.DataFrame({"imp.P_PROP": [1.1, 1.2, 1.3], "y": ["0.1", "oops", "0.3"]})
    axes = [{"keys": ["imp.P_PROP"], "distribution": object()}]
    with pytest.raises(ValueError):
        sensitivity_table(frame, axes, ["y"], "random")
