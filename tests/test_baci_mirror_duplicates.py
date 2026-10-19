"""A BACI code that repeats copper another code already declares (#88).

BACI 178 "Congo" carries DR Congo's cathode as its partners declare it (#84). DR Congo's
own declarations (180) fall short of ICSG production in 2012–2019 and 2024, which 178
fills; in 2020–2023 they match it, and 178 counts the copper twice. The parse drops a
declared code's rows, either side, in its duplicate years only.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "baci_parse"))
from baci_parse_feed import drop_mirror_duplicates  # noqa: E402

CATHODE, CONC = 740311, 260300
DUP = {178: (2020, 2023)}


def _agg(rows):
    """rows: (year, sector, exporter, importer, q_t)."""
    return pd.DataFrame([dict(Year=y, Sector=k, Region_from=i, Region_to=j,
                              q=q, v=8.0 * q)
                         for y, k, i, j, q in rows])


def test_a_declared_codes_rows_are_dropped_in_its_duplicate_years_only():
    """Regression (#88): on opt_20261015_093646_ddd9d3d9, 178's 2020–2023 cathode sat
    on top of a code 180 that already matched ICSG's production, and DR Congo's refined
    output rose 777 kt CC above its prior over 2020–2023 (183 on the run before #84)."""
    agg = _agg([(2019, CATHODE, 178, 784, 242e3),
                (2020, CATHODE, 178, 784, 245e3),
                (2023, CONC,    178, 156, 18e3),
                (2021, CATHODE, 180, 178, 1e3),
                (2020, CATHODE, 180, 784, 110e3),
                (2024, CATHODE, 178, 784, 150e3)])
    out = drop_mirror_duplicates(agg, DUP)
    assert_array_equal(out[["Year", "Region_from", "Region_to"]].to_numpy(),
                       [[2019, 178, 784], [2020, 180, 784], [2024, 178, 784]])


def test_a_declaration_matching_no_rows_raises():
    agg = _agg([(2019, CATHODE, 178, 784, 242e3)])
    with pytest.raises(ValueError, match="178"):
        drop_mirror_duplicates(agg, DUP)
