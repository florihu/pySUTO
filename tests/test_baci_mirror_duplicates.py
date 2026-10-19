"""A BACI code that duplicates another code's declarations from a given year (#88).

BACI 178 "Congo" carries DR Congo's cathode as its partners declare it (#84). Until
2019 it fills DR Congo's own declarations (180), which fall 134–358 kt CC a year short
of ICSG production; from 2020 they cover it, and 178's flows count the same copper a
second time. The parse drops a declared code's rows from its first duplicate year.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "baci_parse"))
from baci_parse_feed import drop_mirror_duplicates  # noqa: E402

CATHODE, CONC = 740311, 260300
DUP = {178: 2020}


def _agg(rows):
    """rows: (year, sector, exporter, importer, q_t)."""
    return pd.DataFrame([dict(Year=y, Sector=k, Region_from=i, Region_to=j,
                              q=q, v=8.0 * q)
                         for y, k, i, j, q in rows])


def test_a_declared_codes_rows_are_dropped_from_its_first_duplicate_year():
    """Regression (#88): 178's 2020–2023 cathode landed on top of a code 180 that
    already matched ICSG's production, and the solver invented 829 kt CC of DR
    Congo's refined output to carry it."""
    agg = _agg([(2019, CATHODE, 178, 784, 242e3),
                (2020, CATHODE, 178, 784, 245e3),
                (2020, CONC,    178, 156, 3e3),
                (2021, CATHODE, 180, 178, 1e3),
                (2020, CATHODE, 180, 784, 110e3)])
    out = drop_mirror_duplicates(agg, DUP)
    assert out[["Year", "Region_from", "Region_to"]].values.tolist() == \
        [[2019, 178, 784], [2020, 180, 784]]


def test_a_declaration_matching_no_rows_raises():
    agg = _agg([(2019, CATHODE, 178, 784, 242e3)])
    with pytest.raises(ValueError, match="178"):
        drop_mirror_duplicates(agg, DUP)
