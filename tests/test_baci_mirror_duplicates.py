"""A BACI code that repeats copper another code already declares (#88), and the parse's
HS scope (#91).

BACI 178 "Congo" carries DR Congo's refined copper as its partners declare it (#84). DR
Congo's own declarations (180, all of HS 7403.1x, #91) fall short of ICSG production
through 2014 and in 2024, which 178 fills; in 2015–2023 they reach or exceed it, and 178
counts the copper twice. The parse drops a declared code's rows, either side, in its
duplicate years only.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest
from numpy.testing import assert_array_equal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "baci_parse"))
from baci_parse_feed import HS_CODES, copper_price, drop_mirror_duplicates  # noqa: E402

CATHODE, CONC = 740311, 260300
DUP = {178: (2015, 2023)}
CURRENT_CONC = os.path.join(os.path.dirname(__file__), "..", "data", "input", "feed", "baci",
                            "current", "conc.xlsx")


def _agg(rows):
    """rows: (year, sector, exporter, importer, q_t)."""
    return pd.DataFrame([dict(Year=y, Sector=k, Region_from=i, Region_to=j,
                              q=q, v=8.0 * q)
                         for y, k, i, j, q in rows])


def test_a_declared_codes_rows_are_dropped_in_its_duplicate_years_only():
    """Regression (#88): on opt_20261015_093646_ddd9d3d9, 178's 2020–2023 cathode sat
    on top of a code 180 that already matched ICSG's production, and DR Congo's refined
    output rose 777 kt CC above its prior over 2020–2023 (183 on the run before #84)."""
    agg = _agg([(2014, CATHODE, 178, 784, 121e3),
                (2015, CATHODE, 178, 784, 80e3),
                (2023, CONC,    178, 156, 18e3),
                (2021, CATHODE, 180, 178, 1e3),
                (2020, CATHODE, 180, 784, 110e3),
                (2024, CATHODE, 178, 784, 150e3)])
    out = drop_mirror_duplicates(agg, DUP)
    assert_array_equal(out[["Year", "Region_from", "Region_to"]].to_numpy(),
                       [[2014, 178, 784], [2020, 180, 784], [2024, 178, 784]])


def test_a_declaration_matching_no_rows_raises():
    agg = _agg([(2014, CATHODE, 178, 784, 121e3)])
    with pytest.raises(ValueError, match="178"):
        drop_mirror_duplicates(agg, DUP)


def test_the_parse_reads_every_code_the_tracked_sector_map_maps():
    """The parse's code list and the concordance's sector map are one scope (#91)."""
    smap = pd.read_excel(CURRENT_CONC, sheet_name="sector_map")
    assert HS_CODES == set(smap["src_sector"].astype(int))
    assert {740311, 740312, 740313, 740319} <= HS_CODES


def test_the_copper_price_stays_the_cathode_unit_value():
    """Wire bars, billets and other refined copper join the Cathode sector, but ADR 0033's
    copper price stays HS 740311's unit value: a 740319 leg at another price must not
    move it (#91)."""
    agg = pd.DataFrame([dict(Year=2003, Sector=740311, q=1000.0, v=1800.0),
                        dict(Year=2003, Sector=740319, q=1000.0, v=9000.0)])
    assert copper_price(agg).to_dict() == {2003: pytest.approx(1800.0)}
