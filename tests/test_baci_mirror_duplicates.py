"""Two BACI codes that declare the same copper (#88, #95), transit declarations (#95), and
the parse's HS scope (#91).

BACI 178 "Congo" carries DR Congo's refined copper as its partners declare it (#84); 180
is DR Congo's own. In 2015–2023 the two repeat each other's shipments, so per year,
product, partner and direction the parse keeps the larger declaration; outside that
window both stay. DR Congo's refined copper declared into Tanzania and Mozambique only
passes through their ports, and its final buyer declares DR Congo as origin too, so
those legs are dropped in every year.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "baci_parse"))
from baci_parse_feed import (  # noqa: E402
    HS_CODES, copper_price, drop_transit_declarations, reconcile_mirror_duplicates)

CATHODE, OTHER_REFINED, CONC = 740311, 740319, 260300
UAE, CHINA, ITALY, SOUTH_AFRICA, TANZANIA, MOZAMBIQUE = 784, 156, 381, 710, 834, 508
DUP     = {178: (180, 2015, 2023)}
TRANSIT = {(178, 180): (TANZANIA, MOZAMBIQUE)}
CURRENT_CONC = os.path.join(os.path.dirname(__file__), "..", "data", "input", "feed", "baci",
                            "current", "conc.xlsx")


def _agg(rows):
    """rows: (year, sector, exporter, importer, q_t)."""
    return pd.DataFrame([dict(Year=y, Sector=k, Region_from=i, Region_to=j,
                              q=q, v=8.0 * q)
                         for y, k, i, j, q in rows])


def _rows(out):
    return sorted(map(tuple, out[["Year", "Sector", "Region_from", "Region_to", "q"]]
                      .to_numpy().tolist()))


# ---------------------------------------------------------------------------
# mirror duplicates — the larger declaration per partner (#95)
# ---------------------------------------------------------------------------

def test_the_larger_declaration_per_partner_is_kept_and_counted_once():
    """Regression (#95): #88 dropped 178 outright in 2015–2023, and on
    opt_20261019_142026_83bfe843 the UAE re-exported cathode it no longer imported
    (Δz² +25,625). Its 178 declaration (242 kt in 2019) is the larger of the two and
    is kept; 180's 121 kt is the same copper and goes."""
    agg = _agg([(2019, CATHODE,       178, UAE, 242e3),
                (2019, CATHODE,       180, UAE, 106e3),
                (2019, OTHER_REFINED, 180, UAE,  15e3)])
    out = reconcile_mirror_duplicates(agg, DUP)
    assert _rows(out) == [(2019, CATHODE, 178, UAE, 242e3)]


def test_refined_codes_are_pooled_per_partner_not_compared_code_by_code():
    """One shipment declared 740311 by the UAE (under 178) and 740319 by DR Congo (180)
    is still one shipment: a per-code max would keep both, 52–71 kt a year twice in
    2015–2017. The winning code keeps every one of its rows, each under its own HS code."""
    agg = _agg([(2015, CATHODE,       178, UAE,    80e3),
                (2015, CATHODE,       180, UAE,     8e3),
                (2015, OTHER_REFINED, 180, UAE,    52e3),
                (2015, CATHODE,       178, CHINA,  30e3),
                (2015, CATHODE,       180, CHINA,  25e3),
                (2015, OTHER_REFINED, 180, CHINA,  20e3)])
    out = reconcile_mirror_duplicates(agg, DUP)
    assert _rows(out) == [(2015, CATHODE,       178, UAE,   80e3),
                          (2015, CATHODE,       180, CHINA, 25e3),
                          (2015, OTHER_REFINED, 180, CHINA, 20e3)]


def test_a_partner_declared_under_one_code_only_is_kept():
    """Per product and direction: 178's concentrate to China is not compared with 180's
    cathode to China, and 180's import from the UAE is not compared with 178's export
    to it."""
    agg = _agg([(2020, CATHODE, 180, CHINA, 500e3),
                (2020, CONC,    178, CHINA,  18e3),
                (2020, CATHODE, 178, ITALY,  50e3),
                (2020, CATHODE, 178, UAE,   245e3),
                (2020, CATHODE, UAE, 180,     1e3)])
    out = reconcile_mirror_duplicates(agg, DUP)
    assert _rows(out) == _rows(agg)


def test_rows_outside_the_window_are_untouched():
    """Through 2014 and in 2024 180 falls short of ICSG production and 178 fills it."""
    agg = _agg([(2014, CATHODE, 178, UAE, 121e3),
                (2014, CATHODE, 180, UAE,  40e3),
                (2024, CATHODE, 178, UAE, 150e3),
                (2024, CATHODE, 180, UAE, 130e3),
                (2016, CATHODE, 178, UAE, 131e3),
                (2016, CATHODE, 180, UAE,  53e3)])
    out = reconcile_mirror_duplicates(agg, DUP)
    assert _rows(out) == _rows(agg.iloc[[0, 1, 2, 3, 4]])


@pytest.mark.parametrize("code, rows", [
    (178, [(2014, CATHODE, 178, UAE, 121e3), (2019, CATHODE, 180, UAE, 106e3)]),
    (180, [(2019, CATHODE, 178, UAE, 242e3)])])
def test_a_mirror_declaration_matching_no_rows_raises(code, rows):
    with pytest.raises(ValueError, match=f"code {code}"):
        reconcile_mirror_duplicates(_agg(rows), DUP)


# ---------------------------------------------------------------------------
# transit declarations (#95)
# ---------------------------------------------------------------------------

def test_transit_declarations_drop_only_refined_rows_into_the_declared_partners():
    """Dropped in every year, from either DR Congo code. Concentrate (not analysed),
    South Africa's own exports to Tanzania and Tanzania's exports stay."""
    agg = _agg([(2019, CATHODE,       180, TANZANIA,     265e3),
                (2019, OTHER_REFINED, 180, TANZANIA,       5e3),
                (2023, CATHODE,       180, MOZAMBIQUE,   126e3),
                (2012, CATHODE,       178, TANZANIA,      10e3),
                (2019, CONC,          180, TANZANIA,      30e3),
                (2019, CATHODE,       180, CHINA,        500e3),
                (2019, CATHODE,       SOUTH_AFRICA, TANZANIA, 4e3),
                (2019, CATHODE,       TANZANIA, CHINA,    20e3)])
    out = drop_transit_declarations(agg, TRANSIT)
    assert _rows(out) == _rows(agg.iloc[4:])


def test_a_transit_declaration_matching_no_rows_raises():
    agg = _agg([(2019, CATHODE, 180, TANZANIA, 265e3),
                (2019, CONC,    180, MOZAMBIQUE, 30e3)])
    with pytest.raises(ValueError, match=str(MOZAMBIQUE)):
        drop_transit_declarations(agg, TRANSIT)


# ---------------------------------------------------------------------------
# the parse's HS scope (#91)
# ---------------------------------------------------------------------------

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
