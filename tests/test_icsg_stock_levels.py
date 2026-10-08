"""
test_icsg_stock_levels.py
=========================
The yearbook parsers return every stock level ICSG prints (#35).

A stock level is the end-of-period quantity ICSG reports for a region and one
stock type; the net addition to stock computed from it is what a stock-change
leg carries (GLOSSARY.md).  Each test feeds a parser a toy sheet shaped as
``read_excel`` hands it over and checks one series that used to be lost.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "icsg_parse"))
import icsg_2005  # noqa: E402
import icsg_2014  # noqa: E402
import icsg_2023f  # noqa: E402


def _level(df, region, typ, year):
    hit = df[(df["Region"] == region) & (df["Type"] == typ) & (df["Year"] == year)]
    assert len(hit) <= 1, f"{region}/{typ}/{year}: {len(hit)} rows"
    return None if hit.empty else hit["Value"].iloc[0]


# ======================================================================
# 2014 yearbook, exchange stocks (Table 8A, read with header=2)
# ======================================================================

def _exchange_sheet_2014():
    """The loader already took the year row as column labels."""
    return pd.DataFrame(
        [["COUNTRY",               np.nan, np.nan],
         ["LONDON METAL EXCHANGE", np.nan, np.nan],
         ["Belgium",               0.0,    1.075],
         ["United States",         35.0,   20.5],
         [" Total LME",            35.0,   21.575],
         ["COMEX",                 43.7,   50.0],
         ["SHFE",                  31.7,   40.0],
         ["TOTAL EXCHANGES",       110.4,  111.575]],
        columns=["STOCK TYPE AND", 2004, 2005])


def test_2014_exchange_sheet_yields_a_level_per_location_and_year():
    """Regression: the cleaner relabelled the columns from a data row, found no
    year column and returned nothing, so MESC had no 2005-2013 levels."""
    df = icsg_2014.clean_exchange_stocks(_exchange_sheet_2014(), id=2014)
    assert _level(df, "United States", "LME", 2005) == pytest.approx(20.5)
    assert _level(df, "United States", "COMEX", 2004) == pytest.approx(43.7)
    assert _level(df, "China", "SHFE", 2005) == pytest.approx(40.0)
    assert set(df["Stock_type"]) == {"Exchanges"}
    assert not df["Region"].str.contains("total", case=False).any()


def test_a_reported_zero_level_is_kept_and_a_blank_is_not():
    """A zero is a report; dropping it turned Belgium's 0 -> 1.075 kt into a
    gap, and a gap yields no net addition to stock."""
    sheet = _exchange_sheet_2014()
    sheet.loc[3, 2005] = np.nan                       # United States LME blank
    df = icsg_2014.clean_exchange_stocks(sheet, id=2014)
    assert _level(df, "Belgium", "LME", 2004) == 0.0
    assert _level(df, "United States", "LME", 2005) is None


# ======================================================================
# 2014 yearbook, producer / consumer / merchant stocks (Table 8B)
# ======================================================================

def _pmc_sheet_2014():
    """read_excel(header=2) leaves the year row as the first data row."""
    rows = [["STOCK TYPE AND",                np.nan,    2004,  2005.0, 2006],
            [np.nan,                          "COUNTRY", np.nan, np.nan, np.nan],
            ["China (total) 1/",              np.nan,    30.0,  47.0,   73.0],
            ["Germany Total  3/",             np.nan,    70.0,  70.0,   70.0],
            ["Iran (producers)",              np.nan,    np.nan, 0.0,   5.0],
            ["UU.SS. (p(prorodduucceersrs))", np.nan,    1010.33, 828.2, 2828.11],
            ["U.S. (consumers)",              np.nan,    45.3,  50.8,   61.7],
            ["U.S. Total",                    np.nan,    55.7,  59.0,   89.8],
            ["TOTAL COUNTRY STOCKS",          np.nan,    741.4, 653.5,  821.8]]
    return pd.DataFrame(rows, columns=["Thousand metric tonnes", "a", "b", "c", "d"])


def test_2014_lone_totals_are_kept():
    """China, Germany and the Philippines print only a total; a label with
    'total' in it was dropped, so their stocks never reached the account."""
    df = icsg_2014.clean_pmc_stocks(_pmc_sheet_2014())
    assert _level(df, "China", "Total", 2005) == pytest.approx(47.0)
    assert _level(df, "Germany", "Total", 2006) == pytest.approx(70.0)


def test_2014_us_producers_is_total_minus_consumers():
    """The row prints with superimposed digits (1010.33 for 10.4), which put
    +/-2 Mt producer-stock changes into the feed."""
    df = icsg_2014.clean_pmc_stocks(_pmc_sheet_2014())
    assert _level(df, "U.S.", "Producers", 2004) == pytest.approx(10.4)
    assert _level(df, "U.S.", "Producers", 2005) == pytest.approx(8.2)
    assert _level(df, "U.S.", "Producers", 2006) == pytest.approx(28.1)
    assert _level(df, "U.S.", "Total", 2006) == pytest.approx(89.8)


def test_2014_reported_zero_producer_level_is_kept():
    df = icsg_2014.clean_pmc_stocks(_pmc_sheet_2014())
    assert _level(df, "Iran", "Producers", 2005) == 0.0
    assert _level(df, "Iran", "Producers", 2004) is None


# ======================================================================
# 2005 yearbook (read with header=0), the cleaners the 2014 ones copy
# ======================================================================

def test_2005_lone_totals_are_kept():
    """China's only stock series 1995-2004 is its printed total."""
    sheet = pd.DataFrame([["STOCK TYPE AND COUNTRY", 1995,  1996],
                          ["China (total)",          126.0, 99.0],
                          ["Philippines Total",      7.8,   6.8],
                          ["TOTAL COUNTRY STOCKS",   624.5, 639.6]],
                         columns=["TABLE 8B", "a", "b"])
    df = icsg_2005.clean_pmc_stocks(sheet)
    assert _level(df, "China", "Total", 1996) == pytest.approx(99.0)
    assert _level(df, "Philippines", "Total", 1995) == pytest.approx(7.8)


def test_2005_reported_zero_exchange_level_is_kept():
    """Korea's LME stock is 0 until 2004; dropping the zeros left no pair to
    difference when it reached 1.5 kt."""
    sheet = pd.DataFrame([["Thousand metric tonnes",  np.nan,   np.nan],
                          ["STOCK TYPE AND COUNTRY",  1995,     1996],
                          ["LONDON METAL EXCHANGE",   np.nan,   np.nan],
                          ["Korean rep.",             "0.000",  "1,500"],
                          ["Total LME",               "0.000",  "1,500"],
                          ["COMEX",                   "21,740", "26,750"],
                          ["SHFE",                    "0.000",  "25,082"],
                          ["TOTAL EXCHANGES",         "21,740", "53,332"]],
                         columns=["TABLE 8A", "a", "b"])
    df = icsg_2005.clean_exchange_stocks(sheet, id=2005)
    assert _level(df, "Korean rep.", "LME", 1995) == 0.0
    assert _level(df, "Korean rep.", "LME", 1996) == pytest.approx(1.5)
    assert _level(df, "China", "SHFE", 1995) == 0.0


# ======================================================================
# 2023 / 2025 yearbooks, one stock sheet (read with header=2)
# ======================================================================

def _stock_sheet_2023(avg_column):
    rows = [["COUNTRY",                                  np.nan, np.nan],
            ["METAL EXCHANGES",                          np.nan, np.nan],
            ["LME",                                      np.nan, np.nan],
            ["Belgium",                                  0.5,    0.7],
            ["COMEX",                                    80.0,   90.0],
            ["SHFE",                                     60.0,   70.0],
            ["COUNTRY STOCKS",                           np.nan, np.nan],
            ["South Africa (producers) 1/",              13.0,   13.0],
            ["Japan Total (only total data since 2021)", 110.0,  94.4],
            ["United Kingdom (consumers) 1/",            7.0,    7.0],
            ["U.S. Total",                               55.0,   60.0],
            ["REFINED STOCK SUMMARY",                    np.nan, np.nan]]
    df = pd.DataFrame(rows, columns=["STOCK TYPE AND", 2023, 2024])
    if avg_column:
        df["Unnamed: 3"] = ["AVG"] + [1.0] * (len(rows) - 1)
    return df


def test_2025_sheet_keeps_its_last_year():
    """The 2025 sheet ends with 2024; the 2023 sheet ends with an averages
    column.  Cutting the last column by position lost 2024 for every type."""
    df = icsg_2023f.clean_stock(_stock_sheet_2023(avg_column=False))
    assert _level(df, "Belgium", "LME", 2024) == pytest.approx(0.7)
    assert _level(df, "South Africa", "Producers", 2024) == pytest.approx(13.0)


def test_2023_sheet_drops_its_averages_column():
    df = icsg_2023f.clean_stock(_stock_sheet_2023(avg_column=True))
    assert set(df["Year"]) == {2023, 2024}


def test_2023_region_is_the_whole_name_before_the_type():
    """The first word of 'South Africa (producers)' is 'South', which the
    combine step discards as an incomplete entry."""
    df = icsg_2023f.clean_stock(_stock_sheet_2023(avg_column=True))
    pmc = set(df.loc[df["Stock_type"] == "PMC", "Region"])
    assert pmc == {"South Africa", "Japan", "United Kingdom", "U.S."}
    assert _level(df, "Japan", "Total", 2024) == pytest.approx(94.4)
