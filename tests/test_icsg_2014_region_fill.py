"""Which country a row belongs to, when the yearbook's label is not on it.

In the ICSG 2014 mine table a country that reports two source types occupies
three rows and its name is typed on the *middle* one, so a plain backfill
reaches the right label from the Concentrates row above but carries the Total
row onto the next country.  Ten years of Australia's mine total were booked as
Azerbaijan's that way, 899 kt/yr against the 0.3-2.6 kt/yr Azerbaijan actually
mines; the same slip gave Russia's total to Saudi Arabia, Kazakhstan's to Laos
and Congo D.R.'s to Cyprus.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "icsg_parse"))
from icsg_2014 import clean_flows  # noqa: E402

_RENAME = {"COUNTRY": "Region", "Source": "Type"}


def _sheet(rows):
    """A mine sheet as read_excel hands it over: blank labels stay blank."""
    return pd.DataFrame(rows, columns=["COUNTRY", "Source", "2004", "2005"])


def _mine(rows):
    return clean_flows(_sheet(rows), "Mining_1", "Mining", "Supply", _RENAME)


def _value(df, region, typ, year=2004):
    hit = df[(df["Region"] == region) & (df["Type"] == typ)
             & (df["Year"] == year)]
    return None if hit.empty else hit["Value"].iloc[0]


# the shape the yearbook actually uses: label on the SX-EW row
_BLOCK = [[None,        "Concentrates", 795.8, 865.3],
          ["Australia", "SX-EW",         58.3,  50.9],
          [None,        "Total",        854.1, 916.2],
          ["Azerbeijan", "Concentrates", None,  None]]


def test_total_row_belongs_to_the_block_above_it():
    """Once the total reaches its true owner the existing drop rule removes it
    as redundant — Australia already reports both of its source types."""
    df = _mine(_BLOCK)
    assert _value(df, "Australia", "Total") is None
    assert not (df["Value"] == pytest.approx(854.1)).any()


def test_total_row_is_not_carried_onto_the_next_country():
    df = _mine(_BLOCK)
    assert _value(df, "Azerbeijan", "Total") is None


def test_rows_above_the_label_still_reach_it():
    """The backfill is right for the Concentrates row — do not regress it."""
    df = _mine(_BLOCK)
    assert _value(df, "Australia", "Concentrates") == pytest.approx(795.8)


def test_a_country_reporting_one_source_is_untouched():
    df = _mine([["Albania", "Concentrates", 1.2, 1.4]])
    assert _value(df, "Albania", "Concentrates") == pytest.approx(1.2)


def test_the_sheets_own_world_total_is_not_given_to_a_country():
    """WORLD/TOTALS is shaped like a country block, so the label must be
    filled before the aggregate rows are dropped — otherwise the grand total
    attaches to whichever country stood above it."""
    df = _mine([["Zimbabwe", "Primary",    7.2,     7.2],
                [None,       "Electrowon", 2705.8,  2693.7],
                ["WORLD",    "Primary",   11142.5, 11717.3],
                ["TOTALS",   "Secondary",  2069.4,  2160.7],
                [None,       "Total",     15917.8, 16571.7]])
    assert _value(df, "Zimbabwe", "Total") is None
    assert not df["Region"].isin(["WORLD", "TOTALS"]).any()
    assert df["Value"].max() == pytest.approx(7.2)


def test_the_block_total_survives_when_its_owner_reports_nothing_else():
    """The existing drop rule only removes a Total that a specific type
    duplicates; a genuine lone total must still come through."""
    df = _mine([[None,     "Concentrates", None, None],
                ["Panama", "SX-EW",        None, None],
                [None,     "Total",        12.0, 13.0],
                ["Peru",   "Concentrates", 1.0,  1.0]])
    assert _value(df, "Panama", "Total") == pytest.approx(12.0)
