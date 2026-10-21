"""Which source type a value of a merged 2005 yearbook cell belongs to.

The 2005 refinery table merges a country's source types into one cell, and the
cell lists every type the country reports in some year: Russia's reads
'Electrowon Primary Secondary' over '500.0 / 50.0' in 1995, a year the
yearbook's SX-EW table gives Russia no output.  Paired positionally, the 500 kt
of primary refined copper became electrowon, which the combination step drops
for the refinery table, and the 50 kt of secondary became primary: Russia's
1995-2003 refinery intake fell to 50-296 kt against a smelter output of
545-842 kt.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "icsg_parse"))
from icsg_2005 import clean_flows, sxew_region_years  # noqa: E402

_RENAME = {"COUNTRY": "Region", "Feed Source  1/": "Type", "Semis- Type": "Type"}


def _sheet(rows):
    return pd.DataFrame(rows, columns=["COUNTRY", "Feed Source  1/", "1995", "1996"])


def _refining(rows, sxew):
    return clean_flows(_sheet(rows), "Refining_2", "Refining", "Supply", _RENAME, sxew=sxew)


def _value(df, region, typ, year):
    hit = df[(df["Region"] == region) & (df["Type"] == typ) & (df["Year"] == year)]
    return None if hit.empty else hit["Value"].iloc[0]


_RUSSIA = [["Russian Fed.", "Electrowon Primary Secondary", "500.0_x000d_\n50.0",
            "513.0_x000d_\n57.0"],
           ["Russian Fed.", "Total", 550.0, 570.0]]


def test_a_cell_without_sx_ew_output_pairs_its_values_with_the_other_types():
    df = _refining(_RUSSIA, sxew=set())
    assert _value(df, "Russia", "Primary", 1995) == 500.0
    assert _value(df, "Russia", "Secondary", 1995) == 50.0
    assert _value(df, "Russia", "Secondary", 1996) == 57.0
    assert _value(df, "Russia", "SX-EW", 1995) is None


def test_a_cell_whose_region_year_reports_sx_ew_keeps_the_positional_pairing():
    df = _refining(_RUSSIA, sxew={("Russia", 1995)})
    assert _value(df, "Russia", "SX-EW", 1995) == 500.0
    assert _value(df, "Russia", "Primary", 1995) == 50.0
    assert _value(df, "Russia", "Primary", 1996) == 513.0     # 1996 has no SX-EW


def test_a_cell_with_a_value_for_every_type_is_unchanged():
    rows = [["Iran", "Electrowon Primary Secondary", "1.5_x000d_\n90.0_x000d_\n4.0",
             "9.5_x000d_\n95.0_x000d_\n3.0"]]
    df = _refining(rows, sxew=set())
    assert _value(df, "Iran", "SX-EW", 1995) == 1.5
    assert _value(df, "Iran", "Primary", 1995) == 90.0
    assert _value(df, "Iran", "Secondary", 1996) == 3.0


def test_the_sx_ew_table_gives_the_region_years_it_reports():
    sheet = pd.DataFrame([["Russian Fed.", None, None], ["Iran", None, 1.5],
                          ["Congo   1/", 33.4, 43.8]], columns=["COUNTRY", "1995", "1996"])
    df = clean_flows(sheet, "SXEW", "SXEW", "Supply", _RENAME)
    assert sxew_region_years(df) == {("Iran", 1996), ("Congo", 1995), ("Congo", 1996)}
