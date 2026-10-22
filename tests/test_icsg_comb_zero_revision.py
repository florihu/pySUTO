"""A later yearbook's revision to zero supersedes the older vintage's value (#111).

The 2025 yearbook revised DR Congo's primary refining to 0 for 2015-2024.  The
parsers dropped exact zeros, so the 2023 yearbook's 35 kt survived the
newest-vintage dedup and the feed read a refinery the source no longer has.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "icsg_parse"))
from icsg_comb import latest_vintage, remove_total_if_specific_exists  # noqa: E402

_COLS = ["Region", "Type", "Year", "Value", "Process", "Supply_Use", "Unit",
         "Year_loaded_from_file"]
_KEY = [c for c in _COLS if c not in ("Year_loaded_from_file", "Value")]


def _row(value, typ="Primary", vintage=2023, region="Congo, D.R.", year=2019):
    return dict(Region=region, Type=typ, Year=year, Value=value, Process="Refining",
                Supply_Use="Supply", Unit="kt", Year_loaded_from_file=vintage)


def _frame(*rows):
    return pd.DataFrame(list(rows), columns=_COLS)


def test_a_revision_to_zero_removes_the_older_value():
    out = latest_vintage(_frame(_row(35.0, vintage=2023), _row(0.0, vintage=2025)), _KEY)
    assert out.empty


def test_the_order_of_the_rows_does_not_matter():
    out = latest_vintage(_frame(_row(0.0, vintage=2025), _row(35.0, vintage=2023)), _KEY)
    assert out.empty


def test_a_newer_value_supersedes_an_older_one():
    out = latest_vintage(_frame(_row(35.0, vintage=2023), _row(40.0, vintage=2025)), _KEY)
    assert len(out) == 1
    assert out["Value"].iloc[0] == pytest.approx(40.0)


def test_a_zero_with_nothing_older_leaves_no_row():
    assert latest_vintage(_frame(_row(0.0, vintage=2025)), _KEY).empty


def test_other_cells_are_untouched():
    out = latest_vintage(_frame(_row(35.0, vintage=2023), _row(0.0, vintage=2025),
                                _row(757.5, typ="SX-EW", vintage=2025)), _KEY)
    assert list(out["Type"]) == ["SX-EW"]
    assert out["Value"].iloc[0] == pytest.approx(757.5)


def test_a_zero_specific_does_not_drop_the_total():
    out = remove_total_if_specific_exists(_frame(_row(0.0), _row(15.0, typ="Total")))
    assert out.loc[out["Type"] == "Total", "Value"].tolist() == [15.0]


def test_a_nonzero_specific_still_drops_the_total():
    out = remove_total_if_specific_exists(_frame(_row(12.0), _row(15.0, typ="Total")))
    assert out["Type"].tolist() == ["Primary"]
