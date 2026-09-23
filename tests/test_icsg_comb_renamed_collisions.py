"""Two feed streams renamed onto one type must add, not shadow each other.

`REN_TYPE` folds ICSG's 'Low-grade' smelter feed into 'Primary', so a region
reporting both ends up with two rows of the same vintage sharing a key.  The
cross-vintage dedup that follows exists to keep the newest *restatement* of a
series; it cannot tell a restatement from a second stream, and dropped one —
reading Zambia's primary smelter 2004-2007 as 20 / 10 / 10 / 1.8 kt against
mine production of 320-350, which no refinery could have consumed.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "icsg_parse"))
from icsg_comb import REN_TYPE, sum_renamed_collisions, type_ren  # noqa: E402

_COLS = ["Region", "Type", "Year", "Value", "Process", "Supply_Use", "Unit",
         "Year_loaded_from_file"]
_KEY = [c for c in _COLS if c not in ("Year_loaded_from_file", "Value")]


def _row(typ, value, region="Zambia", proc="Smelting", year=2004, vintage=2014):
    return dict(Region=region, Type=typ, Year=year, Value=value, Process=proc,
                Supply_Use="Supply", Unit="kt", Year_loaded_from_file=vintage)


def _only(df, **where):
    hit = df
    for k, v in where.items():
        hit = hit[hit[k] == v]
    assert len(hit) == 1, f"expected one row, got {len(hit)}"
    return hit["Value"].iloc[0]


def test_low_grade_is_renamed_onto_primary():
    """The premise: the rename is what creates the collision."""
    assert REN_TYPE[("Low-grade", "Smelting")] == "Primary"


def test_the_two_streams_add():
    f = pd.DataFrame([_row("Primary", 280.1), _row("Low-grade", 20.0)],
                     columns=_COLS)
    _, f = type_ren(pd.DataFrame({"Type": [], "Region": []}), f)
    out = sum_renamed_collisions(f, _KEY)
    assert len(out) == 1
    assert _only(out, Type="Primary") == pytest.approx(300.1)


def test_neither_stream_is_silently_dropped():
    """Regression: keep='last' kept 20.0 and discarded 280.1."""
    f = pd.DataFrame([_row("Primary", 280.1), _row("Low-grade", 20.0)],
                     columns=_COLS)
    _, f = type_ren(pd.DataFrame({"Type": [], "Region": []}), f)
    out = sum_renamed_collisions(f, _KEY)
    assert out["Value"].iloc[0] not in (pytest.approx(280.1),
                                        pytest.approx(20.0))


def test_a_restatement_across_vintages_is_left_to_the_dedup():
    """Same key, DIFFERENT vintage: these must stay two rows so the dedup can
    keep the newer one.  Summing them would double-count a restatement."""
    f = pd.DataFrame([_row("Primary", 245.0, vintage=2005),
                      _row("Primary", 280.1, vintage=2014)], columns=_COLS)
    out = sum_renamed_collisions(f, _KEY)
    assert len(out) == 2
    assert set(out["Value"].round(1)) == {245.0, 280.1}


def test_rows_that_do_not_collide_are_untouched():
    f = pd.DataFrame([_row("Primary", 280.1), _row("Secondary", 12.0),
                      _row("Primary", 244.8, year=2005)], columns=_COLS)
    out = sum_renamed_collisions(f, _KEY)
    assert len(out) == 3
    assert _only(out, Type="Primary", Year=2004) == pytest.approx(280.1)
    assert _only(out, Type="Secondary") == pytest.approx(12.0)


def test_the_column_set_and_order_survive():
    f = pd.DataFrame([_row("Primary", 280.1), _row("Low-grade", 20.0)],
                     columns=_COLS)
    _, f = type_ren(pd.DataFrame({"Type": [], "Region": []}), f)
    assert list(sum_renamed_collisions(f, _KEY).columns) == _COLS
