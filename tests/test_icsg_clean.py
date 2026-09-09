"""ICSG production series → the input each one consumed (ADR 0021).

The refined series measures what the *electrolytic refinery* made, so it maps
to the anode entering ER.  Placing it two stages back, on the matte entering
CFR, assumed every anode a region makes is refined at home — false for the
anode/blister exporters (Zambia, Bulgaria, Congo D.R.), whose matte was then
left with no anchored outlet.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "icsg_parse"))
from icsg_clean import apply_input_divisors, refining_anode_rows  # noqa: E402

_COLS = ["Region", "Type", "Year", "Value", "Process", "Supply_Use", "Unit",
         "Year_loaded_from_file"]
# eff_ER 0.997, eff_CFR 0.9875, eff_Smelting 0.965 — the live nominals
_EFF = pd.DataFrame({"Process": ["Smelting", "CFR", "ER"],
                     "Value_median": [0.965, 0.9875, 0.997]})


def _row(region, proc, typ, value, year=2020, su="Supply"):
    return dict(Region=region, Type=typ, Year=year, Value=value, Process=proc,
                Supply_Use=su, Unit="kt", Year_loaded_from_file=2024)


def _frame(rows):
    return pd.DataFrame(rows)[_COLS]


def test_anode_row_is_total_refined_not_primary_only():
    """Scrap-derived copper reaches ER as anode too, so the ER anchor is
    primary + secondary — 19 % of world refined output rides on that."""
    df = _frame([_row("Germany", "Refining", "Primary", 300.0),
                 _row("Germany", "Refining", "Secondary", 200.0)])
    a = refining_anode_rows(df)
    assert len(a) == 1
    assert a["Type"].iloc[0] == "Anode"
    assert np.isclose(a["Value"].iloc[0], 500.0)


def test_secondary_only_refiner_gets_an_anchor():
    """Austria refines 100 % secondary; under the old placement its ER feed
    had no anchor at all."""
    df = _frame([_row("Austria", "Refining", "Secondary", 103.0)])
    a = refining_anode_rows(df)
    assert len(a) == 1 and np.isclose(a["Value"].iloc[0], 103.0)


def test_anode_rows_are_per_region_year():
    df = _frame([_row("Zambia", "Refining", "Primary", 10.0, year=2021),
                 _row("Zambia", "Refining", "Primary", 20.0, year=2022),
                 _row("Chile",  "Refining", "Primary", 30.0, year=2021)])
    a = refining_anode_rows(df)
    assert len(a) == 3
    assert set(zip(a["Region"], a["Year"])) == {("Zambia", 2021), ("Zambia", 2022),
                                                ("Chile", 2021)}


def test_each_series_divided_by_the_stages_it_crosses():
    df = _frame([_row("Zambia", "Smelting", "Primary", 965.0),
                 _row("Zambia", "Refining", "Primary", 199.4),
                 _row("Zambia", "Refining", "Secondary", 0.0)])
    out = apply_input_divisors(df, _EFF)

    sm = out[(out["Process"] == "Smelting") & (out["Type"] == "Primary")]
    assert np.isclose(sm["Value"].iloc[0], 965.0 / 0.965)        # ÷ eff_Smelting

    an = out[(out["Process"] == "Refining") & (out["Type"] == "Anode")]
    assert np.isclose(an["Value"].iloc[0], 199.4 / 0.997)        # ÷ eff_ER only

    sc = out[(out["Process"] == "Refining") & (out["Type"] == "Secondary")]
    assert np.isclose(sc["Value"].iloc[0], 0.0 / (0.9875 * 0.997))
    # every converted series becomes a Use row
    assert (out["Supply_Use"] == "Use").all()


def test_refined_series_no_longer_lands_on_matte():
    """The regression this change is about: nothing maps the refined series to
    the matte entering CFR any more, so the Matte node keeps only the outlet
    its hard balance gives it."""
    df = _frame([_row("Bulgaria", "Refining", "Primary", 57.0),
                 _row("Bulgaria", "Refining", "Secondary", 3.0)])
    out = apply_input_divisors(df, _EFF)
    assert "Matte" not in set(out["Type"])
    assert "Primary" not in set(out[out["Process"] == "Refining"]["Type"])
    # the secondary row survives as the Scrap→CFR anchor
    assert (out["Type"] == "Secondary").sum() == 1


def test_anode_anchor_exceeds_the_old_matte_anchor_for_a_home_refiner():
    """Germany refines ~49 % secondary: the old placement anchored 300/(eff_CFR
    ·eff_ER); the new one anchors the full 500 one stage later."""
    df = _frame([_row("Germany", "Refining", "Primary", 300.0),
                 _row("Germany", "Refining", "Secondary", 200.0)])
    out = apply_input_divisors(df, _EFF)
    new = out.loc[out["Type"] == "Anode", "Value"].iloc[0]
    old = 300.0 / (0.9875 * 0.997)
    assert new > old
    assert np.isclose(new, 500.0 / 0.997)
