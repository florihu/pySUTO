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
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "icsg_parse"))
from icsg_clean import (aggregate_splits, apply_input_divisors,  # noqa: E402
                        merge_territories, refining_anode_rows, secondary_excess,
                        split_aggregates, territory_aliases)

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


# ---------------------------------------------------------------------------
# ADR 0030 — secondary-smelter copper enters the chain once
# ---------------------------------------------------------------------------
_K = 0.9875 * 0.997


def test_refinery_scrap_is_the_direct_route_only():
    """Austria 2024: 60 kt secondary smelter inside 123 kt secondary refined
    (= total refined). The refinery scrap input is the difference, not S_ref."""
    df = _frame([_row("Austria", "Smelting", "Secondary", 60.0),
                 _row("Austria", "Refining", "Secondary", 123.0)])
    out = apply_input_divisors(df, _EFF)
    sc = out[(out["Process"] == "Refining") & (out["Type"] == "Secondary")]["Value"].iloc[0]
    sm = out[(out["Process"] == "Smelting") & (out["Type"] == "Secondary")]["Value"].iloc[0]
    an = out[out["Type"] == "Anode"]["Value"].iloc[0]
    assert np.isclose(sc, (123.0 - 60.0 * _K) / _K)
    assert np.isclose(sm, 60.0 / 0.965)          # smelter scrap unchanged
    assert np.isclose(an, 123.0 / 0.997)         # total refined anchor unchanged
    # what enters (in refined-copper terms) now equals what leaves
    assert np.isclose(sm * 0.965 * _K + sc * _K, 123.0)


def test_excess_smelter_route_clips_to_zero_and_is_listed():
    """Bulgaria-shaped: smelter route above secondary refined → zero direct
    route, and the region-year is reported, not dropped silently."""
    df = _frame([_row("Bulgaria", "Smelting", "Secondary", 40.0),
                 _row("Bulgaria", "Refining", "Secondary", 15.0)])
    out = apply_input_divisors(df, _EFF)
    sc = out[(out["Process"] == "Refining") & (out["Type"] == "Secondary")]["Value"].iloc[0]
    assert sc == 0.0
    ex = secondary_excess(df, _EFF)
    assert len(ex) == 1
    assert ex["Region"].iloc[0] == "Bulgaria" and ex["Year"].iloc[0] == 2020
    assert np.isclose(ex["excess_kt"].iloc[0], 40.0 * _K - 15.0)
    # a nested region-year is not listed
    assert secondary_excess(_frame([_row("Austria", "Smelting", "Secondary", 60.0),
                                    _row("Austria", "Refining", "Secondary", 123.0)]), _EFF).empty


def test_subtraction_is_per_region_year():
    df = _frame([_row("Germany", "Smelting", "Secondary", 100.0, year=2020),
                 _row("Germany", "Refining", "Secondary", 200.0, year=2020),
                 _row("Germany", "Smelting", "Secondary", 500.0, year=2021),
                 _row("Germany", "Refining", "Secondary", 600.0, year=2021),
                 _row("China",   "Smelting", "Secondary", 900.0, year=2020),
                 _row("China",   "Refining", "Secondary", 1000.0, year=2020)])
    out = apply_input_divisors(df, _EFF)
    sc = out[(out["Process"] == "Refining") & (out["Type"] == "Secondary")].set_index(["Region", "Year"])["Value"]
    assert np.isclose(sc[("Germany", 2020)], (200.0 - 100.0 * _K) / _K)
    assert np.isclose(sc[("Germany", 2021)], (600.0 - 500.0 * _K) / _K)
    assert np.isclose(sc[("China", 2020)], (1000.0 - 900.0 * _K) / _K)


def test_refinery_scrap_without_smelter_row_is_unchanged():
    """No smelter series → nothing is booked twice → the old divisor alone."""
    df = _frame([_row("Belgium", "Refining", "Secondary", 150.0)])
    out = apply_input_divisors(df, _EFF)
    sc = out[(out["Process"] == "Refining") & (out["Type"] == "Secondary")]["Value"].iloc[0]
    assert np.isclose(sc, 150.0 / _K)
    # and a smelter series without a refined-secondary partner is untouched too
    df2 = _frame([_row("Iran", "Smelting", "Secondary", 90.0)])
    out2 = apply_input_divisors(df2, _EFF)
    assert np.isclose(out2[out2["Type"] == "Secondary"]["Value"].iloc[0], 90.0 / 0.965)
    assert secondary_excess(df2, _EFF).empty


# ---------------------------------------------------------------------------
# One territory, one label (ADR 0032)
# ---------------------------------------------------------------------------
# ICSG renames territories mid-series ("Taiwan, China" from 2013) and publishes
# historical joint reporting under a joint label ("Belgium-Luxembourg" to 2013).
# The ADR 0030 nesting subtraction keys on (Region, Year), so a region whose two
# series carry different labels was never nested at all.

def _aliases():
    return {"Belgium-Luxembourg": "Belgium", "Taiwan, China": "Taiwan"}


def test_two_labels_for_one_territory_are_merged_before_nesting():
    """Belgium 1995-2013: the refined-secondary row is labelled
    Belgium-Luxembourg while the smelter row is labelled Belgium, so the
    subtraction found nothing to subtract and left the smelter route inside."""
    df = _frame([_row("Belgium-Luxembourg", "Refining", "Secondary", 158.0),
                 _row("Belgium",            "Smelting", "Secondary", 150.6)])
    out = apply_input_divisors(merge_territories(df, _aliases()), _EFF)
    sc = out[(out["Process"] == "Refining") & (out["Type"] == "Secondary")]["Value"].iloc[0]
    assert np.isclose(sc, (158.0 - 150.6 * _K) / _K)
    # without the merge the subtraction is skipped entirely
    bug = apply_input_divisors(df, _EFF)
    assert np.isclose(bug[(bug["Process"] == "Refining")
                          & (bug["Type"] == "Secondary")]["Value"].iloc[0], 158.0 / _K)


def test_merged_rows_carry_the_root_label():
    df = _frame([_row("Belgium-Luxembourg", "Refining", "Secondary", 158.0),
                 _row("Taiwan, China",      "Cathode",  "Cathode",   437.5)])
    out = merge_territories(df, _aliases())
    assert set(out["Region"]) == {"Belgium", "Taiwan"}


def test_handover_year_published_under_both_labels_is_not_summed():
    """2013 is republished under the new label; the feed used to add the two."""
    df = _frame([_row("Belgium",            "Refining", "Primary", 229.0, year=2013),
                 _row("Belgium-Luxembourg", "Refining", "Primary", 229.0, year=2013)])
    out = merge_territories(df, _aliases())
    assert len(out) == 1
    assert np.isclose(out["Value"].iloc[0], 229.0)


def test_a_genuine_conflict_between_the_two_labels_is_not_silently_dropped():
    df = _frame([_row("Belgium",            "Refining", "Primary", 9.9,   year=2013),
                 _row("Belgium-Luxembourg", "Refining", "Primary", 160.5, year=2013)])
    with pytest.raises(AssertionError, match="two labels"):
        merge_territories(df, _aliases())


def test_row_members_are_never_merged():
    """ROW is a bucket of 48 countries, not a territory: merging it would nest
    Serbia's smelter route against Egypt's refinery."""
    df = _frame([_row("Egypt",  "Refining", "Secondary", 4.1),
                 _row("Serbia", "Smelting", "Secondary", 80.0)])
    out = apply_input_divisors(merge_territories(df, _aliases()), _EFF)
    sc = out[(out["Process"] == "Refining") & (out["Type"] == "Secondary")]["Value"].iloc[0]
    assert np.isclose(sc, 4.1 / _K)          # Egypt keeps its whole series


def test_territory_aliases_reads_the_concordance_and_excludes_row():
    a = territory_aliases()
    assert a == {"Belgium-Luxembourg": "Belgium", "Taiwan, China": "Taiwan"}


def test_merge_is_a_no_op_without_aliases():
    df = _frame([_row("Chile", "Refining", "Primary", 100.0)])
    pd.testing.assert_frame_equal(merge_territories(df, {}), df)


# ---------------------------------------------------------------------------
# #73, #74: an aggregate row is split to its declared member, and its
# handover-year duplicate is not summed with the members
# ---------------------------------------------------------------------------

SPLIT = pd.DataFrame([dict(src_region="Scandinavia", root_region="Sweden",
                           year_from=1995, year_to=2003, handover_year=2004)])


def _nordic(sweden_2004=181.0, sweden_extra=()):
    """Scandinavia's refined output before and in the 2004 handover, as ICSG
    publishes it: the aggregate alone in 2001, both forms in 2004."""
    return pd.DataFrame([
        _row("Scandinavia", "Refining", "Primary", 300.0, year=2001),
        _row("Scandinavia", "Refining", "Primary", 347.7, year=2004),
        _row("Sweden",      "Refining", "Primary", sweden_2004, year=2004),
        _row("Finland",     "Refining", "Primary", 131.1, year=2004),
        _row("Norway",      "Refining", "Primary", 35.6, year=2004),
        _row("Scandinavia", "Cathode_Pool", "PSC", 4.0, year=2001),
        _row("Scandinavia", "Cathode_Pool", "PSC", -0.2, year=2004),
        _row("Sweden",      "MESC", "Cathode", 21.8, year=2001, su="Use"),
        *sweden_extra,
    ])[_COLS]


def _value(df, region, proc, typ, year):
    m = ((df.Region == region) & (df.Process == proc) & (df.Type == typ)
         & (df.Year == year))
    return df.loc[m, "Value"]


def test_a_members_share_is_read_off_the_handover_year_and_applied_before_it():
    """Regression (#74): Sweden's smelting and refining sat inside
    "Scandinavia" (→ ROW) until 2003, so Sweden had a mine and trade but no
    processing, and the model built its cathode through a phantom SX-EW chain.
    Its share of each series is what ICSG publishes for it in 2004."""
    out = split_aggregates(_nordic(), SPLIT)
    share = 181.0 / 347.7
    assert _value(out, "Sweden", "Refining", "Primary", 2001).item() == \
        pytest.approx(300.0 * share, rel=1e-12)
    assert _value(out, "Scandinavia", "Refining", "Primary", 2001).item() == \
        pytest.approx(300.0 * (1.0 - share), rel=1e-12)
    total = (_value(out, "Sweden", "Refining", "Primary", 2001).item()
             + _value(out, "Scandinavia", "Refining", "Primary", 2001).item())
    assert total == pytest.approx(300.0, rel=1e-12), "the aggregate's total is kept"


def test_the_handover_year_aggregate_is_not_summed_with_its_members():
    """Regression (#73): in 2004 ICSG publishes Scandinavia and its members,
    and the feed added both, so the Nordic flows entered twice, once via ROW."""
    out = split_aggregates(_nordic(), SPLIT)
    assert _value(out, "Scandinavia", "Refining", "Primary", 2004).empty
    refined_2004 = out[(out.Process == "Refining") & (out.Year == 2004)].Value.sum()
    assert refined_2004 == pytest.approx(181.0 + 131.1 + 35.6)


def test_a_series_the_member_does_not_report_stays_with_the_aggregate():
    """Sweden reports no producer stock change in 2004, so there is no share to
    read and Scandinavia's PSC rows stay whole, in every year."""
    out = split_aggregates(_nordic(), SPLIT)
    assert _value(out, "Scandinavia", "Cathode_Pool", "PSC", 2001).item() == 4.0
    assert _value(out, "Scandinavia", "Cathode_Pool", "PSC", 2004).item() == -0.2
    assert _value(out, "Sweden", "Cathode_Pool", "PSC", 2001).empty


def test_the_members_own_rows_are_untouched():
    out = split_aggregates(_nordic(), SPLIT)
    assert _value(out, "Sweden", "MESC", "Cathode", 2001).item() == 21.8
    assert _value(out, "Finland", "Refining", "Primary", 2004).item() == 131.1


def test_a_member_larger_than_its_aggregate_raises():
    with pytest.raises(ValueError, match="share"):
        split_aggregates(_nordic(sweden_2004=400.0), SPLIT)


def test_a_member_row_the_split_would_double_raises():
    own = _row("Sweden", "Refining", "Primary", 50.0, year=2001)
    with pytest.raises(ValueError, match="Sweden"):
        split_aggregates(_nordic(sweden_extra=[own]), SPLIT)


def test_no_declaration_is_a_no_op():
    df = _nordic()
    pd.testing.assert_frame_equal(split_aggregates(df, SPLIT.iloc[:0]), df)


def test_the_declaration_is_read_with_integer_years(tmp_path):
    path = tmp_path / "params.csv"
    path.write_text("src_region,root_region,year_from,year_to,handover_year,source,comment\n"
                    "Scandinavia,Sweden,1995,2003,2004,icsg v13,#74\n")
    got = aggregate_splits(path)
    assert got.loc[0, ["year_from", "year_to", "handover_year"]].tolist() == [1995, 2003, 2004]


def test_a_handover_year_inside_the_member_years_raises(tmp_path):
    path = tmp_path / "params.csv"
    path.write_text("src_region,root_region,year_from,year_to,handover_year,source,comment\n"
                    "Scandinavia,Sweden,1995,2003,2003,icsg v13,#74\n")
    with pytest.raises(ValueError, match="handover_year"):
        aggregate_splits(path)
