"""
test_icsg_nas.py
================
Net addition to stock (NAS) from ICSG stock levels (#35, GLOSSARY.md).

A NAS is the difference between a stock level and the same series' level in
the preceding year of the same vintage; it never straddles two vintages or a
reporting gap.  PSC is the producer NAS (a lone printed total counts as
producers), MESC the exchange NAS by exchange location, USC the sum of the
consumer, merchant and government NAS of a region-year.
"""
from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "icsg_parse"))
from icsg_comb import stock_levels  # noqa: E402
from process_icsg import net_addition_to_stock  # noqa: E402

_FLOW_COLS = ["Region", "Type", "Year", "Value", "Process", "Supply_Use", "Unit",
              "Year_loaded_from_file"]


def _levels(rows):
    """rows: (region, stock_type, type, year, value, vintage)"""
    return pd.DataFrame(
        [dict(Region=r, Type=t, Year=y, Value=v, Unit="kt", Stock_type=s,
              Year_loaded_from_file=vin) for r, s, t, y, v, vin in rows])


def _nas(df, region, process, year):
    hit = df[(df["Region"] == region) & (df["Process"] == process) & (df["Year"] == year)]
    assert len(hit) <= 1, f"{region}/{process}/{year}: {len(hit)} rows"
    return None if hit.empty else hit["Value"].iloc[0]


def test_nas_is_the_change_from_the_preceding_year_only():
    """A blank year yields no NAS for itself or for the year after it, instead
    of booking a two-year change in one year."""
    df = net_addition_to_stock(_levels([
        ("Chile", "PMC", "Producers", 2004, 285.5, 2014),
        ("Chile", "PMC", "Producers", 2005, 207.5, 2014),
        ("Chile", "PMC", "Producers", 2007, 223.7, 2014)]))
    assert _nas(df, "Chile", "Cathode_Pool", 2005) == pytest.approx(-78.0)
    assert _nas(df, "Chile", "Cathode_Pool", 2006) is None
    assert _nas(df, "Chile", "Cathode_Pool", 2007) is None
    assert list(df.columns) == _FLOW_COLS
    row = df.iloc[0]
    assert (row["Type"], row["Supply_Use"]) == ("PSC", "Supply")


def test_each_vintage_differences_its_own_levels_and_the_newest_wins():
    """Regression: levels were deduplicated across vintages before
    differencing, so the first year of every later vintage had no NAS and a
    NAS could mix one vintage's level with another's."""
    df = net_addition_to_stock(_levels([
        ("Chile", "PMC", "Producers", 2003, 502.3, 2005),
        ("Chile", "PMC", "Producers", 2004, 343.0, 2005),
        ("Chile", "PMC", "Producers", 2004, 285.5, 2014),   # restated level
        ("Chile", "PMC", "Producers", 2005, 207.5, 2014),
        ("Chile", "PMC", "Producers", 2004, 280.0, 2023),   # a later restatement
        ("Chile", "PMC", "Producers", 2005, 210.0, 2023),
        ("Chile", "PMC", "Producers", 2006, 300.0, 2023)]))
    assert _nas(df, "Chile", "Cathode_Pool", 2004) == pytest.approx(343.0 - 502.3)
    assert _nas(df, "Chile", "Cathode_Pool", 2005) == pytest.approx(210.0 - 280.0)
    assert _nas(df, "Chile", "Cathode_Pool", 2006) == pytest.approx(90.0)
    assert df.loc[df["Year"] == 2004, "Year_loaded_from_file"].tolist() == [2005]
    assert df.loc[df["Year"] == 2005, "Year_loaded_from_file"].tolist() == [2023]


def test_usc_is_the_sum_of_the_user_side_types():
    """Regression: consumers, merchants and government shared one key and an
    unstable sort kept one of them (Germany 1996 kept 3.4 of 3.4 and 2.2)."""
    df = net_addition_to_stock(_levels([
        ("Germany", "PMC", "Consumers",  1995, 20.1, 2005),
        ("Germany", "PMC", "Consumers",  1996, 23.5, 2005),
        ("Germany", "PMC", "Merchants",  1995, 2.7,  2005),
        ("Germany", "PMC", "Merchants",  1996, 4.9,  2005),
        ("Germany", "PMC", "Government", 1996, 1.0,  2005)]))  # no 1995 level
    assert _nas(df, "Germany", "USC", 1996) == pytest.approx(3.4 + 2.2)


def test_mesc_sums_every_exchange_at_one_location():
    """COMEX and the LME's U.S. warehouses are both the United States."""
    df = net_addition_to_stock(_levels([
        ("United States", "Exchanges", "LME",   2004, 35.0, 2014),
        ("United States", "Exchanges", "LME",   2005, 20.5, 2014),
        ("United States", "Exchanges", "COMEX", 2004, 43.7, 2014),
        ("United States", "Exchanges", "COMEX", 2005, 50.0, 2014)]))
    assert _nas(df, "United States", "MESC", 2005) == pytest.approx(-14.5 + 6.3)


def test_a_lone_total_is_producer_stock():
    """China prints only a total; it is booked as PSC (GLOSSARY.md)."""
    df = net_addition_to_stock(_levels([
        ("China", "PMC", "Total", 2004, 30.0, 2014),
        ("China", "PMC", "Total", 2005, 47.0, 2014)]))
    assert _nas(df, "China", "Cathode_Pool", 2005) == pytest.approx(17.0)


def test_a_total_its_types_add_up_to_is_not_counted_twice():
    df = net_addition_to_stock(_levels([
        ("Japan", "PMC", "Producers", 2004, 68.7,  2014),
        ("Japan", "PMC", "Producers", 2005, 75.4,  2014),
        ("Japan", "PMC", "Consumers", 2004, 27.6,  2014),
        ("Japan", "PMC", "Consumers", 2005, 16.4,  2014),
        ("Japan", "PMC", "Total",     2004, 96.4,  2014),   # rounding: 96.3
        ("Japan", "PMC", "Total",     2005, 91.8,  2014)]))
    assert _nas(df, "Japan", "Cathode_Pool", 2005) == pytest.approx(6.7)
    assert _nas(df, "Japan", "USC", 2005) == pytest.approx(-11.2)


def test_types_that_contradict_the_total_give_way_to_it(caplog):
    """Japan prints 0.000 for every type from 2021, when it reports only a
    total: read as levels they booked a -95.8 kt producer draw-down.  The year
    the total takes over from the types books no change (triage Q9); from then
    on the total is a lone total."""
    rows = []
    for year, prod, cons, total in [(2020, 95.8, 14.2, 110.0),
                                    (2021, 0.0,  0.0,  94.4),
                                    (2022, 0.0,  0.0,  98.3)]:
        rows += [("Japan", "PMC", "Producers", year, prod,  2025),
                 ("Japan", "PMC", "Consumers", year, cons,  2025),
                 ("Japan", "PMC", "Total",     year, total, 2025)]
    with caplog.at_level(logging.WARNING):
        df = net_addition_to_stock(_levels(rows))
    assert _nas(df, "Japan", "Cathode_Pool", 2021) is None
    assert _nas(df, "Japan", "Cathode_Pool", 2022) == pytest.approx(3.9)
    assert _nas(df, "Japan", "USC", 2021) is None
    assert "Japan" in caplog.text and "2021" in caplog.text


def test_a_total_that_takes_over_from_its_types_books_no_change_that_year():
    """Germany prints its types to 2003 and only a total from 2004: the
    total's 2004 change mixes a stock change with a change of reporting."""
    df = net_addition_to_stock(_levels([
        ("Germany", "PMC", "Producers", 2003, 47.0, 2005),
        ("Germany", "PMC", "Consumers", 2003, 33.0, 2005),
        ("Germany", "PMC", "Total",     2003, 80.0, 2005),
        ("Germany", "PMC", "Total",     2004, 70.0, 2005),
        ("Germany", "PMC", "Total",     2005, 70.0, 2005)]))
    assert _nas(df, "Germany", "Cathode_Pool", 2004) is None
    assert _nas(df, "Germany", "Cathode_Pool", 2005) == pytest.approx(0.0)


def test_a_type_that_stops_reporting_is_warned_and_yields_no_change(caplog):
    """Korea's user-side types are blank in 2005-06: no USC row, and the
    vanishing is logged (the printed total equals the producers row)."""
    rows = [("South Korea", "PMC", "Producers", 2004, 4.0,  2014),
            ("South Korea", "PMC", "Consumers", 2004, 8.0,  2014),
            ("South Korea", "PMC", "Total",     2004, 12.0, 2014),
            ("South Korea", "PMC", "Producers", 2005, 45.0, 2014),
            ("South Korea", "PMC", "Total",     2005, 45.0, 2014)]
    with caplog.at_level(logging.WARNING):
        df = net_addition_to_stock(_levels(rows))
    assert _nas(df, "South Korea", "USC", 2005) is None
    gap = [r.message for r in caplog.records if "Consumers" in r.message]
    assert len(gap) == 1 and "South Korea" in gap[0] and "2005" in gap[0]


def test_two_levels_for_one_series_year_raise():
    rows = [("Chile", "PMC", "Producers", 2004, 285.5, 2014),
            ("Chile", "PMC", "Producers", 2004, 280.0, 2014)]
    with pytest.raises(ValueError, match="Chile"):
        net_addition_to_stock(_levels(rows))


def test_the_combine_step_keeps_every_vintages_levels_and_totals():
    """Regression: icsg_comb kept only the newest vintage's level of a year and
    renamed or dropped totals before any NAS was taken."""
    s = _levels([("Chile",   "PMC", "Producers", 2004, 343.0, 2005),
                 ("Chile",   "PMC", "Producers", 2004, 285.5, 2014),
                 ("Germany", "PMC", "Total",     2004, 70.0,  2005),
                 ("Germany", "PMC", "Producers", 2004, 47.0,  2005)])
    out = stock_levels(s)
    assert len(out) == 4
    assert set(out["Type"]) == {"Producers", "Total"}


def test_the_combine_step_raises_on_two_levels_in_one_vintage():
    s = _levels([("Chile", "PMC", "Producers", 2004, 343.0, 2005),
                 ("Chile", "PMC", "Producers", 2004, 340.0, 2005)])
    with pytest.raises(ValueError, match="Chile"):
        stock_levels(s)
