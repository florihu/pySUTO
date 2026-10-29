"""The Stage-5 and limitation numbers of manuscript_numbers.py, on toy frames.

The manuscript quotes the default posterior, not the #87 scale-total ablation built
from the same opt run, and its limitations name the mine chains the solver keeps
where ICSG reports no mine output, the stock legs beyond |z| 5 and the regions that
carry the scale-total objective.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "report"))
import manuscript_numbers as M  # noqa: E402


def test_the_default_posterior_is_the_latest_run_without_the_ablation():
    sidecars = {"uncprop_20261021_134829_aa": {"stage_config": {"exclude_scale_totals": False}},
                "uncprop_20261020_082104_bb": {"stage_config": {}},
                "uncprop_20261022_081020_cc": {"stage_config": {"exclude_scale_totals": True}}}
    assert M.stage5_runs(sidecars) == ("uncprop_20261021_134829_aa", "uncprop_20261022_081020_cc")
    assert M.stage5_runs({}) == (None, None)


def _ic(rows):
    return pd.DataFrame(rows, columns=["reg", "Process", "Type", "Supply_Use", "Year", "Value"])


def test_mine_region_years_are_those_with_positive_icsg_concentrate_output():
    ic = _ic([["Chile", "Mining", "Concentrate", "Supply", 2000, 1500.0],
              ["Belgium", "Mining", "Concentrate", "Supply", 2000, 0.0],
              ["Belgium", "Smelting", "Concentrate", "Use", 2000, 300.0],
              [None, "Mining", "Concentrate", "Supply", 2000, 9.0]])
    assert M.mine_region_years(ic) == {("Chile", "2000")}


def _edges(rows):
    return pd.DataFrame(rows, columns=["sector_from", "region_from", "sector_to", "region_to",
                                       "valuation", "time", "prior", "post"])


def test_the_phantom_mine_chain_is_measured_outside_the_mine_region_years():
    e = _edges([["MF", "Belgium", "Concentrate", "Belgium", "CC", "2000", 10.0, 12.0],
                ["CO", "Belgium", "Mining", "Belgium", "CC", "2000", 14.0, 15.0],
                ["Concentrate", "Belgium", "Smelting", "Belgium", "CC", "2000", 4.0, 3.0],
                ["Concentrate", "Belgium", "Smelting", "Japan", "CC", "2000", 8.0, 9.0],
                ["MF", "Chile", "Concentrate", "Chile", "CC", "2000", 900.0, 880.0],
                ["MF", "Belgium", "Concentrate", "Belgium", "Mass", "2000", 40.0, 48.0]])
    p = M.phantom_mine_chain(e, mine={("Chile", "2000")})
    assert p["mf_conc_post_kt"] == pytest.approx(12.0)
    assert p["co_mining_post_kt"] == pytest.approx(15.0)
    assert p["export_share"] == pytest.approx(9.0 / 12.0)
    assert p["by_region"] == {"Belgium": pytest.approx(12.0)}


def test_scale_total_shares_rank_the_regions_by_their_z_squared():
    s = pd.DataFrame({"type": ["sum", "sum", "sum", "ratio"],
                      "region": ["Congo, D.R.", "Congo, D.R.", "Zambia", "Zambia"],
                      "z": [3.0, 4.0, 5.0, 100.0]})
    shares = M.sum_shares_by_region(s)
    assert list(shares) == ["Congo, D.R.", "Zambia"]
    assert shares["Congo, D.R."] == pytest.approx(25.0 / 50.0)
    assert sum(shares.values()) == pytest.approx(1.0, rel=1e-12)


def test_a_regions_scale_total_z_squared_is_located_in_its_largest_years():
    s = pd.DataFrame({"type": ["sum"] * 4 + ["ratio"],
                      "region": ["Uzbekistan"] * 4 + ["Uzbekistan"],
                      "year": ["1995", "1995", "1997", "2010", "1995"],
                      "z": [6.0, 0.0, 2.0, 1.0, 50.0]})
    top = M.top_year_shares(s, ["Uzbekistan"], n=2)
    assert list(top["Uzbekistan"]) == ["1995", "1997"]
    assert top["Uzbekistan"]["1995"] == pytest.approx(36.0 / 41.0, rel=1e-12)


def _ledger(rows):
    return pd.DataFrame(rows, columns=["sector", "region", "time", "X", "P", "U",
                                       "imports_exhausted"])


def test_exports_beyond_production_are_read_per_region_and_year():
    """#111's contradictions: DR Congo exports anode ICSG gives it no smelter for (P
    missing), Zambia exports more cathode than it refines; small excesses drop out."""
    led = _ledger([["Anode", "Congo, D.R.", 2012, 96.1, float("nan"), 25.1, True],
                   ["Anode", "Congo, D.R.", 2021, 97.8, 131.8, 36.1, False],
                   ["Cathode", "Zambia", 2014, 1362.6, 499.4, 18.0, False],
                   ["Cathode", "Zambia", 2021, 333.4, 335.9, 16.0, False],
                   ["Cathode", "Chile", 2014, 2700.0, 1000.0, 50.0, False],
                   ["Anode", "Zambia", 2016, 3.0, 0.0, 0.0, False]])
    regions = ["Congo, D.R.", "Zambia"]
    assert M.exports_beyond_production(led, "Anode", regions) == {
        "Congo, D.R.": {"2012": [pytest.approx(96.1), None]}}
    assert M.exports_beyond_production(led, "Cathode", regions) == {
        "Zambia": {"2014": [pytest.approx(1362.6), pytest.approx(499.4)]}}


def test_withheld_usage_counts_the_exhausted_groups_and_their_icsg_use():
    led = _ledger([["Anode", "Congo, D.R.", 2012, 96.1, float("nan"), 25.1, True],
                   ["Cathode", "South Africa", 2021, 55.3, 0.0, 30.0, True],
                   ["Cathode", "South Africa", 2016, 53.4, 68.8, 55.4, False]])
    w = M.withheld_usage(led)
    assert w["groups"] == 2
    assert w["usage_kt"] == pytest.approx(55.1)
    assert w["by_region"] == {"South Africa": pytest.approx(30.0),
                              "Congo, D.R.": pytest.approx(25.1)}


def test_the_degenerate_family_is_counted_per_year_within_the_fringe():
    """#99's legs: cross-region Cathode→FU, either valuation, prior 0, a* within
    1e-4 kt of its bound; a leg with a prior or real slack is not one."""
    e = _edges([["Cathode", "Chile", "FU", "China", "CC", "2000", 0.0, 1.4e-6],
                ["Cathode", "Chile", "FU", "China", "Mass", "2000", 0.0, 0.0],
                ["Cathode", "Chile", "FU", "Japan", "CC", "2001", 0.0, 5e-5],
                ["Cathode", "Chile", "FU", "Japan", "CC", "2001", 0.0, 1e-3],
                ["Cathode", "Chile", "FU", "Chile", "CC", "2001", 0.0, 0.0],
                ["Cathode", "Chile", "FU", "Peru", "CC", "2001", 3.0, 0.0]])
    assert M.degenerate_family_counts(e) == {"edge_years": 3, "per_year": [1, 2]}


def test_the_interval_shift_is_the_largest_bound_move_pinned_or_free():
    t = pd.DataFrame({"quantity":           ["a", "b", "c"],
                      "sigma_birge":        [0.10, 0.20, 0.30],
                      "sigma_birge_pinned": [0.10, 0.21, 0.30],
                      "sigma_birge_free":   [0.12, 0.20, float("nan")]})
    s = M.interval_shift(t, z=2.0)
    assert s["max"] == pytest.approx(0.04)
    assert s["quantity"] == "a"
