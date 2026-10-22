"""The Stage-5 and limitation numbers of manuscript_numbers.py, on toy frames.

The manuscript quotes the default posterior, not the #87 scale-total ablation built
from the same opt run, and its limitations name the mine chains the solver keeps
where ICSG reports no mine output, the stock legs beyond |z| 5 and the regions that
carry the scale-total objective.
"""
import os
import sys

import numpy as np
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
    assert np.isclose(sum(shares.values()), 1.0)
