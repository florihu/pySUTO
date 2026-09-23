"""
test_piot.py
============
Stage 6 · Slice 5 — the PIOT as a construct family: the market-share process
network at level 'Z' (issue #25, ADR 0037). The toy chain is test_psut's.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_psut import (  # noqa: E402
    CC_TOTALS_2020, PERIODS, PROCS, VALUATIONS, _full_vector, _legacy_results, _make_toy_acc,
)


@pytest.fixture
def psut():
    from pysuto.psut import PSUT
    acc = _make_toy_acc()
    return PSUT.from_results(_legacy_results(acc, _full_vector(acc)), acc)


@pytest.fixture
def world(psut):
    return psut.aggregate(region="World", keep_crossings=True)


def _hand_psut(rows, nodes):
    """A PSUT straight from rows — no accounting structure needed."""
    from pysuto.psut import PSUT
    table = pd.DataFrame(rows, columns=["period", "valuation", "block", "edge_id",
                                        "from_region", "from_sector", "from_entity",
                                        "to_region", "to_sector", "to_entity", "val"])
    nodes = pd.DataFrame(nodes, columns=["sector", "entity", "region", "entity_sub"])
    return PSUT(table, nodes, valuations=["CC"], primary="CC", acc_id="hand")


# ---------------------------------------------------------------------------
# 1) Single producer → identity edges; the blocks and the traded split carry over
# ---------------------------------------------------------------------------

def test_single_producer_gives_identity_edges(world):
    piot = world.to_piot("market_share", level="Z")
    D = piot.D("2020")
    assert D.shape == (2, 2)
    assert D.loc[("World", "Mining"), ("World", "Ore")] == 1.0
    assert D.loc[("World", "Smelting"), ("World", "Cathode")] == 1.0
    assert D.to_numpy().sum() == 2.0
    Z = piot.Z("2020")
    assert Z.index.equals(Z.columns) and Z.index.names == ["region", "process"]
    assert Z.loc[("World", "Mining"), ("World", "Smelting")] == 160.0
    assert Z.to_numpy().sum() == 160.0

    t = piot.table.query("period == '2020' and valuation == 'CC'")
    assert set(t["block"]) == {"Z", "Y", "E", "O", "P"}
    assert t.groupby("block")["val"].sum().to_dict() == pytest.approx(
        {"Z": 160.0, "Y": 151.0, "E": 160.0, "O": 5.0, "P": 14.0})
    z = t[t["block"] == "Z"].set_index("region_crossed")["val"].to_dict()
    assert z == {False: 140.0, True: 20.0}                          # the traded split survives
    assert t.loc[(t["block"] == "Y") & t["region_crossed"], "val"].sum() == 6.0
    assert set(t.loc[t["block"] == "Z", "commodity"]) == {"Ore"}
    assert set(t.loc[t["block"] == "Y", "commodity"]) == {"Cathode"}
    assert set(t.loc[t["block"] == "E", "commodity"]) == {"Nature"}
    assert set(t.loc[t["block"] == "P", "commodity"]) == {"Slag"}
    assert (t.loc[t["block"] == "Z", "to_entity"] == "Process").all()
    assert (t.loc[t["block"] == "Y", "to_entity"] == "SB").all()
    assert (piot.construct, piot.level, piot.primary) == ("market_share", "Z", "CC")
    assert piot.aggregation == world.aggregation and piot.acc_id == "toy"
    assert "PIOT(construct='market_share', level='Z'" in repr(piot)


# ---------------------------------------------------------------------------
# 2) Two producers: the market share splits use and final use between them
# ---------------------------------------------------------------------------

def test_market_share_splits_use_and_final_use_between_producers():
    nodes = [("P1", "Process", "A", None), ("P2", "Process", "A", None),
             ("F", "Flow", "A", None), ("Q", "Process", "A", None), ("FD", "SB", "A", "FD")]
    rows = [("2020", "CC", "S", 0, "A", "P1", "Process", "A", "F", "Flow", 60.0),
            ("2020", "CC", "S", 1, "A", "P2", "Process", "A", "F", "Flow", 40.0),
            ("2020", "CC", "U", 2, "A", "F", "Flow", "A", "Q", "Process", 80.0),
            ("2020", "CC", "Y", 3, "A", "F", "Flow", "A", "FD", "SB", 20.0)]
    piot = _hand_psut(rows, nodes).to_piot()
    D = piot.D()
    assert D.loc[("A", "P1"), ("A", "F")] == pytest.approx(0.6)
    assert D.loc[("A", "P2"), ("A", "F")] == pytest.approx(0.4)
    edges = piot.table.set_index(["block", "from_sector", "to_sector"])["val"]
    assert edges[("Z", "P1", "Q")] == pytest.approx(48.0)
    assert edges[("Z", "P2", "Q")] == pytest.approx(32.0)
    assert edges[("Y", "P1", "FD")] == pytest.approx(12.0)
    assert edges[("Y", "P2", "FD")] == pytest.approx(8.0)
    # every producer's supply is fully attributed: Σ_q edge(p→q, f) + Σ_d edge(p→d, f) = S[p, f]
    per_producer = piot.table.groupby("from_sector")["val"].sum()
    assert per_producer["P1"] == pytest.approx(60.0) and per_producer["P2"] == pytest.approx(40.0)
    Z = piot.Z()
    assert Z.loc[("A", "P1"), ("A", "Q")] == pytest.approx(48.0) and Z.to_numpy().sum() == pytest.approx(80.0)
    assert "region_crossed" not in piot.table.columns          # none stamped upstream


def test_use_of_an_unsupplied_commodity_raises():
    nodes = [("F", "Flow", "A", None), ("Q", "Process", "A", None)]
    rows = [("2020", "CC", "U", 0, "A", "F", "Flow", "A", "Q", "Process", 80.0)]
    with pytest.raises(ValueError, match="nobody supplies"):
        _hand_psut(rows, nodes).to_piot()


# ---------------------------------------------------------------------------
# 3) Conservation on the three-region toy, every period and valuation
# ---------------------------------------------------------------------------

def test_allocation_conserves_every_producers_supply(psut):
    piot = psut.to_piot()
    key = ["period", "valuation", "from_region", "from_sector"]
    supply = psut.table[psut.table["block"] == "S"].groupby(key + ["to_region", "to_sector"])["val"].sum()
    attributed = (piot.table[piot.table["block"].isin(["Z", "Y"])]
                      .groupby(key + ["commodity_region", "commodity"])["val"].sum())
    attributed.index = attributed.index.set_names(supply.index.names)
    pd.testing.assert_series_equal(attributed.sort_index(), supply.sort_index())
    for block_in, block_out in (("U", "Z"), ("Y", "Y")):
        a = psut.table[psut.table["block"] == block_in].groupby(["period", "valuation"])["val"].sum()
        b = piot.table[piot.table["block"] == block_out].groupby(["period", "valuation"])["val"].sum()
        pd.testing.assert_series_equal(a, b)
    for block in ("E", "O", "P"):
        a = psut.table[psut.table["block"] == block].groupby(["period", "valuation"])["val"].sum()
        b = piot.table[piot.table["block"] == block].groupby(["period", "valuation"])["val"].sum()
        pd.testing.assert_series_equal(a, b)
    Z = piot.Z("2020")
    assert Z.index.equals(PROCS)                                # region C present, all zero
    assert (Z.loc["C"] == 0).all().all() and (Z["C"] == 0).all().all()
    assert Z.loc[("B", "Mining"), ("A", "Smelting")] == 20.0    # the cross-region use, re-attributed
    assert piot.Z("2020", "Mass").to_numpy().sum() == pytest.approx(3 * 160.0)


# ---------------------------------------------------------------------------
# 4) The family's other members are stubs, loudly
# ---------------------------------------------------------------------------

def test_other_constructs_and_levels_are_stubs(psut):
    with pytest.raises(NotImplementedError, match="#20"):
        psut.to_piot("suh_product")
    with pytest.raises(NotImplementedError, match="#20"):
        psut.to_piot(level="A")
    with pytest.raises(NotImplementedError, match="#20"):
        psut.to_piot(level="L")
    with pytest.raises(ValueError, match="construct"):
        psut.to_piot("bogus")
    with pytest.raises(ValueError, match="level"):
        psut.to_piot(level="Q")
    piot = psut.to_piot()
    with pytest.raises(NotImplementedError):
        piot.A()
    with pytest.raises(NotImplementedError):
        piot.L()
    with pytest.raises(NotImplementedError):
        piot.footprint()


def test_periods_and_resolve(psut):
    piot = psut.to_piot()
    assert piot.periods() == PERIODS and piot.latest() == "2021"
    assert piot.resolve() == ("2021", "CC") and piot.resolve(2020, "Mass") == ("2020", "Mass")
    with pytest.raises(KeyError, match="PIOT"):
        piot.Z("1999")
    assert piot.valuations == VALUATIONS


# ---------------------------------------------------------------------------
# 5) The process-network Sankey through the shared renderer
# ---------------------------------------------------------------------------

def test_process_network_sankey(world):
    pytest.importorskip("plotly")
    from pysuto.diag.psut import SankeySpec, sankey_tables
    piot = world.to_piot()
    nodes, links = sankey_tables(piot, SankeySpec(period="2020"))
    assert set(nodes["label"]) == {"Nature", "Scrap", "Mining", "Smelting", "Use", "Slag"}   # no commodity nodes
    assert "commodity" in links.columns
    z = links[links["block"] == "Z"]
    assert set(z["cls"]) == {"use", "traded"} and set(z["commodity"]) == {"Ore"}
    assert z["value"].sum() == pytest.approx(160.0)
    assert any("Z · use · Ore" == h for h in links["hover"]) and any("Z · traded · Ore" == h for h in links["hover"])
    fig = piot.sankey("2020")
    d = fig.to_dict()
    assert "process-network Sankey" in d["layout"]["title"]["text"]
    assert sum(d["data"][0]["link"]["value"]) == pytest.approx(160.0 + 151.0 + 160.0 + 5.0 + 14.0)
    # the two Sankeys agree on every boundary total
    _, psut_links = sankey_tables(world, SankeySpec(period="2020"))
    for block in ("E", "O", "P", "Y"):
        assert (links.loc[links["block"] == block, "value"].sum()
                == pytest.approx(psut_links.loc[psut_links["block"] == block, "value"].sum()))
    assert CC_TOTALS_2020["S"] == 311.0                          # the toy's supply, for the record
