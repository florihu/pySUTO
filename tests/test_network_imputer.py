"""
test_network_imputer.py
=======================
Unit suite for pysuto.impute.network.NetworkImputer — the process-recipe
traversal imputer.

A small synthetic copper chain (one region, one year) exercises:
  * forward recipe closure (product + SEM waste) and the Mass twin
  * two-hop propagation across a Flow node (fixpoint)
  * backward multi-input subtraction (pooled Scrap)
  * the Ore branch split resolved from the SXEW anchor (+ negative: left NaN)
  * the Mining WR mass-residual
  * determinacy / no-overwrite
  * CP pooling (eff = 1)
  * conc-by-construction: written CC/Mass twins satisfy CC = conc·Mass
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.agg import AggregationResult
from pysuto.impute import NetworkImputer, derive_process_recipes


# =============================================================================
# Synthetic accounting structure
# =============================================================================

# node: (sector, entity, entity_sub)   — single region "A", single time "2020"
NODES = [
    ("CO",          "SB",      "Nature"),
    ("Mining",      "Process", None),
    ("Ore",         "Flow",    None),
    ("WR",          "SB",      "SEM"),
    ("MF",          "Process", None),
    ("Concentrate", "Flow",    None),
    ("Tailings",    "SB",      "SEM"),
    ("Scrap",       "SB",      "SEM"),
    ("Smelting",    "Process", None),
    ("Matte",       "Flow",    None),
    ("Slag",        "SB",      "SEM"),
    ("SXEW",        "Process", None),
    ("CSXEW",       "Flow",    None),
    ("CER",         "Flow",    None),
    ("CP",          "Process", None),
    ("Cathode",     "Flow",    None),
    ("PSC",         "SB",      "SEM"),
]

# directed (from_sector, to_sector) — same region; CC + Mass twin for each
EDGES = [
    ("CO",          "Mining"),      # use
    ("Mining",      "Ore"),         # supply
    ("Mining",      "WR"),          # supply (SEM waste)
    ("Ore",         "MF"),          # use
    ("Ore",         "SXEW"),        # use (branch)
    ("MF",          "Concentrate"), # supply
    ("MF",          "Tailings"),    # supply (SEM waste)
    ("Concentrate", "Smelting"),    # use
    ("Scrap",       "Smelting"),    # use (pooled primary)
    ("Smelting",    "Matte"),       # supply
    ("Smelting",    "Slag"),        # supply (SEM waste)
    ("SXEW",        "CSXEW"),       # supply
    ("SXEW",        "Tailings"),    # supply (SEM waste)
    ("CSXEW",       "CP"),          # use
    ("CER",         "CP"),          # use
    ("CP",          "Cathode"),     # supply
    ("CP",          "PSC"),         # supply (SEM, becomes 0 with eff=1)
]

EFF = {  # (input, process, product): nominal
    ("Ore",         "MF",       "Concentrate"): 0.80,
    ("Concentrate", "Smelting", "Matte"):       0.965,
    ("Ore",         "SXEW",     "CSXEW"):       0.725,
}

CONC = {  # commodity sector -> Cu mass fraction
    "Ore": 0.009, "Concentrate": 0.275, "Matte": 0.525, "CSXEW": 0.9975,
    "CER": 0.9975, "Cathode": 0.9975, "Scrap": 0.94,
    "WR": 0.00025, "Tailings": 0.0055, "Slag": 0.0275, "CO": 0.01,
}
ORE_GRADE_A = 0.005  # region/time specific CO grade (overrides conc_co_supply)


class _Bounds:
    def __init__(self, n):
        self.l = np.zeros(n, dtype=float)
        self.u = np.full(n, 1e9, dtype=float)


class MockAcc:
    """Minimal AccountingStructure stand-in for the synthetic chain."""

    def __init__(self):
        self.node_index = pd.DataFrame(
            [(s, e, "A", sub) for s, e, sub in NODES],
            columns=["sector", "entity", "region", "entity_sub"],
        )
        self.node_index.index = range(len(NODES))
        self.node_to_id = {(s, e, "A"): i for i, (s, e, _) in enumerate(NODES)}
        sec_to_id = {s: i for i, (s, _e, _sub) in enumerate(NODES)}

        rows = []
        eid = 0
        for fs, ts in EDGES:
            for val in ("Mass", "CC"):
                rows.append((eid, sec_to_id[fs], sec_to_id[ts], "2020", val))
                eid += 1
        self.edge_index = pd.DataFrame(
            rows, columns=["edge_id", "from_id", "to_id", "time", "valuation"]
        )
        self.n_edges = len(rows)
        self.bounds = _Bounds(self.n_edges)

        ent = self.node_index["entity"]
        fe = self.edge_index["from_id"].map(ent).to_numpy()
        te = self.edge_index["to_id"].map(ent).to_numpy()
        self.masks = {
            "supply": (fe == "Process") & np.isin(te, ["Flow", "SB"]),
            "use":    np.isin(fe, ["Flow", "SB"]) & (te == "Process"),
            "tradable": np.zeros(self.n_edges, dtype=bool),
            "sb":     (te == "SB"),
        }

    @property
    def edge_meta(self):
        ei = self.edge_index.reset_index(drop=True)
        nf = self.node_index.loc[ei["from_id"].values].reset_index(drop=True).add_prefix("from_")
        nt = self.node_index.loc[ei["to_id"].values].reset_index(drop=True).add_prefix("to_")
        return pd.concat([ei, nf, nt], axis=1)

    @property
    def process_recipes(self):
        return derive_process_recipes(self)

    # edge_id helpers for tests --------------------------------------------
    def eid(self, fs, ts, val):
        m = self.edge_meta
        hit = m[(m["from_sector"] == fs) & (m["to_sector"] == ts)
                & (m["valuation"] == val)]
        return int(hit["edge_id"].iloc[0])


# =============================================================================
# Param-CSV generation
# =============================================================================

_COLS = [
    "param_id", "param_type", "description",
    "a_sector_from", "a_entity_from", "a_region_from",
    "a_sector_to", "a_entity_to", "a_region_to", "a_time", "a_valuation",
    "b_sector_from", "b_entity_from", "b_region_from",
    "b_sector_to", "b_entity_to", "b_region_to", "b_time", "b_valuation",
    "value_nominal", "unc_type", "value_sd", "value_min", "value_max", "epsilon",
]


def _row(**kw):
    r = {c: "*" for c in _COLS}
    r.update(param_type="x", description="", unc_type="sd", value_sd="0.001",
             value_min="", value_max="", epsilon="0")
    r.update(kw)
    return r


def _write(path, rows):
    pd.DataFrame(rows, columns=_COLS).to_csv(path, sep=";", index=False)


def write_eff_csv(path):
    rows = []
    for (inp, proc, prod), v in EFF.items():
        rows.append(_row(
            param_id=f"eff_{proc.lower()}",
            a_sector_from=inp, a_entity_from="Flow", a_region_from="*",
            a_sector_to=proc, a_entity_to="Process", a_region_to="same",
            a_time="*", a_valuation="CC",
            b_sector_from=proc, b_entity_from="Process", b_region_from="same",
            b_sector_to=prod, b_entity_to="Flow", b_region_to="same",
            b_time="same", b_valuation="same",
            value_nominal=str(v),
        ))
    _write(path, rows)


def _conc_pair(suffix, sector, entity, side, val):
    """Build use (commodity on to) or supply (commodity on from) conc row."""
    b = dict(b_sector_from="same", b_entity_from="same", b_region_from="same",
             b_sector_to="same", b_entity_to="same", b_region_to="same",
             b_time="same", b_valuation="CC")
    if side == "use":
        a = dict(a_sector_to=sector, a_entity_to=entity, a_valuation="Mass")
    else:
        a = dict(a_sector_from=sector, a_entity_from=entity, a_valuation="Mass")
    return _row(param_id=f"conc_{suffix}", value_nominal=str(val), **a, **b)


def write_conc_csv(path):
    rows = []
    flow_use = ["Ore", "Concentrate", "Matte", "CSXEW", "CER", "Cathode"]
    for s in flow_use:
        rows.append(_conc_pair(f"{s.lower()}_use", s, "Flow", "use", CONC[s]))
        rows.append(_conc_pair(f"{s.lower()}_supply", s, "Flow", "supply", CONC[s]))
    # SB commodities
    rows.append(_conc_pair("co_supply", "CO", "SB", "supply", CONC["CO"]))
    rows.append(_conc_pair("scrap_supply", "Scrap", "SB", "supply", CONC["Scrap"]))
    rows.append(_conc_pair("wr_use", "WR", "SB", "use", CONC["WR"]))
    rows.append(_conc_pair("tailings_use", "Tailings", "SB", "use", CONC["Tailings"]))
    rows.append(_conc_pair("slag_use", "Slag", "SB", "use", CONC["Slag"]))
    _write(path, rows)


def write_ore_grade_csv(path):
    rows = [_row(
        param_id="ore_grade_A_2020",
        a_sector_from="CO", a_entity_from="SB", a_region_from="A",
        a_time="2020", a_valuation="Mass",
        b_sector_from="same", b_entity_from="same", b_region_from="same",
        b_sector_to="same", b_entity_to="same", b_region_to="same",
        b_time="same", b_valuation="CC",
        value_nominal=str(ORE_GRADE_A),
    )]
    _write(path, rows)


@pytest.fixture
def acc():
    return MockAcc()


@pytest.fixture
def imputer(tmp_path, acc):
    eff = tmp_path / "eff.csv"
    conc = tmp_path / "conc.csv"
    og = tmp_path / "ore_grade.csv"
    write_eff_csv(eff)
    write_conc_csv(conc)
    write_ore_grade_csv(og)
    imp = NetworkImputer(str(eff), str(conc), str(og), p=1.3, cv_cap=float("inf"))
    imp.prepare(acc)
    return imp


# helpers ---------------------------------------------------------------------

def _run(imputer, acc, seeds):
    """seeds: dict (from_sector, to_sector, valuation) -> value. Returns f."""
    f = np.full(acc.n_edges, np.nan)
    sigma = np.full(acc.n_edges, np.nan)
    for (fs, ts, v), val in seeds.items():
        e = acc.eid(fs, ts, v)
        f[e] = val
        sigma[e] = 0.01 * abs(val) if val else 0.01
    imputer.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)
    return f, sigma


# =============================================================================
# Tests
# =============================================================================

def test_recipe_skeleton(acc):
    recs = {r.process_sector: r for r in derive_process_recipes(acc)}
    assert set(recs) == {"Mining", "MF", "Smelting", "SXEW", "CP"}
    smelt = recs["Smelting"]
    # pooled inputs: Concentrate + Scrap (both use edges into Smelting)
    in_secs = {acc.edge_meta.loc[acc.edge_meta.edge_id == i, "from_sector"].iloc[0]
               for i in smelt.inputs}
    assert in_secs == {"Concentrate", "Scrap"}
    assert acc.edge_meta.loc[acc.edge_meta.edge_id == smelt.product, "to_sector"].iloc[0] == "Matte"
    waste_secs = {acc.edge_meta.loc[acc.edge_meta.edge_id == w, "to_sector"].iloc[0]
                  for w in smelt.wastes}
    assert waste_secs == {"Slag"}


def test_forward_product_waste_and_mass_twin(imputer, acc):
    f, sigma = _run(imputer, acc, {("Ore", "MF", "CC"): 100.0})
    # MF: Concentrate = 0.8*100, Tailings = 0.2*100
    assert f[acc.eid("MF", "Concentrate", "CC")] == pytest.approx(80.0)
    assert f[acc.eid("MF", "Tailings", "CC")] == pytest.approx(20.0)
    # Mass twins via conc
    assert f[acc.eid("MF", "Concentrate", "Mass")] == pytest.approx(80.0 / 0.275)
    assert f[acc.eid("MF", "Tailings", "Mass")] == pytest.approx(20.0 / 0.0055)
    # σ on the product = eff * σ_in (σ_in=1.0 here), no param term (sd=0.001*…)
    assert sigma[acc.eid("MF", "Concentrate", "CC")] > 0


def test_two_hop_fixpoint(imputer, acc):
    # Seed only Ore->MF CC; expect Matte = 0.965 * (0.8*100) across the Flow node
    f, _ = _run(imputer, acc, {("Ore", "MF", "CC"): 100.0})
    # MF->Concentrate -> [Concentrate flow balance] -> Concentrate->Smelting -> Matte
    assert f[acc.eid("Concentrate", "Smelting", "CC")] == pytest.approx(80.0)
    assert f[acc.eid("Smelting", "Matte", "CC")] == pytest.approx(0.965 * 80.0)
    assert f[acc.eid("Smelting", "Slag", "CC")] == pytest.approx((1 - 0.965) * 80.0)


def test_backward_multi_input_subtraction(imputer, acc):
    # Matte + Scrap known, Concentrate missing -> Concentrate = Matte/eff - Scrap
    f, _ = _run(imputer, acc, {
        ("Smelting", "Matte", "CC"): 96.5,
        ("Scrap", "Smelting", "CC"): 10.0,
    })
    S = 96.5 / 0.965  # = 100 total input
    assert f[acc.eid("Concentrate", "Smelting", "CC")] == pytest.approx(S - 10.0)


def test_ore_split_from_sxew_anchor(imputer, acc):
    # Mining->Ore total known + CSXEW (SXEW product) known -> split resolves:
    #   Ore->SXEW = CSXEW/eff_sxew ; Ore->MF = Ore_total - Ore->SXEW
    f, _ = _run(imputer, acc, {
        ("Mining", "Ore", "CC"): 100.0,
        ("SXEW", "CSXEW", "CC"): 36.25,   # = 0.725 * 50
    })
    assert f[acc.eid("Ore", "SXEW", "CC")] == pytest.approx(50.0)
    assert f[acc.eid("Ore", "MF", "CC")] == pytest.approx(50.0)


def test_ore_split_without_anchor_left_nan(imputer, acc):
    # Only Mining->Ore known: the MF/SXEW split is free -> both left NaN
    f, _ = _run(imputer, acc, {("Mining", "Ore", "CC"): 100.0})
    assert np.isnan(f[acc.eid("Ore", "MF", "CC")])
    assert np.isnan(f[acc.eid("Ore", "SXEW", "CC")])


def test_mining_wr_mass_residual(imputer, acc):
    f, sigma = _run(imputer, acc, {("CO", "Mining", "CC"): 100.0})
    assert f[acc.eid("Mining", "Ore", "CC")] == pytest.approx(100.0)   # eff=1
    assert f[acc.eid("Mining", "WR", "CC")] == pytest.approx(0.0)
    co_mass = f[acc.eid("CO", "Mining", "Mass")]
    ore_mass = f[acc.eid("Mining", "Ore", "Mass")]
    assert co_mass == pytest.approx(100.0 / ORE_GRADE_A)   # ore_grade, not conc_co
    assert ore_mass == pytest.approx(100.0 / 0.009)
    assert f[acc.eid("Mining", "WR", "Mass")] == pytest.approx(co_mass - ore_mass)


def test_cp_pool_eff_one(imputer, acc):
    f, _ = _run(imputer, acc, {
        ("CSXEW", "CP", "CC"): 30.0,
        ("CER", "CP", "CC"): 70.0,
    })
    assert f[acc.eid("CP", "Cathode", "CC")] == pytest.approx(100.0)
    assert f[acc.eid("CP", "PSC", "CC")] == pytest.approx(0.0)


def test_no_overwrite_observed(imputer, acc):
    # Observed product should not be overwritten by the recipe
    f = np.full(acc.n_edges, np.nan)
    sigma = np.full(acc.n_edges, np.nan)
    e_in = acc.eid("Ore", "MF", "CC")
    e_out = acc.eid("MF", "Concentrate", "CC")
    f[e_in] = 100.0; sigma[e_in] = 1.0
    f[e_out] = 999.0; sigma[e_out] = 2.0   # observed, "wrong" vs 0.8*100
    imputer.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)
    assert f[e_out] == 999.0   # untouched
    assert sigma[e_out] == 2.0


def test_conc_by_construction(imputer, acc):
    # Every written CC/Mass twin satisfies CC = conc * Mass
    f, _ = _run(imputer, acc, {("Ore", "MF", "CC"): 100.0})
    for s in ("Concentrate", "Matte"):
        cc = f[acc.eid("MF" if s == "Concentrate" else "Smelting", s, "CC")]
        mass = f[acc.eid("MF" if s == "Concentrate" else "Smelting", s, "Mass")]
        assert cc == pytest.approx(CONC[s] * mass, rel=1e-9)
