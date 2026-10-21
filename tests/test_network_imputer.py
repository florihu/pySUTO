"""
test_network_imputer.py
=======================
Unit suite for pysuto.impute.network.NetworkImputer — the process-recipe
traversal imputer.

A small synthetic copper chain (one region, one year) exercises:
  * forward recipe closure (product + SEM waste) and the Mass twin
  * two-hop propagation across a Flow node (fixpoint)
  * backward multi-input subtraction (pooled Scrap)
  * the Ore branch split resolved from the SXEW anchor (+ negative residual: clipped
    to the edge bound, σ-widened, and not propagated downstream)
  * the Mining WR Mass edge is k·Ore at the flat σ (satellite, fixed in Stage 3; ADR 0038)
  * determinacy / no-overwrite
  * CP pooling (eff = 1)
  * conc-by-construction: written CC/Mass twins satisfy CC = conc·Mass

A second, **two-year** fixture (``acc2``) guards the time-pooling regression: the
Flow-node balance must be keyed per ``(node, year)``. With a single year the bug is
invisible (one edge per node), so these tests use two years with distinct per-year
values and assert no cross-year contamination — they fail on the pre-fix code where
``_build_flow_balances`` pooled all years onto the time-agnostic node id.
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
    "WR": 0.0, "Tailings": 0.0055, "Slag": 0.0275, "CO": 0.01,  # WR copperless (conc=0)
}
ORE_GRADE_A = 0.005  # region/time specific CO grade (overrides conc_co_supply)


class _Bounds:
    def __init__(self, n):
        self.l = np.zeros(n, dtype=float)
        self.u = np.full(n, 1e9, dtype=float)


class MockAcc:
    """Minimal AccountingStructure stand-in for the synthetic chain."""

    def __init__(self, times=("2020",)):
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
            for t in times:
                for val in ("Mass", "CC"):
                    rows.append((eid, sec_to_id[fs], sec_to_id[ts], t, val))
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
    def eid(self, fs, ts, val, time="2020"):
        m = self.edge_meta
        hit = m[(m["from_sector"] == fs) & (m["to_sector"] == ts)
                & (m["valuation"] == val) & (m["time"] == time)]
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


def write_eff_csv(path, stripping_ratio: float | None = 2.0):
    rows = []
    for (inp, proc, prod), v in EFF.items():
        rows.append(_row(
            param_id=f"eff_{proc.lower()}", param_type="efficiency",
            a_sector_from=inp, a_entity_from="Flow", a_region_from="*",
            a_sector_to=proc, a_entity_to="Process", a_region_to="same",
            a_time="*", a_valuation="CC",
            b_sector_from=proc, b_entity_from="Process", b_region_from="same",
            b_sector_to=prod, b_entity_to="Flow", b_region_to="same",
            b_time="same", b_valuation="same",
            value_nominal=str(v),
        ))
    if stripping_ratio is not None:
        rows.append(_stripping_row(stripping_ratio))
    _write(path, rows)


def _stripping_row(k: float, region: str = "*"):
    """WR_Mass = k · Ore_Mass: A = Mining→Ore Mass, B = Mining→WR Mass."""
    return _row(
        param_id="strippingratio", param_type="stripping_ratio",
        a_sector_from="Mining", a_entity_from="Process", a_region_from=region,
        a_sector_to="Ore", a_entity_to="Flow", a_region_to="same",
        a_time="*", a_valuation="Mass",
        b_sector_from="Mining", b_entity_from="Process", b_region_from="same",
        b_sector_to="WR", b_entity_to="SB", b_region_to="same",
        b_time="same", b_valuation="same",
        value_nominal=str(k),
    )


def write_stripping_csv(path, k_by_region: dict):
    """The stripping_ratio params file (own family since ADR 0038); "*" = all regions."""
    _write(path, [_stripping_row(k, region) for region, k in k_by_region.items()])


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


def test_negative_balance_clipped_not_propagated(imputer, acc):
    """A negative Ore->SXEW residual (Mining->Ore - Ore->MF < 0, as in a region with
    little/no SXEW) is clipped to the edge bound (0) with σ widened to the clip gap,
    tagged `_clipped`, and does NOT propagate: the SXEW branch closes at 0 and
    CP->Cathode keeps the full CER (pyro) contribution instead of being dragged to 0."""
    f = np.full(acc.n_edges, np.nan)
    sigma = np.full(acc.n_edges, np.nan)
    seeds = {
        ("Mining", "Ore", "CC"): 100.0,   # ore supply
        ("Ore", "MF", "CC"):     120.0,   # flotation feed > ore  -> Ore->SXEW = -20
        ("CER", "CP", "CC"):     70.0,    # pyro leg into the CP pool
    }
    for (fs, ts, v), val in seeds.items():
        e = acc.eid(fs, ts, v)
        f[e] = val
        sigma[e] = 0.01 * val
    entries = imputer.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)

    e_sxew = acc.eid("Ore", "SXEW", "CC")
    # clipped to the lower bound, not left negative
    assert f[e_sxew] == pytest.approx(0.0)
    rec = next(e for e in entries if e["edge_id"] == e_sxew)
    assert rec["method"].endswith("_clipped")
    assert sigma[e_sxew] >= 20.0            # σ widened to at least the clip gap (|−20|)

    # negativity did not propagate down the SXEW branch
    assert f[acc.eid("SXEW", "CSXEW", "CC")] == pytest.approx(0.0)
    assert f[acc.eid("CSXEW", "CP", "CC")]   == pytest.approx(0.0)
    # CP->Cathode keeps the CER contribution (eff_CP = 1), NOT dragged to 0
    assert f[acc.eid("CP", "Cathode", "CC")] == pytest.approx(70.0)


def _stripping_imputer(tmp_path, acc, k_by_region):
    eff, conc, og, sr = (tmp_path / n for n in ("eff.csv", "conc.csv", "og.csv", "sr.csv"))
    write_eff_csv(eff, stripping_ratio=None)
    write_conc_csv(conc)
    write_ore_grade_csv(og)
    write_stripping_csv(sr, k_by_region)
    imp = NetworkImputer(str(eff), str(conc), str(og), p=1.3, cv_cap=1.5,
                         stripping_ratio_csv=str(sr), stripping_sigma=1.0e6)
    imp.prepare(acc)
    return imp


def test_mining_fills_waste_rock_at_the_flat_sigma(tmp_path, acc):
    """WR is a satellite mass flow fixed in Stage 3 by the hard stripping relation
    (ADR 0038): the prior carries WR Mass = k·Ore Mass, consistent with that row,
    at the flat σ — WR has no data of its own, so the cv cap must not tighten it
    (an informative WR prior would count the Ore prior twice through the hard row)."""
    imp = _stripping_imputer(tmp_path, acc, {"*": 2.0})
    f, sigma = _run(imp, acc, {("CO", "Mining", "CC"): 100.0})
    assert f[acc.eid("Mining", "Ore", "Mass")] == pytest.approx(100.0 / 0.009)
    assert f[acc.eid("CO", "Mining", "Mass")] == pytest.approx(100.0 / ORE_GRADE_A)
    assert f[acc.eid("Mining", "WR", "CC")] == pytest.approx(0.0)      # copperless
    assert f[acc.eid("Mining", "WR", "Mass")] == pytest.approx(2.0 * 100.0 / 0.009)
    assert sigma[acc.eid("Mining", "WR", "Mass")] == pytest.approx(1.0e6)


def test_waste_rock_takes_the_most_specific_stripping_row(tmp_path, acc):
    """A region row beats the global row, as for eff and conc."""
    imp = _stripping_imputer(tmp_path, acc, {"*": 2.0, "A": 3.0})
    f, _ = _run(imp, acc, {("CO", "Mining", "CC"): 100.0})
    assert f[acc.eid("Mining", "WR", "Mass")] == pytest.approx(3.0 * 100.0 / 0.009)


def test_cp_pool_eff_one(imputer, acc):
    f, _ = _run(imputer, acc, {
        ("CSXEW", "CP", "CC"): 30.0,
        ("CER", "CP", "CC"): 70.0,
    })
    assert f[acc.eid("CP", "Cathode", "CC")] == pytest.approx(100.0)
    assert f[acc.eid("CP", "PSC", "CC")] == pytest.approx(0.0)


def test_effless_cp_psc_zero_carries_the_resolution_sigma(tmp_path, acc):
    """#36 item 2: CP has no eff row, so the recipe writes PSC = (1 − 1.0)·Σin = 0 at
    σ 0 — a missing parameter, not an observation, and Stage 3 floored it to a σ
    1e-6 pin. On the signed leg (structural l < 0) the zero is priced at the feeds'
    resolution instead; its Mass twin follows through the conc row. Nothing else
    moves: every value is unchanged, and Mining's eff-less copperless WR CC, an
    unsigned output, keeps its exact zero."""
    for val in ("CC", "Mass"):
        acc.bounds.l[acc.eid("CP", "PSC", val)] = -1.0e6      # signed, as in the acc
    eff, conc, og = tmp_path / "eff.csv", tmp_path / "conc.csv", tmp_path / "og.csv"
    write_eff_csv(eff); write_ore_grade_csv(og)
    write_conc_csv(conc)
    rows = pd.read_csv(conc, sep=";", dtype=str, keep_default_na=False).to_dict("records")
    _write(conc, rows + [_conc_pair("psc_use", "PSC", "SB", "use", 0.9975)])

    def run(**kw):
        imp = NetworkImputer(str(eff), str(conc), str(og), p=1.3, cv_cap=float("inf"), **kw)
        imp.prepare(acc)
        return _run(imp, acc, {("CSXEW", "CP", "CC"): 30.0, ("CER", "CP", "CC"): 70.0,
                               ("CO", "Mining", "CC"): 100.0})

    f0, s0 = run()
    f1, s1 = run(effless_stock_sigma=0.3)
    e_cc, e_m = acc.eid("CP", "PSC", "CC"), acc.eid("CP", "PSC", "Mass")
    np.testing.assert_array_equal(f1, f0)
    assert f1[e_cc] == 0.0 and f1[e_m] == 0.0                 # the prior stays 0
    assert s0[e_cc] == 0.0 and s0[e_m] == 0.0                 # the hidden pin
    assert s1[e_cc] == pytest.approx(0.3)
    assert s1[e_m] == pytest.approx(0.3 / 0.9975)
    e_wr = acc.eid("Mining", "WR", "CC")
    assert f1[e_wr] == 0.0 and s1[e_wr] == 0.0
    moved = ~((s1 == s0) | (np.isnan(s1) & np.isnan(s0)))
    np.testing.assert_array_equal(np.flatnonzero(moved), sorted([e_cc, e_m]))


def test_cp_pool_partial_primary_counts_missing_as_zero(imputer, acc):
    # ADR 0010 amendment: a missing pooled primary counts as 0 when ≥1 primary
    # is known — CP's CER/CSXEW are parallel technology routes (a region with
    # refining but no SX-EW, e.g. Japan, must still close its CP recipe).
    f, _ = _run(imputer, acc, {("CER", "CP", "CC"): 70.0})
    assert f[acc.eid("CP", "Cathode", "CC")] == pytest.approx(70.0)
    # the absent route is never written — it stays an honest gap
    assert np.isnan(f[acc.eid("CSXEW", "CP", "CC")])


def test_cp_pool_all_primaries_missing_still_blocks(imputer, acc):
    # Guard: with NO known primary the gate stays closed (no fabricated zeros),
    # even while unrelated recipes fire.
    f, _ = _run(imputer, acc, {("CO", "Mining", "CC"): 100.0})
    assert np.isnan(f[acc.eid("CP", "Cathode", "CC")])


def test_pure_secondary_process_closes_on_scrap_alone(imputer, acc):
    """Round P.1: a 100 %-secondary process has NO primary by construction.

    Seeding only the SEM scrap leg must still close the recipe at convergence.
    Slovakia's smelter is 100 % scrap-fed: ICSG reports ~46.6 kt/yr into it and
    the prior carried 0 kt out of it for 15 region-years, which pinned its Matte
    scale total at c = 0.011 kt and strangled the whole Slovak chain.
    """
    f, _ = _run(imputer, acc, {("Scrap", "Smelting", "CC"): 50.0})
    assert f[acc.eid("Smelting", "Matte", "CC")] == pytest.approx(0.965 * 50.0)
    assert f[acc.eid("Smelting", "Slag", "CC")] == pytest.approx(0.035 * 50.0)
    # the absent primary route is never written — it stays an honest gap
    assert np.isnan(f[acc.eid("Concentrate", "Smelting", "CC")])


def test_process_with_neither_primary_nor_secondary_still_blocks(imputer, acc):
    """The guard that keeps the relaxation honest: no known input of ANY kind
    means no fabricated output (the ADR 0010 rule this amends, not replaces)."""
    f, _ = _run(imputer, acc, {("CO", "Mining", "CC"): 100.0})
    assert np.isnan(f[acc.eid("Smelting", "Matte", "CC")])


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


# =============================================================================
# Two-year time-pooling regression (the CP-pool gap fix)
# =============================================================================

@pytest.fixture
def acc2():
    return MockAcc(times=("2020", "2021"))


@pytest.fixture
def imputer2(tmp_path, acc2):
    eff = tmp_path / "eff.csv"
    conc = tmp_path / "conc.csv"
    og = tmp_path / "ore_grade.csv"
    write_eff_csv(eff)
    write_conc_csv(conc)
    write_ore_grade_csv(og)
    imp = NetworkImputer(str(eff), str(conc), str(og), p=1.3, cv_cap=float("inf"))
    imp.prepare(acc2)
    return imp


def _run2(imputer, acc, seeds):
    """seeds: (from_sector, to_sector, valuation, time) -> value.

    Returns (f, method) where method maps edge_id -> log method.
    """
    f = np.full(acc.n_edges, np.nan)
    sigma = np.full(acc.n_edges, np.nan)
    for (fs, ts, v, t), val in seeds.items():
        e = acc.eid(fs, ts, v, t)
        f[e] = val
        sigma[e] = 0.01 * abs(val) if val else 0.01
    entries = imputer.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)
    return f, {en["edge_id"]: en["method"] for en in entries}


def test_flow_balance_fires_per_year_no_cross_contamination(imputer2, acc2):
    # Concentrate flow: seed the supply leg MF->Concentrate per year with DIFFERENT
    # values; the only use leg Concentrate->Smelting must fill from THAT year's
    # supply. Pre-fix the two years pool onto one node (2 unknown use legs) and the
    # balance never fires -> both stay NaN. This test is the direct guard.
    f, method = _run2(imputer2, acc2, {
        ("MF", "Concentrate", "CC", "2020"): 80.0,
        ("MF", "Concentrate", "CC", "2021"): 40.0,
    })
    e20 = acc2.eid("Concentrate", "Smelting", "CC", "2020")
    e21 = acc2.eid("Concentrate", "Smelting", "CC", "2021")
    assert f[e20] == pytest.approx(80.0)   # 2020 supply — not leaked from 2021
    assert f[e21] == pytest.approx(40.0)
    assert method[e20] == "network_balance"
    assert method[e21] == "network_balance"


def test_cp_pool_closes_per_year_via_flow_balance(imputer2, acc2):
    # The reported gap: CSXEW->CP is derived from the SXEW->CSXEW flow per year, then
    # the CP recipe closes Cathode = CSXEW->CP + CER->CP with PSC = 0 — independently
    # per year (distinct values prove no cross-year leak). CER->CP is seeded because
    # the synthetic chain has no ER supply into the CER flow.
    f, method = _run2(imputer2, acc2, {
        ("SXEW", "CSXEW", "CC", "2020"): 30.0,
        ("SXEW", "CSXEW", "CC", "2021"): 50.0,
        ("CER", "CP", "CC", "2020"): 70.0,
        ("CER", "CP", "CC", "2021"): 20.0,
    })
    for t, csx, cer in (("2020", 30.0, 70.0), ("2021", 50.0, 20.0)):
        e_csxcp = acc2.eid("CSXEW", "CP", "CC", t)
        assert f[e_csxcp] == pytest.approx(csx)          # filled from the flow
        assert method[e_csxcp] == "network_balance"
        assert f[acc2.eid("CP", "Cathode", "CC", t)] == pytest.approx(csx + cer)
        assert f[acc2.eid("CP", "PSC", "CC", t)] == pytest.approx(0.0)


# =============================================================================
# Two-region bilateral-import gate fixture (ADR 0010)
# =============================================================================
#
# Minimal cross-region topology to exercise the bilateral_import recipe tier and
# the flow-balance cross-region-as-0 relaxation. Regions A, B:
#
#   Conc(A) ─┐                          (domestic primary into Sm_A)
#   Conc(B) ─┴► Sm(A) ─► Matte(A) ─┬─► CFR(A)   (domestic use of Matte_A)
#             (import)   │          └─► CFR(B)   (cross-region export of Matte_A)
#                        └► Slag(A)   (SEM waste)
#
_BI_NODES = [
    ("Conc",  "Flow",    "A", None),
    ("Conc",  "Flow",    "B", None),
    ("Sm",    "Process", "A", None),
    ("Matte", "Flow",    "A", None),
    ("Slag",  "SB",      "A", "SEM"),
    ("CFR",   "Process", "A", None),
    ("CFR",   "Process", "B", None),
]
# (from_sector, from_region, to_sector, to_region)
_BI_EDGES = [
    ("Conc",  "A", "Sm",    "A"),   # domestic primary
    ("Conc",  "B", "Sm",    "A"),   # cross-region import (bilateral_import)
    ("Sm",    "A", "Matte", "A"),   # product
    ("Sm",    "A", "Slag",  "A"),   # SEM waste
    ("Matte", "A", "CFR",   "A"),   # domestic use of Matte_A (flow-balance target)
    ("Matte", "A", "CFR",   "B"),   # cross-region export of Matte_A
]
_BI_EFF  = 0.9
_BI_CONC = {"Conc": 0.275, "Matte": 0.525, "Slag": 0.0275}


class _BiAcc:
    """Minimal two-region AccountingStructure stand-in for the gate tests."""

    def __init__(self):
        self.node_index = pd.DataFrame(
            _BI_NODES, columns=["sector", "entity", "region", "entity_sub"],
        )
        self.node_index.index = range(len(_BI_NODES))
        nid = {(s, r): i for i, (s, _e, r, _sub) in enumerate(_BI_NODES)}
        rows, eid = [], 0
        for fs, fr, ts, tr in _BI_EDGES:
            for val in ("Mass", "CC"):
                rows.append((eid, nid[(fs, fr)], nid[(ts, tr)], "2020", val))
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

    def eid(self, fs, fr, ts, tr, val):
        m = self.edge_meta
        hit = m[(m["from_sector"] == fs) & (m["from_region"] == fr)
                & (m["to_sector"] == ts) & (m["to_region"] == tr)
                & (m["valuation"] == val)]
        return int(hit["edge_id"].iloc[0])


def _write_bi_eff(path):
    _write(path, [_row(
        param_id="eff_sm", param_type="efficiency",
        a_sector_from="Conc", a_entity_from="Flow", a_region_from="*",
        a_sector_to="Sm", a_entity_to="Process", a_region_to="same",
        a_time="*", a_valuation="CC",
        b_sector_from="Sm", b_entity_from="Process", b_region_from="same",
        b_sector_to="Matte", b_entity_to="Flow", b_region_to="same",
        b_time="same", b_valuation="same",
        value_nominal=str(_BI_EFF),
    )])


def _write_bi_conc(path):
    rows = []
    for s in ("Conc", "Matte"):
        rows.append(_conc_pair(f"{s.lower()}_use", s, "Flow", "use", _BI_CONC[s]))
        rows.append(_conc_pair(f"{s.lower()}_supply", s, "Flow", "supply", _BI_CONC[s]))
    rows.append(_conc_pair("slag_use", "Slag", "SB", "use", _BI_CONC["Slag"]))
    _write(path, rows)


@pytest.fixture
def bi_imputer(tmp_path):
    acc = _BiAcc()
    eff = tmp_path / "bi_eff.csv"
    conc = tmp_path / "bi_conc.csv"
    _write_bi_eff(eff)
    _write_bi_conc(conc)
    imp = NetworkImputer(str(eff), str(conc), p=1.3, cv_cap=float("inf"))
    imp.prepare(acc)
    return imp, acc


def _bi_run(imp, acc, seeds):
    f = np.full(acc.n_edges, np.nan)
    sigma = np.full(acc.n_edges, np.nan)
    for (fs, fr, ts, tr, v), val in seeds.items():
        e = acc.eid(fs, fr, ts, tr, v)
        f[e] = val
        sigma[e] = 0.01 * abs(val) if val else 0.01
    imp.impute(f, sigma, np.isnan(f), step=0, acc_struct=acc)
    return f, sigma


def test_bilateral_import_classification(bi_imputer):
    imp, acc = bi_imputer
    prod = acc.eid("Sm", "A", "Matte", "A", "CC")
    rec = next(r for r in imp._recipes if r.product == prod)
    dom = acc.eid("Conc", "A", "Sm", "A", "CC")
    imp_leg = acc.eid("Conc", "B", "Sm", "A", "CC")
    assert dom in rec.primary and dom not in rec.bilateral_import
    assert imp_leg in rec.bilateral_import and imp_leg not in rec.primary


def test_gate_opens_with_missing_import(bi_imputer):
    # domestic primary known, cross-region import MISSING → gate opens, import = 0.
    imp, acc = bi_imputer
    f, _ = _bi_run(imp, acc, {("Conc", "A", "Sm", "A", "CC"): 100.0})
    prod = acc.eid("Sm", "A", "Matte", "A", "CC")
    assert f[prod] == pytest.approx(_BI_EFF * 100.0)  # eff·(domestic + 0)


def test_import_pooled_when_known(bi_imputer):
    imp, acc = bi_imputer
    f, _ = _bi_run(imp, acc, {
        ("Conc", "A", "Sm", "A", "CC"): 100.0,
        ("Conc", "B", "Sm", "A", "CC"): 50.0,
    })
    prod = acc.eid("Sm", "A", "Matte", "A", "CC")
    assert f[prod] == pytest.approx(_BI_EFF * 150.0)  # eff·(domestic + import)


def test_backward_subtracts_import(bi_imputer):
    # product + import known, domestic primary missing → domestic = S − import.
    imp, acc = bi_imputer
    f, _ = _bi_run(imp, acc, {
        ("Sm", "A", "Matte", "A", "CC"): 90.0,
        ("Conc", "B", "Sm", "A", "CC"): 50.0,
    })
    dom = acc.eid("Conc", "A", "Sm", "A", "CC")
    assert f[dom] == pytest.approx(90.0 / _BI_EFF - 50.0)  # 100 − 50


def test_flow_balance_missing_cross_region_as_zero(bi_imputer):
    # Matte supply known; domestic use unknown; cross-region export MISSING → 0.
    imp, acc = bi_imputer
    f, _ = _bi_run(imp, acc, {("Sm", "A", "Matte", "A", "CC"): 135.0})
    dom_use = acc.eid("Matte", "A", "CFR", "A", "CC")
    exp_leg = acc.eid("Matte", "A", "CFR", "B", "CC")
    assert f[dom_use] == pytest.approx(135.0)  # export counted as 0
    assert np.isnan(f[exp_leg])                # cross-region leg never written


# =============================================================================
# Region-resolved ore-grade overlay (ADR 0015)
# =============================================================================

def test_region_resolved_grade_overlay_twin_map(tmp_path):
    """Region/time-resolved ore_grade rows overlay the global conc twin map
    per edge (most-specific wins), leaving un-overlaid sectors on the global
    value — the mechanism that makes CO/Ore Mass twins region×year-aware."""
    from pysuto.param_relation import (
        ParamRelationResolver, conc_twin_map, get_edge_cache,
    )

    acc = _BiAcc()
    conc = tmp_path / "c.csv"
    og = tmp_path / "og.csv"
    _write_bi_conc(conc)  # global Conc = 0.275

    def _og_row(pid, region, val):
        return _row(
            param_id=pid, param_type="concentration",
            a_sector_from="Conc", a_entity_from="Flow", a_region_from=region,
            a_time="2020", a_valuation="Mass",
            b_sector_from="same", b_entity_from="same", b_region_from="same",
            b_sector_to="same", b_entity_to="same", b_region_to="same",
            b_time="same", b_valuation="CC",
            value_nominal=str(val),
        )

    _write(og, [_og_row("oregradeconcsupplya2020", "A", 0.2),
                _og_row("oregradeconcsupplyb2020", "B", 0.4)])

    get_edge_cache(acc)
    resolver = ParamRelationResolver(acc)
    m = conc_twin_map(str(conc), resolver)
    conc_twin_map(str(og), resolver, m)

    cc_a = acc.eid("Conc", "A", "Sm", "A", "CC")
    cc_b = acc.eid("Conc", "B", "Sm", "A", "CC")
    assert m[cc_a][0] == acc.eid("Conc", "A", "Sm", "A", "Mass")
    assert m[cc_a][1] == pytest.approx(0.2)   # region A overlay wins
    assert m[cc_b][1] == pytest.approx(0.4)   # region B overlay wins
    cc_matte = acc.eid("Matte", "A", "CFR", "A", "CC")
    assert m[cc_matte][1] == pytest.approx(_BI_CONC["Matte"])  # untouched


# =============================================================================
# Stock-change legs are never written (Round X, ADR 0040)
# =============================================================================

_STOCK_NODES = NODES + [("FU", "SB", "FD"), ("MESC", "SB", "FD")]
_STOCK_EDGES = EDGES + [("Cathode", "FU"), ("Cathode", "MESC")]


@pytest.fixture
def stock_acc(monkeypatch):
    import test_network_imputer as mod
    monkeypatch.setattr(mod, "NODES", _STOCK_NODES)
    monkeypatch.setattr(mod, "EDGES", _STOCK_EDGES)
    return MockAcc()


def _stock_imputer(tmp_path, acc, stock_sectors):
    eff = tmp_path / "eff.csv"; conc = tmp_path / "conc.csv"; og = tmp_path / "ore_grade.csv"
    write_eff_csv(eff); write_conc_csv(conc); write_ore_grade_csv(og)
    imp = NetworkImputer(str(eff), str(conc), str(og), p=1.3, cv_cap=float("inf"),
                         stock_sectors=stock_sectors)
    imp.prepare(acc)
    return imp


def test_flow_balance_never_writes_a_stock_leg(stock_acc, tmp_path):
    """Cathode node: supply 100, FU 80 known, MESC unknown. Without the guard the
    balance books the 20 kt residual on the stock leg (the sink pre-baked into the
    prior); with stock_sectors=("MESC",) the leg stays NaN for the net-stock seed."""
    seeds = {("CP", "Cathode", "CC", "2020"): 100.0, ("Cathode", "FU", "CC", "2020"): 80.0}
    e_mesc = stock_acc.eid("Cathode", "MESC", "CC")

    f, method = _run2(_stock_imputer(tmp_path, stock_acc, ()), stock_acc, seeds)
    assert f[e_mesc] == pytest.approx(20.0) and method[e_mesc] == "network_balance"

    f, method = _run2(_stock_imputer(tmp_path, stock_acc, ("MESC",)), stock_acc, seeds)
    assert np.isnan(f[e_mesc]) and e_mesc not in method


def test_flow_balance_counts_a_missing_stock_leg_as_zero(stock_acc, tmp_path):
    """Supply 100 known, FU and MESC unknown: the stock leg is non-writable and
    counts as 0, so FU closes at 100 (symmetric with the cross-region tier)."""
    seeds = {("CP", "Cathode", "CC", "2020"): 100.0}
    e_fu, e_mesc = stock_acc.eid("Cathode", "FU", "CC"), stock_acc.eid("Cathode", "MESC", "CC")
    f, method = _run2(_stock_imputer(tmp_path, stock_acc, ("MESC",)), stock_acc, seeds)
    assert f[e_fu] == pytest.approx(100.0) and method[e_fu] == "network_balance"
    assert np.isnan(f[e_mesc])


def test_flow_balance_never_derives_supply_from_uses_that_include_a_stock_leg(stock_acc, tmp_path):
    """FU 80 known, MESC unknown (counts as 0), CP→Cathode unknown: the balance must
    not write production = 80 from the uses — production is the recipe's job and the
    stock leg's 0 is an expectation, not a report. Without a stock leg on the node
    (CSXEW: SXEW→CSXEW unknown, CSXEW→CP known) the supply closure still fires."""
    seeds = {("Cathode", "FU", "CC", "2020"): 80.0, ("CSXEW", "CP", "CC", "2020"): 30.0}
    imp = _stock_imputer(tmp_path, stock_acc, ("MESC",))
    f, method = _run2(imp, stock_acc, seeds)
    e_prod = stock_acc.eid("CP", "Cathode", "CC")
    assert not method.get(e_prod, "").startswith("network_balance")   # incl. _capped/_clipped
    e_sx = stock_acc.eid("SXEW", "CSXEW", "CC")
    assert f[e_sx] == pytest.approx(30.0) and method[e_sx] == "network_balance"


def test_flow_balance_never_refills_an_observed_zero(stock_acc, tmp_path):
    """Supply 100 known, the domestic FU leg fed as an observed zero (#76): it is
    covered, so the balance leaves it at 0 instead of closing it at 100."""
    seeds = {("CP", "Cathode", "CC", "2020"): 100.0, ("Cathode", "FU", "CC", "2020"): 0.0}
    e_fu = stock_acc.eid("Cathode", "FU", "CC")
    f, method = _run2(_stock_imputer(tmp_path, stock_acc, ("MESC",)), stock_acc, seeds)
    assert f[e_fu] == 0.0 and e_fu not in method
