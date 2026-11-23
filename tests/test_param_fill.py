"""
test_trade_imputer.py
=====================
Unit suite for the trade fill (a forward ``Fill`` of the conc relation source, scope
cross-region Flow→Process content edges) + the trade→network handoff
that revives the deep refining chain (ADR 0010).

A minimal **two-region** copper chain ``Matte→CFR→Anode→ER→CER→CP→Cathode``
(region A), with a cross-region anode **import** ``Anode(B)→ER(A)`` and a
cross-region anode **export** ``Anode(A)→ER(B)``, exercises:

  * the trade fill seeds the import CC leg from its BACI Mass twin (``trade_seed``);
  * with the import seeded, the NetworkImputer recipe gate opens, the Anode
    flow-balance closes the lone domestic ``Anode(A)→ER(A)`` leg (the export
    route is uncovered → treated as 0), and ``ER→CER→CP`` fill via the recipe /
    flow-balance instead of falling to flat-prior;
  * the trade-seed σ matches the inverted conc twin;
  * no conc row ⇒ the trade fill is a no-op;
  * the source's twin and pair readings, and the parametric σ rule's both directions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.acc import ValuationRoles
from pysuto.impute import FillImputer, NetworkImputer, run_imputer
from pysuto.lp import lp_combine
from pysuto.relation.sources import Fill, ParamSource, trade_scope


# =============================================================================
# Two-region synthetic refining chain
# =============================================================================

# (sector, entity, region, entity_sub)
_NODES = [
    ("Matte",   "Flow",    "A", None),
    ("CFR",     "Process", "A", None),
    ("Anode",   "Flow",    "A", None),
    ("Anode",   "Flow",    "B", None),
    ("ER",      "Process", "A", None),
    ("ER",      "Process", "B", None),
    ("CER",     "Flow",    "A", None),
    ("CP",      "Process", "A", None),
    ("Cathode", "Flow",    "A", None),
]
# (from_sector, from_region, to_sector, to_region)
_EDGES = [
    ("Matte",   "A", "CFR",     "A"),   # domestic primary of CFR  (seeded)
    ("CFR",     "A", "Anode",   "A"),   # product of CFR
    ("Anode",   "A", "ER",      "A"),   # domestic primary of ER (← Anode_A balance)
    ("Anode",   "B", "ER",      "A"),   # cross-region import of ER (← trade-seed)
    ("Anode",   "A", "ER",      "B"),   # cross-region export of Anode_A (left NaN → 0)
    ("ER",      "A", "CER",     "A"),   # product of ER
    ("CER",     "A", "CP",      "A"),   # primary of CP
    ("CP",      "A", "Cathode", "A"),   # product of CP
]
_EFF  = {"CFR": 0.95, "ER": 0.997, "CP": 0.98}
_CONC = {"Matte": 0.525, "Anode": 0.94, "CER": 0.9975, "Cathode": 0.9975}


class _Bounds:
    def __init__(self, n):
        self.l = np.zeros(n, dtype=float)
        self.u = np.full(n, 1e9, dtype=float)


class _ChainAcc:
    def __init__(self):
        self.node_index = pd.DataFrame(
            _NODES, columns=["sector", "entity", "region", "entity_sub"],
        )
        self.node_index.index = range(len(_NODES))
        nid = {(s, r): i for i, (s, _e, r, _sub) in enumerate(_NODES)}
        rows, eid = [], 0
        for fs, fr, ts, tr in _EDGES:
            for val in ("Mass", "CC"):
                rows.append((eid, nid[(fs, fr)], nid[(ts, tr)], "2020", val))
                eid += 1
        self.edge_index = pd.DataFrame(
            rows, columns=["edge_id", "from_id", "to_id", "time", "valuation"]
        )
        self.n_edges = len(rows)
        self.bounds = _Bounds(self.n_edges)
        self.valuation_roles = ValuationRoles(content="CC", gross="Mass")
        self.boundary_roles = {}                          # the chain has no SB node
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

    def eid(self, fs, fr, ts, tr, val):
        m = self.edge_meta
        hit = m[(m["from_sector"] == fs) & (m["from_region"] == fr)
                & (m["to_sector"] == ts) & (m["to_region"] == tr)
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
    r.update(param_type="concentration", description="", unc_type="sd", value_sd="0.001",
             value_min="", value_max="", epsilon="0")
    r.update(kw)
    return r


def _write(path, rows):
    pd.DataFrame(rows, columns=_COLS).to_csv(path, sep=";", index=False)


def _eff_row(inp, proc, prod, v):
    return _row(
        param_id=f"eff_{proc.lower()}", param_type="efficiency",
        a_sector_from=inp, a_entity_from="Flow", a_region_from="*",
        a_sector_to=proc, a_entity_to="Process", a_region_to="same",
        a_time="*", a_valuation="CC",
        b_sector_from=proc, b_entity_from="Process", b_region_from="same",
        b_sector_to=prod, b_entity_to="Flow", b_region_to="same",
        b_time="same", b_valuation="same",
        value_nominal=str(v),
    )


def _conc_pair(sector, side, val):
    b = dict(b_sector_from="same", b_entity_from="same", b_region_from="same",
             b_sector_to="same", b_entity_to="same", b_region_to="same",
             b_time="same", b_valuation="CC")
    if side == "use":
        a = dict(a_sector_to=sector, a_entity_to="Flow", a_valuation="Mass")
    else:
        a = dict(a_sector_from=sector, a_entity_from="Flow", a_valuation="Mass")
    return _row(param_id=f"conc_{sector.lower()}_{side}", value_nominal=str(val), **a, **b)


def write_eff(path):
    _write(path, [
        _eff_row("Matte", "CFR", "Anode", _EFF["CFR"]),
        _eff_row("Anode", "ER",  "CER",   _EFF["ER"]),
        _eff_row("CER",   "CP",  "Cathode", _EFF["CP"]),
    ])


def write_conc(path, with_anode=True):
    rows = []
    for s in ("Matte", "Anode", "CER", "Cathode"):
        if s == "Anode" and not with_anode:
            continue
        rows.append(_conc_pair(s, "use", _CONC[s]))
        rows.append(_conc_pair(s, "supply", _CONC[s]))
    _write(path, rows)


# =============================================================================
# Helpers
# =============================================================================

def _build(tmp_path, with_anode_conc=True):
    acc = _ChainAcc()
    eff = tmp_path / "eff.csv"
    conc = tmp_path / "conc.csv"
    write_eff(eff)
    write_conc(conc, with_anode=with_anode_conc)
    trade = FillImputer(
        Fill(ParamSource.from_params(str(conc), acc, twin=True), "forward",
             trade_scope(acc), id="trade"),
        "trade", p=1.3,
    )
    network = NetworkImputer(str(eff), str(conc), p=1.3, cv_cap=float("inf"), q=3.0, alpha_min=0.05)
    network.prepare(acc)
    return acc, trade, network


def _run(acc, trade, network, seeds):
    """seeds: (fs, fr, ts, tr, val) -> value. Returns (f, sigma, method_by_edge)."""
    f = np.full(acc.n_edges, np.nan)
    sigma = np.full(acc.n_edges, np.nan)
    for (fs, fr, ts, tr, v), val in seeds.items():
        e = acc.eid(fs, fr, ts, tr, v)
        f[e] = val
        sigma[e] = 0.01 * abs(val) if val else 0.01
    entries = run_imputer(trade, f, sigma, 0, acc)       # trade first
    entries += run_imputer(network, f, sigma, 1, acc)    # then network (internal fixpoint)
    method = {e["edge_id"]: e["method"] for e in entries}
    return f, sigma, method


# =============================================================================
# Tests
# =============================================================================

def test_trade_seeds_import_leg(tmp_path):
    acc, trade, network = _build(tmp_path)
    f, sigma, method = _run(acc, trade, network, {
        ("Matte", "A", "CFR", "A", "CC"): 100.0,
        ("Anode", "B", "ER",  "A", "Mass"): 20.0,
    })
    imp = acc.eid("Anode", "B", "ER", "A", "CC")
    assert f[imp] == pytest.approx(_CONC["Anode"] * 20.0)  # conc·Mass
    assert method[imp] == "trade_seed"
    # σ matches the inverted conc twin: lp([conc·σ_m, |f_cc|·conc_sig/conc])
    f_cc = _CONC["Anode"] * 20.0
    expect = lp_combine([_CONC["Anode"] * 0.2, abs(f_cc) * 0.001 / _CONC["Anode"]], 1.3)
    assert sigma[imp] == pytest.approx(expect)


def test_chain_revives_after_trade_seed(tmp_path):
    acc, trade, network = _build(tmp_path)
    f, _sigma, method = _run(acc, trade, network, {
        ("Matte", "A", "CFR", "A", "CC"): 100.0,
        ("Anode", "B", "ER",  "A", "Mass"): 20.0,
    })
    cfr_anode = acc.eid("CFR", "A", "Anode", "A", "CC")
    dom_in    = acc.eid("Anode", "A", "ER", "A", "CC")
    er_cer    = acc.eid("ER", "A", "CER", "A", "CC")
    cer_cp    = acc.eid("CER", "A", "CP", "A", "CC")
    cp_cat    = acc.eid("CP", "A", "Cathode", "A", "CC")
    imp_cc    = _CONC["Anode"] * 20.0

    # CFR product from domestic Matte (imports into CFR = none)
    assert f[cfr_anode] == pytest.approx(_EFF["CFR"] * 100.0)
    assert method[cfr_anode] == "network_forward"
    # Anode_A flow-balance closes the domestic leg (export route uncovered → 0)
    assert f[dom_in] == pytest.approx(_EFF["CFR"] * 100.0)
    assert method[dom_in] == "network_balance"
    # ER product = eff·(domestic + seeded import) — the revival, not flat-prior
    assert f[er_cer] == pytest.approx(_EFF["ER"] * (_EFF["CFR"] * 100.0 + imp_cc))
    assert method[er_cer] == "network_forward"
    # propagation onward through CER flow-balance and CP recipe
    assert f[cer_cp] == pytest.approx(f[er_cer])
    assert method[cer_cp] == "network_balance"
    assert f[cp_cat] == pytest.approx(_EFF["CP"] * f[cer_cp])
    assert method[cp_cat] == "network_forward"
    # the uncovered export route is never written by the network
    assert np.isnan(f[acc.eid("Anode", "A", "ER", "B", "CC")])


def test_no_conc_row_is_noop(tmp_path):
    # Drop the Anode conc rows → no twin for the import leg → trade fill no-op.
    acc, trade, network = _build(tmp_path, with_anode_conc=False)
    f = np.full(acc.n_edges, np.nan)
    sigma = np.full(acc.n_edges, np.nan)
    m = acc.eid("Anode", "B", "ER", "A", "Mass")
    f[m] = 20.0
    sigma[m] = 0.2
    entries = run_imputer(trade, f, sigma, 0, acc)
    imp = acc.eid("Anode", "B", "ER", "A", "CC")
    assert entries == []
    assert np.isnan(f[imp])


def test_trade_and_network_scopes_partition_the_edges(tmp_path):
    # The declared scopes are the ownership partition (ADR 0010): the trade seed owns
    # cross-region Flow→Process CC edges, the network same-region ones; no edge is both.
    acc, trade, network = _build(tmp_path)
    assert not (trade.writer_scope(acc) & network.writer_scope(acc)).any()
    em = acc.edge_meta
    cross = (em["from_region"] != em["to_region"]).to_numpy()
    assert not network.writer_scope(acc)[em["edge_id"].to_numpy()[cross]].any()


def test_chain_runs_strict_and_every_write_is_in_scope(tmp_path):
    acc, trade, network = _build(tmp_path)
    f = np.full(acc.n_edges, np.nan)
    sigma = np.full(acc.n_edges, np.nan)
    for (fs, fr, ts, tr, v), val in {
        ("Matte", "A", "CFR", "A", "CC"): 100.0,
        ("Anode", "B", "ER",  "A", "Mass"): 20.0,
    }.items():
        e = acc.eid(fs, fr, ts, tr, v)
        f[e] = val
        sigma[e] = 0.01 * abs(val)
    seeded  = run_imputer(trade, f, sigma, 0, acc)        # strict: raises on a breach
    derived = run_imputer(network, f, sigma, 1, acc)
    assert {e["edge_id"] for e in seeded} <= set(np.flatnonzero(trade.writer_scope(acc)))
    assert {e["edge_id"] for e in derived} <= set(np.flatnonzero(network.writer_scope(acc)))


# =============================================================================
# The params relation source and the parametric σ rule
# =============================================================================

def _conc_csv(tmp_path, anode=0.94):
    conc = tmp_path / "conc.csv"
    write_conc(conc)
    df = pd.read_csv(conc, sep=";", dtype=str)
    df.loc[df["param_id"].str.startswith("conc_anode"), "value_nominal"] = str(anode)
    df.to_csv(conc, sep=";", index=False)
    return str(conc)


def test_a_source_relation_is_the_conc_twin_with_the_rows_sigma(tmp_path):
    acc = _ChainAcc()
    src = ParamSource.from_params(_conc_csv(tmp_path), acc, twin=True)
    imp = acc.eid("Anode", "B", "ER", "A", "CC")
    rel = next(r for r in src.relations if r.edges[0] == imp)
    assert rel.edges.tolist() == [imp, acc.eid("Anode", "B", "ER", "A", "Mass")]
    assert np.array_equal(rel.coefficients, [1.0, -0.94])
    assert rel.meta["sigma_param"] == 0.001 and rel.sigma == 0.001 / 0.94


def test_the_twin_reading_drops_a_zero_ratio_the_pair_reading_keeps(tmp_path):
    acc = _ChainAcc()
    csv = _conc_csv(tmp_path, anode=0.0)
    cc = acc.eid("Anode", "B", "ER", "A", "CC")
    twin = ParamSource.from_params(csv, acc, twin=True)
    pair = ParamSource.from_params(csv, acc)
    assert cc not in {int(r.edges[0]) for r in twin.relations}
    assert cc in {int(r.edges[0]) for r in pair.relations}


def test_a_both_fill_writes_either_side_and_a_forward_one_only_the_cc(tmp_path):
    acc = _ChainAcc()
    src = ParamSource.from_params(_conc_csv(tmp_path), acc)
    m, cc = (acc.eid("Anode", "B", "ER", "A", v) for v in ("Mass", "CC"))
    for rule, seeds, target in (("both", {cc: 18.8}, m), ("forward", {m: 20.0}, cc)):
        f = np.full(acc.n_edges, np.nan)
        sigma = np.full(acc.n_edges, np.nan)
        for e, val in seeds.items():
            f[e], sigma[e] = val, 0.01 * val
        imp = FillImputer(Fill(src, rule, id="conc"), "parametric")
        (entry,) = [e for e in run_imputer(imp, f, sigma, 0, acc)
                    if e["edge_id"] == target]
        assert entry["method"] == "parametric" and entry["param_id"].startswith("conc_")
        assert f[target] == pytest.approx(20.0 if rule == "both" else 18.8)
    f = np.full(acc.n_edges, np.nan)
    sigma = np.full(acc.n_edges, np.nan)
    f[cc], sigma[cc] = 18.8, 0.1
    assert run_imputer(FillImputer(Fill(src, "forward", id="conc")), f, sigma, 0, acc) == []


def test_the_trade_sigma_rule_is_forward_only(tmp_path):
    acc = _ChainAcc()
    src = ParamSource.from_params(_conc_csv(tmp_path), acc, twin=True)
    with pytest.raises(ValueError, match="forward"):
        FillImputer(Fill(src, "both"), "trade")
