"""
test_trade_imputer.py
=====================
Unit suite for pysuto.impute.trade.TradeImputer + the trade→network handoff
that revives the deep refining chain (ADR 0010).

A minimal **two-region** copper chain ``Matte→CFR→Anode→ER→CER→CP→Cathode``
(region A), with a cross-region anode **import** ``Anode(B)→ER(A)`` and a
cross-region anode **export** ``Anode(A)→ER(B)``, exercises:

  * TradeImputer seeds the import CC leg from its BACI Mass twin (``trade_seed``);
  * with the import seeded, the NetworkImputer recipe gate opens, the Anode
    flow-balance closes the lone domestic ``Anode(A)→ER(A)`` leg (the export
    route is uncovered → treated as 0), and ``ER→CER→CP`` fill via the recipe /
    flow-balance instead of falling to flat-prior;
  * the trade-seed σ matches the inverted conc twin;
  * no conc row ⇒ TradeImputer is a no-op.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.impute import NetworkImputer, TradeImputer, derive_process_recipes
from pysuto.lp import lp_combine


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
    trade = TradeImputer(str(conc), p=1.3, cv_cap=float("inf"))
    network = NetworkImputer(str(eff), str(conc), p=1.3, cv_cap=float("inf"))
    trade.prepare(acc)
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
    entries = trade.impute(f, sigma, np.isnan(f), 0, acc)       # trade first
    entries += network.impute(f, sigma, np.isnan(f), 1, acc)    # then network (internal fixpoint)
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
    # Drop the Anode conc rows → no twin for the import leg → TradeImputer no-op.
    acc, trade, network = _build(tmp_path, with_anode_conc=False)
    f = np.full(acc.n_edges, np.nan)
    sigma = np.full(acc.n_edges, np.nan)
    m = acc.eid("Anode", "B", "ER", "A", "Mass")
    f[m] = 20.0
    sigma[m] = 0.2
    entries = trade.impute(f, sigma, np.isnan(f), 0, acc)
    imp = acc.eid("Anode", "B", "ER", "A", "CC")
    assert entries == []
    assert np.isnan(f[imp])


def test_ownership_partition_holds_on_chain(tmp_path):
    from pysuto.diag import ownership_violations
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
    entries  = trade.impute(f, sigma, np.isnan(f), 0, acc)
    entries += network.impute(f, sigma, np.isnan(f), 1, acc)
    log = pd.DataFrame(entries)
    assert ownership_violations(log, acc.edge_meta).empty


def test_ownership_violations_flags_breaches():
    from pysuto.diag import ownership_violations
    edge_meta = pd.DataFrame({
        "edge_id":     [0, 1, 2, 3],
        "from_region": ["A", "A", "A", "B"],
        "to_region":   ["A", "B", "A", "A"],
    })
    log = pd.DataFrame({
        "edge_id": [0, 1, 2, 3],
        "method":  ["network_forward",  # 0 same-region net → ok
                    "network_balance",  # 1 cross-region net → VIOLATION
                    "trade_seed",       # 2 same-region trade → VIOLATION
                    "trade_seed"],      # 3 cross-region trade → ok
    })
    bad = ownership_violations(log, edge_meta)
    assert set(bad["edge_id"]) == {1, 2}
