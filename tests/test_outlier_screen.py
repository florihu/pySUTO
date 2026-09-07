"""Unit tests for the bilateral-leg plausibility screen (Stage 2, σ inflation)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from collect_aggregate_impute import (  # noqa: E402
    OUTLIER_LEG_CV,
    _screen_outlier_trade_legs,
)

# nodes: (sector, entity, region)
_NODES = [
    ("Concentrate", "Flow", "PNG"),      # 0 exporter flow — implausible
    ("Concentrate", "Flow", "Chile"),    # 1 exporter flow — plausible
    ("Concentrate", "Flow", "Nowhere"),  # 2 exporter flow — no covered production
    ("Sm", "Process", "Philippines"),    # 3 importer process
    ("MF", "Process", "PNG"),            # 4 producer
    ("MF", "Process", "Chile"),          # 5 producer
]
# edges: (from, to) node ids; each in Mass+CC
_EDGES = [
    (4, 0),  # MF→Concentrate PNG (production)
    (5, 1),  # MF→Concentrate Chile (production)
    (0, 3),  # PNG→Philippines bilateral
    (1, 3),  # Chile→Philippines bilateral
    (2, 3),  # Nowhere→Philippines bilateral (no production reference)
]


class _Acc:
    def __init__(self):
        self.node_index = pd.DataFrame(_NODES, columns=["sector", "entity", "region"])
        rows, eid = [], 0
        for f, t in _EDGES:
            for val in ("Mass", "CC"):
                rows.append((eid, f, t, "2020", val))
                eid += 1
        self.edge_index = pd.DataFrame(
            rows, columns=["edge_id", "from_id", "to_id", "time", "valuation"])
        self.n_edges = len(rows)


class _Agg:
    def __init__(self, n):
        self.f_agg = np.zeros(n)
        self.sigma_agg = np.full(n, 1.0)
        self.coverage = np.zeros(n, dtype=bool)


class _Log:
    def warning(self, *a, **k): pass
    def info(self, *a, **k): pass


def _conc_csv(tmp_path):
    p = tmp_path / "params.csv"
    pd.DataFrame([{
        "param_id": "concconcentratesupply", "param_type": "concentration",
        "a_sector_from": "Concentrate", "a_valuation": "Mass",
        "value_nominal": 0.275,
    }]).to_csv(p, index=False)
    return str(p)


def test_screen_inflates_only_implausible_exporter(tmp_path):
    acc, agg = _Acc(), _Agg(10)

    def set_edge(i, val, f, covered=True):
        m = acc.edge_index
        eid = int(m[(m.index // 2 == i) & (m["valuation"] == val)]["edge_id"].iloc[0])
        agg.f_agg[eid] = f
        agg.coverage[eid] = covered
        return eid

    set_edge(0, "CC", 100.0)                    # PNG production 100 CC
    set_edge(1, "CC", 100.0)                    # Chile production 100 CC
    png = set_edge(2, "Mass", 2000.0)           # implied 550 CC = 5.5× → flag
    chl = set_edge(3, "Mass", 400.0)            # implied 110 CC = 1.1× → keep
    now = set_edge(4, "Mass", 9999.0)           # no production → skipped

    report = _screen_outlier_trade_legs(acc, agg, _conc_csv(tmp_path), _Log())
    assert list(report["region_from"]) == ["PNG"]
    assert np.isclose(agg.sigma_agg[png], OUTLIER_LEG_CV * 2000.0)
    assert agg.sigma_agg[chl] == 1.0
    assert agg.sigma_agg[now] == 1.0


def test_screen_clean_returns_empty_and_keeps_zero_sigma_guard(tmp_path):
    acc, agg = _Acc(), _Agg(10)
    # all-plausible world: production 100, exports 200 Mass (55 CC)
    m = acc.edge_index
    for i, val, f in ((0, "CC", 100.0), (2, "Mass", 200.0)):
        eid = int(m[(m.index // 2 == i) & (m["valuation"] == val)]["edge_id"].iloc[0])
        agg.f_agg[eid] = f
        agg.coverage[eid] = True
    report = _screen_outlier_trade_legs(acc, agg, _conc_csv(tmp_path), _Log())
    assert len(report) == 0
    assert (agg.sigma_agg == 1.0).all()


# nodes for the import-adjusted reference (Round C): a re-exporter fed by
# imports, and an exporter with imports but no production
_REEXP_NODES = [
    ("Concentrate", "Flow", "Peru"),     # 0 upstream exporter (skipped: no ref)
    ("Concentrate", "Flow", "Japan"),    # 1 re-exporter — imports cover exports
    ("Concentrate", "Flow", "Hubless"),  # 2 exporter — imports+prod ≪ exports
    ("Sm", "Process", "Japan"),          # 3 import target
    ("Sm", "Process", "Philippines"),    # 4 export target
    ("MF", "Process", "Japan"),          # 5 producer
    ("Sm", "Process", "Hubless"),        # 6 import target
    ("MF", "Process", "Hubless"),        # 7 producer
]
_REEXP_EDGES = [
    (5, 1),  # MF→Concentrate Japan (production)
    (0, 3),  # Peru→Japan bilateral (Japan import)
    (1, 4),  # Japan→Philippines bilateral (Japan export)
    (0, 6),  # Peru→Hubless bilateral (Hubless import)
    (2, 4),  # Hubless→Philippines bilateral (Hubless export)
    (7, 2),  # MF→Concentrate Hubless (production)
]


class _ReexpAcc(_Acc):
    def __init__(self):
        self.node_index = pd.DataFrame(
            _REEXP_NODES, columns=["sector", "entity", "region"])
        rows, eid = [], 0
        for f, t in _REEXP_EDGES:
            for val in ("Mass", "CC"):
                rows.append((eid, f, t, "2020", val))
                eid += 1
        self.edge_index = pd.DataFrame(
            rows, columns=["edge_id", "from_id", "to_id", "time", "valuation"])
        self.n_edges = len(rows)


def test_screen_reference_includes_covered_imports(tmp_path):
    acc, agg = _ReexpAcc(), _Agg(12)

    def set_edge(i, val, f):
        m = acc.edge_index
        eid = int(m[(m.index // 2 == i) & (m["valuation"] == val)]["edge_id"].iloc[0])
        agg.f_agg[eid] = f
        agg.coverage[eid] = True
        return eid

    set_edge(0, "CC", 10.0)                # Japan production 10 CC
    set_edge(1, "Mass", 400.0)             # Japan imports 110 CC
    jpn = set_edge(2, "Mass", 400.0)       # Japan exports 110 CC:
    #   old ratio 110/10 = 11 → false positive; new 110/120 ≈ 0.9 → keep
    set_edge(3, "Mass", 40.0)              # Hubless imports 11 CC
    hub = set_edge(4, "Mass", 400.0)       # Hubless exports 110 CC
    set_edge(5, "CC", 1.0)                 # Hubless production 1 CC
    #   → ratio 110/12 ≈ 9.2 against production+imports → flag

    report = _screen_outlier_trade_legs(acc, agg, _conc_csv(tmp_path), _Log())
    assert list(report["region_from"]) == ["Hubless"]
    assert np.isclose(report["ratio"].iloc[0], 110.0 / 12.0)
    assert np.isclose(agg.sigma_agg[hub], OUTLIER_LEG_CV * 400.0)
    assert agg.sigma_agg[jpn] == 1.0


# ---------------------------------------------------------------------------
# Size-damped leg σ (ADR 0017)
# ---------------------------------------------------------------------------

from collect_aggregate_impute import _damp_leg_sigma  # noqa: E402


class _DampAcc:
    """Two sectors, three bilateral legs each (small/median/big) + extras."""

    def __init__(self):
        nodes = [
            ("Concentrate", "Flow", "A"),   # 0
            ("Concentrate", "Flow", "B"),   # 1
            ("Cathode", "Flow", "A"),       # 2
            ("Sm", "Process", "C"),         # 3
            ("Sm", "Process", "A"),         # 4  (domestic target for node 0)
            ("ER", "Process", "C"),         # 5
        ]
        self.node_index = pd.DataFrame(nodes, columns=["sector", "entity", "region"])
        edges = [
            (0, 3),  # Concentrate A->C  (bilateral)
            (1, 3),  # Concentrate B->C  (bilateral)
            (0, 4),  # Concentrate A->A  (domestic — untouched)
            (2, 5),  # Cathode A->C      (bilateral)
        ]
        rows, eid = [], 0
        for f, t in edges:
            for val in ("Mass", "CC"):
                rows.append((eid, f, t, "2020", val))
                eid += 1
        # one more Concentrate bilateral leg in another year (median pooling)
        rows.append((eid, 1, 3, "2021", "Mass")); eid += 1
        self.edge_index = pd.DataFrame(
            rows, columns=["edge_id", "from_id", "to_id", "time", "valuation"])
        self.n_edges = eid


def test_damping_anchors_at_sector_median_and_caps():
    acc = _DampAcc()
    agg = _Agg(acc.n_edges)
    # Concentrate bilateral Mass legs: 1 (small), 100 (median), 10000 (big)
    vals = {0: 1.0, 2: 100.0, 8: 10000.0}
    for eid, v in vals.items():
        agg.f_agg[eid] = v
        agg.coverage[eid] = True
        agg.sigma_agg[eid] = 0.21 * v          # pedigree cv 0.21
    # Cathode bilateral leg at 50 (its own sector median of one)
    agg.f_agg[6] = 50.0; agg.coverage[6] = True; agg.sigma_agg[6] = 0.21 * 50.0
    # domestic Concentrate leg — must stay untouched
    agg.f_agg[4] = 77.0; agg.coverage[4] = True; agg.sigma_agg[4] = 0.21 * 77.0

    _damp_leg_sigma(acc, agg, gamma=0.5, cv_cap=1.0, logger=_Log())

    # median leg (a_ref=100) keeps its σ exactly
    assert np.isclose(agg.sigma_agg[2], 0.21 * 100.0)
    # big leg: cv = 0.21·(100/10000)^0.5 = 0.021
    assert np.isclose(agg.sigma_agg[8] / 10000.0, 0.021)
    # small leg: cv = 0.21·(100/1)^0.5 = 2.1 → capped at 1.0
    assert np.isclose(agg.sigma_agg[0] / 1.0, 1.0)
    # Cathode anchored at its own median → unchanged, not at Concentrate's
    assert np.isclose(agg.sigma_agg[6], 0.21 * 50.0)
    # domestic leg untouched
    assert np.isclose(agg.sigma_agg[4], 0.21 * 77.0)
    # CC twins untouched at this stage (trade_seed inherits later)
    assert agg.sigma_agg[1] == 1.0 and agg.sigma_agg[3] == 1.0


def test_damping_disabled_and_screen_overrides(tmp_path):
    acc = _DampAcc()
    agg = _Agg(acc.n_edges)
    agg.f_agg[0] = 10.0; agg.coverage[0] = True; agg.sigma_agg[0] = 2.1
    _damp_leg_sigma(acc, agg, gamma=1.0, cv_cap=1.0, logger=_Log())
    assert np.isclose(agg.sigma_agg[0], 2.1)          # off at gamma=1
    _damp_leg_sigma(acc, agg, gamma=None, cv_cap=1.0, logger=_Log())
    assert np.isclose(agg.sigma_agg[0], 2.1)

    # damping then screen: flagged exporter's leg ends at the screen's cv 5
    acc2, agg2 = _Acc(), _Agg(10)
    m = acc2.edge_index
    for i, val, fv in ((0, "CC", 100.0), (2, "Mass", 2000.0)):
        eid = int(m[(m.index // 2 == i) & (m["valuation"] == val)]["edge_id"].iloc[0])
        agg2.f_agg[eid] = fv
        agg2.coverage[eid] = True
        agg2.sigma_agg[eid] = 0.21 * fv
    png = int(m[(m.index // 2 == 2) & (m["valuation"] == "Mass")]["edge_id"].iloc[0])
    _damp_leg_sigma(acc2, agg2, gamma=0.5, cv_cap=1.0, logger=_Log())
    _screen_outlier_trade_legs(acc2, agg2, _conc_csv(tmp_path), _Log())
    assert np.isclose(agg2.sigma_agg[png], 5.0 * 2000.0)
