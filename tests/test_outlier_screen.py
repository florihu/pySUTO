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
