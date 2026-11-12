"""Tests of the case-parametrised Stage 2 algorithms in ``pysuto.impute`` (#144) on a toy
accounting structure that shares no label with copper: valuations ``Gr``/``Ct``, sectors
``Widget``/``Gadget``/``Mill``/``Shop``/``Zed``."""
import numpy as np
import pandas as pd

from pysuto.acc import ValuationRoles
from pysuto.impute import (
    damp_leg_sigma,
    screen_contradicted_nodes,
    screen_outlier_trade_legs,
    time_interp_scope,
)

# nodes: (sector, entity, region)
_NODES = [
    ("Widget", "Flow",    "North"),   # 0 exporter flow
    ("Widget", "Flow",    "South"),   # 1 exporter flow
    ("Gadget", "Flow",    "North"),   # 2 exporter flow
    ("Shop",   "Process", "East"),    # 3 importer
    ("Mill",   "Process", "North"),   # 4 producer
    ("Mill",   "Process", "South"),   # 5 producer
]
# (from, to), each in Gr + Ct
_EDGES = [(4, 0), (5, 1), (0, 3), (1, 3), (2, 3)]


class _Acc:
    def __init__(self):
        self.node_index = pd.DataFrame(_NODES, columns=["sector", "entity", "region"])
        self.valuation_roles = ValuationRoles(content="Ct", gross="Gr")
        rows, eid = [], 0
        for f, t in _EDGES:
            for val in ("Gr", "Ct"):
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


def _set(agg, eid, f):
    agg.f_agg[eid] = f
    agg.coverage[eid] = True


def test_time_interp_scope_takes_the_stock_labels_as_given():
    ei = pd.DataFrame({"region_from": ["A", "A", "A"],
                       "region_to":   ["A", "B", "A"],
                       "sector_to":   ["Use", "Use", "Tank"]})
    scope = time_interp_scope(ei, True, ("Tank", "Vault"), _Log())
    np.testing.assert_array_equal(scope, [True, False, False])
    scope = time_interp_scope(ei, False, ("Tank",), _Log())
    np.testing.assert_array_equal(scope, [True, True, False])
    assert time_interp_scope(ei, False, (), _Log()) is None


def test_damping_pins_a_ref_by_the_cases_sector_label():
    acc, agg = _Acc(), _Agg(10)
    for eid, f in ((4, 100.0), (6, 400.0)):          # Widget bilateral Gr legs
        _set(agg, eid, f)
        agg.sigma_agg[eid] = 0.2 * f
    _set(agg, 8, 50.0)                                # Gadget, its own median
    agg.sigma_agg[8] = 0.2 * 50.0
    damp_leg_sigma(acc, agg, 0.5, 1.0, _Log(), a_ref_override={"Widget": 100.0})
    # a_ref pinned at 100: cv = 0.2·(100/400)^0.5 = 0.1 on the 400 leg
    assert np.isclose(agg.sigma_agg[6] / 400.0, 0.1)
    assert np.isclose(agg.sigma_agg[4], 0.2 * 100.0)
    assert np.isclose(agg.sigma_agg[8], 0.2 * 50.0)
    assert agg.sigma_agg[5] == 1.0                    # Ct twin untouched


def test_outlier_screen_reads_labels_and_roles_from_args(tmp_path):
    acc, agg = _Acc(), _Agg(10)
    csv = tmp_path / "params.csv"
    pd.DataFrame([{"param_id": "concwidget", "param_type": "concentration",
                   "a_sector_from": "Widget", "a_valuation": "Gr",
                   "value_nominal": 0.5}]).to_csv(csv, index=False)
    _set(agg, 1, 100.0)          # North production, Ct
    _set(agg, 3, 100.0)          # South production, Ct
    _set(agg, 4, 1000.0)         # North exports 500 Ct = 5x -> flagged
    _set(agg, 6, 150.0)          # South exports 75 Ct = 0.75x -> kept
    report = screen_outlier_trade_legs(acc, agg, str(csv), _Log(), k=3.0, cv=2.0)
    assert list(report["region_from"]) == ["North"]
    assert np.isclose(agg.sigma_agg[4], 2.0 * 1000.0)
    assert agg.sigma_agg[6] == 1.0


def test_contradicted_node_screen_judges_the_given_sector():
    acc = _Acc()
    acc.node_index = pd.DataFrame({"sector": ["Mill", "Zed", "Shop"],
                                   "entity": ["Process", "Flow", "Process"],
                                   "region": ["A", "A", "A"]})
    acc.edge_index = pd.DataFrame(
        {"edge_id": [0, 1], "from_id": [0, 1], "to_id": [1, 2],
         "time": ["2020", "2020"], "valuation": ["Ct", "Ct"]})
    agg = _Agg(2)
    agg.f_agg[:] = [100.0, 20.0]
    agg.coverage[:] = True
    report = screen_contradicted_nodes(acc, agg, _Log(), sectors=("Zed",), cv=1.0)
    assert list(report["sector"]) == ["Zed"]
    assert np.isclose(agg.sigma_agg[1], 20.0) and agg.sigma_agg[0] == 1.0
