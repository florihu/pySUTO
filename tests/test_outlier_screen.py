"""Copper-case tests of the Stage 2 σ screens in ``pysuto.cases.copper`` (#121): the
bilateral-leg plausibility screen, size-damped leg σ and the contradicted-node screen.
The toy accounting structures use copper's sector labels, which the functions hold."""
import numpy as np
import pandas as pd

from pysuto.acc import ValuationRoles
from pysuto.cases.copper import (
    CASE,
    damp_leg_sigma as _damp_leg_sigma,
    screen_contradicted_nodes,
    screen_outlier_trade_legs as _screen,
)

# The copper case's screen knobs (the Stage 2 driver passes them from the case spec).
OUTLIER_LEG_K  = CASE.imp.OUTLIER_LEG_K
OUTLIER_LEG_CV = CASE.imp.OUTLIER_LEG_CV


def _screen_outlier_trade_legs(acc, agg, conc_csv, logger, grade_csv=None):
    return _screen(acc, agg, conc_csv, logger, grade_csv, k=OUTLIER_LEG_K,
                   cv=OUTLIER_LEG_CV)

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
        self.valuation_roles = ValuationRoles(content="CC", gross="Mass")
        self.boundary_roles      = {}
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
        self.valuation_roles = ValuationRoles(content="CC", gross="Mass")
        self.boundary_roles      = {}
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


class _DampAcc:
    """Two sectors, three bilateral legs each (small/median/big) + extras."""

    def __init__(self):
        self.valuation_roles = ValuationRoles(content="CC", gross="Mass")
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


# ---- ADR 0029 plumbing: the screen prices a leg at its exporter's grade ----

def _conc_csv_by_region(tmp_path, rows):
    """rows: (region, nominal); region "*" is the wildcard."""
    p = tmp_path / "params_region.csv"
    pd.DataFrame([{
        "param_id": f"cg{r.lower().replace('*', 'star')}",
        "param_type": "concentration", "a_sector_from": "Concentrate",
        "a_region_from": r, "a_valuation": "Mass", "value_nominal": v,
    } for r, v in rows]).to_csv(p, index=False)
    return str(p)


def test_screen_uses_each_exporters_own_grade(tmp_path):
    """Same production and same export Mass; only the grade differs.

    With one scalar per sector both exporters were judged alike. Priced at
    their own grades, only the high-grade one crosses OUTLIER_LEG_K.
    """
    acc, agg = _Acc(), _Agg(10)
    m = acc.edge_index

    def set_edge(i, val, f):
        eid = int(m[(m.index // 2 == i) & (m["valuation"] == val)]["edge_id"].iloc[0])
        agg.f_agg[eid] = f
        agg.coverage[eid] = True
        return eid

    set_edge(0, "CC", 100.0)                 # PNG production 100 CC
    set_edge(1, "CC", 100.0)                 # Chile production 100 CC
    png = set_edge(2, "Mass", 1000.0)
    chl = set_edge(3, "Mass", 1000.0)

    csv = _conc_csv_by_region(tmp_path, [("*", 0.275), ("PNG", 0.40), ("Chile", 0.20)])
    report = _screen_outlier_trade_legs(acc, agg, csv, _Log())
    # PNG: 1000 × 0.40 = 400 CC = 4.0× production → flagged
    # Chile: 1000 × 0.20 = 200 CC = 2.0× production → kept
    assert list(report["region_from"]) == ["PNG"]
    assert np.isclose(agg.sigma_agg[png], OUTLIER_LEG_CV * 1000.0)
    assert agg.sigma_agg[chl] == 1.0


# ---------------------------------------------------------------------------
# Contradicted-node screen (ADR 0019)
# ---------------------------------------------------------------------------

class _MatteAcc:
    """Smelting(Process) → Matte(Flow) → CFR(Process) in one region, CC, 2020.

    Edges: 0 Smelting→Matte (inflow), 1 Matte→CFR (outflow)."""

    def __init__(self):
        self.valuation_roles = ValuationRoles(content="CC", gross="Mass")
        self.node_index = pd.DataFrame(
            {"sector": ["Smelting", "Matte", "CFR"],
             "entity": ["Process", "Flow", "Process"],
             "region": ["A", "A", "A"]}, index=[0, 1, 2])
        self.edge_index = pd.DataFrame(
            {"edge_id": [0, 1], "from_id": [0, 1], "to_id": [1, 2],
             "time": ["2020", "2020"], "valuation": ["CC", "CC"]})
        self.n_edges = 2


def _matte_agg(outflow):
    agg = _Agg(2)
    agg.f_agg[:] = [100.0, outflow]
    agg.coverage[:] = True
    return agg


def test_contradicted_node_screen_inflates_the_outflow_when_asked():
    """Outflow prior 20 < 0.5 × inflow prior 100: the node is reported and, with a cv,
    its outflow edge's σ becomes cv·|f| (the inflow edge keeps its σ)."""
    agg = _matte_agg(20.0)
    report = screen_contradicted_nodes(_MatteAcc(), agg, _Log(), sectors=("Matte",),
                                       cv=1.0)
    assert list(report["sector"]) == ["Matte"]
    assert np.isclose(report["ratio"].iloc[0], 0.2)
    assert np.isclose(agg.sigma_agg[1], 20.0) and agg.sigma_agg[0] == 1.0


def test_contradicted_node_screen_without_a_cv_only_detects():
    agg = _matte_agg(20.0)
    report = screen_contradicted_nodes(_MatteAcc(), agg, _Log(), sectors=("Matte",),
                                       cv=None)
    assert len(report) == 1
    assert (agg.sigma_agg == 1.0).all()


def test_contradicted_node_screen_leaves_a_consistent_node_alone():
    agg = _matte_agg(80.0)
    report = screen_contradicted_nodes(_MatteAcc(), agg, _Log(), sectors=("Matte",),
                                       cv=1.0)
    assert len(report) == 0 and (agg.sigma_agg == 1.0).all()


def test_contradicted_node_screen_judges_only_the_given_sectors():
    agg = _matte_agg(20.0)
    report = screen_contradicted_nodes(_MatteAcc(), agg, _Log(), sectors=("Anode",),
                                       cv=1.0)
    assert len(report) == 0 and (agg.sigma_agg == 1.0).all()


# ---------------------------------------------------------------------------
# The case's hooks wire the screens (#121)
# ---------------------------------------------------------------------------

def _ctx(tmp_path, acc, agg):
    from pathlib import Path
    from types import SimpleNamespace
    _conc_csv(tmp_path)                                  # tmp_path/params.csv
    inputs = SimpleNamespace(params=SimpleNamespace(resolve=lambda _id: Path(tmp_path)))
    return SimpleNamespace(case=CASE, acc=acc, agg_result=agg, inputs=inputs,
                           logger=_Log(), reports={})


def test_pre_impute_damps_then_screens_and_files_the_report(tmp_path):
    from pysuto.cases.copper import pre_impute
    acc, agg = _Acc(), _Agg(10)
    m = acc.edge_index
    for i, val, fv in ((0, "CC", 100.0), (2, "Mass", 2000.0)):
        eid = int(m[(m.index // 2 == i) & (m["valuation"] == val)]["edge_id"].iloc[0])
        agg.f_agg[eid], agg.coverage[eid], agg.sigma_agg[eid] = fv, True, 0.21 * fv
    png = int(m[(m.index // 2 == 2) & (m["valuation"] == "Mass")]["edge_id"].iloc[0])
    ctx = _ctx(tmp_path, acc, agg)

    pre_impute(ctx)

    assert list(ctx.reports) == ["outlier_leg_screen"]
    assert list(ctx.reports["outlier_leg_screen"]["region_from"]) == ["PNG"]
    assert np.isclose(agg.sigma_agg[png], OUTLIER_LEG_CV * 2000.0)   # screen wins


def test_post_impute_files_the_contradicted_node_report(tmp_path):
    from pysuto.cases.copper import post_impute
    ctx = _ctx(tmp_path, _MatteAcc(), _matte_agg(20.0))

    post_impute(ctx)

    assert list(ctx.reports) == ["contradicted_node_screen"]
    assert len(ctx.reports["contradicted_node_screen"]) == 1
    assert (ctx.agg_result.sigma_agg == 1.0).all()       # copper: detector only
