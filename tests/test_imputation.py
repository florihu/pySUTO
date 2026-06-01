"""
test_imputation.py
==================
pytest suite for pysuto/impute.py under the structural model.

Covers:
    ImputationLog           — basic operations, as_dataframe, for_edge
    ImputationEngine        — pipeline ordering, early stop, log attachment
    ParametricImputer       — CSV bidirectional cell pairs, lambda directed fills
    FormulaImputer          — balance equations, region binding, clipping
    TimeImputer             — linear, locf, nocb, gap inflation, window
    FlatPriorImputer        — flat-prior cascade, density-aware CV

Test shape conventions:
    f, sigma, missing all have shape (n_edges, n_timesteps) = (6, 3) under
    the MockAcc fixture, where each row is one structural edge series across
    time_range ["2020", "2021", "2022"].
"""
from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd
import pytest

from pysuto.agg import AggregationResult
from pysuto.impute import (
    ImputationLog,
    ImputationEngine,
    BaseImputer,
    ParametricRelation,
    ParametricImputer,
    ParamRef,
    SourceTerm,
    FormulaRelation,
    FormulaImputer,
    TimeImputer,
    FlatPriorImputer,
    _reg_sensitivity,
)


# =============================================================================
# Regularized sensitivity s(α)
# =============================================================================

class TestRegSensitivity:
    def test_saturates_to_one_for_large_alpha(self):
        assert _reg_sensitivity(100.0, q=3.0, alpha_min=0.05) == pytest.approx(1.0, rel=1e-6)

    def test_linear_damping_for_small_alpha(self):
        a, am, q = 1e-4, 0.05, 3.0
        assert _reg_sensitivity(a, q, am) == pytest.approx(a / am, rel=1e-3)

    def test_negative_uses_abs(self):
        assert _reg_sensitivity(-2.0, q=3.0, alpha_min=0.05) == pytest.approx(
            _reg_sensitivity(2.0, q=3.0, alpha_min=0.05)
        )

    def test_at_alpha_equals_alpha_min(self):
        for q in (2.0, 3.0, 4.0):
            assert _reg_sensitivity(0.05, q=q, alpha_min=0.05) == pytest.approx(
                1.0 / 2 ** (1.0 / q)
            )

    def test_zero_alpha_returns_zero(self):
        assert _reg_sensitivity(0.0, q=3.0, alpha_min=0.05) == 0.0


# =============================================================================
# Fixtures — minimal mock accounting structure
# =============================================================================

class MockAcc:
    """
    Small structural accounting structure.

    Nodes (node_index):
        0  Mining / Process / A
        1  Ore    / Flow    / A
        2  Mining / Process / B
        3  Ore    / Flow    / B

    Structural edges (no time column):
        0  0→1  Mass   (supply: Mining_A → Ore_A, Mass)
        1  2→3  Mass   (supply: Mining_B → Ore_B, Mass)
        2  0→1  CC     (same pair, CC valuation)
        3  2→3  CC     (same pair, CC valuation)
        4  1→0  Mass   (use: Ore_A → Mining_A, Mass)
        5  3→2  Mass   (use: Ore_B → Mining_B, Mass)

    Time axis: time_range = ["2020", "2021", "2022"] → n_timeps = 3.
    All per-edge value arrays are (n_edges, n_timeps) = (6, 3).
    """

    class _Bounds:
        def __init__(self, n: int) -> None:
            self.l = np.zeros(n, dtype=float)
            self.u = np.full(n, 1e9, dtype=float)

    def __init__(self):
        self.node_index = pd.DataFrame(
            {
                "sector": ["Mining", "Ore",  "Mining", "Ore"],
                "entity": ["Process","Flow", "Process","Flow"],
                "region": ["A",      "A",    "B",      "B"],
            },
            index=[0, 1, 2, 3],
        )
        self.edge_index = pd.DataFrame(
            {
                "edge_id":   [0, 1, 2, 3, 4, 5],
                "from_id":   [0, 2, 0, 2, 1, 3],
                "to_id":     [1, 3, 1, 3, 0, 2],
                "valuation": ["Mass","Mass","CC", "CC", "Mass","Mass"],
            }
        )
        self.time_range = ["2020", "2021", "2022"]
        self.n_timeps   = 3
        self.bounds = MockAcc._Bounds(6)
        self.n_edges = 6
        self.masks = {
            "supply":   np.array([True, True, True, True, False, False], dtype=bool),
            "use":      np.array([False, False, False, False, True, True], dtype=bool),
            "tradable": np.zeros(6, dtype=bool),
            "sb":       np.zeros(6, dtype=bool),
        }


@pytest.fixture
def acc():
    return MockAcc()


# Helpers for building (n_edges, n_timeps) test arrays compactly.

_N_EDGES = 6
_N_TIMEPS = 3


def _nan2d():
    return np.full((_N_EDGES, _N_TIMEPS), np.nan, dtype=float)


def _agg_result(a_agg, sigma_agg=None, feed_ids=("test",), F=None, Sigma=None):
    """
    Legacy helper preserved for cross-test reuse (e.g. test_imp_diag.py).

    Accepts either 1-D arrays of length ``n_edges`` (placed at ``t_idx=0``,
    other columns left as ``NaN``) or 2-D arrays of shape
    ``(n_edges, n_timesteps)``. The 1-D path lets older diag tests
    construct an AggregationResult without rewriting every line.
    """
    a_agg_arr = np.asarray(a_agg, dtype=float)
    if a_agg_arr.ndim == 1:
        a2 = np.full((_N_EDGES, _N_TIMEPS), np.nan, dtype=float)
        n = min(a_agg_arr.shape[0], _N_EDGES)
        a2[:n, 0] = a_agg_arr[:n]
        a_agg_arr = a2
    if sigma_agg is None:
        sigma_arr = np.where(np.isfinite(a_agg_arr), 1.0, np.nan)
    else:
        sigma_arr = np.asarray(sigma_agg, dtype=float)
        if sigma_arr.ndim == 1:
            s2 = np.full((_N_EDGES, _N_TIMEPS), np.nan, dtype=float)
            n = min(sigma_arr.shape[0], _N_EDGES)
            s2[:n, 0] = sigma_arr[:n]
            sigma_arr = s2
    return _agg_result_2d(a_agg_arr, sigma_arr, feed_ids=feed_ids, F=F, Sigma=Sigma)


def _agg_result_2d(a_agg, sigma_agg=None, feed_ids=("test",), F=None, Sigma=None):
    """Wrap pre-built 2-D arrays in an AggregationResult for engine tests."""
    a_agg = np.asarray(a_agg, dtype=float)
    if sigma_agg is None:
        sigma_agg = np.where(np.isfinite(a_agg), 1.0, np.nan)
    sigma_agg = np.asarray(sigma_agg, dtype=float)
    if F is None:
        F = a_agg[np.newaxis, :, :]
    if Sigma is None:
        Sigma = sigma_agg[np.newaxis, :, :]
    return AggregationResult(
        a_agg     = a_agg.copy(),
        sigma_agg = sigma_agg.copy(),
        Omega     = np.where(np.isfinite(F), 1.0, 0.0),
        F         = F,
        Sigma     = Sigma,
        feed_ids  = list(feed_ids),
        coverage  = np.isfinite(a_agg),
    )


# =============================================================================
# ImputationLog
# =============================================================================

class TestImputationLog:

    def _entry(self, **overrides) -> dict:
        e = {
            "edge_id":      0,
            "t_idx":        0,
            "step":         0,
            "method":       "test",
            "source_edges": [],
            "coefficients": [],
            "f_before":     np.nan,
            "f_after":      5.0,
            "sigma_before": np.nan,
            "sigma_after":  0.5,
            "description":  "x",
            "param_id":     None,
        }
        e.update(overrides)
        return e

    def test_empty_log(self):
        log = ImputationLog()
        assert len(log) == 0
        assert log.as_dataframe().empty

    def test_append_and_len(self):
        log = ImputationLog()
        log.append(self._entry())
        assert len(log) == 1

    def test_as_dataframe_columns(self):
        log = ImputationLog()
        log.append(self._entry(method="parametric", source_edges=[1],
                               coefficients=[2.0], f_after=10.0,
                               sigma_after=1.0, description="d",
                               param_id="p1"))
        df = log.as_dataframe()
        assert set(df.columns) == {
            "edge_id","t_idx","step","method","source_edges","coefficients",
            "f_before","f_after","sigma_before","sigma_after",
            "description","param_id",
        }

    def test_for_edge_filters_correctly(self):
        log = ImputationLog()
        log.extend([
            self._entry(edge_id=0, t_idx=0, method="a", f_after=1.0, sigma_after=0.1),
            self._entry(edge_id=1, t_idx=0, method="b", f_after=2.0, sigma_after=0.2),
        ])
        df = log.for_edge(0)
        assert len(df) == 1
        assert df.iloc[0]["edge_id"] == 0


# =============================================================================
# ImputationEngine
# =============================================================================

class TestImputationEngine:

    def test_engine_attaches_log(self, acc):
        a_agg = _nan2d(); a_agg[0, 0] = 1.0
        ar = _agg_result_2d(a_agg)
        engine = ImputationEngine(acc_struct=acc)
        result = engine.run(ar)
        assert result.imputation_log is not None
        assert isinstance(result.imputation_log, ImputationLog)

    def test_engine_updates_f_agg(self, acc):
        a_agg = _nan2d(); a_agg[0, 0] = 10.0
        sigma = _nan2d(); sigma[0, 0] = 1.0
        ar = _agg_result_2d(a_agg, sigma)

        # Cell pair: (edge 0, t=0) → (edge 2, t=0), coeff=0.5
        imputer = ParametricImputer(_csv_pairs=[
            (0, 0, 2, 0, 0.5, 0.05, 0.0, "test", "desc")
        ])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imputer)
        result = engine.run(ar)

        assert result.a_agg[2, 0] == pytest.approx(5.0)

    def test_engine_pipeline_ordering(self, acc):
        a_agg = _nan2d(); a_agg[0, 0] = 10.0
        sigma = _nan2d(); sigma[0, 0] = 1.0
        ar = _agg_result_2d(a_agg, sigma)

        # First imputer: edge 0 → edge 1 at t=0 (f_1 = 2 * f_0 = 20)
        imp1 = ParametricImputer(_csv_pairs=[(0, 0, 1, 0, 2.0, 0.1, 0.0, "p1", "")])
        # Second imputer: edge 1 → edge 3 at t=0 (f_3 = 3 * f_1)
        imp2 = ParametricImputer(_csv_pairs=[(1, 0, 3, 0, 3.0, 0.1, 0.0, "p2", "")])

        engine = ImputationEngine(acc_struct=acc)
        engine.register(imp1)
        engine.register(imp2)
        result = engine.run(ar)

        assert result.a_agg[1, 0] == pytest.approx(20.0)
        assert result.a_agg[3, 0] == pytest.approx(60.0)

    def test_engine_early_stop_when_nothing_missing(self, acc):
        a_agg = np.arange(_N_EDGES * _N_TIMEPS, dtype=float).reshape(_N_EDGES, _N_TIMEPS)
        ar = _agg_result_2d(a_agg)

        class CountingImputer(BaseImputer):
            def __init__(self):
                self.called = 0
            def impute(self, f, sigma, missing, step, acc_struct):
                self.called += 1
                return []

        counter = CountingImputer()
        engine  = ImputationEngine(acc_struct=acc)
        engine.register(counter)
        engine.run(ar)
        assert counter.called == 0


# =============================================================================
# ParametricImputer — CSV bidirectional cell pairs
# =============================================================================

class TestParametricImputerCSV:

    def test_fill_b_from_a(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0]  = 100.0;  sigma[0, 0] = 5.0
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 0, 2, 0, 0.3, 0.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[2, 0] == pytest.approx(30.0)
        assert sigma[2, 0] == pytest.approx(0.3 * 5.0)

    def test_fill_a_from_b_inverse(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[2, 0]  = 30.0;  sigma[2, 0] = 1.5
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 0, 2, 0, 0.3, 0.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0, 0] == pytest.approx(100.0)
        expected = _reg_sensitivity(1.0 / 0.3, imp.q, imp.alpha_min) * 1.5
        assert sigma[0, 0] == pytest.approx(expected)

    def test_forward_uses_bare_alpha_small_coeff(self, acc):
        f     = _nan2d();  f[0, 0]     = 1.0
        sigma = _nan2d();  sigma[0, 0] = 1.0
        missing = np.isnan(f)
        imp = ParametricImputer(
            _csv_pairs=[(0, 0, 2, 0, 1e-3, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        imp.impute(f, sigma, missing, 0, acc)
        assert f[2, 0] == pytest.approx(1e-3)
        assert sigma[2, 0] == pytest.approx(1e-3, rel=1e-6)

    def test_forward_uses_bare_alpha_large_coeff(self, acc):
        f     = _nan2d();  f[0, 0]     = 1.0
        sigma = _nan2d();  sigma[0, 0] = 1.0
        missing = np.isnan(f)
        imp = ParametricImputer(
            _csv_pairs=[(0, 0, 2, 0, 100.0, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        imp.impute(f, sigma, missing, 0, acc)
        assert f[2, 0] == pytest.approx(100.0)
        assert sigma[2, 0] == pytest.approx(100.0, rel=1e-6)

    def test_both_known_no_fill(self, acc):
        f = _nan2d(); f[0, 0] = 10.0; f[1, 0] = 20.0
        sigma = _nan2d(); sigma[0, 0] = 1.0; sigma[1, 0] = 2.0
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 0, 1, 0, 2.0, 0.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0, 0] == pytest.approx(10.0)
        assert f[1, 0] == pytest.approx(20.0)

    def test_both_missing_no_fill(self, acc):
        f = _nan2d()
        sigma = _nan2d()
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 0, 2, 0, 0.5, 0.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[0, 0]) and np.isnan(f[2, 0])

    def test_sigma_propagation_includes_sigma_param(self, acc):
        f     = _nan2d();  f[0, 0]     = 10.0
        sigma = _nan2d();  sigma[0, 0] = 0.0
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 0, 2, 0, 2.0, 3.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        # σ_b = sqrt(4*0 + 9) = 3
        assert sigma[2, 0] == pytest.approx(3.0)

    def test_epsilon_added_in_quadrature(self, acc):
        f     = _nan2d();  f[0, 0]     = 10.0
        sigma = _nan2d();  sigma[0, 0] = 0.0
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 0, 2, 0, 1.0, 0.0, 0.1, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        # f_b = 10, sigma_b = sqrt(0 + (0.1*10)²) = 1.0
        assert f[2, 0] == pytest.approx(10.0)
        assert sigma[2, 0] == pytest.approx(1.0)

    def test_csv_bounds_not_applied_in_imputer(self, acc):
        f     = _nan2d();  f[0, 0]     = 100.0
        sigma = _nan2d();  sigma[0, 0] = 1.0
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 0, 2, 0, 2.0, 0.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[2, 0] == pytest.approx(200.0)

    def test_log_entries_returned(self, acc):
        f     = _nan2d();  f[0, 0]     = 5.0
        sigma = _nan2d();  sigma[0, 0] = 0.5
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 0, 2, 0, 1.0, 0.0, 0.0, "pid", "desc")])
        entries = imp.impute(f, sigma, missing, 3, acc)
        assert len(entries) == 1
        e = entries[0]
        assert e["edge_id"]  == 2
        assert e["t_idx"]    == 0
        assert e["step"]     == 3
        assert e["method"]   == "parametric"
        assert e["param_id"] == "pid"
        assert np.isnan(e["f_before"])

    def test_backward_alpha_regularization_caps_large_inverse(self, acc):
        f     = _nan2d();  f[2, 0]     = 1.0
        sigma = _nan2d();  sigma[2, 0] = 1.0
        missing = np.isnan(f)
        imp = ParametricImputer(
            _csv_pairs=[(0, 0, 2, 0, 0.01, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0, 0] == pytest.approx(100.0)
        assert sigma[0, 0] == pytest.approx(1.0, rel=1e-6)

    def test_backward_alpha_regularization_floor_for_small_inverse(self, acc):
        f     = _nan2d();  f[2, 0]     = 1.0
        sigma = _nan2d();  sigma[2, 0] = 1.0
        missing = np.isnan(f)
        imp = ParametricImputer(
            _csv_pairs=[(0, 0, 2, 0, 1000.0, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        imp.impute(f, sigma, missing, 0, acc)
        expected = _reg_sensitivity(1e-3, 3.0, 0.05) * 1.0
        assert sigma[0, 0] == pytest.approx(expected)
        assert sigma[0, 0] == pytest.approx(1e-3 / 0.05, rel=1e-3)
        assert sigma[0, 0] < 1.0


class TestParametricImputerFromCSV:

    def _write_csv(self, content: str, tmp_path) -> str:
        p = tmp_path / "params.csv"
        p.write_text(textwrap.dedent(content))
        return str(p)

    def test_from_csv_concentration_wildcard(self, acc, tmp_path):
        """Concentration row pairs Mass and CC edges. ``a_time=*`` broadcasts
        every period; the resolver expands one cell pair per (a,b,t)."""
        csv = (
            "param_id;param_type;description;"
            "a_sector_from;a_entity_from;a_region_from;"
            "a_sector_to;a_entity_to;a_region_to;"
            "a_time;a_valuation;"
            "b_sector_from;b_entity_from;b_region_from;"
            "b_sector_to;b_entity_to;b_region_to;"
            "b_time;b_valuation;"
            "value_nominal;unc_type;value_sd;value_min;value_max;"
            "pedigree_reliability;pedigree_completeness;"
            "pedigree_temporal_correlation;pedigree_geographical_correlation;"
            "pedigree_other_correlation;"
            "epsilon;bound_lower;bound_upper;source;comment\n"
            "conc_ore;concentration;test;"
            "*;*;*;Ore;Flow;*;*;Mass;"
            "same;same;same;same;same;same;same;CC;"
            "0.5;sd;0.05;;;;;;;;"
            "0.0;0.0;1.0;src;cmt\n"
        )
        path = tmp_path / "p.csv"
        path.write_text(csv)

        imp = ParametricImputer.from_csv(str(path), acc)
        a_ids = {pair[0] for pair in imp._csv_pairs}
        # Each (edge_a, edge_b) pair is replicated across n_timesteps.
        # Region A: edge 0 ↔ edge 2; region B: edge 1 ↔ edge 3.
        assert 0 in a_ids or 1 in a_ids
        # 2 regions × 3 timesteps = 6 cell pairs
        assert len(imp._csv_pairs) >= 6

    def test_from_csv_skips_comment_rows(self, acc, tmp_path):
        csv = (
            "param_id;param_type;description;"
            "a_sector_from;a_entity_from;a_region_from;"
            "a_sector_to;a_entity_to;a_region_to;"
            "a_time;a_valuation;"
            "b_sector_from;b_entity_from;b_region_from;"
            "b_sector_to;b_entity_to;b_region_to;"
            "b_time;b_valuation;"
            "value_nominal;unc_type;value_sd;value_min;value_max;"
            "pedigree_reliability;pedigree_completeness;"
            "pedigree_temporal_correlation;pedigree_geographical_correlation;"
            "pedigree_other_correlation;"
            "epsilon;bound_lower;bound_upper;source;comment\n"
            "# this is a comment row and should be skipped\n"
        )
        path = tmp_path / "p.csv"
        path.write_text(csv)
        imp = ParametricImputer.from_csv(str(path), acc)
        assert imp._csv_pairs == []

    def test_from_csv_pedigree_sigma(self, acc, tmp_path):
        csv = (
            "param_id;param_type;description;"
            "a_sector_from;a_entity_from;a_region_from;"
            "a_sector_to;a_entity_to;a_region_to;"
            "a_time;a_valuation;"
            "b_sector_from;b_entity_from;b_region_from;"
            "b_sector_to;b_entity_to;b_region_to;"
            "b_time;b_valuation;"
            "value_nominal;unc_type;value_sd;value_min;value_max;"
            "pedigree_reliability;pedigree_completeness;"
            "pedigree_temporal_correlation;pedigree_geographical_correlation;"
            "pedigree_other_correlation;"
            "epsilon;bound_lower;bound_upper;source;comment\n"
            "p1;concentration;d;"
            "*;*;*;Ore;Flow;*;*;Mass;"
            "same;same;same;same;same;same;same;CC;"
            "1.0;pedigree;;;;"
            "2;2;1;1;2;"
            "0.0;0.0;1.0;s;c\n"
        )
        path = tmp_path / "p.csv"
        path.write_text(csv)
        imp = ParametricImputer.from_csv(str(path), acc)
        # Each pair tuple is (a_id, ti_a, b_id, ti_b, coeff, sig_p, ...);
        # sigma_param is index 5 — must be > 0 for every pair.
        assert all(pair[5] > 0 for pair in imp._csv_pairs)

    def test_from_csv_same_region_constraint(self, acc, tmp_path):
        csv = (
            "param_id;param_type;description;"
            "a_sector_from;a_entity_from;a_region_from;"
            "a_sector_to;a_entity_to;a_region_to;"
            "a_time;a_valuation;"
            "b_sector_from;b_entity_from;b_region_from;"
            "b_sector_to;b_entity_to;b_region_to;"
            "b_time;b_valuation;"
            "value_nominal;unc_type;value_sd;value_min;value_max;"
            "pedigree_reliability;pedigree_completeness;"
            "pedigree_temporal_correlation;pedigree_geographical_correlation;"
            "pedigree_other_correlation;"
            "epsilon;bound_lower;bound_upper;source;comment\n"
            "p1;efficiency;d;"
            "Mining;Process;*;Ore;Flow;same;*;Mass;"
            "same;same;same;same;same;same;same;CC;"
            "0.8;sd;0.05;;;;;;;;"
            "0.0;0.0;1.0;s;c\n"
        )
        path = tmp_path / "p.csv"
        path.write_text(csv)
        imp = ParametricImputer.from_csv(str(path), acc)
        # 2 regions × 3 timesteps = 6 cell pairs.
        assert len(imp._csv_pairs) >= 6


# =============================================================================
# ParametricImputer — lambda relations
# =============================================================================

class TestParametricImputerLambda:

    def test_directed_fill_from_source(self, acc):
        f     = _nan2d();  f[4, 0]     = 50.0   # Ore_A→Mining_A, Mass, t=0
        sigma = _nan2d();  sigma[4, 0] = 5.0
        missing = np.isnan(f)

        rel = ParametricRelation(
            description="test",
            target =lambda ei: (ei["sector_from"] == "Mining") & (ei["valuation"] == "Mass")
                               & (ei["edge_id"].isin([0, 1])),
            sources=lambda ei: (ei["sector_from"] == "Ore") & (ei["valuation"] == "Mass")
                               & (ei["edge_id"].isin([4, 5])),
            coefficients=[0.1],
        )
        imp = ParametricImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)

        # target 0 (Mining_A→Ore_A) is filled from source 4 in region A at t=0.
        assert f[0, 0] == pytest.approx(5.0)
        # t=1, t=2 have no source — remain NaN.
        assert np.isnan(f[0, 1])
        assert np.isnan(f[0, 2])

    def test_missing_source_skips_target(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        missing = np.isnan(f)

        rel = ParametricRelation(
            description="test",
            target =lambda ei: ei["edge_id"] == 0,
            sources=lambda ei: ei["edge_id"] == 4,  # source also NaN
            coefficients=[1.0],
        )
        imp = ParametricImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[0, 0])

    def test_log_entry_contains_description(self, acc):
        f     = _nan2d();  f[4, 0]     = 10.0
        sigma = _nan2d();  sigma[4, 0] = 1.0
        missing = np.isnan(f)

        rel = ParametricRelation(
            description="my description",
            target =lambda ei: ei["edge_id"] == 0,
            sources=lambda ei: ei["edge_id"] == 4,
            coefficients=[1.0],
        )
        imp = ParametricImputer(relations=[rel])
        entries = imp.impute(f, sigma, missing, 0, acc)
        assert len(entries) >= 1
        assert all(e["description"] == "my description" for e in entries)


# =============================================================================
# FormulaImputer
# =============================================================================

class TestFormulaImputer:

    def test_single_term_no_region_binding(self, acc):
        """f_target = f_source at t=0."""
        f     = _nan2d();  f[0, 0]     = 80.0   # Mining_A→Ore_A, Mass
        sigma = _nan2d();  sigma[0, 0] = 8.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="copy",
            target=lambda ei: ei["edge_id"] == 4,
            terms=[
                SourceTerm(
                    selector=lambda ei: ei["edge_id"] == 0,
                    coefficient=1.0,
                    region_binding="none",
                    time_binding=True,
                    aggregate="single",
                )
            ],
            clip_lower=None,
        )
        imp = FormulaImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[4, 0] == pytest.approx(80.0)
        assert sigma[4, 0] == pytest.approx(8.0)

    def test_multi_term_supply_minus_export(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 100.0; sigma[0, 0] = 0.0
        f[4, 0] = 30.0;  sigma[4, 0] = 0.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="balance",
            target=lambda ei: ei["edge_id"] == 1,
            terms=[
                SourceTerm(
                    selector=lambda ei: ei["edge_id"] == 0,
                    coefficient=1.0,
                    region_binding="none",
                    time_binding=True,
                    aggregate="single",
                ),
                SourceTerm(
                    selector=lambda ei: ei["edge_id"] == 4,
                    coefficient=-1.0,
                    region_binding="none",
                    time_binding=True,
                    aggregate="single",
                ),
            ],
            clip_lower=None,
        )
        imp = FormulaImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[1, 0] == pytest.approx(70.0)

    def test_clip_lower_zero(self, acc):
        """Negative result clipped to bounds.l[1] = 0 (default)."""
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0; sigma[0, 0] = 1.0
        f[4, 0] = 50.0; sigma[4, 0] = 1.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="clip test",
            target=lambda ei: ei["edge_id"] == 1,
            terms=[
                SourceTerm(lambda ei: ei["edge_id"] == 0,  1.0, "none", True, "single"),
                SourceTerm(lambda ei: ei["edge_id"] == 4, -1.0, "none", True, "single"),
            ],
            clip_lower=0.0,
        )
        imp = FormulaImputer(relations=[rel])
        entries = imp.impute(f, sigma, missing, 0, acc)
        assert f[1, 0] == pytest.approx(0.0)
        # Only the t=0 cell was filled (others have NaN sources).
        filled = [e for e in entries if e["edge_id"] == 1 and e["t_idx"] == 0]
        assert filled and filled[0]["method"] == "formula_clipped"

    def test_no_clip_when_positive(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 80.0; sigma[0, 0] = 0.0
        f[4, 0] = 30.0; sigma[4, 0] = 0.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="no clip",
            target=lambda ei: ei["edge_id"] == 1,
            terms=[
                SourceTerm(lambda ei: ei["edge_id"] == 0,  1.0, "none", True, "single"),
                SourceTerm(lambda ei: ei["edge_id"] == 4, -1.0, "none", True, "single"),
            ],
            clip_lower=0.0,
        )
        imp = FormulaImputer(relations=[rel])
        entries = imp.impute(f, sigma, missing, 0, acc)
        filled = [e for e in entries if e["edge_id"] == 1 and e["t_idx"] == 0]
        assert filled and filled[0]["method"] == "formula"
        assert f[1, 0] == pytest.approx(50.0)

    def test_region_binding_from_matches_target_from(self, acc):
        """Supply edges 0 (region A) and 1 (region B), targets 4 (A) and 5 (B)."""
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 100.0; sigma[0, 0] = 10.0
        f[1, 0] = 200.0; sigma[1, 0] = 20.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="regional",
            target=lambda ei: (ei["sector_from"] == "Ore") & (ei["valuation"] == "Mass"),
            terms=[
                SourceTerm(
                    selector=lambda ei: (ei["sector_from"] == "Mining")
                                       & (ei["valuation"] == "Mass"),
                    coefficient=1.0,
                    region_binding="to_matches_target_from",
                    time_binding=True,
                    aggregate="single",
                ),
            ],
            clip_lower=None,
        )
        imp = FormulaImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[4, 0] == pytest.approx(100.0)
        assert f[5, 0] == pytest.approx(200.0)

    def test_missing_source_skips_target(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="skip",
            target=lambda ei: ei["edge_id"] == 0,
            terms=[SourceTerm(lambda ei: ei["edge_id"] == 4, 1.0, "none", True, "single")],
            clip_lower=0.0,
        )
        imp = FormulaImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[0, 0])

    def test_aggregate_sum_over_multiple_sources(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0; sigma[0, 0] = 0.0
        f[1, 0] = 20.0; sigma[1, 0] = 0.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="sum",
            target=lambda ei: ei["edge_id"] == 4,
            terms=[
                SourceTerm(
                    selector=lambda ei: ei["edge_id"].isin([0, 1]),
                    coefficient=1.0,
                    region_binding="none",
                    time_binding=True,
                    aggregate="sum",
                ),
            ],
            clip_lower=None,
        )
        imp = FormulaImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[4, 0] == pytest.approx(30.0)

    def test_param_ref_identity(self, acc):
        f     = _nan2d();  f[0, 0]     = 10.0
        sigma = _nan2d();  sigma[0, 0] = 0.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="param_ref",
            target=lambda ei: ei["edge_id"] == 4,
            terms=[
                SourceTerm(
                    selector=lambda ei: ei["edge_id"] == 0,
                    coefficient=ParamRef("eff_mine", "identity"),
                    region_binding="none",
                    time_binding=True,
                    aggregate="single",
                )
            ],
            clip_lower=None,
        )
        imp = FormulaImputer(relations=[rel], params={"eff_mine": 0.9})
        imp.impute(f, sigma, missing, 0, acc)
        assert f[4, 0] == pytest.approx(9.0)

    def test_param_ref_inverse(self, acc):
        f     = _nan2d();  f[0, 0]     = 10.0
        sigma = _nan2d();  sigma[0, 0] = 0.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="inverse",
            target=lambda ei: ei["edge_id"] == 4,
            terms=[
                SourceTerm(
                    selector=lambda ei: ei["edge_id"] == 0,
                    coefficient=ParamRef("eff_mine", "inverse"),
                    region_binding="none",
                    time_binding=True,
                    aggregate="single",
                )
            ],
            clip_lower=None,
        )
        imp = FormulaImputer(relations=[rel], params={"eff_mine": 0.5})
        imp.impute(f, sigma, missing, 0, acc)
        assert f[4, 0] == pytest.approx(20.0)


# =============================================================================
# TimeImputer
# =============================================================================

class TestTimeImputer:
    """
    Each row of f is one structural-edge series across time_range
    ["2020", "2021", "2022"], so t_idx 0/1/2 correspond to those years.
    """

    def test_linear_interpolation_between_two_known(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0;  sigma[0, 0] = 2.0   # 2020
        f[0, 2] = 20.0;  sigma[0, 2] = 4.0   # 2022
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear")
        entries = imp.impute(f, sigma, missing, 0, acc)

        assert f[0, 1] == pytest.approx(15.0)
        p = 1.3
        expected_sigma = (abs(0.5*2)**p + abs(0.5*4)**p) ** (1/p)
        assert sigma[0, 1] == pytest.approx(expected_sigma)
        assert any(e["edge_id"] == 0 and e["t_idx"] == 1
                   and e["method"] == "time_linear" for e in entries)

    def test_linear_interpolation_p2_recovers_quadrature(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0;  sigma[0, 0] = 2.0
        f[0, 2] = 20.0;  sigma[0, 2] = 4.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear", p=2.0)
        imp.impute(f, sigma, missing, 0, acc)
        assert sigma[0, 1] == pytest.approx(np.sqrt(5.0))

    def test_locf(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 42.0;  sigma[0, 0] = 3.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="locf")
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0, 1] == pytest.approx(42.0)

    def test_nocb(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 2] = 99.0;  sigma[0, 2] = 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="nocb")
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0, 1] == pytest.approx(99.0)

    def test_gap_inflation_applied(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0;  sigma[0, 0] = 1.0   # 2020
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="locf", gap_inflation=2.0)
        imp.impute(f, sigma, missing, 0, acc)
        # gap=1 → σ * 2
        assert sigma[0, 1] == pytest.approx(2.0)
        # gap=2 → σ * 4
        assert sigma[0, 2] == pytest.approx(4.0)

    def test_window_limits_lookback(self, acc):
        """Window=1 with LOCF: cascades column-by-column."""
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 5.0;  sigma[0, 0] = 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=1, method="locf")
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0, 1] == pytest.approx(5.0)
        assert f[0, 2] == pytest.approx(5.0)

    def test_window_zero_fills_nothing(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 5.0;  sigma[0, 0] = 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=0, method="locf")
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[0, 1])
        assert np.isnan(f[0, 2])

    def test_mean_method(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0;  sigma[0, 0] = 2.0
        f[0, 2] = 30.0;  sigma[0, 2] = 4.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="mean")
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0, 1] == pytest.approx(20.0)

    def test_already_known_not_overwritten(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0; sigma[0, 0] = 1.0
        f[0, 1] = 99.0; sigma[0, 1] = 1.0
        f[0, 2] = 20.0; sigma[0, 2] = 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear")
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0, 1] == pytest.approx(99.0)

    def test_independent_series_do_not_interact(self, acc):
        """Region-A series (edge 0) and region-B series (edge 1) are independent rows."""
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0; sigma[0, 0] = 1.0
        f[0, 2] = 30.0; sigma[0, 2] = 1.0
        # edge 1 has no observations
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear")
        imp.impute(f, sigma, missing, 0, acc)

        assert not np.isnan(f[0, 1])
        assert np.isnan(f[1, 0]) and np.isnan(f[1, 1]) and np.isnan(f[1, 2])

    def test_log_entries_for_time_linear(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 5.0;  sigma[0, 0] = 1.0
        f[0, 2] = 15.0; sigma[0, 2] = 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear")
        entries = imp.impute(f, sigma, missing, 0, acc)
        methods = {e["method"] for e in entries}
        assert "time_linear" in methods


# =============================================================================
# ImputationEngine — bounds enforcement
# =============================================================================

class TestImputationEngineBounds:

    def test_engine_clips_fill_exceeding_upper_bound(self, acc):
        acc.bounds.u[2] = 3.0
        a_agg = _nan2d(); a_agg[0, 0] = 10.0
        sigma = _nan2d(); sigma[0, 0] = 1.0
        ar = _agg_result_2d(a_agg, sigma)
        imputer = ParametricImputer(_csv_pairs=[(0, 0, 2, 0, 0.5, 0.0, 0.0, "p", "d")])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imputer)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = engine.run(ar)
        assert result.a_agg[2, 0] == pytest.approx(3.0)

    def test_engine_warns_on_clip_with_key_message_fields(self, acc):
        acc.bounds.u[2] = 3.0
        a_agg = _nan2d(); a_agg[0, 0] = 10.0
        sigma = _nan2d(); sigma[0, 0] = 1.0
        ar = _agg_result_2d(a_agg, sigma)
        imputer = ParametricImputer(_csv_pairs=[(0, 0, 2, 0, 0.5, 0.0, 0.0, "p", "d")])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imputer)
        with pytest.warns(UserWarning, match=r"clipped") as rec:
            engine.run(ar)
        msg = str(rec[0].message)
        assert "3" in msg

    def test_engine_no_warn_when_fill_within_bounds(self, acc):
        a_agg = _nan2d(); a_agg[0, 0] = 10.0
        sigma = _nan2d(); sigma[0, 0] = 1.0
        ar = _agg_result_2d(a_agg, sigma)
        imputer = ParametricImputer(_csv_pairs=[(0, 0, 2, 0, 0.5, 0.0, 0.0, "p", "d")])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imputer)
        import warnings
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            engine.run(ar)
        user_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


# =============================================================================
# FormulaImputer — acc.bounds clip
# =============================================================================

class TestFormulaImputerAccBounds:

    def _make_balance_rel(self, clip_lower=0.0):
        return FormulaRelation(
            description="balance",
            target=lambda ei: ei["edge_id"] == 1,
            terms=[
                SourceTerm(lambda ei: ei["edge_id"] == 0,  1.0, "none", True, "single"),
                SourceTerm(lambda ei: ei["edge_id"] == 4, -1.0, "none", True, "single"),
            ],
            clip_lower=clip_lower,
        )

    def test_formula_clips_to_zero_when_acc_bounds_lower_is_zero(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0; sigma[0, 0] = 0.0
        f[4, 0] = 50.0; sigma[4, 0] = 0.0
        missing = np.isnan(f)
        imp = FormulaImputer(relations=[self._make_balance_rel(clip_lower=0.0)])
        entries = imp.impute(f, sigma, missing, 0, acc)
        assert f[1, 0] == pytest.approx(0.0)
        cell_entries = [e for e in entries if e["edge_id"] == 1 and e["t_idx"] == 0]
        assert cell_entries and cell_entries[0]["method"] == "formula_clipped"

    def test_formula_allows_negative_when_acc_bounds_permit(self, acc):
        acc.bounds.l[1] = -1e6
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0; sigma[0, 0] = 0.0
        f[4, 0] = 50.0; sigma[4, 0] = 0.0
        missing = np.isnan(f)
        imp = FormulaImputer(relations=[self._make_balance_rel(clip_lower=0.0)])
        entries = imp.impute(f, sigma, missing, 0, acc)
        assert f[1, 0] == pytest.approx(-40.0)
        cell_entries = [e for e in entries if e["edge_id"] == 1 and e["t_idx"] == 0]
        assert cell_entries and cell_entries[0]["method"] == "formula"


# =============================================================================
# TimeImputer — interpolate_only=True
# =============================================================================

class TestTimeImputerInterpolateOnly:

    def test_interpolate_only_fills_interior_gap(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0; sigma[0, 0] = 2.0
        f[0, 2] = 20.0; sigma[0, 2] = 2.0
        missing = np.isnan(f)
        imp = TimeImputer(window=5, method="linear", interpolate_only=True)
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0, 1] == pytest.approx(15.0)

    def test_interpolate_only_skips_leading_gap(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 2] = 20.0; sigma[0, 2] = 1.0   # only 2022 known
        missing = np.isnan(f)
        imp = TimeImputer(window=5, method="linear", interpolate_only=True)
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[0, 0])
        assert np.isnan(f[0, 1])

    def test_interpolate_only_skips_trailing_gap(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0; sigma[0, 0] = 1.0
        missing = np.isnan(f)
        imp = TimeImputer(window=5, method="linear", interpolate_only=True)
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[0, 1])
        assert np.isnan(f[0, 2])


# =============================================================================
# TimeImputer scoping
# =============================================================================

class TestTimeImputerScoping:

    def test_scope_edges_mask_restricts_fills(self, acc):
        """Boolean mask drops untouched-edge rows entirely."""
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0; sigma[0, 0] = 1.0
        f[0, 2] = 20.0; sigma[0, 2] = 1.0
        f[1, 0] = 30.0; sigma[1, 0] = 1.0
        f[1, 2] = 50.0; sigma[1, 2] = 1.0
        missing = np.isnan(f)

        scope = np.zeros(_N_EDGES, dtype=bool)
        scope[0] = True  # only Mining_A→Ore_A series

        imp = TimeImputer(window=5, method="linear", interpolate_only=True,
                          scope_edges=scope)
        imp.impute(f, sigma, missing, 0, acc)

        assert f[0, 1] == pytest.approx(15.0)
        assert np.isnan(f[1, 1])

    def test_scope_feed_id_resolves_via_attach(self, acc):
        """scope_feed_id resolves to a mask via attach_agg_result(agg_result)."""
        # Two feeds: crude_ore covers edge 0 (t=0, 2); other covers edge 1.
        F     = np.full((2, _N_EDGES, _N_TIMEPS), np.nan, dtype=float)
        Sigma = np.full((2, _N_EDGES, _N_TIMEPS), np.nan, dtype=float)
        F[0, 0, [0, 2]] = [10.0, 20.0]; Sigma[0, 0, [0, 2]] = 1.0  # crude_ore
        F[1, 1, [0, 2]] = [30.0, 50.0]; Sigma[1, 1, [0, 2]] = 1.0  # other

        a_agg     = np.nanmean(F, axis=0)
        sigma_agg = np.nanmean(Sigma, axis=0)
        coverage  = np.isfinite(a_agg)

        ar = AggregationResult(
            a_agg     = a_agg,
            sigma_agg = sigma_agg,
            Omega     = np.where(np.isfinite(F), 1.0, 0.0),
            F         = F,
            Sigma     = Sigma,
            feed_ids  = ["crude_ore", "other"],
            coverage  = coverage,
        )

        f_work     = a_agg.copy()
        sigma_work = sigma_agg.copy()
        missing    = np.isnan(f_work)

        imp = TimeImputer(window=5, method="linear", interpolate_only=True,
                          scope_feed_id="crude_ore")
        imp.attach_agg_result(ar)
        imp.impute(f_work, sigma_work, missing, 0, acc)

        # Mining_A→Ore_A (in crude_ore): interpolated at t=1.
        assert f_work[0, 1] == pytest.approx(15.0)
        # Mining_B→Ore_B (not in crude_ore): untouched.
        assert np.isnan(f_work[1, 1])

    def test_scope_times_restricts_target_years(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0; sigma[0, 0] = 1.0
        f[0, 2] = 20.0; sigma[0, 2] = 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear", interpolate_only=True,
                          scope_times=["2021"])
        imp.impute(f, sigma, missing, 0, acc)

        assert f[0, 1] == pytest.approx(15.0)
        assert not np.isnan(f[0, 0])  # anchor untouched

    def test_scope_combined_with_extrapolation(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 0] = 10.0; sigma[0, 0] = 2.0   # 2020 anchor
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear", interpolate_only=False,
                          gap_inflation=1.5,
                          scope_times=["2021", "2022"])
        entries = imp.impute(f, sigma, missing, 0, acc)

        assert f[0, 1] == pytest.approx(10.0)
        assert f[0, 2] == pytest.approx(10.0)
        assert sigma[0, 1] == pytest.approx(2.0 * 1.5)
        assert sigma[0, 2] == pytest.approx(2.0 * 1.5 ** 2)
        methods = {e["method"] for e in entries
                   if e["edge_id"] == 0 and e["t_idx"] in (1, 2)}
        assert methods == {"time_linear"}

    def test_scope_edges_and_feed_id_mutually_exclusive(self):
        with pytest.raises(ValueError, match="at most one"):
            TimeImputer(scope_edges=np.zeros(_N_EDGES, dtype=bool),
                        scope_feed_id="x")

    def test_scope_feed_id_not_in_agg_raises(self, acc):
        a_agg = _nan2d(); a_agg[0, 0] = 1.0
        ar = _agg_result_2d(a_agg)
        imp = TimeImputer(scope_feed_id="nope")
        with pytest.raises(ValueError, match="not in"):
            imp.attach_agg_result(ar)


# =============================================================================
# ParametricImputer — specificity dedup
# =============================================================================

class TestParametricImputerSpecificityDedup:

    _CSV_HEADER = (
        "param_id;param_type;description;"
        "a_sector_from;a_entity_from;a_region_from;"
        "a_sector_to;a_entity_to;a_region_to;"
        "a_time;a_valuation;"
        "b_sector_from;b_entity_from;b_region_from;"
        "b_sector_to;b_entity_to;b_region_to;"
        "b_time;b_valuation;"
        "value_nominal;unc_type;value_sd;value_min;value_max;"
        "pedigree_reliability;pedigree_completeness;"
        "pedigree_temporal_correlation;pedigree_geographical_correlation;"
        "pedigree_other_correlation;"
        "epsilon;bound_lower;bound_upper;source;comment\n"
    )

    def test_specific_row_wins_over_wildcard_for_same_pair(self, acc, tmp_path):
        rows = (
            # generic row: wildcard region, coeff=0.5
            "generic;concentration;d;"
            "*;*;*;Ore;Flow;*;*;Mass;"
            "same;same;same;same;same;same;same;CC;"
            "0.5;sd;0.0;;;;;;;;"
            "0.0;0.0;1000.0;s;c\n"
            # specific row: region A only, coeff=0.9
            "specific;concentration;d;"
            "Mining;Process;A;Ore;Flow;A;*;Mass;"
            "same;same;same;same;same;same;same;CC;"
            "0.9;sd;0.0;;;;;;;;"
            "0.0;0.0;1000.0;s;c\n"
        )
        path = tmp_path / "dedup.csv"
        path.write_text(self._CSV_HEADER + rows)
        imp = ParametricImputer.from_csv(str(path), acc)

        # region-A pair: specific row (coeff=0.9) must win
        f_a   = _nan2d(); f_a[0, 0]   = 10.0
        sig_a = _nan2d(); sig_a[0, 0] = 0.0
        imp.impute(f_a, sig_a, np.isnan(f_a), 0, acc)
        assert f_a[2, 0] == pytest.approx(9.0)

        # region-B pair: only wildcard row applies (coeff=0.5)
        f_b   = _nan2d(); f_b[1, 0]   = 10.0
        sig_b = _nan2d(); sig_b[1, 0] = 0.0
        imp.impute(f_b, sig_b, np.isnan(f_b), 0, acc)
        assert f_b[3, 0] == pytest.approx(5.0)


# =============================================================================
# Pipeline chaining
# =============================================================================

class TestPipelineChaining:

    def test_pipeline_chaining_no_overwrite(self, acc):
        a_agg = _nan2d(); a_agg[0, 0] = 10.0
        sigma = _nan2d(); sigma[0, 0] = 1.0
        ar = _agg_result_2d(a_agg, sigma)
        imp1 = ParametricImputer(_csv_pairs=[(0, 0, 1, 0, 2.0, 0.0, 0.0, "p1", "")])
        imp2 = ParametricImputer(_csv_pairs=[(0, 0, 1, 0, 99.0, 0.0, 0.0, "p2", "")])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imp1)
        engine.register(imp2)
        result = engine.run(ar)
        assert result.a_agg[1, 0] == pytest.approx(20.0)

    def test_pipeline_chaining_forward_use(self, acc):
        a_agg = _nan2d(); a_agg[0, 0] = 10.0
        sigma = _nan2d(); sigma[0, 0] = 1.0
        ar = _agg_result_2d(a_agg, sigma)
        imp1 = ParametricImputer(_csv_pairs=[(0, 0, 1, 0, 2.0, 0.0, 0.0, "p1", "")])
        imp2 = ParametricImputer(_csv_pairs=[(1, 0, 3, 0, 3.0, 0.0, 0.0, "p2", "")])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imp1)
        engine.register(imp2)
        result = engine.run(ar)
        assert result.a_agg[1, 0] == pytest.approx(20.0)
        assert result.a_agg[3, 0] == pytest.approx(60.0)


# =============================================================================
# FlatPriorImputer
# =============================================================================

class TestFlatPriorImputer:

    def test_supply_uses_supply_cv(self, acc):
        """Edge 0 (supply) → σ = 0.001 · group scale."""
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 2] = 100.0;  sigma[0, 2] = 1.0   # Mining_A → Ore_A, Mass, 2022
        missing = np.isnan(f)
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.impute(f, sigma, missing, 0, acc)
        # Level-1 group (Mining, Process, A, Mass) → only edge 0; its row has
        # one finite cell (=100) → scale = 100 → σ = 0.001·100 = 0.1.
        assert f[0, 0] == 0.0
        assert sigma[0, 0] == pytest.approx(0.001 * 100.0)

    def test_use_uses_use_cv(self, acc):
        """Edge 4 (Ore_A → Mining_A, use): level-2 cascade kicks in."""
        f     = _nan2d()
        sigma = _nan2d()
        f[5, 0] = 200.0; sigma[5, 0] = 1.0
        missing = np.isnan(f)
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.impute(f, sigma, missing, 0, acc)
        # Level-1 group (Ore, Flow, A, Mass) has only edge 4 → all-NaN → empty.
        # Level-2 group (Ore, Mass) has edges 4 and 5 → finite cell = 200 → mean = 200.
        assert f[4, 0] == 0.0
        assert sigma[4, 0] == pytest.approx(0.3 * 200.0)

    def test_default_cv_for_unmasked_edges(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 2] = 50.0; sigma[0, 2] = 1.0
        missing = np.isnan(f)
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001},
            default_cv=0.7,
            acc_struct=acc,
        )
        imp.impute(f, sigma, missing, 0, acc)
        # Edge 4 (use) has no rule → default_cv=0.7. Narrow + broader (Ore, Mass)
        # groups are empty → level 3 (global median of finite, non-zero |f|) = 50.
        assert sigma[4, 0] == pytest.approx(0.7 * 50.0)
        # Edge 0 (supply): level-1 group has its own row with one finite (=50) → scale=50.
        assert sigma[0, 0] == pytest.approx(0.001 * 50.0)

    def test_first_match_wins_with_overlapping_masks(self, acc):
        acc.masks["use"] = acc.masks["use"].copy()
        acc.masks["use"][0] = True   # edge 0 now in both supply and use

        f     = _nan2d()
        sigma = _nan2d()
        f[0, 1] = 80.0; sigma[0, 1] = 1.0
        missing = np.isnan(f)

        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.impute(f, sigma, missing, 0, acc)
        # Edge 0's row has one finite (=80) → scale=80; supply wins → σ = 0.001·80.
        assert sigma[0, 0] == pytest.approx(0.001 * 80.0)

    def test_all_groups_empty_falls_back_to_unit_scale(self, acc):
        """All-NaN ⇒ global fallback scale = 1.0; σ = CV(mask) per edge,
        broadcast across the time axis."""
        f     = _nan2d()
        sigma = _nan2d()
        missing = np.isnan(f)
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.impute(f, sigma, missing, 0, acc)
        assert np.all(f == 0.0)
        expected = np.where(acc.masks["supply"], 0.001, 0.3)[:, None]
        assert np.allclose(sigma, expected)

    def test_log_entries_returned(self, acc):
        f     = _nan2d()
        sigma = _nan2d()
        f[0, 2] = 100.0; sigma[0, 2] = 1.0
        missing = np.isnan(f)
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        entries = imp.impute(f, sigma, missing, 5, acc)
        # (6 × 3) - 1 known cell = 17 missing → 17 entries.
        assert len(entries) == _N_EDGES * _N_TIMEPS - 1
        for e in entries:
            assert e["method"] == "flat_prior"
            assert e["step"] == 5
            assert e["f_after"] == 0.0

    def test_density_aware_cv_interpolation(self, acc):
        """With (cv_lo, cv_hi) and a 2-D coverage array, CV interpolates by
        per-edge density over (sector_from, region_from). Region-A rows
        fully covered → cv_lo; region-B rows fully uncovered → cv_hi."""
        coverage = np.zeros((_N_EDGES, _N_TIMEPS), dtype=bool)
        # Mark every cell of the region-A edges (0, 2, 4) as covered.
        coverage[[0, 2, 4], :] = True

        class _Agg:
            pass
        agg = _Agg(); agg.coverage = coverage

        imp = FlatPriorImputer(
            cv_by_mask={"supply": (0.04, 0.30), "use": (0.20, 0.60)},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.attach_agg_result(agg)
        cv = imp._resolve_cv()
        # Edge 4 (use, from=Ore_A): density(Ore, A) = 1 → cv = 0.20.
        # Edge 5 (use, from=Ore_B): density(Ore, B) = 0 → cv = 0.60.
        assert cv[4] == pytest.approx(0.20)
        assert cv[5] == pytest.approx(0.60)
        # Edge 0 (supply, from=Mining_A): density(Mining, A) = 1 → cv = 0.04.
        # Edge 1 (supply, from=Mining_B): density(Mining, B) = 0 → cv = 0.30.
        assert cv[0] == pytest.approx(0.04)
        assert cv[1] == pytest.approx(0.30)

    def test_unknown_mask_name_raises(self, acc):
        with pytest.raises(KeyError, match="bogus"):
            FlatPriorImputer(
                cv_by_mask={"bogus": 0.1},
                default_cv=0.5,
                acc_struct=acc,
            )

    def test_cv_hi_above_one_raises(self, acc):
        with pytest.raises(ValueError, match="row_use"):
            FlatPriorImputer(
                cv_by_mask={"supply": (0.04, 0.3), "use": (0.2, 0.5),
                            "row_use": (0.3, 1.05)},
                default_cv=0.3,
                acc_struct=acc,
                masks={"row_use": np.zeros(acc.n_edges, dtype=bool)},
            )

    def test_default_cv_above_one_raises(self, acc):
        with pytest.raises(ValueError, match="default_cv"):
            FlatPriorImputer(
                cv_by_mask={"supply": (0.04, 0.3)},
                default_cv=1.5,
                acc_struct=acc,
            )

    def test_masks_overlay_resolves_caller_supplied(self, acc):
        custom = np.zeros(acc.n_edges, dtype=bool)
        custom[0] = True
        imp = FlatPriorImputer(
            cv_by_mask={"my_overlay": (0.1, 0.2), "supply": (0.04, 0.3)},
            default_cv=0.3,
            acc_struct=acc,
            masks={"my_overlay": custom},
        )
        names = [n for n, _ in imp._masks]
        assert names == ["my_overlay", "supply"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
