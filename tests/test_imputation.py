"""
test_imputation.py
==================
pytest suite for pysuto/impute.py

Covers:
    ImputationLog           — basic operations, as_dataframe, for_edge
    ImputationEngine        — pipeline ordering, early stop, log attachment
    ParametricImputer       — CSV bidirectional pairs, lambda directed fills
    FormulaImputer          — balance equations, region binding, clipping
    TimeImputer             — linear, locf, nocb, gap inflation, window
"""
from __future__ import annotations

import io
import textwrap
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from pysuto.agg import AggregationResult, AggregationMath
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

# Lp combiner unit tests live in tests/test_lp.py.
# sigma_from_row unit tests live in tests/test_param_relation.py.


# =============================================================================
# Regularized sensitivity s(α) — used in ParametricImputer σ propagation
# =============================================================================

class TestRegSensitivity:
    def test_saturates_to_one_for_large_alpha(self):
        # |α| >> α_min → s(α) → 1
        assert _reg_sensitivity(100.0, q=3.0, alpha_min=0.05) == pytest.approx(1.0, rel=1e-6)

    def test_linear_damping_for_small_alpha(self):
        # |α| << α_min → s(α) ≈ |α|/α_min
        a, am, q = 1e-4, 0.05, 3.0
        assert _reg_sensitivity(a, q, am) == pytest.approx(a / am, rel=1e-3)

    def test_negative_uses_abs(self):
        assert _reg_sensitivity(-2.0, q=3.0, alpha_min=0.05) == pytest.approx(
            _reg_sensitivity(2.0, q=3.0, alpha_min=0.05)
        )

    def test_at_alpha_equals_alpha_min(self):
        # s(α_min) = 1 / 2^(1/q)
        for q in (2.0, 3.0, 4.0):
            assert _reg_sensitivity(0.05, q=q, alpha_min=0.05) == pytest.approx(
                1.0 / 2 ** (1.0 / q)
            )

    def test_zero_alpha_returns_zero(self):
        assert _reg_sensitivity(0.0, q=3.0, alpha_min=0.05) == 0.0


# =============================================================================
# ParametricImputer consumed-set selection
# =============================================================================

class TestParametricConsumes:
    """ParametricImputer must not build per-pair fills from pooled (agg) rows
    or the stripping_ratio family — those are owned by NetworkImputer + the
    ratio constraint builders. Guards against a spurious Scrap = Matte/eff fill
    once Smelting/CFR eff rows are broadened to a_sector_from=*."""

    def test_skips_agg_rows(self):
        from pysuto.impute.parametric import _parametric_consumes
        for v in ("true", "True", "1", "yes", "y", "t"):
            assert _parametric_consumes({"agg": v}) is False

    def test_skips_stripping_ratio(self):
        from pysuto.impute.parametric import _parametric_consumes
        assert _parametric_consumes({"param_type": "stripping_ratio"}) is False

    def test_consumes_plain_efficiency_and_concentration(self):
        from pysuto.impute.parametric import _parametric_consumes
        assert _parametric_consumes({"agg": "", "param_type": "efficiency"}) is True
        assert _parametric_consumes({"param_type": "concentration"}) is True
        assert _parametric_consumes({}) is True


# =============================================================================
# Fixtures — minimal mock accounting structure
# =============================================================================

class MockAcc:
    """
    Small accounting structure for tests.

    Nodes (node_index):
        0  Mining / Process / A
        1  Ore    / Flow    / A
        2  Mining / Process / B
        3  Ore    / Flow    / B

    Edges (edge_index), edge_id order:
        0  0→1  t=2020  Mass   (supply: Mining_A → Ore_A, Mass)
        1  2→3  t=2020  Mass   (supply: Mining_B → Ore_B, Mass)
        2  0→1  t=2020  CC     (same pair, CC valuation)
        3  2→3  t=2020  CC     (same pair, CC valuation)
        4  1→0  t=2020  Mass   (use: Ore_A → Mining_A, Mass)
        5  3→2  t=2020  Mass   (use: Ore_B → Mining_B, Mass)
        6  0→1  t=2021  Mass   (same as 0, next time step)
        7  2→3  t=2021  Mass   (same as 1, next time step)
        8  0→1  t=2022  Mass   (same as 0, t+2)
        9  2→3  t=2022  Mass   (same as 1, t+2)

    Bounds (bounds.l / bounds.u):
        Default l=0, u=1e9 for all edges.
        Tests that need non-default bounds mutate acc.bounds.l[e] or acc.bounds.u[e] directly.
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
                "edge_id":   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "from_id":   [0, 2, 0, 2, 1, 3, 0, 2, 0, 2],
                "to_id":     [1, 3, 1, 3, 0, 2, 1, 3, 1, 3],
                "time":      ["2020","2020","2020","2020","2020","2020","2021","2021","2022","2022"],
                "valuation": ["Mass","Mass","CC",  "CC",  "Mass","Mass","Mass","Mass","Mass","Mass"],
            }
        )
        self.bounds = MockAcc._Bounds(10)
        self.n_edges = 10
        # Edges 0..3, 6..9: Process→Flow → supply
        # Edges 4,5: Flow→Process → use
        self.masks = {
            "supply": np.array(
                [True, True, True, True, False, False, True, True, True, True], dtype=bool,
            ),
            "use": np.array(
                [False, False, False, False, True, True, False, False, False, False], dtype=bool,
            ),
            "tradable": np.zeros(10, dtype=bool),
            "sb":       np.zeros(10, dtype=bool),
        }


@pytest.fixture
def acc():
    return MockAcc()


def _agg_result(f_values, sigma_values=None, n_feeds=1):
    """Build a minimal AggregationResult from flat arrays."""
    N = len(f_values)
    f     = np.array(f_values, dtype=float)
    sigma = np.array(sigma_values if sigma_values is not None
                     else [1.0 if not np.isnan(v) else np.nan for v in f_values], dtype=float)
    # Wrap in F/Sigma matrices for AggregationResult constructor
    F     = f[np.newaxis, :]
    Sigma = sigma[np.newaxis, :]
    return AggregationResult(
        f_agg     = f.copy(),
        sigma_agg = sigma.copy(),
        Omega     = np.where(np.isfinite(F), 1.0, 0.0),
        F         = F,
        Sigma     = Sigma,
        feed_ids  = ["test"],
        coverage  = np.isfinite(f),
    )


# =============================================================================
# 1. ImputationLog
# =============================================================================

class TestImputationLog:

    def test_empty_log(self):
        log = ImputationLog()
        assert len(log) == 0
        assert log.as_dataframe().empty

    def test_append_and_len(self):
        log = ImputationLog()
        log.append({"edge_id": 0, "step": 0, "method": "test",
                     "source_edges": [], "coefficients": [],
                     "f_before": np.nan, "f_after": 5.0,
                     "sigma_before": np.nan, "sigma_after": 0.5,
                     "description": "x", "param_id": None})
        assert len(log) == 1

    def test_as_dataframe_columns(self):
        log = ImputationLog()
        log.append({"edge_id": 0, "step": 0, "method": "parametric",
                     "source_edges": [1], "coefficients": [2.0],
                     "f_before": np.nan, "f_after": 10.0,
                     "sigma_before": np.nan, "sigma_after": 1.0,
                     "description": "d", "param_id": "p1"})
        df = log.as_dataframe()
        assert set(df.columns) == {
            "edge_id","step","imputer_id","pass","method",
            "source_edges","coefficients",
            "f_before","f_after","sigma_before","sigma_after",
            "description","param_id",
        }

    def test_for_edge_filters_correctly(self):
        log = ImputationLog()
        log.extend([
            {"edge_id": 0, "step": 0, "method": "a", "source_edges": [],
             "coefficients": [], "f_before": np.nan, "f_after": 1.0,
             "sigma_before": np.nan, "sigma_after": 0.1, "description": "", "param_id": None},
            {"edge_id": 1, "step": 0, "method": "b", "source_edges": [],
             "coefficients": [], "f_before": np.nan, "f_after": 2.0,
             "sigma_before": np.nan, "sigma_after": 0.2, "description": "", "param_id": None},
        ])
        df = log.for_edge(0)
        assert len(df) == 1
        assert df.iloc[0]["edge_id"] == 0


# =============================================================================
# 2. ImputationEngine
# =============================================================================

class TestImputationEngine:

    def test_engine_attaches_log(self, acc):
        ar = _agg_result([1.0, np.nan, np.nan, np.nan, np.nan, np.nan,
                          np.nan, np.nan, np.nan, np.nan])
        engine = ImputationEngine(acc_struct=acc)
        result = engine.run(ar)
        assert result.imputation_log is not None
        assert isinstance(result.imputation_log, ImputationLog)

    def test_engine_updates_f_agg(self, acc):
        # Edge 0 known (f=10), edge 2 missing — ParametricImputer fills 2 from 0
        f = [10.0, np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan]
        ar = _agg_result(f, [1.0, np.nan, np.nan, np.nan, np.nan, np.nan,
                              np.nan, np.nan, np.nan, np.nan])

        # CSV-based pair: edge 0 ↔ edge 2, coeff=0.5
        imputer = ParametricImputer(_csv_pairs=[
            (0, 2, 0.5, 0.05, 0.0, "test", "desc")
        ])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imputer)
        result = engine.run(ar)

        assert result.f_agg[2] == pytest.approx(5.0)

    def test_engine_pipeline_ordering(self, acc):
        """Second imputer sees fills produced by the first."""
        f = [10.0, np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan]
        ar = _agg_result(f, [1.0]+[np.nan]*9)

        # First imputer: edge 0 → edge 1 (f_1 = 2 * f_0 = 20)
        imp1 = ParametricImputer(_csv_pairs=[(0, 1, 2.0, 0.1, 0.0, "p1", "")])
        # Second imputer: edge 1 → edge 3 (f_3 = 3 * f_1 — edge 1 filled by imp1)
        imp2 = ParametricImputer(_csv_pairs=[(1, 3, 3.0, 0.1, 0.0, "p2", "")])

        engine = ImputationEngine(acc_struct=acc)
        engine.register(imp1)
        engine.register(imp2)
        result = engine.run(ar)

        assert result.f_agg[1] == pytest.approx(20.0)
        assert result.f_agg[3] == pytest.approx(60.0)

    def test_engine_early_stop_when_nothing_missing(self, acc):
        """If all edges are covered, engine skips all imputers."""
        f = [float(i) for i in range(10)]
        ar = _agg_result(f)

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
# 2b. ImputationEngine — fixpoint cascade + imputer-id provenance
# =============================================================================

class TestImputationCascade:
    """register_cascade iterates the conc/eff-style imputers to a fixpoint."""

    def _two_hop(self):
        # edge 0 known; A fills 2 from 0 (f2 = 2·f0); B fills 3 from 2 (f3 = 3·f2).
        # Registering B before A means B can only succeed on a *second* pass.
        ar = _agg_result([10.0] + [np.nan] * 9, [1.0] + [np.nan] * 9)
        imp_A = ParametricImputer(_csv_pairs=[(0, 2, 2.0, 0.1, 0.0, "A", "")], id="A")
        imp_B = ParametricImputer(_csv_pairs=[(2, 3, 3.0, 0.1, 0.0, "B", "")], id="B")
        return ar, imp_A, imp_B

    def test_single_pass_leaves_chain_incomplete(self, acc):
        """Terminal-only, B before A: B runs too early → edge 3 stays NaN."""
        ar, imp_A, imp_B = self._two_hop()
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imp_B)
        engine.register(imp_A)
        result = engine.run(ar)
        assert result.f_agg[2] == pytest.approx(20.0)   # A filled it
        assert np.isnan(result.f_agg[3])                 # B ran before edge 2 existed

    def test_cascade_fills_chain_via_fixpoint(self, acc):
        """Same imputers/order as a cascade → pass 2 lets B fill edge 3."""
        ar, imp_A, imp_B = self._two_hop()
        engine = ImputationEngine(acc_struct=acc)
        engine.register_cascade([imp_B, imp_A], max_iter=10)
        result = engine.run(ar)
        assert result.f_agg[2] == pytest.approx(20.0)
        assert result.f_agg[3] == pytest.approx(60.0)
        passes = result.imputation_metadata.as_dataframe()["pass"].dropna()
        assert int(passes.max()) == 2                    # converged after 2 productive passes

    def test_max_iter_caps_passes(self, acc):
        """max_iter=1 allows a single pass → the second hop stays unfilled."""
        ar, imp_A, imp_B = self._two_hop()
        engine = ImputationEngine(acc_struct=acc)
        engine.register_cascade([imp_B, imp_A], max_iter=1)
        result = engine.run(ar)
        assert result.f_agg[2] == pytest.approx(20.0)    # A filled on pass 1
        assert np.isnan(result.f_agg[3])                 # B needed pass 2 → capped

    def test_no_dead_pass_recorded(self, acc):
        """The trailing zero-fill sweep that proves convergence is discarded."""
        ar, imp_A, imp_B = self._two_hop()
        engine = ImputationEngine(acc_struct=acc)
        engine.register_cascade([imp_B, imp_A], max_iter=10)
        result = engine.run(ar)
        md = result.imputation_metadata.as_dataframe()
        # 2 productive passes × 2 imputers = 4 records; no pass 3.
        assert set(md["pass"].dropna().astype(int)) == {1, 2}
        assert (md["pass"] == 3).sum() == 0
        assert len(md) == 4

    def test_provenance_in_log_and_meta(self, acc):
        ar, imp_A, imp_B = self._two_hop()
        engine = ImputationEngine(acc_struct=acc)
        engine.register_cascade([imp_B, imp_A], max_iter=10)
        result = engine.run(ar)
        log = result.imputation_log.as_dataframe()
        row2 = log[log["edge_id"] == 2].iloc[0]
        row3 = log[log["edge_id"] == 3].iloc[0]
        assert row2["imputer_id"] == "A" and int(row2["pass"]) == 1
        assert row3["imputer_id"] == "B" and int(row3["pass"]) == 2
        md = result.imputation_metadata.as_dataframe()
        assert {"imputer_id", "pass", "config"} <= set(md.columns)
        assert set(md["imputer_id"]) == {"A", "B"}
        assert all(isinstance(c, dict) for c in md["config"])

    def test_terminal_imputer_has_null_pass(self, acc):
        ar, imp_A, imp_B = self._two_hop()
        imp_C = ParametricImputer(_csv_pairs=[(0, 4, 1.0, 0.1, 0.0, "C", "")], id="C")
        engine = ImputationEngine(acc_struct=acc)
        engine.register_cascade([imp_B, imp_A], max_iter=10)
        engine.register(imp_C)                            # terminal
        result = engine.run(ar)
        assert result.f_agg[4] == pytest.approx(10.0)
        rowC = result.imputation_log.as_dataframe()
        rowC = rowC[rowC["edge_id"] == 4].iloc[0]
        assert rowC["imputer_id"] == "C"
        assert pd.isna(rowC["pass"])

    def test_default_id_is_class_name(self):
        assert ParametricImputer(_csv_pairs=[]).id == "ParametricImputer"

    def test_backward_compat_terminal_only(self, acc):
        """No cascade → old single-pass behaviour: steps 0,1 and null pass."""
        ar = _agg_result([10.0] + [np.nan] * 9, [1.0] + [np.nan] * 9)
        imp1 = ParametricImputer(_csv_pairs=[(0, 1, 2.0, 0.1, 0.0, "p1", "")], id="one")
        imp2 = ParametricImputer(_csv_pairs=[(1, 3, 3.0, 0.1, 0.0, "p2", "")], id="two")
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imp1)
        engine.register(imp2)
        result = engine.run(ar)
        assert result.f_agg[1] == pytest.approx(20.0)
        assert result.f_agg[3] == pytest.approx(60.0)
        md = result.imputation_metadata.as_dataframe()
        assert list(md["step"]) == [0, 1]
        assert md["pass"].isna().all()


# =============================================================================
# 3. ParametricImputer — CSV bidirectional pairs
# =============================================================================

class TestParametricImputerCSV:

    def test_fill_b_from_a(self, acc):
        """A known, B missing (forward) → f_b = coeff × f_a; σ uses bare coeff."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]  = 100.0;  sigma[0] = 5.0
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 2, 0.3, 0.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[2] == pytest.approx(30.0)
        # Forward branch uses bare coeff: σ_b = |coeff| · σ_a
        assert sigma[2] == pytest.approx(0.3 * 5.0)

    def test_fill_a_from_b_inverse(self, acc):
        """B known, A missing (backward) → f_a = f_b / coeff; σ uses s(1/coeff)."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[2]  = 30.0;  sigma[2] = 1.5
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 2, 0.3, 0.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0] == pytest.approx(100.0)
        # Backward branch is regularized: σ_a = s(1/coeff) · σ_b
        # 1/0.3 ≈ 3.33; with α_min=0.05, q=3 this saturates close to 1
        expected = _reg_sensitivity(1.0 / 0.3, imp.q, imp.alpha_min) * 1.5
        assert sigma[0] == pytest.approx(expected)

    def test_forward_uses_bare_alpha_small_coeff(self, acc):
        """Forward branch with very small α: σ propagation is NOT regularized."""
        f     = np.full(10, np.nan);  f[0]     = 1.0
        sigma = np.full(10, np.nan);  sigma[0] = 1.0
        missing = np.isnan(f)
        imp = ParametricImputer(
            _csv_pairs=[(0, 2, 1e-3, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        imp.impute(f, sigma, missing, 0, acc)
        assert f[2] == pytest.approx(1e-3)
        # Forward uses bare α: σ_b = α · σ_a = 1e-3 · 1 = 1e-3
        # (regularized would give s(α) · σ_a ≈ (α/α_min) · σ_a = 0.02)
        assert sigma[2] == pytest.approx(1e-3, rel=1e-6)

    def test_forward_uses_bare_alpha_large_coeff(self, acc):
        """Forward branch with large α: σ amplifies linearly (no cap)."""
        f     = np.full(10, np.nan);  f[0]     = 1.0
        sigma = np.full(10, np.nan);  sigma[0] = 1.0
        missing = np.isnan(f)
        imp = ParametricImputer(
            _csv_pairs=[(0, 2, 100.0, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        imp.impute(f, sigma, missing, 0, acc)
        assert f[2] == pytest.approx(100.0)
        # Forward uses bare α: σ_b = 100 · 1 = 100  (not capped)
        assert sigma[2] == pytest.approx(100.0, rel=1e-6)

    def test_both_known_no_fill(self, acc):
        """Both A and B known → no fill, values unchanged."""
        f = np.array([10.0, 20.0] + [np.nan]*8)
        sigma = np.array([1.0, 2.0] + [np.nan]*8)
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 1, 2.0, 0.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0] == pytest.approx(10.0)
        assert f[1] == pytest.approx(20.0)

    def test_both_missing_no_fill(self, acc):
        f = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 2, 0.5, 0.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[0]) and np.isnan(f[2])

    def test_sigma_propagation_includes_sigma_param(self, acc):
        """σ_b² = coeff² σ_a² + σ_param²."""
        f     = np.full(10, np.nan);  f[0]     = 10.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.0
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 2, 2.0, 3.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        # σ_b = sqrt(4*0 + 9) = 3
        assert sigma[2] == pytest.approx(3.0)

    def test_epsilon_added_in_quadrature(self, acc):
        f     = np.full(10, np.nan);  f[0]     = 10.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.0
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 2, 1.0, 0.0, 0.1, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        # f_b = 10, sigma_b = sqrt(0 + (0.1*10)²) = 1.0
        assert f[2] == pytest.approx(10.0)
        assert sigma[2] == pytest.approx(1.0)

    def test_csv_bounds_not_applied_in_imputer(self, acc):
        """ParametricImputer ignores per-row CSV bounds (they are *ratio* bounds
        consumed by RatioBoundBuilder, not absolute caps on the target flow)."""
        f     = np.full(10, np.nan);  f[0]     = 100.0
        sigma = np.full(10, np.nan);  sigma[0] = 1.0
        missing = np.isnan(f)
        # coeff=2 → natural fill 200; the imputer no longer reads bound_upper.
        imp = ParametricImputer(_csv_pairs=[(0, 2, 2.0, 0.0, 0.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[2] == pytest.approx(200.0)

    def test_log_entries_returned(self, acc):
        f     = np.full(10, np.nan);  f[0]     = 5.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.5
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 2, 1.0, 0.0, 0.0, "pid", "desc")])
        entries = imp.impute(f, sigma, missing, 3, acc)
        assert len(entries) == 1
        e = entries[0]
        assert e["edge_id"]  == 2
        assert e["step"]     == 3
        assert e["method"]   == "parametric"
        assert e["param_id"] == "pid"
        assert np.isnan(e["f_before"])

    def test_backward_alpha_regularization_caps_large_inverse(self, acc):
        """Backward branch with small coeff (1/coeff large): σ capped at σ_b."""
        f     = np.full(10, np.nan);  f[2]     = 1.0
        sigma = np.full(10, np.nan);  sigma[2] = 1.0
        missing = np.isnan(f)
        # coeff = 0.01 → inv = 100 (large); s(100) ≈ 1, so σ_a is capped at σ_b.
        imp = ParametricImputer(
            _csv_pairs=[(0, 2, 0.01, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        imp.impute(f, sigma, missing, 0, acc)
        # f uses bare inverse: f_a = (1/0.01) * 1 = 100
        assert f[0] == pytest.approx(100.0)
        # σ uses s(inv) ≈ 1, so σ_a ≈ σ_b = 1.0  (vs. 100 without regularization)
        assert sigma[0] == pytest.approx(1.0, rel=1e-6)

    def test_backward_alpha_regularization_floor_for_small_inverse(self, acc):
        """Backward branch with very large coeff (1/coeff small): σ floored, not bare α."""
        f     = np.full(10, np.nan);  f[2]     = 1.0
        sigma = np.full(10, np.nan);  sigma[2] = 1.0
        missing = np.isnan(f)
        # coeff = 1000 → inv = 1e-3; s(1e-3) ≈ 1e-3 / α_min = 0.02
        imp = ParametricImputer(
            _csv_pairs=[(0, 2, 1000.0, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        imp.impute(f, sigma, missing, 0, acc)
        expected = _reg_sensitivity(1e-3, 3.0, 0.05) * 1.0
        assert sigma[0] == pytest.approx(expected)
        assert sigma[0] == pytest.approx(1e-3 / 0.05, rel=1e-3)
        assert sigma[0] < 1.0


class TestParametricImputerFromCSV:

    def _write_csv(self, content: str, tmp_path) -> str:
        p = tmp_path / "params.csv"
        p.write_text(textwrap.dedent(content))
        return str(p)

    def test_from_csv_concentration_wildcard(self, acc, tmp_path):
        """
        Concentration row: A-edge has a_sector_to=Ore, a_valuation=Mass.
        B-edge: all 'same' except b_valuation=CC.
        Edge 0 (Mining→Ore, Mass) should pair with edge 2 (Mining→Ore, CC).
        """
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
        # Edges 0 and 1 have to_node = Ore/Flow; each should pair with CC counterpart
        a_ids = {pair[0] for pair in imp._csv_pairs}
        b_ids = {pair[1] for pair in imp._csv_pairs}
        # edge 0 (Mass) should pair with edge 2 (CC); edge 1 (Mass) with edge 3 (CC)
        assert 0 in a_ids or 2 in a_ids   # one of the two valuations is A
        assert len(imp._csv_pairs) >= 2    # at least 2 pairs (one per region)

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
        """unc_type=pedigree produces non-zero sigma from scores."""
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
        # Each pair should have a positive sigma_param
        assert all(pair[3] > 0 for pair in imp._csv_pairs)

    def test_from_csv_same_region_constraint(self, acc, tmp_path):
        """
        a_region_to=same filters A edges to same-region only.
        Edges 0 and 2 are same-region (Mining_A→Ore_A); cross-region edges
        are not in this mock structure, so all edges should qualify.
        """
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
        # Should have pairs for regions A and B (and times 2020, 2021, 2022)
        assert len(imp._csv_pairs) >= 2


# =============================================================================
# 4. ParametricImputer — lambda relations
# =============================================================================

class TestParametricImputerLambda:

    def test_directed_fill_from_source(self, acc):
        """target filled from source in the same region and time."""
        f     = np.full(10, np.nan);  f[4]     = 50.0   # Ore_A→Mining_A, Mass
        sigma = np.full(10, np.nan);  sigma[4] = 5.0
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

        # target 0 (Mining_A→Ore_A) should be filled from source 4 (Ore_A→Mining_A)
        # same region A, same time 2020
        assert f[0] == pytest.approx(5.0)

    def test_missing_source_skips_target(self, acc):
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        missing = np.isnan(f)

        rel = ParametricRelation(
            description="test",
            target =lambda ei: ei["edge_id"] == 0,
            sources=lambda ei: ei["edge_id"] == 4,  # source also NaN
            coefficients=[1.0],
        )
        imp = ParametricImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[0])

    def test_log_entry_contains_description(self, acc):
        f     = np.full(10, np.nan);  f[4]     = 10.0
        sigma = np.full(10, np.nan);  sigma[4] = 1.0
        missing = np.isnan(f)

        rel = ParametricRelation(
            description="my description",
            target =lambda ei: ei["edge_id"] == 0,
            sources=lambda ei: ei["edge_id"] == 4,
            coefficients=[1.0],
        )
        imp = ParametricImputer(relations=[rel])
        entries = imp.impute(f, sigma, missing, 0, acc)
        assert len(entries) == 1
        assert entries[0]["description"] == "my description"


# =============================================================================
# 5. FormulaImputer
# =============================================================================

class TestFormulaImputer:

    def test_single_term_no_region_binding(self, acc):
        """f_target = 1.0 * f_source (no region filter needed: only one match)."""
        f     = np.full(10, np.nan);  f[0]     = 80.0   # Mining_A→Ore_A, Mass
        sigma = np.full(10, np.nan);  sigma[0] = 8.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="copy",
            target=lambda ei: ei["edge_id"] == 4,  # Ore_A→Mining_A, Mass
            terms=[
                SourceTerm(
                    selector=lambda ei: ei["edge_id"] == 0,
                    coefficient=1.0,
                    region_binding="none",
                    time_binding=False,
                    aggregate="single",
                )
            ],
            clip_lower=None,
        )
        imp = FormulaImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[4] == pytest.approx(80.0)
        assert sigma[4] == pytest.approx(8.0)

    def test_multi_term_supply_minus_export(self, acc):
        """
        f_target = f_supply - f_export (two terms, one positive, one negative).
        Supply edge 0 (f=100), export edge 4 (f=30) → target = 70.
        """
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]     = 100.0;  sigma[0] = 0.0
        f[4]     = 30.0;   sigma[4] = 0.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="balance",
            target=lambda ei: ei["edge_id"] == 1,   # Mining_B→Ore_B, Mass (target to fill)
            terms=[
                SourceTerm(
                    selector=lambda ei: ei["edge_id"] == 0,
                    coefficient=1.0,
                    region_binding="none",
                    time_binding=False,
                    aggregate="single",
                ),
                SourceTerm(
                    selector=lambda ei: ei["edge_id"] == 4,
                    coefficient=-1.0,
                    region_binding="none",
                    time_binding=False,
                    aggregate="single",
                ),
            ],
            clip_lower=None,
        )
        imp = FormulaImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[1] == pytest.approx(70.0)

    def test_missing_zero_counts_nan_source_as_zero(self, acc):
        """missing="zero": a NaN subtract term contributes 0 instead of blocking
        the fill (CONTEXT.md "missing counts as 0"); the default "block" on the
        same shape still aborts."""
        def _rel(missing):
            return FormulaRelation(
                description="balance",
                target=lambda ei: ei["edge_id"] == 1,
                terms=[
                    SourceTerm(selector=lambda ei: ei["edge_id"] == 0,
                               coefficient=1.0, region_binding="none",
                               time_binding=False, aggregate="single"),
                    SourceTerm(selector=lambda ei: ei["edge_id"] == 4,
                               coefficient=-1.0, region_binding="none",
                               time_binding=False, aggregate="single",
                               missing=missing),
                ],
                clip_lower=None,
            )
        # subtract source (edge 4) left NaN
        f     = np.full(10, np.nan);  f[0] = 100.0
        sigma = np.full(10, np.nan);  sigma[0] = 5.0
        FormulaImputer(relations=[_rel("zero")]).impute(f, sigma, np.isnan(f), 0, acc)
        assert f[1] == pytest.approx(100.0), "NaN subtract term must count as 0"
        # default "block" semantics unchanged
        f     = np.full(10, np.nan);  f[0] = 100.0
        sigma = np.full(10, np.nan);  sigma[0] = 5.0
        FormulaImputer(relations=[_rel("block")]).impute(f, sigma, np.isnan(f), 0, acc)
        assert np.isnan(f[1]), "default must still block on a NaN source"

    def test_clip_lower_zero(self, acc):
        """Negative result is clipped to clip_lower=0."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]  = 10.0;  sigma[0] = 1.0
        f[4]  = 50.0;  sigma[4] = 1.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="clip test",
            target=lambda ei: ei["edge_id"] == 1,
            terms=[
                SourceTerm(lambda ei: ei["edge_id"] == 0,  1.0, "none", False, "single"),
                SourceTerm(lambda ei: ei["edge_id"] == 4, -1.0, "none", False, "single"),
            ],
            clip_lower=0.0,
        )
        imp = FormulaImputer(relations=[rel])
        entries = imp.impute(f, sigma, missing, 0, acc)
        assert f[1] == pytest.approx(0.0)
        assert entries[0]["method"] == "formula_clipped"

    def test_no_clip_when_positive(self, acc):
        """Positive result not clipped; method is 'formula' not 'formula_clipped'."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]  = 80.0;  sigma[0] = 0.0
        f[4]  = 30.0;  sigma[4] = 0.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="no clip",
            target=lambda ei: ei["edge_id"] == 1,
            terms=[
                SourceTerm(lambda ei: ei["edge_id"] == 0,  1.0, "none", False, "single"),
                SourceTerm(lambda ei: ei["edge_id"] == 4, -1.0, "none", False, "single"),
            ],
            clip_lower=0.0,
        )
        imp = FormulaImputer(relations=[rel])
        entries = imp.impute(f, sigma, missing, 0, acc)
        assert entries[0]["method"] == "formula"
        assert f[1] == pytest.approx(50.0)

    def test_region_binding_from_matches_target_from(self, acc):
        """
        Supply edges 0 (region A) and 1 (region B) both known.
        Target edges 4 (region A) and 5 (region B).
        region_binding='to_matches_target_from' selects supply edge whose
        to-region matches target's from-region.
        """
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 100.0;  sigma[0] = 10.0   # Mining_A→Ore_A
        f[1] = 200.0;  sigma[1] = 20.0   # Mining_B→Ore_B
        missing = np.isnan(f)

        # Target: use edges 4 (Ore_A→Mining_A) and 5 (Ore_B→Mining_B)
        # Source: supply edges 0 and 1 — bound by "to_region == target's from_region"
        # Edge 4's from_region = A; edge 0's to_region = A → should use edge 0
        rel = FormulaRelation(
            description="regional",
            target=lambda ei: (ei["sector_from"] == "Ore") & (ei["valuation"] == "Mass")
                              & (ei["time"] == "2020"),
            terms=[
                SourceTerm(
                    selector=lambda ei: (ei["sector_from"] == "Mining")
                                       & (ei["valuation"] == "Mass")
                                       & (ei["time"] == "2020"),
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

        # Edge 4 (Ore_A→Mining_A): from_region=A → source should be edge 0 (to_region=A)
        assert f[4] == pytest.approx(100.0)
        # Edge 5 (Ore_B→Mining_B): from_region=B → source should be edge 1 (to_region=B)
        assert f[5] == pytest.approx(200.0)

    def test_missing_source_skips_target(self, acc):
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        missing = np.isnan(f)  # everything NaN

        rel = FormulaRelation(
            description="skip",
            target=lambda ei: ei["edge_id"] == 0,
            terms=[SourceTerm(lambda ei: ei["edge_id"] == 4, 1.0, "none", False, "single")],
            clip_lower=0.0,
        )
        imp = FormulaImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[0])

    def test_aggregate_sum_over_multiple_sources(self, acc):
        """aggregate='sum' sums all matching edges (no 'single' constraint)."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]  = 10.0;  sigma[0] = 0.0   # source 1
        f[1]  = 20.0;  sigma[1] = 0.0   # source 2
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="sum",
            target=lambda ei: ei["edge_id"] == 4,
            terms=[
                SourceTerm(
                    selector=lambda ei: ei["edge_id"].isin([0, 1]),
                    coefficient=1.0,
                    region_binding="none",
                    time_binding=False,
                    aggregate="sum",
                ),
            ],
            clip_lower=None,
        )
        imp = FormulaImputer(relations=[rel])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[4] == pytest.approx(30.0)

    def test_param_ref_identity(self, acc):
        """ParamRef with transform='identity' uses value_nominal directly."""
        f     = np.full(10, np.nan);  f[0]     = 10.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="param_ref",
            target=lambda ei: ei["edge_id"] == 4,
            terms=[
                SourceTerm(
                    selector=lambda ei: ei["edge_id"] == 0,
                    coefficient=ParamRef("eff_mine", "identity"),
                    region_binding="none",
                    time_binding=False,
                    aggregate="single",
                )
            ],
            clip_lower=None,
        )
        imp = FormulaImputer(relations=[rel], params={"eff_mine": 0.9})
        imp.impute(f, sigma, missing, 0, acc)
        assert f[4] == pytest.approx(9.0)

    def test_param_ref_inverse(self, acc):
        """ParamRef with transform='inverse' uses 1/value_nominal."""
        f     = np.full(10, np.nan);  f[0]     = 10.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.0
        missing = np.isnan(f)

        rel = FormulaRelation(
            description="inverse",
            target=lambda ei: ei["edge_id"] == 4,
            terms=[
                SourceTerm(
                    selector=lambda ei: ei["edge_id"] == 0,
                    coefficient=ParamRef("eff_mine", "inverse"),
                    region_binding="none",
                    time_binding=False,
                    aggregate="single",
                )
            ],
            clip_lower=None,
        )
        imp = FormulaImputer(relations=[rel], params={"eff_mine": 0.5})
        imp.impute(f, sigma, missing, 0, acc)
        assert f[4] == pytest.approx(20.0)   # 10 / 0.5


# =============================================================================
# 6. TimeImputer
# =============================================================================

class TestTimeImputer:
    """
    Edge series for time tests:
        Edges 0, 6, 8 form one series (Mining_A→Ore_A, Mass) at t=2020,2021,2022
        Edges 1, 7, 9 form one series (Mining_B→Ore_B, Mass) at t=2020,2021,2022
    """

    def test_linear_interpolation_between_two_known(self, acc):
        """Known at t=2020 (edge 0) and t=2022 (edge 8); fill t=2021 (edge 6)."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0;  sigma[0] = 2.0   # t=2020
        f[8] = 20.0;  sigma[8] = 4.0   # t=2022
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear")  # default p=1.3
        entries = imp.impute(f, sigma, missing, 0, acc)

        # λ = 0.5 → f_6 = 15.0
        assert f[6] == pytest.approx(15.0)
        # σ_6 = ( |0.5·2|^p + |0.5·4|^p )^(1/p)
        p = 1.3
        expected_sigma = (abs(0.5*2)**p + abs(0.5*4)**p) ** (1/p)
        assert sigma[6] == pytest.approx(expected_sigma)
        assert any(e["edge_id"] == 6 and e["method"] == "time_linear" for e in entries)

    def test_linear_interpolation_p2_recovers_quadrature(self, acc):
        """At p=2, σ_linear matches the legacy quadrature value."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0;  sigma[0] = 2.0
        f[8] = 20.0;  sigma[8] = 4.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear", p=2.0)
        imp.impute(f, sigma, missing, 0, acc)
        # σ² = 0.25·4 + 0.25·16 = 5 → σ = sqrt(5)
        assert sigma[6] == pytest.approx(np.sqrt(5.0))

    def test_locf(self, acc):
        """LOCF: fill t=2021 from t=2020."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 42.0;  sigma[0] = 3.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="locf")
        imp.impute(f, sigma, missing, 0, acc)
        assert f[6] == pytest.approx(42.0)

    def test_nocb(self, acc):
        """NOCB: fill t=2021 from t=2022."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[8] = 99.0;  sigma[8] = 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="nocb")
        imp.impute(f, sigma, missing, 0, acc)
        assert f[6] == pytest.approx(99.0)

    def test_gap_inflation_applied(self, acc):
        """LOCF with gap_inflation=2.0: σ doubles per time step."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0;  sigma[0] = 1.0   # t=2020
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="locf", gap_inflation=2.0)
        imp.impute(f, sigma, missing, 0, acc)
        # gap=1 step → σ_6 = 1 * 2^1 = 2
        assert sigma[6] == pytest.approx(2.0)
        # gap=2 steps → σ_8 = 1 * 2^2 = 4
        assert sigma[8] == pytest.approx(4.0)

    def test_window_limits_lookback(self, acc):
        """
        Window=1 looks at most 1 step.  With LOCF, filling cascades:
        t=2021 is filled from t=2020 (gap=1), then t=2022 is filled
        from the newly-filled t=2021 (gap=1).  Both should be filled.
        """
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 5.0;  sigma[0] = 1.0   # t=2020 only
        missing = np.isnan(f)

        imp = TimeImputer(window=1, method="locf")
        imp.impute(f, sigma, missing, 0, acc)
        # t=2021 (edge 6): gap=1 from t=2020 → filled
        assert f[6] == pytest.approx(5.0)
        # t=2022 (edge 8): cascades from newly-filled t=2021 (gap=1) → also filled
        assert f[8] == pytest.approx(5.0)

    def test_window_zero_fills_nothing(self, acc):
        """Window=0 cannot look at any adjacent step; no fills."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 5.0;  sigma[0] = 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=0, method="locf")
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[6])
        assert np.isnan(f[8])

    def test_mean_method(self, acc):
        """mean: average of known observations within window."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0;  sigma[0] = 2.0   # t=2020
        f[8] = 30.0;  sigma[8] = 4.0   # t=2022
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="mean")
        imp.impute(f, sigma, missing, 0, acc)
        # mean of 10 and 30 = 20
        assert f[6] == pytest.approx(20.0)

    def test_already_known_not_overwritten(self, acc):
        """Known edges must not be overwritten."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0;  sigma[0] = 1.0
        f[6] = 99.0;  sigma[6] = 1.0   # already known
        f[8] = 20.0;  sigma[8] = 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear")
        imp.impute(f, sigma, missing, 0, acc)
        assert f[6] == pytest.approx(99.0)   # unchanged

    def test_independent_series_do_not_interact(self, acc):
        """Region-A series and Region-B series are independent."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0;  sigma[0] = 1.0   # Mining_A t=2020
        f[8] = 30.0;  sigma[8] = 1.0   # Mining_A t=2022
        # Mining_B has no observations → should remain NaN
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear")
        imp.impute(f, sigma, missing, 0, acc)

        # Mining_A t=2021 filled
        assert not np.isnan(f[6])
        # Mining_B t=2020, 2021, 2022 remain NaN
        assert np.isnan(f[1]) and np.isnan(f[7]) and np.isnan(f[9])

    def test_log_entries_for_time_linear(self, acc):
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 5.0;  sigma[0] = 1.0
        f[8] = 15.0; sigma[8] = 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear")
        entries = imp.impute(f, sigma, missing, 0, acc)
        methods = {e["method"] for e in entries}
        assert "time_linear" in methods


# sigma_from_row unit tests live in tests/test_param_relation.py.


# =============================================================================
# 8. ImputationEngine — bounds enforcement
# =============================================================================

class TestImputationEngineBounds:

    def test_engine_clips_fill_exceeding_upper_bound(self, acc):
        """Engine clips a fill that exceeds acc.bounds.u after the imputer writes it."""
        acc.bounds.u[2] = 3.0   # tight upper bound on edge 2
        # ParametricImputer fills f[2] = 0.5*10 = 5.0 (CSV ratio bounds aren't applied here);
        # the engine then clips to acc.bounds.u[2] = 3.0 post-step.
        f = [10.0] + [np.nan] * 9
        ar = _agg_result(f, [1.0] + [np.nan] * 9)
        imputer = ParametricImputer(_csv_pairs=[(0, 2, 0.5, 0.0, 0.0, "p", "d")])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imputer)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = engine.run(ar)
        assert result.f_agg[2] == pytest.approx(3.0)

    def test_engine_warns_on_clip_with_key_message_fields(self, acc):
        """Engine emits UserWarning containing 'clipped' and the bound value."""
        acc.bounds.u[2] = 3.0
        f = [10.0] + [np.nan] * 9
        ar = _agg_result(f, [1.0] + [np.nan] * 9)
        imputer = ParametricImputer(_csv_pairs=[(0, 2, 0.5, 0.0, 0.0, "p", "d")])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imputer)
        with pytest.warns(UserWarning, match=r"clipped") as rec:
            engine.run(ar)
        msg = str(rec[0].message)
        assert "3" in msg   # the bound value appears in the message

    def test_engine_no_warn_when_fill_within_bounds(self, acc):
        """No UserWarning when the imputed value is within acc.bounds."""
        # default bounds: l=0, u=1e9; f[2] = 0.5*10 = 5.0 — well within bounds
        f = [10.0] + [np.nan] * 9
        ar = _agg_result(f, [1.0] + [np.nan] * 9)
        imputer = ParametricImputer(_csv_pairs=[(0, 2, 0.5, 0.0, 0.0, "p", "d")])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imputer)
        import warnings
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            engine.run(ar)
        user_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


# =============================================================================
# 9. FormulaImputer — acc.bounds clip (new behaviour)
# =============================================================================

class TestFormulaImputerAccBounds:

    def _make_balance_rel(self, clip_lower=0.0):
        """f_target = f[0] - f[4]; target = edge 1."""
        return FormulaRelation(
            description="balance",
            target=lambda ei: ei["edge_id"] == 1,
            terms=[
                SourceTerm(lambda ei: ei["edge_id"] == 0,  1.0, "none", False, "single"),
                SourceTerm(lambda ei: ei["edge_id"] == 4, -1.0, "none", False, "single"),
            ],
            clip_lower=clip_lower,
        )

    def test_formula_clips_to_zero_when_acc_bounds_lower_is_zero(self, acc):
        """Default bounds.l=0: formula result -40 clips to 0 → method='formula_clipped'."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]  = 10.0;  sigma[0] = 0.0
        f[4]  = 50.0;  sigma[4] = 0.0
        missing = np.isnan(f)
        imp = FormulaImputer(relations=[self._make_balance_rel(clip_lower=0.0)])
        entries = imp.impute(f, sigma, missing, 0, acc)
        assert f[1] == pytest.approx(0.0)
        assert entries[0]["method"] == "formula_clipped"

    def test_formula_allows_negative_when_acc_bounds_permit(self, acc):
        """
        acc.bounds.l[1] = -1e6 overrides rel.clip_lower=0 — proves FormulaImputer
        now uses acc.bounds as the authoritative clip source.
        """
        acc.bounds.l[1] = -1e6
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]  = 10.0;  sigma[0] = 0.0
        f[4]  = 50.0;  sigma[4] = 0.0
        missing = np.isnan(f)
        # rel.clip_lower=0.0 is set — if it were still used, f[1] would be 0
        imp = FormulaImputer(relations=[self._make_balance_rel(clip_lower=0.0)])
        entries = imp.impute(f, sigma, missing, 0, acc)
        assert f[1] == pytest.approx(-40.0)          # NOT clipped to 0
        assert entries[0]["method"] == "formula"      # NOT "formula_clipped"


# =============================================================================
# 10. TimeImputer — interpolate_only=True
# =============================================================================

class TestTimeImputerInterpolateOnly:
    """
    Series: edges 0 (t=2020), 6 (t=2021), 8 (t=2022) — Mining_A→Ore_A, Mass.
    """

    def test_interpolate_only_fills_interior_gap(self, acc):
        """Both before and after anchors exist → interior gap is filled."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]  = 10.0;  sigma[0] = 2.0   # t=2020
        f[8]  = 20.0;  sigma[8] = 2.0   # t=2022
        missing = np.isnan(f)
        imp = TimeImputer(window=5, method="linear", interpolate_only=True)
        imp.impute(f, sigma, missing, 0, acc)
        assert f[6] == pytest.approx(15.0)   # interpolated at λ=0.5

    def test_interpolate_only_skips_leading_gap(self, acc):
        """Only a future anchor exists — no 'before' anchor → leading gaps stay NaN."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[8]  = 20.0;  sigma[8] = 1.0   # only t=2022 known
        missing = np.isnan(f)
        imp = TimeImputer(window=5, method="linear", interpolate_only=True)
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[0])   # t=2020 — no before anchor
        assert np.isnan(f[6])   # t=2021 — no before anchor

    def test_interpolate_only_skips_trailing_gap(self, acc):
        """Only a past anchor exists — no 'after' anchor → trailing gaps stay NaN."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]  = 10.0;  sigma[0] = 1.0   # only t=2020 known
        missing = np.isnan(f)
        imp = TimeImputer(window=5, method="linear", interpolate_only=True)
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[6])   # t=2021 — no after anchor
        assert np.isnan(f[8])   # t=2022 — no after anchor


# =============================================================================
# 10b. TimeImputer scoping (edge / feed / time / valuation)
# =============================================================================

class TestTimeImputerScoping:
    """The new scope_* parameters restrict which edges/years get filled while
    still letting historical observations anchor LOCF/NOCB extrapolation."""

    def test_scope_edges_mask_restricts_fills(self, acc):
        """Boolean mask drops untouched-edge series entirely."""
        # Two parallel series: 0/6/8 (Mining_A→Ore_A, Mass) and 1/7/9 (Mining_B→Ore_B, Mass).
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        # Anchor both series at t=2020 and t=2022 so linear interpolation fires.
        f[0] = 10.0; sigma[0] = 1.0
        f[8] = 20.0; sigma[8] = 1.0
        f[1] = 30.0; sigma[1] = 1.0
        f[9] = 50.0; sigma[9] = 1.0
        missing = np.isnan(f)

        scope = np.zeros(10, dtype=bool)
        scope[[0, 6, 8]] = True  # only Mining_A→Ore_A series

        imp = TimeImputer(window=5, method="linear", interpolate_only=True,
                          scope_edges=scope)
        imp.impute(f, sigma, missing, 0, acc)

        assert f[6] == pytest.approx(15.0)        # in scope — interpolated
        assert np.isnan(f[7])                     # out of scope — untouched

    def test_scope_feed_id_resolves_via_attach(self, acc):
        """scope_feed_id resolves to a mask via attach_agg_result(agg_result)."""
        # Two feeds. Feed "crude_ore" only covers edges 0 and 8.
        F     = np.full((2, 10), np.nan)
        Sigma = np.full((2, 10), np.nan)
        F[0, [0, 8]] = [10.0, 20.0]; Sigma[0, [0, 8]] = 1.0   # crude_ore
        F[1, [1, 9]] = [30.0, 50.0]; Sigma[1, [1, 9]] = 1.0   # other
        f_agg     = np.nanmean(F, axis=0)
        sigma_agg = np.nanmean(Sigma, axis=0)
        coverage  = np.isfinite(f_agg)

        ar = AggregationResult(
            f_agg     = f_agg,
            sigma_agg = sigma_agg,
            Omega     = np.where(np.isfinite(F), 1.0, 0.0),
            F         = F,
            Sigma     = Sigma,
            feed_ids  = ["crude_ore", "other"],
            coverage  = coverage,
        )

        f_work     = f_agg.copy()
        sigma_work = sigma_agg.copy()
        missing    = np.isnan(f_work)

        imp = TimeImputer(window=5, method="linear", interpolate_only=True,
                          scope_feed_id="crude_ore")
        imp.attach_agg_result(ar)
        imp.impute(f_work, sigma_work, missing, 0, acc)

        # Mining_A→Ore_A (in crude_ore) — t=2021 interpolated.
        assert f_work[6] == pytest.approx(15.0)
        # Mining_B→Ore_B (NOT in crude_ore) — t=2021 untouched.
        assert np.isnan(f_work[7])

    def test_scope_times_restricts_target_years(self, acc):
        """scope_times filters target years but leaves history available as anchor."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0; sigma[0] = 1.0   # t=2020 anchor
        f[8] = 20.0; sigma[8] = 1.0   # t=2022 anchor
        missing = np.isnan(f)

        # Only fill 2022 targets (none missing for the 0/6/8 series since 8 is set).
        # Try scope_times=["2021"] — should fill only edge 6.
        imp = TimeImputer(window=5, method="linear", interpolate_only=True,
                          scope_times=["2021"])
        imp.impute(f, sigma, missing, 0, acc)

        assert f[6] == pytest.approx(15.0)
        assert not np.isnan(f[0])    # untouched, was anchor
        # Edge 1 and 9 are missing entries in the other series — no anchor there
        # so even though they'd be candidates, no fill possible.

    def test_scope_combined_with_extrapolation(self, acc):
        """Historical anchor at 2020 → LOCF extrapolation to 2021 + 2022
        when scope_times restricts targets but anchor stays visible."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0; sigma[0] = 2.0   # t=2020 anchor only
        missing = np.isnan(f)

        imp = TimeImputer(window=5, method="linear", interpolate_only=False,
                          gap_inflation=1.5,
                          scope_times=["2021", "2022"])
        entries = imp.impute(f, sigma, missing, 0, acc)

        # Both 2021 and 2022 filled by single-sided "linear" path (LOCF-like).
        assert f[6] == pytest.approx(10.0)
        assert f[8] == pytest.approx(10.0)
        # σ inflated by gap_inflation ^ gap_years.
        assert sigma[6] == pytest.approx(2.0 * 1.5)
        assert sigma[8] == pytest.approx(2.0 * 1.5 ** 2)
        methods = {e["method"] for e in entries if e["edge_id"] in (6, 8)}
        assert methods == {"time_linear"}

    def test_scope_edges_and_feed_id_mutually_exclusive(self):
        """Passing both scope_edges and scope_feed_id raises at construction."""
        with pytest.raises(ValueError, match="at most one"):
            TimeImputer(scope_edges=np.zeros(10, dtype=bool),
                        scope_feed_id="x")

    def test_scope_feed_id_not_in_agg_raises(self, acc):
        """Resolving an unknown feed name raises a clear error."""
        ar = _agg_result([1.0] + [np.nan]*9)
        imp = TimeImputer(scope_feed_id="nope")
        with pytest.raises(ValueError, match="not in"):
            imp.attach_agg_result(ar)


# =============================================================================
# 11. ParametricImputer — specificity dedup
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
        """
        Two rows targeting Mining→Ore (Mass ↔ CC):
          Row 1: a_region_from=*  → coeff=0.5  (matches A and B)
          Row 2: a_region_from=A  → coeff=0.9  (matches A only, more specific)
        For region-A pair (edges 0 ↔ 2): coeff=0.9 must win.
        For region-B pair (edges 1 ↔ 3): coeff=0.5 applies (only match).
        """
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

        # --- region-A pair: specific row (coeff=0.9) must win ---
        f_a     = np.full(10, np.nan);  f_a[0]     = 10.0
        sig_a   = np.full(10, np.nan);  sig_a[0]   = 0.0
        imp.impute(f_a, sig_a, np.isnan(f_a), 0, acc)
        assert f_a[2] == pytest.approx(9.0), (
            f"Expected 9.0 (coeff=0.9 from specific row), got {f_a[2]}"
        )

        # --- region-B pair: only wildcard row applies (coeff=0.5) ---
        f_b     = np.full(10, np.nan);  f_b[1]     = 10.0
        sig_b   = np.full(10, np.nan);  sig_b[1]   = 0.0
        imp.impute(f_b, sig_b, np.isnan(f_b), 0, acc)
        assert f_b[3] == pytest.approx(5.0), (
            f"Expected 5.0 (coeff=0.5 from wildcard row), got {f_b[3]}"
        )


# =============================================================================
# 12. Pipeline chaining
# =============================================================================

class TestPipelineChaining:

    def test_pipeline_chaining_no_overwrite(self, acc):
        """
        Level A: a later imputer must not overwrite an edge already filled by an
        earlier step, even if its pair would produce a different value.
        """
        f  = [10.0] + [np.nan] * 9
        ar = _agg_result(f, [1.0] + [np.nan] * 9)
        # imp1 fills edge 1: f[1] = 2.0 * 10 = 20
        imp1 = ParametricImputer(_csv_pairs=[(0, 1, 2.0, 0.0, 0.0, "p1", "")])
        # imp2 would fill edge 1 with 99 * 10 = 990 if edge 1 were still missing
        imp2 = ParametricImputer(_csv_pairs=[(0, 1, 99.0, 0.0, 0.0, "p2", "")])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imp1)
        engine.register(imp2)
        result = engine.run(ar)
        assert result.f_agg[1] == pytest.approx(20.0)   # imp1's fill preserved

    def test_pipeline_chaining_forward_use(self, acc):
        """
        Level B: imputer 2 reads a value written by imputer 1 and derives a
        further edge from it — proves the filled value is visible downstream.
        """
        f  = [10.0] + [np.nan] * 9
        ar = _agg_result(f, [1.0] + [np.nan] * 9)
        # imp1: f[1] = 2.0 * f[0] = 20
        imp1 = ParametricImputer(_csv_pairs=[(0, 1, 2.0, 0.0, 0.0, "p1", "")])
        # imp2: f[3] = 3.0 * f[1]; edge 1 was NaN initially, filled by imp1
        imp2 = ParametricImputer(_csv_pairs=[(1, 3, 3.0, 0.0, 0.0, "p2", "")])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imp1)
        engine.register(imp2)
        result = engine.run(ar)
        assert result.f_agg[1] == pytest.approx(20.0)   # filled by imp1
        assert result.f_agg[3] == pytest.approx(60.0)   # derived from imp1's fill by imp2


# =============================================================================
# FlatPriorImputer
# =============================================================================

class TestFlatPriorImputer:

    def test_supply_uses_supply_cv(self, acc):
        """Edge 0 (supply) → σ = 0.001 · group scale."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[8]  = 100.0;  sigma[8] = 1.0   # Mining_A (region A) → Ore_A, Mass, t=2022
        # group of edge 0 (sector_from=Mining, entity_from=Process, region_from=A, val=Mass)
        # also includes edges 6, 8 → scale = mean(|f|) over finite = 100.0
        missing = np.isnan(f)
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.impute(f, sigma, missing, 0, acc)
        # edge 0 is supply, group has only edge 8 finite → scale=100, σ = 0.001·100 = 0.1
        assert f[0] == 0.0
        assert sigma[0] == pytest.approx(0.001 * 100.0)

    def test_use_uses_use_cv(self, acc):
        """Edge 4 (Ore_A → Mining_A, use): narrow group empty → level-2 cascade kicks in."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[5]  = 200.0;  sigma[5] = 1.0
        # Level 1 (Ore, Flow, A, Mass) has only edge 4 (NaN) → empty.
        # Level 2 (Ore, Mass) has edges 4 and 5 → edge 5 finite (=200) → mean = 200.
        missing = np.isnan(f)
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.impute(f, sigma, missing, 0, acc)
        assert f[4] == 0.0
        assert sigma[4] == pytest.approx(0.3 * 200.0)

    def test_default_cv_for_unmasked_edges(self, acc):
        """If a mask is omitted from cv_by_mask, edges in that mask fall through to default_cv
        (no density modulation), and the 3-level scale cascade still applies."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[8]  = 50.0;  sigma[8] = 1.0
        missing = np.isnan(f)
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001},   # no 'use' rule
            default_cv=0.7,
            acc_struct=acc,
        )
        imp.impute(f, sigma, missing, 0, acc)
        # Edge 4 (use) has no rule → default_cv=0.7. Its narrow + broader (Ore, Mass) groups
        # are both empty (only edge 5 is in (Ore, Mass) and it's NaN), so scale falls all the
        # way to level 3: global median of finite, non-zero |f| = median([50]) = 50.
        assert sigma[4] == pytest.approx(0.7 * 50.0)
        # Edge 0 (supply, Mining→Ore Mass, region A) — narrow group (Mining, Process, A, Mass)
        # has edges 0, 6, 8 — edge 8 finite (=50) → scale = 50 → σ = 0.001·50.
        assert sigma[0] == pytest.approx(0.001 * 50.0)

    def test_first_match_wins_with_overlapping_masks(self, acc):
        """Insertion order in cv_by_mask defines priority."""
        # Force overlap: declare both supply and use both pointing to same mask via
        # custom ordering. With this fixture, edge 0 only matches supply (no overlap).
        # Validate via insertion order: same edge gets first listed mask's CV.
        # We synthesize overlap by setting use_mask[0]=True artificially (test-only).
        acc.masks["use"] = acc.masks["use"].copy()
        acc.masks["use"][0] = True   # edge 0 now matches BOTH supply and use

        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[6]  = 80.0;  sigma[6] = 1.0
        missing = np.isnan(f)

        # supply listed first → edge 0 gets supply CV
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.impute(f, sigma, missing, 0, acc)
        # edge 0's group includes edge 6 (finite) → scale=80, σ = 0.001·80 = 0.08
        assert sigma[0] == pytest.approx(0.001 * 80.0)

    def test_all_groups_empty_falls_back_to_unit_scale(self, acc):
        """When every level of the cascade is empty (no finite, non-zero |f| anywhere)
        the global-median fallback degrades to scale = 1.0, so σ = CV per edge.
        Previously this case produced σ = 0 (the pin-band bug from the May-13 copper REP)."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        # Nothing finite anywhere → all three cascade levels return 0 → global_scale = 1.0
        missing = np.isnan(f)
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.impute(f, sigma, missing, 0, acc)
        # f is still 0 for every filled edge.
        assert np.all(f == 0.0)
        # σ = CV(mask) · 1.0 per edge; supply edges get 0.001, use edges get 0.3.
        # (legacy single-CV form, no density modulation.)
        expected = np.where(acc.masks["supply"], 0.001, 0.3)
        assert np.allclose(sigma, expected)

    def test_log_entries_returned(self, acc):
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[8]  = 100.0;  sigma[8] = 1.0
        missing = np.isnan(f)
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        entries = imp.impute(f, sigma, missing, 5, acc)
        # 9 missing edges → 9 entries
        assert len(entries) == 9
        for e in entries:
            assert e["method"] == "flat_prior"
            assert e["step"] == 5
            assert e["f_after"] == 0.0

    def test_density_aware_cv_interpolation(self, acc):
        """With (cv_lo, cv_hi) and a coverage array, CV interpolates by per-edge density
        over (sector_from, region_from). Verified on an uncovered region vs a covered one."""
        # All 10 edges; build a coverage array where region-A is fully covered (density=1
        # → cv=cv_lo) and region-B is fully uncovered (density=0 → cv=cv_hi).
        from_region = np.array(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])
        coverage = (from_region == "A")  # True for A, False for B

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
        # Edges with region_from == A (covered) → cv_lo for their mask.
        # Edges with region_from == B (uncovered) → cv_hi for their mask.
        # Edge 4 is use (from=Ore_A) → density(Ore,A)=1 → cv = 0.20.
        # Edge 5 is use (from=Ore_B) → density(Ore,B)=0 → cv = 0.60.
        assert cv[4] == pytest.approx(0.20)
        assert cv[5] == pytest.approx(0.60)
        # Edge 0 is supply (from=Mining_A) → density(Mining,A)=1 → cv = 0.04.
        # Edge 1 is supply (from=Mining_B) → density(Mining,B)=0 → cv = 0.30.
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
        """cv_hi > 1.0 in any mask tuple → ValueError mentioning the mask name."""
        with pytest.raises(ValueError, match="row_use"):
            FlatPriorImputer(
                cv_by_mask={"supply": (0.04, 0.3), "use": (0.2, 0.5),
                            "row_use": (0.3, 1.05)},
                default_cv=0.3,
                acc_struct=acc,
                masks={"row_use": np.zeros(acc.n_edges, dtype=bool)},
            )

    def test_default_cv_above_one_raises(self, acc):
        """default_cv > 1.0 → ValueError."""
        with pytest.raises(ValueError, match="default_cv"):
            FlatPriorImputer(
                cv_by_mask={"supply": (0.04, 0.3)},
                default_cv=1.5,
                acc_struct=acc,
            )

    def test_masks_overlay_resolves_caller_supplied(self, acc):
        """A mask name absent from acc.masks but present in the masks= overlay
        resolves to the caller-supplied boolean array."""
        custom = np.zeros(acc.n_edges, dtype=bool)
        custom[0] = True
        imp = FlatPriorImputer(
            cv_by_mask={"my_overlay": (0.1, 0.2), "supply": (0.04, 0.3)},
            default_cv=0.3,
            acc_struct=acc,
            masks={"my_overlay": custom},
        )
        # First-match-wins: edge 0 is in my_overlay → uses (0.1, 0.2).
        names = [n for n, _ in imp._masks]
        assert names == ["my_overlay", "supply"]


class TestCathodeFdSplitSelectors:
    """The cathode final-demand split FormulaRelations must select SB-entity
    targets: FU/USC/MESC nodes carry entity == "SB", not "Process". With the
    old "Process" predicate the formulas matched zero edges and silently never
    fired (domestic Cathode→FU fell to flat_prior_noisy_zero)."""

    def _load_script(self):
        import importlib.util, pathlib
        repo = pathlib.Path(__file__).resolve().parents[1]
        mod_path = repo / "scripts" / "collect_aggregate_impute.py"
        try:
            spec = importlib.util.spec_from_file_location(
                "collect_aggregate_impute", mod_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except ImportError as e:
            pytest.skip(f"collect_aggregate_impute not importable: {e}")
        return mod

    def _toy_ei(self):
        # one domestic Cathode→FU SB edge + decoys (cross-region / wrong entity)
        return pd.DataFrame([
            dict(sector_from="Cathode", entity_from="Flow", region_from="A",
                 sector_to="FU",   entity_to="SB",   region_to="A", valuation="Mass"),
            dict(sector_from="Cathode", entity_from="Flow", region_from="A",
                 sector_to="FU",   entity_to="SB",   region_to="B", valuation="Mass"),
            dict(sector_from="Cathode", entity_from="Flow", region_from="A",
                 sector_to="USC",  entity_to="SB",   region_to="A", valuation="Mass"),
            dict(sector_from="Cathode", entity_from="Flow", region_from="A",
                 sector_to="MESC", entity_to="SB",   region_to="A", valuation="Mass"),
            dict(sector_from="Cathode", entity_from="Flow", region_from="A",
                 sector_to="MF",   entity_to="Process", region_to="A", valuation="Mass"),
        ])

    def test_fd_split_targets_match_sb_entities(self):
        mod = self._load_script()
        ei = self._toy_ei()
        assert mod.cathode_fu_domestic.target(ei).tolist() == [True, False, False, False, False], \
            "cathode_fu_domestic must match exactly the domestic Cathode→FU SB edge"
        assert mod.cathode_usc_domestic.target(ei).tolist() == [False, False, True, False, False]
        assert mod.cathode_mesc_domestic.target(ei).tolist() == [False, False, False, True, False]

    def test_fd_split_subtract_terms_match_sb_entities(self):
        mod = self._load_script()
        ei = self._toy_ei()
        term = mod._cathode_domestic_subtract("USC")
        assert term.selector(ei).tolist() == [False, False, True, False, False], \
            "_cathode_domestic_subtract must select the domestic SB leg"

    def test_fd_split_missing_policy(self):
        """FU (the free-sink anchor) counts missing USC/MESC/exports as 0 so it
        fires wherever production is known; USC/MESC keep "block" so they stay
        reserved for the ShareImputer zero-mean seed (Part B not preempted)."""
        mod = self._load_script()
        fu_missing = [t.missing for t in mod.cathode_fu_domestic.terms]
        assert fu_missing == ["block", "zero", "zero", "zero"], \
            "FU: supply blocks, exports/USC/MESC count missing as 0"
        for rel in (mod.cathode_usc_domestic, mod.cathode_mesc_domestic):
            assert all(t.missing == "block" for t in rel.terms), \
                "USC/MESC must keep full block semantics"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


# =============================================================================
# ADR 0026 — bilateral trade legs are not time-interpolated
# =============================================================================

class _TradeAcc:
    """Two parallel Mass series over 2020–2022, one domestic, one bilateral.

    Nodes: 0 Concentrate/Flow/A · 1 Smelting/Process/A · 2 Smelting/Process/B

    Edges (edge_id order), all Mass:
        0,1,2  0→1  2020/2021/2022   domestic  (Concentrate.A → Smelting.A)
        3,4,5  0→2  2020/2021/2022   bilateral (Concentrate.A → Smelting.B)
    """

    class _Bounds:
        def __init__(self, n):
            self.l = np.zeros(n, dtype=float)
            self.u = np.full(n, 1e9, dtype=float)

    def __init__(self):
        self.node_index = pd.DataFrame(
            {"sector": ["Concentrate", "Smelting", "Smelting"],
             "entity": ["Flow",        "Process",  "Process"],
             "region": ["A",           "A",        "B"]},
            index=[0, 1, 2],
        )
        self.edge_index = pd.DataFrame({
            "edge_id":   [0, 1, 2, 3, 4, 5],
            "from_id":   [0, 0, 0, 0, 0, 0],
            "to_id":     [1, 1, 1, 2, 2, 2],
            "time":      ["2020", "2021", "2022", "2020", "2021", "2022"],
            "valuation": ["Mass"] * 6,
        })
        self.bounds = _TradeAcc._Bounds(6)
        self.n_edges = 6
        self.masks = {}


def _same_region_scope(acc):
    """The mask the Stage-2 driver builds for TIME_SCOPE_SAME_REGION."""
    from pysuto.param_relation import expand_edge_index
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    return (ei["region_from"].to_numpy() == ei["region_to"].to_numpy())


class TestTimeImputerSameRegionScope:
    """ADR 0026: BACI is a complete bilateral matrix, so a route absent in a
    year is a recorded zero, not a gap. The interpolator must not invent it."""

    def test_bilateral_leg_is_not_interpolated_while_domestic_one_is(self):
        acc = _TradeAcc()
        f     = np.full(6, np.nan)
        sigma = np.full(6, np.nan)
        # Both series observed in 2020 and 2022, missing in 2021.
        f[0], sigma[0] = 10.0, 1.0
        f[2], sigma[2] = 20.0, 1.0
        f[3], sigma[3] = 30.0, 1.0
        f[5], sigma[5] = 50.0, 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=3, method="linear", min_obs=2,
                          interpolate_only=True,
                          scope_edges=_same_region_scope(acc),
                          id="time_interp")
        imp.impute(f, sigma, missing, 0, acc)

        assert f[1] == pytest.approx(15.0), "domestic series still interpolates"
        assert np.isnan(f[4]), "bilateral leg must stay an honest gap (ADR 0026)"

    def test_unscoped_imputer_fills_the_bilateral_leg(self):
        """The pre-ADR-0026 behaviour, pinned so the regression is visible."""
        acc = _TradeAcc()
        f     = np.full(6, np.nan)
        sigma = np.full(6, np.nan)
        f[0], sigma[0] = 10.0, 1.0
        f[2], sigma[2] = 20.0, 1.0
        f[3], sigma[3] = 30.0, 1.0
        f[5], sigma[5] = 50.0, 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=3, method="linear", min_obs=2,
                          interpolate_only=True, id="time_interp")
        imp.impute(f, sigma, missing, 0, acc)

        assert f[1] == pytest.approx(15.0)
        assert f[4] == pytest.approx(40.0)

    def test_scope_excludes_bilateral_legs_as_sources_too(self):
        """A scoped-out edge is dropped from the series entirely, so it can
        neither receive a fill nor anchor one."""
        acc = _TradeAcc()
        scope = _same_region_scope(acc)
        assert scope.tolist() == [True, True, True, False, False, False]

        f     = np.full(6, np.nan)
        sigma = np.full(6, np.nan)
        # Only the bilateral series is observed; nothing may be filled anywhere.
        f[3], sigma[3] = 30.0, 1.0
        f[5], sigma[5] = 50.0, 1.0
        missing = np.isnan(f)

        imp = TimeImputer(window=3, method="linear", min_obs=2,
                          interpolate_only=True, scope_edges=scope,
                          id="time_interp")
        entries = imp.impute(f, sigma, missing, 0, acc)

        assert entries == []
        assert np.isnan(f[4])
