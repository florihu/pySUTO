"""
test_imputation.py
==================
pytest suite for pysuto/impute.py

Covers:
    ImputationLog           — basic operations, as_dataframe, for_edge
    ImputationEngine        — pipeline ordering, early stop, log attachment
    FillImputer             — the params source's pair fills, both directions (#136)
    TimeImputer             — linear, locf, nocb, gap inflation, window
"""
from __future__ import annotations

import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pysuto.acc import ValuationRoles
from pysuto.agg import AggregationResult
from pysuto.impute import (
    ImputationLog,
    ImputationEngine,
    BaseImputer,
    FillImputer,
    TimeImputer,
    FlatPriorImputer,
    PriorLedger,
    ScopeError,
    run_imputer,
)
from pysuto.impute.base import _reg_sensitivity
from pysuto.relation.ratio_relation import ratio_relation
from pysuto.relation.sources import Fill, ParamSource
import copper_case  # noqa: F401  (puts scripts/ on sys.path, #189)

# Lp combiner unit tests live in tests/test_lp.py.
# ParamRow.sigma unit tests live in tests/test_relation_params.py.


# =============================================================================
# Regularized sensitivity s(α) — used in the fill's backward σ propagation
# =============================================================================

class TestRegSensitivity:
    def test_saturates_to_one_for_large_alpha(self):
        # |α| >> α_min → s(α) → 1
        assert _reg_sensitivity(100.0, q=3.0, alpha_min=0.05) == pytest.approx(1.0, rel=1e-6)

    def test_linear_damping_for_small_alpha(self):
        # |α| << α_min → s(α) ≈ |α|/α_min
        a, am, q = 1e-4, 0.05, 3.0
        # the asymptote of s(a) for a << alpha_min, not an identity: 1e-3 relative
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
        self.valuation_roles = ValuationRoles(content="CC", gross="Mass")
        self.boundary_roles = {}                          # no SB node
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


def _fill(pairs, id="", q=3.0, alpha_min=0.05):
    """A both-direction parametric :class:`FillImputer` over ``(a_id, b_id, coeff,
    sigma_param, epsilon, param_id, description)`` pairs, as the params source builds them."""
    rels = [ratio_relation([b], [a], r, variant="pair", specs=(), time="2020",
                           valuation="Mass", sigma=sig / abs(r) if r else 0.0, source=pid,
                           meta={"sigma_param": sig, "epsilon": eps, "description": desc})
            for a, b, r, sig, eps, pid, desc in pairs]
    return FillImputer(Fill(ParamSource(rels), "both", id=id), "parametric", q=q,
                       alpha_min=alpha_min)


def _from_csv(path, acc):
    """The both-direction parametric fill of the params file at ``path``."""
    return FillImputer(Fill(ParamSource.from_params(str(path), acc), "both"), "parametric")


def _pairs(imp):
    """``(a_id, b_id, coeff, sigma_param)`` of a fill's relations."""
    return [(int(r.edges[1]), int(r.edges[0]), float(-r.coefficients[1]),
             r.meta["sigma_param"]) for r in imp.fill.source.relations]


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
            "description","param_id","relaxed","assumed_absent",
        }


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
        # Edge 0 known (f=10), edge 2 missing — the fill fills 2 from 0
        f = [10.0, np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan]
        ar = _agg_result(f, [1.0, np.nan, np.nan, np.nan, np.nan, np.nan,
                              np.nan, np.nan, np.nan, np.nan])

        # CSV-based pair: edge 0 ↔ edge 2, coeff=0.5
        imputer = _fill([
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
        imp1 = _fill([(0, 1, 2.0, 0.1, 0.0, "p1", "")])
        # Second imputer: edge 1 → edge 3 (f_3 = 3 * f_1 — edge 1 filled by imp1)
        imp2 = _fill([(1, 3, 3.0, 0.1, 0.0, "p2", "")])

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
            def writer_scope(self, acc):
                return np.ones(acc.n_edges, dtype=bool)
            def impute(self, ledger, step, acc_struct):
                self.called += 1

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
        imp_A = _fill([(0, 2, 2.0, 0.1, 0.0, "A", "")], id="A")
        imp_B = _fill([(2, 3, 3.0, 0.1, 0.0, "B", "")], id="B")
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

    def test_dead_pass_is_logged_as_discarded(self, acc, caplog):
        """The engine says which steps the dead pass would have had and where the
        committed steps end (#203); the records themselves are untouched."""
        ar, imp_A, imp_B = self._two_hop()
        engine = ImputationEngine(acc_struct=acc)
        engine.register_cascade([imp_B, imp_A], max_iter=10)
        with caplog.at_level("INFO", logger="pysuto.impute.engine"):
            result = engine.run(ar)
        msgs = [r.getMessage() for r in caplog.records if "discarded" in r.getMessage()]
        assert msgs == ["pass 3 filled nothing: its steps 4-5 are discarded, the "
                        "committed steps end at 3"]
        assert list(result.imputation_metadata.as_dataframe()["step"]) == [0, 1, 2, 3]

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
        imp_C = _fill([(0, 4, 1.0, 0.1, 0.0, "C", "")], id="C")
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
        assert _fill([]).id == "FillImputer"

    def test_backward_compat_terminal_only(self, acc):
        """No cascade → old single-pass behaviour: steps 0,1 and null pass."""
        ar = _agg_result([10.0] + [np.nan] * 9, [1.0] + [np.nan] * 9)
        imp1 = _fill([(0, 1, 2.0, 0.1, 0.0, "p1", "")], id="one")
        imp2 = _fill([(1, 3, 3.0, 0.1, 0.0, "p2", "")], id="two")
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
# 3. FillImputer — bidirectional pairs
# =============================================================================

class TestFillImputerPairs:

    def test_fill_b_from_a(self, acc):
        """A known, B missing (forward) → f_b = coeff × f_a; σ uses bare coeff."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]  = 100.0;  sigma[0] = 5.0
        imp = _fill([(0, 2, 0.3, 0.0, 0.0, "p", "d")])
        run_imputer(imp, f, sigma, 0, acc)
        assert f[2] == pytest.approx(30.0)
        # Forward branch uses bare coeff: σ_b = |coeff| · σ_a
        assert sigma[2] == pytest.approx(0.3 * 5.0)

    def test_fill_a_from_b_inverse(self, acc):
        """B known, A missing (backward) → f_a = f_b / coeff; σ uses s(1/coeff)."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[2]  = 30.0;  sigma[2] = 1.5
        imp = _fill([(0, 2, 0.3, 0.0, 0.0, "p", "d")])
        run_imputer(imp, f, sigma, 0, acc)
        assert f[0] == pytest.approx(100.0)
        # Backward branch is regularized: σ_a = s(1/coeff) · σ_b
        # 1/0.3 ≈ 3.33; with α_min=0.05, q=3 this saturates close to 1
        expected = _reg_sensitivity(1.0 / 0.3, imp.q, imp.alpha_min) * 1.5
        assert sigma[0] == pytest.approx(expected)

    def test_forward_uses_bare_alpha_small_coeff(self, acc):
        """Forward branch with very small α: σ propagation is NOT regularized."""
        f     = np.full(10, np.nan);  f[0]     = 1.0
        sigma = np.full(10, np.nan);  sigma[0] = 1.0
        imp = _fill([(0, 2, 1e-3, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        run_imputer(imp, f, sigma, 0, acc)
        assert f[2] == pytest.approx(1e-3)
        # Forward uses bare α: σ_b = α · σ_a = 1e-3 · 1 = 1e-3
        # (regularized would give s(α) · σ_a ≈ (α/α_min) · σ_a = 0.02)
        assert sigma[2] == pytest.approx(1e-3, rel=1e-6)

    def test_forward_uses_bare_alpha_large_coeff(self, acc):
        """Forward branch with large α: σ amplifies linearly (no cap)."""
        f     = np.full(10, np.nan);  f[0]     = 1.0
        sigma = np.full(10, np.nan);  sigma[0] = 1.0
        imp = _fill([(0, 2, 100.0, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        run_imputer(imp, f, sigma, 0, acc)
        assert f[2] == pytest.approx(100.0)
        # Forward uses bare α: σ_b = 100 · 1 = 100  (not capped)
        assert sigma[2] == pytest.approx(100.0, rel=1e-6)

    def test_both_known_no_fill(self, acc):
        """Both A and B known → no fill, values unchanged."""
        f = np.array([10.0, 20.0] + [np.nan]*8)
        sigma = np.array([1.0, 2.0] + [np.nan]*8)
        imp = _fill([(0, 1, 2.0, 0.0, 0.0, "p", "d")])
        run_imputer(imp, f, sigma, 0, acc)
        assert f[0] == pytest.approx(10.0)
        assert f[1] == pytest.approx(20.0)

    def test_both_missing_no_fill(self, acc):
        f = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        imp = _fill([(0, 2, 0.5, 0.0, 0.0, "p", "d")])
        run_imputer(imp, f, sigma, 0, acc)
        assert np.isnan(f[0]) and np.isnan(f[2])

    def test_sigma_propagation_includes_sigma_param(self, acc):
        """σ_param (ratio units) enters in flow units: |f_b|·σ_r/|r| = |a|·σ_r (#148)."""
        f     = np.full(10, np.nan);  f[0]     = 10.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.0
        imp = _fill([(0, 2, 2.0, 3.0, 0.0, "p", "d")])
        run_imputer(imp, f, sigma, 0, acc)
        # σ_b = |a|·σ_r = 10 · 3 = 30 kt (was σ_r = 3, a ratio unit added to kt)
        assert sigma[2] == pytest.approx(30.0, rel=1e-12)

    @pytest.mark.parametrize("size", [1000.0, 0.1])
    def test_sigma_param_forward_is_analytic_and_scale_free(self, acc, size):
        """Forward: σ_b = |a|·σ_r with the other terms zero, so CV_b = σ_r/|r| at any size."""
        r, sig_r = 0.28, 0.0375
        f     = np.full(10, np.nan);  f[0]     = size
        sigma = np.full(10, np.nan);  sigma[0] = 0.0
        imp = _fill([(0, 2, r, sig_r, 0.0, "p", "d")])
        run_imputer(imp, f, sigma, 0, acc)
        assert sigma[2] == pytest.approx(size * sig_r, rel=1e-12)
        assert sigma[2] / abs(f[2]) == pytest.approx(sig_r / r, rel=1e-12)

    @pytest.mark.parametrize("size", [1000.0, 0.1])
    def test_sigma_param_backward_is_analytic_and_scale_free(self, acc, size):
        """Backward: σ_a = |b|·σ_r/r² with the other terms zero, so CV_a = σ_r/|r|."""
        r, sig_r = 0.28, 0.0375
        f     = np.full(10, np.nan);  f[2]     = size
        sigma = np.full(10, np.nan);  sigma[2] = 0.0
        imp = _fill([(0, 2, r, sig_r, 0.0, "p", "d")])
        run_imputer(imp, f, sigma, 0, acc)
        assert sigma[0] == pytest.approx(size * sig_r / r ** 2, rel=1e-12)
        assert sigma[0] / abs(f[0]) == pytest.approx(sig_r / r, rel=1e-12)

    def test_sigma_param_forward_with_zero_ratio_keeps_the_parameter_term(self, acc):
        """A zero ratio (WR is copperless) fills 0 with σ = |a|·σ_r, not a σ-0 pin."""
        f     = np.full(10, np.nan);  f[0]     = 500.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.0
        imp = _fill([(0, 2, 0.0, 0.001, 0.0, "p", "d")])
        run_imputer(imp, f, sigma, 0, acc)
        assert f[2] == 0.0
        assert sigma[2] == pytest.approx(500.0 * 0.001, rel=1e-12)

    def test_epsilon_added_in_quadrature(self, acc):
        f     = np.full(10, np.nan);  f[0]     = 10.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.0
        imp = _fill([(0, 2, 1.0, 0.0, 0.1, "p", "d")])
        run_imputer(imp, f, sigma, 0, acc)
        # f_b = 10, sigma_b = sqrt(0 + (0.1*10)²) = 1.0
        assert f[2] == pytest.approx(10.0)
        assert sigma[2] == pytest.approx(1.0)

    def test_csv_bounds_not_applied_in_imputer(self, acc):
        """FillImputer ignores per-row CSV bounds (they are *ratio* bounds
        consumed by RatioBoundBuilder, not absolute caps on the target flow)."""
        f     = np.full(10, np.nan);  f[0]     = 100.0
        sigma = np.full(10, np.nan);  sigma[0] = 1.0
        # coeff=2 → natural fill 200; the imputer no longer reads bound_upper.
        imp = _fill([(0, 2, 2.0, 0.0, 0.0, "p", "d")])
        run_imputer(imp, f, sigma, 0, acc)
        assert f[2] == pytest.approx(200.0)

    def test_log_entries_returned(self, acc):
        f     = np.full(10, np.nan);  f[0]     = 5.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.5
        imp = _fill([(0, 2, 1.0, 0.0, 0.0, "pid", "desc")])
        entries = run_imputer(imp, f, sigma, 3, acc)
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
        # coeff = 0.01 → inv = 100 (large); s(100) ≈ 1, so σ_a is capped at σ_b.
        imp = _fill([(0, 2, 0.01, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        run_imputer(imp, f, sigma, 0, acc)
        # f uses bare inverse: f_a = (1/0.01) * 1 = 100
        assert f[0] == pytest.approx(100.0)
        # σ uses s(inv) ≈ 1, so σ_a ≈ σ_b = 1.0  (vs. 100 without regularization)
        assert sigma[0] == pytest.approx(1.0, rel=1e-6)

    def test_backward_alpha_regularization_floor_for_small_inverse(self, acc):
        """Backward branch with very large coeff (1/coeff small): σ floored, not bare α."""
        f     = np.full(10, np.nan);  f[2]     = 1.0
        sigma = np.full(10, np.nan);  sigma[2] = 1.0
        # coeff = 1000 → inv = 1e-3; s(1e-3) ≈ 1e-3 / α_min = 0.02
        imp = _fill([(0, 2, 1000.0, 0.0, 0.0, "p", "d")],
            q=3.0, alpha_min=0.05,
        )
        run_imputer(imp, f, sigma, 0, acc)
        expected = _reg_sensitivity(1e-3, 3.0, 0.05) * 1.0
        assert sigma[0] == pytest.approx(expected, rel=1e-12)
        # the small-alpha asymptote |a|/alpha_min, not an identity: 1e-3 relative
        assert sigma[0] == pytest.approx(1e-3 / 0.05, rel=1e-3)
        assert sigma[0] < 1.0


class TestParamSourceFromCSV:

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

        imp = _from_csv(path, acc)
        # Edges 0 and 1 have to_node = Ore/Flow; each should pair with CC counterpart
        a_ids = {pair[0] for pair in _pairs(imp)}
        # edge 0 (Mass) should pair with edge 2 (CC); edge 1 (Mass) with edge 3 (CC)
        assert 0 in a_ids or 2 in a_ids   # one of the two valuations is A
        assert len(_pairs(imp)) >= 2    # at least 2 pairs (one per region)

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
        imp = _from_csv(path, acc)
        assert _pairs(imp) == []

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
        imp = _from_csv(path, acc)
        # Each pair should have a positive sigma_param
        assert all(pair[3] > 0 for pair in _pairs(imp))

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
        imp = _from_csv(path, acc)
        # Should have pairs for regions A and B (and times 2020, 2021, 2022)
        assert len(_pairs(imp)) >= 2


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

        imp = TimeImputer(window=5, method="linear")  # default p=1.3
        entries = run_imputer(imp, f, sigma, 0, acc)

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

        imp = TimeImputer(window=5, method="linear", p=2.0)
        run_imputer(imp, f, sigma, 0, acc)
        # σ² = 0.25·4 + 0.25·16 = 5 → σ = sqrt(5)
        assert sigma[6] == pytest.approx(np.sqrt(5.0))

    def test_locf(self, acc):
        """LOCF: fill t=2021 from t=2020."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 42.0;  sigma[0] = 3.0

        imp = TimeImputer(window=5, method="locf")
        run_imputer(imp, f, sigma, 0, acc)
        assert f[6] == pytest.approx(42.0)

    def test_nocb(self, acc):
        """NOCB: fill t=2021 from t=2022."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[8] = 99.0;  sigma[8] = 1.0

        imp = TimeImputer(window=5, method="nocb")
        run_imputer(imp, f, sigma, 0, acc)
        assert f[6] == pytest.approx(99.0)

    def test_gap_inflation_applied(self, acc):
        """LOCF with gap_inflation=2.0: σ doubles per time step."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0;  sigma[0] = 1.0   # t=2020

        imp = TimeImputer(window=5, method="locf", gap_inflation=2.0)
        run_imputer(imp, f, sigma, 0, acc)
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

        imp = TimeImputer(window=1, method="locf")
        run_imputer(imp, f, sigma, 0, acc)
        # t=2021 (edge 6): gap=1 from t=2020 → filled
        assert f[6] == pytest.approx(5.0)
        # t=2022 (edge 8): cascades from newly-filled t=2021 (gap=1) → also filled
        assert f[8] == pytest.approx(5.0)

    def test_window_zero_fills_nothing(self, acc):
        """Window=0 cannot look at any adjacent step; no fills."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 5.0;  sigma[0] = 1.0

        imp = TimeImputer(window=0, method="locf")
        run_imputer(imp, f, sigma, 0, acc)
        assert np.isnan(f[6])
        assert np.isnan(f[8])

    def test_mean_method(self, acc):
        """mean: average of known observations within window."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0;  sigma[0] = 2.0   # t=2020
        f[8] = 30.0;  sigma[8] = 4.0   # t=2022

        imp = TimeImputer(window=5, method="mean")
        run_imputer(imp, f, sigma, 0, acc)
        # mean of 10 and 30 = 20
        assert f[6] == pytest.approx(20.0)

    def test_already_known_not_overwritten(self, acc):
        """Known edges must not be overwritten."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0;  sigma[0] = 1.0
        f[6] = 99.0;  sigma[6] = 1.0   # already known
        f[8] = 20.0;  sigma[8] = 1.0

        imp = TimeImputer(window=5, method="linear")
        run_imputer(imp, f, sigma, 0, acc)
        assert f[6] == pytest.approx(99.0)   # unchanged

    def test_independent_series_do_not_interact(self, acc):
        """Region-A series and Region-B series are independent."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 10.0;  sigma[0] = 1.0   # Mining_A t=2020
        f[8] = 30.0;  sigma[8] = 1.0   # Mining_A t=2022
        # Mining_B has no observations → should remain NaN

        imp = TimeImputer(window=5, method="linear")
        run_imputer(imp, f, sigma, 0, acc)

        # Mining_A t=2021 filled
        assert not np.isnan(f[6])
        # Mining_B t=2020, 2021, 2022 remain NaN
        assert np.isnan(f[1]) and np.isnan(f[7]) and np.isnan(f[9])

    def test_log_entries_for_time_linear(self, acc):
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0] = 5.0;  sigma[0] = 1.0
        f[8] = 15.0; sigma[8] = 1.0

        imp = TimeImputer(window=5, method="linear")
        entries = run_imputer(imp, f, sigma, 0, acc)
        methods = {e["method"] for e in entries}
        assert "time_linear" in methods


# ParamRow.sigma unit tests live in tests/test_relation_params.py.


# =============================================================================
# 8. ImputationEngine — bounds enforcement
# =============================================================================

class TestImputationEngineBounds:

    def test_engine_clips_fill_exceeding_upper_bound(self, acc):
        """Engine clips a fill that exceeds acc.bounds.u after the imputer writes it."""
        acc.bounds.u[2] = 3.0   # tight upper bound on edge 2
        # FillImputer fills f[2] = 0.5*10 = 5.0 (CSV ratio bounds aren't applied here);
        # the engine then clips to acc.bounds.u[2] = 3.0 post-step.
        f = [10.0] + [np.nan] * 9
        ar = _agg_result(f, [1.0] + [np.nan] * 9)
        imputer = _fill([(0, 2, 0.5, 0.0, 0.0, "p", "d")])
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
        imputer = _fill([(0, 2, 0.5, 0.0, 0.0, "p", "d")])
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
        imputer = _fill([(0, 2, 0.5, 0.0, 0.0, "p", "d")])
        engine = ImputationEngine(acc_struct=acc)
        engine.register(imputer)
        import warnings
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            engine.run(ar)
        user_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 0


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
        imp = TimeImputer(window=5, method="linear", interpolate_only=True)
        run_imputer(imp, f, sigma, 0, acc)
        assert f[6] == pytest.approx(15.0)   # interpolated at λ=0.5

    def test_interpolate_only_skips_leading_gap(self, acc):
        """Only a future anchor exists — no 'before' anchor → leading gaps stay NaN."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[8]  = 20.0;  sigma[8] = 1.0   # only t=2022 known
        imp = TimeImputer(window=5, method="linear", interpolate_only=True)
        run_imputer(imp, f, sigma, 0, acc)
        assert np.isnan(f[0])   # t=2020 — no before anchor
        assert np.isnan(f[6])   # t=2021 — no before anchor

    def test_interpolate_only_skips_trailing_gap(self, acc):
        """Only a past anchor exists — no 'after' anchor → trailing gaps stay NaN."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]  = 10.0;  sigma[0] = 1.0   # only t=2020 known
        imp = TimeImputer(window=5, method="linear", interpolate_only=True)
        run_imputer(imp, f, sigma, 0, acc)
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

        scope = np.zeros(10, dtype=bool)
        scope[[0, 6, 8]] = True  # only Mining_A→Ore_A series

        imp = TimeImputer(window=5, method="linear", interpolate_only=True,
                          scope_edges=scope)
        run_imputer(imp, f, sigma, 0, acc)

        assert f[6] == pytest.approx(15.0)        # in scope — interpolated
        assert np.isnan(f[7])                     # out of scope — untouched

    def test_scope_feed_id_resolves_via_attach(self, acc):
        """scope_feed_id resolves to a mask via attach(agg_result)."""
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

        imp = TimeImputer(window=5, method="linear", interpolate_only=True,
                          scope_feed_id="crude_ore")
        imp.attach(ar)
        run_imputer(imp, f_work, sigma_work, 0, acc)

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

        # Only fill 2022 targets (none missing for the 0/6/8 series since 8 is set).
        # Try scope_times=["2021"] — should fill only edge 6.
        imp = TimeImputer(window=5, method="linear", interpolate_only=True,
                          scope_times=["2021"])
        run_imputer(imp, f, sigma, 0, acc)

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

        imp = TimeImputer(window=5, method="linear", interpolate_only=False,
                          gap_inflation=1.5,
                          scope_times=["2021", "2022"])
        entries = run_imputer(imp, f, sigma, 0, acc)

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
            imp.attach(ar)


# =============================================================================
# 11. ParamSource — specificity dedup
# =============================================================================

class TestParamSourceSpecificityDedup:

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
        imp = _from_csv(path, acc)

        # --- region-A pair: specific row (coeff=0.9) must win ---
        f_a     = np.full(10, np.nan);  f_a[0]     = 10.0
        sig_a   = np.full(10, np.nan);  sig_a[0]   = 0.0
        run_imputer(imp, f_a, sig_a, 0, acc)
        assert f_a[2] == pytest.approx(9.0), (
            f"Expected 9.0 (coeff=0.9 from specific row), got {f_a[2]}"
        )

        # --- region-B pair: only wildcard row applies (coeff=0.5) ---
        f_b     = np.full(10, np.nan);  f_b[1]     = 10.0
        sig_b   = np.full(10, np.nan);  sig_b[1]   = 0.0
        run_imputer(imp, f_b, sig_b, 0, acc)
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
        imp1 = _fill([(0, 1, 2.0, 0.0, 0.0, "p1", "")])
        # imp2 would fill edge 1 with 99 * 10 = 990 if edge 1 were still missing
        imp2 = _fill([(0, 1, 99.0, 0.0, 0.0, "p2", "")])
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
        imp1 = _fill([(0, 1, 2.0, 0.0, 0.0, "p1", "")])
        # imp2: f[3] = 3.0 * f[1]; edge 1 was NaN initially, filled by imp1
        imp2 = _fill([(1, 3, 3.0, 0.0, 0.0, "p2", "")])
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
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.attach(SimpleNamespace(coverage=np.ones(10, dtype=bool)))
        run_imputer(imp, f, sigma, 0, acc)
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
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.attach(SimpleNamespace(coverage=np.ones(10, dtype=bool)))
        run_imputer(imp, f, sigma, 0, acc)
        assert f[4] == 0.0
        assert sigma[4] == pytest.approx(0.3 * 200.0)

    def test_default_cv_for_unmasked_edges(self, acc):
        """If a mask is omitted from cv_by_mask, edges in that mask fall through to default_cv
        (no density modulation), and the 3-level scale cascade still applies."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[8]  = 50.0;  sigma[8] = 1.0
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001},   # no 'use' rule
            default_cv=0.7,
            acc_struct=acc,
        )
        imp.attach(SimpleNamespace(coverage=np.ones(10, dtype=bool)))
        run_imputer(imp, f, sigma, 0, acc)
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

        # supply listed first → edge 0 gets supply CV
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.attach(SimpleNamespace(coverage=np.ones(10, dtype=bool)))
        run_imputer(imp, f, sigma, 0, acc)
        # edge 0's group includes edge 6 (finite) → scale=80, σ = 0.001·80 = 0.08
        assert sigma[0] == pytest.approx(0.001 * 80.0)

    def test_all_groups_empty_falls_back_to_unit_scale(self, acc):
        """When every level of the cascade is empty (no finite, non-zero |f| anywhere)
        the global-median fallback degrades to scale = 1.0, so σ = CV per edge.
        Previously this case produced σ = 0 (the pin-band bug from the May-13 copper REP)."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        # Nothing finite anywhere → all three cascade levels return 0 → global_scale = 1.0
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.attach(SimpleNamespace(coverage=np.ones(10, dtype=bool)))
        run_imputer(imp, f, sigma, 0, acc)
        # f is still 0 for every filled edge.
        assert np.all(f == 0.0)
        # σ = CV(mask) · 1.0 per edge; supply edges get 0.001, use edges get 0.3.
        # (legacy single-CV form, no density modulation.)
        expected = np.where(acc.masks["supply"], 0.001, 0.3)
        assert np.allclose(sigma, expected, rtol=1e-12)

    def test_log_entries_returned(self, acc):
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[8]  = 100.0;  sigma[8] = 1.0
        imp = FlatPriorImputer(
            cv_by_mask={"supply": 0.001, "use": 0.3},
            default_cv=0.5,
            acc_struct=acc,
        )
        imp.attach(SimpleNamespace(coverage=np.ones(10, dtype=bool)))
        entries = run_imputer(imp, f, sigma, 5, acc)
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
        imp.attach(agg)
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


def _load_driver():
    """The copper case module, which holds the Stage-2 model (#121)."""
    import copper
    return copper


class TestCathodeFdSplitRetired:
    """The cathode final-demand split FormulaRelations are retired — USC/MESC in
    Round X (ADR 0040: a stock change is never a balance residual), FU after it
    (the Flow-node balance closes the domestic Cathode→FU leg first; see
    test_network_imputer.test_flow_balance_counts_a_missing_stock_leg_as_zero).
    The copper case carries no domestic-use formula, so none can write a stock leg."""

    def test_driver_carries_no_domestic_use_formula(self):
        """The copper case builds the cascade; it holds none of the retired relations."""
        import ast
        import inspect
        mod = _load_driver()
        # code names only: the comments there say why the relations are retired
        used = {n.id for n in ast.walk(ast.parse(inspect.getsource(mod)))
                if isinstance(n, ast.Name)}
        used |= {a.name for n in ast.walk(ast.parse(inspect.getsource(mod)))
                 if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        for name in ("cathode_fu_domestic", "cathode_usc_domestic",
                     "cathode_mesc_domestic", "CONC_FORMULA_RELATIONS",
                     "FormulaImputer", "FormulaRelation"):
            assert not hasattr(mod, name), f"{name} is retired"
            assert name not in used, f"{name} is retired"


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
    from pysuto.relation.param_relation import expand_edge_index
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    return (ei["region_from"].to_numpy() == ei["region_to"].to_numpy())


class TestTimeInterpScopeExcludesStockLegs:
    """ADR 0040: the driver's TimeImputer scope never contains a stock-change
    leg, with or without the same-region restriction (ADR 0026)."""

    @staticmethod
    def _ei():
        # domestic use · cross-region use · domestic MESC leg · cross-region USC leg
        return pd.DataFrame({
            "region_from": ["A", "A", "A", "A"],
            "region_to":   ["A", "B", "A", "B"],
            "sector_to":   ["FU", "FU", "MESC", "USC"],
        })

    def test_same_region_scope_drops_stock_legs(self):
        import logging
        mod = _load_driver()
        scope = mod.time_interp_scope(self._ei(), True, ("MESC", "USC"),
                                       logging.getLogger("test"))
        np.testing.assert_array_equal(scope, [True, False, False, False])

    def test_unscoped_time_imputer_still_drops_stock_legs(self):
        import logging
        mod = _load_driver()
        scope = mod.time_interp_scope(self._ei(), False, ("MESC", "USC"),
                                       logging.getLogger("test"))
        np.testing.assert_array_equal(scope, [True, True, False, False])
        assert mod.time_interp_scope(self._ei(), False, (),
                                      logging.getLogger("test")) is None

    def test_stock_leg_is_neither_filled_nor_a_source(self):
        """End to end through TimeImputer: a MESC series with a 2021 gap stays a
        gap, while the parallel domestic series interpolates."""
        import logging
        acc = _TradeAcc()
        acc.node_index = pd.DataFrame(
            {"sector": ["Cathode", "FU", "MESC"],
             "entity": ["Flow",    "SB", "SB"],
             "region": ["A",       "A",  "A"]},
            index=[0, 1, 2],
        )
        from pysuto.relation.param_relation import expand_edge_index
        ei = expand_edge_index(acc.edge_index, acc.node_index)
        scope = _load_driver().time_interp_scope(ei, True, ("MESC", "USC"),
                                                  logging.getLogger("test"))
        f     = np.full(6, np.nan)
        sigma = np.full(6, np.nan)
        f[0], sigma[0] = 10.0, 1.0
        f[2], sigma[2] = 20.0, 1.0
        f[3], sigma[3] = 30.0, 1.0
        f[5], sigma[5] = 50.0, 1.0
        imp = TimeImputer(window=3, method="linear", min_obs=2, interpolate_only=True,
                          scope_edges=scope, id="time_interp")
        run_imputer(imp, f, sigma, 0, acc)
        assert f[1] == pytest.approx(15.0), "Cathode→FU still interpolates"
        assert np.isnan(f[4]), "Cathode→MESC must stay a gap (ADR 0040)"


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

        imp = TimeImputer(window=3, method="linear", min_obs=2,
                          interpolate_only=True,
                          scope_edges=_same_region_scope(acc),
                          id="time_interp")
        run_imputer(imp, f, sigma, 0, acc)

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

        imp = TimeImputer(window=3, method="linear", min_obs=2,
                          interpolate_only=True, id="time_interp")
        run_imputer(imp, f, sigma, 0, acc)

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

        imp = TimeImputer(window=3, method="linear", min_obs=2,
                          interpolate_only=True, scope_edges=scope,
                          id="time_interp")
        entries = run_imputer(imp, f, sigma, 0, acc)

        assert entries == []
        assert np.isnan(f[4])


# =============================================================================
# 8. PriorLedger — write-time no-overwrite + writer scope (#131)
# =============================================================================

class _Writer(BaseImputer):
    """Writes value 7.0 (σ 0.7) on every edge in ``targets`` inside ``scope``."""

    def __init__(self, targets, scope, id="writer"):
        self.id = id
        self._targets = list(targets)
        self._scope = np.zeros(10, dtype=bool)
        self._scope[list(scope)] = True

    def writer_scope(self, acc):
        return self._scope

    def impute(self, ledger, step, acc_struct):
        for e in self._targets:
            ledger.write(e, 7.0, 0.7, step, "scripted", [], [])


class TestPriorLedger:

    def _ledger(self, strict):
        ledger = PriorLedger(np.full(10, np.nan), np.full(10, np.nan), strict_scope=strict)
        scope = np.zeros(10, dtype=bool)
        scope[:3] = True
        ledger.open("w", 2, scope)
        return ledger

    def test_in_scope_write_is_stamped_with_imputer_and_pass(self):
        ledger = self._ledger(strict=True)
        assert ledger.write(1, 4.0, 0.4, 0, "m", [0], [2.0], "d", "pid")
        (entry,) = ledger.close()
        assert (entry["edge_id"], entry["imputer_id"], entry["pass"]) == (1, "w", 2)
        assert (entry["f_after"], entry["sigma_after"]) == (4.0, 0.4)
        assert ledger.f[1] == 4.0 and ledger.sigma[1] == 0.4

    def test_out_of_scope_write_raises_when_strict(self):
        ledger = self._ledger(strict=True)
        with pytest.raises(ScopeError, match="outside its writer scope"):
            ledger.write(5, 4.0, 0.4, 0, "m", [], [])
        assert np.isnan(ledger.f[5])

    def test_out_of_scope_write_is_dropped_and_counted_otherwise(self):
        ledger = self._ledger(strict=False)
        assert not ledger.write(5, 4.0, 0.4, 0, "m", [], [])
        assert np.isnan(ledger.f[5])
        assert ledger.close() == []

    def test_an_edge_that_holds_a_value_is_never_overwritten(self):
        ledger = self._ledger(strict=True)
        ledger.f[1], ledger.sigma[1] = 9.0, 0.9
        assert not ledger.write(1, 4.0, 0.4, 0, "m", [], [])
        assert ledger.f[1] == 9.0 and ledger.close() == []

    def test_write_many_checks_scope_and_skips_filled_edges(self):
        ledger = self._ledger(strict=False)
        ledger.f[0] = 1.0
        n = ledger.write_many(np.array([0, 1, 2, 6]), np.zeros(4), np.ones(4), 0, "m")
        assert n == 2
        assert [e["edge_id"] for e in ledger.close()] == [1, 2]
        with pytest.raises(ScopeError):
            self._ledger(strict=True).write_many(np.array([6]), np.zeros(1), np.ones(1), 0, "m")

    def test_a_scope_of_the_wrong_shape_is_refused(self):
        ledger = PriorLedger(np.full(10, np.nan), np.full(10, np.nan))
        with pytest.raises(ValueError, match="writer scope"):
            ledger.open("w", None, np.ones(4, dtype=bool))

    def test_engine_raises_strict_on_an_out_of_scope_write(self, acc):
        engine = ImputationEngine(acc_struct=acc, strict_scope=True)
        engine.register(_Writer(targets=[5], scope=[0, 1, 2]))
        with pytest.raises(ScopeError):
            engine.run(_agg_result([1.0] + [np.nan] * 9))

    def test_engine_drops_the_out_of_scope_write_in_production(self, acc):
        engine = ImputationEngine(acc_struct=acc)
        engine.register(_Writer(targets=[1, 5], scope=[0, 1, 2]))
        result = engine.run(_agg_result([1.0] + [np.nan] * 9))
        assert result.f_agg[1] == 7.0 and np.isnan(result.f_agg[5])
        assert result.imputation_log.as_dataframe()["edge_id"].tolist() == [1]

    def test_no_imputer_reads_its_hooks_through_hasattr(self):
        import pathlib
        import pysuto.impute as pkg
        for path in pathlib.Path(pkg.__file__).parent.glob("*.py"):
            assert "hasattr(" not in path.read_text(encoding="utf-8"), path.name

    def test_flat_prior_scope_is_every_edge(self, acc):
        imp = FlatPriorImputer(acc, mode="noisy_zero", sigma_high=1e3)
        assert imp.writer_scope(acc).all()

    def test_parametric_scope_is_the_endpoints_of_its_pairs(self, acc):
        imp = _fill([(0, 2, 0.5, 0.05, 0.0, "a", ""),
                                            (4, 3, 2.0, 0.05, 0.0, "b", "")])
        assert np.flatnonzero(imp.writer_scope(acc)).tolist() == [0, 2, 3, 4]

    def test_parametric_cannot_write_outside_its_pairs(self, acc):
        f, sigma = np.full(10, np.nan), np.full(10, np.nan)
        f[0], sigma[0] = 10.0, 1.0
        imp = _fill([(0, 2, 0.5, 0.05, 0.0, "a", "")])
        ledger = PriorLedger(f, sigma, strict_scope=True)
        ledger.open(imp.id, None, imp.writer_scope(acc))
        imp.impute(ledger, 0, acc)
        assert f[2] == pytest.approx(5.0)
        with pytest.raises(ScopeError):
            ledger.write(7, 1.0, 0.1, 0, "m", [], [])


# =============================================================================
# U2 review fixes (#282): write validation, flat-prior coverage, clip log row
# =============================================================================

class TestLedgerRefusesUnusableWrites:

    def _ledger(self):
        ledger = PriorLedger(np.full(10, np.nan), np.full(10, np.nan), strict_scope=True)
        ledger.open("w", None, np.ones(10, dtype=bool))
        return ledger

    @pytest.mark.parametrize("val, sig", [(np.nan, 1.0), (np.inf, 1.0),
                                          (1.0, np.nan), (1.0, -0.1)])
    def test_write_raises_and_leaves_the_edge_missing(self, val, sig):
        ledger = self._ledger()
        with pytest.raises(ValueError, match="non-finite"):
            ledger.write(1, val, sig, 0, "m", [], [])
        assert np.isnan(ledger.f[1]) and ledger.close() == []

    def test_write_many_raises_on_one_bad_entry(self):
        ledger = self._ledger()
        with pytest.raises(ValueError, match="non-finite"):
            ledger.write_many(np.array([1, 2]), np.array([1.0, np.nan]),
                              np.ones(2), 0, "m")
        assert np.isnan(ledger.f[1])

    def test_a_zero_sigma_is_a_valid_write(self):
        assert self._ledger().write(1, 4.0, 0.0, 0, "m", [], [])


class TestFlatPriorNeedsCoverage:

    def _imp(self, acc):
        return FlatPriorImputer(cv_by_mask={"supply": 0.001, "use": 0.3},
                                default_cv=0.5, acc_struct=acc)

    def test_impute_without_attach_raises(self, acc):
        f, sigma = np.full(10, np.nan), np.full(10, np.nan)
        with pytest.raises(RuntimeError, match="no coverage attached"):
            run_imputer(self._imp(acc), f, sigma, 0, acc)

    def test_attach_without_coverage_raises(self, acc):
        with pytest.raises(ValueError, match="no coverage"):
            self._imp(acc).attach(SimpleNamespace())


class TestEngineClipRecordsTheClippedValue:

    def test_the_log_row_holds_the_clipped_f_after_and_the_old_sigma(self, acc):
        """U2-G4-4 / U2-S1-4: log and array agree; sigma is untouched (policy: #238)."""
        acc.bounds.u[2] = 3.0
        ar = _agg_result([10.0] + [np.nan] * 9, [1.0] + [np.nan] * 9)
        engine = ImputationEngine(acc_struct=acc)
        engine.register(_fill([(0, 2, 0.5, 0.0, 0.0, "p", "d")]))
        with pytest.warns(UserWarning, match="clipped"):
            result = engine.run(ar)
        row = result.imputation_log.as_dataframe().query("edge_id == 2").iloc[0]
        assert result.f_agg[2] == pytest.approx(3.0, rel=1e-12)
        assert row["f_after"] == pytest.approx(3.0, rel=1e-12)
        assert row["sigma_after"] == pytest.approx(result.sigma_agg[2], rel=1e-12)


class TestCascadeWithoutFixpointWarns:

    def test_max_iter_reached_with_missing_edges_logs_a_warning(self, acc, caplog):
        """U2-S1-5: the warning names max_iter and the edges still missing."""
        ar = _agg_result([10.0] + [np.nan] * 9, [1.0] + [np.nan] * 9)
        imp_A = _fill([(0, 2, 2.0, 0.1, 0.0, "A", "")], id="A")
        imp_B = _fill([(2, 3, 3.0, 0.1, 0.0, "B", "")], id="B")
        engine = ImputationEngine(acc_struct=acc)
        engine.register_cascade([imp_B, imp_A], max_iter=1)
        with caplog.at_level("WARNING"):
            engine.run(ar)
        assert any("without reaching a fixpoint" in r.getMessage() for r in caplog.records)
