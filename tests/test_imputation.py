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
    _expand_edge_index,
    _sigma_from_row,
)


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
    """

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
            "edge_id","step","method","source_edges","coefficients",
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
            (0, 2, 0.5, 0.05, 0.0, 0.0, np.inf, "test", "desc")
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
        imp1 = ParametricImputer(_csv_pairs=[(0, 1, 2.0, 0.1, 0.0, 0.0, np.inf, "p1", "")])
        # Second imputer: edge 1 → edge 3 (f_3 = 3 * f_1 — edge 1 filled by imp1)
        imp2 = ParametricImputer(_csv_pairs=[(1, 3, 3.0, 0.1, 0.0, 0.0, np.inf, "p2", "")])

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
# 3. ParametricImputer — CSV bidirectional pairs
# =============================================================================

class TestParametricImputerCSV:

    def test_fill_b_from_a(self, acc):
        """A known, B missing → f_b = coeff × f_a."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[0]  = 100.0;  sigma[0] = 5.0
        missing = np.isnan(f)
        # coeff=0.3, sigma_param=0, epsilon=0 → f[2] = 30, sigma[2] = 0.3*5 = 1.5
        imp = ParametricImputer(_csv_pairs=[(0, 2, 0.3, 0.0, 0.0, 0.0, np.inf, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[2] == pytest.approx(30.0)
        assert sigma[2] == pytest.approx(1.5)

    def test_fill_a_from_b_inverse(self, acc):
        """B known, A missing → f_a = f_b / coeff."""
        f     = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        f[2]  = 30.0;  sigma[2] = 1.5
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 2, 0.3, 0.0, 0.0, 0.0, np.inf, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0] == pytest.approx(100.0)
        assert sigma[0] == pytest.approx(1.5 / 0.3)

    def test_both_known_no_fill(self, acc):
        """Both A and B known → no fill, values unchanged."""
        f = np.array([10.0, 20.0] + [np.nan]*8)
        sigma = np.array([1.0, 2.0] + [np.nan]*8)
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 1, 2.0, 0.0, 0.0, 0.0, np.inf, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[0] == pytest.approx(10.0)
        assert f[1] == pytest.approx(20.0)

    def test_both_missing_no_fill(self, acc):
        f = np.full(10, np.nan)
        sigma = np.full(10, np.nan)
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 2, 0.5, 0.0, 0.0, 0.0, np.inf, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert np.isnan(f[0]) and np.isnan(f[2])

    def test_sigma_propagation_includes_sigma_param(self, acc):
        """σ_b² = coeff² σ_a² + σ_param²."""
        f     = np.full(10, np.nan);  f[0]     = 10.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.0
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 2, 2.0, 3.0, 0.0, 0.0, np.inf, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        # σ_b = sqrt(4*0 + 9) = 3
        assert sigma[2] == pytest.approx(3.0)

    def test_epsilon_added_in_quadrature(self, acc):
        f     = np.full(10, np.nan);  f[0]     = 10.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.0
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 2, 1.0, 0.0, 0.1, 0.0, np.inf, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        # f_b = 10, sigma_b = sqrt(0 + (0.1*10)²) = 1.0
        assert f[2] == pytest.approx(10.0)
        assert sigma[2] == pytest.approx(1.0)

    def test_bound_clips_filled_value(self, acc):
        f     = np.full(10, np.nan);  f[0]     = 100.0
        sigma = np.full(10, np.nan);  sigma[0] = 1.0
        missing = np.isnan(f)
        # coeff=2 would give 200, but bound_upper=50 clips to 50
        imp = ParametricImputer(_csv_pairs=[(0, 2, 2.0, 0.0, 0.0, 0.0, 50.0, "p", "d")])
        imp.impute(f, sigma, missing, 0, acc)
        assert f[2] == pytest.approx(50.0)

    def test_log_entries_returned(self, acc):
        f     = np.full(10, np.nan);  f[0]     = 5.0
        sigma = np.full(10, np.nan);  sigma[0] = 0.5
        missing = np.isnan(f)
        imp = ParametricImputer(_csv_pairs=[(0, 2, 1.0, 0.0, 0.0, 0.0, np.inf, "pid", "desc")])
        entries = imp.impute(f, sigma, missing, 3, acc)
        assert len(entries) == 1
        e = entries[0]
        assert e["edge_id"]  == 2
        assert e["step"]     == 3
        assert e["method"]   == "parametric"
        assert e["param_id"] == "pid"
        assert np.isnan(e["f_before"])


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

        imp = TimeImputer(window=5, method="linear")
        entries = imp.impute(f, sigma, missing, 0, acc)

        # λ = 0.5 → f_6 = 15.0
        assert f[6] == pytest.approx(15.0)
        # σ² = 0.25*4 + 0.25*16 = 1+4 = 5 → σ ≈ 2.236
        assert sigma[6] == pytest.approx(np.sqrt(5.0))
        assert any(e["edge_id"] == 6 and e["method"] == "time_linear" for e in entries)

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


# =============================================================================
# 7. sigma_from_row helper
# =============================================================================

class TestSigmaFromRow:

    def test_sd(self):
        row = {"unc_type": "sd", "value_sd": "0.5"}
        assert _sigma_from_row(pd.Series(row), 1.0) == pytest.approx(0.5)

    def test_min_max(self):
        row = {"unc_type": "min_max", "value_min": "0.2", "value_max": "0.6"}
        # (0.6 - 0.2) / 4 = 0.1
        assert _sigma_from_row(pd.Series(row), 0.4) == pytest.approx(0.1)

    def test_pedigree_positive(self):
        row = {
            "unc_type": "pedigree",
            "pedigree_reliability": "2",
            "pedigree_completeness": "2",
            "pedigree_temporal_correlation": "1",
            "pedigree_geographical_correlation": "1",
            "pedigree_other_correlation": "2",
        }
        sigma = _sigma_from_row(pd.Series(row), 1.0)
        assert sigma > 0

    def test_unknown_unc_type_raises(self):
        with pytest.raises(ValueError):
            _sigma_from_row(pd.Series({"unc_type": "unknown"}), 1.0)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
