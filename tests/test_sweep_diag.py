"""test_sweep_diag.py — the sweep diagnostic (#107).

A toy sweep is persisted under ``tmp_path``: the base chain is a solved
``_make_wide_sigma_run`` (two periods), one point solves the same QP with edge e0's
prior σ loosened ×3 (a σ knob), and the manifest holds three points — one that reuses
the base runs, the loosened one, and a failed one. Edges 0–4 are 2020's, 5–9 2021's.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from pysuto.diag import SweepDiag, SweepStatsDiagnostic
from pysuto.diag.stats.sweep import MaxAbs, Outcome, sensitivity_table
from pysuto.registry import ConstraintRunRegistry, OptimRunRegistry, SweepRunRegistry
from pysuto.solver import ClarabelSolver
from tests.test_solver import _make_wide_sigma_run

AXES = [{"keys": ["const.ANCHOR_CV"], "values": [0.01, 0.03, 0.05]}]
EDGES = pd.DataFrame({"category":    ["A→B", "B→C", "B→C", "C→D", "C→D"] * 2,
                      "sector_from": ["A", "B", "B", "C", "C"] * 2,
                      "sector_to":   ["B", "C", "C", "D", "D"] * 2,
                      "region_from": ["X", "X", "Y", "X", "Y"] * 2,
                      "region_to":   ["Y", "X", "Y", "X", "X"] * 2,
                      "valuation":   ["CC", "CC", "Mass", "Mass", "Mass"] * 2,
                      "time":        ["2020"] * 5 + ["2021"] * 5})


def _solve_and_persist(root, run):
    ConstraintRunRegistry(str(root / "const")).register(run)
    s = ClarabelSolver(const_dir=str(root / "const"), opt_dir=str(root / "opt"),
                       verbose_first_solve=False, prior_mode="cold", qp_variables="defects")
    res = s.solve_all(run.run_id)
    return s.persist(run_id=run.run_id, results=res, output_dir=str(root / "opt"),
                     const_run_id=run.run_id)


@pytest.fixture
def sweep(tmp_path):
    root = tmp_path / "data" / "proc"
    base = dataclasses.replace(_make_wide_sigma_run(periods=("2020", "2021")),
                               run_id="const_base")
    loose = dataclasses.replace(base, run_id="const_loose",
                                sigma_a=base.sigma_a * np.tile([3.0, 1, 1, 1, 1], 2))
    base_opt, loose_opt = _solve_and_persist(root, base), _solve_and_persist(root, loose)
    decl = {"steps": ["const", "opt"], "mode": "grid", "n_points": 3, "seed": None,
            "axes": AXES, "fixed": {}}

    def pt(i, value, status, const, opt):
        return {"index": i, "overrides": {"const.ANCHOR_CV": value}, "status": status,
                "runs": {"const": const, "opt": opt}, "reused": [], "error": None}
    manifest = {"declaration": decl, "commit": None, "points": [
        pt(0, 0.01, "done", "const_base", base_opt),
        pt(1, 0.03, "done", "const_loose", loose_opt),
        pt(2, 0.05, "failed", None, None)]}
    sid = SweepRunRegistry(str(root / "sweep")).register(
        label="toy", manifest=manifest, stage_config=decl,
        parents={"inputs": {}, "upstream": {"feed_run_ids": [], "imp_run_id": None,
                                            "const_run_id": "const_base",
                                            "opt_run_id": base_opt}})
    return SweepStatsDiagnostic(sid, root=str(root), edges=EDGES)


def _flows(stats, index):
    """(posterior, prior) of a done point, in global edge order."""
    run_id = stats.points_frame().loc[index, "run_opt"]
    res = OptimRunRegistry(stats.root + "/opt").load(run_id)
    F, P = np.zeros(len(EDGES)), np.zeros(len(EDGES))
    for r in res.values():
        F[r.cols_t], P[r.cols_t] = r.F_opt, r.a_prior
    return F, P


class TestOutcomes:

    def test_one_row_per_point_with_its_axis_value_and_status(self, sweep):
        df = sweep.points_frame()
        assert list(df["index"]) == [0, 1, 2]
        assert list(df["status"]) == ["done", "done", "failed"]
        assert list(df["const.ANCHOR_CV"]) == [0.01, 0.03, 0.05]
        assert df.loc[2, ["mass_moved_total", "J_base_at_point"]].isna().all()

    def test_a_point_that_is_the_base_moves_nothing_and_costs_nothing(self, sweep):
        r = sweep.points_frame().loc[0]
        assert r["mass_moved_total"] == 0.0
        assert r["J_base_at_point"] == 0.0
        assert r["max_hard_violation_base"] <= 1e-6

    def test_a_moved_point_costs_more_under_the_base_qp_than_the_base_optimum(self, sweep):
        r = sweep.points_frame().loc[1]
        assert r["mass_moved_total"] > 0.0
        assert r["J_base_at_point"] > 0.0               # the base posterior is its optimum
        assert r["max_hard_violation_base"] <= 1e-6     # same hard rows: still feasible
        fam = r[["J_base_at_point_edges", "J_base_at_point_sum_rows",
                 "J_base_at_point_soft_ratios", "J_base_at_point_other_soft"]].sum()
        assert fam == pytest.approx(r["J_base_at_point"], rel=1e-10, abs=1e-10)

    def test_mass_moved_splits_by_valuation_and_by_category(self, sweep):
        r = sweep.points_frame().loc[1]
        assert r["mass_moved_CC"] + r["mass_moved_Mass"] == pytest.approx(
            r["mass_moved_total"], rel=1e-12)
        cat = sweep.category_frame()
        assert cat[cat["index"] == 1]["mass_moved"].sum() == pytest.approx(
            r["mass_moved_total"], rel=1e-12)
        assert set(cat["category"]) == {"A→B CC", "B→C CC", "B→C Mass", "C→D Mass"}

    def test_retention_of_the_base_point_equals_the_bases(self, sweep):
        cat = sweep.category_frame()
        p0 = cat[cat["index"] == 0]
        np.testing.assert_allclose(p0["retention"], p0["retention_base"], rtol=1e-12)
        yr = sweep.retention_by_year()
        assert set(yr["year"]) == {"2020", "2021"}

    def test_the_own_objective_is_not_an_outcome(self, sweep):
        assert not [c for c in sweep.points_frame().columns if c.startswith("J_own")]

    def test_write_produces_the_tables(self, sweep, tmp_path):
        paths = sweep.write(str(tmp_path / "out"))
        names = {p.split("/")[-1] for p in paths}
        assert names == {"points.csv", "categories.csv", "retention_by_year.csv",
                         "sensitivity.csv"}


def test_a_sweep_whose_points_keep_no_const_run_is_scored_all_the_same(sweep):
    """An opt-only sweep, or one that kept only Stage 4 (#108), names no const run a
    point could be loaded with: every outcome needs only the base const and the
    points' opt runs."""
    expected = sweep.points_frame()
    reg = SweepRunRegistry(sweep.root + "/sweep")
    man = reg.load_manifest(sweep.sweep_run_id)
    for pt in man["points"]:
        pt["runs"].pop("const")
    man["declaration"]["steps"] = ["opt"]
    reg.update(sweep.sweep_run_id, manifest=man)
    df = SweepStatsDiagnostic(sweep.sweep_run_id, root=sweep.root, edges=EDGES).points_frame()
    for col in ("J_base_at_point", "mass_moved_total"):
        np.testing.assert_allclose(df[col], expected[col], rtol=1e-12)


class TestDeclaredOutcomes:

    def test_every_measure_is_zero_at_a_point_that_is_the_base(self, sweep):
        sweep.outcome_defs = (Outcome("m", "mass_moved", {"valuation": "CC"}),
                              Outcome("r", "retention", {"category": "A→B"}),
                              Outcome("t", "total"))
        r = sweep.points_frame(refresh=True).loc[0]
        assert (r["m"], r["r"], r["t"]) == (0.0, 0.0, 0.0)

    def test_each_measure_follows_its_definition_over_the_selected_edges(self, sweep):
        sel = {"sector_from": "B", "valuation": ["CC", "Mass"]}       # edges 1, 2, 6, 7
        sweep.outcome_defs = tuple(Outcome(m, m, sel)
                                   for m in ("mass_moved", "retention", "total"))
        r = sweep.points_frame(refresh=True).loc[1]
        (F_p, P_p), (F_b, P_b) = _flows(sweep, 1), _flows(sweep, 0)
        m = np.zeros(10, dtype=bool)
        m[[1, 2, 6, 7]] = True
        assert r["mass_moved"] == pytest.approx(np.abs(F_p - F_b)[m].sum(), rel=1e-12)
        assert r["retention"] == pytest.approx(
            F_p[m].sum() / P_p[m].sum() - F_b[m].sum() / P_b[m].sum(), rel=1e-12)
        assert r["total"] == pytest.approx(F_p[m].sum() / F_b[m].sum() - 1.0, rel=1e-12)
        assert r["mass_moved"] > 0.0

    def test_a_callable_selector_picks_edges_by_any_rule(self, sweep):
        cross_2021 = Outcome("x", "mass_moved",
                             lambda e: (e.region_from != e.region_to) & (e.time == "2021"))
        sweep.outcome_defs = (cross_2021,)
        r = sweep.points_frame(refresh=True).loc[1]
        (F_p, _), (F_b, _) = _flows(sweep, 1), _flows(sweep, 0)
        assert r["x"] == pytest.approx(np.abs(F_p - F_b)[[5, 9]].sum(), rel=1e-12)

    def test_max_abs_is_the_largest_magnitude_of_its_parts(self, sweep):
        parts = tuple(Outcome(c, "total", {"category": c}) for c in ("A→B", "B→C", "C→D"))
        sweep.outcome_defs = parts + (MaxAbs("worst", parts),)
        r = sweep.points_frame(refresh=True).loc[1]
        assert r["worst"] == max(abs(r[c]) for c in ("A→B", "B→C", "C→D"))
        assert r["worst"] > 0.0

    def test_an_unknown_measure_or_column_is_refused(self, sweep):
        with pytest.raises(ValueError, match="measure"):
            Outcome("bad", "share")
        sweep.outcome_defs = (Outcome("bad", "total", {"sector": "A"}),)
        with pytest.raises(KeyError, match="sector"):
            sweep.points_frame(refresh=True)

    def test_declared_outcomes_are_ranked(self, sweep):
        sweep.outcome_defs = (Outcome("t", "total"),)
        sweep.points_frame(refresh=True)
        assert "t" in sweep.outcomes()
        assert "t" in set(sweep.sensitivity()["outcome"])


class TestSensitivityTable:

    def test_srrc_and_spearman_are_one_for_an_outcome_rank_linear_in_one_axis(self):
        rng = np.random.default_rng(3)
        x1, x2 = rng.uniform(1.1, 1.5, 20), rng.uniform(0.0, 1.0, 20)
        frame = pd.DataFrame({"a": x1, "b": x2, "y": np.exp(x1)})
        axes = [{"keys": ["a"], "distribution": "uniform"},
                {"keys": ["b"], "distribution": "uniform"}]
        t = sensitivity_table(frame, axes, ["y"], mode="random").set_index(
            ["axis", "statistic"])["value"]
        assert t[("a", "srrc")] == pytest.approx(1.0, rel=1e-9)
        assert t[("b", "srrc")] == pytest.approx(0.0, abs=1e-9)
        assert t[("a", "spearman")] == pytest.approx(1.0, rel=1e-12)
        assert t[("(rank regression)", "r2")] == pytest.approx(1.0, rel=1e-9)

    def test_the_grid_main_effect_is_the_spread_of_the_level_means(self):
        frame = pd.DataFrame({"m": ["soft", "hard"] * 3, "w": [2, 2, 3, 3, 5, 5],
                              "y": [1.0, 11.0, 2.0, 12.0, 3.0, 13.0]})
        axes = [{"keys": ["m"], "values": ["soft", "hard"]},
                {"keys": ["w"], "values": [2, 3, 5]}]
        t = sensitivity_table(frame, axes, ["y"], mode="grid").set_index(
            ["axis", "statistic"])["value"]
        assert t[("m", "main_effect_range")] == pytest.approx(10.0, rel=1e-12)
        assert t[("w", "main_effect_range")] == pytest.approx(2.0, rel=1e-12)
        assert t[("m×w", "interaction_range")] == pytest.approx(0.0, abs=1e-12)

    def test_a_nominal_axis_has_a_main_effect_but_no_rank_statistic(self):
        orders = [["trade", "network"], ["network", "trade"]]
        frame = pd.DataFrame({"o": [orders[0], orders[1]] * 4,
                              "a": np.linspace(1.1, 1.5, 8),
                              "y": [1.0, 3.0] * 4})
        axes = [{"keys": ["o"], "values": orders},
                {"keys": ["a"], "distribution": "uniform"}]
        t = sensitivity_table(frame, axes, ["y"], mode="random")
        o = t[t["axis"] == "o"].set_index("statistic")["value"]
        assert o["main_effect_range"] == pytest.approx(2.0, rel=1e-12)
        assert np.isnan(o["srrc"]) and np.isnan(o["spearman"])

    def test_a_nominal_axis_is_controlled_for_so_it_does_not_leak_into_the_others(self):
        """y = a + 1000·[o is the second order]: rank(y) is exactly 0.5·rank(a) plus a
        shift per level, so with o's levels in the regression SRRC(a) is 0.5; left out,
        o's effect would leak into a's coefficient."""
        orders = [["trade", "network"], ["network", "trade"]]
        a = np.linspace(1.1, 1.5, 10)
        o = [orders[i % 2] for i in range(10)]
        y = a + 1000.0 * np.array([i % 2 for i in range(10)])
        frame = pd.DataFrame({"o": o, "a": a, "y": y})
        axes = [{"keys": ["o"], "values": orders},
                {"keys": ["a"], "distribution": "uniform"}]
        t = sensitivity_table(frame, axes, ["y"], mode="random").set_index(
            ["axis", "statistic"])["value"]
        assert t[("a", "srrc")] == pytest.approx(0.5, rel=1e-9)

    def test_a_tied_axis_is_labelled_by_all_its_keys(self):
        frame = pd.DataFrame({"imp.P_PROP": np.linspace(1.1, 1.5, 6),
                              "y": np.linspace(0, 1, 6)})
        axes = [{"keys": ["imp.P_PROP", "const.P_PROP"], "distribution": "uniform"}]
        t = sensitivity_table(frame, axes, ["y"], mode="random")
        assert set(t[t["statistic"] != "r2"]["axis"]) == {"imp.P_PROP=const.P_PROP"}


def test_the_diag_draws_its_four_figures_and_names_the_top_axis(sweep, tmp_path):
    diag = SweepDiag(sweep)
    paths = diag.plot(str(tmp_path / "figs"), show=False)
    assert len(paths) == 4 and all(p.endswith(".png") for p in paths)
    text = diag.summary()
    assert "const.ANCHOR_CV" in text and "failed" in text
    assert "top axis per outcome" in text and "mass_moved_total" in text
    assert "J_own" not in text
    assert "hard-row floor" in text and "breaking" not in text
