"""test_sweep_diag.py — the sweep diagnostic (#107, #117).

A toy sweep is persisted under ``tmp_path``: the base chain is a solved
``_make_wide_sigma_run`` (two periods), one point solves the same QP with edge e0's
prior σ loosened ×3 (a σ knob), and the manifest holds three points — one that reuses
the base runs, the loosened one, and a failed one. Edges 0–4 are 2020's, 5–9 2021's.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pandas as pd
import pytest

from pysuto.diag import SweepDiag, SweepStatsDiagnostic
from pysuto.diag.stats.sweep import (Criterion, EveryCategory, MaxAbs, Outcome, _max_abs,
                                     sensitivity_by_year, sensitivity_table, top_axis)
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


def _toy_run():
    return dataclasses.replace(_make_wide_sigma_run(periods=("2020", "2021")),
                               run_id="const_base")


@pytest.fixture
def sweep(tmp_path):
    return _persist_sweep(tmp_path, _toy_run())


@pytest.fixture
def odd_sweep(tmp_path):
    """The toy sweep on a base whose C→D edge e3 is signed (structural lower bound
    below 0) and whose A→B CC category has a zero prior in both years."""
    run = _toy_run()
    l, a0 = run.l.copy(), run.a0.copy()
    l[[3, 8]], a0[[0, 5]] = -1e4, 0.0
    td = {t: {**d, "a_prior": np.where(np.arange(5) == 0, 0.0, d["a_prior"])}
          for t, d in run.timestep_data.items()}
    return _persist_sweep(tmp_path, dataclasses.replace(run, l=l, a0=a0,
                                                        timestep_data=td))


def _persist_sweep(tmp_path, base):
    root = tmp_path / "data" / "proc"
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
        assert names == {"points.csv", "outcomes.csv", "categories.csv",
                         "retention_by_year.csv",
                         "sensitivity.csv", "year_profiles.csv",
                         "sensitivity_by_year.csv", "verdicts.csv"}


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
    df = SweepStatsDiagnostic(sweep.sweep_run_id, root=sweep.root,
                              edges=EDGES).points_frame()
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

    def test_max_abs_is_nan_when_any_part_is_wherever_it_sits(self):
        assert np.isnan(_max_abs([np.nan, 1.0])) and np.isnan(_max_abs([1.0, np.nan]))
        assert _max_abs([-3.0, 2.0]) == 3.0

    def test_a_selector_that_matches_no_edge_is_refused(self, sweep):
        """A misspelt value would otherwise give NaN at every point, which the ranking
        drops without a word."""
        sweep.outcome_defs = (Outcome("chile", "total", {"region_from": "Chlie"}),)
        with pytest.raises(ValueError, match="no edge"):
            sweep.points_frame(refresh=True)

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

    def test_a_max_abs_outcome_is_scored_but_never_ranked(self, sweep):
        """|·| folds a one-way effect into a V that no additive rank regression fits."""
        parts = tuple(Outcome(c, "total", {"category": c}) for c in ("A→B", "B→C"))
        sweep.outcome_defs = parts + (MaxAbs("worst", parts),)
        assert "worst" in sweep.points_frame(refresh=True).columns
        assert "worst" not in sweep.outcomes()
        assert "worst" not in set(sweep.sensitivity()["outcome"])


class TestWindows:

    BC = {"category": "B→C"}                                    # edges 1, 2, 6, 7

    def test_a_pooled_retention_is_the_ratio_of_sums_over_its_window(self, sweep):
        sweep.outcome_defs = (Outcome("all", "retention", self.BC),
                              Outcome("span", "retention", self.BC, ("2020", "2021")),
                              Outcome("y21", "retention", self.BC, ("2021", "2021")))
        r = sweep.points_frame(refresh=True).loc[1]
        (F_p, P_p), (F_b, P_b) = _flows(sweep, 1), _flows(sweep, 0)
        w = [6, 7]
        assert r["y21"] == pytest.approx(
            F_p[w].sum() / P_p[w].sum() - F_b[w].sum() / P_b[w].sum(), rel=1e-12)
        assert r["span"] == r["all"]                     # the default is the full span

    def test_a_one_year_window_equals_that_years_profile_value(self, sweep):
        sweep.outcome_defs = (Outcome("all", "retention", self.BC),
                              Outcome("y21", "retention", self.BC, ("2021", "2021")))
        pts = sweep.points_frame(refresh=True)
        yp = sweep.year_profiles()
        assert list(yp.columns) == ["index", "outcome", "year", "value"]
        assert set(yp[yp["outcome"] == "all"]["year"]) == {"2020", "2021"}
        assert set(yp[yp["outcome"] == "y21"]["year"]) == {"2021"}
        at = yp.set_index(["index", "outcome", "year"])["value"]
        for i in (0, 1):
            assert at[(i, "all", "2021")] == pytest.approx(pts.loc[i, "y21"], rel=1e-12,
                                                           abs=1e-15)
        assert set(yp["index"]) == {0, 1}                # the failed point has none

    def test_only_retention_outcomes_get_a_year_profile(self, sweep):
        sweep.outcome_defs = (Outcome("r", "retention", self.BC), Outcome("t", "total"),
                              Outcome("m", "mass_moved"))
        sweep.points_frame(refresh=True)
        assert set(sweep.year_profiles()["outcome"]) == {"r"}

    def test_a_window_of_years_that_are_not_time_labels_is_refused(self):
        with pytest.raises(ValueError, match="str"):
            Outcome("w", "retention", None, (2020, 2021))

    def test_a_window_outside_the_base_years_is_refused(self, sweep):
        sweep.outcome_defs = (Outcome("w", "retention", self.BC, ("2019", "2021")),)
        with pytest.raises(ValueError, match="2019"):
            sweep.points_frame(refresh=True)

    def test_the_window_is_named_wherever_the_outcome_is_ranked(self, sweep):
        sweep.outcome_defs = (Outcome("all", "retention", self.BC),
                              Outcome("y21", "retention", self.BC, ("2021", "2021")))
        sweep.points_frame(refresh=True)
        win = sweep.sensitivity().groupby("outcome")["window"].first()
        assert (win["all"], win["y21"]) == ("2020–2021", "2021")


class TestCategoriesWithoutRetention:

    def test_a_signed_or_zero_prior_category_has_no_retention_but_its_mass_moved(
            self, odd_sweep):
        cat = odd_sweep.category_frame()
        p1 = cat[cat["index"] == 1].set_index("category")
        assert p1.loc["C→D Mass", "no_retention"] == "signed"
        assert p1.loc["A→B CC", "no_retention"] == "zero prior"
        assert p1.loc["B→C CC", "no_retention"] == ""
        for c in ("C→D Mass", "A→B CC"):
            assert np.isnan(p1.loc[c, "retention"]) and np.isnan(p1.loc[c, "retention_base"])
            assert np.isfinite(p1.loc[c, "mass_moved"])
        assert np.isfinite(p1.loc["B→C CC", "retention"])
        assert p1.loc["A→B CC", "mass_moved"] > 0.0      # e0's σ is the knob
        yr = odd_sweep.retention_by_year()
        undefined = yr["category"].isin(["C→D Mass", "A→B CC"])
        assert yr[undefined]["retention"].isna().all()
        assert yr[~undefined]["retention"].notna().all()

    def test_a_retention_outcome_over_a_signed_edge_is_refused(self, odd_sweep):
        odd_sweep.outcome_defs = (Outcome("cd", "retention", {"category": "C→D"}),)
        with pytest.raises(ValueError, match="signed"):
            odd_sweep.points_frame(refresh=True)

    def test_every_category_expands_to_one_retention_outcome_per_category(self, sweep):
        explicit = Outcome("bc", "retention", {"category": "B→C", "valuation": "Mass"})
        sweep.outcome_defs = (EveryCategory("Mass"), explicit)
        df = sweep.points_frame(refresh=True)
        made = [c for c in df.columns if c.startswith("d_retention[")]
        assert made == ["d_retention[B→C Mass]", "d_retention[C→D Mass]"]
        np.testing.assert_allclose(df["d_retention[B→C Mass]"], df["bc"], rtol=1e-12)

    def test_the_expansion_skips_the_categories_without_retention(self, odd_sweep):
        odd_sweep.outcome_defs = (EveryCategory("CC"), EveryCategory("Mass"))
        df = odd_sweep.points_frame(refresh=True)
        assert [c for c in df.columns if c.startswith("d_retention[")] == [
            "d_retention[B→C CC]", "d_retention[B→C Mass]"]

    def test_a_declared_outcome_equal_to_an_expanded_one_is_scored_once(self, sweep):
        stage = Outcome("d_retention[B→C CC]", "retention",
                        {"category": "B→C", "valuation": "CC"})
        sweep.outcome_defs = (stage, EveryCategory("CC"))
        sweep.points_frame(refresh=True)
        assert sweep.outcomes().count("d_retention[B→C CC]") == 1

    def test_categories_follow_the_declared_chain_order_not_the_alphabet(self, sweep):
        renamed = EDGES.replace({"A": "Mine", "B": "Conc", "C": "Smelt", "D": "Ref"})
        renamed["category"] = renamed["sector_from"] + "→" + renamed["sector_to"]
        stats = SweepStatsDiagnostic(sweep.sweep_run_id, root=sweep.root, edges=renamed)
        assert list(stats.category_table()["category"]) == [
            "Mine→Conc CC", "Conc→Smelt CC", "Conc→Smelt Mass", "Smelt→Ref Mass"]


class TestCriteria:

    AB = Outcome("ab", "retention", {"category": "A→B"})       # e0's σ is the knob
    BC = Outcome("bc", "retention", {"category": "B→C"})

    def test_a_criterion_holds_at_the_base_and_fails_where_a_point_passes_its_limit(
            self, sweep):
        sweep.outcome_defs = (self.AB,)
        moved = abs(sweep.points_frame(refresh=True).loc[1, "ab"])
        assert moved > 0.0
        sweep.criteria = (Criterion("loose", (self.AB,), 2 * moved),
                          Criterion("tight", (self.AB,), moved / 2))
        sweep.points_frame(refresh=True)
        v = sweep.verdicts().set_index("item")
        assert list(v.columns) == ["limit", "points_meeting", "points_done",
                                   "worst_point", "worst_outcome", "worst_window",
                                   "worst_value"]
        assert (v.loc["loose", "points_meeting"], v.loc["loose", "points_done"]) == (2, 2)
        assert (v.loc["tight", "points_meeting"], v.loc["tight", "points_done"]) == (1, 2)
        assert v.loc["tight", "worst_point"] == 1
        assert v.loc["tight", "worst_outcome"] == "ab"
        assert v.loc["tight", "worst_window"] == "2020–2021"
        assert v.loc["tight", "worst_value"] == pytest.approx(moved, rel=1e-12)
        at = sweep.criterion_frame().set_index(["item", "index"])
        assert at.loc[("tight", 0), "value"] == 0.0 and at.loc[("tight", 0), "meets"]

    def test_a_criterion_over_a_max_abs_names_the_part_that_set_its_worst(self, sweep):
        worst = MaxAbs("worst", (self.AB, self.BC))
        sweep.outcome_defs = ()
        sweep.criteria = (Criterion("A6", (worst,), 1e-12),)
        pts = sweep.points_frame(refresh=True)
        assert {"ab", "bc"} <= set(pts.columns)          # scored without being declared
        top = max(("ab", "bc"), key=lambda c: abs(pts.loc[1, c]))
        v = sweep.verdicts().iloc[0]
        assert (v["item"], v["worst_point"], v["worst_outcome"]) == ("A6", 1, top)
        assert v["worst_value"] == pytest.approx(abs(pts.loc[1, top]), rel=1e-12)

    def test_a_criterion_needs_outcomes_and_a_positive_limit(self):
        with pytest.raises(ValueError, match="limit"):
            Criterion("A6", (self.AB,), 0.0)
        with pytest.raises(ValueError, match="outcome"):
            Criterion("A6", (), 0.02)


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

    def test_a_list_axis_swing_is_the_spread_of_its_level_means(self):
        frame = pd.DataFrame({"m": ["soft", "hard"] * 3, "w": [2, 2, 3, 3, 5, 5],
                              "y": [1.0, 11.0, 2.0, 12.0, 3.0, 13.0]})
        axes = [{"keys": ["m"], "values": ["soft", "hard"]},
                {"keys": ["w"], "values": [2, 3, 5]}]
        t = sensitivity_table(frame, axes, ["y"], mode="grid").set_index(
            ["axis", "statistic"])["value"]
        assert t[("m", "swing")] == pytest.approx(10.0, rel=1e-12)
        assert t[("w", "swing")] == pytest.approx(2.0, rel=1e-12)
        assert t[("m×w", "interaction_range")] == pytest.approx(0.0, abs=1e-12)
        # Two levels have an order, so their swing has the sign of their SRRC: second
        # level's mean minus the first's.
        axes[0] = {"keys": ["m"], "values": ["hard", "soft"]}
        t = sensitivity_table(frame, axes, ["y"], mode="grid").set_index(
            ["axis", "statistic"])["value"]
        assert t[("m", "swing")] == pytest.approx(-10.0, rel=1e-12)

    def test_a_numeric_axis_swing_is_its_top_third_mean_minus_its_bottom_third_mean(self):
        """Nine points, thirds by the axis's value (not by row order): y averages 8 over
        x ∈ {0.7, 0.8, 0.9} and 2 over x ∈ {0.1, 0.2, 0.3}, so the swing is +6 in y's
        own unit — whatever the middle third does."""
        x = np.arange(1, 10) / 10.0
        y = np.array([1.0, 2.0, 3.0, 50.0, -50.0, 0.0, 7.0, 8.0, 9.0])
        perm = np.random.default_rng(5).permutation(9)
        frame = pd.DataFrame({"a": x[perm], "b": np.linspace(0, 1, 9), "y": y[perm]})
        axes = [{"keys": ["a"], "distribution": "uniform"},
                {"keys": ["b"], "distribution": "uniform"}]
        t = sensitivity_table(frame, axes, ["y"], mode="random").set_index(
            ["axis", "statistic"])["value"]
        assert t[("a", "swing")] == pytest.approx(6.0, rel=1e-12)
        top, bottom = np.argsort(frame["b"])[-3:], np.argsort(frame["b"])[:3]
        assert t[("b", "swing")] == pytest.approx(
            frame["y"].to_numpy()[top].mean() - frame["y"].to_numpy()[bottom].mean(),
            rel=1e-12)

    def test_the_per_year_table_is_the_sensitivity_table_of_each_years_slice(self):
        rng = np.random.default_rng(11)
        frame = pd.DataFrame({"index": np.arange(12), "a": rng.uniform(0, 1, 12),
                              "o": [["x"], ["y"]] * 6})
        axes = [{"keys": ["a"], "distribution": "uniform"},
                {"keys": ["o"], "values": [["x"], ["y"]]}]
        profiles = pd.DataFrame([(i, out, yr, rng.normal())
                                 for i in range(12) for out in ("r1", "r2")
                                 for yr in ("2020", "2021", "2022")],
                                columns=["index", "outcome", "year", "value"])
        t = sensitivity_by_year(frame, axes, profiles, mode="random")
        assert list(t.columns) == ["year", "outcome", "axis", "statistic", "value", "n"]
        for yr in ("2020", "2021", "2022"):
            wide = profiles[profiles["year"] == yr].pivot(
                index="index", columns="outcome", values="value")
            expect = sensitivity_table(frame.join(wide, on="index"), axes, ["r1", "r2"],
                                       mode="random")
            got = t[t["year"] == yr].drop(columns="year").reset_index(drop=True)
            pd.testing.assert_frame_equal(got, expect, rtol=1e-12)

    def test_a_two_level_nominal_axis_is_ranked_in_its_declared_order(self):
        """Its standardised indicator (second level vs first) is its standardised rank,
        so it gets an SRRC and a Spearman ρ like a numeric axis (#108: the cascade
        order must be judged by the same rule)."""
        orders = [["trade", "network"], ["network", "trade"]]
        frame = pd.DataFrame({"o": [orders[0], orders[1]] * 4,
                              "a": np.linspace(1.1, 1.5, 8),
                              "y": [1.0, 3.0] * 4})
        axes = [{"keys": ["o"], "values": orders},
                {"keys": ["a"], "distribution": "uniform"}]
        t = sensitivity_table(frame, axes, ["y"], mode="random")
        o = t[t["axis"] == "o"].set_index("statistic")["value"]
        assert o["swing"] == pytest.approx(2.0, rel=1e-12)
        assert o["srrc"] == pytest.approx(1.0, rel=1e-9)
        assert o["spearman"] == pytest.approx(1.0, rel=1e-12)

    def test_a_nominal_axis_of_three_levels_has_no_rank_statistic(self):
        levels = [["a"], ["b"], ["c"]]
        frame = pd.DataFrame({"o": levels * 4, "x": np.linspace(0, 1, 12),
                              "y": [1.0, 3.0, 2.0] * 4})
        axes = [{"keys": ["o"], "values": levels},
                {"keys": ["x"], "distribution": "uniform"}]
        t = sensitivity_table(frame, axes, ["y"], mode="random")
        o = t[t["axis"] == "o"].set_index("statistic")["value"]
        assert np.isnan(o["srrc"]) and np.isnan(o["spearman"])
        assert o["swing"] == pytest.approx(2.0, rel=1e-12)

    def test_the_top_axis_is_read_off_spearman_where_the_ranks_are_not_additive(self):
        sens = pd.DataFrame([("y", "a", "srrc", 0.9, 9), ("y", "b", "srrc", 0.1, 9),
                             ("y", "a", "spearman", 0.2, 9), ("y", "b", "spearman", 0.6, 9),
                             ("y", "(rank regression)", "r2", 0.3, 9)],
                            columns=["outcome", "axis", "statistic", "value", "n"])
        assert top_axis(sens, "y") == ("b", "spearman", 0.6)
        sens.loc[4, "value"] = 0.8
        assert top_axis(sens, "y") == ("a", "srrc", 0.9)

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


SW_R = ("sw_r_01", "sw_r_02", "sw_r_04", "sw_r_05", "sw_r_06", "sw_r_07")


def test_the_diag_draws_six_figures_and_prints_a_verdict_per_criterion(sweep, tmp_path):
    ab = Outcome("d_retention[A→B CC]", "retention", {"category": "A→B", "valuation": "CC"})
    sweep.outcome_defs = (ab, EveryCategory("CC"), MaxAbs("worst", (ab,)))
    sweep.criteria = (Criterion("A6", (ab,), 0.02), Criterion("E", (MaxAbs("w", (ab,)),),
                                                             1e-12))
    diag = SweepDiag(sweep, valuation="CC")
    paths = diag.plot(str(tmp_path / "figs"), show=False)
    assert [os.path.basename(p)[:7] for p in paths] == list(SW_R)
    assert all(p.endswith(".png") and os.path.exists(p) for p in paths)
    text = diag.summary()
    lines = text.splitlines()
    assert any(ln.split()[:1] == ["A6"] for ln in lines)
    assert any(ln.split()[:1] == ["E"] and "1/2" in ln for ln in lines)
    assert not any(ln.split()[:1] in (["worst"], ["w"]) for ln in lines)
    assert "const.ANCHOR_CV" in text and "failed points: [2]" in text
    assert "top axis per outcome" in text and "mass_moved_total" in text
    assert "J_own" not in text
    assert "hard-row floor" in text and "breaking" not in text


def test_the_diag_draws_its_figures_with_nothing_declared(odd_sweep, tmp_path):
    """No criterion and no retention outcome: SW-R-02, 05, 06 and 07 say so on an empty
    panel, and SW-R-04 names the categories it leaves out."""
    diag = SweepDiag(odd_sweep, valuation="CC")
    paths = diag.plot(str(tmp_path / "figs"), show=False)
    assert [os.path.basename(p)[:7] for p in paths] == list(SW_R)
    text = diag.summary()
    assert "no criterion declared" in text
    assert "A→B CC (zero prior)" in text and "C→D Mass (signed)" in text
