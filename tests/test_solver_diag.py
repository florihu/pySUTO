"""
test_solver_diag.py
===================
Smoke tests for :class:`pysuto.diag.SolverDiag`.

The class is purely observational, so the tests focus on:
  1. the 12-figure portfolio is produced from a live ``results`` dict + ConstraintRun
     (toy fixtures without an accounting structure render placeholders, never fail)
  2. ``from_opt_id`` round-trip reconstructs the same set of figures from disk
  3. ``summary()`` returns a non-empty string mentioning the worst-ε timestep
  4. ``compare()`` only renders the hueable plots and rejects unhueable IDs
  5. ``regions=`` suffixes the filenames and the ranking CSVs
  6. the block-run (year-local) path with an injected acc exercises the real
     sector / identity figures and picks the worst-Σz² year for the rocket
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend before any pyplot import

import pytest


# Reuse the toy fixture pattern from test_solver.py
from test_solver import _make_toy_run  # type: ignore

N_FIGURES = 12
STEMS = [
    "sd_r_01_solver_health",
    "sd_r_08_useful_output",
    "sd_r_09_flow_total_output",
    "sd_r_10_boundary_flows",
    "sd_r_11_objective_decomposition",
    "sd_r_12_convergence_traces",
    "sd_r_13_objective_share_by_sector",
    "sd_r_14_top_identities",
    "sd_r_15a_profiles_edges",
    "sd_r_15b_profiles_sum_rows",
    "sd_r_15c_profiles_soft_ratios",
    "sd_t_07_rocket",
]
HUEABLE = ["SD-R-01", "SD-R-11", "SD-R-13"]


@pytest.fixture
def toy_results(monkeypatch):
    """Solve the toy run and return (results, const_run)."""
    from pysuto.solver import ClarabelSolver
    run = _make_toy_run()
    s = ClarabelSolver(const_dir="unused", verbose_first_solve=False)
    monkeypatch.setattr(s, "load_run", lambda _rid: run)
    results = s.solve_all("toy_run")
    return results, run


# ---------------------------------------------------------------------------
# 1. Live construction → full portfolio
# ---------------------------------------------------------------------------

class TestPlotPortfolio:

    def test_all_pngs_produced(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        paths = diag.plot(odir=str(tmp_path))
        # 12 PNGs + 1 provenance .md; the toy has no acc and no typed row_meta,
        # so no ranking CSVs are written (frames are None).
        assert len(paths) == N_FIGURES + 1
        for p in paths:
            assert os.path.exists(p)
            if p.endswith(".png"):
                assert os.path.getsize(p) > 1_000  # > 1 KB → not an empty fig

    def test_filenames_carry_plot_id(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        SolverDiag(results, run, label="toy").plot(odir=str(tmp_path))
        names = {os.path.basename(p) for p in os.listdir(tmp_path)}
        for stem in STEMS:
            assert f"{stem}.png" in names

    def test_toy_frames_are_none_but_portfolio_complete(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        assert diag.acc is None
        assert diag.stats.edge_frame() is None
        assert diag.stats.constraint_frame() is None          # row_meta lacks time/type
        assert diag._decomposition() is None
        pngs = [p for p in diag.plot(odir=str(tmp_path)) if p.endswith(".png")]
        assert len(pngs) == N_FIGURES

    def test_empty_label_does_not_crash_provenance(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        paths = SolverDiag(results, run).plot(odir=str(tmp_path), plot_ids=["SD-R-01"])
        assert any(p.endswith("provenance.md") for p in paths)


# ---------------------------------------------------------------------------
# 2. summary() text
# ---------------------------------------------------------------------------

class TestSummary:

    def test_summary_mentions_worst_eps_timestep(self, toy_results, capsys):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        text = diag.summary()
        assert "worst ε" in text
        assert any(t in text for t in run.timestep_data.keys())


# ---------------------------------------------------------------------------
# 3. from_opt_id round-trip
# ---------------------------------------------------------------------------

class TestFromOptId:

    def test_roundtrip_recreates_portfolio(self, toy_results, tmp_path, monkeypatch):
        """Persist toy results, then reconstruct via from_opt_id and plot."""
        from pysuto.solver import ClarabelSolver
        from pysuto.registry import ConstraintRunRegistry
        from pysuto.diag import SolverDiag

        results, run = toy_results
        opt_dir   = tmp_path / "opt"
        const_dir = tmp_path / "const"
        const_dir.mkdir()

        ConstraintRunRegistry(str(const_dir)).register(run)

        s = ClarabelSolver(const_dir="unused", verbose_first_solve=False)
        opt_run_id = s.persist(
            run_id     = run.run_id,
            results    = results,
            output_dir = str(opt_dir),
            const_run_id = run.run_id,
        )

        diag = SolverDiag.from_opt_id(
            opt_run_id,
            output_dir = str(opt_dir),
            const_dir  = str(const_dir),
        )
        paths = diag.plot()  # default odir = {opt_dir}/{opt_run_id}/diag
        assert len(paths) == N_FIGURES + 1
        for p in paths:
            assert os.path.exists(p)
        expected = str(opt_dir / opt_run_id / "diag")
        assert all(p.startswith(expected) for p in paths)


# ---------------------------------------------------------------------------
# 3b. Year-local (block-path) results — the only fixture with real edge labels
# ---------------------------------------------------------------------------

def _solve_block_run(tmp_path, monkeypatch):
    """Build an engine ConstraintRun (G_blocks populated) and solve it →
    year-local OptimResults + the ConstraintRun + the in-memory acc."""
    from test_constraints import _make_toy_acc, _make_toy_agg_result
    from pysuto.constraints import ConstraintEngine, MassBalanceBuilder
    from pysuto.solver import ClarabelSolver

    acc = _make_toy_acc()
    agg = _make_toy_agg_result(acc)
    engine = ConstraintEngine(acc, storage_dir=str(tmp_path / "const"))
    engine.register(MassBalanceBuilder())
    cr = engine.run(agg, timesteps=["2020", "2021"], persist=False)

    s = ClarabelSolver(const_dir=str(tmp_path / "const"), verbose_first_solve=False)
    monkeypatch.setattr(s, "load_run", lambda _rid: cr)
    return s.solve_all(cr.run_id), cr, acc


class TestBlockRunDiag:

    def test_mass_balance_summary_near_zero(self, tmp_path, monkeypatch):
        """SolverDiag built from year-local results reassembles the complete
        residuals, so the mass-balance summary reads ~0 (the artifact the REP
        traced to cross-year placeholders is gone)."""
        import re
        from pysuto.diag import SolverDiag

        results, cr, acc = _solve_block_run(tmp_path, monkeypatch)
        assert all(r.cols_t is not None for r in results.values())

        diag = SolverDiag(results, cr, label="block", acc=acc)
        text = diag.summary()
        m = re.search(r"mass-balance rows:\s*([0-9.eE+-]+)", text)
        assert m, text
        assert float(m.group(1)) < 1e-6, text

    def test_full_portfolio_renders_from_year_local(self, tmp_path, monkeypatch):
        """Every figure renders from year-local results; with the acc injected
        the frames exist, so one ranking CSV per family is written too — but no
        per-year edge / constraint table (those are written on demand, #46)."""
        from pysuto.diag import SolverDiag
        results, cr, acc = _solve_block_run(tmp_path, monkeypatch)
        diag = SolverDiag(results, cr, label="block", acc=acc)
        paths = diag.plot(odir=str(tmp_path / "diag"))
        assert len(paths) == N_FIGURES + 3 + 1
        for p in paths:
            assert os.path.exists(p)
        names = {os.path.basename(p) for p in paths}
        assert {"ranking_edges.csv", "ranking_sum_rows.csv", "ranking_soft_ratios.csv"} <= names
        on_disk = os.listdir(tmp_path / "diag")   # the directory, not the returned paths
        assert not [n for n in on_disk if n.startswith(("edges_", "constraints_"))], on_disk
        # the edge frame is real here: the sector figures are not placeholders
        assert os.path.getsize(tmp_path / "diag" / "sd_r_08_useful_output.png") > 20_000

    def test_worst_year_is_argmax_of_zsq_total(self, tmp_path, monkeypatch):
        from pysuto.diag import SolverDiag
        results, cr, acc = _solve_block_run(tmp_path, monkeypatch)
        diag = SolverDiag(results, cr, label="block", acc=acc)
        dec = diag._decomposition().set_index("year")
        assert dec["zsq_edges"].notna().all()
        first, worst, last = diag._resolve_timesteps(None)
        assert str(worst) == dec["zsq_total"].idxmax()
        assert (str(first), str(last)) == ("2020", "2021")

    def test_rocket_has_no_identity_labels(self, tmp_path, monkeypatch):
        from matplotlib.text import Annotation
        from pysuto.diag import SolverDiag
        results, cr, acc = _solve_block_run(tmp_path, monkeypatch)
        fig = SolverDiag(results, cr, label="block", acc=acc)._fig_rocket()
        assert not any(isinstance(t, Annotation) for ax in fig.axes for t in ax.texts)

    def test_regions_filter_suffixes_outputs(self, tmp_path, monkeypatch):
        from pysuto.diag import SolverDiag
        results, cr, acc = _solve_block_run(tmp_path, monkeypatch)
        diag = SolverDiag(results, cr, label="block", acc=acc)
        paths = diag.plot(odir=str(tmp_path / "diag"), regions=["A"],
                          plot_ids=["SD-R-08", "SD-R-13", "SD-T-07"])
        names = sorted(os.path.basename(p) for p in paths)
        assert "sd_r_08_useful_output__a.png" in names
        assert "sd_r_13_objective_share_by_sector__a.png" in names
        assert "sd_t_07_rocket__a.png" in names
        assert "ranking_edges__a.csv" in names
        assert diag._family("edges", regions=["A"])["from_region"].isin(["A"]).any()
        assert len(diag._family("edges", regions=["A"])) < len(diag._family("edges"))


# ---------------------------------------------------------------------------
# 4. compare()
# ---------------------------------------------------------------------------

class TestCompare:

    def test_compare_renders_only_hueable_plots(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        d1 = SolverDiag(results, run, label="a")
        d2 = SolverDiag(results, run, label="b")
        paths = SolverDiag.compare(
            {"a": d1, "b": d2},
            odir=str(tmp_path),
        )
        assert len(paths) == len(HUEABLE)
        names = {os.path.basename(p) for p in paths}
        assert not any("sd_r_14" in n for n in names)
        assert not any("sd_t_" in n for n in names)

    def test_compare_rejects_non_hueable_ids(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        d1 = SolverDiag(results, run, label="a")
        with pytest.raises(ValueError, match="non-hueable"):
            SolverDiag.compare(
                {"a": d1},
                odir=str(tmp_path),
                plot_ids=["SD-T-07"],  # not hueable
            )

    def test_compare_block_runs_with_regions(self, tmp_path, monkeypatch):
        from pysuto.diag import SolverDiag
        results, cr, acc = _solve_block_run(tmp_path, monkeypatch)
        d1 = SolverDiag(results, cr, label="a", acc=acc)
        d2 = SolverDiag(results, cr, label="b", acc=acc)
        paths = SolverDiag.compare({"a": d1, "b": d2}, odir=str(tmp_path / "cmp"), regions=["A"])
        assert len(paths) == len(HUEABLE)
        assert all(os.path.basename(p).endswith("__a.png") for p in paths)


# ---------------------------------------------------------------------------
# 5. plot_ids selection
# ---------------------------------------------------------------------------

class TestPlotSelection:

    def test_subset_renders_only_selected(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        paths = diag.plot(odir=str(tmp_path),
                          plot_ids=["SD-R-01", "SD-R-12", "SD-T-07"])
        pngs = sorted(os.path.basename(p) for p in paths if p.endswith(".png"))
        assert pngs == sorted([
            "sd_r_01_solver_health.png",
            "sd_r_12_convergence_traces.png",
            "sd_t_07_rocket.png",
        ])
        on_disk = {n for n in os.listdir(tmp_path) if n.endswith(".png")}
        assert on_disk == set(pngs)

    def test_render_order_is_registry_order(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        paths = diag.plot(odir=str(tmp_path),
                          plot_ids=["SD-T-07", "SD-R-12", "SD-R-01"])
        pngs = [os.path.basename(p) for p in paths if p.endswith(".png")]
        assert pngs == [
            "sd_r_01_solver_health.png",
            "sd_r_12_convergence_traces.png",
            "sd_t_07_rocket.png",
        ]

    def test_unknown_id_raises(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        with pytest.raises(ValueError, match="Unknown plot_ids"):
            diag.plot(odir=str(tmp_path), plot_ids=["SD-R-01", "SD-X-99"])


# ---------------------------------------------------------------------------
# 6. Registry integrity — the dict is the single source of truth
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_registry_is_the_portfolio(self):
        from pysuto.diag import SolverDiag
        assert [s.filename for s in SolverDiag.PLOTS.values()] == STEMS
        assert [pid for pid, s in SolverDiag.PLOTS.items() if s.hueable] == HUEABLE

    def test_every_spec_method_exists(self):
        from pysuto.diag import SolverDiag
        for pid, spec in SolverDiag.PLOTS.items():
            assert callable(getattr(SolverDiag, spec.method, None)), \
                f"{pid}: missing method {spec.method!r}"

    def test_every_spec_has_description(self):
        from pysuto.diag import SolverDiag
        for pid, spec in SolverDiag.PLOTS.items():
            assert spec.desc.strip(), f"{pid}: empty description"

    def test_filenames_are_unique(self):
        from pysuto.diag import SolverDiag
        stems = [spec.filename for spec in SolverDiag.PLOTS.values()]
        assert len(stems) == len(set(stems))


# ---------------------------------------------------------------------------
# 7. SD-R-10 palette: SB-input and SB-output share one legend, so they must
#    share one palette (Round X: Scrap and MESC were drawn in the same colour)
# ---------------------------------------------------------------------------

class TestBoundaryPalette:

    @staticmethod
    def _frame(years):
        import pandas as pd
        rows = []
        specs = [
            # SB-input: Scrap, CO ; SB-output: MESC, FU, USC, PSC, WR, Tailings, Slag
            # — nine named groups, one more than the per-panel palette holds
            ("Scrap", "SB", "Smelting", "Process", 10.0),
            ("CO", "SB", "Mining", "Process", 100.0),
            ("Cathode", "Flow", "MESC", "SB", 5.0),
            ("Cathode", "Flow", "FU", "SB", 50.0),
            ("Cathode", "Flow", "USC", "SB", 4.0),
            ("CP", "Process", "PSC", "SB", 3.0),
            ("Mining", "Process", "WR", "SB", 200.0),
            ("MF", "Process", "Tailings", "SB", 150.0),
            ("Smelting", "Process", "Slag", "SB", 20.0),
        ]
        for y in years:
            for fs, fe, ts, te, v in specs:
                rows.append({"from_sector": fs, "from_region": "A", "from_entity": fe,
                             "to_sector": ts, "to_region": "A", "to_entity": te,
                             "valuation": "CC", "a_prior": v, "a_post": 1.1 * v,
                             "year": y})
        return pd.DataFrame(rows)

    def test_sb_input_and_sb_output_sectors_get_distinct_colours(self, toy_results, monkeypatch):
        import matplotlib.pyplot as plt
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        monkeypatch.setattr(diag, "_edge_frame", lambda: self._frame(diag._years))
        fig = diag._fig_boundary_flows()
        try:
            handles = fig.legends[0].get_lines()
            by_label = {h.get_label(): h.get_color() for h in handles}
            groups = {"Scrap", "CO", "MESC", "FU", "USC", "PSC", "WR", "Tailings", "Slag"}
            assert groups <= set(by_label)
            colours = [by_label[g] for g in sorted(groups)]
            assert len(set(colours)) == 9           # no two sectors share a colour
        finally:
            plt.close(fig)

    def test_totals_data_honours_a_given_palette(self, toy_results, monkeypatch):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        frame = self._frame(diag._years)
        given = {"Scrap": "#000000", "CO": "#111111", "other": "#222222"}
        sub = frame[frame["from_entity"] == "SB"]
        _, _, _, colors = diag._totals_data(sub, "from_sector", colors=given)
        assert colors is given

    def test_more_named_sectors_than_palette_slots_raises(self):
        from pysuto.diag import SolverDiag
        twelve = [f"s{i:02d}" for i in range(12)]
        colours = SolverDiag._series_colors(twelve + ["other"])
        assert len({colours[n] for n in twelve}) == 12
        with pytest.raises(ValueError, match="13 named series"):
            SolverDiag._series_colors(twelve + ["s12"])

    def test_directions_with_different_valuations_share_columns(self, toy_results, monkeypatch):
        """SB-input carries only Mass, SB-output CC and Mass: one column per
        valuation of either direction, each panel titled by its own valuation."""
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        frame = self._frame(diag._years)
        sb_in = frame["from_entity"] == "SB"
        frame.loc[sb_in, "valuation"] = "Mass"
        mass_out = frame[~sb_in].assign(valuation="Mass")
        frame = pd.concat([frame, mass_out], ignore_index=True)
        monkeypatch.setattr(diag, "_edge_frame", lambda: frame)
        fig = diag._fig_boundary_flows()
        try:
            axes = np.array(fig.axes).reshape(4, 2)
            assert [a.get_title(loc="left") for a in axes[0]] == [
                "SB-input (entity_from = SB) · valuation = CC",
                "SB-input (entity_from = SB) · valuation = Mass"]
            assert [t.get_text() for t in axes[0][0].texts] == ["n=0"]
            assert len(axes[0][1].get_lines()) > 0          # Mass drawn in the Mass column
            assert all(len(a.get_lines()) > 0 for a in axes[2])
        finally:
            plt.close(fig)
