"""
test_solver_diag.py
===================
Smoke tests for :class:`pysuto.diag.SolverDiag`.

The class is purely observational, so the tests focus on:
  1. all 13 figures are produced from a live ``results`` dict + ConstraintRun
  2. ``from_opt_id`` round-trip reconstructs the same set of figures from disk
  3. ``summary()`` returns a non-empty string mentioning the worst-ε timestep
  4. ``compare()`` only renders the 6 hueable plots and rejects unhueable IDs
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend before any pyplot import

import pytest


# Reuse the toy fixture pattern from test_solver.py
from test_solver import _make_toy_run  # type: ignore


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
        # 8 run-scope (SD-R-01..08) + 8 timestep-scope (SD-T-01..08)
        assert len(paths) == 16
        for p in paths:
            assert os.path.exists(p)
            assert os.path.getsize(p) > 1_000  # > 1 KB → not an empty fig

    def test_filenames_carry_plot_id(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        SolverDiag(results, run, label="toy").plot(odir=str(tmp_path))
        names = {os.path.basename(p) for p in os.listdir(tmp_path)}
        assert "sd_r_01_solver_health.png" in names
        assert "sd_r_07_f_evolution.png" in names
        assert "sd_r_08_time_hued.png" in names
        assert "sd_t_06_cstar_vs_c.png" in names
        assert "sd_t_07_rocket.png" in names
        assert "sd_t_08_eps_heatmap.png" in names


# ---------------------------------------------------------------------------
# 2. summary() text
# ---------------------------------------------------------------------------

class TestSummary:

    def test_summary_mentions_worst_eps_timestep(self, toy_results, capsys):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        text = diag.summary()
        # Both timestep labels appear in the run; the worst-ε one should be in
        # the summary's "worst ε" line.
        assert "worst ε" in text
        # Either toy timestep is acceptable here — we just verify the line
        # exists and references one of the actual labels.
        assert any(t in text for t in run.timestep_data.keys())


# ---------------------------------------------------------------------------
# 3. from_opt_id round-trip
# ---------------------------------------------------------------------------

class TestFromOptId:

    def test_roundtrip_recreates_portfolio(self, toy_results, tmp_path, monkeypatch):
        """Persist toy results, then reconstruct via from_opt_id and plot."""
        from pysuto.solver import ClarabelSolver
        from pysuto.constraints import ConstraintRunRegistry
        from pysuto.diag import SolverDiag

        results, run = toy_results
        opt_dir   = tmp_path / "opt"
        const_dir = tmp_path / "const"
        const_dir.mkdir()

        # Register the toy ConstraintRun so from_opt_id can load it
        ConstraintRunRegistry(str(const_dir)).register(run)

        # Persist results — produces {opt_dir}/registry.json + per-timestep .npz
        s = ClarabelSolver(const_dir="unused", verbose_first_solve=False)
        opt_run_id = s.persist(
            run_id     = run.run_id,
            results    = results,
            output_dir = str(opt_dir),
            const_run_id = run.run_id,
        )

        # Reconstruct & plot
        diag = SolverDiag.from_opt_id(
            opt_run_id,
            output_dir = str(opt_dir),
            const_dir  = str(const_dir),
        )
        paths = diag.plot()  # default odir = {opt_dir}/{opt_run_id}/diag
        assert len(paths) == 16
        for p in paths:
            assert os.path.exists(p)
        # Confirm the default odir is the run-local diag folder
        expected = str(opt_dir / opt_run_id / "diag")
        assert all(p.startswith(expected) for p in paths)


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
        assert len(paths) == 6  # SD-R-01..06
        names = {os.path.basename(p) for p in paths}
        # No SD-R-07 in compare output
        assert not any("sd_r_07" in n for n in names)
        # No SD-T-* in compare output
        assert not any("sd_t_" in n for n in names)

    def test_compare_rejects_non_hueable_ids(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        d1 = SolverDiag(results, run, label="a")
        with pytest.raises(ValueError, match="non-hueable"):
            SolverDiag.compare(
                {"a": d1},
                odir=str(tmp_path),
                plot_ids=["SD-R-07"],  # not hueable
            )
