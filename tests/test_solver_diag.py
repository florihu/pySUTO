"""
test_solver_diag.py
===================
Smoke tests for :class:`pysuto.diag.SolverDiag`.

The class is purely observational, so the tests focus on:
  1. all 16 figures are produced from a live ``results`` dict + ConstraintRun
  2. ``from_opt_id`` round-trip reconstructs the same set of figures from disk
  3. ``summary()`` returns a non-empty string mentioning the worst-ε timestep
  4. ``compare()`` only renders the 5 hueable plots and rejects unhueable IDs
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
        # 9 run-scope (SD-R-01..09) + 13 timestep-scope (SD-T-01..06, 07, 08a, 08b,
        # 09a, 09b, 09c, 09d) + 1 provenance .md  ⇒  22 PNGs + 1 .md = 23 paths
        assert len(paths) == 23
        for p in paths:
            assert os.path.exists(p)
            if p.endswith(".png"):
                assert os.path.getsize(p) > 1_000  # > 1 KB → not an empty fig

    def test_filenames_carry_plot_id(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        SolverDiag(results, run, label="toy").plot(odir=str(tmp_path))
        names = {os.path.basename(p) for p in os.listdir(tmp_path)}
        assert "sd_r_01_solver_health.png" in names
        assert "sd_r_06_f_evolution.png" in names
        assert "sd_r_07_time_hued.png" in names
        assert "sd_r_08_process_outputs.png" in names
        assert "sd_r_09_flow_outputs.png" in names
        assert "sd_t_06_cstar_vs_c.png" in names
        assert "sd_t_07_rocket.png" in names
        assert "sd_t_08a_eps_heatmap_ratio.png" in names
        assert "sd_t_08b_eps_heatmap_sum.png" in names


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
        from pysuto.registry import ConstraintRunRegistry
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
        assert len(paths) == 23  # 22 PNGs + 1 provenance .md
        for p in paths:
            assert os.path.exists(p)
        # Confirm the default odir is the run-local diag folder
        expected = str(opt_dir / opt_run_id / "diag")
        assert all(p.startswith(expected) for p in paths)


# ---------------------------------------------------------------------------
# 3b. Year-local (block-path) results
# ---------------------------------------------------------------------------

def _solve_block_run(tmp_path, monkeypatch):
    """Build an engine ConstraintRun (G_blocks populated) and solve it →
    year-local OptimResults + the ConstraintRun."""
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
    return s.solve_all(cr.run_id), cr


class TestBlockRunDiag:

    def test_mass_balance_summary_near_zero(self, tmp_path, monkeypatch):
        """SolverDiag built from year-local results reassembles the complete
        residuals, so the mass-balance summary reads ~0 (the artifact the REP
        traced to cross-year placeholders is gone)."""
        import re
        from pysuto.diag import SolverDiag

        results, cr = _solve_block_run(tmp_path, monkeypatch)
        # Sanity: results really are year-local.
        assert all(r.cols_t is not None for r in results.values())

        diag = SolverDiag(results, cr, label="block")
        text = diag.summary()
        m = re.search(r"mass-balance rows:\s*([0-9.eE+-]+)", text)
        assert m, text
        assert float(m.group(1)) < 1e-6, text

    def test_full_portfolio_renders_from_year_local(self, tmp_path, monkeypatch):
        """Every figure renders from year-local results without shape errors."""
        from pysuto.diag import SolverDiag
        results, cr = _solve_block_run(tmp_path, monkeypatch)
        paths = SolverDiag(results, cr, label="block").plot(
            odir=str(tmp_path / "diag"),
        )
        assert len(paths) == 23
        for p in paths:
            assert os.path.exists(p)


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
        assert len(paths) == 5  # SD-R-01..05
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


# ---------------------------------------------------------------------------
# 5. plot_ids selection
# ---------------------------------------------------------------------------

class TestPlotSelection:

    def test_subset_renders_only_selected(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        paths = diag.plot(odir=str(tmp_path),
                          plot_ids=["SD-R-01", "SD-T-05", "SD-T-07"])
        pngs = sorted(os.path.basename(p) for p in paths if p.endswith(".png"))
        assert pngs == sorted([
            "sd_r_01_solver_health.png",
            "sd_t_05_distribution_of_a.png",
            "sd_t_07_rocket.png",
        ])
        # No other figure leaked onto disk
        on_disk = {n for n in os.listdir(tmp_path) if n.endswith(".png")}
        assert on_disk == set(pngs)

    def test_render_order_is_registry_order(self, toy_results, tmp_path):
        from pysuto.diag import SolverDiag
        results, run = toy_results
        diag = SolverDiag(results, run, label="toy")
        # Pass ids in reverse registry order; output must still follow registry order.
        paths = diag.plot(odir=str(tmp_path),
                          plot_ids=["SD-T-07", "SD-T-05", "SD-R-01"])
        pngs = [os.path.basename(p) for p in paths if p.endswith(".png")]
        assert pngs == [
            "sd_r_01_solver_health.png",
            "sd_t_05_distribution_of_a.png",
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
