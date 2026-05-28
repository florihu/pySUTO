"""
test_solver.py
==============
Test suite for :class:`pysuto.solver.ClarabelSolver`.

Toy problem (3 edges, 2 timesteps)
----------------------------------
edges:  a = [a0, a1, a2]
bounds: l = [0, 0, 0],   u = [10, 10, 10]
prior:  a_0 = [1, 2, 3], sigma_a = [1, 1, 1]

Constraint matrix G (3 rows):
  zero   (hard): a0 + a1 - a2 = 0           (balance)
  free   (soft): a0           = c_obs[t] + e,   sigma_c = 0.5
  nonneg (hard): a0 - 2*a1    <= 0          (ratio bound)

timestep_data:
  "2020": c = [0, 1.0, 0],  sigma_c = [0, 0.5, 0]
  "2021": c = [0, 1.5, 0],  sigma_c = [0, 0.5, 0]

The optimum sits near the prior with a0 pulled toward c_obs each timestep,
the balance row exactly satisfied, and the ratio bound non-binding.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Helpers — direct ConstraintRun stubs (no engine, no disk)
# ---------------------------------------------------------------------------

def _make_toy_run(run_id: str = "toy_run"):
    from pysuto.constraints import ConstraintRun

    G = sp.csr_matrix(np.array([
        [ 1.0,  1.0, -1.0],   # zero,   hard  (mass balance)
        [ 1.0,  0.0,  0.0],   # free,   soft  (observation on a0)
        [ 1.0, -2.0,  0.0],   # nonneg, hard  (a0 <= 2*a1)
    ]))
    row_meta = pd.DataFrame({
        "constraint_id": ["balance", "obs_a0", "ratio_bound"],
        "is_hard":       [True, False, True],
        "cone_type":     ["zero", "free", "nonneg"],
    })
    return ConstraintRun(
        G              = G,
        row_meta       = row_meta,
        a0             = np.array([1.0, 2.0, 3.0]),
        sigma_a        = np.array([1.0, 1.0, 1.0]),
        u              = np.array([10.0, 10.0, 10.0]),
        l              = np.array([0.0, 0.0, 0.0]),
        timestep_data  = {
            "2020": {"c": np.array([0.0, 1.0, 0.0]),
                     "sigma_c": np.array([0.0, 0.5, 0.0])},
            "2021": {"c": np.array([0.0, 1.5, 0.0]),
                     "sigma_c": np.array([0.0, 0.5, 0.0])},
        },
        run_id     = run_id,
        acc_id     = "toy_acc",
        agg_id     = "toy_agg",
        created_at = "2026-04-29T00:00:00",
    )


def _make_infeasible_run(run_id: str = "bad_run"):
    """Hard nonneg row 'a0 <= -1' contradicts the l=0 lower bound."""
    from pysuto.constraints import ConstraintRun

    G = sp.csr_matrix(np.array([[1.0, 0.0]]))
    row_meta = pd.DataFrame({
        "constraint_id": ["bad_bound"],
        "is_hard":       [True],
        "cone_type":     ["nonneg"],
    })
    return ConstraintRun(
        G             = G,
        row_meta      = row_meta,
        a0            = np.zeros(2),
        sigma_a       = np.ones(2),
        u             = np.full(2, 10.0),
        l             = np.zeros(2),
        timestep_data = {
            "2020": {"c": np.array([-1.0]), "sigma_c": np.array([0.0])},
        },
        run_id     = run_id,
        acc_id     = "bad_acc",
        agg_id     = "bad_agg",
        created_at = "2026-04-29T00:00:00",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_run():
    return _make_toy_run()


@pytest.fixture
def toy_solver(monkeypatch, toy_run):
    """ClarabelSolver with load_run patched to return the toy ConstraintRun."""
    from pysuto.solver import ClarabelSolver
    s = ClarabelSolver(const_dir="unused", verbose_first_solve=False)
    monkeypatch.setattr(s, "load_run", lambda _rid: toy_run)
    return s


# ---------------------------------------------------------------------------
# Core solve_all behaviour
# ---------------------------------------------------------------------------

class TestSolveAll:

    def test_one_result_per_timestep(self, toy_solver):
        results = toy_solver.solve_all("toy_run")
        assert set(results.keys()) == {"2020", "2021"}

    def test_status_solved(self, toy_solver):
        results = toy_solver.solve_all("toy_run")
        for t, r in results.items():
            assert r.solver_status == "Solved", (t, r.solver_status)

    def test_F_opt_within_bounds(self, toy_solver, toy_run):
        results = toy_solver.solve_all("toy_run")
        for r in results.values():
            assert (r.F_opt >= toy_run.l - 1e-6).all()
            assert (r.F_opt <= toy_run.u + 1e-6).all()

    def test_hard_residuals_satisfied(self, toy_solver, toy_run):
        """Zero-cone rows: residual ~= 0; nonneg-cone rows: residual <= 0."""
        results = toy_solver.solve_all("toy_run")
        ct = toy_run.row_meta["cone_type"].to_numpy()
        zero_idx   = np.where(ct == "zero")[0]
        nonneg_idx = np.where(ct == "nonneg")[0]
        for t, r in results.items():
            assert np.allclose(r.residuals[zero_idx], 0.0, atol=1e-6), \
                (t, r.residuals[zero_idx])
            assert (r.residuals[nonneg_idx] <= 1e-6).all(), \
                (t, r.residuals[nonneg_idx])

    def test_soft_observation_pulls_a0(self, toy_solver):
        """Different soft target c_obs across years -> different F_opt[0]."""
        results = toy_solver.solve_all("toy_run")
        a0_2020 = results["2020"].F_opt[0]
        a0_2021 = results["2021"].F_opt[0]
        # 2021 has c_obs=1.5 vs 2020 c_obs=1.0; warm-started a_0 also differs
        assert a0_2021 > a0_2020, (a0_2020, a0_2021)

    def test_filter_timesteps(self, toy_solver):
        results = toy_solver.solve_all("toy_run", timesteps=["2020"])
        assert set(results.keys()) == {"2020"}

    def test_unknown_timestep_raises(self, toy_solver):
        with pytest.raises(KeyError):
            toy_solver.solve_all("toy_run", timesteps=["3000"])


# ---------------------------------------------------------------------------
# Convergence-info capture (get_info + termination callback)
# ---------------------------------------------------------------------------

class TestConvergenceCapture:

    EXPECTED_CONV_KEYS = (
        "cost_primal", "cost_dual", "gap_abs", "gap_rel",
        "res_primal", "res_dual", "res_primal_inf", "res_dual_inf",
        "ktratio", "mu", "sigma", "step_length",
        "iterations", "solve_time", "status",
        "linsolver_name", "linsolver_nnzA", "linsolver_nnzL",
    )

    def test_convergence_dict_complete(self, toy_solver):
        results = toy_solver.solve_all("toy_run")
        for r in results.values():
            for k in self.EXPECTED_CONV_KEYS:
                assert k in r.convergence, f"missing convergence key: {k}"
            assert r.convergence["status"] == "Solved"
            assert r.convergence["iterations"] == r.iterations
            assert r.convergence["res_primal"] < 1e-6

    def test_iter_history_shape_and_dtype(self, toy_solver):
        from pysuto.solver import ITER_DTYPE
        results = toy_solver.solve_all("toy_run")
        for r in results.values():
            assert r.iter_history is not None
            assert r.iter_history.dtype == ITER_DTYPE
            # callback fires once per iter starting at iter 0
            assert r.iter_history.shape[0] == r.iterations + 1

    def test_iter_history_monotonic(self, toy_solver):
        """The 'iter' column is 0..N strictly increasing."""
        results = toy_solver.solve_all("toy_run")
        for r in results.values():
            iters = r.iter_history["iter"]
            assert iters[0] == 0
            assert (np.diff(iters) == 1).all()

    def test_iter_history_disabled(self, monkeypatch, toy_run):
        from pysuto.solver import ClarabelSolver
        s = ClarabelSolver(
            const_dir="unused",
            verbose_first_solve=False,
            capture_iter_history=False,
        )
        monkeypatch.setattr(s, "load_run", lambda _rid: toy_run)
        results = s.solve_all("toy_run")
        for r in results.values():
            assert r.iter_history is None
            # convergence dict still populated via get_info()
            assert r.convergence["status"] == "Solved"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersist:

    def test_roundtrip(self, toy_solver, tmp_path):
        from pysuto.solver import ITER_DTYPE
        results = toy_solver.solve_all("toy_run")
        out_id  = toy_solver.persist(
            run_id       = "toy_run",
            results      = results,
            output_dir   = str(tmp_path),
            const_run_id = "toy_const",
        )
        run_dir = tmp_path / out_id

        # Harmonized output: opt_<ts>_<code>
        assert out_id.startswith("opt_"), out_id

        # Top-level registry.json — 4 fields per entry only.
        with open(tmp_path / "registry.json") as fh:
            registry = json.load(fh)
        assert out_id in registry
        reg = registry[out_id]
        assert set(reg.keys()) == {"run_id", "stage", "label", "created_at"}
        assert reg["stage"] == "opt"

        # Sidecar carries parents.upstream.const_run_id + summary.timesteps + per_timestep.
        with open(run_dir / "metadata.json") as fh:
            sidecar = json.load(fh)
        assert sidecar["parents"]["upstream"]["const_run_id"] == "toy_const"
        assert set(sidecar["stage_summary"]["timesteps"]) == {"2020", "2021"}
        for t in ("2020", "2021"):
            entry = sidecar["stage_summary"]["per_timestep"][t]
            assert entry["solver_status"] == "Solved"
            conv = entry["convergence"]
            assert conv["status"] == "Solved"
            assert "linsolver_name" in conv
            assert isinstance(conv["iterations"], int)

        # per-timestep npz with iter_history
        for t, r in results.items():
            npz = np.load(run_dir / "timesteps" / f"t_{t}.npz")
            assert set(npz.files) >= {
                "F_opt", "eps_opt", "c_star", "residuals", "iter_history",
            }
            np.testing.assert_array_equal(npz["F_opt"], r.F_opt)
            assert npz["iter_history"].dtype == ITER_DTYPE
            assert npz["iter_history"].shape == r.iter_history.shape

    def test_no_iter_history_when_disabled(self, monkeypatch, toy_run, tmp_path):
        from pysuto.solver import ClarabelSolver
        s = ClarabelSolver(
            const_dir="unused",
            verbose_first_solve=False,
            capture_iter_history=False,
        )
        monkeypatch.setattr(s, "load_run", lambda _rid: toy_run)
        results = s.solve_all("toy_run")
        out_id  = s.persist(
            run_id="toy_run", results=results,
            output_dir=str(tmp_path), const_run_id="toy_const",
        )
        npz = np.load(tmp_path / out_id / "timesteps" / "t_2020.npz")
        assert "iter_history" not in npz.files


# ---------------------------------------------------------------------------
# Feasibility pre-check
# ---------------------------------------------------------------------------

class TestEngineToSolverIntegration:
    """Slice 4: end-to-end with a real ConstraintRun (G_blocks populated by
    engine.run). Per-t solver must produce solved status and F_opt that
    satisfies the per-t mass balance."""

    def _build_real_run(self, tmp_path):
        # Reuse test_constraints' toy_acc / toy_agg_result via local builders.
        # Use persist=False to keep cr.G_blocks intact (registry persistence
        # of G_blocks lands in Slice 8; until then, save/load drops them).
        from tests.test_constraints import _make_toy_acc, _make_toy_agg_result
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder

        acc = _make_toy_acc()
        agg = _make_toy_agg_result(acc)
        storage_dir = str(tmp_path / "const")
        engine = ConstraintEngine(acc, storage_dir=storage_dir)
        engine.register(MassBalanceBuilder())
        cr = engine.run(agg, timesteps=["2020", "2021"], persist=False)
        return acc, cr, storage_dir

    @staticmethod
    def _solver_with_inmem_run(storage_dir, cr, monkeypatch):
        """Build a ClarabelSolver whose load_run returns the in-memory cr,
        so G_blocks survives (Slice 8 will fix the on-disk round-trip)."""
        from pysuto.solver import ClarabelSolver
        s = ClarabelSolver(const_dir=storage_dir, verbose_first_solve=False)
        monkeypatch.setattr(s, "load_run", lambda _rid: cr)
        return s

    def test_solver_solves_each_t_on_engine_built_run(self, tmp_path, monkeypatch):
        acc, cr, storage_dir = self._build_real_run(tmp_path)
        solver = self._solver_with_inmem_run(storage_dir, cr, monkeypatch)
        results = solver.solve_all(cr.run_id)

        assert set(results.keys()) == {"2020", "2021"}
        for t, r in results.items():
            assert r.solver_status == "Solved", (t, r.solver_status)
            assert r.F_opt.shape == (acc.n_edges,)

    def test_per_t_mass_balance_residuals_near_zero(self, tmp_path, monkeypatch):
        """For each t, G_blocks[t] @ F_opt[edge_cols_by_t[t]] ≈ 0 (mass balance)."""
        acc, cr, storage_dir = self._build_real_run(tmp_path)
        solver = self._solver_with_inmem_run(storage_dir, cr, monkeypatch)
        results = solver.solve_all(cr.run_id)

        for t, r in results.items():
            cols_t      = cr.edge_cols_by_t[t]
            per_t_F     = r.F_opt[cols_t]
            block_resid = cr.G_blocks[t] @ per_t_F
            assert np.allclose(block_resid, 0.0, atol=1e-6), (t, block_resid)

    def test_warm_start_reuses_solver_instance_when_G_blocks_identical(
        self, tmp_path, monkeypatch,
    ):
        """When every G_blocks[t] is numerically identical (typical: no
        time-varying ratios), the per-t loop creates one DefaultSolver and
        reuses it via update(P, q, b)."""
        import pysuto.solver as solver_mod
        acc, cr, storage_dir = self._build_real_run(tmp_path)

        # Sanity: G_blocks identical (toy_acc + MassBalance: ±1 coefficients,
        # same topology per t).
        b1 = cr.G_blocks["2020"].toarray()
        b2 = cr.G_blocks["2021"].toarray()
        assert np.array_equal(b1, b2), "Test premise: G_blocks must match"

        init_count = [0]
        original_cls = solver_mod.clarabel.DefaultSolver
        def counting_factory(*args, **kwargs):
            init_count[0] += 1
            return original_cls(*args, **kwargs)
        monkeypatch.setattr(
            solver_mod.clarabel, "DefaultSolver", counting_factory,
        )

        solver = self._solver_with_inmem_run(storage_dir, cr, monkeypatch)
        solver.solve_all(cr.run_id, feasibility_check_first_only=True)

        # 1 feasibility-check solver + 1 solve solver (reused across 2 timesteps)
        assert init_count[0] == 2, \
            f"Expected 2 Clarabel inits (1 feas + 1 reused); got {init_count[0]}"

    def test_no_warm_start_when_G_blocks_differ(self, tmp_path, monkeypatch):
        """Time-varying ratio (different r per year) → G_blocks differ across t
        → solver falls back to fresh Clarabel instance per timestep.
        Numerical results still satisfy the per-year ratio."""
        import csv
        import pysuto.solver as solver_mod
        from pysuto.solver import ClarabelSolver
        from pysuto.constraints import (
            ConstraintEngine, MassBalanceBuilder, RatioConstraintBuilder,
        )
        from tests.test_constraints import _make_toy_acc, _make_toy_agg_result

        csv_path = str(tmp_path / "time_varying.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "param_id",
                "a_sector_from", "a_entity_from", "a_region_from",
                "a_sector_to",   "a_entity_to",   "a_region_to",
                "a_time",        "a_valuation",
                "b_sector_from", "b_entity_from", "b_region_from",
                "b_sector_to",   "b_entity_to",   "b_region_to",
                "b_time",        "b_valuation",
                "value_nominal", "unc_type", "value_sd",
                "value_min", "value_max", "bound_lower", "bound_upper",
            ])
            w.writeheader()
            for year, r_val in (("2020", "0.30"), ("2021", "0.70")):
                w.writerow({
                    "param_id":       f"r_{year}",
                    "a_sector_from":  "Mining", "a_entity_from": "Process",
                    "a_region_from":  "A",
                    "a_sector_to":    "Ore",    "a_entity_to":   "Flow",
                    "a_region_to":    "A",
                    "a_time":         year,     "a_valuation":   "Mass",
                    "b_sector_from":  "Ore",    "b_entity_from": "Flow",
                    "b_region_from":  "A",
                    "b_sector_to":    "Mining", "b_entity_to":   "Process",
                    "b_region_to":    "A",
                    "b_time":         "same",   "b_valuation":   "same",
                    "value_nominal":  r_val,    "unc_type":      "sd",
                    "value_sd":       "0.05",   "value_min":     "",
                    "value_max":     "",        "bound_lower":   "",
                    "bound_upper":   "",
                })

        acc = _make_toy_acc()
        agg = _make_toy_agg_result(acc)
        storage_dir = str(tmp_path / "const")
        engine = ConstraintEngine(acc, storage_dir=storage_dir)
        engine.register(MassBalanceBuilder())
        engine.register(RatioConstraintBuilder.from_csv(csv_path, acc))
        cr = engine.run(agg, timesteps=["2020", "2021"], persist=False)

        # Sanity: blocks really do differ (different r values).
        assert not ClarabelSolver._per_t_problem_identical(cr, ["2020", "2021"]), \
            "Test premise: G_blocks must differ"

        init_count = [0]
        original_cls = solver_mod.clarabel.DefaultSolver
        def counting_factory(*args, **kwargs):
            init_count[0] += 1
            return original_cls(*args, **kwargs)
        # Note: counting_factory is set up BEFORE the solver_with_inmem_run
        # call so DefaultSolver is patched at the point feasibility_check runs.
        monkeypatch.setattr(
            solver_mod.clarabel, "DefaultSolver", counting_factory,
        )
        solver = self._solver_with_inmem_run(storage_dir, cr, monkeypatch)
        results = solver.solve_all(cr.run_id, feasibility_check_first_only=True)

        # 1 feasibility-check init + 2 fresh per-t inits (no warm-start)
        assert init_count[0] == 3, \
            f"Expected 3 Clarabel inits (1 feas + 2 fresh); got {init_count[0]}"
        # All timesteps solved
        for r in results.values():
            assert r.solver_status == "Solved"

    def test_per_t_path_matches_legacy_path_F_opt(self, tmp_path, monkeypatch):
        """Numerical equivalence: solver's per-t path and legacy full-G path
        produce identical F_opt at the t-edges (to solver tol_gap_abs)."""
        from pysuto.solver import ClarabelSolver
        import copy
        acc, cr, storage_dir = self._build_real_run(tmp_path)

        # ── new per-t path: serve cr (with G_blocks) in-memory.
        solver_new = ClarabelSolver(
            const_dir=storage_dir, verbose_first_solve=False,
        )
        monkeypatch.setattr(solver_new, "load_run", lambda _rid: cr)
        new_results = solver_new.solve_all(cr.run_id)

        # ── legacy path: same cr but with G_blocks emptied.
        cr_legacy = copy.copy(cr)
        cr_legacy.G_blocks       = {}
        cr_legacy.edge_cols_by_t = {}
        solver_legacy = ClarabelSolver(
            const_dir=storage_dir, verbose_first_solve=False,
        )
        monkeypatch.setattr(solver_legacy, "load_run", lambda _rid: cr_legacy)
        legacy_results = solver_legacy.solve_all(cr.run_id)

        for t in ("2020", "2021"):
            cols_t   = cr.edge_cols_by_t[t]
            new_F_t  = new_results[t].F_opt[cols_t]
            leg_F_t  = legacy_results[t].F_opt[cols_t]
            assert np.allclose(new_F_t, leg_F_t, atol=1e-7, rtol=1e-7), \
                f"t={t}: per-t F_opt differs from legacy: " \
                f"max abs diff {np.max(np.abs(new_F_t - leg_F_t)):.3e}"


class TestFeasibilityCheck:

    def test_infeasible_raises(self, monkeypatch):
        from pysuto.solver import ClarabelSolver
        run = _make_infeasible_run()
        s   = ClarabelSolver(const_dir="unused", verbose_first_solve=False)
        monkeypatch.setattr(s, "load_run", lambda _rid: run)
        with pytest.raises(RuntimeError, match="Feasibility check FAILED"):
            s.solve_all("bad_run")

    def test_first_only_is_default(self, toy_solver):
        """Default feasibility_check_first_only=True still solves every ts."""
        results = toy_solver.solve_all("toy_run")
        assert set(results.keys()) == {"2020", "2021"}

    def test_check_all_timesteps(self, toy_solver):
        """feasibility_check_first_only=False also passes on the toy run."""
        results = toy_solver.solve_all(
            "toy_run", feasibility_check_first_only=False,
        )
        assert set(results.keys()) == {"2020", "2021"}
