"""
test_solver.py
==============
Test suite for :class:`pysuto.solver.Solver` and its :class:`ClarabelBackend`.

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

# The keyword arguments that belong to the ClarabelBackend rather than the Solver.
_BACKEND_KEYS = {"tol_feas", "tol_gap_abs", "tol_gap_rel", "reduced_tol_feas",
                 "reduced_tol_gap_abs", "reduced_tol_gap_rel", "max_iter",
                 "direct_solve_method", "max_threads", "capture_iter_history"}


def make_solver(**kw):
    """A :class:`~pysuto.solver.Solver` over a :class:`ClarabelBackend`; the backend's
    settings are split off ``kw``."""
    from pysuto.clarabel_backend import ClarabelBackend
    from pysuto.solver import Solver
    backend = ClarabelBackend(**{k: kw.pop(k) for k in list(kw) if k in _BACKEND_KEYS})
    return Solver(backend, **kw)


# ---------------------------------------------------------------------------
# Helpers — direct ConstraintRun stubs (no engine, no disk)
# ---------------------------------------------------------------------------

def _make_toy_run(run_id: str = "toy_run"):
    """Two periods over the block ``G_t`` (3 rows, 3 edges); the arrays below are
    the period's, so each period's edges are its own three columns."""
    G_t = sp.csr_matrix(np.array([
        [ 1.0,  1.0, -1.0],   # zero,   hard  (mass balance)
        [ 1.0,  0.0,  0.0],   # free,   soft  (observation on a0)
        [ 1.0, -2.0,  0.0],   # nonneg, hard  (a0 <= 2*a1)
    ]))
    a_prior, sigma_a = np.array([1.0, 2.0, 3.0]), np.ones(3)
    sigma_c = np.array([0.0, 0.5, 0.0])
    run = _block_run(
        G_t, ["zero", "free", "nonneg"],
        {"2020": (np.array([0.0, 1.0, 0.0]), sigma_c, a_prior, sigma_a),
         "2021": (np.array([0.0, 1.5, 0.0]), sigma_c, a_prior, sigma_a)},
        run_id=run_id,
    )
    run.row_meta = run.row_meta.drop(columns="type")       # untyped rows, as before
    return run


def _make_infeasible_run(run_id: str = "bad_run"):
    """Hard nonneg row 'a0 <= -1' contradicts the l=0 lower bound."""
    return _block_run(
        sp.csr_matrix(np.array([[1.0, 0.0]])), ["nonneg"],
        {"2020": (np.array([-1.0]), np.array([0.0]), np.zeros(2), np.ones(2))},
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_run():
    return _make_toy_run()


@pytest.fixture
def toy_solver(monkeypatch, toy_run):
    """Solver (ClarabelBackend) with load_run patched to return the toy ConstraintRun."""
    s = make_solver(const_dir="unused", verbose_first_solve=False, prior_mode="cold")
    monkeypatch.setattr(s, "load_run", lambda _rid: toy_run)
    return s


# ---------------------------------------------------------------------------
# Core solve_all behaviour
# ---------------------------------------------------------------------------

class TestSolveAll:

    def test_one_result_per_timestep(self, toy_solver):
        results = toy_solver.solve_all("toy_run")
        assert set(results.keys()) == {"2020", "2021"}

    def test_results_are_year_local(self, toy_solver, toy_run):
        """F_opt/residuals/c_star cover the period's edges and rows, with the
        global index arrays cols_t/rows_t."""
        results = toy_solver.solve_all("toy_run")
        for t, r in results.items():
            np.testing.assert_array_equal(r.cols_t, toy_run.edge_cols_by_t[t])
            assert r.F_opt.shape == (3,)
            assert r.residuals.shape == r.c_star.shape == (3,)

    def test_status_solved(self, toy_solver):
        results = toy_solver.solve_all("toy_run")
        for t, r in results.items():
            assert r.solver_status == "solved", (t, r.solver_status)

    def test_F_opt_within_bounds(self, toy_solver, toy_run):
        results = toy_solver.solve_all("toy_run")
        for t, r in results.items():
            assert (r.F_opt >= toy_run.l[r.cols_t] - 1e-6).all()
            assert (r.F_opt <= toy_run.u[r.cols_t] + 1e-6).all()

    def test_hard_residuals_satisfied(self, toy_solver, toy_run):
        """Hard equality rows: residual ~= 0; ``le`` rows: residual <= 0."""
        results = toy_solver.solve_all("toy_run")
        sense = toy_run.row_meta["sense"].to_numpy()[:3]
        hard  = toy_run.row_meta["is_hard"].to_numpy().astype(bool)[:3]
        zero_idx   = np.where(hard & (sense == "eq"))[0]
        nonneg_idx = np.where(hard & (sense == "le"))[0]
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
        from pysuto.clarabel_backend import ITER_DTYPE
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
        s = make_solver(
            const_dir="unused",
            verbose_first_solve=False,
            capture_iter_history=False,
            prior_mode="cold",
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

    def test_sidecar_records_the_sigma_conditioning_floors(self, monkeypatch, toy_run,
                                                          tmp_path):
        """Stage 5 reads these back (#142)."""
        from pysuto._conditioning import DEFAULT_CONDITIONING
        s = make_solver(const_dir="unused", verbose_first_solve=False, prior_mode="cold",
                        sigma_c_floor=5e-9)
        monkeypatch.setattr(s, "load_run", lambda _rid: toy_run)
        out_id = s.persist(run_id="toy_run", results=s.solve_all("toy_run"),
                           output_dir=str(tmp_path), const_run_id="toy_const")
        with open(tmp_path / out_id / "metadata.json") as fh:
            cfg = json.load(fh)["stage_config"]
        assert {k: cfg[k] for k in DEFAULT_CONDITIONING} == {**DEFAULT_CONDITIONING,
                                                              "sigma_c_floor": 5e-9}

    def test_roundtrip(self, toy_solver, tmp_path):
        from pysuto.clarabel_backend import ITER_DTYPE
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
            assert entry["solver_status"] == "solved"
            conv = entry["convergence"]
            assert conv["status"] == "Solved"
            assert "linsolver_name" in conv
            assert isinstance(conv["iterations"], int)

        # per-timestep npz with iter_history and the block index
        for t, r in results.items():
            npz = np.load(run_dir / "timesteps" / f"t_{t}.npz")
            assert set(npz.files) >= {
                "F_opt", "eps_opt", "c_star", "residuals", "iter_history",
            }
            assert {"cols_t", "rows_t"} <= set(npz.files)
            np.testing.assert_array_equal(npz["F_opt"], r.F_opt)
            assert npz["iter_history"].dtype == ITER_DTYPE
            assert npz["iter_history"].shape == r.iter_history.shape

    def test_block_run_roundtrips_year_local_arrays(self, tmp_path, monkeypatch):
        """A block run persists + reloads cols_t/rows_t and year-local
        F_opt/residuals; the legacy run (above) carries no block index."""
        from tests.test_constraints import _make_toy_acc, _make_toy_agg_result
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder
        from pysuto.registry import OptimRunRegistry

        acc = _make_toy_acc()
        agg = _make_toy_agg_result(acc)
        engine = ConstraintEngine(acc, storage_dir=str(tmp_path / "const"), sigma_a_cv_min=0.05, sigma_a_abs_floor=1e-6, sigma_c_cv_min=0.05, sigma_c_abs_floor=1e-3, nodata_anchor_cv=0.3)
        engine.register(MassBalanceBuilder())
        cr = engine.run(agg, timesteps=["2020", "2021"], persist=False)

        solver = make_solver(
            const_dir=str(tmp_path / "const"), verbose_first_solve=False,
            prior_mode="cold",
        )
        monkeypatch.setattr(solver, "load_run", lambda _rid: cr)
        results = solver.solve_all(cr.run_id)
        out_id = solver.persist(
            run_id=cr.run_id, results=results,
            output_dir=str(tmp_path / "opt"), const_run_id="c",
        )

        npz = np.load(tmp_path / "opt" / out_id / "timesteps" / "t_2020.npz")
        assert {"cols_t", "rows_t"} <= set(npz.files)

        loaded = OptimRunRegistry(str(tmp_path / "opt")).load(out_id)
        for t, r in loaded.items():
            np.testing.assert_array_equal(r.cols_t, cr.edge_cols_by_t[t])
            assert r.F_opt.shape == r.cols_t.shape
            assert r.residuals.shape == r.rows_t.shape

    def test_no_iter_history_when_disabled(self, monkeypatch, toy_run, tmp_path):
        s = make_solver(
            const_dir="unused",
            verbose_first_solve=False,
            capture_iter_history=False,
            prior_mode="cold",
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

def _make_waste_toy_acc():
    """Src(SB) → P(Process) → F(Flow), P → W(SB): only P is balanced (F has no
    output), a = b + w, so a tight a < b books the gap on the loose w."""
    import logging
    from pysuto.acc import AccountingStructure, EdgeBounds
    acc = AccountingStructure.__new__(AccountingStructure)
    acc.logger = logging.getLogger("test_waste_toy")
    acc.id     = "waste_toy"
    acc.node_index = pd.DataFrame(
        [("Src", "SB", "A"), ("P", "Process", "A"), ("F", "Flow", "A"), ("W", "SB", "A")],
        columns=["sector", "entity", "region"],
    )
    acc.n_nodes, acc.time_range, acc.valuation = 4, ["2020"], ["Mass"]
    acc.edge_index = pd.DataFrame([
        {"edge_id": 0, "from_id": 0, "to_id": 1, "time": "2020", "valuation": "Mass"},  # a
        {"edge_id": 1, "from_id": 1, "to_id": 2, "time": "2020", "valuation": "Mass"},  # b
        {"edge_id": 2, "from_id": 1, "to_id": 3, "time": "2020", "valuation": "Mass"},  # w
    ])
    acc.n_edges = 3
    acc.masks = {
        "supply":   np.array([False, True, False]),
        "use":      np.zeros(3, dtype=bool),
        "tradable": np.zeros(3, dtype=bool),
        "sb":       np.array([True, False, True]),
    }
    acc.ruleset = None
    bounds               = EdgeBounds.__new__(EdgeBounds)
    bounds.n_edges       = 3
    bounds.DEFAULT_LOWER = EdgeBounds.DEFAULT_LOWER
    bounds.DEFAULT_UPPER = EdgeBounds.DEFAULT_UPPER
    bounds.l             = np.zeros(3)
    bounds.u             = np.full(3, EdgeBounds.DEFAULT_UPPER)
    bounds.n_exceptions  = 0
    acc.bounds = bounds
    return acc


class TestLowerBoundsEnforced:
    """ADR 0039, issue #28: without the bound rows the QP books a balance gap on
    a loose residual edge as a negative flow; with them the edge stops at 0."""

    def _solve(self, tmp_path, monkeypatch, enforce):
        from pysuto.constraints import ConstraintEngine, FlowBoundsBuilder, MassBalanceBuilder
        acc = _make_waste_toy_acc()

        class _Agg:
            f_agg     = np.array([100.0, 120.0, 0.0])
            sigma_agg = np.array([1.0, 1.0, 1000.0])
            flat_prior = np.zeros(3, dtype=bool)

        storage_dir = str(tmp_path / "const")
        engine = ConstraintEngine(acc, storage_dir=storage_dir, sigma_a_cv_min=0.05, sigma_a_abs_floor=1e-6, sigma_c_cv_min=0.05, sigma_c_abs_floor=1e-3, nodata_anchor_cv=0.3)
        engine.register(MassBalanceBuilder())
        engine.register(FlowBoundsBuilder(enforce_lower=enforce))
        cr = engine.run(_Agg(), timesteps=["2020"], persist=False)
        s = make_solver(const_dir=storage_dir, verbose_first_solve=False,
                        prior_mode="cold")
        monkeypatch.setattr(s, "load_run", lambda _rid: cr)
        r = s.solve_all(cr.run_id)["2020"]
        return cr, r, dict(zip(np.asarray(r.cols_t, dtype=int), r.F_opt))

    def test_without_bounds_the_residual_edge_goes_negative(self, tmp_path, monkeypatch):
        _, r, F = self._solve(tmp_path, monkeypatch, enforce=False)
        assert r.solver_status == "solved"
        assert F[2] < -10.0                                  # w ≈ a − b ≈ −20

    def test_with_bounds_the_residual_edge_stops_at_zero(self, tmp_path, monkeypatch):
        cr, r, F = self._solve(tmp_path, monkeypatch, enforce=True)
        assert r.solver_status == "solved"
        assert F[2] >= -1e-7 and F[2] == pytest.approx(0.0, abs=1e-6)
        assert F[0] == pytest.approx(F[1], rel=1e-6)         # a = b once w is 0
        bound = (cr.row_meta["type"] == "bound").to_numpy()
        assert bound.sum() == 3
        assert (cr.row_meta.loc[bound, "sense"] == "le").all()


class TestEngineToSolverIntegration:
    """Slice 4: end-to-end with a real ConstraintRun (G_blocks populated by
    engine.run). Per-t solver must produce solved status and F_opt that
    satisfies the per-t mass balance."""

    def _build_real_run(self, tmp_path):
        # Reuse test_constraints' toy_acc / toy_agg_result via local builders.
        # persist=False keeps the run in memory. G_blocks are persisted and
        # round-tripped by ConstraintRunRegistry; this test just needs no disk.
        from tests.test_constraints import _make_toy_acc, _make_toy_agg_result
        from pysuto.constraints import ConstraintEngine, MassBalanceBuilder

        acc = _make_toy_acc()
        agg = _make_toy_agg_result(acc)
        storage_dir = str(tmp_path / "const")
        engine = ConstraintEngine(acc, storage_dir=storage_dir, sigma_a_cv_min=0.05, sigma_a_abs_floor=1e-6, sigma_c_cv_min=0.05, sigma_c_abs_floor=1e-3, nodata_anchor_cv=0.3)
        engine.register(MassBalanceBuilder())
        cr = engine.run(agg, timesteps=["2020", "2021"], persist=False)
        return acc, cr, storage_dir

    @staticmethod
    def _solver_with_inmem_run(storage_dir, cr, monkeypatch):
        """Build a Solver whose load_run returns the in-memory cr
        (G_blocks round-trip through the registry; this only skips the disk)."""
        s = make_solver(const_dir=storage_dir, verbose_first_solve=False,
                        prior_mode="cold")
        monkeypatch.setattr(s, "load_run", lambda _rid: cr)
        return s

    def test_solver_solves_each_t_on_engine_built_run(self, tmp_path, monkeypatch):
        acc, cr, storage_dir = self._build_real_run(tmp_path)
        solver = self._solver_with_inmem_run(storage_dir, cr, monkeypatch)
        results = solver.solve_all(cr.run_id)

        assert set(results.keys()) == {"2020", "2021"}
        for t, r in results.items():
            assert r.solver_status == "solved", (t, r.solver_status)
            # Year-local: F_opt covers only this period's edges.
            assert r.F_opt.shape == cr.edge_cols_by_t[t].shape

    def test_block_path_emits_year_local_result(self, tmp_path, monkeypatch):
        """Block path persists only the period's solved values: F_opt over
        cols_t, residuals/c_star over rows_t, with cols_t/rows_t index arrays.
        No cross-year placeholders (REP §8.2, Option 2)."""
        acc, cr, storage_dir = self._build_real_run(tmp_path)
        solver = self._solver_with_inmem_run(storage_dir, cr, monkeypatch)
        results = solver.solve_all(cr.run_id)

        row_time = cr.row_meta["time"].astype(str).to_numpy()
        for t, r in results.items():
            cols_t = cr.edge_cols_by_t[t]
            rows_t = np.where(row_time == t)[0]
            assert r.cols_t is not None and r.rows_t is not None, t
            np.testing.assert_array_equal(r.cols_t, cols_t)
            np.testing.assert_array_equal(r.rows_t, rows_t)
            assert r.F_opt.shape == cols_t.shape
            assert r.residuals.shape == rows_t.shape
            assert r.c_star.shape == rows_t.shape

    def test_per_t_mass_balance_residuals_near_zero(self, tmp_path, monkeypatch):
        """For each t, the stored year-local residual (G_blocks[t] @ F_opt) ≈ 0
        on the mass-balance rows."""
        acc, cr, storage_dir = self._build_real_run(tmp_path)
        solver = self._solver_with_inmem_run(storage_dir, cr, monkeypatch)
        results = solver.solve_all(cr.run_id)

        for t, r in results.items():
            block_resid = cr.G_blocks[t] @ r.F_opt
            assert np.allclose(block_resid, 0.0, atol=1e-6), (t, block_resid)
            assert np.allclose(r.residuals, block_resid, atol=1e-9), t

    def test_warm_start_reuses_solver_instance_when_G_blocks_identical(
        self, tmp_path, monkeypatch,
    ):
        """When every G_blocks[t] is numerically identical (typical: no
        time-varying ratios), the per-t loop creates one DefaultSolver and
        reuses it via update(P, q, b)."""
        import pysuto.clarabel_backend as solver_mod
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
        solver.solve_all(cr.run_id)

        # 2 feasibility probes (one per period) + 1 solve instance (reused across 2 periods)
        assert init_count[0] == 3, \
            f"Expected 3 Clarabel inits (2 probes + 1 reused); got {init_count[0]}"

    def test_no_warm_start_when_G_blocks_differ(self, tmp_path, monkeypatch):
        """Time-varying ratio (different r per year) → G_blocks differ across t
        → solver falls back to fresh Clarabel instance per timestep.
        Numerical results still satisfy the per-year ratio."""
        import csv
        import pysuto.clarabel_backend as solver_mod
        from pysuto.constraints import (
            ConstraintEngine, MassBalanceBuilder, RatioConstraintBuilder,
        )
        from tests.test_constraints import _make_toy_acc, _make_toy_agg_result

        csv_path = str(tmp_path / "time_varying.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "param_id", "param_type",
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
                    "param_type": "concentration", "param_id":       f"r{year}",
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
        engine = ConstraintEngine(acc, storage_dir=storage_dir, sigma_a_cv_min=0.05, sigma_a_abs_floor=1e-6, sigma_c_cv_min=0.05, sigma_c_abs_floor=1e-3, nodata_anchor_cv=0.3)
        engine.register(MassBalanceBuilder())
        engine.register(RatioConstraintBuilder.from_csv(csv_path, acc, sigma_rel_floor=0.02, scale_abs_floor=1.0))
        cr = engine.run(agg, timesteps=["2020", "2021"], persist=False)

        # Sanity: blocks really do differ (different r values).
        assert not np.array_equal(cr.G_blocks["2020"].toarray(),
                                  cr.G_blocks["2021"].toarray()), \
            "Test premise: G_blocks must differ"

        init_count = [0]
        original_cls = solver_mod.clarabel.DefaultSolver
        def counting_factory(*args, **kwargs):
            init_count[0] += 1
            return original_cls(*args, **kwargs)
        # Note: counting_factory is set up BEFORE the solver_with_inmem_run
        # call so DefaultSolver is patched at the point the probe runs.
        monkeypatch.setattr(
            solver_mod.clarabel, "DefaultSolver", counting_factory,
        )
        solver = self._solver_with_inmem_run(storage_dir, cr, monkeypatch)
        results = solver.solve_all(cr.run_id)

        # 2 probe inits + 2 fresh per-t inits (the backend finds no matching structure)
        assert init_count[0] == 4, \
            f"Expected 4 Clarabel inits (2 probes + 2 fresh); got {init_count[0]}"
        # All timesteps solved
        for r in results.values():
            assert r.solver_status == "solved"

    def test_block_path_warm_start_chains_posterior(self, tmp_path, monkeypatch):
        """Block-path warm start (the default) stores the per-period prior on
        each OptimResult and chains it: a_prior(t>0) == a_post(t-1)."""
        import dataclasses
        acc, cr, storage_dir = self._build_real_run(tmp_path)
        cr = dataclasses.replace(cr, acc_ref=dict(TestWarmPriorAcc.ACC_REF))

        solver = make_solver(const_dir=storage_dir, verbose_first_solve=False)
        monkeypatch.setattr(solver, "load_run", lambda _rid: cr)
        TestWarmPriorAcc._patch_acc_load(monkeypatch, acc)     # the warm prior's acc
        res = solver.solve_all(cr.run_id)

        for t in ("2020", "2021"):
            assert res[t].a_prior is not None
            assert len(res[t].a_prior) == len(cr.edge_cols_by_t[t])

        # 2021's prior is 2020's posterior (template-aligned).
        assert np.allclose(res["2021"].a_prior, res["2020"].F_opt,
                           atol=1e-9, rtol=1e-9)


def _toy_acc_inputs(root):
    """acc input toy v1 → v2, two specs with different regions (another edge universe),
    so a structure built from one is refused for a run on the other (#53, #119)."""
    from test_acc_roles import COPPER_LABELS, acc_input_versions
    return acc_input_versions(root, "toy", [dict(COPPER_LABELS, regions=["A"]),
                                            dict(COPPER_LABELS, regions=["A", "B"])])


class TestWarmPriorAcc:
    """#53 item 5: a warm run aligns each year's prior by edge template, so it loads
    the built acc from ``acc_dir``, checks it against the const run's acc ref and
    records that ref in the opt sidecar; it never falls back to a positional
    carry-forward. A cold run reads no acc and records none."""

    ACC_REF = {"id": "toy", "version": 2, "date": "2026-11-11T00:00:00Z",
               "code": "c0ffee00", "label": "toy"}

    def _solver(self, tmp_path, monkeypatch, prior_mode="warm"):
        import dataclasses
        acc, cr, storage_dir = TestEngineToSolverIntegration()._build_real_run(tmp_path)
        cr = dataclasses.replace(cr, acc_ref=dict(self.ACC_REF))
        acc_dir = str(tmp_path / "proc" / "acc")
        _toy_acc_inputs(tmp_path / "input")
        s = make_solver(const_dir=storage_dir, opt_dir=str(tmp_path / "opt"),
                        acc_dir=acc_dir, input_dir=str(tmp_path / "input"),
                        verbose_first_solve=False, prior_mode=prior_mode)
        monkeypatch.setattr(s, "load_run", lambda _rid: cr)
        return s, cr, acc, acc_dir

    @staticmethod
    def _patch_acc_load(monkeypatch, acc):
        from pysuto import acc_store
        calls = []

        def fake_load(id, **kwargs):
            calls.append((id, kwargs.get("storage_dir")))
            return acc

        monkeypatch.setattr(acc_store, "load", fake_load)
        return calls

    def test_an_acc_that_cannot_be_loaded_raises_naming_the_acc_dir(self, tmp_path,
                                                                    monkeypatch):
        s, cr, _, acc_dir = self._solver(tmp_path, monkeypatch)
        with pytest.raises(RuntimeError, match=rf"{acc_dir}"):
            s.solve_all(cr.run_id)

    def test_the_acc_comes_from_acc_dir_and_the_opt_sidecar_records_its_ref(
            self, tmp_path, monkeypatch):
        from pysuto.registry import OptimRunRegistry
        s, cr, acc, acc_dir = self._solver(tmp_path, monkeypatch)
        calls = self._patch_acc_load(monkeypatch, acc)
        res = s.solve_all(cr.run_id)
        assert calls == [("toy", acc_dir)]
        assert np.allclose(res["2021"].a_prior, res["2020"].F_opt, atol=1e-9, rtol=1e-9)
        opt_id = s.persist(run_id=cr.run_id, results=res, const_run_id=cr.run_id)
        sidecar = OptimRunRegistry(str(tmp_path / "opt")).load_sidecar(opt_id)
        assert sidecar["parents"]["inputs"] == {"acc": self.ACC_REF}

    def test_an_acc_built_from_another_version_raises(self, tmp_path, monkeypatch):
        import os
        s, cr, acc, acc_dir = self._solver(tmp_path, monkeypatch)
        self._patch_acc_load(monkeypatch, acc)
        os.makedirs(os.path.join(acc_dir, "toy"))
        with open(os.path.join(acc_dir, "toy", "build_record.json"), "w") as f:
            json.dump({"id": "toy", "version": 1, "code": "0ld0ld00"}, f)
        with pytest.raises(ValueError,
                           match=r"toy v1.*toy v2; they differ in \['regions'\]"):
            s.solve_all(cr.run_id)

    def test_a_const_run_without_an_acc_ref_raises_and_records_nothing(self, tmp_path,
                                                                       monkeypatch):
        """An in-memory const run (``acc_ref`` ``{}``) names no acc version to check
        the structure against; the warm path must never persist ``{"acc": {}}``."""
        import dataclasses
        s, cr, acc, _ = self._solver(tmp_path, monkeypatch)
        monkeypatch.setattr(s, "load_run",
                            lambda _rid: dataclasses.replace(cr, acc_ref={}))
        self._patch_acc_load(monkeypatch, acc)
        with pytest.raises(ValueError, match="acc ref"):
            s.solve_all(cr.run_id)
        assert s._solve_inputs == {}

    def test_a_cold_run_reads_no_acc_and_records_no_inputs(self, tmp_path, monkeypatch):
        from pysuto.registry import OptimRunRegistry
        s, cr, acc, _ = self._solver(tmp_path, monkeypatch, prior_mode="cold")
        calls = self._patch_acc_load(monkeypatch, acc)
        res = s.solve_all(cr.run_id)
        opt_id = s.persist(run_id=cr.run_id, results=res, const_run_id=cr.run_id)
        assert calls == []
        sidecar = OptimRunRegistry(str(tmp_path / "opt")).load_sidecar(opt_id)
        assert sidecar["parents"]["inputs"] == {}


class TestParallelSolve:
    """Parallel solve (#70): cold periods solved concurrently, one Clarabel
    instance per period, the linear solver pinned to one thread."""

    def test_warm_prior_with_several_workers_raises(self):
        with pytest.raises(ValueError, match="warm"):
            make_solver(prior_mode="warm", n_workers=2)

    def test_fewer_than_one_worker_raises(self):
        with pytest.raises(ValueError, match="n_workers"):
            make_solver(prior_mode="cold", n_workers=0)

    @staticmethod
    def _cold_solver(tmp_path, monkeypatch, **kw):
        """Engine-built toy (2 periods, identical G_blocks) + a cold solver
        serving it in memory."""
        _, cr, storage_dir = TestEngineToSolverIntegration()._build_real_run(tmp_path)
        s = make_solver(const_dir=storage_dir, verbose_first_solve=False,
                        prior_mode="cold", **kw)
        monkeypatch.setattr(s, "load_run", lambda _rid: cr)
        return cr, s

    @staticmethod
    def _record_clarabel_inits(monkeypatch):
        """Wrap clarabel.DefaultSolver; return the list of settings each
        instance was built with (list.append is thread-safe)."""
        import pysuto.clarabel_backend as solver_mod
        built = []
        original_cls = solver_mod.clarabel.DefaultSolver
        def recording_factory(*args, **kwargs):
            built.append(args[5])                    # (P, q, A, b, cones, settings)
            return original_cls(*args, **kwargs)
        monkeypatch.setattr(solver_mod.clarabel, "DefaultSolver", recording_factory)
        return built

    def test_parallel_solve_factorises_each_period_on_its_own(self, tmp_path, monkeypatch):
        """Serial reuses one instance across identical periods (see
        test_warm_start_reuses_solver_instance_when_G_blocks_identical); a
        parallel solve builds one per period."""
        built = self._record_clarabel_inits(monkeypatch)
        cr, s = self._cold_solver(tmp_path, monkeypatch, n_workers=2)
        res = s.solve_all(cr.run_id)

        assert len(built) == 4, f"2 probes + 2 periods expected; got {len(built)}"
        assert all(r.solver_status == "solved" for r in res.values())

    @pytest.mark.parametrize("n_workers", [1, 2])
    def test_period_solves_use_one_linear_solver_thread(
        self, tmp_path, monkeypatch, n_workers,
    ):
        """Multi-threaded faer is slower than one thread on a period QP and
        gives the same answer; the per-period probes are pinned the same way."""
        built = self._record_clarabel_inits(monkeypatch)
        cr, s = self._cold_solver(tmp_path, monkeypatch, n_workers=n_workers,
                                  direct_solve_method="qdldl")
        s.solve_all(cr.run_id)

        assert len(built) >= 3, "2 probes and the period solves expected"
        for st in built:
            assert st.max_threads == 1
            assert st.direct_solve_method == "qdldl"

    def test_parallel_solve_matches_each_period_solved_alone_bit_for_bit(
        self, tmp_path, monkeypatch,
    ):
        """Reference = each period solved serially on its own instance. A serial
        run over this toy's identical G_blocks would reuse the factorisation for
        2021 (update() skips re-equilibration, ~1e-10 off); copper's blocks
        differ, so its serial runs build a fresh instance per period too."""
        cr, serial   = self._cold_solver(tmp_path, monkeypatch, n_workers=1)
        _,  parallel = self._cold_solver(tmp_path, monkeypatch, n_workers=2)
        monkeypatch.setattr(parallel, "load_run", lambda _rid: cr)
        res_s = {t: serial.solve_all(cr.run_id, timesteps=[t])[t]
                 for t in ("2020", "2021")}
        res_p = parallel.solve_all(cr.run_id)

        assert list(res_p) == ["2020", "2021"]
        for t in res_s:
            np.testing.assert_array_equal(res_p[t].F_opt,     res_s[t].F_opt)
            np.testing.assert_array_equal(res_p[t].eps_opt,   res_s[t].eps_opt)
            np.testing.assert_array_equal(res_p[t].residuals, res_s[t].residuals)
            assert res_p[t].objective == res_s[t].objective

    def test_parallel_solve_matches_a_serial_run_to_solver_tolerance(
        self, tmp_path, monkeypatch,
    ):
        """Against one serial run over both periods (KKT reuse on this toy's
        identical G_blocks): same period order, same values to solver tolerance."""
        cr, serial   = self._cold_solver(tmp_path, monkeypatch, n_workers=1)
        _,  parallel = self._cold_solver(tmp_path, monkeypatch, n_workers=2)
        monkeypatch.setattr(parallel, "load_run", lambda _rid: cr)
        res_s = serial.solve_all(cr.run_id)
        res_p = parallel.solve_all(cr.run_id)

        assert list(res_p) == list(res_s)
        for t in res_s:
            assert np.allclose(res_p[t].F_opt, res_s[t].F_opt, rtol=1e-7, atol=1e-7), t

    def test_sidecar_records_how_the_periods_were_solved(self, tmp_path, monkeypatch):
        cr, s = self._cold_solver(tmp_path, monkeypatch, n_workers=8)
        results = s.solve_all(cr.run_id)
        out_id  = s.persist(
            run_id       = cr.run_id,
            results      = results,
            output_dir   = str(tmp_path / "opt"),
            stage_config = {"prior_mode": "cold", "max_iter": 2000},
        )
        with open(tmp_path / "opt" / out_id / "metadata.json") as fh:
            sidecar = json.load(fh)

        cfg = sidecar["stage_config"]
        assert cfg["n_workers"] == 2                  # 8 requested, 2 periods
        assert cfg["max_threads"] == 1
        assert cfg["direct_solve_method"] == "faer"
        assert cfg["max_iter"] == 2000                # the caller's keys are kept
        assert sidecar["stage_summary"]["solve_wall_s"] > 0
        assert set(sidecar["stage_summary"]["per_timestep"]) == {"2020", "2021"}

    def test_failed_periods_are_named_and_nothing_is_returned(self, tmp_path, monkeypatch):
        """Clarabel reports NumericalError on every period (the toy itself
        solves in 0 iterations, so max_iter cannot make it fail); the probe is
        skipped so the failure comes from the period solves themselves. Both
        periods meet at a barrier before either fails: otherwise, under CPU
        load, 2021 is cancelled before it starts and only 2020 is named (#83)."""
        import threading
        import types
        import pysuto.clarabel_backend as solver_mod

        both_started = threading.Barrier(2, timeout=30)

        class _NumericalError:
            def __init__(self, inner):
                self._inner = inner
            def solve(self):
                both_started.wait()
                sol = self._inner.solve()
                return types.SimpleNamespace(status="NumericalError", x=sol.x,
                                             iterations=sol.iterations,
                                             obj_val=sol.obj_val)
            def __getattr__(self, name):
                return getattr(self._inner, name)

        original_cls = solver_mod.clarabel.DefaultSolver
        monkeypatch.setattr(solver_mod.clarabel, "DefaultSolver",
                            lambda *a, **k: _NumericalError(original_cls(*a, **k)))
        opt_dir = tmp_path / "opt"
        cr, s = self._cold_solver(tmp_path, monkeypatch, n_workers=2,
                                  opt_dir=str(opt_dir))
        with pytest.raises(RuntimeError, match="NumericalError") as exc:
            s.solve_all(cr.run_id, feasibility_check="none")
        assert "'2020'" in str(exc.value) and "'2021'" in str(exc.value)
        assert not list(opt_dir.glob("opt_*")), "a failed solve registered a run"


# ---------------------------------------------------------------------------
# QP variables (#33): the flows or their defects
# ---------------------------------------------------------------------------

def _block_run(G_t, kinds, period_data, run_id="qp_toy"):
    """A block-path ConstraintRun whose periods share the block ``G_t``. ``kinds``
    gives each row's kind: "free" (soft eq), "zero" (hard eq) or "nonneg" (hard le).
    ``period_data`` maps each period to its ``(c, sigma_c, a_prior,
    sigma_a)`` over the block's rows and edges. Single-entry ``−1`` rows are typed
    ``bound``, the other ``le`` rows ``ratio``, the soft rows ``sum``."""
    from pysuto.constraints import ConstraintRun
    m, n = G_t.shape
    periods = list(period_data)
    kinds = np.asarray(kinds)
    is_bound = (G_t.getnnz(axis=1) == 1) & (np.asarray(G_t.sum(axis=1)).ravel() == -1.0)
    row_type = np.where(kinds == "free", "sum", np.where(is_bound, "bound", "ratio"))
    row_meta = pd.concat([pd.DataFrame({
        "constraint_id": [f"r{i}_{t}" for i in range(m)],
        "type":          row_type,
        "is_hard":       kinds != "free",
        "sense":         np.where(kinds == "nonneg", "le", "eq"),
        "time":          t,
    }) for t in periods], ignore_index=True)
    timestep_data = {}
    for k, (t, (c, sigma_c, a_prior, _)) in enumerate(period_data.items()):
        c_full, sc_full = np.zeros(m * len(periods)), np.zeros(m * len(periods))
        c_full[k * m:(k + 1) * m], sc_full[k * m:(k + 1) * m] = c, sigma_c
        timestep_data[t] = {"c": c_full, "sigma_c": sc_full,
                            "a_prior": np.asarray(a_prior, dtype=float)}
    return ConstraintRun(
        G              = sp.block_diag([G_t] * len(periods), format="csr"),
        row_meta       = row_meta,
        a0             = np.concatenate([d[2] for d in period_data.values()]),
        sigma_a        = np.concatenate([d[3] for d in period_data.values()]),
        u              = np.full(n * len(periods), 1e4),
        l              = np.zeros(n * len(periods)),
        timestep_data  = timestep_data,
        run_id         = run_id,
        acc_id         = "toy_acc",
        agg_id         = "toy_agg",
        created_at     = "2026-10-20T00:00:00",
        G_blocks       = {t: G_t.copy() for t in periods},
        edge_cols_by_t = {t: np.arange(k * n, (k + 1) * n) for k, t in enumerate(periods)},
    )


def _make_wide_sigma_run(periods=("2020",), seed=33):
    """Five edges per period, σ_a over eight orders (1e-3 … 1e5)::

        zero    e0 − e1 − e2 − e3 − e4 = 0    (balance; pins the loose e1)
        free    e0      = 110,  σ_c = 5
        free    e1 + e2 = 105,  σ_c = 3
        nonneg  e2 − 0.55·e0 ≤ 0              (hard band, binding)
        nonneg  −e_i ≤ 0                      (lower bounds; e3's binds)

    Every edge is held by its prior or pinned by a hard row, and the tight edge e4 has
    a near-zero prior, so Σ(a_prior/σ_a)² stays within an order of J and the flows
    solve converges as well as the defects solve. Periods after the first share the
    block (a serial solve reuses its KKT factorisation) with the soft rows' c and the
    priors scaled by U(0.9, 1.1), seeded."""
    rng = np.random.default_rng(seed)
    G_t = sp.csr_matrix(np.vstack([
        [1.0,   -1.0, -1.0, -1.0, -1.0],
        [1.0,    0.0,  0.0,  0.0,  0.0],
        [0.0,    1.0,  1.0,  0.0,  0.0],
        [-0.55,  0.0,  1.0,  0.0,  0.0],
        -np.eye(5),
    ]))
    kinds = ["zero", "free", "free", "nonneg"] + ["nonneg"] * 5
    c         = np.array([0.0, 110.0, 105.0, 0.0] + [0.0] * 5)
    sigma_c   = np.array([0.0, 5.0, 3.0, 0.0] + [0.0] * 5)
    a_prior   = np.array([100.0, 40.0, 80.0, -20.0, 0.002])
    sigma_a   = np.array([5.0, 1e5, 2.0, 2.0, 1e-3])
    period_data = {}
    for k, t in enumerate(periods):
        scale_c = np.ones(9)
        scale_c[1:3] = rng.uniform(0.9, 1.1, 2) if k else 1.0
        scale_a = rng.uniform(0.9, 1.1, 5) if k else np.ones(5)
        period_data[t] = (c * scale_c, sigma_c, a_prior * scale_a, sigma_a)
    return _block_run(G_t, kinds, period_data)


class TestQpVariables:
    """#33: solving for the defects d = a − a_prior (q = 0, b = c − G·a_prior) is the
    same QP as solving for the flows, whose cost carries the constant Σ(a_prior/σ_a)²;
    the result comes back in flow space either way."""

    @staticmethod
    def _solve(run, tmp_path, monkeypatch, qp_variables, n_workers=1):
        s = make_solver(const_dir=str(tmp_path / "const"), opt_dir=str(tmp_path / "opt"),
                        verbose_first_solve=False, prior_mode="cold",
                        n_workers=n_workers, qp_variables=qp_variables)
        monkeypatch.setattr(s, "load_run", lambda _rid: run)
        return s, s.solve_all(run.run_id)

    @staticmethod
    def _J(run, t, F):
        """Σ((F − a_prior)/σ_a)² + Σ_free((G·F − c)/σ_c)² of period ``t``."""
        rows = np.where(run.row_meta["time"].to_numpy() == t)[0]
        cols = run.edge_cols_by_t[t]
        r    = run.G_blocks[t] @ F - run.timestep_data[t]["c"][rows]
        free = ~run.row_meta["is_hard"].to_numpy().astype(bool)[rows]
        sc   = run.timestep_data[t]["sigma_c"][rows]
        return (float((((F - run.a0[cols]) / run.sigma_a[cols]) ** 2).sum())
                + float(((r[free] / sc[free]) ** 2).sum()))

    def _assert_same_optimum(self, run, flows, defects):
        assert list(defects) == list(flows)
        for t in flows:
            f, d = flows[t], defects[t]
            assert f.solver_status == d.solver_status == "solved", t
            for k in ("F_opt", "eps_opt", "residuals", "c_star"):
                assert np.allclose(getattr(d, k), getattr(f, k), rtol=1e-7, atol=1e-7), \
                    (t, k, np.max(np.abs(getattr(d, k) - getattr(f, k))))
            np.testing.assert_array_equal(d.a_prior, f.a_prior)
            assert self._J(run, t, d.F_opt) == pytest.approx(self._J(run, t, f.F_opt),
                                                             rel=1e-6)

    def test_defects_give_the_flows_optimum_with_binding_bound_and_band(
        self, tmp_path, monkeypatch,
    ):
        run = _make_wide_sigma_run()
        _, flows   = self._solve(run, tmp_path, monkeypatch, "flows")
        _, defects = self._solve(run, tmp_path, monkeypatch, "defects")
        self._assert_same_optimum(run, flows, defects)

        F = defects["2020"].F_opt
        assert F[3] == pytest.approx(0.0, abs=1e-6)            # e3's lower bound binds
        assert F[2] == pytest.approx(0.55 * F[0], rel=1e-6)    # the band binds
        assert F[0] == pytest.approx(F[1:].sum(), rel=1e-7)    # the balance holds

    @pytest.mark.parametrize("n_workers", [1, 2])
    def test_defects_give_the_flows_optimum_on_both_period_paths(
        self, tmp_path, monkeypatch, n_workers,
    ):
        """n_workers=1 reuses one Clarabel instance across the two identical periods
        (update(P, q, b)); n_workers=2 solves them concurrently."""
        run = _make_wide_sigma_run(periods=("2020", "2021"))
        _, flows   = self._solve(run, tmp_path, monkeypatch, "flows",   n_workers)
        _, defects = self._solve(run, tmp_path, monkeypatch, "defects", n_workers)
        self._assert_same_optimum(run, flows, defects)
        for t in defects:
            assert defects[t].F_opt[3] == pytest.approx(0.0, abs=1e-6), t

    def test_objective_keeps_the_flows_convention_under_both_choices(
        self, tmp_path, monkeypatch,
    ):
        """objective = J − Σ(a_prior/σ_a)² either way, so runs compare like with like;
        Clarabel's own cost (convergence) is J under "defects"."""
        run = _make_wide_sigma_run()
        _, flows   = self._solve(run, tmp_path, monkeypatch, "flows")
        _, defects = self._solve(run, tmp_path, monkeypatch, "defects")
        const = float(((run.a0 / run.sigma_a) ** 2).sum())
        J = self._J(run, "2020", defects["2020"].F_opt)
        assert defects["2020"].objective == pytest.approx(flows["2020"].objective, rel=1e-7)
        assert flows["2020"].objective + const == pytest.approx(J, rel=1e-6)
        assert defects["2020"].convergence["cost_primal"] == pytest.approx(J, rel=1e-6)

    def test_flows_is_the_default_and_the_sidecar_records_the_choice(
        self, tmp_path, monkeypatch,
    ):
        assert make_solver(const_dir=str(tmp_path / "c"),
                           opt_dir=str(tmp_path / "o")).qp_variables == "flows"
        run = _make_wide_sigma_run()
        s, results = self._solve(run, tmp_path, monkeypatch, "defects")
        out_id = s.persist(run_id=run.run_id, results=results,
                           output_dir=str(tmp_path / "opt"))
        with open(tmp_path / "opt" / out_id / "metadata.json") as fh:
            assert json.load(fh)["stage_config"]["qp_variables"] == "defects"

    def test_unknown_qp_variables_raises(self, tmp_path):
        with pytest.raises(ValueError, match="qp_variables"):
            make_solver(const_dir=str(tmp_path / "c"), opt_dir=str(tmp_path / "o"),
                        qp_variables="whitened")


class TestObjectiveAt:
    """#102: ``objective_at`` scores any year-local posterior under a const run's QP,
    so a sweep compares points on one objective (each answer under the base σ)."""

    _solve = staticmethod(TestQpVariables._solve)
    _J     = staticmethod(TestQpVariables._J)

    @pytest.mark.parametrize("qp_variables", ["flows", "defects"])
    def test_it_reproduces_the_solvers_objective_at_its_own_posterior(
        self, tmp_path, monkeypatch, qp_variables,
    ):
        run = _make_wide_sigma_run(periods=("2020", "2021"))
        s, res = self._solve(run, tmp_path, monkeypatch, qp_variables)
        for t, r in res.items():
            cols = run.edge_cols_by_t[t]
            const = float(((run.a0[cols] / run.sigma_a[cols]) ** 2).sum())
            got = s.objective_at(run, t, r.F_opt)
            assert got["total"] == pytest.approx(r.objective + const, rel=1e-10)
            assert got["total"] == pytest.approx(self._J(run, t, r.F_opt), rel=1e-10)

    def test_under_defects_it_is_clarabels_own_cost(self, tmp_path, monkeypatch):
        run = _make_wide_sigma_run()
        s, res = self._solve(run, tmp_path, monkeypatch, "defects")
        got = s.objective_at(run, "2020", res["2020"].F_opt)
        assert got["total"] == pytest.approx(res["2020"].convergence["cost_primal"],
                                             rel=1e-6)

    def test_a_feasible_perturbation_scores_higher(self, tmp_path, monkeypatch):
        """Raising e0 and e1 together keeps the balance and the band, so the point
        stays feasible and must cost more than the optimum (a nudge that breaks a
        hard row may cost less: the optimum is over the feasible set only)."""
        run = _make_wide_sigma_run()
        s, res = self._solve(run, tmp_path, monkeypatch, "defects")
        F = res["2020"].F_opt
        base = s.objective_at(run, "2020", F)["total"]
        for step in (0.5, 1.0):
            G = F.copy()
            G[[0, 1]] += step
            assert s.objective_at(run, "2020", G)["total"] > base

    def test_the_family_split_sums_to_the_total(self, tmp_path, monkeypatch):
        run = _make_wide_sigma_run()
        s, res = self._solve(run, tmp_path, monkeypatch, "flows")
        got = s.objective_at(run, "2020", res["2020"].F_opt)
        assert set(got) == {"total", "edges", "sum_rows", "soft_ratios", "other_soft",
                            "max_hard_violation"}
        assert got["max_hard_violation"] <= 1e-6
        parts = got["edges"] + got["sum_rows"] + got["soft_ratios"] + got["other_soft"]
        assert parts == pytest.approx(got["total"], rel=1e-12)
        assert got["sum_rows"] > 0 and got["soft_ratios"] == 0 and got["other_soft"] == 0

    def test_a_hand_worked_point(self, tmp_path):
        """Prior (100, 40, 80, −20, 0.002) at σ (5, 1e5, 2, 2, 1e-3); F = prior except
        e0 = 105: edges (5/5)² = 1. Rows: e0 = 110 at σ 5 → ((105 − 110)/5)² = 1;
        e1 + e2 = 105 at σ 3 → ((120 − 105)/3)² = 25. Total 27."""
        run = _make_wide_sigma_run()
        s = make_solver(const_dir=str(tmp_path / "c"), opt_dir=str(tmp_path / "o"))
        F = np.array([105.0, 40.0, 80.0, -20.0, 0.002])
        got = s.objective_at(run, "2020", F)
        assert got["edges"] == pytest.approx(1.0, rel=1e-12)
        assert got["sum_rows"] == pytest.approx(26.0, rel=1e-12)
        assert got["total"] == pytest.approx(27.0, rel=1e-12)
        # Hard rows: balance 105 − 40 − 80 + 20 − 0.002 = 4.998; band 80 − 0.55·105 =
        # 22.25 > 0; bound −e3 = 20 > 0. The largest breach is the band's.
        assert got["max_hard_violation"] == pytest.approx(22.25, rel=1e-12)

    def test_a_posterior_of_the_wrong_length_is_refused(self, tmp_path):
        run = _make_wide_sigma_run()
        s = make_solver(const_dir=str(tmp_path / "c"), opt_dir=str(tmp_path / "o"))
        with pytest.raises(ValueError, match="2020"):
            s.objective_at(run, "2020", np.zeros(4))


class TestPreFixBlockRunGuard:

    def _block_const_run(self):
        from pysuto.constraints import ConstraintRun
        G = sp.csr_matrix(np.eye(2))
        rm = pd.DataFrame({
            "constraint_id": ["a", "b"], "is_hard": [True, True],
            "sense": ["eq", "eq"], "time": ["2020", "2021"],
        })
        return ConstraintRun(
            G=G, row_meta=rm, a0=np.zeros(2), sigma_a=np.ones(2),
            u=np.full(2, 10.0), l=np.zeros(2),
            timestep_data={"2020": {"c": np.zeros(2), "sigma_c": np.zeros(2)}},
            run_id="c", acc_id="a", agg_id="g", created_at="2026-05-28T00:00:00",
            G_blocks={"2020": G, "2021": G},
            edge_cols_by_t={"2020": np.array([0]), "2021": np.array([1])},
        )

    def test_warns_on_full_vector_results_from_block_const(self):
        """A const run with G_blocks but full-vector (cols_t=None) opt results
        is a pre-fix block run carrying cross-year placeholders → warn."""
        from pysuto.registry.optim import warn_if_prefix_block_run
        from pysuto.solver import OptimResult
        const_run = self._block_const_run()
        results = {"2020": OptimResult(
            timestep="2020", F_opt=np.zeros(2), eps_opt=np.zeros(0),
            c_star=np.zeros(2), residuals=np.zeros(2), objective=0.0,
            iterations=0, solve_time_s=0.0, solver_status="Solved",
            convergence={},
        )}
        with pytest.warns(UserWarning, match="pre-fix block run"):
            warn_if_prefix_block_run(results, const_run)

    def test_silent_on_year_local_block_results(self, recwarn):
        from pysuto.registry.optim import warn_if_prefix_block_run
        from pysuto.solver import OptimResult
        const_run = self._block_const_run()
        results = {"2020": OptimResult(
            timestep="2020", F_opt=np.zeros(1), eps_opt=np.zeros(0),
            c_star=np.zeros(1), residuals=np.zeros(1), objective=0.0,
            iterations=0, solve_time_s=0.0, solver_status="Solved",
            convergence={}, cols_t=np.array([0]), rows_t=np.array([0]),
        )}
        warn_if_prefix_block_run(results, const_run)
        assert not any(issubclass(w.category, UserWarning) for w in recwarn.list)

    def test_silent_on_legacy_const(self, recwarn):
        """A const run with no G_blocks (legacy) + full-vector results is a
        legitimate legacy run → no warning."""
        from pysuto.registry.optim import warn_if_prefix_block_run
        const_run = self._block_const_run()
        const_run.G_blocks = {}
        const_run.edge_cols_by_t = {}
        from pysuto.solver import OptimResult
        results = {"2020": OptimResult(
            timestep="2020", F_opt=np.zeros(2), eps_opt=np.zeros(0),
            c_star=np.zeros(2), residuals=np.zeros(2), objective=0.0,
            iterations=0, solve_time_s=0.0, solver_status="Solved",
            convergence={},
        )}
        warn_if_prefix_block_run(results, const_run)
        assert not any(issubclass(w.category, UserWarning) for w in recwarn.list)


def _make_second_year_infeasible_run(bad=("2021",)):
    """One edge, two periods: a hard balance e0 = c and the bound −e0 ≤ 0. c = 5 is
    feasible; c = −5 in the years ``bad`` is not."""
    G_t = sp.csr_matrix(np.array([[1.0], [-1.0]]))
    data = {t: (np.array([-5.0 if t in bad else 5.0, 0.0]), np.zeros(2),
                np.array([5.0]), np.array([1.0])) for t in ("2020", "2021")}
    return _block_run(G_t, ["zero", "nonneg"], data, run_id="half_bad")


class TestFeasibilityCheckMode:
    """#104, ADR 0042: FEASIBILITY_CHECK = "all" | "none" — a sweep point skips the
    probe; a period that cannot be solved still raises, only not localised by it.
    The probe runs per period, so a bad later year is found before any solve."""

    @staticmethod
    def _solver(run, monkeypatch, **kw):
        s = make_solver(const_dir="unused", verbose_first_solve=False,
                        prior_mode="cold", **kw)
        monkeypatch.setattr(s, "load_run", lambda _rid: run)
        return s

    def test_all_probes_every_year_before_solving(self, monkeypatch):
        s = self._solver(_make_second_year_infeasible_run(), monkeypatch)
        with pytest.raises(RuntimeError, match="Feasibility check FAILED at timestep '2021'"):
            s.solve_all("half_bad", feasibility_check="all")

    def test_the_probe_is_the_default(self, monkeypatch):
        s = self._solver(_make_second_year_infeasible_run(), monkeypatch)
        with pytest.raises(RuntimeError, match="Feasibility check FAILED"):
            s.solve_all("half_bad")

    def test_the_probe_names_the_offending_rows(self, monkeypatch):
        s = self._solver(_make_second_year_infeasible_run(), monkeypatch)
        with pytest.raises(RuntimeError) as exc:
            s.solve_all("half_bad", feasibility_check="all")
        msg = str(exc.value)
        assert "Hard rows with the largest" in msg and "cid='r0_2021'" in msg
        assert "ADR 0019" in msg and "remove" not in msg
        assert "r0_2020" not in msg                     # the feasible year is not blamed

    def test_the_probe_runs_through_the_period_pool(self, monkeypatch):
        s = self._solver(_make_second_year_infeasible_run(), monkeypatch, n_workers=2)
        with pytest.raises(RuntimeError,
                           match="Feasibility check FAILED at timestep '2021'") as exc:
            s.solve_all("half_bad")
        assert "Parallel feasibility check failed in 1 period(s) ['2021']" in str(exc.value)

    def test_none_skips_the_probe_and_the_period_solve_still_raises(self, monkeypatch):
        s = self._solver(_make_second_year_infeasible_run(bad=("2020",)), monkeypatch)
        with pytest.raises(RuntimeError) as exc:
            s.solve_all("half_bad", feasibility_check="none")
        assert "Feasibility check FAILED" not in str(exc.value)
        assert "'2020'" in str(exc.value)

    def test_none_solves_a_feasible_run(self, monkeypatch):
        s = self._solver(_make_second_year_infeasible_run(bad=()), monkeypatch)
        res = s.solve_all("half_bad", feasibility_check="none")
        assert [r.F_opt[0] for r in res.values()] == pytest.approx([5.0, 5.0], rel=1e-7)

    def test_the_first_period_only_probe_is_gone(self, monkeypatch):
        s = self._solver(_make_second_year_infeasible_run(bad=()), monkeypatch)
        with pytest.raises(ValueError, match="feasibility_check='first'"):
            s.solve_all("half_bad", feasibility_check="first")
        with pytest.raises(TypeError):
            s.solve_all("half_bad", feasibility_check_first_only=True)

    def test_an_unknown_mode_is_refused(self, monkeypatch):
        s = self._solver(_make_second_year_infeasible_run(bad=()), monkeypatch)
        with pytest.raises(ValueError, match="feasibility_check"):
            s.solve_all("half_bad", feasibility_check="some")


class TestFeasibilityCheck:

    def test_infeasible_raises(self, monkeypatch):
        run = _make_infeasible_run()
        s   = make_solver(const_dir="unused", verbose_first_solve=False,
                          prior_mode="cold")
        monkeypatch.setattr(s, "load_run", lambda _rid: run)
        with pytest.raises(RuntimeError, match="Feasibility check FAILED"):
            s.solve_all("bad_run")

    def test_the_probe_is_the_default_and_passes_on_the_toy(self, toy_solver):
        """Default feasibility_check="all" probes every period and still solves them."""
        results = toy_solver.solve_all("toy_run")
        assert set(results.keys()) == {"2020", "2021"}


# ---------------------------------------------------------------------------
# The QP backend contract (ADR 0042)
# ---------------------------------------------------------------------------

def _parity_run():
    """Three edges, two periods: a hard balance ``a0 + a1 − a2 = 0``, two soft
    observations (``a0 = 6``, ``a1 = 2``) and the hard band ``a0 − a1 ≤ 0.5``, which
    binds (the prior alone has a0 − a1 = 1). Well conditioned, so a generic NLP solver
    reaches Clarabel's optimum."""
    G_t = sp.csr_matrix(np.array([[1.0, 1.0, -1.0], [1.0, 0.0, 0.0],
                                  [0.0, 1.0, 0.0], [1.0, -1.0, 0.0]]))
    data = {t: (np.array([0.0, 6.0 + k, 2.0, 0.5]), np.array([0.0, 0.5, 0.5, 0.0]),
                np.array([4.0, 3.0, 6.0]), np.ones(3))
            for k, t in enumerate(("2020", "2021"))}
    return _block_run(G_t, ["zero", "free", "free", "nonneg"], data, run_id="parity")


class _SlsqpBackend:
    """A tests-only QP backend on ``scipy.optimize.minimize(method="SLSQP")``."""
    name = "slsqp"

    def __init__(self, status="solved"):
        self._status = status

    @property
    def settings(self) -> dict:
        # 1e-14 ends in SLSQP exit mode 8 with SciPy 1.13.1's pip wheel, at the same x (#196)
        return {"method": "SLSQP", "ftol": 1e-12}

    def solve(self, problem, reuse=False, verbose=False):
        from scipy.optimize import minimize
        from pysuto.qp import QPResult
        P, q = problem.P.toarray(), problem.q
        A_eq, A_le = problem.A_eq.toarray(), problem.A_le.toarray()
        cons = []
        if A_eq.shape[0]:
            cons.append({"type": "eq", "fun": lambda x: A_eq @ x - problem.b_eq,
                         "jac": lambda x: A_eq})
        if A_le.shape[0]:
            cons.append({"type": "ineq", "fun": lambda x: problem.b_le - A_le @ x,
                         "jac": lambda x: -A_le})
        res = minimize(lambda x: 0.5 * x @ P @ x + q @ x, np.zeros(problem.n),
                       jac=lambda x: P @ x + q, constraints=cons, method="SLSQP",
                       options={"ftol": 1e-12, "maxiter": 500})
        return QPResult(x=res.x, status=self._status if res.success else "failed",
                        raw_status=str(res.message), iterations=int(res.nit),
                        solve_time_s=0.0, objective=float(res.fun))


class _StubBackend:
    """Returns a fixed status without solving."""
    name = "stub"
    settings = {}

    def __init__(self, status, raw_status):
        self._status, self._raw = status, raw_status

    def solve(self, problem, reuse=False, verbose=False):
        from pysuto.qp import QPResult
        return QPResult(x=np.zeros(problem.n), status=self._status,
                        raw_status=self._raw, iterations=3, solve_time_s=0.0,
                        objective=0.0)


class TestQPBackendContract:

    @staticmethod
    def _solve(backend, run, **kw):
        from pysuto.solver import Solver
        s = Solver(backend, const_dir="unused", verbose_first_solve=False,
                   prior_mode="cold", **kw)
        s.load_run = lambda _rid: run
        return s, s.solve_all(run.run_id, feasibility_check="none")

    @pytest.mark.parametrize("qp_variables", ["flows", "defects"])
    def test_a_slsqp_backend_matches_clarabel(self, qp_variables):
        """Soft rows, a hard equality and a binding ``≤`` row: the orchestrator's QP
        has the same optimum whichever backend solves it."""
        run = _parity_run()
        _, ref = self._solve(make_solver().backend, run, qp_variables=qp_variables)
        _, alt = self._solve(_SlsqpBackend(), run, qp_variables=qp_variables)
        for t in ref:
            assert np.allclose(alt[t].F_opt, ref[t].F_opt, rtol=1e-6, atol=1e-6), t
            assert np.allclose(alt[t].eps_opt, ref[t].eps_opt, rtol=1e-6, atol=1e-6), t
            assert alt[t].objective == pytest.approx(ref[t].objective, rel=1e-6)
            assert alt[t].backend == "slsqp" and ref[t].backend == "clarabel"
            F = ref[t].F_opt
            assert F[0] - F[1] == pytest.approx(0.5, abs=1e-6)       # the band binds
            assert F[0] + F[1] - F[2] == pytest.approx(0.0, abs=1e-6)

    def test_a_failed_status_raises_naming_the_raw_status(self):
        with pytest.raises(RuntimeError, match=r"stub status=MaxIterations.*no fallback"):
            self._solve(_StubBackend("failed", "MaxIterations"), _parity_run())

    def test_solved_inaccurate_is_accepted(self):
        _, res = self._solve(_SlsqpBackend(status="solved_inaccurate"), _parity_run())
        assert {r.solver_status for r in res.values()} == {"solved_inaccurate"}

    def test_the_backend_status_is_kept_beside_the_neutral_one(self):
        _, res = self._solve(make_solver().backend, _parity_run())
        r = res["2020"]
        assert (r.solver_status, r.backend, r.backend_status) == ("solved", "clarabel",
                                                                  "Solved")
        assert r.convergence["status"] == "Solved"

    def test_the_orchestrator_and_the_neutral_modules_import_no_solver(self):
        import ast
        import pysuto.qp
        import pysuto.solver
        import pysuto.registry.optim
        for mod in (pysuto.qp, pysuto.solver, pysuto.registry.optim):
            tree = ast.parse(open(mod.__file__).read())
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported += [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    imported.append(node.module or "")
            assert not any(m.split(".")[0] == "clarabel" for m in imported), mod

    def test_a_persisted_run_loads_with_clarabel_unimportable(self, tmp_path, monkeypatch):
        """Reading an opt run needs no numerical solver: with ``clarabel`` blocked and the
        pysuto modules freshly imported, the registry still loads it."""
        import sys
        s, res = self._solve(make_solver().backend, _parity_run())
        out_id = s.persist(run_id="parity", results=res, output_dir=str(tmp_path / "opt"),
                           const_run_id="c")
        for name in [m for m in sys.modules if m == "pysuto" or m.startswith("pysuto.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setitem(sys.modules, "clarabel", None)
        from pysuto.registry import OptimRunRegistry
        loaded = OptimRunRegistry(str(tmp_path / "opt")).load(out_id)
        assert set(loaded) == {"2020", "2021"}
        assert loaded["2020"].solver_status == "solved"
        assert "pysuto.clarabel_backend" not in sys.modules

    def test_a_run_written_before_the_split_loads_with_neutral_statuses(self, tmp_path):
        """The pre-change format: Clarabel's status in ``solver_status``, no backend
        fields."""
        from pysuto.registry import OptimRunRegistry
        s, res = self._solve(make_solver().backend, _parity_run())
        out_id = s.persist(run_id="parity", results=res, output_dir=str(tmp_path),
                           const_run_id="c")
        path = tmp_path / out_id / "metadata.json"
        sidecar = json.loads(path.read_text())
        for entry in sidecar["stage_summary"]["per_timestep"].values():
            del entry["backend"], entry["backend_status"]
            entry["solver_status"] = "AlmostSolved"
        sidecar["stage_summary"]["per_timestep"]["2021"]["solver_status"] = "Solved"
        path.write_text(json.dumps(sidecar))

        loaded = OptimRunRegistry(str(tmp_path)).load(out_id)
        assert loaded["2020"].solver_status == "solved_inaccurate"
        assert loaded["2021"].solver_status == "solved"
        assert {r.backend for r in loaded.values()} == {"clarabel"}
        assert loaded["2020"].backend_status == "AlmostSolved"

    def test_the_sidecar_records_the_backend_and_its_settings(self, tmp_path):
        s, res = self._solve(make_solver(tol_gap_rel=1e-7, max_iter=321).backend,
                             _parity_run())
        out_id = s.persist(run_id="parity", results=res, output_dir=str(tmp_path),
                           const_run_id="c")
        sidecar = json.loads((tmp_path / out_id / "metadata.json").read_text())
        cfg = sidecar["stage_config"]
        assert cfg["backend"] == "clarabel"
        assert cfg["backend_settings"]["tol_gap_rel"] == 1e-7
        assert cfg["backend_settings"]["max_iter"] == 321
        assert cfg["backend_settings"]["direct_solve_method"] == "faer"
        assert cfg["max_threads"] == 1 and cfg["direct_solve_method"] == "faer"
        for entry in sidecar["stage_summary"]["per_timestep"].values():
            assert (entry["solver_status"], entry["backend"], entry["backend_status"]) \
                == ("solved", "clarabel", "Solved")


class TestClarabelBackend:
    """What the backend owns: the cones, the status map, factorisation reuse and the
    cold retry that reuse makes necessary."""

    @staticmethod
    def _problem(shift=0.0, n=3):
        """min ½‖x − t‖² s.t. x0 + x1 = 1, x2 ≤ 2 (a two-row, ``n``-variable QP)."""
        from pysuto.qp import QPProblem
        target = np.arange(n, dtype=float) + shift
        return QPProblem(P=sp.eye(n, format="csc"), q=-target,
                         A_eq=sp.csr_matrix(([1.0, 1.0], ([0, 0], [0, 1])), shape=(1, n)),
                         b_eq=np.array([1.0]),
                         A_le=sp.csr_matrix(([1.0], ([0], [2])), shape=(1, n)),
                         b_le=np.array([2.0]), n_a=n, n_eps=0)

    @staticmethod
    def _count_inits(monkeypatch):
        """Wrap clarabel.DefaultSolver; return the list of ``(instance, settings)``.
        An instance's ``stall`` flag makes its next ``solve()`` report NumericalError."""
        import types
        import pysuto.clarabel_backend as mod
        built = []
        original = mod.clarabel.DefaultSolver

        class _Instance:
            stall = False
            def __init__(self, inner):
                self._inner = inner
            def solve(self):
                if self.stall:
                    return types.SimpleNamespace(status="NumericalError", x=None,
                                                 iterations=0, obj_val=float("nan"))
                return self._inner.solve()
            def __getattr__(self, name):
                return getattr(self._inner, name)

        def factory(*args, **kwargs):
            built.append((_Instance(original(*args, **kwargs)), args[5]))
            return built[-1][0]
        monkeypatch.setattr(mod.clarabel, "DefaultSolver", factory)
        return built

    def test_equalities_are_zero_cones_and_le_rows_nonneg(self):
        from pysuto.clarabel_backend import ClarabelBackend
        res = ClarabelBackend().solve(self._problem(shift=1.0))
        assert (res.status, res.raw_status) == ("solved", "Solved")
        assert res.x[2] == pytest.approx(2.0, abs=1e-7)              # 3 clipped to ≤ 2
        assert res.x[0] + res.x[1] == pytest.approx(1.0, abs=1e-7)
        assert res.info["status"] == "Solved"
        assert res.iter_history is not None
        assert res.iter_history.shape[0] == res.iterations + 1

    def test_a_failed_solve_maps_to_failed_with_the_raw_status(self):
        from pysuto.clarabel_backend import ClarabelBackend
        res = ClarabelBackend(max_iter=1).solve(self._problem(shift=1.0))
        assert res.status == "failed" and res.raw_status == "MaxIterations"

    def test_history_is_optional(self):
        from pysuto.clarabel_backend import ClarabelBackend
        res = ClarabelBackend(capture_iter_history=False).solve(self._problem())
        assert res.iter_history is None and res.info is not None

    def test_reuse_keeps_the_instance_for_the_same_structure_only(self, monkeypatch):
        from pysuto.clarabel_backend import ClarabelBackend
        built = self._count_inits(monkeypatch)
        be = ClarabelBackend()
        be.solve(self._problem())
        be.solve(self._problem(shift=0.5), reuse=True)               # same A: update()
        assert len(built) == 1
        be.solve(self._problem(shift=0.7), reuse=False)              # asked afresh
        assert len(built) == 2
        be.solve(self._problem(n=4), reuse=True)                     # other structure
        assert len(built) == 3

    def test_a_stalled_reused_solve_is_retried_cold(self, monkeypatch):
        from pysuto.clarabel_backend import ClarabelBackend
        built = self._count_inits(monkeypatch)
        be = ClarabelBackend()
        be.solve(self._problem())
        built[0][0].stall = True
        res = be.solve(self._problem(shift=0.5), reuse=True)
        assert len(built) == 2 and res.status == "solved"            # the fresh one won

    def test_a_failure_on_a_fresh_instance_is_not_retried(self, monkeypatch):
        from pysuto.clarabel_backend import ClarabelBackend
        built = self._count_inits(monkeypatch)
        res = ClarabelBackend(max_iter=1).solve(self._problem(shift=1.0), reuse=True)
        assert res.status == "failed" and len(built) == 1

    def test_the_probe_uses_looser_tolerances_and_the_period_settings(self, monkeypatch):
        from pysuto.clarabel_backend import ClarabelBackend
        built = self._count_inits(monkeypatch)
        be = ClarabelBackend(tol_gap_rel=1e-9, direct_solve_method="qdldl")
        probe = self._problem()
        probe.feasibility_only = True
        be.solve(probe)
        be.solve(self._problem())
        st_probe, st_period = built[0][1], built[1][1]
        assert (st_probe.tol_feas, st_probe.tol_gap_rel) == (1e-7, 1e-7)
        assert (st_period.tol_feas, st_period.tol_gap_rel) == (1e-8, 1e-9)
        assert st_probe.direct_solve_method == st_period.direct_solve_method == "qdldl"
        assert st_probe.max_threads == st_period.max_threads == 1


# ---------------------------------------------------------------------------
# Row realisation (#11): (sense, is_hard) → layout
# ---------------------------------------------------------------------------

class TestRowLayout:
    """One test per cell of the {sense} × {is_hard} table."""

    def test_soft_eq_is_a_slack_row_first(self):
        from pysuto.solver import row_layout
        lay = row_layout(["eq"], [False])
        assert (lay.n_slack, lay.n_eq, lay.n_le) == (1, 0, 0)

    def test_hard_eq_is_an_equality_row(self):
        from pysuto.solver import row_layout
        lay = row_layout(["eq"], [True])
        assert (lay.n_slack, lay.n_eq, lay.n_le) == (0, 1, 0)

    def test_hard_le_is_an_inequality_row(self):
        from pysuto.solver import row_layout
        lay = row_layout(["le"], [True])
        assert (lay.n_slack, lay.n_eq, lay.n_le) == (0, 0, 1)

    def test_soft_le_is_not_implemented(self):
        from pysuto.solver import row_layout
        with pytest.raises(NotImplementedError, match="soft inequality"):
            row_layout(["eq", "le"], [True, False])

    def test_unknown_sense_raises(self):
        from pysuto.solver import row_layout
        with pytest.raises(ValueError, match="unknown row sense"):
            row_layout(["eq", "ge"], [True, True])

    def test_perm_orders_soft_then_hard_eq_then_le_keeping_original_order(self):
        from pysuto.solver import row_layout
        sense   = ["le", "eq", "eq", "le", "eq", "eq"]
        is_hard = [True, True, False, True, False, True]
        lay = row_layout(sense, is_hard)
        np.testing.assert_array_equal(lay.perm, [2, 4, 1, 5, 0, 3])
        assert (lay.n_slack, lay.n_eq, lay.n_le) == (2, 2, 2)
