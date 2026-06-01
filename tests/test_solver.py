"""
test_solver.py
==============
Test suite for :class:`pysuto.solver.ClarabelSolver` under the structural
model.

Toy problem (3 edges, 2 periods)
--------------------------------
edges:  a = [a0, a1, a2]
bounds: l = [0, 0, 0],   u = [10, 10, 10]
prior:  a_0 = [1, 2, 3], sigma_a = [1, 1, 1]   (held constant across t)

Structural constraint rows (3 total):
  zero   (hard): a0 + a1 - a2 = 0           (balance)
  free   (soft): a0           = c_obs[t]    (observation, σ_c = 0.5)
  nonneg (hard): a0 - 2 a1   <= 0          (ratio bound)

Per-period overlays:
  c       [t=2020] = [0, 1.0, 0],  σ_c = [0, 0.5, 0]
  c       [t=2021] = [0, 1.5, 0],  σ_c = [0, 0.5, 0]
  active  : every row True at every period.

Everything is wired through StructG + 2-D c/sigma_c/active so the tests
exercise the structural pieces directly.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers — direct ConstraintRun stubs (no engine, no disk)
# ---------------------------------------------------------------------------

def _make_struct_g(G_dense: np.ndarray, n_timesteps: int, row_meta: pd.DataFrame):
    """Wrap a 2-D dense matrix in a single-period StructG broadcast across t."""
    import scipy.sparse as sp
    from pysuto.constraints import StructG

    G_csr = sp.csr_matrix(G_dense)
    data  = np.broadcast_to(G_csr.data[:, None], (G_csr.nnz, n_timesteps)).copy()
    return StructG(
        indptr   = G_csr.indptr,
        indices  = G_csr.indices,
        data     = data,
        shape    = tuple(G_csr.shape),
        row_meta = row_meta,
    )


def _make_toy_run(run_id: str = "toy_run"):
    from pysuto.constraints import ConstraintRun

    timesteps = ["2020", "2021"]
    G_dense = np.array([
        [ 1.0,  1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 1.0, -2.0,  0.0],
    ])
    row_meta = pd.DataFrame({
        "constraint_id": ["balance", "obs_a0", "ratio_bound"],
        "is_hard":       [True, False, True],
        "cone_type":     ["zero", "free", "nonneg"],
    })
    G = _make_struct_g(G_dense, n_timesteps=len(timesteps), row_meta=row_meta)

    n_edges = 3
    sigma_a = np.tile(np.array([1.0, 1.0, 1.0])[:, None], (1, len(timesteps)))
    c       = np.array([[0.0, 0.0],
                        [1.0, 1.5],
                        [0.0, 0.0]])
    sigma_c = np.array([[0.0, 0.0],
                        [0.5, 0.5],
                        [0.0, 0.0]])
    active  = np.ones((3, len(timesteps)), dtype=bool)

    return ConstraintRun(
        G          = G,
        row_meta   = row_meta,
        a0         = np.array([1.0, 2.0, 3.0]),
        sigma_a    = sigma_a,
        u          = np.array([10.0, 10.0, 10.0]),
        l          = np.array([0.0, 0.0, 0.0]),
        c          = c,
        sigma_c    = sigma_c,
        active     = active,
        timesteps  = timesteps,
        run_id     = run_id,
        acc_id     = "toy_acc",
        agg_id     = "toy_agg",
        created_at = "2026-04-29T00:00:00",
    )


def _make_infeasible_run(run_id: str = "bad_run"):
    """Hard nonneg row 'a0 <= -1' contradicts the l=0 lower bound."""
    from pysuto.constraints import ConstraintRun

    timesteps = ["2020"]
    row_meta = pd.DataFrame({
        "constraint_id": ["bad_bound"],
        "is_hard":       [True],
        "cone_type":     ["nonneg"],
    })
    G = _make_struct_g(np.array([[1.0, 0.0]]), n_timesteps=1, row_meta=row_meta)

    return ConstraintRun(
        G          = G,
        row_meta   = row_meta,
        a0         = np.zeros(2),
        sigma_a    = np.ones((2, 1)),
        u          = np.full(2, 10.0),
        l          = np.zeros(2),
        c          = np.array([[-1.0]]),
        sigma_c    = np.array([[0.0]]),
        active     = np.array([[True]]),
        timesteps  = timesteps,
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

    def test_result_shapes(self, toy_solver, toy_run):
        """a_star is full-length (n_edges,); c_star/residuals indexed by row."""
        results = toy_solver.solve_all("toy_run")
        n_edges = toy_run.n_edges
        M       = toy_run.n_constraints
        for r in results.values():
            assert r.a_star.shape    == (n_edges,)
            assert r.residuals.shape == (M,)
            assert r.c_star.shape    == (M,)
            # ``F_opt`` survives as an alias for back-compat.
            assert np.shares_memory(r.F_opt, r.a_star)

    def test_status_solved(self, toy_solver):
        results = toy_solver.solve_all("toy_run")
        for r in results.values():
            assert r.solver_status in {"Solved", "AlmostSolved"}

    def test_a_star_within_bounds(self, toy_solver, toy_run):
        results = toy_solver.solve_all("toy_run")
        for r in results.values():
            assert np.all(r.a_star >= toy_run.l - 1e-8)
            assert np.all(r.a_star <= toy_run.u + 1e-8)

    def test_hard_residuals_satisfied(self, toy_solver, toy_run):
        results = toy_solver.solve_all("toy_run")
        hard_mask = toy_run.row_meta["is_hard"].to_numpy()
        for r in results.values():
            # Hard rows: residual ≈ 0 (zero) or ≤ 0 (nonneg).
            for i, is_hard in enumerate(hard_mask):
                if not is_hard:
                    continue
                cone = toy_run.row_meta["cone_type"].iloc[i]
                if cone == "zero":
                    assert abs(r.residuals[i]) < 1e-6
                else:
                    assert r.residuals[i] <= 1e-6

    def test_filter_timesteps(self, toy_solver):
        results = toy_solver.solve_all("toy_run", timesteps=["2020"])
        assert list(results.keys()) == ["2020"]

    def test_unknown_timestep_raises(self, toy_solver):
        with pytest.raises(KeyError):
            toy_solver.solve_all("toy_run", timesteps=["2099"])


# ---------------------------------------------------------------------------
# Prior seam (AggregationPrior vs WarmStartPrior)
# ---------------------------------------------------------------------------

class TestPriorSeam:

    def test_warmstart_uses_a0_at_t0(self, toy_solver):
        """At t_idx=0 WarmStartPrior returns ``run.a0``."""
        from pysuto.solver import WarmStartPrior
        run = toy_solver.load_run("toy_run")
        prior = WarmStartPrior(run.a0)
        assert np.array_equal(prior.mean(0, []), run.a0)

    def test_warmstart_carries_previous_optimum(self, toy_solver):
        """At t_idx=1 WarmStartPrior returns the last history entry."""
        from pysuto.solver import WarmStartPrior
        run = toy_solver.load_run("toy_run")
        prior = WarmStartPrior(run.a0)
        fake_prev = np.array([7.0, 8.0, 9.0])
        assert np.array_equal(prior.mean(1, [fake_prev]), fake_prev)

    def test_aggregation_prior_returns_agg_slice(self):
        from pysuto.solver import AggregationPrior
        a_agg = np.array([[1.0, 2.0], [3.0, 4.0]])
        prior = AggregationPrior(a_agg)
        assert np.array_equal(prior.mean(0, []), [1.0, 3.0])
        assert np.array_equal(prior.mean(1, []), [2.0, 4.0])

    def test_warmstart_solve_differs_from_aggregation(self, toy_solver):
        """The two prior policies must produce DIFFERENT optima at t=2021."""
        from pysuto.solver import AggregationPrior, WarmStartPrior
        run = toy_solver.load_run("toy_run")
        # Make the AggregationPrior visibly different from the warm-start prior.
        a_agg = np.tile(np.array([5.0, 4.0, 3.0])[:, None], (1, 2))
        warm = toy_solver.solve_all("toy_run", prior=WarmStartPrior(run.a0))
        agg  = toy_solver.solve_all("toy_run", prior=AggregationPrior(a_agg))
        assert not np.allclose(warm["2021"].a_star, agg["2021"].a_star)


# ---------------------------------------------------------------------------
# Activation mask
# ---------------------------------------------------------------------------

class TestActivationMask:

    def test_inactive_rows_dropped_from_qp(self, toy_solver, toy_run):
        """Marking the soft observation row inactive at t=2021 must leave the
        prior unmodified (no observation pulls a0)."""
        run = toy_run
        # Deactivate the soft observation row at t=2021.
        run.active = run.active.copy()
        run.active[1, 1] = False

        from pysuto.solver import WarmStartPrior
        results = toy_solver.solve_all("toy_run", prior=WarmStartPrior(run.a0))

        # At t=2021, the OptimResult records the deactivated row.
        active_2021 = results["2021"].active
        assert active_2021 is not None
        assert active_2021[1] == False
        # eps on row 1 stays 0 (inactive ⇒ no slack variable).
        assert results["2021"].residuals[1] == 0.0


# ---------------------------------------------------------------------------
# Convergence capture
# ---------------------------------------------------------------------------

class TestConvergenceCapture:

    def test_convergence_dict_complete(self, toy_solver):
        results = toy_solver.solve_all("toy_run")
        keys = {"cost_primal", "cost_dual", "gap_abs", "res_primal",
                "res_dual", "ktratio", "mu", "step_length", "iterations",
                "status"}
        for r in results.values():
            assert keys.issubset(r.convergence.keys())

    def test_iter_history_dtype(self, toy_solver):
        results = toy_solver.solve_all("toy_run")
        for r in results.values():
            if r.iter_history is None:
                continue
            assert r.iter_history.dtype.names == (
                "iter", "cost_primal", "cost_dual", "gap_abs",
                "res_primal", "res_dual", "ktratio", "mu", "step_length",
            )

    def test_iter_history_disabled(self, monkeypatch, toy_run):
        from pysuto.solver import ClarabelSolver
        s = ClarabelSolver(const_dir="unused", verbose_first_solve=False,
                           capture_iter_history=False)
        monkeypatch.setattr(s, "load_run", lambda _rid: toy_run)
        results = s.solve_all("toy_run")
        for r in results.values():
            assert r.iter_history is None


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------

class TestPersist:

    def test_roundtrip(self, toy_solver, tmp_path):
        from pysuto.registry import OptimRunRegistry
        results   = toy_solver.solve_all("toy_run")
        registry  = OptimRunRegistry(str(tmp_path / "opt"))
        opt_id    = registry.register(
            results,
            parents={"inputs": {}, "upstream": {"const_run_id": "toy_run"}},
            label="rt",
        )

        reloaded = OptimRunRegistry(str(tmp_path / "opt")).load(opt_id)
        assert set(reloaded.keys()) == set(results.keys())
        for t in results:
            np.testing.assert_allclose(reloaded[t].a_star,    results[t].a_star)
            np.testing.assert_allclose(reloaded[t].residuals, results[t].residuals)
            assert reloaded[t].solver_status == results[t].solver_status

    def test_persist_via_solver(self, toy_solver, tmp_path):
        results = toy_solver.solve_all("toy_run")
        opt_id  = toy_solver.persist(
            "toy_run", results,
            output_dir=str(tmp_path / "opt"),
            const_run_id="toy_run", label="via_solver",
        )
        assert opt_id.startswith("opt_")


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

class TestFeasibilityCheck:

    def test_infeasible_raises(self, monkeypatch):
        from pysuto.solver import ClarabelSolver
        run = _make_infeasible_run()
        s = ClarabelSolver(const_dir="unused", verbose_first_solve=False)
        monkeypatch.setattr(s, "load_run", lambda _rid: run)
        with pytest.raises(RuntimeError, match="Feasibility check FAILED"):
            s.solve_all("bad_run")

    def test_first_only_is_default(self, toy_solver):
        # Should not raise; first period feasibility ok.
        toy_solver.solve_all("toy_run", feasibility_check_first_only=True)

    def test_check_all_timesteps(self, toy_solver):
        toy_solver.solve_all("toy_run", feasibility_check_first_only=False)
