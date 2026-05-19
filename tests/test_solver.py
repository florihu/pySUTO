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
