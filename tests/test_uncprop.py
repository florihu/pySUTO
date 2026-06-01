"""
test_uncprop.py
===============
Test suite for :mod:`pysuto.uncprop` (Stage 5) under the structural model.

Tests fall into three buckets:

* Math (closed-form 2×2; PSD + monotone shrinkage on a random sparse run)
* CI clipping (lower clipped to ``l``, upper clipped to ``u``)
* Plumbing (end-to-end engine→solver→uncprop with full registry round-trip)

All fabrications go through the structural ``ConstraintRun`` (StructG +
2-D c/sigma_c/sigma_a/active) — no pre-refactor block fixtures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fabrication helpers
# ---------------------------------------------------------------------------

def _struct_g(G_dense: np.ndarray, n_timesteps: int, row_meta: pd.DataFrame):
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


def _toy_const_run(
    run_id:       str,
    G_dense:      np.ndarray,
    row_meta:     pd.DataFrame,
    a0:           np.ndarray,
    sigma_a:      np.ndarray,
    c:            np.ndarray,
    sigma_c:      np.ndarray,
    timesteps:    list[str],
    u:            np.ndarray | None = None,
    l:            np.ndarray | None = None,
    active:       np.ndarray | None = None,
):
    """Build a ConstraintRun from per-period 2-D arrays."""
    from pysuto.constraints import ConstraintRun
    n_edges = G_dense.shape[1]
    n_t     = len(timesteps)
    if u is None: u = np.full(n_edges,  1e9)
    if l is None: l = np.full(n_edges, -1e9)
    sigma_a_2d = (sigma_a if sigma_a.ndim == 2
                  else np.tile(sigma_a[:, None], (1, n_t)))
    if active is None:
        active = np.ones((G_dense.shape[0], n_t), dtype=bool)
    G = _struct_g(G_dense, n_timesteps=n_t, row_meta=row_meta)
    return ConstraintRun(
        G          = G,
        row_meta   = row_meta,
        a0         = a0,
        sigma_a    = sigma_a_2d,
        u          = u,
        l          = l,
        c          = c,
        sigma_c    = sigma_c,
        active     = active,
        timesteps  = timesteps,
        run_id     = run_id,
        acc_id     = "toy_acc",
        agg_id     = "toy_agg",
        created_at = "2026-05-26T00:00:00",
    )


def _make_toy_2x2(run_id: str = "toy_2x2"):
    """2 edges, 1 hard equality (a0 = a1). Closed-form posterior is
    σ²_{a*} = 0.5 → σ_{a*} = √0.5 on both edges."""
    return _toy_const_run(
        run_id    = run_id,
        G_dense   = np.array([[1.0, -1.0]]),
        row_meta  = pd.DataFrame({
            "constraint_id": ["balance"],
            "is_hard":       [True],
            "cone_type":     ["zero"],
        }),
        a0        = np.array([1.0, 1.0]),
        sigma_a   = np.array([1.0, 1.0]),
        c         = np.array([[0.0]]),
        sigma_c   = np.array([[0.0]]),
        timesteps = ["2020"],
        u         = np.full(2,  10.0),
        l         = np.full(2, -10.0),
    )


def _make_toy_3edge_2t(run_id: str = "toy_3e"):
    """3 edges, mixed cones, 2 periods. σ_c on the soft row varies in t."""
    G_dense = np.array([
        [ 1.0,  1.0, -1.0],   # zero,   hard
        [ 1.0,  0.0,  0.0],   # free,   soft
        [ 1.0, -2.0,  0.0],   # nonneg, hard
    ])
    row_meta = pd.DataFrame({
        "constraint_id": ["balance", "obs_a0", "ratio_bound"],
        "is_hard":       [True, False, True],
        "cone_type":     ["zero", "free", "nonneg"],
    })
    c = np.array([[0.0, 0.0],
                  [1.0, 1.5],
                  [0.0, 0.0]])
    sigma_c = np.array([[0.0, 0.0],
                        [0.5, 0.7],
                        [0.0, 0.0]])
    return _toy_const_run(
        run_id    = run_id,
        G_dense   = G_dense,
        row_meta  = row_meta,
        a0        = np.array([1.0, 2.0, 3.0]),
        sigma_a   = np.array([1.0, 1.0, 1.0]),
        c         = c,
        sigma_c   = sigma_c,
        timesteps = ["2020", "2021"],
        u         = np.full(3, 10.0),
        l         = np.zeros(3),
    )


def _solve_and_persist(const_run, tmp_path):
    from pysuto.solver import ClarabelSolver
    from pysuto.registry import ConstraintRunRegistry

    const_dir   = str(tmp_path / "const")
    opt_dir     = str(tmp_path / "opt")
    uncprop_dir = str(tmp_path / "uncprop")

    const_reg = ConstraintRunRegistry(const_dir)
    const_reg.register(const_run)

    solver  = ClarabelSolver(
        const_dir=const_dir, opt_dir=opt_dir, verbose_first_solve=False,
        capture_iter_history=False,
    )
    results = solver.solve_all(const_run.run_id)
    opt_run_id = solver.persist(
        const_run.run_id, results, output_dir=opt_dir,
        const_run_id=const_run.run_id,
    )
    return opt_run_id, {
        "const_dir":   const_dir,
        "opt_dir":     opt_dir,
        "uncprop_dir": uncprop_dir,
    }


# ---------------------------------------------------------------------------
# Math: closed form
# ---------------------------------------------------------------------------

class TestToyClosedForm:

    def test_2x2_marginal_matches_hand_calc(self, tmp_path):
        from pysuto.uncprop import KalmanUncProp
        run = _make_toy_2x2()
        opt_id, dirs = _solve_and_persist(run, tmp_path)
        uncprop = KalmanUncProp(**dirs)
        result  = uncprop.compute(opt_id, alpha=0.05)
        ts = result.per_timestep["2020"]
        # Closed form: G = [1, -1], σ_a = [1, 1], σ_c = 0 + ridge.
        # S ≈ 2 → d = 0.5 → σ²_a* = 0.5 → σ_a* ≈ √0.5.
        np.testing.assert_allclose(ts.sigma_a_star, np.sqrt(0.5), atol=1e-5)


# ---------------------------------------------------------------------------
# PSD + monotone shrinkage on a small random sparse run
# ---------------------------------------------------------------------------

class TestPsdAndMonotone:

    def test_random_sparse_invariants(self, tmp_path):
        from pysuto.uncprop import KalmanUncProp
        rng = np.random.default_rng(7)
        n_edges = 8
        n_rows  = 5

        # Random sparse G with mixed cones.
        G_dense = rng.standard_normal((n_rows, n_edges)) * (rng.random((n_rows, n_edges)) < 0.4)
        # Ensure no all-zero rows.
        for i in range(n_rows):
            if not G_dense[i].any():
                G_dense[i, 0] = 1.0

        row_meta = pd.DataFrame({
            "constraint_id": [f"r{i}" for i in range(n_rows)],
            "is_hard":       [True, False, True, False, False],
            "cone_type":     ["zero", "free", "nonneg", "free", "free"],
        })
        sigma_a = rng.uniform(0.5, 2.0, size=n_edges)
        n_t     = 1
        a0      = rng.uniform(1.0, 5.0, size=n_edges)
        c       = np.zeros((n_rows, n_t))
        sigma_c = np.where(
            np.array([True, False, True, False, False])[:, None],
            0.0,
            rng.uniform(0.5, 2.0, size=(n_rows, n_t)),
        )

        run = _toy_const_run(
            run_id="rnd", G_dense=G_dense, row_meta=row_meta, a0=a0,
            sigma_a=sigma_a, c=c, sigma_c=sigma_c, timesteps=["2020"],
            u=np.full(n_edges, 1e6), l=np.full(n_edges, -1e6),
        )

        opt_id, dirs = _solve_and_persist(run, tmp_path)
        uncprop = KalmanUncProp(**dirs)
        res = uncprop.compute(opt_id, alpha=0.05)
        ts  = res.per_timestep["2020"]

        # PSD: σ_a* ≥ 0 everywhere.
        assert np.all(ts.sigma_a_star >= 0.0)
        # Monotone shrinkage: σ_a* ≤ σ_a (within numerical tolerance).
        assert np.all(ts.sigma_a_star <= sigma_a + 1e-7)
        # trace(post) ≤ trace(prior).
        trace_prior = float((sigma_a ** 2).sum())
        trace_post  = float((ts.sigma_a_star ** 2).sum())
        assert trace_post <= trace_prior + 1e-7


# ---------------------------------------------------------------------------
# Hard-residual diagnostic
# ---------------------------------------------------------------------------

class TestHardConstraintResidual:

    def test_diagnostics_max_resid_hard_small(self, tmp_path):
        from pysuto.uncprop import KalmanUncProp
        run = _make_toy_3edge_2t()
        opt_id, dirs = _solve_and_persist(run, tmp_path)
        uncprop = KalmanUncProp(**dirs)
        res     = uncprop.compute(opt_id)
        for t, ts in res.per_timestep.items():
            assert ts.diagnostics["max_resid_hard"] < 1e-6, (
                f"t={t}: max_resid_hard={ts.diagnostics['max_resid_hard']:.3e}"
            )


# ---------------------------------------------------------------------------
# CI clipping
# ---------------------------------------------------------------------------

class TestCIClipping:

    def test_lower_clipped_to_l_upper_clipped_to_u(self, tmp_path):
        from pysuto.uncprop import KalmanUncProp
        run = _make_toy_3edge_2t()
        # Force tight bounds so clipping must happen.
        run.l = np.full(3, 0.5)
        run.u = np.full(3, 2.5)
        opt_id, dirs = _solve_and_persist(run, tmp_path)
        uncprop = KalmanUncProp(**dirs)
        res = uncprop.compute(opt_id, alpha=0.05)
        for ts in res.per_timestep.values():
            assert np.all(ts.lower >= run.l - 1e-12)
            assert np.all(ts.upper <= run.u + 1e-12)


# ---------------------------------------------------------------------------
# Activation: inactive rows shouldn't contribute to shrinkage
# ---------------------------------------------------------------------------

class TestActivationAffectsShrinkage:

    def test_deactivating_soft_row_reduces_shrinkage(self, tmp_path):
        from pysuto.uncprop import KalmanUncProp

        run_full = _make_toy_3edge_2t()
        opt_id_full, dirs_full = _solve_and_persist(run_full, tmp_path / "full")
        unc_full = KalmanUncProp(**dirs_full).compute(opt_id_full, alpha=0.05)

        # Same setup but soft row (row 1) deactivated at every period.
        run_off = _make_toy_3edge_2t(run_id="toy_3e_off")
        run_off.active = run_off.active.copy()
        run_off.active[1, :] = False
        opt_id_off, dirs_off = _solve_and_persist(run_off, tmp_path / "off")
        unc_off = KalmanUncProp(**dirs_off).compute(opt_id_off, alpha=0.05)

        for t in run_full.timesteps:
            tp_full = unc_full.per_timestep[t].diagnostics["trace_post"]
            tp_off  = unc_off.per_timestep[t].diagnostics["trace_post"]
            assert tp_off >= tp_full - 1e-9, (
                f"t={t}: deactivating the soft observation row should not "
                f"INCREASE shrinkage (trace_post_off={tp_off:.3e}, "
                f"trace_post_full={tp_full:.3e})"
            )


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------

class TestUncPropRegistryRoundtrip:

    def test_persist_and_reload(self, tmp_path):
        from pysuto.uncprop import KalmanUncProp
        from pysuto.registry import UncPropRunRegistry

        run = _make_toy_3edge_2t()
        opt_id, dirs = _solve_and_persist(run, tmp_path)
        uncprop = KalmanUncProp(**dirs)
        result = uncprop.compute(opt_id, alpha=0.05)
        unc_id = uncprop.persist(result, opt_id, label="rt")

        registry = UncPropRunRegistry(dirs["uncprop_dir"])
        loaded   = registry.load(unc_id)
        assert loaded.method == result.method
        assert loaded.alpha  == pytest.approx(result.alpha)
        for t in result.per_timestep:
            np.testing.assert_allclose(
                loaded.per_timestep[t].sigma_a_star,
                result.per_timestep[t].sigma_a_star,
            )
            np.testing.assert_allclose(
                loaded.per_timestep[t].lower, result.per_timestep[t].lower,
            )
            np.testing.assert_allclose(
                loaded.per_timestep[t].upper, result.per_timestep[t].upper,
            )
