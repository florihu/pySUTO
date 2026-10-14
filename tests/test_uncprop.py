"""
test_uncprop.py
===============
Test suite for :mod:`pysuto.uncprop` (Stage 5 — posterior uncertainty).

Tests fall into two buckets:

* Math (closed-form / invariants / limits / Cholesky reuse)
* Plumbing (registry round-trip, CI clipping, end-to-end load_from_opt_run_id)

All tests use fabricated ConstraintRuns / OptimResults — no ICSG fixture
files, no real CHOLMOD requirement (the scipy fallback covers most tests).
The one test that needs CHOLMOD specifically (``test_warm_factor_caching``)
guards itself with ``pytest.importorskip``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Fabrication helpers
# ---------------------------------------------------------------------------

def _make_toy_2x2(run_id: str = "toy_2x2"):
    """2 edges, 1 hard mass-balance row (a0 = a1). Closed form is hand-checkable.

    σ_a = [1, 1], σ_c = [0]  ⇒  S = 1+1 = 2, d = 0.5, σ²_a* = 0.5, σ_a* = √0.5.
    """
    from pysuto.constraints import ConstraintRun

    G = sp.csr_matrix(np.array([[1.0, -1.0]]))
    row_meta = pd.DataFrame({
        "constraint_id": ["balance"],
        "is_hard":       [True],
        "cone_type":     ["zero"],
    })
    return ConstraintRun(
        G              = G,
        row_meta       = row_meta,
        a0             = np.array([1.0, 1.0]),
        sigma_a        = np.array([1.0, 1.0]),
        u              = np.full(2, 10.0),
        l              = np.full(2, -10.0),
        timestep_data  = {"2020": {"c": np.array([0.0]),
                                    "sigma_c": np.array([0.0])}},
        run_id     = run_id,
        acc_id     = "toy_acc",
        agg_id     = "toy_agg",
        created_at = "2026-05-26T00:00:00",
    )


def _make_toy_3edge_2t(run_id: str = "toy_3e"):
    """3 edges, mixed cone types, 2 timesteps. Same shape as test_solver's toy.

    Reused for the warm-factor-caching test (need ≥2 timesteps to see the
    symbolic factor get reused).
    """
    from pysuto.constraints import ConstraintRun

    G = sp.csr_matrix(np.array([
        [ 1.0,  1.0, -1.0],   # zero,   hard
        [ 1.0,  0.0,  0.0],   # free,   soft
        [ 1.0, -2.0,  0.0],   # nonneg, hard
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
        u              = np.full(3, 10.0),
        l              = np.zeros(3),
        timestep_data  = {
            "2020": {"c": np.array([0.0, 1.0, 0.0]),
                     "sigma_c": np.array([0.0, 0.5, 0.0])},
            "2021": {"c": np.array([0.0, 1.5, 0.0]),
                     "sigma_c": np.array([0.0, 0.7, 0.0])},  # σ_c varies in t
        },
        run_id     = run_id,
        acc_id     = "toy_acc",
        agg_id     = "toy_agg",
        created_at = "2026-05-26T00:00:00",
    )


def _compute_block_toy(tmp_path, monkeypatch):
    """2 years × 3 edges, one hard mass balance (a0 = a1) per year, σ_a = 1, on
    the block (year-local) path. Returns (const_run, opt_results, UncPropResult)."""
    from pysuto.uncprop import KalmanUncProp
    from pysuto.solver import OptimResult
    from pysuto.constraints import ConstraintRun

    G_per_t = sp.csr_matrix(np.array([[1.0, -1.0, 0.0]]))
    G_global = sp.block_diag([G_per_t, G_per_t]).tocsr()
    row_meta = pd.DataFrame({
        "constraint_id": ["mb_2020", "mb_2021"],
        "is_hard":       [True, True],
        "cone_type":     ["zero", "zero"],
        "time":          ["2020", "2021"],
    })
    cr = ConstraintRun(
        G=G_global, row_meta=row_meta,
        a0=np.array([1.0, 1.0, 2.0, 1.5, 1.5, 2.5]),
        sigma_a=np.ones(6), u=np.full(6, 10.0), l=np.zeros(6),
        timestep_data={
            "2020": {"c": np.zeros(2), "sigma_c": np.array([0.0, 1e8])},
            "2021": {"c": np.zeros(2), "sigma_c": np.array([1e8, 0.0])},
        },
        run_id="blk", acc_id="a", agg_id="g",
        created_at="2026-05-28T00:00:00",
        G_blocks={"2020": G_per_t.tocsr(), "2021": G_per_t.tocsr()},
        edge_cols_by_t={"2020": np.array([0, 1, 2]),
                        "2021": np.array([3, 4, 5])},
    )
    opt_results = {
        "2020": OptimResult(
            timestep="2020", F_opt=np.array([1.0, 1.0, 2.0]),
            eps_opt=np.zeros(0), c_star=np.zeros(1), residuals=np.zeros(1),
            objective=0.0, iterations=0, solve_time_s=0.0,
            solver_status="Solved", convergence={},
            cols_t=np.array([0, 1, 2]), rows_t=np.array([0]),
        ),
        "2021": OptimResult(
            timestep="2021", F_opt=np.array([1.5, 1.5, 2.5]),
            eps_opt=np.zeros(0), c_star=np.zeros(1), residuals=np.zeros(1),
            objective=0.0, iterations=0, solve_time_s=0.0,
            solver_status="Solved", convergence={},
            cols_t=np.array([3, 4, 5]), rows_t=np.array([1]),
        ),
    }
    prop = KalmanUncProp(
        opt_dir=str(tmp_path / "opt"), const_dir=str(tmp_path / "const"),
        uncprop_dir=str(tmp_path / "uncprop"),
    )
    monkeypatch.setattr(prop._opt_registry, "load_sidecar",
                        lambda _rid: {"parents": {"upstream": {"const_run_id": cr.run_id}}})
    monkeypatch.setattr(prop._opt_registry, "load", lambda _rid: opt_results)
    monkeypatch.setattr(prop._const_registry, "load", lambda _rid: cr)
    return cr, opt_results, prop.compute("opt_fake", alpha=0.05)


def _solve_and_persist(const_run, tmp_path):
    """Persist const_run, solve it, persist opt run. Returns (opt_run_id, dirs)."""
    from pysuto.solver import ClarabelSolver
    from pysuto.registry import ConstraintRunRegistry

    const_dir   = str(tmp_path / "const")
    opt_dir     = str(tmp_path / "opt")
    uncprop_dir = str(tmp_path / "uncprop")

    const_reg = ConstraintRunRegistry(const_dir)
    const_reg.register(const_run)

    solver  = ClarabelSolver(
        const_dir=const_dir, opt_dir=opt_dir, verbose_first_solve=False,
    )
    results = solver.solve_all(const_run.run_id)
    opt_run_id = solver.persist(const_run.run_id, results, output_dir=opt_dir)

    return opt_run_id, dict(const=const_dir, opt=opt_dir, uncprop=uncprop_dir)


# ---------------------------------------------------------------------------
# 1) Math — toy 2x2 closed form
# ---------------------------------------------------------------------------

class TestToyClosedForm:

    def test_2x2_marginal_matches_hand_calc(self):
        from pysuto.uncprop import KalmanUncProp

        run = _make_toy_2x2()
        # Patch the registries by instantiating, then call the internals
        # directly — closed-form check needs no disk I/O.
        prop = KalmanUncProp.__new__(KalmanUncProp)
        prop.block_size                  = 1024
        prop.sigma_a_floor_abs           = 1e-8
        prop.sigma_a_floor_rel           = 1e-5
        prop.sigma_a_ceiling_factor      = 1e5
        prop.sigma_c_floor               = 1e-9

        from pysuto._conditioning import condition_prior
        _, sigma_a = condition_prior(run.a0, run.sigma_a)
        sigma_a2 = sigma_a ** 2

        G = run.G.tocsc()
        hard_mask = run.row_meta["is_hard"].to_numpy().astype(bool)
        handle = prop._init_factor(G, sigma_a2, run, hard_mask)
        sigma_c2 = prop._build_sigma_c2(run, "2020", hard_mask)
        prop._refresh_factor(handle, G, sigma_a2, sigma_c2)
        d = prop._marginal_diagonal(handle, G, G.shape[1])
        var_post = sigma_a2 * (1.0 - sigma_a2 * d)

        # Analytical: σ²_a* = 0.5 for both edges
        np.testing.assert_allclose(var_post, [0.5, 0.5], atol=1e-12)


# ---------------------------------------------------------------------------
# 2) Math — PSD / monotonicity invariants on random sparse G
# ---------------------------------------------------------------------------

class TestPsdAndMonotone:

    def test_random_sparse_invariants(self):
        from pysuto.constraints import ConstraintRun
        from pysuto.uncprop import KalmanUncProp

        rng = np.random.default_rng(0)
        m, n = 30, 200
        density = 0.08
        G = sp.random(m, n, density=density, random_state=rng,
                      data_rvs=lambda k: rng.standard_normal(k)).tocsr()
        # Ensure no all-zero columns (would give σ_a* = σ_a trivially; still valid)
        col_nnz = np.array((G != 0).sum(axis=0)).flatten()
        for j in np.where(col_nnz == 0)[0]:
            G[rng.integers(m), j] = 1.0
        G = G.tocsr()

        # Half hard, half soft
        is_hard = np.zeros(m, dtype=bool)
        is_hard[: m // 2] = True
        cone_type = np.where(is_hard, "zero", "free")
        row_meta = pd.DataFrame({
            "constraint_id": [f"r{i}" for i in range(m)],
            "is_hard":       is_hard,
            "cone_type":     cone_type,
        })

        run = ConstraintRun(
            G=G, row_meta=row_meta,
            a0=rng.standard_normal(n),
            sigma_a=np.abs(rng.standard_normal(n)) + 0.5,
            u=np.full(n,  100.0),
            l=np.full(n, -100.0),
            timestep_data={
                "2020": {"c": rng.standard_normal(m),
                         "sigma_c": np.where(is_hard, 0.0, 0.3)},
            },
            run_id="rand", acc_id="rand", agg_id="rand",
            created_at="2026-05-26T00:00:00",
        )

        prop = KalmanUncProp.__new__(KalmanUncProp)
        prop.block_size             = 64
        prop.sigma_a_floor_abs      = 1e-8
        prop.sigma_a_floor_rel      = 1e-5
        prop.sigma_a_ceiling_factor = 1e5
        prop.sigma_c_floor          = 1e-9

        from pysuto._conditioning import condition_prior
        _, sigma_a = condition_prior(run.a0, run.sigma_a)
        sigma_a2 = sigma_a ** 2
        Gc = G.tocsc()
        handle = prop._init_factor(Gc, sigma_a2, run, is_hard)
        sigma_c2 = prop._build_sigma_c2(run, "2020", is_hard)
        prop._refresh_factor(handle, Gc, sigma_a2, sigma_c2)
        d = prop._marginal_diagonal(handle, Gc, n)
        var_post = sigma_a2 * (1.0 - sigma_a2 * d)

        # PSD-ness: σ²_a* ≥ 0 (within float tolerance)
        assert (var_post >= -1e-10).all(), var_post.min()
        # Monotonicity: σ²_a* ≤ σ²_a
        assert (var_post <= sigma_a2 + 1e-10).all(), (var_post - sigma_a2).max()


# ---------------------------------------------------------------------------
# 3) Math — hard-constraint residual diagnostic
# ---------------------------------------------------------------------------

class TestHardConstraintResidual:

    def test_diagnostics_max_resid_hard_small(self, tmp_path):
        """After solving the toy 3-edge run, max_resid_hard from uncprop's
        diagnostics should be near zero (a* satisfies hard rows)."""
        from pysuto.uncprop import KalmanUncProp

        run = _make_toy_3edge_2t()
        opt_run_id, dirs = _solve_and_persist(run, tmp_path)

        prop = KalmanUncProp(
            opt_dir=dirs["opt"], const_dir=dirs["const"],
            uncprop_dir=dirs["uncprop"],
        )
        result = prop.compute(opt_run_id, alpha=0.05)
        for t, ts in result.per_timestep.items():
            assert ts.diagnostics["max_resid_hard"] < 1e-6, \
                (t, ts.diagnostics["max_resid_hard"])


# ---------------------------------------------------------------------------
# 4) Math — Byron (1978) limit: all-hard problem
# ---------------------------------------------------------------------------

class TestByronLimit:

    def test_all_hard_matches_byron(self):
        """For an all-hard problem, σ²_a* = σ²_a − σ²_a·G_hᵀ(G_h·diag(σ²_a)·G_hᵀ)⁻¹·G_h·σ²_a.

        On the toy 2x2 (σ_a = [1,1], one balance), this collapses to
        σ²_a* = 0.5 for both edges. Same value as the closed-form test,
        but framed as "the Σ_c → 0 limit of the general formula".
        """
        from pysuto.uncprop import KalmanUncProp

        run = _make_toy_2x2()
        # σ_c already 0 on the hard row — this IS the Byron limit
        prop = KalmanUncProp.__new__(KalmanUncProp)
        prop.block_size                  = 1024
        prop.sigma_a_floor_abs           = 1e-8
        prop.sigma_a_floor_rel           = 1e-5
        prop.sigma_a_ceiling_factor      = 1e5
        prop.sigma_c_floor               = 1e-9

        from pysuto._conditioning import condition_prior
        _, sigma_a = condition_prior(run.a0, run.sigma_a)
        sigma_a2 = sigma_a ** 2
        Gc = run.G.tocsc()
        hard_mask = run.row_meta["is_hard"].to_numpy().astype(bool)
        handle = prop._init_factor(Gc, sigma_a2, run, hard_mask)
        sigma_c2 = prop._build_sigma_c2(run, "2020", hard_mask)
        prop._refresh_factor(handle, Gc, sigma_a2, sigma_c2)
        d = prop._marginal_diagonal(handle, Gc, Gc.shape[1])
        var_post = sigma_a2 * (1.0 - sigma_a2 * d)

        # Byron reference: same hand-calc
        sa2 = sigma_a2
        G_dense = run.G.toarray()
        S = G_dense @ np.diag(sa2) @ G_dense.T          # 1×1
        var_byron = sa2 - sa2 * (G_dense ** 2).sum(axis=0) * sa2 / S[0, 0]

        np.testing.assert_allclose(var_post, var_byron, atol=1e-12)


# ---------------------------------------------------------------------------
# 5) Math — warm factor caching (CHOLMOD only)
# ---------------------------------------------------------------------------

class TestWarmFactorCaching:

    def test_symbolic_factor_analyzed_once(self, tmp_path):
        """Across 2 timesteps with different σ_c, CHOLMOD's ``analyze`` /
        top-level ``cholesky`` should fire exactly once. Per-t refresh
        is ``cholesky_inplace`` on the cached factor.
        """
        pytest.importorskip("sksparse")
        from unittest.mock import patch
        from pysuto.uncprop import KalmanUncProp
        import sksparse.cholmod as cholmod_mod

        run = _make_toy_3edge_2t()
        opt_run_id, dirs = _solve_and_persist(run, tmp_path)

        prop = KalmanUncProp(
            opt_dir=dirs["opt"], const_dir=dirs["const"],
            uncprop_dir=dirs["uncprop"],
        )

        with patch.object(cholmod_mod, "cholesky",
                          wraps=cholmod_mod.cholesky) as spy:
            prop.compute(opt_run_id, alpha=0.05)

        assert spy.call_count == 1, \
            f"Expected 1 symbolic factor analyze, got {spy.call_count}"


# ---------------------------------------------------------------------------
# 6) Plumbing — registry round-trip
# ---------------------------------------------------------------------------

class TestRegistryRoundtrip:

    def test_register_then_load_returns_same_arrays(self, tmp_path):
        from pysuto.registry import UncPropRunRegistry
        from pysuto.uncprop import UncPropResult, UncPropTimestep

        sigma = np.array([0.1, 0.2, 0.3])
        lower = np.array([-0.5, 0.0, 0.8])
        upper = np.array([ 0.5, 1.0, 1.6])
        ts = UncPropTimestep(
            timestep="2020", sigma_a_star=sigma, lower=lower, upper=upper,
            diagnostics={"trace_prior": 1.0, "trace_post": 0.14,
                         "max_resid_hard": 1e-9, "psd_min_diag": 0.1},
        )
        result = UncPropResult(
            method="kalman", alpha=0.05, per_timestep={"2020": ts},
        )

        reg = UncPropRunRegistry(str(tmp_path / "uncprop"))
        rid = reg.register(
            result,
            parents={"inputs": {}, "upstream": {"opt_run_id": "opt_x",
                                                  "const_run_id": "const_y"}},
        )
        assert rid.startswith("uncprop_")

        # Fresh registry instance to force a disk read
        reg2 = UncPropRunRegistry(str(tmp_path / "uncprop"))
        loaded = reg2.load(rid)

        assert loaded.method == "kalman"
        assert loaded.alpha == 0.05
        assert set(loaded.per_timestep) == {"2020"}
        np.testing.assert_array_equal(loaded.per_timestep["2020"].sigma_a_star, sigma)
        np.testing.assert_array_equal(loaded.per_timestep["2020"].lower, lower)
        np.testing.assert_array_equal(loaded.per_timestep["2020"].upper, upper)

        sidecar = reg2.load_sidecar(rid)
        parents = sidecar["parents"]["upstream"]
        assert parents["opt_run_id"]   == "opt_x"
        assert parents["const_run_id"] == "const_y"


# ---------------------------------------------------------------------------
# 7) Plumbing — CI clipping to [l, u]
# ---------------------------------------------------------------------------

class TestCIClipping:

    def test_lower_clipped_to_l_upper_clipped_to_u(self, tmp_path):
        """Construct a problem with a wide σ so that the un-clipped Gaussian
        CI would exit ``[l, u]`` for at least one edge, then verify both
        bounds are clipped.
        """
        from pysuto.constraints import ConstraintRun
        from pysuto.uncprop import KalmanUncProp
        from pysuto.registry import ConstraintRunRegistry, OptimRunRegistry
        from pysuto.solver import OptimResult

        # Single edge, no constraints (G has zero rows is awkward — use a
        # trivial soft "observation" on the edge with huge σ_c so σ_a*
        # ≈ σ_a). a* lives mid-range; the CI overshoots both bounds.
        G = sp.csr_matrix(np.array([[1.0]]))
        row_meta = pd.DataFrame({
            "constraint_id": ["obs"], "is_hard": [False], "cone_type": ["free"],
        })
        run = ConstraintRun(
            G=G, row_meta=row_meta,
            a0=np.array([0.5]),
            sigma_a=np.array([5.0]),     # huge prior std
            u=np.array([1.0]),           # narrow bounds
            l=np.array([0.0]),
            timestep_data={"2020": {"c": np.array([0.5]),
                                     "sigma_c": np.array([1e6])}},  # no info from row
            run_id="clip", acc_id="clip", agg_id="clip",
            created_at="2026-05-26T00:00:00",
        )

        # Persist const + a synthetic opt run so .compute() loads them.
        const_dir = str(tmp_path / "const")
        opt_dir   = str(tmp_path / "opt")
        ConstraintRunRegistry(const_dir).register(run)

        opt_results = {"2020": OptimResult(
            timestep="2020",
            F_opt=np.array([0.5]),
            eps_opt=np.zeros(1),
            c_star=np.array([0.5]),
            residuals=np.array([0.0]),
            objective=0.0, iterations=0, solve_time_s=0.0,
            solver_status="Solved", convergence={}, iter_history=None,
        )}
        opt_reg = OptimRunRegistry(opt_dir)
        opt_run_id = opt_reg.register(
            opt_results,
            parents={"inputs": {}, "upstream": {"const_run_id": run.run_id}},
        )

        prop = KalmanUncProp(opt_dir=opt_dir, const_dir=const_dir,
                             uncprop_dir=str(tmp_path / "uncprop"))
        result = prop.compute(opt_run_id, alpha=0.05)
        ts = result.per_timestep["2020"]

        # σ ≈ 5 (un-shrunk because σ_c is enormous); a* = 0.5
        # un-clipped lower ≈ -9.3, un-clipped upper ≈ 10.3
        assert ts.lower[0] == pytest.approx(0.0)   # clipped to l
        assert ts.upper[0] == pytest.approx(1.0)   # clipped to u


# ---------------------------------------------------------------------------
# 8) Plumbing — end-to-end load_from_opt_run_id
# ---------------------------------------------------------------------------

class TestLoadFromOptRunId:

    def test_full_compute_and_persist(self, tmp_path):
        from pysuto.uncprop import KalmanUncProp
        from pysuto.registry import UncPropRunRegistry

        run = _make_toy_3edge_2t()
        opt_run_id, dirs = _solve_and_persist(run, tmp_path)

        prop = KalmanUncProp(
            opt_dir=dirs["opt"], const_dir=dirs["const"],
            uncprop_dir=dirs["uncprop"],
        )
        result = prop.compute(opt_run_id, alpha=0.05)

        # Shape checks
        assert set(result.per_timestep) == {"2020", "2021"}
        for t, ts in result.per_timestep.items():
            assert ts.sigma_a_star.shape == (3,)
            assert ts.lower.shape == (3,)
            assert ts.upper.shape == (3,)
            assert (ts.upper >= ts.lower - 1e-10).all()
            # CI must lie in [l, u]
            assert (ts.lower >= run.l - 1e-10).all()
            assert (ts.upper <= run.u + 1e-10).all()

        # Persist → reload → equal arrays
        uncprop_run_id = prop.persist(result, opt_run_id, label="toy")
        loaded = UncPropRunRegistry(dirs["uncprop"]).load(uncprop_run_id)
        for t in result.per_timestep:
            a = result.per_timestep[t]
            b = loaded.per_timestep[t]
            np.testing.assert_array_equal(a.sigma_a_star, b.sigma_a_star)
            np.testing.assert_array_equal(a.lower,        b.lower)
            np.testing.assert_array_equal(a.upper,        b.upper)

    def test_per_t_schur_matches_legacy_sigma_a_star(self, tmp_path, monkeypatch):
        """Slice 7: uncprop using G_blocks (per-t Schur) produces numerically
        equivalent sigma_a_star to the legacy global-G Schur path."""
        import copy
        from pysuto.uncprop import KalmanUncProp
        from pysuto.solver import OptimResult
        from pysuto.constraints import ConstraintRun, SIGMA_C_INACTIVE

        # Hand-crafted per-t ConstraintRun: 5 edges × 2 timesteps = 10 edges
        # global. Per t: 1 hard mass balance (a0 == a1) + 1 soft ratio
        # (a0 == 0.5·a3, σ_c=0.5). Block is full-rank → S invertible.
        G_per_t = sp.csr_matrix(np.array([
            [1.0, -1.0,  0.0,  0.0,  0.0],
            [1.0,  0.0,  0.0, -0.5,  0.0],
        ]))
        G_global = sp.block_diag([G_per_t, G_per_t]).tocsr()

        row_meta = pd.DataFrame({
            "constraint_id": ["mb_2020", "ratio_2020", "mb_2021", "ratio_2021"],
            "is_hard":       [True, False, True, False],
            "cone_type":     ["zero", "free", "zero", "free"],
            "time":          ["2020", "2020", "2021", "2021"],
        })
        # Legacy timestep_data: σ_c full-length, non-t rows deactivated.
        sc_inactive = float(SIGMA_C_INACTIVE)
        ts_data = {
            "2020": {"c": np.zeros(4),
                     "sigma_c": np.array([0.0, 0.5, sc_inactive, sc_inactive])},
            "2021": {"c": np.zeros(4),
                     "sigma_c": np.array([sc_inactive, sc_inactive, 0.0, 0.5])},
        }
        a0      = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 1.5, 1.5, 2.5, 3.5, 4.5])
        sigma_a = np.ones(10)
        l       = np.zeros(10)
        u       = np.full(10, 10.0)

        cr = ConstraintRun(
            G              = G_global,
            row_meta       = row_meta,
            a0             = a0,
            sigma_a        = sigma_a,
            u              = u,
            l              = l,
            timestep_data  = ts_data,
            run_id         = "per_t",
            acc_id         = "per_t_acc",
            agg_id         = "per_t_agg",
            created_at     = "2026-05-27T00:00:00",
            G_blocks       = {"2020": G_per_t.tocsr(),
                              "2021": G_per_t.tocsr()},
            edge_cols_by_t = {"2020": np.array([0, 1, 2, 3, 4]),
                              "2021": np.array([5, 6, 7, 8, 9])},
        )

        # Fabricate OptimResults satisfying mb (a0=a1) and ratio (a0=0.5·a3)
        # at each t. The block path consumes YEAR-LOCAL results (F_opt over
        # cols_t); the legacy path consumes a full-length vector.
        opt_year_local = {
            "2020": OptimResult(
                timestep="2020",
                F_opt=np.array([1.0, 1.0, 2.0, 2.0, 4.0]),
                eps_opt=np.zeros(0), c_star=np.zeros(2),
                residuals=np.zeros(2), objective=0.0, iterations=0,
                solve_time_s=0.0, solver_status="Solved", convergence={},
                cols_t=np.array([0, 1, 2, 3, 4]), rows_t=np.array([0, 1]),
            ),
            "2021": OptimResult(
                timestep="2021",
                F_opt=np.array([1.5, 1.5, 2.5, 3.0, 4.5]),
                eps_opt=np.zeros(0), c_star=np.zeros(2),
                residuals=np.zeros(2), objective=0.0, iterations=0,
                solve_time_s=0.0, solver_status="Solved", convergence={},
                cols_t=np.array([5, 6, 7, 8, 9]), rows_t=np.array([2, 3]),
            ),
        }
        opt_full = {
            "2020": OptimResult(
                timestep="2020",
                F_opt=np.array([1.0, 1.0, 2.0, 2.0, 4.0, 0, 0, 0, 0, 0]),
                eps_opt=np.zeros(0), c_star=np.zeros(4),
                residuals=np.zeros(4), objective=0.0, iterations=0,
                solve_time_s=0.0, solver_status="Solved", convergence={},
            ),
            "2021": OptimResult(
                timestep="2021",
                F_opt=np.array([0, 0, 0, 0, 0, 1.5, 1.5, 2.5, 3.0, 4.5]),
                eps_opt=np.zeros(0), c_star=np.zeros(4),
                residuals=np.zeros(4), objective=0.0, iterations=0,
                solve_time_s=0.0, solver_status="Solved", convergence={},
            ),
        }

        const_dir = str(tmp_path / "const")
        # ── Per-t Schur path: serve cr (G_blocks intact) in-memory.
        prop_new = KalmanUncProp(
            opt_dir=str(tmp_path / "opt"),
            const_dir=const_dir,
            uncprop_dir=str(tmp_path / "uncprop"),
        )
        fake_sidecar = {"parents": {"upstream": {"const_run_id": cr.run_id}}}
        monkeypatch.setattr(prop_new._opt_registry, "load_sidecar",
                            lambda _rid: fake_sidecar)
        monkeypatch.setattr(prop_new._opt_registry, "load",
                            lambda _rid: opt_year_local)
        monkeypatch.setattr(prop_new._const_registry, "load",
                            lambda _rid: cr)
        result_new = prop_new.compute("opt_fake", alpha=0.05)

        # ── Legacy path: clear G_blocks → dispatch to legacy global-G Schur.
        cr_legacy = copy.copy(cr)
        cr_legacy.G_blocks       = {}
        cr_legacy.edge_cols_by_t = {}
        prop_legacy = KalmanUncProp(
            opt_dir=str(tmp_path / "opt"),
            const_dir=const_dir,
            uncprop_dir=str(tmp_path / "uncprop"),
        )
        monkeypatch.setattr(prop_legacy._opt_registry, "load_sidecar",
                            lambda _rid: fake_sidecar)
        monkeypatch.setattr(prop_legacy._opt_registry, "load",
                            lambda _rid: opt_full)
        monkeypatch.setattr(prop_legacy._const_registry, "load",
                            lambda _rid: cr_legacy)
        result_legacy = prop_legacy.compute("opt_fake", alpha=0.05)

        # Block output is year-local; legacy is full. They must agree on the
        # t-edges (legacy non-t shrinkage is a separate, expected artifact).
        for t in ("2020", "2021"):
            new_sa  = result_new.per_timestep[t].sigma_a_star
            leg_sa  = result_legacy.per_timestep[t].sigma_a_star
            cols_t  = cr.edge_cols_by_t[t]
            assert new_sa.shape == cols_t.shape
            assert np.allclose(new_sa, leg_sa[cols_t], atol=1e-7, rtol=1e-7), \
                f"t={t}: per-t σ_a* differs from legacy on t-edges " \
                f"(max abs {np.max(np.abs(new_sa - leg_sa[cols_t])):.3e})"

    def test_block_uncprop_output_is_year_local(self, tmp_path, monkeypatch):
        """Block-path uncprop stores year-local sigma_a_star/lower/upper over
        cols_t, with lower ≤ a* ≤ upper, and round-trips through the registry."""
        from pysuto.registry import UncPropRunRegistry

        cr, opt_results, result = _compute_block_toy(tmp_path, monkeypatch)

        for t in ("2020", "2021"):
            ts = result.per_timestep[t]
            cols_t = cr.edge_cols_by_t[t]
            assert ts.sigma_a_star.shape == cols_t.shape
            assert ts.lower.shape == cols_t.shape
            assert ts.upper.shape == cols_t.shape
            assert ts.cols_t is not None
            np.testing.assert_array_equal(ts.cols_t, cols_t)
            f = opt_results[t].F_opt
            assert np.all(ts.lower <= f + 1e-9) and np.all(f - 1e-9 <= ts.upper)

        # Round-trip: registry persists + reloads cols_t and year-local arrays.
        reg = UncPropRunRegistry(str(tmp_path / "uncprop"))
        out_id = reg.register(
            result, parents={"inputs": {}, "upstream": {"const_run_id": cr.run_id}},
        )
        npz = np.load(tmp_path / "uncprop" / out_id / "timesteps" / "t_2020.npz")
        assert "cols_t" in npz.files
        reloaded = reg.load(out_id)
        np.testing.assert_array_equal(
            reloaded.per_timestep["2020"].cols_t, cr.edge_cols_by_t["2020"],
        )

    def test_block_trace_contraction_is_year_local(self, tmp_path, monkeypatch):
        """The block path's trace_prior / trace_post sum over the year's own
        edges: per year σ² = [1, 1, 1] → σ*² = [½, ½, 1], so trace_prior = 3 and
        trace_post = 2 (trace contraction ⅓) — not the all-years 6 and 5 (⅙)."""
        _, _, result = _compute_block_toy(tmp_path, monkeypatch)
        for t in ("2020", "2021"):
            diag = result.per_timestep[t].diagnostics
            assert diag["trace_prior"] == pytest.approx(3.0)
            assert diag["trace_post"]  == pytest.approx(2.0)

    def test_missing_const_run_id_raises(self, tmp_path):
        """If opt sidecar has no upstream.const_run_id, compute() should fail
        cleanly with KeyError rather than crashing deep inside the loader."""
        from pysuto.uncprop import KalmanUncProp
        from pysuto.registry import OptimRunRegistry
        from pysuto.solver import OptimResult

        opt_dir = str(tmp_path / "opt")
        results = {"2020": OptimResult(
            timestep="2020", F_opt=np.zeros(1), eps_opt=np.zeros(0),
            c_star=np.zeros(0), residuals=np.zeros(0),
            objective=0.0, iterations=0, solve_time_s=0.0,
            solver_status="Solved", convergence={},
        )}
        opt_reg = OptimRunRegistry(opt_dir)
        opt_run_id = opt_reg.register(
            results, parents={"inputs": {}, "upstream": {}},  # no const_run_id
        )
        prop = KalmanUncProp(
            opt_dir=opt_dir, const_dir=str(tmp_path / "const"),
            uncprop_dir=str(tmp_path / "uncprop"),
        )
        with pytest.raises(KeyError, match="const_run_id"):
            prop.compute(opt_run_id, alpha=0.05)
