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


def _compute_block_toy(tmp_path, monkeypatch, **compute_kw):
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
    return cr, opt_results, prop.compute("opt_fake", alpha=0.05, **compute_kw)


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


# ---------------------------------------------------------------------------
# 9) Math — the QP's active set (#27): slack inequality rows drop out, binding
#    lower bounds pin their edge, binding bands act as equalities, linearly
#    dependent hard rows are dropped, intervals clip at the bound row's own l
# ---------------------------------------------------------------------------

# One year, four edges e0..e3, σ_a² = [4, 1, 1, 1]; e3 is signed (no bound row)
# and zero-pinned in the ConstraintRun (l = u = 0), as a no-data edge would be.
_FOUR_EDGE_ROWS = [  # (coefficients on e0..e3, type, variant, is_hard, cone, c, σ_c)
    ([1, -1, -1,  0], "massbal", "massbal", True,  "zero",   0.0, 0.0),   # r0
    ([-1, 0,  0,  0], "bound",   "lower",   True,  "nonneg", 0.0, 0.0),   # r1
    ([0, -1,  0,  0], "bound",   "lower",   True,  "nonneg", 0.0, 0.0),   # r2
    ([0,  0, -1,  0], "bound",   "lower",   True,  "nonneg", 0.0, 0.0),   # r3
    ([0,  1,  0, -2], "ratio",   "band",    True,  "nonneg", 0.0, 0.0),   # r4
    ([0,  0,  0,  1], "sum",     "anchor",  False, "free",   4.0, 1.0),   # r5
]
_FOUR_EDGE_SIGMA_A2 = np.array([4.0, 1.0, 1.0, 1.0])


def _one_year_posterior(rows, sigma_a2, a_star, monkeypatch, tmp_path):
    """Stage 5 (block path) on a one-year toy given as row tuples → UncPropTimestep."""
    return _one_year_result(rows, sigma_a2, a_star, monkeypatch, tmp_path).per_timestep["2020"]


def _one_year_result(rows, sigma_a2, a_star, monkeypatch, tmp_path, **compute_kw):
    """Stage 5 (block path) on a one-year toy given as row tuples → UncPropResult.
    The ConstraintRun zero-pins the last edge (l = u = 0)."""
    from pysuto.constraints import ConstraintRun
    from pysuto.solver import OptimResult
    from pysuto.uncprop import KalmanUncProp

    G = sp.csr_matrix(np.array([r[0] for r in rows], dtype=float))
    n = G.shape[1]
    c = np.array([r[5] for r in rows])
    row_meta = pd.DataFrame({
        "constraint_id": [f"r{i}" for i in range(len(rows))],
        "type":      [r[1] for r in rows], "variant": [r[2] for r in rows],
        "is_hard":   [r[3] for r in rows], "cone_type": [r[4] for r in rows],
        "time":      "2020",
    })
    a_star = np.asarray(a_star, dtype=float)
    u = np.full(n, 10.0)
    u[-1] = 0.0
    cr = ConstraintRun(
        G=G, row_meta=row_meta, a0=a_star, sigma_a=np.sqrt(sigma_a2), u=u, l=np.zeros(n),
        timestep_data={"2020": {"c": c, "sigma_c": np.array([r[6] for r in rows])}},
        run_id="as", acc_id="a", agg_id="g", created_at="2026-10-14T00:00:00",
        G_blocks={"2020": G}, edge_cols_by_t={"2020": np.arange(n)},
    )
    opt = {"2020": OptimResult(
        timestep="2020", F_opt=a_star, eps_opt=np.zeros(0), c_star=c,
        residuals=G @ a_star - c, objective=0.0, iterations=0, solve_time_s=0.0,
        solver_status="Solved", convergence={},
        cols_t=np.arange(n), rows_t=np.arange(len(rows)),
    )}
    prop = KalmanUncProp(opt_dir=str(tmp_path / "opt"), const_dir=str(tmp_path / "const"),
                         uncprop_dir=str(tmp_path / "uncprop"))
    monkeypatch.setattr(prop._opt_registry, "load_sidecar",
                        lambda _rid: {"parents": {"upstream": {"const_run_id": "as"}}})
    monkeypatch.setattr(prop._opt_registry, "load", lambda _rid: opt)
    monkeypatch.setattr(prop._const_registry, "load", lambda _rid: cr)
    return prop.compute("opt_fake", alpha=0.05, **compute_kw)


def _four_edge_posterior(a_star, monkeypatch, tmp_path, extra_rows=()):
    return _one_year_posterior(_FOUR_EDGE_ROWS + list(extra_rows), _FOUR_EDGE_SIGMA_A2,
                               a_star, monkeypatch, tmp_path)


def _dense_posterior(rows, sigma_a2, hard):
    """Reference σ*² on an explicitly reduced system of ``_FOUR_EDGE_ROWS``:
    S = G·diag(σ_a²)·Gᵀ + diag(σ_c²), solved densely."""
    G = np.array([_FOUR_EDGE_ROWS[i][0] for i in rows], dtype=float)
    sc2 = np.array([0.0 if h else _FOUR_EDGE_ROWS[i][6] ** 2 for i, h in zip(rows, hard)])
    S = G @ np.diag(sigma_a2) @ G.T + np.diag(sc2)
    return sigma_a2 - sigma_a2 ** 2 * np.einsum("ij,ij->j", G, np.linalg.solve(S, G))


class TestActiveSet:

    def test_slack_inequality_rows_drop_out(self, tmp_path, monkeypatch):
        ts = _four_edge_posterior([5, 3, 2, 4], monkeypatch, tmp_path)
        ref = _dense_posterior([0, 5], _FOUR_EDGE_SIGMA_A2, [True, False])
        np.testing.assert_allclose(ts.sigma_a_star ** 2, ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(ts.bound_active, [False] * 4)

    def test_binding_lower_bound_pins_its_edge_as_an_equality_would(self, tmp_path,
                                                                     monkeypatch):
        """The reference keeps the bound row r3 as a hard equality with e2's full
        prior variance — independent of the implementation's σ_a² → 0 pin."""
        ts = _four_edge_posterior([3, 3, 0, 4], monkeypatch, tmp_path)
        ref = _dense_posterior([0, 3, 5], _FOUR_EDGE_SIGMA_A2, [True, True, False])
        assert ts.sigma_a_star[2] == 0.0
        np.testing.assert_allclose(ts.sigma_a_star ** 2, ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(ts.bound_active, [False, False, True, False])

    def test_bind_tol_is_the_pin_boundary(self, tmp_path, monkeypatch):
        inside  = _four_edge_posterior([3, 3 - 0.9e-6, 0.9e-6, 4], monkeypatch, tmp_path)
        outside = _four_edge_posterior([3, 3 - 1.1e-6, 1.1e-6, 4], monkeypatch, tmp_path)
        assert inside.diagnostics["n_bound_active"] == 1 and inside.sigma_a_star[2] == 0.0
        assert outside.diagnostics["n_bound_active"] == 0 and outside.sigma_a_star[2] > 0.0

    def test_binding_band_acts_as_an_equality(self, tmp_path, monkeypatch):
        ts = _four_edge_posterior([8, 8, 0, 4], monkeypatch, tmp_path)   # e1 − 2·e3 = 0
        ref = _dense_posterior([0, 3, 4, 5], _FOUR_EDGE_SIGMA_A2, [True, True, True, False])
        np.testing.assert_allclose(ts.sigma_a_star ** 2, ref, rtol=1e-10, atol=1e-12)
        d = ts.diagnostics
        counts = (d["n_bound_active"], d["n_bands_binding"], d["n_hard_dependent"])
        assert counts == (1, 1, 0)

    def test_linearly_dependent_hard_rows_are_dropped(self, tmp_path, monkeypatch):
        """Two copies of the mass balance (one negated, one doubled) make S
        singular with every hard row at σ_c = 0; the dependent copies go."""
        dupes = [([-1, 1, 1, 0], "massbal", "massbal", True, "zero", 0.0, 0.0),
                 ([2, -2, -2, 0], "massbal", "massbal", True, "zero", 0.0, 0.0)]
        ts = _four_edge_posterior([5, 3, 2, 4], monkeypatch, tmp_path, extra_rows=dupes)
        ref = _dense_posterior([0, 5], _FOUR_EDGE_SIGMA_A2, [True, False])
        np.testing.assert_allclose(ts.sigma_a_star ** 2, ref, rtol=1e-10, atol=1e-12)
        assert ts.diagnostics["n_hard_dependent"] == 2

    def test_rank_is_tested_on_the_rows_not_their_prior_variance(self, tmp_path,
                                                                 monkeypatch):
        """Rows [1, 1] and [1, 0] are independent and pin both edges, although
        with σ_a = [1e3, 1e-3] they are nearly parallel in the prior metric."""
        rows = [([1, 1], "massbal", "massbal", True, "zero", 0.0, 0.0),
                ([1, 0], "massbal", "massbal", True, "zero", 0.0, 0.0)]
        sigma_a2 = np.array([1e6, 1e-6])
        ts = _one_year_posterior(rows, sigma_a2, [0.0, 0.0], monkeypatch, tmp_path)
        assert ts.diagnostics["n_hard_dependent"] == 0
        assert (ts.sigma_a_star ** 2 <= 1e-6 * sigma_a2).all()          # κ ≥ 1 − 1e-6

    def test_contraction_is_exact_where_S_is_singular_to_working_precision(
            self, tmp_path, monkeypatch):
        """The copper pattern of #27: two independent hard rows share a 1e6 kt
        edge e2 and differ only on two small edges e0, e1, so forming S rounds
        it to singular (σ_a = 1e-6 kt: the LU raises) or skews it silently
        (σ_a = 0.017 kt: the LU gave κ = 0.644). With v = (−1, −1, 1) spanning
        the rows' null space, κ_i = 1 − v_i² / (σ_i² · vᵀΣ⁻¹v): 1 − 1/(2 + σ²/σ_2²)
        on each small edge, and 1 − 1/(6 + σ²/σ_2²) once a soft row with σ_c = σ
        measures e0 + e1 (vᵀΣ⁻¹v gains 4/σ_c²). The small edges come first, so
        the leverage needs its rows sorted by norm."""
        rows = [([1, 0, 1], "massbal", "massbal", True,  "zero", 0.0, 0.0),
                ([0, 1, 1], "massbal", "massbal", True,  "zero", 0.0, 0.0)]
        for s2 in (1e-12, 3e-4):
            sigma_a2 = np.array([s2, s2, 1e12])
            soft = ([1, 1, 0], "sum", "anchor", False, "free", 0.0, np.sqrt(s2))
            for extra, k in (([], 2.0), ([soft], 6.0)):
                kappa = 1.0 - 1.0 / (k + s2 / 1e12)
                ts = _one_year_posterior(rows + extra, sigma_a2, [0.0, 0.0, 0.0],
                                         monkeypatch, tmp_path)
                assert ts.diagnostics["n_hard_dependent"] == 0
                np.testing.assert_allclose(ts.sigma_a_star[:2] ** 2,
                                           (1.0 - kappa) * sigma_a2[:2], rtol=1e-10)
                assert (ts.sigma_a_star ** 2 <= sigma_a2).all()

    def test_rows_emptied_by_elimination_are_dropped(self, tmp_path, monkeypatch):
        # e0, e1, e2 bound-active: the mass balance has no live edge left
        ts = _four_edge_posterior([0, 0, 0, 4], monkeypatch, tmp_path)
        np.testing.assert_allclose(ts.sigma_a_star, [0, 0, 0, np.sqrt(0.5)], rtol=1e-12)

    def test_intervals_clip_at_the_bound_rows_own_l_only(self, tmp_path, monkeypatch):
        from scipy.stats import norm
        z = norm.ppf(0.975)
        ts = _four_edge_posterior([1, 0.5, 0.5, 0.5], monkeypatch, tmp_path)
        s = ts.sigma_a_star
        assert ts.lower[1] == 0.0 and 0.5 - z * s[1] < 0       # bound row → clipped at 0
        assert ts.lower[3] == pytest.approx(0.5 - z * s[3])    # signed edge: may go < 0
        assert ts.lower[3] < 0
        assert ts.upper[3] == pytest.approx(0.5 + z * s[3])    # not clipped to the pin
        assert ts.upper[3] > 0

        rows = list(_FOUR_EDGE_ROWS)
        rows[1] = ([-1, 0, 0, 0], "bound", "lower", True, "nonneg", -2.0, 0.0)   # e0 ≥ 2
        ts = _one_year_posterior(rows, _FOUR_EDGE_SIGMA_A2, [2.5, 1.5, 1, 4],
                                 monkeypatch, tmp_path)
        assert 2.5 - z * ts.sigma_a_star[0] < 2.0 and ts.lower[0] == 2.0

    def test_misaligned_opt_rows_raise(self):
        from types import SimpleNamespace
        from pysuto.uncprop import KalmanUncProp
        opt_t = SimpleNamespace(cols_t=np.arange(4), rows_t=np.arange(5))
        with pytest.raises(ValueError, match="rows_t"):
            KalmanUncProp._check_partition("2020", np.arange(4), np.arange(6), opt_t)

    def test_bound_active_mask_round_trips_through_the_registry(self, tmp_path,
                                                                monkeypatch):
        from pysuto.registry import UncPropRunRegistry
        from pysuto.uncprop import UncPropResult
        ts = _four_edge_posterior([3, 3, 0, 4], monkeypatch, tmp_path)
        reg = UncPropRunRegistry(str(tmp_path / "uncprop"))
        rid = reg.register(UncPropResult(method="kalman", alpha=0.05,
                                         per_timestep={"2020": ts}),
                           parents={"inputs": {}, "upstream": {}})
        np.testing.assert_array_equal(reg.load(rid).per_timestep["2020"].bound_active,
                                      ts.bound_active)


# ---------------------------------------------------------------------------
# 10) Math — aggregates (#81): the posterior covariance of linear totals wᵀa*,
#     posterior draws, and the active-set overrides of #99's degenerate family
# ---------------------------------------------------------------------------

# Aggregates over e0..e3 of _FOUR_EDGE_ROWS, one column each.
_AGG_NAMES = ["e0", "e1+e2", "balance", "e3", "mixed", "total"]
_AGG_W = np.array([
    [1, 0, 0, 0],                 # e0
    [0, 1, 1, 0],                 # e1 + e2, equal to e0 under the hard balance
    [1, -1, -1, 0],               # the balance itself: known exactly
    [0, 0, 0, 1],                 # e3
    [0, 2, 0, 0.5],               # mixed weights
    [1, 1, 1, 1],                 # every edge
], dtype=float).T                 # (4 edges × 6 aggregates)


def _agg_spec(W=_AGG_W, names=_AGG_NAMES):
    from pysuto.uncprop import AggregateSpec
    return AggregateSpec(names=list(names), W=sp.csr_matrix(W))


def _dense_cov(rows, sigma_a2, hard):
    """Reference posterior covariance Σ − Σ·Gᵀ·S⁻¹·G·Σ on an explicitly reduced
    system of ``_FOUR_EDGE_ROWS`` (S = G·Σ·Gᵀ + diag(σ_c²), solved densely)."""
    G = np.array([_FOUR_EDGE_ROWS[i][0] for i in rows], dtype=float)
    sc2 = np.array([0.0 if h else _FOUR_EDGE_ROWS[i][6] ** 2 for i, h in zip(rows, hard)])
    Sig = np.diag(sigma_a2)
    S = G @ Sig @ G.T + np.diag(sc2)
    return Sig - Sig @ G.T @ np.linalg.solve(S, G @ Sig)


class TestAggregates:

    def test_covariance_is_the_dense_posterior_of_the_aggregates(self, tmp_path,
                                                                 monkeypatch):
        """e2's lower bound binds: the reference keeps its bound row as a hard
        equality with the full prior, independent of the implementation's pin."""
        a_star = np.array([3.0, 3.0, 0.0, 4.0])
        res = _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, a_star, monkeypatch,
                               tmp_path, aggregates=_agg_spec())
        ag  = res.aggregates
        ref = _dense_cov([0, 3, 5], _FOUR_EDGE_SIGMA_A2, [True, True, False])
        s   = res.per_timestep["2020"].sigma_a_star
        assert ag.names == _AGG_NAMES
        np.testing.assert_allclose(ag.cov_post, _AGG_W.T @ ref @ _AGG_W, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(ag.cov_prior, _AGG_W.T @ np.diag(_FOUR_EDGE_SIGMA_A2) @ _AGG_W,
                                   rtol=1e-12)
        np.testing.assert_allclose(ag.value, _AGG_W.T @ a_star, rtol=1e-12)
        np.testing.assert_allclose(ag.sigma_quadrature, np.sqrt((_AGG_W ** 2).T @ s ** 2),
                                   rtol=1e-12)
        np.testing.assert_allclose(np.diag(ag.cov_post)[[0, 3]], s[[0, 3]] ** 2, rtol=1e-10)
        assert ag.cov_post[2, 2] == pytest.approx(0.0, abs=1e-11)

    def test_a_balance_makes_a_total_better_known_than_its_parts(self, tmp_path,
                                                                 monkeypatch):
        """e0 = e1 + e2 with σ_a² = (4, 1, 1): var(e1 + e2) = 4/3 exactly, against
        5/3 from the edge variances in quadrature (cov(e1, e2) = −1/6)."""
        res = _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [5, 3, 2, 4],
                               monkeypatch, tmp_path, aggregates=_agg_spec())
        ag = res.aggregates
        assert ag.cov_post[1, 1] == pytest.approx(4.0 / 3.0, rel=1e-10)
        assert ag.sigma_quadrature[1] ** 2 == pytest.approx(5.0 / 3.0, rel=1e-10)

    def test_years_add_and_never_covary(self, tmp_path, monkeypatch):
        """Each year is its own QP: an aggregate over two years has the sum of the
        years' variances, and aggregates of different years have no covariance."""
        W = np.zeros((6, 5))
        W[0, 0] = 1.0                              # e0 (2020)
        W[3, 1] = 1.0                              # e3 (2021)
        W[[0, 3], 2] = 1.0                         # e0 + e3
        W[[0, 3], 3], W[[1, 4], 3] = 1.0, -1.0     # both years' balances: exactly 0
        W[[2, 5], 4] = 1.0                         # the free edges e2 + e5
        names = ["e0", "e3", "e0+e3", "balances", "e2+e5"]
        _, _, res = _compute_block_toy(tmp_path, monkeypatch, aggregates=_agg_spec(W, names))
        year = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
        ref = np.kron(np.eye(2), year)             # the block-diagonal posterior
        np.testing.assert_allclose(res.aggregates.cov_post, W.T @ ref @ W, rtol=1e-10, atol=1e-12)
        assert res.aggregates.cov_post[0, 1] == 0.0

    def test_draws_keep_every_hard_row_and_reproduce_the_posterior(self, tmp_path,
                                                                   monkeypatch):
        """e2 pinned, the band e1 = 2·e3 binding: every draw satisfies both hard
        rows and the pin exactly; their covariance converges to the posterior."""
        res = _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [8, 8, 0, 4], monkeypatch,
                               tmp_path, draws={"2020": 50_000}, rng=np.random.default_rng(81))
        d = res.draws["2020"]
        assert d.shape == (50_000, 4)
        assert (d[:, 2] == 0.0).all()
        G_hard = np.array([_FOUR_EDGE_ROWS[i][0] for i in (0, 4)], dtype=float)
        np.testing.assert_allclose(d @ G_hard.T, 0.0, atol=1e-12)
        ref = _dense_cov([0, 3, 4, 5], _FOUR_EDGE_SIGMA_A2, [True, True, True, False])
        np.testing.assert_allclose(np.cov(d, rowvar=False), ref, atol=0.02)

    def test_draws_need_a_generator(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="rng"):
            _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [5, 3, 2, 4], monkeypatch,
                             tmp_path, draws={"2020": 10})

    def test_force_free_unpins_a_binding_bound(self, tmp_path, monkeypatch):
        res = _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [3, 3, 0, 4], monkeypatch,
                               tmp_path, force_free=np.array([False, False, True, False]))
        ts  = res.per_timestep["2020"]
        ref = _dense_posterior([0, 5], _FOUR_EDGE_SIGMA_A2, [True, False])
        np.testing.assert_allclose(ts.sigma_a_star ** 2, ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(ts.bound_active, [False] * 4)
        assert (ts.diagnostics["n_forced_free"], ts.diagnostics["n_forced_pinned"]) == (1, 0)

    def test_force_pinned_pins_a_slack_bound(self, tmp_path, monkeypatch):
        """The reference keeps e1's bound row r2 as a hard equality."""
        res = _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [5, 3, 2, 4], monkeypatch,
                               tmp_path, force_pinned=np.array([False, True, False, False]))
        ts  = res.per_timestep["2020"]
        ref = _dense_posterior([0, 2, 5], _FOUR_EDGE_SIGMA_A2, [True, True, False])
        assert ts.sigma_a_star[1] == 0.0
        np.testing.assert_allclose(ts.sigma_a_star ** 2, ref, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(ts.bound_active, [False, True, False, False])
        assert (ts.diagnostics["n_forced_free"], ts.diagnostics["n_forced_pinned"]) == (0, 1)

    def test_aggregate_rows_must_span_every_edge(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="one row per edge"):
            _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [5, 3, 2, 4], monkeypatch,
                             tmp_path, aggregates=_agg_spec(_AGG_W[:3], _AGG_NAMES))

    def test_aggregates_need_the_block_path(self, tmp_path, monkeypatch):
        from pysuto.solver import OptimResult
        from pysuto.uncprop import KalmanUncProp
        cr  = _make_toy_2x2()
        opt = {"2020": OptimResult(
            timestep="2020", F_opt=np.ones(2), eps_opt=np.zeros(0), c_star=np.zeros(1),
            residuals=np.zeros(1), objective=0.0, iterations=0, solve_time_s=0.0,
            solver_status="Solved", convergence={},
        )}
        prop = KalmanUncProp(opt_dir=str(tmp_path / "opt"), const_dir=str(tmp_path / "const"),
                             uncprop_dir=str(tmp_path / "uncprop"))
        monkeypatch.setattr(prop._opt_registry, "load_sidecar",
                            lambda _rid: {"parents": {"upstream": {"const_run_id": cr.run_id}}})
        monkeypatch.setattr(prop._opt_registry, "load", lambda _rid: opt)
        monkeypatch.setattr(prop._const_registry, "load", lambda _rid: cr)
        with pytest.raises(ValueError, match="block path"):
            prop.compute("opt_fake", aggregates=_agg_spec(np.eye(2), ["a", "b"]))
