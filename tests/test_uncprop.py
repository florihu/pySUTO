"""
test_uncprop.py
===============
Test suite for :mod:`pysuto.uncprop` (Stage 5 — posterior uncertainty).

Tests fall into two buckets:

* Math (the active set, aggregates, draws)
* Plumbing (registry round-trip, CI clipping, end-to-end load_from_opt_run_id)

All tests use fabricated ConstraintRuns / OptimResults — no ICSG fixture
files. Stage 5 has one path, the per-year block path; a run without
``G_blocks`` is refused.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pysuto._conditioning import DEFAULT_CONDITIONING


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
        "sense":     ["eq"],
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
    """2 years × 3 edges, mixed cone types, on the block (year-local) path: per
    year a hard balance, a soft observation and a hard ``le`` row. σ_c varies in
    t; the other year's rows are deactivated in ``timestep_data``."""
    from pysuto.constraints import ConstraintRun, SIGMA_C_INACTIVE

    G_t = sp.csr_matrix(np.array([
        [ 1.0,  1.0, -1.0],   # zero,   hard
        [ 1.0,  0.0,  0.0],   # free,   soft
        [ 1.0, -2.0,  0.0],   # nonneg, hard
    ]))
    row_meta = pd.DataFrame({
        "constraint_id": ["balance_2020", "obs_a0_2020", "ratio_bound_2020",
                          "balance_2021", "obs_a0_2021", "ratio_bound_2021"],
        "is_hard":       [True, False, True] * 2,
        "sense":         ["eq", "eq", "le"] * 2,
        "time":          ["2020"] * 3 + ["2021"] * 3,
    })
    off = float(SIGMA_C_INACTIVE)
    return ConstraintRun(
        G              = sp.block_diag([G_t, G_t]).tocsr(),
        row_meta       = row_meta,
        a0             = np.array([1.0, 2.0, 3.0] * 2),
        sigma_a        = np.ones(6),
        u              = np.full(6, 10.0),
        l              = np.zeros(6),
        timestep_data  = {
            "2020": {"c": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
                     "sigma_c": np.array([0.0, 0.5, 0.0, off, off, off])},
            "2021": {"c": np.array([0.0, 0.0, 0.0, 0.0, 1.5, 0.0]),
                     "sigma_c": np.array([off, off, off, 0.0, 0.7, 0.0])},
        },
        run_id         = run_id,
        acc_id         = "toy_acc",
        agg_id         = "toy_agg",
        created_at     = "2026-05-26T00:00:00",
        G_blocks       = {"2020": G_t.tocsr(), "2021": G_t.tocsr()},
        edge_cols_by_t = {"2020": np.arange(3), "2021": np.arange(3, 6)},
    )


def _stub_runs(prop, cr, opt_results, opt_stage_config=None):
    """Make ``prop`` read ``cr`` and ``opt_results`` from memory: its two registries
    return them whatever the run id (the one place that reaches their loaders)."""
    sidecar = {"parents": {"upstream": {"const_run_id": cr.run_id}}}
    if opt_stage_config is not None:
        sidecar["stage_config"] = opt_stage_config
    prop._opt_registry.load_sidecar   = lambda _rid: sidecar
    prop._opt_registry.load           = lambda _rid: opt_results
    prop._const_registry.load         = lambda _rid: cr


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
        "sense":     ["eq", "eq"],
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
    _stub_runs(prop, cr, opt_results)
    return cr, opt_results, prop.compute("opt_fake", alpha=0.05, **compute_kw)


def _solve_and_persist(const_run, tmp_path, stage_config=None):
    """Persist a block-path const_run, solve one QP per year over its ``cols_t``
    (the prior of year 1 is the run's, of each later year the previous optimum),
    persist the opt run. Returns (opt_run_id, dirs)."""
    import clarabel
    from pysuto._conditioning import condition_prior, condition_sigma_c
    from pysuto.solver import OptimResult, row_layout
    from pysuto.registry import ConstraintRunRegistry, OptimRunRegistry

    const_dir   = str(tmp_path / "const")
    opt_dir     = str(tmp_path / "opt")
    uncprop_dir = str(tmp_path / "uncprop")
    ConstraintRunRegistry(const_dir).register(const_run)

    rm = const_run.row_meta
    row_time = rm["time"].astype(str).to_numpy()
    a_prior_all, sigma_a_all = condition_prior(const_run.a_prior_global, const_run.sigma_a)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-8

    results = {}
    for t, td in const_run.timestep_data.items():
        cols_t = const_run.edge_cols_by_t[t]
        rows_t = np.where(row_time == t)[0]
        G_t    = const_run.G_blocks[t].tocsr()
        lay = row_layout(rm["sense"].to_numpy()[rows_t], rm["is_hard"].to_numpy()[rows_t])
        n_slack, n_eq, n_le = lay.n_slack, lay.n_eq, lay.n_le
        n_a, M = len(cols_t), len(rows_t)
        a_prior, sigma_a = a_prior_all[cols_t], sigma_a_all[cols_t]
        A = sp.hstack([G_t[lay.perm],
                       sp.vstack([-sp.eye(n_slack, format="csr"),
                                  sp.csr_matrix((n_eq + n_le, n_slack))], format="csr")],
                      format="csc")
        cones = [clarabel.ZeroConeT(n_slack + n_eq)]
        if n_le:
            cones.append(clarabel.NonnegativeConeT(n_le))
        c  = np.asarray(td["c"], dtype=float)[rows_t]
        sc = condition_sigma_c(np.asarray(td["sigma_c"], dtype=float)[rows_t])
        P = sp.diags(2.0 * np.concatenate([1.0 / sigma_a ** 2,
                                           1.0 / sc[lay.perm[:n_slack]] ** 2]), format="csc")
        q = np.concatenate([-2.0 * a_prior / sigma_a ** 2, np.zeros(n_slack)])
        sol = clarabel.DefaultSolver(P, q, A, c[lay.perm], cones, settings).solve()
        x = np.asarray(sol.x)
        eps = np.zeros(M)
        eps[lay.perm[:n_slack]] = x[n_a:]
        results[t] = OptimResult(
            timestep=t, F_opt=x[:n_a], eps_opt=x[n_a:], c_star=c + eps,
            residuals=G_t @ x[:n_a] - c, objective=float(sol.obj_val),
            iterations=int(sol.iterations), solve_time_s=0.0, solver_status="solved",
            convergence={}, a_prior=a_prior.copy(), cols_t=cols_t, rows_t=rows_t)
    opt_run_id = OptimRunRegistry(opt_dir).register(
        results, parents={"inputs": {}, "upstream": {"const_run_id": const_run.run_id}},
        stage_config=stage_config)
    return opt_run_id, dict(const=const_dir, opt=opt_dir, uncprop=uncprop_dir)


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
            cols_t = run.edge_cols_by_t[t]
            assert ts.sigma_a_star.shape == (3,)
            assert ts.lower.shape == (3,)
            assert ts.upper.shape == (3,)
            assert (ts.upper >= ts.lower - 1e-10).all()
            # CI must lie in [l, u]
            assert (ts.lower >= run.l[cols_t] - 1e-10).all()
            assert (ts.upper <= run.u[cols_t] + 1e-10).all()

        # Persist → reload → equal arrays
        uncprop_run_id = prop.persist(result, opt_run_id, label="toy")
        loaded = UncPropRunRegistry(dirs["uncprop"]).load(uncprop_run_id)
        for t in result.per_timestep:
            a = result.per_timestep[t]
            b = loaded.per_timestep[t]
            np.testing.assert_array_equal(a.sigma_a_star, b.sigma_a_star)
            np.testing.assert_array_equal(a.lower,        b.lower)
            np.testing.assert_array_equal(a.upper,        b.upper)

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
# 8b) σ conditioning floors (#142): Stage 5 applies the floors its opt run recorded
# ---------------------------------------------------------------------------

_TUNED = {"sigma_a_floor_abs": 2e-8, "sigma_a_floor_rel": 3e-5,
          "sigma_a_ceiling_factor": 4e5, "sigma_c_floor": 5e-9}


class TestRecordedConditioning:

    def test_floors_recorded_by_a_solved_toy_run_are_the_ones_stage_5_applies(
            self, tmp_path):
        from pysuto.uncprop import KalmanUncProp
        from pysuto.registry import UncPropRunRegistry

        opt_run_id, dirs = _solve_and_persist(_make_toy_3edge_2t(), tmp_path,
                                              stage_config=dict(_TUNED))
        prop = KalmanUncProp(opt_dir=dirs["opt"], const_dir=dirs["const"],
                             uncprop_dir=dirs["uncprop"])
        result = prop.compute(opt_run_id)
        assert {k: getattr(prop, k) for k in _TUNED} == _TUNED

        run_id = prop.persist(result, opt_run_id)
        cfg = UncPropRunRegistry(dirs["uncprop"]).load_sidecar(run_id)["stage_config"]
        assert {k: cfg[k] for k in _TUNED} == _TUNED

    def test_a_sidecar_without_floors_warns_and_uses_the_defaults(self, tmp_path, caplog):
        import logging
        from pysuto.uncprop import KalmanUncProp

        opt_run_id, dirs = _solve_and_persist(_make_toy_3edge_2t(), tmp_path)
        prop = KalmanUncProp(opt_dir=dirs["opt"], const_dir=dirs["const"],
                             uncprop_dir=dirs["uncprop"])
        with caplog.at_level(logging.WARNING, logger="pysuto.conditioning"):
            prop.compute(opt_run_id)
        assert {k: getattr(prop, k) for k in DEFAULT_CONDITIONING} == DEFAULT_CONDITIONING
        assert "records no" in caplog.text and opt_run_id in caplog.text

    def test_an_explicit_argument_overrides_the_record(self, tmp_path):
        from pysuto.uncprop import KalmanUncProp

        opt_run_id, dirs = _solve_and_persist(_make_toy_3edge_2t(), tmp_path,
                                              stage_config=dict(_TUNED))
        prop = KalmanUncProp(opt_dir=dirs["opt"], const_dir=dirs["const"],
                             uncprop_dir=dirs["uncprop"], sigma_c_floor=7e-9)
        prop.compute(opt_run_id)
        assert prop.sigma_c_floor == 7e-9
        assert prop.sigma_a_floor_abs == _TUNED["sigma_a_floor_abs"]

    def test_the_solver_defaults_are_the_conditioning_defaults(self):
        import inspect
        from pysuto.solver import Solver

        params = inspect.signature(Solver.__init__).parameters
        assert {k: params[k].default for k in DEFAULT_CONDITIONING} == DEFAULT_CONDITIONING


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
        "is_hard":   [r[3] for r in rows], "sense": ["le" if r[4] == "nonneg" else "eq" for r in rows],
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
    _stub_runs(prop, cr, opt, opt_stage_config=compute_kw.pop("opt_stage_config", None))
    compute_kw.setdefault("alpha", 0.05)
    return prop.compute("opt_fake", **compute_kw)


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

    def test_a_run_without_g_blocks_is_refused(self, tmp_path, monkeypatch):
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
        _stub_runs(prop, cr, opt)
        with pytest.raises(ValueError, match="no G_blocks"):
            prop.compute("opt_fake")
        with pytest.raises(ValueError, match="no G_blocks"):
            prop.compute("opt_fake", aggregates=_agg_spec(np.eye(2), ["a", "b"]))

    def test_draws_must_name_a_year_of_the_run(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="lacks"):
            _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [5, 3, 2, 4], monkeypatch,
                             tmp_path, draws={"2021": 10}, rng=np.random.default_rng(81))

    def test_force_masks_are_boolean_and_disjoint(self, tmp_path, monkeypatch):
        mask = np.array([False, True, False, False])
        with pytest.raises(ValueError, match="both"):
            _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [5, 3, 2, 4], monkeypatch,
                             tmp_path, force_pinned=mask, force_free=mask)
        with pytest.raises(ValueError, match="boolean"):
            _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [5, 3, 2, 4], monkeypatch,
                             tmp_path, force_pinned=mask.astype(int))


# ---------------------------------------------------------------------------
# 11) The review's U5 findings (#285): guards, the interval's alpha, diagnostics
# ---------------------------------------------------------------------------

_SLACK_ONLY_ROWS = [([1, 0, 0, 0], "ratio", "band", True, "nonneg", 100.0, 0.0)]  # slack at a*


class TestGuardsAndDiagnostics:

    def test_the_interval_half_width_follows_alpha(self, tmp_path, monkeypatch):
        from scipy.stats import norm
        a_star = [5, 3, 2, 4]
        for alpha in (0.05, 0.2):
            r = _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, a_star, monkeypatch,
                                 tmp_path, alpha=alpha)
            ts = r.per_timestep["2020"]
            half = norm.ppf(1.0 - alpha / 2.0) * ts.sigma_a_star
            free = ts.sigma_a_star > 0
            np.testing.assert_allclose((ts.upper - np.asarray(a_star, float))[free],
                                       half[free], rtol=1e-12)
            assert r.alpha == alpha

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_outside_the_open_unit_interval_raises(self, tmp_path, monkeypatch, alpha):
        with pytest.raises(ValueError, match="alpha"):
            _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [5, 3, 2, 4], monkeypatch,
                             tmp_path, alpha=alpha)

    def test_a_year_without_kept_rows_keeps_its_prior_sigma(self, tmp_path, monkeypatch):
        r = _one_year_result(_SLACK_ONLY_ROWS, _FOUR_EDGE_SIGMA_A2, [1, 3, 2, 4], monkeypatch,
                             tmp_path, aggregates=_agg_spec(),
                             draws={"2020": 4000}, rng=np.random.default_rng(285))
        ts = r.per_timestep["2020"]
        assert ts.diagnostics["n_rows_kept"] == 0
        np.testing.assert_allclose(ts.sigma_a_star, np.sqrt(_FOUR_EDGE_SIGMA_A2), rtol=1e-12)
        np.testing.assert_allclose(r.aggregates.cov_post, r.aggregates.cov_prior, rtol=1e-12)
        d = r.draws["2020"]
        assert d.shape == (4000, 4)
        np.testing.assert_allclose(d.std(axis=0), np.sqrt(_FOUR_EDGE_SIGMA_A2), rtol=0.1)

    def test_an_opt_result_without_its_partition_is_refused(self, tmp_path):
        from pysuto.constraints import ConstraintRun
        from pysuto.solver import OptimResult
        from pysuto.uncprop import KalmanUncProp
        G = sp.csr_matrix(np.array([[1.0, -1.0]]))
        cr = ConstraintRun(
            G=G, row_meta=pd.DataFrame({"constraint_id": ["b"], "is_hard": [True],
                                        "sense": ["eq"], "time": ["2020"]}),
            a0=np.ones(2), sigma_a=np.ones(2), u=np.full(2, 10.0), l=np.zeros(2),
            timestep_data={"2020": {"c": np.zeros(1), "sigma_c": np.zeros(1)}},
            run_id="np", acc_id="a", agg_id="g", created_at="2026-11-23T00:00:00",
            G_blocks={"2020": G}, edge_cols_by_t={"2020": np.arange(2)})
        opt = {"2020": OptimResult(
            timestep="2020", F_opt=np.ones(2), eps_opt=np.zeros(0), c_star=np.zeros(1),
            residuals=np.zeros(1), objective=0.0, iterations=0, solve_time_s=0.0,
            solver_status="Solved", convergence={})}
        prop = KalmanUncProp(opt_dir=str(tmp_path / "o"), const_dir=str(tmp_path / "c"),
                             uncprop_dir=str(tmp_path / "u"))
        _stub_runs(prop, cr, opt)
        with pytest.raises(ValueError, match="no cols_t / rows_t"):
            prop.compute("opt_fake")

    def test_dependent_kept_rows_are_refused_by_the_whitened_qr(self):
        from pysuto.uncprop import KalmanUncProp
        G_k = sp.csr_matrix(np.array([[1.0], [2.0]]))     # two hard rows on one live edge
        with pytest.raises(ValueError, match="dependent"):
            KalmanUncProp._whitened_qr(G_k, np.ones(1), np.zeros(2))

    def test_kappa_max_is_recorded_and_in_the_unit_interval(self, tmp_path, monkeypatch):
        r = _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [5, 3, 2, 4], monkeypatch,
                             tmp_path)
        d = r.per_timestep["2020"].diagnostics
        assert 0.0 < d["kappa_max"] <= 1.0 + 1e-12
        assert d["max_resid_hard"] < 1e-6

    def test_a_violated_hard_equality_is_logged(self, tmp_path, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            r = _one_year_result(_FOUR_EDGE_ROWS, _FOUR_EDGE_SIGMA_A2, [5, 3, 1, 4],
                                 monkeypatch, tmp_path)     # e0 - e1 - e2 = 1, not 0
        assert r.per_timestep["2020"].diagnostics["max_resid_hard"] == pytest.approx(1.0)
        assert "hard equality row is violated" in caplog.text

    def test_persist_before_compute_is_refused(self, tmp_path):
        from pysuto.uncprop import KalmanUncProp, UncPropResult
        prop = KalmanUncProp(opt_dir=str(tmp_path / "o"), const_dir=str(tmp_path / "c"),
                             uncprop_dir=str(tmp_path / "u"))
        with pytest.raises(RuntimeError, match="before compute"):
            prop.persist(UncPropResult(method="kalman", alpha=0.05), "opt_x")

    @pytest.mark.parametrize("mode", ["cold", "warm"])
    def test_the_opt_runs_prior_mode_is_recorded_and_a_warm_one_warns(
            self, tmp_path, caplog, mode):
        from pysuto.registry import UncPropRunRegistry
        from pysuto.uncprop import KalmanUncProp
        run = _make_toy_3edge_2t()
        opt_run_id, dirs = _solve_and_persist(run, tmp_path,
                                              stage_config={"prior_mode": mode})
        prop = KalmanUncProp(opt_dir=dirs["opt"], const_dir=dirs["const"],
                             uncprop_dir=dirs["uncprop"])
        with caplog.at_level("WARNING"):
            result = prop.compute(opt_run_id)
        assert ("prior_mode='warm'" in caplog.text) == (mode == "warm")
        run_id = prop.persist(result, opt_run_id)
        sidecar = UncPropRunRegistry(dirs["uncprop"]).load_sidecar(run_id)
        assert sidecar["stage_config"]["prior_mode"] == mode
