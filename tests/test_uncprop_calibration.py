"""
test_uncprop_calibration.py
===========================
Tests for Stage 5's calibration checks (#87): the **standardised shift**
``z/√κ``, the **Birge ratio** ``R = Σz²/dof`` and its scaled interval, and the
scale-total ablation of :class:`pysuto.uncprop.KalmanUncProp`.

The statistical tests draw a linear-Gaussian chain from the model itself: per
year, independent three-edge nodes with one hard balance and three soft rows;
the prior mean is drawn around the truth and the soft rows observe it, both with
the model's σ (or with σ times a factor, so the model understates it). ``a*`` is
the closed-form GLS estimate — the QP's optimum without bounds. Under the model
``z/√κ ~ N(0, 1)`` per edge and ``Σz²`` of one node is χ² with 4 degrees of
freedom (its four rows), so every tolerance below is a multiple of a known
standard deviation.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.stats import norm

SCALE_TOTALS = ("supplytotout", "usetotout", "smelterscrap", "directrefscrap")

# One node: edges e0, e1, e2; (coefficients, type, variant, is_hard, cone).
_NODE_ROWS = [
    ([1.0,  1.0, -1.0], "massbal", "massbal",      True,  "zero"),
    ([1.0,  0.0,  0.0], "sum",     "anchor",       False, "free"),
    ([0.0,  1.0,  1.0], "sum",     "supplytotout", False, "free"),
    ([-0.5, 0.0,  1.0], "ratio",   "conc",         False, "free"),
]


# ---------------------------------------------------------------------------
# Fabrication helpers
# ---------------------------------------------------------------------------

def _gaussian_chain(rng, n_years, n_nodes, understate=1.0):
    """A block-path chain drawn from the model → (ConstraintRun, opt results).

    The model's σ_a / σ_c are the truth's divided by ``understate``. ``a*`` is
    the GLS estimate ``a_prior + Σ Gᵀ S⁻¹ (c − G·a_prior)`` per year."""
    from pysuto.constraints import ConstraintRun
    from pysuto.solver import OptimResult

    years = [str(2000 + k) for k in range(n_years)]
    G_node = np.array([r[0] for r in _NODE_ROWS])
    G_t = sp.block_diag([sp.csr_matrix(G_node)] * n_nodes).tocsr()
    G_d = G_t.toarray()
    n_e, n_r = G_t.shape[1], G_t.shape[0]
    E, M = n_e * n_years, n_r * n_years
    hard = np.tile([r[3] for r in _NODE_ROWS], n_nodes * n_years)

    sigma_a = rng.uniform(0.5, 3.0, E)
    sigma_c = np.where(hard, 0.0, rng.uniform(0.3, 2.0, M))
    a_true  = rng.uniform(50.0, 150.0, E)
    G = sp.block_diag([G_t] * n_years).tocsr()
    a_prior = a_true + understate * sigma_a * rng.standard_normal(E)
    c = G @ a_true + understate * sigma_c * rng.standard_normal(M)

    row_meta = pd.DataFrame({
        "constraint_id": [f"r{i}" for i in range(M)],
        "type":      np.tile([r[1] for r in _NODE_ROWS], n_nodes * n_years),
        "variant":   np.tile([r[2] for r in _NODE_ROWS], n_nodes * n_years),
        "is_hard":   hard,
        "cone_type": np.tile([r[4] for r in _NODE_ROWS], n_nodes * n_years),
        "time":      np.repeat(years, n_r),
    })
    opt, cols_by_t = {}, {}
    for k, y in enumerate(years):
        cols = np.arange(k * n_e, (k + 1) * n_e)
        rows = np.arange(k * n_r, (k + 1) * n_r)
        s2 = sigma_a[cols] ** 2
        S = G_d @ np.diag(s2) @ G_d.T + np.diag(sigma_c[rows] ** 2)
        a_star = a_prior[cols] + s2 * (G_d.T @ np.linalg.solve(S, c[rows] - G_d @ a_prior[cols]))
        resid = G_d @ a_star - c[rows]
        cols_by_t[y] = cols
        opt[y] = OptimResult(
            timestep=y, F_opt=a_star, eps_opt=resid[~hard[rows]],
            c_star=c[rows] + np.where(hard[rows], 0.0, resid), residuals=resid,
            objective=0.0, iterations=0, solve_time_s=0.0, solver_status="Solved",
            convergence={}, cols_t=cols, rows_t=rows, a_prior=a_prior[cols],
        )
    cr = ConstraintRun(
        G=G, row_meta=row_meta, a0=a_prior, sigma_a=sigma_a,
        u=np.full(E, 1e4), l=np.zeros(E),
        timestep_data={y: {"c": c, "sigma_c": sigma_c} for y in years},
        run_id="gauss", acc_id="none", agg_id="", created_at="2026-10-20T00:00:00",
        G_blocks={y: G_t for y in years}, edge_cols_by_t=cols_by_t,
    )
    return cr, opt


def _prop(cr, opt, tmp_path, monkeypatch, **kw):
    """KalmanUncProp over the in-memory chain (registries monkeypatched)."""
    from pysuto.uncprop import KalmanUncProp
    prop = KalmanUncProp(opt_dir=str(tmp_path / "opt"), const_dir=str(tmp_path / "const"),
                         uncprop_dir=str(tmp_path / "uncprop"), **kw)
    monkeypatch.setattr(prop._opt_registry, "load_sidecar",
                        lambda _rid: {"parents": {"upstream": {"const_run_id": cr.run_id}}})
    monkeypatch.setattr(prop._opt_registry, "load", lambda _rid: opt)
    monkeypatch.setattr(prop._const_registry, "load", lambda _rid: cr)
    return prop


def _stats(rng, tmp_path, monkeypatch, n_years=40, n_nodes=50, understate=1.0):
    from pysuto.diag.stats import UncPropStatsDiagnostic
    cr, opt = _gaussian_chain(rng, n_years, n_nodes, understate)
    unc = _prop(cr, opt, tmp_path, monkeypatch).compute("opt_fake", alpha=0.05)
    return UncPropStatsDiagnostic(unc, opt, cr)


# ---------------------------------------------------------------------------
# 1. Data drawn from the model: z/√κ ~ N(0, 1) and R ≈ 1; σ understated by f:
#    z/√κ ~ N(0, f²) and R ≈ f²
# ---------------------------------------------------------------------------

_N_NODES = 40 * 50          # independent nodes in the default chain
_DOF     = 4 * _N_NODES     # one hard + three soft rows per node


def _moments_ok(zk, scale2):
    """The three shifts of one node are correlated N(0, scale2) draws, so over
    N nodes Var(mean(zk²)/scale2) ≤ 2·3²/(9N) = 2/N and Var(mean(zk)/√scale2)
    ≤ 3²/(9N) = 1/N (perfect correlation bounds both); four of those bounds."""
    m2 = float(np.mean(zk ** 2)) / scale2
    m1 = float(np.mean(zk)) / np.sqrt(scale2)
    assert abs(m2 - 1.0) < 4.0 * np.sqrt(2.0 / _N_NODES), m2
    assert abs(m1) < 4.0 * np.sqrt(1.0 / _N_NODES), m1


class TestDrawnFromTheModel:

    def test_standardised_shifts_are_standard_normal(self, tmp_path, monkeypatch):
        f = _stats(np.random.default_rng(87), tmp_path, monkeypatch).edge_frame()
        assert f["calibrated"].all()                   # no bounds, κ well above 0.01
        _moments_ok(f["z_kappa"].to_numpy(), 1.0)

    def test_birge_ratio_is_one(self, tmp_path, monkeypatch):
        """Σz² is χ²(dof) exactly: SD(R) = √(2/dof); five of them."""
        b = _stats(np.random.default_rng(87), tmp_path, monkeypatch).birge()
        run = b.set_index("year").loc["all"]
        assert run["dof"] == _DOF
        assert abs(run["birge_ratio"] - 1.0) < 5.0 * np.sqrt(2.0 / _DOF)

    def test_understated_sigma_gives_its_square_as_birge_ratio(self, tmp_path,
                                                               monkeypatch):
        st = _stats(np.random.default_rng(88), tmp_path, monkeypatch, understate=3.0)
        run = st.birge().set_index("year").loc["all"]
        assert abs(run["birge_ratio"] / 9.0 - 1.0) < 5.0 * np.sqrt(2.0 / _DOF)
        _moments_ok(st.edge_frame()["z_kappa"].to_numpy(), 9.0)

    def test_understated_sigma_fattens_the_tail_share(self, tmp_path, monkeypatch):
        """|z/√κ| > 3 when z/√κ ~ N(0, 9) is |N(0, 1)| > 1: 31.7 %. Shares of
        6,000 correlated edge-years: ≤ √(9/(4·9·2000)) = 1.1 pp SD per share."""
        st = _stats(np.random.default_rng(89), tmp_path, monkeypatch, understate=3.0)
        cal = st.calibration(st.edge_frame())
        assert len(cal) == 1
        assert abs(cal["share_gt3"].item() - 2 * norm.sf(1.0)) < 4 * 0.011


# ---------------------------------------------------------------------------
# 2. Σz² and dof — the QP objective at a* and the active-set constraint count
# ---------------------------------------------------------------------------

def _birge_stub(diagnostics):
    """One year, edges e0–e2, rows: hard balance, two soft rows, one hard band.

    z_edges = (a* − a_prior)/σ_a = [2, −1, 0.5]          → Σ 5.25
    soft residuals / σ_c   = [3/1.5, −1/0.5] = [2, −2]  → Σ 8   (hard rows ignored)
    """
    from pysuto.diag.stats import UncPropStatsDiagnostic
    cr = SimpleNamespace(
        G=sp.csr_matrix(np.eye(4, 3)),
        row_meta=pd.DataFrame({"type": ["massbal", "sum", "ratio", "ratio"],
                               "variant": ["massbal", "supplytotout", "conc", "band"],
                               "cone_type": ["zero", "free", "free", "nonneg"],
                               "time": "2020"}),
        a0=np.array([10.0, 5.0, 2.0]), sigma_a=np.array([2.0, 1.0, 4.0]),
        l=np.zeros(3), u=np.full(3, 100.0),
        timestep_data={"2020": {"c": np.zeros(4),
                                "sigma_c": np.array([0.0, 1.5, 0.5, 0.0])}},
    )
    unc = SimpleNamespace(alpha=0.05, per_timestep={"2020": SimpleNamespace(
        sigma_a_star=np.ones(3), lower=np.zeros(3), upper=np.ones(3),
        cols_t=np.arange(3), bound_active=np.zeros(3, dtype=bool),
        diagnostics=diagnostics)})
    opt = {"2020": SimpleNamespace(F_opt=np.array([14.0, 4.0, 4.0]),
                                   a_prior=np.array([10.0, 5.0, 2.0]),
                                   residuals=np.array([1e-9, 3.0, -1.0, -7.0]),
                                   cols_t=np.arange(3), rows_t=np.arange(4))}
    return UncPropStatsDiagnostic(unc, opt, cr)


class TestBirgeRatio:

    def test_zsq_is_edge_defects_plus_soft_residuals_over_dof(self):
        st = _birge_stub({"n_rows_kept": 3, "n_bound_active": 2,
                          "n_scale_totals_left_out": 0})
        b = st.birge().set_index("year")
        y = b.loc["2020"]
        assert y["zsq_edges"] == pytest.approx(5.25, rel=1e-12)
        assert y["zsq_soft_rows"] == pytest.approx(8.0, rel=1e-12)
        assert y["n_soft_rows"] == 2
        assert y["dof"] == 5
        assert y["birge_ratio"] == pytest.approx(13.25 / 5, rel=1e-12)
        assert b.loc["all", "birge_ratio"] == pytest.approx(13.25 / 5, rel=1e-12)
        z = norm.ppf(0.975)
        assert y["z_alpha"] == pytest.approx(z)
        assert y["z_alpha_birge"] == pytest.approx(z * np.sqrt(13.25 / 5))

    def test_dof_counts_the_scale_totals_an_ablation_left_out(self):
        """a* was fitted with them: the ablation must not change R."""
        st = _birge_stub({"n_rows_kept": 2, "n_bound_active": 2,
                          "n_scale_totals_left_out": 1})
        assert st.birge().set_index("year").loc["2020", "dof"] == 5

    def test_without_active_set_counts_the_ratio_is_undefined(self):
        st = _birge_stub({})
        b = st.birge().set_index("year")
        assert np.isnan(b.loc["2020", "dof"]) and np.isnan(b.loc["all", "birge_ratio"])

    def test_zsq_is_the_clarabel_objective_at_a_star(self, tmp_path, monkeypatch):
        """Clarabel's objective ½xᵀPx + qᵀx omits Σ a_prior²/σ_a²; added back it
        is Σz² over edges and soft rows (GLOSSARY.md "Objective decomposition")."""
        from test_uncprop_diag import _block_run, _propagate  # type: ignore
        from pysuto.diag.stats import UncPropStatsDiagnostic
        results, cr, acc = _block_run(tmp_path, monkeypatch)
        unc = _propagate(results, cr, tmp_path, monkeypatch)
        st = UncPropStatsDiagnostic(unc, results, cr, acc)
        b = st.birge().set_index("year")
        f = st.edge_frame()
        for y, res in results.items():
            fy = f[f["year"] == y]
            const = float(((fy["a_prior"] / fy["sigma_a"]) ** 2).sum())
            assert b.loc[y, "zsq_soft_rows"] > 0
            assert b.loc[y, "zsq_total"] == pytest.approx(res.objective + const, rel=1e-6)

    def test_zsq_matches_solverdiags_objective_decomposition(self, tmp_path, monkeypatch):
        from test_uncprop_diag import _block_run, _propagate  # type: ignore
        from pysuto.diag import SolverDiag
        from pysuto.diag.stats import UncPropStatsDiagnostic
        results, cr, acc = _block_run(tmp_path, monkeypatch)
        unc = _propagate(results, cr, tmp_path, monkeypatch)
        b = UncPropStatsDiagnostic(unc, results, cr, acc).birge().set_index("year")
        od = SolverDiag(results, cr, acc=acc)._decomposition().set_index("year")
        for y in results:
            assert b.loc[y, "zsq_edges"] == pytest.approx(od.loc[y, "zsq_edges"], rel=1e-10)
            soft = od.loc[y, ["zsq_sum_rows", "zsq_soft_ratios"]].fillna(0.0).sum()
            assert b.loc[y, "zsq_soft_rows"] == pytest.approx(soft, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. The Birge-scaled interval sits beside the stored one
# ---------------------------------------------------------------------------

def _interval_stub():
    """One year, three edges; e1 and e2 have lower-bound rows (−a ≤ −l) with
    l = 2 and l = 0; e0 has none. a* = [5, 3, 3], σ* = [1, 1, 1]."""
    from pysuto.diag.stats import UncPropStatsDiagnostic
    z = norm.ppf(0.975)
    a = np.array([5.0, 3.0, 3.0])
    lower = np.array([5 - z, max(3 - z, 2.0), 3 - z])
    cr = SimpleNamespace(
        G=sp.csr_matrix(np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])),
        row_meta=pd.DataFrame({"type": ["bound", "bound"], "variant": ["lower", "lower"],
                               "cone_type": ["nonneg", "nonneg"], "time": "2020"}),
        a0=a.copy(), sigma_a=np.full(3, 2.0), l=np.zeros(3), u=np.full(3, 100.0),
        timestep_data={"2020": {"c": np.array([-2.0, 0.0]), "sigma_c": np.zeros(2)}},
    )
    unc = SimpleNamespace(alpha=0.05, per_timestep={"2020": SimpleNamespace(
        sigma_a_star=np.ones(3), lower=lower, upper=a + z, cols_t=np.arange(3),
        bound_active=np.zeros(3, dtype=bool), diagnostics={})})
    opt = {"2020": SimpleNamespace(F_opt=a, a_prior=a.copy(), residuals=np.array([-1.0, -3.0]),
                                   cols_t=np.arange(3), rows_t=np.arange(2))}
    return UncPropStatsDiagnostic(unc, opt, cr), lower, a + z


class TestBirgeInterval:

    def test_scaled_by_sqrt_r_and_clipped_at_the_bound_rows_own_l(self):
        st, lower, upper = _interval_stub()
        f = st.edge_frame()
        bi = st.birge_interval(f, birge_ratio=4.0)
        z2 = 2.0 * norm.ppf(0.975)                     # z_α·√R
        np.testing.assert_allclose(bi["upper_birge"], [5 + z2, 3 + z2, 3 + z2], rtol=1e-12)
        np.testing.assert_allclose(bi["lower_birge"], [5 - z2, 2.0, 0.0], rtol=1e-12)

    def test_a_ratio_below_one_narrows_inside_the_bound(self):
        st, _, _ = _interval_stub()
        bi = st.birge_interval(st.edge_frame(), birge_ratio=0.25)
        h = 0.5 * norm.ppf(0.975)
        np.testing.assert_allclose(bi["lower_birge"], [5 - h, 3 - h, 3 - h], rtol=1e-12)

    def test_stored_lower_upper_are_untouched(self):
        st, lower, upper = _interval_stub()
        f = st.edge_frame()
        st.birge_interval(f, birge_ratio=4.0)
        np.testing.assert_array_equal(f["lower"].to_numpy(), lower)
        np.testing.assert_array_equal(f["upper"].to_numpy(), upper)
        assert "lower_birge" not in f.columns


# ---------------------------------------------------------------------------
# 4. Scale-total ablation: exactly the scale totals leave S_t, nothing else
# ---------------------------------------------------------------------------

# One year, edges e0..e3 (σ_a² = [4, 1, 1, 2]), no bounds. Every scale-total
# variant is a soft sum row; the adversaries share a variant or a type with them.
_ABL_ROWS = [  # (coefficients, type, variant, is_hard, cone, c, σ_c)
    ([1, -1, -1,  0], "massbal", "massbal",        True,  "zero", 0.0, 0.0),
    ([1,  0,  0,  0], "sum",     "supplytotout",   False, "free", 5.0, 0.5),
    ([0,  1,  0,  0], "sum",     "usetotout",      False, "free", 3.0, 0.4),
    ([0,  0,  1,  1], "sum",     "smelterscrap",   False, "free", 4.0, 0.3),
    ([0,  0,  0,  1], "sum",     "directrefscrap", False, "free", 2.0, 0.2),
    ([0,  1,  1,  0], "sum",     "anchor",         False, "free", 5.0, 1.0),
    ([0,  1,  0, -1], "ratio",   "usetotout",      False, "free", 1.0, 0.7),
]
_ABL_SIGMA_A2 = np.array([4.0, 1.0, 1.0, 2.0])
_ABL_A_STAR   = np.array([5.0, 3.0, 2.0, 2.0])


def _ablation_run():
    from pysuto.constraints import ConstraintRun
    from pysuto.solver import OptimResult
    G = sp.csr_matrix(np.array([r[0] for r in _ABL_ROWS], dtype=float))
    c = np.array([r[5] for r in _ABL_ROWS])
    rm = pd.DataFrame({
        "constraint_id": [f"r{i}" for i in range(len(_ABL_ROWS))],
        "type": [r[1] for r in _ABL_ROWS], "variant": [r[2] for r in _ABL_ROWS],
        "is_hard": [r[3] for r in _ABL_ROWS], "cone_type": [r[4] for r in _ABL_ROWS],
        "time": "2020",
    })
    cr = ConstraintRun(
        G=G, row_meta=rm, a0=_ABL_A_STAR.copy(), sigma_a=np.sqrt(_ABL_SIGMA_A2),
        u=np.full(4, 10.0), l=np.zeros(4),
        timestep_data={"2020": {"c": c, "sigma_c": np.array([r[6] for r in _ABL_ROWS])}},
        run_id="abl", acc_id="a", agg_id="g", created_at="2026-10-20T00:00:00",
        G_blocks={"2020": G}, edge_cols_by_t={"2020": np.arange(4)},
    )
    opt = {"2020": OptimResult(
        timestep="2020", F_opt=_ABL_A_STAR, eps_opt=np.zeros(0), c_star=c,
        residuals=G @ _ABL_A_STAR - c, objective=0.0, iterations=0, solve_time_s=0.0,
        solver_status="Solved", convergence={}, cols_t=np.arange(4),
        rows_t=np.arange(len(_ABL_ROWS)),
    )}
    return cr, opt


def _dense_sigma2(rows):
    """Reference σ*² with S over the given rows of ``_ABL_ROWS`` only."""
    G = np.array([_ABL_ROWS[i][0] for i in rows], dtype=float)
    sc2 = np.array([0.0 if _ABL_ROWS[i][3] else _ABL_ROWS[i][6] ** 2 for i in rows])
    S = G @ np.diag(_ABL_SIGMA_A2) @ G.T + np.diag(sc2)
    return _ABL_SIGMA_A2 - _ABL_SIGMA_A2 ** 2 * np.einsum("ij,ij->j", G, np.linalg.solve(S, G))


_NOT_SCALE = [i for i, r in enumerate(_ABL_ROWS)
              if not (r[1] == "sum" and r[2] in SCALE_TOTALS)]


class TestScaleTotalAblation:

    def test_scale_total_rows_are_read_from_type_and_variant(self):
        from pysuto.uncprop import SCALE_TOTAL_VARIANTS, KalmanUncProp
        assert set(SCALE_TOTAL_VARIANTS) == set(SCALE_TOTALS)
        cr, _ = _ablation_run()
        mask = KalmanUncProp._scale_total_mask(cr.row_meta)
        picked = cr.row_meta.loc[mask, ["type", "variant"]]
        assert set(map(tuple, picked.to_numpy())) == {("sum", v) for v in SCALE_TOTALS}
        np.testing.assert_array_equal(np.flatnonzero(~mask), _NOT_SCALE)

    def test_ablation_leaves_exactly_the_scale_totals_out_of_S(self, tmp_path,
                                                                monkeypatch):
        cr, opt = _ablation_run()
        base = _prop(cr, opt, tmp_path, monkeypatch).compute("o").per_timestep["2020"]
        abl = _prop(cr, opt, tmp_path, monkeypatch,
                    exclude_scale_totals=True).compute("o").per_timestep["2020"]
        np.testing.assert_allclose(abl.sigma_a_star ** 2, _dense_sigma2(_NOT_SCALE),
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(base.sigma_a_star ** 2,
                                   _dense_sigma2(range(len(_ABL_ROWS))),
                                   rtol=1e-10, atol=1e-12)
        assert abl.diagnostics["n_scale_totals_left_out"] == 4
        assert base.diagnostics["n_scale_totals_left_out"] == 0
        assert base.diagnostics["n_rows_kept"] - abl.diagnostics["n_rows_kept"] == 4

    def test_default_keeps_the_scale_totals(self, tmp_path, monkeypatch):
        import inspect
        from pysuto.uncprop import KalmanUncProp
        knob = inspect.signature(KalmanUncProp).parameters["exclude_scale_totals"]
        assert knob.default is False
        cr, opt = _ablation_run()
        base = _prop(cr, opt, tmp_path, monkeypatch).compute("o").per_timestep["2020"]
        expl = _prop(cr, opt, tmp_path, monkeypatch,
                     exclude_scale_totals=False).compute("o").per_timestep["2020"]
        for k in ("sigma_a_star", "lower", "upper"):
            np.testing.assert_array_equal(getattr(base, k), getattr(expl, k))

    def test_the_knob_is_recorded_in_stage_config(self, tmp_path, monkeypatch):
        from pysuto.registry import UncPropRunRegistry
        cr, opt = _ablation_run()
        prop = _prop(cr, opt, tmp_path, monkeypatch, exclude_scale_totals=True)
        rid = prop.persist(prop.compute("o"), "o")
        cfg = UncPropRunRegistry(str(tmp_path / "uncprop")).load_sidecar(rid)["stage_config"]
        assert cfg["exclude_scale_totals"] is True

    def test_the_full_g_path_refuses_the_ablation(self, tmp_path, monkeypatch):
        import dataclasses
        cr, opt = _ablation_run()
        cr = dataclasses.replace(cr, G_blocks={}, edge_cols_by_t={})
        with pytest.raises(ValueError, match="block path"):
            _prop(cr, opt, tmp_path, monkeypatch,
                  exclude_scale_totals=True).compute("o")
