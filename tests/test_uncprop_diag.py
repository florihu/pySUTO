"""
test_uncprop_diag.py
====================
Smoke and round-trip tests for :class:`pysuto.diag.UncPropDiag`.

The chain is real but tiny: the block-path toy of ``test_solver_diag`` (engine
ConstraintRun with mass balances — region B's two are redundant — plus one soft
row per year, solved by Clarabel) propagated by ``KalmanUncProp``. The tests cover:
  1. the portfolio (2 figures + ``contraction_summary.csv``) renders live;
  2. ``from_uncprop_id`` reloads result, opt run, const run, imp run (prior
     classes) and the sidecar's σ conditioning, and renders under ``diag/``;
  3. registry, plot selection, ``regions=`` suffix, ``timesteps=`` validation;
  4. ``summary()`` reports the trace contraction and the quadrant shares.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless backend before any pyplot import

import numpy as np
import pytest


STEMS = ["ud_r_01_contraction_shift", "ud_r_02_prior_posterior_bands"]


def _block_run(tmp_path, monkeypatch, agg_id=""):
    """The block-path toy of ``test_solver_diag`` plus one soft row per year
    (``Mining B → Ore B`` = 50 kt, σ_c 2), solved by Clarabel → (year-local
    results, ConstraintRun, acc). Region B is a closed two-node loop whose Mining
    and Ore balances are each other's negatives: Stage 5 must drop one (#27)."""
    import dataclasses
    import pandas as pd
    import scipy.sparse as sp
    from test_constraints import _make_toy_acc, _make_toy_agg_result  # type: ignore
    from pysuto.constraints import ConstraintEngine, MassBalanceBuilder
    from pysuto.solver import ClarabelSolver

    acc = _make_toy_acc()
    engine = ConstraintEngine(acc, storage_dir=str(tmp_path / "const"))
    engine.register(MassBalanceBuilder())
    cr = engine.run(_make_toy_agg_result(acc), timesteps=["2020", "2021"], persist=False)

    rm   = cr.row_meta
    em   = acc.edge_meta.reset_index(drop=True)
    soft = em.index[(em["from_sector"] == "Mining") & (em["from_region"] == "B")]
    years = em.loc[soft, "time"].astype(str).tolist()
    soft_G = sp.csr_matrix((np.ones(len(soft)), (np.arange(len(soft)), soft)),
                           shape=(len(soft), cr.G.shape[1]))
    soft_rm = pd.DataFrame([{**{c: None for c in rm.columns},
                             "constraint_id": f"soft_mining_b_{y}", "type": "sum",
                             "variant": "anchor", "is_hard": False, "cone_type": "free",
                             "time": y, "valuation": "Mass"} for y in years])
    G  = sp.vstack([cr.G.tocsr(), soft_G]).tocsr()
    rm = pd.concat([rm, soft_rm], ignore_index=True)
    rm["is_hard"] = rm["is_hard"].astype(bool)
    row_time = rm["time"].astype(str).to_numpy()
    n_soft = len(soft)
    cr = dataclasses.replace(
        cr, G=G, row_meta=rm, agg_id=agg_id,
        timestep_data={t: {**d,
                           "c": np.r_[np.asarray(d["c"]), np.full(n_soft, 50.0)],
                           "sigma_c": np.r_[np.asarray(d["sigma_c"]),
                                            np.full(n_soft, 2.0)]}
                       for t, d in cr.timestep_data.items()},
        G_blocks={t: G[np.flatnonzero(row_time == t)][:, cols].tocsr()
                  for t, cols in cr.edge_cols_by_t.items()},
    )
    s = ClarabelSolver(const_dir=str(tmp_path / "const"), verbose_first_solve=False)
    monkeypatch.setattr(s, "load_run", lambda _rid: cr)
    return s.solve_all(cr.run_id), cr, acc


def _propagate(results, cr, tmp_path, monkeypatch):
    """Stage 5 on the in-memory chain (registries monkeypatched)."""
    from pysuto.uncprop import KalmanUncProp
    prop = KalmanUncProp(opt_dir=str(tmp_path / "opt"), const_dir=str(tmp_path / "const"),
                         uncprop_dir=str(tmp_path / "uncprop"))
    monkeypatch.setattr(prop._opt_registry, "load_sidecar",
                        lambda _rid: {"parents": {"upstream": {"const_run_id": cr.run_id}}})
    monkeypatch.setattr(prop._opt_registry, "load", lambda _rid: results)
    monkeypatch.setattr(prop._const_registry, "load", lambda _rid: cr)
    return prop.compute("opt_fake", alpha=0.05)


@pytest.fixture
def chain(tmp_path, monkeypatch):
    results, cr, acc = _block_run(tmp_path, monkeypatch)
    return _propagate(results, cr, tmp_path, monkeypatch), results, cr, acc


@pytest.fixture
def diag(chain):
    from pysuto.diag import UncPropDiag
    unc, results, cr, acc = chain
    return UncPropDiag(unc, results, cr, label="toy", acc=acc)


# ---------------------------------------------------------------------------
# 1. Live construction → portfolio
# ---------------------------------------------------------------------------

class TestLivePortfolio:

    def test_figures_and_summary_csv_written(self, diag, tmp_path):
        paths = diag.plot(odir=str(tmp_path / "diag"))
        names = [os.path.basename(p) for p in paths]
        assert names == [f"{s}.png" for s in STEMS] + ["contraction_summary.csv"]
        for p in paths:
            assert os.path.getsize(p) > 0

    def test_contraction_summary_covers_every_edge_year(self, diag, chain, tmp_path):
        import pandas as pd
        unc = chain[0]
        diag.plot(odir=str(tmp_path / "diag"), plot_ids=[])
        s = pd.read_csv(tmp_path / "diag" / "contraction_summary.csv")
        n = sum(ts.sigma_a_star.size for ts in unc.per_timestep.values())
        assert s["n_edges"].sum() == n
        assert set(s["quadrant"]) <= {"prior-carried", "confirmed", "overruled",
                                      "unexplained move"}

    def test_redundant_balance_is_dropped_not_fatal(self, chain):
        unc = chain[0]
        assert all(ts.diagnostics["n_hard_dependent"] == 1
                   for ts in unc.per_timestep.values())

    def test_frame_bound_active_is_stage5s_pinned_set(self, diag, chain):
        unc = chain[0]
        f = diag.stats.edge_frame()
        for t, ts in unc.per_timestep.items():
            ft = f[f["year"] == t].set_index("edge_id").loc[ts.cols_t]
            np.testing.assert_array_equal(ft["bound_active"].to_numpy(), ts.bound_active)

    def test_frame_kappa_matches_stage5_sigma(self, diag, chain):
        unc = chain[0]
        f = diag.stats.edge_frame()
        ts = unc.per_timestep["2020"]
        f20 = f[f["year"] == "2020"].set_index("edge_id").loc[ts.cols_t]
        np.testing.assert_allclose(f20["sigma_a_star"].to_numpy(), ts.sigma_a_star,
                                   rtol=1e-12)
        assert ((f20["kappa"] >= -1e-12) & (f20["kappa"] <= 1 + 1e-12)).all()

    def test_live_without_odir_raises(self, diag):
        with pytest.raises(ValueError, match="odir is required"):
            diag.plot()


class TestSummary:

    def test_summary_reports_trace_contraction_and_shares(self, diag, capsys):
        text = diag.summary()
        assert "trace κ" in text
        assert "prior-carried" in text and "overruled" in text
        assert "no imputation run" in text
        assert text in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 2. from_uncprop_id round trip
# ---------------------------------------------------------------------------

class TestFromUncpropId:

    def test_roundtrip_renders_under_the_run_dir(self, tmp_path, monkeypatch):
        from pysuto.diag import UncPropDiag
        from pysuto.registry import ConstraintRunRegistry
        from pysuto.solver import ClarabelSolver
        from pysuto.uncprop import KalmanUncProp

        results, cr, acc = _block_run(tmp_path, monkeypatch)
        dirs = {k: str(tmp_path / k) for k in ("const", "opt", "uncprop", "imp")}
        ConstraintRunRegistry(dirs["const"]).register(cr)
        const_run_id = cr.run_id
        s = ClarabelSolver(const_dir=dirs["const"], verbose_first_solve=False)
        opt_run_id = s.persist(run_id=const_run_id, results=results,
                               output_dir=dirs["opt"], const_run_id=const_run_id)

        prop = KalmanUncProp(opt_dir=dirs["opt"], const_dir=dirs["const"],
                             uncprop_dir=dirs["uncprop"], sigma_a_floor_abs=1e-6)
        unc_id = prop.persist(prop.compute(opt_run_id), opt_run_id)

        diag = UncPropDiag.from_uncprop_id(
            unc_id, uncprop_dir=dirs["uncprop"], opt_dir=dirs["opt"],
            const_dir=dirs["const"], imp_dir=dirs["imp"], acc=acc,
        )
        assert diag.label == unc_id
        assert diag.stats._cond["sigma_a_floor_abs"] == pytest.approx(1e-6)
        paths = diag.plot()
        expected = os.path.join(dirs["uncprop"], unc_id, "diag")
        assert len(paths) == len(STEMS) + 1
        assert all(p.startswith(expected) and os.path.exists(p) for p in paths)

    def test_roundtrip_reads_prior_classes_from_the_imp_run(self, tmp_path, monkeypatch):
        """With the const run's agg_id registered, prior classes come from the
        imp run's coverage and imputation log (2020 feed-covered, 2021 imputed
        except the Waste edge, logged as a flat prior)."""
        from types import SimpleNamespace
        from pysuto.diag import UncPropDiag
        from pysuto.impute import ImputationLog
        from pysuto.registry import ConstraintRunRegistry, ImputationRunRegistry
        from pysuto.solver import ClarabelSolver
        from pysuto.uncprop import KalmanUncProp

        dirs = {k: str(tmp_path / k) for k in ("const", "opt", "uncprop", "imp")}
        log = ImputationLog()
        log.extend([{"edge_id": e, "imputer_id": "network", "method": "network_forward"}
                    for e in (5, 6, 7, 8)]
                   + [{"edge_id": 9, "imputer_id": "flat_prior",
                       "method": "flat_prior_noisy_zero"}])
        z = np.zeros((1, 10))
        agg = SimpleNamespace(f_agg=np.ones(10), sigma_agg=np.ones(10), Omega=z, F=z,
                              Sigma=z, coverage=np.arange(10) < 5, feed_ids=["toy"],
                              imputation_log=log, imputation_metadata=None)
        imp_id = ImputationRunRegistry(dirs["imp"]).register(
            agg, label="toy", parents={"inputs": {}, "upstream": {}})

        results, cr, acc = _block_run(tmp_path, monkeypatch, agg_id=imp_id)
        ConstraintRunRegistry(dirs["const"]).register(cr)
        opt_run_id = ClarabelSolver(const_dir=dirs["const"], verbose_first_solve=False) \
            .persist(run_id=cr.run_id, results=results, output_dir=dirs["opt"],
                     const_run_id=cr.run_id)
        prop = KalmanUncProp(opt_dir=dirs["opt"], const_dir=dirs["const"],
                             uncprop_dir=dirs["uncprop"])
        unc_id = prop.persist(prop.compute(opt_run_id), opt_run_id)

        diag = UncPropDiag.from_uncprop_id(
            unc_id, uncprop_dir=dirs["uncprop"], opt_dir=dirs["opt"],
            const_dir=dirs["const"], imp_dir=dirs["imp"], acc=acc,
        )
        f = diag.stats.edge_frame().set_index("edge_id")
        assert set(f.loc[f["year"] == "2020", "prior_class"]) == {"from a feed"}
        assert list(f.loc[[5, 6, 7, 8, 9], "prior_class"]) == ["imputed"] * 4 + ["flat"]
        assert "no imputation run" not in diag.summary()

    def test_unknown_run_id_raises(self, tmp_path):
        from pysuto.diag import UncPropDiag
        with pytest.raises(KeyError, match="uncprop_nope"):
            UncPropDiag.from_uncprop_id("uncprop_nope", uncprop_dir=str(tmp_path / "u"))


# ---------------------------------------------------------------------------
# 3. Registry, selection, filters
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_every_spec_method_exists_and_stems_are_unique(self):
        from pysuto.diag import UncPropDiag
        assert list(UncPropDiag.PLOTS) == ["UD-R-01", "UD-R-02"]
        for spec in UncPropDiag.PLOTS.values():
            assert callable(getattr(UncPropDiag, spec.method))
            assert spec.desc
        stems = [s.filename for s in UncPropDiag.PLOTS.values()]
        assert stems == STEMS and len(set(stems)) == len(stems)

    def test_subset_and_unknown_id(self, diag, tmp_path):
        paths = diag.plot(odir=str(tmp_path / "d"), plot_ids=["UD-R-02"])
        assert [os.path.basename(p) for p in paths] == [
            "ud_r_02_prior_posterior_bands.png", "contraction_summary.csv"]
        with pytest.raises(ValueError, match="UD-R-99"):
            diag.plot(odir=str(tmp_path / "d"), plot_ids=["UD-R-99"])

    def test_regions_filter_suffixes_outputs(self, diag, tmp_path):
        paths = diag.plot(odir=str(tmp_path / "d"), regions=["A"])
        names = sorted(os.path.basename(p) for p in paths)
        assert names == ["contraction_summary__a.csv",
                         "ud_r_01_contraction_shift__a.png",
                         "ud_r_02_prior_posterior_bands__a.png"]
        f = diag._frame(regions=["A"])
        assert 0 < len(f) and ((f["from_region"] == "A") | (f["to_region"] == "A")).all()

    def test_timesteps_narrow_the_frame_and_reject_unknown_years(self, diag):
        assert set(diag._frame(timesteps=["2021"])["year"]) == {"2021"}
        with pytest.raises(KeyError, match="1999"):
            diag._frame(timesteps=["1999"])
