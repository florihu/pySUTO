"""
test_uncprop_diag.py
====================
Smoke and round-trip tests for :class:`pysuto.diag.UncPropDiag`.

The chain is real but tiny: the block-path toy of ``test_solver_diag`` (engine
ConstraintRun with mass balances, made full-rank, solved by Clarabel) propagated
by ``KalmanUncProp``. The tests cover:
  1. the portfolio (2 figures + ``contraction_summary.csv``) renders live;
  2. ``from_uncprop_id`` reloads result, opt run, const run and the sidecar's
     σ conditioning, and renders under the uncprop run's ``diag/``;
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


def _full_rank_block_run(tmp_path, monkeypatch):
    """The block-path toy of ``test_solver_diag`` minus region B's Ore balance,
    solved by Clarabel → (year-local results, ConstraintRun, acc). Region B is a
    closed two-node loop whose Mining and Ore balances are each other's negatives;
    with both, the hard rows are rank-deficient and Stage 5's ``S_t`` is singular."""
    import dataclasses
    from test_constraints import _make_toy_acc, _make_toy_agg_result  # type: ignore
    from pysuto.constraints import ConstraintEngine, MassBalanceBuilder
    from pysuto.solver import ClarabelSolver

    acc = _make_toy_acc()
    engine = ConstraintEngine(acc, storage_dir=str(tmp_path / "const"))
    engine.register(MassBalanceBuilder())
    cr = engine.run(_make_toy_agg_result(acc), timesteps=["2020", "2021"], persist=False)

    rm   = cr.row_meta
    keep = ~((rm["sector"] == "Ore") & (rm["region"] == "B")).to_numpy()
    G    = cr.G.tocsr()[keep]
    rm   = rm[keep].reset_index(drop=True)
    row_time = rm["time"].astype(str).to_numpy()
    cr = dataclasses.replace(
        cr, G=G, row_meta=rm,
        timestep_data={t: {**d, "c": np.asarray(d["c"])[keep],
                              "sigma_c": np.asarray(d["sigma_c"])[keep]}
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
    results, cr, acc = _full_rank_block_run(tmp_path, monkeypatch)
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

        results, cr, acc = _full_rank_block_run(tmp_path, monkeypatch)
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
