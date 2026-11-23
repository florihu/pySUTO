"""
test_toy_case.py
================
The toy case (#59), the repo's first six-stage end-to-end test. One module-scoped chain
runs every stage of :mod:`pysuto.tutorial.toy` into a temporary run tree, and the acceptance
tests read that tree. The unit tests check the generator itself (the truth balances, each
synthetic feed is noisy at its pedigree CV, every imputer writes) and the tutorial's
exercise hooks. The structure is the toy module's own ``ACC_SPEC``: no test fixture of a
toy world is declared here.
"""

from __future__ import annotations

import ast
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuto.acc import ALL_SPEC_KEYS, AccountingStructure, check_built_acc
from pysuto.tutorial import toy
from pysuto.diag import FeedDiagnostics, SankeySpec, TruthDiag, sankey_tables
from pysuto.registry import (ConstraintRunRegistry, DataFeedRegistry,
                             ImputationRunRegistry, OptimRunRegistry, PsutRunRegistry,
                             UncPropRunRegistry)

PLANTED_KEY = ("from_sector", "from_region", "to_sector", "to_region", "time",
               "valuation")


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """All six stages of the toy case, run once into a temporary run tree."""
    root = tmp_path_factory.mktemp("toy")
    t0   = time.perf_counter()
    ids  = {"feed": toy.run_stage1(root)}
    ids["imp"]     = toy.run_stage2(root, ids["feed"])
    ids["const"]   = toy.run_stage3(root, ids["imp"])
    ids["opt"]     = toy.run_stage4(root, ids["const"])
    ids["uncprop"] = toy.run_stage5(root, ids["opt"])
    # no PNG: Kaleido needs Chrome, which the suite does not (the two PNG tests below
    # fake the figure; the tutorial job draws the real one)
    ids["psut"]    = toy.run_stage6(root, ids["opt"], png=False)
    return {"root": root, "ids": ids, "seconds": time.perf_counter() - t0,
            "acc": AccountingStructure.load(toy.ACC_ID,
                                            storage_dir=str(root / "proc" / "acc"))}


def _edge_key(edge_meta: pd.DataFrame, edge_id: int) -> tuple:
    row = edge_meta.loc[edge_id]
    return tuple(str(row[k]) for k in PLANTED_KEY)


def _node_balance(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Inflow and outflow of every Process and Flow node, per (time, valuation)."""
    keys = ["time", "valuation", "region", "sector"]
    inflow = (frame.groupby(["time", "valuation", "to_region", "to_sector"])[value]
                   .sum().rename_axis(keys).rename("inflow"))
    outflow = (frame.groupby(["time", "valuation", "from_region", "from_sector"])[value]
                    .sum().rename_axis(keys).rename("outflow"))
    both = pd.concat([inflow, outflow], axis=1).fillna(0.0).reset_index()
    entity = both["sector"].map({s["name"]: s["entity"] for s in toy.ACC_SPEC["sectors"]})
    return both[entity.isin(["Process", "Flow"]).to_numpy()]


# =============================================================================
# The generator
# =============================================================================

def test_the_acc_spec_builds_through_from_spec_and_survives_a_reload(tmp_path):
    acc = toy.build_acc(tmp_path)
    assert acc.n_nodes == 13 * 3
    assert acc.n_edges == 420                    # (11 · 3 + 9) edges × 5 years × 2
    assert acc.valuation == [toy.GROSS, toy.CONTENT]
    assert int(acc.masks["cross_region"].sum()) == 6 * 5 * 2
    assert int((~acc.masks["massbal_scope"]).sum()) == 3 * 5 * 2   # Mining → WasteRock
    stock = acc.edge_meta["to_sector"].eq("Stock").to_numpy()
    assert np.all(acc.bounds.l[stock] < 0.0) and np.all(acc.bounds.l[~stock] == 0.0)
    # #53's record; the same version needs no registry to be compared
    check_built_acc(str(tmp_path), toy.ACC_REF, None, ALL_SPEC_KEYS, require_record=True)
    loaded = AccountingStructure.load(toy.ACC_ID, storage_dir=str(tmp_path))
    pd.testing.assert_frame_equal(loaded.edge_index, acc.edge_index)
    np.testing.assert_array_equal(loaded.masks["massbal_scope"],
                                  acc.masks["massbal_scope"])


def test_truth_balances_every_process_and_flow_node_in_both_valuations():
    truth = toy.truth()
    assert len(truth) == 420
    satellite = truth["from_sector"].eq("Mining") & truth["to_sector"].eq("WasteRock")
    nodes = _node_balance(truth[~satellite], "value")
    assert len(nodes) == 6 * 3 * 5 * 2           # sectors × regions × years × valuations
    np.testing.assert_allclose(nodes["inflow"], nodes["outflow"], rtol=0.0, atol=1e-12)
    assert truth.loc[truth["to_sector"].eq("Stock"), "value"].lt(0).any()   # signed leg
    portua_mine = truth["from_region"].eq("Portua") & truth["to_sector"].eq("Mining")
    assert truth.loc[portua_mine, "value"].eq(0.0).all()                    # no mines


def test_world_totals_sum_the_truth_over_regions_and_balance_every_node():
    wt = toy.world_totals()
    assert list(wt.columns) == ["time", "valuation", "from_sector", "to_sector", "unit",
                                "value"]
    assert "from_region" not in wt.columns and "to_region" not in wt.columns
    truth = toy.truth()
    one = wt[(wt["time"] == "2003") & (wt["valuation"] == toy.CONTENT)
             & (wt["from_sector"] == "Milling") & (wt["to_sector"] == "Concentrate")]
    mask = ((truth["time"] == "2003") & (truth["valuation"] == toy.CONTENT)
            & (truth["from_sector"] == "Milling") & (truth["to_sector"] == "Concentrate"))
    assert len(one) == 1
    assert one["value"].iloc[0] == pytest.approx(truth.loc[mask, "value"].sum(),
                                                 rel=1e-12)
    assert one["unit"].iloc[0] == toy.UNITS[toy.CONTENT]
    assert wt["value"].sum() == pytest.approx(truth["value"].sum(), rel=1e-12)
    # world-level balances close for every Process and Flow node but Mining (ADR 0038)
    inside = wt[wt["to_sector"].ne("WasteRock")]
    nodes  = _node_balance(inside.assign(from_region="W", to_region="W"), "value")
    np.testing.assert_allclose(nodes["inflow"], nodes["outflow"], rtol=0.0, atol=1e-10)


def test_check_0_1_accepts_the_world_balance_and_rejects_a_wrong_one():
    inflow, outflow = toy._ref_0_1()
    assert inflow == pytest.approx(outflow, rel=1e-12)
    assert toy.check_0_1(inflow, outflow).startswith("✓ 0.1")
    with pytest.raises(AssertionError):
        toy.check_0_1(inflow + 1.0, outflow)
    wt = toy.world_totals()
    ore = wt[(wt["time"] == "2003") & (wt["valuation"] == toy.GROSS)
             & (wt["from_sector"] == "Ore")]["value"].sum()
    mill_in, mill_out = toy._ref_0_1("Milling", "2003", toy.GROSS)
    assert mill_in == pytest.approx(ore) and mill_in == pytest.approx(mill_out)
    with pytest.raises(ValueError, match="'Mining'"):
        toy.check_0_1(1.0, 1.0, sector="Mining")
    with pytest.raises(ValueError, match="'Use'"):
        toy.check_0_1(1.0, 1.0, sector="Use")


def test_the_truth_and_the_feeds_are_reproducible_from_the_seed():
    pd.testing.assert_frame_equal(toy.truth(), toy.truth())
    a, b = toy.make_feeds(toy.truth()), toy.make_feeds(toy.truth())
    for feed_id in toy.FEED_IDS:
        pd.testing.assert_frame_equal(a[feed_id].raw, b[feed_id].raw)
    assert not toy.truth(seed=1).equals(toy.truth())


def test_each_feed_is_noisy_at_its_pedigree_cv_over_its_covered_edges(chain):
    """Root level, against the truth, planted leg excluded. This checks the generator's
    rescaled noise and Stage 1's mapping together: the Survey and the Customs feed map
    1:1 onto root edges, so their empirical CV is their pedigree CV exactly. The Bureau's
    refined-output edges pool two series each, so at root level its CV sits a little
    below its pedigree CV and is held to a 10 % band."""
    root, acc = chain["root"], chain["acc"]
    truth     = toy.truth_vector(toy.truth(), acc)
    planted   = toy._edge_id(acc, *(toy.reveal()[k] for k in PLANTED_KEY))
    registry  = DataFeedRegistry(odir=str(root / "proc" / "feed" / "registry"))
    registry.load_index()
    ratio = {}
    for feed_id in toy.FEED_IDS:
        result  = registry.load_result(feed_id)
        covered = result["sigma_root"] > 0
        covered[planted] = False
        rel     = (result["F_root"][covered] - truth[covered]) / truth[covered]
        cv_ped  = registry.load_metadata(feed_id).to_pedigree_scores().total_cv()
        ratio[feed_id] = float(np.sqrt(np.mean(rel ** 2))) / cv_ped
    assert ratio["survey"]  == pytest.approx(1.0, rel=1e-10)
    assert ratio["customs"] == pytest.approx(1.0, rel=1e-10)
    assert abs(ratio["bureau"] - 1.0) <= 0.10, ratio


def test_every_registered_imputer_writes_and_the_time_imputer_fills_the_two_gaps(chain):
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    imp      = ImputationRunRegistry(str(root / "proc" / "imp"))
    log      = imp.load(ids["imp"]).imputation_log.as_dataframe()
    registered = {r["imputer_id"] for r in imp.load_sidecar(ids["imp"])["imputers"]}
    assert registered == set(toy.CASE.imp.CASCADE_ORDER) | {"time_interp", "flat_prior"}
    cascade = imp.load_sidecar(ids["imp"])["stage_config"]["case"]["imp"]["cascade"]
    assert [(e["id"], e["kind"]) for e in cascade] == [
        ("trade", "fill"), ("network", "step"), ("conc", "fill")]
    assert cascade[0]["source"]["params_id"] == "conc"
    assert all((log["imputer_id"] == i).any() for i in registered), \
        log.groupby("imputer_id").size().to_dict()

    bureau   = toy.make_feeds(toy.truth())["bureau"].raw
    bureau   = bureau[bureau["Unit"] == toy._BUREAU_UNIT]  # the overlap table has no gaps
    gaps     = ({(r, t) for r in toy.REGIONS for t in toy.YEARS}
                - set(zip(bureau["Region"], bureau["Year"])))
    assert len(gaps) == 2
    em       = acc.edge_meta.set_index("edge_id")
    filled   = em.loc[log.loc[log["imputer_id"] == "time_interp", "edge_id"].astype(int)]
    assert (filled["from_region"] == filled["to_region"]).all()
    assert set(zip(filled["from_region"], filled["time"])) == gaps


def test_two_feeds_meet_on_exactly_one_edge_and_stage_2_pools_them(chain):
    """The Survey and the Bureau both observe Isla's ore extraction in gross mass,
    in all five years: the one place where Stage 2's aggregation has two observations
    to weigh, by precision, with a posterior σ below both (barely: Stage 2's Lp rule at
    p = 1.3 pools σ conservatively, and on Montara's edge it exceeds the smaller σ in two
    of five years)."""
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    agg = np.load(root / "proc" / "imp" / ids["imp"] / "agg_result.npz",
                  allow_pickle=True)
    F, Sigma, feeds = agg["F"], agg["Sigma"], list(agg["feed_ids"])
    both = np.isfinite(Sigma).sum(axis=0) == 2
    assert np.isfinite(Sigma).sum(axis=0).max() == 2
    assert {_edge_key(acc.edge_meta, int(e)) for e in np.flatnonzero(both)} == {
        ("Deposit", "Isla", "Mining", "Isla", t, toy.GROSS) for t in toy.YEARS}
    for e in np.flatnonzero(both):
        k = [i for i in range(len(feeds)) if np.isfinite(Sigma[i, e])]
        assert {feeds[i] for i in k} == {"survey", "bureau"}
        w = 1.0 / Sigma[k, e] ** 2
        assert agg["f_agg"][e] == pytest.approx(np.sum(w * F[k, e]) / w.sum(), rel=1e-12)
        assert agg["sigma_agg"][e] < Sigma[k, e].min()


def test_reveal_names_the_one_customs_row_that_is_off_by_its_factor(chain):
    planted = toy.reveal()
    feed    = toy.make_feeds(toy.truth())[planted["feed"]]
    code    = dict(feed.maps["region_map_df"][["src_region", "root_region"]].values)
    raw     = feed.raw.assign(from_region=feed.raw["Region"].map(code),
                              to_region=feed.raw["Partner"].map(code))
    tv      = toy.truth().set_index(list(PLANTED_KEY))["value"]
    ratio   = raw["Value"].to_numpy() / np.array([
        tv.loc[("Concentrate", r["from_region"], "Smelting", r["to_region"], r["Year"],
                toy.GROSS)] for _, r in raw.iterrows()])
    off = np.flatnonzero(ratio > 100.0)
    assert off.size == 1
    row = raw.iloc[off[0]]
    assert ("Concentrate", row["from_region"], "Smelting", row["to_region"], row["Year"],
            toy.GROSS) == tuple(planted[k] for k in PLANTED_KEY)
    cv = feed.metadata.to_pedigree_scores().total_cv()
    assert abs(ratio[off[0]] / planted["factor"] - 1.0) < 5 * cv


def test_write_params_writes_the_three_families_the_builders_read(tmp_path):
    paths = toy.write_params(tmp_path)
    assert set(paths) == {"conc", "eff", "stripping_ratio"}
    for pid, path in paths.items():
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)
        assert not df["param_id"].str.contains("_").any()        # ADR 0005
        assert df["param_id"].is_unique
    eff = pd.read_csv(paths["eff"], sep=None, engine="python", dtype=str)
    assert eff.set_index("param_id").loc["effsmelting", "agg"] == "true"   # pooled


# =============================================================================
# The six stages (acceptance)
# =============================================================================

def test_six_stages_run_into_a_run_tree_in_under_60_s(chain):
    ids = chain["ids"]
    assert chain["seconds"] < 60.0, chain["seconds"]
    assert set(ids["feed"]) == set(toy.FEED_IDS)
    for stage in ("imp", "const", "opt", "uncprop", "psut"):
        assert ids[stage].startswith(f"{stage}_")


def test_every_hard_row_closes_to_1e_10(chain):
    root, ids = chain["root"], chain["ids"]
    const = ConstraintRunRegistry(str(root / "proc" / "const")).load(ids["const"])
    hard  = const.row_meta["is_hard"].to_numpy().astype(bool)
    assert hard.sum() == 180 + 15                 # mass balances + stripping rows
    for t, res in OptimRunRegistry(str(root / "proc" / "opt")).load(ids["opt"]).items():
        assert res.solver_status == "solved"
        assert np.abs(res.residuals[hard[res.rows_t]]).max() <= 1e-10, t


def test_the_toy_registers_no_scale_totals(chain):
    """2026-11-16: no scale totals in notebooks 00-07. The 90 ``usefulsupply`` sum rows
    only restated the hard Flow balances (largest |z| 5.3e-13) and are gone."""
    root, ids = chain["root"], chain["ids"]
    const = ConstraintRunRegistry(str(root / "proc" / "const")).load(ids["const"])
    assert set(const.row_meta["type"]) == {"massbal", "ratio"}
    assert not hasattr(toy.CASE.const, "SUM_SIGMA_FRACTION")


def test_the_planted_leg_is_the_top_soft_residual(chain):
    soft = toy.soft_residuals(chain["root"], chain["ids"]["opt"])
    assert str(soft["edge_id"].dtype) == "Int64" and soft["constraint_id"].dtype == object
    top  = soft.iloc[0]
    assert top["kind"] == "edge" and top["constraint_id"] is None
    assert _edge_key(chain["acc"].edge_meta, int(top["edge_id"])) == \
        tuple(toy.reveal()[k] for k in PLANTED_KEY)
    assert abs(top["z"]) > 1.5 * abs(soft["z"].iloc[2])   # the leg and its twin lead


def test_the_planted_leg_is_recovered_inside_its_interval(chain):
    """The x1000 leg's posterior mean lands near its truth (seed 59, a* 23.63 kt vs truth
    22.73 kt on ``opt_20261123_134417_2ad7561c``), inside its (1 − alpha) interval."""
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    leg   = toy._edge_id(acc, *(toy.reveal()[k] for k in PLANTED_KEY))
    truth = toy.truth_vector(toy.truth(), acc)[leg]
    a_star = toy._global(OptimRunRegistry(str(root / "proc" / "opt")).load(ids["opt"]),
                         "F_opt", acc.n_edges)[leg]
    post   = UncPropRunRegistry(str(root / "proc" / "uncprop")).load(ids["uncprop"])
    (ts,)  = [t for t in post.per_timestep.values() if leg in set(np.asarray(t.cols_t))]
    k      = int(np.flatnonzero(np.asarray(ts.cols_t) == leg)[0])
    assert ts.lower[k] <= truth <= ts.upper[k]
    assert abs(a_star - truth) < 0.10 * truth


def test_stage5_coverage_of_the_420_edges_lies_in_its_recorded_band(chain):
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    truth   = toy.truth_vector(toy.truth(), acc)
    covered = np.zeros(acc.n_edges, dtype=bool)
    post    = UncPropRunRegistry(str(root / "proc" / "uncprop")).load(ids["uncprop"])
    for ts in post.per_timestep.values():
        cols = np.asarray(ts.cols_t)
        covered[cols] = (ts.lower <= truth[cols]) & (truth[cols] <= ts.upper)
    coverage = covered.mean()
    # Band fixed on the first run of this test (2026-11-11, seed 59): observed 0.936,
    # 393 of 420 edges inside their 95 % interval. Never loosen it. With copper's
    # cascade order (review of 9c797e3): 0.933, 392 of 420. After #60's hooks (the
    # inert sum rows removed, one edge observed by two feeds): 0.943, 396 of 420.
    assert coverage == pytest.approx(0.943, abs=0.003), coverage   # 396 of 420, ±1 edge
    assert toy.scoreboard(root, ids).coverage() == pytest.approx(coverage)


def test_the_psut_supply_and_use_blocks_balance_per_node(chain):
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    table = PsutRunRegistry(str(root / "proc" / "psut")).load(ids["psut"]).table
    assert set(table["block"]) == {"S", "U", "Y", "E", "O", "P"}
    scoped = table[acc.masks["massbal_scope"][table["edge_id"].to_numpy()]]
    nodes  = _node_balance(scoped.rename(columns={"period": "time"}), "val")
    assert len(nodes) == 180
    np.testing.assert_allclose(nodes["inflow"], nodes["outflow"], rtol=0.0, atol=1e-8)


def test_the_solved_footprints_track_the_planted_truth(chain):
    """Stage 6 scored against the truth: the PIOT footprint of the solved chain and of the
    planted flows, world total of the nature extraction per period and valuation. Band
    fixed on the first run of this test (2026-11-23, seed 59): the largest relative gap
    is 0.065 (2001, Content) over the five years and both valuations. Never loosen it."""
    from types import SimpleNamespace
    from pysuto.psut import PSUT
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    solved = PsutRunRegistry(str(root / "proc" / "psut")).load(ids["psut"])
    truth  = toy.truth_vector(toy.truth(), acc)
    planted = PSUT.from_results(
        {t: SimpleNamespace(timestep=t, F_opt=truth, cols_t=None) for t in solved.periods()},
        acc)
    a, b = solved.to_piot(level="L"), planted.to_piot(level="L")
    gaps = []
    for period in solved.periods():
        for valuation in solved.valuations:
            fa = a.footprint("E", period, valuation, by=None)
            fb = b.footprint("E", period, valuation, by=None)
            gaps.append(abs(fa / fb - 1.0))
    assert len(gaps) == 10 and max(gaps) < 0.08, gaps


def test_the_sankey_tables_sum_to_the_psut_totals(chain, caplog):
    """Per block, the PSUT's total is the Sankey's links plus the links it drops; in the
    planted year the dropped part (negative posteriors, a negative stock change) is not
    zero, and the Sankey says so."""
    psut = PsutRunRegistry(str(chain["root"] / "proc" / "psut")).load(
        chain["ids"]["psut"])
    for period in psut.periods():
        for valuation in psut.valuations:
            with caplog.at_level("WARNING", logger="pysuto.diag.psut"):
                caplog.clear()
                _, links, dropped = sankey_tables(
                    psut, SankeySpec(period=period, valuation=valuation),
                    with_dropped=True)
            rows   = psut.table[(psut.table["period"] == period)
                                & (psut.table["valuation"] == valuation)]
            total  = rows.groupby("block")["val"].sum()
            shown  = links.groupby("block")["value"].sum().reindex(total.index,
                                                                   fill_value=0.0)
            np.testing.assert_allclose(
                shown + dropped.reindex(total.index, fill_value=0.0), total,
                rtol=1e-12, atol=1e-12)
            if period == toy.reveal()["time"]:
                assert dropped.sum() < 0.0, (period, valuation)
                assert "≤ 0 left out" in caplog.text


# =============================================================================
# Diagnostics and the tutorial's hooks
# =============================================================================

def test_stage6_titles_its_sankey_png_in_the_content_unit(chain, monkeypatch):
    seen = {}

    class _Figure:
        def write_image(self, *args, **kwargs):
            pass

    def _sankey(self, period=None, valuation=None, **spec):
        seen.update(spec)
        return _Figure()

    monkeypatch.setattr(toy.PSUT, "sankey", _sankey)
    toy.run_stage6(chain["root"], chain["ids"]["opt"])
    assert seen.get("unit") == toy.UNITS[toy.CONTENT]


def test_stage6_raises_with_the_cause_when_the_requested_png_cannot_be_written(
        chain, monkeypatch):
    class _Figure:
        def write_image(self, *args, **kwargs):
            raise ValueError("Chrome not found")

    monkeypatch.setattr(toy.PSUT, "sankey", lambda self, *a, **k: _Figure())
    with pytest.raises(RuntimeError, match="Chrome not found.*plotly_get_chrome"):
        toy.run_stage6(chain["root"], chain["ids"]["opt"])
    toy.run_stage6(chain["root"], chain["ids"]["opt"], png=False)


def test_truth_diag_draws_its_three_figures_and_summarises(chain, tmp_path, capsys):
    board = toy.scoreboard(chain["root"], chain["ids"])
    paths = board.plot(str(tmp_path))
    assert [p.rsplit("/", 1)[-1] for p in paths] == [
        "truth_error_by_stage.png", "truth_calibration.png", "truth_residual_rank.png"]
    assert all((tmp_path / p.rsplit("/", 1)[-1]).stat().st_size > 0 for p in paths)
    assert board.residual_rank()["planted"].iloc[0]
    errors = board.error_by_stage()
    assert list(errors.index) == ["feed", "aggregate", "prior", "reconciled"]
    assert errors.loc["reconciled", "median"] < errors.loc["prior", "median"]
    retention = board.retention(toy.CONTENT)
    assert retention.index.names == ["category", "year"]
    assert f"truth [{toy.UNITS[toy.CONTENT]}]" in retention.columns
    assert set(retention.index.get_level_values("year")) == set(toy.YEARS)
    assert "Metal → Stock" not in retention.index.get_level_values("category")  # signed
    text = board.summary()
    assert "coverage" in text and text in capsys.readouterr().out


def test_truth_diag_refuses_arrays_of_the_wrong_length(chain):
    em = chain["acc"].edge_meta
    with pytest.raises(ValueError, match="estimate 'prior'"):
        TruthDiag({"prior": np.zeros(3)}, np.zeros(len(em)), None, em)


def test_scoreboard_reads_the_seed_from_the_run_tree_and_refuses_another(chain):
    root, ids = chain["root"], chain["ids"]
    assert toy.scoreboard(root, ids, seed=toy.SEED).coverage() == \
        toy.scoreboard(root, ids).coverage()
    with pytest.raises(ValueError, match="seed=1, but"):
        toy.scoreboard(root, ids, seed=1)


def test_feed_diagnostics_draw_the_synthetic_feeds(chain, tmp_path):
    acc = chain["acc"]
    for feed_id, feed in toy.make_feeds(toy.truth()).items():
        ct, _, F_src, result, _ = toy._feed_steps(feed, acc)
        path = FeedDiagnostics(feed_id, feed_id, toy.ACC_ID, feed.raw, ct, F_src,
                               result).plot(odir=str(tmp_path / feed_id))
        assert (tmp_path / feed_id / "feed.png").exists() and path.endswith("feed.png")


def test_the_exercise_checks_accept_the_reference_and_reject_a_wrong_answer(chain):
    root = chain["root"]
    bureau_cv = toy._metadata("bureau").to_pedigree_scores().total_cv()

    cv = toy._ref_1_1(root)
    assert cv < bureau_cv                         # the diversification invariant
    assert toy.check_1_1(root, cv).startswith("✓ 1.1")
    with pytest.raises(AssertionError):
        toy.check_1_1(root, bureau_cv)

    gaps = toy._ref_2_1(root)
    assert toy.check_2_1(root, gaps).startswith("✓ 2.1")
    with pytest.raises(AssertionError):
        toy.check_2_1(root, sorted(gaps)[:1])

    n_hard, n_soft = toy._ref_3_1(root)
    assert n_hard == 39 and n_soft > 0            # 36 mass balances + 3 stripping rows
    assert toy.check_3_1(root, n_hard, n_soft).startswith("✓ 3.1")
    with pytest.raises(AssertionError):
        toy.check_3_1(root, n_hard + 1, n_soft)

    soft = toy.soft_residuals(root)
    cid  = soft.loc[soft["kind"] == "row", "constraint_id"].iloc[0]
    z    = toy._ref_4_1(root, cid)
    assert toy.check_4_1(root, cid, z).startswith("✓ 4.1")
    with pytest.raises(AssertionError):
        toy.check_4_1(root, cid, z + 1.0)

    ratio = toy._ref_5_1(root)
    assert 0.0 < ratio <= 1.0 + 1e-9
    assert toy.check_5_1(root, ratio).startswith("✓ 5.1")
    with pytest.raises(AssertionError):
        toy.check_5_1(root, 1.5)

    supply, use = toy._ref_6_1(root)
    assert toy.check_6_1(root, supply, use).startswith("✓ 6.1")
    with pytest.raises(AssertionError):
        toy.check_6_1(root, supply + 1.0, use)
    with pytest.raises(ValueError, match="'Mining'"):
        toy.check_6_1(root, supply, use, sector="Mining")


def test_imports_flow_one_way_tutorial_on_top_and_truth_diag_a_leaf():
    """The library imports no ``pysuto.tutorial`` (``__main__.py``, the
    ``python -m pysuto`` driver, and ``_step.py``, its step child, are drivers and do); ``diag/truth.py``
    imports neither the cases nor ``scripts/``."""
    src = Path(toy.__file__).resolve().parents[1]
    imports_cases   = re.compile(
        r"^\s*(from|import)\s+(pysuto\.tutorial|\.+tutorial)\b",
        re.MULTILINE)
    imports_scripts = re.compile(r"^\s*(from|import)\s+scripts\b", re.MULTILINE)
    importers = [str(p.relative_to(src)) for p in src.rglob("*.py")
                 if p.parent.name != "tutorial"
                 and p.name not in ("__main__.py", "_step.py")
                 and imports_cases.search(p.read_text(encoding="utf-8"))]
    assert importers == []
    truth_py = (src / "diag" / "truth.py").read_text(encoding="utf-8")
    assert not imports_cases.search(truth_py) and not imports_scripts.search(truth_py)


def test_solution_returns_each_reference_source_and_refuses_an_unknown_key():
    for key in ("0.1", "1.1", "2.1", "3.1", "4.1", "5.1", "6.1"):
        assert toy.solution(key).startswith(f"def _ref_{key.replace('.', '_')}(")
    with pytest.raises(KeyError, match="'9.9'"):
        toy.solution("9.9")


def test_the_toy_module_carries_none_of_copper_s_labels():
    """The toy declares its own labels (#123): no copper label is a string literal."""
    tree     = ast.parse(Path(toy.__file__).read_text(encoding="utf-8"))
    literals = {n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert literals.isdisjoint({"CC", "Mass", "SEM", "FD", "Nature"})
    assert (toy.GROSS, toy.CONTENT) == ("Gross", "Content")


# =============================================================================
# The flat-prior flag (#137)
# =============================================================================

def test_stage_2_persists_the_flat_prior_flag_of_the_log_s_flat_prior_edges(chain):
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    imp  = ImputationRunRegistry(str(root / "proc" / "imp"))
    log  = imp.load(ids["imp"]).imputation_log.as_dataframe()
    flag = np.load(root / "proc" / "imp" / ids["imp"] / "agg_result.npz")["flat_prior"]
    assert flag.dtype == bool and flag.shape == (acc.n_edges,)
    expected = np.zeros(acc.n_edges, dtype=bool)
    expected[log.loc[log["imputer_id"] == "flat_prior", "edge_id"].astype(int)] = True
    assert flag.any()
    np.testing.assert_array_equal(flag, expected)
    np.testing.assert_array_equal(imp.load(ids["imp"]).flat_prior, expected)


def _register_flagless(root, ids):
    imp = ImputationRunRegistry(str(root / "proc" / "imp"))
    agg = imp.load(ids["imp"])
    agg.flat_prior = None                              # an imp run from before #137
    old = imp.register(agg, label="no flag", parents={
        "inputs": {"acc": {}, "params": []}, "upstream": {"feed_run_ids": []}})
    assert "flat_prior" not in np.load(root / "proc" / "imp" / old / "agg_result.npz").files
    return imp, old


def test_an_imp_run_from_before_the_flag_derives_it_from_its_log(chain):
    root, ids = chain["root"], chain["ids"]
    imp, old = _register_flagless(root, ids)
    np.testing.assert_array_equal(imp.load(old).flat_prior, imp.load(ids["imp"]).flat_prior)


def test_a_const_run_on_an_imp_run_without_flag_or_log_raises_naming_it(chain):
    root, ids = chain["root"], chain["ids"]
    _, old = _register_flagless(root, ids)
    os.remove(root / "proc" / "imp" / old / "imputation_log.parquet")
    with pytest.raises(ValueError, match=old):
        toy.run_stage3(root, old)


def test_stage_5_conditions_the_same_prior_vector_as_stage_4(chain):
    """Stage 5 conditions ``const.a0``, Stage 4 ``const.a_prior_global``: on the block
    path the two are one vector (the sigma floor is relative to |a0|), #285."""
    root, ids = chain["root"], chain["ids"]
    const = ConstraintRunRegistry(str(root / "proc" / "const")).load(ids["const"])
    assert const.G_blocks
    np.testing.assert_array_equal(const.a_prior_global, np.asarray(const.a0, dtype=float))
