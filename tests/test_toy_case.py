"""
test_toy_case.py
================
The toy case (#59), the repo's first six-stage end-to-end test. One module-scoped chain
runs every stage of :mod:`pysuto.cases.toy` into a temporary run tree, and the acceptance
tests read that tree. The unit tests check the generator itself (the truth balances, each
synthetic feed is noisy at its pedigree CV, every imputer writes) and the tutorial's
exercise hooks. The structure is the toy module's own ``ACC_SPEC``: no test fixture of a
toy world is declared here.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuto.acc import AccountingStructure, check_built_acc
from pysuto.cases import toy
from pysuto.diag import FeedDiagnostics, SankeySpec, TruthDiag, sankey_tables
from pysuto.feedpipe import (build_F_src, propagate_with_row_penalty,
                             step_B_build_concordance, step_C_build_weight_matrix)
from pysuto.registry import (ConstraintRunRegistry, DataFeedRegistry, ImputationRunRegistry,
                             OptimRunRegistry, PsutRunRegistry, UncPropRunRegistry)

PLANTED_KEY = ("from_sector", "from_region", "to_sector", "to_region", "time", "valuation")


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
    ids["psut"]    = toy.run_stage6(root, ids["opt"])
    return {"root": root, "ids": ids, "seconds": time.perf_counter() - t0,
            "acc": AccountingStructure.load(toy.ACC_ID, storage_dir=str(root / "acc"))}


def _edge_key(edge_meta: pd.DataFrame, edge_id: int) -> tuple:
    row = edge_meta.loc[edge_id]
    return tuple(str(row[k]) for k in PLANTED_KEY)


def _node_balance(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Inflow and outflow of every Process and Flow node, per (time, valuation)."""
    keys = ["time", "valuation", "region", "sector"]
    inflow = (frame.groupby(["time", "valuation", "to_region", "to_sector"])[value].sum()
                   .rename_axis(keys).rename("inflow"))
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
    assert int(acc.masks["cross_region"].sum()) == 6 * 5 * 2
    assert int((~acc.masks["massbal_scope"]).sum()) == 3 * 5 * 2   # Mining → WasteRock
    stock = acc.edge_meta["to_sector"].eq("Stock").to_numpy()
    assert np.all(acc.bounds.l[stock] < 0.0) and np.all(acc.bounds.l[~stock] == 0.0)
    check_built_acc(str(tmp_path), toy.ACC_REF, require_record=True)      # #53's record
    loaded = AccountingStructure.load(toy.ACC_ID, storage_dir=str(tmp_path))
    pd.testing.assert_frame_equal(loaded.edge_index, acc.edge_index)
    np.testing.assert_array_equal(loaded.masks["massbal_scope"], acc.masks["massbal_scope"])


def test_truth_balances_every_process_and_flow_node_in_both_valuations():
    truth = toy.truth()
    assert len(truth) == 420
    satellite = truth["from_sector"].eq("Mining") & truth["to_sector"].eq("WasteRock")
    nodes = _node_balance(truth[~satellite], "value")
    assert len(nodes) == 6 * 3 * 5 * 2           # 6 balanced sectors × regions × years × 2
    np.testing.assert_allclose(nodes["inflow"], nodes["outflow"], rtol=0.0, atol=1e-12)
    assert truth.loc[truth["to_sector"].eq("Stock"), "value"].lt(0).any()   # the signed leg
    portua_mine = truth["from_region"].eq("Portua") & truth["to_sector"].eq("Mining")
    assert truth.loc[portua_mine, "value"].eq(0.0).all()                    # no mines


def test_the_truth_and_the_feeds_are_reproducible_from_the_seed():
    pd.testing.assert_frame_equal(toy.truth(), toy.truth())
    a, b = toy.make_feeds(toy.truth()), toy.make_feeds(toy.truth())
    for feed_id in toy.FEED_IDS:
        pd.testing.assert_frame_equal(a[feed_id].raw, b[feed_id].raw)
    assert not toy.truth(seed=1).equals(toy.truth())


def test_each_feed_is_noisy_at_its_pedigree_cv_over_its_covered_edges(chain):
    """Root level, against the truth, planted leg excluded. Survey and Customs map 1:1;
    the Bureau's fan-in edges pool two series, so its CV sits a little below."""
    root, acc = chain["root"], chain["acc"]
    truth     = toy.truth_vector(toy.truth(), acc)
    planted   = toy._edge_id(acc, *(toy.reveal()[k] for k in PLANTED_KEY))
    registry  = DataFeedRegistry(odir=str(root / "feed"))
    registry.load_index()
    for feed_id in toy.FEED_IDS:
        result  = registry.load_result(feed_id)
        covered = result["sigma_root"] > 0
        covered[planted] = False
        rel     = (result["F_root"][covered] - truth[covered]) / truth[covered]
        cv_emp  = float(np.sqrt(np.mean(rel ** 2)))
        cv_ped  = registry.load_metadata(feed_id).to_pedigree_scores().total_cv()
        assert abs(cv_emp / cv_ped - 1.0) <= 0.10, (feed_id, cv_emp, cv_ped)


def test_every_registered_imputer_writes_and_the_time_imputer_fills_the_two_gaps(chain):
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    imp      = ImputationRunRegistry(str(root / "imp"))
    log      = imp.load(ids["imp"]).imputation_log.as_dataframe()
    registered = {r["imputer_id"] for r in imp.load_sidecar(ids["imp"])["imputers"]}
    assert registered == set(toy.CASCADE) | {"time_interp", "flat_prior"}
    assert all((log["imputer_id"] == i).any() for i in registered), \
        log.groupby("imputer_id").size().to_dict()

    bureau   = toy.make_feeds(toy.truth())["bureau"].raw
    gaps     = ({(r, t) for r in toy.REGIONS for t in toy.YEARS}
                - set(zip(bureau["Region"], bureau["Year"])))
    assert len(gaps) == 2
    em       = acc.edge_meta.set_index("edge_id")
    filled   = em.loc[log.loc[log["imputer_id"] == "time_interp", "edge_id"].astype(int)]
    assert (filled["from_region"] == filled["to_region"]).all()
    assert set(zip(filled["from_region"], filled["time"])) == gaps


def test_reveal_names_the_one_customs_row_that_is_off_by_its_factor(chain):
    planted = toy.reveal()
    feed    = toy.make_feeds(toy.truth())[planted["feed"]]
    code    = dict(feed.maps["region_map_df"][["src_region", "root_region"]].values)
    raw     = feed.raw.assign(from_region=feed.raw["Region"].map(code),
                              to_region=feed.raw["Partner"].map(code))
    tv      = toy.truth().set_index(list(PLANTED_KEY))["value"]
    ratio   = raw["Value"].to_numpy() / np.array([
        tv.loc[("Concentrate", r["from_region"], "Smelting", r["to_region"], r["Year"],
                "Mass")] for _, r in raw.iterrows()])
    off = np.flatnonzero(ratio > 100.0)
    assert off.size == 1
    row = raw.iloc[off[0]]
    assert ("Concentrate", row["from_region"], "Smelting", row["to_region"], row["Year"],
            "Mass") == tuple(planted[k] for k in PLANTED_KEY)
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
    assert eff.set_index("param_id").loc["effsmelting", "agg"] == "true"   # pooled inputs


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
    const = ConstraintRunRegistry(str(root / "const")).load(ids["const"])
    hard  = const.row_meta["is_hard"].to_numpy().astype(bool)
    assert hard.sum() == 180 + 15                 # mass balances + stripping rows
    for t, res in OptimRunRegistry(str(root / "opt")).load(ids["opt"]).items():
        assert res.solver_status == "Solved"
        assert np.abs(res.residuals[hard[res.rows_t]]).max() <= 1e-10, t


def test_the_planted_leg_is_the_top_soft_residual(chain):
    soft = toy.soft_residuals(chain["root"], chain["ids"]["opt"])
    top  = soft.iloc[0]
    assert top["kind"] == "edge"
    assert _edge_key(chain["acc"].edge_meta, int(top["id"])) == \
        tuple(toy.reveal()[k] for k in PLANTED_KEY)
    assert abs(top["z"]) > 1.5 * abs(soft["z"].iloc[2])   # the leg and its CC twin lead


def test_stage5_coverage_of_the_420_edges_lies_in_its_recorded_band(chain):
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    truth   = toy.truth_vector(toy.truth(), acc)
    covered = np.zeros(acc.n_edges, dtype=bool)
    post    = UncPropRunRegistry(str(root / "uncprop")).load(ids["uncprop"])
    for ts in post.per_timestep.values():
        cols = np.asarray(ts.cols_t)
        covered[cols] = (ts.lower <= truth[cols]) & (truth[cols] <= ts.upper)
    coverage = covered.mean()
    # Band fixed on the first run of this test (2026-11-11, seed 59): observed 0.936,
    # 393 of 420 edges inside their 95 % interval. Never loosen it.
    assert 0.90 <= coverage <= 0.97, coverage
    assert toy.scoreboard(root, ids).coverage() == pytest.approx(coverage)


def test_the_psut_supply_and_use_blocks_balance_per_node(chain):
    root, ids, acc = chain["root"], chain["ids"], chain["acc"]
    table = PsutRunRegistry(str(root / "psut")).load(ids["psut"]).table
    assert set(table["block"]) == {"S", "U", "Y", "E", "O", "P"}
    scoped = table[acc.masks["massbal_scope"][table["edge_id"].to_numpy()]]
    nodes  = _node_balance(scoped.rename(columns={"period": "time"}), "val")
    assert len(nodes) == 180
    np.testing.assert_allclose(nodes["inflow"], nodes["outflow"], rtol=0.0, atol=1e-8)


def test_the_sankey_tables_sum_to_the_psut_totals(chain):
    """Per block, over the PSUT's positive entries: the Sankey draws no link ≤ 0."""
    psut = PsutRunRegistry(str(chain["root"] / "psut")).load(chain["ids"]["psut"])
    for period in psut.periods():
        for valuation in psut.valuations:
            _, links = sankey_tables(psut, SankeySpec(period=period, valuation=valuation))
            rows = psut.table[(psut.table["period"] == period)
                              & (psut.table["valuation"] == valuation)
                              & (psut.table["val"] > 0)]
            pd.testing.assert_series_equal(
                links.groupby("block")["value"].sum().sort_index(),
                rows.groupby("block")["val"].sum().sort_index(),
                check_names=False, rtol=1e-12)


# =============================================================================
# Diagnostics and the tutorial's hooks
# =============================================================================

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
    text = board.summary()
    assert "coverage" in text and text in capsys.readouterr().out


def test_truth_diag_refuses_arrays_of_the_wrong_length(chain):
    em = chain["acc"].edge_meta
    with pytest.raises(ValueError, match="estimate 'prior'"):
        TruthDiag({"prior": np.zeros(3)}, np.zeros(len(em)), None, em)


def test_feed_diagnostics_draw_the_synthetic_feeds(chain, tmp_path):
    acc = chain["acc"]
    for feed_id, feed in toy.make_feeds(toy.truth()).items():
        ct = step_B_build_concordance(feed.raw, acc, **feed.maps, feed_id=feed_id,
                                      aux_col=feed.aux_col, allow_unmatched_edges=False)
        wm = step_C_build_weight_matrix(ct)
        F_src = build_F_src(feed.raw, ct, label=feed_id)
        result, _ = propagate_with_row_penalty(wm, F_src, feed.metadata, acc)
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
    cid  = soft.loc[soft["kind"] == "row", "id"].iloc[0]
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


def test_imports_flow_one_way_cases_on_top_and_truth_diag_a_leaf():
    src = Path(toy.__file__).resolve().parents[1]
    imports_cases = re.compile(r"^\s*(from|import)\s+(pysuto\.cases|\.+cases)\b", re.MULTILINE)
    importers = [str(p.relative_to(src)) for p in src.rglob("*.py")
                 if p.parent.name != "cases"
                 and imports_cases.search(p.read_text(encoding="utf-8"))]
    assert importers == []
    truth_imports = re.findall(r"^(?:from|import) (\S+)",
                               (src / "diag" / "truth.py").read_text(encoding="utf-8"),
                               flags=re.MULTILINE)
    assert set(truth_imports) == {"__future__", "os", "typing", "numpy", "pandas",
                                  "scipy.stats", "._plot"}


def test_solution_returns_each_reference_source_and_refuses_an_unknown_key():
    for key in ("1.1", "2.1", "3.1", "4.1", "5.1", "6.1"):
        assert toy.solution(key).startswith(f"def _ref_{key.replace('.', '_')}(")
    with pytest.raises(KeyError, match="'9.9'"):
        toy.solution("9.9")
