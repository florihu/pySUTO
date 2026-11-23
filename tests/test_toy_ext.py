"""
test_toy_ext.py
===============
The toy case's extras (#61, :mod:`pysuto.tutorial.toy_ext`): the three ``break_*``
experiments and the three seam examples. One module-scoped base chain (Stages 1-5) in a
temporary run tree; every experiment runs into that tree under its own registry label.
The assertions are the qualitative facts the notebooks teach, not the numbers of one run.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

from pysuto.tutorial import toy, toy_ext
from pysuto.impute.ledger import run_imputer
from pysuto.registry import (ConstraintRunRegistry, DataFeedRegistry,
                             OptimRunRegistry)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """The toy chain, Stages 1-5, in a temporary run tree; its run ids."""
    root = tmp_path_factory.mktemp("toy_ext")
    feed = toy.run_stage1(root)
    imp  = toy.run_stage2(root, feed)
    const = toy.run_stage3(root, imp)
    opt  = toy.run_stage4(root, const)
    uncprop = toy.run_stage5(root, opt)
    return {"root": root, "base": {"imp": imp, "const": const, "opt": opt,
                                   "uncprop": uncprop}}


def _core_latest(root) -> dict:
    """The latest ``"toy"``-labelled run of each stage: what the core chapters read."""
    return toy_ext.base_ids(root)


@pytest.fixture(scope="module")
def contradiction(tree):
    return toy_ext.break_contradiction(tree["root"], row_cvs=(0.01, 1.0))


def test_the_variant_refuses_the_core_label():
    with pytest.raises(ValueError, match="core label"):
        with toy_ext._variant(toy.LABEL):
            pass


def test_the_variant_restores_the_core_label_after_an_error():
    with pytest.raises(RuntimeError):
        with toy_ext._variant("x"):
            assert toy.LABEL == "x"
            raise RuntimeError
    assert toy.LABEL == "toy"


# =============================================================================
# break_contradiction
# =============================================================================

def test_loosening_the_contradicted_row_hides_its_residual(contradiction):
    t = contradiction["table"]
    assert list(t.index) == ["base", "cv=0.01", "cv=1", "fixed"]
    tight, loose = t.loc["cv=0.01", "planted_row_z"], t.loc["cv=1", "planted_row_z"]
    assert abs(tight) > 10.0                    # copper's failure: a residual of tens of σ
    assert abs(loose) < abs(tight) / 20.0       # a hundredfold σ, a residual to match


def test_loosening_the_row_leaves_the_solution_where_it_was(contradiction):
    t = contradiction["table"]
    tight, loose = t.loc["cv=0.01"], t.loc["cv=1"]
    # σ moved the residual by two orders of magnitude and the planted edge by under 1 %
    # of its truth: the hard balance, not the anchor's σ, decides where the mass goes
    assert abs(tight["planted_a_star"] - loose["planted_a_star"]) \
        < 0.01 * tight["planted_truth"]
    assert loose["coverage"] == pytest.approx(tight["coverage"], abs=0.003)


def test_the_anchors_cost_coverage_at_every_row_cv(contradiction):
    t = contradiction["table"]
    assert (t.loc[["cv=0.01", "cv=1", "fixed"], "coverage"]
            < t.loc["base", "coverage"]).all()


def test_fixing_the_input_removes_the_residual(contradiction):
    t = contradiction["table"]
    assert abs(t.loc["fixed", "planted_row_z"]) < 1.0
    assert abs(t.loc["fixed", "planted_row_z"]) \
        < abs(t.loc["cv=0.01", "planted_row_z"]) / 20.0


def test_check_8_1_accepts_the_table_s_z_ratio_and_rejects_another(contradiction):
    t      = contradiction["table"]
    factor = t.loc["cv=0.01", "planted_row_z"] / t.loc["cv=1", "planted_row_z"]
    assert toy_ext.check_8_1(t, factor).startswith("✓ 8.1")
    with pytest.raises(AssertionError):
        toy_ext.check_8_1(t, factor * 1.1)


def test_an_experiment_never_becomes_the_latest_core_run(tree, contradiction):
    assert _core_latest(tree["root"]) == tree["base"]
    feeds = DataFeedRegistry(odir=str(toy.run_tree(tree["root"])["feed"]))
    feeds.load_index()
    assert feeds.latest_by_label("customs").run_id \
        == toy_ext._feed_run_ids(tree["root"], tree["base"]["imp"])["customs"]


# =============================================================================
# break_phantom
# =============================================================================

def test_stage_4_refuses_a_leg_larger_than_its_exporters_output(tree):
    out = toy_ext.break_phantom(tree["root"])
    row = out["table"].iloc[0]
    assert row["prior"] > row["output"]                 # the leg the exporter cannot make
    assert row["a_star"] < 0.10 * row["prior"]
    assert row["posterior_over_prior"] < 0.10
    # the refused leg's residual: measured z = -4.51 (seed 59); the bound is about half
    assert row["z"] < -2.5
    assert _core_latest(tree["root"]) == tree["base"]


# =============================================================================
# break_tolerance
# =============================================================================

def test_without_an_inequality_cone_both_gaps_return_the_same_solution(tree):
    out = toy_ext.break_tolerance(tree["root"], bands=False)
    assert out["summary"]["iterations"] == (0, 0)       # KKT solved directly
    assert out["summary"]["max_abs_diff"] == 0.0
    assert out["runs"]["const"] == tree["base"]["const"]


def test_with_the_non_negativity_cone_the_gap_decides_where_the_solver_stops(tree):
    out = toy_ext.break_tolerance(tree["root"], bands=True)
    s = out["summary"]
    assert s["iterations"][0] < s["iterations"][1]
    assert s["max_abs_diff"] > 1e-3                     # the loose solve is another point
    assert s["max_diff_over_sigma"] < 1                 # ... yet inside σ (measured 0.24)
    assert s["objective"][0] != s["objective"][1]
    assert len(out["table"]) == 6
    assert out["table"]["diff_over_sigma"].is_monotonic_decreasing
    assert _core_latest(tree["root"]) == tree["base"]


# =============================================================================
# The seams
# =============================================================================

def test_scrap_share_seed_writes_only_unset_scrap_legs_at_a_share_of_the_input(tree):
    acc = toy.load_acc(toy.run_tree(tree["root"]))
    em, content = acc.edge_meta, acc.valuation_roles.content
    f, sigma = np.full(acc.n_edges, np.nan), np.full(acc.n_edges, np.nan)
    conc = em[(em["from_sector"] == "Concentrate") & (em["to_sector"] == "Smelting")
              & (em["valuation"] == content) & (em["to_region"] == "Isla")
              & (em["time"] == "2003")]
    ids = conc["edge_id"].to_numpy()
    f[ids], sigma[ids] = [2.0, 1.0, 3.0], 0.1
    scrap = em[(em["from_sector"] == "Scrap") & (em["to_sector"] == "Smelting")
               & (em["valuation"] == content)]
    held = int(scrap["edge_id"].iloc[0])        # an edge already written must stay
    f[held] = 7.0
    seed = toy_ext.ScrapShareSeed(share=0.10, cv=0.5)
    seed.prepare(acc)
    log = run_imputer(seed, f, sigma, step=0, acc=acc)
    target = scrap[(scrap["to_region"] == "Isla") & (scrap["time"] == "2003")]["edge_id"]
    assert [e["edge_id"] for e in log] == list(target)
    assert f[target.iloc[0]] == pytest.approx(0.6)
    assert sigma[target.iloc[0]] == pytest.approx(0.3)
    assert f[held] == 7.0
    assert np.isnan(f[em["valuation"] == toy.GROSS]).all()


def test_net_stock_row_is_one_soft_row_per_region_and_year(tree):
    acc = toy.load_acc(toy.run_tree(tree["root"]))
    block = toy_ext.NetStockRow(sigma=0.1).build_static(acc)
    assert block.G_rows.shape == (len(toy.REGIONS) * len(toy.YEARS), acc.n_edges)
    assert (block.G_rows.sum(axis=1) == 1).all() and not block.row_meta["is_hard"].any()
    assert (block.sigma_c == 0.1).all() and (block.c == 0.0).all()
    hit = np.asarray(block.G_rows.indices)
    assert set(acc.edge_meta.loc[hit, "to_sector"]) == {"Stock"}
    assert set(acc.edge_meta.loc[hit, "valuation"]) == {toy.CONTENT}


def test_the_three_seams_run_through_stages_2_to_4(tree):
    out = toy_ext.run_seams(tree["root"])
    runs, root = out["runs"], tree["root"]
    tree_ = toy.run_tree(root)
    assert out["n_seeded"] == len(toy._BUREAU_GAPS)    # the scrap legs the Bureau skipped
    feeds = DataFeedRegistry(odir=str(tree_["feed"]))
    feeds.load_index()
    hit = np.flatnonzero(feeds.load_result(runs["feed"]["ministry"])["F_root"])
    em = toy.load_acc(tree_).edge_meta.loc[hit]
    # the Ministry covers exactly the two region-years the Bureau did not publish
    assert set(zip(em["from_region"], em["time"])) == set(toy._BUREAU_GAPS)
    assert len(hit) == len(toy._BUREAU_GAPS)
    meta = ConstraintRunRegistry(str(tree_["const"])).load(runs["const"]).row_meta
    assert int((meta["variant"] == "netstock").sum()) \
        == len(toy.REGIONS) * len(toy.YEARS)
    sidecar = OptimRunRegistry(str(toy.run_tree(root)["opt"])).load_sidecar(runs["opt"])
    assert sidecar["label"] == "toy_seams"
    assert _core_latest(root) == tree["base"]


def test_each_seam_example_is_under_25_lines_of_code():
    for seam in (toy_ext.ScrapShareSeed, toy_ext.NetStockRow, toy_ext.fourth_feed):
        node = ast.parse(textwrap.dedent(inspect.getsource(seam))).body[0]
        doc = node.body[0]
        body = node.body[1:] if isinstance(doc, ast.Expr) \
            and isinstance(doc.value, ast.Constant) else node.body
        lines = {n for b in body for n in range(b.lineno, b.end_lineno + 1)}
        assert len(lines) < 25, (seam.__name__, len(lines))
