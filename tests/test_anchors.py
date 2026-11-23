"""
test_anchors.py
===============
``pysuto.constraints.anchors`` (#145): the Stage 3 anchor families take their labels as
parameters. The structure is the toy case's own acc — ``Content``/``Gross`` valuations,
``Smelting`` consuming ``Scrap`` — so no copper label is declared here.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pysuto.constraints as constraints_pkg
from pysuto.tutorial import toy
from pysuto.constraints import ConstraintEngine, anchors


@pytest.fixture(scope="module")
def toy_acc(tmp_path_factory):
    return toy.build_acc(tmp_path_factory.mktemp("anchors_acc"))


@pytest.fixture
def f_agg(toy_acc):
    return np.random.default_rng(0).uniform(1.0, 50.0, toy_acc.n_edges)


def _agg(f_agg, covered: bool):
    return SimpleNamespace(f_agg=f_agg, coverage=np.full(len(f_agg), covered))


# =============================================================================
# the cv rule and its resolver
# =============================================================================

def test_scale_total_cv_floors_feed_backed_totals_only():
    kw = dict(anchor_cv=0.01, fabricated_cv=0.02, feed_share_observed=0.5, abs_floor=0.3)
    assert anchors.scale_total_cv(1.0, 1.0, **kw) * 1.0 == pytest.approx(0.3)
    assert anchors.scale_total_cv(1.0, 1000.0, **kw) == pytest.approx(0.01)
    assert anchors.scale_total_cv(0.0, 1.0, **kw) == 0.02      # fabricated: no floor
    assert anchors.scale_total_cv(1.0, 0.0, **kw) is None


def test_resolver_classifies_a_group_by_its_feed_share(toy_acc, f_agg):
    kw = dict(anchor_cv=0.01, fabricated_cv=0.5, feed_share_observed=0.5, abs_floor=0.3)
    covered = anchors.scale_total_resolver(toy_acc, _agg(f_agg, True), **kw)
    fabricated = anchors.scale_total_resolver(toy_acc, _agg(f_agg, False), **kw)
    ids = [0, 1, 2]
    assert covered({}, ids) == pytest.approx(max(0.01, 0.3 / f_agg[ids].sum()))
    assert fabricated({}, ids) == 0.5


def test_resolver_applies_the_contradicted_node_cv_only_to_a_flagged_node(toy_acc,
                                                                         monkeypatch):
    f_agg = np.ones(toy_acc.n_edges)
    flagged = ("Concentrate", "Montara", "2001")
    monkeypatch.setattr(anchors, "contradicted_flow_nodes",
                        lambda acc, agg, sectors: ({flagged}, None))
    kw = dict(anchor_cv=0.01, fabricated_cv=0.01, feed_share_observed=0.5, abs_floor=0.3,
              unclosed_sectors=("Concentrate",))
    key = dict(zip(("sector_from", "region_from", "time"), flagged))
    other = dict(key, region_from="Portua")
    off = anchors.scale_total_resolver(toy_acc, _agg(f_agg, True), **kw)
    on = anchors.scale_total_resolver(toy_acc, _agg(f_agg, True), unclosed_anchor_cv=1.0,
                                      **kw)
    assert off(key, [0]) == pytest.approx(0.3)                 # detector only
    assert on(key, [0]) == 1.0
    assert on(other, [0]) == pytest.approx(0.3)


# =============================================================================
# supply / use scale totals
# =============================================================================

def test_scale_total_anchors_pin_each_group_to_its_prior(toy_acc, f_agg):
    builders = anchors.scale_total_anchors(toy_acc, f_agg, None, anchor_cv=0.01, p=1.3,
                                           supply_prefix="sup", use_prefix="use")
    assert len(builders) == 2
    em = toy_acc.edge_meta
    for builder, prefix, entity in zip(builders, ("sup", "use"), ("Process", "Flow")):
        block = builder.build_static(toy_acc)
        sectors = {s["name"] for s in toy.ACC_SPEC["sectors"] if s["entity"] == entity}
        assert set(block.row_meta["variant"]) == {prefix}
        assert set(block.row_meta["sector"]) <= sectors
        # the target of every row is the prior's group sum; sigma_c = anchor_cv·|c|
        G = block.G_rows.tocsr()
        c = np.asarray(G @ f_agg)
        np.testing.assert_allclose(block.c, c, rtol=1e-12)
        np.testing.assert_allclose(block.sigma_c, 0.01 * np.abs(c), rtol=1e-12)
        # year-locked and unit-coherent: one valuation and one time per row
        for i in range(G.shape[0]):
            cols = G.indices[G.indptr[i]:G.indptr[i + 1]]
            assert em["time"].iloc[cols].nunique() == 1
            assert em["valuation"].iloc[cols].nunique() == 1


def test_scale_total_anchors_use_the_resolvers_cv(toy_acc, f_agg):
    builders = anchors.scale_total_anchors(toy_acc, f_agg, lambda key, ids: 0.25,
                                           anchor_cv=0.01, p=1.3)
    block = builders[0].build_static(toy_acc)
    np.testing.assert_allclose(block.sigma_c, 0.25 * np.abs(block.c), rtol=1e-12)


# =============================================================================
# scrap-level anchors
# =============================================================================

def test_scrap_level_anchors_cover_covered_content_edges_of_the_source_only(toy_acc,
                                                                           f_agg):
    cov = np.random.default_rng(1).random(toy_acc.n_edges) < 0.6
    builders = anchors.scrap_level_anchors(
        toy_acc, f_agg, cov, None, source_sector="Scrap",
        consumers={"lvlsmelt": "Smelting"}, anchor_cv=0.01, p=1.3)
    assert len(builders) == 1
    block = builders[0].build_static(toy_acc)
    meta = block.row_meta
    assert set(meta["variant"]) == {"lvlsmelt"}
    assert set(meta["sector"]) == {"Smelting"}                  # the consuming node
    assert set(meta["valuation"]) == {toy.CONTENT}              # the content role only
    em = toy_acc.edge_meta
    scope = (em["from_sector"].eq("Scrap") & em["to_sector"].eq("Smelting")
             & em["valuation"].eq(toy.CONTENT)).to_numpy() & cov
    assert scope.any()
    assert scope[np.unique(block.G_rows.tocsr().indices)].all()
    expected = pd.Series(f_agg[scope]).groupby(
        [em["to_region"].to_numpy()[scope], em["time"].to_numpy()[scope]]).sum()
    np.testing.assert_allclose(np.sort(block.c), np.sort(expected.to_numpy()), rtol=1e-12)


def test_scrap_level_anchors_keep_each_route_apart(toy_acc, f_agg):
    cov = np.ones(toy_acc.n_edges, dtype=bool)
    builders = anchors.scrap_level_anchors(
        toy_acc, f_agg, cov, None, source_sector="Metal",
        consumers={"a": "Use", "b": "Stock"}, anchor_cv=0.01, p=1.3)
    assert [set(b.build_static(toy_acc).row_meta["sector"]) for b in builders] == \
           [{"Use"}, {"Stock"}]


# =============================================================================
# the no-data exemption
# =============================================================================

def test_nodata_anchor_exempt_is_off_without_data_aware_sectors(toy_acc):
    assert anchors.nodata_anchor_exempt(toy_acc, "unused.csv", param_prefix="x",
                                        exclude=("Mining",), data_aware=()) == (None, 0)


def test_nodata_anchor_exempt_spares_data_aware_sectors_in_listed_region_years(
        toy_acc, monkeypatch):
    rows = [SimpleNamespace(param_id="gradeore",
                            selectors={"a_region_from": "Montara", "a_time": "2001"}),
            SimpleNamespace(param_id="other",
                            selectors={"a_region_from": "Portua", "a_time": "2002"})]
    monkeypatch.setattr(anchors, "read_params", lambda path, accepts: rows)
    mask, n = anchors.nodata_anchor_exempt(toy_acc, "x.csv", param_prefix="grade",
                                           exclude=("Mining",), data_aware=("Mining",))
    assert n == 1
    em = toy_acc.edge_meta
    touches = em["from_sector"].eq("Mining") | em["to_sector"].eq("Mining")
    listed = em["from_region"].eq("Montara") & em["time"].eq("2001")
    np.testing.assert_array_equal(mask, (touches & listed).to_numpy())


# =============================================================================
# the library keeps no case's sector list
# =============================================================================

def test_engine_spares_no_sector_from_the_pin_unless_the_case_names_them(toy_acc,
                                                                         tmp_path):
    assert ConstraintEngine(toy_acc, storage_dir=str(tmp_path), sigma_a_cv_min=0.05, sigma_a_abs_floor=1e-6, sigma_c_cv_min=0.05, sigma_c_abs_floor=1e-3, nodata_anchor_cv=0.3)._pin_exclude_sectors == ()
    named = ConstraintEngine(toy_acc, storage_dir=str(tmp_path),
                             pin_exclude_sectors=("Ore",),
                             sigma_a_cv_min=0.05, sigma_a_abs_floor=1e-6, sigma_c_cv_min=0.05, sigma_c_abs_floor=1e-3, nodata_anchor_cv=0.3)
    assert named._pin_exclude_sectors == ("Ore",)


def test_the_toy_case_spares_no_sector_from_the_pin():
    assert toy.CASE.const.PIN_EXCLUDE_SECTORS == ()


def test_no_application_sector_label_is_a_string_literal_in_constraints():
    labels = ("CO", "Mining", "Ore", "WR", "Tailings", "Slag", "PSC", "Scrap", "Smelting",
              "CFR", "Matte", "Anode", "Cathode", "CC", "Mass")
    pattern = re.compile(r"""(["'])(%s)\1""" % "|".join(labels))
    hits = []
    for path in Path(constraints_pkg.__file__).parent.glob("*.py"):
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.name}:{no}: {line.strip()}")
    assert hits == []
