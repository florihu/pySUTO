"""
test_feed_spec.py
=================
One feed pipeline (#143): a :class:`pysuto.feedpipe.FeedSpec` run by ``run_feed`` over a
data dir builds, propagates and registers a feed run, and the copper feed scripts attach
no root logging handler when imported.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysuto.tutorial import toy
import copper_case  # noqa: F401  (puts scripts/ on sys.path, #189)
from copper_feed import BACI_FEED, CRUDE_ORE_FEED, ICSG_FEED, icsg_sector_key
from pysuto.feedpipe import FeedSpec, run_feed

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _survey_spec(feed):
    """The toy's survey feed (ore extraction, 1:1) as a FeedSpec over its own maps."""
    maps = {k.replace("_df", ""): v for k, v in feed.maps.items()}
    return FeedSpec(
        feed_id               = feed.feed_id,
        load_raw              = lambda path: pd.read_csv(path).assign(
                                    Year=lambda d: d["Year"].astype(str)),
        load_tables           = lambda path: maps,
        allow_unmatched_edges = False,
    )


@pytest.fixture
def toy_data_dir(tmp_path):
    """A data dir holding the toy's accounting structure and its survey feed's values."""
    tree = toy._tree(tmp_path)
    entry = toy.write_inputs(tmp_path).acc.entry(toy.ACC_ID)
    acc = toy.build_acc(tree["acc"], entry)
    feed = toy.make_feeds(toy.truth(), toy.SEED)["survey"]
    current = tmp_path / "input" / "feed" / feed.feed_id / "current"
    current.mkdir(parents=True)
    feed.raw.to_csv(current / "values.csv", index=False)
    return tmp_path, acc, feed


def test_a_toy_feed_spec_produces_a_registered_feed_run(toy_data_dir):
    data_dir, acc, feed = toy_data_dir
    entry = run_feed(_survey_spec(feed), acc_id=toy.ACC_ID, data_dir=str(data_dir),
                     diag=False)

    assert entry.run_id.startswith("feed_survey_")
    assert str(data_dir / "proc" / "feed" / "registry") in str(entry.run_dir)
    sidecar = json.loads((Path(entry.run_dir) / "metadata.json").read_text())
    assert sidecar["parents"]["inputs"]["acc"]["id"] == toy.ACC_ID
    assert sidecar["parents"]["inputs"]["feed"]["id"] == "survey"
    assert sidecar["parents"]["upstream"] == {}

    # the run's F_root is the toy's own steps B-D over the same raw rows and maps
    _, _, _, expected, _ = toy._feed_steps(feed, acc)
    got = np.load(os.path.join(entry.run_dir, "result.npz"))["F_root"]
    # the CSV round trip reads floats to the last digit or two
    np.testing.assert_allclose(got, expected.F_root, rtol=1e-12, atol=0)


def test_run_feed_writes_nothing_outside_its_data_dir(toy_data_dir, tmp_path_factory):
    data_dir, _, feed = toy_data_dir
    cwd = tmp_path_factory.mktemp("cwd")
    old = os.getcwd()
    os.chdir(cwd)
    try:
        run_feed(_survey_spec(feed), acc_id=toy.ACC_ID, data_dir=str(data_dir),
                 diag=False)
    finally:
        os.chdir(old)
    assert not (cwd / "data").exists()


def test_a_run_log_handler_is_removed_when_the_run_ends(toy_data_dir, tmp_path):
    data_dir, _, feed = toy_data_dir
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    log = tmp_path / "logs" / "survey.log"
    run_feed(_survey_spec(feed), acc_id=toy.ACC_ID, data_dir=str(data_dir), diag=False,
             run_log=str(log))
    assert log.exists() and "Registering feed" in log.read_text()
    assert root.handlers == handlers and root.level == level


def test_importing_the_feed_scripts_attaches_no_root_handler(monkeypatch):
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    monkeypatch.syspath_prepend(SCRIPTS)
    for name in ("icsg_feed", "baci_feed", "crude_ore_feed"):
        monkeypatch.delitem(sys.modules, name, raising=False)   # restored at teardown
        __import__(name)
    assert root.handlers == before and root.level == level


def test_the_copper_feed_specs_name_their_feeds():
    assert [s.feed_id for s in (ICSG_FEED, BACI_FEED, CRUDE_ORE_FEED)] == [
        "icsg", "baci", "crude_ore"]
    assert BACI_FEED.edge_scope and not ICSG_FEED.edge_scope


def test_icsg_sector_key_disambiguates_use_rows_only():
    assert icsg_sector_key("Smelting", "Concentrate", "Supply") == "Smelting"
    assert icsg_sector_key("Smelting", "Srap", "Use") == "Smelting_Scrap"
    assert icsg_sector_key("Smelting", "Concentrate", "Use") == "Smelting_Concentrate"
    assert icsg_sector_key("Refining", "Scrap", "Use") == "Refining_Scrap"
    assert icsg_sector_key("Refining", "Anode", "Use") == "Refining_Anode"
    assert icsg_sector_key("Mining", "Ore", "Use") == "Mining"


# ---------------------------------------------------------------------------
# Edge scope (ADR 0014): a scoped feed never builds unscoped (#280, U1a-G4-4)
# ---------------------------------------------------------------------------

def _write_sidecar(data_dir, feed, mask=None):
    """Register a ``meta.xlsx`` for the toy survey feed; ``mask`` is its Edge_scope_mask."""
    rows = [("Feed_name", feed.feed_id), ("Reliability", 2), ("Completeness", 1),
            ("Temporal_correlation", 1), ("Geographical_correlation", 1),
            ("Other_correlation", 1)]
    if mask is not None:
        rows.append(("Edge_scope_mask", mask))
    path = data_dir / "input" / "feed" / feed.feed_id / "current" / "meta.xlsx"
    pd.DataFrame(rows, columns=["Meta_name", "Value"]).to_excel(path, index=False)


def _scoped(spec):
    return dataclasses.replace(spec, edge_scope=True)


def test_a_scoped_feed_without_a_sidecar_raises(toy_data_dir):
    data_dir, _, feed = toy_data_dir
    with pytest.raises(ValueError, match="names no Edge_scope_mask"):
        run_feed(_scoped(_survey_spec(feed)), acc_id=toy.ACC_ID, data_dir=str(data_dir),
                 diag=False)


def test_a_scoped_feed_whose_sidecar_names_no_mask_raises(toy_data_dir):
    data_dir, _, feed = toy_data_dir
    _write_sidecar(data_dir, feed)
    with pytest.raises(ValueError, match="names no Edge_scope_mask"):
        run_feed(_scoped(_survey_spec(feed)), acc_id=toy.ACC_ID, data_dir=str(data_dir),
                 diag=False)


def test_a_scoped_feed_whose_mask_the_structure_lacks_raises(toy_data_dir):
    data_dir, _, feed = toy_data_dir
    _write_sidecar(data_dir, feed, mask="no_such_mask")
    with pytest.raises(ValueError, match="not present in acc_struct.masks"):
        run_feed(_scoped(_survey_spec(feed)), acc_id=toy.ACC_ID, data_dir=str(data_dir),
                 diag=False)


def test_a_scope_that_keeps_every_row_leaves_the_feed_unchanged(toy_data_dir):
    data_dir, _, feed = toy_data_dir
    unscoped = run_feed(_survey_spec(feed), acc_id=toy.ACC_ID, data_dir=str(data_dir),
                        diag=False)
    _write_sidecar(data_dir, feed, mask="massbal_scope")
    scoped = run_feed(_scoped(_survey_spec(feed)), acc_id=toy.ACC_ID,
                      data_dir=str(data_dir), diag=False)
    got = np.load(os.path.join(scoped.run_dir, "result.npz"))["F_root"]
    want = np.load(os.path.join(unscoped.run_dir, "result.npz"))["F_root"]
    np.testing.assert_array_equal(got, want)
