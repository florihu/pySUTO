"""
test_sidecar_provenance.py
==========================
Every sidecar records what produced it and how long it took (#210): the code version, the
Python and numerical-stack versions, the stage's wall time; a feed run records its
configuration (``p`` and the spec's knobs) in ``stage_config``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from pysuto.feedpipe import FeedSpec, run_feed
from pysuto.registry import build_sidecar, stage_timer
from pysuto.tutorial import toy

ENV_KEYS = {"pysuto_version", "git_commit", "git_dirty", "python",
            "numpy", "scipy", "pandas", "pyarrow", "clarabel"}
STAGE_DIRS = {"feed": "feed/registry", "imp": "imp", "const": "const", "opt": "opt",
              "uncprop": "uncprop", "psut": "psut"}


def _sidecar(path) -> dict:
    return json.loads(Path(path, "metadata.json").read_text())


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """The six toy stages into one data directory; no PNG (Kaleido needs Chrome)."""
    root = tmp_path_factory.mktemp("provenance")
    ids = {"feed": toy.run_stage1(root)}
    ids["imp"]     = toy.run_stage2(root, ids["feed"])
    ids["const"]   = toy.run_stage3(root, ids["imp"])
    ids["opt"]     = toy.run_stage4(root, ids["const"])
    ids["uncprop"] = toy.run_stage5(root, ids["opt"])
    ids["psut"]    = toy.run_stage6(root, ids["opt"], png=False)
    return root, ids


def test_every_stage_sidecar_records_the_environment_and_the_wall_time(chain):
    root, ids = chain
    runs = [(s, rid) for s, rid in ids.items() if s != "feed"]
    runs += [("feed", rid) for rid in ids["feed"].values()]
    for stage, run_id in runs:
        sidecar = _sidecar(root / "proc" / STAGE_DIRS[stage] / run_id)
        env = sidecar["environment"]
        assert set(env) == ENV_KEYS, stage
        assert env["pysuto_version"] and env["python"]
        for pkg in ("numpy", "scipy", "pandas", "pyarrow", "clarabel"):
            assert env[pkg], (stage, pkg)
        assert set(sidecar["footprint"]) == {"wall_s", "peak_rss_mb"}
        assert sidecar["footprint"]["wall_s"] > 0, stage


def test_the_new_blocks_stay_outside_the_parents(chain):
    root, ids = chain
    sidecar = _sidecar(root / "proc" / "opt" / ids["opt"])
    assert set(sidecar["parents"]) == {"inputs", "upstream"}


def _survey_spec(feed, knobs):
    maps = {k.replace("_df", ""): v for k, v in feed.maps.items()}
    return FeedSpec(
        feed_id               = feed.feed_id,
        load_raw              = lambda path: pd.read_csv(path).assign(
                                    Year=lambda d: d["Year"].astype(str)),
        load_tables           = lambda path: maps,
        allow_unmatched_edges = False,
        knobs                 = knobs,
    )


def test_a_feed_run_sidecar_records_p_the_hooks_and_the_knobs(tmp_path):
    tree = toy._tree(tmp_path)
    entry = toy.write_inputs(tmp_path).acc.entry(toy.ACC_ID)
    toy.build_acc(tree["acc"], entry)
    feed = toy.make_feeds(toy.truth(), toy.SEED)["survey"]
    current = tmp_path / "input" / "feed" / feed.feed_id / "current"
    current.mkdir(parents=True)
    feed.raw.to_csv(current / "values.csv", index=False)
    knobs = {"CAP": 5.0, "EDGES": (("MF", "Concentrate"),)}

    run = run_feed(_survey_spec(feed, knobs), acc_id=toy.ACC_ID, data_dir=str(tmp_path),
                   diag=False)

    sidecar = _sidecar(run.run_dir)
    config = sidecar["stage_config"]
    assert config["p"] == 1.3
    assert config["knobs"] == {"CAP": 5.0, "EDGES": [["MF", "Concentrate"]]}
    assert config["allow_unmatched_edges"] is False
    assert config["hooks"]["load_raw"].endswith("<lambda>")
    assert set(config["hooks"]) == {"load_raw", "load_tables"}
    assert sidecar["footprint"]["wall_s"] > 0
    assert set(sidecar["environment"]) == ENV_KEYS


def test_a_sidecar_built_outside_a_stage_timer_has_no_wall_time():
    parents = {"inputs": {}, "upstream": {}}
    kw = dict(run_id="x", stage="imp", label="", created_at="", parents=parents)
    assert build_sidecar(**kw)["footprint"]["wall_s"] is None
    with stage_timer():
        time.sleep(0.01)
        assert build_sidecar(**kw)["footprint"]["wall_s"] >= 0.01
    assert build_sidecar(**kw)["footprint"]["wall_s"] is None


def test_the_environment_is_computed_once_per_process(monkeypatch):
    import pysuto.registry._sidecar as sidecar
    parents = {"inputs": {}, "upstream": {}}
    kw = dict(run_id="x", stage="imp", label="", created_at="", parents=parents)
    build_sidecar(**kw)
    monkeypatch.setattr(sidecar, "_git", lambda *a: pytest.fail("repeated subprocess"))
    build_sidecar(**kw)
