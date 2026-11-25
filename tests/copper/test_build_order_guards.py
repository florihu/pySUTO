"""The two build-order guards, through the entry points that call them (#221).

#53: ``load_trade_shares`` (the ICSG feed's reader) refuses weights whose ``parents.json``
names another BACI feed run or input version. #63: ``build_baci_reexport.main`` refuses an
ICSG input parsed against another BACI version than the one it nets, and a source version
whose region map differs from ``current/``'s. The check functions have their own tests
(``test_feedprep_trade_shares``, ``test_feedprep_reexport``); these cross the call sites,
on registries built in ``tmp_path``.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

import copper_case  # noqa: F401  (puts scripts/ on sys.path, #189)
import copper_feed
import build_baci_reexport as R
from pysuto.input_registry import InputRegistries

RUNS = "20261019_100000_aaaaaaaa"


def _write(root, kind, id_, name, text):
    f = root / kind / id_ / "current" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(text)
    return f


def _registry_with_trade_shares(tmp_path):
    """baci, icsg, conc, eff and one BACI feed run registered; trade_shares built on them
    as ``build_trade_shares`` records it, then registered."""
    root, runs = tmp_path / "input", tmp_path / "runs"
    for kind, id_, name in (("feed", "baci", "values.csv"), ("feed", "icsg", "values.csv"),
                            ("params", "conc", "params.csv"), ("params", "eff", "params.csv")):
        _write(root, kind, id_, name, "x\n1\n")
    (runs / f"feed_baci_{RUNS}").mkdir(parents=True)
    inputs = InputRegistries.initialize(root=root)
    _write(root, "params", "trade_shares", "params.csv", "weight\n1.0\n")
    _write(root, "params", "trade_shares", "parents.json",
           json.dumps(copper_feed.trade_share_parents(inputs, str(runs))))
    return root, runs, InputRegistries.initialize(root=root)


def test_load_trade_shares_returns_weights_built_on_the_run_in_force(tmp_path):
    _, runs, inputs = _registry_with_trade_shares(tmp_path)
    df, entry = copper_feed.load_trade_shares(inputs, str(runs))
    assert list(df["weight"]) == [1.0] and entry["version"] == 1


def test_load_trade_shares_refuses_weights_older_than_the_baci_feed_run(tmp_path):
    """baci_feed re-run, build_trade_shares skipped: the stale weights cost 6.2 pp of z²."""
    _, runs, inputs = _registry_with_trade_shares(tmp_path)
    (runs / "feed_baci_20261020_100000_bbbbbbbb").mkdir()
    with pytest.raises(ValueError, match="feed_baci_20261020_100000_bbbbbbbb"):
        copper_feed.load_trade_shares(inputs, str(runs))


def test_load_trade_shares_refuses_weights_built_on_another_baci_input_version(tmp_path):
    root, runs, _ = _registry_with_trade_shares(tmp_path)
    _write(root, "feed", "baci", "values.csv", "x\n2\n")
    inputs = InputRegistries.initialize(root=root)                   # baci v2
    with pytest.raises(ValueError, match="baci"):
        copper_feed.load_trade_shares(inputs, str(runs))


def test_load_trade_shares_without_the_artifact_is_the_uniform_fan(tmp_path):
    root = tmp_path / "input"
    _write(root, "feed", "baci", "values.csv", "x\n1\n")
    assert copper_feed.load_trade_shares(InputRegistries.initialize(root=root),
                                         str(tmp_path / "runs")) == (None, None)


# ---------------------------------------------------------------------------
# build_baci_reexport.main (#63, #77)
# ---------------------------------------------------------------------------

def _region_map_xlsx(path, root_of_490):
    with pd.ExcelWriter(path) as w:
        pd.DataFrame({"src_region": [276, 490], "root_region": ["Germany", root_of_490]}
                     ).to_excel(w, sheet_name="region_map", index=False)


def _netting_registry(tmp_path, monkeypatch, icsg_baci_version):
    root = tmp_path / "input"
    _write(root, "feed", "baci", "values.csv", "x\n1\n")
    _write(root, "feed", "icsg", "values.csv", "x\n1\n")
    _write(root, "feed", "icsg", "parents.json",
           json.dumps({"inputs": {"feed": {"baci": icsg_baci_version}}}))
    _region_map_xlsx(root / "feed" / "baci" / "current" / "conc.xlsx", "Taiwan")
    InputRegistries.initialize(root=root)
    monkeypatch.setattr(R, "_INPUT_ROOT", str(root))
    monkeypatch.setattr(R, "OUT_DIR", str(root / "feed" / "baci" / "current"))
    return root


def test_the_netting_refuses_an_icsg_input_parsed_against_another_baci_version(
        tmp_path, monkeypatch):
    _netting_registry(tmp_path, monkeypatch, icsg_baci_version=2)
    with pytest.raises(ValueError, match="parsed against baci v2.*nets baci v1"):
        R.main(1)


def test_the_netting_refuses_an_icsg_input_that_records_no_baci_version(
        tmp_path, monkeypatch):
    root = _netting_registry(tmp_path, monkeypatch, icsg_baci_version=1)
    (root / "feed" / "icsg" / "v1" / "parents.json").unlink()
    with pytest.raises(ValueError, match="records no baci version"):
        R.main(1)


def test_the_netting_refuses_a_source_whose_region_map_differs_from_current(
        tmp_path, monkeypatch):
    root = _netting_registry(tmp_path, monkeypatch, icsg_baci_version=1)
    _region_map_xlsx(root / "feed" / "baci" / "current" / "conc.xlsx", "China")
    with pytest.raises(ValueError, match="map differently"):
        R.main(1)
