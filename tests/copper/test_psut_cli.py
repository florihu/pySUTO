"""test_psut_cli.py
================
``scripts/psut.py <opt_run_id>``: one parquet per period, the provenance and the Sankey
HTML. Split from ``tests/test_psut.py`` (#304); the fixtures come from there.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # test_psut
from test_psut import (  # noqa: E402,F401
    F, PERIODS, _persist_opt_chain, patched_acc_load, toy_acc)


def test_cli_writes_one_parquet_per_period(tmp_path, monkeypatch, toy_acc, F, patched_acc_load):
    from pysuto.registry import PsutRunRegistry
    opt_id, _, _ = _persist_opt_chain(tmp_path, toy_acc, F)
    monkeypatch.chdir(tmp_path)                       # logs/ lands in the tmp dir

    mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "psut.py")
    spec = importlib.util.spec_from_file_location("psut_cli", mod_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    run_id = cli.main(opt_id, label="cli", primary="CC", data_dir=str(tmp_path))

    run_dir = tmp_path / "proc" / "psut" / run_id
    for t in PERIODS:
        assert (run_dir / "timesteps" / f"t_{t}.parquet").exists()
    sidecar = PsutRunRegistry(str(tmp_path / "proc" / "psut")).load_sidecar(run_id)
    assert sidecar["parents"]["upstream"] == {"opt_run_id": opt_id, "const_run_id": "const_toy"}
    assert sidecar["parents"]["inputs"]["acc"]["id"] == "toy"
    assert sidecar["label"] == "cli"
    assert sidecar["stage_config"]["primary"] == "CC"


def test_cli_records_the_acc_version_the_const_run_names(tmp_path, monkeypatch, toy_acc,
                                                         F, patched_acc_load):
    """#53 item 4: the psut sidecar carries the const run's full acc ref, not the id."""
    from pysuto.registry import PsutRunRegistry
    opt_id, _, _ = _persist_opt_chain(tmp_path, toy_acc, F)
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "psut_cli_acc", os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "psut.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    run_id = cli.main(opt_id, data_dir=str(tmp_path))
    sidecar = PsutRunRegistry(str(tmp_path / "proc" / "psut")).load_sidecar(run_id)
    assert sidecar["parents"]["inputs"]["acc"] == {"id": "toy", "version": 1}


def test_cli_writes_sankey_html(tmp_path, monkeypatch, toy_acc, F, patched_acc_load):
    pytest.importorskip("plotly")
    opt_id, _, _ = _persist_opt_chain(tmp_path, toy_acc, F)
    monkeypatch.chdir(tmp_path)

    mod_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "psut.py")
    spec = importlib.util.spec_from_file_location("psut_cli_sankey", mod_path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    figs = tmp_path / "figs"
    cli.main(opt_id, data_dir=str(tmp_path), sankey=["2020", "2021"], figs_dir=str(figs), png=False)
    for t in PERIODS:
        for name in ("supply_use", "process_network"):
            html = figs / f"sankey_{name}_World_{t}_CC.html"
            assert html.exists() and "sankey" in html.read_text()[:200000]
            assert " kt CC" in html.read_text()          # the content valuation's unit (#286)
    assert not list(figs.glob("*.png"))                                 # png=False
    cli.main(opt_id, data_dir=str(tmp_path), sankey=[], figs_dir=str(figs), png=False)   # flag alone → latest
    assert (figs / "sankey_supply_use_World_2021_CC.html").exists()
