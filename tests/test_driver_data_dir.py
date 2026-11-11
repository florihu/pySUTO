"""test_driver_data_dir.py — the Stage 2–5 drivers run on a data tree outside the
checkout (#58), and check the built acc they load against their acc ref (#53).

A *case* here is a toy data tree in ``tmp_path/case/data`` (an acc input registry, the
structure built from it, a constraint run on it) and a config naming its acc. Every
driver runs from an empty working directory, which must hold no ``data/`` afterwards:
the registries create their directory when opened, so a path left relative to the
working directory would show up there.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend before any pyplot import

import numpy as np
import pandas as pd
import pytest

from pysuto.registry import stage_lock

REPO = Path(__file__).resolve().parents[1]
STAGE_DRIVERS = ["collect_aggregate_impute.py", "constraints.py", "run_solver.py",
                 "uncertainty_propagation.py"]


def _driver(script: str):
    spec = importlib.util.spec_from_file_location(f"_data_dir_{script[:-3]}",
                                                  REPO / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def _edit_acc_input(case: dict, rules: bytes) -> dict:
    """Edit the case's acc input and register it; return the new entry."""
    from pysuto.input_registry import InputRegistry
    (case["data"] / "input" / "acc" / "toy" / "current" / "rules.ods").write_bytes(rules)
    return InputRegistry("acc", root=case["data"] / "input").initialize().entry("toy")


def _make_case(root: Path) -> dict:
    """acc input toy v1 → v2, the structure built from v2 (with its source record),
    a two-period constraint run on it (mass balances) and a config naming acc toy."""
    from pysuto.acc import AccountingStructure, write_source_record
    from pysuto.constraints import ConstraintEngine, MassBalanceBuilder

    data = root / "case" / "data"
    current = data / "input" / "acc" / "toy" / "current"
    current.mkdir(parents=True)
    for name in ("root.ods", "bounds.ods", "rules.ods"):
        (current / name).write_bytes(name.encode())
    case = {"data": data}
    _edit_acc_input(case, b"rules v1")
    entry = _edit_acc_input(case, b"rules v2")

    acc_dir = str(data / "proc" / "acc")
    nodes = pd.DataFrame({"sector": ["Mining", "Ore", "Smelting"],
                          "entity": ["Process", "Flow", "Process"],
                          "region": ["A", "A", "A"]})
    acc = AccountingStructure(
        id="toy", storage_dir=acc_dir,
        root={"nodes": nodes, "time_range": ["2020", "2021"], "valuation": ["Mass"]},
        rules=[lambda ni, nj: (ni["entity"] == "Process") != (nj["entity"] == "Process")])
    acc.save()
    write_source_record(acc_dir, entry)

    class _Agg:
        f_agg     = np.arange(1.0, acc.n_edges + 1)
        sigma_agg = np.ones(acc.n_edges)

    engine = ConstraintEngine(AccountingStructure.load("toy", storage_dir=acc_dir),
                              storage_dir=str(data / "proc" / "const"))
    engine.register(MassBalanceBuilder())
    cr = engine.run(_Agg(), timesteps=["2020", "2021"], persist=True, label="case",
                    parents={"inputs": {"acc": entry}, "upstream": {}})

    config = root / "case" / "config.yaml"
    config.write_text("pipeline:\n  acc_id: toy\n")
    cwd = root / "cwd"
    cwd.mkdir()
    case.update(acc_dir=acc_dir, entry=entry, const_run_id=cr.run_id,
                config=str(config), cwd=cwd)
    return case


@pytest.fixture
def case(tmp_path, monkeypatch):
    c = _make_case(tmp_path)
    monkeypatch.chdir(c["cwd"])
    yield c
    assert not (c["cwd"] / "data").exists(), "a driver touched the CWD's data/"


# ---------------------------------------------------------------------------
# Import, lock, paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", STAGE_DRIVERS)
def test_importing_a_driver_reads_no_config_and_touches_no_data(script, tmp_path,
                                                                monkeypatch):
    """The working directory has no config/config.yaml: an import-time load_config
    would raise FileNotFoundError."""
    monkeypatch.chdir(tmp_path)
    _driver(script)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("script,args", [
    ("collect_aggregate_impute.py", ["--feed-id", "feed_x"]),
    ("constraints.py",              ["--imp-run-id", "imp_x", "lbl"]),
    ("run_solver.py",               ["--const-run-id", "const_x"]),
    ("uncertainty_propagation.py",  ["opt_x"]),
])
def test_a_driver_takes_the_stage_lock_of_its_data_dir(script, args, tmp_path):
    data, cwd = tmp_path / "case" / "data", tmp_path / "cwd"
    cwd.mkdir()
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    with stage_lock("held_by_test", root=str(data / "proc")):
        r = subprocess.run([sys.executable, str(REPO / "scripts" / script), *args,
                            "--data-dir", str(data)],
                           cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "LockHeld" in r.stderr and "held_by_test" in r.stderr, r.stderr[-800:]
    assert not (cwd / "data").exists()


def test_stages_4_and_5_read_and_write_only_the_given_data_dir(case):
    """The cold chain of the case: an opt run, then its uncprop run, both under the
    case's proc/; the opt sidecar of a cold run records no inputs (unchanged)."""
    from pysuto.registry import OptimRunRegistry, UncPropRunRegistry
    proc = case["data"] / "proc"
    opt_id = _driver("run_solver.py").main(case["const_run_id"], "case", "cold",
                                           data_dir=str(case["data"]))
    sidecar = OptimRunRegistry(str(proc / "opt")).load_sidecar(opt_id)
    assert sidecar["parents"] == {"inputs": {},
                                  "upstream": {"const_run_id": case["const_run_id"]}}
    assert (proc / "opt" / opt_id / "diag").is_dir()

    _driver("uncertainty_propagation.py").main(opt_id, "case", data_dir=str(case["data"]))
    unc = UncPropRunRegistry(str(proc / "uncprop"))
    assert len(unc.run_ids()) == 1
    assert (proc / "uncprop" / unc.run_ids()[0] / "diag").is_dir()


def test_a_warm_stage_4_records_the_acc_it_loaded_from_the_data_dir(case, monkeypatch):
    """#53 item 5: the warm prior's acc comes from <data-dir>/proc/acc, is checked
    against the const run's acc ref, and that ref lands in the opt sidecar."""
    from pysuto.registry import OptimRunRegistry
    rs = _driver("run_solver.py")
    monkeypatch.setattr(rs, "N_WORKERS", 1)                 # warm is a chain
    opt_id = rs.main(case["const_run_id"], "case", "warm", data_dir=str(case["data"]))
    sidecar = OptimRunRegistry(str(case["data"] / "proc" / "opt")).load_sidecar(opt_id)
    assert sidecar["parents"]["inputs"] == {"acc": case["entry"]}
    assert sidecar["stage_config"]["prior_mode"] == "warm"


def test_a_warm_stage_4_refuses_an_acc_built_from_another_version(case, monkeypatch):
    from pysuto.acc import write_source_record
    write_source_record(case["acc_dir"], {**case["entry"], "version": 1})
    rs = _driver("run_solver.py")
    monkeypatch.setattr(rs, "N_WORKERS", 1)
    with pytest.raises(ValueError, match=r"toy v1.*toy v2"):
        rs.main(case["const_run_id"], "case", "warm", data_dir=str(case["data"]))


# ---------------------------------------------------------------------------
# Stages 2 and 3 check the built acc against their own acc input entry (#53)
# ---------------------------------------------------------------------------

def test_stage_3_refuses_an_acc_built_from_an_older_input_version(case):
    """The input moved on to v3; the structure is still the one built from v2."""
    _edit_acc_input(case, b"rules v3")
    with pytest.raises(ValueError, match=r"toy v2.*toy v3"):
        _driver("constraints.py").main("imp_x", "case", data_dir=str(case["data"]),
                                       config=case["config"])


def test_stage_3_reads_the_acc_id_from_the_given_config(case):
    """The built acc matches its entry, so Stage 3 gets as far as the imp run."""
    with pytest.raises(KeyError, match="imp_x"):
        _driver("constraints.py").main("imp_x", "case", data_dir=str(case["data"]),
                                       config=case["config"])


def test_stage_2_refuses_an_acc_built_from_an_older_input_version(case):
    _edit_acc_input(case, b"rules v3")
    with pytest.raises(ValueError, match=r"toy v2.*toy v3"):
        _driver("collect_aggregate_impute.py").main(
            ["feed_x"], "case", data_dir=str(case["data"]), config=case["config"])


def test_stage_2_pins_the_acc_version_an_earlier_imp_run_recorded(case):
    """Replaying an imp run built on toy v2 pins v2 although v3 is the latest, so the
    v2 structure passes and Stage 2 gets as far as the feed lookup."""
    from pysuto.registry import ImputationRunRegistry
    from pysuto.registry._sidecar import build_sidecar
    _edit_acc_input(case, b"rules v3")
    reg = ImputationRunRegistry(storage_dir=str(case["data"] / "proc" / "imp"))
    reg._write_sidecar(reg._make_run_dir("imp_ref"), build_sidecar(
        run_id="imp_ref", stage="imp", label="", created_at="2026-11-11T00:00:00Z",
        parents={"inputs": {"acc": case["entry"], "params": []}, "upstream": {}}))
    reg._add_entry(run_id="imp_ref", label="", created_at="2026-11-11T00:00:00Z")
    reg.save()
    with pytest.raises(KeyError, match="feed_x"):
        _driver("collect_aggregate_impute.py").main(
            ["feed_x"], "case", data_dir=str(case["data"]), config=case["config"],
            pin_inputs_from="imp_ref")


# ---------------------------------------------------------------------------
# ConstDiagnostic's default figure directory is the run's own (#58)
# ---------------------------------------------------------------------------

def test_const_diag_from_a_run_dir_draws_into_that_run_dir(case):
    from pysuto.diag import ConstDiagnostic
    run_dir = case["data"] / "proc" / "const" / case["const_run_id"]
    paths = ConstDiagnostic.from_saved(str(run_dir)).plot()
    assert paths and all(Path(p).parent == run_dir / "diag" for p in paths)


def test_const_diag_without_a_run_dir_needs_an_odir(case):
    from pysuto.diag import ConstDiagnostic
    from pysuto.registry import ConstraintRunRegistry
    cr = ConstraintRunRegistry(str(case["data"] / "proc" / "const")).load(
        case["const_run_id"])
    with pytest.raises(ValueError, match="odir"):
        ConstDiagnostic(cr).plot()
