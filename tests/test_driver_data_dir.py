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


@pytest.fixture
def driver(monkeypatch):
    """Load a driver script by path as a module, ``scripts/`` on the path for the test."""
    monkeypatch.syspath_prepend(str(REPO / "scripts"))

    def load(script: str):
        spec = importlib.util.spec_from_file_location(f"_data_dir_{script[:-3]}",
                                                      REPO / "scripts" / script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    return load


def _edit_acc_input(case: dict, rules: bytes) -> dict:
    """Edit the case's acc input and register it; return the new entry."""
    from pysuto.input_registry import InputRegistry
    (case["data"] / "input" / "acc" / "toy" / "current" / "rules.ods").write_bytes(rules)
    return InputRegistry("acc", root=case["data"] / "input").initialize().entry("toy")


def _make_case(root: Path) -> dict:
    """acc input toy v1 → v2, the structure built from v2 (with its build record),
    a two-period constraint run on it (mass balances) and a config naming acc toy."""
    from pysuto.acc import AccountingStructure, write_build_record
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
    write_build_record(acc_dir, entry)

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
                                                                monkeypatch, driver):
    """The working directory has no config/config.yaml: an import-time load_config
    would raise FileNotFoundError."""
    monkeypatch.chdir(tmp_path)
    driver(script)
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


def test_stages_4_and_5_read_and_write_only_the_given_data_dir(case, driver):
    """The cold chain of the case: an opt run, then its uncprop run, both under the
    case's proc/; the opt sidecar of a cold run records no inputs (unchanged)."""
    from pysuto.registry import OptimRunRegistry, UncPropRunRegistry
    proc = case["data"] / "proc"
    opt_id = driver("run_solver.py").main(case["const_run_id"], "case", "cold",
                                          data_dir=str(case["data"]))
    sidecar = OptimRunRegistry(str(proc / "opt")).load_sidecar(opt_id)
    assert sidecar["parents"] == {"inputs": {},
                                  "upstream": {"const_run_id": case["const_run_id"]}}
    assert (proc / "opt" / opt_id / "diag").is_dir()

    driver("uncertainty_propagation.py").main(opt_id, "case", data_dir=str(case["data"]))
    unc = UncPropRunRegistry(str(proc / "uncprop"))
    assert len(unc.run_ids()) == 1
    assert (proc / "uncprop" / unc.run_ids()[0] / "diag").is_dir()


def test_a_warm_stage_4_records_the_acc_it_loaded_from_the_data_dir(case, monkeypatch,
                                                                    driver):
    """#53 item 5: the warm prior's acc comes from <data-dir>/proc/acc, is checked
    against the const run's acc ref, and that ref lands in the opt sidecar."""
    from pysuto.registry import OptimRunRegistry
    rs = driver("run_solver.py")
    monkeypatch.setattr(rs, "N_WORKERS", 1)                 # warm is a chain
    opt_id = rs.main(case["const_run_id"], "case", "warm", data_dir=str(case["data"]))
    sidecar = OptimRunRegistry(str(case["data"] / "proc" / "opt")).load_sidecar(opt_id)
    assert sidecar["parents"]["inputs"] == {"acc": case["entry"]}
    assert sidecar["stage_config"]["prior_mode"] == "warm"


def test_a_warm_stage_4_refuses_an_acc_built_from_another_version(case, monkeypatch,
                                                                   driver):
    from pysuto.acc import write_build_record
    write_build_record(case["acc_dir"], {**case["entry"], "version": 1})
    rs = driver("run_solver.py")
    monkeypatch.setattr(rs, "N_WORKERS", 1)
    with pytest.raises(ValueError, match=r"toy v1.*toy v2"):
        rs.main(case["const_run_id"], "case", "warm", data_dir=str(case["data"]))


def test_a_cold_stage_4_checks_the_acc_its_diagnostics_load_before_solving(case,
                                                                           driver):
    """#53: the cold solve reads no acc, but the diagnostics do; the build record is
    checked before the solve, so a refused run persists nothing."""
    from pysuto.acc import write_build_record
    write_build_record(case["acc_dir"], {**case["entry"], "version": 1})
    with pytest.raises(ValueError, match=r"toy v1.*toy v2"):
        driver("run_solver.py").main(case["const_run_id"], "case", "cold",
                                     data_dir=str(case["data"]))
    assert not (case["data"] / "proc" / "opt").exists()


# ---------------------------------------------------------------------------
# Stages 2 and 3 check the built acc against their own acc input entry (#53)
# ---------------------------------------------------------------------------

def test_stage_3_refuses_an_acc_built_from_an_older_input_version(case, driver,
                                                                  monkeypatch):
    """The input moved on to v3; the structure is still the one built from v2. The
    record is checked before the structure is loaded."""
    from pysuto.acc import AccountingStructure
    _edit_acc_input(case, b"rules v3")
    monkeypatch.setattr(AccountingStructure, "load", classmethod(_no_load))
    with pytest.raises(ValueError, match=r"toy v2.*toy v3"):
        driver("constraints.py").main("imp_x", "case", data_dir=str(case["data"]),
                                       config=case["config"])


def test_stage_3_reads_the_acc_id_from_the_given_config(case, driver):
    """The built acc matches its entry, so Stage 3 gets as far as the imp run."""
    with pytest.raises(KeyError, match="imp_x"):
        driver("constraints.py").main("imp_x", "case", data_dir=str(case["data"]),
                                       config=case["config"])


def test_stage_2_refuses_an_acc_built_from_an_older_input_version(case, driver,
                                                                  monkeypatch):
    from pysuto.acc import AccountingStructure
    _edit_acc_input(case, b"rules v3")
    monkeypatch.setattr(AccountingStructure, "load", classmethod(_no_load))
    with pytest.raises(ValueError, match=r"toy v2.*toy v3"):
        driver("collect_aggregate_impute.py").main(
            ["feed_x"], "case", data_dir=str(case["data"]), config=case["config"])


def _register_imp_ref(case) -> str:
    """An imp run of the case built on toy v2 (its acc entry), recording no params."""
    from pysuto.registry import ImputationRunRegistry
    from pysuto.registry._sidecar import build_sidecar
    reg = ImputationRunRegistry(storage_dir=str(case["data"] / "proc" / "imp"))
    reg._write_sidecar(reg._make_run_dir("imp_ref"), build_sidecar(
        run_id="imp_ref", stage="imp", label="", created_at="2026-11-11T00:00:00Z",
        parents={"inputs": {"acc": case["entry"], "params": []}, "upstream": {}}))
    reg._add_entry(run_id="imp_ref", label="", created_at="2026-11-11T00:00:00Z")
    reg.save()
    return "imp_ref"


def test_stage_2_pins_the_acc_version_an_earlier_imp_run_recorded(case, driver):
    """Replaying an imp run built on toy v2 pins v2 although v3 is the latest, so the
    v2 structure passes and Stage 2 gets as far as the feed lookup."""
    _edit_acc_input(case, b"rules v3")
    imp_ref = _register_imp_ref(case)
    with pytest.raises(KeyError, match="feed_x"):
        driver("collect_aggregate_impute.py").main(
            ["feed_x"], "case", data_dir=str(case["data"]), config=case["config"],
            pin_inputs_from=imp_ref)


def test_a_pinned_stage_2_refuses_a_structure_without_a_build_record(case, driver):
    """Unpinned, a structure built before #53 only warns; a pinned acc version cannot
    be verified against it."""
    from pysuto.acc import BUILD_RECORD
    os.remove(os.path.join(case["acc_dir"], "toy", BUILD_RECORD))
    imp_ref = _register_imp_ref(case)
    with pytest.raises(ValueError, match="no build record"):
        driver("collect_aggregate_impute.py").main(
            ["feed_x"], "case", data_dir=str(case["data"]), config=case["config"],
            pin_inputs_from=imp_ref)


# ---------------------------------------------------------------------------
# Stages 2 and 3 build from the case spec and record it in the sidecar (#120)
# ---------------------------------------------------------------------------

# Every stage_config key the Stage 2 and 3 sidecars recorded before #120 (Stage 3's
# registry adds "p"); the serialised case spec is recorded beside them under "case".
IMP_KEYS = {
    "p", "q", "alpha_min", "cascade_max_iter", "cascade_order", "flat_mode",
    "outlier_leg_k", "outlier_leg_cv", "outlier_ref", "unclosed_sectors", "unclosed_cv",
    "unclosed_k", "leg_sigma_a_ref", "leg_sigma_gamma", "leg_sigma_cv_cap",
    "flat_sigma_high", "flat_default_cv", "time", "stock_sectors", "stock_seed_cv",
    "stock_seed_abs_floor", "effless_stock_sigma", "input_pins", "overrides",
}
CONST_KEYS = {
    "p", "sigma_a_cv_min", "sigma_a_abs_floor", "sigma_c_cv_min", "sigma_c_abs_floor",
    "prior_box_k", "nodata_anchor_cv", "nodata_anchor_exclude",
    "nodata_anchor_data_aware", "nodata_anchor_grade_region_years",
    "pin_flat_prior_zeros", "ratio_sigma_rel_floor", "ratio_scale_abs_floor",
    "stock_bounds_scale_abs_floor", "ore_grade_basis", "fabricated_anchor_cv",
    "feed_share_observed", "stock_bounds_mode", "stock_bounds_skip_reported",
    "scrap_share_mode", "scrap_level_anchor", "massbal_mask", "enforce_lower_bounds",
    "stripping_ratio_hard", "overrides",
}


def _toy_imputers(ctx):
    from pysuto.impute import FlatPriorImputer
    spec = ctx.case.imp
    return [FlatPriorImputer(acc_struct=ctx.acc, mode="noisy_zero",
                             sigma_high=spec.FLAT_SIGMA_HIGH, p=spec.P_PROP,
                             id="flat_prior")]


def _toy_builders(ctx):
    from pysuto.constraints import MassBalanceBuilder
    return [MassBalanceBuilder()]


_HOOK_CALLS: list = []


def _toy_pre_impute(ctx):
    _HOOK_CALLS.append(("pre", ctx.agg_result.imputation_log is None))


def _toy_post_impute(ctx):
    _HOOK_CALLS.append(("post", ctx.agg_result.imputation_log is not None))


def _toy_case():
    """The copper case's knobs with the toy's cascade, hooks, builders and acc."""
    from dataclasses import replace
    from pysuto.cases.copper import CASE
    return replace(CASE, id="toy", acc_id="toy",
                   imp=replace(CASE.imp, imputers=_toy_imputers,
                               pre_impute=_toy_pre_impute, post_impute=_toy_post_impute),
                   const=replace(CASE.const, builders=_toy_builders,
                                 nodata_anchor_exempt=None))


def _qualname(fn) -> str:
    return f"{fn.__module__}.{fn.__qualname__}"


def _as_json(obj):
    import json
    from pysuto.registry import _NpEncoder
    return json.loads(json.dumps(obj, cls=_NpEncoder))


def _register_feed_run(case) -> str:
    """A feed run of the case built on toy v2, covering its first two edges."""
    from pysuto.acc import AccountingStructure
    from pysuto.registry import DataFeedRegistry
    from pysuto.registry._sidecar import build_sidecar
    n = AccountingStructure.load("toy", storage_dir=case["acc_dir"]).n_edges
    f, sigma = np.zeros(n), np.zeros(n)
    f[:2], sigma[:2] = (5.0, 3.0), (0.5, 0.3)
    reg = DataFeedRegistry(odir=str(case["data"] / "proc" / "feed" / "registry"))
    run_dir = reg._make_run_dir("feed_toy_ref")
    np.savez(os.path.join(run_dir, "result.npz"), F_root=f, sigma_root=sigma)
    reg._write_sidecar(run_dir, build_sidecar(
        run_id="feed_toy_ref", stage="feed", label="", created_at="2026-11-11T00:00:00Z",
        parents={"inputs": {"acc": case["entry"], "feed": {"id": "toy", "version": 1}},
                 "upstream": {}}))
    reg._add_entry(run_id="feed_toy_ref", label="", created_at="2026-11-11T00:00:00Z")
    reg.save()
    return "feed_toy_ref"


def _register_imp_run(case) -> str:
    """An imp run of the case built on toy v2: every edge observed by one feed."""
    from pysuto.acc import AccountingStructure
    from pysuto.agg import AggregationResult
    from pysuto.impute import ImputationLog
    from pysuto.registry import ImputationRunRegistry
    n = AccountingStructure.load("toy", storage_dir=case["acc_dir"]).n_edges
    f = np.arange(1.0, n + 1)
    agg = AggregationResult(f_agg=f, sigma_agg=0.1 * f, Omega=np.ones((1, n)),
                            F=f[None, :], Sigma=0.1 * f[None, :], feed_ids=["toy"],
                            coverage=np.ones(n, dtype=bool))
    agg.imputation_log = ImputationLog()
    return ImputationRunRegistry(storage_dir=str(case["data"] / "proc" / "imp")).register(
        agg, label="case", parents={"inputs": {"acc": case["entry"], "params": []},
                                    "upstream": {"feed_run_ids": []}})


class _NoDiag:
    """Stands in for the Stage 2 diagnostics, which draw the copper case's 1995."""

    def __init__(self, *args, **kwargs):
        pass

    def summary(self):
        pass

    def plot(self, *args, **kwargs):
        return []

    def write(self, *args, **kwargs):
        return []


def test_stage_2_builds_from_the_case_and_records_it(case, driver, monkeypatch):
    """The toy case's cascade is what runs; the sidecar carries the serialised Stage 2
    section beside every key it recorded before, and the override lands on both."""
    from pysuto.input_registry import InputRegistry
    from pysuto.registry import ImputationRunRegistry
    conc = case["data"] / "input" / "params" / "conc" / "current"
    conc.mkdir(parents=True)
    (conc / "params.csv").write_text("param_id,param_type,a_sector_from,a_valuation,"
                                     "value_nominal\nconcore,concentration,Ore,Mass,0.01\n")
    InputRegistry("params", root=case["data"] / "input").initialize()
    feed = _register_feed_run(case)
    mod = driver("collect_aggregate_impute.py")
    for name in ("AggDiagnostic", "ImpDiagnostic", "CoverageDiagnostic",
                 "StatsDiagnostic"):
        monkeypatch.setattr(mod, name, _NoDiag)
    toy = _toy_case().with_knobs("imp", {"TIME_WINDOW": 5})
    _HOOK_CALLS.clear()

    imp_id = mod.main([feed], "case", {"TIME_WINDOW": 5}, case=toy,
                      data_dir=str(case["data"]), config=case["config"])

    assert _HOOK_CALLS == [("pre", True), ("post", True)]    # before and after imputing

    sidecar = (ImputationRunRegistry(str(case["data"] / "proc" / "imp"))
               .load_sidecar(imp_id))
    cfg = sidecar["stage_config"]
    assert set(cfg) == IMP_KEYS | {"case"}
    assert cfg["case"] == _as_json(toy.describe("imp"))
    assert cfg["case"]["id"] == "toy"
    assert cfg["case"]["imp"]["imputers"] == _qualname(_toy_imputers)
    assert cfg["case"]["imp"]["pre_impute"] == _qualname(_toy_pre_impute)
    assert cfg["case"]["imp"]["TIME_WINDOW"] == cfg["time"]["window"] == 5
    assert cfg["overrides"] == {"TIME_WINDOW": 5}
    assert {s["imputer_id"] for s in sidecar["imputers"]} == {"flat_prior"}


def test_stage_3_builds_from_the_case_and_records_it(case, driver):
    """The toy case's builders are what run; the sidecar carries the serialised Stage 3
    section beside every key it recorded before, and the override lands on both."""
    from pysuto.registry import ConstraintRunRegistry
    imp_id = _register_imp_run(case)
    knobs = {"ANCHOR_CV": 0.02, "DIAG_PLOT": False}
    toy = _toy_case().with_knobs("const", knobs)

    const_id = driver("constraints.py").main(imp_id, "case", knobs, case=toy,
                                             data_dir=str(case["data"]),
                                             config=case["config"])

    reg = ConstraintRunRegistry(str(case["data"] / "proc" / "const"))
    cfg = reg.load_sidecar(const_id)["stage_config"]
    assert set(cfg) == CONST_KEYS | {"case"}
    assert cfg["case"] == _as_json(toy.describe("const"))
    assert cfg["case"]["const"]["builders"] == _qualname(_toy_builders)
    assert cfg["case"]["const"]["nodata_anchor_exempt"] is None
    assert cfg["case"]["const"]["ANCHOR_CV"] == 0.02
    assert cfg["overrides"] == knobs
    assert cfg["nodata_anchor_grade_region_years"] == 0
    assert set(reg.load(const_id).row_meta["type"]) == {"massbal"}


# ---------------------------------------------------------------------------
# ConstDiagnostic's default figure directory is the run's own (#58)
# ---------------------------------------------------------------------------

def _no_load(cls, id, **kwargs):
    raise AssertionError(f"the structure {id!r} was loaded before its build record "
                         f"was checked")


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
