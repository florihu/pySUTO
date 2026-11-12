"""test_generic_drivers.py — the Stage 2–6 drivers take ``--case`` and run the toy case
(#124).

One module-scoped chain seeds the toy's Stage 1 (the case's own ``run_stage1``) into a
data dir under ``tmp_path`` and runs the five stage drivers as subprocesses with
``--case toy --data-dir …`` from an empty working directory, which must hold no ``data/``
afterwards. The chain is compared with the toy's own ``run_stage4`` on the same const run
(the drivers add no policy to the library) and with the numbers of ``test_toy_case.py``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pysuto.case import CaseSpec, check_case
from pysuto.cases import case_ids, load_case, toy
from pysuto.registry import (ConstraintRunRegistry, ImputationRunRegistry,
                             OptimRunRegistry, PsutRunRegistry, UncPropRunRegistry)

REPO = Path(__file__).resolve().parents[1]


def _run(cwd: Path, script: str, *args: str) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, str(REPO / "scripts" / script), *args],
                          cwd=str(cwd), capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert proc.returncode == 0, (f"{script} failed:\n{proc.stdout[-1500:]}\n"
                                  f"{proc.stderr[-3000:]}")
    return proc


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """The toy's Stage 1, then Stages 2–6 through the drivers."""
    root, cwd = tmp_path_factory.mktemp("toy_data"), tmp_path_factory.mktemp("cwd")
    feeds = toy.run_stage1(root)
    case  = ["--case", "toy", "--data-dir", str(root)]
    ids   = {"feed": feeds}

    def run_id(name: str, *args: str) -> str:
        path = cwd / f"{name}.id"
        _run(cwd, *args, "--run-id-file", str(path))
        return path.read_text().strip()

    ids["imp"] = run_id("imp", "collect_aggregate_impute.py", *case, "--label", "toy",
                        *[a for f in feeds for a in ("--feed-id", f)])
    ids["const"] = run_id("const", "constraints.py", *case, "--imp-run-id", ids["imp"],
                          "toy")
    ids["opt"] = run_id("opt", "run_solver.py", *case, "--const-run-id", ids["const"],
                        "--label", "toy")
    _run(cwd, "uncertainty_propagation.py", ids["opt"], *case, "--label", "toy")
    ids["uncprop"] = UncPropRunRegistry(str(root / "proc" / "uncprop")
                                        ).latest_by_label("toy").run_id
    _run(cwd, "psut.py", ids["opt"], *case, "--label", "toy")
    ids["psut"] = PsutRunRegistry(str(root / "proc" / "psut")).latest_by_label("toy").run_id
    return {"root": root, "cwd": cwd, "ids": ids}


def test_the_drivers_write_nothing_under_the_working_directory_s_data(chain):
    assert not (chain["cwd"] / "data").exists()
    for stage in ("imp", "const", "opt", "uncprop", "psut"):
        assert (chain["root"] / "proc" / stage).is_dir()


def test_every_sidecar_records_the_case_id(chain):
    root, ids = chain["root"], chain["ids"]
    registries = {"imp": ImputationRunRegistry, "const": ConstraintRunRegistry,
                  "opt": OptimRunRegistry, "uncprop": UncPropRunRegistry,
                  "psut": PsutRunRegistry}
    for stage, registry in registries.items():
        sidecar = registry(str(root / "proc" / stage)).load_sidecar(ids[stage])
        assert sidecar["stage_config"]["case"]["id"] == "toy", stage


def test_the_driver_chain_reproduces_the_toy_s_own_numbers(chain):
    """Same posterior as ``toy.run_stage4`` on the same const run, and the scoreboard of
    ``test_toy_case.py``: the planted leg leads the soft residuals and 392 of the 420
    edges lie inside their interval."""
    root, ids = chain["root"], chain["ids"]
    own = toy.run_stage4(root, ids["const"])
    registry = OptimRunRegistry(str(root / "proc" / "opt"))
    theirs, ours = registry.load(ids["opt"]), registry.load(own)
    assert set(theirs) == set(ours) == set(toy.YEARS)
    for t in toy.YEARS:
        assert theirs[t].solver_status == "Solved"
        np.testing.assert_allclose(theirs[t].F_opt, ours[t].F_opt, rtol=1e-6, atol=1e-8)
    soft = toy.soft_residuals(root, ids["opt"])
    assert soft.iloc[0]["z"] == pytest.approx(-4.779, abs=5e-4)
    board = toy.scoreboard(root, ids)
    assert round(board.coverage() * 420) == 392


def test_the_psut_primary_defaults_to_the_acc_s_content_valuation(chain):
    root, ids = chain["root"], chain["ids"]
    psut = PsutRunRegistry(str(root / "proc" / "psut")).load(ids["psut"])
    assert psut.primary == toy.CONTENT


def test_an_unknown_case_raises_listing_the_known_ones():
    with pytest.raises(ValueError, match="'nope'.*copper.*toy"):
        load_case("nope")
    assert {"copper", "toy"} <= set(case_ids())
    assert isinstance(load_case("toy"), CaseSpec) and load_case("copper").id == "copper"


def test_a_driver_refuses_an_unknown_case_with_a_usage_error(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_solver.py"), "--case", "nope",
         "--const-run-id", "const_x", "--data-dir", str(tmp_path)],
        cwd=str(tmp_path), capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert proc.returncode == 2 and "unknown case 'nope'" in proc.stderr


def test_a_run_of_another_case_is_refused():
    sidecar = {"run_id": "const_x", "stage_config": {"case": {"id": "copper"}}}
    with pytest.raises(ValueError, match="made by case 'copper'.*--case toy"):
        check_case(sidecar, toy.CASE)
    check_case({"run_id": "old", "stage_config": {}}, toy.CASE)        # no record: passes
    check_case(sidecar, load_case("copper"))


# ---------------------------------------------------------------------------
# The library holds no copper default the toy has to override (#124)
# ---------------------------------------------------------------------------

def test_the_constraint_engine_exempts_no_sector_from_the_anchor_by_default(tmp_path):
    from pysuto.constraints import ConstraintEngine
    engine = ConstraintEngine(acc_struct=None, storage_dir=str(tmp_path / "const"))
    assert engine._nodata_anchor_exclude == ()


def test_saving_a_weight_matrix_to_a_path_creates_no_default_directory(tmp_path,
                                                                       monkeypatch):
    acc  = toy.build_acc(tmp_path / "acc")
    feed = toy.make_feeds(toy.truth())["survey"]
    wm   = toy._feed_steps(feed, acc)[1]
    monkeypatch.chdir(tmp_path)
    wm.save(path=str(tmp_path / "out" / "wm.pkl"))
    assert (tmp_path / "out" / "wm.pkl").is_file()
    assert not (tmp_path / "data").exists()


def test_building_an_acc_prints_nothing(tmp_path, capsys):
    toy.build_acc(tmp_path)
    assert capsys.readouterr().out == ""


def test_the_default_diagnostic_year_is_the_first_year_of_the_run(chain):
    from pysuto.diag import ConstDiagnostic
    from pysuto.diag.stats.const import ConstStatsDiagnostic
    cr = ConstraintRunRegistry(str(chain["root"] / "proc" / "const")).load(
        chain["ids"]["const"])
    assert ConstDiagnostic(cr).year == toy.YEARS[0]
    assert ConstStatsDiagnostic(cr).years == [toy.YEARS[0]]


def _load_driver(monkeypatch, script: str):
    import importlib.util
    monkeypatch.syspath_prepend(str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(f"_g124_{script[:-3]}",
                                                  REPO / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Stop(Exception):
    pass


def test_copper_s_psut_primary_resolves_to_cc_through_driver_knobs(monkeypatch, tmp_path):
    """One resolution point: ``psut.main`` applies the case's ``driver_knobs``, so the
    library call and the command line (which passes ``--primary`` None) agree."""
    mod = _load_driver(monkeypatch, "psut.py")
    seen = {}

    class _Registry:
        def __init__(self, *a, **k): pass
        def load_sidecar(self, run_id): return {"run_id": run_id}

    def _from_opt_run(*a, **k):
        seen["primary"] = k["primary"]
        raise _Stop

    monkeypatch.setattr(mod, "OptimRunRegistry", _Registry)
    monkeypatch.setattr(mod.PSUT, "from_opt_run", staticmethod(_from_opt_run))
    monkeypatch.chdir(tmp_path)
    for kwargs, expected in (({}, "CC"), ({"primary": "Mass"}, "Mass")):
        with pytest.raises(_Stop):
            mod.main("opt_x", case=load_case("copper"), data_dir=str(tmp_path), **kwargs)
        assert seen["primary"] == expected
    with pytest.raises(_Stop):                    # a case without the knob: the content valuation
        mod.main("opt_x", case=toy.CASE, data_dir=str(tmp_path))
    assert seen["primary"] is None


def test_driver_knobs_are_checked_by_the_knob_protocol():
    from dataclasses import replace
    knobs = ("PRIMARY",)
    ns = {"PRIMARY": None}
    assert load_case("copper").driver_knobs_for("psut", knobs, ns) == {"PRIMARY": "CC"}
    misspelt = replace(toy.CASE, driver_knobs={"psut": {"PRIMARI": "CC"}})
    with pytest.raises(KeyError, match="PRIMARI"):
        misspelt.driver_knobs_for("psut", knobs, ns)
    opt_ns = {"QP_VARIABLES": "defects"}
    wrong = replace(toy.CASE, driver_knobs={"opt": {"QP_VARIABLES": 3}})
    with pytest.raises(TypeError, match="QP_VARIABLES"):
        wrong.driver_knobs_for("opt", ("QP_VARIABLES",), opt_ns)


def test_the_opt_sidecar_records_the_case_s_driver_knobs(chain):
    sidecar = OptimRunRegistry(str(chain["root"] / "proc" / "opt")).load_sidecar(
        chain["ids"]["opt"])
    assert sidecar["stage_config"]["case_knobs"] == {"QP_VARIABLES": "flows"}
