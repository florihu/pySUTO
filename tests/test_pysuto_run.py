"""test_pysuto_run.py — ``python -m pysuto run`` (#125): the toy case's six stages in one
command, under its own labels.

One module-scoped chain runs ``python -m pysuto run --case toy --from feed --to psut
--data-dir <tmp>`` as a subprocess from an empty working directory, which must hold no
``data/`` afterwards. The result is read back through the registries, its sidecars'
parents are walked from the PSUT to the feeds, and the toy's own checks of
``test_toy_case.py`` run on it (hard rows close, the planted leg is the top soft
residual, Stage 5 coverage lies in its recorded band, the PSUT balances). The command's
argument handling and the step commands are checked without running a driver.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import test_toy_case as toy_checks
from pysuto import acc_store
from pysuto.tutorial import toy
from pysuto.pipeline import ALL_STEPS, Chain, Pipeline
from pysuto.registry import (ConstraintRunRegistry, ImputationRunRegistry,
                             OptimRunRegistry, PsutRunRegistry, UncPropRunRegistry)

REPO = Path(__file__).resolve().parents[1]


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "pysuto", "run", *args], cwd=str(cwd),
                          capture_output=True, text=True, encoding="utf-8",
                          env={**os.environ, "PYTHONPATH": str(REPO / "src")})


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    root, cwd = tmp_path_factory.mktemp("toy_data"), tmp_path_factory.mktemp("cwd")
    proc = _run(cwd, "--case", "toy", "--from", "feed", "--to", "psut",
                "--data-dir", str(root))
    assert proc.returncode == 0, f"{proc.stdout[-1500:]}\n{proc.stderr[-3000:]}"
    printed = {line.split()[0]: line.split()[1:] for line in proc.stdout.splitlines()}
    ids = {"feed": dict(zip(toy.FEED_IDS, printed["feed"]))}
    ids.update({s: printed[s][0] for s in ("imp", "const", "opt", "uncprop", "psut")})
    return {"root": root, "cwd": cwd, "ids": ids, "stdout": proc.stdout,
            "acc": acc_store.load(toy.ACC_ID,
                                            storage_dir=str(root / "proc" / "acc"))}


@pytest.fixture(scope="module")
def file_chain(tmp_path_factory):
    """The same chain through ``--case <file>``, the file re-exporting the toy's CASE."""
    root, cwd = tmp_path_factory.mktemp("toy_file_data"), tmp_path_factory.mktemp("cwd2")
    case_file = cwd / "zinc_case.py"
    case_file.write_text("from pysuto.tutorial.toy import CASE\n")
    proc = _run(cwd, "--case", "./zinc_case.py", "--from", "feed", "--to", "psut",
                "--data-dir", str(root))
    assert proc.returncode == 0, f"{proc.stdout[-1500:]}\n{proc.stderr[-3000:]}"
    printed = {line.split()[0]: line.split()[1:] for line in proc.stdout.splitlines()}
    return {"root": root, "case_file": case_file,
            "ids": {s: printed[s][0] for s in ("imp", "const", "opt", "uncprop", "psut")}}


def _stored(run_dir: Path) -> dict:
    """Every stored array and frame of a run directory (diagnostics left out) by file and
    key, and its ``parents.inputs`` without the registry date and code stamps."""
    out = {}
    for path in sorted(run_dir.rglob("*")):
        rel = path.relative_to(run_dir)
        if "diag" in rel.parts or not path.is_file():
            continue
        if path.suffix == ".parquet":
            out[(str(rel), None)] = pd.read_parquet(path)
        elif path.suffix in (".npy", ".npz"):
            data = np.load(path, allow_pickle=True)
            for key in ([None] if path.suffix == ".npy" else list(data.keys())):
                out[(str(rel), key)] = data if key is None else data[key]
    inputs = json.loads((run_dir / "metadata.json").read_text())["parents"]["inputs"]
    out["inputs"] = json.loads(re.sub(r'"(date|code)": "[^"]*"', '"x": 0',
                                      json.dumps(inputs)))
    return out


def test_a_case_file_outside_src_runs_the_chain_bit_identical_to_the_alias(
        chain, file_chain):
    """#187 AC1: every stored array and frame and ``parents.inputs`` equal; only the
    provenance of the sidecars differs."""
    regs = {"imp": ImputationRunRegistry, "const": ConstraintRunRegistry,
            "opt": OptimRunRegistry, "uncprop": UncPropRunRegistry,
            "psut": PsutRunRegistry}
    for stage, reg in regs.items():
        a_id, b_id = chain["ids"][stage], file_chain["ids"][stage]
        a = _stored(chain["root"] / "proc" / stage / a_id)
        b = _stored(file_chain["root"] / "proc" / stage / b_id)
        assert a.keys() == b.keys() and len(a) > 1, stage
        for key in a:
            if key == "inputs":
                assert a[key] == b[key], stage
            elif isinstance(a[key], pd.DataFrame):
                assert a[key].equals(b[key]), (stage, key)
            else:
                nan = a[key].dtype.kind in "fc"     # NaN: "not covered", equal to itself
                assert np.array_equal(a[key], b[key], equal_nan=nan), (stage, key)
        ca = reg(str(chain["root"] / "proc" / stage)).load_sidecar(
            a_id)["stage_config"]["case"]
        cb = reg(str(file_chain["root"] / "proc" / stage)).load_sidecar(
            b_id)["stage_config"]["case"]
        assert ca["case_source"] == "alias:toy"
        assert cb["case_source"] == f"path:{file_chain['case_file']}"
        assert re.fullmatch(r"[0-9a-f]{64}", ca["case_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", cb["case_sha256"])
        assert ca["case_sha256"] != cb["case_sha256"]
        assert {k: v for k, v in ca.items() if not k.startswith("case_")} == \
            {k: v for k, v in cb.items() if not k.startswith("case_")}


def test_the_command_leaves_six_stages_linked_through_their_sidecars(chain):
    root, ids = chain["root"], chain["ids"]
    proc = root / "proc"
    assert not (chain["cwd"] / "data").exists()
    regs = {"imp": ImputationRunRegistry, "const": ConstraintRunRegistry,
            "opt": OptimRunRegistry, "uncprop": UncPropRunRegistry,
            "psut": PsutRunRegistry}
    side = {s: r(str(proc / s)).load_sidecar(ids[s]) for s, r in regs.items()}
    for stage in regs:
        assert (proc / stage / ids[stage]).is_dir()
    for run_id in ids["feed"].values():
        assert run_id.startswith("feed_")
        assert (proc / "feed" / "registry" / run_id).is_dir()
    feed_ids = side["imp"]["parents"]["upstream"]["feed_run_ids"]
    assert feed_ids == list(ids["feed"].values())
    assert side["const"]["parents"]["upstream"]["imp_run_id"] == ids["imp"]
    assert side["opt"]["parents"]["upstream"]["const_run_id"] == ids["const"]
    assert side["uncprop"]["parents"]["upstream"]["opt_run_id"] == ids["opt"]
    assert side["psut"]["parents"]["upstream"]["opt_run_id"] == ids["opt"]
    for stage, sidecar in side.items():
        assert sidecar["label"] == "toy" and \
            sidecar["stage_config"]["case"]["id"] == "toy", stage


def test_the_toy_s_checks_hold_on_the_one_command_chain(chain):
    toy_checks.test_every_hard_row_closes_to_1e_10(chain)
    toy_checks.test_the_planted_leg_is_the_top_soft_residual(chain)
    toy_checks.test_stage5_coverage_of_the_420_edges_lies_in_its_recorded_band(chain)
    toy_checks.test_the_psut_supply_and_use_blocks_balance_per_node(chain)


def test_a_feed_step_needs_the_case_s_stage_1_hook(tmp_path, monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr("pysuto.pipeline.resolve_case",
                        lambda ref: replace(toy.CASE, stage1=None))
    with pytest.raises(ValueError, match="declares none"):
        Pipeline(steps=ALL_STEPS, case="x", data_dir=str(tmp_path))


def test_a_start_past_the_feed_step_needs_a_base_and_a_feed_start_refuses_one(tmp_path):
    no_base = _run(tmp_path, "--case", "toy", "--from", "imp",
                   "--data-dir", str(tmp_path))
    assert no_base.returncode == 2 and "--base" in no_base.stderr
    both = _run(tmp_path, "--case", "toy", "--from", "feed", "--base", "opt_x",
                "--data-dir", str(tmp_path))
    assert both.returncode == 2 and "--base" in both.stderr
    back = _run(tmp_path, "--case", "toy", "--from", "opt", "--to", "imp",
                "--data-dir", str(tmp_path))
    assert back.returncode == 2 and "comes after" in back.stderr


def test_the_command_runs_six_real_children_and_looks_up_no_scripts_directory(
        tmp_path, monkeypatch, capsys):
    """#186: each step, the feed hook included, is a fresh ``python -m pysuto._step``
    child of the parent's own ``pysuto``; no driver script or ``REPO_ROOT`` is
    involved."""
    import pysuto.pipeline as pipeline_module
    from pysuto.__main__ import main

    launched, real_run = [], subprocess.run

    def record(cmd, *args, **kwargs):
        launched.append((list(cmd), kwargs.get("env", {}).get("PYTHONPATH", "")))
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", record)
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--case", "toy", "--data-dir", str(tmp_path / "d")]) == 0
    assert not hasattr(pipeline_module, "REPO_ROOT")
    assert len(launched) == 6
    # the parent's own pysuto: its source tree first, or nothing added for an installed
    # wheel (the rule's two branches are tested in test_pipeline.py)
    expected = pipeline_module._child_env().get("PYTHONPATH", "")
    for cmd, pythonpath in launched:
        assert cmd[1:3] == ["-m", "pysuto._step"]
        assert not any("scripts" in part for part in cmd)
        assert pythonpath == expected
    printed = {l.split()[0] for l in capsys.readouterr().out.splitlines() if l.split()}
    assert {"feed", "imp", "const", "opt", "uncprop", "psut"} <= printed
    for stage in ("imp", "const", "opt", "uncprop", "psut"):
        assert len(list((tmp_path / "d" / "proc" / stage).glob(f"{stage}_*"))) == 1
    log = (tmp_path / "d" / "logs" / "pipeline" / "toy_opt.log").read_text()
    assert "run_opt DONE" in log                    # the child's output is the step log


def test_a_child_that_fails_raises_step_failed_with_its_returncode_and_log_tail(tmp_path):
    from pysuto.pipeline import StepFailed
    base = Chain(feed_run_ids=(), imp="imp_x", const="const_missing", opt="opt_x")
    pipe = Pipeline(steps=("opt",), base=base, case="toy",
                    cwd=str(tmp_path), data_dir="d")
    with pytest.raises(StepFailed) as exc:
        pipe.run({}, label="fails")
    assert exc.value.step == "opt" and exc.value.returncode != 0
    assert "const_missing" in exc.value.tail
    assert Path(exc.value.log_path).is_file()


def test_the_child_parses_knobs_as_literals_against_the_stage_defaults(monkeypatch):
    import pysuto._step as step
    from dataclasses import replace
    seen = {}
    monkeypatch.setattr("pysuto.stages.run_opt", lambda case, up, label, knobs, **kw:
                        seen.update(knobs=knobs, kw=kw) or "opt_x")
    spec = step.StepSpec(step="opt", case="toy", label="l", upstream="const_1",
                         data_dir="d", config=None, dirs={"const_dir": "c"},
                         knobs=["TOL_GAP_REL=1e-6", "QP_VARIABLES='flows'"])
    assert step.run_spec(spec) == "opt_x"
    assert seen["knobs"] == {"TOL_GAP_REL": 1e-6, "QP_VARIABLES": "flows"}
    assert seen["kw"] == {"data_dir": "d", "const_dir": "c"}
    with pytest.raises(TypeError, match="TOL_GAP_REL"):
        step.run_spec(replace(spec, knobs=["TOL_GAP_REL='x'"]))
    with pytest.raises(KeyError, match="NOT_A_KNOB"):
        step.run_spec(replace(spec, knobs=["NOT_A_KNOB=1"]))
    with pytest.raises(KeyError, match="takes no knobs"):
        step.run_spec(replace(spec, step="uncprop", knobs=["ALPHA=0.1"]))


def test_uncprop_and_psut_take_no_knobs():
    base = Chain(feed_run_ids=("feed_a",), imp="imp_1", const="const_1", opt="opt_1")
    pipe = Pipeline(steps=ALL_STEPS[1:], base=base, case="toy",
                    data_dir="d")
    with pytest.raises(KeyError, match="uncprop step takes no knobs"):
        pipe.check_overrides({"uncprop.ALPHA": 0.1})


# =============================================================================
# Review fixes U8 (#279): the entry past Stage 1, the real-child --config, the stage
# guards
# =============================================================================


def _printed(proc) -> dict:
    """The command's result lines (stage name, run ids); log lines share its stdout."""
    return {l.split()[0]: l.split()[1:] for l in proc.stdout.splitlines()
            if l.split() and l.split()[0] in ALL_STEPS}


def _copy_data(chain, tmp_path) -> Path:
    import shutil
    dest = tmp_path / "data"
    shutil.copytree(chain["root"], dest)
    return dest


def test_run_from_imp_with_a_base_runs_only_the_later_stages_and_prints_only_them(
        chain, tmp_path):
    """U8-G6-2 and G4-4: the main entry past Stage 1 reads its upstream from ``--base``
    and prints one line per stage it ran, none for the base chain's feed or imp."""
    data = _copy_data(chain, tmp_path)
    proc = _run(tmp_path, "--case", "toy", "--from", "const", "--to", "opt",
                "--base", chain["ids"]["opt"], "--data-dir", str(data))
    assert proc.returncode == 0, f"{proc.stdout[-1500:]}\n{proc.stderr[-3000:]}"
    printed = _printed(proc)
    assert set(printed) == {"const", "opt"}
    assert printed["const"][0] != chain["ids"]["const"]
    assert printed["opt"][0] != chain["ids"]["opt"]
    side = json.loads((data / "proc" / "const" / printed["const"][0]
                       / "metadata.json").read_text())
    assert side["parents"]["upstream"]["imp_run_id"] == chain["ids"]["imp"]


def test_a_case_without_a_stage_1_hook_ends_a_feed_start_with_exit_2(tmp_path):
    """U8-G6-2: the exit is the real command's, not a monkeypatched constructor."""
    (tmp_path / "hookless.py").write_text(
        "from dataclasses import replace\n"
        "from pysuto.tutorial.toy import CASE as _TOY\n"
        "CASE = replace(_TOY, stage1=None)\n")
    proc = _run(tmp_path, "--case", "./hookless.py", "--from", "feed", "--to", "feed",
                "--data-dir", str(tmp_path / "d"))
    assert proc.returncode == 2 and "declares none" in proc.stderr


def test_a_config_names_the_acc_id_a_real_child_runs_on(chain, tmp_path):
    """U8-G6-3: ``--config`` reaches the child, which replaces the case's acc id by the
    config's; an acc nobody built fails the imp step by name."""
    data = _copy_data(chain, tmp_path)
    cfg = tmp_path / "other.yaml"
    cfg.write_text("pipeline:\n  acc_id: no_such_acc\n")
    proc = _run(tmp_path, "--case", "toy", "--from", "imp", "--to", "imp",
                "--base", chain["ids"]["opt"], "--data-dir", str(data),
                "--config", str(cfg))
    assert proc.returncode != 0 and "no_such_acc" in proc.stderr + proc.stdout


def test_a_const_step_refuses_an_imp_run_built_on_another_acc_version(chain, tmp_path):
    """U8-G6-4: the imp/const acc-mismatch guard, a provenance guard of Stage 3."""
    data = _copy_data(chain, tmp_path)
    path = data / "proc" / "imp" / chain["ids"]["imp"] / "metadata.json"
    side = json.loads(path.read_text())
    side["parents"]["inputs"]["acc"]["version"] = 999
    path.write_text(json.dumps(side))
    proc = _run(tmp_path, "--case", "toy", "--from", "const", "--to", "const",
                "--base", chain["ids"]["opt"], "--data-dir", str(data))
    assert proc.returncode != 0 and "acc mismatch" in proc.stderr + proc.stdout


def test_a_case_screens_reports_are_written_as_csv_under_the_imp_runs_diag(
        chain, tmp_path):
    """U8-G6-4: ``ctx.reports`` goes to ``<imp run>/diag/<name>.csv``."""
    data = _copy_data(chain, tmp_path)
    (tmp_path / "screened.py").write_text(
        "from dataclasses import replace\n"
        "import pandas as pd\n"
        "from pysuto.tutorial.toy import CASE as _TOY\n"
        "def _screen(ctx):\n"
        "    ctx.reports['demo_screen'] = pd.DataFrame({'a': [1, 2]})\n"
        "CASE = replace(_TOY, imp=replace(_TOY.imp, pre_impute=_screen))\n")
    proc = _run(tmp_path, "--case", "./screened.py", "--from", "imp", "--to", "imp",
                "--base", chain["ids"]["opt"], "--data-dir", str(data))
    assert proc.returncode == 0, f"{proc.stdout[-1500:]}\n{proc.stderr[-3000:]}"
    imp_id = _printed(proc)["imp"][0]
    out = pd.read_csv(data / "proc" / "imp" / imp_id / "diag" / "demo_screen.csv")
    assert out["a"].tolist() == [1, 2]


def test_the_imp_sidecar_records_the_feed_collectors_coverage_threshold(chain):
    """U8-G3-1: ``MIN_FLOW_KT`` is a Stage 2 knob, recorded with the case's others."""
    side = json.loads((chain["root"] / "proc" / "imp" / chain["ids"]["imp"]
                       / "metadata.json").read_text())
    assert side["stage_config"]["case"]["imp"]["MIN_FLOW_KT"] == 0.01


def test_the_steps_write_utf8_when_the_locale_encodes_stdout_as_cp1252(tmp_path):
    """#293: a pipe on Windows encodes with cp1252, which lacks the ``σ`` of the diag
    summary the imp body prints; the step child writes UTF-8 whatever the parent's
    ``PYTHONIOENCODING`` says, and its log files are UTF-8."""
    root = tmp_path / "data"
    proc = subprocess.run(
        [sys.executable, "-m", "pysuto", "run", "--case", "toy", "--to", "imp",
         "--data-dir", str(root)], cwd=str(tmp_path), capture_output=True, text=True,
        encoding="utf-8",
        env={**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONIOENCODING": "cp1252"})
    assert proc.returncode == 0, f"{proc.stdout[-1500:]}\n{proc.stderr[-3000:]}"
    step_log = (root / "logs" / "pipeline" / "toy_imp.log").read_text(encoding="utf-8")
    assert "σ" in step_log
    for log in (root / "logs").glob("*.log"):
        log.read_text(encoding="utf-8")                  # raises on a non-UTF-8 byte
