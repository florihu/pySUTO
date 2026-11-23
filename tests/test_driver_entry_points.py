"""test_driver_entry_points.py
==========================
The scripts' entry points (#219): the feed drivers answer ``--help`` without building,
``solver_stats.py`` requires its run id, ``run_solver.main`` defaults to cold and no
driver names a file or log that does not exist.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FEEDS   = ["icsg_feed.py", "baci_feed.py", "crude_ore_feed.py"]


def _run(script, args, cwd):
    env = {**os.environ, "PYTHONPATH": str(SCRIPTS.parent / "src")}
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args], cwd=cwd,
                          capture_output=True, text=True, env=env, timeout=300)


@pytest.mark.parametrize("script", FEEDS)
def test_a_feed_driver_answers_help_and_writes_nothing(script, tmp_path):
    res = _run(script, ["--help"], tmp_path)
    assert res.returncode == 0, res.stderr
    assert "usage:" in res.stdout and "config/config.yaml" in res.stdout
    assert list(tmp_path.iterdir()) == []                 # no data/, no logs/


@pytest.mark.parametrize("script", FEEDS)
def test_a_feed_driver_refuses_an_option_it_does_not_take(script, tmp_path):
    res = _run(script, ["--nonsense"], tmp_path)
    assert res.returncode == 2
    assert list(tmp_path.iterdir()) == []


def test_solver_stats_requires_the_run_id(tmp_path):
    res = _run("solver_stats.py", [], tmp_path)
    assert res.returncode == 2 and "opt_run_id" in res.stderr
    assert list(tmp_path.iterdir()) == []


def test_run_solver_main_defaults_to_the_cold_prior():
    spec = importlib.util.spec_from_file_location("run_solver_script",
                                                  SCRIPTS / "run_solver.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert inspect.signature(mod.main).parameters["prior_mode"].default == "cold"


def test_no_driver_names_a_file_or_log_that_does_not_exist():
    stale = re.compile(r"\binit_(run_solver|uncertainty_propagation|copper_constraints"
                       r"|psut|solve_diag)\b")
    for path in sorted(SCRIPTS.glob("*.py")):
        assert not stale.search(path.read_text(encoding="utf-8")), path.name
