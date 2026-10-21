"""test_driver_knobs.py — each Stage 2–4 driver's knob surface (#100).

The drivers are imported by path under a private module name (import-safe: their
``main()`` runs only under ``__main__``) and run as subprocesses for the CLI checks.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pysuto.knobs import parse_overrides

REPO    = Path(__file__).resolve().parents[1]
DRIVERS = {
    "imp":   ("collect_aggregate_impute.py", ["--feed-id", "feed_x"]),
    "const": ("constraints.py",              ["--imp-run-id", "imp_x", "lbl"]),
    "opt":   ("run_solver.py",               ["--const-run-id", "const_x"]),
}


def _load(stage):
    script = REPO / "scripts" / DRIVERS[stage][0]
    spec = importlib.util.spec_from_file_location(f"_knobs_driver_{stage}", script)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def _run(stage, *extra):
    script = REPO / "scripts" / DRIVERS[stage][0]
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run([sys.executable, str(script), *DRIVERS[stage][1], *extra],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("stage", sorted(DRIVERS))
class TestDriverKnobs:

    def test_every_knob_is_an_upper_case_global_of_the_driver(self, stage):
        mod = _load(stage)
        assert len(set(mod.KNOBS)) == len(mod.KNOBS)
        for name in mod.KNOBS:
            assert name.isupper() and hasattr(mod, name), name

    def test_paths_and_the_acc_id_are_not_knobs(self, stage):
        mod = _load(stage)
        assert not [k for k in mod.KNOBS if k.endswith("_DIR") or k == "ACC_ID"]

    def test_every_default_round_trips_through_its_repr(self, stage):
        mod = _load(stage)
        ns = vars(mod)
        pairs = [f"{k}={ns[k]!r}" for k in mod.KNOBS]
        assert parse_overrides(pairs, mod.KNOBS, ns) == {k: ns[k] for k in mod.KNOBS}

    def test_an_unknown_knob_fails_before_anything_is_loaded(self, stage):
        r = _run(stage, "--set", "NOT_A_KNOB=1")
        assert r.returncode != 0
        assert "NOT_A_KNOB" in r.stderr and "is not a knob" in r.stderr
        assert "Traceback" not in r.stderr


def test_a_knob_of_the_wrong_type_fails_before_anything_is_loaded():
    r = _run("opt", "--set", "TOL_GAP_REL='x'")
    assert r.returncode != 0
    assert "TOL_GAP_REL" in r.stderr and "expected float" in r.stderr
