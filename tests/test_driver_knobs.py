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


class TestCascadeOrder:
    """#104: the Stage 2 cascade order is a knob; it must stay a permutation of the
    five cascade imputers, and the terminal pass (time, share, flat) stays fixed."""

    def test_the_default_order_is_todays_cascade(self):
        mod = _load("imp")
        assert mod.CASCADE_ORDER == ("trade", "network", "conc", "ore_grade", "eff")
        assert "CASCADE_ORDER" in mod.KNOBS

    def test_the_cascade_is_built_in_the_given_order(self):
        mod = _load("imp")
        by_id = {k: object() for k in mod.CASCADE_ORDER}
        order = ("network", "trade", "eff", "conc", "ore_grade")
        assert mod.cascade_in_order(by_id, order) == [by_id[k] for k in order]

    @pytest.mark.parametrize("order", [
        ("trade", "network", "conc", "ore_grade", "time"),        # unknown
        ("trade", "trade", "conc", "ore_grade", "eff"),           # repeated
        ("trade", "network", "conc", "ore_grade"),                # missing
    ])
    def test_an_order_that_is_not_a_permutation_is_refused(self, order):
        mod = _load("imp")
        with pytest.raises(ValueError, match="permutation"):
            mod.cascade_in_order({k: object() for k in mod.CASCADE_ORDER}, order)

    def test_a_bad_order_fails_on_the_command_line_before_anything_is_loaded(self):
        r = _run("imp", "--set", 'CASCADE_ORDER=("trade","trade","conc","ore_grade","eff")')
        assert r.returncode != 0 and "permutation" in r.stderr
        assert "Traceback" not in r.stderr


def test_the_feasibility_check_knob_replaces_the_bool():
    mod = _load("opt")
    assert mod.FEASIBILITY_CHECK == "first"
    assert "FEASIBILITY_CHECK" in mod.KNOBS and "FEASIBILITY_CHECK_FIRST_ONLY" not in mod.KNOBS
    assert not hasattr(mod, "FEASIBILITY_CHECK_FIRST_ONLY")
