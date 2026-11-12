"""test_driver_knobs.py — each Stage 2–4 driver's knob surface (#100).

The Stage 2 and 3 knobs are fields of the copper case's sections (#120), read from
``pysuto.cases.copper``; Stage 4's are its driver's module globals, so that driver is
imported by path under a private module name (import-safe: its ``main()`` runs only
under ``__main__``). All three drivers run as subprocesses for the CLI checks.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pysuto.cases.copper import CASE, cascade_in_order
from pysuto.knobs import parse_overrides

REPO    = Path(__file__).resolve().parents[1]
DRIVERS = {
    "imp":   ("collect_aggregate_impute.py", ["--feed-id", "feed_x"]),
    "const": ("constraints.py",              ["--imp-run-id", "imp_x", "lbl"]),
    "opt":   ("run_solver.py",               ["--const-run-id", "const_x"]),
}
# The stages whose knobs are the case spec's fields (#120).
CASE_STAGES = ("imp", "const")


def _load(stage):
    """Stage 4's driver as a module."""
    script = REPO / "scripts" / DRIVERS[stage][0]
    spec = importlib.util.spec_from_file_location(f"_knobs_driver_{stage}", script)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def _surface(stage):
    """``(KNOBS, defaults)`` of the stage: the copper case section's UPPER_CASE fields
    for Stages 2 and 3, the driver's ``KNOBS`` and module globals for Stage 4."""
    if stage in CASE_STAGES:
        defaults = getattr(CASE, stage).knobs()
        return tuple(defaults), defaults
    mod = _load(stage)
    return tuple(mod.KNOBS), {k: getattr(mod, k) for k in mod.KNOBS}


def _run(stage, *extra):
    script = REPO / "scripts" / DRIVERS[stage][0]
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run([sys.executable, str(script), *DRIVERS[stage][1], *extra],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("stage", sorted(DRIVERS))
class TestDriverKnobs:

    def test_every_knob_is_an_upper_case_name_with_a_default(self, stage):
        knobs, defaults = _surface(stage)
        assert len(set(knobs)) == len(knobs)
        assert set(knobs) == set(defaults)
        for name in knobs:
            assert name.isupper(), name

    def test_paths_and_the_acc_id_are_not_knobs(self, stage):
        knobs, _ = _surface(stage)
        assert not [k for k in knobs if k.endswith("_DIR") or k == "ACC_ID"]

    def test_every_default_round_trips_through_its_repr(self, stage):
        knobs, ns = _surface(stage)
        pairs = [f"{k}={ns[k]!r}" for k in knobs]
        assert parse_overrides(pairs, knobs, ns) == {k: ns[k] for k in knobs}

    def test_every_knob_value_can_be_recorded_in_the_sidecar(self, stage):
        """stage_config["overrides"] goes through the registries' JSON encoder."""
        import json
        from pysuto.registry import _NpEncoder
        knobs, ns = _surface(stage)
        back = json.loads(json.dumps({k: ns[k] for k in knobs}, cls=_NpEncoder))
        assert set(back) == set(knobs)

    def test_an_unknown_knob_fails_before_anything_is_loaded(self, stage):
        r = _run(stage, "--set", "NOT_A_KNOB=1")
        assert r.returncode != 0
        assert "NOT_A_KNOB" in r.stderr and "is not a knob" in r.stderr
        assert "Traceback" not in r.stderr
        knobs, _ = _surface(stage)
        assert ", ".join(knobs) in r.stderr     # the driver's KNOBS are the section's


@pytest.mark.parametrize("stage, flags", [("imp", ["--imp-dir"]),
                                          ("const", ["--imp-dir", "--const-dir"]),
                                          ("opt", ["--const-dir"])])
def test_each_driver_takes_the_directories_a_sweep_scratch_store_needs(stage, flags):
    """A sweep that keeps only Stage 4 points imp and const runs elsewhere (#108)."""
    r = _run(stage, "--help")
    assert r.returncode == 0, r.stderr[-500:]
    for flag in flags:
        assert flag in r.stdout


@pytest.mark.parametrize("stage, pair, knob, expected", [
    ("opt",   "TOL_GAP_REL='x'", "TOL_GAP_REL", "expected float"),
    ("imp",   "TIME_WINDOW=1.5", "TIME_WINDOW", "expected int"),
    ("const", "ANCHOR_CV='x'",   "ANCHOR_CV",   "expected float"),
])
def test_a_knob_of_the_wrong_type_fails_before_anything_is_loaded(stage, pair, knob,
                                                                  expected):
    r = _run(stage, "--set", pair)
    assert r.returncode != 0
    assert knob in r.stderr and expected in r.stderr
    assert "Traceback" not in r.stderr


class TestCascadeOrder:
    """#104: the Stage 2 cascade order is a knob; it must stay a permutation of the
    five cascade imputers, and the terminal pass (time, share, flat) stays fixed."""

    def test_the_default_order_is_todays_cascade(self):
        assert CASE.imp.CASCADE_ORDER == ("trade", "network", "conc", "ore_grade", "eff")
        assert CASE.imp.cascade_ids == CASE.imp.CASCADE_ORDER
        knobs, _ = _surface("imp")
        assert "CASCADE_ORDER" in knobs and "cascade_ids" not in knobs

    def test_the_cascade_is_built_in_the_given_order(self):
        by_id = {k: object() for k in CASE.imp.CASCADE_ORDER}
        order = ("network", "trade", "eff", "conc", "ore_grade")
        assert cascade_in_order(by_id, order) == [by_id[k] for k in order]

    @pytest.mark.parametrize("order", [
        ("trade", "network", "conc", "ore_grade", "time"),        # unknown
        ("trade", "trade", "conc", "ore_grade", "eff"),           # repeated
        ("trade", "network", "conc", "ore_grade"),                # missing
    ])
    def test_an_order_that_is_not_a_permutation_is_refused(self, order):
        with pytest.raises(ValueError, match="permutation"):
            cascade_in_order({k: object() for k in CASE.imp.CASCADE_ORDER}, order)

    def test_a_bad_order_fails_on_the_command_line_before_anything_is_loaded(self):
        r = _run("imp", "--set", 'CASCADE_ORDER=("trade","trade","conc","ore_grade","eff")')
        assert r.returncode != 0 and "permutation" in r.stderr
        assert "Traceback" not in r.stderr


def test_the_constraint_diagnostics_are_a_knob_on_by_default():
    """#108: a sweep deletes its const runs from the scratch store, so drawing their
    figures is time it switches off."""
    assert CASE.const.DIAG_PLOT is True
    assert "DIAG_PLOT" in _surface("const")[0]


def test_the_feasibility_check_knob_replaces_the_bool():
    mod = _load("opt")
    assert mod.FEASIBILITY_CHECK == "first"
    assert "FEASIBILITY_CHECK" in mod.KNOBS
    assert "FEASIBILITY_CHECK_FIRST_ONLY" not in mod.KNOBS
    assert not hasattr(mod, "FEASIBILITY_CHECK_FIRST_ONLY")
