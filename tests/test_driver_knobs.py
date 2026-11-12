"""test_driver_knobs.py — each Stage 2–4 driver's knob surface (#100).

The drivers are imported by path under a private module name (import-safe: their
``main()`` runs only under ``__main__``) and run as subprocesses for the CLI checks.
The Stage 2 and 3 knobs are fields of the copper case's sections (#120); Stage 4's are
its driver's module globals.
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
# The stages whose knobs are the case spec's fields (#120).
CASE_STAGES = ("imp", "const")


def _load(stage):
    script = REPO / "scripts" / DRIVERS[stage][0]
    spec = importlib.util.spec_from_file_location(f"_knobs_driver_{stage}", script)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(mod)
    return mod


def _defaults(stage, mod) -> dict:
    """Every knob of the stage at its default: the copper case section's fields for
    Stages 2 and 3, the driver's module globals for Stage 4."""
    if stage in CASE_STAGES:
        return getattr(mod.CASE, stage).knobs()
    return {k: getattr(mod, k) for k in mod.KNOBS}


def _run(stage, *extra):
    script = REPO / "scripts" / DRIVERS[stage][0]
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.run([sys.executable, str(script), *DRIVERS[stage][1], *extra],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("stage", sorted(DRIVERS))
class TestDriverKnobs:

    def test_every_knob_is_an_upper_case_name_with_a_default(self, stage):
        mod = _load(stage)
        defaults = _defaults(stage, mod)
        assert len(set(mod.KNOBS)) == len(mod.KNOBS)
        assert set(mod.KNOBS) == set(defaults)
        for name in mod.KNOBS:
            assert name.isupper(), name

    def test_paths_and_the_acc_id_are_not_knobs(self, stage):
        mod = _load(stage)
        assert not [k for k in mod.KNOBS if k.endswith("_DIR") or k == "ACC_ID"]

    def test_every_default_round_trips_through_its_repr(self, stage):
        mod = _load(stage)
        ns = _defaults(stage, mod)
        pairs = [f"{k}={ns[k]!r}" for k in mod.KNOBS]
        assert parse_overrides(pairs, mod.KNOBS, ns) == {k: ns[k] for k in mod.KNOBS}

    def test_every_knob_value_can_be_recorded_in_the_sidecar(self, stage):
        """stage_config["overrides"] goes through the registries' JSON encoder."""
        import json
        from pysuto.registry import _NpEncoder
        mod = _load(stage)
        ns = _defaults(stage, mod)
        back = json.loads(json.dumps({k: ns[k] for k in mod.KNOBS}, cls=_NpEncoder))
        assert set(back) == set(mod.KNOBS)

    def test_an_unknown_knob_fails_before_anything_is_loaded(self, stage):
        r = _run(stage, "--set", "NOT_A_KNOB=1")
        assert r.returncode != 0
        assert "NOT_A_KNOB" in r.stderr and "is not a knob" in r.stderr
        assert "Traceback" not in r.stderr


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


# Today's knob names (#100, #104, #108); the case spec (#120) must keep the set.
KNOB_NAMES = {
    "imp": {
        "P_PROP", "Q_REG", "ALPHA_MIN_REG", "CASCADE_MAX_ITER", "CASCADE_ORDER",
        "TIME_WINDOW", "TIME_MIN_OBS", "TIME_GAP_INFLATION", "TIME_METHOD",
        "TIME_INTERPOLATE_ONLY", "TIME_SCOPE_SAME_REGION",
        "STOCK_SEED_CV", "STOCK_SEED_ABS_FLOOR_KT", "EFFLESS_STOCK_SIGMA_KT",
        "FLAT_MODE", "FLAT_SIGMA_HIGH", "FLAT_DEFAULT_CV",
        "FLAT_CV_LO_SUPPLY", "FLAT_CV_HI_SUPPLY", "FLAT_CV_LO_USE", "FLAT_CV_HI_USE",
        "OUTLIER_LEG_K", "OUTLIER_LEG_CV", "UNCLOSED_SECTORS", "UNCLOSED_CV",
        "LEG_SIGMA_AREF", "LEG_SIGMA_GAMMA", "LEG_SIGMA_CV_CAP",
    },
    "const": {
        "P_PROP", "SIGMA_A_CV_MIN", "SIGMA_A_ABS_FLOOR", "ANCHOR_CV", "SIGMA_C_CV_MIN",
        "FABRICATED_ANCHOR_CV", "FEED_SHARE_OBSERVED", "UNCLOSED_SECTORS",
        "UNCLOSED_ANCHOR_CV", "SIGMA_C_ABS_FLOOR", "SCALE_TOTAL_ABS_FLOOR", "PRIOR_BOX_K",
        "NODATA_ANCHOR_CV", "NODATA_ANCHOR_EXCLUDE", "NODATA_ANCHOR_DATA_AWARE",
        "PIN_FLAT_PRIOR_ZEROS", "RATIO_SIGMA_REL_FLOOR", "RATIO_SCALE_ABS_FLOOR",
        "STOCK_BOUNDS_SCALE_ABS_FLOOR", "STOCK_BOUNDS_MODE", "SCRAP_SHARE_MODE",
        "SCRAP_LEVEL_ANCHOR", "MASSBAL_MASK", "ENFORCE_LOWER_BOUNDS",
        "STRIPPING_RATIO_HARD", "STOCK_BOUNDS_SKIP_REPORTED", "DIAG_PLOT",
    },
}


@pytest.mark.parametrize("stage", CASE_STAGES)
class TestCaseSpecKnobs:
    """#120: the Stage 2 and 3 knobs are the copper case section's UPPER_CASE fields;
    the drivers keep none of them."""

    def test_the_knob_names_are_unchanged(self, stage):
        assert set(_load(stage).KNOBS) == KNOB_NAMES[stage]

    def test_the_driver_keeps_no_knob_value(self, stage):
        mod = _load(stage)
        assert [k for k in mod.KNOBS if hasattr(mod, k)] == []

    def test_the_drivers_knobs_are_the_case_sections(self, stage):
        mod = _load(stage)
        assert mod.KNOBS == tuple(getattr(mod.CASE, stage).knobs())


class TestCascadeOrder:
    """#104: the Stage 2 cascade order is a knob; it must stay a permutation of the
    five cascade imputers, and the terminal pass (time, share, flat) stays fixed."""

    def test_the_default_order_is_todays_cascade(self):
        mod = _load("imp")
        assert mod.CASE.imp.CASCADE_ORDER == ("trade", "network", "conc", "ore_grade",
                                              "eff")
        assert mod.CASE.imp.cascade_ids == mod.CASE.imp.CASCADE_ORDER
        assert "CASCADE_ORDER" in mod.KNOBS and "cascade_ids" not in mod.KNOBS

    def test_the_cascade_is_built_in_the_given_order(self):
        mod = _load("imp")
        by_id = {k: object() for k in mod.CASE.imp.CASCADE_ORDER}
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
            mod.cascade_in_order({k: object() for k in mod.CASE.imp.CASCADE_ORDER}, order)

    def test_a_bad_order_fails_on_the_command_line_before_anything_is_loaded(self):
        r = _run("imp", "--set", 'CASCADE_ORDER=("trade","trade","conc","ore_grade","eff")')
        assert r.returncode != 0 and "permutation" in r.stderr
        assert "Traceback" not in r.stderr


def test_the_constraint_diagnostics_are_a_knob_on_by_default():
    """#108: a sweep deletes its const runs from the scratch store, so drawing their
    figures is time it switches off."""
    mod = _load("const")
    assert mod.CASE.const.DIAG_PLOT is True
    assert "DIAG_PLOT" in mod.KNOBS


def test_the_feasibility_check_knob_replaces_the_bool():
    mod = _load("opt")
    assert mod.FEASIBILITY_CHECK == "first"
    assert "FEASIBILITY_CHECK" in mod.KNOBS
    assert "FEASIBILITY_CHECK_FIRST_ONLY" not in mod.KNOBS
    assert not hasattr(mod, "FEASIBILITY_CHECK_FIRST_ONLY")
