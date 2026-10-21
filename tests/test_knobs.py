"""test_knobs.py — the driver knob surface (#100): ``--set KNOB=VALUE`` parsing."""
from __future__ import annotations

import pytest

from pysuto.knobs import apply_overrides, parse_overrides, write_run_id


def _ns():
    return {"P_PROP": 1.3, "TIME_WINDOW": 3, "TIME_METHOD": "linear",
            "TIME_INTERPOLATE_ONLY": True, "UNCLOSED_CV": None,
            "STOCK_SECTORS": ("MESC", "USC"), "LEG_SIGMA_AREF": {"Anode": 0.2},
            "ACC_DIR": "data/proc/acc"}


KNOBS = ("P_PROP", "TIME_WINDOW", "TIME_METHOD", "TIME_INTERPOLATE_ONLY",
         "UNCLOSED_CV", "STOCK_SECTORS", "LEG_SIGMA_AREF")


class TestParseOverrides:

    def test_a_float_knob_takes_a_float_literal(self):
        assert parse_overrides(["P_PROP=1.5"], KNOBS, _ns()) == {"P_PROP": 1.5}

    def test_a_name_outside_knobs_is_refused_naming_the_knobs(self):
        with pytest.raises(KeyError, match="TIME_WINDOW"):
            parse_overrides(["ACC_DIR='x'"], KNOBS, _ns())

    def test_a_value_of_another_type_is_refused(self):
        with pytest.raises(TypeError, match="P_PROP"):
            parse_overrides(["P_PROP='x'"], KNOBS, _ns())
        with pytest.raises(TypeError, match="TIME_WINDOW"):
            parse_overrides(["TIME_WINDOW=3.0"], KNOBS, _ns())

    def test_an_int_is_accepted_for_a_float_knob(self):
        assert parse_overrides(["P_PROP=2"], KNOBS, _ns()) == {"P_PROP": 2}

    def test_a_bool_is_neither_an_int_nor_a_float(self):
        with pytest.raises(TypeError):
            parse_overrides(["TIME_WINDOW=True"], KNOBS, _ns())
        with pytest.raises(TypeError):
            parse_overrides(["TIME_INTERPOLATE_ONLY=1"], KNOBS, _ns())
        assert parse_overrides(["TIME_INTERPOLATE_ONLY=False"], KNOBS, _ns()) == {
            "TIME_INTERPOLATE_ONLY": False}

    def test_none_switches_any_knob_off_and_a_none_default_takes_any_value(self):
        assert parse_overrides(["P_PROP=None"], KNOBS, _ns()) == {"P_PROP": None}
        assert parse_overrides(["UNCLOSED_CV=1.0"], KNOBS, _ns()) == {"UNCLOSED_CV": 1.0}

    def test_tuples_and_dicts_round_trip_as_literals(self):
        got = parse_overrides(['STOCK_SECTORS=("USC",)',
                               'LEG_SIGMA_AREF={"Anode": 0.5}'], KNOBS, _ns())
        assert got == {"STOCK_SECTORS": ("USC",), "LEG_SIGMA_AREF": {"Anode": 0.5}}
        with pytest.raises(TypeError):
            parse_overrides(['STOCK_SECTORS=["USC"]'], KNOBS, _ns())

    def test_a_str_knob_takes_a_quoted_literal_or_a_bare_word(self):
        assert parse_overrides(['TIME_METHOD="nearest"'], KNOBS, _ns()) == {
            "TIME_METHOD": "nearest"}
        assert parse_overrides(["TIME_METHOD=nearest"], KNOBS, _ns()) == {
            "TIME_METHOD": "nearest"}
        with pytest.raises(ValueError):
            parse_overrides(["P_PROP=one"], KNOBS, _ns())

    def test_a_pair_without_equals_is_refused_and_a_later_pair_wins(self):
        with pytest.raises(ValueError):
            parse_overrides(["P_PROP"], KNOBS, _ns())
        got = parse_overrides(["P_PROP=1.1", "P_PROP=1.2"], KNOBS, _ns())
        assert got == {"P_PROP": 1.2}

    def test_a_float_repr_round_trips_exactly(self):
        x = 0.1 + 0.2
        assert parse_overrides([f"P_PROP={x!r}"], KNOBS, _ns())["P_PROP"] == x


class TestApplyAndRunIdFile:

    def test_apply_overrides_sets_the_globals_and_leaves_the_rest(self):
        ns = _ns()
        apply_overrides(ns, {"P_PROP": 1.5})
        assert ns["P_PROP"] == 1.5 and ns["TIME_WINDOW"] == 3

    def test_write_run_id_writes_the_id_and_ignores_no_path(self, tmp_path):
        f = tmp_path / "sub" / "rid.txt"
        write_run_id(str(f), "imp_20261021_120000_abcdef01")
        assert f.read_text() == "imp_20261021_120000_abcdef01\n"
        write_run_id(None, "imp_x")
