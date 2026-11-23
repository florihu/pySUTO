"""
test_copper_modes.py
====================
The three copper mode knobs (``FLAT_MODE``, ``STOCK_BOUNDS_MODE``, ``SCRAP_SHARE_MODE``)
are free strings that ``--set`` can reach: a value outside the declared ones raises
before the hook reads any input, instead of selecting a branch (or dropping the
scrap-share band) silently (#278, U7-G3-3). The hooks run here with a stub context: the
mode check comes first, so no params data is needed.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

import copper_case  # noqa: F401  (puts scripts/ on sys.path, #189)
from copper import CASE, builders, terminal


def _ctx(**section_changes):
    section = "imp" if "FLAT_MODE" in section_changes else "const"
    case = dataclasses.replace(
        CASE, **{section: dataclasses.replace(getattr(CASE, section), **section_changes)})
    return SimpleNamespace(case=case, inputs=None, acc=None, agg_result=None,
                           logger=None, param_refs=[])


def test_an_unknown_flat_mode_raises_naming_the_knob():
    with pytest.raises(ValueError, match=r"FLAT_MODE = 'noisy_zeros' is not one of"):
        terminal(_ctx(FLAT_MODE="noisy_zeros"))


@pytest.mark.parametrize("knob, value", [
    ("STOCK_BOUNDS_MODE", "Soft"),
    ("SCRAP_SHARE_MODE",  "of"),
])
def test_an_unknown_const_mode_raises_naming_the_knob(knob, value):
    with pytest.raises(ValueError, match=rf"{knob} = '{value}' is not one of"):
        builders(_ctx(**{knob: value}))
