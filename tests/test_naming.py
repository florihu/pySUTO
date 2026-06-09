"""
test_naming.py
==============
Unit tests for the constraint-id grammar helpers in
:mod:`pysuto.constraints.naming` — the single home of the
``{type}_{variant}_{specs}_{year}`` grammar and the ``ROW_META_COLS`` schema.
"""
from __future__ import annotations

import numpy as np
import pytest

from pysuto.constraints.naming import (
    ROW_META_COLS,
    ConstraintIdError,
    build_constraint_id,
    empty_row_meta_dict,
    render_edge,
    _safe,
)


def test_build_orders_slots_and_year_last():
    cid = build_constraint_id("sum", "usefulsupply", "Cathode", "Chile", "Mass", year="1995")
    assert cid == "sum_usefulsupply_Cathode_Chile_Mass_1995"
    assert cid.split("_") == ["sum", "usefulsupply", "Cathode", "Chile", "Mass", "1995"]


def test_year_none_renders_na():
    assert build_constraint_id("sum", "mylabel", "Mass", year=None).endswith("_na")
    assert _safe(None, field="x") == "na"
    assert _safe(np.nan, field="x") == "na"


def test_render_edge_is_underscore_free():
    s = render_edge("Smelting", "Chile", "Anode", "Chile")
    assert s == "Smelting.Chile-Anode.Chile"
    assert "_" not in s


@pytest.mark.parametrize("bad_field", [
    ("type",    {"type_": "su_m", "variant": "v"}),
    ("variant", {"type_": "sum", "variant": "has_underscore"}),
    ("spec",    {"type_": "sum", "variant": "v", "specs": ("Ca_thode",)}),
])
def test_guard_rejects_underscore(bad_field):
    _name, kwargs = bad_field
    specs = kwargs.pop("specs", ())
    with pytest.raises(ConstraintIdError):
        build_constraint_id(kwargs["type_"], kwargs["variant"], *specs, year="1995")


def test_render_edge_guard_on_underscore_value():
    with pytest.raises(ConstraintIdError):
        render_edge("Sme_lting", "Chile", "Anode", "Chile")


def test_empty_row_meta_dict_matches_schema():
    d = empty_row_meta_dict()
    assert list(d.keys()) == ROW_META_COLS
    assert all(v is None for v in d.values())
