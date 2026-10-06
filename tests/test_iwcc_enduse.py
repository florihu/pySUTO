"""Unit tests for the IWCC end-use payload (scripts/explo/iwcc_enduse.py, #45).

A toy workbook in the 2025-release layout of sheet "Copper Content" -- rows 2-4
group/subgroup/leaf headers (with the source's broken words), row 5 the
descriptions with "Total" in column 17, data from row 7 -- goes in; the page's
``enduse`` payload comes out.
"""
import json
import os
import sys

import numpy as np
import pytest
from openpyxl import Workbook

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "explo"))
from iwcc_enduse import build_payload  # noqa: E402

YEARS  = [2012, 2013, 2014]
# leaf j in year i: one-decimal values, so the 0.1 kt rounding is the identity
LEAVES = [[100.0 * (j + 1) + 10.0 * i + 0.5 for j in range(16)] for i in range(3)]

GROUPS = ["Building construction", "Infrastructure", "Equipment — industrial",
          "Equipment — transport", "Equipment — other", "Diverse"]
CATEGORIES = [
    ("Plumbing", "Building construction"), ("Building plant", "Building construction"),
    ("Architecture", "Building construction"), ("Communications", "Building construction"),
    ("Electrical power", "Building construction"),
    ("Power utility", "Infrastructure"), ("Telecommunications", "Infrastructure"),
    ("Electrical", "Equipment — industrial"), ("Non-electrical", "Equipment — industrial"),
    ("Automotive electrical", "Equipment — transport"),
    ("Automotive non-electrical", "Equipment — transport"),
    ("Other vehicles", "Equipment — transport"),
    ("Consumer & general products", "Equipment — other"),
    ("Cooling equipment", "Equipment — other"),
    ("Electronic equipment", "Equipment — other"),
    ("Others", "Diverse"),
]


def _rows(years=YEARS, leaves=LEAVES, totals=None):
    """The sheet as a list of 18-cell rows, header quirks of the source included."""
    totals = [sum(r) for r in leaves] if totals is None else totals
    blank = [None] * 18
    head  = ["WORLD TOTAL", None, None, None, None,
             "Basis: Copper Content (thousand tonnes)"] + [None] * 12
    group = [" ", "Building Construction", None, None, None, None, "Infrastructure",
             None, "Equipment Manufacture"] + [None] * 9
    sub   = [None] * 8 + ["Industrial ", None, "Transport", None, None,
                          "Other Equipment", None, None, None, None]
    leaf  = [None, "Plumbing", "Building Plant", "Architecture", "Commun ications",
             "Electrical Power", "Power Utility", "Telecom munications", "Electrical",
             "Non Electrical", "Automotive Electrical", "Automotive Non Electrical",
             "Other Vehicles", "Consumer and general products", "Cooling equipment",
             "Electronic equipment", "Diverse", " "]
    desc  = [None] * 16 + ["Others", "Total"]
    data  = [[y] + list(v) + [t] for y, v, t in zip(years, leaves, totals)]
    return [head, blank, group, sub, leaf, desc, blank] + data


def _write(tmp_path, rows):
    wb = Workbook()
    ws = wb.active
    ws.title = "Copper Content"
    for r in rows:
        ws.append(r)
    path = tmp_path / "End-Use-Summary-Copper-content.xlsx"
    wb.save(path)
    return path


def test_toy_sheet_round_trips_to_the_payload(tmp_path):
    p = build_payload(_write(tmp_path, _rows()))

    assert p["source"] == "IWCC end-use summary, 2025 release, copper-content basis"
    assert p["years"] == [2012, 2013, 2014]
    assert all(type(y) is int for y in p["years"])
    assert p["groups"] == GROUPS
    assert [(c["label"], c["group"]) for c in p["categories"]] == CATEGORIES

    kt = np.array([c["kt"] for c in p["categories"]])
    assert kt[0].tolist() == [100.5, 110.5, 120.5]          # Plumbing
    assert kt[15].tolist() == [1600.5, 1610.5, 1620.5]      # Others
    assert np.allclose(kt, np.array(LEAVES).T, rtol=1e-12)
    assert np.allclose(p["total"], [13608.0, 13768.0, 13928.0], rtol=1e-12)
    assert np.allclose((kt / np.array(p["total"])).sum(axis=0), 1.0, rtol=1e-12)
    assert json.loads(json.dumps(p)) == p                    # the page reads it as JSON


def test_leaves_that_do_not_sum_to_total_raise(tmp_path):
    rows = _rows(totals=[13608.0, 13768.0 + 5.0, 13928.0])
    with pytest.raises(ValueError, match=r"do not sum to Total in \[2013\]"):
        build_payload(_write(tmp_path, rows))


def test_a_missing_category_column_raises(tmp_path):
    rows = [r[:9] + r[10:] for r in _rows()]                 # "Non Electrical" gone
    with pytest.raises(ValueError, match="expected 18 columns"):
        build_payload(_write(tmp_path, rows))


@pytest.mark.parametrize("cell", ["2013e", 2013.5, None])
def test_a_non_integer_year_cell_raises(tmp_path, cell):
    rows = _rows()
    rows[8][0] = cell
    with pytest.raises(ValueError, match="non-year cells"):
        build_payload(_write(tmp_path, rows))


def test_swapped_category_columns_raise(tmp_path):
    rows = [r[:6] + [r[7], r[6]] + r[8:] for r in _rows()]   # Telecom before Power Utility
    with pytest.raises(ValueError, match="category header row"):
        build_payload(_write(tmp_path, rows))


def test_a_header_block_shifted_down_one_row_raises(tmp_path):
    rows = _rows()
    rows.insert(0, [None] * 18)
    with pytest.raises(ValueError, match="category header row"):
        build_payload(_write(tmp_path, rows))


def test_values_are_rounded_to_a_tenth_of_a_kt(tmp_path):
    leaves = [list(r) for r in LEAVES]
    leaves[0][0], leaves[0][1] = 100.53, 200.46                      # Total 13607.99
    p = build_payload(_write(tmp_path, _rows(leaves=leaves)))
    assert p["categories"][0]["kt"][0] == 100.5
    assert p["categories"][1]["kt"][0] == 200.5
    assert p["total"][0] == 13608.0
