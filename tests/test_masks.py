"""
test_masks.py
=============
Cover mask resolution, the AccountingStructure ``masks`` dict, and the
FlatPrior ROW multiplicative modulation.
"""

from __future__ import annotations

import logging
import textwrap

import numpy as np
import pandas as pd
import pytest

from pysuto.masks import MaskRuleSet, coerce_mask, load_masks_csv


# ---------------------------------------------------------------------------
# Toy edge/node index fixture
# ---------------------------------------------------------------------------

def _make_toy_indices():
    """5 nodes, 6 edges. Mixed regions including 'ROW'."""
    node_index = pd.DataFrame(
        [
            ("Mining", "Process", "A"),     # 0
            ("Ore",    "Flow",    "A"),     # 1
            ("Mining", "Process", "B"),     # 2
            ("Ore",    "Flow",    "B"),     # 3
            ("Ore",    "Flow",    "ROW"),   # 4
        ],
        columns=["sector", "entity", "region"],
    )

    edge_rows = [
        # edge_id, from_id, to_id, time, valuation
        (0, 0, 1, "2020", "Mass"),  # supply Mining/A → Ore/A           domestic
        (1, 2, 3, "2020", "Mass"),  # supply Mining/B → Ore/B           domestic
        (2, 1, 0, "2020", "Mass"),  # use    Ore/A   → Mining/A         domestic
        (3, 4, 0, "2020", "Mass"),  # use    Ore/ROW → Mining/A         ROW (region_from=ROW)
        (4, 2, 4, "2020", "Mass"),  # supply Mining/B → Ore/ROW         ROW (region_to=ROW)
        (5, 4, 2, "2020", "Mass"),  # use    Ore/ROW → Mining/B         ROW (region_from=ROW)
    ]
    edge_index = pd.DataFrame(edge_rows,
        columns=["edge_id", "from_id", "to_id", "time", "valuation"])
    return node_index, edge_index


# ---------------------------------------------------------------------------
# MaskRuleSet.from_csv + resolve
# ---------------------------------------------------------------------------

def test_resolve_row_mask_via_csv(tmp_path):
    node_index, edge_index = _make_toy_indices()

    csv = tmp_path / "masks.csv"
    csv.write_text(textwrap.dedent("""\
        mask_name,logic,sector_from,sector_to,region_from,region_to,valuation,time,entity_from,entity_to
        row,include,*,*,ROW,*,*,*,*,*
        row,include,*,*,*,ROW,*,*,*,*
    """))

    masks = load_masks_csv(csv, edge_index, node_index)
    assert set(masks) == {"row"}
    expected = np.array([False, False, False, True, True, True])
    assert np.array_equal(masks["row"], expected)


def test_include_then_exclude(tmp_path):
    node_index, edge_index = _make_toy_indices()

    csv = tmp_path / "masks.csv"
    csv.write_text(textwrap.dedent("""\
        mask_name,logic,sector_from,sector_to,region_from,region_to,valuation,time,entity_from,entity_to
        domestic,include,*,*,*,*,*,*,*,*
        domestic,exclude,*,*,ROW,*,*,*,*,*
        domestic,exclude,*,*,*,ROW,*,*,*,*
    """))

    masks = load_masks_csv(csv, edge_index, node_index)
    expected = np.array([True, True, True, False, False, False])
    assert np.array_equal(masks["domestic"], expected)


def test_missing_required_columns_raises(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("foo,bar\n1,2\n")

    with pytest.raises(ValueError, match="mask_name"):
        MaskRuleSet.from_csv(csv)


def test_unknown_logic_raises(tmp_path):
    csv = tmp_path / "bad_logic.csv"
    csv.write_text("mask_name,logic\nrow,intersect\n")

    with pytest.raises(ValueError, match="include.*exclude"):
        MaskRuleSet.from_csv(csv)


def test_partial_columns_treated_as_wildcards(tmp_path):
    """Omitting selector columns is the same as filling them with '*'."""
    node_index, edge_index = _make_toy_indices()

    csv = tmp_path / "masks.csv"
    csv.write_text(textwrap.dedent("""\
        mask_name,logic,region_from
        row_from,include,ROW
    """))

    masks = load_masks_csv(csv, edge_index, node_index)
    expected = np.array([False, False, False, True, False, True])
    assert np.array_equal(masks["row_from"], expected)


# ---------------------------------------------------------------------------
# coerce_mask: array + callable inputs
# ---------------------------------------------------------------------------

def test_coerce_mask_accepts_array():
    node_index, edge_index = _make_toy_indices()
    arr = np.array([1, 0, 1, 0, 1, 0], dtype=bool)
    out = coerce_mask("custom", arr, edge_index, node_index, n_edges=6)
    assert out.dtype == bool
    assert np.array_equal(out, arr)


def test_coerce_mask_accepts_callable():
    node_index, edge_index = _make_toy_indices()

    def predicate(ei_exp: pd.DataFrame) -> np.ndarray:
        return ei_exp["region_to"].values == "ROW"

    out = coerce_mask("row_to", predicate, edge_index, node_index, n_edges=6)
    expected = np.array([False, False, False, False, True, False])
    assert np.array_equal(out, expected)


def test_coerce_mask_rejects_wrong_shape():
    node_index, edge_index = _make_toy_indices()
    arr = np.zeros(3, dtype=bool)
    with pytest.raises(ValueError, match=r"shape"):
        coerce_mask("bad", arr, edge_index, node_index, n_edges=6)


# ---------------------------------------------------------------------------
# AccountingStructure.register_mask + apply_mask_overlay
# ---------------------------------------------------------------------------

def _make_acc_struct_with_masks():
    """Return a minimal AccountingStructure-like object compatible with
    register_mask / apply_mask_overlay."""
    from pysuto.acc import AccountingStructure

    node_index, edge_index = _make_toy_indices()
    acc = AccountingStructure.__new__(AccountingStructure)
    acc.logger = logging.getLogger("test_acc")
    acc.id = "toy"
    acc.node_index = node_index
    acc.edge_index = edge_index
    acc.n_edges = len(edge_index)
    acc.masks = {
        "supply":   np.array([True, True, False, False, True, False]),
        "use":      np.array([False, False, True, True, False, True]),
        "tradable": np.zeros(6, dtype=bool),
        "sb":       np.zeros(6, dtype=bool),
    }
    return acc


def test_register_mask_array():
    acc = _make_acc_struct_with_masks()
    arr = np.array([1, 0, 0, 0, 1, 0], dtype=bool)
    acc.register_mask("custom", arr)
    assert "custom" in acc.masks
    assert np.array_equal(acc.masks["custom"], arr)


def test_register_mask_predicate():
    acc = _make_acc_struct_with_masks()
    acc.register_mask("row", lambda ei: ei["region_from"].values == "ROW")
    expected = np.array([False, False, False, True, False, True])
    assert np.array_equal(acc.masks["row"], expected)


def test_register_mask_overwrites(caplog):
    acc = _make_acc_struct_with_masks()
    with caplog.at_level(logging.WARNING, logger="test_acc"):
        acc.register_mask("supply", np.ones(6, dtype=bool))
    assert any("overwriting" in r.message for r in caplog.records)
    assert acc.masks["supply"].all()


def test_apply_mask_overlay_overrides(tmp_path):
    """Separate-kind overlay wins on name collision."""
    acc = _make_acc_struct_with_masks()
    acc.masks["row"] = np.zeros(6, dtype=bool)  # acc-local placeholder

    csv = tmp_path / "rules.csv"
    csv.write_text(textwrap.dedent("""\
        mask_name,logic,region_from,region_to
        row,include,ROW,*
        row,include,*,ROW
    """))

    acc.apply_mask_overlay(str(csv))
    expected = np.array([False, False, False, True, True, True])
    assert np.array_equal(acc.masks["row"], expected)
