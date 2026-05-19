"""
test_flatprior_row.py
=====================
Verify the multiplicative ROW modulation in :class:`FlatPriorImputer`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.impute.flat_prior import FlatPriorImputer


class _Bounds:
    def __init__(self, n: int):
        self.l = np.zeros(n, dtype=float)
        self.u = np.full(n, 1e9, dtype=float)


class _RowAcc:
    """
    5-edge fixture with explicit ROW labelling.

    edge_id | from_id | to_id | sector_from | region_from | region_to | bucket
    0       | 0       | 1     | Mining      | A           | A         | supply, domestic (anchor)
    1       | 2       | 3     | Mining      | A           | A         | supply, domestic (imputed)
    2       | 3       | 2     | Ore         | A           | A         | use,    domestic (imputed)
    3       | 4       | 5     | Mining      | ROW         | A         | supply, ROW      (imputed)
    4       | 5       | 4     | Ore         | A           | ROW       | use,    ROW      (imputed)
    """

    def __init__(self):
        self.id = "row_test"
        self.n_edges = 5
        self.bounds = _Bounds(5)

        self.node_index = pd.DataFrame(
            {
                "sector": ["Mining", "Ore", "Mining", "Ore", "Mining", "Ore"],
                "entity": ["Process", "Flow", "Process", "Flow", "Process", "Flow"],
                "region": ["A", "A", "A", "A", "ROW", "A"],
            },
            index=[0, 1, 2, 3, 4, 5],
        )
        self.edge_index = pd.DataFrame(
            {
                "edge_id":   [0, 1, 2, 3, 4],
                "from_id":   [0, 2, 3, 4, 5],
                "to_id":     [1, 3, 2, 5, 4],
                "time":      ["2020"] * 5,
                "valuation": ["Mass"] * 5,
            }
        )
        self.masks = {
            "supply":   np.array([True,  True,  False, True,  False]),
            "use":      np.array([False, False, True,  False, True]),
            "tradable": np.zeros(5, dtype=bool),
            "sb":       np.zeros(5, dtype=bool),
            "row":      np.array([False, False, False, True,  True]),
        }


def _attach_full_coverage(imp, n_edges):
    """Attach a coverage array of all True so density ρ=1 → CV=cv_lo."""
    class _Agg:
        pass

    a = _Agg()
    a.coverage = np.ones(n_edges, dtype=bool)
    imp.attach_agg_result(a)


def test_row_factor_one_is_no_op():
    acc = _RowAcc()
    f = np.full(5, np.nan)
    f[0] = 50.0       # finite anchor for level-2 scale group (Mining, Mass)
    sigma = np.full(5, np.nan)
    sigma[0] = 1.0
    missing = np.isnan(f)

    imp = FlatPriorImputer(
        cv_by_mask={"supply": 0.1, "use": 0.2},
        default_cv=0.5,
        acc_struct=acc,
        row_cv_factor=1.0,
    )
    _attach_full_coverage(imp, acc.n_edges)
    imp.impute(f, sigma, missing, 0, acc)

    # With factor=1.0, sigma of ROW supply (edge 3) should equal sigma of
    # domestic-imputed supply (edge 1) — both fall through to level-2 scale.
    assert sigma[3] / sigma[1] == pytest.approx(1.0)
    assert sigma[4] / sigma[2] == pytest.approx(1.0)


def test_row_factor_multiplies_cv_for_row_edges():
    acc = _RowAcc()
    f = np.full(5, np.nan)
    f[0] = 50.0
    sigma = np.full(5, np.nan)
    sigma[0] = 1.0
    missing = np.isnan(f)

    imp = FlatPriorImputer(
        cv_by_mask={"supply": 0.1, "use": 0.2},
        default_cv=0.5,
        acc_struct=acc,
        row_mask_name="row",
        row_cv_factor=2.5,
    )
    _attach_full_coverage(imp, acc.n_edges)
    imp.impute(f, sigma, missing, 0, acc)

    # Edge 1 (supply, domestic, imputed) vs Edge 3 (supply, ROW, imputed):
    # same level-2 scale group (Mining, Mass), so sigma ratio = ROW factor.
    assert sigma[3] / sigma[1] == pytest.approx(2.5)
    # Same comparison in the use bucket: edge 2 (use, domestic) vs edge 4 (use, ROW).
    # Both use level-3 global fallback (no finite |f| in use sector_from group).
    assert sigma[4] / sigma[2] == pytest.approx(2.5)


def test_row_factor_applies_to_default():
    """Edges that match no listed mask still get the ROW factor."""
    acc = _RowAcc()
    f = np.full(5, np.nan)
    f[0] = 50.0
    sigma = np.full(5, np.nan)
    sigma[0] = 1.0
    missing = np.isnan(f)

    imp = FlatPriorImputer(
        cv_by_mask={},            # no buckets → everything is default
        default_cv=0.4,
        acc_struct=acc,
        row_mask_name="row",
        row_cv_factor=3.0,
    )
    _attach_full_coverage(imp, acc.n_edges)
    imp.impute(f, sigma, missing, 0, acc)

    assert sigma[3] / sigma[1] == pytest.approx(3.0)
    assert sigma[4] / sigma[2] == pytest.approx(3.0)


def test_missing_row_mask_disables_modulation():
    """If row_mask_name is set but not present, modulation is skipped."""
    acc = _RowAcc()
    del acc.masks["row"]

    imp = FlatPriorImputer(
        cv_by_mask={"supply": 0.1, "use": 0.2},
        default_cv=0.5,
        acc_struct=acc,
        row_mask_name="row",
        row_cv_factor=2.0,
    )
    assert imp._row_mask is None
