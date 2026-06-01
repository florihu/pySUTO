"""
test_flatprior_row.py
=====================
Verify ROW-specific CV inflation works through the new ``masks=`` overlay
+ pre-inflated ``cv_by_mask`` tuples pattern. ROW inflation lives in the
caller (the init script), not in ``FlatPriorImputer``.
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
                "valuation": ["Mass"] * 5,
            }
        )
        # Single timestep — keeps fixtures compact while honouring the 2-D
        # (n_edges, n_timesteps) contract introduced by the structural refactor.
        self.time_range = ["2020"]
        self.n_timeps   = 1
        self.masks = {
            "supply":   np.array([True,  True,  False, True,  False]),
            "use":      np.array([False, False, True,  False, True]),
            "tradable": np.zeros(5, dtype=bool),
            "sb":       np.zeros(5, dtype=bool),
        }


def _attach_full_coverage(imp, n_edges, n_timeps=1):
    """Attach a coverage array of all True so density ρ=1 → CV=cv_lo."""
    class _Agg:
        pass

    a = _Agg()
    a.coverage = np.ones((n_edges, n_timeps), dtype=bool)
    imp.attach_agg_result(a)


def _row_sub_masks(acc):
    """Replicate the init-script's ROW mask computation against ``_RowAcc``.

    Returns ``(row_supply, row_use)`` boolean arrays of shape ``(n_edges,)``.
    Uses node_index["region"] directly (the fixture is small enough that we
    skip ``expand_edge_index``).
    """
    region = acc.node_index["region"].to_numpy()
    region_from = region[acc.edge_index["from_id"].to_numpy()]
    region_to   = region[acc.edge_index["to_id"].to_numpy()]
    row = (region_from == "ROW") | (region_to == "ROW")
    return acc.masks["supply"] & row, acc.masks["use"] & row


def test_row_inflation_one_is_no_op():
    """row_inflation=1.0 → ROW supply edges get same CV as non-ROW supply edges."""
    acc = _RowAcc()
    f = np.full((5, 1), np.nan)
    f[0, 0] = 50.0       # finite anchor for level-2 scale group (Mining, Mass)
    sigma = np.full((5, 1), np.nan)
    sigma[0, 0] = 1.0
    missing = np.isnan(f)

    row_supply, row_use = _row_sub_masks(acc)
    inflation = 1.0
    imp = FlatPriorImputer(
        cv_by_mask={
            "row_supply": (0.1 * inflation, 0.1 * inflation),
            "row_use":    (0.2 * inflation, 0.2 * inflation),
            "supply":     (0.1, 0.1),
            "use":        (0.2, 0.2),
        },
        default_cv=0.5,
        acc_struct=acc,
        masks={"row_supply": row_supply, "row_use": row_use},
    )
    _attach_full_coverage(imp, acc.n_edges)
    imp.impute(f, sigma, missing, 0, acc)

    # Inflation=1 → sigma of ROW supply (edge 3) equals sigma of domestic supply (edge 1).
    assert sigma[3, 0] / sigma[1, 0] == pytest.approx(1.0)
    assert sigma[4, 0] / sigma[2, 0] == pytest.approx(1.0)


def test_row_inflation_multiplies_cv_for_row_edges():
    """row_inflation=2.5 → sigma[row_supply]/sigma[supply] == 2.5 within same scale group."""
    acc = _RowAcc()
    f = np.full((5, 1), np.nan)
    f[0, 0] = 50.0
    sigma = np.full((5, 1), np.nan)
    sigma[0, 0] = 1.0
    missing = np.isnan(f)

    row_supply, row_use = _row_sub_masks(acc)
    inflation = 2.5
    imp = FlatPriorImputer(
        cv_by_mask={
            "row_supply": (0.1 * inflation, 0.1 * inflation),
            "row_use":    (0.2 * inflation, 0.2 * inflation),
            "supply":     (0.1, 0.1),
            "use":        (0.2, 0.2),
        },
        default_cv=0.5,
        acc_struct=acc,
        masks={"row_supply": row_supply, "row_use": row_use},
    )
    _attach_full_coverage(imp, acc.n_edges)
    imp.impute(f, sigma, missing, 0, acc)

    # Edge 1 (supply, domestic) vs edge 3 (supply, ROW): same level-2 scale group
    # (Mining, Mass), so sigma ratio = inflation factor.
    assert sigma[3, 0] / sigma[1, 0] == pytest.approx(2.5)
    # Same comparison in the use bucket: edge 2 (use, domestic) vs edge 4 (use, ROW).
    assert sigma[4, 0] / sigma[2, 0] == pytest.approx(2.5)


def test_overlay_takes_priority_over_acc_masks():
    """Caller-supplied masks overlay names already in acc.masks."""
    acc = _RowAcc()
    # Replace supply with a different boolean via the overlay; the imputer
    # must prefer the overlay-supplied array.
    custom_supply = np.array([False, False, False, False, True])  # only edge 4
    imp = FlatPriorImputer(
        cv_by_mask={"supply": (0.1, 0.1)},
        default_cv=0.5,
        acc_struct=acc,
        masks={"supply": custom_supply},
    )
    name, mask = imp._masks[0]
    assert name == "supply"
    assert np.array_equal(mask, custom_supply)
