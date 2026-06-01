"""
mass_balance.py
===============
``MassBalanceBuilder`` — Type-1 hard equality rows: Σ inflows − Σ outflows = 0
per (balanced node × timestep).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pysuto.logging import get_logger

from .base import BaseConstraintBuilder, ConstraintBlock, _empty_block

_log = get_logger(__name__)


class MassBalanceBuilder(BaseConstraintBuilder):
    """
    Hard zero-cone constraint: Σ supply_edges − Σ use_edges = 0
    for every (Flow | Process) node × valuation that carries ≥1 incoming and
    ≥1 outgoing structural edge.

    SB nodes are open-boundary endpoints and are excluded.

    Under the structural model one row is emitted per (balanced_node,
    valuation) — never replicated per timestep. The same boolean ±1
    coefficients hold every period; the row is always active.
    """

    _BALANCE_ENTITIES = {"Flow", "Process"}

    def __init__(self, mask: str | None = None):
        self._mask_name = mask

    def build_static(self, acc_struct) -> ConstraintBlock:
        ei         = acc_struct.edge_index
        node_idx   = acc_struct.node_index
        n_edges    = acc_struct.n_edges
        valuations = list(acc_struct.valuation)

        v_arr = ei["valuation"].values

        if self._mask_name is not None:
            if self._mask_name not in acc_struct.masks:
                raise ValueError(
                    f"MassBalanceBuilder(mask={self._mask_name!r}): "
                    f"mask not in acc_struct.masks. "
                    f"Available: {sorted(acc_struct.masks)}"
                )
            scope_mask = acc_struct.masks[self._mask_name]
        else:
            scope_mask = np.ones(n_edges, dtype=bool)

        rows_list, cols_list, vals_list = [], [], []
        c_list, sigma_list, meta_list   = [], [], []
        row_id = 0

        for valuation in valuations:
            v_mask = scope_mask & (v_arr == valuation)
            # Any node with ≥1 in AND ≥1 out edge of this valuation is balanced.
            # Splitting by valuation guarantees each row is unit-coherent
            # (no Mass+CC mixing in G).
            nodes_with_incoming = set(ei.loc[v_mask, "to_id"].values)
            nodes_with_outgoing = set(ei.loc[v_mask, "from_id"].values)
            balanced = nodes_with_incoming & nodes_with_outgoing

            for nid in sorted(balanced):
                entity = str(node_idx.loc[nid, "entity"])
                if entity not in self._BALANCE_ENTITIES:
                    continue

                inc_mask = v_mask & (ei["to_id"].values == nid)
                out_mask = v_mask & (ei["from_id"].values == nid)

                inc_ids = ei.loc[inc_mask, "edge_id"].values
                out_ids = ei.loc[out_mask, "edge_id"].values

                if len(inc_ids) == 0 or len(out_ids) == 0:
                    continue

                for eid in inc_ids:
                    rows_list.append(row_id); cols_list.append(eid); vals_list.append(1.0)
                for eid in out_ids:
                    rows_list.append(row_id); cols_list.append(eid); vals_list.append(-1.0)

                c_list.append(0.0)
                sigma_list.append(0.0)
                n_row = node_idx.loc[nid]
                meta_list.append({
                    "constraint_id": f"mass_balance_node{nid}_{valuation}",
                    "type":          "mass_balance",
                    "is_hard":       True,
                    "cone_type":     "zero",
                    "valuation":     valuation,
                    "sector":        str(n_row["sector"]),
                    "region":        str(n_row["region"]),
                    "entity":        str(n_row["entity"]),
                    "entity_sub":    str(n_row.get("entity_sub", "")) if "entity_sub" in node_idx.columns else None,
                })
                row_id += 1

        if row_id == 0:
            return _empty_block(n_edges)

        G = sp.csr_matrix(
            (vals_list, (rows_list, cols_list)),
            shape=(row_id, n_edges),
        )
        _log.info("MassBalanceBuilder.build_static: %d rows (structural)", row_id)
        return ConstraintBlock(
            G_rows   = G,
            c        = np.array(c_list),
            sigma_c  = np.array(sigma_list),
            row_meta = pd.DataFrame(meta_list),
        )


# =============================================================================
# SumConstraintBuilder
# =============================================================================

