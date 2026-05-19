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
    for every (Flow | Process) node that has ≥1 supply edge AND ≥1 use edge,
    per timestep.

    SB nodes are open-boundary endpoints and are excluded.
    G is time-invariant: one row per (balanced_node, time) pair.
    """

    _BALANCE_ENTITIES = {"Flow", "Process"}

    def build_static(self, acc_struct) -> ConstraintBlock:
        ei        = acc_struct.edge_index
        node_idx  = acc_struct.node_index
        n_edges   = acc_struct.n_edges
        times     = acc_struct.time_range

        # Expand edge_index with decoded node attributes
        ni = node_idx.loc[ei["from_id"].values]
        nj = node_idx.loc[ei["to_id"].values]
        ef_arr = ni["entity"].values
        et_arr = nj["entity"].values
        sf_arr = acc_struct.supply_edge_mask
        uf_arr = acc_struct.use_edge_mask
        t_arr  = ei["time"].values

        rows_list, cols_list, vals_list = [], [], []
        c_list, sigma_list, meta_list   = [], [], []
        row_id = 0

        for time in times:
            t_mask = (t_arr == str(time))
            # Any node that has at least one incoming AND one outgoing edge is balanced.
            # Incoming: edges where to_id == n  (supply edges for Flow nodes,
            #                                    use edges for Process nodes)
            # Outgoing: edges where from_id == n (supply edges for Process nodes,
            #                                     use edges for Flow nodes)
            nodes_with_incoming = set(ei.loc[t_mask, "to_id"].values)
            nodes_with_outgoing = set(ei.loc[t_mask, "from_id"].values)
            balanced = nodes_with_incoming & nodes_with_outgoing

            for nid in sorted(balanced):
                entity = str(node_idx.loc[nid, "entity"])
                if entity not in self._BALANCE_ENTITIES:
                    continue

                # incoming edges: to_id == nid  → coefficient +1
                inc_mask = t_mask & (ei["to_id"].values == nid)
                # outgoing edges: from_id == nid → coefficient −1
                out_mask = t_mask & (ei["from_id"].values == nid)

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
                meta_list.append({
                    "constraint_id": f"mass_balance_node{nid}_t{time}",
                    "type": "mass_balance",
                    "is_hard": True,
                    "cone_type": "zero",
                    "time":      str(time),
                })
                row_id += 1

        if row_id == 0:
            return _empty_block(n_edges)

        G = sp.csr_matrix(
            (vals_list, (rows_list, cols_list)),
            shape=(row_id, n_edges),
        )
        _log.info("MassBalanceBuilder.build_static: %d rows", row_id)
        return ConstraintBlock(
            G_rows   = G,
            c        = np.array(c_list),
            sigma_c  = np.array(sigma_list),
            row_meta = pd.DataFrame(meta_list),
        )


# =============================================================================
# SumConstraintBuilder
# =============================================================================

