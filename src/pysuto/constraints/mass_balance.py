"""
mass_balance.py
===============
``MassBalanceBuilder`` — Type-1 hard equality rows: Σ inflows − Σ outflows = 0
per (balanced node × timestep). The rows are :class:`~pysuto.relation.relation.Relation`
objects built by :func:`pysuto.relation.balance.node_balance_relations` and emitted by
:func:`pysuto.constraints.emit.emit`.
"""

from __future__ import annotations

import numpy as np

from pysuto.logging import get_logger
from pysuto.relation.balance import node_balance_relations

from .base import BaseConstraintBuilder, ConstraintBlock
from .emit import emit

_log = get_logger(__name__)


class MassBalanceBuilder(BaseConstraintBuilder):
    """
    Hard equality constraint: Σ supply_edges − Σ use_edges = 0
    for every (Flow | Process) node that has ≥1 supply edge AND ≥1 use edge,
    per timestep.

    SB nodes are open-boundary endpoints and are excluded.
    G is time-invariant: one row per (balanced_node, time) pair.
    """

    def __init__(self, mask: str | None = None):
        """
        Parameters
        ----------
        mask : str | None
            If set, restrict mass-balance summation to edges where
            ``acc_struct.masks[mask] == True``. Nodes that no longer have at
            least one incoming AND one outgoing in-scope edge are skipped.
            Raises ``ValueError`` at build time if the mask is not present
            in ``acc_struct.masks``.
        """
        self._mask_name = mask

    def build_static(self, acc_struct) -> ConstraintBlock:
        n_edges = acc_struct.n_edges

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

        # One relation per (time, valuation, node). Splitting by valuation keeps each
        # emitted row unit-coherent (no Mass+CC mixing in G). ``node_balance_relations``
        # sorts by (node, time), so regroup by time in the structure's period order,
        # nodes ascending within a period.
        by_valuation = {}
        for valuation in acc_struct.valuation:
            by_time = {}
            for rel in node_balance_relations(acc_struct, valuation, scope_mask):
                by_time.setdefault(rel.time, []).append(rel)
            by_valuation[valuation] = by_time

        relations = []
        for time in acc_struct.time_range:
            for valuation in acc_struct.valuation:
                relations.extend(by_valuation[valuation].get(str(time), ()))

        _log.info("MassBalanceBuilder.build_static: %d rows", len(relations))
        return emit(relations, n_edges)
