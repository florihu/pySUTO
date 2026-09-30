"""
flow_bound.py
=============
``FlowBoundsBuilder`` — enforced lower bounds (ADR 0039): one non-negative-cone
row ``−a_i ≤ −l_i`` per edge whose *structural* lower bound (the accounting
structure's ``bounds.l``) is 0. Upper bounds and the Stage-3 flat-prior zero pin
are not rows; they stay on ``ConstraintRun.u / .l`` for the feasibility probe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pysuto.logging import get_logger

from .base import BaseConstraintBuilder, ConstraintBlock, _empty_block
from .naming import ROW_META_COLS, TYPE_BOUND, build_constraint_id, render_edge

_log = get_logger(__name__)


class FlowBoundsBuilder(BaseConstraintBuilder):
    """
    Hard lower bounds from the accounting structure (ADR 0039).

    Emits one non-negative-cone row ``−a_i ≤ −l_i`` for every edge whose
    structural lower bound ``acc_struct.bounds.l`` is 0 — non-negativity of every
    flow except the signed stock edges, whose ``l`` is negative and needs no row.
    Rows are hard (``σ_c = 0``), year-local (the edge's own year) and always
    active. Reading the structural bounds, not the pinned ``ConstraintRun.l``,
    keeps the zero pin out of the QP and leaves no-data stock changes signed.

    ``enforce_lower=False`` restores the pre-ADR-0039 no-op (bounds stored only).
    """

    def __init__(self, enforce_lower: bool = True):
        self._enforce_lower = bool(enforce_lower)

    def build_static(self, acc_struct) -> ConstraintBlock:
        n = acc_struct.n_edges
        if not self._enforce_lower:
            _log.info("FlowBoundsBuilder: lower bounds not enforced (stored in l/u only)")
            return _empty_block(n)
        ids = np.flatnonzero(acc_struct.bounds.l == 0.0)
        if ids.size == 0:
            return _empty_block(n)
        em  = acc_struct.edge_meta.iloc[ids]
        fs, fr = em["from_sector"].astype(str).to_numpy(), em["from_region"].astype(str).to_numpy()
        ts, tr = em["to_sector"].astype(str).to_numpy(),   em["to_region"].astype(str).to_numpy()
        val, tim = em["valuation"].astype(str).to_numpy(), em["time"].astype(str).to_numpy()
        m = ids.size
        row_meta = pd.DataFrame({c: [None] * m for c in ROW_META_COLS})
        row_meta["constraint_id"] = [
            build_constraint_id(TYPE_BOUND, "lower", render_edge(*e), v, year=t)
            for *e, v, t in zip(fs, fr, ts, tr, val, tim)
        ]
        row_meta["type"]      = TYPE_BOUND
        row_meta["variant"]   = "lower"
        row_meta["is_hard"]   = True
        row_meta["cone_type"] = "nonneg"
        row_meta["time"]      = tim
        row_meta["valuation"] = val
        row_meta["bound_dir"] = "lower"
        row_meta["num_sector_from"], row_meta["num_region_from"] = fs, fr
        row_meta["num_sector_to"],   row_meta["num_region_to"]   = ts, tr
        row_meta["num_valuation"] = val
        G = sp.csr_matrix((-np.ones(m), (np.arange(m), ids)), shape=(m, n))
        _log.info("FlowBoundsBuilder: %d lower-bound rows (structural l = 0); "
                  "%d signed edges without a row", m, n - m)
        return ConstraintBlock(
            G_rows   = G,
            c        = -acc_struct.bounds.l[ids].astype(float),
            sigma_c  = np.zeros(m),
            row_meta = row_meta,
        )


# =============================================================================
# ConstraintRun  — output of ConstraintEngine.run()
# =============================================================================

