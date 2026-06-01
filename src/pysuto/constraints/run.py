"""
run.py
======
``ConstraintRun`` — the persisted output of one :class:`ConstraintEngine`
run under the structural model. Saved to disk by
:class:`ConstraintRunRegistry` (in :mod:`pysuto.registry`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .struct_g import StructG


@dataclass
class ConstraintRun:
    """
    Full output of one ConstraintEngine.run() call.

    G          : :class:`StructG` — structurally-invariant sparsity pattern
                 plus a per-period coefficient tape of shape
                 ``(nnz, n_timesteps)``.
    row_meta   : DataFrame, one row per G row (constraint_id, type, is_hard,
                 cone_type, ...). No ``time`` column under the structural
                 model — per-period applicability lives in ``active``.
    a0         : (n_edges,)            — structural warm-start seed
                 (= ``a_agg[:, 0]``; the 1995 anchor).
    sigma_a    : (n_edges, n_timesteps) — per-period prior σ.
    u, l       : (n_edges,)            — flow upper / lower bounds.
    c          : (n_constraints, n_timesteps) — per-period RHS.
    sigma_c    : (n_constraints, n_timesteps) — per-period row σ
                 (always 0 on hard rows).
    active     : (n_constraints, n_timesteps) bool — per-period activation
                 mask. Hard rows: when False the engine drops the row from
                 the QP at that t. Free rows: when False the row is
                 excluded from the objective at that t.
    timesteps  : list[str] — the labels solved for, in order. Indexed by
                 the second axis of ``c`` / ``sigma_c`` / ``sigma_a`` /
                 ``active`` / ``G.data``.
    run_id, acc_id, agg_id, created_at, label : provenance.
    """
    G:          StructG
    row_meta:   pd.DataFrame
    a0:         np.ndarray
    sigma_a:    np.ndarray
    u:          np.ndarray
    l:          np.ndarray
    c:          np.ndarray
    sigma_c:    np.ndarray
    active:     np.ndarray
    timesteps:  list
    run_id:     str
    acc_id:     str
    agg_id:     str
    created_at: str
    label:      str = ""
    _p:         float = 1.3

    @property
    def n_edges(self) -> int:
        return int(self.G.n_edges)

    @property
    def n_constraints(self) -> int:
        return int(self.G.n_constraints)

    @property
    def n_timesteps(self) -> int:
        return int(self.G.n_timesteps)
