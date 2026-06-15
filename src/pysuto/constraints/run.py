"""
run.py
======
``ConstraintRun`` — the persisted output of one :class:`ConstraintEngine`
run. Saved to disk by :class:`ConstraintRunRegistry`
(in :mod:`pysuto.registry`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class ConstraintRun:
    """
    Full output of one ConstraintEngine.run() call.

    G              : sparse (M × E), legacy monolithic constraint matrix
    G_blocks       : dict t → sparse (M_t × E_t); per-t partition of G with cols
                     restricted to edges whose acc.edge_index.time == t. No
                     cross-time coupling exists in any builder, so G is exactly
                     block-diagonal by t and the partition is lossless.
    edge_cols_by_t : dict t → np.ndarray of global edge_id columns belonging to t
    row_meta       : DataFrame — one row per G row (constraint_id, type, is_hard, cone_type, time)
    a0             : (E,) — Stage-2 f_agg (time-replicated global vector). Legacy
                     global cold prior; the optimiser prior now lives per-period
                     as ``timestep_data[t]["a_prior"]`` (see ``a_prior_global``).
    sigma_a        : (E,) — sigma_agg
    u              : (E,) — flow upper bounds
    l              : (E,) — flow lower bounds
    timestep_data  : dict mapping timestep label →
                     {"c": ..., "sigma_c": ..., "a_prior": ...}, where
                     ``a_prior`` is the per-period cold seed (f_agg sliced to the
                     year's edges, cols_t order). Block path only.
    run_id         : unique identifier
    acc_id         : AccountingStructure id
    agg_id         : AggregationResult id (if available)
    created_at     : ISO timestamp
    """
    G:              sp.csr_matrix
    row_meta:       pd.DataFrame
    a0:             np.ndarray
    sigma_a:        np.ndarray
    u:              np.ndarray
    l:              np.ndarray
    timestep_data:  dict
    run_id:         str
    acc_id:         str
    agg_id:         str
    created_at:     str
    label:          str = ""
    _p:             float = 1.3
    G_blocks:       dict = field(default_factory=dict)
    edge_cols_by_t: dict = field(default_factory=dict)

    @property
    def a_prior_global(self) -> np.ndarray:
        """Global (E,) cold prior reconstructed from per-period ``a_prior``.

        Scatters each period's cold seed ``timestep_data[t]["a_prior"]`` (over
        ``edge_cols_by_t[t]``) into one E-length vector. Equals the legacy
        ``a0`` (= Stage-2 f_agg) on the block path — a convenience accessor for
        readers that want the prior as a single global vector (uncprop,
        constraint-stage diagnostics). Falls back to ``a0`` when no per-period
        ``a_prior`` is present (legacy runs / full-vector path).
        """
        out = np.zeros(self.G.shape[1], dtype=float)
        have = False
        for t, cols in self.edge_cols_by_t.items():
            ap = self.timestep_data.get(t, {}).get("a_prior")
            if ap is not None:
                out[np.asarray(cols)] = ap
                have = True
        return out if have else np.asarray(self.a0, dtype=float)

