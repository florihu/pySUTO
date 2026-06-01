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
    a0             : (E,) — a_agg at the first timestep (t=0 prior for optimiser)
    sigma_a        : (E,) — sigma_agg
    u              : (E,) — flow upper bounds
    l              : (E,) — flow lower bounds
    timestep_data  : dict mapping timestep label → {"c": ..., "sigma_c": ...}
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

