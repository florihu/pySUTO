"""
run.py
======
``ConstraintRun`` — the persisted output of one :class:`ConstraintEngine`
run. Saved to disk by :class:`ConstraintRunRegistry`
(in :mod:`pysuto.registry`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class ConstraintRun:
    """
    Full output of one ConstraintEngine.run() call.

    G          : sparse (M × E), time-invariant
    row_meta   : DataFrame — one row per G row (constraint_id, type, is_hard, cone_type)
    a0         : (E,) — f_agg at the first timestep (t=0 prior for optimiser)
    sigma_a    : (E,) — sigma_agg
    u          : (E,) — flow upper bounds
    l          : (E,) — flow lower bounds
    timestep_data : dict mapping timestep label → {"c": ..., "sigma_c": ...}
    run_id     : unique identifier
    acc_id     : AccountingStructure id
    agg_id     : AggregationResult id (if available)
    created_at : ISO timestamp
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

