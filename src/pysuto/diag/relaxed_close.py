"""
relaxed_close.py
================
Read-only analysis of the relaxed recipe closes of an imputation log (ADR 0010, #150).

Not a diagnostic panel: a function over ``ImputationLog.as_dataframe()`` (or the stored
parquet) that counts the relaxed closes whose assumed-absent primaries a later imputer
filled.  Split out of ``imp.py``, which holds the step panels.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def relaxed_close_exposure(log_df: pd.DataFrame,
                           edge_valuation: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Relaxed recipe closes whose assumed-absent primaries were filled afterwards.

    A relaxed close (ADR 0010) counts a missing primary as 0 once the network has
    converged; a later imputer may still derive that primary, and no-overwrite keeps
    the close. Read-only: the log is not changed.

    Parameters
    ----------
    log_df : DataFrame
        ``ImputationLog.as_dataframe()`` (or the stored parquet) carrying the
        ``relaxed`` and ``assumed_absent`` columns; row order is the fill order.
    edge_valuation : (E,) array, optional
        Valuation label per edge id (``acc.edge_index["valuation"]``); without it
        every row is labelled ``""``.

    Returns
    -------
    DataFrame
        One row per exposed relaxed fill: ``edge_id``, ``valuation``, ``f_after``
        (kt in the valuation's unit), ``imputer_id``, ``pass``, ``step`` and
        ``later_edges`` (the assumed-absent edges that received a later fill).
    """
    pos = {}
    for i, e in enumerate(log_df["edge_id"].to_numpy(dtype=np.int64)):
        pos.setdefault(int(e), i)
    rows = []
    for i in np.flatnonzero(log_df["relaxed"].to_numpy(dtype=bool)):
        later = [int(e) for e in log_df["assumed_absent"].iat[i]
                 if pos.get(int(e), -1) > i]
        if later:
            e = int(log_df["edge_id"].iat[i])
            rows.append({
                "edge_id": e,
                "valuation": "" if edge_valuation is None else str(edge_valuation[e]),
                "f_after": float(log_df["f_after"].iat[i]),
                "imputer_id": log_df["imputer_id"].iat[i],
                "pass": log_df["pass"].iat[i],
                "step": int(log_df["step"].iat[i]),
                "later_edges": later,
            })
    return pd.DataFrame(rows, columns=["edge_id", "valuation", "f_after", "imputer_id",
                                       "pass", "step", "later_edges"])
