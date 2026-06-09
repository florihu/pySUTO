"""
_core.py
========
Shared, stateless builders behind the per-bucket stats CSVs, so the same logic
serves every stage instead of being copied per diagnostic.

* :func:`grouped_stats` — group-by-``(sector, region, valuation)`` percentile
  aggregator for a **per-edge** value. Used by :class:`StatsDiagnostic` (Stage 2,
  imputed-val distribution) and :class:`SolverStatsDiagnostic` (Stage 4, solver
  defect distribution). The value is masked to ``NaN`` upstream wherever it should
  be excluded; every aggregator here skips NaN, counts it (``n_nan``), and reports
  ``val_pct`` = non-NaN share.
* :func:`soft_constraint_table` — flat per-**soft-constraint** residual/violation
  table. Used by both :class:`ConstStatsDiagnostic` (Stage 3, prior residual
  ``G·a0 − c``) and :class:`SolverStatsDiagnostic` (Stage 4, solved
  ``c_star − c``). Same schema, parametrised only by the achieved LHS ``lhs = G·a``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_COLUMNS = ["min", "p25", "p50", "p75", "p95", "mean", "max", "sum",
            "n_val", "n_nan", "val_pct"]

_SOFT_TYPES = ("sum_eq", "ratio_eq")


def grouped_stats(
    edges: pd.DataFrame,
    side: str,
    value_col: str,
    z_col: str | None = None,
) -> pd.DataFrame:
    """Distribution stats of ``value_col`` grouped by one endpoint.

    Parameters
    ----------
    edges : pd.DataFrame
        Per-edge frame exposing ``f"{side}_sector"``, ``f"{side}_region"``,
        ``"valuation"``, ``value_col`` and (optionally) ``z_col``.
    side : str
        ``"from"`` or ``"to"`` — which endpoint defines the group.
    value_col : str
        Column summarised into the percentile/mean/sum columns (NaN-skipping).
    z_col : str, optional
        If given, append a ``median_abs_z`` column = per-group median of
        ``abs(z_col)``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``(sector, region, valuation)``; columns are the fixed stats
        block (plus ``median_abs_z`` when ``z_col`` is given). Empty input → an
        empty DataFrame.
    """
    if edges.empty:
        return pd.DataFrame()

    keys = [f"{side}_sector", f"{side}_region", "valuation"]
    g = edges.groupby(keys)
    v = g[value_col]

    out = pd.DataFrame({
        "min":   v.min(),
        "p25":   v.quantile(0.25),
        "p50":   v.quantile(0.50),
        "p75":   v.quantile(0.75),
        "p95":   v.quantile(0.95),
        "mean":  v.mean(),
        "max":   v.max(),
        "sum":   v.sum(),
        "n_val": v.count(),
        "n_nan": g.size() - v.count(),
    })
    out["val_pct"] = 100.0 * out["n_val"] / g.size()

    cols = list(_COLUMNS)
    if z_col is not None:
        out["median_abs_z"] = g[z_col].apply(lambda s: s.abs().median())
        cols = cols + ["median_abs_z"]

    out["n_val"] = out["n_val"].astype(int)
    out["n_nan"] = out["n_nan"].astype(int)
    out = out[cols]
    out.index.names = ["sector", "region", "valuation"]
    return out


def soft_constraint_table(
    row_meta: pd.DataFrame,
    c,
    sigma_c,
    lhs,
    year: str,
    soft_types: tuple = _SOFT_TYPES,
) -> pd.DataFrame:
    """Per-soft-constraint residual/violation table for one ``year``.

    Stage-neutral: it works off the achieved left-hand side ``lhs = G·a`` —
    ``G·a0`` (prior) at the constraint stage, ``c_star`` (= ``G·a*`` on soft rows)
    at the solver stage. ``residual = lhs − c`` is the codebase-reserved residual
    ``G·a − c``; on a soft row this equals the slack the solver calls ``eps``.

    Parameters
    ----------
    row_meta : pd.DataFrame
        One row per constraint (``constraint_id``, ``type``, ``time``, optionally
        ``valuation``). Positionally aligned with ``c`` / ``sigma_c`` / ``lhs``.
    c, sigma_c, lhs : array-like (M,)
        Target RHS, per-row tolerance, achieved LHS — full constraint-row length.
    year : str
        Restrict to rows whose ``time`` matches.
    soft_types : tuple
        Legacy fallback only — used when ``row_meta`` has no ``cone_type``
        column. A soft row is canonically a **free-cone** row (``cone_type ==
        "free"``), which is exactly the old ``sum_eq`` ∪ ``ratio_eq`` set.

    Returns
    -------
    pd.DataFrame
        Indexed by ``constraint_id``; columns
        ``[type, variant, valuation, c, lhs, residual, sigma_c, z]`` with
        ``z = residual/σ_c``.
    """
    rm = row_meta.reset_index(drop=True)
    # Soft ⟺ free cone (slack-bearing). Fall back to the legacy type whitelist
    # only when cone_type is absent (toy fixtures).
    if "cone_type" in rm.columns:
        soft = rm["cone_type"].astype(str) == "free"
    else:
        soft = rm["type"].isin(soft_types)
    mask = soft & (rm["time"].astype(str) == str(year))
    idx = np.flatnonzero(mask.to_numpy())

    c       = np.asarray(c, dtype=float)[idx]
    sigma_c = np.asarray(sigma_c, dtype=float)[idx]
    lhs     = np.asarray(lhs, dtype=float)[idx]
    residual = lhs - c
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma_c > 0, residual / sigma_c, np.nan)

    if "valuation" in rm.columns:
        valuation = rm["valuation"].to_numpy()[idx]
    else:
        valuation = np.full(idx.size, np.nan)

    if "variant" in rm.columns:
        variant = rm["variant"].to_numpy()[idx]
    else:
        variant = np.full(idx.size, np.nan)

    return pd.DataFrame(
        {
            "type":      rm["type"].to_numpy()[idx],
            "variant":   variant,
            "valuation": valuation,
            "c":         c,
            "lhs":       lhs,
            "residual":  residual,
            "sigma_c":   sigma_c,
            "z":         z,
        },
        index=pd.Index(rm["constraint_id"].to_numpy()[idx], name="constraint_id"),
    )
