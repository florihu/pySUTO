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
* :func:`constraint_table` — flat per-**constraint** row-level table over **all**
  rows (hard + soft). Used by both :class:`ConstStatsDiagnostic` (Stage 3,
  ``residual = lhs_prior − c``) and :class:`SolverStatsDiagnostic` (Stage 4,
  ``residual = lhs_post − c``). Harmonised schema: ``lhs_prior = G·a0`` appears in
  both stages' tables; ``lhs_post = c_star`` only at the solver stage, and
  ``residual``/``z`` always derive from the stage's achieved LHS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_COLUMNS = ["min", "p25", "p50", "p75", "p95", "mean", "max", "sum",
            "n_val", "n_nan", "val_pct"]


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


def constraint_table(
    row_meta: pd.DataFrame,
    c,
    sigma_c,
    lhs_prior,
    year: str,
    lhs_post=None,
) -> pd.DataFrame:
    """Row-level per-constraint table for one ``year`` — **all** rows (hard + soft).

    Harmonised across stages: ``lhs_prior = G·a0`` means the same thing in both
    stages' tables; ``lhs_post`` (= ``c_star = G·a*``) is given only at the solver
    stage. ``residual = lhs − c`` (the codebase-reserved residual ``G·a − c``,
    = the slack ``eps`` on a soft row) and ``z = residual/σ_c`` always derive from
    the stage's achieved LHS: ``lhs_post`` when given, else ``lhs_prior``. ``z`` is
    NaN where ``σ_c ≤ 0`` (hard rows).

    Parameters
    ----------
    row_meta : pd.DataFrame
        One row per constraint (``constraint_id``, ``type``, ``time``, optionally
        ``variant`` / ``cone_type`` / ``valuation``). Positionally aligned with
        ``c`` / ``sigma_c`` / ``lhs_prior`` / ``lhs_post``.
    c, sigma_c, lhs_prior : array-like (M,)
        Target RHS, per-row tolerance, prior LHS ``G·a0`` — full row length.
    year : str
        Restrict to rows whose ``time`` matches.
    lhs_post : array-like (M,), optional
        Solved LHS ``c_star``; adds the ``lhs_post`` column and rebases
        ``residual``/``z`` on it.

    Returns
    -------
    pd.DataFrame
        Columns ``[constraint_id, type, variant, cone_type, valuation, c,
        lhs_prior, (lhs_post,) residual, residual_abs, sigma_c, z, z_abs]``.
    """
    rm = row_meta.reset_index(drop=True)
    mask = rm["time"].astype(str) == str(year)
    idx = np.flatnonzero(mask.to_numpy())

    c         = np.asarray(c, dtype=float)[idx]
    sigma_c   = np.asarray(sigma_c, dtype=float)[idx]
    lhs_prior = np.asarray(lhs_prior, dtype=float)[idx]
    if lhs_post is not None:
        lhs_post = np.asarray(lhs_post, dtype=float)[idx]
    achieved = lhs_post if lhs_post is not None else lhs_prior
    residual = achieved - c
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sigma_c > 0, residual / sigma_c, np.nan)

    def _col(name):
        if name in rm.columns:
            return rm[name].to_numpy()[idx]
        return np.full(idx.size, np.nan)

    out = {
        "constraint_id": rm["constraint_id"].to_numpy()[idx],
        "type":          rm["type"].to_numpy()[idx],
        "variant":       _col("variant"),
        "cone_type":     _col("cone_type"),
        "valuation":     _col("valuation"),
        "c":             c,
        "lhs_prior":     lhs_prior,
    }
    if lhs_post is not None:
        out["lhs_post"] = lhs_post
    out.update({
        "residual":     residual,
        "residual_abs": np.abs(residual),
        "sigma_c":      sigma_c,
        "z":            z,
        "z_abs":        np.abs(z),
    })
    return pd.DataFrame(out)
