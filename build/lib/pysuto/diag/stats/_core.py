"""
_core.py
========
Shared, stateless builders behind the per-bucket stats CSVs and the Stage-4
figure frames, so the same logic serves every stage instead of being copied per
diagnostic.

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
* :func:`filter_regions`, :func:`sector_totals`, :func:`boundary_edges` — frame
  aggregators over the Stage-4 all-years edge frame (sector totals + retention,
  system-boundary edge split) consumed by the ``SolverDiag`` figures.
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
    year,
    lhs_post=None,
    extra_cols=(),
) -> pd.DataFrame:
    """Row-level per-constraint table for one ``year`` — **all** rows (hard + soft).

    Harmonised across stages: ``lhs_prior = G·a0`` means the same thing in both
    stages' tables; ``lhs_post`` (= ``c_star``, which equals ``G·a*`` on soft rows
    and ``c`` itself on hard rows) is given only at the solver stage.
    ``residual = lhs − c`` (the codebase-reserved residual ``G·a − c``, = the slack
    ``eps`` on a soft row) and ``z = residual/σ_c`` always derive from the stage's
    achieved LHS: ``lhs_post`` when given, else ``lhs_prior``. ``z`` is NaN where
    ``σ_c ≤ 0`` (hard rows).

    Parameters
    ----------
    row_meta : pd.DataFrame
        One row per constraint (``constraint_id``, ``type``, ``time``, optionally
        ``variant`` / ``cone_type`` / ``valuation``). Positionally aligned with
        ``c`` / ``sigma_c`` / ``lhs_prior`` / ``lhs_post``.
    c, sigma_c, lhs_prior : array-like (M,)
        Target RHS, per-row tolerance, prior LHS ``G·a0`` — full row length.
    year : str | None
        Restrict to rows whose ``time`` matches; ``None`` keeps every row (the
        caller has already assembled per-year ``c`` / ``sigma_c`` / ``lhs_post``
        into full-length vectors).
    lhs_post : array-like (M,), optional
        Solved LHS ``c_star``; adds the ``lhs_post`` column and rebases
        ``residual``/``z`` on it.
    extra_cols : sequence of str
        Further ``row_meta`` columns to carry through (only those present are
        appended, after the standard columns). Default ``()`` leaves the schema
        unchanged.

    Returns
    -------
    pd.DataFrame
        Columns ``[constraint_id, type, variant, cone_type, valuation, c,
        lhs_prior, (lhs_post,) residual, residual_abs, sigma_c, z, z_abs,
        *extra_cols]``.
    """
    rm = row_meta.reset_index(drop=True)
    if year is None:
        idx = np.arange(len(rm))
    else:
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
    for name in extra_cols:
        if name in rm.columns and name not in out:
            out[name] = rm[name].to_numpy()[idx]
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Frame aggregators (Stage-4 all-years edge frame)
# ---------------------------------------------------------------------------

def filter_regions(frame, cols, regions):
    """Rows of ``frame`` touching any of ``regions`` on any of ``cols``.

    ``regions=None`` (or a None frame) returns the input unchanged. Columns
    absent from the frame are ignored; a row is kept when at least one present
    column is in ``regions`` (the "region touch" rule: an edge touches a region
    on either endpoint, a pair-scoped ratio row on either numerator endpoint).
    """
    if frame is None or regions is None:
        return frame
    wanted = {str(r) for r in regions}
    mask = np.zeros(len(frame), dtype=bool)
    for col in cols:
        if col in frame.columns:
            mask |= frame[col].isin(wanted).to_numpy()
    return frame[mask]


def sector_totals(edges, group_col, years=None, valuations=None) -> pd.DataFrame:
    """Σa* and Σa_prior per ``(group_col, valuation, year)`` plus **retention**.

    Returns the long frame ``group, valuation, year, sum_post, sum_prior,
    retention`` where ``retention = sum_post / sum_prior`` (NaN when
    ``sum_prior`` is not > 0). When ``years`` / ``valuations`` are given the
    grid is completed (missing cells → NaN sums) so time series line up.
    """
    keys = [group_col, "valuation", "year"]
    g = (edges.groupby(keys, observed=True, sort=True)[["a_post", "a_prior"]]
              .sum()
              .rename(columns={"a_post": "sum_post", "a_prior": "sum_prior"}))
    if years is not None or valuations is not None:
        lv = g.index.levels if len(g.index) else None
        groups = sorted(set(g.index.get_level_values(0))) if len(g.index) else []
        vals   = list(valuations) if valuations is not None else \
                 sorted(set(g.index.get_level_values(1))) if len(g.index) else []
        yrs    = [str(y) for y in years] if years is not None else \
                 sorted(set(g.index.get_level_values(2))) if len(g.index) else []
        full = pd.MultiIndex.from_product([groups, vals, yrs], names=keys)
        g = g.reindex(full)
    g = g.reset_index().rename(columns={group_col: "group"})
    g["year"] = g["year"].astype(str)
    sp = g["sum_prior"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g["retention"] = np.where(sp > 0, g["sum_post"].to_numpy(dtype=float) / sp, np.nan)
    return g


def boundary_edges(edges) -> pd.DataFrame:
    """The system-boundary subset of an edge frame: exactly one ``SB`` endpoint.

    Adds ``direction`` (``sb_input`` = ``from_entity == "SB"``, ``sb_output`` =
    ``to_entity == "SB"``), ``sb_sector`` (the SB node's sector) and
    ``sb_entity_sub`` (its boundary category, ``""`` when the frame carries no
    ``*_entity_sub`` columns). Requires ``from_entity`` / ``to_entity``.
    """
    fe = edges["from_entity"].astype(str).to_numpy()
    te = edges["to_entity"].astype(str).to_numpy()
    is_in  = (fe == "SB") & (te != "SB")
    is_out = (te == "SB") & (fe != "SB")
    keep = is_in | is_out
    out = edges.loc[keep].copy()
    inn = is_in[keep]
    out["direction"] = np.where(inn, "sb_input", "sb_output")
    out["sb_sector"] = np.where(inn, out["from_sector"].astype(str).to_numpy(),
                                     out["to_sector"].astype(str).to_numpy())
    if {"from_entity_sub", "to_entity_sub"} <= set(out.columns):
        sub = np.where(inn, out["from_entity_sub"].astype(object).to_numpy(),
                            out["to_entity_sub"].astype(object).to_numpy())
        out["sb_entity_sub"] = pd.Series(sub, index=out.index).fillna("").astype(str) \
                                 .replace({"nan": "", "None": ""}).to_numpy()
    else:
        out["sb_entity_sub"] = ""
    return out
