"""
objective.py
============
Stateless builders for the **objective share** view of an opt run (pandas /
numpy, and the row-type and sense names of ``pysuto.constraints.naming``).

The QP objective is ``Σz²`` over edges (defects, ``z = (a*−a_prior)/σ_a``) plus
``Σz²`` over soft rows (residuals, ``z = (G·a*−c)/σ_c``). Every item's ``z²`` is
therefore its cost in the objective, and its **objective share** is that ``z²``
over the family's (or the run's) ``Σz²``. Three families:

* ``edges``       — edge defects;
* ``sum_rows``    — residuals on ``row_meta.type == "sum"`` rows (all soft; today
                    every one is a scale total or a scrap level anchor);
* ``soft_ratios`` — residuals on ``type == "ratio"`` rows in the free cone.

The three families partition the soft equality rows of today's builders (every one is a
``sum`` or a ``ratio`` row); unlike ``Solver.objective_at``, which adds ``other_soft`` for
a soft row of any other type, the families have no such bucket, so
:func:`objective_decomposition` raises on a soft row of another type rather than drop it
from ``zsq_total``.

A large ``|z|`` on a soft row is priced disagreement, not failure (GLOSSARY.md
§"Constraint table") — nothing here is a "violation".

Inputs are the all-years frames of :class:`SolverStatsDiagnostic`
(:meth:`edge_frame` / :meth:`constraint_frame`); :func:`family_frame`
harmonises one family into ``family, identity, label, sector_key, year, z,
delta`` (+ the source columns) and the other builders rank / aggregate it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pysuto.constraints.naming import SENSE_EQ, TYPE_MASSBAL, TYPE_RATIO, TYPE_SUM

from ._core import filter_regions

FAMILIES = ("edges", "sum_rows", "soft_ratios")
FAMILY_LABELS = {
    "edges":       "edge defects",
    "sum_rows":    "sum-row residuals",
    "soft_ratios": "soft-ratio residuals",
}
# "Region touch" columns per family (any match keeps the row).
REGION_COLS = {
    "edges":       ("from_region", "to_region"),
    "sum_rows":    ("region",),
    "soft_ratios": ("num_region_from", "num_region_to", "region"),
}
_HARMONISED = ["family", "identity", "label", "sector_key", "year", "z", "delta"]
_RANK_COLS = ["rank", "identity", "label", "sector_key", "zsq", "share",
              "sum_abs_delta", "n_years", "z_absmax", "year_absmax", "n_years_gt3"]


def _s(series: pd.Series) -> np.ndarray:
    """Series → unicode array with ``""`` for missing values (categorical-safe)."""
    return series.astype(object).where(series.notna(), "").astype(str).to_numpy(dtype=str)


def family_frame(edge_frame, constraint_frame, family: str, regions=None):
    """One family as a harmonised frame (``None`` when its source frame is None).

    ``sector_key``: edges → ``from_sector→to_sector``; sum rows → ``sector``
    (``variant`` when null); soft ratios → ``num_sector_from→num_sector_to``
    for pair-scoped rows, ``variant`` for set-scoped ones. ``delta`` is the
    defect (edges) or the residual (rows). ``regions`` applies the region-touch
    rule of :data:`REGION_COLS`.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if family == "edges":
        if edge_frame is None:
            return None
        src = filter_regions(edge_frame, REGION_COLS[family], regions)
        key = np.char.add(np.char.add(_s(src["from_sector"]), "→"), _s(src["to_sector"]))
        delta = src["defect"].to_numpy(dtype=float)
    else:
        if constraint_frame is None:
            return None
        typ = _s(constraint_frame["type"])
        if family == "sum_rows":
            src = constraint_frame[typ == TYPE_SUM]
        else:
            sense = _s(constraint_frame["sense"])
            hard  = constraint_frame["is_hard"].astype(bool).to_numpy()
            src = constraint_frame[(typ == TYPE_RATIO) & (sense == SENSE_EQ) & ~hard]
        src = filter_regions(src, REGION_COLS[family], regions)
        variant = _s(src["variant"]) if "variant" in src.columns else np.full(len(src), "", dtype=str)
        if family == "sum_rows":
            sector = _s(src["sector"]) if "sector" in src.columns else np.full(len(src), "", dtype=str)
            key = np.where(sector != "", sector, variant)
        else:
            if "num_sector_from" in src.columns:
                nsf, nst = _s(src["num_sector_from"]), _s(src["num_sector_to"])
                key = np.where(nsf != "", np.char.add(np.char.add(nsf, "→"), nst), variant)
            else:
                key = variant
        delta = src["residual"].to_numpy(dtype=float)

    out = pd.DataFrame({
        "family":     family,
        "identity":   src["identity"].to_numpy(dtype=np.int64),
        "label":      src["label"].to_numpy(),
        "sector_key": np.asarray(key, dtype=object),
        "year":       _s(src["year"]),
        "z":          src["z"].to_numpy(dtype=float),
        "delta":      delta,
    }, index=src.index)
    rest = src.drop(columns=[c for c in _HARMONISED if c in src.columns])
    return pd.concat([out, rest], axis=1).reset_index(drop=True)


def share_by_key(fam, key: str = "sector_key", top=None):
    """``Σz²`` per ``key`` value with its share of the family total.

    Columns ``[key, zsq, share, n_rows, n_identities]`` sorted by ``zsq``
    descending; NaN ``z`` ignored. With ``top``, everything past the first
    ``top`` keys is folded into one ``"other"`` row.
    """
    if fam is None:
        return None
    ok = fam["z"].notna().to_numpy()
    f = fam.loc[ok, [key, "identity", "z"]].copy()
    f["zsq"] = f["z"].to_numpy(dtype=float) ** 2
    g = f.groupby(key, sort=False)
    out = pd.DataFrame({
        "zsq":          g["zsq"].sum(),
        "n_rows":       g.size(),
        "n_identities": g["identity"].nunique(),
    })
    total = float(out["zsq"].sum())
    out["share"] = out["zsq"] / total if total > 0 else np.nan
    out = out.sort_values("zsq", ascending=False).reset_index()
    if top is not None and len(out) > top:
        head, tail = out.iloc[:top], out.iloc[top:]
        other = pd.DataFrame([{
            key: "other",
            "zsq": tail["zsq"].sum(), "n_rows": int(tail["n_rows"].sum()),
            "n_identities": int(tail["n_identities"].sum()), "share": tail["share"].sum(),
        }])
        out = pd.concat([head, other], ignore_index=True)
    return out[[key, "zsq", "share", "n_rows", "n_identities"]]


MULTI = "multi"
_SHARE_REGION_COLS = ["family", "region", "year", "zsq", "share", "n_rows"]


def _col(fam: pd.DataFrame, name: str):
    """Column as a unicode array (``""`` for missing), or ``None`` when absent."""
    return _s(fam[name]) if name in fam.columns else None


def assign_region(fam: pd.DataFrame) -> np.ndarray:
    """Region bucket of every row of a :func:`family_frame` (read from columns).

    * ``edges``       — ``from_region`` when equal to ``to_region``, else ``multi``;
    * ``sum_rows``    — the row's ``region``; ``multi`` when it has none (world totals);
    * ``soft_ratios`` — a pair-scoped row takes ``num_region_from`` when equal to
      ``num_region_to``, else ``multi``; a set-scoped row its ``region``, else ``multi``.

    ``multi`` is the bucket of everything that spans regions or names none.
    Limitation: a sum row grouped on both ends carries only ``region_to`` in
    ``row_meta``, so it is credited to the destination, not to ``multi``.
    """
    n = len(fam)
    multi = np.full(n, MULTI, dtype=object)
    family = fam["family"].iloc[0] if n else None
    if family == "edges":
        a, b = _col(fam, "from_region"), _col(fam, "to_region")
        return np.where((a == b) & (a != ""), a, multi)
    own = _col(fam, "region")
    own = own if own is not None else np.full(n, "", dtype=str)
    if family == "sum_rows":
        return np.where(own != "", own, multi)
    a, b = _col(fam, "num_region_from"), _col(fam, "num_region_to")
    if a is None:
        return np.where(own != "", own, multi)
    pair = np.where((a == b) & (a != ""), a, multi)
    return np.where(a != "", pair, np.where(own != "", own, multi))


def share_by_region(fam):
    """Each region's share of one family's ``Σz²``, overall and per year.

    Rows are bucketed by :func:`assign_region` (spanning / region-less rows go
    to ``multi``). Columns ``family, region, year, zsq, share, n_rows``: the
    ``year == "all"`` rows give the region's ``Σ_t z²`` and its share of the
    family total; the per-year rows give the region-year's ``Σz²`` and *its*
    share of the same family total (so each block sums to 1). The ``"all"``
    block comes first, each block sorted by ``zsq`` descending; NaN ``z``
    ignored. ``None`` for a ``None`` frame.
    """
    if fam is None:
        return None
    ok = fam["z"].notna().to_numpy()
    f = fam.loc[ok]
    d = pd.DataFrame({
        "region": assign_region(fam)[ok],
        "year":   f["year"].to_numpy(),
        "zsq":    f["z"].to_numpy(dtype=float) ** 2,
    })
    total = float(d["zsq"].sum())
    name = fam["family"].iloc[0] if len(fam) else ""
    blocks = []
    for keys, year in ((["region"], "all"), (["region", "year"], None)):
        g = d.groupby(keys, sort=False)["zsq"].agg(["sum", "size"]).reset_index()
        g = g.rename(columns={"sum": "zsq", "size": "n_rows"})
        if year is not None:
            g["year"] = year
        g["family"] = name
        g["share"] = g["zsq"] / total if total > 0 else np.nan
        blocks.append(g.sort_values("zsq", ascending=False, kind="stable"))
    return pd.concat(blocks, ignore_index=True)[_SHARE_REGION_COLS]


def rank_identities(fam, top=None):
    """Identities of one family ranked by ``Σ_t z²`` (descending).

    Columns: ``rank, identity, label, sector_key, zsq, share, sum_abs_delta,
    n_years, z_absmax, year_absmax, n_years_gt3``. ``share`` is relative to the
    family's total ``Σz²`` *including* rows without an identity, so it is
    comparable with :func:`share_by_key`. Unidentified rows (``identity == -1``)
    are excluded and counted in ``.attrs["n_unidentified"]``; the family total
    is ``.attrs["zsq_total"]``.
    """
    if fam is None:
        return None
    z_all = fam["z"].to_numpy(dtype=float)
    total = float(np.nansum(z_all ** 2))
    ident = fam["identity"].to_numpy()
    unid = ident < 0
    keep = (~unid) & np.isfinite(z_all)
    f = fam.loc[keep, ["identity", "label", "sector_key", "year", "z", "delta"]].copy()
    empty = pd.DataFrame(columns=_RANK_COLS)
    if f.empty:
        empty.attrs["n_unidentified"] = int(unid.sum())
        empty.attrs["zsq_total"] = total
        return empty
    z = f["z"].to_numpy(dtype=float)
    f["zsq"]  = z ** 2
    f["absz"] = np.abs(z)
    f["absd"] = np.abs(f["delta"].to_numpy(dtype=float))
    f["gt3"]  = f["absz"] > 3
    g = f.groupby("identity", sort=False)
    out = pd.DataFrame({
        "label":         g["label"].first(),
        "sector_key":    g["sector_key"].first(),
        "zsq":           g["zsq"].sum(),
        "sum_abs_delta": g["absd"].sum(),
        "n_years":       g["year"].nunique(),
        "z_absmax":      g["absz"].max(),
        "n_years_gt3":   g["gt3"].sum().astype(int),
    })
    idx = g["absz"].idxmax()
    out["year_absmax"] = pd.Series(f.loc[idx.to_numpy(), "year"].to_numpy(), index=idx.index)
    out["share"] = out["zsq"] / total if total > 0 else np.nan
    out = out.sort_values("zsq", ascending=False).reset_index()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out = out[_RANK_COLS]
    if top is not None:
        out = out.head(top).reset_index(drop=True)
    out.attrs["n_unidentified"] = int(unid.sum())
    out.attrs["zsq_total"] = total
    return out


def profiles(fam, identities, years=None) -> pd.DataFrame:
    """Signed ``z`` per ``identity × year`` (wide) for the given identities, in
    the given order; columns reindexed to ``years`` when given."""
    ids = list(identities)
    sub = fam[fam["identity"].isin(ids)]
    wide = sub.pivot_table(index="identity", columns="year", values="z", aggfunc="first")
    wide = wide.reindex(index=ids)
    if years is not None:
        wide = wide.reindex(columns=[str(y) for y in years])
    return wide


def objective_decomposition(edge_frame, constraint_frame, years=None, regions=None) -> pd.DataFrame:
    """Per-year ``Σz²`` per family (+ total), the max hard mass-balance
    ``|G·a*−c|`` (from ``resid_solver``) and the share of ``|z| > 3`` per family.

    Columns: ``year, zsq_edges, zsq_sum_rows, zsq_soft_ratios, zsq_total,
    max_abs_resid_massbal, share_gt3_<family>…, n_<family>…``. A family whose
    frame is missing yields NaN columns rather than an error.

    ``max_abs_resid_massbal`` is one maximum over the mass-balance rows of both
    valuations, in kt on a gross row and kt CC on a content row: a hard row has no
    σ_c to standardise it by, so the column stays unnormalised and is labelled so.
    A soft equality row whose ``type`` is neither ``sum`` nor ``ratio`` raises
    ``ValueError`` instead of dropping out of ``zsq_total``.
    """
    if constraint_frame is not None:
        soft = ((_s(constraint_frame["sense"]) == SENSE_EQ)
                & ~constraint_frame["is_hard"].astype(bool).to_numpy())
        other = soft & ~np.isin(_s(constraint_frame["type"]), (TYPE_SUM, TYPE_RATIO))
        if other.any():
            raise ValueError(
                f"{int(other.sum())} soft row(s) of type "
                f"{sorted(set(_s(constraint_frame['type'])[other]))} belong to no "
                f"objective family; their z² would be missing from zsq_total")
    fams = {f: family_frame(edge_frame, constraint_frame, f, regions=regions) for f in FAMILIES}
    if years is None:
        seen: set = set()
        for fam in fams.values():
            if fam is not None:
                seen.update(fam["year"].tolist())
        if constraint_frame is not None:
            seen.update(_s(constraint_frame["year"]).tolist())
        years = sorted(seen)
    years = [str(y) for y in years]
    out = pd.DataFrame(index=pd.Index(years, name="year"))
    for f, fam in fams.items():
        if fam is None or fam.empty:
            out[f"zsq_{f}"] = np.nan
            out[f"share_gt3_{f}"] = np.nan
            out[f"n_{f}"] = np.nan
            continue
        z = fam["z"].to_numpy(dtype=float)
        ok = np.isfinite(z)
        d = pd.DataFrame({"year": fam["year"].to_numpy()[ok],
                          "zsq": z[ok] ** 2, "gt3": np.abs(z[ok]) > 3})
        g = d.groupby("year")
        out[f"zsq_{f}"]       = g["zsq"].sum().reindex(years)
        out[f"share_gt3_{f}"] = g["gt3"].mean().reindex(years)
        out[f"n_{f}"]         = g.size().reindex(years)
    out["zsq_total"] = out[[f"zsq_{f}" for f in FAMILIES]].sum(axis=1, min_count=1)
    if constraint_frame is not None and "resid_solver" in constraint_frame.columns:
        hard = constraint_frame[_s(constraint_frame["type"]) == TYPE_MASSBAL]
        hard = filter_regions(hard, ("region",), regions)
        a = pd.DataFrame({"year": _s(hard["year"]),
                          "a": np.abs(hard["resid_solver"].to_numpy(dtype=float))})
        out["max_abs_resid_massbal"] = a.groupby("year")["a"].max().reindex(years)
    else:
        out["max_abs_resid_massbal"] = np.nan
    cols = (["zsq_" + f for f in FAMILIES] + ["zsq_total", "max_abs_resid_massbal"]
            + ["share_gt3_" + f for f in FAMILIES] + ["n_" + f for f in FAMILIES])
    return out[cols].reset_index()
