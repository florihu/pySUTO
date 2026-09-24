"""
solver.py
=========
``SolverStatsDiagnostic`` — post-solve row-level tables: the per-year CSVs the
REP scripts read, and the cached all-years **frames** the ``SolverDiag`` figures
draw from. Both come out of one code path, so figures and CSVs agree by
construction.

Per-year CSV deliverables (schemas frozen):

**Edge table** (``edges_<year>.csv``) — one row per edge active in ``year``, with
endpoint metadata plus ``a_prior`` (the per-period prior the objective used),
``a_post`` (= ``F_opt``), ``sigma_a``, the per-edge **defect** ``= a_post − a_prior``,
``defect_abs`` and ``z = defect/σ_a`` (NaN where ``σ_a ≤ 0``).

**Constraint table** (``constraints_<year>.csv``) — one row per constraint row
(hard + soft) active in ``year``, built by the stage-harmonised
:func:`pysuto.diag.stats._core.constraint_table`: ``lhs_prior = G·a0``,
``lhs_post = c_star``, ``residual = lhs_post − c`` (= the slack the solver calls
``eps`` on a soft row), ``z = residual/σ_c``. The Stage-3 twin
:class:`ConstStatsDiagnostic` emits the same schema minus ``lhs_post``.
**Note:** ``c_star = c + ε`` and ``ε = 0`` on hard rows, so ``lhs_post = G·a*``
holds on soft rows only and the CSV ``residual`` is identically 0 on hard rows;
the true hard-row residual is ``OptimResult.residuals`` (frame column
``resid_solver`` below).

All-years frames (:meth:`edge_frame`, :meth:`constraint_frame`): the CSV columns
for every result year plus ``year``, the cross-year **identity** (edges: the acc
edge template; rows: factorised over the ``row_meta`` identity columns, ``-1``
where the columns do not single the row out within its year) and a readable
``label``; the constraint frame additionally carries ``lhs_prior_t = G·a_prior(t)``
(the per-period prior LHS, ≠ ``lhs_prior`` under warm priors) and
``resid_solver``. The per-year tables are slices of these frames.

Terminology: **defect** = per-edge ``a*−a_prior``; **residual** = per-constraint
``G·a − c`` (reserved — see ``OptimResult.residuals``). They are distinct.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ._core import constraint_table

# CSV schema of the edge table (present columns only, in this order).
_EDGE_META_COLS = ["edge_id", "from_sector", "from_region", "from_entity",
                   "to_sector", "to_region", "to_entity", "valuation"]
# Extra edge_meta columns carried by the frame (never by the CSV).
_EDGE_FRAME_EXTRA = ["from_id", "to_id", "from_entity_sub", "to_entity_sub"]
_EDGE_VALUE_COLS = ["a_prior", "a_post", "sigma_a", "defect", "defect_abs", "z"]

# row_meta columns carried by the constraint frame (never by the CSV).
_ROW_EXTRA_COLS = ("is_hard", "sector", "region", "entity", "entity_sub",
                   "ratio_kind", "bound_dir",
                   "num_sector_from", "num_region_from", "num_sector_to",
                   "num_region_to", "num_valuation",
                   "denom_sector_from", "denom_region_from", "denom_sector_to",
                   "denom_region_to", "denom_valuation")
# A row's cross-year identity is the tuple of EVERY row_meta column except the
# display key and the year (read from columns, never parsed from constraint_id
# — CONTEXT.md §"Constraint identity vocabulary"). cone_type / bound_dir are
# part of it: a pair's soft row and its two hard band rows are three identities.
_ROW_NON_IDENTITY_COLS = ("constraint_id", "time")
_ROW_CATEGORY_COLS = ("type", "variant", "cone_type", "valuation") + _ROW_EXTRA_COLS
_CONST_CSV_COLS = ["constraint_id", "type", "variant", "cone_type", "valuation",
                   "c", "lhs_prior", "lhs_post", "residual", "residual_abs",
                   "sigma_c", "z", "z_abs"]


def resolve_years(years, available) -> list[str]:
    """Coerce a ``years`` spec to a validated list of year strings.

    ``years`` is a sequence of year-likes (coerced via ``str()``) or the literal
    ``"all"`` → every entry of ``available`` (the run's timestep keys). An
    unknown year raises ``KeyError`` listing what is available.
    """
    avail = [str(k) for k in available]
    if isinstance(years, str):
        if years != "all":
            raise ValueError(f"years must be a sequence or 'all', got {years!r}")
        return avail
    out = [str(y) for y in years]
    unknown = [y for y in out if y not in avail]
    if unknown:
        raise KeyError(f"no timestep for year(s) {unknown} (have {avail})")
    return out


def _strings(series: pd.Series) -> np.ndarray:
    """Object series → str array with ``""`` for missing values."""
    return series.astype(object).where(series.notna(), "").astype(str).to_numpy()


class SolverStatsDiagnostic:
    """Post-solve row-level edge + constraint tables (CSVs per ``year``) and the
    cached all-years frames behind the ``SolverDiag`` figures.

    Parameters
    ----------
    results : dict[timestep -> OptimResult]
        The (reassembled) per-timestep solver results, as exposed by
        ``SolverDiag.results``. Each ``OptimResult`` supplies global-length
        ``F_opt`` and ``c_star`` (and ``residuals``).
    constraint_run : ConstraintRun
        Supplies ``G``, ``a0``, ``sigma_a``, ``row_meta`` and ``timestep_data``.
    acc_struct : optional
        ``AccountingStructure`` — supplies ``edge_meta`` (and, when available,
        ``edge_template_code``). ``None`` disables the edge side: the
        constraint tables still work, :meth:`edge_frame` returns ``None`` and
        :meth:`edge_table` raises.
    years : sequence of str | "all"
        The years :meth:`write` emits both deliverables for (default ``("1995",)``).
        The frames always cover every result year.
    """

    def __init__(self, results, constraint_run, acc_struct=None, years=("1995",)) -> None:
        self._res = results
        self._cr = constraint_run
        self._acc = acc_struct
        self.years = resolve_years(years, results)
        self._all_years = [str(k) for k in results]
        self._lhs_prior = None            # cached G·a0 (year-independent)
        self._edge_frame_cache = None
        self._constraint_frame_cache = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _result_for_year(self, year: str):
        """The ``OptimResult`` whose timestep matches ``year`` (key may be int/str)."""
        for k in self._res:
            if str(k) == year:
                return self._res[k]
        raise KeyError(f"no solver result for year {year!r} "
                       f"(have {[str(k) for k in self._res]})")

    def _timestep_data_for_year(self, year: str) -> dict:
        for k in self._cr.timestep_data:
            if str(k) == year:
                return self._cr.timestep_data[k]
        raise KeyError(f"no constraint timestep_data for year {year!r}")

    def _lhs_prior_full(self) -> np.ndarray:
        """``G·a0`` over all rows — computed once, cached."""
        if self._lhs_prior is None:
            a0 = np.asarray(self._cr.a0, dtype=float)
            self._lhs_prior = np.asarray(self._cr.G.dot(a0)).ravel()
        return self._lhs_prior

    def _cold_prior(self) -> np.ndarray:
        """Global cold prior (``a_prior_global`` when available, else legacy ``a0``)."""
        cold = getattr(self._cr, "a_prior_global", None)
        return np.asarray(cold if cold is not None else self._cr.a0, dtype=float)

    def _prior_vector(self, year: str) -> np.ndarray:
        """The prior the objective used for ``year``, as a global ``(E,)`` vector.

        Warm = ``a_post(t-1)``, cold = ``f_agg(t)``: the year-local
        ``OptimResult.a_prior`` scattered into global edge order over the cold
        prior; a legacy full-length ``a_prior`` is used as is; runs predating
        ``a_prior`` fall back to the cold prior.
        """
        res = self._result_for_year(year)
        cold = self._cold_prior()
        ap = getattr(res, "a_prior", None)
        if ap is not None and getattr(res, "cols_t", None) is not None:
            out = cold.copy()
            out[np.asarray(res.cols_t)] = np.asarray(ap, dtype=float)
            return out
        if ap is not None:
            return np.asarray(ap, dtype=float)
        return cold

    # ------------------------------------------------------------------
    # All-years frames
    # ------------------------------------------------------------------

    def edge_frame(self):
        """All result years' edges: CSV columns + ``year``, ``identity``, ``label``
        (+ ``from_id``/``to_id``/``*_entity_sub`` when the acc carries them).
        ``None`` when no ``edge_meta`` is available. Cached."""
        if self._edge_frame_cache is not None:
            return self._edge_frame_cache
        em = getattr(self._acc, "edge_meta", None) if self._acc is not None else None
        if em is None:
            return None
        em = em.reset_index(drop=True)
        E = len(em)
        time = em["time"].astype(str).to_numpy()
        a_post  = np.full(E, np.nan)
        a_prior = np.full(E, np.nan)
        for y in self._all_years:
            m = time == y
            if not m.any():
                continue
            res = self._result_for_year(y)
            a_post[m]  = np.asarray(res.F_opt, dtype=float)[m]
            a_prior[m] = self._prior_vector(y)[m]
        sigma_a = np.asarray(self._cr.sigma_a, dtype=float)
        defect = a_post - a_prior
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sigma_a > 0, defect / sigma_a, np.nan)

        cols = [c for c in _EDGE_META_COLS + _EDGE_FRAME_EXTRA if c in em.columns]
        out = em[cols].copy()
        out["a_prior"]    = a_prior
        out["a_post"]     = a_post
        out["sigma_a"]    = sigma_a
        out["defect"]     = defect
        out["defect_abs"] = np.abs(defect)
        out["z"]          = z
        out["year"]       = time
        out["identity"]   = self._edge_identity(em)
        keep = np.isin(time, np.asarray(self._all_years))
        out = out[keep].reset_index(drop=True)
        out["label"] = self._edge_labels(out)
        self._edge_frame_cache = out
        return out

    def _edge_identity(self, em: pd.DataFrame) -> np.ndarray:
        """Cross-year edge identity: the acc's edge template code, else a
        factorisation of ``(from_id, to_id, valuation)`` (or of the endpoint
        labels when ids are absent)."""
        code = getattr(self._acc, "edge_template_code", None)
        if code is not None and len(code) == len(em):
            return np.asarray(code, dtype=np.int64)
        if {"from_id", "to_id", "valuation"} <= set(em.columns):
            keys = ["from_id", "to_id", "valuation"]
        else:
            keys = [c for c in ("from_sector", "from_region", "to_sector",
                                "to_region", "valuation") if c in em.columns]
        return em.groupby(keys, sort=False, dropna=False).ngroup().to_numpy(dtype=np.int64)

    @staticmethod
    def _edge_labels(frame: pd.DataFrame) -> np.ndarray:
        """``from_sector|from_region→to_sector|to_region|valuation`` per identity."""
        first = frame.drop_duplicates("identity").set_index("identity")

        def _c(name):
            return _strings(first[name]) if name in first.columns else np.full(len(first), "", dtype=object)

        lab = (_c("from_sector") + "|" + _c("from_region") + "→"
               + _c("to_sector") + "|" + _c("to_region") + "|" + _c("valuation"))
        return frame["identity"].map(pd.Series(lab, index=first.index)).to_numpy()

    def constraint_frame(self):
        """All result years' constraint rows: CSV columns + the ``row_meta``
        identity columns + ``year``, ``identity``, ``label``, ``lhs_prior_t``,
        ``resid_solver``. ``None`` when ``row_meta`` lacks ``time``/``type``
        (toy fixtures). Cached."""
        if self._constraint_frame_cache is not None:
            return self._constraint_frame_cache
        rm = self._cr.row_meta.reset_index(drop=True)
        if not {"time", "type"} <= set(rm.columns):
            return None
        M = len(rm)
        time = rm["time"].astype(str).to_numpy()
        c_full  = np.full(M, np.nan)
        sc_full = np.full(M, np.nan)
        cs_full = np.full(M, np.nan)
        lhs_t   = np.full(M, np.nan)
        resid   = np.full(M, np.nan)
        G = self._cr.G
        for y in self._all_years:
            m = time == y
            if not m.any():
                continue
            td  = self._timestep_data_for_year(y)
            res = self._result_for_year(y)
            c_full[m]  = np.asarray(td["c"], dtype=float)[m]
            sc_full[m] = np.asarray(td["sigma_c"], dtype=float)[m]
            cs_full[m] = np.asarray(res.c_star, dtype=float)[m]
            lhs_t[m]   = np.asarray(G.dot(self._prior_vector(y))).ravel()[m]
            r = getattr(res, "residuals", None)
            if r is not None:
                resid[m] = np.asarray(r, dtype=float)[m]

        tbl = constraint_table(rm, c_full, sc_full, self._lhs_prior_full(), None,
                               lhs_post=cs_full, extra_cols=_ROW_EXTRA_COLS)
        tbl["year"]         = time
        tbl["lhs_prior_t"]  = lhs_t
        tbl["resid_solver"] = resid
        keep = np.isin(time, np.asarray(self._all_years))
        tbl = tbl[keep].reset_index(drop=True)
        tbl["identity"] = self._row_identity(tbl, list(rm.columns))
        tbl["label"]    = self._row_labels(tbl)
        for col in _ROW_CATEGORY_COLS:
            if col in tbl.columns:
                tbl[col] = tbl[col].astype("category")
        self._constraint_frame_cache = tbl
        return tbl

    @staticmethod
    def _row_identity(tbl: pd.DataFrame, row_meta_cols) -> np.ndarray:
        """Factorise the identity columns (every ``row_meta`` column but
        ``constraint_id`` / ``time``); ``-1`` where the tuple recurs within one
        year (the columns cannot single the row out — e.g. set-scoped rows of
        const runs that predate the ``region`` column)."""
        cols = [c for c in row_meta_cols if c not in _ROW_NON_IDENTITY_COLS and c in tbl.columns]
        ident = tbl.groupby(cols, sort=False, dropna=False).ngroup().to_numpy(dtype=np.int64)
        dup = pd.DataFrame({"i": ident, "y": tbl["year"].to_numpy()}) \
                .duplicated(keep=False).to_numpy()
        return np.where(dup, -1, ident)

    @staticmethod
    def _row_labels(tbl: pd.DataFrame) -> np.ndarray:
        """Readable identity label per row, from columns only: node-scoped
        ``variant sector·region valuation``; pair-scoped ``variant
        from.region→to.region num/denom``; set-scoped ``variant region
        valuation``; else ``type variant valuation``."""
        first = tbl.drop_duplicates("identity").set_index("identity")

        def _c(name):
            return _strings(first[name]) if name in first.columns else np.full(len(first), "", dtype=object)

        variant, valuation = _c("variant"), _c("valuation")
        sector, region = _c("sector"), _c("region")
        nsf, nrf, nst, nrt = _c("num_sector_from"), _c("num_region_from"), _c("num_sector_to"), _c("num_region_to")
        nval, dval = _c("num_valuation"), _c("denom_valuation")
        node = (sector != "") & (region != "")
        pair = nsf != ""
        sets = (~node) & (~pair) & (region != "")
        lab = np.select(
            [node, pair, sets],
            [variant + " " + sector + "·" + region + " " + valuation,
             variant + " " + nsf + "." + nrf + "→" + nst + "." + nrt + " " + nval + "/" + dval,
             variant + " " + region + " " + valuation],
            default=_c("type") + " " + variant + " " + valuation,
        )
        lab = np.char.strip(lab.astype(str))
        labels = pd.Series(lab, index=first.index)
        out = tbl["identity"].map(labels).to_numpy()
        # unidentified rows (identity -1) share one label slot; keep their own
        unid = tbl["identity"].to_numpy() < 0
        if unid.any():
            out = out.astype(object)
            out[unid] = "(unidentified) " + _strings(tbl.loc[unid, "constraint_id"])
        return out

    # ------------------------------------------------------------------
    # Per-year tables (CSV schemas)
    # ------------------------------------------------------------------

    def edge_table(self, year: str) -> pd.DataFrame:
        """One row per edge with ``edge_meta.time == year`` (a slice of
        :meth:`edge_frame`, CSV columns only)."""
        ef = self.edge_frame()
        if ef is None:
            raise RuntimeError("edge_table requires an AccountingStructure with "
                               "edge_meta (acc_struct is None or lacks it)")
        cols = [c for c in _EDGE_META_COLS if c in ef.columns] + _EDGE_VALUE_COLS
        mask = (ef["year"] == str(year)).to_numpy()
        return ef.loc[mask, cols].reset_index(drop=True)

    def constraint_table(self, year: str) -> pd.DataFrame:
        """One row per constraint row (hard + soft) active in ``year`` (a slice
        of :meth:`constraint_frame`, CSV columns only)."""
        cf = self.constraint_frame()
        if cf is None:  # row_meta without time/type: keep the direct path (raises as before)
            td = self._timestep_data_for_year(str(year))
            res = self._result_for_year(str(year))
            return constraint_table(self._cr.row_meta, td["c"], td["sigma_c"],
                                    self._lhs_prior_full(), year, lhs_post=res.c_star)
        mask = (cf["year"] == str(year)).to_numpy()
        out = cf.loc[mask, _CONST_CSV_COLS].reset_index(drop=True)
        for col in ("type", "variant", "cone_type", "valuation"):
            if str(out[col].dtype) == "category":
                out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def write(self, odir: str) -> list[str]:
        """Write ``edges_<year>.csv`` + ``constraints_<year>.csv`` per year."""
        os.makedirs(odir, exist_ok=True)
        paths: list[str] = []
        for year in self.years:
            p_edges = os.path.join(odir, f"edges_{year}.csv")
            p_const = os.path.join(odir, f"constraints_{year}.csv")
            self.edge_table(year).to_csv(p_edges, index=False)
            self.constraint_table(year).to_csv(p_const, index=False)
            print(f"   Edges       ({year}) → {p_edges}  ✓")
            print(f"   Constraints ({year}) → {p_const}  ✓")
            paths += [p_edges, p_const]
        return paths

    def write_rankings(self, odir: str, regions=None, suffix: str = "") -> list[str]:
        """Write ``ranking_<family>{suffix}.csv`` — the full identity ranking by
        objective share — for every family whose frame exists."""
        from .objective import FAMILIES, family_frame, rank_identities

        os.makedirs(odir, exist_ok=True)
        paths: list[str] = []
        ef, cf = self.edge_frame(), self.constraint_frame()
        for fam in FAMILIES:
            frame = family_frame(ef, cf, fam, regions=regions)
            if frame is None:
                continue
            ranked = rank_identities(frame)
            p = os.path.join(odir, f"ranking_{fam}{suffix}.csv")
            ranked.to_csv(p, index=False)
            print(f"   ranking ({fam}) → {p}  ✓")
            paths.append(p)
        return paths
