"""
problem.py
==========
``ProblemStatsDiagnostic`` — the problem one Stage-4 period is handed, as the
frames behind the Results' problem figure (Fig. 3) and its ``numbers.json``.

One chain (opt → const → imp) and one year. Vocabulary follows GLOSSARY.md:

* ``provenance``      — per edge, ``"feed"`` (**from a feed**: a feed covers the
  edge) or the ``imputer_id`` that wrote it; an edge with neither, or with two,
  raises;
* **chain profile**   — kt per domestic ``(from_sector, to_sector)`` pair in
  :data:`CHAIN_ORDER` and per traded commodity (cross-region legs by supplying
  sector), stacked by provenance; zero-flow bars and the signed stock-change
  edges (``l < 0``) are dropped;
* ``family``          — the constraint family, read from the ``row_meta``
  columns (never from a ``constraint_id``);
* ``prior_breach``    — **prior breach**: a hard row the prior does not satisfy,
  ``|G·a_prior − c| > tol`` on a zero-cone row, ``G·a_prior − c > tol`` on a
  nonneg row;
* ``binding``         — the row is in the **active set** (ADR 0044): a nonneg row
  with residual ``G·a* − c ≥ −tol``;
* ``on_flow``         — at least one of the row's edges carries ``|a*| > tol``;
* **region slice**    — a region's columns and the rows touching them, laid out
  in Clarabel's standard form ``[G_free −I; G_zero 0; G_nonneg 0]``.

``tol`` is :data:`~pysuto.uncprop.DEFAULT_BIND_TOL`, the Stage-5 bind
tolerance. A row's edges are its nonzero coefficients: entries ``G`` stores as
zero are ignored everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from pysuto.uncprop import DEFAULT_BIND_TOL as BIND_TOL_KT

# Supply-chain order of the sectors; sinks (FU, Tailings, Slag, WR, stock legs)
# sort after every chain sector.
CHAIN_ORDER = ["CO", "Mining", "Ore", "MF", "SXEW", "Concentrate", "Smelting",
               "Matte", "CFR", "Anode", "ER", "CER", "CSXEW", "CP", "Cathode", "Scrap"]

# Provenance categories in display order; an imputer outside the list is
# appended after them.
PROVENANCE_ORDER = ["feed", "network", "trade", "conc", "ore_grade", "eff",
                    "time_interp", "net_stock_share", "flat_prior"]
# Provenances that give an edge no prior of its own (0 kt seeds).
NO_PRIOR_OF_ITS_OWN = ("net_stock_share", "flat_prior")

FAMILIES = ["mass balance", "stripping", "conc band", "eff band", "scrap-share band",
            "lower bound", "soft ratio", "scale total & scrap anchor"]
CONES = ["free", "zero", "nonneg"]                 # Clarabel's cone order

_BAND_FAMILY = {"conc": "conc band", "eff": "eff band", "scrap_share": "scrap-share band"}
_VALUATION_ORDER = ["Mass", "CC"]


def _position(values: pd.Series, order: list) -> np.ndarray:
    """Each value's index in ``order``; values outside it sort last."""
    rank = {v: i for i, v in enumerate(order)}
    return values.map(rank).fillna(len(order)).to_numpy(dtype=np.int64)


def row_family(row_meta: pd.DataFrame) -> np.ndarray:
    """The constraint family of every row, from ``type`` / ``variant`` /
    ``ratio_kind`` / ``is_hard``. A row matching no family raises ``ValueError``."""
    typ  = row_meta["type"].astype(str).to_numpy()
    var  = row_meta["variant"].astype(str).to_numpy()
    hard = row_meta["is_hard"].to_numpy(dtype=bool)
    kind = row_meta["ratio_kind"].astype(object).to_numpy()

    out = np.full(len(row_meta), "", dtype=object)
    out[typ == "massbal"] = "mass balance"
    out[typ == "bound"]   = "lower bound"
    out[typ == "sum"]     = "scale total & scrap anchor"
    ratio = typ == "ratio"
    strip = var == "strippingratio"
    out[ratio & strip]          = "stripping"
    out[ratio & ~hard & ~strip] = "soft ratio"
    for k, fam in _BAND_FAMILY.items():
        out[ratio & hard & ~strip & (kind == k)] = fam

    bad = out == ""
    if bad.any():
        first = row_meta.loc[bad, ["type", "variant", "is_hard"]].head(3).to_dict("records")
        raise ValueError(f"{int(bad.sum())} row(s) match no constraint family "
                         f"(first: {first})")
    return out


@dataclass
class RegionSlice:
    """A region's slice of one period's QP in standard form.

    ``A`` is ``[G_slice | S]``: the slice's rows over its edge columns, then the
    slack block ``S`` (``−I`` on the free rows, which come first). Row arrays
    follow ``row_ids``; column arrays follow ``edge_ids``.
    """
    region:          str
    year:            str
    A:               sp.csr_matrix
    edge_ids:        np.ndarray          # global edge ids, column order
    valuation:       np.ndarray
    is_leg:          np.ndarray          # bool: cross-region leg
    row_ids:         np.ndarray          # global row ids, row order
    family:          np.ndarray
    cone:            np.ndarray
    complete:        np.ndarray          # bool: every nonzero of the row is in the slice
    binding_on_flow: np.ndarray          # bool
    n_edges:         int
    n_slack:         int
    n_legs_omitted:  int                 # the region's legs with no prior of their own


class ProblemStatsDiagnostic:
    """The problem one Stage-4 period is handed, as frames.

    Parameters
    ----------
    opt_result : OptimResult
        The year's result on the block path: ``F_opt``, ``a_prior`` and
        ``residuals`` year-local over ``cols_t`` / ``rows_t``.
    constraint_run : ConstraintRun
        Supplies ``G_blocks[year]`` / ``edge_cols_by_t[year]``, ``row_meta``,
        ``timestep_data[year]["c"]`` (over all rows) and the structural lower
        bound ``l``.
    imp_result : AggregationResult
        The imputation run behind the const run: ``f_agg``, ``coverage`` and
        ``imputation_log.as_dataframe()``.
    acc_struct : AccountingStructure
        ``edge_meta`` (endpoints, time, valuation) for every global edge.
    year : str
        The period, a time label.
    """

    def __init__(self, opt_result, constraint_run, imp_result, acc_struct,
                 year: str) -> None:
        if not isinstance(year, str):
            raise TypeError(f"year is a time label (str), got "
                            f"{type(year).__name__} {year!r}")
        self.year = year
        self._opt = opt_result
        self._cr  = constraint_run
        self._imp = imp_result
        self._em  = acc_struct.edge_meta.reset_index(drop=True)
        self._prov_cache  = None
        self._edge_cache  = None
        self._row_cache   = None

    @classmethod
    def from_opt_id(
        cls,
        opt_run_id: str,
        year:       str,
        opt_dir:    str = "data/proc/opt",
        const_dir:  str = "data/proc/const",
        imp_dir:    str = "data/proc/imp",
        acc_dir:    str = "data/proc/acc",
    ) -> "ProblemStatsDiagnostic":
        """Follow the opt run's ``parents.upstream`` to its const and imp runs
        and load the year from disk."""
        # Local import to keep diag at the leaf of the import graph (as SolverDiag).
        from pysuto.acc import AccountingStructure
        from pysuto.registry import (
            ConstraintRunRegistry, ImputationRunRegistry, OptimRunRegistry,
        )

        opt_reg  = OptimRunRegistry(opt_dir)
        upstream = opt_reg.load_sidecar(opt_run_id).get("parents", {}).get("upstream", {})
        const_run_id = upstream.get("const_run_id")
        if not const_run_id:
            raise KeyError(f"opt run '{opt_run_id}' sidecar lacks "
                           "parents.upstream.const_run_id")
        const_run = ConstraintRunRegistry(const_dir).load(const_run_id)
        if not const_run.agg_id:
            raise KeyError(f"const run '{const_run_id}' records no imp run "
                           "(parents.upstream)")
        imp_result = ImputationRunRegistry(imp_dir).load(const_run.agg_id)
        acc = AccountingStructure.load(id=const_run.acc_id, storage_dir=acc_dir)
        results = opt_reg.load(opt_run_id)
        if year not in results:
            raise KeyError(f"opt run '{opt_run_id}' has no period {year!r}")
        return cls(results[year], const_run, imp_result, acc, year)

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def provenance(self) -> np.ndarray:
        """``"feed"`` or the ``imputer_id`` of every global edge. Cached."""
        if self._prov_cache is not None:
            return self._prov_cache
        coverage = np.asarray(self._imp.coverage, dtype=bool)
        log = self._imp.imputation_log.as_dataframe()
        ids = log["edge_id"].to_numpy(dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= coverage.size):
            raise ValueError(f"imputation_log edge_id outside [0, {coverage.size})")
        dup = pd.Index(ids).duplicated()
        if dup.any():
            raise ValueError(f"{int(dup.sum())} edge(s) imputed more than once "
                             f"(first: {ids[dup][:5].tolist()})")
        both = coverage[ids]
        if both.any():
            raise ValueError(f"{int(both.sum())} edge(s) are both feed-covered and imputed "
                             f"(first: {ids[both][:5].tolist()})")
        out = np.full(coverage.size, "", dtype=object)
        out[coverage] = "feed"
        out[ids] = log["imputer_id"].astype(str).to_numpy()
        missing = out == ""
        if missing.any():
            raise ValueError(f"{int(missing.sum())} edge(s) are neither feed-covered nor "
                             f"imputed (first: {np.flatnonzero(missing)[:5].tolist()})")
        self._prov_cache = out
        return out

    def _cols(self) -> np.ndarray:
        cols  = np.asarray(self._opt.cols_t, dtype=np.int64)
        block = np.asarray(self._cr.edge_cols_by_t[self.year], dtype=np.int64)
        if not np.array_equal(cols, block):
            raise ValueError(f"year {self.year}: the opt result's cols_t and the "
                             "constraint run's block columns differ — the runs do "
                             "not share one partition")
        return cols

    def edge_frame(self) -> pd.DataFrame:
        """One row per edge of the year, in ``cols_t`` order. Cached."""
        if self._edge_cache is not None:
            return self._edge_cache
        cols = self._cols()
        em = self._em.iloc[cols]
        f = pd.DataFrame({
            "edge_id":     cols,
            "valuation":   em["valuation"].astype(str).to_numpy(),
            "from_sector": em["from_sector"].astype(str).to_numpy(),
            "from_region": em["from_region"].astype(str).to_numpy(),
            "to_sector":   em["to_sector"].astype(str).to_numpy(),
            "to_region":   em["to_region"].astype(str).to_numpy(),
            "f":           np.asarray(self._imp.f_agg, dtype=float)[cols],
            "provenance":  self.provenance()[cols],
            "signed":      np.asarray(self._cr.l, dtype=float)[cols] < 0,
        })
        f["leg"] = f["from_region"] != f["to_region"]
        self._edge_cache = f
        return f

    @staticmethod
    def _categories(present) -> list:
        return PROVENANCE_ORDER + sorted(set(present) - set(PROVENANCE_ORDER))

    def provenance_totals(self, valuation: str = "CC") -> pd.DataFrame:
        """Edge count and Σ ``f_agg`` per provenance category for the year, in
        kt of ``valuation`` (kt CC for ``"CC"``)."""
        e = self.edge_frame()
        e = e[e["valuation"] == valuation]
        g = e.groupby("provenance")["f"].agg(n_edges="size", kt="sum")
        g = g.reindex(self._categories(g.index)).fillna(0.0)
        g["n_edges"] = g["n_edges"].astype(np.int64)
        g.index.name = "provenance"
        return g

    def feed_edges_by_year(self, valuation: str = "CC") -> pd.Series:
        """Feed-covered edges per year over the whole run (the caption's range)."""
        cov = np.asarray(self._imp.coverage, dtype=bool)
        m = cov & (self._em["valuation"].astype(str).to_numpy() == valuation)
        years = pd.Series(self._em["time"].astype(str).to_numpy()[m])
        return years.value_counts().sort_index()

    def chain_profile(self, valuation: str = "CC") -> pd.DataFrame:
        """Σ ``f_agg`` (kt of ``valuation``) per domestic sector pair in chain
        order and per traded commodity, one column per provenance category,
        plus ``kind``."""
        e = self.edge_frame()
        e = e[(e["valuation"] == valuation) & ~e["signed"]].copy()
        leg = e["leg"].to_numpy()
        e["bar"]    = np.where(leg, e["from_sector"] + " (trade)",
                               e["from_sector"] + "→" + e["to_sector"])
        e["kind"]   = np.where(leg, "legs", "domestic")
        e["k_kind"] = leg.astype(np.int64)
        e["k_from"] = _position(e["from_sector"], CHAIN_ORDER)
        e["k_to"]   = np.where(leg, 0, _position(e["to_sector"], CHAIN_ORDER))
        e["to_key"] = np.where(leg, "", e["to_sector"])
        p = e.pivot_table(index="bar", columns="provenance", values="f", aggfunc="sum",
                          fill_value=0.0)
        p = p[[c for c in self._categories(p.columns) if c in p.columns]]
        keys = e.drop_duplicates("bar").set_index("bar")
        p = p.join(keys[["kind", "k_kind", "k_from", "k_to", "to_key"]])
        p = p[p[[c for c in p.columns if c in set(e["provenance"])]].abs().sum(axis=1) > 0]
        p = p.sort_values(["k_kind", "k_from", "k_to", "to_key"], kind="mergesort")
        p.index.name = "bar"
        return p.drop(columns=["k_kind", "k_from", "k_to", "to_key"])

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    def _G(self) -> sp.csr_matrix:
        return self._cr.G_blocks[self.year].tocsr()

    def _pattern(self) -> sp.csr_matrix:
        """The block's true nonzeros as a 0/1 matrix (stored zeros dropped)."""
        return (self._G() != 0).astype(np.int64).tocsr()

    def row_frame(self) -> pd.DataFrame:
        """One row per constraint row of the year, in ``rows_t`` order. Cached."""
        if self._row_cache is not None:
            return self._row_cache
        cols = self._cols()                                    # validates the partition
        rows = np.asarray(self._opt.rows_t, dtype=np.int64)
        G = self._G()
        if G.shape != (rows.size, cols.size):
            raise ValueError(f"year {self.year}: block {G.shape} ≠ rows_t × cols_t "
                             f"({rows.size} × {cols.size})")
        rm = self._cr.row_meta.iloc[rows].reset_index(drop=True)
        c = np.asarray(self._cr.timestep_data[self.year]["c"], dtype=float)
        if c.size != len(self._cr.row_meta):
            raise ValueError(f"year {self.year}: c has {c.size} entries, row_meta "
                             f"{len(self._cr.row_meta)} rows — c is stored over all rows")
        r_prior = G @ np.asarray(self._opt.a_prior, dtype=float) - c[rows]
        resid   = np.asarray(self._opt.residuals, dtype=float)
        F       = np.asarray(self._opt.F_opt, dtype=float)

        cone = rm["cone_type"].astype(str).to_numpy()
        zero, nonneg = cone == "zero", cone == "nonneg"
        flowing = (np.abs(F) > BIND_TOL_KT).astype(np.int64)
        on_flow = (self._pattern() @ flowing) > 0
        binding = nonneg & (resid >= -BIND_TOL_KT)
        f = pd.DataFrame({
            "row_id":       rows,
            "family":       row_family(rm),
            "cone":         cone,
            "is_hard":      rm["is_hard"].to_numpy(dtype=bool),
            "prior_resid":  r_prior,
            "resid":        resid,
            "prior_breach": np.where(zero, np.abs(r_prior) > BIND_TOL_KT,
                                     nonneg & (r_prior > BIND_TOL_KT)),
            "binding":      binding,
            "on_flow":      on_flow,
        })
        f["binding_on_flow"]       = f["binding"] & f["on_flow"]
        f["binding_on_zero_flows"] = f["binding"] & ~f["on_flow"]
        self._row_cache = f
        return f

    def family_summary(self) -> pd.DataFrame:
        """Rows per family with its cone and the prior-breach / binding counts.

        ``prior_breach`` is NA on soft (free) families, the binding counts are NA
        unless the family is nonneg: an equality always holds at ``a*``.
        """
        r = self.row_frame()
        g = r.groupby("family").agg(
            cone=("cone", "first"), n_cones=("cone", "nunique"), rows=("row_id", "size"),
            prior_breach=("prior_breach", "sum"), binding=("binding", "sum"),
            binding_on_flow=("binding_on_flow", "sum"),
            binding_on_zero_flows=("binding_on_zero_flows", "sum"))
        if (g["n_cones"] > 1).any():
            raise ValueError(f"families spanning several cones: "
                             f"{g.index[g.n_cones > 1].tolist()}")
        g = g.drop(columns="n_cones").reindex([x for x in FAMILIES if x in g.index])
        counts = ["prior_breach", "binding", "binding_on_flow", "binding_on_zero_flows"]
        g[counts] = g[counts].astype("Int64")
        g.loc[g["cone"] == "free", "prior_breach"] = pd.NA
        g.loc[g["cone"] != "nonneg", counts[1:]] = pd.NA
        g["rows"] = g["rows"].astype(np.int64)
        return g

    def block_sizes(self) -> dict:
        """The year's block: rows (per cone), edges, true and stored nonzeros."""
        r = self.row_frame()
        G = self._G()
        return {"rows": int(G.shape[0]), "edges": int(G.shape[1]),
                "rows_by_cone": {k: int((r["cone"] == k).sum()) for k in CONES},
                "nnz": int(self._pattern().nnz), "nnz_stored": int(G.nnz)}

    # ------------------------------------------------------------------
    # Region slice
    # ------------------------------------------------------------------

    def region_slice(self, region: str) -> RegionSlice:
        """The region's domestic edges and its legs with a prior of their own
        (provenance not in :data:`NO_PRIOR_OF_ITS_OWN`), with every row that has
        a nonzero on them, in standard form."""
        e, r = self.edge_frame(), self.row_frame()
        touch = (e["from_region"] == region) | (e["to_region"] == region)
        no_prior = e["provenance"].isin(NO_PRIOR_OF_ITS_OWN)
        keep = touch & (~e["leg"] | ~no_prior)
        n_omitted = int((touch & e["leg"] & no_prior).sum())

        k = e[keep].copy()
        k["j"] = np.flatnonzero(keep.to_numpy())
        out_leg = (k["from_region"] == region).to_numpy()
        leg = k["leg"].to_numpy()
        k["k_val"]  = _position(k["valuation"], _VALUATION_ORDER)
        k["k_leg"]  = leg.astype(np.int64)
        k["k_from"] = _position(k["from_sector"], CHAIN_ORDER)
        # legs: imports, then exports
        k["k_to"]   = np.where(leg, out_leg.astype(np.int64),
                               _position(k["to_sector"], CHAIN_ORDER))
        k["k_name"] = np.where(leg, np.where(out_leg, k["to_region"], k["from_region"]),
                               k["to_sector"])
        k = k.sort_values(["k_val", "k_leg", "k_from", "k_to", "k_name", "edge_id"],
                          kind="mergesort")
        jcols = k["j"].to_numpy()

        P = self._pattern()
        n_in  = np.asarray(P[:, jcols].sum(axis=1)).ravel()
        n_all = np.asarray(P.sum(axis=1)).ravel()
        i_rows = np.flatnonzero(n_in > 0)
        sel = r.iloc[i_rows].assign(i=i_rows, complete=(n_in == n_all)[i_rows])
        sel["k_cone"] = _position(sel["cone"], CONES)
        sel["k_fam"]  = _position(sel["family"], FAMILIES)
        sel["k_part"] = (~sel["complete"]).astype(np.int64)
        sel = sel.sort_values(["k_cone", "k_fam", "k_part", "i"], kind="mergesort")

        G_slice = self._G()[sel["i"].to_numpy()][:, jcols].tocsr()
        G_slice.eliminate_zeros()
        n_free = int((sel["cone"] == "free").sum())
        S = sp.csr_matrix((-np.ones(n_free), (np.arange(n_free), np.arange(n_free))),
                          shape=(len(sel), n_free))
        return RegionSlice(
            region=region, year=self.year, A=sp.hstack([G_slice, S], format="csr"),
            edge_ids=k["edge_id"].to_numpy(dtype=np.int64),
            valuation=k["valuation"].to_numpy(), is_leg=k["leg"].to_numpy(dtype=bool),
            row_ids=sel["row_id"].to_numpy(dtype=np.int64),
            family=sel["family"].to_numpy(), cone=sel["cone"].to_numpy(),
            complete=sel["complete"].to_numpy(dtype=bool),
            binding_on_flow=sel["binding_on_flow"].to_numpy(dtype=bool),
            n_edges=int(len(k)), n_slack=n_free, n_legs_omitted=n_omitted,
        )
