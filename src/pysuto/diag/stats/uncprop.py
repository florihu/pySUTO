"""
uncprop.py
==========
``UncPropStatsDiagnostic`` — the all-years edge frame behind the Stage-5
``UncPropDiag`` figures, and its per-quadrant summary table.

One row per edge × year of a persisted ``UncPropResult``, joined to the opt run's
``a*`` and prior and to the const run's σ_a. Vocabulary follows GLOSSARY.md:

* ``sigma_a``      — the prior σ **conditioned** with the rules Stage 5 (and the
  QP) applied (:func:`pysuto._conditioning.condition_prior`), so ``kappa`` and
  ``z`` are the numbers the posterior was computed from;
* ``kappa``        — **posterior contraction** ``1 − σ_a*² / σ_a²``;
* ``z``            — the standardised **defect** ``(a* − a_prior) / σ_a``;
* ``weight``       — ``max(|a_prior|, |a*|)`` in kt: keeps both a deleted edge
  (``a* ≈ 0``) and an inflated zero seed (``a_prior = 0``) visible;
* ``prior_class``  — **prior class** (from a feed / imputed / flat / zero-pinned),
  ``"all"`` when no imputation run is supplied;
* ``quadrant``     — prior-carried / confirmed / overruled / unexplained move,
  cut at :data:`KAPPA_CUT` and :data:`Z_CUT`;
* ``bound_active`` — the edge has an ADR 0039 lower-bound row and ``a* ≤`` the
  run's ``bind_tol`` — exactly the edges Stage 5 pinned (σ_a* = 0) in its
  active set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Quadrant cuts (GLOSSARY.md "Prior-carried / confirmed / overruled / unexplained move").
KAPPA_CUT   = 0.5
Z_CUT       = 3.0

QUADRANTS     = ["prior-carried", "confirmed", "overruled", "unexplained move"]
PRIOR_CLASSES = ["from a feed", "imputed", "flat", "zero-pinned"]

# imputation_log rows whose prior carries no information of its own: the flat
# prior (anchored or not) and the fabricated waste-rock satellite (ADR 0038).
_FLAT_IMPUTER_IDS = ("flat_prior",)
_FLAT_METHODS     = ("network_stripping",)

_EDGE_META_COLS = ["from_id", "to_id", "from_sector", "from_region", "from_entity",
                   "to_sector", "to_region", "to_entity", "valuation"]


def classify_priors(coverage, imputation_log: pd.DataFrame, l, u) -> np.ndarray:
    """The **prior class** of every edge, as an ``(E,)`` object array.

    Precedence: from a feed (``coverage``) → zero-pinned (a no-data edge with
    ``l == u == 0``) → flat (``imputer_id`` in the flat set or ``method`` in the
    satellite set) → imputed (any other ``imputation_log`` entry). An edge none
    of these reach raises ``ValueError``: the four classes cover every edge.
    """
    coverage = np.asarray(coverage, dtype=bool)
    l, u = np.asarray(l, dtype=float), np.asarray(u, dtype=float)
    E = coverage.size
    out = np.full(E, "", dtype=object)

    log = imputation_log.drop_duplicates("edge_id", keep="last")
    ids = log["edge_id"].to_numpy(dtype=np.int64)
    if ids.size and (ids.min() < 0 or ids.max() >= E):
        raise ValueError(f"imputation_log edge_id outside [0, {E})")
    flat = (log["imputer_id"].isin(_FLAT_IMPUTER_IDS)
            | log["method"].isin(_FLAT_METHODS)).to_numpy()
    out[ids[~flat]] = "imputed"
    out[ids[flat]]  = "flat"
    out[(l == 0.0) & (u == 0.0) & ~coverage] = "zero-pinned"
    out[coverage]   = "from a feed"

    missing = int((out == "").sum())
    if missing:
        raise ValueError(
            f"{missing} edge(s) are neither feed-covered, imputed nor zero-pinned — "
            "the prior classes must cover every edge"
        )
    return out


def quadrant(kappa, z) -> np.ndarray:
    """The κ–z quadrant of each edge-year (``""`` where κ or z is NaN)."""
    kappa, az = np.asarray(kappa, dtype=float), np.abs(np.asarray(z, dtype=float))
    informed, moved = kappa >= KAPPA_CUT, az > Z_CUT
    out = np.where(informed, np.where(moved, "overruled", "confirmed"),
                             np.where(moved, "unexplained move", "prior-carried"))
    out = out.astype(object)
    out[np.isnan(kappa) | np.isnan(az)] = ""
    return out


def _lower_bound_cols(constraint_run, lower_bound_mask) -> np.ndarray:
    """Global edge columns that carry an ADR 0039 lower-bound row (the rows of
    ``lower_bound_mask``, each with a single nonzero in ``G``)."""
    rows = np.flatnonzero(lower_bound_mask)
    if rows.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.unique(constraint_run.G.tocsr()[rows].tocoo().col).astype(np.int64)


class UncPropStatsDiagnostic:
    """The all-years Stage-5 edge frame and its quadrant summary.

    Parameters
    ----------
    uncprop_result : UncPropResult
        ``per_timestep[t]`` carries ``sigma_a_star`` / ``lower`` / ``upper`` —
        year-local over ``cols_t`` (block path) or full-length (legacy).
    opt_results : dict[timestep -> OptimResult]
        The opt run Stage 5 propagated: ``F_opt`` and ``a_prior`` (year-local
        over ``cols_t`` on the block path).
    constraint_run : ConstraintRun
        Supplies ``a0`` / ``sigma_a`` (conditioned here), ``l`` / ``u``,
        ``row_meta`` and ``G`` (lower-bound rows).
    acc_struct : optional
        ``AccountingStructure`` — ``edge_meta`` (endpoints, valuation) and
        ``edge_template_code`` (cross-year identity). ``None`` → valuation
        ``"all"`` and the edge column as identity.
    imp_result : optional
        The imputation run behind the const run — ``coverage`` and
        ``imputation_log.as_dataframe()``. ``None`` → prior class ``"all"``.
    conditioning : dict, optional
        ``sigma_a_floor_abs`` / ``sigma_a_floor_rel`` / ``sigma_a_ceiling_factor``
        as Stage 5 used them (the uncprop sidecar's ``stage_config``); missing
        keys take :func:`condition_prior`'s defaults.
    """

    def __init__(self, uncprop_result, opt_results, constraint_run, acc_struct=None,
                 imp_result=None, conditioning=None) -> None:
        self._unc   = uncprop_result
        self._opt   = {str(k): v for k, v in opt_results.items()}
        self._cr    = constraint_run
        self._acc   = acc_struct
        self._imp   = imp_result
        self._cond  = dict(conditioning or {})
        self.years  = [str(t) for t in uncprop_result.per_timestep]
        missing = [y for y in self.years if y not in self._opt]
        if missing:
            raise KeyError(f"opt results lack the uncprop year(s) {missing}")
        self._frame_cache = None

    # ------------------------------------------------------------------
    # Edge frame
    # ------------------------------------------------------------------

    def _sigma_a(self) -> np.ndarray:
        # Local import to keep diag at the leaf of the import graph (as SolverDiag).
        from pysuto._conditioning import condition_prior
        kw = {"floor_abs":      self._cond.get("sigma_a_floor_abs"),
              "floor_rel":      self._cond.get("sigma_a_floor_rel"),
              "ceiling_factor": self._cond.get("sigma_a_ceiling_factor")}
        _, s = condition_prior(self._cr.a0, self._cr.sigma_a,
                               **{k: float(v) for k, v in kw.items() if v is not None})
        return s

    def _prior_classes(self):
        if self._imp is None:
            return None
        return classify_priors(self._imp.coverage,
                               self._imp.imputation_log.as_dataframe(),
                               self._cr.l, self._cr.u)

    def edge_frame(self) -> pd.DataFrame:
        """One row per edge × uncprop year. Cached."""
        if self._frame_cache is not None:
            return self._frame_cache

        # Local import to keep diag at the leaf of the import graph (as SolverDiag):
        # Stage 5's own lower-bound rule and default tolerance, so the flag marks
        # exactly the edges its active set pinned.
        from pysuto.uncprop import DEFAULT_BIND_TOL, KalmanUncProp

        E        = int(self._cr.G.shape[1])
        a0       = np.asarray(self._cr.a0, dtype=float)
        sigma_a  = self._sigma_a()
        classes  = self._prior_classes()
        bind_tol = float(self._cond.get("bind_tol", DEFAULT_BIND_TOL))
        has_lb   = np.zeros(E, dtype=bool)
        has_lb[_lower_bound_cols(self._cr,
                                 KalmanUncProp._lower_bound_mask(self._cr.row_meta))] = True

        parts = []
        for t, ts in self._unc.per_timestep.items():
            y   = str(t)
            res = self._opt[y]
            cols = (np.asarray(ts.cols_t, dtype=np.int64) if ts.cols_t is not None
                    else np.arange(E, dtype=np.int64))
            year_local = getattr(res, "cols_t", None) is not None
            if year_local and not np.array_equal(np.asarray(res.cols_t), cols):
                raise ValueError(
                    f"year {y}: the opt result's cols_t and the uncprop timestep's "
                    "cols_t differ — the two runs do not share one partition"
                )
            F = np.asarray(res.F_opt, dtype=float)
            a_post = F if year_local else F[cols]
            ap = getattr(res, "a_prior", None)
            if ap is None:
                a_prior = a0[cols]
            else:
                ap = np.asarray(ap, dtype=float)
                a_prior = ap if year_local else ap[cols]
            parts.append(pd.DataFrame({
                "edge_id":     cols,
                "year":         y,
                "a_prior":      a_prior,
                "a_post":       a_post,
                "sigma_a":      sigma_a[cols],
                "sigma_a_star": np.asarray(ts.sigma_a_star, dtype=float),
                "lower":        np.asarray(ts.lower, dtype=float),
                "upper":        np.asarray(ts.upper, dtype=float),
            }))
        f = pd.concat(parts, ignore_index=True)

        s2 = f["sigma_a"].to_numpy() ** 2
        f["kappa"]  = 1.0 - f["sigma_a_star"].to_numpy() ** 2 / s2
        f["z"]      = (f["a_post"] - f["a_prior"]).to_numpy() / f["sigma_a"].to_numpy()
        f["weight"] = np.maximum(f["a_prior"].abs(), f["a_post"].abs())
        cols = f["edge_id"].to_numpy()
        f["prior_class"]  = classes[cols] if classes is not None else "all"
        f["quadrant"]     = quadrant(f["kappa"], f["z"])
        f["bound_active"] = has_lb[cols] & (f["a_post"].to_numpy() <= bind_tol)

        em = getattr(self._acc, "edge_meta", None) if self._acc is not None else None
        if em is not None:
            em = em.reset_index(drop=True)
            for c in _EDGE_META_COLS:
                if c in em.columns:
                    f[c] = em[c].to_numpy()[cols]
            code = getattr(self._acc, "edge_template_code", None)
            f["identity"] = (np.asarray(code, dtype=np.int64)[cols]
                             if code is not None and len(code) == len(em) else cols)
        else:
            f["identity"] = cols
        if "valuation" not in f.columns:
            f["valuation"] = "all"
        f["label"] = self._labels(f)
        self._frame_cache = f
        return f

    @staticmethod
    def _labels(f: pd.DataFrame) -> np.ndarray:
        """``from_sector|from_region→to_sector|to_region|valuation`` (or the edge column)."""
        need = ["from_sector", "from_region", "to_sector", "to_region"]
        if not set(need) <= set(f.columns):
            return ("edge " + f["edge_id"].astype(str)).to_numpy()
        s = {c: f[c].astype(str) for c in need + ["valuation"]}
        return (s["from_sector"] + "|" + s["from_region"] + "→" + s["to_sector"] + "|"
                + s["to_region"] + "|" + s["valuation"]).to_numpy()

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    @staticmethod
    def trace_contraction(frame: pd.DataFrame) -> pd.Series:
        """``1 − Σσ_a*² / Σσ_a²`` per year over the frame's edges."""
        g = frame.assign(s2=frame["sigma_a"] ** 2, s2_star=frame["sigma_a_star"] ** 2) \
                 .groupby("year", sort=True)[["s2", "s2_star"]].sum()
        return 1.0 - g["s2_star"] / g["s2"]

    @staticmethod
    def contraction_summary(frame: pd.DataFrame) -> pd.DataFrame:
        """Edge count and Σ weight (kt) per year × valuation × prior class × quadrant."""
        g = frame.groupby(["year", "valuation", "prior_class", "quadrant"],
                          sort=True, observed=True)
        out = g.agg(n_edges=("edge_id", "size"),
                    weight_kt=("weight", "sum")).reset_index()
        out["n_edges"] = out["n_edges"].astype(np.int64)
        return out
