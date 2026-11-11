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
* ``bound_active`` — the edges Stage 5 pinned (σ_a* = 0) because their lower
  bound binds at ``a*``, read from the persisted active set (all ``False``
  where Stage 5 recorded none: the full-G path);
* ``std_shift``    — the **standardised shift** ``z/√κ`` (NaN where κ = 0): under
  the active-set linearisation ``a* − a_prior ~ N(0, σ_a²·κ)``, so it is
  N(0, 1) when σ is right;
* ``calibrated``   — the edge-years the calibration reads: not bound-active,
  ``κ ≥`` :data:`CALIB_KAPPA_MIN` and not of prior class ``flat`` (a prior with
  no information of its own, up to σ_a = 1e6, has ``z ≈ 0`` by construction).

Calibration (#87): :meth:`UncPropStatsDiagnostic.calibration` tabulates the tail
share ``|z/√κ| > 3`` (N(0, 1) expects :data:`CALIB_EXPECTED_SHARE`) and the
quantiles :data:`CALIB_QUANTILES` of ``|z/√κ|`` per valuation × prior class;
:meth:`~UncPropStatsDiagnostic.birge` the **Birge ratio** ``R = Σz²/dof`` per year
and over the run, where ``Σz²`` is the QP objective at ``a*`` (edge defects plus
soft-row residuals, as in SolverDiag's objective decomposition) and ``dof`` the
active-set problem's independent constraints (Stage 5's ``n_rows_kept`` +
``n_scale_totals_left_out`` + the bound-active edges whose prior is off the bound);
:meth:`~UncPropStatsDiagnostic.birge_interval` the Birge-scaled interval
``a* ± z_α·√R·σ_a*`` beside the stored one. On a scale-total ablation run
(``exclude_scale_totals`` in the conditioning) ``z`` comes from the full fit but
``κ`` from the reduced ``S(t)``, so ``z/√κ`` is not N(0, 1) there and the
calibration and the Birge interval are not produced; the Birge ratio still is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from pysuto._conditioning import condition_sigma_c
from pysuto.uncprop import DEFAULT_BIND_TOL, KalmanUncProp

# Quadrant cuts (GLOSSARY.md "Prior-carried / confirmed / overruled / unexplained move").
KAPPA_CUT   = 0.5
Z_CUT       = 3.0

# Calibration of the shifts (#87): below CALIB_KAPPA_MIN, z/√κ is the ratio of two
# numbers near zero; under N(0, 1) a share CALIB_EXPECTED_SHARE lies beyond Z_CUT.
CALIB_KAPPA_MIN       = 0.01
CALIB_EXPECTED_SHARE  = float(2.0 * norm.sf(Z_CUT))
CALIB_QUANTILES       = (0.5, 0.9, 0.99)
_CALIB_EXCLUDED_CLASS = "flat"


def quantile_label(q: float) -> str:
    """The name of the ``q`` quantile (``q50``)."""
    return f"q{q * 100:g}"


def quantile_column(q: float) -> str:
    """The calibration-table column of the ``q`` quantile of ``|z/√κ|`` (``q50_abs``)."""
    return f"{quantile_label(q)}_abs"

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
        self._birge_cache = None
        self.ablation = bool(self._cond.get("exclude_scale_totals", False))
        self._z_alpha = float(norm.ppf(1.0 - float(uncprop_result.alpha) / 2.0))

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

        E       = int(self._cr.G.shape[1])
        a0      = np.asarray(self._cr.a0, dtype=float)
        sigma_a = self._sigma_a()
        classes = self._prior_classes()

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
                "bound_active": (np.asarray(ts.bound_active, dtype=bool)
                                 if getattr(ts, "bound_active", None) is not None
                                 else np.zeros(cols.size, dtype=bool)),
            }))
        f = pd.concat(parts, ignore_index=True)

        s2 = f["sigma_a"].to_numpy() ** 2
        f["kappa"]  = 1.0 - f["sigma_a_star"].to_numpy() ** 2 / s2
        f["z"]      = (f["a_post"] - f["a_prior"]).to_numpy() / f["sigma_a"].to_numpy()
        f["weight"] = np.maximum(f["a_prior"].abs(), f["a_post"].abs())
        cols = f["edge_id"].to_numpy()
        f["prior_class"]  = classes[cols] if classes is not None else "all"
        f["quadrant"]     = quadrant(f["kappa"], f["z"])
        kappa = f["kappa"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            f["std_shift"] = np.where(kappa > 0.0, f["z"].to_numpy() / np.sqrt(kappa), np.nan)
        f["calibrated"] = (~f["bound_active"].to_numpy() & (kappa >= CALIB_KAPPA_MIN)
                           & (f["prior_class"].to_numpy() != _CALIB_EXCLUDED_CLASS)
                           & np.isfinite(f["std_shift"].to_numpy()))

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
    def calibration(frame: pd.DataFrame, birge_ratio: float = float("nan")) -> pd.DataFrame:
        """Calibration of the shifts per valuation × prior class.

        Columns: ``n_edge_years`` (the whole cell), ``n_calibrated`` and their Σ
        ``weight_kt``; over the calibrated edge-years, the share with
        ``|z/√κ| > 3`` (``share_gt3``; N(0, 1) expects :data:`CALIB_EXPECTED_SHARE`),
        the same share weighted by ``max(|a_prior|, |a*|)``, the quantiles
        :data:`CALIB_QUANTILES` (columns :func:`quantile_column`) and ``max_abs`` of
        ``|z/√κ|``, and both shares beyond the Birge-scaled cut ``3·√R`` (NaN
        without ``birge_ratio``). A cell without a calibrated edge-year has NaN
        statistics.
        """
        cut_birge = Z_CUT * np.sqrt(birge_ratio)
        birge = np.isfinite(cut_birge)
        rows = []
        for (val, cls), d in frame.groupby(["valuation", "prior_class"], sort=True,
                                           observed=True):
            c = d[d["calibrated"].to_numpy(dtype=bool)]
            a = np.abs(c["std_shift"].to_numpy(dtype=float))
            w = c["weight"].to_numpy(dtype=float)
            q = (np.quantile(a, CALIB_QUANTILES) if a.size
                 else np.full(len(CALIB_QUANTILES), np.nan))
            rows.append({
                "valuation":                val,
                "prior_class":              cls,
                "n_edge_years":             len(d),
                "n_calibrated":             a.size,
                "weight_kt":                float(w.sum()),
                "share_gt3":                _share(a > Z_CUT),
                "share_gt3_weighted":       _share(a > Z_CUT, w),
                **{quantile_column(qq): v for qq, v in zip(CALIB_QUANTILES, q)},
                "max_abs":                  float(a.max()) if a.size else np.nan,
                "share_gt3_birge":          _share(a > cut_birge) if birge else np.nan,
                "share_gt3_birge_weighted": _share(a > cut_birge, w) if birge else np.nan,
            })
        return pd.DataFrame(rows)

    def birge(self) -> pd.DataFrame:
        """The **Birge ratio** ``R = Σz²/dof`` per year and over the run (row
        ``year == "all"``). Cached.

        ``zsq_edges`` sums ``z²`` over the year's edges (conditioned σ_a, the
        per-period prior the QP used); ``zsq_soft_rows`` sums the squared
        standardised residual ``(G·a* − c)/σ_c`` over its soft rows (``cone_type ==
        "free"``, the QP's slack rows; σ_c conditioned as the QP did). Their sum
        ``zsq_total`` is the QP objective at ``a*``.

        ``dof = n_rows_kept + n_scale_totals_left_out + n_bound_active −
        n_bound_active_at_prior`` from Stage 5's per-year diagnostics, NaN where
        Stage 5 recorded none (the full-G path). An ablation's left-out scale totals
        count, because ``a*`` was fitted with them. A bound-active edge whose prior
        already sits on its bound (``|a_prior − l| ≤ bind_tol``) does not: its
        ``a* = l = a_prior`` is zero by construction, not a draw
        (``n_bound_active_at_prior``). ``z_alpha`` and ``z_alpha_birge =
        z_alpha·√R`` are the half-widths of the raw and the Birge-scaled interval in
        units of ``σ_a*``; the latter is NaN on an ablation run.
        """
        if self._birge_cache is not None:
            return self._birge_cache
        f = self.edge_frame()
        lb = self._lower_bounds()[f["edge_id"].to_numpy()]
        bind_tol = float(self._cond.get("bind_tol", DEFAULT_BIND_TOL))
        with np.errstate(invalid="ignore"):
            at_prior = (f["bound_active"].to_numpy()
                        & (np.abs(f["a_prior"].to_numpy() - lb) <= bind_tol))
        g = f.assign(zsq=f["z"].to_numpy() ** 2, at_prior=at_prior).groupby("year")
        zsq_e, n_e, n_at = g["zsq"].sum(), g.size(), g["at_prior"].sum()
        cone   = self._cr.row_meta["cone_type"].to_numpy()
        tdata  = {str(k): v for k, v in self._cr.timestep_data.items()}
        per_ts = {str(k): v for k, v in self._unc.per_timestep.items()}
        floor  = self._cond.get("sigma_c_floor")
        kw     = {} if floor is None else {"floor": float(floor)}

        rows = []
        for y in self.years:
            res    = self._opt[y]
            rows_t = getattr(res, "rows_t", None)
            n_soft, zsq_r = np.nan, np.nan
            if rows_t is not None:
                rows_t = np.asarray(rows_t, dtype=np.int64)
                soft   = cone[rows_t] == "free"
                sc     = condition_sigma_c(
                    np.asarray(tdata[y]["sigma_c"], dtype=float)[rows_t], **kw)
                r      = np.asarray(res.residuals, dtype=float)[soft] / sc[soft]
                n_soft, zsq_r = int(soft.sum()), float(np.sum(r ** 2))
            d = getattr(per_ts[y], "diagnostics", None) or {}
            rows.append({
                "year":                    y,
                "n_edges":                 int(n_e[y]),
                "zsq_edges":               float(zsq_e[y]),
                "n_soft_rows":             n_soft,
                "zsq_soft_rows":           zsq_r,
                "n_rows_kept":             d.get("n_rows_kept", np.nan),
                "n_scale_totals_left_out": d.get("n_scale_totals_left_out", 0),
                "n_bound_active":          d.get("n_bound_active", np.nan),
                "n_bound_active_at_prior": int(n_at[y]),
            })
        b = pd.DataFrame(rows)
        b.loc[len(b)] = {"year": "all", **b.drop(columns="year").sum(skipna=False).to_dict()}
        b["zsq_total"]     = b["zsq_edges"] + b["zsq_soft_rows"]
        b["dof"]           = (b["n_rows_kept"] + b["n_scale_totals_left_out"]
                              + b["n_bound_active"] - b["n_bound_active_at_prior"])
        b["birge_ratio"]   = b["zsq_total"] / b["dof"]
        b["z_alpha"]       = self._z_alpha
        b["z_alpha_birge"] = (np.nan if self.ablation
                              else self._z_alpha * np.sqrt(b["birge_ratio"]))
        for col in ("n_edges", "n_soft_rows", "n_rows_kept", "n_scale_totals_left_out",
                    "n_bound_active", "n_bound_active_at_prior", "dof"):
            if b[col].notna().all():
                b[col] = b[col].astype(np.int64)
        self._birge_cache = b[["year", "n_edges", "zsq_edges", "n_soft_rows", "zsq_soft_rows",
                               "zsq_total", "n_rows_kept", "n_scale_totals_left_out",
                               "n_bound_active", "n_bound_active_at_prior", "dof",
                               "birge_ratio", "z_alpha", "z_alpha_birge"]]
        return self._birge_cache

    def birge_ratio(self) -> float:
        """The run-level Birge ratio — the ``"all"`` row of :meth:`birge`."""
        b = self.birge()
        return float(b.loc[b["year"] == "all", "birge_ratio"].iloc[0])

    def birge_interval(self, frame: pd.DataFrame, birge_ratio=None) -> pd.DataFrame:
        """The Birge-scaled interval ``a* ± z_α·√R·σ_a*`` for every row of ``frame``
        (columns ``lower_birge`` / ``upper_birge`` on its index).

        ``R`` defaults to the run-level :meth:`birge_ratio`. The lower end clips at
        the edge's lower-bound row ``l`` the way Stage 5 clips the stored one
        (``min(max(·, l), a*)``). ``frame``'s ``lower`` / ``upper`` are not touched.
        Raises on an ablation run, whose ``σ_a*`` is not the fitted estimator's.
        """
        if self.ablation:
            raise ValueError("no Birge interval on a scale-total ablation run: its σ_a* "
                             "comes from the reduced S(t), its a* from the full fit")
        R    = self.birge_ratio() if birge_ratio is None else float(birge_ratio)
        a    = frame["a_post"].to_numpy(dtype=float)
        half = self._z_alpha * np.sqrt(R) * frame["sigma_a_star"].to_numpy(dtype=float)
        lower, upper = a - half, a + half
        lb  = self._lower_bounds()[frame["edge_id"].to_numpy(dtype=np.int64)]
        has = ~np.isnan(lb)
        lower[has] = np.minimum(np.maximum(lower[has], lb[has]), a[has])
        return pd.DataFrame({"lower_birge": lower, "upper_birge": upper}, index=frame.index)

    def _lower_bounds(self) -> np.ndarray:
        """Each edge's lower-bound row ``l`` (NaN without one), by Stage 5's own
        extraction (:meth:`KalmanUncProp._lower_bound_values`) over each year's
        block of the partition; all NaN on the full-G path."""
        out = np.full(int(self._cr.G.shape[1]), np.nan)
        blocks = {str(k): v for k, v in (getattr(self._cr, "G_blocks", None) or {}).items()}
        if not blocks:
            return out
        is_lower = KalmanUncProp._lower_bound_mask(self._cr.row_meta)
        tdata = {str(k): v for k, v in self._cr.timestep_data.items()}
        for y in self.years:
            res = self._opt[y]
            rows_t, cols_t = getattr(res, "rows_t", None), getattr(res, "cols_t", None)
            if rows_t is None or cols_t is None:
                continue
            rows_t = np.asarray(rows_t, dtype=np.int64)
            c_t = np.asarray(tdata[y]["c"], dtype=float)[rows_t]
            out[np.asarray(cols_t, dtype=np.int64)] = KalmanUncProp._lower_bound_values(
                blocks[y].tocsr(), c_t, is_lower[rows_t])
        return out

    @staticmethod
    def contraction_summary(frame: pd.DataFrame) -> pd.DataFrame:
        """Edge count and Σ weight (kt) per year × valuation × prior class × quadrant."""
        g = frame.groupby(["year", "valuation", "prior_class", "quadrant"],
                          sort=True, observed=True)
        out = g.agg(n_edges=("edge_id", "size"),
                    weight_kt=("weight", "sum")).reset_index()
        out["n_edges"] = out["n_edges"].astype(np.int64)
        return out


def _share(mask: np.ndarray, weight=None) -> float:
    """Share of ``True`` in ``mask``, weighted by ``weight`` when given; NaN when
    there is nothing to share."""
    if weight is None:
        return float(mask.mean()) if mask.size else np.nan
    total = float(np.sum(weight))
    return float(np.sum(weight[mask]) / total) if total > 0.0 else np.nan
