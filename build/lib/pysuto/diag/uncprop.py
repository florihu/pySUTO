"""
uncprop.py
==========
``UncPropDiag`` — the Stage-5 (uncprop run) figure portfolio.

Every number is read from the all-years edge frame of
:class:`pysuto.diag.stats.UncPropStatsDiagnostic`. Vocabulary follows GLOSSARY.md:
**posterior contraction** ``κ = 1 − σ_a*²/σ_a²`` (how informative the constraints
are), the standardised **defect** ``z = (a*−a_prior)/σ_a`` (whether they agree
with the prior), **prior class** and the four κ–z quadrants (prior-carried /
confirmed / overruled / unexplained move). The weight of an edge-year is
``max(|a_prior|, |a*|)`` in kt, so deleted and inflated edges both count.

Plot catalogue — :attr:`UncPropDiag.PLOTS` is the registry (ids are stable)
--------------------------------------------------------------------------
* **UD-R-01** ``ud_r_01_contraction_shift`` — κ (x) vs signed z (y, symlog),
  hexbin coloured by Σ weight; rows = prior class, columns = valuation; all
  years pooled (``timesteps=`` narrows them); guides at the quadrant cuts;
  bound-active edges as open circles.
* **UD-R-02** ``ud_r_02_prior_posterior_bands`` — per identity over the years:
  the prior ``a_prior ± z_α·σ_a`` (dashed) and the posterior ``[lower, upper]``
  persisted by Stage 5 (filled) around ``a*``; the top-12 identities by Σ_t
  weight per valuation.
* **UD-R-03** ``ud_r_03_shift_calibration`` — QQ plot of the standardised shift
  ``z/√κ`` against N(0, 1), one panel per prior class × valuation, over the
  calibrated edge-years (not bound-active, κ ≥ 0.01, not flat); reference lines
  N(0, 1) and N(0, R) with the run's Birge ratio R (#87).

``plot()`` also writes ``contraction_summary.csv`` (edge count and Σ weight per
year × valuation × prior class × quadrant), ``calibration.csv`` (tail share and
quantiles of ``|z/√κ|`` per valuation × prior class) and ``birge.csv`` (the Birge
ratio per year and over the run, with the raw and Birge-scaled half-widths);
``plot(regions=[...])`` restricts the figures and the first two CSVs to edges
touching those regions (suffix ``__<region>``). ``birge.csv`` is run-level: the
QP objective and its degrees of freedom do not split by region, so neither
``regions`` nor ``timesteps`` filters it.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ._plot import _CAT8, _INK, _savefig, _seq_cmap
from .solver import PlotSpec, _apply_modern_style, _short, _slug
from .stats._core import filter_regions
from .stats.uncprop import (
    CALIB_EXPECTED_SHARE, CALIB_KAPPA_MIN, CALIB_QUANTILES, KAPPA_CUT, PRIOR_CLASSES,
    QUADRANTS, Z_CUT, UncPropStatsDiagnostic, quantile_column, quantile_label,
)

# Why an ablation run (#87) gets no shift calibration and no Birge interval.
_ABLATION_NOTE  = ("not produced on a scale-total ablation run: z comes from the full "
                   "fit, κ and σ_a* from the reduced S(t)")
_CALIBRATION_ID = "UD-R-03"                      # the figure an ablation run skips

_N_BANDS  = 12                                   # identities per valuation in UD-R-02
_Z_TICKS  = (-1000, -100, -10, -3, 0, 3, 10, 100, 1000)
_HEX_GRID = (30, 24)                             # hexbin cells (κ, z)
_QQ_BULK  = 1500                                 # UD-R-03: bulk points per panel …
_QQ_TAIL  = 500                                  # … plus every point of each tail


def _symlog(z) -> np.ndarray:
    """``sign(z)·log10(1 + |z|)`` — linear-ish near 0, logarithmic in the tails."""
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.log10(1.0 + np.abs(z))


def _qq_index(n: int) -> np.ndarray:
    """Ranks of a sorted sample to draw in a QQ panel: the ``_QQ_TAIL`` smallest
    and largest in full, the bulk thinned to ``_QQ_BULK`` evenly spaced ranks."""
    if n <= _QQ_BULK + 2 * _QQ_TAIL:
        return np.arange(n)
    bulk = np.linspace(_QQ_TAIL, n - _QQ_TAIL - 1, _QQ_BULK).round().astype(np.int64)
    return np.unique(np.r_[np.arange(_QQ_TAIL), bulk, np.arange(n - _QQ_TAIL, n)])


class UncPropDiag:
    """
    Stage-5 posterior diagnostics — renders after ``KalmanUncProp.compute()``.

    Two construction modes:

    * **Live**: pass the ``UncPropResult``, the opt run's results dict and the
      ``ConstraintRun`` directly (a notebook, a test).
    * **Persisted**: ``UncPropDiag.from_uncprop_id(uncprop_run_id)`` reloads the
      result, follows the sidecar's ``parents.upstream`` to the opt and const
      runs and ``const_run.agg_id`` to the imputation run, and re-renders at any
      later time.

    Parameters
    ----------
    uncprop_result : UncPropResult
    opt_results : dict[str, OptimResult]
        The opt run Stage 5 propagated (year-local on the block path).
    constraint_run : ConstraintRun
    label : str
        Figure-title label.
    acc : AccountingStructure | None
        Edge labels and identities. Loaded lazily from ``constraint_run.acc_id``
        when omitted (several GB for ``copper``); without it every edge is
        valuation ``"all"`` and its own identity.
    imp_result : AggregationResult | None
        The imputation run behind the const run (``coverage`` +
        ``imputation_log``) — the source of the prior class. ``None`` → one
        class ``"all"``.
    conditioning : dict | None
        Stage 5's σ_a conditioning (its sidecar ``stage_config``).
    """

    # ---- Plot registry — single source of truth (insertion order = render order) ----
    PLOTS: "dict[str, PlotSpec]" = {
        "UD-R-01": PlotSpec("_fig_contraction_shift", "ud_r_01_contraction_shift",     "Contraction–shift: κ vs signed z per edge-year, hexbin Σ max(|a_prior|,|a*|); rows = prior class, cols = valuation; bound-active edges circled"),
        "UD-R-02": PlotSpec("_fig_bands",             "ud_r_02_prior_posterior_bands", "Prior a_prior ± z_α·σ_a (dashed) vs posterior [lower, upper] (filled) around a* over the years, top-12 identities by Σ_t weight per valuation"),
        "UD-R-03": PlotSpec("_fig_calibration",       "ud_r_03_shift_calibration",     "Shift calibration: QQ of z/√κ vs N(0, 1) per prior class × valuation (not bound-active, κ ≥ 0.01, not flat), with N(0, R) at the run's Birge ratio"),
    }

    def __init__(self, uncprop_result, opt_results, constraint_run, label: str = "",
                 acc=None, imp_result=None, conditioning: Optional[dict] = None) -> None:
        self.label          = label
        self._unc           = uncprop_result
        self._opt           = opt_results
        self._cr            = constraint_run
        self._acc           = acc
        self._acc_resolved  = acc is not None
        self._imp           = imp_result
        self._conditioning  = conditioning
        self._stats_obj: Optional[UncPropStatsDiagnostic] = None
        self._default_odir: Optional[str] = None
        self._cal_cache: dict = {}

    @property
    def opt_results(self) -> dict:
        """The opt run's per-timestep ``OptimResult`` dict this diag was built from."""
        return self._opt

    @property
    def constraint_run(self):
        """The ``ConstraintRun`` this diag was built from."""
        return self._cr

    @property
    def acc(self):
        """The ``AccountingStructure`` (injected or lazily loaded); ``None`` when
        it cannot be loaded (toy fixtures)."""
        if not self._acc_resolved:
            self._acc_resolved = True
            try:
                from pysuto.acc import AccountingStructure
                self._acc = AccountingStructure.load(self._cr.acc_id)
            except Exception:
                self._acc = None
        return self._acc

    @property
    def stats(self) -> UncPropStatsDiagnostic:
        """The stats object owning the all-years edge frame (built once)."""
        if self._stats_obj is None:
            self._stats_obj = UncPropStatsDiagnostic(
                self._unc, self._opt, self._cr, self.acc,
                imp_result=self._imp, conditioning=self._conditioning,
            )
        return self._stats_obj

    # ------------------------------------------------------------------
    # Construction from a persisted run
    # ------------------------------------------------------------------

    @classmethod
    def from_uncprop_id(
        cls,
        uncprop_run_id: str,
        uncprop_dir: str = "data/proc/uncprop",
        opt_dir:     str = "data/proc/opt",
        const_dir:   str = "data/proc/const",
        imp_dir:     str = "data/proc/imp",
        label:       str = "",
        acc=None,
    ) -> "UncPropDiag":
        """Reconstruct the uncprop result, its opt + const runs and the
        imputation run from disk."""
        # Local import to keep diag at the leaf of the import graph (as SolverDiag).
        from pysuto.registry import (
            ConstraintRunRegistry, ImputationRunRegistry, OptimRunRegistry,
            UncPropRunRegistry,
        )

        unc_reg = UncPropRunRegistry(uncprop_dir)
        if uncprop_run_id not in unc_reg._index:
            raise KeyError(
                f"uncprop_run_id '{uncprop_run_id}' not in {unc_reg._index_path}. "
                f"Known: {sorted(unc_reg.run_ids())}"
            )
        sidecar  = unc_reg.load_sidecar(uncprop_run_id)
        upstream = sidecar.get("parents", {}).get("upstream", {})
        opt_run_id, const_run_id = upstream.get("opt_run_id"), upstream.get("const_run_id")
        if not opt_run_id or not const_run_id:
            raise KeyError(
                f"uncprop run '{uncprop_run_id}' sidecar lacks parents.upstream "
                f"opt_run_id / const_run_id (got {upstream})"
            )

        result      = unc_reg.load(uncprop_run_id)
        opt_results = OptimRunRegistry(opt_dir).load(opt_run_id)
        const_run   = ConstraintRunRegistry(const_dir).load(const_run_id)

        imp_result = None
        if const_run.agg_id:
            imp_reg = ImputationRunRegistry(imp_dir)
            if const_run.agg_id in imp_reg._index:
                imp_result = imp_reg.load(const_run.agg_id)

        instance = cls(
            result, opt_results, const_run, label=label or uncprop_run_id, acc=acc,
            imp_result=imp_result, conditioning=sidecar.get("stage_config", {}),
        )
        instance._default_odir = os.path.join(uncprop_dir, uncprop_run_id, "diag")
        return instance

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def _frame(self, regions=None, timesteps=None) -> pd.DataFrame:
        f = filter_regions(self.stats.edge_frame(), ("from_region", "to_region"), regions)
        if timesteps is not None:
            years = {str(t) for t in timesteps}
            unknown = years - set(self.stats.years)
            if unknown:
                raise KeyError(f"timestep(s) {sorted(unknown)} not in the uncprop run "
                               f"(have {self.stats.years})")
            f = f[f["year"].isin(years)]
        return f

    @staticmethod
    def _valuations(frame) -> list[str]:
        """Valuation labels present in a frame (sorted — no label literals)."""
        return sorted(frame["valuation"].dropna().astype(str).unique().tolist())

    @staticmethod
    def _classes(frame) -> list[str]:
        present = set(frame["prior_class"].astype(str))
        return ([c for c in PRIOR_CLASSES if c in present]
                + sorted(present - set(PRIOR_CLASSES)))

    @staticmethod
    def _region_suffix(regions) -> str:
        return "__" + "-".join(_slug(r) for r in regions) if regions else ""

    def _title(self, base: str, regions=None, timesteps=None) -> str:
        t = f"{base}  —  {self.label}" if self.label else base
        if timesteps is not None:
            t += f"  ·  years: {', '.join(str(y) for y in timesteps)}"
        if regions:
            t += f"  ·  regions: {', '.join(str(r) for r in regions)}"
        return t

    def _placeholder(self, text: str, regions=None):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 3))
        fig.suptitle(self._title(text, regions), fontsize=12)
        return fig

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Print and return a text digest: trace contraction per year; the share of
        Σ weight per prior class × quadrant, per valuation; the shift calibration
        ``z/√κ`` per valuation × prior class (one line instead on an ablation run);
        the Birge ratio over the run and per year, with the bound-active edges whose
        prior sits on the bound that ``dof`` leaves out; and the raw beside the
        Birge-scaled interval (not on an ablation run)."""
        f  = self.stats.edge_frame()
        tc = self.stats.trace_contraction(f)
        years = self.stats.years
        lines = [f"UncPropDiag ({self.label or 'unlabeled'})"]
        ablation = self.stats.ablation
        lines.append(f"  years     : {len(years)} ({years[0]}–{years[-1]})  "
                     f"method={getattr(self._unc, 'method', '?')}  "
                     f"alpha={getattr(self._unc, 'alpha', float('nan')):g}"
                     + ("  scale totals left out of S(t) (ablation)" if ablation else ""))
        cells = [f"{y} {v:.3f}" for y, v in tc.items()]
        for k in range(0, len(cells), 6):
            head = "  trace κ   : " if k == 0 else " " * 14
            lines.append(head + "  ".join(cells[k:k + 6]))
        n_ba = int(f["bound_active"].sum())
        lines.append(f"  edge-years: {len(f):,}  bound-active={n_ba:,}"
                     + ("" if self._imp is not None
                        else "  (no imputation run: prior class 'all')"))
        for val in self._valuations(f):
            fv = f[f["valuation"] == val]
            total = float(fv["weight"].sum())
            lines.append(f"  {val} — share of Σ max(|a_prior|,|a*|) = {total:,.1f} kt, "
                         "all years")
            pv = (fv.groupby(["prior_class", "quadrant"], observed=True)["weight"].sum()
                    .unstack("quadrant").reindex(columns=QUADRANTS).fillna(0.0))
            pv = pv.reindex([c for c in self._classes(fv) if c in pv.index])
            pv.loc["total"] = pv.sum(axis=0)
            lines.append("    " + f"{'':<12}" + "".join(f"{q:>18}" for q in QUADRANTS))
            for cls, row in pv.iterrows():
                share = row / total * 100.0 if total > 0 else row * np.nan
                lines.append("    " + f"{cls:<12}" + "".join(f"{s:>17.1f}%" for s in share))
        lines += self._calibration_lines()
        text = "\n".join(lines)
        print(text)
        return text

    def _calibration(self, regions=None, timesteps=None) -> pd.DataFrame:
        """:meth:`UncPropStatsDiagnostic.calibration` of the filtered frame at the
        run's Birge ratio, computed once per filter."""
        key = (tuple(regions) if regions else None,
               tuple(str(t) for t in timesteps) if timesteps is not None else None)
        if key not in self._cal_cache:
            self._cal_cache[key] = self.stats.calibration(self._frame(regions, timesteps),
                                                          self.stats.birge_ratio())
        return self._cal_cache[key]

    def _calibration_lines(self) -> list[str]:
        """Summary lines of the shift calibration (per valuation × prior class) and
        the Birge ratio (run and per year) with the scaled interval (#87)."""
        st = self.stats
        lines: list[str] = []
        if st.ablation:
            lines.append(f"  shift calibration z/√κ, Birge-scaled interval: {_ABLATION_NOTE}")
        else:
            cal = self._calibration()
            expected = " / ".join(f"{quantile_label(q)} {norm.ppf(0.5 + q / 2):.2f}"
                                  for q in CALIB_QUANTILES)
            lines.append(f"  shift calibration z/√κ — not bound-active, κ ≥ {CALIB_KAPPA_MIN:g}, "
                         f"not flat; N(0, 1): {CALIB_EXPECTED_SHARE * 100:.2f} % beyond "
                         f"{Z_CUT:g}, |z/√κ| {expected}")
            head = (f"{'':<12}{'n':>10}{'>3':>9}{'>3 wt':>9}"
                    + "".join(f"{quantile_label(q):>8}" for q in CALIB_QUANTILES)
                    + f"{'max':>10}{'>3·√R':>9}{'>3·√R wt':>10}")
            for val in self._valuations(cal):
                cv = cal[cal["valuation"] == val].set_index("prior_class")
                lines.append(f"    {val}")
                lines.append("    " + head)
                for cls in self._classes(cv.reset_index()):
                    r = cv.loc[cls]
                    lines.append(
                        "    " + f"{cls:<12}{int(r['n_calibrated']):>10,}"
                        + f"{r['share_gt3'] * 100:>8.2f}%{r['share_gt3_weighted'] * 100:>8.2f}%"
                        + "".join(f"{r[quantile_column(q)]:>8.2f}" for q in CALIB_QUANTILES)
                        + f"{r['max_abs']:>10.1f}"
                        + f"{r['share_gt3_birge'] * 100:>8.2f}%"
                        + f"{r['share_gt3_birge_weighted'] * 100:>9.2f}%")

        b = st.birge()
        run = b[b["year"] == "all"].iloc[0]
        R = float(run["birge_ratio"])
        lines.append("  Birge ratio R = Σz²/dof — Σz² the QP objective at a* (edge defects + "
                     "soft-row residuals); dof = rows kept + scale totals left out + "
                     "bound-active edges whose prior is off the bound")
        if not np.isfinite(R):
            lines.append("    run      : R undefined — Stage 5 recorded no active-set counts "
                         "(full-G path)")
            return lines
        lines.append(f"    run      : R = {R:.3f}   Σz² {run['zsq_total']:,.0f} (edges "
                     f"{run['zsq_edges']:,.0f} + soft rows {run['zsq_soft_rows']:,.0f})   "
                     f"dof {run['dof']:,.0f}")
        lines.append(f"    left out : {run['n_bound_active_at_prior']:,.0f} of "
                     f"{run['n_bound_active']:,.0f} bound-active edge-years have their prior "
                     f"on the bound (a* = l = a_prior): no degree of freedom")
        cells = [f"{y} {r:.2f}" for y, r in zip(b["year"], b["birge_ratio"]) if y != "all"]
        for k in range(0, len(cells), 6):
            head = "    per year : " if k == 0 else " " * 15
            lines.append(head + "  ".join(cells[k:k + 6]))
        if not st.ablation:
            alpha = float(getattr(self._unc, "alpha", float("nan")))
            lines.append(f"    {(1.0 - alpha) * 100:g} % interval: raw a* ± "
                         f"{run['z_alpha']:.3f}·σ_a*   Birge-scaled a* ± "
                         f"{run['z_alpha']:.3f}·√R·σ_a* = a* ± {run['z_alpha_birge']:.3f}·σ_a*")
        return lines

    def plot(
        self,
        odir: Optional[str] = None,
        show: bool = False,
        dpi:  int  = 150,
        plot_ids: Optional[list] = None,
        regions:  Optional[list] = None,
        timesteps: Optional[list] = None,
    ) -> list[str]:
        """Render figures from the :attr:`PLOTS` registry plus
        ``contraction_summary.csv``, ``calibration.csv`` and ``birge.csv``. A
        scale-total ablation run gets neither UD-R-03 nor ``calibration.csv``.

        Parameters
        ----------
        odir : str | None
            Output directory. ``None`` → ``{uncprop_dir}/{uncprop_run_id}/diag``
            when built by :meth:`from_uncprop_id`; otherwise required.
        plot_ids : list[str] | None
            Subset of registry ids (rendered in registry order). An unknown id
            raises ``ValueError``.
        regions : list[str] | None
            Restrict to edges touching these regions; filenames get a
            ``__<region>`` suffix. ``birge.csv`` is run-level and never filtered.
        timesteps : list[str] | None
            Years pooled by UD-R-01 / UD-R-03 and counted in the per-edge CSVs
            (default: all). UD-R-02 always spans every year.
        """
        import matplotlib.pyplot as plt
        if odir is None:
            odir = self._default_odir
        if odir is None:
            raise ValueError(
                "odir is required. Either pass odir=..., or build the diag via "
                "UncPropDiag.from_uncprop_id(...) so the default lands under the run dir."
            )
        os.makedirs(odir, exist_ok=True)

        selected = set(self.PLOTS) if plot_ids is None else set(plot_ids)
        unknown = selected - set(self.PLOTS)
        if unknown:
            raise ValueError(f"Unknown plot_ids {sorted(unknown)}. "
                             f"Valid ids: {list(self.PLOTS)}")

        suffix = self._region_suffix(regions)
        paths: list[str] = []
        for pid, spec in self.PLOTS.items():
            if pid not in selected:
                continue
            if pid == _CALIBRATION_ID and self.stats.ablation:
                print(f"   {pid}  {_ABLATION_NOTE}")
                continue
            fig = getattr(self, spec.method)(regions=regions, timesteps=timesteps)
            p = _savefig(fig, os.path.join(odir, f"{spec.filename}{suffix}.png"), dpi=dpi)
            print(f"   {pid}  → {p}  ✓")
            paths.append(p)
            if not show:
                plt.close(fig)

        csv = os.path.join(odir, f"contraction_summary{suffix}.csv")
        self.stats.contraction_summary(self._frame(regions, timesteps)).to_csv(csv, index=False)
        paths.append(csv)
        if not self.stats.ablation:
            cal = os.path.join(odir, f"calibration{suffix}.csv")
            self._calibration(regions, timesteps).to_csv(cal, index=False)
            paths.append(cal)
        birge = os.path.join(odir, "birge.csv")
        self.stats.birge().to_csv(birge, index=False)
        paths.append(birge)
        return paths

    # ------------------------------------------------------------------
    # UD-R-01 — contraction–shift
    # ------------------------------------------------------------------

    def _fig_contraction_shift(self, regions=None, timesteps=None):
        import matplotlib.pyplot as plt
        import matplotlib.transforms as mtransforms
        from matplotlib.colors import LogNorm

        f = self._frame(regions, timesteps)
        base = "UD-R-01  Contraction–shift: κ vs z, colour Σ max(|a_prior|,|a*|)"
        if f.empty:
            return self._placeholder(base + " — no edges", regions)

        classes, vals = self._classes(f), self._valuations(f)
        y_all = _symlog(f["z"].to_numpy())
        y_lim = max(float(np.nanmax(np.abs(y_all))) * 1.05, float(_symlog(10.0)))
        ticks = [v for v in _Z_TICKS if abs(_symlog(v)) <= y_lim]

        fig, axes = plt.subplots(len(classes), len(vals), squeeze=False, sharex=True,
                                 sharey=True, layout="constrained",
                                 figsize=(4.4 * len(vals) + 1.2, 3.0 * len(classes) + 1.6))
        for j, val in enumerate(vals):
            hexes = []
            for i, cls in enumerate(classes):
                ax = axes[i, j]
                d = f[(f["prior_class"] == cls) & (f["valuation"] == val)]
                w = d["weight"].to_numpy()
                pos = w > 0
                if pos.any():
                    hb = ax.hexbin(d["kappa"].to_numpy()[pos],
                                   _symlog(d["z"].to_numpy()[pos]), C=w[pos],
                                   reduce_C_function=np.sum, gridsize=_HEX_GRID,
                                   extent=(0.0, 1.0, -y_lim, y_lim), mincnt=1,
                                   cmap=_seq_cmap(), linewidths=0)
                    hexes.append(hb)
                ba = d[d["bound_active"] & pos]
                if len(ba):
                    ax.scatter(ba["kappa"], _symlog(ba["z"]), s=12, facecolors="none",
                               edgecolors=_INK["primary"], linewidths=0.6,
                               label="bound-active")
                ax.axvline(KAPPA_CUT, color=_INK["muted"], lw=0.8, ls="--")
                for yv in (_symlog(Z_CUT), -_symlog(Z_CUT)):
                    ax.axhline(yv, color=_INK["muted"], lw=0.8, ls="--")
                ax.axhline(0.0, color=_INK["muted"], lw=0.5)
                tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
                for x, yv, txt, ha in ((0.02, y_lim * 0.9, "unexplained move", "left"),
                                       (0.98, y_lim * 0.9, "overruled", "right"),
                                       (0.02, 0.15, "prior-carried", "left"),
                                       (0.98, 0.15, "confirmed", "right")):
                    ax.text(x, yv, txt, transform=tr, ha=ha, va="top", fontsize=7,
                            color=_INK["muted"])
                ax.set_yticks([_symlog(v) for v in ticks])
                ax.set_yticklabels([str(v) for v in ticks])
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-y_lim, y_lim)
                n_ba = int(d["bound_active"].sum())
                ax.set_title(f"{cls} · {val}\n{len(d):,} edge-years · "
                             f"{w.sum():,.0f} kt ({val}) · {n_ba:,} bound-active", fontsize=9)
                if i == len(classes) - 1:
                    ax.set_xlabel("posterior contraction κ")
                if j == 0:
                    ax.set_ylabel("defect z (symlog)")
                _apply_modern_style(ax)
            if hexes:
                vmax = max(float(np.nanmax(h.get_array())) for h in hexes)
                vmin = min(float(np.nanmin(h.get_array())) for h in hexes)
                norm = LogNorm(vmin=max(vmin, vmax * 1e-9), vmax=vmax)
                for h in hexes:
                    h.set_norm(norm)
                cb = fig.colorbar(hexes[0], ax=list(axes[:, j]), location="bottom",
                                  fraction=0.04)
                cb.set_label(f"Σ max(|a_prior|, |a*|) per cell [kt ({val})]")
        handles, labels = [], []
        for ax in axes.ravel():
            h, lab = ax.get_legend_handles_labels()
            if h:
                handles, labels = h[:1], lab[:1]
                break
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
        fig.suptitle(self._title(base, regions, timesteps), fontsize=11)
        return fig

    # ------------------------------------------------------------------
    # UD-R-02 — prior vs posterior bands over time
    # ------------------------------------------------------------------

    def _fig_bands(self, regions=None, timesteps=None):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        f = self._frame(regions)
        base = (f"UD-R-02  Prior vs posterior bands — top {_N_BANDS} identities by "
                f"Σ_t max(|a_prior|,|a*|) per valuation")
        if f.empty:
            return self._placeholder(base + " — no edges", regions)

        z_alpha = float(norm.ppf(1.0 - float(self._unc.alpha) / 2.0))
        years = sorted(f["year"].unique())
        x_of = {y: i for i, y in enumerate(years)}
        ncols = 3

        slots = []                                   # (valuation, identity) or None
        for val in self._valuations(f):
            fv = f[f["valuation"] == val]
            top = fv.groupby("identity")["weight"].sum().nlargest(_N_BANDS).index.tolist()
            slots += [(val, i) for i in top]
            slots += [None] * (-len(top) % ncols)
        nrows = max(1, len(slots) // ncols)

        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(4.6 * ncols, 2.4 * nrows + 1.0))
        for ax, slot in zip(axes.ravel(), slots):
            if slot is None:
                ax.set_visible(False)
                continue
            val, ident = slot
            d = f[(f["valuation"] == val) & (f["identity"] == ident)].sort_values("year")
            x = d["year"].map(x_of).to_numpy()
            ap, post = d["a_prior"].to_numpy(), d["a_post"].to_numpy()
            half = z_alpha * d["sigma_a"].to_numpy()
            ax.fill_between(x, d["lower"].to_numpy(), d["upper"].to_numpy(),
                            color=_CAT8[0], alpha=0.3, lw=0)
            ax.plot(x, post, color=_CAT8[0], lw=1.4)
            ax.plot(x, ap, color=_CAT8[1], lw=1.0, ls=":")
            ax.plot(x, ap - half, color=_CAT8[1], lw=0.9, ls="--")
            ax.plot(x, ap + half, color=_CAT8[1], lw=0.9, ls="--")
            vis = np.concatenate([ap, post, d["lower"].to_numpy(), d["upper"].to_numpy()])
            lo, hi = float(np.nanmin(vis)), float(np.nanmax(vis))
            pad = 0.1 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(-0.5, len(years) - 0.5)
            every = max(1, len(years) // 6)
            ax.set_xticks(range(0, len(years), every))
            ax.set_xticklabels(years[::every], fontsize=7)
            ax.set_title(_short(d["label"].iloc[0], 52), fontsize=8)
            ax.set_ylabel(f"kt ({val})", fontsize=8)
            _apply_modern_style(ax)
        for ax in axes.ravel()[len(slots):]:
            ax.set_visible(False)

        level = f"{(1.0 - float(self._unc.alpha)) * 100:g} %"
        handles = [
            Line2D([0], [0], color=_CAT8[0], lw=1.4, label="posterior a*"),
            Patch(color=_CAT8[0], alpha=0.3, label=f"posterior {level} interval (Stage 5)"),
            Line2D([0], [0], color=_CAT8[1], lw=1.0, ls=":", label="prior a_prior"),
            Line2D([0], [0], color=_CAT8[1], lw=0.9, ls="--",
                   label=f"prior ± z·σ_a ({level}, clipped to the axes)"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 0.995))
        fig.suptitle(self._title(base, regions), fontsize=11, y=1.03)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # UD-R-03 — calibration of the shifts
    # ------------------------------------------------------------------

    def _fig_calibration(self, regions=None, timesteps=None):
        import matplotlib.pyplot as plt

        base = (f"UD-R-03  Shift calibration: QQ of z/√κ vs N(0, 1) — not bound-active, "
                f"κ ≥ {CALIB_KAPPA_MIN:g}, not flat")
        if self.stats.ablation:
            return self._placeholder(f"{base} — {_ABLATION_NOTE}", regions)
        f = self._frame(regions, timesteps)
        c = f[f["calibrated"].to_numpy(dtype=bool)]
        if c.empty:
            return self._placeholder(base + " — no calibrated edge-years", regions)

        R = self.stats.birge_ratio()
        cal = self._calibration(regions, timesteps).set_index(["prior_class", "valuation"])
        classes, vals = self._classes(c), self._valuations(c)
        y_lim = max(float(np.nanmax(np.abs(_symlog(c["std_shift"].to_numpy())))) * 1.05,
                    float(_symlog(10.0)))
        ticks = [v for v in _Z_TICKS if abs(_symlog(v)) <= y_lim]
        n_max = int(c.groupby(["prior_class", "valuation"], observed=True).size().max())
        x_lim = float(norm.ppf(1.0 - 0.5 / n_max)) * 1.05 + 0.2
        xs = np.linspace(-x_lim, x_lim, 200)

        fig, axes = plt.subplots(len(classes), len(vals), squeeze=False, sharex=True,
                                 sharey=True, layout="constrained",
                                 figsize=(4.4 * len(vals) + 1.2, 3.0 * len(classes) + 1.6))
        for i, cls in enumerate(classes):
            for j, val in enumerate(vals):
                ax = axes[i, j]
                y = np.sort(c.loc[(c["prior_class"] == cls) & (c["valuation"] == val),
                                  "std_shift"].to_numpy(dtype=float))
                if y.size:
                    x = norm.ppf((np.arange(1, y.size + 1) - 0.5) / y.size)
                    k = _qq_index(y.size)
                    ax.scatter(x[k], _symlog(y[k]), s=6, color=_CAT8[0], linewidths=0,
                               label="calibrated edge-years")
                ax.plot(xs, _symlog(xs), color=_INK["muted"], lw=1.0, ls="--",
                        label="N(0, 1)")
                if np.isfinite(R):
                    ax.plot(xs, _symlog(np.sqrt(R) * xs), color=_CAT8[1], lw=1.0,
                            label=f"N(0, R), Birge R = {R:.3g}")
                for yv in (_symlog(Z_CUT), -_symlog(Z_CUT)):
                    ax.axhline(yv, color=_INK["muted"], lw=0.6, ls=":")
                ax.set_yticks([_symlog(v) for v in ticks])
                ax.set_yticklabels([str(v) for v in ticks])
                ax.set_xlim(-x_lim, x_lim)
                ax.set_ylim(-y_lim, y_lim)
                if (cls, val) in cal.index and y.size:
                    r = cal.loc[(cls, val)]
                    tail = (f"|z/√κ| > {Z_CUT:g}: {r['share_gt3'] * 100:.2f} % "
                            f"({r['share_gt3_weighted'] * 100:.2f} % of weight)")
                else:
                    tail = "no calibrated edge-years"
                ax.set_title(f"{cls} · {val}\n{y.size:,} edge-years · {tail}", fontsize=9)
                if i == len(classes) - 1:
                    ax.set_xlabel("N(0, 1) quantile")
                if j == 0:
                    ax.set_ylabel("z/√κ (symlog)")
                _apply_modern_style(ax)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        for ax in axes.ravel():
            h, lab = ax.get_legend_handles_labels()
            if len(h) > len(handles):
                handles, labels = h, lab
        fig.legend(handles, labels, loc="outside lower center", ncol=len(handles),
                   fontsize=8, frameon=False)
        fig.suptitle(self._title(base, regions, timesteps)
                     + f"\nN(0, 1) puts {CALIB_EXPECTED_SHARE * 100:.2f} % beyond ±{Z_CUT:g}",
                     fontsize=11)
        return fig
