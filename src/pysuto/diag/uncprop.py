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

``plot()`` also writes ``contraction_summary.csv`` (edge count and Σ weight per
year × valuation × prior class × quadrant); ``plot(regions=[...])`` restricts both
figures and the CSV to edges touching those regions (suffix ``__<region>``).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from ._plot import _CAT8, _INK, _savefig, _seq_cmap
from .solver import PlotSpec, _apply_modern_style, _short, _slug
from .stats._core import filter_regions
from .stats.uncprop import (
    KAPPA_CUT, PRIOR_CLASSES, QUADRANTS, Z_CUT, UncPropStatsDiagnostic,
)

_N_BANDS  = 12                                   # identities per valuation in UD-R-02
_Z_TICKS  = (-1000, -100, -10, -3, 0, 3, 10, 100, 1000)
_HEX_GRID = (30, 24)                             # hexbin cells (κ, z)


def _symlog(z) -> np.ndarray:
    """``sign(z)·log10(1 + |z|)`` — linear-ish near 0, logarithmic in the tails."""
    z = np.asarray(z, dtype=float)
    return np.sign(z) * np.log10(1.0 + np.abs(z))


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
        """Print and return a text digest: trace contraction per year and the
        share of Σ weight per prior class × quadrant, per valuation."""
        f  = self.stats.edge_frame()
        tc = self.stats.trace_contraction(f)
        years = self.stats.years
        lines = [f"UncPropDiag ({self.label or 'unlabeled'})"]
        lines.append(f"  years     : {len(years)} ({years[0]}–{years[-1]})  "
                     f"method={getattr(self._unc, 'method', '?')}  "
                     f"alpha={getattr(self._unc, 'alpha', float('nan')):g}")
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
        text = "\n".join(lines)
        print(text)
        return text

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
        ``contraction_summary.csv``.

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
            ``__<region>`` suffix.
        timesteps : list[str] | None
            Years pooled by UD-R-01 and counted in the CSV (default: all).
            UD-R-02 always spans every year.
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
            fig = getattr(self, spec.method)(regions=regions, timesteps=timesteps)
            p = _savefig(fig, os.path.join(odir, f"{spec.filename}{suffix}.png"), dpi=dpi)
            print(f"   {pid}  → {p}  ✓")
            paths.append(p)
            if not show:
                plt.close(fig)

        csv = os.path.join(odir, f"contraction_summary{suffix}.csv")
        summary = self.stats.contraction_summary(self._frame(regions, timesteps))
        summary.to_csv(csv, index=False)
        paths.append(csv)
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
        from scipy.stats import norm

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
