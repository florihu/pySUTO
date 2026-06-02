"""
agg.py
======
``AggDiagnostic`` — Stage 2 aggregation diagnostic suite.

Per-feed coverage, Omega weights, f_agg/σ_agg distributions.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._plot import _C, _savefig


class AggDiagnostic:
    """
    Stage 2a aggregation diagnostics — runs immediately after
    ``FeedCollector.aggregate()`` and BEFORE imputation.

    Produces one figure ``<label>_agg_distributions.png`` with a 2×2 panel:

        [0,0]  log10 |f_agg| histogram of covered edges
        [0,1]  log10 σ_agg histogram
        [1,0]  log10 CV histogram (CV = σ_agg / |f_agg|), with vertical
               reference lines at CV = 0.1, 1, 10, 100
        [1,1]  per-feed scatter: |f_root| vs σ_root for every feed × edge,
               one colour per feed_id, log–log axes. The diagonal CV=1 line
               and CV=10 line are drawn for reference. This panel reveals
               which Stage-1 feed is producing σ ≫ |f| outliers.

    ``summary()`` prints CV percentiles and the top-N (feed, edge) pairs
    by σ to stdout.

    Parameters
    ----------
    agg_result : AggregationResult
        Output of ``FeedCollector.aggregate()`` — uses ``f_agg``, ``sigma_agg``,
        ``coverage`` (Stage 2a aggregated quantities) and ``F``, ``Sigma``,
        ``feed_ids`` (Stage 1 per-feed observations).
    label : str
        Optional label for figure titles and output filenames.
    """

    def __init__(self, agg_result, acc_struct=None, label: str = "",
                 year: str | None = "1995") -> None:
        # Snapshot the arrays we need at construction time, because
        # ImputationEngine.run() mutates agg_result.f_agg / sigma_agg in place.
        # The per-feed F / Sigma matrices are not mutated by imputation.
        self.label      = label
        self.acc_struct = acc_struct
        self.year       = year
        self._f_agg     = np.array(agg_result.f_agg, dtype=float, copy=True)
        self._sigma_agg = np.array(agg_result.sigma_agg, dtype=float, copy=True)
        self._coverage  = np.array(agg_result.coverage, dtype=bool, copy=True)
        self._F         = np.array(agg_result.F, dtype=float, copy=True)
        self._Sigma     = np.array(agg_result.Sigma, dtype=float, copy=True)
        self._feed_ids  = list(agg_result.feed_ids)
        # Per-edge time labels, when acc_struct is provided. Used to stratify
        # the per-feed root distributions by year.
        if acc_struct is not None:
            ei = acc_struct.edge_index.sort_values("edge_id")
            self._edge_times = ei["time"].astype(str).to_numpy()
        else:
            self._edge_times = None
        # Distribution panels restrict to a single year when both the year and
        # the per-edge time labels are available; None ⇒ pool all years. If the
        # year is absent from the data, fall back to all years rather than
        # rendering empty panels.
        if self._edge_times is not None and year is not None:
            mask = self._edge_times == str(year)
            self._year_mask = mask if mask.any() else None
        else:
            self._year_mask = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(
        self,
        odir: str | None = None,
        show: bool = False,
        dpi:  int  = 150,
    ) -> list[str]:
        """Produce and optionally save the aggregation diagnostic figure."""
        import matplotlib.pyplot as plt

        paths: list[str] = []

        fig = self._fig_distributions()
        if odir is not None:
            p = _savefig(fig, os.path.join(odir, "agg_distributions.png"), dpi=dpi)
            paths.append(p)
            print(f"   Aggregation distributions   → {p}  ✓")
        if show:
            plt.show()
        plt.close(fig)

        # New: per-timestep root distributions (viridis by year) when
        # acc_struct is available so we can stratify edge columns by time.
        if self._edge_times is not None:
            fig_root = self._fig_root_by_year()
            if fig_root is not None:
                if odir is not None:
                    p = _savefig(fig_root, os.path.join(odir, "agg_root_by_year.png"), dpi=dpi)
                    paths.append(p)
                    print(f"   Aggregation root-by-year    → {p}  ✓")
                if show:
                    plt.show()
                plt.close(fig_root)
        return paths

    def summary(self) -> str:
        """Print and return a one-block text summary of f, σ, CV, and outliers."""
        f_agg     = self._f_agg
        sigma_agg = self._sigma_agg
        coverage  = self._coverage
        F_mat     = self._F
        Sigma_mat = self._Sigma

        # Aggregated CV
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_agg = np.where(np.abs(f_agg) > 0, sigma_agg / np.abs(f_agg), np.nan)
        cv_valid = cv_agg[coverage & np.isfinite(cv_agg)]

        def _pct(x: np.ndarray, q: float) -> float:
            return float(np.nanpercentile(x, q)) if x.size else float("nan")

        # Per-feed (feed × edge) distributions for outlier ranking
        finite = np.isfinite(F_mat) & np.isfinite(Sigma_mat)
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_feed = np.where(np.abs(F_mat) > 0, Sigma_mat / np.abs(F_mat), np.nan)

        # Top-5 (feed, edge) pairs by σ
        idx_sorted = np.argsort(-np.where(finite, Sigma_mat, -np.inf), axis=None)[:5]
        rows, cols = np.unravel_index(idx_sorted, Sigma_mat.shape)

        lines: list[str] = []
        lines.append(f"AggDiagnostic ({self.label or 'unlabeled'})")
        lines.append(f"  covered edges: {int(coverage.sum())} / {f_agg.size}")
        lines.append(
            f"  f_agg  : p50={_pct(np.abs(f_agg[coverage]), 50):.3g}  "
            f"p99={_pct(np.abs(f_agg[coverage]), 99):.3g}  "
            f"max={float(np.nanmax(np.abs(f_agg[coverage]))) if coverage.any() else float('nan'):.3g}"
        )
        lines.append(
            f"  σ_agg  : p50={_pct(sigma_agg[coverage], 50):.3g}  "
            f"p99={_pct(sigma_agg[coverage], 99):.3g}  "
            f"max={float(np.nanmax(sigma_agg[coverage])) if coverage.any() else float('nan'):.3g}"
        )
        lines.append(
            f"  CV_agg : p50={_pct(cv_valid, 50):.3g}  "
            f"p90={_pct(cv_valid, 90):.3g}  "
            f"p99={_pct(cv_valid, 99):.3g}  "
            f"max={float(np.nanmax(cv_valid)) if cv_valid.size else float('nan'):.3g}"
        )
        lines.append("  top-5 (feed, edge) by σ_root:")
        for r, c in zip(rows, cols):
            if not finite[r, c]:
                continue
            fid = self._feed_ids[int(r)]
            lines.append(
                f"    feed='{fid}' edge={int(c)}  "
                f"f={F_mat[r, c]:.3g}  σ={Sigma_mat[r, c]:.3g}  "
                f"CV={cv_feed[r, c]:.3g}"
            )
        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # Internal: 2×2 panel
    # ------------------------------------------------------------------

    def _fig_distributions(self):
        import matplotlib.pyplot as plt

        f_agg     = self._f_agg
        sigma_agg = self._sigma_agg
        F_mat     = self._F
        Sigma_mat = self._Sigma
        feed_ids  = self._feed_ids

        # Restrict every panel to the active year (when set); otherwise pool all.
        ymask     = (self._year_mask if self._year_mask is not None
                     else np.ones(f_agg.shape, dtype=bool))
        coverage  = self._coverage & ymask

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        title = f"Aggregation diagnostics — {self.label}" if self.label else "Aggregation diagnostics"
        if self._year_mask is not None:
            title += f"  ({self.year})"
        fig.suptitle(title, fontsize=13, fontweight="bold")

        # [0,0] log10 |f_agg|
        ax = axes[0, 0]
        f_pos = np.abs(f_agg[coverage])
        f_pos = f_pos[f_pos > 0]
        if f_pos.size:
            ax.hist(np.log10(f_pos), bins=60, color=_C["blue"], alpha=0.8)
        ax.set_title("|f_agg|  (log10)", fontweight="bold")
        ax.set_xlabel("log10 |f_agg|")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)

        # [0,1] log10 σ_agg
        ax = axes[0, 1]
        s_pos = sigma_agg[coverage]
        s_pos = s_pos[s_pos > 0]
        if s_pos.size:
            ax.hist(np.log10(s_pos), bins=60, color=_C["orange"], alpha=0.8)
        ax.set_title("σ_agg  (log10)", fontweight="bold")
        ax.set_xlabel("log10 σ_agg")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)

        # [1,0] log10 CV with reference lines at 0.1, 1, 10, 100
        ax = axes[1, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_agg = np.where(np.abs(f_agg) > 0, sigma_agg / np.abs(f_agg), np.nan)
        cv_valid = cv_agg[coverage & np.isfinite(cv_agg) & (cv_agg > 0)]
        if cv_valid.size:
            ax.hist(np.log10(cv_valid), bins=60, color=_C["purple"], alpha=0.8)
        for ref, lbl in [(0.1, "0.1"), (1.0, "1"), (10.0, "10"), (100.0, "100")]:
            ax.axvline(np.log10(ref), color=_C["red"], lw=0.8, ls="--", alpha=0.6)
            ax.text(np.log10(ref), ax.get_ylim()[1] * 0.95, lbl,
                    rotation=90, fontsize=8, va="top", ha="right",
                    color=_C["red"], alpha=0.8)
        ax.set_title("CV = σ_agg / |f_agg|  (log10)", fontweight="bold")
        ax.set_xlabel("log10 CV")
        ax.set_ylabel("count")
        ax.grid(alpha=0.3)

        # [1,1] per-feed scatter |f_root| vs σ_root
        ax = axes[1, 1]
        cmap = plt.get_cmap("tab10")
        for k, fid in enumerate(feed_ids):
            f_k = F_mat[k]
            s_k = Sigma_mat[k]
            mask = np.isfinite(f_k) & np.isfinite(s_k) & (np.abs(f_k) > 0) & (s_k > 0) & ymask
            if not mask.any():
                continue
            ax.scatter(
                np.abs(f_k[mask]), s_k[mask],
                s=4, alpha=0.4, color=cmap(k % 10), label=str(fid),
            )
        # Reference CV=1 (σ=|f|) and CV=10 lines
        if F_mat.size and Sigma_mat.size:
            mn = max(1e-12, float(np.nanmin(np.abs(F_mat[F_mat != 0]))) if np.any(F_mat != 0) else 1e-6)
            mx = float(np.nanmax(np.abs(F_mat))) if np.isfinite(F_mat).any() else 1.0
            xx = np.array([mn, mx])
            ax.plot(xx, xx,        "k--", lw=0.8, alpha=0.5, label="CV=1")
            ax.plot(xx, 10.0 * xx, color=_C["red"], lw=0.8, ls=":", alpha=0.6, label="CV=10")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("per-feed |f_root| vs σ_root", fontweight="bold")
        ax.set_xlabel("|f_root|")
        ax.set_ylabel("σ_root")
        if len(feed_ids) <= 12:
            ax.legend(fontsize=7, framealpha=0.6, loc="lower right")
        ax.grid(alpha=0.3, which="both")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # ------------------------------------------------------------------
    # Per-timestep f_root / σ_root distributions (viridis by year)
    # ------------------------------------------------------------------

    def _fig_root_by_year(self):
        """Two-panel log10|f_root| and log10 σ_root histograms stratified by
        edge time. Viridis colormap across years with a shared colorbar.
        Pools over all feeds; requires ``self._edge_times``.
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        edge_times = self._edge_times
        if edge_times is None:
            return None

        def _sort_key(t: str) -> tuple:
            try:
                return (0, int(t))
            except (ValueError, TypeError):
                return (1, str(t))
        years_unique = sorted(set(edge_times.tolist()), key=_sort_key)
        if not years_unique:
            return None

        # Numeric year axis for the colormap when years parse as floats; fall
        # back to ordinal positions when they don't.
        try:
            year_vals = [float(y) for y in years_unique]
        except (ValueError, TypeError):
            year_vals = list(range(len(years_unique)))
        norm = Normalize(vmin=min(year_vals), vmax=max(year_vals))
        cmap = plt.get_cmap("viridis")

        F_mat     = self._F
        Sigma_mat = self._Sigma

        # Shared bin edges from full data range so per-year curves are
        # comparable.
        finite_f = np.abs(F_mat[np.isfinite(F_mat) & (F_mat != 0)])
        finite_s = Sigma_mat[np.isfinite(Sigma_mat) & (Sigma_mat > 0)]
        if not finite_f.size or not finite_s.size:
            return None
        f_bins = np.linspace(np.log10(finite_f.min()), np.log10(finite_f.max()), 60)
        s_bins = np.linspace(np.log10(finite_s.min()), np.log10(finite_s.max()), 60)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        title = f"Per-timestep root distributions — {self.label}" if self.label else "Per-timestep root distributions"
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for year, yval in zip(years_unique, year_vals):
            col_mask = (edge_times == year)
            if not col_mask.any():
                continue
            color = cmap(norm(yval))

            f_vals = np.abs(F_mat[:, col_mask]).ravel()
            f_vals = f_vals[np.isfinite(f_vals) & (f_vals > 0)]
            if f_vals.size:
                axes[0].hist(np.log10(f_vals), bins=f_bins, histtype="step",
                             color=color, linewidth=1.0, alpha=0.85)

            s_vals = Sigma_mat[:, col_mask].ravel()
            s_vals = s_vals[np.isfinite(s_vals) & (s_vals > 0)]
            if s_vals.size:
                axes[1].hist(np.log10(s_vals), bins=s_bins, histtype="step",
                             color=color, linewidth=1.0, alpha=0.85)

        axes[0].set_title("|f_root| per year", fontweight="bold")
        axes[0].set_xlabel("log10 |f_root|")
        axes[0].set_ylabel("count")
        axes[0].grid(alpha=0.3)

        axes[1].set_title("σ_root per year", fontweight="bold")
        axes[1].set_xlabel("log10 σ_root")
        axes[1].set_ylabel("count")
        axes[1].grid(alpha=0.3)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[1], pad=0.02)
        cbar.set_label("year")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        return fig


# ===========================================================================
# SolverDiag — Stage 3b post-solve diagnostics
# ===========================================================================

# Per constraint-type colour binding — reused across every SolverDiag figure.
# Real ConstraintEngine.row_meta uses {mass_balance, sum_eq, ratio_eq,
# ratio_bound}; flow bounds live in u/l and never appear in row_meta. Toy
# fixtures may omit the "type" column entirely, in which case we fall back
# to cone_type.
