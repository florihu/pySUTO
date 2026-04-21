"""
diag.py
=======
Diagnostic visualisation classes for all three pipeline stages.

    A  FeedDiagnostics      — Stage 1: feed reading, concordance coverage,
                              F_src distribution, CV propagation
    B  ImputationDiagnostics — Stage 2: aggregation coverage pre/post
                              imputation, fill counts by method, value deltas
    C  OptimDiagnostics     — Stage 3: optimisation residuals, σ_opt vs σ_agg,
                              convergence across timesteps

Design conventions
------------------
- Each class is constructed with its data; nothing is computed at import time.
- `.plot(odir=..., show=False)` writes a PNG and returns the figure path.
- `.summary()` prints a text block to stdout and returns it as a string.
- All classes can be used standalone or called from pipeline steps.
- Matplotlib is imported lazily inside methods so the module can be imported
  in headless environments without triggering a backend error.
- No pysuto-internal imports in the class bodies; types are duck-typed so
  this module stays at the leaf of the import graph.

Usage
-----
    from pysuto.diag import FeedDiagnostics, ImputationDiagnostics, OptimDiagnostics

    # Stage 1 — after run_feed_pipeline or init_icsg_feed
    diag_a = FeedDiagnostics(
        feed_id  = "icsg",
        version  = 0,
        acc_id   = "copper",
        raw_df   = raw_df,
        ct       = ct,
        F_src    = F_src,
        result   = result,
        conc_file= CONC_FILE,         # optional — enables unmatched-region breakdown
    )
    diag_a.plot(odir="data/proc/diagnostics")
    diag_a.summary()

    # Stage 2 — after FeedCollector.aggregate() + ImputationEngine.run()
    diag_b = ImputationDiagnostics(
        agg_pre  = agg_result_before_imputation,
        agg_post = agg_result_after_imputation,
        acc_struct = acc,
    )
    diag_b.plot(odir="data/proc/diagnostics")
    diag_b.summary()

    # Stage 3 — after OptimPipeline.run()
    diag_c = OptimDiagnostics(run=optim_run)
    diag_c.plot(odir="data/proc/diagnostics")
    diag_c.summary()
"""

from __future__ import annotations

import os
from typing import Optional, Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shared colour palette
# ---------------------------------------------------------------------------

_C = {
    "green":  "#2ECC71",
    "red":    "#E74C3C",
    "orange": "#E67E22",
    "blue":   "#3498DB",
    "purple": "#9B59B6",
    "grey":   "#95A5A6",
    "navy":   "#2C3E50",
}


def _savefig(fig, path: str, dpi: int = 150) -> str:
    """Save figure to *path*, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ===========================================================================
# A  FeedDiagnostics — Stage 1
# ===========================================================================

class FeedDiagnostics:
    """
    Six-panel diagnostic figure for a single data feed after Stage 1.

    Panels
    ------
    [0,0]  Row coverage          — matched vs unmatched raw rows (bar)
    [0,1]  Unmatched breakdown   — reason for unmatched rows (horizontal bar)
    [0,2]  Matched value by process — aggregated kt per process category (bar)
    [1,0]  F_src distribution    — histogram of non-zero source-edge values
    [1,1]  CV_root distribution  — histogram of per-root-edge CV (%)
    [1,2]  Value coverage by year — matched vs total flow over time (fill)

    Parameters
    ----------
    feed_id   : identifier string for titles and filenames
    version   : version integer or string
    acc_id    : accounting structure id for titles
    raw_df    : raw input DataFrame (must have columns Value, Year, Process, Region)
    ct        : ConcordanceTable (must have .unmatched_raw_rows, .n_src_edges,
                .n_root_edges)
    F_src     : source-edge flow vector, shape (n_src_edges,)
    result    : PropagationResult (must have .cv_root, .cv_src, .F_root)
    conc_file : optional path to concordance Excel workbook; enables the
                region-based unmatched-reason breakdown in panel [0,1].
                If None, all unmatched rows are labelled "reason unknown".
    """

    def __init__(
        self,
        feed_id:   str,
        version:   Any,
        acc_id:    str,
        raw_df:    pd.DataFrame,
        ct,
        F_src:     np.ndarray,
        result,
        conc_file: Optional[str] = None,
    ) -> None:
        self.feed_id   = feed_id
        self.version   = version
        self.acc_id    = acc_id
        self.raw_df    = raw_df
        self.ct        = ct
        self.F_src     = F_src
        self.result    = result
        self.conc_file = conc_file

        self._unmatched_idx = set(getattr(ct, "unmatched_raw_rows", []))
        self._matched_mask  = ~raw_df.index.isin(self._unmatched_idx)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(
        self,
        odir: str = "data/proc/diagnostics",
        show: bool = False,
        dpi:  int  = 150,
    ) -> str:
        """
        Render and save the six-panel diagnostic figure.

        Returns the path of the saved PNG.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            f"Feed diagnostics — {self.feed_id}  v{self.version}  (acc: {self.acc_id})",
            fontsize=14, fontweight="bold",
        )

        self._panel_row_coverage(axes[0, 0])
        self._panel_unmatched_reason(axes[0, 1])
        self._panel_matched_by_sector(axes[0, 2])
        self._panel_fsrc_distribution(axes[1, 0])
        self._panel_cvroot_distribution(axes[1, 1])
        self._panel_year_coverage(axes[1, 2])

        plt.tight_layout()

        path = os.path.join(odir, f"{self.feed_id}_v{self.version}_feed.png")
        _savefig(fig, path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)
        print(f"   Feed diagnostics → {path}  ✓")
        return path

    def summary(self) -> str:
        """Print and return a text summary block."""
        raw_df     = self.raw_df
        total      = len(raw_df)
        matched_n  = int(self._matched_mask.sum())
        unmatch_n  = len(self._unmatched_idx)
        val_match  = raw_df.loc[self._matched_mask, "Value"].sum()
        val_total  = raw_df["Value"].sum()

        lines = [
            f"{'─'*60}",
            f"  FEED DIAGNOSTICS — {self.feed_id}  v{self.version}  (acc: {self.acc_id})",
            f"{'─'*60}",
            f"  Raw rows total  : {total:>8,}",
            f"  Rows matched    : {matched_n:>8,}  ({matched_n / total * 100:.1f} %)",
            f"  Rows unmatched  : {unmatch_n:>8,}  ({unmatch_n / total * 100:.1f} %)",
            f"  Value matched   : {val_match:>10,.1f} kt",
            f"  Value total     : {val_total:>10,.1f} kt",
            f"  Value coverage  :   {val_match / val_total * 100:.1f} %",
            f"  Src edges       : {self.ct.n_src_edges:>8,}",
            f"  Root edges      : {self.ct.n_root_edges:>8,}",
            f"  CV_src          :   {self.result.cv_src * 100:.2f} %",
        ]

        if self._unmatched_idx:
            lines.append(f"\n  Top unmatched regions:")
            unm = raw_df.loc[list(self._unmatched_idx)]
            for reg, cnt in unm["Region"].value_counts().head(10).items():
                lines.append(f"    {reg:<32} {cnt:>5} rows")

        lines.append(f"{'─'*60}")
        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # Private panel renderers
    # ------------------------------------------------------------------

    def _panel_row_coverage(self, ax) -> None:
        counts = [int(self._matched_mask.sum()), len(self._unmatched_idx)]
        bars   = ax.bar(
            ["Matched", "Unmatched"], counts,
            color=[_C["green"], _C["red"]], edgecolor="white",
        )
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                f"{cnt:,}\n({cnt / len(self.raw_df) * 100:.1f} %)",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_title("Row coverage", fontweight="bold")
        ax.set_ylabel("# raw rows")
        ax.set_ylim(0, max(counts) * 1.2)
        ax.grid(axis="y", alpha=0.3)

    def _panel_unmatched_reason(self, ax) -> None:
        if not self._unmatched_idx:
            ax.text(0.5, 0.5, "All rows matched ✓",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=12, color=_C["green"])
            ax.set_title("Unmatched — reason", fontweight="bold")
            return

        unm = self.raw_df.loc[list(self._unmatched_idx)].copy()
        unm["reason"] = "reason unknown"

        if self.conc_file and os.path.isfile(self.conc_file):
            try:
                mapped_regs = pd.read_excel(
                    self.conc_file, sheet_name="region_map",
                )["src_region"].tolist()
                unm.loc[~unm["Region"].isin(mapped_regs), "reason"] = "unmapped region"
            except Exception:
                pass  # conc_file readable but sheet absent — leave "reason unknown"

        rc = unm["reason"].value_counts()
        ax.barh(rc.index, rc.values, color=_C["orange"], edgecolor="white")
        ax.set_title("Unmatched — reason", fontweight="bold")
        ax.set_xlabel("# rows")
        ax.grid(axis="x", alpha=0.3)

    def _panel_matched_by_sector(self, ax) -> None:
        import matplotlib.ticker as mticker
        # derive flow sector from concordance table (raw_id → src_sector_from)
        sector_of = (
            self.ct.tab[['raw_id', 'src_sector_from']]
            .drop_duplicates('raw_id')
            .set_index('raw_id')['src_sector_from']
        )
        matched = self.raw_df.loc[self._matched_mask].copy()
        matched['_sector'] = matched.index.map(sector_of)
        pv = matched.groupby('_sector')['Value'].sum().sort_values()
        ax.barh(pv.index, pv.values, color=_C["blue"], edgecolor="white")
        ax.set_title("Matched value by flow sector (kt)", fontweight="bold")
        ax.set_xlabel("kt")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(axis="x", alpha=0.3)

    def _panel_fsrc_distribution(self, ax) -> None:
        nz = self.F_src[self.F_src > 0]
        if len(nz):
            ax.hist(np.log10(nz), bins=40, color=_C["purple"], edgecolor="white", linewidth=0.5)
        ax.set_title("F_src distribution (log₁₀ kt, non-zero)", fontweight="bold")
        ax.set_xlabel("log₁₀(kt)")
        ax.set_ylabel("# src edges")
        ax.grid(axis="y", alpha=0.3)

    def _panel_cvroot_distribution(self, ax) -> None:
        cv_pct = self.result.cv_root * 100
        valid  = cv_pct[np.isfinite(cv_pct) & (cv_pct > 0)]
        if len(valid):
            ax.hist(np.log10(valid), bins=30, color=_C["red"], edgecolor="white", linewidth=0.5)
        ax.set_title("CV_root distribution (log₁₀ %)", fontweight="bold")
        ax.set_xlabel("log₁₀(CV %)")
        ax.set_ylabel("# root edges")
        ax.grid(axis="y", alpha=0.3)

    def _panel_year_coverage(self, ax) -> None:
        import matplotlib.ticker as mticker
        yr_all   = self.raw_df.groupby("Year")["Value"].sum().sort_index()
        yr_match = (
            self.raw_df.loc[self._matched_mask]
            .groupby("Year")["Value"].sum()
            .reindex(yr_all.index, fill_value=0)
        )
        yrs = yr_all.index.astype(int)
        ax.fill_between(yrs, yr_all.values,   alpha=0.3, color=_C["red"],
                        label="Total (incl. unmatched)")
        ax.fill_between(yrs, yr_match.values, alpha=0.7, color=_C["green"],
                        label="Matched")
        ax.set_title("Value coverage by year (kt)", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("kt")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
        )


# ===========================================================================
# B  ImputationDiagnostics — Stage 2
# ===========================================================================

class ImputationDiagnostics:
    """
    Diagnostic figure comparing aggregation state before and after imputation.

    Panels
    ------
    [0,0]  Coverage heatmap (pre)  — which feeds cover which edge classes
    [0,1]  Coverage heatmap (post) — same view after imputation fills NaNs
    [1,0]  Fill count by method    — how many edges each imputer filled
    [1,1]  Imputed value distribution — histogram of imputed f_agg values
    [1,2]  σ_agg: pre vs post      — scatter showing uncertainty change

    Parameters
    ----------
    agg_pre   : AggregationResult before ImputationEngine.run()
                Must have .f_agg [N_r], .sigma_agg [N_r], .Omega [N_f × N_r]
    agg_post  : AggregationResult after ImputationEngine.run()
                Additionally must have .imputation_log — list of dicts with
                keys: edge_id, method, value, sigma
    acc_struct: AccountingStructure — used for edge labels (optional; pass
                None to fall back to integer indices)
    label     : optional run label for figure titles
    """

    def __init__(
        self,
        agg_pre,
        agg_post,
        acc_struct=None,
        label: str = "",
    ) -> None:
        self.agg_pre    = agg_pre
        self.agg_post   = agg_post
        self.acc_struct = acc_struct
        self.label      = label

        # Derived
        self._imputed_mask = (
            np.isnan(agg_pre.f_agg) & np.isfinite(agg_post.f_agg)
        )
        self._imp_log: list[dict] = getattr(agg_post, "imputation_log", [])

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(
        self,
        odir: str = "data/proc/diagnostics",
        show: bool = False,
        dpi:  int  = 150,
    ) -> str:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        suffix = f" — {self.label}" if self.label else ""
        fig.suptitle(
            f"Imputation diagnostics{suffix}",
            fontsize=14, fontweight="bold",
        )

        self._panel_coverage_pre(axes[0, 0])
        self._panel_coverage_post(axes[0, 1])
        self._panel_fill_count_by_method(axes[0, 2])
        self._panel_imputed_value_dist(axes[1, 0])
        self._panel_sigma_pre_post(axes[1, 1])
        self._panel_feed_weight_dist(axes[1, 2])

        plt.tight_layout()

        tag   = self.label.replace(" ", "_") or "imputation"
        path  = os.path.join(odir, f"{tag}_imputation.png")
        _savefig(fig, path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)
        print(f"   Imputation diagnostics → {path}  ✓")
        return path

    def summary(self) -> str:
        """Print and return a text summary block."""
        pre, post   = self.agg_pre, self.agg_post
        n_edges     = len(pre.f_agg)
        n_covered   = int(np.isfinite(pre.f_agg).sum())
        n_imputed   = int(self._imputed_mask.sum())
        n_missing   = n_edges - n_covered - n_imputed

        # Fill counts by method
        method_counts: dict[str, int] = {}
        for rec in self._imp_log:
            m = rec.get("method", "unknown")
            method_counts[m] = method_counts.get(m, 0) + 1

        lines = [
            f"{'─'*60}",
            f"  IMPUTATION DIAGNOSTICS{' — ' + self.label if self.label else ''}",
            f"{'─'*60}",
            f"  Root edges total  : {n_edges:>8,}",
            f"  Covered by feeds  : {n_covered:>8,}  ({n_covered / n_edges * 100:.1f} %)",
            f"  Filled by imputer : {n_imputed:>8,}  ({n_imputed / n_edges * 100:.1f} %)",
            f"  Still missing     : {n_missing:>8,}  ({n_missing / n_edges * 100:.1f} %)",
        ]

        if method_counts:
            lines.append(f"\n  Fill counts by method:")
            for method, cnt in sorted(method_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {method:<30} {cnt:>6,}")

        if n_imputed:
            imp_vals = post.f_agg[self._imputed_mask]
            lines += [
                f"\n  Imputed value range : {imp_vals.min():>10,.1f} – {imp_vals.max():,.1f} kt",
                f"  Imputed value mean  : {imp_vals.mean():>10,.1f} kt",
            ]

        lines.append(f"{'─'*60}")
        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # Private panel renderers
    # ------------------------------------------------------------------

    def _panel_coverage_pre(self, ax) -> None:
        self._coverage_panel(ax, self.agg_pre, title="Coverage — pre-imputation")

    def _panel_coverage_post(self, ax) -> None:
        self._coverage_panel(ax, self.agg_post, title="Coverage — post-imputation")

    def _coverage_panel(self, ax, agg, title: str) -> None:
        """
        Bar chart: covered / imputed / missing edge counts.
        If Omega is available, also shows the mean number of feeds per edge.
        """
        n_edges   = len(agg.f_agg)
        n_covered = int(np.isfinite(agg.f_agg).sum())
        n_missing = n_edges - n_covered

        bars = ax.bar(
            ["Covered", "Missing"],
            [n_covered, n_missing],
            color=[_C["green"], _C["grey"]],
            edgecolor="white",
        )
        for bar, cnt in zip(bars, [n_covered, n_missing]):
            pct = cnt / n_edges * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + n_edges * 0.01,
                f"{cnt:,}\n({pct:.1f} %)",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("# root edges")
        ax.set_ylim(0, n_edges * 1.2)
        ax.grid(axis="y", alpha=0.3)

    def _panel_fill_count_by_method(self, ax) -> None:
        if not self._imp_log:
            ax.text(0.5, 0.5, "No imputation log available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title("Fill count by method", fontweight="bold")
            return

        method_counts: dict[str, int] = {}
        for rec in self._imp_log:
            m = rec.get("method", "unknown")
            method_counts[m] = method_counts.get(m, 0) + 1

        methods = list(method_counts.keys())
        counts  = [method_counts[m] for m in methods]
        colors  = [_C["blue"], _C["purple"], _C["orange"], _C["navy"]][:len(methods)]

        ax.bar(methods, counts, color=colors, edgecolor="white")
        ax.set_title("Fill count by method", fontweight="bold")
        ax.set_ylabel("# edges filled")
        ax.grid(axis="y", alpha=0.3)

    def _panel_imputed_value_dist(self, ax) -> None:
        imp_vals = self.agg_post.f_agg[self._imputed_mask]
        if len(imp_vals):
            ax.hist(imp_vals, bins=30, color=_C["orange"],
                    edgecolor="white", linewidth=0.5)
        ax.set_title("Imputed f_agg distribution (kt)", fontweight="bold")
        ax.set_xlabel("kt")
        ax.set_ylabel("# edges")
        ax.grid(axis="y", alpha=0.3)

    def _panel_sigma_pre_post(self, ax) -> None:
        """
        Scatter: σ_agg before vs after imputation for covered edges.
        Points that changed lie off the diagonal; imputed edges highlighted.
        """
        pre_sigma  = self.agg_pre.sigma_agg
        post_sigma = self.agg_post.sigma_agg
        both_finite = np.isfinite(pre_sigma) & np.isfinite(post_sigma)

        if both_finite.any():
            ax.scatter(
                pre_sigma[both_finite], post_sigma[both_finite],
                s=4, alpha=0.4, color=_C["blue"], label="feed-covered",
            )
        # Imputed edges — sigma changed from NaN to finite
        imp_post = post_sigma[self._imputed_mask]
        if len(imp_post):
            ax.scatter(
                np.zeros(len(imp_post)), imp_post,
                s=10, alpha=0.6, color=_C["orange"], label="imputed",
            )

        lo = min(pre_sigma[both_finite].min() if both_finite.any() else 0, 0)
        hi = post_sigma[np.isfinite(post_sigma)].max() if np.isfinite(post_sigma).any() else 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="no change")
        ax.set_title("σ_agg: pre vs post imputation (kt)", fontweight="bold")
        ax.set_xlabel("σ_agg pre (kt)")
        ax.set_ylabel("σ_agg post (kt)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def _panel_feed_weight_dist(self, ax) -> None:
        """
        Histogram of max feed weight per root edge from Ω.
        Shows how concentrated vs distributed the coverage is.
        """
        Omega = getattr(self.agg_pre, "Omega", None)
        if Omega is None or Omega.shape[0] == 0:
            ax.text(0.5, 0.5, "Ω not available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title("Max feed weight per edge", fontweight="bold")
            return

        max_w = np.nanmax(Omega, axis=0)
        max_w = max_w[np.isfinite(max_w)]
        if len(max_w):
            ax.hist(max_w, bins=20, color=_C["purple"],
                    edgecolor="white", linewidth=0.5)
        ax.axvline(1.0, color=_C["red"], lw=1, ls="--",
                   label="single-feed monopoly")
        ax.set_title("Max Ω weight per root edge", fontweight="bold")
        ax.set_xlabel("max Ω[k, j]")
        ax.set_ylabel("# root edges")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)


# ===========================================================================
# C  OptimDiagnostics — Stage 3
# ===========================================================================

class OptimDiagnostics:
    """
    Diagnostic figure for an optimisation run across T timesteps.

    Panels
    ------
    [0,0]  Residual norm by timestep  — ||G f_opt − c||₂ over time
    [0,1]  σ_opt vs σ_agg            — scatter; points below diagonal = reduced unc.
    [0,2]  Uncertainty reduction      — histogram of (σ_agg − σ_opt) / σ_agg
    [1,0]  F_opt vs F_agg            — scatter; points off diagonal = constraint pull
    [1,1]  Solver status by timestep  — bar coloured by status (ok / infeasible / …)
    [1,2]  Warm-start delta           — ||f_opt[t] − f_opt[t-1]||₂ over time

    Parameters
    ----------
    run : OptimRun — must have .results (list of OptimResult, indexed by timestep)
          Each OptimResult must have: .F_opt, .sigma_opt, .F_agg, .sigma_agg,
          .residual_norm, .solver_status, .timestep
    label : optional run label for figure titles
    """

    def __init__(self, run, label: str = "") -> None:
        self.run   = run
        self.label = label
        self._results: list = list(run.results.values()) if hasattr(run.results, "values") \
                              else list(run.results)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(
        self,
        odir: str = "data/proc/diagnostics",
        show: bool = False,
        dpi:  int  = 150,
    ) -> str:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        suffix = f" — {self.label}" if self.label else ""
        fig.suptitle(
            f"Optimisation diagnostics{suffix}",
            fontsize=14, fontweight="bold",
        )

        self._panel_residual_norm(axes[0, 0])
        self._panel_sigma_opt_vs_agg(axes[0, 1])
        self._panel_unc_reduction(axes[0, 2])
        self._panel_fopt_vs_fagg(axes[1, 0])
        self._panel_solver_status(axes[1, 1])
        self._panel_warmstart_delta(axes[1, 2])

        plt.tight_layout()

        tag  = self.label.replace(" ", "_") or "optim"
        path = os.path.join(odir, f"{tag}_optim.png")
        _savefig(fig, path, dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)
        print(f"   Optimisation diagnostics → {path}  ✓")
        return path

    def summary(self) -> str:
        """Print and return a text summary block."""
        results = self._results
        if not results:
            text = "  No optimisation results to summarise."
            print(text)
            return text

        timesteps = [getattr(r, "timestep", i) for i, r in enumerate(results)]
        res_norms = [getattr(r, "residual_norm", np.nan) for r in results]
        statuses  = [getattr(r, "solver_status", "?") for r in results]

        # σ reduction statistics (all timesteps pooled)
        all_sigma_agg = np.concatenate([
            np.asarray(getattr(r, "sigma_agg", [])).ravel() for r in results
        ])
        all_sigma_opt = np.concatenate([
            np.asarray(getattr(r, "sigma_opt", [])).ravel() for r in results
        ])
        finite = np.isfinite(all_sigma_agg) & np.isfinite(all_sigma_opt) & (all_sigma_agg > 0)
        rel_red = (all_sigma_agg[finite] - all_sigma_opt[finite]) / all_sigma_agg[finite]

        ok_count  = sum(1 for s in statuses if str(s).lower() in ("ok", "optimal", "0"))
        bad_count = len(statuses) - ok_count

        lines = [
            f"{'─'*60}",
            f"  OPTIMISATION DIAGNOSTICS{' — ' + self.label if self.label else ''}",
            f"{'─'*60}",
            f"  Timesteps run      : {len(results):>6,}",
            f"  Solver OK          : {ok_count:>6,}",
            f"  Solver non-OK      : {bad_count:>6,}",
            f"  Mean residual norm : {np.nanmean(res_norms):>10.4f}",
            f"  Max  residual norm : {np.nanmax(res_norms):>10.4f}",
        ]

        if finite.any():
            lines += [
                f"\n  σ reduction (σ_agg → σ_opt) across all edges × timesteps:",
                f"    Mean : {rel_red.mean() * 100:>7.2f} %",
                f"    P25  : {np.percentile(rel_red, 25) * 100:>7.2f} %",
                f"    P75  : {np.percentile(rel_red, 75) * 100:>7.2f} %",
            ]

        lines.append(f"{'─'*60}")
        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # Private panel renderers
    # ------------------------------------------------------------------

    def _panel_residual_norm(self, ax) -> None:
        results   = self._results
        timesteps = [getattr(r, "timestep", i) for i, r in enumerate(results)]
        res_norms = [getattr(r, "residual_norm", np.nan) for r in results]

        ax.plot(timesteps, res_norms, color=_C["red"], marker="o",
                markersize=4, linewidth=1.2)
        ax.set_title("Residual norm ‖Gf_opt − c‖₂ by timestep", fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("‖residual‖₂")
        ax.grid(alpha=0.3)

    def _panel_sigma_opt_vs_agg(self, ax) -> None:
        """Scatter σ_opt vs σ_agg pooled across all timesteps. Points below the
        diagonal indicate uncertainty reduction from the optimiser."""
        all_agg, all_opt = [], []
        for r in self._results:
            sa = np.asarray(getattr(r, "sigma_agg", [])).ravel()
            so = np.asarray(getattr(r, "sigma_opt", [])).ravel()
            if len(sa) == len(so):
                all_agg.append(sa)
                all_opt.append(so)

        if not all_agg:
            ax.text(0.5, 0.5, "No σ data available",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("σ_opt vs σ_agg (kt)", fontweight="bold")
            return

        sa = np.concatenate(all_agg)
        so = np.concatenate(all_opt)
        finite = np.isfinite(sa) & np.isfinite(so)

        ax.scatter(sa[finite], so[finite], s=2, alpha=0.3, color=_C["blue"])
        hi = max(sa[finite].max(), so[finite].max())
        ax.plot([0, hi], [0, hi], "k--", lw=0.8, alpha=0.5, label="no change")
        ax.set_title("σ_opt vs σ_agg (kt)", fontweight="bold")
        ax.set_xlabel("σ_agg (kt)")
        ax.set_ylabel("σ_opt (kt)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def _panel_unc_reduction(self, ax) -> None:
        """Histogram of relative uncertainty reduction (σ_agg − σ_opt) / σ_agg."""
        all_red = []
        for r in self._results:
            sa = np.asarray(getattr(r, "sigma_agg", [])).ravel()
            so = np.asarray(getattr(r, "sigma_opt", [])).ravel()
            if len(sa) == len(so):
                pos = (sa > 0) & np.isfinite(sa) & np.isfinite(so)
                all_red.append((sa[pos] - so[pos]) / sa[pos])

        if not all_red:
            ax.text(0.5, 0.5, "No σ data available",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Relative uncertainty reduction (%)", fontweight="bold")
            return

        red = np.concatenate(all_red) * 100
        ax.hist(red, bins=40, color=_C["green"], edgecolor="white", linewidth=0.5)
        ax.axvline(0, color=_C["red"], lw=1, ls="--", label="no change")
        ax.set_title("Relative uncertainty reduction (%)", fontweight="bold")
        ax.set_xlabel("(σ_agg − σ_opt) / σ_agg  (%)")
        ax.set_ylabel("# edges × timesteps")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    def _panel_fopt_vs_fagg(self, ax) -> None:
        """Scatter F_opt vs F_agg pooled across timesteps. Deviation from the
        diagonal shows where the constraint pulls the optimised value."""
        all_agg, all_opt = [], []
        for r in self._results:
            fa = np.asarray(getattr(r, "F_agg", [])).ravel()
            fo = np.asarray(getattr(r, "F_opt", [])).ravel()
            if len(fa) == len(fo):
                all_agg.append(fa)
                all_opt.append(fo)

        if not all_agg:
            ax.text(0.5, 0.5, "No F data available",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("F_opt vs F_agg (kt)", fontweight="bold")
            return

        fa = np.concatenate(all_agg)
        fo = np.concatenate(all_opt)
        finite = np.isfinite(fa) & np.isfinite(fo)

        ax.scatter(fa[finite], fo[finite], s=2, alpha=0.3, color=_C["navy"])
        hi = max(fa[finite].max(), fo[finite].max())
        ax.plot([0, hi], [0, hi], "k--", lw=0.8, alpha=0.5, label="no constraint pull")
        ax.set_title("F_opt vs F_agg (kt)", fontweight="bold")
        ax.set_xlabel("F_agg (kt)")
        ax.set_ylabel("F_opt (kt)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    def _panel_solver_status(self, ax) -> None:
        from collections import Counter
        statuses  = [str(getattr(r, "solver_status", "unknown")) for r in self._results]
        timesteps = [getattr(r, "timestep", i) for i, r in enumerate(self._results)]
        status_set = sorted(set(statuses))

        color_map = {
            s: c for s, c in zip(
                status_set,
                [_C["green"], _C["red"], _C["orange"], _C["grey"]],
            )
        }

        bottom = np.zeros(len(timesteps))
        for s in status_set:
            vals = np.array([1 if st == s else 0 for st in statuses], dtype=float)
            ax.bar(timesteps, vals, bottom=bottom,
                   color=color_map[s], label=s, edgecolor="white", linewidth=0.3)
            bottom += vals

        ax.set_title("Solver status by timestep", fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Status (stacked)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    def _panel_warmstart_delta(self, ax) -> None:
        """
        L2 norm of the change in F_opt between consecutive timesteps.
        A large jump indicates that warm-starting carried limited benefit.
        """
        results   = self._results
        timesteps = [getattr(r, "timestep", i) for i, r in enumerate(results)]
        deltas    = [np.nan]   # first timestep has no predecessor

        for i in range(1, len(results)):
            fo_prev = np.asarray(getattr(results[i - 1], "F_opt", [])).ravel()
            fo_curr = np.asarray(getattr(results[i],     "F_opt", [])).ravel()
            if len(fo_prev) == len(fo_curr) and len(fo_curr):
                diff = fo_curr - fo_prev
                deltas.append(float(np.sqrt(np.nansum(diff ** 2))))
            else:
                deltas.append(np.nan)

        ax.plot(timesteps, deltas, color=_C["purple"], marker="o",
                markersize=4, linewidth=1.2)
        ax.set_title("Warm-start delta ‖F_opt[t] − F_opt[t−1]‖₂", fontweight="bold")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("‖Δ F_opt‖₂ (kt)")
        ax.grid(alpha=0.3)
