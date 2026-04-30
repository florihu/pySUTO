"""
diag.py
=======
Diagnostic visualisation classes for all three pipeline stages.

    A  FeedDiagnostics  — Stage 1: feed reading, concordance coverage,
                          F_src distribution, CV propagation
    B  ImpDiagnostic    — Stage 2: per-step F_root/σ distributions (log scale),
                          coverage and fill-count evolution, TimeImputer timeseries
    C  OptimDiagnostics — Stage 3: optimisation residuals, σ_opt vs σ_agg,
                          convergence across timesteps
    D  ConstDiagnostic  — Stage 3a: G-matrix sparsity, prior distribution,
                          constraint residuals ε₀ = c − G·a₀

Design conventions
------------------
- Each class is constructed with its data; nothing is computed at import time.
- `.plot(odir=..., show=False)` writes PNG(s) and returns a list of paths.
- `.summary()` prints a text block to stdout and returns it as a string.
- All classes can be used standalone or called from pipeline steps.
- Matplotlib is imported lazily inside methods so the module can be imported
  in headless environments without triggering a backend error.
- No pysuto-internal imports in the class bodies; types are duck-typed so
  this module stays at the leaf of the import graph.

Usage
-----
    from pysuto.diag import FeedDiagnostics, ImpDiagnostic, OptimDiagnostics, ConstDiagnostic

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
    diag_b = ImpDiagnostic(agg_result, acc_struct=acc, label="copper")
    diag_b.plot(odir="data/proc/diagnostics", show=False)
    diag_b.summary()

    # Load from disk for post-hoc analysis
    diag_b = ImpDiagnostic.from_saved("data/proc/agg", acc_struct=acc)
    diag_b.plot(show=True)

    # Stage 3 — after OptimPipeline.run()
    diag_c = OptimDiagnostics(run=optim_run)
    diag_c.plot(odir="data/proc/diagnostics")
    diag_c.summary()

    # Stage 3a — after ConstraintEngine.run()
    diag_d = ConstDiagnostic(constraint_run, label="copper")
    diag_d.plot(odir="data/proc/const/<run_id>/diag", show=False)
    diag_d.summary()

    # Or load from disk for post-hoc analysis
    diag_d = ConstDiagnostic.from_saved("data/proc/const/<run_id>")
    diag_d.plot(show=True)

    # Or trigger automatically during the constraint run
    engine.run(agg_result, timesteps=[...], diag_plot=True)
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
# B  ImpDiagnostic — Stage 2
# ===========================================================================

class ImpDiagnostic:
    """
    Per-step imputation diagnostics for Stage 2 of the pySUTO pipeline.

    Produces two figures:

    Figure 1 — 2×2 panel  (<label>_imputation_steps.png)
        [0,0]  F_root distribution per step (log scale, overlapping histograms)
        [0,1]  Coverage count per step (# finite edges at each pipeline step)
        [1,0]  CV (σ/|f|) distribution per step (log scale)
        [1,1]  Count of edges imputed per step

    Figure 2 — σ-by-method boxplot  (<label>_imputation_sigma_box.png)
        Single-panel log boxplot grouped by log["method"], with a leftmost
        "covered" baseline box (Stage 2a observed σ before any imputation).
        Outliers shown as red dots. n=… annotation above each box.

    Figure 3 — timeseries  (<label>_imputation_timeseries.png)
        Up to 6 panels showing time-series of TimeImputer-filled flows,
        distinguishing raw (feed-covered) from imputed points.
        Only produced when acc_struct is provided.

    Parameters
    ----------
    agg_result : AggregationResult
        Result after ImputationEngine.run(). Must have .imputation_log set.
    acc_struct : optional
        AccountingStructure — required for Figure 2 (timeseries). If None,
        Figure 2 is skipped.
    label : str
        Optional label for figure titles and output filenames.
    """

    _TIME_METHODS = {"time_linear", "time_mean", "locf", "nocb"}
    _PALETTE      = ["navy", "blue", "green", "orange", "purple", "red"]

    def __init__(self, agg_result, acc_struct=None, label: str = "") -> None:
        self.agg_result = agg_result
        self.acc_struct = acc_struct
        self.label      = label
        self._log_df    = agg_result.imputation_log.as_dataframe()
        self._snapshots = self._build_snapshots()

    @classmethod
    def from_saved(
        cls,
        run_id: str,
        storage_dir: str = "data/proc/imp",
        acc_struct=None,
        label: str = "",
    ) -> "ImpDiagnostic":
        """
        Load an imputation run from ImputationRunRegistry and construct
        ImpDiagnostic. ``run_id`` must be a registered run in ``storage_dir``.
        """
        from pysuto.registry import ImputationRunRegistry
        registry = ImputationRunRegistry(storage_dir=storage_dir)
        agg      = registry.load(run_id)
        return cls(agg, acc_struct=acc_struct, label=label or run_id)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(
        self,
        odir: str | None = None,
        show: bool = True,
        dpi:  int  = 150,
    ) -> list[str]:
        """Produce and optionally save the three diagnostic figures."""
        import matplotlib.pyplot as plt

        tag   = self.label.replace(" ", "_") or "imputation"
        paths = []

        # Figure 1 — per-step trajectories (2×2)
        fig1 = self._fig1_steps()
        if odir is not None:
            p = _savefig(fig1, os.path.join(odir, f"{tag}_imputation_steps.png"), dpi=dpi)
            paths.append(p)
            print(f"   Imputation step diagnostics → {p}  ✓")
        if show:
            plt.show()
        plt.close(fig1)

        # Figure 2 — σ-by-method boxplot (1×1)
        fig2 = self._fig2_sigma_box()
        if fig2 is not None:
            if odir is not None:
                p = _savefig(fig2, os.path.join(odir, f"{tag}_imputation_sigma_box.png"), dpi=dpi)
                paths.append(p)
                print(f"   Imputation σ boxplot       → {p}  ✓")
            if show:
                plt.show()
            plt.close(fig2)

        # Figure 3 — TimeImputer timeseries (n×3, optional)
        if self.acc_struct is not None:
            fig3 = self._fig3_timeseries()
            if fig3 is not None:
                if odir is not None:
                    p = _savefig(fig3, os.path.join(odir, f"{tag}_imputation_timeseries.png"), dpi=dpi)
                    paths.append(p)
                    print(f"   Imputation timeseries      → {p}  ✓")
                if show:
                    plt.show()
                plt.close(fig3)

        return paths

    def summary(self) -> str:
        """Print and return a text summary of the imputation pipeline."""
        log_df  = self._log_df
        agg     = self.agg_result
        n_edges = len(agg.f_agg)
        n_final = int(np.isfinite(agg.f_agg).sum())
        n_imp   = len(log_df)
        n_miss  = n_edges - n_final

        lines = [
            f"{'─'*60}",
            f"  IMPUTATION DIAGNOSTICS{' — ' + self.label if self.label else ''}",
            f"{'─'*60}",
            f"  Root edges total       : {n_edges:>8,}",
            f"  Covered after pipeline : {n_final:>8,}  ({n_final/n_edges*100:.1f} %)",
            f"  Filled by imputers     : {n_imp:>8,}  ({n_imp/n_edges*100:.1f} %)",
            f"  Still missing          : {n_miss:>8,}  ({n_miss/n_edges*100:.1f} %)",
        ]

        if not log_df.empty:
            lines.append(f"\n  Fills per step:")
            for step, grp in log_df.groupby("step"):
                dom = grp["method"].value_counts().index[0]
                lines.append(f"    step {step}  {dom:<22}  {len(grp):>6,} fills")

        lines.append(f"{'─'*60}")
        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # Snapshot reconstruction
    # ------------------------------------------------------------------

    def _build_snapshots(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Reconstruct per-step (f, sigma) arrays from ImputationLog."""
        log_df = self._log_df
        if log_df.empty:
            f = self.agg_result.f_agg
            s = self.agg_result.sigma_agg
            return [("pre", f.copy(), s.copy())]

        imputed_ids = log_df["edge_id"].values

        f0 = self.agg_result.f_agg.copy()
        s0 = self.agg_result.sigma_agg.copy()
        f0[imputed_ids] = np.nan
        s0[imputed_ids] = np.nan

        snapshots = [("pre", f0.copy(), s0.copy())]
        f_curr = f0.copy()
        s_curr = s0.copy()

        for step in sorted(log_df["step"].unique()):
            mask   = log_df["step"] == step
            ids    = log_df.loc[mask, "edge_id"].values
            f_curr[ids] = log_df.loc[mask, "f_after"].values
            s_curr[ids] = log_df.loc[mask, "sigma_after"].values
            dom    = log_df.loc[mask, "method"].value_counts().index[0]
            snapshots.append((f"step {step} ({dom})", f_curr.copy(), s_curr.copy()))

        return snapshots

    # ------------------------------------------------------------------
    # Figure 1 — 2×2 step panels
    # ------------------------------------------------------------------

    def _fig1_steps(self):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        suffix = f" — {self.label}" if self.label else ""
        fig.suptitle(
            f"Imputation pipeline — per-step diagnostics{suffix}",
            fontsize=13, fontweight="bold",
        )
        self._panel_f_dist(axes[0, 0])
        self._panel_coverage(axes[0, 1])
        self._panel_cv_dist(axes[1, 0])
        self._panel_imputed_count(axes[1, 1])
        plt.tight_layout()
        return fig

    def _panel_f_dist(self, ax) -> None:
        colors = [_C[k] for k in self._PALETTE]
        for i, (lbl, f, _) in enumerate(self._snapshots):
            vals = f[np.isfinite(f) & (f > 0)]
            if not len(vals):
                continue
            ax.hist(np.log10(vals), bins=60, alpha=0.45,
                    color=colors[i % len(colors)], label=lbl,
                    density=True, histtype="stepfilled", edgecolor="none")
        ax.set_title("F_root distribution per step", fontweight="bold")
        ax.set_xlabel("log₁₀(f)  [log kt]")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, framealpha=0.6)
        ax.grid(axis="y", alpha=0.3)

    def _panel_coverage(self, ax) -> None:
        colors  = [_C[k] for k in self._PALETTE]
        labels  = [s[0] for s in self._snapshots]
        counts  = [int(np.isfinite(s[1]).sum()) for s in self._snapshots]
        n_total = len(self._snapshots[0][1])

        bars = ax.barh(
            labels, counts,
            color=[colors[i % len(colors)] for i in range(len(labels))],
            edgecolor="white",
        )
        for bar, cnt in zip(bars, counts):
            ax.text(
                cnt + n_total * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}  ({cnt/n_total*100:.1f} %)",
                va="center", fontsize=8,
            )
        ax.set_title("Coverage per step (# finite edges)", fontweight="bold")
        ax.set_xlabel("# root edges")
        ax.set_xlim(0, n_total * 1.28)
        ax.grid(axis="x", alpha=0.3)

    def _panel_cv_dist(self, ax) -> None:
        """
        Per-step CV (= σ / |f|) histogram on log₁₀ scale.

        Replaces the old σ-histogram panel: σ is ∝ |f| through Stage 1
        propagation, so a σ histogram is a translated copy of the F
        histogram and adds no information. CV strips out the magnitude
        coupling and exposes whether a step actually inflates relative
        uncertainty.
        """
        colors = [_C[k] for k in self._PALETTE]
        for i, (lbl, fv, sv) in enumerate(self._snapshots):
            with np.errstate(divide="ignore", invalid="ignore"):
                cv = np.where(np.abs(fv) > 0, sv / np.abs(fv), np.nan)
            vals = cv[np.isfinite(cv) & (cv > 0)]
            if not len(vals):
                continue
            ax.hist(np.log10(vals), bins=60, alpha=0.45,
                    color=colors[i % len(colors)], label=lbl,
                    density=True, histtype="stepfilled", edgecolor="none")
        ax.set_title("CV (σ/|f|) distribution per step", fontweight="bold")
        ax.set_xlabel("log₁₀(CV)")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, framealpha=0.6)
        ax.grid(axis="y", alpha=0.3)

    def _panel_imputed_count(self, ax) -> None:
        log_df = self._log_df
        if log_df.empty:
            ax.text(0.5, 0.5, "No fills recorded",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Fills per step", fontweight="bold")
            return

        step_counts = log_df.groupby("step").size()
        steps  = [str(s) for s in step_counts.index]
        counts = step_counts.values
        colors = [_C[k] for k in self._PALETTE]

        bars = ax.bar(
            steps, counts,
            color=[colors[i % len(colors)] for i in range(len(steps))],
            edgecolor="white",
        )
        for bar, cnt in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{cnt:,}", ha="center", va="bottom", fontsize=8,
            )
        ax.set_title("Edges imputed per step", fontweight="bold")
        ax.set_xlabel("pipeline step")
        ax.set_ylabel("# edges filled")
        ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------
    # Figure 2 — σ-by-method log boxplot
    # ------------------------------------------------------------------

    # method label → palette key (kept consistent with step palette via reuse)
    _METHOD_COLORS = {
        "covered":                "grey",
        "parametric":             "blue",
        "parametric_capped":      "red",
        "formula":                "green",
        "formula_clipped":        "orange",
        "formula_capped":         "red",
        "formula_clipped_capped": "red",
        "time_linear":            "purple",
        "time_mean":              "purple",
        "locf":                   "navy",
        "nocb":                   "navy",
        "flat_prior":             "red",
    }

    def _fig2_sigma_box(self):
        """
        Single-panel σ boxplot grouped by `method` from the imputation log,
        with a leftmost "covered" baseline box (Stage 2a observed σ before
        any imputation). Outliers shown as red dots. Logarithmic y-axis.
        """
        import matplotlib.pyplot as plt

        log_df = self._log_df
        groups: list[tuple[str, np.ndarray]] = []

        # 1. Covered baseline = Stage-2a observed σ (pre-snapshot before any fills)
        if self._snapshots:
            _, _, s_pre = self._snapshots[0]
            covered = s_pre[np.isfinite(s_pre) & (s_pre > 0)]
            if len(covered):
                groups.append(("covered", covered))

        # 2. One box per method in the log (skip empty-σ rows)
        if not log_df.empty:
            for method, sub in log_df.groupby("method", sort=False):
                vals = pd.to_numeric(sub["sigma_after"], errors="coerce").to_numpy()
                vals = vals[np.isfinite(vals) & (vals > 0)]
                if len(vals):
                    groups.append((str(method), vals))

        if not groups:
            return None

        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(groups)), 6))
        suffix = f" — {self.label}" if self.label else ""
        fig.suptitle(
            f"σ by imputer method{suffix}",
            fontsize=13, fontweight="bold",
        )

        labels  = [g[0] for g in groups]
        data    = [np.log10(g[1]) for g in groups]
        ns      = [len(g[1]) for g in groups]
        colors  = [
            _C[self._METHOD_COLORS.get(lbl, "grey")]
            for lbl in labels
        ]

        bp = ax.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(
                marker="o", markersize=2.5,
                markerfacecolor=_C["red"], markeredgecolor=_C["red"],
                alpha=0.5,
            ),
            medianprops=dict(color="black", linewidth=1.2),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor("black")

        # Annotate box counts
        ymax = max(d.max() for d in data)
        for i, n in enumerate(ns):
            ax.text(i + 1, ymax + 0.2, f"n={n:,}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("log₁₀(σ)  [log kt]")
        ax.set_xlabel("method")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Figure 3 — timeseries (requires acc_struct)
    # ------------------------------------------------------------------

    def _fig3_timeseries(self):
        import matplotlib.pyplot as plt

        acc     = self.acc_struct
        log_df  = self._log_df
        time_df = log_df[log_df["method"].isin(self._TIME_METHODS)].copy()
        if time_df.empty:
            return None

        ei      = acc.edge_index.set_index("edge_id")
        time_df = time_df.join(ei[["from_id", "to_id", "time", "valuation"]], on="edge_id")

        # Top-6 groups by mean |f_after|
        group_keys = ["from_id", "to_id", "valuation"]
        grp_mean   = (
            time_df.groupby(group_keys)["f_after"]
            .apply(lambda x: np.abs(x).mean())
            .sort_values(ascending=False)
        )
        top_groups = grp_mean.head(6).index.tolist()
        if not top_groups:
            return None

        n     = len(top_groups)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
        suffix = f" — {self.label}" if self.label else ""
        fig.suptitle(f"TimeImputer timeseries{suffix}", fontsize=13, fontweight="bold")

        try:
            time_order = {t: float(t) for t in acc.time_range}
        except (ValueError, TypeError):
            time_order = {t: i for i, t in enumerate(acc.time_range)}

        agg_f        = self.agg_result.f_agg
        imputed_set  = set(log_df["edge_id"].values)
        time_imp_set = set(time_df["edge_id"].values)

        for panel_idx, key in enumerate(top_groups):
            ax = axes[panel_idx // ncols][panel_idx % ncols]
            from_id, to_id, valuation = key

            group_edges = ei[
                (ei["from_id"] == from_id) &
                (ei["to_id"]   == to_id)   &
                (ei["valuation"] == valuation)
            ]

            raw_t, raw_f, imp_t, imp_f = [], [], [], []
            for eid, row in group_edges.iterrows():
                f_val = agg_f[eid]
                if not np.isfinite(f_val):
                    continue
                t_num = time_order.get(str(row["time"]), 0)
                if eid in time_imp_set:
                    imp_t.append(t_num)
                    imp_f.append(f_val)
                elif eid not in imputed_set:
                    raw_t.append(t_num)
                    raw_f.append(f_val)

            if raw_t:
                ax.scatter(raw_t, raw_f, s=30, color=_C["grey"],
                           label="raw", zorder=3, alpha=0.85)
            if imp_t:
                ax.scatter(imp_t, imp_f, s=50, marker="x", color=_C["orange"],
                           label="imputed", zorder=4, linewidths=1.8)

            n_from = acc.decode_node(int(from_id))
            n_to   = acc.decode_node(int(to_id))
            ax.set_title(
                f"{n_from[0]}/{n_from[2]} → {n_to[0]}/{n_to[2]}  [{valuation}]",
                fontsize=8, fontweight="bold",
            )
            ax.set_xlabel("time")
            ax.set_ylabel("f (kt)")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        plt.tight_layout()
        return fig


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


# ===========================================================================
# D  ConstDiagnostic — Stage 3a
# ===========================================================================

class ConstDiagnostic:
    """
    Three-figure diagnostic for a ConstraintRun produced by ConstraintEngine.

    Figure 1 — G sparsity  (fig1_G_sparsity.png)
        Coarsened black/white grid showing where G has non-zero entries.

    Figure 2 — Prior overview  (fig2_prior_overview.png)
        [0,0]  Constraint type breakdown — stacked bar (hard vs soft per type)
        [0,1]  a₀ distribution           — log₁₀ histogram of prior flow values
        [0,2]  σ_a distribution          — log₁₀ histogram of prior uncertainties

    Figure 3 — Constraint residuals  (fig3_residuals.png)
        [0,0]  |c| by year   — overlapping log₁₀ histograms, light→dark per timestep
                               (soft constraints only)
        [0,1]  σ_c by year   — overlapping log₁₀ histograms, light→dark per timestep
                               (soft constraints only)
        [1,0]  |ε₀| by cone  — log₁₀ histogram of |c − G·a₀|, coloured by cone_type
                               (reference: first timestep)
        [1,1]  |ε₀| vs |c|   — log-log scatter, coloured by cone_type
                               (reference: first timestep)

    Parameters
    ----------
    constraint_run :
        ConstraintRun dataclass (must have .G, .row_meta, .a0, .sigma_a,
        .timestep_data, .run_id).
    label : str
        Optional label for figure titles and summary output.
    """

    # cone_type → display colour
    _CONE_COLORS = {
        "zero":   _C["navy"],
        "nonneg": _C["orange"],
        "free":   _C["green"],
    }

    # constraint type → bar colour
    _TYPE_COLORS = {
        "mass_balance": _C["blue"],
        "sum_eq":       _C["green"],
        "ratio_eq":     _C["purple"],
        "ratio_bound":  _C["orange"],
        "flow_bound":   _C["red"],
    }

    def __init__(self, constraint_run, label: str = "") -> None:
        self.cr    = constraint_run
        self.label = label

    # ------------------------------------------------------------------
    # Classmethod — load from disk
    # ------------------------------------------------------------------

    @classmethod
    def from_saved(cls, run_dir: str, label: str = "") -> "ConstDiagnostic":
        """
        Reconstruct a ConstDiagnostic by loading a persisted ConstraintRun
        directly from its directory (no registry required).
        """
        import glob
        import types
        import scipy.sparse as _sp

        run_dir = run_dir.rstrip("/\\")

        G        = _sp.load_npz(os.path.join(run_dir, "G.npz"))
        row_meta = pd.read_parquet(os.path.join(run_dir, "row_meta.parquet"))
        a0       = np.load(os.path.join(run_dir, "a0.npy"))
        sigma_a  = np.load(os.path.join(run_dir, "sigma_a.npy"))

        ts_dir        = os.path.join(run_dir, "timesteps")
        timestep_data = {}
        for path in sorted(glob.glob(os.path.join(ts_dir, "t_*.npz"))):
            t_label = os.path.basename(path)[2:-4]   # strip "t_" prefix and ".npz"
            npz     = np.load(path)
            timestep_data[t_label] = {"c": npz["c"], "sigma_c": npz["sigma_c"]}

        cr         = types.SimpleNamespace()
        cr.G       = G
        cr.row_meta      = row_meta
        cr.a0            = a0
        cr.sigma_a       = sigma_a
        cr.timestep_data = timestep_data
        cr.run_id        = os.path.basename(run_dir)

        return cls(cr, label=label)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def plot(
        self,
        odir: str | None = None,
        show: bool = False,
        dpi:  int  = 150,
    ) -> list[str]:
        """Produce and save all three diagnostic figures."""
        import matplotlib.pyplot as plt

        run_id = getattr(self.cr, "run_id", "const_diag")
        if odir is None:
            odir = os.path.join("data/proc/const", run_id, "diag")

        paths = []

        for fig, fname, desc in [
            (self._fig1_G_sparsity(),    "fig1_G_sparsity.png",    "G sparsity    "),
            (self._fig2_prior_overview(), "fig2_prior_overview.png", "prior overview"),
            (self._fig3_residuals(),      "fig3_residuals.png",      "residuals     "),
        ]:
            p = _savefig(fig, os.path.join(odir, fname), dpi=dpi)
            paths.append(p)
            print(f"   ConstDiagnostic {desc} → {p}  ✓")
            if show:
                plt.show()
            plt.close(fig)

        return paths

    def summary(self) -> str:
        """Print and return a text summary of the constraint run."""
        cr  = self.cr
        rm  = cr.row_meta
        M, E = cr.G.shape
        nnz  = cr.G.nnz
        n_hard = int(rm["is_hard"].sum()) if len(rm) else 0
        n_soft = len(rm) - n_hard

        lines = [
            f"{'─'*60}",
            f"  CONSTRAINT DIAGNOSTICS{' — ' + self.label if self.label else ''}",
            f"{'─'*60}",
            f"  G shape          : {M:>8,} rows × {E:,} edges",
            f"  Non-zeros (nnz)  : {nnz:>8,}  ({100*nnz/(M*E):.4f} % dense)"
            if M * E else f"  Non-zeros (nnz)  : {nnz:>8,}",
            f"  Hard constraints : {n_hard:>8,}  ({n_hard/len(rm)*100:.1f} %)"
            if len(rm) else "  Hard constraints : 0",
            f"  Soft constraints : {n_soft:>8,}  ({n_soft/len(rm)*100:.1f} %)"
            if len(rm) else "  Soft constraints : 0",
            f"  Timesteps        : {len(cr.timestep_data):>8,}",
        ]

        if cr.timestep_data:
            t0    = next(iter(cr.timestep_data))
            c_t0  = cr.timestep_data[t0]["c"]
            eps0  = c_t0 - cr.G.dot(cr.a0)
            abs_e = np.abs(eps0)
            finite_e = abs_e[np.isfinite(abs_e)]
            if len(finite_e):
                soft_mask = rm["cone_type"].values == "free" if len(rm) else np.array([], bool)
                soft_e    = np.abs(eps0[soft_mask]) if soft_mask.any() else np.array([])
                soft_mean = soft_e[np.isfinite(soft_e)].mean() if len(soft_e) else float("nan")
                lines += [
                    f"\n  |ε₀| = |c − G·a₀| at t={t0}:",
                    f"    Mean (all)  : {finite_e.mean():>12.4f}",
                    f"    Max  (all)  : {finite_e.max():>12.4f}",
                    f"    Mean (soft) : {soft_mean:>12.4f}",
                ]

        lines.append(f"{'─'*60}")
        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # Figure builders
    # ------------------------------------------------------------------

    def _fig1_G_sparsity(self):
        import matplotlib.pyplot as plt

        G = self.cr.G
        M, E   = G.shape
        nnz    = G.nnz
        density = 100 * nnz / (M * E) if M * E else 0

        n_rows_grid = min(400, M)
        n_cols_grid = min(600, E)

        if nnz > 0:
            coo      = G.tocoo()
            row_bins = np.floor(coo.row * n_rows_grid / M).astype(int).clip(0, n_rows_grid - 1)
            col_bins = np.floor(coo.col * n_cols_grid / E).astype(int).clip(0, n_cols_grid - 1)
            grid = np.zeros((n_rows_grid, n_cols_grid), dtype=np.float32)
            np.add.at(grid, (row_bins, col_bins), 1)
            grid = (grid > 0).astype(np.float32)
        else:
            grid = np.zeros((n_rows_grid, n_cols_grid), dtype=np.float32)

        suffix = f" — {self.label}" if self.label else ""
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(grid, cmap="Greys", aspect="auto", interpolation="nearest")
        ax.set_title(
            f"G sparsity{suffix}   "
            f"{M:,}×{E:,} rows×edges   "
            f"{nnz:,} non-zeros   ({density:.4f} % dense)\n"
            f"display grid: {n_rows_grid}×{n_cols_grid} px  "
            f"(each cell = any non-zero present in bin)",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel(f"Edge index  (0 – {E - 1:,})")
        ax.set_ylabel(f"Constraint row  (0 – {M - 1:,})")
        plt.tight_layout()
        return fig

    def _fig2_prior_overview(self):
        import matplotlib.pyplot as plt

        suffix = f" — {self.label}" if self.label else ""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"Constraint prior overview{suffix}", fontsize=13, fontweight="bold")

        self._panel_type_bar(axes[0])
        self._panel_a0_dist(axes[1])
        self._panel_sigma_a_dist(axes[2])

        plt.tight_layout()
        return fig

    def _fig3_residuals(self):
        import matplotlib.pyplot as plt

        suffix = f" — {self.label}" if self.label else ""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Constraint residuals{suffix}", fontsize=13, fontweight="bold")

        self._panel_c_by_year(axes[0, 0])
        self._panel_sigma_c_by_year(axes[0, 1])
        self._panel_eps0_dist(axes[1, 0])
        self._panel_eps0_vs_c(axes[1, 1])

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Figure 2 panels
    # ------------------------------------------------------------------

    def _panel_type_bar(self, ax) -> None:
        rm     = self.cr.row_meta
        types  = rm["type"].unique().tolist() if len(rm) else []

        if not types:
            ax.text(0.5, 0.5, "No constraints", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Constraints by type", fontweight="bold")
            return

        x      = np.arange(len(types))
        width  = 0.6
        totals = []

        for i, t in enumerate(types):
            sub    = rm[rm["type"] == t]
            n_soft = int((~sub["is_hard"]).sum())
            n_hard = int(sub["is_hard"].sum())
            color  = self._TYPE_COLORS.get(t, _C["grey"])
            ax.bar(i, n_soft, width, color=color, alpha=0.45, edgecolor="white")
            ax.bar(i, n_hard, width, bottom=n_soft, color=color, alpha=0.9,
                   edgecolor="white", hatch="//")
            totals.append(n_soft + n_hard)

        max_total = max(totals) if totals else 1
        for i, total in enumerate(totals):
            ax.text(i, total + max_total * 0.01, f"{total:,}",
                    ha="center", va="bottom", fontsize=7)

        # Proxy legend entries for hard vs soft
        import matplotlib.patches as mpatches
        ax.legend(
            handles=[
                mpatches.Patch(facecolor=_C["grey"], alpha=0.45, label="soft"),
                mpatches.Patch(facecolor=_C["grey"], alpha=0.9, hatch="//", label="hard"),
            ],
            fontsize=8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=25, ha="right", fontsize=8)
        ax.set_title("Constraints by type", fontweight="bold")
        ax.set_ylabel("# constraint rows")
        ax.grid(axis="y", alpha=0.3)

    def _panel_a0_dist(self, ax) -> None:
        a0   = self.cr.a0
        vals = a0[np.isfinite(a0) & (a0 > 0)]
        if len(vals):
            ax.hist(np.log10(vals), bins=60, color=_C["blue"],
                    alpha=0.8, edgecolor="none", density=True)
        ax.set_title("Prior a₀ distribution", fontweight="bold")
        ax.set_xlabel("log₁₀(a₀)")
        ax.set_ylabel("density")
        ax.grid(axis="y", alpha=0.3)

    def _panel_sigma_a_dist(self, ax) -> None:
        sa   = self.cr.sigma_a
        vals = sa[np.isfinite(sa) & (sa > 0)]
        if len(vals):
            ax.hist(np.log10(vals), bins=60, color=_C["purple"],
                    alpha=0.8, edgecolor="none", density=True)
        ax.set_title("Prior σ_a distribution", fontweight="bold")
        ax.set_xlabel("log₁₀(σ_a)")
        ax.set_ylabel("density")
        ax.grid(axis="y", alpha=0.3)

    # ------------------------------------------------------------------
    # Figure 3 panels
    # ------------------------------------------------------------------

    def _soft_mask(self) -> np.ndarray:
        return self.cr.row_meta["cone_type"].values == "free"

    def _eps0_ref(self):
        """Return (eps0, c_t0, t0_label) for the first timestep, or (None, None, None)."""
        cr = self.cr
        if not cr.timestep_data:
            return None, None, None
        t0   = next(iter(cr.timestep_data))
        c_t0 = cr.timestep_data[t0]["c"]
        eps0 = c_t0 - cr.G.dot(cr.a0)
        return eps0, c_t0, t0

    def _panel_c_by_year(self, ax) -> None:
        import matplotlib.pyplot as plt

        cr         = self.cr
        soft_mask  = self._soft_mask()

        if not soft_mask.any():
            ax.text(0.5, 0.5, "No soft constraints", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("|c| distribution (soft constraints)", fontweight="bold")
            return

        timesteps = list(cr.timestep_data.keys())
        n    = len(timesteps)
        cmap = plt.cm.Blues

        for i, t_label in enumerate(timesteps):
            c    = cr.timestep_data[t_label]["c"][soft_mask]
            vals = np.abs(c)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if not len(vals):
                continue
            color = cmap(0.3 + 0.7 * i / max(n - 1, 1))
            ax.hist(np.log10(vals), bins=40, alpha=0.5, color=color,
                    histtype="stepfilled", edgecolor="none",
                    label=t_label, density=True)

        ax.set_title("|c| distribution (soft constraints)", fontweight="bold")
        ax.set_xlabel("log₁₀(|c|)")
        ax.set_ylabel("density")
        if n <= 12:
            ax.legend(fontsize=6, framealpha=0.6)
        ax.grid(axis="y", alpha=0.3)

    def _panel_sigma_c_by_year(self, ax) -> None:
        import matplotlib.pyplot as plt

        cr        = self.cr
        soft_mask = self._soft_mask()

        if not soft_mask.any():
            ax.text(0.5, 0.5, "No soft constraints", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("σ_c distribution (soft constraints)", fontweight="bold")
            return

        timesteps = list(cr.timestep_data.keys())
        n    = len(timesteps)
        cmap = plt.cm.Greens

        for i, t_label in enumerate(timesteps):
            sc   = cr.timestep_data[t_label]["sigma_c"][soft_mask]
            vals = sc[np.isfinite(sc) & (sc > 0)]
            if not len(vals):
                continue
            color = cmap(0.3 + 0.7 * i / max(n - 1, 1))
            ax.hist(np.log10(vals), bins=40, alpha=0.5, color=color,
                    histtype="stepfilled", edgecolor="none",
                    label=t_label, density=True)

        ax.set_title("σ_c distribution (soft constraints)", fontweight="bold")
        ax.set_xlabel("log₁₀(σ_c)")
        ax.set_ylabel("density")
        if n <= 12:
            ax.legend(fontsize=6, framealpha=0.6)
        ax.grid(axis="y", alpha=0.3)

    def _panel_eps0_dist(self, ax) -> None:
        eps0, _c, t0 = self._eps0_ref()
        if eps0 is None:
            ax.text(0.5, 0.5, "No timestep data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("|ε₀| by cone type", fontweight="bold")
            return

        rm = self.cr.row_meta
        for cone_type, color in self._CONE_COLORS.items():
            mask = rm["cone_type"].values == cone_type
            if not mask.any():
                continue
            vals = np.abs(eps0[mask])
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if not len(vals):
                continue
            ax.hist(np.log10(vals), bins=40, alpha=0.6, color=color,
                    histtype="stepfilled", edgecolor="none",
                    label=cone_type, density=True)

        ax.set_title(f"|ε₀| = |c − G·a₀| by cone type  (t={t0})", fontweight="bold")
        ax.set_xlabel("log₁₀(|ε₀|)")
        ax.set_ylabel("density")
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, framealpha=0.6)
        ax.grid(axis="y", alpha=0.3)

    def _panel_eps0_vs_c(self, ax) -> None:
        eps0, c_t0, t0 = self._eps0_ref()
        if eps0 is None:
            ax.text(0.5, 0.5, "No timestep data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("|ε₀| vs |c|", fontweight="bold")
            return

        rm       = self.cr.row_meta
        all_cx   = np.abs(c_t0)
        all_epsx = np.abs(eps0)

        for cone_type, color in self._CONE_COLORS.items():
            mask  = rm["cone_type"].values == cone_type
            cx    = all_cx[mask]
            epsx  = all_epsx[mask]
            valid = np.isfinite(cx) & np.isfinite(epsx) & (cx > 0) & (epsx > 0)
            if not valid.any():
                continue
            ax.scatter(cx[valid], epsx[valid], s=6, alpha=0.4, color=color,
                       label=cone_type, linewidths=0)

        # Diagonal reference line
        valid_all = np.isfinite(all_cx) & np.isfinite(all_epsx) & (all_cx > 0) & (all_epsx > 0)
        if valid_all.any():
            lo = min(all_cx[valid_all].min(), all_epsx[valid_all].min())
            hi = max(all_cx[valid_all].max(), all_epsx[valid_all].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="ε₀ = c")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"|ε₀| vs |c|  (t={t0})", fontweight="bold")
        ax.set_xlabel("|c|")
        ax.set_ylabel("|ε₀|")
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8, framealpha=0.6)
        ax.grid(alpha=0.3, which="both")
