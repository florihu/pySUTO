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
    feed_run_id : full run identifier used for titles and filenames.
    feed_id   : short feed identifier (e.g. "baci").
    acc_id    : accounting structure id for titles.
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
        feed_run_id: str,
        feed_id:     str,
        acc_id:      str,
        raw_df:      pd.DataFrame,
        ct,
        F_src:       np.ndarray,
        result,
        conc_file:   Optional[str] = None,
    ) -> None:
        self.feed_run_id = feed_run_id
        self.feed_id     = feed_id
        self.acc_id      = acc_id
        self.raw_df      = raw_df
        self.ct          = ct
        self.F_src       = F_src
        self.result      = result
        self.conc_file   = conc_file

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
            f"Feed diagnostics — {self.feed_run_id}  (acc: {self.acc_id})",
            fontsize=14, fontweight="bold",
        )

        self._panel_row_coverage(axes[0, 0])
        self._panel_unmatched_reason(axes[0, 1])
        self._panel_matched_by_sector(axes[0, 2])
        self._panel_fsrc_distribution(axes[1, 0])
        self._panel_cvroot_distribution(axes[1, 1])
        self._panel_year_coverage(axes[1, 2])

        plt.tight_layout()

        path = os.path.join(odir, f"{self.feed_run_id}_feed.png")
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
            f"  FEED DIAGNOSTICS — {self.feed_run_id}  (acc: {self.acc_id})",
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
            ax.hist(valid, bins=30, color=_C["red"], edgecolor="white", linewidth=0.5)
        ax.set_title("CV_root distribution (%)", fontweight="bold")
        ax.set_xlabel("CV (%)")
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

    Figure 2b — CV-by-method boxplot  (<label>_imputation_cv_box.png)
        Mirror of fig 2 with linear y-axis and CV (σ/|f|) instead of σ.

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

    # method label → palette key used by both boxplots and step coloring.
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

    def __init__(self, agg_result, acc_struct=None, label: str = "") -> None:
        self.agg_result = agg_result
        self.acc_struct = acc_struct
        self.label      = label
        self._log_df    = agg_result.imputation_log.as_dataframe()
        self._snapshots = self._build_snapshots()
        # Single source of truth: snapshot label → matplotlib color.  All
        # panels (fig 1 a..d, both boxplots) read from this map so step N
        # always renders in the same colour everywhere.
        self._step_color = self._build_step_color_map()

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

        # Figure 2b — CV-by-method boxplot (1×1, parallel to σ box)
        fig_cv = self._fig_cv_box()
        if fig_cv is not None:
            if odir is not None:
                p = _savefig(fig_cv, os.path.join(odir, f"{tag}_imputation_cv_box.png"), dpi=dpi)
                paths.append(p)
                print(f"   Imputation CV boxplot      → {p}  ✓")
            if show:
                plt.show()
            plt.close(fig_cv)

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

    def _step_dominant_method(self, step: int) -> str:
        """Most-common method label across all fills for one pipeline step."""
        if self._log_df.empty:
            return "covered"
        sub = self._log_df[self._log_df["step"] == step]
        if sub.empty:
            return "covered"
        return str(sub["method"].value_counts().index[0])

    def _build_step_color_map(self) -> dict[str, str]:
        """
        Map snapshot labels to matplotlib colors via each step's dominant
        method. The "pre" snapshot is grey; everything else inherits from
        ``_METHOD_COLORS``.
        """
        mapping: dict[str, str] = {}
        for label, _, _ in self._snapshots:
            if label == "pre":
                mapping[label] = _C["grey"]
                continue
            method = label.split("(", 1)[1].rstrip(")") if "(" in label else label
            mapping[label] = _C[self._METHOD_COLORS.get(method, "grey")]
        return mapping

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
        for lbl, f, _ in self._snapshots:
            vals = f[np.isfinite(f) & (f > 0)]
            if not len(vals):
                continue
            ax.hist(np.log10(vals), bins=60, alpha=0.45,
                    color=self._step_color[lbl], label=lbl,
                    density=True, histtype="stepfilled", edgecolor="none")
        ax.set_title("F_root distribution per step", fontweight="bold")
        ax.set_xlabel("log₁₀(f)  [log kt]")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, framealpha=0.6)
        ax.grid(axis="y", alpha=0.3)

    def _panel_coverage(self, ax) -> None:
        labels  = [s[0] for s in self._snapshots]
        counts  = [int(np.isfinite(s[1]).sum()) for s in self._snapshots]
        n_total = len(self._snapshots[0][1])

        bars = ax.barh(
            labels, counts,
            color=[self._step_color[lbl] for lbl in labels],
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
        Per-step CV (= σ / |f|) histogram on a linear axis. CV strips out
        the magnitude coupling and exposes whether a step actually inflates
        relative uncertainty.
        """
        for lbl, fv, sv in self._snapshots:
            with np.errstate(divide="ignore", invalid="ignore"):
                cv = np.where(np.abs(fv) > 0, sv / np.abs(fv), np.nan)
            vals = cv[np.isfinite(cv) & (cv > 0)]
            if not len(vals):
                continue
            ax.hist(vals, bins=60, alpha=0.45,
                    color=self._step_color[lbl], label=lbl,
                    density=True, histtype="stepfilled", edgecolor="none")
        ax.set_title("CV (σ/|f|) distribution per step", fontweight="bold")
        ax.set_xlabel("CV")
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
        steps  = [int(s) for s in step_counts.index]
        labels = [f"step {s} ({self._step_dominant_method(s)})" for s in steps]
        counts = step_counts.values
        # Read from the shared step→color map; identical to the other
        # panels, so step N renders in the same colour everywhere.
        colors = [self._step_color.get(lbl, _C["grey"]) for lbl in labels]

        bars = ax.bar(
            [str(s) for s in steps], counts,
            color=colors,
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
    # Figure 2b — CV-by-method boxplot (linear, parallel to σ box)
    # ------------------------------------------------------------------

    def _fig_cv_box(self):
        """
        Single-panel CV boxplot grouped by `method` from the imputation log,
        with a leftmost "covered" baseline box (Stage 2a observed CV before
        any imputation). Linear y-axis.
        """
        import matplotlib.pyplot as plt

        log_df = self._log_df
        groups: list[tuple[str, np.ndarray]] = []

        # 1. Covered baseline = Stage-2a observed CV (pre-snapshot before fills).
        if self._snapshots:
            _, f_pre, s_pre = self._snapshots[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                cv_pre = np.where(np.abs(f_pre) > 0, s_pre / np.abs(f_pre), np.nan)
            covered = cv_pre[np.isfinite(cv_pre) & (cv_pre > 0)]
            if len(covered):
                groups.append(("covered", covered))

        # 2. One box per method in the log.
        if not log_df.empty:
            for method, sub in log_df.groupby("method", sort=False):
                f_vals = pd.to_numeric(sub["f_after"], errors="coerce").to_numpy()
                s_vals = pd.to_numeric(sub["sigma_after"], errors="coerce").to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    cv = np.where(np.abs(f_vals) > 0, s_vals / np.abs(f_vals), np.nan)
                cv = cv[np.isfinite(cv) & (cv > 0)]
                if len(cv):
                    groups.append((str(method), cv))

        if not groups:
            return None

        fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(groups)), 6))
        suffix = f" — {self.label}" if self.label else ""
        fig.suptitle(
            f"CV by imputer method{suffix}",
            fontsize=13, fontweight="bold",
        )

        labels  = [g[0] for g in groups]
        data    = [g[1] for g in groups]
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

        ymax = max(d.max() for d in data)
        for i, n in enumerate(ns):
            ax.text(i + 1, ymax * 1.02, f"n={n:,}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("CV (σ/|f|)")
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


# ===========================================================================
# AggDiagnostic — post-aggregation distribution diagnostics (Stage 2a)
# ===========================================================================

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

    def __init__(self, agg_result, label: str = "") -> None:
        # Snapshot the arrays we need at construction time, because
        # ImputationEngine.run() mutates agg_result.f_agg / sigma_agg in place.
        # The per-feed F / Sigma matrices are not mutated by imputation.
        self.label      = label
        self._f_agg     = np.array(agg_result.f_agg, dtype=float, copy=True)
        self._sigma_agg = np.array(agg_result.sigma_agg, dtype=float, copy=True)
        self._coverage  = np.array(agg_result.coverage, dtype=bool, copy=True)
        self._F         = np.array(agg_result.F, dtype=float, copy=True)
        self._Sigma     = np.array(agg_result.Sigma, dtype=float, copy=True)
        self._feed_ids  = list(agg_result.feed_ids)

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

        tag   = self.label.replace(" ", "_") or "agg"
        paths: list[str] = []

        fig = self._fig_distributions()
        if odir is not None:
            p = _savefig(fig, os.path.join(odir, f"{tag}_agg_distributions.png"), dpi=dpi)
            paths.append(p)
            print(f"   Aggregation distributions   → {p}  ✓")
        if show:
            plt.show()
        plt.close(fig)
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
        coverage  = self._coverage
        F_mat     = self._F
        Sigma_mat = self._Sigma
        feed_ids  = self._feed_ids

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        title = f"Aggregation diagnostics — {self.label}" if self.label else "Aggregation diagnostics"
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
            mask = np.isfinite(f_k) & np.isfinite(s_k) & (np.abs(f_k) > 0) & (s_k > 0)
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


# ===========================================================================
# SolverDiag — Stage 3b post-solve diagnostics
# ===========================================================================

# Per constraint-type colour binding — reused across every SolverDiag figure.
# Real ConstraintEngine.row_meta uses {mass_balance, sum_eq, ratio_eq,
# ratio_bound}; flow bounds live in u/l and never appear in row_meta. Toy
# fixtures may omit the "type" column entirely, in which case we fall back
# to cone_type.
_TYPE_COLORS = {
    "mass_balance": _C["navy"],     # Type 1 (hard equality)
    "sum_eq":       _C["blue"],     # Type 2 (soft sum)
    "ratio_eq":     _C["purple"],   # Type 3 (soft ratio)
    "ratio_bound":  _C["orange"],   # Type 4 (hard ratio bound)
    "flow_bound":   _C["green"],    # Type 5 (hard flow bound — synthetic, from u/l)
}

# Hue palette for SolverDiag.compare(). Order matters: first run gets blue.
_HUE_PALETTE = ["blue", "orange", "green", "purple", "red", "navy"]

# Plot IDs hueable under SolverDiag.compare(). Excluded: SD-R-07 (per-edge
# semantics differ across runs) and every SD-T-* (cross-sections).
_HUEABLE_IDS = ("SD-R-01", "SD-R-02", "SD-R-03", "SD-R-04", "SD-R-05", "SD-R-06")


def _apply_modern_style(ax) -> None:
    """Tighten the look of one axes: no top/right spines, light grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)


class SolverDiag:
    """
    Stage 3b solver diagnostics — runs after ``ClarabelSolver.solve_all()``.

    Two construction modes:

    * **Live**: pass the ``results`` dict + ``ConstraintRun`` directly
      (typical inside a solver script, before/after ``persist()``).
    * **Persisted**: ``SolverDiag.from_opt_id(opt_run_id)`` reconstructs
      the run from ``{output_dir}/registry.json`` and per-timestep ``.npz``
      files, plus the matching ``ConstraintRun`` from ``ConstraintRunRegistry``.

    Plot portfolio (see README §Stage 3c for the full ID table):

    * Run-scope  : SD-R-01 … SD-R-07 (x = timestep)
    * Per-timestep cross-section : SD-T-01 … SD-T-06
      (columns = t_min, t_worst, t_max; histograms only)

    Parameters
    ----------
    results : dict[str, OptimResult]
        Output of ``ClarabelSolver.solve_all()``.
    constraint_run : ConstraintRun
        The ConstraintRun the solver consumed — supplies G, row_meta,
        a0, sigma_a, u, l, and per-timestep c / sigma_c.
    label : str
        Optional label for figure titles and output filenames.
    sigma_opt : dict[str, np.ndarray] | None
        Optional per-timestep σ_opt from ``OptUncertainty`` — when provided
        unlocks the σ_opt / σ_a panel in SD-T-03.
    """

    # ---- IDs (kept in plot() iteration order) ------------------------------
    RUN_PLOT_IDS = (
        "SD-R-01",  # Solver health
        "SD-R-02",  # Mass-balance leakage
        "SD-R-03",  # Soft-slack stress
        "SD-R-04",  # Hard-inequality activity
        "SD-R-05",  # Prior–posterior distance
        "SD-R-06",  # Warm-start delta
        "SD-R-07",  # F*(t) evolution
    )
    TIMESTEP_PLOT_IDS = (
        "SD-T-01",  # Residual digest
        "SD-T-02",  # Soft-slack z-scores
        "SD-T-03",  # Solution vs prior cross-section
        "SD-T-04",  # Per-type binding rate
        "SD-T-05",  # |a*| magnitude histogram
        "SD-T-06",  # c* vs c
    )

    def __init__(
        self,
        results,
        constraint_run,
        label: str = "",
        sigma_opt: Optional[dict] = None,
    ) -> None:
        # Ordered list of timestep labels = key order in results dict
        self.label         = label
        self._results      = dict(results)
        self._timesteps    = list(self._results.keys())
        self._const_run    = constraint_run
        self._sigma_opt    = dict(sigma_opt) if sigma_opt else None

        # Default odir cache — populated by from_opt_id; None for live mode
        self._default_odir: Optional[str] = None

        # Pre-compute row classification & masks once
        self._row_info = self._classify_rows(constraint_run)

        # Pre-compute per-timestep ||eps||_2 for t_worst selection
        self._eps_norms = {
            t: float(np.linalg.norm(self._results[t].eps_opt))
            for t in self._timesteps
        }

    # ------------------------------------------------------------------
    # Construction from persisted run
    # ------------------------------------------------------------------

    @classmethod
    def from_opt_id(
        cls,
        opt_run_id: str,
        output_dir: str = "data/proc/opt",
        const_dir:  str = "data/proc/const",
        label:      str = "",
    ) -> "SolverDiag":
        """Reconstruct results + ConstraintRun from disk."""
        import json
        # Local import to keep diag.py at the leaf of the import graph for
        # CI environments without the full Stage-3 stack installed.
        from pysuto.solver import OptimResult, ITER_DTYPE
        from pysuto.constraints import ConstraintRunRegistry

        reg_path = os.path.join(output_dir, "registry.json")
        if not os.path.exists(reg_path):
            raise FileNotFoundError(f"No registry at {reg_path}")
        with open(reg_path) as fh:
            registry = json.load(fh)
        if opt_run_id not in registry:
            raise KeyError(
                f"opt_run_id '{opt_run_id}' not in {reg_path}. "
                f"Known: {sorted(registry.keys())}"
            )
        entry = registry[opt_run_id]

        # Reconstruct ConstraintRun
        const_run = ConstraintRunRegistry(const_dir).load(entry["const_run_id"])

        # Reconstruct each OptimResult from the .npz + per_timestep entry
        ts_dir = os.path.join(output_dir, opt_run_id, "timesteps")
        results: dict[str, OptimResult] = {}
        for t in entry["timesteps"]:
            npz = np.load(os.path.join(ts_dir, f"t_{t}.npz"))
            per = entry["per_timestep"][t]
            results[t] = OptimResult(
                timestep      = t,
                F_opt         = npz["F_opt"],
                eps_opt       = npz["eps_opt"],
                c_star        = npz["c_star"],
                residuals     = npz["residuals"],
                objective     = float(per["objective"]),
                iterations    = int(per["iterations"]),
                solve_time_s  = float(per["solve_time_s"]),
                solver_status = str(per["solver_status"]),
                convergence   = dict(per["convergence"]),
                iter_history  = npz["iter_history"] if "iter_history" in npz.files else None,
            )

        instance = cls(results, const_run, label=label or opt_run_id)
        instance._default_odir = os.path.join(output_dir, opt_run_id, "diag")
        return instance

    # ------------------------------------------------------------------
    # Row classification helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_rows(const_run) -> dict:
        """
        Pre-compute masks over row_meta for quick filtering across figures.

        Returns dict with int-array indices keyed by:
            'mass_balance', 'sum_eq', 'ratio_eq', 'ratio_bound',
            'zero', 'free', 'nonneg'
        and a string array 'type_of_row' parallel to row_meta.
        """
        rm = const_run.row_meta
        ct = rm["cone_type"].to_numpy() if "cone_type" in rm.columns else np.array([], dtype=str)

        if "type" in rm.columns:
            tp = rm["type"].astype(str).to_numpy()
        else:
            # Toy fixtures: synthesise type from (cone_type, is_hard)
            is_hard = rm["is_hard"].to_numpy() if "is_hard" in rm.columns else np.zeros(len(rm), dtype=bool)
            tp = np.full(len(rm), "unknown", dtype=object)
            tp[(ct == "zero")   & is_hard] = "mass_balance"
            tp[(ct == "free")  & ~is_hard] = "sum_eq"           # plausible default
            tp[(ct == "nonneg") & is_hard] = "ratio_bound"
            tp = tp.astype(str)

        info = {
            "type_of_row": tp,
            "cone_of_row": ct,
            "mass_balance": np.where(tp == "mass_balance")[0],
            "sum_eq":       np.where(tp == "sum_eq")[0],
            "ratio_eq":     np.where(tp == "ratio_eq")[0],
            "ratio_bound":  np.where(tp == "ratio_bound")[0],
            "zero":         np.where(ct == "zero")[0],
            "free":         np.where(ct == "free")[0],
            "nonneg":       np.where(ct == "nonneg")[0],
        }
        return info

    def _t_worst(self) -> str:
        """Timestep with largest ||eps||_2."""
        return max(self._eps_norms, key=self._eps_norms.get)

    def _resolve_timesteps(self, timesteps) -> tuple[str, str, str]:
        if timesteps is None:
            ts_min  = self._timesteps[0]
            ts_max  = self._timesteps[-1]
            ts_wrst = self._t_worst()
            return ts_min, ts_wrst, ts_max
        if len(timesteps) != 3:
            raise ValueError(f"timesteps must be a triple, got {timesteps!r}")
        for t in timesteps:
            if t not in self._results:
                raise KeyError(f"timestep '{t}' not in results")
        return tuple(timesteps)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Print and return a one-block text digest of the run."""
        n = len(self._timesteps)
        statuses = [self._results[t].solver_status for t in self._timesteps]
        n_solved = sum(s == "Solved" for s in statuses)
        n_almost = sum(s == "AlmostSolved" for s in statuses)
        total_iter = sum(self._results[t].iterations for t in self._timesteps)
        total_time = sum(self._results[t].solve_time_s for t in self._timesteps)

        worst_t = self._t_worst()
        worst_eps = self._eps_norms[worst_t]

        rinfo = self._row_info
        n_mb   = len(rinfo["mass_balance"])
        n_soft = len(rinfo["sum_eq"]) + len(rinfo["ratio_eq"])
        n_rb   = len(rinfo["ratio_bound"])

        # Max hard-row residual across timesteps (mass-balance only)
        if n_mb:
            mb_max = max(
                float(np.max(np.abs(self._results[t].residuals[rinfo["mass_balance"]])))
                for t in self._timesteps
            )
        else:
            mb_max = 0.0

        lines = []
        lines.append(f"SolverDiag ({self.label or 'unlabeled'})")
        lines.append(f"  timesteps : {n}  (solved={n_solved}, almost={n_almost})")
        lines.append(f"  KKT       : total_iter={total_iter}  total_time={total_time:.2f}s")
        lines.append(f"  rows      : mass_balance={n_mb}  soft={n_soft}  ratio_bound={n_rb}")
        lines.append(f"  worst ε   : t={worst_t}  ‖ε‖₂={worst_eps:.3e}")
        lines.append(f"  max |Gx-c| on mass-balance rows: {mb_max:.3e}")
        text = "\n".join(lines)
        print(text)
        return text

    def plot(
        self,
        odir: Optional[str] = None,
        timesteps: Optional[tuple] = None,
        show: bool = False,
        dpi:  int  = 150,
    ) -> list[str]:
        """Render the full 13-figure portfolio.

        Parameters
        ----------
        odir : str | None
            Output directory. If None, defaults to ``{output_dir}/{opt_run_id}/diag``
            when this instance came from ``from_opt_id``; otherwise required.
        timesteps : tuple[str, str, str] | None
            (t_min_col, t_mid_col, t_max_col) for Scope-2 figures. If None,
            defaults to (first, worst-ε, last).
        """
        import matplotlib.pyplot as plt  # noqa: F401 — pre-import for backend setup
        if odir is None:
            odir = self._default_odir
        if odir is None:
            raise ValueError(
                "odir is required. Either pass odir=..., or build the diag via "
                "SolverDiag.from_opt_id(...) so the default lands under the run dir."
            )
        os.makedirs(odir, exist_ok=True)

        ts_triple = self._resolve_timesteps(timesteps)
        tag = self.label.replace(" ", "_") or "solver"

        paths: list[str] = []

        # ---- Scope 1: run-scope ----------------------------------------
        for fn, pid, fname in [
            (self._fig_solver_health,    "SD-R-01", "sd_r_01_solver_health"),
            (self._fig_mass_balance,     "SD-R-02", "sd_r_02_mass_balance"),
            (self._fig_soft_slack,       "SD-R-03", "sd_r_03_soft_slack"),
            (self._fig_hard_active,      "SD-R-04", "sd_r_04_hard_active"),
            (self._fig_prior_posterior,  "SD-R-05", "sd_r_05_prior_posterior"),
            (self._fig_warmstart_delta,  "SD-R-06", "sd_r_06_warmstart_delta"),
        ]:
            fig = fn(hue=None)
            p = _savefig(fig, os.path.join(odir, f"{tag}_{fname}.png"), dpi=dpi)
            print(f"   {pid}  → {p}  ✓")
            paths.append(p)
            if not show:
                import matplotlib.pyplot as plt; plt.close(fig)

        # SD-R-07 has no hue parameter
        fig = self._fig_f_evolution()
        p = _savefig(fig, os.path.join(odir, f"{tag}_sd_r_07_f_evolution.png"), dpi=dpi)
        print(f"   SD-R-07  → {p}  ✓")
        paths.append(p)
        if not show:
            import matplotlib.pyplot as plt; plt.close(fig)

        # ---- Scope 2: timestep cross-section --------------------------
        for fn, pid, fname in [
            (self._fig_residual_digest,    "SD-T-01", "sd_t_01_residual_digest"),
            (self._fig_zscore_soft,        "SD-T-02", "sd_t_02_zscore_soft"),
            (self._fig_solution_vs_prior,  "SD-T-03", "sd_t_03_solution_vs_prior"),
            (self._fig_binding_rate,       "SD-T-04", "sd_t_04_binding_rate"),
            (self._fig_solution_magnitude, "SD-T-05", "sd_t_05_solution_magnitude"),
            (self._fig_cstar_vs_c,         "SD-T-06", "sd_t_06_cstar_vs_c"),
        ]:
            fig = fn(timesteps=ts_triple)
            p = _savefig(fig, os.path.join(odir, f"{tag}_{fname}.png"), dpi=dpi)
            print(f"   {pid}  → {p}  ✓")
            paths.append(p)
            if not show:
                import matplotlib.pyplot as plt; plt.close(fig)

        return paths

    # ------------------------------------------------------------------
    # Multi-run comparison
    # ------------------------------------------------------------------

    @classmethod
    def compare(
        cls,
        diags: dict,
        odir: str,
        plot_ids: Optional[list] = None,
        label: str = "compare",
        dpi:   int = 150,
    ) -> list[str]:
        """
        Overlay run-scope diagnostics from multiple SolverDiag instances.

        Parameters
        ----------
        diags : dict[str, SolverDiag]
            ``{hue_label: diag}`` — up to 6 entries recommended (palette size).
        odir : str
            Output directory. Files land at ``{odir}/{label}_{plot_id}.png``.
        plot_ids : list[str] | None
            Subset of hueable IDs to render. Default is all of _HUEABLE_IDS.
        """
        if not diags:
            raise ValueError("compare() requires at least one diag")
        if plot_ids is None:
            plot_ids = list(_HUEABLE_IDS)
        not_hueable = set(plot_ids) - set(_HUEABLE_IDS)
        if not_hueable:
            raise ValueError(
                f"plot_ids contain non-hueable plots: {sorted(not_hueable)}. "
                f"Hueable: {_HUEABLE_IDS}"
            )
        os.makedirs(odir, exist_ok=True)

        # Build (hue_label, diag, color) tuples once
        hue: list = []
        for k, (lbl, dg) in enumerate(diags.items()):
            color = _C[_HUE_PALETTE[k % len(_HUE_PALETTE)]]
            hue.append((lbl, dg, color))

        # Map plot_id → (method_name, filename_stem)
        plot_dispatch = {
            "SD-R-01": ("_fig_solver_health",    "sd_r_01_solver_health"),
            "SD-R-02": ("_fig_mass_balance",     "sd_r_02_mass_balance"),
            "SD-R-03": ("_fig_soft_slack",       "sd_r_03_soft_slack"),
            "SD-R-04": ("_fig_hard_active",      "sd_r_04_hard_active"),
            "SD-R-05": ("_fig_prior_posterior",  "sd_r_05_prior_posterior"),
            "SD-R-06": ("_fig_warmstart_delta",  "sd_r_06_warmstart_delta"),
        }

        # Render using the first diag as host; each method accepts hue=[(label,diag,color),...]
        host = next(iter(diags.values()))
        paths: list[str] = []
        for pid in plot_ids:
            method_name, stem = plot_dispatch[pid]
            fig = getattr(host, method_name)(hue=hue)
            p = _savefig(fig, os.path.join(odir, f"{label}_{stem}.png"), dpi=dpi)
            print(f"   {pid} (compare) → {p}  ✓")
            paths.append(p)
            import matplotlib.pyplot as plt; plt.close(fig)
        return paths

    # ==================================================================
    # Internal: data extractors  (used by figure methods)
    # ==================================================================

    def _ts_index(self) -> np.ndarray:
        """X-axis array: integer indices [0..T-1] for plotting line series."""
        return np.arange(len(self._timesteps))

    def _ts_labels_for_xticks(self) -> list[str]:
        return [str(t) for t in self._timesteps]

    def _conv_series(self, key: str) -> np.ndarray:
        """Pull ``OptimResult.convergence[key]`` over time."""
        return np.array(
            [float(self._results[t].convergence.get(key, np.nan))
             for t in self._timesteps]
        )

    def _scalar_series(self, attr: str) -> np.ndarray:
        return np.array(
            [float(getattr(self._results[t], attr)) for t in self._timesteps]
        )

    def _mb_residuals(self, t: str) -> np.ndarray:
        idx = self._row_info["mass_balance"]
        if idx.size == 0:
            return np.zeros(0)
        return self._results[t].residuals[idx]

    def _eps_for_type(self, t: str, type_key: str) -> np.ndarray:
        """ε values for soft rows of a given type (sum_eq or ratio_eq)."""
        idx = self._row_info[type_key]
        if idx.size == 0:
            return np.zeros(0)
        # eps_opt is in free-cone row order (length n_free); but residuals
        # are in original row order, so derive ε from c_star - c.
        c   = np.asarray(self._const_run.timestep_data[t]["c"])
        eps = self._results[t].c_star - c
        return eps[idx]

    def _sigma_c_for_type(self, t: str, type_key: str) -> np.ndarray:
        idx = self._row_info[type_key]
        if idx.size == 0:
            return np.zeros(0)
        sc = np.asarray(self._const_run.timestep_data[t]["sigma_c"])
        return sc[idx]

    def _nonneg_slack(self, t: str) -> np.ndarray:
        """s = c - G·a for nonneg-cone rows (slack ≥ 0 ⇔ constraint satisfied)."""
        idx = self._row_info["nonneg"]
        if idx.size == 0:
            return np.zeros(0)
        return -self._results[t].residuals[idx]  # s = c - Gx = -(Gx - c)

    def _flow_bound_slack(self, t: str) -> tuple[np.ndarray, np.ndarray]:
        """Flow-bound slacks (u_i - a_i, a_i - l_i) over all edges."""
        F = self._results[t].F_opt
        u = np.asarray(self._const_run.u, dtype=float)
        l = np.asarray(self._const_run.l, dtype=float)
        return (u - F), (F - l)

    # ==================================================================
    # Figures — Scope 1 (run-scope)
    # ==================================================================

    def _fig_solver_health(self, hue=None):
        """SD-R-01: 4 stacked panels."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
        fig.suptitle(self._title("Solver health — SD-R-01"), fontsize=13, fontweight="bold")

        runs = hue if hue else [(self.label or "run", self, _C["blue"])]
        x_master = self._ts_index()

        for lbl, dg, color in runs:
            x = dg._ts_index()
            axes[0].plot(x, dg._scalar_series("objective"),
                         color=color, lw=1.6, marker="o", ms=3, label=lbl)
            axes[1].plot(x, dg._scalar_series("iterations"),
                         color=color, lw=1.6, marker="o", ms=3, label=lbl)
            axes[2].plot(x, dg._scalar_series("solve_time_s"),
                         color=color, lw=1.6, marker="o", ms=3, label=lbl)
            axes[3].semilogy(x, np.maximum(dg._conv_series("res_primal"), 1e-16),
                             color=color, lw=1.4, ls="-",  label=f"{lbl}: res_primal")
            axes[3].semilogy(x, np.maximum(dg._conv_series("res_dual"),   1e-16),
                             color=color, lw=1.4, ls="--", label=f"{lbl}: res_dual")
            axes[3].semilogy(x, np.maximum(dg._conv_series("gap_abs"),    1e-16),
                             color=color, lw=1.4, ls=":",  label=f"{lbl}: gap_abs")

        axes[0].set_ylabel("objective");      axes[0].set_title("Objective", loc="left", fontsize=10)
        axes[1].set_ylabel("iterations");     axes[1].set_title("Iterations", loc="left", fontsize=10)
        axes[2].set_ylabel("solve_time (s)"); axes[2].set_title("Solve time", loc="left", fontsize=10)
        axes[3].set_ylabel("residual");       axes[3].set_title("Final KKT residuals", loc="left", fontsize=10)
        axes[3].axhspan(1e-9, 1e-7, color=_C["grey"], alpha=0.10)  # tolerance band

        for ax in axes:
            _apply_modern_style(ax)
        axes[-1].set_xticks(x_master)
        axes[-1].set_xticklabels(self._ts_labels_for_xticks(), rotation=45, ha="right", fontsize=8)
        axes[-1].set_xlabel("timestep")

        if hue or len(runs) > 1:
            axes[3].legend(fontsize=7, ncol=min(3, len(runs)*3), frameon=False, loc="upper right")
        else:
            axes[3].legend(fontsize=8, frameon=False, loc="upper right")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_mass_balance(self, hue=None):
        """SD-R-02: max/L2 of mass-balance residual per timestep (semilogy)."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(13, 5))
        fig.suptitle(self._title("Mass-balance leakage — SD-R-02"),
                     fontsize=13, fontweight="bold")

        runs = hue if hue else [(self.label or "run", self, _C["navy"])]
        for lbl, dg, color in runs:
            x = dg._ts_index()
            mx = np.array([float(np.max(np.abs(dg._mb_residuals(t)))) if dg._mb_residuals(t).size else np.nan
                           for t in dg._timesteps])
            l2 = np.array([float(np.linalg.norm(dg._mb_residuals(t)))
                           for t in dg._timesteps])
            ax.semilogy(x, np.maximum(mx, 1e-16),
                        color=color, lw=1.6, marker="o", ms=3, ls="-",  label=f"{lbl}: max|·|")
            ax.semilogy(x, np.maximum(l2, 1e-16),
                        color=color, lw=1.4, marker="s", ms=3, ls="--", label=f"{lbl}: ‖·‖₂")

        ax.axhspan(1e-9, 1e-7, color=_C["grey"], alpha=0.10, label="solver tol band")
        ax.set_xticks(self._ts_index())
        ax.set_xticklabels(self._ts_labels_for_xticks(), rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("timestep")
        ax.set_ylabel("|G·a − c| on hard balance rows")
        _apply_modern_style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper right",
                  ncol=max(1, len(runs)))
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        return fig

    def _fig_soft_slack(self, hue=None):
        """SD-R-03: ε norms + fraction of soft rows with |ε| > 3·σ_c."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        fig.suptitle(self._title("Soft-slack stress — SD-R-03"),
                     fontsize=13, fontweight="bold")

        runs = hue if hue else [(self.label or "run", self, _C["blue"])]

        for lbl, dg, color in runs:
            x = dg._ts_index()
            # Panel A: ||eps||_inf, ||eps||_2, median |eps|
            eps_inf = np.array([float(np.max(np.abs(dg._results[t].eps_opt))) if dg._results[t].eps_opt.size else np.nan
                                for t in dg._timesteps])
            eps_l2  = np.array([float(np.linalg.norm(dg._results[t].eps_opt))
                                for t in dg._timesteps])
            eps_med = np.array([float(np.median(np.abs(dg._results[t].eps_opt))) if dg._results[t].eps_opt.size else np.nan
                                for t in dg._timesteps])
            axes[0].semilogy(x, np.maximum(eps_inf, 1e-16), color=color, lw=1.6, ls="-",  marker="o", ms=3, label=f"{lbl}: ‖ε‖∞")
            axes[0].semilogy(x, np.maximum(eps_l2,  1e-16), color=color, lw=1.4, ls="--", marker="s", ms=3, label=f"{lbl}: ‖ε‖₂")
            axes[0].semilogy(x, np.maximum(eps_med, 1e-16), color=color, lw=1.2, ls=":",  marker="^", ms=3, label=f"{lbl}: median|ε|")

            # Panel B: fraction of soft rows with |eps| > 3 sigma_c
            if hue is None:
                # No hue → split by constraint type
                for tkey, tcol_key in [("sum_eq", "blue"), ("ratio_eq", "purple")]:
                    frac = np.array([dg._frac_above_3sigma(t, tkey) for t in dg._timesteps])
                    axes[1].plot(x, frac, color=_C[tcol_key], lw=1.6, marker="o", ms=3,
                                 label=tkey)
            else:
                # Hued → aggregate over both soft types
                frac = np.array([
                    dg._frac_above_3sigma(t, None)  # combined
                    for t in dg._timesteps
                ])
                axes[1].plot(x, frac, color=color, lw=1.6, marker="o", ms=3, label=lbl)

        axes[0].set_ylabel("|ε|")
        axes[0].set_title("ε norms over time", loc="left", fontsize=10)
        axes[1].set_ylabel("fraction |ε| > 3·σ_c")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].set_title("Fraction of soft rows breaching 3σ_c", loc="left", fontsize=10)

        for ax in axes:
            _apply_modern_style(ax)
        axes[1].set_xticks(self._ts_index())
        axes[1].set_xticklabels(self._ts_labels_for_xticks(), rotation=45, ha="right", fontsize=8)
        axes[1].set_xlabel("timestep")
        axes[0].legend(fontsize=7, frameon=False, loc="upper right",
                       ncol=max(1, len(runs)))
        axes[1].legend(fontsize=8, frameon=False, loc="upper right")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def _frac_above_3sigma(self, t: str, type_key: Optional[str]) -> float:
        """Helper: fraction of soft rows with |ε| > 3 σ_c (one or all soft types)."""
        if type_key is None:
            eps   = np.concatenate([self._eps_for_type(t, "sum_eq"), self._eps_for_type(t, "ratio_eq")])
            sigma = np.concatenate([self._sigma_c_for_type(t, "sum_eq"), self._sigma_c_for_type(t, "ratio_eq")])
        else:
            eps   = self._eps_for_type(t, type_key)
            sigma = self._sigma_c_for_type(t, type_key)
        if eps.size == 0:
            return float("nan")
        # Avoid div-by-zero
        sig = np.where(sigma > 0, sigma, np.inf)
        return float(np.mean(np.abs(eps) > 3.0 * sig))

    def _fig_hard_active(self, hue=None):
        """SD-R-04: active Type-4 ratio bounds and Type-5 flow bounds."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        fig.suptitle(self._title("Hard-inequality activity — SD-R-04"),
                     fontsize=13, fontweight="bold")

        runs = hue if hue else [(self.label or "run", self, _C["orange"])]
        tol = 1e-7  # solver eps_abs by default

        for lbl, dg, color in runs:
            x = dg._ts_index()

            # Panel A: count of active Type-4 ratio-bound rows
            rb_idx = dg._row_info["ratio_bound"]
            if rb_idx.size:
                active_rb = np.array([
                    int(np.sum(np.abs(-dg._results[t].residuals[rb_idx]) <= tol))
                    for t in dg._timesteps
                ])
            else:
                active_rb = np.zeros(len(dg._timesteps), dtype=int)
            axes[0].plot(x, active_rb, color=color, lw=1.8, marker="o", ms=4, label=lbl)

            # Panel B: count of edges at lower / upper flow bound
            at_lower = np.zeros(len(dg._timesteps), dtype=int)
            at_upper = np.zeros(len(dg._timesteps), dtype=int)
            for i, t in enumerate(dg._timesteps):
                s_u, s_l = dg._flow_bound_slack(t)
                at_upper[i] = int(np.sum(np.abs(s_u) <= tol))
                at_lower[i] = int(np.sum(np.abs(s_l) <= tol))
            if hue is None:
                axes[1].plot(x, at_upper, color=_C["red"],   lw=1.8, marker="^", ms=4, label="at upper")
                axes[1].plot(x, at_lower, color=_C["green"], lw=1.8, marker="v", ms=4, label="at lower")
            else:
                axes[1].plot(x, at_upper + at_lower, color=color, lw=1.8, marker="o", ms=4, label=f"{lbl}: total")

        axes[0].set_ylabel("active count")
        axes[0].set_title("Type-4 ratio-bound rows (s ≈ 0)", loc="left", fontsize=10)
        axes[1].set_ylabel("active count")
        axes[1].set_title("Type-5 flow bounds (|F − u| ≈ 0 or |F − l| ≈ 0)", loc="left", fontsize=10)

        for ax in axes:
            _apply_modern_style(ax)
        axes[1].set_xticks(self._ts_index())
        axes[1].set_xticklabels(self._ts_labels_for_xticks(), rotation=45, ha="right", fontsize=8)
        axes[1].set_xlabel("timestep")
        axes[0].legend(fontsize=8, frameon=False, loc="upper right")
        axes[1].legend(fontsize=8, frameon=False, loc="upper right")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def _fig_prior_posterior(self, hue=None):
        """SD-R-05: ||a* − a0(t)|| in three norms over time."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(13, 6))
        fig.suptitle(self._title("Prior–posterior distance — SD-R-05"),
                     fontsize=13, fontweight="bold")

        runs = hue if hue else [(self.label or "run", self, _C["purple"])]

        for lbl, dg, color in runs:
            x = dg._ts_index()
            a0_t = np.asarray(dg._const_run.a0, dtype=float)
            sa   = np.asarray(dg._const_run.sigma_a, dtype=float)
            sa_safe = np.where(sa > 0, sa, np.inf)

            l_inf, l_2, l_w = [], [], []
            for t in dg._timesteps:
                F = dg._results[t].F_opt
                d = F - a0_t
                l_inf.append(float(np.max(np.abs(d))))
                l_2.append(float(np.linalg.norm(d)))
                l_w.append(float(np.sum((d / sa_safe) ** 2)))
                a0_t = F  # warm-start: next prior = this F
            ax.semilogy(x, np.maximum(l_inf, 1e-16), color=color, lw=1.5, ls="-",  marker="o", ms=3, label=f"{lbl}: L∞")
            ax.semilogy(x, np.maximum(l_2,   1e-16), color=color, lw=1.4, ls="--", marker="s", ms=3, label=f"{lbl}: L₂")
            ax.semilogy(x, np.maximum(l_w,   1e-16), color=color, lw=1.3, ls=":",  marker="^", ms=3, label=f"{lbl}: weighted")

        ax.set_xticks(self._ts_index())
        ax.set_xticklabels(self._ts_labels_for_xticks(), rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("timestep")
        ax.set_ylabel("‖a* − a₀(t)‖")
        ax.set_title("Prior is warm-start: a₀(t>0) = a*(t-1)", loc="left", fontsize=9, color=_C["grey"])
        _apply_modern_style(ax)
        ax.legend(fontsize=7, frameon=False, loc="upper right",
                  ncol=max(1, len(runs)))
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return fig

    def _fig_warmstart_delta(self, hue=None):
        """SD-R-06: consecutive ||a*(t) − a*(t-1)|| over time."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(13, 6))
        fig.suptitle(self._title("Warm-start delta — SD-R-06"),
                     fontsize=13, fontweight="bold")

        runs = hue if hue else [(self.label or "run", self, _C["green"])]

        for lbl, dg, color in runs:
            ts = dg._timesteps
            if len(ts) < 2:
                ax.text(0.5, 0.5, "Need ≥ 2 timesteps for warm-start delta",
                        transform=ax.transAxes, ha="center", color=_C["grey"])
                continue
            l_inf, l_2, l_rel = [], [], []
            for i in range(1, len(ts)):
                F   = dg._results[ts[i]].F_opt
                Fm1 = dg._results[ts[i-1]].F_opt
                d   = F - Fm1
                l_inf.append(float(np.max(np.abs(d))))
                l_2.append(float(np.linalg.norm(d)))
                denom = max(float(np.linalg.norm(Fm1)), 1e-16)
                l_rel.append(float(np.linalg.norm(d) / denom))
            x = np.arange(1, len(ts))
            ax.semilogy(x, np.maximum(l_inf, 1e-16), color=color, lw=1.5, ls="-",  marker="o", ms=3, label=f"{lbl}: L∞")
            ax.semilogy(x, np.maximum(l_2,   1e-16), color=color, lw=1.4, ls="--", marker="s", ms=3, label=f"{lbl}: L₂")
            ax.semilogy(x, np.maximum(l_rel, 1e-16), color=color, lw=1.3, ls=":",  marker="^", ms=3, label=f"{lbl}: rel L₂")

        ax.set_xticks(self._ts_index())
        ax.set_xticklabels(self._ts_labels_for_xticks(), rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("timestep")
        ax.set_ylabel("‖a*(t) − a*(t−1)‖")
        _apply_modern_style(ax)
        ax.legend(fontsize=7, frameon=False, loc="upper right",
                  ncol=max(1, len(runs)))
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        return fig

    def _fig_f_evolution(self):
        """SD-R-07: heatmap of F*(t) for top-K most-variable edges + top-5 lines."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import SymLogNorm

        K = 200
        F = np.stack([self._results[t].F_opt for t in self._timesteps], axis=1)
        # Variability per edge: range across time on log-magnitude.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_abs = np.log10(np.abs(F) + 1e-20)
        var_per_edge = log_abs.max(axis=1) - log_abs.min(axis=1)
        order = np.argsort(-var_per_edge)
        top = order[:min(K, F.shape[0])]
        F_top = F[top]

        fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                                 gridspec_kw={"height_ratios": [3, 2]})
        fig.suptitle(self._title("F*(t) evolution — SD-R-07"),
                     fontsize=13, fontweight="bold")

        # Panel A: heatmap (edges × time)
        ax = axes[0]
        vmax = max(1e-12, float(np.nanmax(np.abs(F_top))) if F_top.size else 1.0)
        norm = SymLogNorm(linthresh=max(1e-9, vmax * 1e-9), vmin=-vmax, vmax=vmax, base=10)
        im = ax.imshow(F_top, aspect="auto", cmap="RdBu_r", norm=norm,
                       interpolation="nearest")
        ax.set_xticks(self._ts_index())
        ax.set_xticklabels(self._ts_labels_for_xticks(), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(f"top-{F_top.shape[0]} variable edges (idx-sorted by range)")
        ax.set_title("Per-edge time evolution (symlog colour)", loc="left", fontsize=10)
        cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04)
        cbar.set_label("F*")

        # Panel B: top-5 most-variable edges as labelled lines
        ax = axes[1]
        cmap = plt.get_cmap("tab10")
        for k in range(min(5, F_top.shape[0])):
            ax.plot(self._ts_index(), F[top[k]], lw=1.8, marker="o", ms=3,
                    color=cmap(k % 10), label=f"edge {int(top[k])}")
        ax.set_xticks(self._ts_index())
        ax.set_xticklabels(self._ts_labels_for_xticks(), rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("timestep")
        ax.set_ylabel("F*")
        ax.set_title("Top-5 most-variable edges", loc="left", fontsize=10)
        _apply_modern_style(ax)
        ax.legend(fontsize=7, frameon=False, loc="best")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # ==================================================================
    # Figures — Scope 2 (per-timestep cross-section)
    # ==================================================================

    def _make_xsection(self, n_rows: int, title: str):
        """Helper: build a (n_rows × 3) cross-section figure skeleton."""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(n_rows, 3,
                                 figsize=(13, max(3.0, 2.6 * n_rows)),
                                 squeeze=False)
        fig.suptitle(self._title(title), fontsize=13, fontweight="bold")
        return fig, axes

    @staticmethod
    def _hist_with_footer(ax, data: np.ndarray, color: str, *, symlog: bool = False,
                          bins: int = 40, ref_lines: tuple = ()) -> None:
        """Shared histogram-cell renderer with summary footer in the axes."""
        finite = data[np.isfinite(data)]
        if finite.size:
            if symlog:
                # Stable symlog binning: edges in [min, max] linspace, switch x-scale later
                edges = np.linspace(finite.min(), finite.max(), bins + 1)
            else:
                edges = bins
            ax.hist(finite, bins=edges, color=color, alpha=0.55,
                    edgecolor=color, linewidth=0.6)
        for r in ref_lines:
            ax.axvline(r, color=_C["grey"], lw=0.8, ls="--", alpha=0.7)
        if symlog:
            ax.set_xscale("symlog", linthresh=max(1e-12, float(np.nanmax(np.abs(finite))) * 1e-6 if finite.size else 1e-9))
        _apply_modern_style(ax)

        # Footer: n / median / p95 / max
        if finite.size:
            footer = (f"n={finite.size}  med={np.median(finite):.2e}  "
                      f"p95={np.percentile(np.abs(finite), 95):.2e}  "
                      f"max={np.max(np.abs(finite)):.2e}")
        else:
            footer = "n=0"
        ax.text(0.99, 0.97, footer, transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=_C["grey"])

    def _fig_residual_digest(self, timesteps):
        """SD-T-01: 5 rows × 3 cols of constraint-residual histograms."""
        ts_triple = self._resolve_timesteps(timesteps)
        fig, axes = self._make_xsection(5, "Constraint-residual digest — SD-T-01")

        row_specs = [
            ("Type-1 mass-balance Gx",  "mass_balance", _TYPE_COLORS["mass_balance"]),
            ("Type-2 soft-sum ε",       "sum_eq",       _TYPE_COLORS["sum_eq"]),
            ("Type-3 soft-ratio ε",     "ratio_eq",     _TYPE_COLORS["ratio_eq"]),
            ("Type-4 ratio-bound slack", "ratio_bound", _TYPE_COLORS["ratio_bound"]),
            ("Type-5 flow-bound slack",  None,          _TYPE_COLORS["flow_bound"]),
        ]

        for col, t in enumerate(ts_triple):
            for row, (rlabel, key, color) in enumerate(row_specs):
                ax = axes[row, col]
                if key == "mass_balance":
                    data = self._mb_residuals(t)
                elif key in ("sum_eq", "ratio_eq"):
                    data = self._eps_for_type(t, key)
                elif key == "ratio_bound":
                    idx = self._row_info["ratio_bound"]
                    data = -self._results[t].residuals[idx] if idx.size else np.zeros(0)
                else:
                    s_u, s_l = self._flow_bound_slack(t)
                    data = np.concatenate([s_u, s_l])
                self._hist_with_footer(ax, data, color, symlog=True)
                if col == 0:
                    ax.set_ylabel(rlabel, fontsize=9)
                if row == 0:
                    ax.set_title(self._col_title(t, ts_triple), fontsize=10, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_zscore_soft(self, timesteps):
        """SD-T-02: 2 rows × 3 cols of |ε/σ_c| z-scores for soft constraints."""
        ts_triple = self._resolve_timesteps(timesteps)
        fig, axes = self._make_xsection(2, "Soft-slack z-scores — SD-T-02")

        for col, t in enumerate(ts_triple):
            for row, (rlabel, key) in enumerate([
                ("Type-2 ε/σ_c (sum_eq)",   "sum_eq"),
                ("Type-3 ε/σ_c (ratio_eq)", "ratio_eq"),
            ]):
                ax = axes[row, col]
                eps = self._eps_for_type(t, key)
                sig = self._sigma_c_for_type(t, key)
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = np.where(sig > 0, eps / sig, np.nan)
                self._hist_with_footer(
                    ax, z[np.isfinite(z)], _TYPE_COLORS[key],
                    bins=40, ref_lines=(-3, -1, 1, 3),
                )
                if col == 0:
                    ax.set_ylabel(rlabel, fontsize=9)
                if row == 0:
                    ax.set_title(self._col_title(t, ts_triple), fontsize=10, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_solution_vs_prior(self, timesteps):
        """SD-T-03: (a*-a0)/sigma_a, |a*-a0|/|a0|, sigma_opt/sigma_a (cond.)."""
        ts_triple = self._resolve_timesteps(timesteps)
        n_rows = 3 if self._sigma_opt else 2
        fig, axes = self._make_xsection(n_rows, "Solution vs prior — SD-T-03")

        a0  = np.asarray(self._const_run.a0, dtype=float)
        sa  = np.asarray(self._const_run.sigma_a, dtype=float)
        sa_safe = np.where(sa > 0, sa, np.inf)
        a0_safe = np.where(np.abs(a0) > 0, np.abs(a0), np.inf)

        for col, t in enumerate(ts_triple):
            F = self._results[t].F_opt
            # Row 1: z-score
            z = (F - a0) / sa_safe
            self._hist_with_footer(axes[0, col], z, _C["blue"],
                                   bins=40, ref_lines=(-3, -1, 1, 3))
            # Row 2: relative change
            rel = (F - a0) / a0_safe
            self._hist_with_footer(axes[1, col], rel, _C["orange"],
                                   bins=40, symlog=True)
            # Row 3: sigma_opt / sigma_a (optional)
            if self._sigma_opt is not None and t in self._sigma_opt:
                so = np.asarray(self._sigma_opt[t], dtype=float)
                ratio = np.where(sa > 0, so / sa_safe, np.nan)
                self._hist_with_footer(axes[2, col], ratio, _C["green"],
                                       bins=40, ref_lines=(1.0,))

            if col == 0:
                axes[0, col].set_ylabel("(a* − a₀)/σ_a", fontsize=9)
                axes[1, col].set_ylabel("(a* − a₀)/|a₀|", fontsize=9)
                if n_rows == 3:
                    axes[2, col].set_ylabel("σ_opt/σ_a", fontsize=9)
            axes[0, col].set_title(self._col_title(t, ts_triple),
                                   fontsize=10, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_binding_rate(self, timesteps):
        """SD-T-04: 2 rows × 3 cols of binding-rate distributions."""
        ts_triple = self._resolve_timesteps(timesteps)
        fig, axes = self._make_xsection(2, "Per-type binding rate — SD-T-04")

        for col, t in enumerate(ts_triple):
            # Row 1: Type-4 ratio-bound expressions (Gx - c sign convention:
            # we want expression value where negative ⇒ slack)
            idx = self._row_info["ratio_bound"]
            v_rb = self._results[t].residuals[idx] if idx.size else np.zeros(0)
            self._hist_with_footer(axes[0, col], v_rb, _TYPE_COLORS["ratio_bound"],
                                   bins=40, symlog=True, ref_lines=(0.0,))
            axes[0, col].axvspan(min(0, axes[0, col].get_xlim()[0]) - 1, 0,
                                 color=_C["grey"], alpha=0.08)

            # Row 2: Type-5 flow-bound expressions (F - u and l - F).
            # Negative = slack (good); positive = violation.
            s_u, s_l = self._flow_bound_slack(t)
            v_fb = np.concatenate([-s_u, -s_l])  # = F - u and -(F - l) flipped — see plan
            self._hist_with_footer(axes[1, col], v_fb, _TYPE_COLORS["flow_bound"],
                                   bins=40, symlog=True, ref_lines=(0.0,))
            axes[1, col].axvspan(min(0, axes[1, col].get_xlim()[0]) - 1, 0,
                                 color=_C["grey"], alpha=0.08)

            if col == 0:
                axes[0, col].set_ylabel("Type-4: a_i − r·a_j", fontsize=9)
                axes[1, col].set_ylabel("Type-5: F − u, l − F", fontsize=9)
            axes[0, col].set_title(self._col_title(t, ts_triple),
                                   fontsize=10, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_solution_magnitude(self, timesteps):
        """SD-T-05: log10|a*| histogram per column with p50/p95/p99 markers."""
        ts_triple = self._resolve_timesteps(timesteps)
        fig, axes = self._make_xsection(1, "Solution magnitude |a*| — SD-T-05")

        # Shared bin edges across the three columns for visual comparability
        F_all = np.concatenate([self._results[t].F_opt for t in ts_triple])
        F_pos = np.abs(F_all)
        F_pos = F_pos[F_pos > 0]
        if F_pos.size:
            lo, hi = np.log10(F_pos.min()), np.log10(F_pos.max())
        else:
            lo, hi = -1.0, 1.0
        bin_edges = np.linspace(lo, hi, 41)

        for col, t in enumerate(ts_triple):
            ax = axes[0, col]
            F = np.abs(self._results[t].F_opt)
            F = F[F > 0]
            if F.size:
                ax.hist(np.log10(F), bins=bin_edges, color=_C["blue"],
                        alpha=0.55, edgecolor=_C["blue"], linewidth=0.6)
                for q, lbl, col_q in [(50, "p50", _C["navy"]),
                                       (95, "p95", _C["orange"]),
                                       (99, "p99", _C["red"])]:
                    val = float(np.percentile(np.log10(F), q))
                    ax.axvline(val, color=col_q, lw=1.2, ls="--", alpha=0.8)
                    ax.text(val, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1.0,
                            lbl, rotation=90, fontsize=7, va="top", ha="right", color=col_q)
            _apply_modern_style(ax)
            ax.set_xlabel("log₁₀ |a*|")
            if col == 0:
                ax.set_ylabel("count", fontsize=9)
            ax.set_title(self._col_title(t, ts_triple), fontsize=10, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_cstar_vs_c(self, timesteps):
        """SD-T-06: c*-c and (c*-c)/(|c|+δ) histograms on soft rows."""
        ts_triple = self._resolve_timesteps(timesteps)
        fig, axes = self._make_xsection(2, "c* vs c on soft rows — SD-T-06")

        # Soft rows = sum_eq ∪ ratio_eq
        soft_idx = np.concatenate([self._row_info["sum_eq"], self._row_info["ratio_eq"]])

        for col, t in enumerate(ts_triple):
            if soft_idx.size:
                c       = np.asarray(self._const_run.timestep_data[t]["c"])[soft_idx]
                cstar   = self._results[t].c_star[soft_idx]
                eps     = cstar - c
                rel_eps = eps / (np.abs(c) + 1e-12)
            else:
                eps     = np.zeros(0)
                rel_eps = np.zeros(0)

            self._hist_with_footer(axes[0, col], eps, _C["blue"],
                                   bins=40, symlog=True, ref_lines=(0.0,))
            self._hist_with_footer(axes[1, col], rel_eps, _C["purple"],
                                   bins=40, ref_lines=(-1, 0, 1))

            if col == 0:
                axes[0, col].set_ylabel("c* − c", fontsize=9)
                axes[1, col].set_ylabel("(c* − c) / (|c| + δ)", fontsize=9)
            axes[0, col].set_title(self._col_title(t, ts_triple),
                                   fontsize=10, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    # ------------------------------------------------------------------
    # Title helpers
    # ------------------------------------------------------------------

    def _title(self, base: str) -> str:
        return f"{base}  —  {self.label}" if self.label else base

    @staticmethod
    def _col_title(t: str, triple: tuple) -> str:
        which = {triple[0]: "t_min", triple[1]: "t_worst", triple[2]: "t_max"}.get(t, "t")
        return f"{which}={t}"
