"""
imp.py
======
``ImpDiagnostic`` — Stage 2 diagnostic suite.

Per-step F_root/σ distributions (log scale), coverage and fill-count
evolution, TimeImputer timeseries.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._plot import _C, _savefig


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

    _TIME_METHODS = {"time_linear", "time_mean"}

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

        paths = []

        # Figure 1 — per-step trajectories (2×2)
        fig1 = self._fig1_steps()
        if odir is not None:
            p = _savefig(fig1, os.path.join(odir, "imputation_steps.png"), dpi=dpi)
            paths.append(p)
            print(f"   Imputation step diagnostics → {p}  ✓")
        if show:
            plt.show()
        plt.close(fig1)

        # Figure 2 — σ-by-method boxplot (1×1)
        fig2 = self._fig2_sigma_box()
        if fig2 is not None:
            if odir is not None:
                p = _savefig(fig2, os.path.join(odir, "imputation_sigma_box.png"), dpi=dpi)
                paths.append(p)
                print(f"   Imputation σ boxplot       → {p}  ✓")
            if show:
                plt.show()
            plt.close(fig2)

        # Figure 2b — CV-by-method boxplot (1×1, parallel to σ box)
        fig_cv = self._fig_cv_box()
        if fig_cv is not None:
            if odir is not None:
                p = _savefig(fig_cv, os.path.join(odir, "imputation_cv_box.png"), dpi=dpi)
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
                    p = _savefig(fig3, os.path.join(odir, "imputation_timeseries.png"), dpi=dpi)
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
        # Under the structural model a_agg is 2-D (n_edges, n_timesteps); the
        # coverage tallies are over the full cell grid.
        n_cells = int(agg.a_agg.size)
        n_final = int(np.isfinite(agg.a_agg).sum())
        n_imp   = len(log_df)
        n_miss  = n_cells - n_final

        lines = [
            f"{'─'*60}",
            f"  IMPUTATION DIAGNOSTICS{' — ' + self.label if self.label else ''}",
            f"{'─'*60}",
            f"  Cells (edges × t)      : {n_cells:>8,}",
            f"  Covered after pipeline : {n_final:>8,}  ({n_final/n_cells*100:.1f} %)",
            f"  Filled by imputers     : {n_imp:>8,}  ({n_imp/n_cells*100:.1f} %)",
            f"  Still missing          : {n_miss:>8,}  ({n_miss/n_cells*100:.1f} %)",
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
        """Reconstruct per-step (f, sigma) arrays from ImputationLog.

        Under the structural model ``agg_result.a_agg`` / ``sigma_agg`` are
        2-D ``(n_edges, n_timesteps)``. The log carries the cell coordinate
        ``(edge_id, t_idx)`` for each fill, so each snapshot tracks the
        full grid.
        """
        log_df = self._log_df
        if log_df.empty:
            f = self.agg_result.a_agg
            s = self.agg_result.sigma_agg
            return [("pre", f.copy(), s.copy())]

        edge_ids = log_df["edge_id"].values.astype(int)
        # Legacy log entries written before the t_idx column existed leave
        # NaN here; default those rows to period 0 so the snapshot loop
        # remains operative.
        t_idxs   = pd.to_numeric(log_df["t_idx"], errors="coerce").fillna(0).astype(int).values

        f0 = self.agg_result.a_agg.copy()
        s0 = self.agg_result.sigma_agg.copy()
        f0[edge_ids, t_idxs] = np.nan
        s0[edge_ids, t_idxs] = np.nan

        snapshots = [("pre", f0.copy(), s0.copy())]
        f_curr = f0.copy()
        s_curr = s0.copy()

        for step in sorted(log_df["step"].unique()):
            mask   = log_df["step"] == step
            e_ids  = log_df.loc[mask, "edge_id"].values.astype(int)
            ti     = (pd.to_numeric(log_df.loc[mask, "t_idx"], errors="coerce")
                        .fillna(0).astype(int).values)
            f_curr[e_ids, ti] = log_df.loc[mask, "f_after"].values
            s_curr[e_ids, ti] = log_df.loc[mask, "sigma_after"].values
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
        self._overlay_year_reference(ax, what="f", year="1995")
        ax.set_title("F_root distribution per step", fontweight="bold")
        ax.set_xlabel("log₁₀(f)  [log kt]")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, framealpha=0.6)
        ax.grid(axis="y", alpha=0.3)

    def _overlay_year_reference(self, ax, what: str, year: str = "1995") -> None:
        """Overlay a fully-observed year's raw distribution as a reference curve.

        Under the structural model ``acc.time_range`` is the canonical period
        axis. The reference year is selected by its index in ``time_range``;
        each snapshot is a 2-D ``(n_edges, n_timesteps)`` grid, so the column
        slice picks out one period across all edges.
        """
        if self.acc_struct is None:
            return
        time_range = [str(t) for t in self.acc_struct.time_range]
        if year not in time_range:
            return
        t_idx = time_range.index(year)
        _, f_pre, s_pre = self._snapshots[0]
        if what == "f":
            vals = np.abs(f_pre[:, t_idx])
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if not len(vals):
                return
            vals = np.log10(vals)
        elif what == "cv":
            with np.errstate(divide="ignore", invalid="ignore"):
                cv = np.where(
                    np.abs(f_pre[:, t_idx]) > 0,
                    s_pre[:, t_idx] / np.abs(f_pre[:, t_idx]),
                    np.nan,
                )
            cv = cv[np.isfinite(cv) & (cv > 0)]
            if not len(cv):
                return
            vals = np.log10(cv)
        else:
            return
        ax.hist(vals, bins=60, density=True, histtype="step",
                color="black", linestyle="--", linewidth=1.4,
                label=f"ref {year}")

    def _panel_coverage(self, ax) -> None:
        labels  = [s[0] for s in self._snapshots]
        counts  = [int(np.isfinite(s[1]).sum()) for s in self._snapshots]
        # Coverage is the count of finite cells over (n_edges × n_timesteps).
        n_total = int(self._snapshots[0][1].size)

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
        Per-step CV (= σ / |f|) histogram on a log10 axis. CV strips out
        the magnitude coupling and exposes whether a step actually inflates
        relative uncertainty. Reference lines at CV = 0.1, 1, 10, 100.
        """
        for lbl, fv, sv in self._snapshots:
            with np.errstate(divide="ignore", invalid="ignore"):
                cv = np.where(np.abs(fv) > 0, sv / np.abs(fv), np.nan)
            vals = cv[np.isfinite(cv) & (cv > 0)]
            if not len(vals):
                continue
            ax.hist(np.log10(vals), bins=60, alpha=0.45,
                    color=self._step_color[lbl], label=lbl,
                    density=True, histtype="stepfilled", edgecolor="none")
        for ref in (0.1, 1.0, 10.0, 100.0):
            ax.axvline(np.log10(ref), color=_C["red"], lw=0.7, ls="--", alpha=0.5)
        self._overlay_year_reference(ax, what="cv", year="1995")
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
        # σ_after p50/p99 readout per step — keeps a σ summary visible
        # even though the distribution panel above plots CV.
        sigma_stats: dict[int, tuple[float, float]] = {}
        for step in steps:
            sub = log_df.loc[log_df["step"] == step, "sigma_after"]
            vals = pd.to_numeric(sub, errors="coerce").to_numpy()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                sigma_stats[step] = (
                    float(np.percentile(vals, 50)),
                    float(np.percentile(vals, 99)),
                )

        for bar, cnt, step in zip(bars, counts, steps):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + counts.max() * 0.01,
                f"{cnt:,}", ha="center", va="bottom", fontsize=8,
            )
            stat = sigma_stats.get(step)
            if stat is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    f"σ p50={stat[0]:.2g}\np99={stat[1]:.2g}",
                    ha="center", va="center", fontsize=7,
                    color="white" if bar.get_height() > counts.max() * 0.15 else "black",
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
        """TimeImputer panel — under the structural model each ``(from, to,
        valuation)`` group is a *single* edge_id, and time lives on the
        second axis of ``a_agg``. Per-cell imputation is keyed on
        ``(edge_id, t_idx)``.
        """
        import matplotlib.pyplot as plt

        acc     = self.acc_struct
        log_df  = self._log_df
        time_df = log_df[log_df["method"].isin(self._TIME_METHODS)].copy()
        if time_df.empty:
            return None

        ei      = acc.edge_index.set_index("edge_id")
        time_df = time_df.join(ei[["from_id", "to_id", "valuation"]], on="edge_id")

        # Rank groups by mean |f_after|, then pick top-6 that have ≥1
        # bilateral-valid imputed point (avoids showing empty panels).
        group_keys = ["from_id", "to_id", "valuation"]
        grp_mean   = (
            time_df.groupby(group_keys)["f_after"]
            .apply(lambda x: np.abs(x).mean())
            .sort_values(ascending=False)
        )
        if grp_mean.empty:
            return None

        time_range = [str(t) for t in acc.time_range]
        try:
            time_num = [float(t) for t in time_range]
        except (ValueError, TypeError):
            time_num = list(range(len(time_range)))

        agg_f        = self.agg_result.a_agg
        # Cell-level imputed sets keyed by (edge_id, t_idx). Tolerate
        # legacy logs without t_idx by defaulting missing entries to 0.
        def _ti(df):
            return pd.to_numeric(df["t_idx"], errors="coerce").fillna(0).astype(int).values

        imputed_cells  = set(zip(log_df["edge_id"].astype(int).values,
                                  _ti(log_df)))
        time_imp_cells = set(zip(time_df["edge_id"].astype(int).values,
                                  _ti(time_df)))

        def _group_edge_id(key) -> int | None:
            from_id, to_id, valuation = key
            grp = ei[
                (ei["from_id"] == from_id) &
                (ei["to_id"]   == to_id)   &
                (ei["valuation"] == valuation)
            ]
            if grp.empty:
                return None
            return int(grp.index[0])

        def _collect(eid: int) -> tuple[list, list, list, list]:
            raw_t, raw_f, imp_t, imp_f = [], [], [], []
            raw_idx_set: set[int] = set()
            imp_candidates: list[tuple[int, float]] = []
            for ti in range(len(time_range)):
                f_val = agg_f[eid, ti]
                if not np.isfinite(f_val):
                    continue
                cell = (eid, ti)
                if cell in time_imp_cells:
                    imp_candidates.append((ti, float(f_val)))
                elif cell not in imputed_cells:
                    raw_idx_set.add(ti)
                    raw_t.append(time_num[ti])
                    raw_f.append(float(f_val))
            for ti, fv in imp_candidates:
                has_before = any((ti - d) in raw_idx_set for d in (1, 2, 3))
                has_after  = any((ti + d) in raw_idx_set for d in (1, 2, 3))
                if has_before and has_after:
                    imp_t.append(time_num[ti])
                    imp_f.append(fv)
            return raw_t, raw_f, imp_t, imp_f

        top_groups = []
        for key in grp_mean.index:
            eid = _group_edge_id(key)
            if eid is None:
                continue
            _, _, imp_t, _ = _collect(eid)
            if imp_t:
                top_groups.append((key, eid))
                if len(top_groups) == 6:
                    break

        if not top_groups:
            return None

        n     = len(top_groups)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
        suffix = f" — {self.label}" if self.label else ""
        fig.suptitle(f"TimeImputer timeseries{suffix}", fontsize=13, fontweight="bold")

        for panel_idx, (key, eid) in enumerate(top_groups):
            ax = axes[panel_idx // ncols][panel_idx % ncols]
            from_id, to_id, valuation = key
            raw_t, raw_f, imp_t, imp_f = _collect(eid)

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

