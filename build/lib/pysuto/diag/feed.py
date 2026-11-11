"""
feed.py
=======
``FeedDiagnostics`` — Stage 1 diagnostic suite.

Reads feed reading, concordance coverage, F_src distribution,
CV propagation.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from ._plot import _C, _savefig


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

        path = os.path.join(odir, "feed.png")
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

