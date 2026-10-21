"""
sweep.py
========
``SweepDiag`` — the figure portfolio of a sweep run (#107).

Every number is read from :class:`pysuto.diag.stats.SweepStatsDiagnostic`: each done
point scored against the base chain by posterior distance (mass moved, retention by
category) and by the cross-evaluated objective ``J_base_at_point`` (the point's answer
under the base QP, GLOSSARY.md "Sweep"), and the axes ranked per outcome.

Plot catalogue — :attr:`SweepDiag.PLOTS` (ids are stable)
---------------------------------------------------------
* **SW-R-01** ``sw_r_01_sensitivity`` — one bar per axis for the chosen outcome: the
  SRRC in a random sweep (numeric axes), the main-effect range in a grid.
* **SW-R-02** ``sw_r_02_outcome_by_knob`` — the outcome against each axis's value, one
  panel per axis (a list axis by level).
* **SW-R-03** ``sw_r_03_cross_objective`` — ``J_base_at_point`` per point, stacked by
  family (edge defects, sum-row, soft-ratio and other soft residuals).
* **SW-R-04** ``sw_r_04_retention`` — retention per point for the twelve categories
  that moved most, with the base's retention marked.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from ._plot import _CAT8, _INK, _savefig
from .solver import PlotSpec, _apply_modern_style, _short
from .stats.sweep import FAMILIES, SweepStatsDiagnostic, axis_label

_N_CATEGORIES = 12
# A point breaks the base's hard rows when it misses one by more than this (kt). The base
# itself sits at its equality floor, ≈ 1.6e-6 kt on opt_20261020_112726_2f451f31 (#33).
HARD_BREACH_TOL_KT = 1e-3
_FAMILY_LABELS = {"edges":       "edge defects",
                  "sum_rows":    "sum-row residuals",
                  "soft_ratios": "soft-ratio residuals",
                  "other_soft":  "other soft residuals"}


class SweepDiag:
    """Figures and a text summary of one sweep run.

    Parameters
    ----------
    stats : the sweep's :class:`SweepStatsDiagnostic`.
    outcome : the outcome SW-R-01 and SW-R-02 rank and plot; default
        ``J_base_at_point``.
    """

    PLOTS: "dict[str, PlotSpec]" = {
        "SW-R-01": PlotSpec("_fig_sensitivity", "sw_r_01_sensitivity",
                            "sensitivity of the outcome to each axis"),
        "SW-R-02": PlotSpec("_fig_outcome_by_knob", "sw_r_02_outcome_by_knob",
                            "the outcome against each axis's value"),
        "SW-R-03": PlotSpec("_fig_cross_objective", "sw_r_03_cross_objective",
                            "J_base_at_point per point by family"),
        "SW-R-04": PlotSpec("_fig_retention", "sw_r_04_retention",
                            "retention of the most-moved categories per point"),
    }

    def __init__(self, stats: SweepStatsDiagnostic,
                 outcome: str = "J_base_at_point") -> None:
        self.stats = stats
        self.outcome = outcome

    # ------------------------------------------------------------------
    def _done(self) -> pd.DataFrame:
        df = self.stats.points_frame()
        return df[df["status"] == "done"].reset_index(drop=True)

    def _stat(self) -> tuple:
        """The statistic that ranks the axes: (column value, label)."""
        if self.stats.mode == "random":
            return "srrc", "SRRC"
        return "main_effect_range", "main-effect range"

    def _ranking(self, outcome: Optional[str] = None) -> pd.DataFrame:
        stat = self._stat()[0]
        t = self.stats.sensitivity()
        t = t[(t["outcome"] == (outcome or self.outcome)) & (t["statistic"] == stat)]
        t = t.dropna(subset=["value"])
        return t.reindex(t["value"].abs().sort_values(ascending=False).index)

    # ------------------------------------------------------------------
    def plot(self, odir: str, show: bool = False) -> list:
        """Write the four figures and the stats tables to ``odir``; return the PNGs."""
        import matplotlib.pyplot as plt
        self.stats.write(odir)
        paths = []
        for spec in self.PLOTS.values():
            fig = getattr(self, spec.method)(plt)
            paths.append(_savefig(fig, os.path.join(odir, spec.filename + ".png")))
            if show:
                plt.show()
            plt.close(fig)
        return paths

    def _fig_sensitivity(self, plt):
        rank = self._ranking()
        stat = self._stat()[1]
        fig, ax = plt.subplots(figsize=(7, 0.5 * max(len(rank), 2) + 1.5))
        if len(rank):
            ax.barh(np.arange(len(rank)), rank["value"], color=_CAT8[0], height=0.6)
            ax.set_yticks(np.arange(len(rank)), [_short(a) for a in rank["axis"]])
            ax.invert_yaxis()
            ax.axvline(0, color=_INK["muted"], lw=1)
        else:
            ax.text(0.5, 0.5, "no axis has a defined statistic", ha="center",
                    transform=ax.transAxes, color=_INK["muted"])
        ax.set_xlabel(f"{stat} of {self.outcome}")
        ax.set_title(f"SW-R-01  sensitivity of {self.outcome} to each axis")
        _apply_modern_style(ax)
        return fig

    def _fig_outcome_by_knob(self, plt):
        done, axes = self._done(), self.stats.axes
        fig, axs = plt.subplots(1, len(axes), figsize=(4 * len(axes), 3.5), squeeze=False)
        for ax, decl in zip(axs[0], axes):
            col = done[decl["keys"][0]] if len(done) else pd.Series(dtype=float)
            y = done[self.outcome] if len(done) else pd.Series(dtype=float)
            if "values" in decl:
                levels = [repr(v) for v in decl["values"]]
                x = [levels.index(repr(v)) for v in col]
                ax.set_xticks(range(len(levels)), [_short(lv, 18) for lv in levels],
                              rotation=30, ha="right")
            else:
                x = col.to_numpy(dtype=float)
                if decl.get("distribution") in ("loguniform", "reciprocal"):
                    ax.set_xscale("log")
            ax.scatter(x, y, s=36, color=_CAT8[0], edgecolor="white", linewidth=1)
            ax.set_xlabel(_short(axis_label(decl), 40))
            ax.set_ylabel(self.outcome)
            _apply_modern_style(ax)
        fig.suptitle(f"SW-R-02  {self.outcome} against each axis")
        fig.tight_layout()
        return fig

    def _fig_cross_objective(self, plt):
        done = self._done()
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(done) + 2), 4))
        bottom_pos = np.zeros(len(done))
        bottom_neg = np.zeros(len(done))
        for k, f in enumerate(FAMILIES):
            v = (done[f"J_base_at_point_{f}"].to_numpy(dtype=float) if len(done)
                 else np.zeros(0))
            base = np.where(v >= 0, bottom_pos, bottom_neg)
            ax.bar(done["index"], v, bottom=base, color=_CAT8[k], width=0.7,
                   edgecolor="white", linewidth=1, label=_FAMILY_LABELS[f])
            bottom_pos += np.where(v >= 0, v, 0.0)
            bottom_neg += np.where(v < 0, v, 0.0)
        ax.axhline(0, color=_INK["muted"], lw=1)
        ax.set_xticks(done["index"], [f"p{i:03d}" for i in done["index"]],
                      rotation=90 if len(done) > 12 else 0)
        ax.set_xlabel("point")
        ax.set_ylabel("J_base(a*_point) − J_base(a*_base)")
        ax.set_title("SW-R-03  each point's answer scored under the base QP")
        ax.legend(frameon=False, fontsize=8)
        _apply_modern_style(ax)
        return fig

    def _fig_retention(self, plt):
        cat = self.stats.category_frame()
        top = (cat.groupby("category")["mass_moved"].sum()
               .sort_values(ascending=False).head(_N_CATEGORIES).index.tolist())
        fig, ax = plt.subplots(figsize=(7, 0.45 * max(len(top), 2) + 1.5))
        for i, c in enumerate(top):
            rows = cat[cat["category"] == c]
            ax.scatter(rows["retention"], np.full(len(rows), i), s=30, color=_CAT8[0],
                       edgecolor="white", linewidth=1, label="point" if i == 0 else None)
            ax.scatter([rows["retention_base"].iloc[0]], [i], marker="|", s=300,
                       color=_INK["primary"], label="base" if i == 0 else None)
        ax.set_yticks(range(len(top)), [_short(c, 40) for c in top])
        ax.invert_yaxis()
        ax.set_xlabel("retention Σa*/Σa_prior (all years)")
        ax.set_title(f"SW-R-04  retention of the {len(top)} most-moved categories")
        if top:
            ax.legend(frameon=False, fontsize=8, loc="upper left",
                      bbox_to_anchor=(1.01, 1.0))
        _apply_modern_style(ax)
        return fig

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """The axis ranking for the outcome, the failed points and the caveats."""
        df = self.stats.points_frame()
        lines = [f"sweep {self.stats.sweep_run_id}: {self.stats.mode}, {len(df)} point(s), "
                 f"base {self.stats.base_opt}",
                 "  status: " + ", ".join(f"{s} {n}" for s, n in
                                          df["status"].value_counts().items())]
        rank = self._ranking()
        stat = self._stat()[1]
        lines.append(f"  axes by |{stat}| of {self.outcome}:")
        lines += [f"    {r.axis:<40s} {r.value:+.4g}  (n = {r.n})"
                  for r in rank.itertuples()]
        lines.append(f"  top axis per outcome (|{stat}|):")
        for out in self.stats.outcomes():
            top = self._ranking(out)
            if len(top):
                lines.append(f"    {out:<34s} {top['axis'].iloc[0]} "
                             f"({top['value'].iloc[0]:+.3g})")
        unranked = {axis_label(a) for a in self.stats.axes} - set(rank["axis"])
        if unranked:
            lines.append(f"  without a {stat}: {', '.join(sorted(unranked))} "
                         f"(see sensitivity.csv)")
        if "max_hard_violation_base" in df.columns:
            lines.append(f"  base's own hard-row floor: "
                         f"{self.stats.base_max_hard_violation:.2g} kt")
            bad = df[df["max_hard_violation_base"] > HARD_BREACH_TOL_KT]
            if len(bad):
                lines.append(f"  points breaking the base's hard rows "
                             f"(> {HARD_BREACH_TOL_KT:g} kt): {list(bad['index'])} — their "
                             f"J_base_at_point is not a comparison at a feasible point")
        failed = df[df["status"] == "failed"]["index"].tolist()
        if failed:
            lines.append(f"  failed points: {failed} (their logs are in the manifest)")
        lines.append("  J_own compares only points whose σ-defining knobs equal the "
                     "base's.")
        text = "\n".join(lines)
        print(text)
        return text
