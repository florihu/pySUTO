"""
sweep.py
========
``SweepDiag`` — the figure portfolio of a sweep run (#107, #117).

Every number is read from :class:`pysuto.diag.stats.SweepStatsDiagnostic`: each done
point scored against the base chain, the axes ranked (SRRC, or Spearman's ρ where the
rank regression's R² is below 0.5) and sized (the swing) per outcome and per year, and
the verdicts of the declared criteria (GLOSSARY.md "Pooled retention", "Swing",
"Year profile", "Criterion"). The figures of one valuation (default ``CC``) cover its
categories and outcomes only.

Plot catalogue — :attr:`SweepDiag.PLOTS` (ids are stable; SW-R-03 is retired)
------------------------------------------------------------------------------
* **SW-R-01** ``sw_r_01_sensitivity_map`` — every axis × every retention outcome and
  the valuation's mass moved: colour = the rank statistic (in a grid the swing, scaled
  per column), text = the swing, influential cells boxed, R² under each column;
  criterion outcomes first, then chain order.
* **SW-R-02** ``sw_r_02_criterion_vs_top_axis`` — each criterion outcome against its
  top axis (log-scaled for a log-uniform axis), the criterion's limit as a band.
* **SW-R-03** retired (#117), never reused: ``J_base_at_point`` by family, whose
  clusters were the two cascade orders. The column stays in points.csv as a check.
* **SW-R-04** ``sw_r_04_retention_boxes`` — Δ pooled retention per category of the
  valuation across the points: box = IQR, whiskers = min–max, the base's retention in
  the label, chain order, criterion bands, the categories without retention named.
* **SW-R-05** ``sw_r_05_criterion_bands`` — every criterion outcome as a row of points
  against its limit band, coloured by the rank of the row's top axis, the count outside
  the band named.
* **SW-R-06** ``sw_r_06_year_envelope`` — per retention outcome of the valuation: the
  per-year Δ across the points (min–max and 10–90 % bands), the points whose pooled
  value breaks the outcome's limit as lines.
* **SW-R-07** ``sw_r_07_sensitivity_by_year`` — per retention outcome: the per-year
  statistic of every axis influential on at least one pooled retention outcome (every
  axis in a grid); filled = SRRC, hollow = ρ where that year's R² is below 0.5.
"""

from __future__ import annotations

import json
import os
import textwrap

import numpy as np
import pandas as pd

from ._plot import _CAT8, _CAT_EXT, _INK, _savefig, _seq_cmap
from .solver import PlotSpec, _apply_modern_style, _short
from .stats.sweep import (R2_MIN, SRRC_MIN, SweepStatsDiagnostic, axis_label, influential,
                          rank_statistic, top_axis)

# A point breaks the base's hard rows when it misses one by more than this (kt). The base
# itself sits at its equality floor, ≈ 1.6e-6 kt on opt_20261020_112726_2f451f31 (#33).
HARD_BREACH_TOL_KT = 1e-3
_NCOLS       = 6                                   # panels per row of SW-R-06 and 07
_LOG         = ("loguniform", "reciprocal")
_STAT_LABEL  = {"srrc": "SRRC", "spearman": "ρ", "swing": "swing"}


def _fmt(v) -> str:
    return f"{float(v):+.2g}" if np.isfinite(v) else ""


class SweepDiag:
    """Figures and a text summary of one sweep run.

    Parameters
    ----------
    stats : the sweep's :class:`SweepStatsDiagnostic`, with its outcomes and criteria.
    valuation : the valuation of SW-R-04, SW-R-06 and the map's mass-moved column (the
        case's, e.g. ``"CC"`` for copper).
    """

    PLOTS: "dict[str, PlotSpec]" = {
        "SW-R-01": PlotSpec("_fig_map", "sw_r_01_sensitivity_map",
                            "every axis against every ranked retention outcome"),
        "SW-R-02": PlotSpec("_fig_criterion_by_top_axis", "sw_r_02_criterion_vs_top_axis",
                            "each criterion outcome against its top axis"),
        "SW-R-04": PlotSpec("_fig_retention_boxes", "sw_r_04_retention_boxes",
                            "Δ pooled retention per category across the points"),
        "SW-R-05": PlotSpec("_fig_criterion_bands", "sw_r_05_criterion_bands",
                            "each criterion outcome's points against its limit"),
        "SW-R-06": PlotSpec("_fig_year_envelope", "sw_r_06_year_envelope",
                            "per-year Δ retention across the points"),
        "SW-R-07": PlotSpec("_fig_sensitivity_by_year", "sw_r_07_sensitivity_by_year",
                            "per-year ranking of the influential axes"),
    }

    def __init__(self, stats: SweepStatsDiagnostic, valuation: str) -> None:
        self.stats = stats
        self.valuation = valuation
        self._memo: dict = {}

    # ------------------------------------------------------------------
    def _done(self) -> pd.DataFrame:
        df = self.stats.points_frame()
        return df[df["status"] == "done"].reset_index(drop=True)

    def _sens(self) -> pd.DataFrame:
        if "sens" not in self._memo:
            self._memo["sens"] = self.stats.sensitivity()
        return self._memo["sens"]

    def _sens_by_year(self) -> pd.DataFrame:
        if "by_year" not in self._memo:
            self._memo["by_year"] = self.stats.sensitivity_by_year()
        return self._memo["by_year"]

    def _criterion_outcomes(self) -> list:
        """(item, outcome, limit) per outcome a criterion judges, a MaxAbs by its parts."""
        return [(c.item, q.name, c.limit) for c in self.stats.criteria for q in c.leaves()]

    def _retention_outcomes(self, valuation=None) -> pd.DataFrame:
        ot = self.stats.outcome_table()
        ot = ot[ot["measure"] == "retention"]
        return ot if valuation is None else ot[ot["valuation"] == valuation]

    def _left_out(self) -> str:
        """The valuation's categories without retention, with the reason."""
        ct = self.stats.category_table()
        ct = ct[(ct["valuation"] == self.valuation) & (ct["no_retention"] != "")]
        return ", ".join(f"{c} ({r})" for c, r in zip(ct["category"], ct["no_retention"]))

    @staticmethod
    def _axis_x(decl: dict, column: pd.Series) -> tuple:
        """Plot positions of an axis's values and, for a list axis, its level labels
        (values compared as JSON, as the statistics compare them)."""
        if "values" in decl:
            levels = [json.dumps(v) for v in decl["values"]]
            return np.array([levels.index(json.dumps(v)) for v in column],
                            dtype=float), levels
        return column.to_numpy(dtype=float), None

    @staticmethod
    def _grid(plt, n: int, ncols: int, w: float, h: float) -> tuple:
        ncols = min(ncols, n)
        nrows = -(-n // ncols)
        fig, axs = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
        flat = list(axs.flat)
        for ax in flat[n:]:
            ax.set_visible(False)
        return fig, flat[:n]

    @staticmethod
    def _empty(plt, title: str, message: str):
        fig, ax = plt.subplots(figsize=(7, 2))
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes,
                color=_INK["muted"])
        ax.set_axis_off()
        ax.set_title(title)
        return fig

    @staticmethod
    def _year_ticks(ax, x: np.ndarray, years: list) -> None:
        sel = x[::max(1, len(x) // 4)]
        ax.set_xticks(sel, [years[k] for k in sel], fontsize=7)

    # ------------------------------------------------------------------
    def plot(self, odir: str, show: bool = False) -> list:
        """Write the six figures and the stats tables to ``odir``; return the PNGs."""
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

    def _map_columns(self) -> tuple:
        """The criterion outcomes that are retention outcomes, then the other retention
        outcomes in chain order, then the valuation's mass moved; and how many lead."""
        ret = self._retention_outcomes()
        crit = list(dict.fromkeys(o for _, o, _ in self._criterion_outcomes()
                                  if o in set(ret["outcome"])))
        pos = {c: k for k, c in enumerate(self.stats.category_table()["category"])}
        rest = ret[~ret["outcome"].isin(crit)]
        rest = rest.iloc[np.argsort([pos.get(c, len(pos)) for c in rest["category"]],
                                    kind="stable")]
        cols = crit + list(rest["outcome"])
        mass = f"mass_moved_{self.valuation}"
        return cols + ([mass] if mass in self.stats.outcomes() else []), len(crit)

    def _fig_map(self, plt):
        cols, n_crit = self._map_columns()
        title = "SW-R-01  sensitivity map"
        if not cols:
            return self._empty(plt, title, "no retention outcome declared")
        sens = self._sens()
        axes = [axis_label(a) for a in self.stats.axes]
        M = np.full((len(axes), len(cols)), np.nan)
        S = np.full_like(M, np.nan)
        heads, boxed = [], []
        for j, out in enumerate(cols):
            stat, r2 = rank_statistic(sens, out)
            t = sens[sens["outcome"] == out]
            S[:, j] = t[t["statistic"] == "swing"].set_index("axis")["value"].reindex(axes)
            if stat == "swing":
                peak = np.nanmax(np.abs(S[:, j])) if np.isfinite(S[:, j]).any() else 0.0
                M[:, j] = S[:, j] / peak if peak > 0 else np.nan
                fit = ""
            else:
                M[:, j] = t[t["statistic"] == stat].set_index("axis")["value"].reindex(axes)
                fit = f"R² {r2:.2f}" + (" (ρ)" if stat == "spearman" else "")
            heads.append(f"{_short(out, 36)}  {self.stats.window_of(out)}  {fit}")
            boxed += [(axes.index(a), j) for a in influential(sens, out)]
        order = np.argsort(-np.where(np.isfinite(M), np.abs(M), -1.0).max(axis=1),
                           kind="stable")
        row_of = {i: k for k, i in enumerate(order)}
        M, S = M[order], S[order]

        fig, ax = plt.subplots(figsize=(3.0 + 0.42 * len(cols), 2.6 + 0.42 * len(axes)))
        im = ax.imshow(np.ma.masked_invalid(M), cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                dark = np.isfinite(M[i, j]) and abs(M[i, j]) > 0.6
                ax.text(j, i, _fmt(S[i, j]), ha="center", va="center", fontsize=5.5,
                        color="white" if dark else _INK["primary"])
        for i, j in boxed:
            ax.add_patch(plt.Rectangle((j - .5, row_of[i] - .5), 1, 1, fill=False,
                                       ec=_INK["primary"], lw=1.4))
        ax.set_yticks(range(len(axes)), [_short(axes[i], 64) for i in order], fontsize=8)
        ax.set_xticks(range(len(cols)), heads, rotation=90, fontsize=6.5)
        if 0 < n_crit < len(cols):
            ax.axvline(n_crit - .5, color="white", lw=4)
        random = self.stats.mode == "random"
        cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cb.set_label(f"SRRC (Spearman ρ where R² < {R2_MIN:g})" if random
                     else "swing ÷ the column's largest |swing|")
        rule = (f"; boxed = influential (|SRRC| ≥ {SRRC_MIN:g}, R² ≥ {R2_MIN:g})"
                if random else "")
        ax.set_title(f"{title}, {len(self._done())} points: colour = "
                     f"{'the rank statistic' if random else 'the scaled swing'}, text = "
                     f"the swing in the outcome's unit{rule}; criterion outcomes left of "
                     f"the gap", fontsize=9, loc="left")
        return fig

    def _fig_criterion_by_top_axis(self, plt):
        crit = self._criterion_outcomes()
        title = "SW-R-02  each criterion outcome against its top axis"
        if not crit:
            return self._empty(plt, title, "no criterion declared")
        from matplotlib.ticker import NullFormatter
        done, sens = self._done(), self._sens()
        decl = {axis_label(a): a for a in self.stats.axes}
        fig, axs = self._grid(plt, len(crit), 4, 3.6, 3.0)
        for ax, (item, out, limit) in zip(axs, crit):
            ax.axhspan(-limit, limit, color=_CAT8[2], alpha=.12, lw=0)
            ax.axhline(0, color=_INK["muted"], lw=.8)
            top = top_axis(sens, out)
            if top:
                a = decl[top[0]]
                x, levels = self._axis_x(a, done[a["keys"][0]])
                ax.scatter(x, done[out], s=18, color=_CAT8[0], edgecolor="white", lw=.6,
                           zorder=3)
                if levels:
                    ax.set_xticks(range(len(levels)), [_short(lv, 14) for lv in levels],
                                  rotation=30, ha="right", fontsize=7)
                elif a.get("distribution") in _LOG:
                    ax.set_xscale("log")
                    ax.xaxis.set_minor_formatter(NullFormatter())
                ax.set_xlabel(f"{_short(top[0], 44)}\n{_STAT_LABEL[top[1]]} {_fmt(top[2])}",
                              fontsize=8)
            ax.set_title(f"{_short(out, 36)}  {self.stats.window_of(out)}\n{item}: "
                         f"|Δ| < {limit:g}", fontsize=8)
            _apply_modern_style(ax)
        fig.suptitle(f"{title} (band = the criterion's limit)", fontsize=10)
        fig.tight_layout()
        return fig

    def _fig_retention_boxes(self, plt):
        v = self.valuation
        title = f"SW-R-04  {v} pooled retention across the points"
        ct = self.stats.category_table()
        keep = ct[(ct["valuation"] == v) & (ct["no_retention"] == "")]["category"].tolist()
        left = self._left_out()
        if not keep:
            return self._empty(plt, title, f"no {v} category has a retention"
                               + (f"\nleft out: {left}" if left else ""))
        cat = self.stats.category_frame()
        cat = cat[cat["category"].isin(keep)]
        delta = {c: (g["retention"] - g["retention_base"]).to_numpy()
                 for c, g in cat.groupby("category")}
        base = cat.groupby("category")["retention_base"].first()
        ot = self.stats.outcome_table()
        limit = ot[ot["category"] != ""].groupby("category")["limit"].min()

        fig, ax = plt.subplots(figsize=(9, 1.6 + 0.27 * len(keep)))
        for i, c in enumerate(keep):
            lim = limit.get(c, np.nan)
            if np.isfinite(lim):
                ax.add_patch(plt.Rectangle((-lim, i - .42), 2 * lim, .84, color=_CAT8[2],
                                           alpha=.15, lw=0))
        bp = ax.boxplot([delta[c] for c in keep], vert=False, positions=range(len(keep)),
                        whis=(0, 100), widths=.6, patch_artist=True, showfliers=False,
                        medianprops={"color": _INK["primary"], "lw": 1})
        for patch, c in zip(bp["boxes"], keep):
            judged = np.isfinite(limit.get(c, np.nan))
            patch.set(facecolor=_CAT8[1] if judged else _CAT8[0], alpha=.75,
                      edgecolor=_INK["primary"], lw=.6)
        ax.axvline(0, color=_INK["muted"], lw=.8)
        ax.set_yticks(range(len(keep)), [f"{c}  ({base[c]:.3f})" for c in keep],
                      fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel(f"Δ pooled retention vs base {self.stats.base_opt} "
                      f"({self.stats.window_of(None)})")
        ax.set_title(f"{title}, {len(self._done())} points: box = IQR, whiskers = min–max, "
                     f"base retention in brackets;\norange = judged by a criterion, band = "
                     f"its limit", fontsize=9, loc="left")
        if left:
            fig.text(0.01, 0.0, textwrap.fill(f"left out (no retention): {left}", 150),
                     fontsize=7, color=_INK["muted"], va="top")
        _apply_modern_style(ax)
        return fig

    def _fig_criterion_bands(self, plt):
        crit = self._criterion_outcomes()
        title = "SW-R-05  criterion outcomes per point against their limits"
        if not crit:
            return self._empty(plt, title, "no criterion declared")
        done, sens = self._done(), self._sens()
        decl = {axis_label(a): a for a in self.stats.axes}
        cmap = _seq_cmap()
        jitter = ((np.arange(len(done)) * 0.618034) % 1.0 - 0.5) * 0.56
        fig, ax = plt.subplots(figsize=(10, 1.4 + 0.45 * len(crit)))
        for i, (item, out, limit) in enumerate(crit):
            ax.add_patch(plt.Rectangle((-limit, i - .4), 2 * limit, .8, color=_CAT8[2],
                                       alpha=.12, lw=0))
            y = done[out].to_numpy(dtype=float)
            top = top_axis(sens, out)
            shade = np.full(len(done), 0.5)
            if top:
                a = decl[top[0]]
                x, levels = self._axis_x(a, done[a["keys"][0]])
                shade = (x / max(len(levels) - 1, 1) if levels
                         else pd.Series(x).rank(pct=True).to_numpy())
            ax.scatter(y, i + jitter, c=cmap(shade), s=16, edgecolor="white", lw=.5,
                       zorder=3)
            outside = int((~(np.abs(y) < limit)).sum())
            note = (f"{_short(top[0], 46)} {_STAT_LABEL[top[1]]} {_fmt(top[2])}" if top
                    else "no ranked axis")
            ax.text(1.01, i, note + (f"   {outside} outside" if outside else ""),
                    transform=ax.get_yaxis_transform(), va="center", fontsize=7.5,
                    color=_INK["primary"] if outside else _INK["muted"])
        ax.axvline(0, color=_INK["muted"], lw=1)
        ax.set_yticks(range(len(crit)), [f"{item}  {_short(out, 34)}  "
                                         f"{self.stats.window_of(out)}"
                                         for item, out, _ in crit], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"Δ vs base {self.stats.base_opt}")
        ax.set_title(f"{title} (band);\ncolour = the rank of the row's top axis (dark low → "
                     f"yellow high), named on the right", fontsize=9, loc="left")
        _apply_modern_style(ax)
        return fig

    def _fig_year_envelope(self, plt):
        v = self.valuation
        title = f"SW-R-06  per-year Δ retention of every {v} retention outcome"
        outs = self._retention_outcomes(v)
        left = self._left_out()
        if not len(outs):
            return self._empty(plt, title, f"no {v} retention outcome declared")
        yp, done = self.stats.year_profiles(), self._done()
        years = list(dict.fromkeys(yp["year"]))
        xpos = {y: k for k, y in enumerate(years)}
        fig, axs = self._grid(plt, len(outs), _NCOLS, 3.0, 2.3)
        for ax, r in zip(axs, outs.itertuples()):
            g = yp[yp["outcome"] == r.outcome].groupby("year", sort=False)["value"]
            lo = g.min()
            x = np.array([xpos[y] for y in lo.index])
            ax.fill_between(x, lo, g.max(), color=_CAT8[0], alpha=.18, lw=0,
                            label="min–max")
            ax.fill_between(x, g.quantile(.1), g.quantile(.9), color=_CAT8[0], alpha=.35,
                            lw=0, label="10–90 %")
            if np.isfinite(r.limit):
                ax.axhspan(-r.limit, r.limit, color=_CAT8[2], alpha=.1, lw=0)
                pooled = done[r.outcome].to_numpy(dtype=float)
                for k, idx in enumerate(done["index"][~(np.abs(pooled) < r.limit)]):
                    b = yp[(yp["outcome"] == r.outcome) & (yp["index"] == idx)]
                    ax.plot([xpos[y] for y in b["year"]], b["value"], color=_CAT8[1],
                            lw=.8, alpha=.8, label="breaks the limit" if k == 0 else None)
            ax.axhline(0, color=_INK["muted"], lw=.8)
            ax.set_title(f"{_short(r.outcome, 34)}  {r.window}", fontsize=7.5)
            self._year_ticks(ax, x, years)
            _apply_modern_style(ax)
        axs[0].legend(frameon=False, fontsize=6.5, loc="upper left")
        fig.suptitle(f"{title} across {len(done)} points (band = a criterion's limit; lines "
                     f"= points whose pooled value breaks it)"
                     + (f"\n{textwrap.fill('left out (no retention): ' + left, 180)}"
                        if left else ""), fontsize=9)
        fig.tight_layout()
        return fig

    def _fig_sensitivity_by_year(self, plt):
        title = "SW-R-07  per-year sensitivity of every retention outcome"
        outs = self._retention_outcomes()["outcome"].tolist()
        if not outs:
            return self._empty(plt, title, "no retention outcome declared")
        sens, random = self._sens(), self.stats.mode == "random"
        axes = [axis_label(a) for a in self.stats.axes]
        if random:
            hit = set().union(*(influential(sens, o) for o in outs))
            axes = [a for a in axes if a in hit]
        if not axes:
            return self._empty(plt, title, "no axis is influential on a pooled retention "
                               "outcome")
        sby = self._sens_by_year()
        years = list(dict.fromkeys(sby["year"]))
        xpos = {y: k for k, y in enumerate(years)}
        palette = _CAT8 + _CAT_EXT
        fig, axs = self._grid(plt, len(outs), _NCOLS, 3.0, 2.3)
        for ax, out in zip(axs, outs):
            t = sby[sby["outcome"] == out]
            r2 = t[t["statistic"] == "r2"].set_index("year")["value"]
            for k, a in enumerate(axes):
                ta = t[t["axis"] == a]
                if random:
                    s = ta[ta["statistic"] == "srrc"].set_index("year")["value"]
                    rho = ta[ta["statistic"] == "spearman"].set_index("year")["value"]
                    srrc = (r2.reindex(s.index) >= R2_MIN).to_numpy()
                    y = np.where(srrc, s.to_numpy(), rho.reindex(s.index).to_numpy())
                else:
                    s = ta[ta["statistic"] == "swing"].set_index("year")["value"]
                    srrc, y = np.ones(len(s), dtype=bool), s.to_numpy()
                x = np.array([xpos[yr] for yr in s.index])
                c = palette[k % len(palette)]
                ax.plot(x, y, color=c, lw=1, label=_short(a, 64))
                ax.scatter(x[srrc], y[srrc], s=9, color=c, zorder=3)
                ax.scatter(x[~srrc], y[~srrc], s=9, facecolor="white", edgecolor=c,
                           zorder=3)
            if random:
                for h in (SRRC_MIN, -SRRC_MIN):
                    ax.axhline(h, ls=":", color=_INK["muted"], lw=.8)
                ax.set_ylim(-1.05, 1.05)
            ax.axhline(0, color=_INK["muted"], lw=.6)
            ax.set_title(f"{_short(out, 34)}  {self.stats.window_of(out)}", fontsize=7.5)
            self._year_ticks(ax, np.array(sorted(set(xpos[yr] for yr in t["year"]))),
                             years)
            _apply_modern_style(ax)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(axes)),
                   frameon=False, fontsize=8)
        what = (f"SRRC per year, filled; Spearman ρ, hollow, where that year's R² < "
                f"{R2_MIN:g}; dotted = ±{SRRC_MIN:g}" if random else "swing per year")
        fig.suptitle(f"{title}: {what}", fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        return fig

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """The verdicts, the top axis per ranked outcome with its swing, the categories
        without retention, the hard-row check and the failed points."""
        df = self.stats.points_frame()
        lines = [f"sweep {self.stats.sweep_run_id}: {self.stats.mode}, {len(df)} point(s), "
                 f"base {self.stats.base_opt}",
                 "  status: " + ", ".join(f"{s} {n}" for s, n in
                                          df["status"].value_counts().items())]
        verdicts = self.stats.verdicts()
        if len(verdicts):
            lines.append("  criteria (a point meets one when every |outcome| is below its "
                         "limit; it holds when every done point does):")
        else:
            lines.append("  no criterion declared")
        for r in verdicts.itertuples():
            worst = (f"; worst {r.worst_value:.3g} at p{int(r.worst_point):03d}: "
                     f"{r.worst_outcome} ({r.worst_window})"
                     if r.worst_outcome is not None else "")
            held = "holds" if r.points_meeting == r.points_done else "fails"
            lines.append(f"    {r.item:<14s} {r.points_meeting}/{r.points_done} points "
                         f"below {r.limit:g} — {held}{worst}")

        sens = self._sens()
        rule = (f"|SRRC| where R² ≥ {R2_MIN:g}, else |Spearman ρ|; * = influential: "
                f"|SRRC| ≥ {SRRC_MIN:g} with R² ≥ {R2_MIN:g}"
                if self.stats.mode == "random" else "|swing|")
        lines.append(f"  top axis per outcome ({rule}), and that axis's swing:")
        for out in self.stats.outcomes():
            top = top_axis(sens, out)
            if not top:
                continue
            sw = sens[(sens["outcome"] == out) & (sens["axis"] == top[0])
                      & (sens["statistic"] == "swing")]["value"]
            star = " *" if top[0] in influential(sens, out) else ""
            lines.append(f"    {out:<40s} {self.stats.window_of(out):<10s} {top[0]} "
                         f"({_STAT_LABEL[top[1]]} {top[2]:+.3g}{star}), swing "
                         f"{_fmt(sw.iloc[0]) if len(sw) else 'n/a'}")

        ct = self.stats.category_table()
        no = ct[ct["no_retention"] != ""]
        if len(no):
            lines.append("  categories without a retention: "
                         + ", ".join(f"{c} ({r})" for c, r in
                                     zip(no["category"], no["no_retention"])))
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
        text = "\n".join(lines)
        print(text)
        return text
