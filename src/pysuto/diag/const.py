"""
const.py
========
``ConstDiagnostic`` — Stage 3 (constraint-build) diagnostic suite.

G-matrix sparsity, prior distribution, constraint residuals
ε₀ = c − G·a₀.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._plot import _C, _grouped_box_grid, _savefig


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

    def __init__(self, constraint_run, label: str = "", year: str = "1995") -> None:
        self.cr    = constraint_run
        self.label = label
        self.year  = str(year)

    # ------------------------------------------------------------------
    # Classmethod — load from disk
    # ------------------------------------------------------------------

    @classmethod
    def from_saved(cls, run_dir: str, label: str = "",
                   year: str = "1995") -> "ConstDiagnostic":
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

        return cls(cr, label=label, year=year)

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
            (self._fig4_sum_residuals(),  "fig4_sum_residuals.png",  "sum residuals "),
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
        """True spy of the focus-year sub-block (rows active at ``self.year`` ×
        the edge columns they touch). Falls back to the full G when the year is
        absent. The no-cross-time-coupling invariant guarantees the columns
        touched by a year's rows are exactly that year's edges."""
        import matplotlib.pyplot as plt

        G = self.cr.G

        rows_y = np.where(self._year_row_mask())[0]
        if rows_y.size:
            cols_y = np.unique(G[rows_y].tocoo().col)
            G_sub  = G[rows_y][:, cols_y]
            scope  = f"({self.year})"
        else:
            G_sub  = G
            scope  = "(all years)"

        M, E = G_sub.shape
        nnz  = G_sub.nnz
        density = 100 * nnz / (M * E) if M * E else 0

        suffix = f" — {self.label}" if self.label else ""
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.spy(G_sub, markersize=0.5, aspect="auto", color=_C["navy"])
        ax.set_title(
            f"G sparsity {scope}{suffix}   "
            f"{M:,}×{E:,} rows×edges   "
            f"{nnz:,} non-zeros   ({density:.4f} % dense)\n"
            f"active sub-block — true spy (one mark per non-zero)",
            fontsize=10, fontweight="bold",
        )
        ax.set_xlabel(f"Edge column  (0 – {E - 1:,})")
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

    def _fig4_sum_residuals(self):
        """Per-family boxplot of log₁₀|ε₀| for the soft sum-anchors active at the
        focus year (single CC panel — sum anchors are CC-only)."""
        suffix = f" — {self.label}" if self.label else ""

        eps0, _c, t = self._eps0_at(self.year)
        rm   = self.cr.row_meta
        mask = ((rm["type"].values == "sum_eq") & self._year_row_mask()
                if eps0 is not None else np.zeros(len(rm), dtype=bool))

        if mask.any():
            family = self._sum_family(rm["constraint_id"][mask]).to_numpy()
            vals   = np.abs(eps0[mask])
            keep   = np.isfinite(vals) & (vals > 0)
            cells  = {(self.year, "CC"): (family[keep], np.log10(vals[keep]))}
        else:
            cells  = {(self.year, "CC"): (np.array([], dtype=str),
                                          np.array([], dtype=float))}

        return _grouped_box_grid(
            cells=cells,
            row_keys=[self.year],
            col_keys=["CC"],
            group_label="anchor family",
            y_label="log₁₀|ε₀|",
            suptitle=f"Sum-anchor prior residuals (CC, {self.year}){suffix}",
            cell_title=lambda r, c: f"{c} · {r}",
        )

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

    def _year_row_mask(self) -> np.ndarray:
        """Boolean row mask for constraints whose time matches the focus year."""
        rm = self.cr.row_meta
        if "time" not in rm.columns or not len(rm):
            return np.zeros(len(rm), dtype=bool)
        return rm["time"].astype(str).values == self.year

    @staticmethod
    def _sum_family(cid_series: pd.Series) -> pd.Series:
        """Map sum-anchor constraint_ids to their family prefix.

        e.g. ``mine_prod_anchor_Argentina_1997_CC`` → ``mine_prod_anchor``.
        """
        return cid_series.astype(str).str.split("_anchor").str[0] + "_anchor"

    def _eps0_at(self, year: str):
        """Return (eps0, c, t_label) for ``year`` (or the first timestep when
        ``year`` is absent), or (None, None, None) when there is no timestep data."""
        cr = self.cr
        if not cr.timestep_data:
            return None, None, None
        t = year if year in cr.timestep_data else next(iter(cr.timestep_data))
        c = cr.timestep_data[t]["c"]
        eps0 = c - cr.G.dot(cr.a0)
        return eps0, c, t

    def _eps0_ref(self):
        """Return (eps0, c_t0, t0_label) for the first timestep, or (None, None, None)."""
        return self._eps0_at(next(iter(self.cr.timestep_data))) \
            if self.cr.timestep_data else (None, None, None)

    def _panel_c_by_year(self, ax) -> None:
        """
        |c| distribution per timestep, all years overlaid with a viridis
        colour ramp (light → recent year). For n > 12 timesteps the per-year
        legend is replaced by a colorbar so individual years remain
        distinguishable; for n ≤ 12 the legend is shown directly.
        """
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        cr         = self.cr
        soft_mask  = self._soft_mask()

        if not soft_mask.any():
            ax.text(0.5, 0.5, "No soft constraints", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("|c| distribution (soft constraints)", fontweight="bold")
            return

        timesteps = list(cr.timestep_data.keys())
        n    = len(timesteps)
        cmap = plt.cm.viridis

        # Outlined steps (not filled) so every year remains visible — with 30+
        # overlaid stepfilled histograms the latest year was drowning out the
        # rest. Compute a common bin grid for fair comparison.
        all_log_vals = []
        for t_label in timesteps:
            c = cr.timestep_data[t_label]["c"][soft_mask]
            v = np.abs(c)
            v = v[np.isfinite(v) & (v > 0)]
            if len(v):
                all_log_vals.append(np.log10(v))
        if all_log_vals:
            lo = float(min(a.min() for a in all_log_vals))
            hi = float(max(a.max() for a in all_log_vals))
            bins = np.linspace(lo, hi, 41)
        else:
            bins = 40

        for i, t_label in enumerate(timesteps):
            c    = cr.timestep_data[t_label]["c"][soft_mask]
            vals = np.abs(c)
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if not len(vals):
                continue
            color = cmap(i / max(n - 1, 1))
            ax.hist(np.log10(vals), bins=bins, color=color,
                    histtype="step", linewidth=1.0,
                    label=t_label, density=True)

        ax.set_title("|c| distribution (soft constraints)", fontweight="bold")
        ax.set_xlabel("log₁₀(|c|)")
        ax.set_ylabel("density")
        if n <= 12:
            ax.legend(fontsize=6, framealpha=0.6)
        else:
            # Numeric x-axis for the colorbar; falls back to integer indices
            # when timestep labels are not parseable as numbers.
            try:
                t_numeric = [float(t) for t in timesteps]
                cbar_lo, cbar_hi = float(min(t_numeric)), float(max(t_numeric))
                cbar_label = "year"
            except (TypeError, ValueError):
                cbar_lo, cbar_hi = 0.0, float(n - 1)
                cbar_label = "timestep index"
            sm = ScalarMappable(norm=Normalize(vmin=cbar_lo, vmax=cbar_hi), cmap=cmap)
            sm.set_array([])
            cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02, fraction=0.04, aspect=30)
            cbar.set_label(cbar_label, fontsize=8)
            cbar.ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)

    def _panel_sigma_c_by_year(self, ax) -> None:
        """
        σ_c distribution at the **first timestep only**.

        Soft sum-constraint rows whose spec belongs to a different year
        are deactivated for this t via ``σ_c = SIGMA_C_INACTIVE`` (≥ 1e10);
        we exclude them so the histogram shows only this year's active
        rows.
        """
        from ..constraints import SIGMA_C_INACTIVE

        cr        = self.cr
        soft_mask = self._soft_mask()

        if not soft_mask.any():
            ax.text(0.5, 0.5, "No soft constraints", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("σ_c distribution (soft constraints)", fontweight="bold")
            return

        if not cr.timestep_data:
            ax.text(0.5, 0.5, "No timestep data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("σ_c distribution (soft constraints)", fontweight="bold")
            return

        t0   = next(iter(cr.timestep_data))
        sc   = cr.timestep_data[t0]["sigma_c"][soft_mask]
        # Drop deactivated sentinel entries (σ_c ≈ SIGMA_C_INACTIVE).
        active = np.isfinite(sc) & (sc > 0) & (sc < SIGMA_C_INACTIVE * 0.5)
        vals = sc[active]

        if not len(vals):
            ax.text(0.5, 0.5, "All σ_c ≤ 0 or non-finite", ha="center",
                    va="center", transform=ax.transAxes)
        else:
            ax.hist(np.log10(vals), bins=40, alpha=0.8, color=_C["green"],
                    histtype="stepfilled", edgecolor="none", density=True)

        ax.set_title(f"σ_c distribution (soft constraints)  (t={t0})",
                     fontweight="bold")
        ax.set_xlabel("log₁₀(σ_c)")
        ax.set_ylabel("density")
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
        """|ε₀| vs |c| for the soft sum-anchors active at the focus year,
        coloured by anchor family."""
        import matplotlib.pyplot as plt

        eps0, c_t, t = self._eps0_at(self.year)
        if eps0 is None:
            ax.text(0.5, 0.5, "No timestep data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("|ε₀| vs |c| (sum anchors)", fontweight="bold")
            return

        rm   = self.cr.row_meta
        mask = (rm["type"].values == "sum_eq") & self._year_row_mask()
        if not mask.any():
            ax.text(0.5, 0.5, "No sum anchors at year", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"|ε₀| vs |c| (sum anchors)  (t={t})", fontweight="bold")
            return

        cx     = np.abs(c_t)[mask]
        epsx   = np.abs(eps0)[mask]
        family = self._sum_family(rm["constraint_id"][mask]).to_numpy()

        fams   = sorted(pd.unique(family))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(fams), 1)))
        for fam, color in zip(fams, colors):
            fm    = family == fam
            valid = fm & np.isfinite(cx) & np.isfinite(epsx) & (cx > 0) & (epsx > 0)
            if not valid.any():
                continue
            ax.scatter(cx[valid], epsx[valid], s=6, alpha=0.5, color=color,
                       label=fam, linewidths=0)

        # Diagonal reference line
        valid_all = np.isfinite(cx) & np.isfinite(epsx) & (cx > 0) & (epsx > 0)
        if valid_all.any():
            lo = min(cx[valid_all].min(), epsx[valid_all].min())
            hi = max(cx[valid_all].max(), epsx[valid_all].max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="ε₀ = c")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"|ε₀| vs |c| (sum anchors by family)  (t={t})", fontweight="bold")
        ax.set_xlabel("|c|")
        ax.set_ylabel("|ε₀|")
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=6, framealpha=0.6)
        ax.grid(alpha=0.3, which="both")


# ===========================================================================
# AggDiagnostic — post-aggregation distribution diagnostics (Stage 2a)
# ===========================================================================

