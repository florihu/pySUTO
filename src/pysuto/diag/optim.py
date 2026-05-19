"""
optim.py
========
``OptimDiagnostics`` — Stage 4 optimisation-side diagnostic suite.

Optimisation residuals, σ_opt vs σ_agg, convergence across timesteps.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._plot import _C, _savefig


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

        path = os.path.join(odir, "optim.png")
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

