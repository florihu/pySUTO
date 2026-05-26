"""
solver.py
=========
``SolverDiag`` — Stage 4 solver-side diagnostic suite.

Owns its own colour palette and styling helper because it renders a
much wider plot surface than the other diagnostics.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._plot import _C, _savefig


_TYPE_COLORS = {
    "mass_balance": _C["navy"],     # Type 1 (hard equality)
    "sum_eq":       _C["blue"],     # Type 2 (soft sum)
    "ratio_eq":     _C["purple"],   # Type 3 (soft ratio)
    "ratio_bound":  _C["orange"],   # Type 4 (hard ratio bound)
    "flow_bound":   _C["green"],    # Type 5 (hard flow bound — synthetic, from u/l)
}

# Display labels for the ratio_bound sub-kinds (sourced from params_id upstream).
# stock_bounds is rendered as "stock share" in figure titles/legends.
_RATIO_KIND_LABELS = {"conc": "conc", "eff": "eff", "stock_bounds": "stock share"}
_RATIO_KINDS_ORDER = ("conc", "eff", "stock_bounds")
_BOUND_DIRS_ORDER  = ("upper", "lower")

# Hue palette for SolverDiag.compare(). Order matters: first run gets blue.
_HUE_PALETTE = ["blue", "orange", "green", "purple", "red", "navy"]

# Plot IDs hueable under SolverDiag.compare(). Excluded: SD-R-07 (per-edge
# semantics differ across runs) and every SD-T-* (cross-sections).
_HUEABLE_IDS = ("SD-R-01", "SD-R-02", "SD-R-03", "SD-R-04", "SD-R-05")


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

    * Run-scope  : SD-R-01 … SD-R-09 (x = timestep); SD-R-01 … SD-R-05 hueable
    * Per-timestep cross-section : SD-T-01 … SD-T-07 (columns = t_min, t_worst, t_max)
    * Constraint ε heatmaps: SD-T-08a (ratio_eq), SD-T-08b (sum_eq)

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
        Optional per-timestep σ_opt (= sigma_a_star from
        :class:`pysuto.uncprop.KalmanUncProp`) — when provided unlocks
        the σ_opt / σ_a panel in SD-T-03.
    """

    # ---- IDs (kept in plot() iteration order) ------------------------------
    RUN_PLOT_IDS = (
        "SD-R-01",  # Solver health
        "SD-R-02",  # Mass-balance leakage
        "SD-R-03",  # Soft-slack stress
        "SD-R-04",  # Hard-inequality activity
        "SD-R-05",  # Warm-start delta
        "SD-R-06",  # F*(t) evolution
        "SD-R-07",  # Time-hued distributions
        "SD-R-08",  # Process-entity outputs (stacked area)
        "SD-R-09",  # Flow-entity outputs (stacked area)
    )
    TIMESTEP_PLOT_IDS = (
        "SD-T-01",  # Residual digest
        "SD-T-02",  # Soft-slack z-scores
        "SD-T-03",  # Solution vs prior cross-section
        "SD-T-04",  # Per-type binding rate
        "SD-T-05",  # Distribution of a
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

        # Provenance fields — populated by from_opt_id; None in live mode
        self._opt_run_id:      Optional[str]  = None
        self._opt_entry:       Optional[dict] = None
        self._imp_index_entry: Optional[dict] = None
        self._feed_entries:    Optional[list] = None

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
        imp_dir:    str = "data/proc/imp",
        feed_dir:   str = "data/proc/feeds",
        label:      str = "",
    ) -> "SolverDiag":
        """Reconstruct results + ConstraintRun from disk."""
        # Local import to keep diag.py at the leaf of the import graph for
        # CI environments without the full Stage-3 stack installed.
        from pysuto.registry import ConstraintRunRegistry, OptimRunRegistry

        opt_reg = OptimRunRegistry(output_dir)
        if opt_run_id not in opt_reg._index:
            raise KeyError(
                f"opt_run_id '{opt_run_id}' not in {opt_reg._index_path}. "
                f"Known: {sorted(opt_reg.run_ids())}"
            )
        sidecar = opt_reg.load_sidecar(opt_run_id)
        results = opt_reg.load(opt_run_id)
        const_run_id = (sidecar.get("parents", {})
                                .get("upstream", {})
                                .get("const_run_id"))

        const_run = ConstraintRunRegistry(const_dir).load(const_run_id)

        instance = cls(results, const_run, label=label or opt_run_id)
        instance._default_odir = os.path.join(output_dir, opt_run_id, "diag")

        # Populate provenance chain
        instance._opt_run_id = opt_run_id
        instance._opt_entry  = opt_reg[opt_run_id].to_index_dict()

        agg_id = const_run.agg_id
        if agg_id:
            try:
                from pysuto.registry import ImputationRunRegistry, DataFeedRegistry
                imp_reg = ImputationRunRegistry(imp_dir)
                if agg_id in imp_reg._index:
                    imp_sidecar = imp_reg.load_sidecar(agg_id)
                    instance._imp_index_entry = imp_sidecar
                    feed_run_ids = (imp_sidecar.get("parents", {})
                                                 .get("upstream", {})
                                                 .get("feed_run_ids", []))
                    feed_reg = DataFeedRegistry(feed_dir)
                    feed_reg.load_index()
                    instance._feed_entries = []
                    for fid in feed_run_ids:
                        try:
                            e = feed_reg.resolve(fid)
                            instance._feed_entries.append(e.to_index_dict())
                        except KeyError:
                            instance._feed_entries.append({
                                "run_id":     fid,
                                "label":      "(not registered)",
                                "created_at": "—",
                                "stage":      "feed",
                            })
            except Exception:
                pass  # best-effort; missing registries are silently skipped

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
        groups: dict[tuple[str, str], np.ndarray] = {}
        if "ratio_kind" in rm.columns and "bound_dir" in rm.columns:
            rk = rm["ratio_kind"].astype(str).to_numpy()
            bd = rm["bound_dir"].astype(str).to_numpy()
            is_rb = (tp == "ratio_bound")
            for kind in _RATIO_KINDS_ORDER:
                for direction in _BOUND_DIRS_ORDER:
                    mask = is_rb & (rk == kind) & (bd == direction)
                    groups[(kind, direction)] = np.where(mask)[0]
        info["ratio_bound_groups"] = groups
        return info

    def _t_worst(self) -> str:
        """Timestep with largest ||eps||_2."""
        return max(self._eps_norms, key=self._eps_norms.get)

    def _t_median(self) -> str:
        """Chronological middle timestep (index T//2)."""
        return self._timesteps[len(self._timesteps) // 2]

    def _resolve_timesteps(self, timesteps) -> tuple[str, str, str]:
        if timesteps is None:
            ts_min = self._timesteps[0]
            ts_max = self._timesteps[-1]
            ts_mid = self._t_median()
            return ts_min, ts_mid, ts_max
        if len(timesteps) != 3:
            raise ValueError(f"timesteps must be a triple, got {timesteps!r}")
        # Coerce to str: timestep keys are stored as strings on disk
        coerced = tuple(str(t) for t in timesteps)
        for t in coerced:
            if t not in self._results:
                raise KeyError(f"timestep '{t}' not in results")
        return coerced  # type: ignore[return-value]

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
        """Render the full 15-figure portfolio.

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

        paths: list[str] = []

        # ---- Scope 1: run-scope ----------------------------------------
        for fn, pid, fname in [
            (self._fig_solver_health,    "SD-R-01", "sd_r_01_solver_health"),
            (self._fig_mass_balance,     "SD-R-02", "sd_r_02_mass_balance"),
            (self._fig_soft_slack,       "SD-R-03", "sd_r_03_soft_slack"),
            (self._fig_hard_active,      "SD-R-04", "sd_r_04_hard_active"),
            (self._fig_warmstart_delta,  "SD-R-05", "sd_r_05_warmstart_delta"),
        ]:
            fig = fn(hue=None)
            p = _savefig(fig, os.path.join(odir, f"{fname}.png"), dpi=dpi)
            print(f"   {pid}  → {p}  ✓")
            paths.append(p)
            if not show:
                import matplotlib.pyplot as plt; plt.close(fig)

        # SD-R-06 has no hue parameter
        fig = self._fig_f_evolution()
        p = _savefig(fig, os.path.join(odir, "sd_r_06_f_evolution.png"), dpi=dpi)
        print(f"   SD-R-06  → {p}  ✓")
        paths.append(p)
        if not show:
            import matplotlib.pyplot as plt; plt.close(fig)

        # SD-R-07 — time-hued histograms (per-timestep colour gradient)
        fig = self._fig_time_hued()
        p = _savefig(fig, os.path.join(odir, "sd_r_07_time_hued.png"), dpi=dpi)
        print(f"   SD-R-07  → {p}  ✓")
        paths.append(p)
        if not show:
            import matplotlib.pyplot as plt; plt.close(fig)

        # SD-R-08 / SD-R-09 — entity-output stacked area by sector
        for fn, pid, fname in [
            (self._fig_process_outputs, "SD-R-08", "sd_r_08_process_outputs"),
            (self._fig_flow_outputs,    "SD-R-09", "sd_r_09_flow_outputs"),
        ]:
            fig = fn()
            p = _savefig(fig, os.path.join(odir, f"{fname}.png"), dpi=dpi)
            print(f"   {pid}  → {p}  ✓")
            paths.append(p)
            if not show:
                import matplotlib.pyplot as plt; plt.close(fig)

        # ---- Scope 2: timestep cross-section --------------------------
        for fn, pid, fname in [
            (self._fig_residual_digest,    "SD-T-01", "sd_t_01_residual_digest"),
            (self._fig_zscore_soft,        "SD-T-02", "sd_t_02_zscore_soft"),
            (self._fig_solution_vs_prior,  "SD-T-03", "sd_t_03_solution_vs_prior"),
            (self._fig_binding_rate,       "SD-T-04", "sd_t_04_binding_rate"),
            (self._fig_solution_magnitude, "SD-T-05", "sd_t_05_distribution_of_a"),
            (self._fig_cstar_vs_c,         "SD-T-06", "sd_t_06_cstar_vs_c"),
            (self._fig_rocket,             "SD-T-07", "sd_t_07_rocket"),
        ]:
            fig = fn(timesteps=ts_triple)
            p = _savefig(fig, os.path.join(odir, f"{fname}.png"), dpi=dpi)
            print(f"   {pid}  → {p}  ✓")
            paths.append(p)
            if not show:
                import matplotlib.pyplot as plt; plt.close(fig)

        # SD-T-08a/b — constraint-level ε heatmaps split by type (not cross-sections)
        fig = self._fig_eps_heatmap_ratio()
        p = _savefig(fig, os.path.join(odir, "sd_t_08a_eps_heatmap_ratio.png"), dpi=dpi)
        print(f"   SD-T-08a  → {p}  ✓")
        paths.append(p)
        if not show:
            import matplotlib.pyplot as plt; plt.close(fig)

        fig = self._fig_eps_heatmap_sum()
        p = _savefig(fig, os.path.join(odir, "sd_t_08b_eps_heatmap_sum.png"), dpi=dpi)
        print(f"   SD-T-08b  → {p}  ✓")
        paths.append(p)
        if not show:
            import matplotlib.pyplot as plt; plt.close(fig)

        # SD-T-09a/b/c/d — |a*−a₀(t)| boxplots by edge metadata
        for fig_fn, sd_id, fname in [
            (self._fig_prior_departure_from_sector,
                "SD-T-09a", "sd_t_09a_prior_dep_from_sector.png"),
            (self._fig_prior_departure_to_sector,
                "SD-T-09b", "sd_t_09b_prior_dep_to_sector.png"),
            (self._fig_prior_departure_from_region,
                "SD-T-09c", "sd_t_09c_prior_dep_from_region.png"),
            (self._fig_prior_departure_to_region,
                "SD-T-09d", "sd_t_09d_prior_dep_to_region.png"),
        ]:
            fig = fig_fn()
            p = _savefig(fig, os.path.join(odir, fname), dpi=dpi)
            print(f"   {sd_id}  → {p}  ✓")
            paths.append(p)
            if not show:
                import matplotlib.pyplot as plt; plt.close(fig)

        prov_path = self._write_provenance(odir)
        if prov_path:
            paths.append(prov_path)

        return paths

    # ------------------------------------------------------------------
    # Provenance trace
    # ------------------------------------------------------------------

    def _write_provenance(self, odir: str) -> Optional[str]:
        """Write {label}_provenance.md to *odir* and print a compact tree to stdout."""
        from datetime import datetime as _dt

        out_path  = os.path.join(odir, "provenance.md")
        now       = _dt.utcnow().isoformat(timespec="seconds") + "Z"

        cr = self._const_run

        # ── Collect node info ──────────────────────────────────────────
        opt_id       = self._opt_run_id or "(not registered)"
        opt_label    = (self._opt_entry or {}).get("label", "") or "—"
        opt_created  = (self._opt_entry or {}).get("created_at", "—")
        # Infer opt run dir two levels above odir (odir = .../run_id/diag/)
        opt_path     = os.path.dirname(os.path.dirname(odir)) if self._opt_run_id else "—"

        const_id      = cr.run_id
        const_label   = getattr(cr, "label", "") or "—"
        const_created = cr.created_at
        n_rows, n_edges = cr.G.shape

        agg_id       = cr.agg_id or "(not registered)"
        imp          = self._imp_index_entry or {}
        imp_label    = imp.get("label", "") or "—"
        imp_created  = imp.get("created_at", "—")
        imp_n_edges  = imp.get("n_edges", "—")
        imp_n_filled = imp.get("n_filled", "—")
        imp_n_steps  = imp.get("n_steps", "—")
        imp_n_before = imp.get("n_covered_before", "—") if imp else "—"
        imp_n_after  = imp.get("n_covered_after", "—") if imp else "—"
        # Infer imp path from the imp_dir used in from_opt_id (best-effort)
        imp_path     = f"…/{agg_id}/" if cr.agg_id else "—"

        feeds = self._feed_entries or []

        # ── Build markdown ─────────────────────────────────────────────
        lines = [
            f"# Provenance: {self.label or tag}",
            "",
            f"Generated: {now}",
            "",
            "## Optimisation run",
            f"- **ID**: `{opt_id}`",
            f"- **Label**: {opt_label}",
            f"- **Created**: {opt_created}",
            f"- **Path**: {opt_path}",
            "",
            "## Constraint run",
            f"- **ID**: `{const_id}`",
            f"- **Label**: {const_label}",
            f"- **Created**: {const_created}",
            f"- **Constraints**: {n_rows} rows × {n_edges} edges",
            "",
            "## Imputation / aggregation run",
            f"- **ID**: `{agg_id}`",
            f"- **Label**: {imp_label}",
            f"- **Created**: {imp_created}",
            f"- **Coverage**: {imp_n_before} → {imp_n_after} edges"
            f"  ({imp_n_filled} filled, {imp_n_steps} steps)",
            f"- **Path**: {imp_path}",
        ]

        if feeds:
            lines += [
                "",
                "### Feeds",
                "| feed_id | feed_run_id | label | created | path |",
                "|---|---|---|---|---|",
            ]
            for fe in feeds:
                fid     = fe.get("feed_id", "—")
                frun    = fe.get("feed_run_id", "—")
                flabel  = fe.get("label", "") or "—"
                fdate   = (fe.get("created_at", "—") or "—")[:10]
                fpath   = fe.get("odir", "—")
                lines.append(f"| {fid} | `{frun}` | {flabel} | {fdate} | {fpath} |")

        os.makedirs(odir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        # ── Compact stdout tree ────────────────────────────────────────
        feed_ids_display = [fe.get("feed_run_id", fe.get("feed_id", "?")) for fe in feeds]
        print(f"   provenance  {opt_id}")
        print(f"               └── {const_id}")
        print(f"               │   └── {agg_id}")
        for i, fid in enumerate(feed_ids_display):
            connector = "└──" if i == len(feed_ids_display) - 1 else "├──"
            print(f"               │       {connector} {fid}")
        print(f"               → {out_path}  ✓")

        return out_path

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
            "SD-R-05": ("_fig_warmstart_delta",  "sd_r_05_warmstart_delta"),
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
        fig.suptitle(self._title("Solver health (SD-R-01)"), fontsize=13, fontweight="bold")

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
        fig.suptitle(self._title("Mass-balance leakage (SD-R-02)"),
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
        fig.suptitle(self._title("Soft-slack stress (SD-R-03)"),
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
            eps_max = np.array([float(np.max(np.abs(dg._results[t].eps_opt))) if dg._results[t].eps_opt.size else np.nan
                                for t in dg._timesteps])
            axes[0].semilogy(x, np.maximum(eps_inf, 1e-16), color=color, lw=1.6, ls="-",  marker="o", ms=3, label=f"{lbl}: ‖ε‖∞")
            axes[0].semilogy(x, np.maximum(eps_l2,  1e-16), color=color, lw=1.4, ls="--", marker="s", ms=3, label=f"{lbl}: ‖ε‖₂")
            axes[0].semilogy(x, np.maximum(eps_med, 1e-16), color=color, lw=1.2, ls=":",  marker="^", ms=3, label=f"{lbl}: median|ε|")
            axes[0].semilogy(x, np.maximum(eps_max, 1e-16), color=color, lw=1.0, ls="-.", marker="D", ms=3, label=f"{lbl}: max|ε|")

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
        """SD-R-04: active Type-4 ratio bounds, split by (ratio_kind × bound_dir).

        3×2 grid: rows = conc/eff/share, cols = upper/lower. Each cell shows
        the fraction of rows of that (kind, dir) bucket that are active
        (slack within solver tolerance) over time.
        """
        import matplotlib.pyplot as plt

        n_rows = len(_RATIO_KINDS_ORDER)
        n_cols = len(_BOUND_DIRS_ORDER)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 8.5), sharex=True)
        fig.suptitle(self._title("Hard ratio-bound activity (SD-R-04)"),
                     fontsize=13, fontweight="bold")

        runs = hue if hue else [(self.label or "run", self, _C["orange"])]
        tol = 1e-7  # solver eps_abs by default

        for lbl, dg, color in runs:
            x = dg._ts_index()
            groups = dg._row_info.get("ratio_bound_groups", {})
            for r, kind in enumerate(_RATIO_KINDS_ORDER):
                for c, direction in enumerate(_BOUND_DIRS_ORDER):
                    ax = axes[r, c]
                    idx = groups.get((kind, direction), np.array([], dtype=int))
                    n = idx.size
                    if n:
                        frac = np.array([
                            np.sum(np.abs(-dg._results[t].residuals[idx]) <= tol) / n
                            for t in dg._timesteps
                        ])
                    else:
                        frac = np.zeros(len(dg._timesteps))
                    ax.plot(x, frac, color=color, lw=1.6, marker="o", ms=3.5, label=lbl)

        for r, kind in enumerate(_RATIO_KINDS_ORDER):
            for c, direction in enumerate(_BOUND_DIRS_ORDER):
                ax = axes[r, c]
                n = self._row_info.get("ratio_bound_groups", {}).get(
                    (kind, direction), np.array([], dtype=int)).size
                label = _RATIO_KIND_LABELS.get(kind, kind)
                ax.set_title(f"{label} · {direction}  (n={n})",
                             loc="left", fontsize=9)
                ax.set_ylim(-0.01, 1.05)
                if c == 0:
                    ax.set_ylabel("active fraction")
                _apply_modern_style(ax)

        for c in range(n_cols):
            axes[-1, c].set_xticks(self._ts_index())
            axes[-1, c].set_xticklabels(self._ts_labels_for_xticks(),
                                       rotation=45, ha="right", fontsize=8)
            axes[-1, c].set_xlabel("timestep")

        if hue is not None:
            axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def _fig_warmstart_delta(self, hue=None):
        """SD-R-05: consecutive ||a*(t) − a*(t-1)|| over time."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(13, 6))
        fig.suptitle(self._title("Warm-start delta (SD-R-05)"),
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
        """SD-R-06: heatmap of F*(t) for top-K most-variable edges + top-10 lines."""
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
        fig.suptitle(self._title("F*(t) evolution (SD-R-06)"),
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

        # Panel B: top-10 most-variable edges as labelled lines (symlog y)
        ax = axes[1]
        cmap = plt.get_cmap("tab10")
        n_top = min(10, F_top.shape[0])
        F_panel = F[top[:n_top]]
        for k in range(n_top):
            ax.plot(self._ts_index(), F_panel[k], lw=1.8, marker="o", ms=3,
                    color=cmap(k % 10), label=f"edge {int(top[k])}")
        # symlog y so sign is preserved while spanning many orders of magnitude
        abs_vals = np.abs(F_panel[F_panel != 0])
        linthresh = float(np.nanpercentile(abs_vals, 5)) if abs_vals.size else 1e-9
        linthresh = max(linthresh, 1e-9)
        ax.set_yscale("symlog", linthresh=linthresh)
        ax.set_xticks(self._ts_index())
        ax.set_xticklabels(self._ts_labels_for_xticks(), rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("timestep")
        ax.set_ylabel("F* (symlog)")
        ax.set_title("Top-10 most-variable edges (symlog y)", loc="left", fontsize=10)
        ax.text(0.99, 0.02, "symlog scale", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=_C["grey"])
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
                                 figsize=(14, max(3.0, 2.8 * n_rows)),
                                 squeeze=False)
        fig.suptitle(self._title(title), fontsize=13, fontweight="bold")
        return fig, axes

    @staticmethod
    def _row_label(ax, text: str) -> None:
        """Left-side row caption that does not collide with histogram bars."""
        ax.set_ylabel(text, fontsize=9, labelpad=8)

    @staticmethod
    def _hist_with_footer(ax, data: np.ndarray, color: str, *, symlog: bool = False,
                          bins: int = 40, ref_lines: tuple = (),
                          linthresh: Optional[float] = None) -> None:
        """Shared histogram-cell renderer with summary footer in the axes.

        When ``symlog`` is True, edges are spaced logarithmically on |data|
        with a linear band of width ``linthresh`` around zero, so a single
        outlier no longer collapses the bulk into one visual bin.
        """
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            ax.text(0.5, 0.5, "n=0", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=_C["grey"])
            _apply_modern_style(ax)
            return

        if symlog:
            abs_finite = np.abs(finite)
            nonzero = abs_finite[abs_finite > 0]
            if nonzero.size == 0:
                edges = np.linspace(-1, 1, bins + 1)
                lt = 1e-9
            else:
                if linthresh is None:
                    lt = max(np.percentile(nonzero, 5), nonzero.max() * 1e-9, 1e-15)
                else:
                    lt = float(linthresh)
                hi = max(nonzero.max(), lt * 10)
                log_hi = np.log10(hi)
                log_lo = np.log10(lt)
                n_log = max(2, bins // 2)
                pos = np.logspace(log_lo, log_hi, n_log + 1)
                # Include the linear band [-lt, 0, lt] as explicit bins so
                # near-zero values don't collapse into the first log bin.
                edges = np.concatenate([-pos[::-1], [-lt, 0.0, lt], pos[1:]])
                edges = np.unique(edges)  # deduplicate ±lt at boundary
            ax.hist(finite, bins=edges, color=color, alpha=0.55,
                    edgecolor=color, linewidth=0.6)
            ax.set_xscale("symlog", linthresh=lt)
            # Limit tick density on symlog (decades only, ≤6 majors)
            from matplotlib.ticker import LogLocator, SymmetricalLogLocator
            ax.xaxis.set_major_locator(
                SymmetricalLogLocator(base=10.0, linthresh=lt, subs=[1.0])
            )
            ax.xaxis.set_minor_locator(
                SymmetricalLogLocator(base=10.0, linthresh=lt,
                                      subs=np.arange(2, 10) * 0.1)
            )
        else:
            ax.hist(finite, bins=bins, color=color, alpha=0.55,
                    edgecolor=color, linewidth=0.6)

        for r in ref_lines:
            ax.axvline(r, color=_C["grey"], lw=0.8, ls="--", alpha=0.7)
        _apply_modern_style(ax)

        footer = (f"n={finite.size}  med={np.median(finite):.2e}  "
                  f"p95={np.percentile(np.abs(finite), 95):.2e}  "
                  f"max={np.max(np.abs(finite)):.2e}")
        ax.text(0.99, 0.97, footer, transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=_C["grey"])
        ax.tick_params(axis="x", labelsize=7)
        # Prune labels to every other major tick if still crowded.
        labels = ax.get_xticklabels()
        if len(labels) > 8:
            for i, lab in enumerate(labels):
                if i % 2:
                    lab.set_visible(False)

    @staticmethod
    def _hist_logabs_signed(ax, data: np.ndarray, *, bins: int = 40,
                            ref_lines_abs: tuple = (),
                            pos_color: str = None, neg_color: str = None) -> None:
        """Histogram of log10|x| with positive/negative values rendered in
        different colours on the same axis — replaces symlog for signed data.

        ``ref_lines_abs`` are positive thresholds (e.g. (1, 3) for z-scores);
        each is drawn as a vertical line at log10(threshold).
        """
        pos_color = pos_color or _C["blue"]
        neg_color = neg_color or _C["red"]
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            ax.text(0.5, 0.5, "n=0", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=_C["grey"])
            _apply_modern_style(ax)
            return

        pos = finite[finite > 0]
        neg = -finite[finite < 0]  # magnitudes only
        nonzero_abs = np.concatenate([pos, neg])
        nonzero_abs = nonzero_abs[nonzero_abs > 0]
        if nonzero_abs.size == 0:
            ax.text(0.5, 0.5, "all zero", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=_C["grey"])
            _apply_modern_style(ax)
            return

        lo = float(np.log10(nonzero_abs.min()))
        hi = float(np.log10(nonzero_abs.max()))
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)

        if pos.size:
            ax.hist(np.log10(pos), bins=edges, color=pos_color, alpha=0.45,
                    edgecolor=pos_color, linewidth=0.6, histtype="stepfilled",
                    label=f"x>0 (n={pos.size})")
        if neg.size:
            ax.hist(np.log10(neg), bins=edges, color=neg_color, alpha=0.45,
                    edgecolor=neg_color, linewidth=0.6, histtype="stepfilled",
                    label=f"x<0 (n={neg.size})")

        for r in ref_lines_abs:
            if r > 0:
                ax.axvline(np.log10(r), color=_C["grey"], lw=0.8, ls="--", alpha=0.7)

        _apply_modern_style(ax)
        n_zero = int(np.sum(finite == 0))
        footer = (f"n={finite.size}  +={pos.size}  −={neg.size}"
                  + (f"  z={n_zero}" if n_zero else "")
                  + f"  max|x|={np.max(np.abs(finite)):.2e}")
        ax.text(0.99, 0.97, footer, transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color=_C["grey"])
        if pos.size or neg.size:
            ax.legend(fontsize=6, frameon=False, loc="upper left")
        ax.tick_params(axis="x", labelsize=7)
        ax.set_xlabel("log₁₀ |x|", fontsize=8)

    def _fig_residual_digest(self, timesteps):
        """SD-T-01: 5 rows × 3 cols of constraint-residual histograms.

        log₁₀|x| with sign overlay (pos = blue, neg = red) for every row.
        """
        ts_triple = self._resolve_timesteps(timesteps)
        fig, axes = self._make_xsection(5, "Constraint-residual digest (SD-T-01)")

        row_specs = [
            ("Type-1 mass-balance Gx",  "mass_balance"),
            ("Type-2 soft-sum ε",       "sum_eq"),
            ("Type-3 soft-ratio ε",     "ratio_eq"),
            ("Type-4 ratio-bound slack", "ratio_bound"),
            ("Type-5 flow-bound slack",  None),
        ]

        for col, t in enumerate(ts_triple):
            for row, (rlabel, key) in enumerate(row_specs):
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
                self._hist_logabs_signed(ax, data, bins=40)
                if col == 0:
                    self._row_label(ax, rlabel)
                if row == 0:
                    ax.set_title(self._col_title(t, ts_triple), fontsize=10, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_zscore_soft(self, timesteps):
        """SD-T-02: 2 rows × 3 cols of ε/σ_c z-scores for soft constraints.

        log₁₀|z| with sign as colour overlay (pos = blue, neg = red). Vertical
        guides at log₁₀(1) and log₁₀(3) mark the ±1σ / ±3σ thresholds.
        """
        ts_triple = self._resolve_timesteps(timesteps)
        fig, axes = self._make_xsection(2, "Soft-slack z-scores (SD-T-02)")

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
                self._hist_logabs_signed(
                    ax, z[np.isfinite(z)], bins=40, ref_lines_abs=(1, 3),
                )
                if col == 0:
                    self._row_label(ax, rlabel)
                if row == 0:
                    ax.set_title(self._col_title(t, ts_triple), fontsize=10, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_solution_vs_prior(self, timesteps):
        """SD-T-03: (a*-a0)/σ_a, (a*-a0)/|a0|, σ_opt/σ_a (cond.).

        Signed rows (z-score, relative change) use log₁₀|x| with sign overlay
        (pos = blue, neg = red). σ_opt/σ_a is unsigned positive, plotted log₁₀.
        """
        ts_triple = self._resolve_timesteps(timesteps)
        n_rows = 3 if self._sigma_opt else 2
        fig, axes = self._make_xsection(n_rows, "Solution vs prior (SD-T-03)")

        a0  = np.asarray(self._const_run.a0, dtype=float)
        sa  = np.asarray(self._const_run.sigma_a, dtype=float)
        sa_safe = np.where(sa > 0, sa, np.inf)
        a0_safe = np.where(np.abs(a0) > 0, np.abs(a0), np.inf)

        for col, t in enumerate(ts_triple):
            F = self._results[t].F_opt
            # Row 1: z-score (signed) → log10|z| with sign overlay
            z = (F - a0) / sa_safe
            self._hist_logabs_signed(axes[0, col], z, bins=40,
                                      ref_lines_abs=(1, 3))
            # Row 2: relative change (signed) → log10|rel| with sign overlay
            rel = (F - a0) / a0_safe
            self._hist_logabs_signed(axes[1, col], rel, bins=40,
                                      ref_lines_abs=(1.0,))
            # Row 3: sigma_opt / sigma_a (positive, unsigned)
            if self._sigma_opt is not None and t in self._sigma_opt:
                so = np.asarray(self._sigma_opt[t], dtype=float)
                ratio = np.where(sa > 0, so / sa_safe, np.nan)
                self._hist_logabs_signed(axes[2, col], ratio, bins=40,
                                          ref_lines_abs=(1.0,))

            if col == 0:
                self._row_label(axes[0, col], "(a* − a₀)/σ_a")
                self._row_label(axes[1, col], "(a* − a₀)/|a₀|")
                if n_rows == 3:
                    self._row_label(axes[2, col], "σ_opt/σ_a")
            axes[0, col].set_title(self._col_title(t, ts_triple),
                                   fontsize=10, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_binding_rate(self, timesteps):
        """SD-T-04: 2 rows × 3 cols of binding-rate distributions.

        log₁₀|expression| with sign overlay (neg = slack/good, pos = violation).
        """
        ts_triple = self._resolve_timesteps(timesteps)
        fig, axes = self._make_xsection(2, "Per-type binding rate (SD-T-04)")

        for col, t in enumerate(ts_triple):
            # Row 1: Type-4 ratio-bound expressions (Gx - c sign convention:
            # negative ⇒ slack; positive ⇒ violation)
            idx = self._row_info["ratio_bound"]
            v_rb = self._results[t].residuals[idx] if idx.size else np.zeros(0)
            self._hist_logabs_signed(axes[0, col], v_rb, bins=40)

            # Row 2: Type-5 flow-bound expressions (F - u and l - F).
            # Negative = slack (good); positive = violation.
            s_u, s_l = self._flow_bound_slack(t)
            v_fb = np.concatenate([-s_u, -s_l])
            self._hist_logabs_signed(axes[1, col], v_fb, bins=40)

            if col == 0:
                self._row_label(axes[0, col], "Type-4: a_i − r·a_j")
                self._row_label(axes[1, col], "Type-5: F − u, l − F")
            axes[0, col].set_title(self._col_title(t, ts_triple),
                                   fontsize=10, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_solution_magnitude(self, timesteps):
        """SD-T-05: distribution of a — log10|a*|, symlog a*, and log10|a₀(t)|."""
        ts_triple = self._resolve_timesteps(timesteps)
        fig, axes = self._make_xsection(3, "Distribution of a (SD-T-05)")

        # Build warm-start prior path: a0(t0)=const_run.a0; a0(t>0)=F*(t-1)
        prior_for: dict = {}
        prev = np.asarray(self._const_run.a0, dtype=float)
        for t in self._timesteps:
            prior_for[t] = prev
            prev = self._results[t].F_opt

        # Shared bin edges for row 0 (log10|a*|) across the three columns
        F_all = np.concatenate([self._results[t].F_opt for t in ts_triple])
        F_pos = np.abs(F_all)
        F_pos = F_pos[F_pos > 0]
        if F_pos.size:
            lo, hi = np.log10(F_pos.min()), np.log10(F_pos.max())
        else:
            lo, hi = -1.0, 1.0
        bin_edges_r0 = np.linspace(lo, hi, 41)

        # Shared bin edges for row 2 (log10|a₀|) across the three columns
        a0_all = np.concatenate([np.abs(prior_for[t]) for t in ts_triple])
        a0_pos = a0_all[a0_all > 0]
        if a0_pos.size:
            lo0, hi0 = np.log10(a0_pos.min()), np.log10(a0_pos.max())
        else:
            lo0, hi0 = -1.0, 1.0
        bin_edges_r2 = np.linspace(lo0, hi0, 41)

        for col, t in enumerate(ts_triple):
            # ---- Row 0: log10|a*| (original) ----
            ax = axes[0, col]
            F = np.abs(self._results[t].F_opt)
            F = F[F > 0]
            if F.size:
                ax.hist(np.log10(F), bins=bin_edges_r0, color=_C["blue"],
                        alpha=0.55, edgecolor=_C["blue"], linewidth=0.6)
                for q, lbl, col_q in [(50, "p50", _C["navy"]),
                                       (95, "p95", _C["orange"]),
                                       (99, "p99", _C["red"])]:
                    val = float(np.percentile(np.log10(F), q))
                    ax.axvline(val, color=col_q, lw=1.2, ls="--", alpha=0.8)
                    ax.text(val, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1.0,
                            lbl, rotation=90, fontsize=7, va="top", ha="right", color=col_q)
            _apply_modern_style(ax)
            ax.set_xlabel("log₁₀ |a*|  (log₁₀-transformed)")
            if col == 0:
                self._row_label(ax, "log₁₀ |a*|")
            ax.set_title(self._col_title(t, ts_triple), fontsize=10, fontweight="bold")

            # ---- Row 1: log₁₀|a*| with sign overlay (pos = blue, neg = red) ----
            ax = axes[1, col]
            F_signed = self._results[t].F_opt
            self._hist_logabs_signed(ax, F_signed, bins=40)
            if col == 0:
                self._row_label(ax, "a* (log₁₀|x| signed)")

            # ---- Row 2: log10|a₀(t)| ----
            ax = axes[2, col]
            a0_t = np.abs(prior_for[t])
            a0_t = a0_t[a0_t > 0]
            if a0_t.size:
                ax.hist(np.log10(a0_t), bins=bin_edges_r2, color=_C["orange"],
                        alpha=0.55, edgecolor=_C["orange"], linewidth=0.6)
            _apply_modern_style(ax)
            ax.set_xlabel("log₁₀ |a₀(t)|  (log₁₀-transformed)")
            if col == 0:
                self._row_label(ax, "a₀(t) — log₁₀|a₀|")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_cstar_vs_c(self, timesteps):
        """SD-T-06: c*-c and (c*-c)/(|c|+δ) histograms on soft rows.

        log₁₀|x| with sign overlay (pos = blue, neg = red). For the relative
        row, log₁₀(1) is marked to show "100% of c" departure.
        """
        ts_triple = self._resolve_timesteps(timesteps)
        fig, axes = self._make_xsection(2, "c* vs c on soft rows (SD-T-06)")

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

            self._hist_logabs_signed(axes[0, col], eps, bins=40)
            self._hist_logabs_signed(axes[1, col], rel_eps, bins=40,
                                      ref_lines_abs=(1.0,))

            if col == 0:
                self._row_label(axes[0, col], "c* − c")
                self._row_label(axes[1, col], "(c* − c) / (|c| + δ)")
            axes[0, col].set_title(self._col_title(t, ts_triple),
                                   fontsize=10, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_rocket(self, timesteps):
        """SD-T-07: log–log scatter 'rocket' plots — warm-start prior throughout.

        All three rows compare the warm-start prior `a*(t−1)` (propagated through
        G where needed) against the current-timestep solution.

        Row 1: log10|G·a*(t−1)| (x) vs log10|c*| (y) on sum_eq rows.
               Colour = |ε|/σ_c (z-score); viridis LogNorm.
        Row 2: log10|ε₀| (x) vs log10|ε_opt| (y) on soft (free-cone) rows,
               where ε₀ = G_free · a*(t−1) − c_free(t).
               Colour = |ε_opt|/σ_c (z-score); viridis LogNorm.
        Row 3: log10|a*(t−1)| (x) vs log10|a*(t)| (y) over all edges.
               Colour = |a*−a₀|/σ_a; viridis LogNorm.
        Diagonal y=x = perfect agreement.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize

        ts_triple = self._resolve_timesteps(timesteps)
        fig, axes = self._make_xsection(3, "Rocket scatter (SD-T-07)")
        cmap = plt.get_cmap("viridis")

        # Row 1: sum_eq only
        se_idx = self._row_info["sum_eq"]

        # Build warm-start prior path: a0(t0)=const_run.a0; a0(t>0)=F*(t-1)
        prior_for: dict = {}
        prev = np.asarray(self._const_run.a0, dtype=float)
        for t in self._timesteps:
            prior_for[t] = prev
            prev = self._results[t].F_opt

        sa = np.asarray(self._const_run.sigma_a, dtype=float)
        sa_safe = np.where(sa > 0, sa, np.inf)

        def _logabs(x: np.ndarray) -> np.ndarray:
            v = np.abs(np.asarray(x, dtype=float))
            v = v[v > 0]
            return np.log10(v) if v.size else np.array([0.0])

        # Row 1 x-axis is c_prior(t) = G_sum_eq · a*(t-1); shared limits across cols
        G_se = self._const_run.G[se_idx, :] if se_idx.size else None
        if se_idx.size and G_se is not None:
            cprior_all = np.concatenate(
                [np.asarray(G_se @ prior_for[t]) for t in ts_triple])
            cstars_all = np.concatenate(
                [self._results[t].c_star[se_idx] for t in ts_triple])
            all_r1 = np.concatenate([_logabs(cprior_all), _logabs(cstars_all)])
            r1_lo = float(all_r1.min()) if all_r1.size else -1.0
            r1_hi = float(all_r1.max()) if all_r1.size else 1.0
        else:
            r1_lo, r1_hi = -1.0, 1.0

        a0_all = np.concatenate([np.abs(prior_for[t]) for t in ts_triple])
        ax_all = np.concatenate([np.abs(self._results[t].F_opt) for t in ts_triple])
        all_r2 = np.concatenate([_logabs(a0_all), _logabs(ax_all)])
        r2_lo = float(all_r2.min()) if all_r2.size else -1.0
        r2_hi = float(all_r2.max()) if all_r2.size else 1.0

        # Colour normalisation for row 1: |ε|/σ_c on sum_eq, shared across cols
        if se_idx.size:
            z_all = np.concatenate([
                np.abs(self._eps_for_type(t, "sum_eq")) /
                np.where(self._sigma_c_for_type(t, "sum_eq") > 0,
                         self._sigma_c_for_type(t, "sum_eq"), np.inf)
                for t in ts_triple
            ])
            z_all = z_all[np.isfinite(z_all) & (z_all > 0)]
            znorm = LogNorm(vmin=max(z_all.min(), 1e-6), vmax=z_all.max()) if z_all.size else Normalize(0, 1)
        else:
            znorm = Normalize(0, 1)

        dpa_all = np.concatenate([
            np.abs(self._results[t].F_opt - prior_for[t]) / sa_safe
            for t in ts_triple
        ])
        dpa_all = dpa_all[np.isfinite(dpa_all) & (dpa_all > 0)]
        anorm = LogNorm(vmin=max(dpa_all.min(), 1e-6), vmax=dpa_all.max()) if dpa_all.size else Normalize(0, 1)

        # ε₀ vs ε_opt limits: free-cone rows only
        free_idx = self._row_info["free"]

        def _eps0_for_t(t):
            """Pre-optimisation soft-row residual: G_free @ a*(t-1) - c_free(t)."""
            if not free_idx.size:
                return np.zeros(0)
            G_free = self._const_run.G[free_idx, :]
            c_free = np.asarray(self._const_run.timestep_data[t]["c"])[free_idx]
            return G_free @ prior_for[t] - c_free

        def _z_free_for_t(t):
            """|ε_opt|/σ_c on free-cone rows (LogNorm colour for row 2)."""
            if not free_idx.size:
                return np.zeros(0)
            eps_opt = np.asarray(self._results[t].eps_opt, dtype=float)
            sig_all = np.asarray(self._const_run.timestep_data[t]["sigma_c"], dtype=float)
            sig_free = sig_all[free_idx][:len(eps_opt)]
            sig_safe = np.where(sig_free > 0, sig_free, np.inf)
            return np.abs(eps_opt) / sig_safe

        eps0_all_vals = np.concatenate(
            [np.abs(_eps0_for_t(t)) for t in ts_triple]
        )
        eps0_all_vals = eps0_all_vals[eps0_all_vals > 0]
        epsopt_all_vals = np.concatenate(
            [np.abs(self._results[t].eps_opt) for t in ts_triple]
        )
        epsopt_all_vals = epsopt_all_vals[epsopt_all_vals > 0]

        if eps0_all_vals.size and epsopt_all_vals.size:
            r2_eps_lo = float(min(np.log10(eps0_all_vals.min()),
                                  np.log10(epsopt_all_vals.min())))
            r2_eps_hi = float(max(np.log10(eps0_all_vals.max()),
                                  np.log10(epsopt_all_vals.max())))
        else:
            r2_eps_lo, r2_eps_hi = -9.0, 0.0

        # Colour normalisation for row 2: |ε_opt|/σ_c on free-cone rows
        z_free_all = np.concatenate([_z_free_for_t(t) for t in ts_triple])
        z_free_all = z_free_all[np.isfinite(z_free_all) & (z_free_all > 0)]
        z_free_norm = (LogNorm(vmin=max(z_free_all.min(), 1e-6), vmax=z_free_all.max())
                       if z_free_all.size else Normalize(0, 1))

        for col, t in enumerate(ts_triple):
            # ---- Row 1: G·a*(t-1) vs c* on sum_eq rows, coloured by |ε|/σ_c ----
            ax = axes[0, col]
            if se_idx.size and G_se is not None:
                c_prior = np.abs(np.asarray(G_se @ prior_for[t], dtype=float))
                cstar   = np.abs(self._results[t].c_star[se_idx])
                eps     = self._eps_for_type(t, "sum_eq")
                sig     = self._sigma_c_for_type(t, "sum_eq")
                z       = np.abs(eps) / np.where(sig > 0, sig, np.inf)
                m       = (c_prior > 0) & (cstar > 0) & np.isfinite(z)
                if m.any():
                    sc = ax.scatter(np.log10(c_prior[m]), np.log10(cstar[m]),
                                    c=z[m], cmap=cmap, norm=znorm,
                                    s=10, alpha=0.7, edgecolors="none")
                    if col == 2:  # colorbar on last column
                        fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05,
                                     label="|ε|/σ_c")
            lo, hi = r1_lo, r1_hi
            ax.plot([lo, hi], [lo, hi], color=_C["grey"], lw=0.9, ls="--")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("log₁₀ |G·a*(t−1)|", fontsize=8)
            _apply_modern_style(ax)
            if col == 0:
                self._row_label(ax, "log₁₀ |c*|")
            ax.set_title(self._col_title(t, ts_triple),
                         fontsize=10, fontweight="bold")

            # ---- Row 2: ε₀ vs ε_opt on free-cone rows, coloured by |ε_opt|/σ_c ----
            ax = axes[1, col]
            eps0_t   = np.abs(_eps0_for_t(t))
            epsopt_t = np.abs(self._results[t].eps_opt)
            zfree_t  = _z_free_for_t(t)
            # Match lengths (eps_opt is already free-cone order; eps0 built from free_idx)
            n_min = min(len(eps0_t), len(epsopt_t), len(zfree_t))
            eps0_t   = eps0_t[:n_min]
            epsopt_t = epsopt_t[:n_min]
            zfree_t  = zfree_t[:n_min]
            m2 = (eps0_t > 0) & (epsopt_t > 0) & np.isfinite(zfree_t)
            if m2.any():
                sc3 = ax.scatter(np.log10(eps0_t[m2]), np.log10(epsopt_t[m2]),
                                 c=zfree_t[m2], cmap=cmap, norm=z_free_norm,
                                 s=8, alpha=0.6, edgecolors="none")
                if col == 2:
                    fig.colorbar(sc3, ax=ax, pad=0.02, fraction=0.05,
                                 label="|ε_opt|/σ_c")
            lo2, hi2 = r2_eps_lo, r2_eps_hi
            ax.plot([lo2, hi2], [lo2, hi2], color=_C["grey"], lw=0.9, ls="--")
            ax.set_xlim(lo2, hi2); ax.set_ylim(lo2, hi2)
            ax.set_xlabel("log₁₀ |ε₀ = G·a*(t−1) − c|", fontsize=8)
            _apply_modern_style(ax)
            if col == 0:
                self._row_label(ax, "log₁₀ |ε_opt|")

            # ---- Row 3: a₀(t) vs a* over all edges, coloured by |a*−a₀|/σ_a ----
            ax = axes[2, col]
            a0    = np.abs(prior_for[t])
            astar = np.abs(self._results[t].F_opt)
            dp    = np.abs(self._results[t].F_opt - prior_for[t]) / sa_safe
            m = (a0 > 0) & (astar > 0) & np.isfinite(dp)
            if m.any():
                sc2 = ax.scatter(np.log10(a0[m]), np.log10(astar[m]),
                                 c=dp[m], cmap=cmap, norm=anorm,
                                 s=8, alpha=0.5, edgecolors="none")
                if col == 2:
                    fig.colorbar(sc2, ax=ax, pad=0.02, fraction=0.05,
                                 label="|a*−a₀|/σ_a")
            lo, hi = r2_lo, r2_hi
            ax.plot([lo, hi], [lo, hi], color=_C["grey"], lw=0.9, ls="--")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_xlabel("log₁₀ |a₀(t)|  (log₁₀-transformed)", fontsize=8)
            _apply_modern_style(ax)
            if col == 0:
                self._row_label(ax, "log₁₀ |a*|")

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_time_hued(self, hue=None):
        """SD-R-07: histograms of c*(soft) and |ε_opt| hued by timestep.

        Two stacked panels — the |a*| distribution panel was removed because
        SD-T-05 row 0 already covers it at the three-timestep cross-section
        granularity. Each timestep is rendered as a step-histogram with
        viridis-coloured outline so the temporal gradient is visible.
        Not hueable across runs.
        """
        import matplotlib.pyplot as plt

        del hue  # Not used: per-timestep hue is built-in.
        fig, axes = plt.subplots(2, 1, figsize=(13, 8))
        fig.suptitle(self._title("Time-hued c* and |ε_opt| (SD-R-07)"),
                     fontsize=13, fontweight="bold")

        cmap = plt.get_cmap("viridis")
        T = len(self._timesteps)

        # --- Panel A: c* on soft rows
        ax = axes[0]
        soft_idx = np.concatenate([self._row_info["sum_eq"],
                                   self._row_info["ratio_eq"]])
        if soft_idx.size:
            C_all = np.concatenate(
                [np.abs(self._results[t].c_star[soft_idx])
                 for t in self._timesteps]
            )
            C_all = C_all[C_all > 0]
            if C_all.size:
                lo, hi = np.log10(C_all.min()), np.log10(C_all.max())
            else:
                lo, hi = -1.0, 1.0
            bin_edges = np.linspace(lo, hi, 41)
            for i, t in enumerate(self._timesteps):
                cstar = np.abs(self._results[t].c_star[soft_idx])
                cstar = cstar[cstar > 0]
                if cstar.size == 0:
                    continue
                color = cmap(i / max(T - 1, 1))
                ax.hist(np.log10(cstar), bins=bin_edges, histtype="step",
                        color=color, lw=1.1, alpha=0.85)
            ax.set_xlabel("log₁₀ |c*| on soft rows")
        else:
            ax.text(0.5, 0.5, "no soft rows", transform=ax.transAxes,
                    ha="center", va="center", color=_C["grey"])
        ax.set_ylabel("count")
        ax.set_title("Per-timestep c* on soft rows (hue = time)",
                     loc="left", fontsize=10)
        _apply_modern_style(ax)

        # --- Panel B: log10|ε_opt| over all soft-row slacks
        ax = axes[1]
        eps_all = np.concatenate(
            [np.abs(self._results[t].eps_opt) for t in self._timesteps]
        )
        eps_pos = eps_all[eps_all > 0]
        if eps_pos.size:
            lo, hi = np.log10(eps_pos.min()), np.log10(eps_pos.max())
        else:
            lo, hi = -1.0, 1.0
        bin_edges = np.linspace(lo, hi, 41)
        for i, t in enumerate(self._timesteps):
            eps_t = np.abs(self._results[t].eps_opt)
            eps_t = eps_t[eps_t > 0]
            if eps_t.size == 0:
                continue
            color = cmap(i / max(T - 1, 1))
            ax.hist(np.log10(eps_t), bins=bin_edges, histtype="step",
                    color=color, lw=1.1, alpha=0.85)
        ax.set_xlabel("log₁₀ |ε_opt|  (log₁₀-transformed)")
        ax.set_ylabel("count")
        ax.set_title("Per-timestep |ε_opt| distribution (hue = time)",
                     loc="left", fontsize=10)
        _apply_modern_style(ax)

        fig.tight_layout(rect=[0, 0, 0.93, 0.95])

        # Shared colorbar — add AFTER tight_layout to keep matplotlib quiet.
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=max(T - 1, 1)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.tolist(), pad=0.01, fraction=0.025)
        n_ticks = min(6, T)
        tick_idx = np.linspace(0, T - 1, n_ticks, dtype=int)
        cbar.set_ticks(tick_idx)
        cbar.set_ticklabels([str(self._timesteps[i]) for i in tick_idx])
        cbar.set_label("timestep")
        return fig

    @staticmethod
    def _draw_eps_heatmap_pair(fig, ax_abs, ax_z, eps_mat, z_mat, ts_labels, row_labels):
        """Render one pair of heatmaps (log₁₀|ε| left, log₁₀(|ε|/σ_c) right)."""
        T = eps_mat.shape[1]
        n_rows = eps_mat.shape[0]
        kw = dict(aspect="auto", interpolation="nearest")

        with np.errstate(divide="ignore", invalid="ignore"):
            log_mat = np.where(eps_mat > 0, np.log10(eps_mat), np.nan)
            log_z   = np.where(z_mat   > 0, np.log10(np.maximum(z_mat, 1e-12)), np.nan)

        vmin_a = np.nanmin(log_mat) if np.any(np.isfinite(log_mat)) else -9
        vmax_a = np.nanmax(log_mat) if np.any(np.isfinite(log_mat)) else 0
        im1 = ax_abs.imshow(log_mat, cmap="YlOrRd", vmin=vmin_a, vmax=vmax_a, **kw)
        fig.colorbar(im1, ax=ax_abs, pad=0.01, fraction=0.04, label="log₁₀|ε|  (log₁₀-transformed)")
        ax_abs.set_xticks(range(T))
        ax_abs.set_xticklabels(ts_labels, rotation=45, ha="right", fontsize=7)
        ax_abs.set_yticks(range(n_rows))
        ax_abs.set_yticklabels(row_labels, fontsize=8)

        vmin_lz = np.nanmin(log_z) if np.any(np.isfinite(log_z)) else -3
        vmax_lz = np.nanmax(log_z) if np.any(np.isfinite(log_z)) else 3
        im2 = ax_z.imshow(log_z, cmap="YlOrRd", vmin=vmin_lz, vmax=vmax_lz, **kw)
        fig.colorbar(im2, ax=ax_z, pad=0.01, fraction=0.04, label="log₁₀(|ε|/σ_c)  (log₁₀-transformed)")
        ax_z.set_xticks(range(T))
        ax_z.set_xticklabels(ts_labels, rotation=45, ha="right", fontsize=7)
        ax_z.set_yticks(range(n_rows))
        ax_z.set_yticklabels(row_labels, fontsize=8)

    def _fig_eps_heatmap_ratio(self):
        """SD-T-08a: ratio_eq realized-ratio heatmap — top-50 by log10-range across t.

        Each row shows ``log10(realized_ratio)`` where realized_ratio =
        ``a*[edge_i] / a*[edge_j]`` per ratio_eq constraint row (the G row reads
        ``1·a[edge_i] − r·a[edge_j] = 0`` so this is the realized analogue of r).
        Row selection: top-50 by ``log10(max) − log10(min)`` of realized_ratio
        across all timesteps (same variability metric as SD-R-06).
        """
        import matplotlib.pyplot as plt

        K = 50
        T = len(self._timesteps)
        ts_labels = [str(t) for t in self._timesteps]
        rm = self._const_run.row_meta
        ri = self._row_info

        re_idx = ri["ratio_eq"]
        if re_idx.size == 0:
            fig = plt.figure(figsize=(8, 3))
            fig.suptitle(self._title("ratio_eq realized-ratio heatmap (SD-T-08a) — no rows"),
                         fontsize=12)
            return fig

        # Extract (edge_i, edge_j) per ratio_eq row from G:
        #   row coefficients are [+1 at edge_i, -r at edge_j] by construction.
        G_re = self._const_run.G[re_idx, :].tocsr()
        n_re = G_re.shape[0]
        edge_i_arr = np.full(n_re, -1, dtype=np.int64)
        edge_j_arr = np.full(n_re, -1, dtype=np.int64)
        for i in range(n_re):
            r = G_re.getrow(i)
            cols, vals = r.indices, r.data
            if cols.size == 0:
                continue
            pos = cols[vals > 0]
            neg = cols[vals < 0]
            if pos.size and neg.size:
                edge_i_arr[i] = int(pos[0])
                edge_j_arr[i] = int(neg[0])

        valid = (edge_i_arr >= 0) & (edge_j_arr >= 0)

        # Per-row realized ratio across all timesteps: r_realized[row, t] = a*[edge_i]/a*[edge_j]
        ratio_mat_all = np.full((n_re, T), np.nan)
        for col_i, t in enumerate(self._timesteps):
            F_t = self._results[t].F_opt
            with np.errstate(divide="ignore", invalid="ignore"):
                num = np.where(valid, F_t[edge_i_arr.clip(min=0)], np.nan)
                den = np.where(valid, F_t[edge_j_arr.clip(min=0)], np.nan)
                ratio_mat_all[:, col_i] = num / den

        # Rank by log10-range of the realized ratio across timesteps
        abs_ratio = np.abs(ratio_mat_all)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_abs = np.where(abs_ratio > 0, np.log10(abs_ratio), np.nan)
        rng = np.nanmax(log_abs, axis=1) - np.nanmin(log_abs, axis=1)
        # Rows with no finite values get -inf and end up at the bottom of the sort
        rng = np.where(np.isfinite(rng), rng, -np.inf)
        order = np.argsort(-rng)[:K]
        n_top = int(np.sum(rng[order] > -np.inf))
        order = order[:n_top] if n_top else order[:0]

        row_labels = [str(rm.iloc[re_idx[i]]["constraint_id"]) for i in order]
        ratio_mat = ratio_mat_all[order, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            log_mat = np.where(np.abs(ratio_mat) > 0,
                               np.log10(np.abs(ratio_mat)), np.nan)

        fig_h = max(6.0, 0.28 * len(order) + 2.0)
        fig = plt.figure(figsize=(11, fig_h))
        fig.suptitle(self._title(f"ratio_eq realized-ratio heatmap — top-{len(order)} (SD-T-08a)"),
                     fontsize=13, fontweight="bold")

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 1, figure=fig, top=0.88, bottom=0.15, left=0.28, right=0.92)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title("log₁₀ |a*[edge_i] / a*[edge_j]|", fontsize=10)
        kw = dict(aspect="auto", interpolation="nearest")
        if log_mat.size:
            vmin = np.nanmin(log_mat) if np.any(np.isfinite(log_mat)) else -3
            vmax = np.nanmax(log_mat) if np.any(np.isfinite(log_mat)) else 3
            im = ax.imshow(log_mat, cmap="YlOrRd", vmin=vmin, vmax=vmax, **kw)
            fig.colorbar(im, ax=ax, pad=0.01, fraction=0.04,
                         label="log₁₀ |realized ratio|")
        ax.set_xticks(range(T))
        ax.set_xticklabels(ts_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        return fig

    def _fig_eps_heatmap_sum(self):
        """SD-T-08b: sum_eq ε heatmap — top-50 nodes by log10-range of |ε| across t."""
        import matplotlib.pyplot as plt
        import re as _re

        K = 50
        T = len(self._timesteps)
        ts_labels = [str(t) for t in self._timesteps]
        rm = self._const_run.row_meta
        ri = self._row_info

        se_cids_all = rm.iloc[ri["sum_eq"]]["constraint_id"].to_numpy()
        nodes_all   = np.array([_re.sub(r"_t\d{4}$", "", c) for c in se_cids_all])
        unique_nodes = np.unique(nodes_all)

        # Rank nodes by log10-range of mean|ε| across ALL timesteps
        node_eps_t: dict = {}
        for node in unique_nodes:
            node_mask = nodes_all == node
            vals = []
            for t in self._timesteps:
                eps_t = np.abs(self._eps_for_type(t, "sum_eq"))
                vals.append(float(eps_t[node_mask].mean()) if node_mask.any() else 0.0)
            node_eps_t[node] = np.asarray(vals, dtype=float)

        def _log10_range(arr: np.ndarray) -> float:
            pos = arr[arr > 0]
            if pos.size < 2:
                return -np.inf
            return float(np.log10(pos.max()) - np.log10(pos.min()))

        node_var = {n: _log10_range(node_eps_t[n]) for n in unique_nodes}
        se_sorted_all = sorted(node_var, key=node_var.get, reverse=True)
        se_sorted = [n for n in se_sorted_all if node_var[n] > -np.inf][:K]

        # Decode node keys → human labels
        try:
            from pysuto.acc import AccountingStructure
            _acc = AccountingStructure.load(self._const_run.acc_id)
            def _get_label(node_key: str) -> str:
                m = _re.search(r"node(\d+)$", node_key)
                if m is None:
                    return node_key
                sector, entity, region = _acc.decode_node(int(m.group(1)))
                return f"{sector}|{region}"
        except Exception:
            def _get_label(node_key: str) -> str:
                return node_key

        se_labels = [_get_label(n) for n in se_sorted]
        n_se = len(se_sorted)

        eps_mat = np.full((n_se, T), np.nan)
        z_mat   = np.full((n_se, T), np.nan)
        for col_i, t in enumerate(self._timesteps):
            eps_t   = np.abs(self._eps_for_type(t, "sum_eq"))
            sigma_t = self._sigma_c_for_type(t, "sum_eq")
            cids_t  = rm.iloc[ri["sum_eq"]]["constraint_id"].to_numpy()
            nds_t   = np.array([_re.sub(r"_t\d{4}$", "", c) for c in cids_t])
            for row_i, node in enumerate(se_sorted):
                mask = nds_t == node
                if not mask.any():
                    continue
                ev = eps_t[mask]
                sv = sigma_t[mask]
                eps_mat[row_i, col_i] = ev.mean()
                safe_sv = np.where(sv > 0, sv, np.inf)
                z_mat[row_i, col_i]   = (ev / safe_sv).mean()

        fig_h = max(6.0, 0.28 * n_se + 2.0)
        fig = plt.figure(figsize=(18, fig_h))
        fig.suptitle(self._title(f"sum_eq ε heatmap — top-{n_se} nodes (SD-T-08b)"),
                     fontsize=13, fontweight="bold")

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, figure=fig, wspace=0.35,
                      top=0.88, bottom=0.15, left=0.28, right=0.96)
        ax_abs = fig.add_subplot(gs[0, 0])
        ax_z   = fig.add_subplot(gs[0, 1])
        ax_abs.set_title("log₁₀|ε|", fontsize=10)
        ax_z.set_title("log₁₀(|ε|/σ_c)", fontsize=10)
        self._draw_eps_heatmap_pair(fig, ax_abs, ax_z, eps_mat, z_mat, ts_labels, se_labels)
        return fig

    # ------------------------------------------------------------------
    # SD-T-09 — |a* − a₀| distribution by edge metadata
    #   2×2 grid per group_col: rows = (first_t, last_t), cols = (Mass, CC).
    #   a₀ is the stage-3 prior (const_run.a0), fixed across both rows.
    # ------------------------------------------------------------------

    _MINING_REGION_HIGHLIGHT = ("ROW", "Australia", "Chile", "China", "Peru", "Congo, D.R.")
    _SD_T_09_VALUATIONS = ("Mass", "CC")

    def _edge_meta(self) -> "pd.DataFrame | None":  # type: ignore[name-defined]
        """Cached AccountingStructure.edge_meta. Returns None if the underlying
        AccountingStructure cannot be loaded (e.g. toy fixtures).
        """
        if hasattr(self, "_edge_meta_cache"):
            return self._edge_meta_cache
        try:
            from pysuto.acc import AccountingStructure
            acc = AccountingStructure.load(self._const_run.acc_id)
            self._edge_meta_cache = acc.edge_meta.reset_index(drop=True)
        except Exception:
            self._edge_meta_cache = None
        return self._edge_meta_cache

    def _fig_prior_departure_box(self, group_col: str, title_suffix: str,
                                  *, highlight: tuple = (),
                                  fig_id: str = ""):
        """SD-T-09 figure: 2×2 grid (year × valuation) boxplot of
        log₁₀|a*(Y) − a₀(stage-3)| grouped by ``group_col``.

        Categories share a global x-axis order derived from the pooled median
        across all four cells, sorted descending. Highlighted categories use
        distinct tab10 face colours; others render grey at low alpha.
        """
        import matplotlib.pyplot as plt

        em = self._edge_meta()
        if em is None:
            fig = plt.figure(figsize=(8, 3))
            fig.suptitle(self._title(f"{fig_id}: edge_index unavailable (acc data missing)"),
                         fontsize=12)
            return fig
        if group_col not in em.columns:
            fig = plt.figure(figsize=(8, 3))
            fig.suptitle(self._title(f"{fig_id}: column '{group_col}' missing from edge_meta"),
                         fontsize=12)
            return fig
        if not self._timesteps:
            fig = plt.figure(figsize=(8, 3))
            fig.suptitle(self._title(f"{fig_id}: no timesteps in run"),
                         fontsize=12)
            return fig

        years = [self._timesteps[0]]
        if len(self._timesteps) > 1:
            years.append(self._timesteps[-1])
        valuations = list(self._SD_T_09_VALUATIONS)

        a0 = np.asarray(self._const_run.a0, dtype=float)
        em_time = em["time"].astype(str).to_numpy()
        em_val  = em["valuation"].astype(str).to_numpy()
        em_lbl  = em[group_col].astype(str).to_numpy()

        # Per-cell (Y, V) bucket: dict[(Y, V)] -> (labels_arr, log_delta_arr)
        cells: dict = {}
        for Y in years:
            F = np.asarray(self._results[Y].F_opt, dtype=float)
            delta_all = np.abs(F - a0)
            for V in valuations:
                mask = (em_time == str(Y)) & (em_val == V)
                d = delta_all[mask]
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_d = np.where(d > 0, np.log10(d), np.nan)
                lbl = em_lbl[mask]
                finite = np.isfinite(log_d)
                cells[(Y, V)] = (lbl[finite], log_d[finite])

        # Pool everything to build the global category order
        pool_lbl = np.concatenate([c[0] for c in cells.values()]) \
            if any(c[0].size for c in cells.values()) else np.array([], dtype=str)
        pool_log = np.concatenate([c[1] for c in cells.values()]) \
            if any(c[1].size for c in cells.values()) else np.array([], dtype=float)

        if pool_log.size == 0:
            fig = plt.figure(figsize=(8, 3))
            fig.suptitle(self._title(f"{fig_id}: no finite |a*−a₀| samples"),
                         fontsize=12)
            return fig

        cats_pool, inv_pool = np.unique(pool_lbl, return_inverse=True)
        medians_pool = np.array([
            np.median(pool_log[inv_pool == i]) if np.any(inv_pool == i) else -np.inf
            for i in range(len(cats_pool))
        ])
        order = np.argsort(-medians_pool)
        cats_sorted = cats_pool[order]

        # Colour assignment per category (highlight or grey)
        tab10 = plt.get_cmap("tab10").colors
        if highlight:
            highlight_set = set(highlight)
            highlight_color = {h: tab10[k % 10] for k, h in enumerate(highlight)}
            face_colors, edge_colors, alphas = [], [], []
            for cat in cats_sorted:
                if cat in highlight_set:
                    face_colors.append(highlight_color[cat])
                    edge_colors.append(highlight_color[cat])
                    alphas.append(0.7)
                else:
                    face_colors.append(_C["grey"])
                    edge_colors.append(_C["grey"])
                    alphas.append(0.25)
        else:
            face_colors = [tab10[i % 10] for i in range(len(cats_sorted))]
            edge_colors = face_colors
            alphas = [0.7] * len(cats_sorted)

        # Figure layout — sharex (categorical) across all cells; sharey per column
        n_rows = len(years)
        n_cols = len(valuations)
        fig_w = max(10.0, 0.3 * len(cats_sorted) + 4.0)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(min(fig_w, 22), 3.2 * n_rows + 1.0),
            sharex=True, sharey="col", squeeze=False,
        )
        fig.suptitle(self._title(f"|a* − a₀| by {title_suffix} ({fig_id})"),
                     fontsize=13, fontweight="bold")

        for r, Y in enumerate(years):
            for c, V in enumerate(valuations):
                ax = axes[r, c]
                lbl_cell, log_cell = cells[(Y, V)]
                if log_cell.size == 0:
                    ax.text(0.5, 0.5, "n=0", transform=ax.transAxes,
                            ha="center", va="center", fontsize=9, color=_C["grey"])
                else:
                    # Bucket cell data into the global category order
                    per_cat = [log_cell[lbl_cell == cat] for cat in cats_sorted]
                    positions = [i for i, v in enumerate(per_cat) if v.size]
                    data = [v for v in per_cat if v.size]
                    bp = ax.boxplot(
                        data, positions=positions,
                        widths=0.6, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="black", lw=1.2),
                    )
                    face_used = [face_colors[i] for i in positions]
                    edge_used = [edge_colors[i] for i in positions]
                    alpha_used = [alphas[i] for i in positions]
                    for patch, face, edge, a in zip(
                        bp["boxes"], face_used, edge_used, alpha_used
                    ):
                        patch.set_facecolor(face)
                        patch.set_edgecolor(edge)
                        patch.set_alpha(a)
                    for whisker_pair in zip(bp["whiskers"][0::2], bp["whiskers"][1::2]):
                        for w, edge in zip(whisker_pair, edge_used):
                            w.set_color(edge)
                    for cap_pair in zip(bp["caps"][0::2], bp["caps"][1::2]):
                        for cp, edge in zip(cap_pair, edge_used):
                            cp.set_color(edge)

                ax.set_xlim(-0.6, len(cats_sorted) - 0.4)
                ax.set_title(f"{Y} · {V}", loc="left", fontsize=10)
                if c == 0:
                    ax.set_ylabel("log₁₀ |a* − a₀|")
                _apply_modern_style(ax)

        # Only bottom row gets xtick labels
        for c in range(n_cols):
            axes[-1, c].set_xticks(range(len(cats_sorted)))
            axes[-1, c].set_xticklabels(cats_sorted, rotation=60, ha="right", fontsize=8)
            axes[-1, c].set_xlabel(title_suffix)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def _fig_prior_departure_from_sector(self):
        """SD-T-09a: |a*−a₀| boxplot grouped by from_sector (2×2 year × val)."""
        return self._fig_prior_departure_box(
            "from_sector", "from-sector", fig_id="SD-T-09a",
        )

    def _fig_prior_departure_to_sector(self):
        """SD-T-09b: |a*−a₀| boxplot grouped by to_sector (2×2 year × val)."""
        return self._fig_prior_departure_box(
            "to_sector", "to-sector", fig_id="SD-T-09b",
        )

    def _fig_prior_departure_from_region(self):
        """SD-T-09c: |a*−a₀| boxplot grouped by from_region; ROW + 5 mining highlighted."""
        return self._fig_prior_departure_box(
            "from_region", "from-region", fig_id="SD-T-09c",
            highlight=self._MINING_REGION_HIGHLIGHT,
        )

    def _fig_prior_departure_to_region(self):
        """SD-T-09d: |a*−a₀| boxplot grouped by to_region; ROW + 5 mining highlighted."""
        return self._fig_prior_departure_box(
            "to_region", "to-region", fig_id="SD-T-09d",
            highlight=self._MINING_REGION_HIGHLIGHT,
        )

    # ------------------------------------------------------------------
    # SD-R-08 / SD-R-09 — entity-output stacked area by sector
    # ------------------------------------------------------------------

    def _fig_entity_outputs(self, *, from_entity: str, fig_id: str,
                             title_phrase: str):
        """Line plot of Σ a* across outgoing edges of ``from_entity``,
        grouped by from_sector. Two panels: cols = (Mass, CC). x = timestep.
        One line per sector.
        """
        import matplotlib.pyplot as plt

        em = self._edge_meta()
        if em is None:
            fig = plt.figure(figsize=(8, 3))
            fig.suptitle(self._title(f"{fig_id}: edge_index unavailable (acc data missing)"),
                         fontsize=12)
            return fig

        required = {"from_entity", "from_sector", "time", "valuation"}
        missing = required - set(em.columns)
        if missing:
            fig = plt.figure(figsize=(8, 3))
            fig.suptitle(self._title(f"{fig_id}: edge_meta missing columns {missing}"),
                         fontsize=12)
            return fig

        em_fe   = em["from_entity"].astype(str).to_numpy()
        em_fs   = em["from_sector"].astype(str).to_numpy()
        em_time = em["time"].astype(str).to_numpy()
        em_val  = em["valuation"].astype(str).to_numpy()

        entity_mask = (em_fe == from_entity)
        if not entity_mask.any():
            fig = plt.figure(figsize=(8, 3))
            fig.suptitle(self._title(
                f"{fig_id}: no edges with from_entity={from_entity!r}"),
                         fontsize=12)
            return fig

        sectors = sorted({s for s in em_fs[entity_mask]})
        valuations = list(self._SD_T_09_VALUATIONS)
        ts = self._timesteps
        first_t = str(ts[0]) if ts else None
        a0 = np.asarray(self._const_run.a0, dtype=float)

        # Build [n_sectors × n_timesteps] sum matrix per valuation, plus
        # the constant a₀-at-first-year reference per sector per valuation.
        per_val: dict = {}
        ref_first: dict = {}
        for V in valuations:
            mat = np.zeros((len(sectors), len(ts)), dtype=float)
            for j, t in enumerate(ts):
                F = np.asarray(self._results[t].F_opt, dtype=float)
                cell_mask = entity_mask & (em_time == str(t)) & (em_val == V)
                if not cell_mask.any():
                    continue
                fs_cell = em_fs[cell_mask]
                F_cell  = F[cell_mask]
                for i, S in enumerate(sectors):
                    mat[i, j] = float(np.sum(F_cell[fs_cell == S]))
            per_val[V] = mat

            ref = np.zeros(len(sectors), dtype=float)
            if first_t is not None:
                ref_mask = entity_mask & (em_time == first_t) & (em_val == V)
                if ref_mask.any():
                    fs_ref = em_fs[ref_mask]
                    a0_ref = a0[ref_mask]
                    for i, S in enumerate(sectors):
                        ref[i] = float(np.sum(a0_ref[fs_ref == S]))
            ref_first[V] = ref

        # Line order = sectors sorted by mean magnitude descending (legend top-down)
        pooled = sum(np.abs(per_val[V]).mean(axis=1) for V in valuations)
        order = np.argsort(-pooled)
        sectors_ord = [sectors[i] for i in order]

        tab20 = plt.get_cmap("tab20").colors
        color_for = {S: tab20[i % 20] for i, S in enumerate(sectors_ord)}

        fig, axes = plt.subplots(
            1, len(valuations),
            figsize=(13, 5.5),
            sharex=True, squeeze=False,
        )
        axes = axes[0]
        fig.suptitle(self._title(f"{title_phrase} ({fig_id})"),
                     fontsize=13, fontweight="bold")

        x = np.arange(len(ts))
        for c, V in enumerate(valuations):
            ax = axes[c]
            mat = per_val[V]
            ref = ref_first[V]
            for i, S in enumerate(sectors_ord):
                ax.plot(x, mat[order[i]], color=color_for[S], lw=1.5,
                        label=S, alpha=0.9)
                ax.hlines(ref[order[i]], xmin=x[0], xmax=x[-1],
                          colors=color_for[S], linestyles="--",
                          lw=1.0, alpha=0.6)
            ax.set_title(f"valuation = {V}", loc="left", fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([str(t) for t in ts], rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("timestep")
            if c == 0:
                ax.set_ylabel("Σ a* across outputs [tonnes]")
            _apply_modern_style(ax)
        # Generic dashed-reference entry in legend
        axes[0].plot([], [], color=_C["grey"], ls="--", lw=1.0,
                     label=f"a₀ ({first_t})" if first_t else "a₀")

        # Single legend on the right
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right",
                   fontsize=7, frameon=False, ncol=1,
                   bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout(rect=[0, 0, 0.88, 0.95])
        return fig

    def _fig_process_outputs(self):
        """SD-R-08: Σ supply-edge a* by from_sector, stacked area per valuation."""
        return self._fig_entity_outputs(
            from_entity="Process",
            fig_id="SD-R-08",
            title_phrase="Process-entity outputs (supply edges) by sector",
        )

    def _fig_flow_outputs(self):
        """SD-R-09: Σ use-edge a* by from_sector, stacked area per valuation."""
        return self._fig_entity_outputs(
            from_entity="Flow",
            fig_id="SD-R-09",
            title_phrase="Flow-entity outputs (use edges) by sector",
        )

    # ------------------------------------------------------------------
    # Title helpers
    # ------------------------------------------------------------------

    def _title(self, base: str) -> str:
        return f"{base}  —  {self.label}" if self.label else base

    @staticmethod
    def _col_title(t: str, triple: tuple) -> str:
        return str(t)
