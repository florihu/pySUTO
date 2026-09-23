"""
solver.py
=========
``SolverDiag`` — the Stage-4 (opt run) figure portfolio.

Every number is read from the all-years frames of
:class:`pysuto.diag.stats.SolverStatsDiagnostic` (edge frame, constraint frame),
so the figures and the per-year ``edges_<year>.csv`` / ``constraints_<year>.csv``
agree by construction. Vocabulary follows CONTEXT.md: **defect** = ``a*−a_prior``
per edge; **residual** = ``G·a−c`` per row; **objective share** = an item's ``z²``
over its family's ``Σz²`` (the QP objective *is* ``Σz²``); **identity** = the
cross-year key of an edge (template) or a row (``row_meta`` columns);
**retention** = ``Σa*/Σa_prior`` over a set of edges.

Plot catalogue — :attr:`SolverDiag.PLOTS` is the registry (ids are stable)
---------------------------------------------------------------------------
Run scope (x = year):

* **SD-R-01** ``sd_r_01_solver_health`` — status strip, iterations, solve time,
  final KKT residuals (``res_primal``, ``res_dual``, ``gap_rel``).      [hueable]
* **SD-R-08** ``sd_r_08_useful_output`` — useful output (Process→Flow) by
  process sector: ``Σa*(t)`` vs ``Σa_prior(t)`` and retention; one column per
  valuation.
* **SD-R-09** ``sd_r_09_flow_total_output`` — total output by flow sector
  (Flow→Process + Flow→SB final demand, the ``usetotout`` group); same layout.
* **SD-R-10** ``sd_r_10_boundary_flows`` — SB-input and SB-output totals and
  retention by SB sector.
* **SD-R-11** ``sd_r_11_objective_decomposition`` — ``Σz²`` per family (edge
  defects / sum-row residuals / soft-ratio residuals) per year, max hard
  mass-balance ``|G·a*−c|``, share of ``|z| > 3``.                    [hueable]
* **SD-R-12** ``sd_r_12_convergence_traces`` — Clarabel per-iteration traces,
  one line per year.
* **SD-R-13** ``sd_r_13_objective_share_by_sector`` — objective share per
  sector key, per family.                                              [hueable]
* **SD-R-14** ``sd_r_14_top_identities`` — top-25 identities per family by
  ``Σ_t z²``, annotated with ``Σ_t|Δ|``.
* **SD-R-15a/b/c** ``sd_r_15{a,b,c}_profiles_{edges,sum_rows,soft_ratios}`` —
  ``z_t`` over years for the top-12 identities of each family.

Timestep scope (columns = first / worst-``Σz²`` / last year):

* **SD-T-07** ``sd_t_07_rocket`` — c-space (``G·a_prior(t)`` vs ``c*`` on sum
  rows) and a-space (``a_prior(t)`` vs ``a*``) log–log scatters, colour ``|z|``,
  top-10 identities labelled.

``plot(regions=[...])`` renders the same portfolio restricted to edges / rows
touching those regions (filename suffix ``__<region>``); ``plot()`` also writes
``ranking_<family>.csv`` (the full identity ranking) next to the figures.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from ._plot import _C, _CAT8, _savefig
from .stats._core import boundary_edges, filter_regions, sector_totals
from .stats.objective import (
    FAMILIES, FAMILY_LABELS, family_frame, objective_decomposition, profiles,
    rank_identities, share_by_key,
)
from .stats.solver import SolverStatsDiagnostic

# Hue palette for SolverDiag.compare(). Order matters: first run gets blue.
_HUE_PALETTE = ["blue", "orange", "green", "purple", "red", "navy"]
# One colour / linestyle per objective family (single-run SD-R-11, SD-R-15).
_FAMILY_COLOR = {"edges": _C["blue"], "sum_rows": _C["orange"], "soft_ratios": _C["purple"]}
_FAMILY_LS    = {"edges": "-",        "sum_rows": "--",         "soft_ratios": ":"}
_STATUS_COLOR = {"Solved": _C["green"], "AlmostSolved": _C["orange"]}   # anything else: red

_LOG_FLOOR   = 1e-6   # values are floored here before log10 (rocket axes)
_MAX_SERIES  = 8      # categorical hues per panel; the tail folds into "other"
_TOP_KEYS    = 20     # sector keys shown in SD-R-13
_TOP_IDS     = 25     # identities shown in SD-R-14
_N_PROFILES  = 12     # identities profiled in SD-R-15
_N_ROCKET_LABELS = 10
# entity_sub tie-break for the SB legend order (label literals — see ADR 0034;
# only an ordering hint, never a filter).
_BOUNDARY_FAMILY_ORDER = {"Nature": 0, "SEM": 1, "FD": 2}


class PlotSpec(NamedTuple):
    """One row of the :attr:`SolverDiag.PLOTS` registry.

    ``method`` is the ``_fig_*`` entrypoint (each returns exactly one Figure),
    ``filename`` the PNG stem (``.png`` appended at save), ``desc`` a one-line
    explanation, and ``hueable`` whether the fig can be overlaid in
    :meth:`SolverDiag.compare`.
    """

    method:   str
    filename: str
    desc:     str
    hueable:  bool = False


def _apply_modern_style(ax) -> None:
    """Tighten the look of one axes: no top/right spines, light grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)


def _slug(text) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()


def _short(text, n: int = 45) -> str:
    s = str(text)
    return s if len(s) <= n else s[: n - 1] + "…"


class SolverDiag:
    """
    Stage-4 solver diagnostics — runs after ``ClarabelSolver.solve_all()``.

    Two construction modes:

    * **Live**: pass the ``results`` dict + ``ConstraintRun`` directly
      (typical inside a solver script, before/after ``persist()``).
    * **Persisted**: ``SolverDiag.from_opt_id(opt_run_id)`` reconstructs
      the run from ``{output_dir}/registry.json`` and per-timestep ``.npz``
      files, plus the matching ``ConstraintRun`` from ``ConstraintRunRegistry``.

    Plot portfolio: see :attr:`PLOTS` — the canonical registry mapping every
    plot id → its ``_fig_*`` method, PNG stem, one-line description, and
    ``.compare()`` hue-eligibility. It is the single source of truth and must
    gain an entry whenever a new figure method is added.

    Parameters
    ----------
    results : dict[str, OptimResult]
        Output of ``ClarabelSolver.solve_all()``.
    constraint_run : ConstraintRun
        The ConstraintRun the solver consumed — supplies G, row_meta,
        a0, sigma_a, u, l, and per-timestep c / sigma_c.
    label : str
        Optional label for figure titles and output filenames.
    acc : AccountingStructure | None
        The accounting structure (edge labels, edge templates). Pass it when
        the caller already holds it — loading ``copper`` costs several GB —
        otherwise it is loaded lazily from ``constraint_run.acc_id``; when that
        fails the edge-based figures render placeholders.
    """

    # ---- Plot registry — single source of truth (insertion order = render order) ----
    # Add an entry here whenever a new _fig_* method is implemented.
    PLOTS: "dict[str, PlotSpec]" = {
        "SD-R-01":  PlotSpec("_fig_solver_health",           "sd_r_01_solver_health",            "Solver health: status strip, iterations, solve time, final KKT residuals per year", hueable=True),
        "SD-R-08":  PlotSpec("_fig_useful_output",           "sd_r_08_useful_output",            "Useful output (Process→Flow) by process sector: Σa*(t) vs Σa_prior(t) + retention, cols = valuation"),
        "SD-R-09":  PlotSpec("_fig_flow_total_output",       "sd_r_09_flow_total_output",        "Total output by flow sector (Flow→Process + Flow→SB): Σa*(t) vs Σa_prior(t) + retention, cols = valuation"),
        "SD-R-10":  PlotSpec("_fig_boundary_flows",          "sd_r_10_boundary_flows",           "System-boundary totals by SB sector: SB-input and SB-output Σa* vs Σa_prior + retention, cols = valuation"),
        "SD-R-11":  PlotSpec("_fig_objective_decomposition", "sd_r_11_objective_decomposition",  "Objective decomposition per year: Σz² edge defects / sum-row residuals / soft-ratio residuals, max hard mass-balance |G·a*−c|, share |z|>3", hueable=True),
        "SD-R-12":  PlotSpec("_fig_convergence_traces",      "sd_r_12_convergence_traces",       "Clarabel per-iteration traces (res_primal, res_dual, gap_abs, step_length), one line per year"),
        "SD-R-13":  PlotSpec("_fig_objective_share_by_sector", "sd_r_13_objective_share_by_sector", "Objective share per sector key, per family (edges: from→to sector; sum rows: sector; soft ratios: numerator from→to sector)", hueable=True),
        "SD-R-14":  PlotSpec("_fig_top_identities",          "sd_r_14_top_identities",           "Top-25 identities per family ranked by Σ_t z², annotated with Σ_t|Δ| in kt"),
        "SD-R-15a": PlotSpec("_fig_profiles_edges",          "sd_r_15a_profiles_edges",          "z_t over years for the top-12 edge identities (cross-year template)"),
        "SD-R-15b": PlotSpec("_fig_profiles_sum_rows",       "sd_r_15b_profiles_sum_rows",       "z_t over years for the top-12 sum-row identities"),
        "SD-R-15c": PlotSpec("_fig_profiles_soft_ratios",    "sd_r_15c_profiles_soft_ratios",    "z_t over years for the top-12 soft-ratio identities"),
        "SD-T-07":  PlotSpec("_fig_rocket",                  "sd_t_07_rocket",                   "Rocket: c-space (G·a_prior(t) vs c*, sum rows) and a-space (a_prior(t) vs a*) at first / worst-Σz² / last year, colour |z|"),
    }

    def __init__(self, results, constraint_run, label: str = "", acc=None) -> None:
        self.label      = label
        self._results   = self._reassemble_if_year_local(dict(results), constraint_run)
        self._timesteps = list(self._results.keys())      # original keys (str on disk)
        self._years     = [str(t) for t in self._timesteps]
        self._const_run = constraint_run

        # Accounting structure: injected, or loaded lazily on first use.
        self._acc          = acc
        self._acc_resolved = acc is not None
        self._stats_obj: Optional[SolverStatsDiagnostic] = None
        self._fam_cache: dict = {}
        self._dec_cache: dict = {}

        # Default odir cache — populated by from_opt_id; None for live mode
        self._default_odir: Optional[str] = None

        # Provenance fields — populated by from_opt_id; None in live mode
        self._opt_run_id:      Optional[str]  = None
        self._opt_entry:       Optional[dict] = None
        self._imp_index_entry: Optional[dict] = None
        self._feed_entries:    Optional[list] = None

        # Row classification (mass-balance rows for summary()) and the ε-norm
        # fallback for the worst year when no frames are available.
        self._row_info = self._classify_rows(constraint_run)
        self._eps_norms = {
            t: float(np.linalg.norm(np.asarray(self._results[t].eps_opt, dtype=float)))
            for t in self._timesteps
        }

    @property
    def results(self) -> dict:
        """The (reassembled) per-timestep ``OptimResult`` dict."""
        return self._results

    @property
    def constraint_run(self):
        """The ``ConstraintRun`` this diag was built from."""
        return self._const_run

    @property
    def acc(self):
        """The ``AccountingStructure`` (injected or lazily loaded); ``None`` when
        it cannot be loaded (toy fixtures)."""
        if not self._acc_resolved:
            self._acc_resolved = True
            try:
                from pysuto.acc import AccountingStructure
                self._acc = AccountingStructure.load(self._const_run.acc_id)
            except Exception:
                self._acc = None
        return self._acc

    @property
    def stats(self) -> SolverStatsDiagnostic:
        """The stats object owning the all-years frames (built once)."""
        if self._stats_obj is None:
            self._stats_obj = SolverStatsDiagnostic(
                self._results, self._const_run, self.acc, years="all",
            )
        return self._stats_obj

    @staticmethod
    def _reassemble_if_year_local(results: dict, constraint_run) -> dict:
        """Block-path (year-local) results store only each period's solved
        slice (F_opt over cols_t, residuals/c_star over rows_t). Scatter the
        pieces back into one complete F (E,) and one complete residuals/c_star
        (M,) — each edge/row belongs to exactly one period, so the assembled
        vectors are the true, placeholder-free solution. Legacy (full-vector)
        results are returned untouched. cols_t/rows_t are retained so the
        per-period prior can be re-sliced."""
        if not any(getattr(r, "cols_t", None) is not None for r in results.values()):
            return results

        E = constraint_run.G.shape[1]
        M = constraint_run.G.shape[0]
        F_complete     = np.zeros(E)
        resid_complete = np.zeros(M)
        cstar_complete = np.zeros(M)
        for r in results.values():
            if r.cols_t is None:
                continue
            F_complete[r.cols_t]     = r.F_opt
            resid_complete[r.rows_t] = r.residuals
            cstar_complete[r.rows_t] = r.c_star

        return {
            t: replace(
                r,
                F_opt     = F_complete,
                residuals = resid_complete,
                c_star    = cstar_complete,
            )
            for t, r in results.items()
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
        acc=None,
    ) -> "SolverDiag":
        """Reconstruct results + ConstraintRun from disk."""
        # Local import to keep diag.py at the leaf of the import graph for
        # CI environments without the full Stage-3 stack installed.
        from pysuto.registry import ConstraintRunRegistry, OptimRunRegistry
        from pysuto.registry.optim import warn_if_prefix_block_run

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
        warn_if_prefix_block_run(results, const_run)

        instance = cls(results, const_run, label=label or opt_run_id, acc=acc)
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
        Pre-compute masks over row_meta for quick filtering.

        Returns dict with int-array indices keyed by the display buckets
        'mass_balance', 'sum_eq', 'ratio_eq', 'ratio_bound', 'zero', 'free',
        'nonneg' and a string array 'type_of_row' parallel to row_meta.

        These buckets are derived from the canonical row_meta family column
        ``type`` ∈ {massbal, sum, ratio} plus ``cone_type`` (which splits ratio
        into eq=free vs bound=nonneg). They are *display labels*, not the
        ``type`` column values.
        """
        rm = const_run.row_meta
        ct = rm["cone_type"].to_numpy() if "cone_type" in rm.columns else np.array([], dtype=str)

        if "type" in rm.columns:
            fam = rm["type"].astype(str).to_numpy()
            tp = fam.astype(object).copy()
            tp[fam == "massbal"] = "mass_balance"
            tp[fam == "sum"]     = "sum_eq"
            ratio_mask = (fam == "ratio")
            tp[ratio_mask & (ct == "free")]   = "ratio_eq"
            tp[ratio_mask & (ct == "nonneg")] = "ratio_bound"
            tp = tp.astype(str)
        else:
            # Toy fixtures: synthesise type from (cone_type, is_hard)
            is_hard = rm["is_hard"].to_numpy() if "is_hard" in rm.columns else np.zeros(len(rm), dtype=bool)
            tp = np.full(len(rm), "unknown", dtype=object)
            tp[(ct == "zero")   & is_hard] = "mass_balance"
            tp[(ct == "free")  & ~is_hard] = "sum_eq"
            tp[(ct == "nonneg") & is_hard] = "ratio_bound"
            tp = tp.astype(str)

        return {
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

    # ------------------------------------------------------------------
    # Frames, families, decomposition (cached)
    # ------------------------------------------------------------------

    def _edge_frame(self):
        return self.stats.edge_frame()

    def _constraint_frame(self):
        return self.stats.constraint_frame()

    @staticmethod
    def _rkey(regions):
        return tuple(str(r) for r in regions) if regions else None

    def _family(self, fam: str, regions=None):
        """Harmonised family frame (cached per family × region filter)."""
        key = (fam, self._rkey(regions))
        if key not in self._fam_cache:
            self._fam_cache[key] = family_frame(
                self._edge_frame(), self._constraint_frame(), fam, regions=regions,
            )
        return self._fam_cache[key]

    def _decomposition(self, regions=None):
        """Per-year objective decomposition (cached per region filter); ``None``
        when neither frame is available."""
        key = self._rkey(regions)
        if key not in self._dec_cache:
            ef, cf = self._edge_frame(), self._constraint_frame()
            if ef is None and cf is None:
                self._dec_cache[key] = None
            else:
                self._dec_cache[key] = objective_decomposition(
                    ef, cf, years=self._years, regions=regions,
                )
        return self._dec_cache[key]

    def _key_for_year(self, year: str):
        """The results-dict key whose ``str()`` is ``year``."""
        for t in self._timesteps:
            if str(t) == str(year):
                return t
        raise KeyError(year)

    def _t_worst(self, regions=None):
        """Year with the largest total Σz² (objective decomposition); falls back
        to the largest ‖ε‖₂ when no frame is available."""
        dec = self._decomposition(regions)
        if dec is not None and dec["zsq_total"].notna().any():
            y = dec.set_index("year")["zsq_total"].idxmax()
            return self._key_for_year(y)
        return max(self._eps_norms, key=self._eps_norms.get)

    def _resolve_timesteps(self, timesteps, regions=None) -> tuple:
        """Default triple = (first, worst-Σz², last)."""
        if timesteps is None:
            return self._timesteps[0], self._t_worst(regions), self._timesteps[-1]
        if len(timesteps) != 3:
            raise ValueError(f"timesteps must be a triple, got {timesteps!r}")
        # Coerce to str: timestep keys are stored as strings on disk
        coerced = tuple(str(t) for t in timesteps)
        for t in coerced:
            if t not in self._results:
                raise KeyError(f"timestep '{t}' not in results")
        return coerced

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
        dec = self._decomposition()
        if dec is not None and dec["zsq_total"].notna().any():
            tot = dec[[f"zsq_{f}" for f in FAMILIES]].sum(min_count=1)
            worst = self._t_worst()
            lines.append(
                "  objective : Σz² "
                + "  ".join(f"{f}={tot[f'zsq_{f}']:.4g}" for f in FAMILIES)
                + f"  (worst year t={worst})"
            )
        worst_eps = max(self._eps_norms, key=self._eps_norms.get)
        lines.append(f"  worst ε   : t={worst_eps}  ‖ε‖₂={self._eps_norms[worst_eps]:.3e}")
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
        plot_ids: Optional[list] = None,
        regions: Optional[list] = None,
    ) -> list[str]:
        """Render figures from the :attr:`PLOTS` registry (+ ranking CSVs + provenance).

        Parameters
        ----------
        odir : str | None
            Output directory. If None, defaults to ``{output_dir}/{opt_run_id}/diag``
            when this instance came from ``from_opt_id``; otherwise required.
        timesteps : tuple[str, str, str] | None
            Columns of the timestep-scope figures (the rocket). ``None`` →
            ``(first, worst-Σz², last)``.
        plot_ids : list[str] | None
            Subset of registry ids to render. ``None`` (default) renders every
            registered figure. Figures always render in canonical registry order
            regardless of the order given here. An unknown id raises ``ValueError``.
        regions : list[str] | None
            Restrict the sector totals, objective-share and rocket figures to
            edges / rows touching these regions; filenames get a ``__<region>``
            suffix. Solver-mechanics figures (SD-R-01, SD-R-12) ignore it.
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

        selected = set(self.PLOTS) if plot_ids is None else set(plot_ids)
        unknown = selected - set(self.PLOTS)
        if unknown:
            raise ValueError(
                f"Unknown plot_ids {sorted(unknown)}. Valid ids: {list(self.PLOTS)}"
            )

        ts_triple = self._resolve_timesteps(timesteps, regions)
        suffix = self._region_suffix(regions)

        paths: list[str] = []
        for pid, spec in self.PLOTS.items():        # canonical registry order
            if pid not in selected:
                continue
            fig = getattr(self, spec.method)(hue=None, timesteps=ts_triple, regions=regions)
            p = _savefig(fig, os.path.join(odir, f"{spec.filename}{suffix}.png"), dpi=dpi)
            print(f"   {pid}  → {p}  ✓")
            paths.append(p)
            if not show:
                import matplotlib.pyplot as plt; plt.close(fig)

        # The identity rankings behind SD-R-13/14/15 — one CSV per family.
        paths += self.stats.write_rankings(odir, regions=regions, suffix=suffix)

        # Provenance is run lineage, not a figure — always appended (no-op in live mode).
        prov_path = self._write_provenance(odir)
        if prov_path:
            paths.append(prov_path)

        return paths

    # ------------------------------------------------------------------
    # Provenance trace
    # ------------------------------------------------------------------

    def _write_provenance(self, odir: str) -> Optional[str]:
        """Write provenance.md to *odir* and print a compact tree to stdout."""
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
            f"# Provenance: {self.label or self._opt_run_id or 'unlabeled'}",
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
        regions: Optional[list] = None,
    ) -> list[str]:
        """
        Overlay run-scope diagnostics from multiple SolverDiag instances.

        Parameters
        ----------
        diags : dict[str, SolverDiag]
            ``{hue_label: diag}`` — up to 6 entries recommended (palette size).
        odir : str
            Output directory. Files land at ``{odir}/{label}_{stem}{suffix}.png``.
        plot_ids : list[str] | None
            Subset of hueable ids to render. Default is every hueable id in
            :attr:`PLOTS`.
        regions : list[str] | None
            Same region filter as :meth:`plot`.
        """
        if not diags:
            raise ValueError("compare() requires at least one diag")
        hueable = [pid for pid, spec in cls.PLOTS.items() if spec.hueable]
        if plot_ids is None:
            plot_ids = list(hueable)
        not_hueable = set(plot_ids) - set(hueable)
        if not_hueable:
            raise ValueError(
                f"plot_ids contain non-hueable plots: {sorted(not_hueable)}. "
                f"Hueable: {hueable}"
            )
        os.makedirs(odir, exist_ok=True)

        # Build (hue_label, diag, color) tuples once
        hue: list = []
        for k, (lbl, dg) in enumerate(diags.items()):
            color = _C[_HUE_PALETTE[k % len(_HUE_PALETTE)]]
            hue.append((lbl, dg, color))

        # Render using the first diag as host; each method accepts hue=[(label,diag,color),...]
        host = next(iter(diags.values()))
        suffix = cls._region_suffix(regions)
        paths: list[str] = []
        for pid in plot_ids:
            spec = cls.PLOTS[pid]
            fig = getattr(host, spec.method)(hue=hue, timesteps=None, regions=regions)
            p = _savefig(fig, os.path.join(odir, f"{label}_{spec.filename}{suffix}.png"), dpi=dpi)
            print(f"   {pid} (compare) → {p}  ✓")
            paths.append(p)
            import matplotlib.pyplot as plt; plt.close(fig)
        return paths

    # ==================================================================
    # Internal: small helpers shared by the figures
    # ==================================================================

    def _ts_index(self) -> np.ndarray:
        """X-axis array: integer indices [0..T-1] for plotting line series."""
        return np.arange(len(self._timesteps))

    def _ts_labels_for_xticks(self) -> list[str]:
        return [str(t) for t in self._timesteps]

    def _conv_series(self, key: str) -> np.ndarray:
        """Pull ``OptimResult.convergence[key]`` over time."""
        return np.array(
            [float((self._results[t].convergence or {}).get(key, np.nan))
             for t in self._timesteps]
        )

    def _scalar_series(self, attr: str) -> np.ndarray:
        return np.array(
            [float(getattr(self._results[t], attr)) for t in self._timesteps]
        )

    @staticmethod
    def _region_suffix(regions) -> str:
        if not regions:
            return ""
        return "__" + "-".join(_slug(r) for r in regions)

    def _title(self, base: str, regions=None) -> str:
        t = f"{base}  —  {self.label}" if self.label else base
        if regions:
            t += f"  ·  regions: {', '.join(str(r) for r in regions)}"
        return t

    def _placeholder(self, text: str, regions=None):
        """A figure carrying only a title — used when the data behind a figure
        is unavailable (no acc, toy row_meta) so the portfolio stays complete."""
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 3))
        fig.suptitle(self._title(text, regions), fontsize=12)
        return fig

    def _set_year_xticks(self, ax, every: int = 1) -> None:
        x = self._ts_index()
        ax.set_xticks(x[::every])
        ax.set_xticklabels(self._ts_labels_for_xticks()[::every], rotation=45, ha="right", fontsize=8)

    @staticmethod
    def _valuations(frame) -> list[str]:
        """Valuation labels present in a frame (sorted — no label literals)."""
        return sorted(frame["valuation"].dropna().astype(str).unique().tolist())

    @staticmethod
    def _series_colors(names) -> dict:
        """Stable colour per series name: slots of the validated categorical
        palette in sorted-name order; ``"other"`` is always grey."""
        named = sorted(n for n in names if n != "other")
        colors = {n: _CAT8[i % len(_CAT8)] for i, n in enumerate(named)}
        colors["other"] = _C["grey"]
        return colors

    # ------------------------------------------------------------------
    # Sector totals machinery (SD-R-08 / 09 / 10)
    # ------------------------------------------------------------------

    @staticmethod
    def _named_groups(tot: pd.DataFrame, order_hint=None) -> list:
        """Groups that get their own line, by mean |Σa*| descending; when more
        than ``_MAX_SERIES`` exist the top ``_MAX_SERIES − 1`` are named and
        the rest fold into ``"other"``. ``order_hint`` (dict name → rank) is
        applied first (SB family order)."""
        mag = tot.groupby("group", observed=True)["sum_post"].apply(lambda s: float(np.nanmean(np.abs(s))))
        mag = mag.fillna(0.0)
        keys = sorted(mag.index, key=lambda g: ((order_hint or {}).get(g, 0), -mag[g]))
        if len(keys) > _MAX_SERIES:
            by_mag = sorted(mag.index, key=lambda g: -mag[g])[: _MAX_SERIES - 1]
            keys = [g for g in keys if g in set(by_mag)]
        return keys

    @staticmethod
    def _fold_groups(tot: pd.DataFrame, named) -> pd.DataFrame:
        """Sum every group not in ``named`` into ``"other"`` per (valuation, year)."""
        named = set(named)
        if tot["group"].isin(named).all():
            return tot
        t = tot.copy()
        t["group"] = np.where(t["group"].isin(named), t["group"], "other")
        g = (t.groupby(["group", "valuation", "year"], sort=False)[["sum_post", "sum_prior"]]
              .sum(min_count=1).reset_index())
        sp = g["sum_prior"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            g["retention"] = np.where(sp > 0, g["sum_post"].to_numpy(dtype=float) / sp, np.nan)
        return g

    def _totals_data(self, world_edges, group_col: str, regions=None, order_hint=None):
        """(valuations, folded totals, ordered groups, colours) for one edge
        subset. The named set and the colours come from the *world* view so a
        region filter never repaints a sector."""
        vals = self._valuations(world_edges)
        world = sector_totals(world_edges, group_col, years=self._years, valuations=vals)
        named = self._named_groups(world, order_hint)
        colors = self._series_colors(named)
        edges = filter_regions(world_edges, ("from_region", "to_region"), regions)
        if edges.empty:
            return vals, None, [], colors
        tot = self._fold_groups(sector_totals(edges, group_col, years=self._years, valuations=vals), named)
        groups = [g for g in named if g in set(tot["group"])]
        if "other" in set(tot["group"]):
            groups.append("other")
        return vals, tot, groups, colors

    def _draw_totals_rows(self, ax_top, ax_bot, vals, tot, groups, colors, unit="kt") -> None:
        """Fill one (totals, retention) row pair: ``ax_top[c]`` / ``ax_bot[c]``
        per valuation ``vals[c]``."""
        x = self._ts_index()
        for c, V in enumerate(vals):
            at, ab = ax_top[c], ax_bot[c]
            sub = tot[tot["valuation"] == V] if tot is not None else None
            if sub is None or sub.empty:
                at.text(0.5, 0.5, "n=0", transform=at.transAxes, ha="center", va="center",
                        fontsize=9, color=_C["grey"])
            else:
                rmax = 1.0
                vals_seen = []
                for g in groups:
                    s = sub[sub["group"] == g].set_index("year").reindex(self._years)
                    if s["sum_post"].notna().sum() == 0 and s["sum_prior"].notna().sum() == 0:
                        continue
                    col = colors.get(g, _C["grey"])
                    post, prior = s["sum_post"].to_numpy(dtype=float), s["sum_prior"].to_numpy(dtype=float)
                    at.plot(x, post, color=col, lw=1.6, label=g)
                    at.plot(x, prior, color=col, lw=1.1, ls="--", alpha=0.7)
                    vals_seen.append(post); vals_seen.append(prior)
                    r = s["retention"].to_numpy(dtype=float)
                    ab.plot(x, r, color=col, lw=1.6, label=g)
                    if np.isfinite(r).any():
                        rmax = max(rmax, float(np.nanmax(r)))
                allv = np.concatenate(vals_seen) if vals_seen else np.array([1.0])
                allv = allv[np.isfinite(allv)]
                if allv.size and (allv <= 0).any():
                    # signed series (net stock changes): symlog keeps sign and zeros readable
                    pos = allv[allv > 0]
                    lin = 10 ** np.floor(np.log10(pos.min())) if pos.size else 1.0
                    at.set_yscale("symlog", linthresh=max(float(lin), 1e-3))
                else:
                    at.set_yscale("log")
                ab.set_ylim(0, min(max(1.5, rmax * 1.05), 5.0))
            ab.axhline(1.0, color=_C["grey"], lw=1.0, ls="--")
            at.set_title(f"valuation = {V}", loc="left", fontsize=10)
            if c == 0:
                at.set_ylabel(f"Σ a* (solid) · Σ a_prior (dashed) [{unit}]")
                ab.set_ylabel("retention Σa*/Σa_prior")
            _apply_modern_style(at)
            _apply_modern_style(ab)

    def _totals_legend(self, fig, groups, colors, loc_rect=(0, 0, 0.87, 0.95)) -> None:
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color=colors.get(g, _C["grey"]), lw=1.8, label=g) for g in groups]
        handles.append(Line2D([0], [0], color=_C["navy"], lw=1.6, label="posterior a*(t)"))
        handles.append(Line2D([0], [0], color=_C["navy"], lw=1.1, ls="--", alpha=0.7, label="prior a_prior(t)"))
        fig.legend(handles=handles, loc="center right", fontsize=7, frameon=False,
                   bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout(rect=list(loc_rect))

    def _fig_totals_by_sector(self, *, subset, fig_id: str, title: str, regions=None):
        """Shared body of SD-R-08 / SD-R-09: 2 rows (Σ totals, retention) ×
        valuation columns for one edge subset grouped by ``from_sector``."""
        import matplotlib.pyplot as plt

        if subset is None:
            return self._placeholder(f"{fig_id}: edge frame unavailable (acc data missing)", regions)
        if subset.empty:
            return self._placeholder(f"{fig_id}: no edges in this subset", regions)
        vals, tot, groups, colors = self._totals_data(subset, "from_sector", regions)
        fig, axes = plt.subplots(2, len(vals), figsize=(6.5 * len(vals) + 1.5, 8.5),
                                 sharex=True, squeeze=False)
        fig.suptitle(self._title(f"{title} ({fig_id})", regions), fontsize=13, fontweight="bold")
        self._draw_totals_rows(axes[0], axes[1], vals, tot, groups, colors)
        for c in range(len(vals)):
            self._set_year_xticks(axes[1][c])
            axes[1][c].set_xlabel("year")
        self._totals_legend(fig, groups, colors)
        return fig

    def _fig_useful_output(self, *, hue=None, timesteps=None, regions=None):
        """SD-R-08: useful output (Process→Flow) by process sector."""
        ef = self._edge_frame()
        subset = None
        if ef is not None:
            if not {"from_entity", "to_entity"} <= set(ef.columns):
                return self._placeholder("SD-R-08: edge frame lacks entity columns", regions)
            subset = ef[(ef["from_entity"].astype(str) == "Process")
                        & (ef["to_entity"].astype(str) == "Flow")]
        return self._fig_totals_by_sector(
            subset=subset, fig_id="SD-R-08",
            title="Useful output (Process→Flow) by process sector", regions=regions,
        )

    def _fig_flow_total_output(self, *, hue=None, timesteps=None, regions=None):
        """SD-R-09: total output by flow sector (Flow→Process + Flow→SB)."""
        ef = self._edge_frame()
        subset = None
        if ef is not None:
            if "from_entity" not in ef.columns:
                return self._placeholder("SD-R-09: edge frame lacks entity columns", regions)
            subset = ef[ef["from_entity"].astype(str) == "Flow"]
        return self._fig_totals_by_sector(
            subset=subset, fig_id="SD-R-09",
            title="Total output by flow sector (Flow→Process + Flow→SB)", regions=regions,
        )

    def _fig_boundary_flows(self, *, hue=None, timesteps=None, regions=None):
        """SD-R-10: SB-input / SB-output totals and retention by SB sector."""
        import matplotlib.pyplot as plt

        ef = self._edge_frame()
        if ef is None:
            return self._placeholder("SD-R-10: edge frame unavailable (acc data missing)", regions)
        if not {"from_entity", "to_entity"} <= set(ef.columns):
            return self._placeholder("SD-R-10: edge frame lacks entity columns", regions)
        be = boundary_edges(ef)
        if be.empty:
            return self._placeholder("SD-R-10: no system-boundary edges present", regions)

        directions = [("sb_input", "SB-input (entity_from = SB)"),
                      ("sb_output", "SB-output (entity_to = SB, incl. final demand)")]
        prepared = []
        for key, dlabel in directions:
            sub = be[be["direction"] == key]
            if sub.empty:
                prepared.append((dlabel, None, None, [], {}))
                continue
            fam_of = (sub.drop_duplicates("sb_sector").set_index("sb_sector")["sb_entity_sub"]
                         .astype(str).to_dict())
            hint = {s: _BOUNDARY_FAMILY_ORDER.get(f, 99) for s, f in fam_of.items()}
            prepared.append((dlabel,) + self._totals_data(sub, "sb_sector", regions, order_hint=hint))

        vals = next((p[1] for p in prepared if p[1] is not None), None)
        if vals is None:
            return self._placeholder("SD-R-10: no system-boundary edges present", regions)
        fig, axes = plt.subplots(4, len(vals), figsize=(6.5 * len(vals) + 1.5, 15),
                                 sharex=True, squeeze=False)
        fig.suptitle(self._title("System-boundary totals by SB sector (SD-R-10)", regions),
                     fontsize=13, fontweight="bold")
        all_groups: list = []
        all_colors: dict = {}
        for d, (dlabel, dvals, tot, groups, colors) in enumerate(prepared):
            top, bot = axes[2 * d], axes[2 * d + 1]
            if dvals is None:
                for c in range(len(vals)):
                    top[c].text(0.5, 0.5, "n=0", transform=top[c].transAxes, ha="center",
                                va="center", fontsize=9, color=_C["grey"])
            else:
                self._draw_totals_rows(top, bot, dvals, tot, groups, colors)
            for c in range(len(vals)):
                top[c].set_title(f"{dlabel} · valuation = {vals[c]}", loc="left", fontsize=10)
            all_groups += [g for g in groups if g not in all_groups]
            all_colors.update(colors)
        for c in range(len(vals)):
            self._set_year_xticks(axes[-1][c])
            axes[-1][c].set_xlabel("year")
        self._totals_legend(fig, all_groups, all_colors)
        return fig

    # ------------------------------------------------------------------
    # Solver performance (SD-R-01 / 11 / 12)
    # ------------------------------------------------------------------

    def _fig_solver_health(self, *, hue=None, timesteps=None, regions=None):
        """SD-R-01: status strip + iterations + solve time + final KKT residuals."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        runs = hue if hue else [(self.label or "run", self, _C["blue"])]
        fig, axes = plt.subplots(4, 1, figsize=(13, 9.5), sharex=True,
                                 gridspec_kw={"height_ratios": [0.5 + 0.15 * len(runs), 1, 1, 1.4]})
        fig.suptitle(self._title("Solver health (SD-R-01)"), fontsize=13, fontweight="bold")
        x_master = self._ts_index()

        for k, (lbl, dg, color) in enumerate(runs):
            x = dg._ts_index()
            statuses = [str(dg._results[t].solver_status) for t in dg._timesteps]
            cols = [_STATUS_COLOR.get(s, _C["red"]) for s in statuses]
            axes[0].scatter(x, np.full(len(x), k), c=cols, s=42, marker="s", edgecolors="none")
            axes[1].plot(x, dg._scalar_series("iterations"),
                         color=color, lw=1.6, marker="o", ms=3, label=lbl)
            axes[2].plot(x, dg._scalar_series("solve_time_s"),
                         color=color, lw=1.6, marker="o", ms=3, label=lbl)
            axes[3].semilogy(x, np.maximum(dg._conv_series("res_primal"), 1e-16),
                             color=color, lw=1.4, ls="-",  label=f"{lbl}: res_primal")
            axes[3].semilogy(x, np.maximum(dg._conv_series("res_dual"),   1e-16),
                             color=color, lw=1.4, ls="--", label=f"{lbl}: res_dual")
            axes[3].semilogy(x, np.maximum(dg._conv_series("gap_rel"),    1e-16),
                             color=color, lw=1.4, ls=":",  label=f"{lbl}: gap_rel")

        axes[0].set_yticks(range(len(runs)))
        axes[0].set_yticklabels([lbl for lbl, _, _ in runs], fontsize=8)
        axes[0].set_ylim(-0.6, len(runs) - 0.4)
        axes[0].set_title("Status", loc="left", fontsize=10)
        status_handles = [Line2D([0], [0], marker="s", ls="", color=c, ms=7, label=s)
                          for s, c in list(_STATUS_COLOR.items()) + [("other", _C["red"])]]
        axes[0].legend(handles=status_handles, fontsize=7, frameon=False, loc="upper right", ncol=3)
        axes[1].set_ylabel("iterations");     axes[1].set_title("Iterations", loc="left", fontsize=10)
        axes[2].set_ylabel("solve_time (s)"); axes[2].set_title("Solve time", loc="left", fontsize=10)
        axes[3].set_ylabel("residual");       axes[3].set_title("Final KKT residuals (res_primal, res_dual, gap_rel)", loc="left", fontsize=10)
        axes[3].axhspan(1e-9, 1e-7, color=_C["grey"], alpha=0.10)  # Clarabel default feasibility band

        for ax in axes:
            _apply_modern_style(ax)
        axes[0].grid(False)
        axes[-1].set_xticks(x_master)
        axes[-1].set_xticklabels(self._ts_labels_for_xticks(), rotation=45, ha="right", fontsize=8)
        axes[-1].set_xlabel("year")
        axes[3].legend(fontsize=7, ncol=min(3, len(runs) * 3), frameon=False, loc="center right")
        if len(runs) > 1:
            axes[1].legend(fontsize=7, frameon=False, loc="upper right")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_objective_decomposition(self, *, hue=None, timesteps=None, regions=None):
        """SD-R-11: Σz² per family per year, max hard mass-balance |G·a*−c|, share |z|>3."""
        import matplotlib.pyplot as plt

        runs = hue if hue else [(self.label or "run", self, _C["blue"])]
        decs = [(lbl, dg, color, dg._decomposition(regions)) for lbl, dg, color in runs]
        if all(d is None for _, _, _, d in decs):
            return self._placeholder("SD-R-11: no frames available (acc / row_meta missing)", regions)

        fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
        fig.suptitle(self._title("Objective decomposition (SD-R-11)", regions),
                     fontsize=13, fontweight="bold")
        for lbl, dg, color, dec in decs:
            if dec is None:
                continue
            d = dec.set_index("year").reindex(dg._years)
            x = dg._ts_index()
            for fam in FAMILIES:
                col = color if hue else _FAMILY_COLOR[fam]
                name = f"{lbl}: {FAMILY_LABELS[fam]}" if hue else FAMILY_LABELS[fam]
                axes[0].semilogy(x, d[f"zsq_{fam}"].clip(lower=1e-12).to_numpy(dtype=float),
                                 color=col, ls=_FAMILY_LS[fam], lw=1.6, label=name)
                axes[2].plot(x, d[f"share_gt3_{fam}"].to_numpy(dtype=float),
                             color=col, ls=_FAMILY_LS[fam], lw=1.6, label=name)
            axes[0].semilogy(x, d["zsq_total"].clip(lower=1e-12).to_numpy(dtype=float),
                             color=color if hue else _C["navy"], lw=2.2, alpha=0.9,
                             label=f"{lbl}: total" if hue else "total")
            axes[1].semilogy(x, d["max_abs_resid_massbal"].clip(lower=1e-16).to_numpy(dtype=float),
                             color=color, lw=1.6, marker="o", ms=3, label=lbl)

        axes[0].set_ylabel("Σz² (objective)")
        axes[0].set_title("Σz² per family — edge defects · sum-row residuals · soft-ratio residuals", loc="left", fontsize=10)
        axes[1].set_ylabel("max |G·a* − c|")
        axes[1].set_title("Max hard mass-balance residual", loc="left", fontsize=10)
        axes[1].axhspan(1e-9, 1e-7, color=_C["grey"], alpha=0.10)
        axes[2].set_ylabel("share of rows with |z| > 3")
        axes[2].set_title("Share |z| > 3 per family (priced disagreement, not violation)", loc="left", fontsize=10)
        axes[2].set_ylim(0, 1)
        for ax in axes:
            _apply_modern_style(ax)
        axes[0].legend(fontsize=7, frameon=False, loc="upper left", ncol=2 if hue else 1)
        if hue:
            axes[1].legend(fontsize=7, frameon=False, loc="upper right")
        self._set_year_xticks(axes[-1])
        axes[-1].set_xlabel("year")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def _fig_convergence_traces(self, *, hue=None, timesteps=None, regions=None):
        """SD-R-12: Clarabel per-iteration traces, one line per year."""
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import ListedColormap, Normalize

        hist = []
        for t in self._timesteps:
            h = getattr(self._results[t], "iter_history", None)
            if h is not None and len(h) > 0:
                hist.append((t, h))
        if not hist:
            return self._placeholder("SD-R-12: no iteration history persisted for this run")

        n = len(hist)
        ramp = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, n))     # one hue, light → dark
        panels = [("res_primal", "primal residual", True), ("res_dual", "dual residual", True),
                  ("gap_abs", "absolute gap", True), ("step_length", "step length", False)]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
        fig.suptitle(self._title("Convergence traces per year (SD-R-12)"), fontsize=13, fontweight="bold")
        for ax, (key, name, log) in zip(axes.flat, panels):
            for i, (t, h) in enumerate(hist):
                try:
                    it = np.asarray(h["iter"], dtype=float)
                    y  = np.asarray(h[key], dtype=float)
                except (KeyError, ValueError, IndexError):
                    continue
                if log:
                    y = np.maximum(y, 1e-16)
                ax.plot(it, y, color=ramp[i], lw=1.2, marker="o" if len(it) == 1 else None, ms=4)
            if log:
                ax.set_yscale("log")
            ax.set_title(name, loc="left", fontsize=10)
            ax.set_xlabel("iteration")
            _apply_modern_style(ax)
        sm = ScalarMappable(norm=Normalize(0, max(n - 1, 1)), cmap=ListedColormap(ramp))
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), pad=0.02, fraction=0.03)
        ticks = sorted({0, n // 2, n - 1})
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(hist[i][0]) for i in ticks])
        cbar.set_label("year")
        return fig

    # ------------------------------------------------------------------
    # Objective share (SD-R-13 / 14 / 15)
    # ------------------------------------------------------------------

    @staticmethod
    def _dot_rank(ax, y, x, color, label=None, size=48):
        """One ranked dot series with hairline stems from the axis minimum."""
        ax.scatter(x, y, s=size, color=color, zorder=3, edgecolors="none", label=label)

    @staticmethod
    def _n_unidentified(fam) -> int:
        if fam is None or fam.empty:
            return 0
        return int((fam["identity"].to_numpy() < 0).sum())

    def _fig_objective_share_by_sector(self, *, hue=None, timesteps=None, regions=None):
        """SD-R-13: objective share per sector key, one panel per family."""
        import matplotlib.pyplot as plt

        runs = hue if hue else [(self.label or "run", self, _C["blue"])]
        host = runs[0][1]
        fig, axes = plt.subplots(1, len(FAMILIES), figsize=(6 * len(FAMILIES), 7.5), squeeze=False)
        axes = axes[0]
        fig.suptitle(self._title("Objective share by sector key (SD-R-13)", regions),
                     fontsize=13, fontweight="bold")
        drew_any = False
        for j, fam in enumerate(FAMILIES):
            ax = axes[j]
            host_fam = host._family(fam, regions)
            host_share = share_by_key(host_fam, top=_TOP_KEYS)
            if host_share is None or host_share.empty:
                ax.text(0.5, 0.5, "no rows", transform=ax.transAxes, ha="center", va="center", color=_C["grey"])
                ax.set_title(FAMILY_LABELS[fam], loc="left", fontsize=10)
                _apply_modern_style(ax)
                continue
            drew_any = True
            keys = list(host_share["sector_key"])
            ys = np.arange(len(keys))[::-1].astype(float)
            n_runs = len(runs)
            xmin = 1.0
            for k, (lbl, dg, color) in enumerate(runs):
                s = share_by_key(dg._family(fam, regions))
                if s is None or s.empty:
                    continue
                s = s.set_index("sector_key")["share"]
                named = [kk for kk in keys if kk != "other"]
                shares = np.array([float(s.get(kk, 0.0)) for kk in named])
                if "other" in keys:
                    shares = np.append(shares, max(1.0 - shares.sum(), 0.0))
                pos = shares > 0
                xmin = min(xmin, float(shares[pos].min())) if pos.any() else xmin
                off = (k - (n_runs - 1) / 2) * 0.28 if n_runs > 1 else 0.0
                ax.hlines(ys[pos] + off, 1e-12, shares[pos], color=_C["grey"], lw=0.8, alpha=0.6)
                self._dot_rank(ax, ys[pos] + off, shares[pos], color, label=lbl if hue else None)
            lo = max(xmin / 3, 1e-6)
            ax.set_xscale("log")
            ax.set_xlim(lo, 1.0)
            ax.set_yticks(ys)
            ax.set_yticklabels([_short(kk, 32) for kk in keys], fontsize=8)
            ax.set_ylim(-0.8, len(keys) - 0.2)
            ax.set_xlabel("objective share of the family (log)")
            total = float(np.nansum(host_fam["z"].to_numpy(dtype=float) ** 2)) if host_fam is not None else 0.0
            ax.set_title(f"{FAMILY_LABELS[fam]} · Σz² = {total:.4g}", loc="left", fontsize=10)
            n_unid = self._n_unidentified(host_fam)
            if n_unid:
                ax.text(0.0, -0.14, f"{n_unid:,} rows without identity (in totals; keyed by variant)",
                        transform=ax.transAxes, fontsize=7, color=_C["grey"])
            _apply_modern_style(ax)
        if not drew_any:
            plt.close(fig)
            return self._placeholder("SD-R-13: no frames available (acc / row_meta missing)", regions)
        if hue:
            axes[0].legend(fontsize=7, frameon=False, loc="lower right")
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        return fig

    def _fig_top_identities(self, *, hue=None, timesteps=None, regions=None):
        """SD-R-14: top identities per family by Σ_t z², annotated with Σ_t|Δ|."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(FAMILIES), figsize=(7.5 * len(FAMILIES), 9.5), squeeze=False)
        axes = axes[0]
        fig.suptitle(self._title(f"Top {_TOP_IDS} identities by Σ_t z² (SD-R-14)", regions),
                     fontsize=13, fontweight="bold")
        drew_any = False
        for j, fam in enumerate(FAMILIES):
            ax = axes[j]
            f = self._family(fam, regions)
            r = rank_identities(f, top=_TOP_IDS) if f is not None else None
            if r is None or r.empty:
                ax.text(0.5, 0.5, "no rows", transform=ax.transAxes, ha="center", va="center", color=_C["grey"])
                ax.set_title(FAMILY_LABELS[fam], loc="left", fontsize=10)
                _apply_modern_style(ax)
                continue
            drew_any = True
            ys = np.arange(len(r))[::-1].astype(float)
            x = np.maximum(r["share"].to_numpy(dtype=float), 1e-12)
            lo = max(float(x.min()) / 3, 1e-6)
            ax.hlines(ys, 1e-12, x, color=_C["grey"], lw=0.8, alpha=0.6)
            self._dot_rank(ax, ys, x, _FAMILY_COLOR[fam])
            for yi, xi, row in zip(ys, x, r.itertuples(index=False)):
                ax.text(xi * 1.15, yi,
                        f"{row.sum_abs_delta:,.0f} kt · {row.n_years}y · max|z| {row.z_absmax:.1f} ({row.year_absmax})",
                        va="center", fontsize=6.5, color=_C["grey"])
            ax.set_xscale("log")
            ax.set_xlim(lo, 1.0)
            ax.set_yticks(ys)
            ax.set_yticklabels([_short(l, 45) for l in r["label"]], fontsize=7.5)
            ax.set_ylim(-0.8, len(r) - 0.2)
            ax.set_xlabel("objective share of the family (log)")
            ax.set_title(f"{FAMILY_LABELS[fam]} · Σz² = {r.attrs.get('zsq_total', float('nan')):.4g}",
                         loc="left", fontsize=10)
            n_unid = r.attrs.get("n_unidentified", 0)
            if n_unid:
                ax.text(0.0, -0.10, f"{n_unid:,} rows without identity excluded from the ranking",
                        transform=ax.transAxes, fontsize=7, color=_C["grey"])
            _apply_modern_style(ax)
        if not drew_any:
            plt.close(fig)
            return self._placeholder("SD-R-14: no frames available (acc / row_meta missing)", regions)
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        return fig

    def _fig_identity_profiles(self, family: str, fig_id: str, regions=None):
        """Shared body of SD-R-15a/b/c: z_t over years for the top identities."""
        import matplotlib.pyplot as plt

        f = self._family(family, regions)
        r = rank_identities(f, top=_N_PROFILES) if f is not None else None
        if r is None or r.empty:
            return self._placeholder(f"{fig_id}: no {FAMILY_LABELS[family]} available", regions)
        P = profiles(f, r["identity"].tolist(), years=self._years)
        ncols, nrows = 4, int(np.ceil(len(r) / 4))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.0 * nrows + 1), sharex=True, sharey=True,
                                 squeeze=False)
        fig.suptitle(self._title(f"z over years — top {len(r)} {FAMILY_LABELS[family]} ({fig_id})", regions),
                     fontsize=13, fontweight="bold")
        x = self._ts_index()
        for i, row in enumerate(r.itertuples(index=False)):
            ax = axes.flat[i]
            z = P.loc[row.identity].to_numpy(dtype=float)
            ax.axhline(0, color=_C["grey"], lw=0.8)
            ax.axhline(3, color=_C["grey"], lw=0.8, ls="--")
            ax.axhline(-3, color=_C["grey"], lw=0.8, ls="--")
            ax.plot(x, z, color=_FAMILY_COLOR[family], lw=1.6, marker="o", ms=3)
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_title(f"#{row.rank} {_short(row.label, 38)} · {100 * row.share:.1f} %",
                         loc="left", fontsize=8.5)
            _apply_modern_style(ax)
        for ax in axes.flat[len(r):]:
            ax.set_visible(False)
        step = max(1, len(x) // 6)
        for c in range(ncols):
            ax = axes[-1][c]
            ax.set_xticks(x[::step])
            ax.set_xticklabels(self._ts_labels_for_xticks()[::step], rotation=45, ha="right", fontsize=7)
        axes[0][0].set_ylabel("z (symlog)")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def _fig_profiles_edges(self, *, hue=None, timesteps=None, regions=None):
        """SD-R-15a."""
        return self._fig_identity_profiles("edges", "SD-R-15a", regions)

    def _fig_profiles_sum_rows(self, *, hue=None, timesteps=None, regions=None):
        """SD-R-15b."""
        return self._fig_identity_profiles("sum_rows", "SD-R-15b", regions)

    def _fig_profiles_soft_ratios(self, *, hue=None, timesteps=None, regions=None):
        """SD-R-15c."""
        return self._fig_identity_profiles("soft_ratios", "SD-R-15c", regions)

    # ------------------------------------------------------------------
    # Rocket (SD-T-07)
    # ------------------------------------------------------------------

    def _fig_rocket(self, *, hue=None, timesteps=None, regions=None):
        """SD-T-07: prior vs posterior log–log scatters at three years.

        Row c: ``log10 |G·a_prior(t)|`` vs ``log10 |c*|`` on sum rows;
        row a: ``log10 |a_prior(t)|`` vs ``log10 |a*|`` on the year's edges.
        Values are floored at ``_LOG_FLOOR`` before the log so a handful of
        numerical zeros cannot squash the populated decades; colour = ``|z|``
        (sequential, log scale, clipped to [1e-2, 1e2]); the top identities of
        the family are labelled where they appear.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm

        ts_triple = self._resolve_timesteps(timesteps, regions)
        specs = [
            ("c-space · sum rows", self._family("sum_rows", regions), "lhs_prior_t", "lhs_post",
             "log₁₀ |G·a_prior(t)|", "log₁₀ |c*|"),
            ("a-space · edges", self._family("edges", regions), "a_prior", "a_post",
             "log₁₀ |a_prior(t)|", "log₁₀ |a*|"),
        ]
        if all(fam is None for _, fam, *_ in specs):
            return self._placeholder("SD-T-07: no frames available (acc / row_meta missing)", regions)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), squeeze=False)
        fig.suptitle(self._title("Rocket: prior vs posterior, colour |z| (SD-T-07)", regions),
                     fontsize=13, fontweight="bold")
        cmap = plt.get_cmap("YlOrRd")
        norm = LogNorm(vmin=1e-2, vmax=1e2, clip=True)
        worst = ts_triple[1]

        for r, (name, fam, xcol, ycol, xlab, ylab) in enumerate(specs):
            if fam is None or fam.empty or xcol not in fam.columns:
                for col in range(3):
                    axes[r][col].text(0.5, 0.5, f"{name}: no rows", transform=axes[r][col].transAxes,
                                      ha="center", va="center", color=_C["grey"])
                    _apply_modern_style(axes[r][col])
                continue
            top = rank_identities(fam, top=_N_ROCKET_LABELS)
            top_ids = set(top["identity"].tolist()) if top is not None and not top.empty else set()
            per_year, lo, hi = {}, np.inf, -np.inf
            for t in ts_triple:
                sub = fam[fam["year"] == str(t)]
                xs = np.log10(np.maximum(np.abs(sub[xcol].to_numpy(dtype=float)), _LOG_FLOOR))
                ys = np.log10(np.maximum(np.abs(sub[ycol].to_numpy(dtype=float)), _LOG_FLOOR))
                zz = np.abs(sub["z"].to_numpy(dtype=float))
                per_year[t] = (sub, xs, ys, zz)
                if xs.size:
                    lo = min(lo, float(np.nanmin(xs)), float(np.nanmin(ys)))
                    hi = max(hi, float(np.nanmax(xs)), float(np.nanmax(ys)))
            if not np.isfinite(lo) or not np.isfinite(hi):
                lo, hi = np.log10(_LOG_FLOOR), 6.0
            pad = 0.3
            for col, t in enumerate(ts_triple):
                ax = axes[r][col]
                sub, xs, ys, zz = per_year[t]
                ok = np.isfinite(zz)
                sc = None
                if ok.any():
                    sc = ax.scatter(xs[ok], ys[ok], c=zz[ok], cmap=cmap, norm=norm,
                                    s=8, alpha=0.65, edgecolors="none")
                if (~ok).any():   # σ = 0 → z undefined: grey
                    ax.scatter(xs[~ok], ys[~ok], color=_C["grey"], s=6, alpha=0.4, edgecolors="none")
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=_C["grey"], lw=0.9, ls="--")
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
                if top_ids:
                    m = sub["identity"].isin(top_ids).to_numpy()
                    for xi, yi, lab in zip(xs[m], ys[m], sub["label"].to_numpy()[m]):
                        ax.annotate(_short(str(lab).replace("|", "·"), 30), (xi, yi),
                                    xytext=(3, 3), textcoords="offset points",
                                    fontsize=6, color=_C["navy"])
                if sc is not None and col == len(ts_triple) - 1:
                    fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05, label="|z|")
                ax.set_xlabel(xlab, fontsize=8)
                if col == 0:
                    ax.set_ylabel(ylab)
                if r == 0:
                    tag = "  · worst Σz²" if (timesteps is None and t == worst and col == 1) else ""
                    ax.set_title(f"{t}{tag}", fontsize=10, fontweight="bold")
                ax.text(0.02, 0.97, f"{name} · n={int(ok.sum() + (~ok).sum())}",
                        transform=ax.transAxes, fontsize=8, color=_C["grey"], va="top")
                _apply_modern_style(ax)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig
