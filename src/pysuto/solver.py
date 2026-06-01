"""
solver.py
=========
Stage 3b — Clarabel-backed σ-weighted QP solver under the structural model.

Loads a persisted :class:`pysuto.constraints.ConstraintRun`, builds a Clarabel
QP per period using :class:`StructG`'s materialised CSR view, and produces
one :class:`OptimResult` per period.

The pre-refactor block/legacy split is collapsed into a single loop: there is
no per-period column subset (``StructG`` has no time axis on the column
side), and per-period activation is read off ``ConstraintRun.active`` rather
than the ``SIGMA_C_INACTIVE`` / ``BOUND_C_INACTIVE`` sentinels.

Public API
----------
- :class:`OptimResult`     — per-period solution + diagnostics
- :class:`Prior`           — protocol; :class:`AggregationPrior`,
                             :class:`WarmStartPrior`
- :class:`QPProblem`       — per-period Clarabel problem assembly
- :class:`ClarabelSolver`  — load, feasibility-check, ``solve_all``, persist
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np
import pandas as pd
import scipy.sparse as sp

import clarabel

from ._conditioning import condition_prior
from .constraints import ConstraintRun
from .registry import ConstraintRunRegistry, OptimRunRegistry
from .logging import get_logger

logger = get_logger("pysuto.solver")


# =============================================================================
# OptimResult
# =============================================================================

ITER_DTYPE = np.dtype([
    ("iter",        np.int32),
    ("cost_primal", np.float64),
    ("cost_dual",   np.float64),
    ("gap_abs",     np.float64),
    ("res_primal",  np.float64),
    ("res_dual",    np.float64),
    ("ktratio",     np.float64),
    ("mu",          np.float64),
    ("step_length", np.float64),
])


@dataclass
class OptimResult:
    """One period's solver output.

    ``a_star`` is the full-length optimum on every structural edge. ``eps_opt``
    / ``c_star`` / ``residuals`` are indexed against ``ConstraintRun.row_meta``
    (the active rows that participated in this period's QP carry the meaningful
    values; inactive rows contribute zero ε and zero residual by construction).
    """
    timestep:      str
    a_star:        np.ndarray      # (n_edges,)
    eps_opt:       np.ndarray      # (n_active_free,) — slack on active free rows
    c_star:        np.ndarray      # (n_constraints,) — c + ε (ε=0 on hard/inactive)
    residuals:     np.ndarray      # (n_constraints,) — G @ a_star - c (per-row)
    objective:     float
    iterations:    int
    solve_time_s:  float
    solver_status: str
    convergence:   dict
    iter_history:  Optional[np.ndarray] = None
    # active mask used at this period (for diagnostics / downstream uncprop)
    active:        Optional[np.ndarray] = None  # (n_constraints,) bool

    # Backwards-compat alias so old callers reading ``.F_opt`` keep working
    # without an explicit rename pass; the canonical name is ``a_star``.
    @property
    def F_opt(self) -> np.ndarray:
        return self.a_star


# =============================================================================
# Prior seam (deepening #2)
# =============================================================================

class Prior(Protocol):
    """Strategy for the period prior mean ``a_0(t)``.

    ``history`` is the ordered list of solved ``a_star`` from previous
    periods (length ``t_idx`` on call ``mean(t_idx, ...)``).
    """
    def mean(self, t_idx: int, history: list[np.ndarray]) -> np.ndarray: ...


@dataclass
class AggregationPrior:
    """``a_0(t) = a_agg[:, t]``. No temporal smoothing — re-index baseline."""
    a_agg: np.ndarray   # (n_edges, n_timesteps)

    def mean(self, t_idx: int, history: list[np.ndarray]) -> np.ndarray:
        return self.a_agg[:, t_idx]


@dataclass
class WarmStartPrior:
    """``a_0(0) = a0``; ``a_0(t>0) = a_star(t-1)``."""
    a0: np.ndarray      # (n_edges,)

    def mean(self, t_idx: int, history: list[np.ndarray]) -> np.ndarray:
        if t_idx == 0 or not history:
            return self.a0
        return history[-1]


# =============================================================================
# QPProblem — per-period assembly
# =============================================================================

@dataclass
class QPProblem:
    """Bundle of arrays / objects Clarabel needs for one period."""
    P:           sp.csc_matrix
    q:           np.ndarray
    A:           sp.csc_matrix
    b:           np.ndarray
    cones:       list
    perm:        np.ndarray         # row permutation [free | zero | nonneg] within active rows
    active_idx:  np.ndarray         # indices of active rows in original row order
    n_active_free: int
    n_active_zero: int
    n_active_nonneg: int
    n_edges:     int

    @classmethod
    def build(
        cls,
        run:    ConstraintRun,
        t_idx:  int,
        a_0:    np.ndarray,
        sigma_a: np.ndarray,
        sigma_c_floor: float = 1e-9,
    ) -> "QPProblem":
        G_t       = run.G.at(t_idx).tocsr()
        c_t       = np.asarray(run.c[:, t_idx],       dtype=float)
        sigma_c_t = np.asarray(run.sigma_c[:, t_idx], dtype=float)
        active    = np.asarray(run.active[:, t_idx],  dtype=bool)
        cone_type = run.row_meta["cone_type"].to_numpy()

        # Filter rows by activation; keep original-row index for back-mapping.
        active_idx = np.where(active)[0]
        if active_idx.size:
            G_t       = G_t[active_idx]
            c_t       = c_t[active_idx]
            sigma_c_t = sigma_c_t[active_idx]
            cone_t    = cone_type[active_idx]
        else:
            G_t       = sp.csr_matrix((0, G_t.shape[1]))
            c_t       = np.empty(0)
            sigma_c_t = np.empty(0)
            cone_t    = np.empty(0, dtype=cone_type.dtype)

        # Cone permutation [free; zero; nonneg].
        idx_free   = np.where(cone_t == "free")[0]
        idx_zero   = np.where(cone_t == "zero")[0]
        idx_nonneg = np.where(cone_t == "nonneg")[0]
        perm = np.concatenate([idx_free, idx_zero, idx_nonneg])

        n_free, n_zero, n_nonneg = idx_free.size, idx_zero.size, idx_nonneg.size
        n_a = run.n_edges
        M_t = G_t.shape[0]

        G_p = G_t[perm] if M_t else G_t

        # Slack column block: S = [-I_{n_free}; 0]
        if n_free:
            S_top = -sp.eye(n_free, format="csr")
            S_bot = sp.csr_matrix((n_zero + n_nonneg, n_free))
            S = sp.vstack([S_top, S_bot], format="csr")
        else:
            S = sp.csr_matrix((M_t, 0))

        A = sp.hstack([G_p, S], format="csc")

        # Hessian: 2 · diag([Σ_a^{-1}; Σ_c^{-1}_free])
        inv_sa = 1.0 / (sigma_a ** 2)
        if n_free:
            sc = sigma_c_t[perm[:n_free]]
            sc = np.where(np.isnan(sc), sigma_c_floor, sc)
            sc = np.maximum(sc, sigma_c_floor)
            inv_scf = 1.0 / (sc ** 2)
        else:
            inv_scf = np.zeros(0)
        P_diag = 2.0 * np.concatenate([inv_sa, inv_scf])
        P = sp.diags(P_diag, format="csc")

        # Linear term q = [-2 Σ_a^{-1} a_0; 0]
        q = np.concatenate([-2.0 * inv_sa * a_0, np.zeros(n_free)])

        # Permuted RHS
        b = c_t[perm] if M_t else np.empty(0)

        cones: list = []
        if n_free   > 0: cones.append(clarabel.ZeroConeT(n_free))
        if n_zero   > 0: cones.append(clarabel.ZeroConeT(n_zero))
        if n_nonneg > 0: cones.append(clarabel.NonnegativeConeT(n_nonneg))

        return cls(
            P=P, q=q, A=A, b=b, cones=cones, perm=perm,
            active_idx=active_idx,
            n_active_free=n_free, n_active_zero=n_zero, n_active_nonneg=n_nonneg,
            n_edges=n_a,
        )


# =============================================================================
# ClarabelSolver
# =============================================================================

_OK_STATUSES = {"Solved", "AlmostSolved"}


class ClarabelSolver:
    """
    σ-weighted QP solver wrapping Clarabel for Stage 3b.

    A single loop over periods. Per period, :class:`QPProblem.build` reads the
    period slice from :class:`StructG` + 2-D ``c`` / ``sigma_c`` / ``active``,
    drops inactive rows from the QP entirely (replacing the pre-refactor σ_c
    sentinels), and produces a fresh Clarabel problem. The KKT factorisation is
    not reused across periods because the operator's data tape varies (e.g.
    ratio coefficient ``r(t)``) and the active row subset may change too.

    The prior seam (``prior=AggregationPrior(a_agg)`` vs
    ``WarmStartPrior(run.a0)``) is selected per call.

    Σ_a conditioning
    ----------------
    A floor and a ceiling on ``sigma_a`` (per period) bound the condition
    number of ``Σ_a^{-1}``::

        floor   = max(σ_a_floor_abs,  σ_a_floor_rel · |a_0|)
        ceiling = σ_a_ceiling_factor · max(finite σ_a)
    """

    def __init__(
        self,
        const_dir: str = "data/proc/const",
        opt_dir:   str = "data/proc/opt",
        sigma_a_floor_abs:      float = 1e-8,
        sigma_a_floor_rel:      float = 1e-5,
        sigma_a_ceiling_factor: float = 1e5,
        sigma_c_floor:          float = 1e-9,
        tol_feas:               float = 1e-8,
        tol_gap_abs:            float = 1e-8,
        tol_gap_rel:            float = 1e-8,
        max_iter:               int   = 200,
        verbose_first_solve:    bool  = True,
        capture_iter_history:   bool  = True,
    ) -> None:
        self.const_dir              = const_dir
        self.opt_dir                = opt_dir
        self.sigma_a_floor_abs      = sigma_a_floor_abs
        self.sigma_a_floor_rel      = sigma_a_floor_rel
        self.sigma_a_ceiling_factor = sigma_a_ceiling_factor
        self.sigma_c_floor          = sigma_c_floor
        self.tol_feas               = tol_feas
        self.tol_gap_abs            = tol_gap_abs
        self.tol_gap_rel            = tol_gap_rel
        self.max_iter               = max_iter
        self.verbose_first_solve    = verbose_first_solve
        self.capture_iter_history   = capture_iter_history

        self._const_registry = ConstraintRunRegistry(const_dir)
        self._opt_registry   = OptimRunRegistry(opt_dir)

    # ------------------------------------------------------------------

    def load_run(self, run_id: str) -> ConstraintRun:
        logger.info("Loading ConstraintRun '%s' from %s", run_id, self.const_dir)
        cr = self._const_registry.load(run_id)
        logger.info(
            "  StructG (%d constraints, %d edges, %d timesteps); active mask present",
            cr.G.n_constraints, cr.G.n_edges, cr.G.n_timesteps,
        )
        return cr

    # ------------------------------------------------------------------

    def _settings(self, verbose: bool) -> clarabel.DefaultSettings:
        s = clarabel.DefaultSettings()
        s.verbose     = verbose
        s.max_iter    = self.max_iter
        s.tol_feas    = self.tol_feas
        s.tol_gap_abs = self.tol_gap_abs
        s.tol_gap_rel = self.tol_gap_rel
        return s

    def _condition_prior(
        self,
        a_0:     np.ndarray,
        sigma_a: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return condition_prior(
            a_0, sigma_a,
            floor_abs      = self.sigma_a_floor_abs,
            floor_rel      = self.sigma_a_floor_rel,
            ceiling_factor = self.sigma_a_ceiling_factor,
        )

    @staticmethod
    def _info_to_dict(info) -> dict:
        ls = getattr(info, "linsolver", None)
        return {
            "cost_primal":     float(info.cost_primal),
            "cost_dual":       float(info.cost_dual),
            "gap_abs":         float(info.gap_abs),
            "gap_rel":         float(info.gap_rel),
            "res_primal":      float(info.res_primal),
            "res_dual":        float(info.res_dual),
            "res_primal_inf":  float(info.res_primal_inf),
            "res_dual_inf":    float(info.res_dual_inf),
            "ktratio":         float(info.ktratio),
            "mu":              float(info.mu),
            "sigma":           float(info.sigma),
            "step_length":     float(info.step_length),
            "iterations":      int(info.iterations),
            "solve_time":      float(info.solve_time),
            "status":          str(info.status),
            "linsolver_name":  str(getattr(ls, "name",   "")) if ls is not None else "",
            "linsolver_nnzA":  int(getattr(ls, "nnzA",   0))  if ls is not None else 0,
            "linsolver_nnzL":  int(getattr(ls, "nnzL",   0))  if ls is not None else 0,
        }

    # ------------------------------------------------------------------
    # Feasibility pre-check
    # ------------------------------------------------------------------

    def feasibility_check(self, run: ConstraintRun, t_idx: int) -> None:
        """Halt if the active hard rows + flow bounds at period ``t_idx`` are
        infeasible. Solves a feasibility-only QP ``min ½ ε ‖a‖²`` on the hard
        subsystem (tiny PSD P; status, not optimum, is the signal)."""
        G_t       = run.G.at(t_idx).tocsr()
        c_t       = np.asarray(run.c[:, t_idx], dtype=float)
        active    = np.asarray(run.active[:, t_idx], dtype=bool)
        cone_type = run.row_meta["cone_type"].to_numpy()

        # Only active hard rows participate.
        hard_active = active & (np.isin(cone_type, ["zero", "nonneg"]))
        idx_active_hard = np.where(hard_active)[0]
        hard_ct = cone_type[idx_active_hard]
        idx_zero   = idx_active_hard[hard_ct == "zero"]
        idx_nonneg = idx_active_hard[hard_ct == "nonneg"]
        n_a = run.n_edges
        n_zero, n_nonneg = idx_zero.size, idx_nonneg.size

        G_zero   = G_t[idx_zero]   if n_zero   else sp.csr_matrix((0, n_a))
        G_nonneg = G_t[idx_nonneg] if n_nonneg else sp.csr_matrix((0, n_a))

        I = sp.eye(n_a, format="csr")
        Aub = sp.vstack([G_nonneg, I, -I], format="csr")
        bub = np.concatenate([c_t[idx_nonneg], run.u, -run.l])

        A = sp.vstack([G_zero, Aub], format="csc")
        b = np.concatenate([c_t[idx_zero], bub])

        cones_fc: list = []
        if n_zero       > 0: cones_fc.append(clarabel.ZeroConeT(n_zero))
        if Aub.shape[0] > 0: cones_fc.append(clarabel.NonnegativeConeT(Aub.shape[0]))

        P_fc = sp.diags(np.full(n_a, 1e-10), format="csc")
        q_fc = np.zeros(n_a)

        settings = self._settings(verbose=False)
        settings.tol_feas    = 1e-7
        settings.tol_gap_abs = 1e-7
        settings.tol_gap_rel = 1e-7

        solver = clarabel.DefaultSolver(P_fc, q_fc, A, b, cones_fc, settings)
        sol    = solver.solve()
        status = str(sol.status)

        if status not in _OK_STATUSES:
            self._raise_infeasible(run, t_idx, status, sol)

    def _raise_infeasible(self, run, t_idx, status, sol):
        try:
            x = np.asarray(sol.x)
            if x.size >= run.n_edges:
                a = x[: run.n_edges]
                G_t = run.G.at(t_idx).tocsr()
                resid = G_t @ a - np.asarray(run.c[:, t_idx])
                hard = run.row_meta["is_hard"].to_numpy()
                active = np.asarray(run.active[:, t_idx], dtype=bool)
                hard_resid = np.where(hard & active, np.abs(resid), 0.0)
                worst = np.argsort(-hard_resid)[:5]
                worst_msg = "; ".join(
                    f"row={int(i)} cid={run.row_meta.iloc[int(i)]['constraint_id']!r} "
                    f"|residual|={hard_resid[i]:.3e}"
                    for i in worst
                )
            else:
                worst_msg = "(no diagnostic — solution vector unavailable)"
        except Exception as e:
            worst_msg = f"(diagnostic failed: {e})"

        t_label = (run.timesteps[t_idx] if 0 <= t_idx < len(run.timesteps)
                   else f"t_idx={t_idx}")
        raise RuntimeError(
            f"Feasibility check FAILED at period {t_label!r}: "
            f"Clarabel status={status}. "
            f"Top hard-row offenders (largest |G·a - c|): {worst_msg}. "
            f"Widen the offending ratio bounds in the params CSV "
            f"(e.g. toward [0, 1]) and rebuild the ConstraintRun."
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve_all(
        self,
        run_id:    str,
        timesteps: Optional[list] = None,
        feasibility_check_first_only: bool = True,
        prior:     Optional[Prior] = None,
    ) -> dict[str, OptimResult]:
        """Solve every requested period in order.

        ``prior`` defaults to :class:`WarmStartPrior(run.a0)`. Pass
        :class:`AggregationPrior(a_agg)` to disable warm-start temporal
        coupling — that mode is the validation anchor against the
        pre-refactor baseline.
        """
        run = self.load_run(run_id)

        if timesteps is None:
            ts_labels = list(run.timesteps)
        else:
            ts_labels = [str(t) for t in timesteps]
            missing = set(ts_labels) - set(run.timesteps)
            if missing:
                raise KeyError(f"timesteps not in run: {sorted(missing)}")
        t_idx_for = {lbl: i for i, lbl in enumerate(run.timesteps)}

        if prior is None:
            prior = WarmStartPrior(run.a0)

        # Feasibility pre-check
        check_labels = ts_labels[:1] if feasibility_check_first_only else ts_labels
        logger.info(
            "Feasibility pre-check on %d/%d periods%s…",
            len(check_labels), len(ts_labels),
            " (first only)" if feasibility_check_first_only else "",
        )
        for lbl in check_labels:
            self.feasibility_check(run, t_idx_for[lbl])
        logger.info("Feasibility pre-check passed.")

        results: dict[str, OptimResult] = {}
        history: list[np.ndarray] = []

        for k, t_label in enumerate(ts_labels):
            t_idx = t_idx_for[t_label]
            a_0_raw = prior.mean(t_idx, history)
            sigma_a_raw = run.sigma_a[:, t_idx]
            a_0, sigma_a = self._condition_prior(a_0_raw, sigma_a_raw)

            qp = QPProblem.build(
                run, t_idx, a_0, sigma_a, sigma_c_floor=self.sigma_c_floor,
            )

            t0 = time.perf_counter()
            verbose = self.verbose_first_solve and k == 0
            settings = self._settings(verbose=verbose)
            solver = clarabel.DefaultSolver(qp.P, qp.q, qp.A, qp.b, qp.cones, settings)

            hist: list[tuple] = []
            if self.capture_iter_history:
                def _cb(info, _hist=hist):
                    _hist.append((
                        int(info.iterations),
                        float(info.cost_primal), float(info.cost_dual),
                        float(info.gap_abs),
                        float(info.res_primal), float(info.res_dual),
                        float(info.ktratio),    float(info.mu),
                        float(info.step_length),
                    ))
                    return False
                solver.set_termination_callback(_cb)

            sol     = solver.solve()
            elapsed = time.perf_counter() - t0
            status  = str(sol.status)

            if self.capture_iter_history:
                solver.unset_termination_callback()

            convergence  = self._info_to_dict(solver.get_info())
            iter_history = (
                np.array(hist, dtype=ITER_DTYPE) if hist else None
            )

            if status not in _OK_STATUSES:
                raise RuntimeError(
                    f"Period {t_label!r}: Clarabel status={status}; halting "
                    f"(no fallback). iterations={sol.iterations}"
                )

            x       = np.asarray(sol.x)
            a_star  = x[: qp.n_edges]
            eps_opt = x[qp.n_edges:] if qp.n_active_free else np.zeros(0)

            # Project ε / c_star / residuals back to the full row order.
            M = run.n_constraints
            eps_full       = np.zeros(M)
            c_star_full    = np.asarray(run.c[:, t_idx], dtype=float).copy()
            residuals_full = np.zeros(M)
            if qp.active_idx.size:
                # Map the cone-permuted slacks back to active-row positions.
                if qp.n_active_free:
                    active_free_pos = qp.active_idx[qp.perm[:qp.n_active_free]]
                    eps_full[active_free_pos] = eps_opt
                # Per-row residuals on active rows.
                G_t = run.G.at(t_idx).tocsr()
                residuals_full[qp.active_idx] = (
                    G_t[qp.active_idx] @ a_star - np.asarray(run.c[qp.active_idx, t_idx])
                )
                c_star_full = c_star_full + eps_full

            results[t_label] = OptimResult(
                timestep      = t_label,
                a_star        = a_star,
                eps_opt       = eps_opt,
                c_star        = c_star_full,
                residuals     = residuals_full,
                objective     = float(sol.obj_val),
                iterations    = int(sol.iterations),
                solve_time_s  = elapsed,
                solver_status = status,
                convergence   = convergence,
                iter_history  = iter_history,
                active        = np.asarray(run.active[:, t_idx], dtype=bool),
            )

            logger.info(
                "t=%s  status=%s  iters=%d  obj=%.4e  gap=%.2e  dres=%.2e  step=%.2f  time=%.2fs  (%d active rows / %d total)",
                t_label, status, sol.iterations, sol.obj_val,
                convergence["gap_abs"], convergence["res_dual"],
                convergence["step_length"], elapsed,
                qp.active_idx.size, run.n_constraints,
            )

            history.append(a_star)

        return results

    def solve_timestep(
        self,
        run_id:  str,
        t_label: str,
        prior:   Optional[Prior] = None,
    ) -> OptimResult:
        """Convenience: solve a single period."""
        return self.solve_all(run_id, timesteps=[t_label], prior=prior)[t_label]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(
        self,
        run_id:     str,
        results:    dict[str, OptimResult],
        output_dir: Optional[str] = None,
        const_run_id: Optional[str] = None,
        label:      str = "",
    ) -> str:
        const_id = const_run_id or run_id
        registry = OptimRunRegistry(output_dir) if output_dir else self._opt_registry
        opt_run_id = registry.register(
            results,
            parents = {
                "inputs":   {},
                "upstream": {"const_run_id": const_id},
            },
            label = label,
        )
        logger.info("OptimRun '%s' persisted to %s",
                    opt_run_id, registry.storage_dir)
        return opt_run_id
