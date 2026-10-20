"""
solver.py
=========
Stage 3b — Clarabel-backed σ-weighted QP solver.

Loads a persisted :class:`pysuto.constraints.ConstraintRun`, transforms it
to Clarabel's standard form, runs a feasibility pre-check on the hard rows,
then solves the QP per timestep using ``solver.update(b=...)`` between
timesteps to retain the KKT factorisation (the practical warm-start
mechanism for this problem since only ``c`` varies across time). Under
``prior_mode='cold'`` with ``n_workers > 1`` the timesteps are instead solved
concurrently (**parallel solve**, #70), each on its own Clarabel instance.

Cone layout (correcting README §665-667):

* ``ZeroConeT(n_free)`` — soft equality residuals ``Gx - ε = c`` (slack
  ε is a free decision variable inside ``x``; the residual itself is
  exact equality).
* ``ZeroConeT(n_zero)`` — hard mass-balance equalities.
* ``NonnegativeConeT(n_nonneg)`` — hard ratio bounds and flow lower bounds
  (ADR 0039; upper bounds and the zero pin reach only the feasibility pre-check).

Public API
----------
- :class:`OptimResult` — per-timestep solution + diagnostics
- :class:`ClarabelSolver` — load, feasibility-check, solve_all, persist
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional

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

# Structured dtype for per-iteration convergence trace.
# Columns mirror Clarabel's verbose console output.
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
    """One timestep's solver output."""
    timestep:      str
    F_opt:         np.ndarray      # (E,)
    eps_opt:       np.ndarray      # (n_free,) — slack on soft rows in original row order
    c_star:        np.ndarray      # (M,) — c + ε (ε=0 for hard rows). Diagnostic.
    residuals:     np.ndarray      # (M,) — G @ F_opt - c (original row order)
    objective:     float           # J − Σ(a_prior/σ_a)², the flows form's cost, under
                                   # either qp_variables (#33)
    iterations:    int
    solve_time_s:  float
    solver_status: str             # str(clarabel SolverStatus enum)
    convergence:   dict            # final get_info() snapshot (JSON-safe scalars)
    iter_history:  Optional[np.ndarray] = None  # structured array of ITER_DTYPE
    # Block path only: F_opt/residuals/c_star are the year-local slices over
    # these global indices. None for the legacy full-vector path.
    cols_t:        Optional[np.ndarray] = None  # global edge cols for this t
    rows_t:        Optional[np.ndarray] = None  # global constraint rows for this t
    # The per-period prior mean actually used in the objective q-vector.
    # Block path: year-local (cols_t order); warm = a_post(t-1), cold = f_agg(t).
    # Legacy path: full-length (E,). None for runs predating the warm-start fix.
    a_prior:       Optional[np.ndarray] = None


# =============================================================================
# ClarabelSolver
# =============================================================================

_OK_STATUSES = {"Solved", "AlmostSolved"}

# Linear-solver threads per period QP: one. Measured on
# const_20261012_113828_12c562c7 against opt_20261012_122124_40b1b370 (#70): the
# 1995 period solved in 48.0 s on all cores and 34.9 s on one, bit-identical
# objective; the cores go to the parallel solve instead.
_PERIOD_MAX_THREADS = 1


class ClarabelSolver:
    """
    σ-weighted QP solver wrapping Clarabel for Stage 3b.

    Modelling decisions (locked in design grilling):

    - **Warm-start prior across time** (``prior_mode='warm'``, the default).
      The per-period prior mean ``a_prior(t)`` in the objective is the
      previous timestep's optimum, aligned by **edge template**
      ``(from_id, to_id, valuation)`` since each year occupies disjoint
      global columns (``cols_t``):

          a_prior(0) = f_agg                    (cold seed, the year-0 a_prior)
          a_prior(t) = a_post(t-1) | template   for t > 0

      ``prior_mode='cold'`` disables the chain: every year uses its own cold
      seed ``timestep_data[t]["a_prior"]`` (= f_agg sliced to year t). The
      actually-used ``a_prior(t)`` is stored on each ``OptimResult``.
      σ_a is held constant across timesteps (conditioned once from the cold
      prior): the prior mean drifts forward but the prior precision does not.
      Consequence: each warm year's optimum depends on every prior year
      through the chained prior.
      *Historical note:* the block path previously wrote ``a_0[cols_t]=F_opt_t``
      into the same (disjoint) year-t columns it had just read, so the
      carry-forward was a silent no-op and every year fell back to the
      time-replicated f_agg. The template-aligned chain below fixes that.
    - Cross-timestep speedup (``reuse_factorization``) uses
      ``solver.update(q=..., b=...)`` to retain the KKT factorisation — a
      *numerical* warm-start, distinct from the *prior* warm-start above.
      ``P`` and ``A`` are time-invariant.
    - Hard rows (zero and nonneg cones) are enforced exactly. The
      feasibility pre-check rules out impossible problems before the
      main solve and halts on any infeasibility.

    Σ_a conditioning
    ----------------
    A floor and a ceiling are applied to ``sigma_a`` to bound the
    condition number of ``Σ_a^{-1}``:

        floor   = max(σ_a_floor_abs,  σ_a_floor_rel · |a_0|)
        ceiling = σ_a_ceiling_factor · max(finite σ_a)

    Defaults give ~12 orders of magnitude condition number — fine for
    Clarabel with default static + dynamic regularisation.

    QP variables (#33)
    ------------------
    ``qp_variables='flows'`` (the default) solves for ``a``: ``q = −2·Σ_a^{-1}·a_prior``
    and Clarabel's cost is J − Σ(a_prior/σ_a)². ``'defects'`` solves the same QP for
    the defect ``d = a − a_prior`` (same P and A, ``q = 0``, ``b = c − G·a_prior``), so
    its cost is J itself. ``F_opt`` comes back as ``a_prior + d`` and ``objective`` in
    the flows convention (J − Σ(a_prior/σ_a)²) either way; ``convergence`` and
    ``iter_history`` keep Clarabel's own cost. gap_rel's denominator is that cost, so
    the same ``tol_gap_rel`` is ~10× tighter in absolute terms under ``'defects'``
    (J ≈ 1e5 against |J − Σ(a_prior/σ_a)²| ≈ 1e6 on copper). Block path only (serial
    and parallel); the feasibility probe has no σ-weighted cost.
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
        reduced_tol_feas:       float | None = None,
        reduced_tol_gap_abs:    float | None = None,
        reduced_tol_gap_rel:    float | None = None,
        max_iter:               int   = 200,
        verbose_first_solve:    bool  = True,
        capture_iter_history:   bool  = True,
        prior_mode:             str   = "warm",
        n_workers:              int   = 1,
        direct_solve_method:    str   = "faer",
        qp_variables:           str   = "flows",
    ) -> None:
        if prior_mode not in ("warm", "cold"):
            raise ValueError(
                f"prior_mode must be 'warm' or 'cold', got {prior_mode!r}"
            )
        if qp_variables not in ("flows", "defects"):
            raise ValueError(
                f"qp_variables must be 'flows' or 'defects', got {qp_variables!r}"
            )
        if n_workers < 1:
            raise ValueError(f"n_workers must be ≥ 1, got {n_workers}")
        if n_workers > 1 and prior_mode == "warm":
            raise ValueError(
                f"n_workers={n_workers} needs prior_mode='cold': a warm run is a "
                f"chain (a_prior(t) = a*(t-1)) and is solved in period order"
            )
        self.const_dir              = const_dir
        self.opt_dir                = opt_dir
        self.prior_mode             = prior_mode
        self.n_workers              = n_workers
        self.direct_solve_method    = direct_solve_method
        self.qp_variables           = qp_variables
        self.sigma_a_floor_abs      = sigma_a_floor_abs
        self.sigma_a_floor_rel      = sigma_a_floor_rel
        self.sigma_a_ceiling_factor = sigma_a_ceiling_factor
        self.sigma_c_floor          = sigma_c_floor
        self.tol_feas               = tol_feas
        self.tol_gap_abs            = tol_gap_abs
        self.tol_gap_rel            = tol_gap_rel
        self.reduced_tol_feas       = reduced_tol_feas
        self.reduced_tol_gap_abs    = reduced_tol_gap_abs
        self.reduced_tol_gap_rel    = reduced_tol_gap_rel
        self.max_iter               = max_iter
        self.verbose_first_solve    = verbose_first_solve
        self.capture_iter_history   = capture_iter_history
        # How the last solve_all ran (worker threads, linear solver, wall-clock);
        # persist() writes it into the sidecar.
        self._solve_config:  dict = {}
        self._solve_summary: dict = {}

        self._const_registry = ConstraintRunRegistry(const_dir)
        self._opt_registry   = OptimRunRegistry(opt_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_run(self, run_id: str) -> ConstraintRun:
        logger.info("Loading ConstraintRun '%s' from %s", run_id, self.const_dir)
        cr = self._const_registry.load(run_id)
        logger.info(
            "  G shape=%s  edges=%d  timesteps=%d",
            cr.G.shape, cr.G.shape[1], len(cr.timestep_data),
        )
        return cr

    # ------------------------------------------------------------------
    # Internal: build settings + transform
    # ------------------------------------------------------------------

    def _settings(self, verbose: bool) -> clarabel.DefaultSettings:
        s = clarabel.DefaultSettings()
        s.verbose         = verbose
        s.max_iter        = self.max_iter
        s.tol_feas        = self.tol_feas
        s.tol_gap_abs     = self.tol_gap_abs
        s.tol_gap_rel     = self.tol_gap_rel
        # Reduced-accuracy acceptance: when the iterate hits the numerical
        # floor (InsufficientProgress), accept it as AlmostSolved if it meets
        # these secondary tolerances. None ⇒ Clarabel defaults.
        if self.reduced_tol_feas is not None:
            s.reduced_tol_feas = self.reduced_tol_feas
        if self.reduced_tol_gap_abs is not None:
            s.reduced_tol_gap_abs = self.reduced_tol_gap_abs
        if self.reduced_tol_gap_rel is not None:
            s.reduced_tol_gap_rel = self.reduced_tol_gap_rel
        # Default static + dynamic regularisation kept on (Clarabel defaults).
        return s

    def _period_settings(self, verbose: bool) -> clarabel.DefaultSettings:
        """Settings for one period's QP: the linear solver pinned to
        ``_PERIOD_MAX_THREADS``. The full-G feasibility probe keeps the default,
        where the threads help (1995 probe of const_20261012_113828_12c562c7:
        432 s on all cores vs 471 s on one, #70)."""
        s = self._settings(verbose)
        s.direct_solve_method = self.direct_solve_method
        s.max_threads         = _PERIOD_MAX_THREADS
        return s

    def _condition_prior(
        self,
        a_0:     np.ndarray,
        sigma_a: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Thin wrapper over :func:`pysuto._conditioning.condition_prior`.

        Kept as a method so existing tests / call sites continue to work;
        the actual rules live in the shared helper so uncprop sees the
        same conditioned prior the QP did.
        """
        return condition_prior(
            a_0, sigma_a,
            floor_abs      = self.sigma_a_floor_abs,
            floor_rel      = self.sigma_a_floor_rel,
            ceiling_factor = self.sigma_a_ceiling_factor,
        )

    def _build_problem(
        self,
        run:      ConstraintRun,
        t_label:  str,
    ) -> dict:
        """
        Build (P, q, A, b, cones) and return everything needed to solve and
        post-process this timestep. Reusable internals are computed once per
        ConstraintRun (see :meth:`_problem_skeleton`); this method assembles
        the per-timestep b only when called via solve_timestep / solve_all.
        """
        skel = self._problem_skeleton(run)
        b    = self._build_b(run, t_label, skel["perm"])
        return {**skel, "b": b}

    def _problem_skeleton(self, run: ConstraintRun) -> dict:
        """
        Build the time-invariant pieces: P, q, A, cones, and the row
        permutation that maps original row order → [free; zero; nonneg].
        Computed once per ConstraintRun.
        """
        G        = run.G
        row_meta = run.row_meta
        a_0, sigma_a = self._condition_prior(run.a_prior_global, run.sigma_a)

        # Permute rows by cone_type → [free; zero; nonneg]
        ct = row_meta["cone_type"].to_numpy()
        idx_free   = np.where(ct == "free")[0]
        idx_zero   = np.where(ct == "zero")[0]
        idx_nonneg = np.where(ct == "nonneg")[0]
        perm = np.concatenate([idx_free, idx_zero, idx_nonneg])

        n_free, n_zero, n_nonneg = idx_free.size, idx_zero.size, idx_nonneg.size
        n_a = G.shape[1]
        M   = G.shape[0]

        # Permuted G (CSR for hstack)
        G_p = G.tocsr()[perm]

        # Slack column block: S = [-I_{n_free}; 0]
        S_top = -sp.eye(n_free, format="csr")
        S_bot = sp.csr_matrix((n_zero + n_nonneg, n_free))
        S     = sp.vstack([S_top, S_bot], format="csr") if n_free else \
                sp.csr_matrix((M, 0))

        # A = [G_p | S]  in CSC for Clarabel
        A = sp.hstack([G_p, S], format="csc")

        # P = 2 · diag([Σ_a^{-1}; Σ_c^{-1}_free]) — diagonal, CSC, upper-triangle
        inv_sa = 1.0 / (sigma_a ** 2)

        # σ_c on free rows may vary per timestep (e.g. SumConstraintBuilder
        # deactivates rows of other years via SIGMA_C_INACTIVE, and
        # fraction_of_flow scales σ_c with year-t flow magnitude). We no
        # longer require σ_c on free rows to be time-invariant; instead
        # the per-t Hessian is rebuilt in solve_all via Clarabel's
        # update(P=…). The initial P uses t_first's σ_c so the solver can
        # be instantiated once.
        first_td = next(iter(run.timestep_data.values()), None)
        if first_td is not None and n_free:
            sc_first = np.array(first_td["sigma_c"], dtype=float, copy=True)
            sc_first = np.where(np.isnan(sc_first), self.sigma_c_floor, sc_first)
            sc_first = np.maximum(sc_first, self.sigma_c_floor)
            sc_free_init = sc_first[perm[:n_free]]
        else:
            sc_free_init = np.zeros(n_free)

        inv_scf_init = 1.0 / (sc_free_init ** 2) if n_free else np.zeros(0)
        P_diag = 2.0 * np.concatenate([inv_sa, inv_scf_init])
        P = sp.diags(P_diag, format="csc")

        cones: list = []
        if n_free   > 0: cones.append(clarabel.ZeroConeT(n_free))
        if n_zero   > 0: cones.append(clarabel.ZeroConeT(n_zero))
        if n_nonneg > 0: cones.append(clarabel.NonnegativeConeT(n_nonneg))

        return {
            "P":          P,
            "A":          A,
            "cones":      cones,
            "perm":       perm,
            "n_free":     n_free,
            "n_zero":     n_zero,
            "n_nonneg":   n_nonneg,
            "n_a":        n_a,
            "M":          M,
            "sigma_a":    sigma_a,
            "inv_sa":     inv_sa,
            "a_0_init":   a_0,
        }

    def _build_P_diag(self, inv_sa: np.ndarray, run: ConstraintRun,
                      t_label: str, perm: np.ndarray, n_free: int) -> np.ndarray:
        """
        Per-timestep Hessian diagonal: ``2 · [Σ_a^{-1}; Σ_c(t)^{-1}_free]``.

        ``Σ_a^{-1}`` is time-invariant (computed once and passed in).
        ``Σ_c(t)^{-1}_free`` is recomputed from ``timestep_data[t]["sigma_c"]``
        with NaN → floor; same conditioning used in ``_problem_skeleton``.
        """
        if n_free:
            sc_t = np.array(run.timestep_data[t_label]["sigma_c"],
                            dtype=float, copy=True)
            sc_t = np.where(np.isnan(sc_t), self.sigma_c_floor, sc_t)
            sc_t = np.maximum(sc_t, self.sigma_c_floor)
            inv_scf = 1.0 / (sc_t[perm[:n_free]] ** 2)
        else:
            inv_scf = np.zeros(0)
        return 2.0 * np.concatenate([inv_sa, inv_scf])

    def _build_q(self, inv_sa: np.ndarray, a_0: np.ndarray, n_free: int) -> np.ndarray:
        """q = [-2 Σ_a^{-1} a_0(t); 0]. Recomputed per timestep when warm-starting."""
        return np.concatenate([-2.0 * inv_sa * a_0, np.zeros(n_free)])

    def _build_b(self, run: ConstraintRun, t_label: str, perm: np.ndarray) -> np.ndarray:
        c_t = run.timestep_data[t_label]["c"]
        return np.asarray(c_t, dtype=float)[perm]

    @staticmethod
    def _info_to_dict(info) -> dict:
        """Flatten Clarabel's ``DefaultInfo`` into JSON-safe scalars."""
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
            "sigma":            float(info.sigma),
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

    def feasibility_check(self, run: ConstraintRun, t_label: str) -> None:
        """
        Halt if the hard rows alone (zero + nonneg cones) plus flow bounds
        are infeasible at this timestep. Solves a feasibility-only QP
        ``min ½ ‖a‖²`` (P=εI, q=0) on the hard subsystem.

        ``ε`` (a tiny diagonal) keeps Clarabel happy with a strictly PSD P;
        the choice does not affect feasibility detection.
        """
        G        = run.G
        row_meta = run.row_meta
        u, l     = run.u, run.l
        c_t      = np.asarray(run.timestep_data[t_label]["c"], dtype=float)

        ct = row_meta["cone_type"].to_numpy()
        idx_zero   = np.where(ct == "zero")[0]
        idx_nonneg = np.where(ct == "nonneg")[0]
        n_a = G.shape[1]

        n_zero, n_nonneg = idx_zero.size, idx_nonneg.size

        # Hard rows from G
        G_zero   = G.tocsr()[idx_zero]   if n_zero   else sp.csr_matrix((0, n_a))
        G_nonneg = G.tocsr()[idx_nonneg] if n_nonneg else sp.csr_matrix((0, n_a))

        # Add flow bounds as inequalities  a ≤ u  and  -a ≤ -l
        #   Only edges with non-default bounds (finite, distinct from sentinel)
        #   really matter, but adding all edges is simpler and Clarabel handles
        #   trivial bounds fine. To keep the pre-check small, restrict to
        #   edges where l > -∞ or u < +∞ effectively differs from extremes:
        n_edges = n_a
        I = sp.eye(n_edges, format="csr")
        # Append all flow-bound rows
        Aub = sp.vstack([G_nonneg, I, -I], format="csr")
        bub = np.concatenate([c_t[idx_nonneg], u, -l])

        A = sp.vstack([G_zero, Aub], format="csc")
        b = np.concatenate([c_t[idx_zero], bub])

        cones_fc: list = []
        if n_zero            > 0: cones_fc.append(clarabel.ZeroConeT(n_zero))
        if Aub.shape[0]      > 0: cones_fc.append(clarabel.NonnegativeConeT(Aub.shape[0]))

        # Tiny PSD P + zero q for a pure feasibility solve
        P_fc = sp.diags(np.full(n_a, 1e-10), format="csc")
        q_fc = np.zeros(n_a)

        settings = self._settings(verbose=False)
        # Slightly looser tolerance on the feasibility probe — we just want a
        # yes/no answer, not a polished optimum.
        settings.tol_feas    = 1e-7
        settings.tol_gap_abs = 1e-7
        settings.tol_gap_rel = 1e-7

        solver = clarabel.DefaultSolver(P_fc, q_fc, A, b, cones_fc, settings)
        sol    = solver.solve()
        status = str(sol.status)

        if status not in _OK_STATUSES:
            self._raise_infeasible(run, t_label, status, sol)

    def _raise_infeasible(self, run, t_label, status, sol):
        # Best-effort diagnostic: which hard rows have the largest residual
        # at the (infeasible) point Clarabel returned. Even when the problem
        # is primal-infeasible Clarabel returns an x; the residual on hard
        # rows tells the user where to widen bounds.
        try:
            x = np.asarray(sol.x)
            if x.size >= run.G.shape[1]:
                a = x[: run.G.shape[1]]
                resid = run.G @ a - np.asarray(run.timestep_data[t_label]["c"])
                row_meta = run.row_meta
                hard = row_meta["is_hard"].to_numpy()
                hard_resid = np.where(hard, np.abs(resid), 0.0)
                worst = np.argsort(-hard_resid)[:5]
                worst_msg = "; ".join(
                    f"row={int(i)} cid={row_meta.iloc[int(i)]['constraint_id']!r} "
                    f"|residual|={hard_resid[i]:.3e}"
                    for i in worst
                )
            else:
                worst_msg = "(no diagnostic — solution vector unavailable)"
        except Exception as e:                              # pragma: no cover
            worst_msg = f"(diagnostic failed: {e})"

        raise RuntimeError(
            f"Feasibility check FAILED at timestep {t_label!r}: "
            f"Clarabel status={status}. "
            f"Top hard-row offenders (largest |G·a - c|): {worst_msg}. "
            f"Widen the offending ratio bounds in the params CSV "
            f"(e.g. toward [0, 1]) and rebuild the ConstraintRun, "
            f"or remove the conflicting constraint."
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve_all(
        self,
        run_id:    str,
        timesteps: Optional[list] = None,
        feasibility_check_first_only: bool = True,
    ) -> dict[str, OptimResult]:
        """
        Solve every requested timestep. Performs the feasibility pre-check
        before solving; halts if any checked timestep is infeasible.

        Parameters
        ----------
        feasibility_check_first_only:
            If True (default), only the first timestep is checked — fast.
            If False, all timesteps are checked before any solve begins.
        """
        run = self.load_run(run_id)

        if timesteps is None:
            ts_labels = list(run.timestep_data.keys())
        else:
            ts_labels = [str(t) for t in timesteps]
            missing = set(ts_labels) - set(run.timestep_data.keys())
            if missing:
                raise KeyError(f"timesteps not in run: {sorted(missing)}")

        # A failed call must not leave the previous call's facts for persist().
        self._solve_config, self._solve_summary = {}, {}
        t0 = time.perf_counter()
        if run.G_blocks:
            results = self._solve_all_via_blocks(
                run, ts_labels, feasibility_check_first_only,
            )
            self._solve_config = {
                "n_workers":           min(self.n_workers, len(ts_labels)),
                "max_threads":         _PERIOD_MAX_THREADS,
                "direct_solve_method": self.direct_solve_method,
                "qp_variables":        self.qp_variables,
            }
        else:
            results = self._solve_all_legacy(
                run, ts_labels, feasibility_check_first_only,
            )
            self._solve_config = {"n_workers": 1}
        # Wall-clock after loading: feasibility probe + period solves.
        self._solve_summary = {"solve_wall_s": time.perf_counter() - t0}
        return results

    def _solve_all_legacy(
        self,
        run:       ConstraintRun,
        ts_labels: list[str],
        feasibility_check_first_only: bool,
    ) -> dict[str, OptimResult]:
        """Legacy path: full-G QP per timestep with solver.update() warm-start.
        Used only when run.G_blocks is empty (test fixtures predating Slice 2)."""
        if self.qp_variables != "flows":
            raise ValueError(
                f"qp_variables={self.qp_variables!r} needs the block path "
                f"(run.G_blocks); the legacy full-G path solves for the flows only"
            )
        # Feasibility pre-check
        check_labels = ts_labels[:1] if feasibility_check_first_only else ts_labels
        logger.info(
            "Feasibility pre-check on %d/%d timesteps%s…",
            len(check_labels), len(ts_labels),
            " (first only)" if feasibility_check_first_only else "",
        )
        for t in check_labels:
            self.feasibility_check(run, t)
        logger.info("Feasibility pre-check passed.")

        # Build skeleton once (P, A, cones are time-invariant)
        skel    = self._problem_skeleton(run)
        cones   = skel["cones"]
        P, A    = skel["P"], skel["A"]
        perm    = skel["perm"]
        n_free  = skel["n_free"]
        n_a     = skel["n_a"]
        M       = skel["M"]
        inv_sa  = skel["inv_sa"]

        # Prior: a_prior(0) = cold seed (conditioned f_agg); a_prior(t>0) =
        # previous F_opt under prior_mode='warm' (full-vector path: cols align,
        # so the carry-forward is genuine), else the cold seed every year.
        a_0_cold = np.array(skel["a_0_init"], dtype=float, copy=True)
        a_0_t    = np.array(a_0_cold, dtype=float, copy=True)

        results: dict[str, OptimResult] = {}
        solver: Optional[clarabel.DefaultSolver] = None

        for k, t in enumerate(ts_labels):
            b = self._build_b(run, t, perm)
            q = self._build_q(inv_sa, a_0_t, n_free)
            # P now varies per timestep because σ_c on free rows can vary
            # (e.g. SumConstraintBuilder deactivates non-t-year rows with
            # σ_c = SIGMA_C_INACTIVE). The diagonal is rebuilt each iter;
            # cone structure, n_free, n_a are time-invariant so reusing
            # the Clarabel instance is fine.
            P_diag = self._build_P_diag(inv_sa, run, t, perm, n_free)
            P_t    = sp.diags(P_diag, format="csc")

            t0 = time.perf_counter()
            verbose = self.verbose_first_solve and k == 0
            if solver is None:
                settings = self._settings(verbose=verbose)
                solver   = clarabel.DefaultSolver(P_t, q, A, b, cones, settings)
            else:
                solver.update(P=P_t, q=q, b=b)

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
                    f"Timestep {t!r}: Clarabel status={status}; halting "
                    f"(no fallback). iterations={sol.iterations}"
                )

            x       = np.asarray(sol.x)
            F_opt   = x[:n_a]
            eps_opt = x[n_a:] if n_free else np.zeros(0)

            # Map ε back into M-length vector at original row positions
            eps_full = np.zeros(M)
            if n_free:
                eps_full[perm[:n_free]] = eps_opt

            c_full        = np.asarray(run.timestep_data[t]["c"])
            c_star        = c_full + eps_full
            residuals     = run.G @ F_opt - c_full

            results[t] = OptimResult(
                timestep      = t,
                F_opt         = F_opt,
                eps_opt       = eps_opt,
                c_star        = c_star,
                residuals     = residuals,
                objective     = float(sol.obj_val),
                iterations    = int(sol.iterations),
                solve_time_s  = elapsed,
                solver_status = status,
                convergence   = convergence,
                iter_history  = iter_history,
                a_prior       = np.array(a_0_t, dtype=float, copy=True),
            )

            logger.info(
                "t=%s  status=%s  iters=%d  obj=%.4e  gap=%.2e  dres=%.2e  step=%.2f  time=%.2fs",
                t, status, sol.iterations, sol.obj_val,
                convergence["gap_abs"], convergence["res_dual"],
                convergence["step_length"], elapsed,
            )

            # Prior for next timestep: warm → a_prior(t+1) ← a*(t); cold → cold seed
            a_0_t = F_opt if self.prior_mode == "warm" else a_0_cold

        return results

    @staticmethod
    def _per_t_problem_identical(run: ConstraintRun, ts_labels: list[str]) -> bool:
        """True when A_t = [G_t | S_t] is identical across every t in ``ts_labels``.

        Requires: same G_blocks (shape, indptr, indices, data) AND same
        cone_type ordering within each per-t row block (so the column-permuted
        slack block S_t and cones list match too). Holds in the typical case
        where all builders are time-invariant (e.g. ratios with a_time="*");
        breaks when a builder emits year-specific coefficients (e.g. ore_grade
        with explicit per-year value_nominal)."""
        if len(ts_labels) <= 1:
            return False
        row_time = run.row_meta["time"].astype(str).to_numpy()
        first_t  = ts_labels[0]
        first_G  = run.G_blocks[first_t].tocsr()
        first_ct = run.row_meta.iloc[
            np.where(row_time == first_t)[0]
        ]["cone_type"].to_numpy()
        for t in ts_labels[1:]:
            G_t = run.G_blocks[t].tocsr()
            if G_t.shape != first_G.shape or G_t.nnz != first_G.nnz:
                return False
            if not np.array_equal(G_t.indptr,  first_G.indptr):  return False
            if not np.array_equal(G_t.indices, first_G.indices): return False
            if not np.array_equal(G_t.data,    first_G.data):    return False
            ct_t = run.row_meta.iloc[
                np.where(row_time == t)[0]
            ]["cone_type"].to_numpy()
            if not np.array_equal(ct_t, first_ct):
                return False
        return True

    def _solve_all_via_blocks(
        self,
        run:       ConstraintRun,
        ts_labels: list[str],
        feasibility_check_first_only: bool,
    ) -> dict[str, OptimResult]:
        """Per-t solve path using run.G_blocks. Each per-t QP is ~30× smaller
        than the legacy full-G QP. When every per-t A_t matches (typical case
        with no time-varying ratios), one Clarabel instance is built and
        reused via solver.update(P, q, b) — keeping the KKT factorisation
        across the whole timestep loop."""
        check_labels = ts_labels[:1] if feasibility_check_first_only else ts_labels
        logger.info(
            "Feasibility pre-check on %d/%d timesteps%s…",
            len(check_labels), len(ts_labels),
            " (first only)" if feasibility_check_first_only else "",
        )
        for t in check_labels:
            self.feasibility_check(run, t)
        logger.info("Feasibility pre-check passed.")

        a_0_global, sigma_a_global = self._condition_prior(
            run.a_prior_global, run.sigma_a,
        )
        inv_sa_global = 1.0 / (sigma_a_global ** 2)
        row_time = run.row_meta["time"].astype(str).to_numpy()

        # Edge-template map for the warm-start carry-forward: a_prior(t)=a_post(t-1)
        # aligned by (from_id,to_id,valuation), since cols_t are disjoint per year.
        # When the AccountingStructure is unavailable (toy fixtures), fall back to
        # a positional carry-forward — per-year cols_t are emitted in template
        # order, so equal-length adjacent years align position-for-position.
        acc_ws = None
        if self.prior_mode == "warm":
            try:
                from pysuto.acc import AccountingStructure
                acc_ws = AccountingStructure.load(run.acc_id)
            except Exception:
                logger.warning(
                    "warm prior: AccountingStructure %r unavailable — using "
                    "positional carry-forward by edge order.", run.acc_id,
                )

        if self.n_workers > 1:
            return self._solve_periods_parallel(
                run, ts_labels, a_0_global, inv_sa_global, row_time,
            )

        reuse_factorization = self._per_t_problem_identical(run, ts_labels)
        results: dict[str, OptimResult] = {}
        solver_t: Optional[clarabel.DefaultSolver] = None
        prev_cols: Optional[np.ndarray] = None
        prev_F:    Optional[np.ndarray] = None

        for k, t in enumerate(ts_labels):
            cols_t  = run.edge_cols_by_t[t]
            # Per-period prior mean a_prior(t): warm → a_post(t-1) aligned by
            # template; cold seed (t0, or every year in cold mode) → conditioned
            # f_agg slice.
            a_prior_t = a_0_global[cols_t].copy()
            if self.prior_mode == "warm" and prev_F is not None:
                if acc_ws is not None:
                    g   = acc_ws.warm_start_gather(prev_cols, cols_t)
                    hit = g >= 0
                    a_prior_t[hit] = prev_F[g[hit]]
                elif len(prev_F) == len(a_prior_t):
                    a_prior_t = prev_F.copy()  # positional fallback (no acc)

            results[t], solver_t = self._solve_period(
                run, t, a_prior_t, inv_sa_global[cols_t], row_time,
                solver_t = solver_t if reuse_factorization else None,
                verbose  = self.verbose_first_solve and k == 0,
            )

            # Carry this period's posterior forward as next year's warm prior,
            # aligned to t+1's edges by template (warm_start_gather above). The
            # old in-place a_0_t_global[cols_t]=F_opt_t write was a no-op (cols_t
            # are disjoint per year, so a later year never read it).
            prev_cols, prev_F = cols_t, results[t].F_opt

        return results

    def _solve_periods_parallel(
        self,
        run:           ConstraintRun,
        ts_labels:     list[str],
        a_0_global:    np.ndarray,
        inv_sa_global: np.ndarray,
        row_time:      np.ndarray,
    ) -> dict[str, OptimResult]:
        """Parallel solve (cold only): every period is its own QP against its
        cold seed, so the periods run concurrently on a thread pool, each with a
        fresh Clarabel instance. Clarabel releases the GIL in ``solve()``: 8
        periods of const_20261012_113828_12c562c7 took 63.3 s on 8 threads vs
        549 s serially in opt_20261012_122124_40b1b370, bit-identical (#70).

        The first failed period cancels the periods not yet started; the
        RuntimeError names every period that failed, and nothing is returned."""
        n_threads = min(self.n_workers, len(ts_labels))
        logger.info("Parallel solve: %d periods on %d threads", len(ts_labels), n_threads)

        def _solve_one_period(t: str) -> OptimResult:
            cols_t = run.edge_cols_by_t[t]
            res, _ = self._solve_period(
                run, t, a_0_global[cols_t].copy(), inv_sa_global[cols_t], row_time,
            )
            return res

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            fut_by_t = {t: pool.submit(_solve_one_period, t) for t in ts_labels}
            for fut in as_completed(fut_by_t.values()):
                if fut.exception() is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
        # Leaving the with-block waited for the periods already running.
        failed = {t: f.exception() for t, f in fut_by_t.items()
                  if not f.cancelled() and f.exception() is not None}
        if failed:
            raise RuntimeError(
                f"Parallel solve failed in {len(failed)} period(s) "
                f"{sorted(failed)}; periods not yet started were cancelled — "
                + " | ".join(str(e) for e in failed.values())
            ) from next(iter(failed.values()))
        return {t: fut_by_t[t].result() for t in ts_labels}

    def _solve_period(
        self,
        run:       ConstraintRun,
        t:         str,
        a_prior_t: np.ndarray,
        inv_sa_t:  np.ndarray,
        row_time:  np.ndarray,
        solver_t:  Optional[clarabel.DefaultSolver] = None,
        verbose:   bool = False,
    ) -> tuple[OptimResult, clarabel.DefaultSolver]:
        """Build period ``t``'s QP, solve it and return its OptimResult with the
        Clarabel instance used. Passing the previous period's ``solver_t``
        keeps its KKT factorisation via ``update()`` (serial path, identical
        per-t problems only); otherwise a fresh instance is built."""
        cols_t  = run.edge_cols_by_t[t]
        rows_t  = np.where(row_time == t)[0]
        G_t     = run.G_blocks[t]
        rm_t    = run.row_meta.iloc[rows_t]
        c_t_full     = np.asarray(run.timestep_data[t]["c"])
        sc_t_full    = np.asarray(run.timestep_data[t]["sigma_c"])
        c_t          = c_t_full[rows_t]
        sc_t         = sc_t_full[rows_t]

        ct       = rm_t["cone_type"].to_numpy()
        idx_free   = np.where(ct == "free")[0]
        idx_zero   = np.where(ct == "zero")[0]
        idx_nonneg = np.where(ct == "nonneg")[0]
        perm_t   = np.concatenate([idx_free, idx_zero, idx_nonneg])
        n_free_t = idx_free.size
        n_zero_t, n_nonneg_t = idx_zero.size, idx_nonneg.size
        n_a_t    = G_t.shape[1]
        m_t      = G_t.shape[0]

        G_p_t = G_t.tocsr()[perm_t]
        S_top = -sp.eye(n_free_t, format="csr") if n_free_t else sp.csr_matrix((0, 0))
        S_bot = sp.csr_matrix((n_zero_t + n_nonneg_t, n_free_t))
        S_t   = sp.vstack([S_top, S_bot], format="csr") if n_free_t else \
                sp.csr_matrix((m_t, 0))
        A_t   = sp.hstack([G_p_t, S_t], format="csc")
        b_t   = c_t[perm_t]

        if n_free_t:
            sc_arr   = np.where(np.isnan(sc_t), self.sigma_c_floor, sc_t)
            sc_arr   = np.maximum(sc_arr, self.sigma_c_floor)
            inv_scf  = 1.0 / (sc_arr[perm_t[:n_free_t]] ** 2)
        else:
            inv_scf = np.zeros(0)
        P_diag_t = 2.0 * np.concatenate([inv_sa_t, inv_scf])
        P_t      = sp.diags(P_diag_t, format="csc")
        # Prior offset (#33): Clarabel solves for x = a − a_shift. "flows": a_shift = 0;
        # "defects": a_shift = a_prior, so q = 0 and Clarabel's cost is J itself rather
        # than J − Σ(a_prior/σ_a)² (2015 of const_20261019_170151_13460e94: 1,288,065
        # against an optimal J of 117,282). const_t moves the reported objective back
        # to the flows convention.
        a_shift  = a_prior_t if self.qp_variables == "defects" else np.zeros(n_a_t)
        q_t      = np.concatenate([-2.0 * inv_sa_t * (a_prior_t - a_shift),
                                   np.zeros(n_free_t)])
        b_t      = b_t - G_p_t @ a_shift
        const_t  = float(inv_sa_t @ (a_shift * (2.0 * a_prior_t - a_shift)))

        cones_t: list = []
        if n_free_t   > 0: cones_t.append(clarabel.ZeroConeT(n_free_t))
        if n_zero_t   > 0: cones_t.append(clarabel.ZeroConeT(n_zero_t))
        if n_nonneg_t > 0: cones_t.append(clarabel.NonnegativeConeT(n_nonneg_t))

        t0 = time.perf_counter()
        reused = solver_t is not None
        if reused:
            solver_t.update(P=P_t, q=q_t, b=b_t)
        else:
            settings = self._period_settings(verbose=verbose)
            solver_t = clarabel.DefaultSolver(
                P_t, q_t, A_t, b_t, cones_t, settings,
            )

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
            solver_t.set_termination_callback(_cb)

        sol     = solver_t.solve()
        elapsed = time.perf_counter() - t0
        status  = str(sol.status)

        if self.capture_iter_history:
            solver_t.unset_termination_callback()

        if status not in _OK_STATUSES and reused:
            # The reused instance keeps the equilibration computed for the
            # first timestep (update() does not re-equilibrate); per-t
            # σ_c now varies with f_agg, so a later year can stall on
            # stale scaling. Cold restart with fresh equilibration.
            logger.warning(
                "t=%s  status=%s on reused solver — cold-restart retry",
                t, status,
            )
            settings = self._period_settings(verbose=False)
            solver_t = clarabel.DefaultSolver(
                P_t, q_t, A_t, b_t, cones_t, settings,
            )
            hist.clear()
            if self.capture_iter_history:
                solver_t.set_termination_callback(_cb)
            sol     = solver_t.solve()
            elapsed = time.perf_counter() - t0
            status  = str(sol.status)
            if self.capture_iter_history:
                solver_t.unset_termination_callback()

        convergence  = self._info_to_dict(solver_t.get_info())
        iter_history = np.array(hist, dtype=ITER_DTYPE) if hist else None

        if status not in _OK_STATUSES:
            raise RuntimeError(
                f"Timestep {t!r}: Clarabel status={status}; halting"
                f"{' (cold restart already tried)' if reused else ''}. "
                f"iterations={sol.iterations}"
            )

        x         = np.asarray(sol.x)
        F_opt_t   = a_shift + x[:n_a_t]
        eps_opt_t = x[n_a_t:] if n_free_t else np.zeros(0)

        # ε back into row-order on the per-t row block
        eps_full_t = np.zeros(m_t)
        if n_free_t:
            eps_full_t[perm_t[:n_free_t]] = eps_opt_t

        # Year-local persistence (REP §8.2): store only this period's
        # solved values. F_opt over cols_t, residuals/c_star over rows_t —
        # no cross-year prior placeholders. Residual is the block residual
        # G_t @ F_opt_t - c_t (rows_t order); mass balance ≈ 0 by construction.
        residuals_t = np.asarray(G_t @ F_opt_t - c_t)
        c_star_t    = c_t + eps_full_t

        result = OptimResult(
            timestep      = t,
            F_opt         = F_opt_t,
            eps_opt       = eps_opt_t,
            c_star        = c_star_t,
            residuals     = residuals_t,
            objective     = float(sol.obj_val) - const_t,
            iterations    = int(sol.iterations),
            solve_time_s  = elapsed,
            solver_status = status,
            convergence   = convergence,
            iter_history  = iter_history,
            cols_t        = cols_t,
            rows_t        = rows_t,
            a_prior       = a_prior_t,
        )

        logger.info(
            "t=%s  status=%s  iters=%d  obj=%.4e  gap=%.2e  dres=%.2e  step=%.2f  time=%.2fs  (per-t: %d cols × %d rows)",
            t, status, sol.iterations, result.objective,
            convergence["gap_abs"], convergence["res_dual"],
            convergence["step_length"], elapsed,
            n_a_t, m_t,
        )
        return result, solver_t

    def solve_timestep(
        self,
        run_id:  str,
        t_label: str,
    ) -> OptimResult:
        """Convenience: solve a single timestep (no warm-start)."""
        return self.solve_all(run_id, timesteps=[t_label])[t_label]

    # --------------------1----------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(
        self,
        run_id:     str,
        results:    dict[str, OptimResult],
        output_dir: Optional[str] = None,
        const_run_id: Optional[str] = None,
        label:      str = "",
        stage_config: Optional[dict] = None,
    ) -> str:
        """Delegate to :class:`OptimRunRegistry`.

        The first positional argument (``run_id``) is the *upstream*
        ``const_run_id`` (kept for backwards-compatible call signature).
        ``output_dir`` overrides the solver's own ``self._opt_registry``
        target — useful for tests pinning to a tmpdir. The newly minted
        *output* id is returned and starts with ``opt_``.

        ``stage_config`` records how the solve was configured. ``prior_mode``
        in particular is not recoverable from the artifacts, so a run that
        omits it cannot be compared to another on that axis (Round P, G5); it
        defaults to this solver's own ``prior_mode`` when the caller passes
        nothing. How the last ``solve_all`` ran — ``n_workers`` (threads
        used), ``max_threads``, ``direct_solve_method`` and ``qp_variables`` — is
        added to it, and its wall-clock lands in ``stage_summary.solve_wall_s``.
        """
        stage_config = ({"prior_mode": self.prior_mode}
                        if stage_config is None else stage_config)
        stage_config = {**stage_config, **self._solve_config}
        const_id = const_run_id or run_id
        registry = OptimRunRegistry(output_dir) if output_dir else self._opt_registry
        opt_run_id = registry.register(
            results,
            parents = {
                "inputs":   {},
                "upstream": {"const_run_id": const_id},
            },
            stage_config  = stage_config,
            stage_summary = dict(self._solve_summary) or None,
            label = label,
        )
        logger.info("OptimRun '%s' persisted to %s",
                    opt_run_id, registry.storage_dir)
        return opt_run_id
