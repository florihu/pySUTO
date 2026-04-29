"""
solver.py
=========
Stage 3b — Clarabel-backed σ-weighted QP solver.

Loads a persisted :class:`pysuto.constraints.ConstraintRun`, transforms it
to Clarabel's standard form, runs a feasibility pre-check on the hard rows,
then solves the QP per timestep using ``solver.update(b=...)`` between
timesteps to retain the KKT factorisation (the practical warm-start
mechanism for this problem since only ``c`` varies across time).

Cone layout (correcting README §665-667):

* ``ZeroConeT(n_free)`` — soft equality residuals ``Gx - ε = c`` (slack
  ε is a free decision variable inside ``x``; the residual itself is
  exact equality).
* ``ZeroConeT(n_zero)`` — hard mass-balance equalities.
* ``NonnegativeConeT(n_nonneg)`` — hard ratio bounds and flow bounds.

Public API
----------
- :class:`OptimResult` — per-timestep solution + diagnostics
- :class:`ClarabelSolver` — load, feasibility-check, solve_all, persist
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

import clarabel

from .constraints import ConstraintRun, ConstraintRunRegistry
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
    objective:     float
    iterations:    int
    solve_time_s:  float
    solver_status: str             # str(clarabel SolverStatus enum)
    convergence:   dict            # final get_info() snapshot (JSON-safe scalars)
    iter_history:  Optional[np.ndarray] = None  # structured array of ITER_DTYPE


# =============================================================================
# ClarabelSolver
# =============================================================================

_OK_STATUSES = {"Solved", "AlmostSolved"}


class ClarabelSolver:
    """
    σ-weighted QP solver wrapping Clarabel for Stage 3b.

    Modelling decisions (locked in design grilling):

    - **Warm-start prior across time.** The prior ``a_0(t)`` in the
      objective is the previous timestep's optimum:

          a_0(0) = ConstraintRun.a0       (Stage 2 f_agg)
          a_0(t) = F_opt(t-1)             for t > 0

      σ_a is held constant across timesteps (= ``ConstraintRun.sigma_a``);
      the prior mean drifts forward but the prior precision does not.
      Consequence to be aware of: each year's optimum depends on every
      prior year through the prior. If year t-1 drifts from its data,
      year t inherits that drift.
    - Cross-timestep speedup uses ``solver.update(q=..., b=...)`` to
      retain the KKT factorisation. ``P`` and ``A`` are time-invariant.
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
    """

    def __init__(
        self,
        const_dir: str = "data/proc/const",
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

        self._registry = ConstraintRunRegistry(const_dir)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_run(self, run_id: str) -> ConstraintRun:
        logger.info("Loading ConstraintRun '%s' from %s", run_id, self.const_dir)
        cr = self._registry.load(run_id)
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
        # Default static + dynamic regularisation kept on (Clarabel defaults).
        return s

    def _condition_prior(
        self,
        a_0:     np.ndarray,
        sigma_a: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make ``a_0`` and ``sigma_a`` safe for Clarabel:

        - Any NaN in ``a_0`` → 0 (flat prior, belt-and-braces in case the
          producing ConstraintRun pre-dates FlatPriorImputer integration).
        - Any NaN in ``sigma_a`` → ceiling (no-information prior).
        - Floor σ_a at ``max(σ_a_floor_abs, σ_a_floor_rel · |a_0|)``.
        - Ceiling σ_a at ``σ_a_ceiling_factor · max(finite σ_a)``.

        Logs counts of NaN substitutions if any.
        """
        a   = np.array(a_0,     dtype=float, copy=True)
        s   = np.array(sigma_a, dtype=float, copy=True)

        n_nan_a = int(np.isnan(a).sum())
        if n_nan_a:
            a[np.isnan(a)] = 0.0
            logger.warning(
                "ClarabelSolver: %d NaN entries in a_0 substituted with 0 "
                "(prefer registering FlatPriorImputer in Stage 2 to avoid this).",
                n_nan_a,
            )

        # Compute ceiling from finite σ before NaN substitution
        finite = s[np.isfinite(s) & (s < 1e30) & (s > 0)]
        if finite.size:
            ceiling = self.sigma_a_ceiling_factor * float(finite.max())
        else:
            ceiling = 1e12

        n_nan_s = int(np.isnan(s).sum())
        if n_nan_s:
            s[np.isnan(s)] = ceiling
            logger.warning(
                "ClarabelSolver: %d NaN entries in sigma_a substituted with "
                "ceiling=%.3e (no-information prior).",
                n_nan_s, ceiling,
            )

        floor = np.maximum(self.sigma_a_floor_abs,
                           self.sigma_a_floor_rel * np.abs(a))
        s = np.maximum(s, floor)
        s = np.minimum(s, ceiling)

        return a, s

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
        a_0, sigma_a = self._condition_prior(run.a0, run.sigma_a)

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

        # σ_c for free rows is invariant under timestep IF σ_c does not vary
        # by timestep. Some builders (SumConstraintBuilder fraction_of_flow)
        # produce σ_c that does vary per timestep — in that case the
        # per-timestep b update would not be enough to keep P valid. We
        # check this and raise a clear error if it occurs.
        sc_free_per_t = []
        for t_label, td in run.timestep_data.items():
            sc_t = np.array(td["sigma_c"], dtype=float, copy=True)
            # NaN → floor (treated as "no info on the constraint" → minimum
            # weight; the σ_c floor is the smallest admissible value).
            sc_t = np.where(np.isnan(sc_t), self.sigma_c_floor, sc_t)
            sc_t = np.maximum(sc_t, self.sigma_c_floor)
            sc_free_per_t.append(sc_t[perm[:n_free]])
        sc_free_arr = np.stack(sc_free_per_t) if sc_free_per_t else np.zeros((0, n_free))

        if sc_free_arr.shape[0] > 0 and not np.allclose(
            sc_free_arr, sc_free_arr[0:1], rtol=0, atol=0
        ):
            raise NotImplementedError(
                "ClarabelSolver currently assumes σ_c is time-invariant on "
                "free-cone rows (so P can be built once and only b updated "
                "across timesteps). Detected per-timestep variation in σ_c "
                "for free rows — re-implement with full re-build per timestep."
            )

        sc_free = sc_free_arr[0] if sc_free_arr.shape[0] else np.zeros(n_free)
        inv_scf = 1.0 / (sc_free ** 2) if n_free else np.zeros(0)

        P_diag = 2.0 * np.concatenate([inv_sa, inv_scf])
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

        # Warm-start prior: a_0(0) = run.a0; a_0(t>0) = previous F_opt
        a_0_t = np.array(skel["a_0_init"], dtype=float, copy=True)

        results: dict[str, OptimResult] = {}
        solver: Optional[clarabel.DefaultSolver] = None

        for k, t in enumerate(ts_labels):
            b = self._build_b(run, t, perm)
            q = self._build_q(inv_sa, a_0_t, n_free)

            t0 = time.perf_counter()
            verbose = self.verbose_first_solve and k == 0
            if solver is None:
                settings = self._settings(verbose=verbose)
                solver   = clarabel.DefaultSolver(P, q, A, b, cones, settings)
            else:
                solver.update(q=q, b=b)

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
            )

            logger.info(
                "t=%s  status=%s  iters=%d  obj=%.4e  gap=%.2e  dres=%.2e  step=%.2f  time=%.2fs",
                t, status, sol.iterations, sol.obj_val,
                convergence["gap_abs"], convergence["res_dual"],
                convergence["step_length"], elapsed,
            )

            # Warm-start prior for next timestep: a_0(t+1) ← a*(t)
            a_0_t = F_opt

        return results

    def solve_timestep(
        self,
        run_id:  str,
        t_label: str,
    ) -> OptimResult:
        """Convenience: solve a single timestep (no warm-start)."""
        return self.solve_all(run_id, timesteps=[t_label])[t_label]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(
        self,
        run_id:     str,
        results:    dict[str, OptimResult],
        output_dir: str = "data/proc/opt",
        const_run_id: Optional[str] = None,
    ) -> str:
        """
        Persist results under ``{output_dir}/{run_id}/``.

        Layout:
            {output_dir}/{optim_run_id}/
            ├── registry.json
            └── timesteps/
                └── t_{label}.npz   (F_opt, eps_opt, c_star, residuals)
        """
        optim_run_id = (
            f"optim_{run_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        )
        run_dir = os.path.join(output_dir, optim_run_id)
        ts_dir  = os.path.join(run_dir, "timesteps")
        os.makedirs(ts_dir, exist_ok=True)

        for t, res in results.items():
            arrays = {
                "F_opt":     res.F_opt,
                "eps_opt":   res.eps_opt,
                "c_star":    res.c_star,
                "residuals": res.residuals,
            }
            if res.iter_history is not None:
                arrays["iter_history"] = res.iter_history
            np.savez(os.path.join(ts_dir, f"t_{t}.npz"), **arrays)

        index = {
            "optim_run_id": optim_run_id,
            "const_run_id": const_run_id or run_id,
            "created_at":   datetime.utcnow().isoformat(),
            "timesteps":    list(results.keys()),
            "per_timestep": {
                t: {
                    "objective":     res.objective,
                    "iterations":    res.iterations,
                    "solve_time_s":  res.solve_time_s,
                    "solver_status": res.solver_status,
                    "convergence":   res.convergence,
                }
                for t, res in results.items()
            },
        }
        with open(os.path.join(run_dir, "registry.json"), "w") as fh:
            json.dump(index, fh, indent=2)

        logger.info("OptimRun '%s' persisted to %s", optim_run_id, run_dir)
        return optim_run_id
