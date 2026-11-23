"""
clarabel_backend.py
===================
:class:`ClarabelBackend` — the production QP backend (ADR 0042).

The only module of the package that imports ``clarabel``. It binds the neutral
:class:`pysuto.qp.QPProblem` to Clarabel's conic form (equality rows → ``ZeroConeT``,
``≤`` rows → ``NonnegativeConeT``) and owns what is specific to Clarabel:

* its settings — termination and reduced-accuracy tolerances (ADR 0039), iteration cap,
  linear solver and threads per solve;
* reuse of the KKT factorisation across consecutive calls (``solver.update(P, q, b)``)
  when the structure matches, and the cold restart that reuse makes necessary;
* its telemetry — the final info snapshot and the per-iteration history (:data:`ITER_DTYPE`);
* the status map: ``Solved`` → ``solved``, ``AlmostSolved`` → ``solved_inaccurate``,
  everything else → ``failed``.

Public API
----------
- :class:`ClarabelBackend`
- :data:`ITER_DTYPE` — dtype of the per-iteration history
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import scipy.sparse as sp

import clarabel

from .logging import get_logger
from .qp import FAILED, SOLVED, SOLVED_INACCURATE, QPProblem, QPResult

logger = get_logger("pysuto.clarabel_backend")

# Structured dtype for the per-iteration convergence trace.
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

_STATUS_MAP = {"Solved": SOLVED, "AlmostSolved": SOLVED_INACCURATE}

# Tolerance of a feasibility probe: a yes/no answer, not a polished optimum.
_PROBE_TOL = 1e-7


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
        "sigma":           float(info.sigma),
        "step_length":     float(info.step_length),
        "iterations":      int(info.iterations),
        "solve_time":      float(info.solve_time),
        "status":          str(info.status),
        "linsolver_name":  str(getattr(ls, "name",   "")) if ls is not None else "",
        "linsolver_nnzA":  int(getattr(ls, "nnzA",   0))  if ls is not None else 0,
        "linsolver_nnzL":  int(getattr(ls, "nnzL",   0))  if ls is not None else 0,
    }


class ClarabelBackend:
    """Solve a :class:`~pysuto.qp.QPProblem` with Clarabel.

    Parameters
    ----------
    tol_feas, tol_gap_abs, tol_gap_rel:
        Termination tolerances.
    reduced_tol_feas, reduced_tol_gap_abs, reduced_tol_gap_rel:
        Reduced-accuracy acceptance: when the iterate hits the numerical floor
        (``InsufficientProgress``) Clarabel accepts it as ``AlmostSolved`` if it meets
        these. ``None`` keeps Clarabel's defaults.
    max_iter:
        Iteration cap.
    direct_solve_method, max_threads:
        Clarabel's linear solver and the threads it uses per solve. One thread:
        measured on const_20261012_113828_12c562c7 against opt_20261012_122124_40b1b370
        (#70), the 1995 period solved in 48.0 s on all cores and 34.9 s on one,
        bit-identical objective; the cores go to the orchestrator's parallel solve.
    capture_iter_history:
        Keep the per-iteration history (:data:`ITER_DTYPE`) of every solve.

    The backend keeps the instance of its latest call. ``solve(..., reuse=True)`` updates
    ``P``, ``q`` and ``b`` on it, keeping its KKT factorisation, when ``A`` and the
    sparsity of ``P`` are identical to that call's; ``update()`` does not re-equilibrate,
    so a stalled solve on a reused instance is retried on a fresh one. A ``reuse=False``
    call builds a fresh instance, so such calls may run on several threads at once;
    ``reuse=True`` is for one caller at a time. Every call still copies ``A`` into the
    structure key and stores its instance as the latest one, also a ``reuse=False`` call
    on one of several threads: the later ``reuse=True`` call of a serial run must find
    the first period's instance, and a call cannot know whether one follows. The stores
    of concurrent calls are single reference assignments, so the race is benign (a
    parallel run never passes ``reuse=True``).

    ``settings`` records the constructor's values: a ``reduced_tol_*`` left ``None``
    is recorded as ``None`` and runs on Clarabel's own default, and the probe's
    tolerance (``_PROBE_TOL``, for ``feasibility_only`` problems) is not recorded. The
    Stage-4 driver passes every tolerance, so a sidecar from it holds the values used.
    Each result's ``info`` carries ``reused`` (this solve continued the kept
    factorisation) and ``cold_restart`` (a reused instance stalled and the solve was
    redone on a fresh one; ``reused`` is then false).
    """

    name = "clarabel"

    def __init__(
        self,
        tol_feas:            float = 1e-8,
        tol_gap_abs:         float = 1e-8,
        tol_gap_rel:         float = 1e-8,
        reduced_tol_feas:    Optional[float] = None,
        reduced_tol_gap_abs: Optional[float] = None,
        reduced_tol_gap_rel: Optional[float] = None,
        max_iter:            int   = 200,
        direct_solve_method: str   = "faer",
        max_threads:         int   = 1,
        capture_iter_history: bool = True,
    ) -> None:
        self.tol_feas             = tol_feas
        self.tol_gap_abs          = tol_gap_abs
        self.tol_gap_rel          = tol_gap_rel
        self.reduced_tol_feas     = reduced_tol_feas
        self.reduced_tol_gap_abs  = reduced_tol_gap_abs
        self.reduced_tol_gap_rel  = reduced_tol_gap_rel
        self.max_iter             = max_iter
        self.direct_solve_method  = direct_solve_method
        self.max_threads          = max_threads
        self.capture_iter_history = capture_iter_history
        self._kept: Optional[tuple] = None        # (structure key, DefaultSolver)

    @property
    def settings(self) -> dict:
        return {
            "tol_feas":             self.tol_feas,
            "tol_gap_abs":          self.tol_gap_abs,
            "tol_gap_rel":          self.tol_gap_rel,
            "reduced_tol_feas":     self.reduced_tol_feas,
            "reduced_tol_gap_abs":  self.reduced_tol_gap_abs,
            "reduced_tol_gap_rel":  self.reduced_tol_gap_rel,
            "max_iter":             self.max_iter,
            "direct_solve_method":  self.direct_solve_method,
            "max_threads":          self.max_threads,
            "capture_iter_history": self.capture_iter_history,
        }

    # ------------------------------------------------------------------
    def _clarabel_settings(self, verbose: bool, probe: bool) -> clarabel.DefaultSettings:
        s = clarabel.DefaultSettings()
        s.verbose             = verbose
        s.max_iter            = self.max_iter
        s.tol_feas            = _PROBE_TOL if probe else self.tol_feas
        s.tol_gap_abs         = _PROBE_TOL if probe else self.tol_gap_abs
        s.tol_gap_rel         = _PROBE_TOL if probe else self.tol_gap_rel
        if self.reduced_tol_feas is not None:
            s.reduced_tol_feas = self.reduced_tol_feas
        if self.reduced_tol_gap_abs is not None:
            s.reduced_tol_gap_abs = self.reduced_tol_gap_abs
        if self.reduced_tol_gap_rel is not None:
            s.reduced_tol_gap_rel = self.reduced_tol_gap_rel
        s.direct_solve_method = self.direct_solve_method
        s.max_threads         = self.max_threads
        return s

    @staticmethod
    def _structure(P: sp.csc_matrix, A: sp.csc_matrix, n_eq: int) -> tuple:
        """What must be identical for ``update(P, q, b)`` to be valid: ``A`` entry for
        entry, the sparsity of ``P`` and the cone split."""
        return (A.shape, n_eq, A.indptr.copy(), A.indices.copy(), A.data.copy(),
                P.indptr.copy(), P.indices.copy())

    @staticmethod
    def _same(a: tuple, b: tuple) -> bool:
        return (a[0] == b[0] and a[1] == b[1]
                and all(np.array_equal(x, y) for x, y in zip(a[2:], b[2:])))

    def _run(self, inst, record: bool):
        """Solve ``inst``; returns ``(sol, history)``."""
        hist: list = []
        if record:
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
            inst.set_termination_callback(_cb)
        sol = inst.solve()
        if record:
            inst.unset_termination_callback()
        return sol, hist

    def solve(self, problem: QPProblem, reuse: bool = False,
              verbose: bool = False) -> QPResult:
        t0 = time.perf_counter()
        n_eq = problem.A_eq.shape[0]
        n_le = problem.A_le.shape[0]
        A = sp.vstack([problem.A_eq, problem.A_le], format="csc")
        b = np.concatenate([problem.b_eq, problem.b_le])
        P = sp.triu(problem.P, format="csc")
        cones: list = []
        if n_eq > 0: cones.append(clarabel.ZeroConeT(n_eq))
        if n_le > 0: cones.append(clarabel.NonnegativeConeT(n_le))
        probe = problem.feasibility_only

        key = self._structure(P, A, n_eq)
        kept = self._kept[1] if (reuse and self._kept is not None
                                 and self._same(self._kept[0], key)) else None
        reused = kept is not None
        cold_restart = False
        if reused:
            inst = kept
            inst.update(P=P, q=problem.q, b=b)
        else:
            inst = clarabel.DefaultSolver(P, problem.q, A, b, cones,
                                          self._clarabel_settings(verbose, probe))
        sol, hist = self._run(inst, self.capture_iter_history)
        status = str(sol.status)

        if status not in _STATUS_MAP and reused:
            # update() keeps the equilibration computed for the first problem; a later
            # period with other σ_c can stall on stale scaling. Cold restart.
            logger.warning("status=%s on a reused instance — cold-restart retry", status)
            reused, cold_restart = False, True
            inst = clarabel.DefaultSolver(P, problem.q, A, b,
                                          cones, self._clarabel_settings(False, probe))
            sol, hist = self._run(inst, self.capture_iter_history)
            status = str(sol.status)
        self._kept = (key, inst)

        return QPResult(
            x            = np.asarray(sol.x),
            status       = _STATUS_MAP.get(status, FAILED),
            raw_status   = status,
            iterations   = int(sol.iterations),
            solve_time_s = time.perf_counter() - t0,
            objective    = float(sol.obj_val),
            info         = {**_info_to_dict(inst.get_info()),
                            "reused": reused, "cold_restart": cold_restart},
            iter_history = np.array(hist, dtype=ITER_DTYPE) if hist else None,
        )
