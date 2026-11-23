"""
solver.py
=========
Stage 4 — the σ-weighted QP orchestrator (ADR 0042).

:class:`Solver` loads a persisted :class:`pysuto.constraints.ConstraintRun`, conditions
the prior, and per period builds one neutral standard-form QP
(:class:`pysuto.qp.QPProblem`) over ``x = [a; ε]``. It owns the model: the σ weights in
``P`` and ``q``, the warm or cold prior, and the slack ``ε`` that realises every soft row
(``G a − ε = c``). An injected **QP backend** (:class:`pysuto.qp.QPBackend`, in practice
:class:`pysuto.clarabel_backend.ClarabelBackend`) solves it and reports a neutral
status; the orchestrator accepts ``solved`` and ``solved_inaccurate`` and raises on
``failed``, quoting the backend's raw status. This module imports no numerical solver.

A feasibility probe precedes the solves: per period, a zero-objective QP over that
period's hard rows and its edges' bounds ``l ≤ a ≤ u``, solved through the same
backend. Cold runs probe and solve their periods on a thread pool of ``n_workers``
(**parallel solve**, #70), each period in its own backend call; warm runs are a chain
and run serially, with the backend asked to reuse its factorisation across periods.

Row layout (solver-neutral, :func:`row_layout`): ``row_meta`` carries only ``sense`` and
``is_hard``, and a period's QP rows are, in this order,

* soft equalities ``G a − ε = c`` — one slack ``ε`` each, a free decision variable;
* hard equalities (mass balance);
* hard ``≤`` rows (ratio bounds and flow lower bounds, ADR 0039; the upper bounds and
  the zero pin reach only the feasibility probe).

Public API
----------
- :class:`OptimResult` — per-timestep solution + diagnostics
- :func:`row_layout` — ``(sense, is_hard)`` → row permutation / slack count / group sizes
- :class:`Solver` — load, solve_all, persist
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from ._conditioning import (
    SIGMA_A_CEILING_FACTOR, SIGMA_A_FLOOR_ABS, SIGMA_A_FLOOR_REL, SIGMA_C_FLOOR,
    condition_prior, condition_sigma_c,
)
from .acc import EDGE_UNIVERSE_ROLE_KEYS, AccountingStructure, check_built_acc
from .constraints import ConstraintRun
from .constraints.naming import SENSE_EQ, SENSE_LE
from .input_registry import InputRegistry
from .qp import FAILED, QPBackend, QPProblem
from .registry import ConstraintRunRegistry, OptimRunRegistry
from .logging import get_logger

logger = get_logger("pysuto.solver")


# ============================================================================
# Row realisation: (sense, is_hard) → how the QP carries the row
# ============================================================================
# Which of the three row groups a row belongs to. Groups are laid out in this order
# in the QP; a soft equality carries one slack column (``Gx - ε = c``).
GROUP_SOFT_EQ = 0
GROUP_HARD_EQ = 1
GROUP_LE      = 2
_N_GROUPS     = 3

# Total over {sense} × {is_hard}. ``None`` is a named gap, not a default.
_ROW_GROUP = {
    (SENSE_EQ, False): GROUP_SOFT_EQ,
    (SENSE_EQ, True):  GROUP_HARD_EQ,
    (SENSE_LE, True):  GROUP_LE,
    (SENSE_LE, False): None,
}


@dataclass(frozen=True)
class RowLayout:
    """Solver-neutral layout of one set of rows: ``perm`` maps the QP row order
    ``[soft eq; hard eq; le]`` to original row positions; ``n_slack`` (= soft
    equalities), ``n_eq`` (hard equalities) and ``n_le`` (``≤`` rows) are the group
    sizes."""
    perm:    np.ndarray
    n_slack: int
    n_eq:    int
    n_le:    int


def row_layout(sense, is_hard) -> RowLayout:
    """Realise per-row ``sense`` / ``is_hard`` arrays as a :class:`RowLayout`.

    Raises ``NotImplementedError`` on a soft ``le`` row (no builder emits one),
    ``ValueError`` on an unknown ``sense``.
    """
    sense   = np.asarray(sense)
    is_hard = np.asarray(is_hard).astype(bool)
    group   = np.full(sense.size, -1, dtype=np.int64)
    for (s, hard), g in _ROW_GROUP.items():
        mask = (sense == s) & (is_hard == hard)
        if not mask.any():
            continue
        if g is None:
            raise NotImplementedError(
                f"{int(mask.sum())} row(s) with sense={s!r}, is_hard={hard}: a soft "
                f"inequality has no realisation in the QP (no builder emits one)"
            )
        group[mask] = g
    if (group < 0).any():
        raise ValueError(
            f"unknown row sense(s) {sorted(set(sense[group < 0].astype(str)))}; "
            f"expected {SENSE_EQ!r} or {SENSE_LE!r}"
        )
    idx = [np.flatnonzero(group == g) for g in range(_N_GROUPS)]
    return RowLayout(
        perm    = np.concatenate(idx),
        n_slack = idx[GROUP_SOFT_EQ].size,
        n_eq    = idx[GROUP_HARD_EQ].size,
        n_le    = idx[GROUP_LE].size,
    )


# =============================================================================
# OptimResult
# =============================================================================

@dataclass
class OptimResult:
    """One timestep's solver output."""
    timestep:      str
    F_opt:         np.ndarray      # (E,)
    eps_opt:       np.ndarray      # (n_slack,) — slack on soft rows in original row order
    c_star:        np.ndarray      # (M,) — c + ε (ε=0 for hard rows). Diagnostic.
    residuals:     np.ndarray      # (M,) — G @ F_opt - c (original row order)
    objective:     float           # J − Σ(a_prior/σ_a)², the flows form's cost, under
                                   # either qp_variables (#33)
    iterations:    int
    solve_time_s:  float
    solver_status: str             # neutral: solved | solved_inaccurate (ADR 0042)
    convergence:   dict            # the backend's final info snapshot (JSON-safe scalars)
    # The backend's per-iteration history, a structured array whose dtype the backend
    # owns (ClarabelBackend: pysuto.clarabel_backend.ITER_DTYPE). None when it keeps none.
    iter_history:  Optional[np.ndarray] = None
    # F_opt/residuals/c_star are the year-local slices over these global indices. None
    # on runs persisted before the year-local fix (full-vector results).
    cols_t:        Optional[np.ndarray] = None  # global edge cols for this t
    rows_t:        Optional[np.ndarray] = None  # global constraint rows for this t
    # The per-period prior mean actually used in the objective q-vector (cols_t order):
    # warm = a_post(t-1), cold = f_agg(t). None for runs predating the warm-start fix.
    a_prior:       Optional[np.ndarray] = None
    # Which backend solved it and its own status string (ADR 0042). "" when not
    # recorded; the registry fills "clarabel" and the raw status on runs written
    # before the split.
    backend:        str = ""
    backend_status: str = ""


# =============================================================================
# Solver
# =============================================================================

# Which periods the feasibility probe covers before the solves (solve_all, #104): every
# requested period ("all") or none ("none", a sweep point). The first-period-only
# probe is gone: a per-period probe is ≈ 1/30 of the full-G one (ADR 0042).
FEASIBILITY_CHECKS = ("all", "none")


class Solver:
    """
    σ-weighted QP orchestrator for Stage 4; the numerics are the injected ``backend``'s.

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
    - Cross-timestep speedup is the backend's: a serial run asks it to ``reuse`` its
      factorisation, which only the backend can honour (``P`` and ``A`` are
      time-invariant when no builder emits year-specific coefficients) — a
      *numerical* warm-start, distinct from the *prior* warm-start above. A parallel
      solve gives every period its own call and no reuse.
    - Hard rows (equalities and ``≤`` rows) are enforced exactly. The
      feasibility probe rules out impossible problems before the main solve and
      halts on any infeasibility.

    Σ_a conditioning
    ----------------
    A floor and a ceiling are applied to ``sigma_a`` to bound the
    condition number of ``Σ_a^{-1}``:

        floor   = max(σ_a_floor_abs,  σ_a_floor_rel · |a_0|)
        ceiling = σ_a_ceiling_factor · max(finite σ_a)

    Defaults give ~12 orders of magnitude condition number.

    QP variables (#33)
    ------------------
    ``qp_variables='flows'`` (the default) solves for ``a``: ``q = −2·Σ_a^{-1}·a_prior``
    and the backend's cost is J − Σ(a_prior/σ_a)². ``'defects'`` solves the same QP for
    the defect ``d = a − a_prior`` (same P and A, ``q = 0``, ``b = c − G·a_prior``), so
    its cost is J itself. ``F_opt`` comes back as ``a_prior + d`` and ``objective`` in
    the flows convention (J − Σ(a_prior/σ_a)²) either way; ``convergence`` and
    ``iter_history`` keep the backend's own cost. gap_rel's denominator is that cost, so
    the same ``tol_gap_rel`` is ~10× tighter in absolute terms under ``'defects'``
    (J ≈ 1e5 against |J − Σ(a_prior/σ_a)²| ≈ 1e6 on copper). The feasibility probe has
    no σ-weighted cost.
    """

    def __init__(
        self,
        backend:   QPBackend,
        const_dir: str = "data/proc/const",
        opt_dir:   str = "data/proc/opt",
        acc_dir:   str = "data/proc/acc",
        input_dir: str = "data/input",
        sigma_a_floor_abs:      float = SIGMA_A_FLOOR_ABS,
        sigma_a_floor_rel:      float = SIGMA_A_FLOOR_REL,
        sigma_a_ceiling_factor: float = SIGMA_A_CEILING_FACTOR,
        sigma_c_floor:          float = SIGMA_C_FLOOR,
        verbose_first_solve:    bool  = True,
        prior_mode:             str   = "warm",
        n_workers:              int   = 1,
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
        self.backend                = backend
        self.const_dir              = const_dir
        self.opt_dir                = opt_dir
        self.acc_dir                = acc_dir
        self.input_dir              = input_dir
        self.prior_mode             = prior_mode
        self.n_workers              = n_workers
        self.qp_variables           = qp_variables
        self.sigma_a_floor_abs      = sigma_a_floor_abs
        self.sigma_a_floor_rel      = sigma_a_floor_rel
        self.sigma_a_ceiling_factor = sigma_a_ceiling_factor
        self.sigma_c_floor          = sigma_c_floor
        self.verbose_first_solve    = verbose_first_solve
        # How the last solve_all ran (worker threads, backend and its settings,
        # wall-clock) and the inputs it read (the acc of a warm run); persist() writes
        # them into the sidecar.
        self._solve_config:  dict = {}
        self._solve_summary: dict = {}
        self._solve_inputs:  dict = {}

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

    # ------------------------------------------------------------------
    # Per-period parallel map
    # ------------------------------------------------------------------

    def _map_periods(self, labels: list, fn: Callable, what: str) -> dict:
        """``{t: fn(t)}`` over ``labels``. One worker: a plain loop, the first
        exception propagates. Several (cold only): a thread pool of ``min(n_workers,
        periods)`` threads — the backends release the GIL in ``solve()`` (8 periods of
        const_20261012_113828_12c562c7 took 63.3 s on 8 threads vs 549 s serially,
        bit-identical, #70). The first failed period cancels the periods not yet
        started; the RuntimeError names every period that failed, and nothing is
        returned."""
        if self.n_workers == 1:
            return {t: fn(t) for t in labels}
        n_threads = min(self.n_workers, len(labels))
        logger.info("%s: %d periods on %d threads", what, len(labels), n_threads)
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            fut_by_t = {t: pool.submit(fn, t) for t in labels}
            for fut in as_completed(fut_by_t.values()):
                if fut.exception() is not None:
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
        # Leaving the with-block waited for the periods already running.
        failed = {t: f.exception() for t, f in fut_by_t.items()
                  if not f.cancelled() and f.exception() is not None}
        if failed:
            raise RuntimeError(
                f"{what} failed in {len(failed)} period(s) "
                f"{sorted(failed)}; periods not yet started were cancelled — "
                + " | ".join(str(e) for e in failed.values())
            ) from next(iter(failed.values()))
        return {t: fut_by_t[t].result() for t in labels}

    # ------------------------------------------------------------------
    # Feasibility probe
    # ------------------------------------------------------------------

    def feasibility_check(self, run: ConstraintRun, t_label: str) -> None:
        """
        Halt if the hard rows of period ``t_label`` (equalities + ``≤`` rows) plus its
        edges' flow bounds are infeasible. Solves a feasibility-only QP
        ``min ½ ‖a‖²`` (P=εI, q=0) on the period's hard subsystem through the backend.

        ``ε`` (a tiny diagonal) keeps the backend happy with a strictly PSD P;
        the choice does not affect feasibility detection. Flow bounds enter as
        ``a ≤ u`` and ``−a ≤ −l`` on every edge of the period; the probe-only bounds
        (the zero pin, the prior box) are part of ``u`` and ``l``.
        """
        cols_t  = run.edge_cols_by_t[t_label]
        rows_t  = np.where(run.row_meta["time"].astype(str).to_numpy() == t_label)[0]
        rm_t    = run.row_meta.iloc[rows_t]
        G_t     = run.G_blocks[t_label].tocsr()
        c_t     = np.asarray(run.timestep_data[t_label]["c"], dtype=float)[rows_t]
        layout  = row_layout(rm_t["sense"].to_numpy(), rm_t["is_hard"].to_numpy())
        n_a     = G_t.shape[1]
        idx_eq  = layout.perm[layout.n_slack:layout.n_slack + layout.n_eq]
        idx_le  = layout.perm[layout.n_slack + layout.n_eq:]

        I = sp.eye(n_a, format="csr")
        problem = QPProblem(
            P     = sp.diags(np.full(n_a, 1e-10), format="csc"),
            q     = np.zeros(n_a),
            A_eq  = G_t[idx_eq],
            b_eq  = c_t[idx_eq],
            A_le  = sp.vstack([G_t[idx_le], I, -I], format="csr"),
            b_le  = np.concatenate([c_t[idx_le], run.u[cols_t], -run.l[cols_t]]),
            n_a   = n_a,
            n_eps = 0,
            feasibility_only = True,
        )
        res = self.backend.solve(problem)
        if res.status == FAILED:
            self._raise_infeasible(t_label, rm_t, G_t, c_t, res)

    def _raise_infeasible(self, t_label, rm_t, G_t, c_t, res):
        # Best-effort diagnostic: which hard rows have the largest residual at the
        # (infeasible) point the backend returned; the residual on hard rows tells the
        # user where to widen bounds.
        try:
            x = np.asarray(res.x)
            n_a = G_t.shape[1]
            if x.size >= n_a:
                resid = G_t @ x[:n_a] - c_t
                hard = rm_t["is_hard"].to_numpy().astype(bool)
                hard_resid = np.where(hard, np.abs(resid), 0.0)
                worst = np.argsort(-hard_resid)[:5]
                worst_msg = "; ".join(
                    f"row={int(i)} cid={rm_t.iloc[int(i)]['constraint_id']!r} "
                    f"|residual|={hard_resid[i]:.3e}"
                    for i in worst
                )
            else:
                worst_msg = "(no diagnostic — solution vector unavailable)"
        except Exception as e:                              # pragma: no cover
            worst_msg = f"(diagnostic failed: {e})"

        raise RuntimeError(
            f"Feasibility check FAILED at timestep {t_label!r}: "
            f"{self.backend.name} status={res.raw_status}. "
            f"Hard rows with the largest |G·a - c| at the solver's last point "
            f"(a hint, not the conflicting set): {worst_msg}. "
            f"A data contradiction is fixed in the input, never by dropping the row "
            f"or loosening σ (ADR 0019, 0035): check the inputs behind these rows "
            f"(the params row of a ratio band, the feed behind a scale total), "
            f"correct the one that is wrong and rebuild the ConstraintRun."
        )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve_all(
        self,
        run_id:    str,
        timesteps: Optional[list] = None,
        feasibility_check: str = "all",
    ) -> dict[str, OptimResult]:
        """
        Solve every requested timestep. Performs the feasibility probe first;
        halts if any probed timestep is infeasible.

        Parameters
        ----------
        feasibility_check:
            Which timesteps the probe covers before any solve: ``"all"`` (default; one
            zero-objective QP per requested period, on the same thread pool as the
            solves) or ``"none"`` (a sweep point, #104). Without the probe an
            infeasible period still raises in its own solve — any ``failed`` status
            does — only not localised by it.
        """
        if feasibility_check not in FEASIBILITY_CHECKS:
            raise ValueError(
                f"feasibility_check={feasibility_check!r}; expected one of "
                f"{FEASIBILITY_CHECKS} (the first-period-only probe is gone: the probe "
                f"is per period, ADR 0042)")
        run = self.load_run(run_id)

        if timesteps is None:
            ts_labels = list(run.timestep_data.keys())
        else:
            ts_labels = [str(t) for t in timesteps]
            missing = set(ts_labels) - set(run.timestep_data.keys())
            if missing:
                raise KeyError(f"timesteps not in run: {sorted(missing)}")

        check_labels = ts_labels if feasibility_check == "all" else []

        # A failed call must not leave the previous call's facts for persist().
        self._solve_config, self._solve_summary, self._solve_inputs = {}, {}, {}
        if not run.G_blocks:
            raise ValueError(
                f"constraint run {run_id!r} has no per-period partition (G_blocks): "
                f"Stage 4 solves one QP per period and the full-G path is gone; "
                f"rebuild the run with Stage 3"
            )
        t0 = time.perf_counter()
        results = self._solve_all_via_blocks(run, ts_labels, check_labels)
        settings = dict(self.backend.settings)
        self._solve_config = {
            "n_workers":        min(self.n_workers, len(ts_labels)),
            "qp_variables":     self.qp_variables,
            "backend":          self.backend.name,
            "backend_settings": settings,
            "sigma_a_floor_abs":      self.sigma_a_floor_abs,
            "sigma_a_floor_rel":      self.sigma_a_floor_rel,
            "sigma_a_ceiling_factor": self.sigma_a_ceiling_factor,
            "sigma_c_floor":          self.sigma_c_floor,
        }
        # Persisted before the backend split under these names (#70); kept.
        for k in ("max_threads", "direct_solve_method"):
            if k in settings:
                self._solve_config[k] = settings[k]
        # Wall-clock after loading: feasibility probe + period solves.
        self._solve_summary = {"solve_wall_s": time.perf_counter() - t0}
        return results

    def _solve_all_via_blocks(
        self,
        run:       ConstraintRun,
        ts_labels: list[str],
        check_labels: list[str],
    ) -> dict[str, OptimResult]:
        """Per-t solve path using run.G_blocks: each per-t QP is ~30× smaller than a
        full-G one. Cold runs with ``n_workers > 1`` probe and solve their periods
        concurrently, one backend call each. Otherwise the periods run in order and
        the backend is asked to ``reuse`` its factorisation (it does so only for
        structurally identical consecutive periods)."""
        if check_labels:
            logger.info("Feasibility pre-check on %d/%d timesteps…",
                        len(check_labels), len(ts_labels))
            t0 = time.perf_counter()
            self._map_periods(check_labels,
                              lambda t: self.feasibility_check(run, t),
                              "Parallel feasibility check")
            logger.info("Feasibility pre-check passed (%.1f s).", time.perf_counter() - t0)

        a_0_global, sigma_a_global = self._condition_prior(
            run.a_prior_global, run.sigma_a,
        )
        inv_sa_global = 1.0 / (sigma_a_global ** 2)
        row_time = run.row_meta["time"].astype(str).to_numpy()

        if self.n_workers > 1:
            def _solve_one_period(t: str) -> OptimResult:
                cols_t = run.edge_cols_by_t[t]
                return self._solve_period(
                    run, t, a_0_global[cols_t].copy(), inv_sa_global[cols_t], row_time,
                )
            return self._map_periods(ts_labels, _solve_one_period, "Parallel solve")

        # Edge-template map for the warm-start carry-forward: a_prior(t)=a_post(t-1)
        # aligned by (from_id,to_id,valuation), since cols_t are disjoint per year.
        acc_ws = None
        if self.prior_mode == "warm":
            acc_ws = self._warm_prior_acc(run)
            self._solve_inputs = {"acc": dict(run.acc_ref)}    # the input it read

        results: dict[str, OptimResult] = {}
        prev_cols: Optional[np.ndarray] = None
        prev_F:    Optional[np.ndarray] = None

        for k, t in enumerate(ts_labels):
            cols_t  = run.edge_cols_by_t[t]
            # Per-period prior mean a_prior(t): warm → a_post(t-1) aligned by
            # template; cold seed (t0, or every year in cold mode) → conditioned
            # f_agg slice.
            a_prior_t = a_0_global[cols_t].copy()
            if self.prior_mode == "warm" and prev_F is not None:
                g   = acc_ws.warm_start_gather(prev_cols, cols_t)
                hit = g >= 0
                a_prior_t[hit] = prev_F[g[hit]]

            results[t] = self._solve_period(
                run, t, a_prior_t, inv_sa_global[cols_t], row_time,
                reuse   = k > 0,         # the first period starts from a fresh instance
                verbose = self.verbose_first_solve and k == 0,
            )

            # Carry this period's posterior forward as next year's warm prior,
            # aligned to t+1's edges by template (warm_start_gather above). The
            # old in-place a_0_t_global[cols_t]=F_opt write was a no-op (cols_t
            # are disjoint per year, so a later year never read it).
            prev_cols, prev_F = cols_t, results[t].F_opt

        return results

    def _warm_prior_acc(self, run: ConstraintRun):
        """The built AccountingStructure a warm run aligns each year's prior by
        (#53): its build record checked against the const run's acc ref (the two acc
        input versions' edge universes, read from ``input_dir``), then loaded from
        ``acc_dir``.

        Raises
        ------
        ValueError
            The const run names no acc id and version, or the structure was built from
            an acc input version with another edge universe.
        RuntimeError
            The structure cannot be loaded from ``acc_dir``.
        """
        check_built_acc(self.acc_dir, run.acc_ref,
                        InputRegistry("acc", root=self.input_dir),
                        EDGE_UNIVERSE_ROLE_KEYS)
        try:
            return AccountingStructure.load(run.acc_id, storage_dir=self.acc_dir)
        except (AssertionError, OSError) as exc:
            raise RuntimeError(
                f"warm prior: AccountingStructure {run.acc_id!r} cannot be loaded from "
                f"{self.acc_dir} ({exc}); a warm run aligns each year's prior to the "
                f"previous year's posterior by edge template"
            ) from exc

    def _period_problem(
        self,
        run:       ConstraintRun,
        t:         str,
        rows_t:    np.ndarray,
        a_prior_t: np.ndarray,
        inv_sa_t:  np.ndarray,
    ):
        """Period ``t``'s neutral QP and what is needed to read its answer back:
        ``(problem, perm, a_shift, const)``. ``perm`` orders the period's rows as the QP
        does (soft, hard equalities, ``≤``); ``a_shift`` is what the backend's variables
        are measured from; ``const`` moves the backend's cost back to the flows
        convention."""
        G_t     = run.G_blocks[t]
        rm_t    = run.row_meta.iloc[rows_t]
        c_t     = np.asarray(run.timestep_data[t]["c"])[rows_t]
        sc_t    = np.asarray(run.timestep_data[t]["sigma_c"])[rows_t]

        layout   = row_layout(rm_t["sense"].to_numpy(), rm_t["is_hard"].to_numpy())
        perm     = layout.perm
        n_slack, n_eq, n_le = layout.n_slack, layout.n_eq, layout.n_le
        n_a      = G_t.shape[1]
        m_t      = G_t.shape[0]

        G_p = G_t.tocsr()[perm]
        S_top = -sp.eye(n_slack, format="csr") if n_slack else sp.csr_matrix((0, 0))
        S_bot = sp.csr_matrix((n_eq + n_le, n_slack))
        S     = sp.vstack([S_top, S_bot], format="csr") if n_slack else \
                sp.csr_matrix((m_t, 0))
        A     = sp.hstack([G_p, S], format="csr")
        b     = c_t[perm]

        if n_slack:
            sc_arr   = condition_sigma_c(sc_t, floor=self.sigma_c_floor)
            inv_scf  = 1.0 / (sc_arr[perm[:n_slack]] ** 2)
        else:
            inv_scf = np.zeros(0)
        P = sp.diags(2.0 * np.concatenate([inv_sa_t, inv_scf]), format="csc")
        # Prior offset (#33): the backend solves for x = a − a_shift. "flows": a_shift = 0;
        # "defects": a_shift = a_prior, so q = 0 and the backend's cost is J itself rather
        # than J − Σ(a_prior/σ_a)² (2015 of const_20261019_170151_13460e94: 1,288,065
        # against an optimal J of 117,282). const moves the reported objective back
        # to the flows convention.
        a_shift = a_prior_t if self.qp_variables == "defects" else np.zeros(n_a)
        q       = np.concatenate([-2.0 * inv_sa_t * (a_prior_t - a_shift),
                                  np.zeros(n_slack)])
        b       = b - G_p @ a_shift
        const   = float(inv_sa_t @ (a_shift * (2.0 * a_prior_t - a_shift)))

        n_top = n_slack + n_eq
        problem = QPProblem(
            P = P, q = q,
            A_eq = A[:n_top], b_eq = b[:n_top],
            A_le = A[n_top:], b_le = b[n_top:],
            n_a = n_a, n_eps = n_slack,
        )
        return problem, perm, a_shift, const

    def _solve_period(
        self,
        run:       ConstraintRun,
        t:         str,
        a_prior_t: np.ndarray,
        inv_sa_t:  np.ndarray,
        row_time:  np.ndarray,
        reuse:     bool = False,
        verbose:   bool = False,
    ) -> OptimResult:
        """Build period ``t``'s QP, solve it through the backend and return its
        OptimResult. ``reuse`` asks the backend to keep its factorisation from the
        previous period (serial path only); a ``failed`` status raises."""
        cols_t  = run.edge_cols_by_t[t]
        rows_t  = np.where(row_time == t)[0]
        G_t     = run.G_blocks[t]
        c_t     = np.asarray(run.timestep_data[t]["c"])[rows_t]
        problem, perm, a_shift, const = self._period_problem(
            run, t, rows_t, a_prior_t, inv_sa_t)
        n_a, n_slack, m_t = problem.n_a, problem.n_eps, G_t.shape[0]

        res = self.backend.solve(problem, reuse=reuse, verbose=verbose)
        if res.status == FAILED:
            raise RuntimeError(
                f"Timestep {t!r}: {self.backend.name} status={res.raw_status}; halting "
                f"(no fallback). iterations={res.iterations}"
            )

        x         = res.x
        F_opt_t   = a_shift + x[:n_a]
        eps_opt_t = x[n_a:] if n_slack else np.zeros(0)

        # ε back into row-order on the per-t row block
        eps_full_t = np.zeros(m_t)
        if n_slack:
            eps_full_t[perm[:n_slack]] = eps_opt_t

        # Year-local persistence (REP §8.2): store only this period's
        # solved values. F_opt over cols_t, residuals/c_star over rows_t —
        # no cross-year prior placeholders. Residual is the block residual
        # G_t @ F_opt_t - c_t (rows_t order); mass balance ≈ 0 by construction.
        residuals_t = np.asarray(G_t @ F_opt_t - c_t)
        c_star_t    = c_t + eps_full_t

        result = OptimResult(
            timestep       = t,
            F_opt          = F_opt_t,
            eps_opt        = eps_opt_t,
            c_star         = c_star_t,
            residuals      = residuals_t,
            objective      = float(res.objective) - const,
            iterations     = int(res.iterations),
            solve_time_s   = res.solve_time_s,
            solver_status  = res.status,
            convergence    = dict(res.info) if res.info else {},
            iter_history   = res.iter_history,
            cols_t         = cols_t,
            rows_t         = rows_t,
            a_prior        = a_prior_t,
            backend        = self.backend.name,
            backend_status = res.raw_status,
        )

        logger.info(
            "t=%s  status=%s (%s)  iters=%d  obj=%.4e  time=%.2fs  (per-t: %d cols × %d rows)",
            t, res.status, res.raw_status, res.iterations, result.objective,
            res.solve_time_s, n_a, m_t,
        )
        return result

    def objective_at(
        self,
        run:     ConstraintRun,
        t:       str,
        F:       np.ndarray,
        a_prior: Optional[np.ndarray] = None,
    ) -> dict:
        """Score a year-local posterior ``F`` (over ``run.edge_cols_by_t[t]``) under
        period ``t``'s QP, conditioned exactly as :meth:`_solve_period` conditions it.

        Returns ``J = Σ((F − a_prior)/σ_a)² + Σ_free((G_t·F − c_t)/σ_c)²`` as ``total``
        — the defects form's cost, and the flows form's ``objective`` plus
        ``Σ(a_prior/σ_a)²`` — split by family as the objective decomposition splits
        it: ``edges`` (defects), ``sum_rows`` (free ``type == "sum"`` rows),
        ``soft_ratios`` (free ``type == "ratio"`` rows) and ``other_soft`` (any other
        free row). ``a_prior`` defaults to the cold seed, the conditioned ``f_agg``
        slice; pass ``OptimResult.a_prior`` to score under a warm run's prior. Lets a
        sweep score each point's answer under the base chain's σ (#102).

        ``max_hard_violation`` (kt) is the largest breach of the period's hard rows by
        ``F`` — ``|G·F − c|`` on hard equality rows, ``max(0, G·F − c)`` on ``le`` rows. A
        posterior from a QP with other hard rows can break these; ``J`` is then not
        a comparison at a feasible point.
        """
        cols_t = run.edge_cols_by_t[t]
        F = np.asarray(F, dtype=float)
        if F.shape != cols_t.shape:
            raise ValueError(f"period {t!r} has {cols_t.size} edges, F has shape {F.shape}")
        a_0, sigma_a = self._condition_prior(run.a_prior_global, run.sigma_a)
        prior = a_0[cols_t] if a_prior is None else np.asarray(a_prior, dtype=float)
        edges = float((((F - prior) / sigma_a[cols_t]) ** 2).sum())

        rows_t = np.where(run.row_meta["time"].astype(str).to_numpy() == t)[0]
        rm_t   = run.row_meta.iloc[rows_t]
        sense  = rm_t["sense"].to_numpy()
        hard   = rm_t["is_hard"].to_numpy().astype(bool)
        free   = (sense == SENSE_EQ) & ~hard
        typ    = rm_t["type"].astype(str).to_numpy()
        c_t    = np.asarray(run.timestep_data[t]["c"], dtype=float)[rows_t]
        sc_t   = condition_sigma_c(np.asarray(run.timestep_data[t]["sigma_c"])[rows_t],
                                   floor=self.sigma_c_floor)
        resid  = run.G_blocks[t] @ F - c_t
        zsq    = np.where(free, (resid / sc_t) ** 2, 0.0)
        breach = np.concatenate([np.abs(resid[hard & (sense == SENSE_EQ)]),
                                 np.maximum(resid[hard & (sense == SENSE_LE)], 0.0), [0.0]])
        out = {
            "edges":       edges,
            "sum_rows":    float(zsq[free & (typ == "sum")].sum()),
            "soft_ratios": float(zsq[free & (typ == "ratio")].sum()),
            "other_soft":  float(zsq[free & (typ != "sum") & (typ != "ratio")].sum()),
        }
        out["total"] = edges + out["sum_rows"] + out["soft_ratios"] + out["other_soft"]
        out["max_hard_violation"] = float(breach.max())
        return out

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
        used), ``qp_variables``, the ``backend`` and its ``backend_settings`` (and, from
        a backend that has them, ``max_threads`` and ``direct_solve_method``) — is
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
                "inputs":   dict(self._solve_inputs),
                "upstream": {"const_run_id": const_id},
            },
            stage_config  = stage_config,
            stage_summary = dict(self._solve_summary) or None,
            label = label,
        )
        logger.info("OptimRun '%s' persisted to %s",
                    opt_run_id, registry.storage_dir)
        return opt_run_id
