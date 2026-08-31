"""
init_run_solver.py
==================
Stage 4 driver: solve the σ-weighted QP using Clarabel, starting from a
persisted ConstraintRun.

Pipeline
--------
1  Load ClarabelSolver + ConstraintRun
2  Feasibility pre-check (defaults to first timestep only)
3  Solve every timestep with the warm-start prior  a_prior(t) = a_post(t-1)
   (prior_mode='warm', the ClarabelSolver default)
4  Persist OptimResults to data/proc/opt/<opt_run_id>/
5  Render a diagnostics subset and initialise the solver stats module
   (per-year edge/constraint CSVs) under data/proc/opt/<opt_run_id>/diag/

Usage::

    python scripts/run_solver.py --const-run-id CONST_RUN_ID [--label LABEL]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
# Pipeline wiring (`const_run_id`) is passed via the `--const-run-id` CLI flag.

CONST_DIR  = os.path.join("data", "proc", "const")
OUTPUT_DIR = os.path.join("data", "proc", "opt")
TIMESTEPS  = None #[1995]           # None → all timesteps in the run
# Sensitivity cold run: scan every year up front so an infeasibility introduced
# by the flat-prior-zero pin halts at — and localises — the first breaking year,
# rather than only checking year 0. See branch sensitivity/no-flat-prior-zeros.
FEASIBILITY_CHECK_FIRST_ONLY = False


# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.acc     import AccountingStructure
from pysuto.solver  import ClarabelSolver
from pysuto.diag    import SolverDiag, SolverStatsDiagnostic
from pysuto.logging import get_logger, add_run_handler

_log = get_logger("init_run_solver")


def main(const_run_id: str, label: str = "", prior_mode: str = "warm") -> None:
    os.makedirs("logs", exist_ok=True)
    add_run_handler(
        f"logs/init_run_solver_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    _log.info("=== init_run_solver START ===")
    _log.info("Loading ClarabelSolver (const_dir=%s)", CONST_DIR)
    # Termination tolerances for the scaled-σ_c QP: the iterate hits the
    # floating-point floor (μ ~1e-15) with rel gap 1e-4–5e-3 and dual residual
    # ~1e-8–7e-7 — just above the 1e-8 defaults — and Clarabel then reports
    # InsufficientProgress. Main tols 1e-7/1e-3 catch most years; the reduced
    # tols accept a floored iterate as AlmostSolved (rel gap ≤1 %, feas ≤1e-5)
    # — still orders of magnitude below any z-interpretation threshold.
    solver = ClarabelSolver(const_dir=CONST_DIR, tol_feas=1e-7,
                            tol_gap_rel=1e-3, tol_gap_abs=1e-1,
                            reduced_tol_feas=1e-5,
                            reduced_tol_gap_rel=1e-2,
                            reduced_tol_gap_abs=1e2, 
                            prior_mode=prior_mode,)

    _log.info("Solving ConstraintRun '%s'", const_run_id)
    results = solver.solve_all(
        const_run_id,
        timesteps                    = TIMESTEPS,
        feasibility_check_first_only = FEASIBILITY_CHECK_FIRST_ONLY,
    )

    optim_run_id = solver.persist(
        run_id       = const_run_id,
        results      = results,
        output_dir   = OUTPUT_DIR,
        const_run_id = const_run_id,
        label        = label,
    )

    total_iter   = sum(r.iterations   for r in results.values())
    total_time   = sum(r.solve_time_s for r in results.values())
    max_residual = max(
        float(abs(r.residuals).max()) if r.residuals.size else 0.0
        for r in results.values()
    )
    _log.info(
        "Solved %d timesteps  total_iter=%d  total_time=%.2fs  max|residual|=%.3e",
        len(results), total_iter, total_time, max_residual,
    )
    _log.info("OptimRun saved as '%s' under %s", optim_run_id, OUTPUT_DIR)

    const_run = solver.load_run(const_run_id)
    diag_dir  = os.path.join(OUTPUT_DIR, optim_run_id, "diag")

    diag = SolverDiag(results, const_run, label=label or optim_run_id)
    diag.summary()
    diag.plot(odir=diag_dir,
              plot_ids=["SD-R-01", "SD-R-08", "SD-R-09",
                        "SD-T-05", "SD-T-07", "SD-T-10", "SD-T-11"])

    # Initialise the solver stats module as part of the run: per-year edge
    # (defect) and constraint (residual) row-level CSVs. Use diag.results — the
    # reassembled global-length vectors the stats tables expect (raw solver
    # results are year-local slices).
    acc = AccountingStructure.load(const_run.acc_id)
    SolverStatsDiagnostic(diag.results, const_run, acc, years="all").write(diag_dir)

    _log.info("=== init_run_solver DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve persisted ConstraintRun.")
    parser.add_argument(
        "--const-run-id", required=True,
        help="ConstraintRun ID to load from data/proc/const/.",
    )
    parser.add_argument(
        "--label", required=False, default="",
        help="Optional label for the optim run sidecar.",
    )

    parser.add_argument(
        "--prior-mode", required=False, default='cold',
        choices=["warm", "cold"],
        help="ClarabelSolver prior mode: 'warm' (default) or 'cold'."
    )

    args = parser.parse_args()
    main(args.const_run_id, args.label, args.prior_mode)
