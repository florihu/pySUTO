"""
run_optim_copper.py
===================
Stage 3b driver: solve the σ-weighted QP for the copper model using
Clarabel, starting from a persisted ConstraintRun.

Pipeline
--------
1  Load ClarabelSolver + ConstraintRun
2  Feasibility pre-check across all timesteps (halts on infeasibility)
3  Solve every timestep with warm-start prior  a_0(t) = a*(t-1)
4  Persist OptimResults to data/proc/optim/

═══════════════════════════════════════════════════════════
USER CONFIGURATION  ←  only section you ever edit
═══════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

CONST_RUN_ID = "const_copper_20260504_125811_9478f497"
CONST_DIR    = os.path.join("data", "proc", "const")
OUTPUT_DIR   = os.path.join("data", "proc", "opt")
TIMESTEPS    = None          # None → all timesteps in the run
FEASIBILITY_CHECK_FIRST_ONLY = True  # True → check only first timestep (fast); False → check all

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.solver  import ClarabelSolver
from pysuto.diag    import SolverDiag
from pysuto.logging import get_logger, add_run_handler

_log = get_logger("run_optim_copper")


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    add_run_handler(
        f"logs/run_optim_copper_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    _log.info("=== run_optim_copper START ===")
    _log.info("Loading ClarabelSolver (const_dir=%s)", CONST_DIR)
    solver = ClarabelSolver(const_dir=CONST_DIR)

    _log.info("Solving ConstraintRun '%s'", CONST_RUN_ID)
    results = solver.solve_all(
        CONST_RUN_ID,
        timesteps=TIMESTEPS,
        feasibility_check_first_only=FEASIBILITY_CHECK_FIRST_ONLY,
    )

    optim_run_id = solver.persist(
        run_id        = CONST_RUN_ID,
        results       = results,
        output_dir    = OUTPUT_DIR,
        const_run_id  = CONST_RUN_ID,
    )

    # Brief end-of-run summary
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

    # ------------------------------------------------------------------
    # Solver diagnostics (Stage 3c) — 15-figure portfolio under {run}/diag
    # ------------------------------------------------------------------
    diag = SolverDiag(results, solver.load_run(CONST_RUN_ID), label="copper")
    diag.summary()
    diag.plot(odir=os.path.join(OUTPUT_DIR, optim_run_id, "diag"))

    _log.info("=== run_optim_copper DONE ===")


if __name__ == "__main__":
    main()
