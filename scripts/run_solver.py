"""
init_run_solver.py
==================
Stage 4 driver: solve the σ-weighted QP using Clarabel, starting from a
persisted ConstraintRun.

Pipeline
--------
1  Load ClarabelSolver + ConstraintRun
2  Feasibility pre-check (defaults to first timestep only)
3  Solve every timestep with warm-start prior  a_0(t) = a*(t-1)
4  Persist OptimResults to data/proc/opt/<opt_run_id>/

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
TIMESTEPS  = None          # None → all timesteps in the run
FEASIBILITY_CHECK_FIRST_ONLY = True

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.solver  import ClarabelSolver
from pysuto.diag    import SolverDiag
from pysuto.logging import get_logger, add_run_handler

_log = get_logger("init_run_solver")


def main(const_run_id: str, label: str = "") -> None:
    os.makedirs("logs", exist_ok=True)
    add_run_handler(
        f"logs/init_run_solver_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    _log.info("=== init_run_solver START ===")
    _log.info("Loading ClarabelSolver (const_dir=%s)", CONST_DIR)
    solver = ClarabelSolver(const_dir=CONST_DIR)

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

    diag = SolverDiag(results, solver.load_run(const_run_id),
                      label=label or optim_run_id)
    diag.summary()
    diag.plot(odir=os.path.join(OUTPUT_DIR, optim_run_id, "diag"))

    _log.info("=== init_run_solver DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve persisted ConstraintRun.")
    parser.add_argument(
        "--const-run-id", required=True,
        help="ConstraintRun ID to load from data/proc/const/.",
    )
    parser.add_argument(
        "--label", default="",
        help="Optional label for the optim run sidecar.",
    )
    args = parser.parse_args()
    main(args.const_run_id, args.label)
