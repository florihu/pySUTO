"""
init_solve_diag.py
==================
Stage 3b+3c driver: optionally solve & persist, then render the full
SolverDiag portfolio (16 figures + provenance .md).

Two modes:
  SOLVE_AND_PERSIST = False  →  load an existing opt_run_id  (Stage 3c only)
  SOLVE_AND_PERSIST = True   →  solve CONST_RUN_ID, persist, then run diag
                                (Stage 3b + 3c in one shot)

Figures land in ``{OUTPUT_DIR}/{OPT_RUN_ID}/diag/``.

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

SOLVE_AND_PERSIST = False         # True → solve + persist; False → load existing opt_run_id

# --- used when SOLVE_AND_PERSIST = True ---
CONST_RUN_ID = "const_copper_20260512_083803_30c16790"
TIMESTEPS    = None               # None → all timesteps; or e.g. (1995, 2010, 2024)
FEASIBILITY_CHECK_FIRST_ONLY = True

# --- used when SOLVE_AND_PERSIST = False ---
OPT_RUN_ID = "optim_const_copper_20260512_083803_30c16790_20260512_093956"

# --- always used ---
OUTPUT_DIR = os.path.join("data", "proc", "opt")
CONST_DIR  = os.path.join("data", "proc", "const")
LABEL      = "copper"

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.solver  import ClarabelSolver
from pysuto.diag    import SolverDiag
from pysuto.logging import get_logger, add_run_handler

_log = get_logger("init_solve_diag")


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    add_run_handler(f"logs/init_solve_diag_{datetime.now():%Y%m%d_%H%M%S}.log")
    _log.info("=== init_solve_diag START ===")

    if SOLVE_AND_PERSIST:
        _log.info("Mode: solve + persist  (CONST_RUN_ID='%s')", CONST_RUN_ID)
        solver = ClarabelSolver(const_dir=CONST_DIR)
        results = solver.solve_all(
            CONST_RUN_ID,
            timesteps=TIMESTEPS,
            feasibility_check_first_only=FEASIBILITY_CHECK_FIRST_ONLY,
        )
        opt_run_id = solver.persist(
            run_id       = CONST_RUN_ID,
            results      = results,
            output_dir   = OUTPUT_DIR,
            const_run_id = CONST_RUN_ID,
            label        = LABEL,
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
        _log.info("OptimRun saved as '%s' under %s", opt_run_id, OUTPUT_DIR)
    else:
        _log.info("Mode: diag only  (OPT_RUN_ID='%s')", OPT_RUN_ID)
        opt_run_id = OPT_RUN_ID

    _log.info("Loading SolverDiag from opt_run_id='%s'", opt_run_id)
    diag = SolverDiag.from_opt_id(
        opt_run_id,
        output_dir = OUTPUT_DIR,
        const_dir  = CONST_DIR,
        label      = LABEL,
    )

    diag.summary()
    paths = diag.plot(timesteps=TIMESTEPS)

    _log.info("Wrote %d figures to %s", len(paths),
              os.path.join(OUTPUT_DIR, opt_run_id, "diag"))
    _log.info("=== init_solve_diag DONE ===")


if __name__ == "__main__":
    main()
