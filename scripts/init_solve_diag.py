"""
init_solve_diag.py
==================
Stage 3c driver: render the SolverDiag 15-figure portfolio for a persisted
optim run, loaded by opt_run_id.

Figures land in ``{OUTPUT_DIR}/{OPT_RUN_ID}/diag/`` and follow the catalogue
IDs ``SD-R-01..08`` (run-scope) and ``SD-T-01..07`` (per-timestep cross-section)
documented in README §Stage 3c.

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

OPT_RUN_ID = "optim_const_copper_20260429_070511_db39a7d4_20260511_084052"
OUTPUT_DIR = os.path.join("data", "proc", "opt")
CONST_DIR  = os.path.join("data", "proc", "const")
LABEL      = "copper"
TIMESTEPS  = (1995, 2010, 2024)   # None → (t_min, t_worst, t_max). Or pass a tuple, e.g. ("2010","2016","2023").

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.diag    import SolverDiag
from pysuto.logging import get_logger, add_run_handler

_log = get_logger("init_solve_diag")


def main() -> None:
    os.makedirs("logs", exist_ok=True)
    add_run_handler(f"logs/init_solve_diag_{datetime.now():%Y%m%d_%H%M%S}.log")

    _log.info("=== init_solve_diag START ===")
    _log.info("Loading SolverDiag from opt_run_id='%s'", OPT_RUN_ID)
    diag = SolverDiag.from_opt_id(
        OPT_RUN_ID,
        output_dir = OUTPUT_DIR,
        const_dir  = CONST_DIR,
        label      = LABEL,
    )

    diag.summary()
    paths = diag.plot(timesteps=TIMESTEPS)   # default odir → {OUTPUT_DIR}/{OPT_RUN_ID}/diag

    _log.info("Wrote %d figures to %s", len(paths),
              os.path.join(OUTPUT_DIR, OPT_RUN_ID, "diag"))
    _log.info("=== init_solve_diag DONE ===")


if __name__ == "__main__":
    main()
