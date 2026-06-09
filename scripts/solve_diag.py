"""
solve_diag.py
=============
Stage 4-diag driver: render the full SolverDiag portfolio for a persisted
opt run.

The constraint run is resolved from the opt run's sidecar
(``parents.upstream.const_run_id``), so only ``opt_run_id`` is required.

Usage::

    python scripts/solve_diag.py <opt_run_id> [--label LABEL]

Figures land in ``data/proc/opt/<opt_run_id>/diag/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join("data", "proc", "opt")
CONST_DIR  = os.path.join("data", "proc", "const")
TIMESTEPS  = None              # None → all timesteps; or e.g. (1995, 2010, 2024)

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.acc     import AccountingStructure
from pysuto.diag    import SolverDiag, SolverStatsDiagnostic
from pysuto.logging import get_logger, add_run_handler

_log = get_logger("init_solve_diag")


def main(opt_run_id: str, label: str = "") -> None:
    os.makedirs("logs", exist_ok=True)
    add_run_handler(f"logs/init_solve_diag_{datetime.now():%Y%m%d_%H%M%S}.log")

    _log.info("=== init_solve_diag START ===")
    _log.info("Loading SolverDiag from opt_run_id='%s'", opt_run_id)

    diag = SolverDiag.from_opt_id(
        opt_run_id,
        output_dir = OUTPUT_DIR,
        const_dir  = CONST_DIR,
        label      = label,
    )

    diag.summary()
    paths = diag.plot(timesteps=TIMESTEPS)

    diag_dir = os.path.join(OUTPUT_DIR, opt_run_id, "diag")
    _log.info("Wrote %d figures to %s", len(paths), diag_dir)

    acc = AccountingStructure.load(diag.constraint_run.acc_id)
    SolverStatsDiagnostic(diag.results, diag.constraint_run, acc, year="1995").write(diag_dir)

    _log.info("=== init_solve_diag DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render SolverDiag portfolio for a persisted opt run.",
    )
    parser.add_argument(
        "opt_run_id",
        help="Opt run ID under data/proc/opt/. The const run is resolved "
             "from this run's sidecar.",
    )
    parser.add_argument(
        "--label", default="",
        help="Optional display label for figures (defaults to opt_run_id).",
    )
    args = parser.parse_args()
    main(args.opt_run_id, args.label)
