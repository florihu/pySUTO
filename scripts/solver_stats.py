"""
solver_stats.py
===============
Stage 4-stats driver: write the row-level edge + constraint CSVs for a
persisted opt run.

The constraint run is resolved from the opt run's sidecar
(``parents.upstream.const_run_id``), so only ``opt_run_id`` is required.

Usage::

    python scripts/solver_stats.py                          # default run, 1995
    python scripts/solver_stats.py <opt_run_id> --years 1995 2000
    python scripts/solver_stats.py <opt_run_id> --years all

CSVs land in ``data/proc/opt/<opt_run_id>/diag/``.
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
DEFAULT_OPT_RUN_ID = "opt_20260611_075036_6d2e540f"

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.diag    import SolverDiag, SolverStatsDiagnostic
from pysuto.logging import get_logger, add_run_handler

_log = get_logger("solver_stats")


def main(opt_run_id: str, years) -> None:
    os.makedirs("logs", exist_ok=True)
    add_run_handler(f"logs/solver_stats_{datetime.now():%Y%m%d_%H%M%S}.log")

    _log.info("=== solver_stats START ===")
    _log.info("Loading SolverDiag from opt_run_id='%s'", opt_run_id)

    diag = SolverDiag.from_opt_id(
        opt_run_id,
        output_dir = OUTPUT_DIR,
        const_dir  = CONST_DIR,
    )
    acc = diag.acc   # loaded once, lazily, from the const run's acc_id

    diag_dir = os.path.join(OUTPUT_DIR, opt_run_id, "diag")
    stats = SolverStatsDiagnostic(diag.results, diag.constraint_run, acc, years=years)
    paths = stats.write(diag_dir)

    _log.info("Wrote %d CSVs to %s", len(paths), diag_dir)
    _log.info("=== solver_stats DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write row-level edge + constraint stats CSVs for a persisted opt run.",
    )
    parser.add_argument(
        "opt_run_id", nargs="?", default=DEFAULT_OPT_RUN_ID,
        help="Opt run ID under data/proc/opt/. The const run is resolved "
             f"from this run's sidecar. Default: {DEFAULT_OPT_RUN_ID}",
    )
    parser.add_argument(
        "--years", nargs="+", default=["1995"],
        help="Years to evaluate (space-separated), or 'all'. Default: 1995",
    )
    args = parser.parse_args()
    years = "all" if args.years == ["all"] else args.years
    main(args.opt_run_id, years)
