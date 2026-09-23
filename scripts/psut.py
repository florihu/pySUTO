"""
psut.py
=======
Stage 6 driver: materialise the solved edge vector a* of a persisted OptimRun
as a tidy physical supply-use table (PSUT) and persist it, one parquet per
period, under data/proc/psut/<psut_run_id>/.

Pipeline
--------
1  PSUT.from_opt_run(opt_run_id, ...) — resolves the AccountingStructure via
   provenance (opt sidecar → const sidecar → acc id), maps year-local F_opt to
   global edge ids through cols_t and classifies every edge into its block
   (S, U, Y, E, O, P; every valuation, ``primary`` = the default for views)
2  PsutRunRegistry.register(psut, parents=..., label=...) — writes
   timesteps/t_<period>.parquet + nodes.parquet + the harmonized sidecar with
   parents.upstream.opt_run_id

Usage::

    python scripts/psut.py <opt_run_id> [--label LABEL] [--primary CC]
                           [--timesteps 2020 2021 ...] [--data-dir data]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
DATA_DIR  = "data"          # holds proc/opt, proc/const, proc/acc; output → proc/psut
PRIMARY   = "CC"          # valuation the PSUT views default to; every valuation is loaded
# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.psut import PSUT
from pysuto.registry import PsutRunRegistry
from pysuto.logging import get_logger, add_run_handler

_log = get_logger("init_psut")


def main(
    opt_run_id: str,
    label:      str = "",
    primary:    str = PRIMARY,
    timesteps:  list | None = None,
    data_dir:   str = DATA_DIR,
) -> str:
    os.makedirs("logs", exist_ok=True)
    add_run_handler(f"logs/init_psut_{datetime.now():%Y%m%d_%H%M%S}.log")

    _log.info("=== init_psut START ===")
    _log.info("opt_run_id=%s  primary=%s  timesteps=%s  data_dir=%s",
              opt_run_id, primary, timesteps or "all", data_dir)

    proc = os.path.join(data_dir, "proc")
    psut = PSUT.from_opt_run(
        opt_run_id,
        timesteps = timesteps,
        primary   = primary,
        opt_dir   = os.path.join(proc, "opt"),
        const_dir = os.path.join(proc, "const"),
        acc_dir   = os.path.join(proc, "acc"),
    )

    registry    = PsutRunRegistry(os.path.join(proc, "psut"))
    psut_run_id = registry.register(
        psut,
        parents = {
            "inputs":   {"acc": {"id": psut.acc_id}},
            "upstream": {"opt_run_id": opt_run_id, "const_run_id": psut.const_run_id},
        },
        label = label,
    )

    _log.info("PsutRun '%s' persisted under %s  %r", psut_run_id, registry.storage_dir, psut)
    _log.info("=== init_psut DONE ===")
    return psut_run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Materialise a persisted OptimRun as a tidy PSUT (one parquet per period)."
    )
    parser.add_argument("opt_run_id", help="OptimRun ID from data/proc/opt/.")
    parser.add_argument("--label", default="", help="Optional label for the psut run sidecar.")
    parser.add_argument("--primary", default=PRIMARY,
                        help=f"Valuation the views default to; every valuation is loaded (default {PRIMARY}).")
    parser.add_argument("--timesteps", nargs="+", default=None,
                        help="Subset of timestep labels (default: every timestep of the run).")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help=f"Data root holding proc/opt, proc/const, proc/acc (default {DATA_DIR!r}).")
    args = parser.parse_args()
    main(args.opt_run_id, args.label, args.primary, args.timesteps, args.data_dir)
