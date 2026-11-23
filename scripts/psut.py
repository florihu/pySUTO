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
   (S, U, Y, E, O, P, W; every valuation, ``primary`` = the default for views)
2  PsutRunRegistry.register(psut, parents=..., label=...) — writes
   timesteps/t_<period>.parquet + nodes.parquet + the harmonized sidecar with
   parents.upstream.opt_run_id
3  --sankey [PERIOD ...] — collapse to World (keep_crossings=True), build the
   market-share PIOT, and write both Sankeys per period — the supply-use
   Sankey of the PSUT and the process-network Sankey of the PIOT — as HTML
   (+ PNG when kaleido can render) under figs/psut/

Usage::

    python scripts/psut.py <opt_run_id> [--label LABEL] [--primary CC]
                           [--timesteps 2020 2021 ...] [--data-dir data]
                           [--sankey [2020 2024 ...]] [--figs-dir figs/psut] [--no-png]
                           [--run-id-file PATH]
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# The body is pysuto.stages.run_psut (#186) and so are the constants it reads (PRIMARY,
# FIGS_DIR): this driver keeps the argument parsing, the stage lock and the log handler.
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
from pysuto.case import CaseSpec, resolve_case
from pysuto.case import add_case_arg, case_from_args
from pysuto.knobs import write_run_id
from pysuto.registry import stage_lock
from pysuto.logging import get_logger, start_run_log
from pysuto.stages import FIGS_DIR, PRIMARY, PSUT_KNOBS as KNOBS, run_psut

# The case the driver runs without --case: the repository's copper case, next to it.
DEFAULT_CASE = str(Path(__file__).resolve().with_name("copper.py"))
CASE = resolve_case(DEFAULT_CASE)

_log = get_logger("init_psut")

DATA_DIR = "data"          # holds proc/opt, proc/const, proc/acc; output → proc/psut


def main(
    opt_run_id: str,
    label:      str = "",
    primary:    str | None = None,       # None: the case's PRIMARY, else the content valuation
    timesteps:  list | None = None,
    data_dir:   str = DATA_DIR,
    sankey:     list | None = None,      # periods to draw; [] → the latest
    figs_dir:   str = FIGS_DIR,
    png:        bool = True,             # also a static PNG per figure, when kaleido can render
    case:       CaseSpec = CASE,         # recorded in the sidecar; must have made the opt run
) -> str:
    start_run_log(data_dir, "init_psut")
    return run_psut(case, opt_run_id, label, primary, timesteps, data_dir=data_dir,
                    sankey=sankey, figs_dir=figs_dir, png=png, logger=_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Materialise a persisted OptimRun as a tidy PSUT (one parquet per period)."
    )
    parser.add_argument("opt_run_id", help="OptimRun ID from data/proc/opt/.")
    parser.add_argument("--label", default="", help="Optional label for the psut run sidecar.")
    parser.add_argument("--primary", default=None,
                        help="Valuation the views default to; every valuation is loaded "
                             "(default: the case's, else the acc's content valuation).")
    parser.add_argument("--timesteps", nargs="+", default=None,
                        help="Subset of timestep labels (default: every timestep of the run).")
    add_case_arg(parser, default=DEFAULT_CASE)
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="Data root holding proc/opt, proc/const, proc/acc "
                             f"(default {DATA_DIR!r}).")
    parser.add_argument("--sankey", nargs="*", default=None, metavar="PERIOD",
                        help="Write the World supply-use and process-network Sankeys for these "
                             "periods (flag without values: the latest period).")
    parser.add_argument("--figs-dir", default=FIGS_DIR,
                        help="Directory for the Sankey HTML/PNG files "
                             f"(default {FIGS_DIR!r}).")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip the static PNG copies (needs kaleido and a browser).")
    parser.add_argument("--run-id-file", default=None, metavar="PATH",
                        help="Write the persisted run id to PATH (pysuto.pipeline).")
    args = parser.parse_args()
    case = case_from_args(parser, args)
    try:
        case.driver_knobs_for("psut", KNOBS, {"PRIMARY": PRIMARY})     # a bad case value is a usage error
    except (KeyError, TypeError) as exc:
        parser.error(exc.args[0] if exc.args else str(exc))
    with stage_lock("psut", root=os.path.join(args.data_dir, "proc")):   # #101
        write_run_id(args.run_id_file,
                     main(args.opt_run_id, args.label, args.primary, args.timesteps,
                          args.data_dir, args.sankey, args.figs_dir, not args.no_png, case))
