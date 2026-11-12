"""
init_copper_constraints.py
==========================
Stage 3a driver: build and persist the constraint matrix for the copper model.

The constraint builders and their registration order are the case's
(pysuto.cases.copper.builders, #120; its docstring lists the constraint types):
hard mass balance, the supply/use scale totals and the scrap level anchors, the
CSV-fed soft ratios and hard ratio bounds, the net-stock and scrap-share bands, and
the hard lower bounds.

═══════════════════════════════════════════════════════════
CONFIGURATION  ←  every knob is a field of the case spec,
                  src/pysuto/cases/copper.py (edit it there)
═══════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
# Upstream wiring (`imp_run_id`) is passed via the `--imp-run-id` CLI flag. main()
# takes the acc id from the case (`--case`, default copper; `--config` may replace it)
# and derives every path from `--data-dir` (default data): proc/acc, proc/imp,
# proc/const, input/. Every knob (P_PROP, σ floors, builder registrations) lives in the
# case spec (pysuto.cases.<id>).

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

# make sure the package is importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataclasses import replace
from pysuto.case import CaseSpec
from pysuto.cases import add_case_arg, case_from_args
from pysuto.cases.copper import CASE
from pysuto.config import load_config
from pysuto.registry import stage_lock
from pysuto.logging import get_logger, add_run_handler
from pysuto.knobs import add_knob_args, overrides_from_cli, write_run_id
from pysuto.stages import run_const

# The knobs a pipeline step may override with --set (#100, pysuto.knobs): the UPPER_CASE
# fields of the case's Stage 3 section. Paths and the acc id are not knobs.
KNOBS = tuple(CASE.const.knobs())

_log = get_logger("init_copper_constraints")


def main(imp_run_id: str, label: str, overrides: dict | None = None, *,
         case: CaseSpec = CASE,
         data_dir: str = "data", config: str | None = None,
         imp_dir: str | None = None, const_dir: str | None = None) -> str:
    """Build and persist the constraint run of ``imp_run_id``; return its id.

    The run builds from ``case``'s Stage 3 section (default the copper case; the
    command line passes it with its ``--set`` overrides applied); ``overrides`` is
    recorded in the sidecar. ``config`` names a config whose ``pipeline.acc_id``
    replaces the case's acc id (default: the case's). ``data_dir`` holds ``proc/`` and
    ``input/``; the imp run is read from ``imp_dir`` (default ``<data_dir>/proc/imp``)
    and the run written to ``const_dir`` (default ``<data_dir>/proc/const``).
    """
    from datetime import datetime
    add_run_handler(f"logs/init_copper_constraints_{datetime.now():%Y%m%d_%H%M%S}.log")
    if config is not None:
        case = replace(case, acc_id=load_config(config).pipeline.acc_id)
    return run_const(case, imp_run_id, label, overrides, data_dir=data_dir,
                     imp_dir=imp_dir, const_dir=const_dir, logger=_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and persist constraint matrix.")
    parser.add_argument(
        "--imp-run-id", required=True,
        help="ImputationRun ID to load from --imp-dir.",
    )
    parser.add_argument("label", help="Label for this constraint run")
    add_case_arg(parser)
    parser.add_argument("--data-dir", default="data",
                        help="Data root holding proc/ and input/ (default 'data').")
    parser.add_argument("--config", default=None,
                        help="Config whose pipeline.acc_id replaces the case's acc id "
                             "(default: the case's).")
    parser.add_argument("--imp-dir", default=None,
                        help="Where the imputation run is read from "
                             "(default <data-dir>/proc/imp).")
    parser.add_argument("--const-dir", default=None,
                        help="Where the run is written (default <data-dir>/proc/const); "
                             "a sweep's scratch store (#108).")
    add_knob_args(parser)
    args = parser.parse_args()
    case = case_from_args(parser, args)
    overrides = overrides_from_cli(parser, args, tuple(case.const.knobs()),
                                   case.const.knobs())
    case = case.with_knobs("const", overrides)
    # registries have no locking (#101)
    with stage_lock("constraints", root=os.path.join(args.data_dir, "proc")):
        write_run_id(args.run_id_file,
                     main(args.imp_run_id, args.label, overrides, case=case,
                          data_dir=args.data_dir, config=args.config,
                          imp_dir=args.imp_dir, const_dir=args.const_dir))
