"""
collect_aggregate_impute.py
===========================
Stage 2 driver: feed collection, inverse-variance aggregation, and imputation.

Steps
-----
A  Load accounting structure and feed registry
B  Collect and aggregate feeds (baci, crude_ore, icsg)
C  Run the case's imputation cascade (copper: pysuto.cases.copper.imputers, #120):
     1  trade seeds, network closure, concentration, ore-grade and efficiency
        ratios, iterated to a fixpoint in CASCADE_ORDER
     2  temporal interpolation (interpolation-only)
     3  net-stock share imputation
     4  flat-prior fallback
D  Save AggregationResult to OUTPUT_DIR

═══════════════════════════════════════════════════════════
CONFIGURATION  ←  every knob is a field of the case spec,
                  src/pysuto/cases/copper.py (edit it there)
═══════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
# Upstream wiring (feed IDs) is passed via repeated `--feed-id` CLI flags. The case
# (`--case`, default copper: pysuto.cases.<id>.CASE) names the acc id, which `--config`
# may replace; every path derives from `--data-dir` (default data): proc/acc,
# proc/feed/registry, proc/imp, input/. Every knob lives in the case spec.

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataclasses     import replace
from pysuto.case     import CaseSpec
from pysuto.cases    import add_case_arg, case_from_args
from pysuto.cases.copper import CASE
from pysuto.config   import load_config
from pysuto.registry import stage_lock
from pysuto.logging  import get_logger, add_run_handler
from pysuto.knobs    import add_knob_args, overrides_from_cli, write_run_id
from pysuto.stages   import run_imp

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION — the copper case (pysuto.cases.copper, #120) declares every
# Stage 2 knob, its value and the imputer cascade; the stage body is
# pysuto.stages.run_imp (#124), which builds from the case this driver passes.
# ─────────────────────────────────────────────────────────────────────────────

# The knobs a pipeline step may override with --set (#100, pysuto.knobs): the UPPER_CASE
# fields of the case's Stage 2 section. Paths, the acc id, CV_GEO4 (a pedigree constant)
# and stock_sectors (ADR 0040 membership, kept in four places, #39) are not knobs.
KNOBS = tuple(CASE.imp.knobs())

def main(feed_ids: list[str], label: str, overrides: dict | None = None, *,
         case: CaseSpec = CASE,
         data_dir: str = "data", config: str | None = None,
         imp_dir: str | None = None, pin_inputs_from: str | None = None) -> str:
    """Run Stage 2 on ``feed_ids``; return the imp run id.

    The run builds from ``case``'s Stage 2 section (default the copper case; the
    command line passes it with its ``--set`` overrides applied); ``overrides`` is
    recorded in the sidecar. ``config`` names a config whose ``pipeline.acc_id``
    replaces the case's acc id (default: the case's). ``data_dir`` holds ``proc/`` and
    ``input/``; the run is written to ``imp_dir`` (default ``<data_dir>/proc/imp``).
    ``pin_inputs_from`` names an imp run in ``<data_dir>/proc/imp`` whose recorded input
    versions this run uses (#53).
    """
    log_path = os.path.join(
        "logs",
        f"collect_aggregate_impute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    os.makedirs("logs", exist_ok=True)
    add_run_handler(log_path)
    logger = get_logger("collect_aggregate_impute")
    logger.info("=== collect_aggregate_impute START ===")
    if config is not None:
        case = replace(case, acc_id=load_config(config).pipeline.acc_id)
    run_id = run_imp(case, feed_ids, label, overrides, data_dir=data_dir,
                     imp_dir=imp_dir, pin_inputs_from=pin_inputs_from, logger=logger)
    logger.info("=== collect_aggregate_impute DONE ===")
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feed collection, aggregation, and imputation.")
    parser.add_argument(
        "--feed-id", action="append", required=True, dest="feed_ids",
        help="Feed run ID to include. Repeat for each feed.",
    )
    parser.add_argument("--label", help="Human-readable label stored in the run's sidecar")
    add_case_arg(parser)
    parser.add_argument("--data-dir", default="data",
                        help="Data root holding proc/ and input/ (default 'data').")
    parser.add_argument("--config", default=None,
                        help="Config whose pipeline.acc_id replaces the case's acc id "
                             "(default: the case's, copper's is config/config.yaml's).")
    parser.add_argument("--imp-dir", default=None,
                        help="Where the run is written (default <data-dir>/proc/imp); a "
                             "sweep's scratch store (#108).")
    parser.add_argument("--pin-inputs-from", default=None, metavar="IMP_RUN_ID",
                        help="Use every input version this imp run of "
                             "<data-dir>/proc/imp recorded in its sidecar, instead of the "
                             "latest (#53).")
    add_knob_args(parser)
    args = parser.parse_args()
    case = case_from_args(parser, args)
    overrides = overrides_from_cli(parser, args, tuple(case.imp.knobs()), case.imp.knobs())
    case = case.with_knobs("imp", overrides)
    if case.imp.validate is not None:
        try:
            case.imp.validate(case.imp)
        except ValueError as exc:
            parser.error(str(exc))
    # registries have no locking (#101)
    with stage_lock("collect_aggregate_impute", root=os.path.join(args.data_dir, "proc")):
        write_run_id(args.run_id_file,
                     main(args.feed_ids, args.label, overrides, case=case,
                          data_dir=args.data_dir, config=args.config,
                          imp_dir=args.imp_dir, pin_inputs_from=args.pin_inputs_from))
