"""
sweep.py
========
Sweep driver (#105): run a set of chains that differ only in knob values, recorded as
one ``sweep`` run under ``data/proc/sweep/`` (GLOSSARY.md "Sweep", "Point", "Axis").

Declare the sweep in the USER CONFIGURATION block — base chain, steps, axes, mode —
then look before you run::

    python scripts/sweep.py --dry-run              # the points, the plan, time and disk
    python scripts/sweep.py                        # run it (hours: in the background)
    python scripts/sweep.py --resume <sweep_id>    # continue an interrupted sweep

Run it from the main checkout's root, never beside another stage driver: each point's
steps take the stage lock (#101). The declaration is copied into the sweep's sidecar,
which, not this file's history, is the record. Score the points with the sweep diag
(#107).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.logging  import get_logger, add_run_handler
from pysuto.pipeline import STEP_GB, Axis, Chain, Pipeline, Sweep

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

# The chain every point starts from: its feed runs feed the imp step, and every point
# is compared with it. The reference chain of ADR 0039 am. (#33).
BASE_OPT_RUN_ID = "opt_20261020_112726_2f451f31"
STEPS           = ("imp", "const", "opt")
MODE            = "random"          # "random" (seeded Latin hypercube) | "grid"
N_POINTS        = 4                 # random mode only
SEED            = 20261021          # random mode only
LABEL           = "knob sweep"

# Overrides set on every point. A sweep through opt passes opt.FEASIBILITY_CHECK =
# "none" unless this or an axis sets it (#104).
FIXED: dict = {}

# Rough minutes per step run, from the 2026-10-21 logs (imp 2.7, const 1.6–4.4, opt
# ≈ 10 of which ≈ 7 the probe a sweep point skips, plus its diag); re-measure on the
# first sweep (its manifest timestamps).
STEP_MINUTES = {"imp": 3.0, "const": 3.0, "opt": 5.0}


# The varied knobs. A placeholder until the knob study (#108) settles them.
AXES = [
    Axis.interval(("imp.P_PROP", "const.P_PROP"), 1.1, 1.5),
]

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

_log = get_logger("sweep")


def build(base_opt_run_id: str) -> Sweep:
    base = Chain.from_opt_run(base_opt_run_id)
    pipeline = Pipeline(steps=STEPS, base=base, cwd=os.getcwd(),
                        scripts_dir=os.path.dirname(os.path.abspath(__file__)))
    kw = {"n_points": N_POINTS, "seed": SEED} if MODE == "random" else {}
    return Sweep(pipeline, AXES, mode=MODE, fixed=FIXED, **kw)


def dry_run(sweep: Sweep) -> None:
    keys = [k for ax in sweep.axes for k in ax.keys]
    _log.info("sweep: %s mode, %d point(s), steps %s, base %s", sweep.mode,
              len(sweep.points()), "→".join(sweep.pipeline.steps), sweep.pipeline.base.opt)
    for i, pt in enumerate(sweep.points()):
        _log.info("  p%03d  %s", i, "  ".join(f"{k}={pt[k]!r}" for k in keys))
    if sweep.fixed:
        _log.info("  every point: %s", sweep.fixed)
    plan = sweep.plan()
    minutes = sum(STEP_MINUTES[s] * n for s, n in plan.items())
    written = sum(STEP_GB[s] * n for s, n in plan.items())
    free = shutil.disk_usage(os.getcwd()).free / 1024 ** 3
    _log.info("plan: %s runs ≈ %.0f min, ≈ %.1f GB written (%.1f GB with the reserve); "
              "%.1f GB free", plan, minutes, written, sweep.disk_need_gb(), free)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a knob sweep of chains.")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the points, the plan, time and disk; run nothing.")
    parser.add_argument("--resume", metavar="SWEEP_ID", default=None,
                        help="Continue a sweep: rerun its points not done.")
    parser.add_argument("--base", metavar="OPT_RUN_ID", default=BASE_OPT_RUN_ID,
                        help=f"Base chain's opt run (default {BASE_OPT_RUN_ID}).")
    args = parser.parse_args()

    sweep = build(args.base)
    if args.dry_run:
        dry_run(sweep)
        return
    os.makedirs("logs", exist_ok=True)
    add_run_handler(f"logs/sweep_{datetime.now():%Y%m%d_%H%M%S}.log")
    dry_run(sweep)
    sid = sweep.run(label=LABEL, resume=args.resume)
    _log.info("sweep run %s under data/proc/sweep/", sid)


if __name__ == "__main__":
    main()
