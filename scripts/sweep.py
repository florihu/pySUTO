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
from pysuto.pipeline import RESERVE_GB, Axis, Chain, Pipeline, Sweep

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

# The knob study of #108; its decisions are recorded on the issue (2026-10-22).
# The chain every point starts from: its feed runs feed the imp step, and every point
# is compared with it. The final reference chain (#110, 2026-10-21).
BASE_OPT_RUN_ID = "opt_20261021_131744_58504ad3"
STEPS           = ("imp", "const", "opt")
MODE            = "random"          # "random" (seeded Latin hypercube) | "grid"
N_POINTS        = 64                # random mode only: ≈ 4× the axes
SEED            = 20261022          # random mode only
LABEL           = "knob study (#108)"

# Keep only the opt runs: each point's imp and const runs (≈ 0.93 GB) go to a RAM-backed
# scratch store and are deleted once no remaining point reuses them (#108).
KEEP_INTERMEDIATE = False
SCRATCH_DIR       = "/dev/shm/pysuto-sweep"

# Overrides set on every point. A sweep through opt passes opt.FEASIBILITY_CHECK =
# "none" unless this or an axis sets it (#104).
FIXED: dict = {}

# Rough minutes per step run, from the final reference chain's run times (imp ≈ 3,
# const ≈ 4.5; opt's period solves 93 s once the probe is skipped, plus loading, saving
# and its diag); re-measure on the first sweep (its manifest timestamps).
STEP_MINUTES = {"imp": 3.0, "const": 4.5, "opt": 3.0}

# collect_aggregate_impute.py's CASCADE_IDS; reversed, every pairwise precedence flips.
CASCADE_SHIPPED  = ("trade", "network", "conc", "ore_grade", "eff")
CASCADE_REVERSED = tuple(reversed(CASCADE_SHIPPED))

# The varied knobs (#108): σ-scale knobs log-uniform over a factor 2 either side of the
# shipped value, shape knobs uniform over a stated interval around it.
AXES = [
    Axis.interval(("imp.P_PROP", "const.P_PROP"), 1.1, 1.5),          # lp.py's range
    Axis.interval("imp.Q_REG", 1.5, 4.5),
    Axis.interval("imp.ALPHA_MIN_REG", 0.025, 0.1, log=True),
    Axis("imp.CASCADE_ORDER", [CASCADE_SHIPPED, CASCADE_REVERSED]),
    Axis("imp.TIME_WINDOW", [2, 3, 4]),
    Axis.interval("imp.TIME_GAP_INFLATION", 1.0, 1.1),
    Axis.interval("imp.STOCK_SEED_CV", 0.005, 0.02, log=True),
    Axis.interval("imp.OUTLIER_LEG_K", 2.0, 4.0),
    Axis.interval("imp.OUTLIER_LEG_CV", 2.5, 10.0, log=True),
    Axis.interval("imp.LEG_SIGMA_GAMMA", 0.3, 0.7),                   # ADR 0017: 0.5, 0.7
    Axis.interval("imp.LEG_SIGMA_CV_CAP", 0.5, 2.0, log=True),
    # Tied: below the engine's σ_c cv floor an anchor would be inflated back up to it.
    Axis.interval(("const.ANCHOR_CV", "const.FABRICATED_ANCHOR_CV",
                   "const.SIGMA_C_CV_MIN"), 0.005, 0.02, log=True),
    Axis.interval("const.NODATA_ANCHOR_CV", 0.025, 0.1, log=True),
    Axis.interval("const.RATIO_SIGMA_REL_FLOOR", 0.01, 0.04, log=True),
    Axis.interval("const.RATIO_SCALE_ABS_FLOOR", 50.0, 200.0, log=True),
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
    return Sweep(pipeline, AXES, mode=MODE, fixed=FIXED,
                 keep_intermediate=KEEP_INTERMEDIATE, scratch_dir=SCRATCH_DIR, **kw)


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
    free = shutil.disk_usage(os.getcwd()).free / 1024 ** 3
    _log.info("plan: %s runs ≈ %.0f min, ≈ %.1f GB written to data/proc (%.1f GB with the "
              "reserve); %.1f GB free", plan, minutes, sweep.disk_need_gb() - RESERVE_GB,
              sweep.disk_need_gb(), free)
    if not sweep.keep_intermediate:
        _log.info("  only %s runs are kept; the others pass through %s (≈ %.2f GB at a "
                  "time)", sweep.pipeline.steps[-1], sweep.scratch_dir,
                  sweep.scratch_need_gb())


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
