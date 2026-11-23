"""
uncertainty_propagation.py
==========================
Stage 5 driver: compute posterior σ_a*(t) and (1−α) confidence bounds for
every edge × timestep of a persisted OptimRun, using the closed-form GLS
expression in :class:`pysuto.uncprop.KalmanUncProp` (ADR 0044).

Pipeline
--------
1  Instantiate KalmanUncProp (σ conditioning floors come from the opt sidecar)
2  compute(opt_run_id, alpha=0.05) — cross-discovers const_run_id via the
   opt sidecar, reduces each period at the QP's active set at a* (#27), and
   reads the posterior contraction κ off a QR of the whitened system
   M = [G·diag(σ_a), diag(σ_c)] (S = M·Mᵀ is never formed)
3  persist(result, opt_run_id, label) — writes per-t .npz arrays and a
   harmonized sidecar under <data-dir>/proc/uncprop/<uncprop_run_id>/
4  Render the UncPropDiag portfolio (UD-R-01/02/03 + contraction_summary.csv,
   calibration.csv, birge.csv) from the persisted run into
   <data-dir>/proc/uncprop/<uncprop_run_id>/diag/ (re-render later with
   scripts/uncprop_diag.py)

Usage::

    python scripts/uncertainty_propagation.py <opt_run_id> [--label LABEL] [--case CASE]
                                              [--data-dir data] [--run-id-file PATH]
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# The body is pysuto.stages.run_uncprop (#186), with its constants (ALPHA, BIND_TOL,
# RANK_TOL, EXCLUDE_SCALE_TOTALS); main() derives every path from `--data-dir`
# (default data): proc/opt, proc/const, proc/imp, proc/acc, proc/uncprop.
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
from pysuto.case    import CaseSpec, resolve_case
from pysuto.case   import add_case_arg, case_from_args
from pysuto.knobs   import write_run_id
from pysuto.logging import get_logger, start_run_log
from pysuto.registry import stage_lock
from pysuto.stages  import run_uncprop

# The case the driver runs without --case: the repository's copper case, next to it.
DEFAULT_CASE = str(Path(__file__).resolve().with_name("copper.py"))
CASE = resolve_case(DEFAULT_CASE)

_log = get_logger("uncertainty_propagation")


def main(opt_run_id: str, label: str = "", *, data_dir: str = "data",
         case: CaseSpec = CASE) -> str:
    """Propagate opt run ``opt_run_id`` of ``<data_dir>/proc/opt``; the run is
    written to ``<data_dir>/proc/uncprop``. ``case`` is recorded in the sidecar and
    must be the case that made the opt run."""
    start_run_log(data_dir, "uncertainty_propagation")
    return run_uncprop(case, opt_run_id, label, data_dir=data_dir, logger=_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute posterior σ_a*(t) and CIs for a persisted OptimRun."
    )
    parser.add_argument("opt_run_id", help="OptimRun ID from <data-dir>/proc/opt/.")
    parser.add_argument(
        "--label", default="",
        help="Optional label for the uncprop run sidecar.",
    )
    add_case_arg(parser, default=DEFAULT_CASE)
    parser.add_argument("--data-dir", default="data",
                        help="Data root holding proc/ (default 'data').")
    parser.add_argument("--run-id-file", default=None, metavar="PATH",
                        help="Write the persisted run id to PATH (pysuto.pipeline).")
    args = parser.parse_args()
    case = case_from_args(parser, args)
    # registries have no locking (#101)
    with stage_lock("uncertainty_propagation", root=os.path.join(args.data_dir, "proc")):
        write_run_id(args.run_id_file,
                     main(args.opt_run_id, args.label, data_dir=args.data_dir, case=case))
