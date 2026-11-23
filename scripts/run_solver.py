"""
run_solver.py
=============
Stage 4 driver: solve the σ-weighted QP using Clarabel, starting from a
persisted ConstraintRun.

Pipeline
--------
1  Load Solver (ClarabelBackend) + ConstraintRun
2  Feasibility pre-check of every timestep (FEASIBILITY_CHECK = "all"; "none" skips it)
3  Solve every timestep from the cold prior  a_prior(t) = conditioned f_agg(t)
   (--prior-mode cold, this driver's default; 'warm' carries a_post(t-1)),
   N_WORKERS timesteps at a time (parallel solve; warm needs N_WORKERS = 1)
4  Persist OptimResults to <data-dir>/proc/opt/<opt_run_id>/
5  Render the SolverDiag portfolio (12 figures + ranking CSVs + provenance)
   under <data-dir>/proc/opt/<opt_run_id>/diag/ (per-year edge/constraint CSVs on
   demand: scripts/solver_stats.py)

Usage::

    python scripts/run_solver.py --const-run-id CONST_RUN_ID [--label LABEL]
                                 [--prior-mode cold|warm] [--case CASE]
                                 [--data-dir data] [--set KNOB=VALUE ...]
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# The body is pysuto.stages.run_opt (#186) and so are the tuning constants (TIMESTEPS,
# FEASIBILITY_CHECK, the Clarabel tolerances, N_WORKERS, QP_VARIABLES, with their
# measurement notes): this driver keeps the argument parsing, the stage lock, the log
# handler and the --set knob protocol over pysuto.stages.OPT_KNOBS.
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pathlib import Path
from pysuto.case    import CaseSpec, resolve_case
from pysuto.case   import add_case_arg, case_from_args
from pysuto.logging import get_logger, start_run_log
from pysuto.knobs   import add_knob_args, overrides_from_cli, write_run_id
from pysuto.registry import stage_lock
from pysuto.stages  import OPT_KNOBS as KNOBS, opt_defaults, run_opt

# The case the driver runs without --case: the repository's copper case, next to it.
DEFAULT_CASE = str(Path(__file__).resolve().with_name("copper.py"))
CASE = resolve_case(DEFAULT_CASE)

_log = get_logger("run_solver")


def main(const_run_id: str, label: str = "", prior_mode: str = "cold",
         overrides: dict | None = None, *, data_dir: str = "data",
         const_dir: str | None = None, case: CaseSpec = CASE) -> str:
    """Solve constraint run ``const_run_id``; return the opt run id.

    ``case`` is recorded in the sidecar and must be the case that made the constraint
    run. The knobs are ``pysuto.stages``' constants, except those the case sets in
    ``driver_knobs["opt"]`` (a ``--set`` of the same knob wins); see
    :func:`pysuto.stages.run_opt`, which this calls.
    """
    start_run_log(data_dir, "run_solver")
    return run_opt(case, const_run_id, label, overrides, prior_mode=prior_mode,
                   data_dir=data_dir, const_dir=const_dir, logger=_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve persisted ConstraintRun.")
    parser.add_argument(
        "--const-run-id", required=True,
        help="ConstraintRun ID to load from --const-dir.",
    )
    parser.add_argument(
        "--label", required=False, default="",
        help="Optional label for the optim run sidecar.",
    )

    parser.add_argument(
        "--prior-mode", required=False, default='cold',
        choices=["warm", "cold"],
        help="Solver prior mode: 'cold' (default) or 'warm'."
    )
    add_case_arg(parser, default=DEFAULT_CASE)
    parser.add_argument("--data-dir", default="data",
                        help="Data root holding proc/ (default 'data').")
    parser.add_argument("--const-dir", default=None,
                        help="Where the constraint run is read from "
                             "(default <data-dir>/proc/const).")

    add_knob_args(parser)
    args = parser.parse_args()
    case = case_from_args(parser, args)
    try:
        # a bad case value is a usage error
        case.driver_knobs_for("opt", KNOBS, opt_defaults())
    except (KeyError, TypeError) as exc:
        parser.error(exc.args[0] if exc.args else str(exc))
    overrides = overrides_from_cli(parser, args, KNOBS, opt_defaults())
    # registries have no locking (#101)
    with stage_lock("run_solver", root=os.path.join(args.data_dir, "proc")):
        write_run_id(args.run_id_file,
                     main(args.const_run_id, args.label, args.prior_mode, overrides,
                          data_dir=args.data_dir, const_dir=args.const_dir, case=case))
