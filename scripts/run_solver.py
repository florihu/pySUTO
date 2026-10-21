"""
init_run_solver.py
==================
Stage 4 driver: solve the σ-weighted QP using Clarabel, starting from a
persisted ConstraintRun.

Pipeline
--------
1  Load ClarabelSolver + ConstraintRun
2  Feasibility pre-check (defaults to first timestep only)
3  Solve every timestep from the cold prior  a_prior(t) = conditioned f_agg(t)
   (--prior-mode cold, this driver's default; 'warm' carries a_post(t-1)),
   N_WORKERS timesteps at a time (parallel solve; warm needs N_WORKERS = 1)
4  Persist OptimResults to data/proc/opt/<opt_run_id>/
5  Render the SolverDiag portfolio (12 figures + ranking CSVs + provenance)
   under data/proc/opt/<opt_run_id>/diag/ (per-year edge/constraint CSVs on
   demand: scripts/solver_stats.py)

Usage::

    python scripts/run_solver.py --const-run-id CONST_RUN_ID [--label LABEL]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
# Pipeline wiring (`const_run_id`) is passed via the `--const-run-id` CLI flag.

CONST_DIR  = os.path.join("data", "proc", "const")
OUTPUT_DIR = os.path.join("data", "proc", "opt")
TIMESTEPS  = None #[1995]           # None → all timesteps in the run
# Sensitivity cold run: scan every year up front so an infeasibility introduced
# by the flat-prior-zero pin halts at — and localises — the first breaking year,
# rather than only checking year 0. See branch sensitivity/no-flat-prior-zeros.
# Back to the first-only default since Round C: the 30-probe scan costs ~350 s
# per probe (~3 h) on the full hard system, and each year's QP reports its own
# status anyway — flip to False only to localise a mid-series infeasibility.
FEASIBILITY_CHECK_FIRST_ONLY = True
# Clarabel iteration cap. The default (200) is too low for the pinned, soft-stock
# 1.37M-row phase-1 feasibility QP (converges but not within 200 iters → spurious
# MaxIterations reported as infeasible). max_iter is a CAP, so easy runs still stop
# early; raising it only helps the hard pinned scan. 2000 solves 1995 feasibility.
MAX_ITER = 2000
# Clarabel termination tolerances (measured in Round W.1, ADR 0039). The former
# production setting (gap_rel 1e-3 / feas 1e-7, reduced 1e-2 / 1e-5) stopped every
# run while the cost was still falling by thousands per iteration: re-solving 2015
# of const_20260929_120611_6ef43f64 at gap_rel 1e-8 lowered the objective by 9,052
# (8 % of that year's misfit) and lifted Chile's concentrate-export retention from
# 0.66 to 0.97. The reported gap (~800) bounds nothing: |objective| is dominated by
# the constant Σ(a0/σ)² (1.17e6 vs a misfit of 1.13e5 in 2015), and the dual cost
# at the stop sits above the true optimum. With the lower-bound rows the same stop
# left W.1 48,800 short of its optimum. These tolerances run each year to its
# numerical floor (InsufficientProgress after ~150–250 iterations, ≈ 30–60 s/year)
# and accept that floor as AlmostSolved through the reduced tolerances (gap_rel
# 1e-5 is not reached there, 1e-4 is).
# #33 (ADR 0039 am. 2026-10-21): solved for the defects, every year of the final chain
# reaches Solved at gap 1e-8 in 64–81 iterations within 0.2 of a certified lower bound on J
# (opt_20261020_112726_2f451f31 vs opt_20261019_171746_3272682b: Σ_t J −66,702).
TOL_FEAS            = 1e-8
TOL_GAP_REL         = 1e-8
TOL_GAP_ABS         = 1e-8
REDUCED_TOL_FEAS    = 1e-6
REDUCED_TOL_GAP_REL = 1e-4
REDUCED_TOL_GAP_ABS = 10.0
# Parallel solve (#70): cold periods are independent QPs, so they run on
# N_WORKERS threads, each with its own Clarabel instance; 1 = the serial loop,
# the only setting --prior-mode warm accepts (a warm run is a chain). Each
# period's linear solver is pinned to one thread: multi-threaded faer was slower
# than one thread (1995 of const_20261012_113828_12c562c7: 48.0 s vs 34.9 s) and
# gave the objective of opt_20261012_122124_40b1b370 bit for bit (#70).
N_WORKERS           = os.cpu_count() or 1
# Clarabel's linear solver for the period QPs. "faer" is what "auto" picked in
# opt_20261012_122124_40b1b370 (sidecar linsolver_name). "qdldl" was rejected on
# measurement (#72): on const_20261019_170151_13460e94 its period solves took 52 s vs
# 234 s, but its posteriors were indistinguishable from faer's on the same QPs (Σ_t J
# +1,337.5, +0.043 %, lower in 13/30 years; opt_20261020_090246_99da4a39 vs
# opt_20261019_171746_3272682b), and Stage 4 fell only 745 → 563 s because the
# feasibility probe dominates (ADR 0039 am.).
DIRECT_SOLVE_METHOD = "faer"
# The period QP's decision variables (#33): "flows" solves for a, so Clarabel's cost
# is J − Σ(a_prior/σ_a)², a constant ≈ 1e6 against J ≈ 1e5; "defects" solves the
# same QP for d = a − a_prior (q = 0, cost = J). Probed on
# const_20261019_170151_13460e94, years 1995/2008/2009/2015/2016/2022
# (scripts/explo/issue33_conditioning_probe.py): at feas/gap 1e-8 "defects" reached
# Solved in every year in 71–81 iterations (20–22 s vs 52–68 s), its J within 0.2 of
# a certified dual lower bound, while "flows" stopped 516–4,349 above that bound
# (opt_20261019_171746_3272682b 945–4,363 above it), hard-row residuals ≤ 9e-7 kt vs
# ≤ 1e-7 kt. The sidecar's objective stays J − Σ(a_prior/σ_a)² under both; gap_rel is
# relative to Clarabel's own cost, so TOL_GAP_REL is ~10× tighter in absolute terms
# under "defects". Measured on the full chain (#33, ADR 0039 am. 2026-10-21):
# opt_20261020_112726_2f451f31 lowers Σ_t J by 66,702 (−2.13 % vs
# opt_20261019_171746_3272682b) in 30/30 years; the feasibility probe has no σ-weighted cost
# and is unaffected.
QP_VARIABLES        = "defects"

# The knobs a pipeline step may override with --set (#100, pysuto.knobs). Paths are
# not knobs.
KNOBS = (
    "TIMESTEPS", "FEASIBILITY_CHECK_FIRST_ONLY", "MAX_ITER",
    "TOL_FEAS", "TOL_GAP_REL", "TOL_GAP_ABS",
    "REDUCED_TOL_FEAS", "REDUCED_TOL_GAP_REL", "REDUCED_TOL_GAP_ABS",
    "N_WORKERS", "DIRECT_SOLVE_METHOD", "QP_VARIABLES",
)


# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.acc     import AccountingStructure
from pysuto.solver  import ClarabelSolver
from pysuto.diag    import SolverDiag
from pysuto.logging import get_logger, add_run_handler
from pysuto.knobs   import add_knob_args, overrides_from_cli, write_run_id
from pysuto.registry import stage_lock

_log = get_logger("init_run_solver")


def main(const_run_id: str, label: str = "", prior_mode: str = "warm",
         overrides: dict | None = None) -> str:
    os.makedirs("logs", exist_ok=True)
    add_run_handler(
        f"logs/init_run_solver_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    _log.info("=== init_run_solver START ===")
    _log.info("Loading ClarabelSolver (const_dir=%s)", CONST_DIR)
    # Termination tolerances: see the USER CONFIGURATION block (Round W.1).
    solver = ClarabelSolver(const_dir=CONST_DIR, tol_feas=TOL_FEAS,
                            tol_gap_rel=TOL_GAP_REL, tol_gap_abs=TOL_GAP_ABS,
                            reduced_tol_feas=REDUCED_TOL_FEAS,
                            reduced_tol_gap_rel=REDUCED_TOL_GAP_REL,
                            reduced_tol_gap_abs=REDUCED_TOL_GAP_ABS,
                            max_iter=MAX_ITER,
                            prior_mode=prior_mode,
                            n_workers=N_WORKERS,
                            direct_solve_method=DIRECT_SOLVE_METHOD,
                            qp_variables=QP_VARIABLES,)

    _log.info("Solving ConstraintRun '%s'", const_run_id)
    results = solver.solve_all(
        const_run_id,
        timesteps                    = TIMESTEPS,
        feasibility_check_first_only = FEASIBILITY_CHECK_FIRST_ONLY,
    )

    optim_run_id = solver.persist(
        run_id       = const_run_id,
        results      = results,
        output_dir   = OUTPUT_DIR,
        const_run_id = const_run_id,
        label        = label,
        stage_config = {
            "prior_mode":                   solver.prior_mode,
            "max_iter":                     MAX_ITER,
            "tol_feas":                     TOL_FEAS,
            "tol_gap_rel":                  TOL_GAP_REL,
            "tol_gap_abs":                  TOL_GAP_ABS,
            "reduced_tol_feas":             REDUCED_TOL_FEAS,
            "reduced_tol_gap_rel":          REDUCED_TOL_GAP_REL,
            "reduced_tol_gap_abs":          REDUCED_TOL_GAP_ABS,
            "feasibility_check_first_only": FEASIBILITY_CHECK_FIRST_ONLY,
            "overrides":                    dict(overrides or {}),
        },
    )

    total_iter   = sum(r.iterations   for r in results.values())
    total_time   = sum(r.solve_time_s for r in results.values())
    max_residual = max(
        float(abs(r.residuals).max()) if r.residuals.size else 0.0
        for r in results.values()
    )
    _log.info(
        "Solved %d timesteps  total_iter=%d  total_time=%.2fs  max|residual|=%.3e",
        len(results), total_iter, total_time, max_residual,
    )
    _log.info("OptimRun saved as '%s' under %s", optim_run_id, OUTPUT_DIR)

    const_run = solver.load_run(const_run_id)
    diag_dir  = os.path.join(OUTPUT_DIR, optim_run_id, "diag")

    # One AccountingStructure for figures and ranking CSVs (loading it is the
    # expensive step); the diag's stats object owns the all-years frames both read from.
    acc  = AccountingStructure.load(const_run.acc_id)
    diag = SolverDiag(results, const_run, label=label or optim_run_id, acc=acc)
    diag.summary()
    diag.plot(odir=diag_dir)

    _log.info("=== init_run_solver DONE ===")
    return optim_run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve persisted ConstraintRun.")
    parser.add_argument(
        "--const-run-id", required=True,
        help="ConstraintRun ID to load from data/proc/const/.",
    )
    parser.add_argument(
        "--label", required=False, default="",
        help="Optional label for the optim run sidecar.",
    )

    parser.add_argument(
        "--prior-mode", required=False, default='cold',
        choices=["warm", "cold"],
        help="ClarabelSolver prior mode: 'cold' (default) or 'warm'."
    )

    add_knob_args(parser)
    args = parser.parse_args()
    overrides = overrides_from_cli(parser, args, KNOBS, globals())
    with stage_lock("run_solver"):   # registries have no locking (#101)
        write_run_id(args.run_id_file,
                     main(args.const_run_id, args.label, args.prior_mode, overrides))
