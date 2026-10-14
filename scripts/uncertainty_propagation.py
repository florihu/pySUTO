"""
init_uncertainty_propagation.py
===============================
Stage 5 driver: compute posterior σ_a*(t) and (1−α) confidence bounds for
every edge × timestep of a persisted OptimRun, using the closed-form GLS
expression in :class:`pysuto.uncprop.KalmanUncProp`.

Pipeline
--------
1  Instantiate KalmanUncProp (defaults match ClarabelSolver's σ conditioning)
2  compute(opt_run_id, alpha=0.05) — cross-discovers const_run_id via the
   opt sidecar, builds S(t) = G·diag(σ²_a)·Gᵀ + diag(σ²_c(t)) per t, and
   solves for the marginal diagonal of the posterior covariance
3  persist(result, opt_run_id, label) — writes per-t .npz arrays and a
   harmonized sidecar under data/proc/uncprop/<uncprop_run_id>/

Usage::

    python scripts/uncertainty_propagation.py <opt_run_id> [--label LABEL]
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
OPT_DIR     = os.path.join("data", "proc", "opt")
CONST_DIR   = os.path.join("data", "proc", "const")
UNCPROP_DIR = os.path.join("data", "proc", "uncprop")
ALPHA       = 0.05
# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.uncprop import KalmanUncProp
from pysuto.logging import get_logger, add_run_handler

_log = get_logger("init_uncertainty_propagation")


def main(opt_run_id: str, label: str = "") -> None:
    os.makedirs("logs", exist_ok=True)
    add_run_handler(
        f"logs/init_uncertainty_propagation_{datetime.now():%Y%m%d_%H%M%S}.log"
    )

    _log.info("=== init_uncertainty_propagation START ===")
    _log.info("opt_run_id=%s  alpha=%.3f", opt_run_id, ALPHA)

    prop = KalmanUncProp(
        opt_dir     = OPT_DIR,
        const_dir   = CONST_DIR,
        uncprop_dir = UNCPROP_DIR,
    )

    result = prop.compute(opt_run_id, alpha=ALPHA)
    uncprop_run_id = prop.persist(result, opt_run_id, label=label)

    contractions = [
        1.0 - ts.diagnostics["trace_post"] / ts.diagnostics["trace_prior"]
        for ts in result.per_timestep.values()
        if ts.diagnostics["trace_prior"] > 0.0
    ]
    mean_contraction = sum(contractions) / len(contractions) if contractions else 0.0
    max_resid_hard = max(
        ts.diagnostics["max_resid_hard"] for ts in result.per_timestep.values()
    )

    _log.info(
        "UncPropRun '%s' persisted under %s  (n_t=%d, mean_trace_contraction=%.3f, max_resid_hard=%.3e)",
        uncprop_run_id, UNCPROP_DIR, len(result.per_timestep),
        mean_contraction, max_resid_hard,
    )
    _log.info("=== init_uncertainty_propagation DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute posterior σ_a*(t) and CIs for a persisted OptimRun."
    )
    parser.add_argument("opt_run_id", help="OptimRun ID from data/proc/opt/.")
    parser.add_argument(
        "--label", default="",
        help="Optional label for the uncprop run sidecar.",
    )
    args = parser.parse_args()
    main(args.opt_run_id, args.label)
