"""
init_copper_constraints.py
==========================
Stage 3a driver: build and persist the constraint matrix for the copper model.

Constraint types (registered in order)
---------------------------------------
1  Hard mass balance        — MassBalanceBuilder
                              Σ incoming − Σ outgoing = 0 per (Flow|Process) node × time
2  Soft sum — Process nodes — SumConstraintBuilder.from_acc_struct_subentity
                              Inputs on LHS (+1), SEM/Nature waste outputs on LHS (−1),
                              useful supply (Flow + FD) as observed RHS c_k
                              σ_c = Gaussian propagation of supply-edge uncertainties
3  Soft sum — Flow nodes    — SumConstraintBuilder.from_acc_struct_subentity
                              All demand (use + FD) on LHS (+1),
                              Process→Flow supply as observed RHS c_k
                              σ_c = Gaussian propagation of supply-edge uncertainties
4  Soft ratio — concentration  — RatioConstraintBuilder (copper_conc_params.csv)
5  Soft ratio — efficiency     — RatioConstraintBuilder (copper_efficiency_params.csv)
6  Hard ratio bounds — conc    — RatioBoundBuilder (copper_conc_params.csv)
7  Hard ratio bounds — eff     — RatioBoundBuilder (copper_efficiency_params.csv)
8  Hard flow bounds           — FlowBoundsBuilder (from acc_struct.bounds)

═══════════════════════════════════════════════════════════
USER CONFIGURATION  ←  only section you ever edit
═══════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

ACC_ID     = "copper"
ACC_DIR    = os.path.join("data", "proc", "acc")
IMP_DIR    = os.path.join("data", "proc", "imp")
PARAMS_DIR = os.path.join("data", "input", "params")
OUTPUT_DIR = os.path.join("data", "proc", "const")

# Power-mean exponent — must match the value used in the imputation pipeline
P_PROP = 1.3

# SB entity_sub values whose outgoing edges are subtracted from the Process
# node LHS (waste to nature / socio-economic system boundary)
SEM_NATURE_SUBS = {"Nature", "SEM"}

# SB entity_sub values treated as final demand (outgoing to these go into c_k
# for Process nodes; default for Flow nodes)
FD_SUBS = {"FD"}

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

# make sure the package is importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.acc import AccountingStructure
from pysuto.registry import ImputationRunRegistry
from pysuto.constraints import (
    ConstraintEngine,
    MassBalanceBuilder,
    SumConstraintBuilder,
    RatioConstraintBuilder,
    RatioBoundBuilder,
    FlowBoundsBuilder,
)
from pysuto.logging import get_logger, add_run_handler

_log = get_logger("init_copper_constraints")


def main(imp_run_id: str, label: str) -> None:
    from datetime import datetime
    add_run_handler(f"logs/init_copper_constraints_{datetime.now():%Y%m%d_%H%M%S}.log")

    # ── A. Load inputs ────────────────────────────────────────────────────
    _log.info("Loading AccountingStructure '%s' from %s", ACC_ID, ACC_DIR)
    acc = AccountingStructure.load(ACC_ID, storage_dir=ACC_DIR)
    _log.info("  nodes=%d  edges=%d  time=%s–%s",
              acc.n_nodes, acc.n_edges, acc.time_range[0], acc.time_range[-1])

    _log.info("Loading imputation run from %s", IMP_DIR)
    imp_registry = ImputationRunRegistry(storage_dir=IMP_DIR)
    agg_result = imp_registry.load(imp_run_id)
    _log.info("  f_agg shape=%s  coverage=%d edges  imp_run_id=%s",
              agg_result.f_agg.shape, agg_result.coverage.sum(), imp_run_id)

    # ── B. Assemble constraint engine ─────────────────────────────────────
    engine = ConstraintEngine(acc_struct=acc, storage_dir=OUTPUT_DIR)

    # 1. Hard mass balance
    _log.info("Registering MassBalanceBuilder")
    engine.register(MassBalanceBuilder())

    # 2. Soft sum — Process nodes
    _log.info("Registering SumConstraintBuilder — Process nodes")
    engine.register(SumConstraintBuilder.from_acc_struct_subentity(
        acc,
        entity_filter={"Process"},
        rhs_out_entity={"Flow"},
        rhs_out_entity_sub=FD_SUBS,
        lhs_neg_out_entity_sub=SEM_NATURE_SUBS,
        p=P_PROP,
    ))

    # 3. Soft sum — Flow nodes
    _log.info("Registering SumConstraintBuilder — Flow nodes")
    engine.register(SumConstraintBuilder.from_acc_struct_subentity(
        acc,
        entity_filter={"Flow"},
        rhs_in_entity={"Process"},
        p=P_PROP,
    ))

    # 4–5. Soft ratio constraints
    _log.info("Registering RatioConstraintBuilder — concentration")
    engine.register(RatioConstraintBuilder.from_csv(
        os.path.join(PARAMS_DIR, "copper_conc_params.csv"), acc_struct=acc))
    _log.info("Registering RatioConstraintBuilder — efficiency")
    engine.register(RatioConstraintBuilder.from_csv(
        os.path.join(PARAMS_DIR, "copper_efficiency_params.csv"), acc_struct=acc))

    # 6–7. Hard ratio bounds
    _log.info("Registering RatioBoundBuilder — concentration")
    engine.register(RatioBoundBuilder.from_csv(
        os.path.join(PARAMS_DIR, "copper_conc_params.csv"), acc_struct=acc))
    _log.info("Registering RatioBoundBuilder — efficiency")
    engine.register(RatioBoundBuilder.from_csv(
        os.path.join(PARAMS_DIR, "copper_efficiency_params.csv"), acc_struct=acc))

    # 8. Hard flow bounds from acc_struct
    _log.info("Registering FlowBoundsBuilder")
    engine.register(FlowBoundsBuilder())

    # ── C. Run and persist ────────────────────────────────────────────────
    _log.info("Running ConstraintEngine over %d timesteps", len(acc.time_range))
    # agg_id auto-derived from agg_result._imp_run_id (set by ImputationRunRegistry).
    constraint_run = engine.run(
        agg_result=agg_result,
        timesteps=[str(t) for t in acc.time_range],
        persist=True,
        diag_plot=True,
        label=label,
    )

    # ── D. Summary ────────────────────────────────────────────────────────
    _log.info("ConstraintRun saved: %s  label=%s", constraint_run.run_id, label)
    _log.info("G shape: %s  (rows × edges)", constraint_run.G.shape)
    _log.info("Constraint type breakdown:\n%s",
              constraint_run.row_meta["type"].value_counts().to_string())
    _log.info("Cone type breakdown:\n%s",
              constraint_run.row_meta["cone_type"].value_counts().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and persist constraint matrix.")
    parser.add_argument("imp_run_id", help="Imputation run ID from init_collect_aggregate_impute.py")
    parser.add_argument("label", help="Label for this constraint run")
    args = parser.parse_args()
    main(args.imp_run_id, args.label)
