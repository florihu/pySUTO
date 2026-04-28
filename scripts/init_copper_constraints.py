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

import os
import sys

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

ACC_ID     = "copper"
ACC_DIR    = os.path.join("data", "proc", "acc")
AGG_DIR    = os.path.join("data", "proc", "imp", "copper_agg_imp_v0")
PARAMS_DIR = os.path.join("data", "input", "params")
OUTPUT_DIR = os.path.join("data", "proc", "const")
AGG_ID     = "copper_agg_imp_v0"

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
from pysuto.agg import AggregationResult
from pysuto.constraints import (
    ConstraintEngine,
    MassBalanceBuilder,
    SumConstraintBuilder,
    RatioConstraintBuilder,
    RatioBoundBuilder,
    FlowBoundsBuilder,
)


def main() -> None:
    # ── A. Load inputs ────────────────────────────────────────────────────
    print(f"Loading AccountingStructure '{ACC_ID}' from {ACC_DIR} …")
    acc = AccountingStructure.load(ACC_ID, storage_dir=ACC_DIR)
    print(f"  nodes={acc.n_nodes}  edges={acc.n_edges}  time={acc.time_range[0]}–{acc.time_range[-1]}")

    print(f"Loading AggregationResult from {AGG_DIR} …")
    agg_result = AggregationResult.load(AGG_DIR)
    print(f"  f_agg shape={agg_result.f_agg.shape}  coverage={agg_result.coverage.sum()} edges")

    # ── B. Assemble constraint engine ─────────────────────────────────────
    engine = ConstraintEngine(acc_struct=acc, storage_dir=OUTPUT_DIR)

    # 1. Hard mass balance
    engine.register(MassBalanceBuilder())

    # 2. Soft sum — Process nodes
    #    Inputs on LHS (+1), SEM/Nature waste outputs on LHS (−1),
    #    useful supply (Flow + FD) is the observed RHS c_k
    engine.register(SumConstraintBuilder.from_acc_struct_subentity(
        acc,
        entity_filter={"Process"},
        rhs_out_entity={"Flow"},
        rhs_out_entity_sub=FD_SUBS,
        lhs_neg_out_entity_sub=SEM_NATURE_SUBS,
    ))

    # 3. Soft sum — Flow nodes
    #    Process→Flow supply is the observed RHS c_k;
    #    all demand (Flow→Process use + Flow→FD SB) on LHS (+1)
    engine.register(SumConstraintBuilder.from_acc_struct_subentity(
        acc,
        entity_filter={"Flow"},
        rhs_in_entity={"Process"},
    ))

    # 4–5. Soft ratio constraints
    engine.register(RatioConstraintBuilder.from_csv(
        os.path.join(PARAMS_DIR, "copper_conc_params.csv"), acc_struct=acc))
    engine.register(RatioConstraintBuilder.from_csv(
        os.path.join(PARAMS_DIR, "copper_efficiency_params.csv"), acc_struct=acc))

    # 6–7. Hard ratio bounds
    engine.register(RatioBoundBuilder.from_csv(
        os.path.join(PARAMS_DIR, "copper_conc_params.csv"), acc_struct=acc))
    engine.register(RatioBoundBuilder.from_csv(
        os.path.join(PARAMS_DIR, "copper_efficiency_params.csv"), acc_struct=acc))

    # 8. Hard flow bounds from acc_struct
    engine.register(FlowBoundsBuilder())

    # ── C. Run and persist ────────────────────────────────────────────────
    print(f"Running ConstraintEngine over {len(acc.time_range)} timesteps …")
    constraint_run = engine.run(
        agg_result=agg_result,
        timesteps=[str(t) for t in acc.time_range],
        agg_id=AGG_ID,
        persist=True,
    )

    # ── D. Summary ────────────────────────────────────────────────────────
    print(f"\nConstraintRun saved: {constraint_run.run_id}")
    print(f"G shape: {constraint_run.G.shape}  (rows × edges)")
    print("\nConstraint type breakdown:")
    print(constraint_run.row_meta["type"].value_counts().to_string())
    print("\nCone type breakdown:")
    print(constraint_run.row_meta["cone_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
