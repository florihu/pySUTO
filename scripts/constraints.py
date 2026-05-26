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
8  Net stock bounds           — RatioBoundBuilder (stock_bounds/params.csv);
                                 each row carries `agg=true` so the builder
                                 sums every matched a-side edge into one
                                 G row and every matched b-side edge into
                                 another, per (region, time, valuation).
9  Hard flow bounds           — FlowBoundsBuilder (from acc_struct.bounds)

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
# Pipeline wiring (acc_id, constraint bindings) lives in config/config.yaml
# under `pipeline:`.  Power-mean exponent and σ-regularisation floors are also
# loaded from config — see `cfg = load_config()` below.

ACC_DIR     = os.path.join("data", "proc", "acc")
IMP_DIR     = os.path.join("data", "proc", "imp")
_INPUT_ROOT = os.path.join("data", "input")
OUTPUT_DIR  = os.path.join("data", "proc", "const")

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
from pysuto.config import load_config
from pysuto.registry import ImputationRunRegistry
from pysuto.input_registry import InputRegistries
from pysuto.constraints import (
    ConstraintEngine,
    MassBalanceBuilder,
    SumConstraintBuilder,
    RatioConstraintBuilder,
    RatioBoundBuilder,
    FlowBoundsBuilder,
)
from pysuto.logging import get_logger, add_run_handler

cfg        = load_config()
ACC_ID     = cfg.pipeline.acc_id
P_PROP     = cfg.propagation.p
IMP_RUN_ID = cfg.pipeline.constraints.imp_run_id

CONST_BUILDER_BY_NAME = {
    "RatioConstraintBuilder": RatioConstraintBuilder,
    "RatioBoundBuilder":      RatioBoundBuilder,
}

_log = get_logger("init_copper_constraints")


def main(label: str) -> None:
    from datetime import datetime
    add_run_handler(f"logs/init_copper_constraints_{datetime.now():%Y%m%d_%H%M%S}.log")

    # ── 0  Initialise input registries (scan-and-bump) ────────────────────
    inputs    = InputRegistries.initialize(root=_INPUT_ROOT)
    acc_entry = inputs.acc.entry(ACC_ID)
    _log.info("0  Input acc: %s v%d (code=%s)",
              acc_entry["id"], acc_entry["version"], acc_entry["code"])

    # ── A. Load inputs ────────────────────────────────────────────────────
    _log.info("Loading AccountingStructure '%s' from %s", ACC_ID, ACC_DIR)
    acc = AccountingStructure.load(ACC_ID, storage_dir=ACC_DIR)
    _log.info("  nodes=%d  edges=%d  time=%s–%s",
              acc.n_nodes, acc.n_edges, acc.time_range[0], acc.time_range[-1])

    _log.info("Loading imputation run from %s", IMP_DIR)
    imp_registry = ImputationRunRegistry(storage_dir=IMP_DIR)
    agg_result = imp_registry.load(IMP_RUN_ID)
    _log.info("  f_agg shape=%s  coverage=%d edges  imp_run_id=%s",
              agg_result.f_agg.shape, agg_result.coverage.sum(), IMP_RUN_ID)

    # ── A2. Cross-stage consistency: imp must reference same acc.id+version
    imp_sidecar = imp_registry.load_sidecar(IMP_RUN_ID)
    imp_acc_ref = (imp_sidecar.get("parents", {})
                              .get("inputs", {})
                              .get("acc"))
    if imp_acc_ref and (imp_acc_ref.get("id"), imp_acc_ref.get("version")) \
                       != (acc_entry["id"], acc_entry["version"]):
        raise RuntimeError(
            f"acc mismatch: imp run was built on "
            f"acc {imp_acc_ref!r}, but this const run is using "
            f"{acc_entry['id']} v{acc_entry['version']}."
        )

    # ── B. Assemble constraint engine ─────────────────────────────────────
    engine = ConstraintEngine(
        acc_struct        = acc,
        storage_dir       = OUTPUT_DIR,
        p                 = P_PROP,
        sigma_a_cv_min    = cfg.constraints.sigma_a_cv_min,
        sigma_a_abs_floor = cfg.constraints.sigma_a_abs_floor,
        sigma_c_cv_min    = cfg.constraints.sigma_c_cv_min,
        sigma_c_abs_floor = cfg.constraints.sigma_c_abs_floor,
    )

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
        sigma_mode="p_mean",
        p=P_PROP,
    ))

    # 4–8. CSV-fed soft ratio + hard ratio-bound builders.
    #      Bindings come from config/config.yaml → pipeline.constraints.bindings.
    #      Two builders may consume the same params_id (e.g. 'conc' → both
    #      RatioConstraintBuilder and RatioBoundBuilder).  stock_bounds joins
    #      the list: its CSV carries `agg=true` so the builder sums all
    #      matched a-side / b-side edges into one G row per
    #      (region, time, valuation) bucket.
    param_refs: list[dict] = []
    seen_param_ids: set[str] = set()
    for binding in cfg.pipeline.constraints.bindings:
        Builder  = CONST_BUILDER_BY_NAME[binding.builder]
        csv_path = inputs.params.resolve(binding.params_id) / "params.csv"
        _log.info("Registering %s ← params/%s v%d",
                  Builder.__name__, binding.params_id,
                  inputs.params.entry(binding.params_id)["version"])
        kwargs = {"kind": binding.params_id} if Builder is RatioBoundBuilder else {}
        engine.register(Builder.from_csv(str(csv_path), acc_struct=acc, **kwargs))
        if binding.params_id not in seen_param_ids:
            param_refs.append(inputs.params.entry(binding.params_id))
            seen_param_ids.add(binding.params_id)

    # 9. Hard flow bounds from acc_struct
    _log.info("Registering FlowBoundsBuilder")
    engine.register(FlowBoundsBuilder())

    # ── C. Run and persist ────────────────────────────────────────────────
    _log.info("Running ConstraintEngine over %d timesteps", len(acc.time_range))
    parents = {
        "inputs": {
            "acc":    acc_entry,
            "params": param_refs,
        },
        "upstream": {"imp_run_id": IMP_RUN_ID},
    }
    constraint_run = engine.run(
        agg_result   = agg_result,
        timesteps    = [str(t) for t in acc.time_range],
        persist      = True,
        diag_plot    = True,
        label        = label,
        parents      = parents,
        stage_config = {
            "sigma_a_cv_min":    cfg.constraints.sigma_a_cv_min,
            "sigma_a_abs_floor": cfg.constraints.sigma_a_abs_floor,
            "sigma_c_cv_min":    cfg.constraints.sigma_c_cv_min,
            "sigma_c_abs_floor": cfg.constraints.sigma_c_abs_floor,
        },
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
    #parser.add_argument("imp_run_id", help="Imputation run ID from init_collect_aggregate_impute.py")
    parser.add_argument("label", help="Label for this constraint run")
    args = parser.parse_args()
    main(args.label)
