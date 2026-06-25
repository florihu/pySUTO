"""
init_copper_constraints.py
==========================
Stage 3a driver: build and persist the constraint matrix for the copper model.

Constraint types (registered in order)
---------------------------------------
1  Hard mass balance        — MassBalanceBuilder
                              Σ incoming − Σ outgoing = 0 per (Flow|Process) node × time
2  Soft sum — supply/use    — SumConstraintBuilder.from_mask_anchor, two families
   total-output SCALE TOTAL   pinned to the model's own prior f_agg
   (scale preservation)       (c = Σ f_agg over each group, σ_c = ANCHOR_CV·|c|,
                              |c|≈0 groups dropped). Both grouped by
                              (sector_from, region_from, time, valuation):
                                • supply total output (process→flow useful output)
                                  → one row per process;
                                • use total output (flow→process use + flow→SB
                                  final demand) → one row per flow, summed over
                                  all destinations.
                              A *scale total*, not an external *control total* —
                              the warm-start (a₀=f_agg at t₀, a_post(t-1) after)
                              solve never sees this fixed Stage-2 total. See
                              ADR 0006.
3  Soft ratio — concentration  — RatioConstraintBuilder (copper_conc_params.csv)
4  Soft ratio — efficiency     — RatioConstraintBuilder (copper_efficiency_params.csv)
5  Hard ratio bounds — conc    — RatioBoundBuilder (copper_conc_params.csv)
6  Hard ratio bounds — eff     — RatioBoundBuilder (copper_efficiency_params.csv)
7  Net stock bounds           — RatioBoundBuilder (stock_bounds/params.csv);
                                 each row carries `agg=true` so the builder
                                 sums every matched a-side edge into one
                                 G row and every matched b-side edge into
                                 another, per (region, time, valuation).
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
# Upstream wiring (`imp_run_id`) is passed via the `--imp-run-id` CLI flag.
# `acc_id` is read from config/config.yaml; every other knob (P_PROP, σ
# floors, builder registrations) lives inline below.

ACC_DIR     = os.path.join("data", "proc", "acc")
IMP_DIR     = os.path.join("data", "proc", "imp")
_INPUT_ROOT = os.path.join("data", "input")
OUTPUT_DIR  = os.path.join("data", "proc", "const")

# Supply/use total-output SCALE TOTALS pin every process's and every flow's
# total-output scale to the model's own Stage-2 prior (c = Σ f_agg), preventing
# the homogeneous scale-collapse. Built generically from the edge topology (no
# per-sector family list) — see section B below. ANCHOR_CV sets their tight σ_c.

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

# make sure the package is importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.acc import AccountingStructure
from pysuto.config import load_config
from pysuto.registry import ImputationRunRegistry
from pysuto.input_registry import InputRegistries
from pysuto.param_relation import expand_edge_index
from pysuto.constraints import (
    ConstraintEngine,
    MassBalanceBuilder,
    SumConstraintBuilder,
    RatioConstraintBuilder,
    RatioBoundBuilder,
    FlowBoundsBuilder,
)
from pysuto.logging import get_logger, add_run_handler

cfg    = load_config()
ACC_ID = cfg.pipeline.acc_id

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION — every knob that used to live in config/config.yaml.
# Note: P_PROP must match scripts/collect_aggregate_impute.py's P_PROP
# (cross-stage Lp exponent).
# ─────────────────────────────────────────────────────────────────────────────
P_PROP            = 1.3
SIGMA_A_CV_MIN    = 0.001
SIGMA_A_ABS_FLOOR = 1.0e-6
# Anchor CV (1%). SIGMA_C_CV_MIN is lowered to match so the engine's σ_c CV
# floor does not inflate the tight cathode-use anchor back up to 5%.
ANCHOR_CV         = 0.01
SIGMA_C_CV_MIN    = 0.01
SIGMA_C_ABS_FLOOR = 1.0e-3
# Prior box ("tight fitting"): box flow bounds to a₀ ± PRIOR_BOX_K·σ_a on edges
# with a real prior, grafting RAS's scale-floor onto the QP to stop the
# homogeneous collapse to l. None = off (legacy behaviour). See
# docs/rep/analysis_opt_20260602_133908_46bc514e_20260603_feed-scale-collapse.md §8.3.
PRIOR_BOX_K       = None
# No-data ("flat-prior") σ_a magnitude anchor: edges with a₀=0 whose σ fell back
# to the global-median (≈1e6) get σ = NODATA_ANCHOR_CV · median|a₀| of anchored
# siblings in the same (from_sector, valuation), so the zero prior pulls and they
# stop absorbing mass routed through data-free subgraphs (e.g. scrap on Taiwan/
# Panama CFR). Extraction front-end {CO,Mining,Ore,WR,Tailings} is excluded (real
# stripping-ratio magnitude prior). 0 disables. See ADR 0013.
NODATA_ANCHOR_CV  = 0.3
# Soft-ratio σ_c scaling (REP analysis_opt_20260611_075036 §8.2): the CSV ratio
# σ_r is rescaled to flow space per row, σ_c = max(σ_r, REL_FLOOR) ·
# max(Σ|a₀,num|, |r|·Σ|a₀,denom|, ABS_FLOOR). REL_FLOOR caps |z| at 1/floor on
# a 100 % ratio violation; ABS_FLOOR (tonnage) keeps both-zero-prior rows finite
# AND soft: ~77 % of soft ratio rows are zero-prior parametric ties, and at
# ABS_FLOOR=1 their σ_c=0.02 t made the Hessian so stiff Clarabel stalled
# (InsufficientProgress @2005). 100 t → σ_c=2 t on those rows (z=50 per 100 t).
RATIO_SIGMA_REL_FLOOR = 0.02
RATIO_SCALE_ABS_FLOOR = 100.0

_log = get_logger("init_copper_constraints")


def main(imp_run_id: str, label: str) -> None:
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
    agg_result = imp_registry.load(imp_run_id)
    _log.info("  f_agg shape=%s  coverage=%d edges  imp_run_id=%s",
              agg_result.f_agg.shape, agg_result.coverage.sum(), imp_run_id)

    # ── A2. Cross-stage consistency: imp must reference same acc.id+version
    imp_sidecar = imp_registry.load_sidecar(imp_run_id)
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
        sigma_a_cv_min    = SIGMA_A_CV_MIN,
        sigma_a_abs_floor = SIGMA_A_ABS_FLOOR,
        sigma_c_cv_min    = SIGMA_C_CV_MIN,
        sigma_c_abs_floor = SIGMA_C_ABS_FLOOR,
        prior_box_k       = PRIOR_BOX_K,
        nodata_anchor_cv  = NODATA_ANCHOR_CV,
    )

    # 1. Hard mass balance
    _log.info("Registering MassBalanceBuilder")
    engine.register(MassBalanceBuilder())

    # 2. Soft sum — supply/use total-output SCALE TOTALS (self-referential).
    #    Pin every process's and every flow's total-output scale to the model's
    #    own Stage-2 prior:  Σ (group edges) = Σ f_agg over the group, with tight
    #    σ_c = ANCHOR_CV·|c|; |c|≈0 groups are dropped by from_mask_anchor. Both
    #    families group by (sector_from, region_from, time, valuation) so each row
    #    is node-scoped, year-locked and unit-coherent.
    #      • supply total output: process→flow useful output → one row per process
    #        (a single process→flow edge is itself the total supply; no summing).
    #      • use total output: flow→process use + flow→SB final demand → one row
    #        per flow, summing over all sector_to destinations = the flow's total
    #        output of an industrial flow.
    #    A *scale total*, not an external *control total* — see ADR 0006.
    ei_exp      = expand_edge_index(acc.edge_index, acc.node_index)
    f_agg       = agg_result.f_agg
    supply_mask = (ei_exp["entity_from"].eq("Process")
                   & ei_exp["entity_to"].eq("Flow")).to_numpy()
    use_mask    = ei_exp["entity_from"].eq("Flow").to_numpy()
    GROUP_KEYS  = ("sector_from", "region_from", "time", "valuation")

    _log.info("Registering SumConstraintBuilder — supply/use total-output scale totals")
    engine.register(SumConstraintBuilder.from_mask_anchor(
        acc,
        mask          = supply_mask,
        anchor_values = f_agg,
        group_keys    = GROUP_KEYS,
        sigma_cv      = ANCHOR_CV,
        id_prefix     = "supplytotout",
    ))
    engine.register(SumConstraintBuilder.from_mask_anchor(
        acc,
        mask          = use_mask,
        anchor_values = f_agg,
        group_keys    = GROUP_KEYS,
        sigma_cv      = ANCHOR_CV,
        id_prefix     = "usetotout",
    ))

    # 3–7. CSV-fed soft ratio + hard ratio-bound builders.
    #      Two builders may consume the same params_id (e.g. 'conc' → both
    #      RatioConstraintBuilder and RatioBoundBuilder).  stock_bounds joins
    #      the list: its CSV carries `agg=true` so the builder sums all
    #      matched a-side / b-side edges into one G row per
    #      (region, time, valuation) bucket.
    param_refs: list[dict] = []
    seen_param_ids: set[str] = set()

    def _register_builder(Builder, params_id: str, **kwargs) -> None:
        csv_path = inputs.params.resolve(params_id) / "params.csv"
        _log.info("Registering %s ← params/%s v%d",
                  Builder.__name__, params_id,
                  inputs.params.entry(params_id)["version"])
        engine.register(Builder.from_csv(str(csv_path), acc_struct=acc, **kwargs))
        if params_id not in seen_param_ids:
            param_refs.append(inputs.params.entry(params_id))
            seen_param_ids.add(params_id)

    # 3. Soft ratio — concentration
    _register_builder(RatioConstraintBuilder, "conc",
                      sigma_rel_floor=RATIO_SIGMA_REL_FLOOR,
                      scale_abs_floor=RATIO_SCALE_ABS_FLOOR)
    # 4. Soft ratio — efficiency
    _register_builder(RatioConstraintBuilder, "eff",
                      sigma_rel_floor=RATIO_SIGMA_REL_FLOOR,
                      scale_abs_floor=RATIO_SCALE_ABS_FLOOR)
    # 5. Hard ratio bounds — concentration
    _register_builder(RatioBoundBuilder, "conc", kind="conc")
    # 6. Hard ratio bounds — efficiency
    _register_builder(RatioBoundBuilder, "eff", kind="eff")
    # 7. Net stock bounds
    _register_builder(RatioBoundBuilder, "stock_bounds", kind="stock_bounds")

    # 8. Hard flow bounds from acc_struct
    _log.info("Registering FlowBoundsBuilder")
    engine.register(FlowBoundsBuilder())

    # ── C. Run and persist ────────────────────────────────────────────────
    _log.info("Running ConstraintEngine over %d timesteps", len(acc.time_range))
    parents = {
        "inputs": {
            "acc":    acc_entry,
            "params": param_refs,
        },
        "upstream": {"imp_run_id": imp_run_id},
    }
    constraint_run = engine.run(
        agg_result   = agg_result,
        timesteps    = [str(t) for t in acc.time_range],
        persist      = True,
        diag_plot    = True,
        label        = label,
        parents      = parents,
        stage_config = {
            "sigma_a_cv_min":    SIGMA_A_CV_MIN,
            "sigma_a_abs_floor": SIGMA_A_ABS_FLOOR,
            "sigma_c_cv_min":    SIGMA_C_CV_MIN,
            "sigma_c_abs_floor": SIGMA_C_ABS_FLOOR,
            "prior_box_k":       PRIOR_BOX_K,
            "nodata_anchor_cv":  NODATA_ANCHOR_CV,
            "ratio_sigma_rel_floor": RATIO_SIGMA_REL_FLOOR,
            "ratio_scale_abs_floor": RATIO_SCALE_ABS_FLOOR,
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
    parser.add_argument(
        "--imp-run-id", required=True,
        help="ImputationRun ID to load from data/proc/imp/.",
    )
    parser.add_argument("label", help="Label for this constraint run")
    args = parser.parse_args()
    main(args.imp_run_id, args.label)
