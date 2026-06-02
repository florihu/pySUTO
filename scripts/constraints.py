"""
init_copper_constraints.py
==========================
Stage 3a driver: build and persist the constraint matrix for the copper model.

Constraint types (registered in order)
---------------------------------------
1  Hard mass balance        — MassBalanceBuilder
                              Σ incoming − Σ outgoing = 0 per (Flow|Process) node × time
2  Soft sum — ICSG totals   — SumConstraintBuilder.from_mask_anchor, one family
   anchor (production +       per ICSG_ANCHOR_FAMILIES entry (production, usage,
   usage)                     producer stock change). Each: Σ over a masked edge
                              family = the genuine ICSG total per process-side
                              region × year (CC), ROW groups excluded; σ_c =
                              ANCHOR_CV·|c| (tight). Reaggregates the ICSG feed
                              F_root, which is disaggregated across the opposite
                              endpoint.
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

# Final-use SB sectors whose incoming Cathode edges are anchored to the
# genuine ICSG total per consuming region (summed over all cathode origins).
ANCHOR_USE_SECTORS = {"FU", "MESC", "USC"}

# ICSG-total anchor families: each genuine published ICSG total becomes one
# soft-sum anchor family. An ICSG flow has one commodity node (entity=Flow) and
# one process/boundary node; the total is published per the *process-side*
# region, so we group by that region (sums over the disaggregated opposite
# endpoint). Each tuple is (sector_from, {sector_to…}, group_region, id_prefix).
# Rows whose group_region is ROW are dropped (ROW totals are an unreliable
# catch-all). Waste (Mining→WR, SXEW→Tailings), CP→PSC, and secondary scrap
# inputs are deliberately excluded — they are not published ICSG totals.
ICSG_ANCHOR_FAMILIES = [
    ("Mining",      {"Ore"},            "region_from", "mine_prod_anchor"),
    ("SXEW",        {"CSXEW"},          "region_from", "sxew_prod_anchor"),
    ("Concentrate", {"Smelting"},       "region_to",   "smelter_thru_anchor"),
    ("Matte",       {"CFR"},            "region_to",   "refine_cfr_anchor"),
    ("Anode",       {"ER"},             "region_to",   "refine_er_anchor"),
    ("Cathode",     ANCHOR_USE_SECTORS, "region_to",   "cathode_use_anchor"),
    ("CP",          {"Cathode"},        "region_from", "cp_cathode_anchor"),
]

# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

# make sure the package is importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.acc import AccountingStructure
from pysuto.config import load_config
from pysuto.registry import ImputationRunRegistry, DataFeedRegistry
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

    # ── A3. ICSG anchor reference ─────────────────────────────────────────
    # Load the F_root of the *same* ICSG feed run this imp run was built on
    # (no "latest" drift). from_mask_anchor re-aggregates it per family over the
    # disaggregated endpoint to recover the genuine published per-region totals.
    feed_run_ids = (imp_sidecar.get("parents", {})
                               .get("upstream", {})
                               .get("feed_run_ids", []))
    icsg_run_ids = [r for r in feed_run_ids if str(r).startswith("feed_icsg_")]
    if len(icsg_run_ids) != 1:
        raise RuntimeError(
            f"expected exactly one icsg feed run in imp provenance, "
            f"got {icsg_run_ids!r}"
        )
    icsg_run_id   = icsg_run_ids[0]
    feed_registry = DataFeedRegistry()
    feed_registry.load_index()
    icsg_F_root = feed_registry.load_result(icsg_run_id)["F_root"]
    _log.info("A3  ICSG anchor reference: %s  (F_root sum=%.1f kt)",
              icsg_run_id, float(icsg_F_root.sum()))

    # ── B. Assemble constraint engine ─────────────────────────────────────
    engine = ConstraintEngine(
        acc_struct        = acc,
        storage_dir       = OUTPUT_DIR,
        p                 = P_PROP,
        sigma_a_cv_min    = SIGMA_A_CV_MIN,
        sigma_a_abs_floor = SIGMA_A_ABS_FLOOR,
        sigma_c_cv_min    = SIGMA_C_CV_MIN,
        sigma_c_abs_floor = SIGMA_C_ABS_FLOOR,
    )

    # 1. Hard mass balance
    _log.info("Registering MassBalanceBuilder")
    engine.register(MassBalanceBuilder())

    # 2. Soft sum — ICSG totals anchor (production + usage)
    #    One family per ICSG_ANCHOR_FAMILIES entry. Each row is
    #    Σ (masked edges) = genuine ICSG total per process-side region × year (CC),
    #    ROW groups dropped, with tight σ_c = ANCHOR_CV·|c|.
    _log.info("Registering SumConstraintBuilder — ICSG totals anchor (%d families)",
              len(ICSG_ANCHOR_FAMILIES))
    ei_exp = expand_edge_index(acc.edge_index, acc.node_index)
    for sector_from, sectors_to, group_region, id_prefix in ICSG_ANCHOR_FAMILIES:
        mask = (
            ei_exp["sector_from"].eq(sector_from)
            & ei_exp["sector_to"].isin(sectors_to)
            & ei_exp[group_region].ne("ROW")
        ).to_numpy()
        engine.register(SumConstraintBuilder.from_mask_anchor(
            acc,
            mask          = mask,
            anchor_values = icsg_F_root,
            group_keys    = (group_region, "time", "valuation"),
            sigma_cv      = ANCHOR_CV,
            id_prefix     = id_prefix,
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
    _register_builder(RatioConstraintBuilder, "conc")
    # 4. Soft ratio — efficiency
    _register_builder(RatioConstraintBuilder, "eff")
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
        "upstream": {"imp_run_id": imp_run_id, "icsg_feed_run_id": icsg_run_id},
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
