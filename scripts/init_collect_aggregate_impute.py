"""
collect_aggregate_impute.py
===========================
Stage 2 driver: feed collection, inverse-variance aggregation, and imputation.

Steps
-----
A  Load accounting structure and feed registry
B  Collect and aggregate feeds (baci, crude_ore, icsg)
C  Run imputation pipeline:
     0  ore-grade params (region/time-specific, higher priority)
     1  global concentration params (generic fallback)
     2  efficiency params
     3  domestic-use FormulaImputer for traded flows
        (Concentrate, Matte, Anode, Cathode × FU/USC/MESC)
     4  temporal interpolation (window=3, interpolation-only)
D  Save AggregationResult to OUTPUT_DIR

═══════════════════════════════════════════════════════════
USER CONFIGURATION  ←  only section you ever edit
═══════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

ACC_ID       = "copper"
ACC_DIR      = os.path.join("data", "proc", "acc")
REGISTRY_DIR = os.path.join("data", "proc", "feed", "registry")

IMP_DIR      = os.path.join("data", "proc", "imp")  # ImputationRunRegistry storage dir
PARAMS_DIR   = os.path.join("data", "input", "params")  # where to find CSVs for ParametricImputer

# Power-mean exponent for σ propagation (Stage 1, Stage 2a, all imputers).
# p ∈ [1.1, 1.5]; p=2.0 recovers Gaussian quadrature. See README.
P_PROP       = 1.3
# ParametricImputer α-regularization (s(α) = |α|/(|α|^q + α_min^q)^(1/q)).
# Caps σ amplification for large |α| and damps very small |α|. See README.
Q_REG        = 3.0
ALPHA_MIN_REG = 0.05
# Default CV for FlatPrior edges that match no listed mask (supply/use/...)
FLAT_DEFAULT_CV = 0.3
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.acc      import AccountingStructure
from pysuto.registry import DataFeedRegistry, ImputationRunRegistry
from pysuto.collector import FeedCollector
from pysuto.impute   import (
    ImputationEngine,
    ParametricImputer,
    FormulaImputer,
    FormulaRelation,
    SourceTerm,
    ParamRef,
    TimeImputer,
    FlatPriorImputer,
)
from pysuto.logging  import get_logger, add_run_handler
from pysuto.diag     import AggDiagnostic, ImpDiagnostic

# ---------------------------------------------------------------------------
# FormulaRelations (module-level — no runtime state, safe to define here)
# ---------------------------------------------------------------------------

concentrate_domestic = FormulaRelation(
    description="Domestic Concentrate use: MF supply/conc - exports, clip 0",
    target=lambda ei: (
        (ei["sector_from"] == "Concentrate") & (ei["entity_from"] == "Flow") &
        (ei["sector_to"]   == "Smelting")    & (ei["entity_to"]   == "Process") &
        (ei["region_from"] == ei["region_to"]) & (ei["valuation"] == "Mass")
    ),
    terms=[
        SourceTerm(
            selector=lambda ei: (
                (ei["sector_from"] == "MF")          & (ei["entity_from"] == "Process") &
                (ei["sector_to"]   == "Concentrate") & (ei["entity_to"]   == "Flow") &
                (ei["valuation"]   == "CC")
            ),
            coefficient=ParamRef("conc_concentrate_supply", "inverse"),
            region_binding="to_matches_target_from",
            aggregate="single",
        ),
        SourceTerm(
            selector=lambda ei: (
                (ei["sector_from"] == "Concentrate") & (ei["entity_from"] == "Flow") &
                (ei["entity_to"]   == "Flow") & (ei["valuation"] == "Mass")
            ),
            coefficient=-1.0,
            region_binding="from_matches_target_from",
            aggregate="sum",
        ),
    ],
    clip_lower=0.0,
    epsilon=0.05,
)

matte_domestic = FormulaRelation(
    description="Domestic Matte use: Smelting supply/conc - exports, clip 0",
    target=lambda ei: (
        (ei["sector_from"] == "Matte") & (ei["entity_from"] == "Flow") &
        (ei["sector_to"]   == "CFR")   & (ei["entity_to"]   == "Process") &
        (ei["region_from"] == ei["region_to"]) & (ei["valuation"] == "Mass")
    ),
    terms=[
        SourceTerm(
            selector=lambda ei: (
                (ei["sector_from"] == "Smelting") & (ei["entity_from"] == "Process") &
                (ei["sector_to"]   == "Matte")    & (ei["entity_to"]   == "Flow") &
                (ei["valuation"]   == "CC")
            ),
            coefficient=ParamRef("conc_matte_supply", "inverse"),
            region_binding="to_matches_target_from",
            aggregate="single",
        ),
        SourceTerm(
            selector=lambda ei: (
                (ei["sector_from"] == "Matte") & (ei["entity_from"] == "Flow") &
                (ei["entity_to"]   == "Flow")  & (ei["valuation"]   == "Mass")
            ),
            coefficient=-1.0,
            region_binding="from_matches_target_from",
            aggregate="sum",
        ),
    ],
    clip_lower=0.0,
    epsilon=0.05,
)

anode_domestic = FormulaRelation(
    description="Domestic Anode use: CFR supply/conc - exports, clip 0",
    target=lambda ei: (
        (ei["sector_from"] == "Anode") & (ei["entity_from"] == "Flow") &
        (ei["sector_to"]   == "ER")    & (ei["entity_to"]   == "Process") &
        (ei["region_from"] == ei["region_to"]) & (ei["valuation"] == "Mass")
    ),
    terms=[
        SourceTerm(
            selector=lambda ei: (
                (ei["sector_from"] == "CFR")   & (ei["entity_from"] == "Process") &
                (ei["sector_to"]   == "Anode") & (ei["entity_to"]   == "Flow") &
                (ei["valuation"]   == "CC")
            ),
            coefficient=ParamRef("conc_anode_supply", "inverse"),
            region_binding="to_matches_target_from",
            aggregate="single",
        ),
        SourceTerm(
            selector=lambda ei: (
                (ei["sector_from"] == "Anode") & (ei["entity_from"] == "Flow") &
                (ei["entity_to"]   == "Flow")  & (ei["valuation"]   == "Mass")
            ),
            coefficient=-1.0,
            region_binding="from_matches_target_from",
            aggregate="sum",
        ),
    ],
    clip_lower=0.0,
    epsilon=0.05,
)

def _cathode_supply_term():
    return SourceTerm(
        selector=lambda ei: (
            (ei["sector_from"] == "CP")      & (ei["entity_from"] == "Process") &
            (ei["sector_to"]   == "Cathode") & (ei["entity_to"]   == "Flow") &
            (ei["valuation"]   == "CC")
        ),
        coefficient=ParamRef("conc_cathode_supply", "inverse"),
        region_binding="to_matches_target_from",
        aggregate="single",
    )

def _cathode_export_term():
    return SourceTerm(
        selector=lambda ei: (
            (ei["sector_from"] == "Cathode") & (ei["entity_from"] == "Flow") &
            (ei["entity_to"]   == "Flow")    & (ei["valuation"]   == "Mass")
        ),
        coefficient=-1.0,
        region_binding="from_matches_target_from",
        aggregate="sum",
    )

def _cathode_domestic_subtract(sector_to: str):
    return SourceTerm(
        selector=lambda ei, s=sector_to: (
            (ei["sector_from"] == "Cathode") & (ei["entity_from"] == "Flow") &
            (ei["sector_to"]   == s)         & (ei["entity_to"]   == "Process") &
            (ei["region_from"] == ei["region_to"]) & (ei["valuation"] == "Mass")
        ),
        coefficient=-1.0,
        region_binding="from_matches_target_from",
        aggregate="single",
    )

cathode_fu_domestic = FormulaRelation(
    description="Domestic Cathode→FU: CP supply/conc - exports - USC - MESC, clip 0",
    target=lambda ei: (
        (ei["sector_from"] == "Cathode") & (ei["entity_from"] == "Flow") &
        (ei["sector_to"]   == "FU")      & (ei["entity_to"]   == "Process") &
        (ei["region_from"] == ei["region_to"]) & (ei["valuation"] == "Mass")
    ),
    terms=[
        _cathode_supply_term(),
        _cathode_export_term(),
        _cathode_domestic_subtract("USC"),
        _cathode_domestic_subtract("MESC"),
    ],
    clip_lower=0.0,
    epsilon=0.05,
)

cathode_usc_domestic = FormulaRelation(
    description="Domestic Cathode→USC: CP supply/conc - exports - FU - MESC, clip 0",
    target=lambda ei: (
        (ei["sector_from"] == "Cathode") & (ei["entity_from"] == "Flow") &
        (ei["sector_to"]   == "USC")     & (ei["entity_to"]   == "Process") &
        (ei["region_from"] == ei["region_to"]) & (ei["valuation"] == "Mass")
    ),
    terms=[
        _cathode_supply_term(),
        _cathode_export_term(),
        _cathode_domestic_subtract("FU"),
        _cathode_domestic_subtract("MESC"),
    ],
    clip_lower=0.0,
    epsilon=0.05,
)

cathode_mesc_domestic = FormulaRelation(
    description="Domestic Cathode→MESC: CP supply/conc - exports - FU - USC, clip 0",
    target=lambda ei: (
        (ei["sector_from"] == "Cathode") & (ei["entity_from"] == "Flow") &
        (ei["sector_to"]   == "MESC")    & (ei["entity_to"]   == "Process") &
        (ei["region_from"] == ei["region_to"]) & (ei["valuation"] == "Mass")
    ),
    terms=[
        _cathode_supply_term(),
        _cathode_export_term(),
        _cathode_domestic_subtract("FU"),
        _cathode_domestic_subtract("USC"),
    ],
    clip_lower=0.0,
    epsilon=0.05,
)


def main(label: str, feed_run_ids: list[str]) -> None:
    log_path = os.path.join(
        "logs",
        f"collect_aggregate_impute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    os.makedirs("logs", exist_ok=True)
    add_run_handler(log_path)
    logger = get_logger("collect_aggregate_impute")
    logger.info("=== collect_aggregate_impute START ===")

    # ── A  Load infrastructure ────────────────────────────────────────────────
    logger.info("A  Loading AccountingStructure '%s' from %s", ACC_ID, ACC_DIR)
    acc = AccountingStructure.load(id=ACC_ID, storage_dir=ACC_DIR)

    logger.info("A  Loading DataFeedRegistry from %s", REGISTRY_DIR)
    registry = DataFeedRegistry(odir=REGISTRY_DIR)
    registry.load_index()

    # ── B  Collect and aggregate ──────────────────────────────────────────────
    logger.info("B  Collecting feeds: %s (p=%.2f)", feed_run_ids, P_PROP)
    collector  = FeedCollector(registry=registry, acc_struct=acc, min_flow_kt=0.01)
    agg_result = collector.select(feed_run_ids).aggregate(p=P_PROP)
    logger.info(
        "B  Aggregation done: %d covered / %d total edges",
        agg_result.n_covered, agg_result.n_edges,
    )

    # Snapshot post-aggregation distributions BEFORE imputation mutates them.
    agg_diag = AggDiagnostic(agg_result, acc_struct=acc, label=label)
    agg_diag.summary()

    # ── C  Imputation pipeline ────────────────────────────────────────────────
    logger.info("C  Building imputation engine")
    engine = ImputationEngine(acc_struct=acc)

    # Step 0: ore-grade (specific, time-varying) — priority via specificity dedup
    engine.register(ParametricImputer.from_csv(
        os.path.join(PARAMS_DIR, "ore_grade_params.csv"), acc,
        p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG,
    ))

    # Step 1: global concentration params (fills gaps left by ore_grade)
    engine.register(ParametricImputer.from_csv(
        os.path.join(PARAMS_DIR, "copper_conc_params.csv"), acc,
        p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG,
    ))

    # Step 2: efficiency params
    engine.register(ParametricImputer.from_csv(
        os.path.join(PARAMS_DIR, "copper_efficiency_params.csv"), acc,
        p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG,
    ))

    # Step 3: domestic use balance for traded flows
    engine.register(FormulaImputer(
        relations=[
            concentrate_domestic,
            matte_domestic,
            anode_domestic,
            cathode_fu_domestic,
            cathode_usc_domestic,
            cathode_mesc_domestic,
        ],
        params_csv=os.path.join(PARAMS_DIR, "copper_conc_params.csv"),
        p=P_PROP,
    ))

    # Step 4: temporal interpolation, no extrapolation
    engine.register(TimeImputer(
        window=3,
        method="linear",
        min_obs=2,
        gap_inflation=1.05,
        interpolate_only=True,
        p=P_PROP,
    ))

    # Step 5: final fallback — flat prior sized by mask CV × group scale.
    engine.register(FlatPriorImputer.from_csv(
        os.path.join(PARAMS_DIR, "flat_prior_cv.csv"),
        acc_struct=acc,
        default_cv=FLAT_DEFAULT_CV,
        p=P_PROP,
    ))

    logger.info("C  Running imputation engine")
    agg_result = engine.run(agg_result)
    logger.info(
        "C  Imputation done: %d log entries",
        len(agg_result.imputation_log),
    )

    # ── D  Register run ───────────────────────────────────────────────────────
    logger.info("D  Registering imputation run in %s", IMP_DIR)
    imp_registry = ImputationRunRegistry(storage_dir=IMP_DIR)
    run_id = imp_registry.register(
        agg_result,
        label    = label,
        feeds    = collector.selected,
        imputers = engine.imputers,
        acc_id   = acc.id,
    )
    run_dir = os.path.join(IMP_DIR, run_id)
    logger.info("D  Registered run_id=%s  label=%s → %s", run_id, label, run_dir)

    # ── E  Diagnostics ────────────────────────────────────────────────────────
    logger.info("E  Running diagnostics")
    diag_dir = os.path.join(run_dir, "diag")
    agg_diag.plot(odir=diag_dir, show=False)
    diag = ImpDiagnostic(agg_result, acc_struct=acc, label=run_id)
    diag.plot(odir=diag_dir, show=False)
    diag.summary()

    logger.info("=== collect_aggregate_impute DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feed collection, aggregation, and imputation.")
    parser.add_argument("label", help="Human-readable label folded into the imputation run_id")
    parser.add_argument("feed_run_ids", nargs="+", help="Feed run IDs to collect (e.g. feed_baci_20260504_122237_f028190f)")
    args = parser.parse_args()
    main(args.label, args.feed_run_ids)
