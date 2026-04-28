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

import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

ACC_ID       = "copper"
ACC_DIR      = os.path.join("data", "proc", "acc")
REGISTRY_DIR = os.path.join("data", "proc", "feed", "registry")

IMP_ID      = "copper_agg_imp_v0"  # for logging and param references           # for logging and param references 
OUTPUT_DIR   = os.path.join("data", "proc", "imp", IMP_ID)  # where to save the final AggregationResult (must not exist yet)
PARAMS_DIR   = os.path.join("data", "input", "params")  # where to find CSVs for ParametricImputer
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.acc      import AccountingStructure
from pysuto.registry import DataFeedRegistry
from pysuto.collector import FeedCollector
from pysuto.impute   import (
    ImputationEngine,
    ParametricImputer,
    FormulaImputer,
    FormulaRelation,
    SourceTerm,
    ParamRef,
    TimeImputer,
)
from pysuto.logging  import get_logger, add_run_handler
from pysuto.diag     import ImpDiagnostic

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log_path = os.path.join(
    "logs",
    f"collect_aggregate_impute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)
os.makedirs("logs", exist_ok=True)
add_run_handler(log_path)
logger = get_logger("collect_aggregate_impute")
logger.info("=== collect_aggregate_impute START ===")

# ---------------------------------------------------------------------------
# A  Load infrastructure
# ---------------------------------------------------------------------------

logger.info("A  Loading AccountingStructure '%s' from %s", ACC_ID, ACC_DIR)
acc = AccountingStructure.load(id=ACC_ID, storage_dir=ACC_DIR)

logger.info("A  Loading DataFeedRegistry from %s", REGISTRY_DIR)
registry = DataFeedRegistry(odir=REGISTRY_DIR)
registry.load_index()

# ---------------------------------------------------------------------------
# B  Collect and aggregate
# ---------------------------------------------------------------------------

logger.info("B  Collecting feeds: baci, crude_ore, icsg")
collector  = FeedCollector(registry=registry, acc_struct=acc, min_flow_kt=0.01)
agg_result = collector.select(["baci", "crude_ore", "icsg"]).aggregate()
logger.info(
    "B  Aggregation done: %d covered / %d total edges",
    agg_result.n_covered, agg_result.n_edges,
)

# ---------------------------------------------------------------------------
# C  Imputation pipeline
# ---------------------------------------------------------------------------

# ── Domestic use FormulaRelations for traded flows ──────────────────────────
#
# Formula: domestic_use_Mass = supply_CC × (1/concentration) − Σ(exports_Mass)
#          clipped at zero.
#
# Cathode additionally subtracts the other two observed domestic use sinks.

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

# Cathode: three sinks (FU, USC, MESC). Each formula subtracts the other two
# observed domestic uses. If any source is NaN the formula skips and
# TimeImputer handles those edges.

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

# ── Assemble and run engine ──────────────────────────────────────────────────

logger.info("C  Building imputation engine")
engine = ImputationEngine(acc_struct=acc)

# Step 0: ore-grade (specific, time-varying) — priority via specificity dedup
engine.register(ParametricImputer.from_csv(
    os.path.join(PARAMS_DIR, "ore_grade_params.csv"), acc,
))

# Step 1: global concentration params (fills gaps left by ore_grade)
engine.register(ParametricImputer.from_csv(
    os.path.join(PARAMS_DIR, "copper_conc_params.csv"), acc,
))

# Step 2: efficiency params
engine.register(ParametricImputer.from_csv(
    os.path.join(PARAMS_DIR, "copper_efficiency_params.csv"), acc,
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
))

# Step 4: temporal interpolation, no extrapolation
engine.register(TimeImputer(
    window=3,
    method="linear",
    min_obs=2,
    gap_inflation=1.05,
    interpolate_only=True,
))

logger.info("C  Running imputation engine")
agg_result = engine.run(agg_result)
logger.info(
    "C  Imputation done: %d log entries",
    len(agg_result.imputation_log),
)

# ---------------------------------------------------------------------------
# D  Save
# ---------------------------------------------------------------------------

logger.info("D  Saving AggregationResult to %s", OUTPUT_DIR)
agg_result.save(OUTPUT_DIR)

# ---------------------------------------------------------------------------
# E  Diagnostics
# ---------------------------------------------------------------------------

logger.info("E  Running imputation diagnostics")
diag = ImpDiagnostic(agg_result, acc_struct=acc, label=IMP_ID)
diag.plot(odir=os.path.join(OUTPUT_DIR, "diag"), show=False)
diag.summary()

logger.info("=== collect_aggregate_impute DONE ===")
