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

import numpy as np

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

ACC_ID       = "copper"
ACC_DIR      = os.path.join("data", "proc", "acc")
REGISTRY_DIR = os.path.join("data", "proc", "feed", "registry")

IMP_DIR      = os.path.join("data", "proc", "imp")          # ImputationRunRegistry storage dir
_INPUT_ROOT  = os.path.join("data", "input")                # root of the input registries

# Propagation / imputer knobs (p, q, alpha_min, flat_default_cv, time imputer
# settings) are loaded from config/config.yaml — see `cfg = load_config()` below.
# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.acc      import AccountingStructure
from pysuto.config   import load_config
from pysuto.registry import DataFeedRegistry, ImputationRunRegistry
from pysuto.input_registry import InputRegistries
from pysuto.collector import FeedCollector
from pysuto.param_relation import expand_edge_index
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

cfg             = load_config()
P_PROP          = cfg.propagation.p
Q_REG           = cfg.propagation.q
ALPHA_MIN_REG   = cfg.propagation.alpha_min
FLAT_DEFAULT_CV = cfg.imputers.flat_prior.default_cv

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


def main(label: str, feed_keys: list[str]) -> None:
    log_path = os.path.join(
        "logs",
        f"collect_aggregate_impute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    os.makedirs("logs", exist_ok=True)
    add_run_handler(log_path)
    logger = get_logger("collect_aggregate_impute")
    logger.info("=== collect_aggregate_impute START ===")

    # ── 0  Initialise input registries (scan-and-bump) ────────────────────────
    inputs = InputRegistries.initialize(root=_INPUT_ROOT)
    acc_entry = inputs.acc.entry(ACC_ID)
    logger.info("0  Input acc: %s v%d (code=%s)",
                acc_entry["id"], acc_entry["version"], acc_entry["code"])

    # ── A  Load infrastructure ────────────────────────────────────────────────
    logger.info("A  Loading AccountingStructure '%s' from %s", ACC_ID, ACC_DIR)
    acc = AccountingStructure.load(id=ACC_ID, storage_dir=ACC_DIR)

    logger.info("A  Loading DataFeedRegistry from %s", REGISTRY_DIR)
    registry = DataFeedRegistry(odir=REGISTRY_DIR)
    registry.load_index()

    # ── B  Collect and aggregate ──────────────────────────────────────────────
    logger.info("B  Collecting feeds: %s (p=%.2f)", feed_keys, P_PROP)
    collector  = FeedCollector(registry=registry, acc_struct=acc, min_flow_kt=0.01)
    agg_result = collector.select(feed_keys).aggregate(p=P_PROP)
    logger.info(
        "B  Aggregation done: %d covered / %d total edges",
        agg_result.n_covered, agg_result.n_edges,
    )

    feed_run_ids = [s["feed_run_id"] for s in collector.selected if s["feed_run_id"]]
    logger.info("B  Resolved feed_run_ids: %s", feed_run_ids)

    # Snapshot post-aggregation distributions BEFORE imputation mutates them.
    agg_diag = AggDiagnostic(agg_result, acc_struct=acc, label=label)
    agg_diag.summary()

    # ── C  Imputation pipeline ────────────────────────────────────────────────
    logger.info("C  Building imputation engine")
    engine = ImputationEngine(acc_struct=acc)

    # Config-table-driven imputer registration.  Each tuple is
    #   (params_id, Builder, kwargs).  The params_id is resolved via the
    #   input registry → data/input/params/<id>/v{latest}/params.csv.
    IMP_BINDINGS: list[tuple[str, type, dict]] = [
        ("eff",        ParametricImputer, dict(p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG)),
        ("ore_grade",  ParametricImputer, dict(p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG)),
        ("conc",       ParametricImputer, dict(p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG)),
    ]

    param_refs: list[dict] = []
    for param_id, Builder, kwargs in IMP_BINDINGS:
        csv_path = inputs.params.resolve(param_id) / "params.csv"
        engine.register(Builder.from_csv(str(csv_path), acc, **kwargs))
        param_refs.append(inputs.params.entry(param_id))
        logger.info("C  registered %s (params=%s v%d)",
                    Builder.__name__, param_id,
                    inputs.params.entry(param_id)["version"])

    # FormulaImputer is lambda-driven; it shares the `conc` params CSV.
    engine.register(FormulaImputer(
        relations=[
            concentrate_domestic,
            matte_domestic,
            anode_domestic,
            cathode_fu_domestic,
            cathode_usc_domestic,
            cathode_mesc_domestic,
        ],
        params_csv=str(inputs.params.resolve("conc") / "params.csv"),
        p=P_PROP,
    ))

    # Temporal interpolation, no extrapolation.
    engine.register(TimeImputer(
        window           = cfg.imputers.time.window,
        method           = cfg.imputers.time.method,
        min_obs          = cfg.imputers.time.min_obs,
        gap_inflation    = cfg.imputers.time.gap_inflation,
        interpolate_only = cfg.imputers.time.interpolate_only,
        p                = P_PROP,
    ))

    # Final fallback — flat prior sized by mask CV × group scale.
    # CVs come from config; the ROW relation (any edge touching ROW as
    # region_from or region_to) is defined inline below and combined with
    # acc.masks["supply"] / acc.masks["use"] to form two row-specific
    # sub-masks whose (cv_lo, cv_hi) are pre-inflated by row_inflation.
    flat_cfg     = cfg.imputers.flat_prior
    supply_mask  = np.asarray(acc.masks["supply"], dtype=bool)
    use_mask     = np.asarray(acc.masks["use"],    dtype=bool)

    ei_exp       = expand_edge_index(acc.edge_index, acc.node_index)
    row_mask     = (
        (ei_exp["region_from"].to_numpy() == "ROW") |
        (ei_exp["region_to"].to_numpy()   == "ROW")
    )

    row_supply   = supply_mask & row_mask
    row_use      = use_mask    & row_mask

    inflation    = flat_cfg.row_inflation
    flat_cv_by_mask = {
        "row_supply": (flat_cfg.cv_lo_supply * inflation,
                       flat_cfg.cv_hi_supply * inflation),
        "row_use":    (flat_cfg.cv_lo_use    * inflation,
                       flat_cfg.cv_hi_use    * inflation),
        "supply":     (flat_cfg.cv_lo_supply, flat_cfg.cv_hi_supply),
        "use":        (flat_cfg.cv_lo_use,    flat_cfg.cv_hi_use),
    }

    engine.register(FlatPriorImputer(
        cv_by_mask = flat_cv_by_mask,
        default_cv = flat_cfg.default_cv,
        acc_struct = acc,
        p          = P_PROP,
        masks      = {"row_supply": row_supply, "row_use": row_use},
    ))

    logger.info("C  Running imputation engine")
    agg_result = engine.run(agg_result)
    logger.info("C  Imputation done: %d log entries", len(agg_result.imputation_log))

    # ── D  Register run ───────────────────────────────────────────────────────
    logger.info("D  Registering imputation run in %s", IMP_DIR)
    imp_registry = ImputationRunRegistry(storage_dir=IMP_DIR)
    parents = {
        "inputs": {
            "acc":    acc_entry,
            "params": param_refs,
        },
        "upstream": {"feed_run_ids": feed_run_ids},
    }
    run_id = imp_registry.register(
        agg_result,
        label        = label,
        parents      = parents,
        imputers     = engine.imputers,
        stage_config = {
            "p":               P_PROP,
            "q":               Q_REG,
            "alpha_min":       ALPHA_MIN_REG,
            "flat_default_cv": FLAT_DEFAULT_CV,
            "time": {
                "window":           cfg.imputers.time.window,
                "min_obs":          cfg.imputers.time.min_obs,
                "gap_inflation":    cfg.imputers.time.gap_inflation,
                "method":           cfg.imputers.time.method,
                "interpolate_only": cfg.imputers.time.interpolate_only,
            },
        },
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
    parser.add_argument("label", help="Human-readable label stored in the run's sidecar")
    parser.add_argument(
        "feed_keys", nargs="+",
        help="Feed selectors: input-feed-id (latest run wins) or exact run_id",
    )
    args = parser.parse_args()
    main(args.label, args.feed_keys)
