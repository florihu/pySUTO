"""
collect_aggregate_impute.py
===========================
Stage 2 driver: feed collection, inverse-variance aggregation, and imputation.

Steps
-----
A  Load accounting structure and feed registry
B  Collect and aggregate feeds (baci, crude_ore, icsg)
C  Run the case's imputation cascade (pysuto.cases.copper.imputers, #120):
     1  trade seeds, network closure, concentration, ore-grade and efficiency
        ratios, iterated to a fixpoint in CASCADE_ORDER
     2  temporal interpolation (interpolation-only)
     3  net-stock share imputation
     4  flat-prior fallback
D  Save AggregationResult to OUTPUT_DIR

═══════════════════════════════════════════════════════════
CONFIGURATION  ←  every knob is a field of the case spec,
                  src/pysuto/cases/copper.py (edit it there)
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
# Upstream wiring (feed IDs) is passed via repeated `--feed-id` CLI flags. main()
# reads `acc_id` from `--config` (default config/config.yaml) and derives every path
# from `--data-dir` (default data): proc/acc, proc/feed/registry, proc/imp, input/.
# Every knob lives in the case spec (pysuto.cases.copper).

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto.acc      import AccountingStructure, check_built_acc
from pysuto.case     import Cascade, CaseContext, CaseSpec
from pysuto.cases.copper import CASE, cascade_in_order
from pysuto.config   import load_config
from pysuto.registry import DataFeedRegistry, ImputationRunRegistry, stage_lock
from pysuto.input_registry import InputRegistries, check_feed_accs, input_pins
from pysuto.collector import FeedCollector
from pysuto.flow_node_balance import K_CLOSE as UNCLOSED_K
from pysuto.impute   import ImputationEngine
from pysuto.logging  import get_logger, add_run_handler
from pysuto.knobs    import add_knob_args, overrides_from_cli, write_run_id
from pysuto.diag     import (
    AggDiagnostic, CoverageDiagnostic, ImpDiagnostic, StatsDiagnostic,
    ownership_violations,
)

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION — the copper case (pysuto.cases.copper, #120) declares every
# Stage 2 knob, its value and the imputer cascade; this driver builds from it.
# ─────────────────────────────────────────────────────────────────────────────

# The knobs a pipeline step may override with --set (#100, pysuto.knobs): the UPPER_CASE
# fields of the case's Stage 2 section. Paths, the acc id, CV_GEO4 (a pedigree constant)
# and stock_sectors (ADR 0040 membership, kept in four places, #39) are not knobs.
KNOBS = tuple(CASE.imp.knobs())

def main(feed_ids: list[str], label: str, overrides: dict | None = None, *,
         case: CaseSpec = CASE,
         data_dir: str = "data", config: str = os.path.join("config", "config.yaml"),
         imp_dir: str | None = None, pin_inputs_from: str | None = None) -> str:
    """Run Stage 2 on ``feed_ids``; return the imp run id.

    The run builds from ``case``'s Stage 2 section (default the copper case; the
    command line passes it with its ``--set`` overrides applied); ``overrides`` is
    recorded in the sidecar. ``data_dir`` holds ``proc/`` and ``input/``; the run is
    written to ``imp_dir`` (default ``<data_dir>/proc/imp``). ``pin_inputs_from`` names
    an imp run in ``<data_dir>/proc/imp`` whose recorded input versions this run uses
    (#53).
    """
    spec = case.imp
    log_path = os.path.join(
        "logs",
        f"collect_aggregate_impute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    os.makedirs("logs", exist_ok=True)
    add_run_handler(log_path)
    logger = get_logger("collect_aggregate_impute")
    logger.info("=== collect_aggregate_impute START ===")

    acc_id       = load_config(config).pipeline.acc_id
    proc         = os.path.join(data_dir, "proc")
    acc_dir      = os.path.join(proc, "acc")
    registry_dir = os.path.join(proc, "feed", "registry")
    if imp_dir is None:
        imp_dir = os.path.join(proc, "imp")
    feed_keys = list(feed_ids)

    # ── 0  Initialise input registries (scan-and-bump) ────────────────────────
    pins = None
    if pin_inputs_from is not None:
        ref_sidecar = (ImputationRunRegistry(storage_dir=os.path.join(proc, "imp"))
                       .load_sidecar(pin_inputs_from))
        pins = input_pins(ref_sidecar, acc_id)
        logger.info("0  Input versions pinned from %s: %s", pin_inputs_from, pins)
    inputs = InputRegistries.initialize(root=os.path.join(data_dir, "input"), pins=pins)
    acc_entry = inputs.acc.entry(acc_id)
    logger.info("0  Input acc: %s v%d (code=%s)",
                acc_entry["id"], acc_entry["version"], acc_entry["code"])

    # ── A  Load infrastructure ────────────────────────────────────────────────
    # A pinned acc version is verifiable only against a structure with a build record.
    check_built_acc(acc_dir, acc_entry, require_record=pins is not None)
    logger.info("A  Loading AccountingStructure '%s' from %s", acc_id, acc_dir)
    acc = AccountingStructure.load(id=acc_id, storage_dir=acc_dir)

    logger.info("A  Loading DataFeedRegistry from %s", registry_dir)
    registry = DataFeedRegistry(odir=registry_dir)
    registry.load_index()

    # ── B  Collect and aggregate ──────────────────────────────────────────────
    logger.info("B  Collecting feeds: %s (p=%.2f)", feed_keys, spec.P_PROP)
    collector  = FeedCollector(registry=registry, acc_struct=acc, min_flow_kt=0.01)
    collector.select(feed_keys)
    feed_run_ids = [s["feed_run_id"] for s in collector.selected]
    logger.info("B  Resolved feed_run_ids: %s", feed_run_ids)
    check_feed_accs({r: registry.load_sidecar(r) for r in feed_run_ids},
                    inputs.acc, acc_entry)
    agg_result = collector.aggregate(p=spec.P_PROP)
    logger.info(
        "B  Aggregation done: %d covered / %d total edges",
        agg_result.n_covered, agg_result.n_edges,
    )

    ctx = CaseContext(case=case, acc=acc, inputs=inputs, data_dir=data_dir,
                      agg_result=agg_result, logger=logger)
    if spec.pre_impute is not None:
        spec.pre_impute(ctx)

    # Snapshot post-aggregation distributions BEFORE imputation mutates them.
    # Distribution panels focus on 1995 (agg_root_by_year stays all-year).
    agg_diag = AggDiagnostic(agg_result, acc_struct=acc, label=label, year="1995")
    agg_diag.summary()

    # ── C  Imputation pipeline ────────────────────────────────────────────────
    # The case's cascade (pysuto.cases.copper.imputers): its fixpoint group in the
    # group's order, then each terminal imputer, registered in the order listed.
    logger.info("C  Building imputation engine")
    engine = ImputationEngine(acc_struct=acc)

    for item in spec.imputers(ctx):
        if isinstance(item, Cascade):
            engine.register_cascade(item.imputers, max_iter=item.max_iter)
        else:
            engine.register(item)

    logger.info("C  Running imputation engine")
    agg_result = engine.run(agg_result)
    _meta_df    = agg_result.imputation_metadata.as_dataframe()
    _passes     = _meta_df["pass"].dropna()
    passes_run  = int(_passes.max()) if not _passes.empty else 0
    logger.info("C  Imputation done: %d log entries, cascade ran %d pass(es)",
                len(agg_result.imputation_log), passes_run)

    ctx.agg_result = agg_result
    if spec.post_impute is not None:
        spec.post_impute(ctx)

    # Ownership-partition check (ADR 0010): trade owns cross-region edges, the
    # network owns same-region edges. A clean run has zero violations; warn loudly
    # if a guard ever regressed.
    _viol = ownership_violations(agg_result.imputation_log.as_dataframe(), acc.edge_meta)
    if len(_viol):
        logger.warning("C  ownership-partition VIOLATIONS: %d edge(s) filled by an "
                       "unintended owner\n%s", len(_viol), _viol.head(20).to_string(index=False))

    # ── D  Register run ───────────────────────────────────────────────────────
    logger.info("D  Registering imputation run in %s", imp_dir)
    imp_registry = ImputationRunRegistry(storage_dir=imp_dir)
    parents = {
        "inputs": {
            "acc":    acc_entry,
            "params": ctx.param_refs,
        },
        "upstream": {"feed_run_ids": feed_run_ids},
    }
    run_id = imp_registry.register(
        agg_result,
        label         = label,
        parents       = parents,
        imputers      = engine.imputers,
        stage_summary = {"cascade_passes_run": passes_run},
        stage_config = {
            "p":               spec.P_PROP,
            "q":               spec.Q_REG,
            "alpha_min":       spec.ALPHA_MIN_REG,
            "cascade_max_iter": spec.CASCADE_MAX_ITER,
            "cascade_order":   list(spec.CASCADE_ORDER),
            "flat_mode":       spec.FLAT_MODE,
            "outlier_leg_k":   spec.OUTLIER_LEG_K,
            "outlier_leg_cv":  spec.OUTLIER_LEG_CV,
            "outlier_ref":     "production+imports",
            "unclosed_sectors": list(spec.UNCLOSED_SECTORS),
            "unclosed_cv":     spec.UNCLOSED_CV,
            "unclosed_k":      UNCLOSED_K,
            "leg_sigma_a_ref":  spec.LEG_SIGMA_AREF,
            "leg_sigma_gamma":  spec.LEG_SIGMA_GAMMA,
            "leg_sigma_cv_cap": spec.LEG_SIGMA_CV_CAP,
            "flat_sigma_high": (spec.FLAT_SIGMA_HIGH if spec.FLAT_MODE == "noisy_zero"
                                else None),
            "flat_default_cv": (spec.FLAT_DEFAULT_CV if spec.FLAT_MODE == "magnitude_aware"
                                else None),
            "time": {
                "window":           spec.TIME_WINDOW,
                "min_obs":          spec.TIME_MIN_OBS,
                "gap_inflation":    spec.TIME_GAP_INFLATION,
                "method":           spec.TIME_METHOD,
                "interpolate_only": spec.TIME_INTERPOLATE_ONLY,
                "scope_same_region": spec.TIME_SCOPE_SAME_REGION,
            },
            "stock_sectors":      list(spec.stock_sectors),
            "stock_seed_cv":      spec.STOCK_SEED_CV,
            "stock_seed_abs_floor": spec.STOCK_SEED_ABS_FLOOR_KT,
            "effless_stock_sigma": spec.EFFLESS_STOCK_SIGMA_KT,
            "input_pins":         (None if pins is None
                                   else {"from": pin_inputs_from, "versions": pins}),
            "overrides":          dict(overrides or {}),
            "case":               case.describe("imp"),
        },
    )
    run_dir = os.path.join(imp_dir, run_id)
    logger.info("D  Registered run_id=%s  label=%s → %s", run_id, label, run_dir)

    # ── E  Diagnostics ────────────────────────────────────────────────────────
    logger.info("E  Running diagnostics")
    diag_dir = os.path.join(run_dir, "diag")
    os.makedirs(diag_dir, exist_ok=True)
    for name, report in ctx.reports.items():      # the case's screens
        report.to_csv(os.path.join(diag_dir, f"{name}.csv"), index=False)
    agg_diag.plot(odir=diag_dir, show=False)
    diag = ImpDiagnostic(agg_result, acc_struct=acc, label=run_id, year="1995")
    diag.plot(odir=diag_dir, show=False)
    diag.summary()
    CoverageDiagnostic(agg_result, acc_struct=acc, year="1995").write(diag_dir)
    StatsDiagnostic(agg_result, acc_struct=acc, year="1995").write(diag_dir)

    logger.info("=== collect_aggregate_impute DONE ===")
    return run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feed collection, aggregation, and imputation.")
    parser.add_argument(
        "--feed-id", action="append", required=True, dest="feed_ids",
        help="Feed run ID to include. Repeat for each feed.",
    )
    parser.add_argument("--label", help="Human-readable label stored in the run's sidecar")
    parser.add_argument("--data-dir", default="data",
                        help="Data root holding proc/ and input/ (default 'data').")
    parser.add_argument("--config", default=os.path.join("config", "config.yaml"),
                        help="Config naming the acc id (default config/config.yaml).")
    parser.add_argument("--imp-dir", default=None,
                        help="Where the run is written (default <data-dir>/proc/imp); a "
                             "sweep's scratch store (#108).")
    parser.add_argument("--pin-inputs-from", default=None, metavar="IMP_RUN_ID",
                        help="Use every input version this imp run of "
                             "<data-dir>/proc/imp recorded in its sidecar, instead of the "
                             "latest (#53).")
    add_knob_args(parser)
    args = parser.parse_args()
    overrides = overrides_from_cli(parser, args, KNOBS, CASE.imp.knobs())
    case = CASE.with_knobs("imp", overrides)
    try:
        cascade_in_order(dict.fromkeys(case.imp.cascade_ids), case.imp.CASCADE_ORDER)
    except ValueError as exc:
        parser.error(str(exc))
    # registries have no locking (#101)
    with stage_lock("collect_aggregate_impute", root=os.path.join(args.data_dir, "proc")):
        write_run_id(args.run_id_file,
                     main(args.feed_ids, args.label, overrides, case=case,
                          data_dir=args.data_dir, config=args.config,
                          imp_dir=args.imp_dir, pin_inputs_from=args.pin_inputs_from))
