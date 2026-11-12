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

import numpy as np

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

from pysuto.acc      import (ALL_SPEC_KEYS, AccountingStructure, check_built_acc,
                             check_feed_accs)
from pysuto.case     import Cascade, CaseContext, CaseSpec
from pysuto.cases.copper import CASE
from pysuto.config   import load_config
from pysuto.registry import DataFeedRegistry, ImputationRunRegistry, stage_lock
from pysuto.input_registry import InputRegistries, input_pins
from pysuto.collector import FeedCollector
from pysuto.param_relation import (conc_for, conc_nominal_lookup, conc_sectors,
                                   expand_edge_index)
from pysuto.flow_node_balance import (K_CLOSE as UNCLOSED_K,
                                      contradicted_flow_nodes)
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _time_interp_scope(ei, same_region: bool, stock_sectors, logger):
    """TimeImputer edge scope: same-region edges only (ADR 0026) when
    ``same_region``, never a stock-change leg (ADR 0040). None = unscoped."""
    scope = None
    if same_region:
        scope = (ei["region_from"].to_numpy() == ei["region_to"].to_numpy())
        logger.info("   time_interp scope: same-region edges only "
                    "(%d of %d; %d cross-region edges excluded, ADR 0026)",
                    int(scope.sum()), scope.size, int((~scope).sum()))
    if stock_sectors:
        # stock-change legs are never interpolated (ADR 0040)
        stock = ei["sector_to"].isin(stock_sectors).to_numpy()
        scope = (~stock) if scope is None else (scope & ~stock)
        logger.info("   time_interp scope: %d stock-change legs (%s) excluded "
                    "(ADR 0040)", int(stock.sum()), "/".join(stock_sectors))
    return scope

def _damp_leg_sigma(acc, agg_result, gamma, cv_cap, logger, a_ref_override=None):
    """Size-damp σ on covered bilateral Flow-origin Mass legs (ADR 0017).

    σ_e = cv_e · a_ref(sector)^(1−γ) · a_e^γ with a_ref = the sector's median
    covered bilateral leg |f|, pooled over years; resulting cv capped at
    cv_cap. Runs BEFORE the outlier screen (its cv-5 inflation overrides).
    """
    if gamma is None or gamma >= 1.0:
        logger.info("B  leg-σ damping disabled (gamma=%s)", gamma)
        return
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    cov = np.asarray(agg_result.coverage, dtype=bool)
    eid = ei["edge_id"].to_numpy()
    f = np.asarray(agg_result.f_agg, dtype=float)
    # entity_from Flow, any destination — Cathode's bilateral legs are
    # Flow→SB (final demand), matching the outlier screen's population.
    bilat = (cov[eid]
             & ei["entity_from"].eq("Flow").to_numpy()
             & (ei["region_from"] != ei["region_to"]).to_numpy()
             & ei["valuation"].eq(acc.valuation_roles.gross).to_numpy()
             & (np.abs(np.nan_to_num(f[eid])) > 0))
    legs = ei[bilat]
    n_damped = 0
    for sector, grp in legs.groupby("sector_from"):
        ids = grp["edge_id"].to_numpy()
        a = np.abs(f[ids])
        a_ref_run = float(np.median(a))
        # ADR 0027: across a measurement series the reference size is pinned to
        # the predecessor run's, so a change in the leg population cannot
        # re-price every leg in the sector under cover of the structural change.
        a_ref = float((a_ref_override or {}).get(sector, a_ref_run))
        if a_ref_override and sector in a_ref_override:
            logger.info("B  leg-σ damping %-12s a_ref PINNED %.4g kt "
                        "(this run would use %.4g)", sector, a_ref, a_ref_run)
        if a_ref <= 0:
            continue
        cv0 = agg_result.sigma_agg[ids] / a
        sigma = cv0 * a_ref ** (1.0 - gamma) * a ** gamma
        sigma = np.minimum(sigma, cv_cap * a)          # σ-cap on tiny legs
        agg_result.sigma_agg[ids] = sigma
        n_damped += len(ids)
        logger.info(
            "B  leg-σ damping %-12s a_ref=%.1f kt  n=%d  cv %0.3f–%0.3f "
            "(pedigree cv med %.3f)",
            sector, a_ref, len(ids), float(np.min(sigma / a)),
            float(np.max(sigma / a)), float(np.median(cv0)))
    logger.info("B  leg-σ damping: gamma=%.2f cv_cap=%.2f — %d legs damped",
                gamma, cv_cap, n_damped)


def _screen_outlier_trade_legs(acc, agg_result, conc_csv: str, logger,
                               grade_csv: str | None = None, *, k: float, cv: float):
    """σ-inflate implausible bilateral Mass legs (see the case's OUTLIER_LEG_K,
    passed as ``k``, and OUTLIER_LEG_CV, passed as ``cv``).

    Per (flow sector, exporter region, year): implied CC exports =
    Σ covered bilateral Flow→* Mass legs · conc_nominal(sector); reference =
    Σ covered Process→Flow CC edges into that exporter's flow node (ICSG
    production) + implied CC imports (Σ covered bilateral Mass legs into the
    region · conc_nominal), so re-exporters are judged against what they
    actually receive. Groups without covered production stay unjudged (the
    transit-hub population; see the case's OUTLIER_LEG_K note).
    Ratio > k ⇒ every covered bilateral Mass leg
    of that group gets sigma_agg = cv·|f_agg|. Returns the report
    DataFrame (flagged groups only; empty frame if none).
    """
    import pandas as pd

    ei = expand_edge_index(acc.edge_index, acc.node_index)
    cov = np.asarray(agg_result.coverage, dtype=bool)
    f = np.nan_to_num(np.asarray(agg_result.f_agg, dtype=float))

    conc_lookup = conc_nominal_lookup(*[c for c in (conc_csv, grade_csv) if c],
                                      gross=acc.valuation_roles.gross)
    sectors = conc_sectors(conc_lookup)

    eid = ei["edge_id"].to_numpy()
    bilat = (cov[eid]
             & ei["entity_from"].eq("Flow").to_numpy()
             & (ei["region_from"] != ei["region_to"]).to_numpy()
             & ei["valuation"].eq(acc.valuation_roles.gross).to_numpy()
             & ei["sector_from"].isin(sectors).to_numpy())
    prodm = (cov[eid]
             & ei["entity_from"].eq("Process").to_numpy()
             & ei["entity_to"].eq("Flow").to_numpy()
             & ei["valuation"].eq(acc.valuation_roles.content).to_numpy())

    legs = ei[bilat].copy()
    # priced with the EXPORTER's concentration — the grade belongs to the origin
    legs["cc"] = f[legs["edge_id"].to_numpy()] \
        * conc_for(conc_lookup, legs["sector_from"], legs["region_from"])
    exp_cc = legs.groupby(["sector_from", "region_from", "time"])["cc"].sum()
    imp_cc = legs.groupby(["sector_from", "region_to", "time"])["cc"].sum()
    imp_cc.index.names = exp_cc.index.names

    prod = ei[prodm].copy()
    prod["cc"] = f[prod["edge_id"].to_numpy()]
    prod_cc = prod.groupby(["sector_to", "region_to", "time"])["cc"].sum()
    prod_cc.index.names = exp_cc.index.names

    both = pd.concat([exp_cc.rename("exports_cc"), prod_cc.rename("prod_cc"),
                      imp_cc.rename("import_cc")], axis=1).fillna(0.0)
    both["ref_cc"] = both["prod_cc"] + both["import_cc"]
    both = both[(both["prod_cc"] > 0) & (both["exports_cc"] > 0)]
    both["ratio"] = both["exports_cc"] / both["ref_cc"]
    flagged = both[both["ratio"] > k].reset_index()

    n_edges = 0
    for _, g in flagged.iterrows():
        m = bilat \
            & ei["sector_from"].eq(g["sector_from"]).to_numpy() \
            & ei["region_from"].eq(g["region_from"]).to_numpy() \
            & ei["time"].eq(g["time"]).to_numpy()
        ids = ei["edge_id"].to_numpy()[m]
        ids = ids[np.abs(f[ids]) > 0]      # a genuine 0 keeps its σ (no 0-pin)
        agg_result.sigma_agg[ids] = cv * np.abs(agg_result.f_agg[ids])
        n_edges += len(ids)

    if len(flagged):
        logger.warning(
            "B  outlier-leg screen: %d exporter-sector-years flagged "
            "(implied CC exports > %.1f× covered production+imports) — σ "
            "inflated to cv=%.1f on %d bilateral Mass legs\n%s",
            len(flagged), k, cv, n_edges,
            flagged.sort_values("ratio", ascending=False).head(10).to_string(index=False))
    else:
        logger.info("B  outlier-leg screen: no exporter-sector-year exceeds "
                    "%.1f× covered production+imports — nothing inflated",
                    k)
    return flagged


def _screen_contradicted_nodes(acc, agg_result, logger, *, sectors: tuple, cv):
    """σ-inflate the outflow edges of contradicted Flow nodes (ADR 0019) of
    ``sectors`` to ``cv``·|f| (the case's UNCLOSED_SECTORS and UNCLOSED_CV; ``None``
    detects only).

    Detection lives in ``pysuto.flow_node_balance`` so Stage 3 applies the
    identical rule to the scale-total anchors. Returns the report DataFrame
    (failing nodes only; empty frame if none).

    Runs **after** imputation: the inflow side of a Flow node (Smelting→Matte)
    is never fed, so pre-cascade its prior is 0 and no node looks contradicted.
    """
    keys, report = contradicted_flow_nodes(
        acc, agg_result, sectors=sectors)
    if not keys:
        logger.info("B  contradicted-node screen: every %s node's outflow "
                    "prior honours its inflow — nothing inflated",
                    "/".join(sectors))
        return report

    ei = expand_edge_index(acc.edge_index, acc.node_index)
    cov = np.asarray(agg_result.coverage, dtype=bool)
    f = np.nan_to_num(np.asarray(agg_result.f_agg, dtype=float))
    eid = ei["edge_id"].to_numpy()
    outflow = cov[eid] & ei["entity_from"].eq("Flow").to_numpy()
    node_key = list(zip(ei["sector_from"], ei["region_from"], ei["time"]))
    hit = outflow & np.array([k in keys for k in node_key])

    ids = eid[hit]
    ids = ids[np.abs(f[ids]) > 0]          # a genuine 0 keeps its σ (no 0-pin)
    if cv is not None:
        agg_result.sigma_agg[ids] = cv * np.abs(agg_result.f_agg[ids])
    logger.warning(
        "B  contradicted-node screen: %d %s node-years whose outflow prior is "
        "< %.2f× their inflow prior (%d outflow edges) — σ action: %s\n%s",
        len(keys), "/".join(sectors), UNCLOSED_K, len(ids),
        f"cv={cv}" if cv is not None
        else "none (ADR 0019 — detector only)",
        report.head(10).to_string(index=False))
    return report


def cascade_in_order(by_id: dict, order) -> list:
    """The cascade imputers of ``by_id`` in ``order``, which must be a permutation of
    ``by_id``'s ids (#104); anything else — an unknown, repeated or missing id —
    raises ``ValueError``."""
    order = tuple(order)
    if sorted(order) != sorted(by_id):
        raise ValueError(f"CASCADE_ORDER {order} must be a permutation of "
                         f"{tuple(by_id)}")
    return [by_id[k] for k in order]


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
    check_built_acc(acc_dir, acc_entry, inputs.acc, ALL_SPEC_KEYS,
                    require_record=pins is not None)
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

    # Size-damped leg σ (ADR 0017), then the plausibility screen — the screen's
    # cv-5 inflation overrides damping on flagged legs.
    _damp_leg_sigma(acc, agg_result, spec.LEG_SIGMA_GAMMA, spec.LEG_SIGMA_CV_CAP, logger,
                    a_ref_override=spec.LEG_SIGMA_AREF)
    outlier_report = _screen_outlier_trade_legs(
        acc, agg_result, str(inputs.params.resolve("conc") / "params.csv"), logger,
        k=spec.OUTLIER_LEG_K, cv=spec.OUTLIER_LEG_CV)

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

    # The time interpolator's scope (ADR 0026, ADR 0040), read by the case's cascade.
    ctx.time_scope = _time_interp_scope(expand_edge_index(acc.edge_index, acc.node_index),
                                        spec.TIME_SCOPE_SAME_REGION, spec.stock_sectors,
                                        logger)
    for item in spec.imputers(ctx):
        if isinstance(item, Cascade):
            engine.register_cascade(cascade_in_order(item.imputers, item.order),
                                    max_iter=item.max_iter)
        else:
            engine.register(item)

    logger.info("C  Running imputation engine")
    agg_result = engine.run(agg_result)
    _meta_df    = agg_result.imputation_metadata.as_dataframe()
    _passes     = _meta_df["pass"].dropna()
    passes_run  = int(_passes.max()) if not _passes.empty else 0
    logger.info("C  Imputation done: %d log entries, cascade ran %d pass(es)",
                len(agg_result.imputation_log), passes_run)

    # ADR 0019 — must run AFTER imputation: a Flow node's inflow edge
    # (Smelting→Matte) carries no feed observation, so before the cascade the
    # inflow prior reads 0 and no contradiction is visible. The outflow edges
    # it inflates are covered, so imputation never overwrites them.
    unclosed_report = _screen_contradicted_nodes(acc, agg_result, logger,
                                                 sectors=spec.UNCLOSED_SECTORS,
                                                 cv=spec.UNCLOSED_CV)
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
    outlier_report.to_csv(os.path.join(diag_dir, "outlier_leg_screen.csv"), index=False)
    unclosed_report.to_csv(
        os.path.join(diag_dir, "contradicted_node_screen.csv"), index=False)
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
