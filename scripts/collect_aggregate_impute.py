"""
collect_aggregate_impute.py
===========================
Stage 2 driver: feed collection, inverse-variance aggregation, and imputation.

Steps
-----
A  Load accounting structure and feed registry
B  Collect and aggregate feeds (baci, crude_ore, icsg)
C  Run imputation pipeline:
     1  concentration ratios (highest priority)
     2  ore-grade ratios
     3  efficiency ratios
     4  domestic-use FormulaImputer for traded flows
        (Concentrate, Matte, Anode, Cathode × FU/USC/MESC)
     5  temporal interpolation (interpolation-only)
     6  flat-prior fallback (ROW-aware CVs)
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
# Upstream wiring (feed IDs) is passed via repeated `--feed-id` CLI flags.
# `acc_id` is read from config/config.yaml; every other knob lives inline below.

ACC_DIR      = os.path.join("data", "proc", "acc")
REGISTRY_DIR = os.path.join("data", "proc", "feed", "registry")

IMP_DIR      = os.path.join("data", "proc", "imp")          # ImputationRunRegistry storage dir
_INPUT_ROOT  = os.path.join("data", "input")                # root of the input registries

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
from pysuto.flow_node_balance import (K_CLOSE as UNCLOSED_K,
                                      contradicted_flow_nodes)
from pysuto.unc import PedigreeScores
from pysuto.impute   import (
    ImputationEngine,
    TradeImputer,
    NetworkImputer,
    ParametricImputer,
    FormulaImputer,
    FormulaRelation,
    SourceTerm,
    ParamRef,
    TimeImputer,
    ShareImputer,
    FlatPriorImputer,
)
from pysuto.logging  import get_logger, add_run_handler
from pysuto.diag     import (
    AggDiagnostic, CoverageDiagnostic, ImpDiagnostic, StatsDiagnostic,
    ownership_violations,
)

cfg    = load_config()
ACC_ID = cfg.pipeline.acc_id

# ─────────────────────────────────────────────────────────────────────────────
# USER CONFIGURATION — every knob that used to live in config/config.yaml.
# Note: P_PROP must match scripts/constraints.py's P_PROP (cross-stage Lp).
# ─────────────────────────────────────────────────────────────────────────────

# Propagation / Lp settings
P_PROP        = 1.3      # Lp/power-mean exponent
Q_REG         = 3.0      # ParametricImputer backward-branch knee sharpness
ALPHA_MIN_REG = 0.05     # inverse-multiplier effective-sensitivity floor

# Imputation cascade — conc/eff/formula iterate to a fixpoint (re-run until a
# full sweep fills no new edges). Safety cap on passes (chain depth ≈ 8).
CASCADE_MAX_ITER = 10

# NetworkImputer — waste-rock mass as a ratio of ore output (WR_Mass = k·Ore_Mass).
# The ratio (k) is now a parameter: the `stripping_ratio` row in the eff params
# CSV (param_type=stripping_ratio), so it is no longer hard-coded here. A direct
# ratio avoids the CO_Mass − Ore_Mass residual going negative when a region's
# ore_grade exceeds the flat conc_ore.

# TimeImputer (general interpolation pass)
TIME_WINDOW           = 3
TIME_MIN_OBS          = 2
TIME_GAP_INFLATION    = 1.05
TIME_METHOD           = "linear"
TIME_INTERPOLATE_ONLY = True

# FlatPriorImputer mode
#   "magnitude_aware" (legacy) — σ = CV(j) · scale(j) via cv_by_mask + 3-level cascade
#   "noisy_zero" (test)        — f=0, σ=FLAT_SIGMA_HIGH; effectively no prior weight
FLAT_MODE       = "noisy_zero"
FLAT_SIGMA_HIGH = 1.0e6      # used only when FLAT_MODE == "noisy_zero"

# FlatPriorImputer — CVs per mask (ROW edges get the geographic pedigree penalty)
#   used only when FLAT_MODE == "magnitude_aware"
FLAT_DEFAULT_CV    = 0.3
FLAT_CV_LO_SUPPLY  = 0.04
FLAT_CV_HI_SUPPLY  = 0.3
FLAT_CV_LO_USE     = 0.2
FLAT_CV_HI_USE     = 0.5
# ROW edges carry the worst geographical pedigree (geographical_correlation=4),
# added in quadrature to the base CV — the same transparent penalty Stage 1
# applies to fed ROW edges (replaces the old ad-hoc FLAT_ROW_INFLATION x1.5).
CV_GEO4 = PedigreeScores._cv_other(4, "medium_sensitive")

# Bilateral-leg plausibility screen (post-aggregation, pre-imputation).
# BACI misclassification/gross-weight outliers (e.g. Papua New Guinea →
# Philippines/ROW "Concentrate" at 14–21 Mt/yr, >20× PNG's production) carry
# customs-grade σ into the QP, which then pays z≈1/cv per deleted leg. Where an
# exporter's implied bilateral CC exports (Σ Mass·conc_nominal) exceed
# OUTLIER_LEG_K × its covered CC supply of that sector-year — ICSG-covered
# production PLUS implied CC imports (Σ covered bilateral Mass in · conc), so
# legitimate re-exporters (Japan 1996–2004, Spain 2011–15: exports > own mine
# production, fed by imports) are not false positives — the exporter's
# bilateral Mass legs get σ inflated to OUTLIER_LEG_CV·|f|; the data stays
# (trace + report CSV), only its false authority goes, mirroring the
# pedigree-penalty style (ADR 0003). Exporters with no covered production are
# still skipped (P=0 groups are the transit-hub population — Round C measures
# them, Round D treats them structurally; σ-nuking ~1100 such groups here
# would be a hub policy in disguise). See REP
# analysis_opt_20260903_084528_45fdd041_masscc-tension-root-causes §4.
OUTLIER_LEG_K  = 3.0
OUTLIER_LEG_CV = 5.0

# Contradicted flow nodes (ADR 0019). A Flow node's hard mass balance cannot
# honour two disagreeing scale-total anchors; because σ_c scales with |c| the
# solver collapses the node to the SMALLER prior, even when that side is the
# less trustworthy one (Russia Matte 1995: inflow 545 kt CC from ICSG
# concentrate use vs outflow 50.8 from an under-reported ICSG refining series
# → the smelting chain collapsed). Where the outflow prior falls below
# UNCLOSED_K × the inflow prior, its covered edges get σ = UNCLOSED_CV·|f| —
# data kept, false authority removed, mirroring the outlier screen — and
# scripts/constraints.py loosens the matching usetotout anchor to match.
# cv 1.0 (not 0.15): flipping the solver's preference needs
# cv > ANCHOR_CV·c_inflow/c_outflow, which is 0.52 at Iran 1996.
# MEASURED AND REJECTED as a remedy (ADR 0019): loosening σ repriced the
# contradiction without resolving it — z² fell to ~baseline while only 0.46 %
# of the model's mass moved and the collapses were untouched, because the
# binding limit is missing downstream capacity, not σ. The DETECTION ships
# (the report CSV localises where ICSG's own series disagree); set
# UNCLOSED_CV to a value to re-enable the inflation.
# ADR 0021 widens the *report* to Anode: moving the refined-production anchor
# from Matte→CFR to Anode→ER removes the Matte contradiction at source, and the
# question that replaces it is whether it reappears one node downstream. The
# remedy stays disabled, so this is diagnostics only.
UNCLOSED_SECTORS = ("Matte", "Anode")
UNCLOSED_CV      = None

# Size-damped σ on bilateral legs (ADR 0017; applied BEFORE the screen above).
# The Laner pedigree cv (~0.21 on BACI Mass legs) is size-blind, so deleting
# any leg entirely costs z = 1/cv ≈ 4.8 flat — per tonne the biggest legs are
# thousands of times cheaper to delete, and the QP eats them first (deleted
# share 7%→45% by size decile; Chile→China/Japan zeroed). Power-law damping,
#     σ_e = cv_pedigree(e) · a_ref(sector)^(1−γ) · a_e^γ ,
# anchored at each sector's median covered bilateral leg (a_ref, pooled over
# years): at the median σ is unchanged, bigger legs get relatively tighter,
# smaller looser (cv capped at LEG_SIGMA_CV_CAP). γ=1.0 or None disables.
LEG_SIGMA_GAMMA  = 0.5
LEG_SIGMA_CV_CAP = 1.0

# ---------------------------------------------------------------------------
# FormulaRelations (module-level — no runtime state, safe to define here)
# ---------------------------------------------------------------------------

# concentrate_domestic / matte_domestic / anode_domestic RETIRED (ADR 0010 §C).
# The deepened NetworkImputer now closes these domestic chain legs structurally
# (CC flow-balance → Mass twin), and more correctly than the hand-written
# formulas did: their export term selected entity_to=="Flow" and so missed the
# bilateral Anode(Flow)→ER(Process) trade legs, over-attributing the domestic
# leg. Verified on the reference feeds — once the recipe gate opens these three
# relations fired 0× (the network's network_mass_twin fills the domestic Mass
# legs first). The cathode FD-split trio below is NOT subsumable (the Cathode
# node has three same-region FD unknowns, beyond the lone-gap flow-balance) and
# is retained.

def _cathode_supply_term():
    return SourceTerm(
        selector=lambda ei: (
            (ei["sector_from"] == "CP")      & (ei["entity_from"] == "Process") &
            (ei["sector_to"]   == "Cathode") & (ei["entity_to"]   == "Flow") &
            (ei["valuation"]   == "CC")
        ),
        coefficient=ParamRef("conccathodesupply", "inverse"),
        region_binding="to_matches_target_from",
        aggregate="single",
    )

def _cathode_export_term(missing: str = "block"):
    return SourceTerm(
        selector=lambda ei: (
            (ei["sector_from"] == "Cathode") & (ei["entity_from"] == "Flow") &
            (ei["entity_to"]   == "Flow")    & (ei["valuation"]   == "Mass")
        ),
        coefficient=-1.0,
        region_binding="from_matches_target_from",
        aggregate="sum",
        missing=missing,
    )

def _cathode_domestic_subtract(sector_to: str, missing: str = "block"):
    return SourceTerm(
        selector=lambda ei, s=sector_to: (
            (ei["sector_from"] == "Cathode") & (ei["entity_from"] == "Flow") &
            (ei["sector_to"]   == s)         & (ei["entity_to"]   == "SB") &
            (ei["region_from"] == ei["region_to"]) & (ei["valuation"] == "Mass")
        ),
        coefficient=-1.0,
        region_binding="from_matches_target_from",
        aggregate="single",
        missing=missing,
    )

# FU uses missing="zero" on the subtract/export terms: absent USC/MESC or
# unobserved export legs count as 0 (CONTEXT.md "missing counts as 0"), so the
# residual fires wherever CP→Cathode production is known — this is what anchors
# the FU sink with a real prior. The supply term keeps missing="block": without
# known production a residual FU prior would be meaningless. USC/MESC below
# deliberately keep full "block" semantics so they stay reserved for the
# ShareImputer's zero-mean throughput-scaled seed (Part B) instead of being
# preempted by a near-zero formula residual.
cathode_fu_domestic = FormulaRelation(
    description="Domestic Cathode→FU: CP supply/conc - exports - USC - MESC, clip 0",
    target=lambda ei: (
        (ei["sector_from"] == "Cathode") & (ei["entity_from"] == "Flow") &
        (ei["sector_to"]   == "FU")      & (ei["entity_to"]   == "SB") &
        (ei["region_from"] == ei["region_to"]) & (ei["valuation"] == "Mass")
    ),
    terms=[
        _cathode_supply_term(),
        _cathode_export_term(missing="zero"),
        _cathode_domestic_subtract("USC", missing="zero"),
        _cathode_domestic_subtract("MESC", missing="zero"),
    ],
    clip_lower=0.0,
    epsilon=0.05,
)

cathode_usc_domestic = FormulaRelation(
    description="Domestic Cathode→USC: CP supply/conc - exports - FU - MESC, clip 0",
    target=lambda ei: (
        (ei["sector_from"] == "Cathode") & (ei["entity_from"] == "Flow") &
        (ei["sector_to"]   == "USC")     & (ei["entity_to"]   == "SB") &
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
        (ei["sector_to"]   == "MESC")    & (ei["entity_to"]   == "SB") &
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


def _damp_leg_sigma(acc, agg_result, gamma, cv_cap, logger):
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
             & ei["valuation"].eq("Mass").to_numpy()
             & (np.abs(np.nan_to_num(f[eid])) > 0))
    legs = ei[bilat]
    n_damped = 0
    for sector, grp in legs.groupby("sector_from"):
        ids = grp["edge_id"].to_numpy()
        a = np.abs(f[ids])
        a_ref = float(np.median(a))
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


def _screen_outlier_trade_legs(acc, agg_result, conc_csv: str, logger):
    """σ-inflate implausible bilateral Mass legs (see OUTLIER_LEG_K above).

    Per (flow sector, exporter region, year): implied CC exports =
    Σ covered bilateral Flow→* Mass legs · conc_nominal(sector); reference =
    Σ covered Process→Flow CC edges into that exporter's flow node (ICSG
    production) + implied CC imports (Σ covered bilateral Mass legs into the
    region · conc_nominal), so re-exporters are judged against what they
    actually receive. Groups without covered production stay unjudged (the
    transit-hub population; see the config comment above).
    Ratio > OUTLIER_LEG_K ⇒ every covered bilateral Mass leg
    of that group gets sigma_agg = OUTLIER_LEG_CV·|f_agg|. Returns the report
    DataFrame (flagged groups only; empty frame if none).
    """
    import pandas as pd

    ei = expand_edge_index(acc.edge_index, acc.node_index)
    cov = np.asarray(agg_result.coverage, dtype=bool)
    f = np.nan_to_num(np.asarray(agg_result.f_agg, dtype=float))

    conc_p = pd.read_csv(conc_csv, sep=None, engine="python", comment="#", dtype=str)
    conc_p = conc_p[conc_p["param_type"] == "concentration"]
    conc_by_sector = {
        r["a_sector_from"]: float(r["value_nominal"])
        for _, r in conc_p.iterrows()
        if r["a_sector_from"] not in ("*", "", None) and r["a_valuation"] == "Mass"
    }

    eid = ei["edge_id"].to_numpy()
    bilat = (cov[eid]
             & ei["entity_from"].eq("Flow").to_numpy()
             & (ei["region_from"] != ei["region_to"]).to_numpy()
             & ei["valuation"].eq("Mass").to_numpy()
             & ei["sector_from"].isin(conc_by_sector).to_numpy())
    prodm = (cov[eid]
             & ei["entity_from"].eq("Process").to_numpy()
             & ei["entity_to"].eq("Flow").to_numpy()
             & ei["valuation"].eq("CC").to_numpy())

    legs = ei[bilat].copy()
    legs["cc"] = f[legs["edge_id"].to_numpy()] \
        * legs["sector_from"].map(conc_by_sector).to_numpy()
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
    flagged = both[both["ratio"] > OUTLIER_LEG_K].reset_index()

    n_edges = 0
    for _, g in flagged.iterrows():
        m = bilat \
            & ei["sector_from"].eq(g["sector_from"]).to_numpy() \
            & ei["region_from"].eq(g["region_from"]).to_numpy() \
            & ei["time"].eq(g["time"]).to_numpy()
        ids = ei["edge_id"].to_numpy()[m]
        ids = ids[np.abs(f[ids]) > 0]      # a genuine 0 keeps its σ (no 0-pin)
        agg_result.sigma_agg[ids] = OUTLIER_LEG_CV * np.abs(agg_result.f_agg[ids])
        n_edges += len(ids)

    if len(flagged):
        logger.warning(
            "B  outlier-leg screen: %d exporter-sector-years flagged "
            "(implied CC exports > %.1f× covered production+imports) — σ "
            "inflated to cv=%.1f on %d bilateral Mass legs\n%s",
            len(flagged), OUTLIER_LEG_K, OUTLIER_LEG_CV, n_edges,
            flagged.sort_values("ratio", ascending=False).head(10).to_string(index=False))
    else:
        logger.info("B  outlier-leg screen: no exporter-sector-year exceeds "
                    "%.1f× covered production+imports — nothing inflated",
                    OUTLIER_LEG_K)
    return flagged


def _screen_contradicted_nodes(acc, agg_result, logger):
    """σ-inflate the outflow edges of contradicted Flow nodes (ADR 0019).

    Detection lives in ``pysuto.flow_node_balance`` so Stage 3 applies the
    identical rule to the scale-total anchors. Returns the report DataFrame
    (failing nodes only; empty frame if none).

    Runs **after** imputation: the inflow side of a Flow node (Smelting→Matte)
    is never fed, so pre-cascade its prior is 0 and no node looks contradicted.
    """
    keys, report = contradicted_flow_nodes(
        acc, agg_result, sectors=UNCLOSED_SECTORS)
    if not keys:
        logger.info("B  contradicted-node screen: every %s node's outflow "
                    "prior honours its inflow — nothing inflated",
                    "/".join(UNCLOSED_SECTORS))
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
    if UNCLOSED_CV is not None:
        agg_result.sigma_agg[ids] = UNCLOSED_CV * np.abs(agg_result.f_agg[ids])
    logger.warning(
        "B  contradicted-node screen: %d %s node-years whose outflow prior is "
        "< %.2f× their inflow prior (%d outflow edges) — σ action: %s\n%s",
        len(keys), "/".join(UNCLOSED_SECTORS), UNCLOSED_K, len(ids),
        f"cv={UNCLOSED_CV}" if UNCLOSED_CV is not None
        else "none (ADR 0019 — detector only)",
        report.head(10).to_string(index=False))
    return report


def main(feed_ids: list[str], label: str) -> None:
    log_path = os.path.join(
        "logs",
        f"collect_aggregate_impute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    os.makedirs("logs", exist_ok=True)
    add_run_handler(log_path)
    logger = get_logger("collect_aggregate_impute")
    logger.info("=== collect_aggregate_impute START ===")

    feed_keys = list(feed_ids)

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

    # Size-damped leg σ (ADR 0017), then the plausibility screen — the screen's
    # cv-5 inflation overrides damping on flagged legs.
    _damp_leg_sigma(acc, agg_result, LEG_SIGMA_GAMMA, LEG_SIGMA_CV_CAP, logger)
    outlier_report = _screen_outlier_trade_legs(
        acc, agg_result, str(inputs.params.resolve("conc") / "params.csv"), logger)

    # Snapshot post-aggregation distributions BEFORE imputation mutates them.
    # Distribution panels focus on 1995 (agg_root_by_year stays all-year).
    agg_diag = AggDiagnostic(agg_result, acc_struct=acc, label=label, year="1995")
    agg_diag.summary()

    # ── C  Imputation pipeline ────────────────────────────────────────────────
    # The conc/eff/formula imputers form a metallurgical cascade: a value
    # derived at one stage (e.g. Smelting→Matte CC via eff) only feeds the next
    # stage's fill (Matte→CFR Mass via conc/formula) on a *subsequent* pass.
    # They are registered as a fixpoint cascade — re-run together until a full
    # sweep fills no new edges — so the chain saturates instead of dying after
    # one hop (see docs/rep/analysis_imp_…_conc-after-eff-mass-gap.md).
    # TimeImputer/FlatPrior run once afterwards (terminal).
    logger.info("C  Building imputation engine")
    engine = ImputationEngine(acc_struct=acc)
    param_refs: list[dict] = []

    def _build_parametric(params_id: str, imp_id: str) -> ParametricImputer:
        csv_path = inputs.params.resolve(params_id) / "params.csv"
        imp = ParametricImputer.from_csv(
            str(csv_path), acc, id=imp_id, p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG,
        )
        param_refs.append(inputs.params.entry(params_id))
        logger.info("C  built ParametricImputer id=%s (params=%s v%d)",
                    imp_id, params_id, inputs.params.entry(params_id)["version"])
        return imp

    # Cascade group (iterated to fixpoint): concentration ratios → ore-grade
    # ratios → process efficiencies → formula-driven domestic-use fills (the
    # FormulaImputer shares the conc params CSV).
    conc_formula = FormulaImputer(
        relations=[
            # chain-3 (concentrate/matte/anode_domestic) retired — see note above;
            # the NetworkImputer now closes these legs. Cathode FD split retained.
            cathode_fu_domestic,
            cathode_usc_domestic,
            cathode_mesc_domestic,
        ],
        params_csv=str(inputs.params.resolve("conc") / "params.csv"),
        id="conc_formula",
        p=P_PROP,
    )
    # TradeImputer runs FIRST: it seeds cross-region Flow→Process import CC legs
    # from the BACI Mass twin (f_CC = conc·f_Mass), so the NetworkImputer recipe
    # gate can pool them and the flow-balance can close the lone domestic leg
    # (ADR 0010). It owns only cross-region edges — the complement of the
    # network's same-region territory.
    trade = TradeImputer(
        conc_csv = str(inputs.params.resolve("conc") / "params.csv"),
        p=P_PROP, id="trade",
    )
    # NetworkImputer is the sole writer of process-incident, same-region chain
    # edges (recipe closure on CC + Mass twins + SEM waste). The eff/conc
    # parametric imputers are demoted to a fallback on whatever the network
    # leaves undetermined; chain-edge provenance shows network_*.
    network = NetworkImputer(
        eff_csv       = str(inputs.params.resolve("eff")       / "params.csv"),
        conc_csv      = str(inputs.params.resolve("conc")      / "params.csv"),
        ore_grade_csv = str(inputs.params.resolve("ore_grade") / "params.csv"),
        p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG, cv_cap=1.5,
        id="network",
    )
    engine.register_cascade(
        [
            trade,
            network,
            _build_parametric("conc",      "conc"),
            _build_parametric("ore_grade", "ore_grade"),
            _build_parametric("eff",       "eff"),
            conc_formula,
        ],
        max_iter=CASCADE_MAX_ITER,
    )

    # 5. Temporal interpolation, no extrapolation (terminal).
    engine.register(TimeImputer(
        window           = TIME_WINDOW,
        method           = TIME_METHOD,
        min_obs          = TIME_MIN_OBS,
        gap_inflation    = TIME_GAP_INFLATION,
        interpolate_only = TIME_INTERPOLATE_ONLY,
        p                = P_PROP,
        id               = "time_interp",
    ))

    # (The scoped crude_ore end-of-series extrapolator that used to sit here
    # is retired — ADR 0023: past the crude_ore feed's last year the ore Mass
    # chain is derived by the NetworkImputer from ICSG production through the
    # carried-forward in-situ grade, so nothing is left for it to fill.)

    # 6.5 Net-stock share imputation. Give Δstock (PSC/USC/MESC) edges informative,
    # cold-smooth priors BEFORE flat_prior would fill them with noisy_zero: PSC by
    # share-transfer (share=PSC/(CSXEW+CER) carried across years, seeding 1995), and
    # MESC/USC by a zero-mean throughput-scaled seed σ=(p95/Z95)·FU. p95 is read from
    # the SAME stock_bounds params the Part-A soft bounds use, so prior and bound agree.
    engine.register(ShareImputer.from_stock_bounds_csv(
        str(inputs.params.resolve("stock_bounds") / "params.csv"),
        scale_abs_floor = 100.0,
        p               = P_PROP,
        id              = "net_stock_share",
    ))
    param_refs.append(inputs.params.entry("stock_bounds"))

    # 7. Final fallback — flat prior.
    # Two modes (see FLAT_MODE config above):
    #   - "noisy_zero":      f=0, σ=FLAT_SIGMA_HIGH (constant); prior weight ≈ 0
    #   - "magnitude_aware": σ = CV(j) · scale(j) via cv_by_mask + 3-level cascade
    if FLAT_MODE == "noisy_zero":
        engine.register(FlatPriorImputer(
            acc_struct = acc,
            mode       = "noisy_zero",
            sigma_high = FLAT_SIGMA_HIGH,
            p          = P_PROP,
            id         = "flat_prior",
        ))
    else:
        # The ROW relation (any edge touching ROW as region_from or region_to)
        # is combined with acc.masks["supply"] / acc.masks["use"] to form two
        # row-specific sub-masks whose (cv_lo, cv_hi) carry the geographic
        # pedigree penalty (CV_GEO4) added in quadrature to the base CV.
        supply_mask = np.asarray(acc.masks["supply"], dtype=bool)
        use_mask    = np.asarray(acc.masks["use"],    dtype=bool)

        ei_exp      = expand_edge_index(acc.edge_index, acc.node_index)
        row_mask    = (
            (ei_exp["region_from"].to_numpy() == "ROW") |
            (ei_exp["region_to"].to_numpy()   == "ROW")
        )

        row_supply  = supply_mask & row_mask
        row_use     = use_mask    & row_mask

        flat_cv_by_mask = {
            "row_supply": (float(np.hypot(FLAT_CV_LO_SUPPLY, CV_GEO4)),
                           float(np.hypot(FLAT_CV_HI_SUPPLY, CV_GEO4))),
            "row_use":    (float(np.hypot(FLAT_CV_LO_USE,    CV_GEO4)),
                           float(np.hypot(FLAT_CV_HI_USE,    CV_GEO4))),
            "supply":     (FLAT_CV_LO_SUPPLY, FLAT_CV_HI_SUPPLY),
            "use":        (FLAT_CV_LO_USE,    FLAT_CV_HI_USE),
        }

        engine.register(FlatPriorImputer(
            acc_struct = acc,
            cv_by_mask = flat_cv_by_mask,
            default_cv = FLAT_DEFAULT_CV,
            p          = P_PROP,
            masks      = {"row_supply": row_supply, "row_use": row_use},
            id         = "flat_prior",
        ))

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
    unclosed_report = _screen_contradicted_nodes(acc, agg_result, logger)

    # Ownership-partition check (ADR 0010): trade owns cross-region edges, the
    # network owns same-region edges. A clean run has zero violations; warn loudly
    # if a guard ever regressed.
    _viol = ownership_violations(agg_result.imputation_log.as_dataframe(), acc.edge_meta)
    if len(_viol):
        logger.warning("C  ownership-partition VIOLATIONS: %d edge(s) filled by an "
                       "unintended owner\n%s", len(_viol), _viol.head(20).to_string(index=False))

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
        label         = label,
        parents       = parents,
        imputers      = engine.imputers,
        stage_summary = {"cascade_passes_run": passes_run},
        stage_config = {
            "p":               P_PROP,
            "q":               Q_REG,
            "alpha_min":       ALPHA_MIN_REG,
            "cascade_max_iter": CASCADE_MAX_ITER,
            "flat_mode":       FLAT_MODE,
            "outlier_leg_k":   OUTLIER_LEG_K,
            "outlier_leg_cv":  OUTLIER_LEG_CV,
            "outlier_ref":     "production+imports",
            "unclosed_sectors": list(UNCLOSED_SECTORS),
            "unclosed_cv":     UNCLOSED_CV,
            "unclosed_k":      UNCLOSED_K,
            "leg_sigma_gamma":  LEG_SIGMA_GAMMA,
            "leg_sigma_cv_cap": LEG_SIGMA_CV_CAP,
            "flat_sigma_high": FLAT_SIGMA_HIGH if FLAT_MODE == "noisy_zero" else None,
            "flat_default_cv": FLAT_DEFAULT_CV if FLAT_MODE == "magnitude_aware" else None,
            "time": {
                "window":           TIME_WINDOW,
                "min_obs":          TIME_MIN_OBS,
                "gap_inflation":    TIME_GAP_INFLATION,
                "method":           TIME_METHOD,
                "interpolate_only": TIME_INTERPOLATE_ONLY,
            },
        },
    )
    run_dir = os.path.join(IMP_DIR, run_id)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feed collection, aggregation, and imputation.")
    parser.add_argument(
        "--feed-id", action="append", required=True, dest="feed_ids",
        help="Feed run ID to include. Repeat for each feed.",
    )
    parser.add_argument("--label", help="Human-readable label stored in the run's sidecar")
    args = parser.parse_args()
    main(args.feed_ids, args.label)
