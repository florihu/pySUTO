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
2b Soft sum — scrap LEVEL      — SumConstraintBuilder.from_mask_anchor, two
   ANCHOR                        families, CC only, COVERED edges only, grouped
                              by (sector_to, region_to, time, valuation):
                                • smelter scrap      Σ(Scrap→Smelting) = the
                                  level ICSG reports in Smelting/Scrap/Use;
                                • direct-to-refinery Σ(Scrap→CFR) = the level
                                  scrap                ICSG reports in
                                                       Refining/Scrap/Use.
                              `Scrap` is SB, so these edges are in NEITHER
                              family above and had no scale row at all; the
                              scrap_share band (7b) states only a *share*, which
                              an idle node satisfies trivially. Registered
                              SEPARATELY so each route is anchored at its own
                              level, not just their sum. A region-year ICSG does
                              not report gets no row and keeps its freedom.
                              SCRAP_LEVEL_ANCHOR, ADR 0031.
3  Soft ratio — concentration  — RatioConstraintBuilder (conc/params.csv)
3b Soft ratio — ore grade      — RatioConstraintBuilder (ore_grade/params.csv,
                                 region×year in-situ CO/Ore grades; owns the
                                 CO/Ore Mass↔CC coupling — ADR 0015)
4  Soft ratio — efficiency     — RatioConstraintBuilder (copper_efficiency_params.csv)
5  Hard ratio bounds — conc    — RatioBoundBuilder (conc/params.csv)
5b Hard ratio bounds — ore grade — RatioBoundBuilder (ore_grade/params.csv,
                                 per-region series band)
6  Hard ratio bounds — eff     — RatioBoundBuilder (copper_efficiency_params.csv)
7  Net stock bounds           — RatioConstraintBuilder or RatioBoundBuilder
                                 (stock_bounds/params.csv, per STOCK_BOUNDS_MODE);
                                 each row carries `agg=true` so the builder
                                 sums every matched a-side edge into one
                                 G row and every matched b-side edge into
                                 another, per (region, time, valuation).
                                 PSC only since Round X: MESC/USC legs have no
                                 row family — they are held at their prior by
                                 the Stage-2 net-stock seed σ (1 % / 0.3 kt,
                                 ADR 0040), and a zero-centred row pulled the
                                 reported stock changes toward zero.
7b Scrap-share bands          — RatioBoundBuilder (scrap_share/params.csv, per
                                 SCRAP_SHARE_MODE); per (region, P∈{Smelting,CFR})
                                 hard band bl·Σ(CC inputs to P) ≤ Σ(Scrap→P) ≤
                                 bu·Σ(CC inputs to P), agg=true.
8  Hard lower bounds          — FlowBoundsBuilder (structural l = 0 → −a ≤ 0, ADR 0039)

═══════════════════════════════════════════════════════════
USER CONFIGURATION  ←  only section you ever edit
═══════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

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
from pysuto.registry import ImputationRunRegistry, stage_lock
from pysuto.input_registry import InputRegistries
from pysuto.param_relation import expand_edge_index
from pysuto.flow_node_balance import contradicted_flow_nodes
from pysuto.constraints.engine import nodata_exempt_mask
from pysuto.constraints import (
    ConstraintEngine,
    MassBalanceBuilder,
    SumConstraintBuilder,
    RatioConstraintBuilder,
    RatioBoundBuilder,
    FlowBoundsBuilder,
)
from pysuto.logging import get_logger, add_run_handler
from pysuto.knobs import add_knob_args, overrides_from_cli, write_run_id

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
# Provenance-aware scale totals: a scale total preserves the model's own prior
# group-sum ("not to add information", GLOSSARY.md §Total-output). Where that
# prior is FABRICATED (twin/recipe/flat imputations, no feed observation behind
# the group) a tight ANCHOR_CV over-preserves it — it fights the observed
# valuation of the same sectors and inverts the data hierarchy (REP
# analysis_opt_20260903_084528_45fdd041_masscc-tension: 3.0M z² on fabricated
# totals; observed totals moved MORE than fabricated ones). Groups whose
# feed-backed share of Σ|f_agg| (agg_result.coverage — feed observations only)
# falls below FEED_SHARE_OBSERVED keep their row but at the loose
# FABRICATED_ANCHOR_CV; feed-backed groups keep ANCHOR_CV.
# NEGATIVE RESULT (Round A, 2026-09-03): loosening was tried at 0.15 and 0.05
# and both collapsed the world chain (CO Mass retention 0.66→0.41/0.45, observed
# China cathode dragged to z≈−20/−16) — the fabricated totals do DOUBLE DUTY:
# besides over-preserving fabricated priors they are the counterweight to the
# NODATA_ANCHOR zero pulls (ADR 0013) on the same interior subgraphs, and
# loosening them hands the interior to the zero anchors. Default therefore
# keeps FABRICATED_ANCHOR_CV = ANCHOR_CV (policy inert); the resolver mechanism
# stays for a future σ design that decouples the two roles (e.g. propagated
# edge σ). See ADR 0016 and the Round-A REP.
FABRICATED_ANCHOR_CV = 0.01
FEED_SHARE_OBSERVED  = 0.5

# Contradicted flow nodes (ADR 0019). A Flow node's hard mass balance cannot
# honour two disagreeing scale-total anchors; since σ_c scales with |c| the
# solver collapses the node to the smaller prior even when that side is the
# less trustworthy one. Where pysuto.flow_node_balance finds the outflow prior
# below K_CLOSE × the inflow prior, that node's usetotout row is loosened to
# UNCLOSED_ANCHOR_CV so the hard balance can settle at the reliable inflow
# instead. This is the first live use of the ADR-0016 resolver, on a DISJOINT
# population: these groups are feed-backed and covered, so ADR 0013's no-data
# zero anchors — the counterweight whose loss made ADR 0016 reject loosening
# fabricated groups — do not act on them. FABRICATED_ANCHOR_CV stays inert.
# cv 1.0 (not 0.15): the flip needs cv > ANCHOR_CV·c_inflow/c_outflow = 0.52
# at Iran 1996.
# MEASURED AND REJECTED as a remedy (ADR 0019) — see the Stage-2 note. The
# detection is still run and logged; set a cv to re-enable the loosening.
UNCLOSED_SECTORS     = ("Matte",)
UNCLOSED_ANCHOR_CV   = None
SIGMA_C_ABS_FLOOR = 1.0e-3
# ADR 0025: a FEED-BACKED scale total cannot be known better than the feeds
# report it. ICSG's raw series are published at 0.1–1 kt (Mining 51 % at
# 0.1 kt, 20 % at integer kt; crude_ore 0.1–1 kt); integer-kt rounding alone
# carries σ = 0.5/√3 = 0.29 kt per reported value. With σ_c = ANCHOR_CV·|c| a
# 1 kt anchor was priced at ≈ 0.017 kt, so a 1 kt miss cost z ≈ −56 (Portugal
# Cathode, the largest cell on the Round-J run, has a 0.3–1.7 kt anchor).
# Applied through the ADR-0016 resolver, in the direction that ADR asked for:
# feed-backed groups (feed share ≥ FEED_SHARE_OBSERVED) get
# σ_c = max(ANCHOR_CV·|c|, SCALE_TOTAL_ABS_FLOOR); FABRICATED groups keep the
# tight cv — their tightness is the counterweight to the nodata zero anchors,
# and a global abs floor (Round K, first variant) let small fabricated chains
# grow (small-node posterior/prior p90 1.06 → 2.33, Chile concentrate export
# retention −8 pp). See ADR 0025 and the Round-K REP.
SCALE_TOTAL_ABS_FLOOR = 0.3


def scale_total_cv(feed_share: float, c_abs: float,
                   anchor_cv: float = ANCHOR_CV,
                   fabricated_cv: float = FABRICATED_ANCHOR_CV,
                   feed_share_observed: float = FEED_SHARE_OBSERVED,
                   abs_floor: float = SCALE_TOTAL_ABS_FLOOR) -> float | None:
    """cv for one scale-total group (ADR 0016 classification, ADR 0025 floor).

    Fabricated groups (feed share below the threshold) keep ``fabricated_cv``.
    Feed-backed groups get the cv that makes ``cv·|c|`` equal the feeds'
    reporting-resolution floor on small totals, ``max(anchor_cv, floor/|c|)``,
    so σ_c = max(anchor_cv·|c|, floor). ``None`` means "use the builder's
    default cv" (a feed-backed group with |c| = 0 cannot be floored via cv).
    """
    if feed_share < feed_share_observed:
        return fabricated_cv
    if c_abs <= 0.0:
        return None
    return max(anchor_cv, abs_floor / c_abs)
# Prior box ("tight fitting"): box flow bounds to a₀ ± PRIOR_BOX_K·σ_a on edges
# with a real prior, grafting RAS's scale-floor onto the QP to stop the
# homogeneous collapse to l. None = off (legacy behaviour). See
# docs/rep/analysis_opt_20260602_133908_46bc514e_20260603_feed-scale-collapse.md §8.3.
PRIOR_BOX_K       = None
# No-data ("flat-prior") σ_a magnitude anchor: edges with a₀=0 whose σ fell back
# to the global-median (≈1e6) get σ = NODATA_ANCHOR_CV · median|a₀| of anchored
# siblings in the same (from_sector, valuation), so the zero prior pulls and they
# stop absorbing mass routed through data-free subgraphs (e.g. scrap on Taiwan/
# Panama CFR). 0.05 ⇒ z=20 per median-sibling-flow deviation: tight, but soft —
# a hard mass balance can still force genuine flow at visible penalty cost.
# This is the EFFECTIVE QP-side mechanism: the flat-prior-zero pin below only
# reaches the feasibility pre-check (ConstraintRun.l/u are never handed to
# Clarabel), so the anchor must not be disabled in its favour. Exclusion list
# is aligned with the pin's (extraction chain + all four SEM residual outputs);
# see NODATA_ANCHOR_EXCLUDE and ADR 0013.
NODATA_ANCHOR_CV  = 0.05
NODATA_ANCHOR_EXCLUDE = ("CO", "Mining", "Ore", "WR", "Tailings", "Slag", "PSC")
# Data-aware exemption (ADR 0013 amendment 3, Round W.3). The extraction chain is
# spared by the anchor only where a relation sets its magnitude: region-years with
# an ore-grade row (ADR 0015). In countries that mine nothing its noisy zeros get
# the ordinary sibling-median anchor, so phantom extraction costs z instead of
# being free. Slag and PSC keep the sector rule. () restores the sector-only rule.
NODATA_ANCHOR_DATA_AWARE = ("CO", "Mining", "Ore", "WR", "Tailings")
# Cold sensitivity run (branch sensitivity/no-flat-prior-zeros): hard-pin every
# genuine no-data ("flat-prior") leaf edge to [0,0] in the flow bounds — "no feed
# data ⇒ no flow". NOTE: the pin's l/u boxes only constrain the feasibility
# pre-check probe, NOT the QP (FlowBoundsBuilder enforces only the structural
# lower bounds, ADR 0039) — the soft NODATA_ANCHOR_CV above is what the solver
# actually feels. The pin stays on to
# keep the pre-check strict and document intent. The pin spares structurally-
# determined edges (extraction chain + all four SEM residual outputs
# {WR,Tailings,Slag,PSC}); see ADR 0013.
PIN_FLAT_PRIOR_ZEROS = True
# Soft-ratio σ_c scaling (REP analysis_opt_20260611_075036 §8.2): the CSV σ (ratio
# units) is made relative, σ_r = σ/|r| (#29), and rescaled to flow space per row,
# σ_c = max(σ_r, REL_FLOOR) · max(Σ|a₀,num|, |r|·Σ|a₀,denom|, ABS_FLOOR). REL_FLOOR caps |z| at 1/floor on
# a 100 % ratio violation; ABS_FLOOR (in kt — the flow unit) keeps both-zero-prior
# rows finite AND soft: ~77 % of soft ratio rows are zero-prior parametric ties,
# and at ABS_FLOOR=1 their σ_c=0.02 kt made the Hessian so stiff Clarabel stalled
# (InsufficientProgress @2005). 100 kt → σ_c=2 kt on those rows (z=50 per 100 kt).
# (The REP and an earlier version of this comment wrote "t"; the value was always
# applied in kt.)
RATIO_SIGMA_REL_FLOOR = 0.02
RATIO_SCALE_ABS_FLOOR = 100.0
# Net-stock rows use their own scale floor. Their σ_c scales off the throughput
# denominator alone (ratio.py scale_mode="denominator"), so the floor is what a
# region with no throughput gets: at the generic 100 kt it was σ_r·100 kt on every
# such row (Round X: 553 kt on the MESC "hub" rows, a free sink). 0.1 kt is the
# yearbook's display resolution; the row stays finite and soft (ADR 0040).
STOCK_BOUNDS_SCALE_ABS_FLOOR = 0.1
# Net-stock ("stock_bound") mode. "soft" routes the PSC net-stock-change bounds
# (stock_bounds v5 carries PSC rows only; MESC/USC are held by their prior σ,
# ADR 0040) to the free-cone soft-ratio path (minimized quadratic slack, σ_c scaled
# off the throughput denominator — see RatioConstraintBuilder scale_mode); "hard"
# keeps the legacy nonneg-cone RatioBoundBuilder. Soft guarantees feasibility (no
# stock row can break) and secures too-high AND too-low symmetrically. The hard
# bounds went infeasible under the flat-prior-zero pin (denominator collapse);
# see docs/rep/analysis_const_20260625_130524_...flat-prior-zero-pin-cold.md.
STOCK_BOUNDS_MODE = "soft"
# Scrap-share bands: per (region, process P∈{Smelting,CFR}) hard nonneg-cone band
# bl·Σ(CC inputs to P) ≤ Σ(Scrap→P, CC) ≤ bu·Σ(CC inputs to P), bands derived
# per-region from observed ICSG scrap shares (scripts/explo/scrap_share_study.py).
# Guards the secondary/primary input mix: without it, no-data Scrap→Process edges
# (in neither scale-total sum family) acted as free copper faucets and inverted
# the primary share (see REP analysis_opt_20260901_080337_99aa122f). "hard" →
# RatioBoundBuilder; "soft" → RatioConstraintBuilder fallback (if the hard band
# ever conflicts with eff/grade bands at the feasibility check); "off" disables.
# Deliberately hard-only (no soft centre row): a soft pull toward median·inputs
# would fight the NODATA_ANCHOR_CV zero anchor on no-data regions.
SCRAP_SHARE_MODE = "hard"
# Scrap LEVEL anchors (Round P, ADR 0031). The band above states a *share*; a
# share band over a node with no throughput is satisfied trivially, so the level
# ICSG reports was never stated anywhere in Stage 3 (the faucet note above is the
# same gap seen from the other side). False restores the Round O configuration
# for an A/B. See section 2b of main().
SCRAP_LEVEL_ANCHOR = True
# Mass-balance scope (ADR 0038). The acc-local mask `massbal_scope` (masks.csv)
# excludes Mining→WR, so waste rock is a satellite mass flow outside the Mining
# balance — the gap between crude ore and ore can no longer be booked on it.
# None balances every edge (the pre-ADR-0038 configuration, for an A/B).
MASSBAL_MASK = "massbal_scope"
# Enforced lower bounds (ADR 0039, Round W.1). Every edge whose structural lower
# bound is 0 gets a hard row −a ≤ 0: non-negativity was part of the documented
# formulation but never reached the QP (bounds lived only in the feasibility
# probe), and residual mass edges went negative. False = pre-ADR-0039 behaviour.
ENFORCE_LOWER_BOUNDS = True
# Stripping ratio as a HARD relation (ADR 0038, Round W.2b). As a soft row the
# waste-rock edge — copperless, outside every balance — is decided by nothing but
# two weak terms, and the interior-point stopping rule never pins it down: W.2a
# (opt_20260929_111551_21703f80) ended at WR/Ore 1.04 instead of 2.0, and still
# 1.03 when 2019 was re-solved at a 1e-6 gap. A zero-cone row enforces
# WR = k·Ore to feasibility tolerance; the imputer leaves WR to the flat prior so
# the relation, not a second copy of the ore prior, sets it. False = soft (W.2a).
STRIPPING_RATIO_HARD = True

# The knobs a pipeline step may override with --set (#100, pysuto.knobs). Paths and
# ACC_ID are not knobs. scale_total_cv's defaults bind at import, so main() passes
# the four knobs it reads explicitly.
KNOBS = (
    "P_PROP", "SIGMA_A_CV_MIN", "SIGMA_A_ABS_FLOOR", "ANCHOR_CV", "SIGMA_C_CV_MIN",
    "FABRICATED_ANCHOR_CV", "FEED_SHARE_OBSERVED", "UNCLOSED_SECTORS",
    "UNCLOSED_ANCHOR_CV", "SIGMA_C_ABS_FLOOR", "SCALE_TOTAL_ABS_FLOOR", "PRIOR_BOX_K",
    "NODATA_ANCHOR_CV", "NODATA_ANCHOR_EXCLUDE", "NODATA_ANCHOR_DATA_AWARE",
    "PIN_FLAT_PRIOR_ZEROS", "RATIO_SIGMA_REL_FLOOR", "RATIO_SCALE_ABS_FLOOR",
    "STOCK_BOUNDS_SCALE_ABS_FLOOR", "STOCK_BOUNDS_MODE", "SCRAP_SHARE_MODE",
    "SCRAP_LEVEL_ANCHOR", "MASSBAL_MASK", "ENFORCE_LOWER_BOUNDS",
    "STRIPPING_RATIO_HARD",
)

_log = get_logger("init_copper_constraints")


def main(imp_run_id: str, label: str, overrides: dict | None = None) -> str:
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

    # ADR 0013 amendment 3: spare the extraction chain only in region-years with an
    # ore-grade row (the copper case's "a relation sets this magnitude").
    anchor_exempt, n_grade_ry = None, 0
    if NODATA_ANCHOR_DATA_AWARE:
        _og = pd.read_csv(inputs.params.resolve("ore_grade") / "params.csv",
                          sep=None, engine="python", dtype=str)
        _co = _og[_og["param_id"].str.startswith("oregradecosupply")]
        _grade_ry = set(zip(_co["a_region_from"], _co["a_time"].astype(str)))
        n_grade_ry = len(_grade_ry)
        anchor_exempt = nodata_exempt_mask(acc, NODATA_ANCHOR_EXCLUDE,
                                           NODATA_ANCHOR_DATA_AWARE, _grade_ry)
        _log.info("No-data anchor: extraction chain spared in %d ore-grade region-years; "
                  "%d edges spared in total", n_grade_ry, int(anchor_exempt.sum()))

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
        nodata_anchor_exclude_sectors = NODATA_ANCHOR_EXCLUDE,
        nodata_anchor_exempt_mask     = anchor_exempt,
        pin_flat_prior_zeros = PIN_FLAT_PRIOR_ZEROS,
    )

    # 1. Hard mass balance
    _log.info("Registering MassBalanceBuilder")
    engine.register(MassBalanceBuilder(mask=MASSBAL_MASK))

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

    # Provenance-aware σ_c: loosen scale totals whose prior group-sum is
    # fabricated (feed-backed share of Σ|f_agg| < FEED_SHARE_OBSERVED). Uses
    # agg_result.coverage — the pre-imputation "did any feed observe this
    # edge?" flag (GLOSSARY.md §Coverage), so imputations never count.
    _cov  = np.asarray(agg_result.coverage, dtype=bool)
    _f_agg_signed = np.nan_to_num(np.asarray(f_agg, dtype=float))
    _absf = np.abs(_f_agg_signed)

    # ADR 0019: the same detection Stage 2 used to σ-inflate the edges.
    _unclosed, _unclosed_rep = contradicted_flow_nodes(
        acc, agg_result, sectors=UNCLOSED_SECTORS)
    _log.info("ADR 0019: %d contradicted %s node-year(s) — anchor action: %s",
              len(_unclosed), "/".join(UNCLOSED_SECTORS),
              f"usetotout cv={UNCLOSED_ANCHOR_CV}"
              if UNCLOSED_ANCHOR_CV is not None else "none (detector only)")

    def _scale_total_cv(key_map, edge_ids):
        # Contradicted node: the outflow anchor loses authority so the hard
        # balance can settle at the reliable inflow prior (ADR 0019).
        if (UNCLOSED_ANCHOR_CV is not None
                and (key_map.get("sector_from"), key_map.get("region_from"),
                     key_map.get("time")) in _unclosed):
            return UNCLOSED_ANCHOR_CV
        ids = np.asarray(edge_ids, dtype=int)
        tot = _absf[ids].sum()
        if tot <= 0.0:
            return FABRICATED_ANCHOR_CV
        feed_share = _absf[ids[_cov[ids]]].sum() / tot
        return scale_total_cv(feed_share, abs(float(_f_agg_signed[ids].sum())),
                              anchor_cv=ANCHOR_CV, fabricated_cv=FABRICATED_ANCHOR_CV,
                              feed_share_observed=FEED_SHARE_OBSERVED,
                              abs_floor=SCALE_TOTAL_ABS_FLOOR)

    _log.info("Registering SumConstraintBuilder — supply/use total-output scale totals")
    engine.register(SumConstraintBuilder.from_mask_anchor(
        acc,
        mask          = supply_mask,
        anchor_values = f_agg,
        group_keys    = GROUP_KEYS,
        sigma_cv      = ANCHOR_CV,
        id_prefix     = "supplytotout",
        sigma_cv_resolver = _scale_total_cv,
    ))
    engine.register(SumConstraintBuilder.from_mask_anchor(
        acc,
        mask          = use_mask,
        anchor_values = f_agg,
        group_keys    = GROUP_KEYS,
        sigma_cv      = ANCHOR_CV,
        id_prefix     = "usetotout",
        sigma_cv_resolver = _scale_total_cv,
    ))

    # 2b. Soft sum — ICSG SCRAP LEVEL anchors (Round P, ADR 0031).
    #     `Scrap` is an SB node, so `Scrap→Process` edges match NEITHER family
    #     above (supply needs entity_from=Process, use needs entity_from=Flow).
    #     ICSG's `Smelting/Scrap/Use` and `Refining/Scrap/Use` therefore entered
    #     Stage 2 as feed-backed edge observations and then had NO scale row at
    #     all: every Stage-3 statement touching them (massbal's zero cone, the
    #     conc CC/Mass ratio, eff, the scrap_share band) is scale-INVARIANT, so a
    #     whole secondary chain can shrink toward zero for the price of one
    #     edge's prior penalty — Slovakia's smelter scrap 46.6 → 0.012 kt/yr and
    #     its anode 43 → 6.5 kt against an unchanged 57 kt anchor (|z| 14 → 52).
    #     The prior does not decay: these runs are `--prior-mode cold`, so a₀ is
    #     re-asserted from f_agg every year and the solver walks away every year
    #     regardless. A per-edge penalty is simply cheaper to pay than to
    #     satisfy; only a scale total prices the GROUP rather than the edge.
    #     ADR 0030 §Measured is the evidence; this closes the level/share gap it
    #     exposed and is what makes Round O mergeable.
    #
    #     TWO registrations, not one. Both routes share sector_from="Scrap", so a
    #     single mask grouped on the scrap side would anchor only their SUM and
    #     leave the solver free to move copper between the smelter and the
    #     refinery route — the exact substitution ADR 0030 constrained. Grouped
    #     on `sector_to`, so row_meta.sector is the CONSUMING node. These are
    #     *scrap level anchors*, NOT input totals: GLOSSARY.md reserves "total
    #     input" for ALL incoming edges, and these sum one input class.
    #
    #     Scoped to CC and to COVERED edges:
    #       • CC only — the Mass twin is network_mass_twin-derived and carries no
    #         independent information (0 of the Mass scrap edges are covered;
    #         CC/Mass is the 0.70 conc nominal on every pair). Anchoring it too
    #         would pin the scrap grade to exactly 0.70 and void ADR 0012's
    #         single wide [0.40, 0.98] band — and it would be *stiffer* than its
    #         CC twin, since fabricated groups keep the tight cv with no ADR 0025
    #         floor. The concscrapsupply ratio already ties the twin.
    #       • covered only — the row asserts a level ICSG REPORTED, and it makes
    #         every group 100 % feed-backed so ADR 0025's 0.3 kt floor applies
    #         unconditionally. A plain |c|>0 mask would add the time_linear
    #         region-years as FABRICATED, un-floored pins (a 0.068 kt anchor
    #         priced at ±0.68 t).
    #     A region-year ICSG does not report gets NO row and stays exactly as
    #     free as before (from_mask_anchor's |c| ≤ min_abs_c skip is
    #     belt-and-braces; the coverage mask already excludes it). This family
    #     can only ever ASSERT a scrap inflow into Smelting or CFR, never forbid
    #     one. Known gap, deliberately left to P.1: where ADR 0030's nesting
    #     clipped the direct route to nil the claim is positive ("no direct
    #     route") but is recorded here as merely unstated.
    #     σ_c: `_scale_total_cv` unchanged. Its ADR 0019 branch is inert here
    #     (UNCLOSED_ANCHOR_CV is None, and it targets Matte usetotout rows).
    _scrap_cc = ((ei_exp["sector_from"].eq("Scrap")
                  & ei_exp["valuation"].eq("CC")).to_numpy() & _cov)
    smelter_scrap_mask   = _scrap_cc & ei_exp["sector_to"].eq("Smelting").to_numpy()
    directref_scrap_mask = _scrap_cc & ei_exp["sector_to"].eq("CFR").to_numpy()
    SCRAP_GROUP_KEYS = ("sector_to", "region_to", "time", "valuation")

    if SCRAP_LEVEL_ANCHOR:
        _log.info("Registering SumConstraintBuilder — ICSG scrap-level anchors "
                  "(smelter scrap %d edges, direct-to-refinery scrap %d edges)",
                  int(smelter_scrap_mask.sum()), int(directref_scrap_mask.sum()))
        for _mask, _prefix in ((smelter_scrap_mask,   "smelterscrap"),
                               (directref_scrap_mask, "directrefscrap")):
            engine.register(SumConstraintBuilder.from_mask_anchor(
                acc,
                mask          = _mask,
                anchor_values = f_agg,
                group_keys    = SCRAP_GROUP_KEYS,
                sigma_cv      = ANCHOR_CV,
                id_prefix     = _prefix,
                sigma_cv_resolver = _scale_total_cv,
            ))
    else:
        _log.info("SCRAP_LEVEL_ANCHOR off — scrap levels unstated (Round O config)")

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
    # 4a. Stripping ratio WR_Mass = k·Ore_Mass (ADR 0038): the only statement about
    #     the waste-rock satellite, hard by default (STRIPPING_RATIO_HARD)
    _register_builder(RatioConstraintBuilder, "stripping_ratio",
                      sigma_rel_floor=RATIO_SIGMA_REL_FLOOR,
                      scale_abs_floor=RATIO_SCALE_ABS_FLOOR,
                      hard=STRIPPING_RATIO_HARD)
    # 3b. Soft ratio — region×year in-situ ore grades (CO/Ore edges; ADR 0015 —
    #     these replace the conc family's retired global CO/Ore rows)
    _register_builder(RatioConstraintBuilder, "ore_grade",
                      sigma_rel_floor=RATIO_SIGMA_REL_FLOOR,
                      scale_abs_floor=RATIO_SCALE_ABS_FLOOR)
    # 5. Hard ratio bounds — concentration
    _register_builder(RatioBoundBuilder, "conc", kind="conc")
    # 5b. Hard ratio bounds — ore grade (per-region series band; ADR 0015)
    _register_builder(RatioBoundBuilder, "ore_grade", kind="conc")
    # 6. Hard ratio bounds — efficiency
    _register_builder(RatioBoundBuilder, "eff", kind="eff")
    # 7. Net stock bounds — soft (free cone, minimized slack) or hard (nonneg cone)
    if STOCK_BOUNDS_MODE == "soft":
        _register_builder(RatioConstraintBuilder, "stock_bounds",
                          sigma_rel_floor=RATIO_SIGMA_REL_FLOOR,
                          scale_abs_floor=STOCK_BOUNDS_SCALE_ABS_FLOOR)
    else:
        _register_builder(RatioBoundBuilder, "stock_bounds", kind="stock_bounds")
    # 7b. Scrap-share bands — hard (nonneg cone) by default; see SCRAP_SHARE_MODE
    if SCRAP_SHARE_MODE == "hard":
        _register_builder(RatioBoundBuilder, "scrap_share", kind="scrap_share")
    elif SCRAP_SHARE_MODE == "soft":
        _register_builder(RatioConstraintBuilder, "scrap_share",
                          sigma_rel_floor=RATIO_SIGMA_REL_FLOOR,
                          scale_abs_floor=RATIO_SCALE_ABS_FLOOR)

    # 8. Hard lower bounds from acc_struct (ADR 0039)
    _log.info("Registering FlowBoundsBuilder (enforce_lower=%s)", ENFORCE_LOWER_BOUNDS)
    engine.register(FlowBoundsBuilder(enforce_lower=ENFORCE_LOWER_BOUNDS))

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
            "nodata_anchor_exclude": list(NODATA_ANCHOR_EXCLUDE),
            "nodata_anchor_data_aware": list(NODATA_ANCHOR_DATA_AWARE),
            "nodata_anchor_grade_region_years": n_grade_ry,
            "pin_flat_prior_zeros": PIN_FLAT_PRIOR_ZEROS,
            "ratio_sigma_rel_floor": RATIO_SIGMA_REL_FLOOR,
            "ratio_scale_abs_floor": RATIO_SCALE_ABS_FLOOR,
            "stock_bounds_scale_abs_floor": STOCK_BOUNDS_SCALE_ABS_FLOOR,
            "ore_grade_basis": "in-situ (ADR 0015)",
            "fabricated_anchor_cv": FABRICATED_ANCHOR_CV,
            "feed_share_observed":  FEED_SHARE_OBSERVED,
            "stock_bounds_mode": STOCK_BOUNDS_MODE,
            "scrap_share_mode":  SCRAP_SHARE_MODE,
            "scrap_level_anchor": SCRAP_LEVEL_ANCHOR,
            "massbal_mask":      MASSBAL_MASK,
            "enforce_lower_bounds": ENFORCE_LOWER_BOUNDS,
            "stripping_ratio_hard": STRIPPING_RATIO_HARD,
            "overrides":         dict(overrides or {}),
        },
    )

    # ── D. Summary ────────────────────────────────────────────────────────
    _log.info("ConstraintRun saved: %s  label=%s", constraint_run.run_id, label)
    _log.info("G shape: %s  (rows × edges)", constraint_run.G.shape)
    _log.info("Constraint type breakdown:\n%s",
              constraint_run.row_meta["type"].value_counts().to_string())
    _log.info("Cone type breakdown:\n%s",
              constraint_run.row_meta["cone_type"].value_counts().to_string())
    return constraint_run.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and persist constraint matrix.")
    parser.add_argument(
        "--imp-run-id", required=True,
        help="ImputationRun ID to load from data/proc/imp/.",
    )
    parser.add_argument("label", help="Label for this constraint run")
    add_knob_args(parser)
    args = parser.parse_args()
    overrides = overrides_from_cli(parser, args, KNOBS, globals())
    with stage_lock("constraints"):   # registries have no locking (#101)
        write_run_id(args.run_id_file, main(args.imp_run_id, args.label, overrides))
