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
7b Scrap-share bands          — RatioBoundBuilder (scrap_share/params.csv, per
                                 SCRAP_SHARE_MODE); per (region, P∈{Smelting,CFR})
                                 hard band bl·Σ(CC inputs to P) ≤ Σ(Scrap→P) ≤
                                 bu·Σ(CC inputs to P), agg=true.
8  Hard flow bounds           — FlowBoundsBuilder (from acc_struct.bounds)

═══════════════════════════════════════════════════════════
USER CONFIGURATION  ←  only section you ever edit
═══════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

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
from pysuto.flow_node_balance import contradicted_flow_nodes
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
# Provenance-aware scale totals: a scale total preserves the model's own prior
# group-sum ("not to add information", CONTEXT.md §Total-output). Where that
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
# Cold sensitivity run (branch sensitivity/no-flat-prior-zeros): hard-pin every
# genuine no-data ("flat-prior") leaf edge to [0,0] in the flow bounds — "no feed
# data ⇒ no flow". NOTE: the pin's l/u boxes only constrain the feasibility
# pre-check probe, NOT the QP (FlowBoundsBuilder is a no-op for G) — the soft
# NODATA_ANCHOR_CV above is what the solver actually feels. The pin stays on to
# keep the pre-check strict and document intent. The pin spares structurally-
# determined edges (extraction chain + all four SEM residual outputs
# {WR,Tailings,Slag,PSC}); see ADR 0013.
PIN_FLAT_PRIOR_ZEROS = True
# Soft-ratio σ_c scaling (REP analysis_opt_20260611_075036 §8.2): the CSV ratio
# σ_r is rescaled to flow space per row, σ_c = max(σ_r, REL_FLOOR) ·
# max(Σ|a₀,num|, |r|·Σ|a₀,denom|, ABS_FLOOR). REL_FLOOR caps |z| at 1/floor on
# a 100 % ratio violation; ABS_FLOOR (tonnage) keeps both-zero-prior rows finite
# AND soft: ~77 % of soft ratio rows are zero-prior parametric ties, and at
# ABS_FLOOR=1 their σ_c=0.02 t made the Hessian so stiff Clarabel stalled
# (InsufficientProgress @2005). 100 t → σ_c=2 t on those rows (z=50 per 100 t).
RATIO_SIGMA_REL_FLOOR = 0.02
RATIO_SCALE_ABS_FLOOR = 100.0
# Net-stock ("stock_bound") mode. "soft" routes the PSC/USC/MESC net-stock-change
# bounds to the free-cone soft-ratio path (minimized quadratic slack, σ_c scaled
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
        nodata_anchor_exclude_sectors = NODATA_ANCHOR_EXCLUDE,
        pin_flat_prior_zeros = PIN_FLAT_PRIOR_ZEROS,
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

    # Provenance-aware σ_c: loosen scale totals whose prior group-sum is
    # fabricated (feed-backed share of Σ|f_agg| < FEED_SHARE_OBSERVED). Uses
    # agg_result.coverage — the pre-imputation "did any feed observe this
    # edge?" flag (CONTEXT.md §Coverage), so imputations never count.
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
        return scale_total_cv(feed_share, abs(float(_f_agg_signed[ids].sum())))

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
                          scale_abs_floor=RATIO_SCALE_ABS_FLOOR)
    else:
        _register_builder(RatioBoundBuilder, "stock_bounds", kind="stock_bounds")
    # 7b. Scrap-share bands — hard (nonneg cone) by default; see SCRAP_SHARE_MODE
    if SCRAP_SHARE_MODE == "hard":
        _register_builder(RatioBoundBuilder, "scrap_share", kind="scrap_share")
    elif SCRAP_SHARE_MODE == "soft":
        _register_builder(RatioConstraintBuilder, "scrap_share",
                          sigma_rel_floor=RATIO_SIGMA_REL_FLOOR,
                          scale_abs_floor=RATIO_SCALE_ABS_FLOOR)

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
            "nodata_anchor_exclude": list(NODATA_ANCHOR_EXCLUDE),
            "pin_flat_prior_zeros": PIN_FLAT_PRIOR_ZEROS,
            "ratio_sigma_rel_floor": RATIO_SIGMA_REL_FLOOR,
            "ratio_scale_abs_floor": RATIO_SCALE_ABS_FLOOR,
            "ore_grade_basis": "in-situ (ADR 0015)",
            "fabricated_anchor_cv": FABRICATED_ANCHOR_CV,
            "feed_share_observed":  FEED_SHARE_OBSERVED,
            "stock_bounds_mode": STOCK_BOUNDS_MODE,
            "scrap_share_mode":  SCRAP_SHARE_MODE,
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
