"""
copper.py
=========
The copper case (GLOSSARY.md "Case", #26 §3, #120): the Stage 2 and Stage 3 model of
the copper application as one :class:`pysuto.case.CaseSpec`, ``CASE``.

- :class:`CopperImp` — every Stage 2 knob at its shipped value; :func:`imputers` builds
  the imputer cascade ``scripts/collect_aggregate_impute.py`` registers.
- :class:`CopperConst` — every Stage 3 knob at its shipped value;
  :func:`nodata_anchor_exempt` selects the no-data anchor's exemption and
  :func:`builders` builds the constraint builders ``scripts/constraints.py`` registers.

The knob values, their notes and the construction steps moved here unchanged from the
two drivers' ``# USER CONFIGURATION`` blocks and ``main()``s (#120), and so did the
script-level algorithms (#121): the cascade order check, ``scale_total_cv`` and the
scale totals' σ resolver stay here; leg-σ damping, the outlier and contradicted-node
screens and the time interpolator's scope moved into ``pysuto.impute`` with their
labels as parameters (#144), and this case passes its own. The hooks ``pre_impute``
and ``post_impute`` and the builders reach them through the :class:`CaseContext`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from pysuto.case import Cascade, CaseSpec, ConstSection, ImpSection
from pysuto.constraints import (
    FlowBoundsBuilder,
    MassBalanceBuilder,
    RatioBoundBuilder,
    RatioConstraintBuilder,
    SumConstraintBuilder,
)
from pysuto.constraints.engine import nodata_exempt_mask
from pysuto.impute import (
    FlatPriorImputer,
    NetworkImputer,
    ParametricImputer,
    ShareImputer,
    TimeImputer,
    TradeImputer,
    damp_leg_sigma,
    screen_contradicted_nodes,
    screen_outlier_trade_legs,
    time_interp_scope,
)
from pysuto.flow_node_balance import K_CLOSE as UNCLOSED_K, contradicted_flow_nodes
from pysuto.param_relation import expand_edge_index
from pysuto.unc import PedigreeScores


# =============================================================================
# Stage 2 — collect, aggregate, impute
# =============================================================================

# ROW edges carry the worst geographical pedigree (geographical_correlation=4),
# added in quadrature to the base CV — the same transparent penalty Stage 1
# applies to fed ROW edges (replaces the old ad-hoc FLAT_ROW_INFLATION x1.5).
# Used only when FLAT_MODE == "magnitude_aware"; a pedigree constant, not a knob.
CV_GEO4 = PedigreeScores._cv_other(4, "medium_sensitive")


@dataclass(frozen=True)
class CopperImp(ImpSection):
    """The copper case's Stage 2 section. Its UPPER_CASE fields are the knobs a
    pipeline step may override with ``--set`` (#100); ``cascade_ids`` and
    ``stock_sectors`` are not knobs."""

    # Note: P_PROP must match CopperConst's P_PROP (cross-stage Lp; pysuto.pipeline TIED).

    # Propagation / Lp settings
    P_PROP:        float = 1.3      # Lp/power-mean exponent
    Q_REG:         float = 3.0      # ParametricImputer backward-branch knee sharpness
    ALPHA_MIN_REG: float = 0.05     # inverse-multiplier effective-sensitivity floor

    # Imputation cascade — conc/eff/formula iterate to a fixpoint (re-run until a
    # full sweep fills no new edges). Safety cap on passes (chain depth ≈ 8).
    CASCADE_MAX_ITER: int = 10
    # Cascade order (#104): imputers are write-once, so where two writer scopes overlap
    # the earlier imputer writes the edge (GLOSSARY "Writer scope"). A knob, so a sweep
    # can run named alternative orders; it must stay a permutation of cascade_ids. The
    # terminal pass (time → share → flat) is fixed: flat must be last and only share may
    # write the stock legs (ADR 0040).
    cascade_ids:   tuple = ("trade", "network", "conc", "ore_grade", "eff")
    CASCADE_ORDER: tuple = ("trade", "network", "conc", "ore_grade", "eff")

    # NetworkImputer — waste rock is a satellite mass flow outside the Mining balance
    # (ADR 0038), fixed in Stage 3 by the hard stripping relation WR_Mass = k·Ore_Mass
    # (the `stripping_ratio` params family). The prior carries WR_Mass = k·Ore_Mass at the
    # flat σ FLAT_SIGMA_HIGH, so it satisfies that row without a weight of its own (an
    # informative WR prior would count the Ore prior twice through the hard row). A direct
    # ratio avoids the CO_Mass − Ore_Mass residual going negative when a region's
    # ore_grade exceeds the flat conc_ore.

    # TimeImputer (general interpolation pass)
    TIME_WINDOW:           int   = 3
    TIME_MIN_OBS:          int   = 2
    TIME_GAP_INFLATION:    float = 1.05
    TIME_METHOD:           str   = "linear"
    TIME_INTERPOLATE_ONLY: bool  = True
    # ADR 0026: the interpolator is scoped to same-region edges. BACI is a *complete*
    # bilateral matrix — a route absent in a year is a recorded zero, not a gap — so
    # interpolating a cross-region leg from its neighbouring years invents trade the
    # source never reported (9,215 Mass legs / 20,252 kt on the pre-ADR-0026 chain,
    # of which 15,266 kt survived into the posterior: Papua New Guinea concentrate,
    # Zambia cathode/matte, Peru). ADR 0010 and GLOSSARY "Trade-seed" already state
    # the intent ("a route with no BACI Mass observation is left NaN, an honest
    # flat-prior gap, never seeded to 0"); the unscoped imputer silently overrode it.
    # Set False to restore the pre-ADR-0026 behaviour.
    TIME_SCOPE_SAME_REGION: bool = True

    # Signed stock-change legs (Cathode→MESC / Cathode→USC). A stock change is a
    # reported value or the zero-centred net-stock seed, nothing else: the time
    # interpolator and the NetworkImputer's Flow-node balance closure are barred from
    # writing these legs (Round X, ADR 0040). Before: 51 balance residuals (Σ|f|
    # 2,780 kt) and 96 interpolated stock changes (3,034 kt) sat in the prior, the
    # sink pre-baked. Not a knob (ADR 0040 membership, kept in four places, #39).
    stock_sectors: tuple = ("MESC", "USC")
    # Net-stock seed σ for MESC/USC (ShareImputer seed mode), in kt: the seed is
    # priced like a feed-backed level — ADR 0025's scale-total rule, 1 % of the
    # co-located Cathode(i)→FU(j) flow, floored at the feeds' 0.3 kt resolution — so
    # a stock leg stays at its prior with small wiggle room. The former band
    # (p95/1.645 × max(FU, 100 kt)) gave the exchange-location legs σ ≈ 553 kt.
    STOCK_SEED_CV:           float = 0.01
    # icsg_feed's OBSERVED_ZERO_* mirror this floor (#76)
    STOCK_SEED_ABS_FLOOR_KT: float = 0.3
    # CP has no `eff` row, so the NetworkImputer's recipe writes CC PSC = (1 − 1.0)·Σin
    # = 0 at σ 0 wherever ICSG does not report it and CP has an input (854 region-years
    # on imp_20261021_100015_ea05007a, 857 with three time-interpolated copies), and
    # Stage 3 floors that to σ 1e-6: a hidden pin of a stock change. The zero stays; it is
    # priced at the seed's floor alone, the feeds' resolution (ADR 0025), not at the
    # rejected 1 % of CP→Cathode (#36, ADR 0041 am.). 0.0 restores the pin.
    # Rejected on measurement (2026-10-21, opt_20261021_115215_653c8677 vs
    # opt_20261021_111957_5c6d13bc): at 0.3 kt the freed legs become a sink again — the
    # near-pin PSC rows' Σz² 0.04 → 1,437 (Slovakia 2016 z +37.8), 18 stock legs newly
    # above |z| 5 — so the pin stays.
    EFFLESS_STOCK_SIGMA_KT:  float = 0.0

    # FlatPriorImputer mode
    #   "magnitude_aware" (legacy) — σ = CV(j) · scale(j) via cv_by_mask + 3-level cascade
    #   "noisy_zero" (test)        — f=0, σ=FLAT_SIGMA_HIGH; effectively no prior weight
    FLAT_MODE:       str   = "noisy_zero"
    # noisy_zero fallback σ; also the WR stripping fill's σ (ADR 0038)
    FLAT_SIGMA_HIGH: float = 1.0e6

    # FlatPriorImputer — CVs per mask (ROW edges get the geographic pedigree penalty
    # CV_GEO4) used only when FLAT_MODE == "magnitude_aware"
    FLAT_DEFAULT_CV:    float = 0.3
    FLAT_CV_LO_SUPPLY:  float = 0.04
    FLAT_CV_HI_SUPPLY:  float = 0.3
    FLAT_CV_LO_USE:     float = 0.2
    FLAT_CV_HI_USE:     float = 0.5

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
    OUTLIER_LEG_K:  float = 3.0
    OUTLIER_LEG_CV: float = 5.0

    # Contradicted flow nodes (ADR 0019). A Flow node's hard mass balance cannot
    # honour two disagreeing scale-total anchors; because σ_c scales with |c| the
    # solver collapses the node to the SMALLER prior, even when that side is the
    # less trustworthy one (Russia Matte 1995: inflow 545 kt CC from ICSG
    # concentrate use vs outflow 50.8 from an under-reported ICSG refining series
    # → the smelting chain collapsed). Where the outflow prior falls below
    # UNCLOSED_K × the inflow prior, its covered edges get σ = UNCLOSED_CV·|f| —
    # data kept, false authority removed, mirroring the outlier screen — and
    # Stage 3 (CopperConst.UNCLOSED_ANCHOR_CV) loosens the matching usetotout anchor
    # to match.
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
    UNCLOSED_SECTORS: tuple           = ("Matte", "Anode")
    UNCLOSED_CV:      Optional[float] = None

    # ADR 0027: pin ADR 0017's per-sector reference size instead of re-deriving it.
    # `a_ref` is the median of the *assembled* covered bilateral-leg population, so
    # any change to that population — the re-export netting removes the small
    # hub-side legs — shifts it (+22 % to +58 % per sector when measured), which
    # re-prices every trade leg in the sector. Across a measurement series that is
    # a σ change confounded with a structural one: the metric moves without mass.
    # Pin it to the values the predecessor run logged, and report the unpinned
    # variant as a sensitivity line. None = derive from this run (the default).
    # Pinned to the Round L.0 run `imp_20260915_095200_64dfaa88` (BACI v5, the
    # un-netted leg population), so the L.1 metric is not confounded by σ.
    LEG_SIGMA_AREF: Optional[dict] = field(default_factory=lambda: {
        "Anode": 0.213367, "Cathode": 0.26827266,
        "Concentrate": 10.26, "Matte": 0.2225045})

    # Size-damped σ on bilateral legs (ADR 0017; applied BEFORE the screen above).
    # The Laner pedigree cv (~0.21 on BACI Mass legs) is size-blind, so deleting
    # any leg entirely costs z = 1/cv ≈ 4.8 flat — per tonne the biggest legs are
    # thousands of times cheaper to delete, and the QP eats them first (deleted
    # share 7%→45% by size decile; Chile→China/Japan zeroed). Power-law damping,
    #     σ_e = cv_pedigree(e) · a_ref(sector)^(1−γ) · a_e^γ ,
    # anchored at each sector's median covered bilateral leg (a_ref, pooled over
    # years): at the median σ is unchanged, bigger legs get relatively tighter,
    # smaller looser (cv capped at LEG_SIGMA_CV_CAP). γ=1.0 or None disables.
    LEG_SIGMA_GAMMA:  Optional[float] = 0.5
    LEG_SIGMA_CV_CAP: float           = 1.0


# ---------------------------------------------------------------------------
# Domestic-use FormulaRelations — all RETIRED; the NetworkImputer closes these legs
# ---------------------------------------------------------------------------

# concentrate_domestic / matte_domestic / anode_domestic RETIRED (ADR 0010 §C).
# The deepened NetworkImputer now closes these domestic chain legs structurally
# (CC flow-balance → Mass twin), and more correctly than the hand-written
# formulas did: their export term selected entity_to=="Flow" and so missed the
# bilateral Anode(Flow)→ER(Process) trade legs, over-attributing the domestic
# leg. Verified on the reference feeds — once the recipe gate opens these three
# relations fired 0× (the network's network_mass_twin fills the domestic Mass
# legs first).
# The cathode final-demand split (cathode_fu/usc/mesc_domestic) is RETIRED too.
# USC/MESC in Round X (ADR 0040): a stock change is a reported value or the
# ShareImputer's zero-mean seed, never a cathode-balance residual (once the
# balance was barred from the stock legs they booked 7,290 kt of one-signed
# stock change). FU with it: the stock legs are no longer writable, so the
# domestic Cathode→FU leg is the Cathode node's lone same-region gap and the
# balance closes it first (production − exports − reported stock change) —
# the FU relation fired 0× on X.0. Its export term carried the same
# entity_to=="Flow" defect (every Cathode export leg ends at an FU SB node), so
# where it did fire (W.1: 226 legs, 28,740 kt) it booked production/conc with
# no export subtracted; the network writes those legs at 8,845 kt.


def cascade_in_order(by_id: dict, order) -> list:
    """The cascade imputers of ``by_id`` in ``order``, which must be a permutation of
    ``by_id``'s ids (#104); anything else — an unknown, repeated or missing id —
    raises ``ValueError``."""
    order = tuple(order)
    if sorted(order) != sorted(by_id):
        raise ValueError(f"CASCADE_ORDER {order} must be a permutation of "
                         f"{tuple(by_id)}")
    return [by_id[k] for k in order]


def validate_imp(spec) -> None:
    """Raise ``ValueError`` unless ``spec.CASCADE_ORDER`` is a permutation of
    ``spec.cascade_ids``."""
    cascade_in_order(dict.fromkeys(spec.cascade_ids), spec.CASCADE_ORDER)


def pre_impute(ctx) -> None:
    """After aggregation, before imputation: size-damped leg σ (ADR 0017), then the
    plausibility screen — the screen's cv-5 inflation overrides damping on flagged
    legs. Its report is ``ctx.reports["outlier_leg_screen"]``."""
    spec, acc, agg_result, logger = ctx.case.imp, ctx.acc, ctx.agg_result, ctx.logger
    damp_leg_sigma(acc, agg_result, spec.LEG_SIGMA_GAMMA, spec.LEG_SIGMA_CV_CAP, logger,
                   a_ref_override=spec.LEG_SIGMA_AREF)
    ctx.reports["outlier_leg_screen"] = screen_outlier_trade_legs(
        acc, agg_result, str(ctx.inputs.params.resolve("conc") / "params.csv"), logger,
        k=spec.OUTLIER_LEG_K, cv=spec.OUTLIER_LEG_CV)


def post_impute(ctx) -> None:
    """After imputation: the contradicted-node screen (ADR 0019) — must run AFTER, a
    Flow node's inflow edge (Smelting→Matte) carries no feed observation, so before
    the cascade the inflow prior reads 0 and no contradiction is visible. The outflow
    edges it inflates are covered, so imputation never overwrites them. Its report is
    ``ctx.reports["contradicted_node_screen"]``."""
    spec = ctx.case.imp
    ctx.reports["contradicted_node_screen"] = screen_contradicted_nodes(
        ctx.acc, ctx.agg_result, ctx.logger, sectors=spec.UNCLOSED_SECTORS,
        cv=spec.UNCLOSED_CV)


def imp_stage_config(ctx) -> dict:
    """The copper knob values the imp sidecar records (``stage_config``), beside the
    generic keys the stage adds."""
    spec = ctx.case.imp
    return {
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
    }


def imputers(ctx) -> list:
    """The copper imputer cascade, in order: the fixpoint :class:`Cascade` (trade,
    network, conc, ore_grade, eff in ``CASCADE_ORDER``), then the terminal time
    interpolator, net-stock share imputer and flat prior. Appends the params entries
    it reads to ``ctx.param_refs``."""
    spec, inputs, acc, logger = ctx.case.imp, ctx.inputs, ctx.acc, ctx.logger
    param_refs = ctx.param_refs

    # The conc/eff/formula imputers form a metallurgical cascade: a value
    # derived at one stage (e.g. Smelting→Matte CC via eff) only feeds the next
    # stage's fill (Matte→CFR Mass via conc/formula) on a *subsequent* pass.
    # They are registered as a fixpoint cascade — re-run together until a full
    # sweep fills no new edges — so the chain saturates instead of dying after
    # one hop (see docs/rep/analysis_imp_…_conc-after-eff-mass-gap.md).
    # TimeImputer/FlatPrior run once afterwards (terminal).

    def _build_parametric(params_id: str, imp_id: str) -> ParametricImputer:
        csv_path = inputs.params.resolve(params_id) / "params.csv"
        imp = ParametricImputer.from_csv(
            str(csv_path), acc, id=imp_id, p=spec.P_PROP, q=spec.Q_REG,
            alpha_min=spec.ALPHA_MIN_REG,
        )
        param_refs.append(inputs.params.entry(params_id))
        logger.info("C  built ParametricImputer id=%s (params=%s v%d)",
                    imp_id, params_id, inputs.params.entry(params_id)["version"])
        return imp

    # Cascade group (iterated to fixpoint): trade seeds → network closure →
    # concentration ratios → ore-grade ratios → process efficiencies. (The
    # domestic-use FormulaImputer is retired — see the note above.)
    # TradeImputer runs FIRST: it seeds cross-region Flow→Process import CC legs
    # from the BACI Mass twin (f_CC = conc·f_Mass), so the NetworkImputer recipe
    # gate can pool them and the flow-balance can close the lone domestic leg
    # (ADR 0010). It owns only cross-region edges — the complement of the
    # network's same-region territory.
    trade = TradeImputer(
        conc_csv = str(inputs.params.resolve("conc") / "params.csv"),
        p=spec.P_PROP, id="trade",
    )
    # NetworkImputer is the sole writer of process-incident, same-region chain
    # edges (recipe closure on CC + Mass twins + SEM waste). The eff/conc
    # parametric imputers are demoted to a fallback on whatever the network
    # leaves undetermined; chain-edge provenance shows network_*.
    network = NetworkImputer(
        eff_csv       = str(inputs.params.resolve("eff")       / "params.csv"),
        conc_csv      = str(inputs.params.resolve("conc")      / "params.csv"),
        ore_grade_csv = str(inputs.params.resolve("ore_grade") / "params.csv"),
        stripping_ratio_csv = str(inputs.params.resolve("stripping_ratio")
                                  / "params.csv"),
        stripping_sigma     = spec.FLAT_SIGMA_HIGH,
        p=spec.P_PROP, q=spec.Q_REG, alpha_min=spec.ALPHA_MIN_REG, cv_cap=1.5,
        id="network",
        stock_sectors=spec.stock_sectors,
        effless_stock_sigma=spec.EFFLESS_STOCK_SIGMA_KT,
    )
    param_refs.append(inputs.params.entry("stripping_ratio"))
    cascade = Cascade(
        imputers = cascade_in_order({
            "trade":     trade,
            "network":   network,
            "conc":      _build_parametric("conc",      "conc"),
            "ore_grade": _build_parametric("ore_grade", "ore_grade"),
            "eff":       _build_parametric("eff",       "eff"),
        }, spec.CASCADE_ORDER),
        max_iter = spec.CASCADE_MAX_ITER,
    )

    # 5. Temporal interpolation, no extrapolation (terminal).
    #    Scoped to same-region edges (ADR 0026) — see TIME_SCOPE_SAME_REGION.
    #    scope_edges drops the excluded edges from the series entirely, so a
    #    cross-region leg is neither a fill target nor an interpolation source.
    time_scope = time_interp_scope(expand_edge_index(acc.edge_index, acc.node_index),
                                   spec.TIME_SCOPE_SAME_REGION, spec.stock_sectors,
                                   logger)
    time_interp = TimeImputer(
        window           = spec.TIME_WINDOW,
        method           = spec.TIME_METHOD,
        min_obs          = spec.TIME_MIN_OBS,
        gap_inflation    = spec.TIME_GAP_INFLATION,
        interpolate_only = spec.TIME_INTERPOLATE_ONLY,
        p                = spec.P_PROP,
        scope_edges      = time_scope,
        id               = "time_interp",
    )

    # (The scoped crude_ore end-of-series extrapolator that used to sit here
    # is retired — ADR 0023: past the crude_ore feed's last year the ore Mass
    # chain is derived by the NetworkImputer from ICSG production through the
    # carried-forward in-situ grade, so nothing is left for it to fill.)

    # 6.5 Net-stock share imputation. Give Δstock (PSC/USC/MESC) edges informative,
    # cold-smooth priors BEFORE flat_prior would fill them with noisy_zero: PSC by
    # share-transfer (share=PSC/(CSXEW+CER) carried across years, seeding 1995; p95
    # read from the SAME stock_bounds params the Part-A soft bounds use), and
    # MESC/USC by a zero-mean seed held at σ = max(STOCK_SEED_CV·|FU leg|,
    # STOCK_SEED_ABS_FLOOR_KT) — see the CopperImp note (ADR 0040).
    share = ShareImputer.from_stock_bounds_csv(
        str(inputs.params.resolve("stock_bounds") / "params.csv"),
        scale_abs_floor = 100.0,
        p               = spec.P_PROP,
        id              = "net_stock_share",
        seed_cv         = spec.STOCK_SEED_CV,
        seed_abs_floor  = spec.STOCK_SEED_ABS_FLOOR_KT,
    )
    param_refs.append(inputs.params.entry("stock_bounds"))

    # 7. Final fallback — flat prior.
    # Two modes (see FLAT_MODE above):
    #   - "noisy_zero":      f=0, σ=FLAT_SIGMA_HIGH (constant); prior weight ≈ 0
    #   - "magnitude_aware": σ = CV(j) · scale(j) via cv_by_mask + 3-level cascade
    if spec.FLAT_MODE == "noisy_zero":
        flat = FlatPriorImputer(
            acc_struct = acc,
            mode       = "noisy_zero",
            sigma_high = spec.FLAT_SIGMA_HIGH,
            p          = spec.P_PROP,
            id         = "flat_prior",
        )
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
            "row_supply": (float(np.hypot(spec.FLAT_CV_LO_SUPPLY, CV_GEO4)),
                           float(np.hypot(spec.FLAT_CV_HI_SUPPLY, CV_GEO4))),
            "row_use":    (float(np.hypot(spec.FLAT_CV_LO_USE,    CV_GEO4)),
                           float(np.hypot(spec.FLAT_CV_HI_USE,    CV_GEO4))),
            "supply":     (spec.FLAT_CV_LO_SUPPLY, spec.FLAT_CV_HI_SUPPLY),
            "use":        (spec.FLAT_CV_LO_USE,    spec.FLAT_CV_HI_USE),
        }

        flat = FlatPriorImputer(
            acc_struct = acc,
            cv_by_mask = flat_cv_by_mask,
            default_cv = spec.FLAT_DEFAULT_CV,
            p          = spec.P_PROP,
            masks      = {"row_supply": row_supply, "row_use": row_use},
            id         = "flat_prior",
        )

    return [cascade, time_interp, share, flat]


# =============================================================================
# Stage 3 — constraints
# =============================================================================

@dataclass(frozen=True)
class CopperConst(ConstSection):
    """The copper case's Stage 3 section. Every field but the two functions is a
    knob a pipeline step may override with ``--set`` (#100).

    Supply/use total-output SCALE TOTALS pin every process's and every flow's
    total-output scale to the model's own Stage-2 prior (c = Σ f_agg), preventing
    the homogeneous scale-collapse. Built generically from the edge topology (no
    per-sector family list) — see :func:`builders`. ANCHOR_CV sets their tight σ_c.
    """

    # Note: P_PROP must match CopperImp's P_PROP (cross-stage Lp exponent; pysuto.pipeline
    # TIED).
    P_PROP:            float = 1.3
    SIGMA_A_CV_MIN:    float = 0.001
    SIGMA_A_ABS_FLOOR: float = 1.0e-6
    # Anchor CV (1%). SIGMA_C_CV_MIN is lowered to match so the engine's σ_c CV
    # floor does not inflate the tight cathode-use anchor back up to 5%.
    ANCHOR_CV:         float = 0.01
    SIGMA_C_CV_MIN:    float = 0.01
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
    FABRICATED_ANCHOR_CV: float = 0.01
    FEED_SHARE_OBSERVED:  float = 0.5

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
    UNCLOSED_SECTORS:   tuple           = ("Matte",)
    UNCLOSED_ANCHOR_CV: Optional[float] = None
    SIGMA_C_ABS_FLOOR:  float           = 1.0e-3
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
    SCALE_TOTAL_ABS_FLOOR: float = 0.3

    # Prior box ("tight fitting"): box flow bounds to a₀ ± PRIOR_BOX_K·σ_a on edges
    # with a real prior, grafting RAS's scale-floor onto the QP to stop the
    # homogeneous collapse to l. None = off (legacy behaviour). See
    # docs/rep/analysis_opt_20260602_133908_46bc514e_20260603_feed-scale-collapse.md §8.3.
    PRIOR_BOX_K: Optional[float] = None
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
    NODATA_ANCHOR_CV:      float = 0.05
    NODATA_ANCHOR_EXCLUDE: tuple = ("CO", "Mining", "Ore", "WR", "Tailings", "Slag",
                                    "PSC")
    # Data-aware exemption (ADR 0013 amendment 3, Round W.3). The extraction chain is
    # spared by the anchor only where a relation sets its magnitude: region-years with
    # an ore-grade row (ADR 0015). In countries that mine nothing its noisy zeros get
    # the ordinary sibling-median anchor, so phantom extraction costs z instead of
    # being free. Slag and PSC keep the sector rule. () restores the sector-only rule.
    NODATA_ANCHOR_DATA_AWARE: tuple = ("CO", "Mining", "Ore", "WR", "Tailings")
    # Cold sensitivity run (branch sensitivity/no-flat-prior-zeros): hard-pin every
    # genuine no-data ("flat-prior") leaf edge to [0,0] in the flow bounds — "no feed
    # data ⇒ no flow". NOTE: the pin's l/u boxes only constrain the feasibility
    # pre-check probe, NOT the QP (FlowBoundsBuilder enforces only the structural
    # lower bounds, ADR 0039) — the soft NODATA_ANCHOR_CV above is what the solver
    # actually feels. The pin stays on to
    # keep the pre-check strict and document intent. The pin spares structurally-
    # determined edges (extraction chain + all four SEM residual outputs
    # {WR,Tailings,Slag,PSC}); see ADR 0013.
    PIN_FLAT_PRIOR_ZEROS: bool = True
    # Soft-ratio σ_c scaling (REP analysis_opt_20260611_075036 §8.2): the CSV σ (ratio
    # units) is made relative, σ_r = σ/|r| (#29), and rescaled to flow space per row,
    # σ_c = max(σ_r, REL_FLOOR) · max(Σ|a₀,num|, |r|·Σ|a₀,denom|, ABS_FLOOR).
    # REL_FLOOR caps |z| at 1/floor on a 100 % ratio violation; ABS_FLOOR (in kt — the flow unit) keeps both-zero-prior
    # rows finite AND soft: ~77 % of soft ratio rows are zero-prior parametric ties,
    # and at ABS_FLOOR=1 their σ_c=0.02 kt made the Hessian so stiff Clarabel stalled
    # (InsufficientProgress @2005). 100 kt → σ_c=2 kt on those rows (z=50 per 100 kt).
    # (The REP and an earlier version of this comment wrote "t"; the value was always
    # applied in kt.)
    RATIO_SIGMA_REL_FLOOR: float = 0.02
    RATIO_SCALE_ABS_FLOOR: float = 100.0
    # Net-stock rows use their own scale floor. Their σ_c scales off the throughput
    # denominator alone (ratio.py scale_mode="denominator"), so the floor is what a
    # region with no throughput gets: at the generic 100 kt it was σ_r·100 kt on every
    # such row (Round X: 553 kt on the MESC "hub" rows, a free sink). 0.1 kt is the
    # yearbook's display resolution; the row stays finite and soft (ADR 0040).
    STOCK_BOUNDS_SCALE_ABS_FLOOR: float = 0.1
    # Net-stock ("stock_bound") mode. "soft" routes the PSC net-stock-change bounds
    # (stock_bounds v5 carries PSC rows only; MESC/USC are held by their prior σ,
    # ADR 0040) to the free-cone soft-ratio path (minimized quadratic slack, σ_c scaled
    # off the throughput denominator — see RatioConstraintBuilder scale_mode); "hard"
    # keeps the legacy nonneg-cone RatioBoundBuilder. Soft guarantees feasibility (no
    # stock row can break) and secures too-high AND too-low symmetrically. The hard
    # bounds went infeasible under the flat-prior-zero pin (denominator collapse);
    # see docs/rep/analysis_const_20260625_130524_...flat-prior-zero-pin-cold.md.
    STOCK_BOUNDS_MODE: str = "soft"
    # The soft PSC rows are centred at zero, so on a region-year whose CP→PSC leg ICSG
    # reports, the row pulls the reported stock change toward zero — the pull ADR 0040
    # retired on MESC/USC (#37, opt_20261019_171746_3272682b: Iran 2016 reported −63.0,
    # posterior −40.5 kt CC, row z −10.1). True emits a row only where the leg is not
    # feed-covered (agg_result.coverage); the near-pin rows on regions with no refinery
    # output stay (#98, ADR 0040 am.). False restores a row on every region-year.
    STOCK_BOUNDS_SKIP_REPORTED: bool = True
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
    SCRAP_SHARE_MODE: str = "hard"
    # Scrap LEVEL anchors (Round P, ADR 0031). The band above states a *share*; a
    # share band over a node with no throughput is satisfied trivially, so the level
    # ICSG reports was never stated anywhere in Stage 3 (the faucet note above is the
    # same gap seen from the other side). False restores the Round O configuration
    # for an A/B. See section 2b of builders().
    SCRAP_LEVEL_ANCHOR: bool = True
    # Mass-balance scope (ADR 0038). The acc-local mask `massbal_scope` (masks.csv)
    # excludes Mining→WR, so waste rock is a satellite mass flow outside the Mining
    # balance — the gap between crude ore and ore can no longer be booked on it.
    # None balances every edge (the pre-ADR-0038 configuration, for an A/B).
    MASSBAL_MASK: Optional[str] = "massbal_scope"
    # Enforced lower bounds (ADR 0039, Round W.1). Every edge whose structural lower
    # bound is 0 gets a hard row −a ≤ 0: non-negativity was part of the documented
    # formulation but never reached the QP (bounds lived only in the feasibility
    # probe), and residual mass edges went negative. False = pre-ADR-0039 behaviour.
    ENFORCE_LOWER_BOUNDS: bool = True
    # Stripping ratio as a HARD relation (ADR 0038, Round W.2b). As a soft row the
    # waste-rock edge — copperless, outside every balance — is decided by nothing but
    # two weak terms, and the interior-point stopping rule never pins it down: W.2a
    # (opt_20260929_111551_21703f80) ended at WR/Ore 1.04 instead of 2.0, and still
    # 1.03 when 2019 was re-solved at a 1e-6 gap. A zero-cone row enforces
    # WR = k·Ore to feasibility tolerance; the imputer leaves WR to the flat prior so
    # the relation, not a second copy of the ore prior, sets it. False = soft (W.2a).
    STRIPPING_RATIO_HARD: bool = True
    # Draw the run's ConstDiagnostic figures and 1995 prior table into <run>/diag/, about
    # half the stage's time. A sweep that deletes its const runs turns it off (#108); no
    # stage reads these files.
    DIAG_PLOT: bool = True


# -----------------------------------------------------------------------------
# Stage 3 algorithms — moved unchanged from scripts/constraints.py (#121)
# -----------------------------------------------------------------------------

def scale_total_cv(feed_share: float, c_abs: float, *, anchor_cv: float,
                   fabricated_cv: float, feed_share_observed: float,
                   abs_floor: float) -> float | None:
    """cv for one scale-total group (ADR 0016 classification, ADR 0025 floor).

    Fabricated groups (feed share below the threshold) keep ``fabricated_cv``.
    Feed-backed groups get the cv that makes ``cv·|c|`` equal the feeds'
    reporting-resolution floor on small totals, ``max(anchor_cv, floor/|c|)``,
    so σ_c = max(anchor_cv·|c|, floor). ``None`` means "use the builder's
    default cv" (a feed-backed group with |c| = 0 cannot be floored via cv).
    The four knobs are the case's ANCHOR_CV, FABRICATED_ANCHOR_CV,
    FEED_SHARE_OBSERVED and SCALE_TOTAL_ABS_FLOOR.
    """
    if feed_share < feed_share_observed:
        return fabricated_cv
    if c_abs <= 0.0:
        return None
    return max(anchor_cv, abs_floor / c_abs)


def scale_total_resolver(ctx):
    """The scale totals' σ resolver (ADR 0016, 0019, 0025): ``(key_map, edge_ids) ->
    cv`` for ``SumConstraintBuilder.from_mask_anchor``, built over ``ctx.agg_result``
    with the case's Stage 3 knobs."""
    spec, acc, agg_result, logger = (ctx.case.const, ctx.acc, ctx.agg_result,
                                     ctx.logger)
    # Provenance-aware σ_c: loosen scale totals whose prior group-sum is
    # fabricated (feed-backed share of Σ|f_agg| < FEED_SHARE_OBSERVED). Uses
    # agg_result.coverage — the pre-imputation "did any feed observe this
    # edge?" flag (GLOSSARY.md §Coverage), so imputations never count.
    f_agg = agg_result.f_agg
    cov   = np.asarray(agg_result.coverage, dtype=bool)
    f_agg_signed = np.nan_to_num(np.asarray(f_agg, dtype=float))
    absf = np.abs(f_agg_signed)

    # ADR 0019: the same detection Stage 2 used to σ-inflate the edges.
    unclosed, unclosed_rep = contradicted_flow_nodes(
        acc, agg_result, sectors=spec.UNCLOSED_SECTORS)
    logger.info("ADR 0019: %d contradicted %s node-year(s) — anchor action: %s",
                len(unclosed), "/".join(spec.UNCLOSED_SECTORS),
                f"usetotout cv={spec.UNCLOSED_ANCHOR_CV}"
                if spec.UNCLOSED_ANCHOR_CV is not None else "none (detector only)")

    def resolve(key_map, edge_ids):
        # Contradicted node: the outflow anchor loses authority so the hard
        # balance can settle at the reliable inflow prior (ADR 0019).
        if (spec.UNCLOSED_ANCHOR_CV is not None
                and (key_map.get("sector_from"), key_map.get("region_from"),
                     key_map.get("time")) in unclosed):
            return spec.UNCLOSED_ANCHOR_CV
        ids = np.asarray(edge_ids, dtype=int)
        tot = absf[ids].sum()
        if tot <= 0.0:
            return spec.FABRICATED_ANCHOR_CV
        feed_share = absf[ids[cov[ids]]].sum() / tot
        return scale_total_cv(feed_share, abs(float(f_agg_signed[ids].sum())),
                              anchor_cv=spec.ANCHOR_CV,
                              fabricated_cv=spec.FABRICATED_ANCHOR_CV,
                              feed_share_observed=spec.FEED_SHARE_OBSERVED,
                              abs_floor=spec.SCALE_TOTAL_ABS_FLOOR)
    return resolve


def nodata_anchor_exempt(ctx) -> tuple:
    """``(mask, n)``: the no-data anchor's per-edge exemption and the number of
    ore-grade region-years it spares (``(None, 0)`` when NODATA_ANCHOR_DATA_AWARE is
    empty)."""
    spec, inputs, acc, _log = ctx.case.const, ctx.inputs, ctx.acc, ctx.logger

    # ADR 0013 amendment 3: spare the extraction chain only in region-years with an
    # ore-grade row (the copper case's "a relation sets this magnitude").
    anchor_exempt, n_grade_ry = None, 0
    if spec.NODATA_ANCHOR_DATA_AWARE:
        _og = pd.read_csv(inputs.params.resolve("ore_grade") / "params.csv",
                          sep=None, engine="python", dtype=str)
        _co = _og[_og["param_id"].str.startswith("oregradecosupply")]
        _grade_ry = set(zip(_co["a_region_from"], _co["a_time"].astype(str)))
        n_grade_ry = len(_grade_ry)
        anchor_exempt = nodata_exempt_mask(acc, spec.NODATA_ANCHOR_EXCLUDE,
                                           spec.NODATA_ANCHOR_DATA_AWARE, _grade_ry)
        _log.info("No-data anchor: extraction chain spared in %d ore-grade region-years; "
                  "%d edges spared in total", n_grade_ry, int(anchor_exempt.sum()))
    return anchor_exempt, n_grade_ry


def const_stage_config(ctx) -> dict:
    """The copper knob values the const sidecar records (``stage_config``), beside the
    generic keys the stage adds."""
    spec = ctx.case.const
    return {
        "sigma_a_cv_min":    spec.SIGMA_A_CV_MIN,
        "sigma_a_abs_floor": spec.SIGMA_A_ABS_FLOOR,
        "sigma_c_cv_min":    spec.SIGMA_C_CV_MIN,
        "sigma_c_abs_floor": spec.SIGMA_C_ABS_FLOOR,
        "prior_box_k":       spec.PRIOR_BOX_K,
        "nodata_anchor_cv":  spec.NODATA_ANCHOR_CV,
        "nodata_anchor_exclude": list(spec.NODATA_ANCHOR_EXCLUDE),
        "nodata_anchor_data_aware": list(spec.NODATA_ANCHOR_DATA_AWARE),
        "nodata_anchor_grade_region_years": ctx.n_anchor_exempt,
        "pin_flat_prior_zeros": spec.PIN_FLAT_PRIOR_ZEROS,
        "ratio_sigma_rel_floor": spec.RATIO_SIGMA_REL_FLOOR,
        "ratio_scale_abs_floor": spec.RATIO_SCALE_ABS_FLOOR,
        "stock_bounds_scale_abs_floor": spec.STOCK_BOUNDS_SCALE_ABS_FLOOR,
        "ore_grade_basis": "in-situ (ADR 0015)",
        "fabricated_anchor_cv": spec.FABRICATED_ANCHOR_CV,
        "feed_share_observed":  spec.FEED_SHARE_OBSERVED,
        "stock_bounds_mode": spec.STOCK_BOUNDS_MODE,
        "stock_bounds_skip_reported": spec.STOCK_BOUNDS_SKIP_REPORTED,
        "scrap_share_mode":  spec.SCRAP_SHARE_MODE,
        "scrap_level_anchor": spec.SCRAP_LEVEL_ANCHOR,
        "massbal_mask":      spec.MASSBAL_MASK,
        "enforce_lower_bounds": spec.ENFORCE_LOWER_BOUNDS,
        "stripping_ratio_hard": spec.STRIPPING_RATIO_HARD,
    }


def builders(ctx) -> list:
    """The copper constraint builders, in registration order. Appends the params
    entries it reads to ``ctx.param_refs``; the scale totals' σ resolver is
    :func:`scale_total_resolver`.

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
                                     reported stock changes toward zero. For the
                                     same reason a PSC leg a feed reports gets no
                                     row (STOCK_BOUNDS_SKIP_REPORTED, #98).
    7b Scrap-share bands          — RatioBoundBuilder (scrap_share/params.csv, per
                                     SCRAP_SHARE_MODE); per (region, P∈{Smelting,CFR})
                                     hard band bl·Σ(CC inputs to P) ≤ Σ(Scrap→P) ≤
                                     bu·Σ(CC inputs to P), agg=true.
    8  Hard lower bounds          — FlowBoundsBuilder (structural l = 0 → −a ≤ 0, ADR 0039)
    """
    spec, inputs, acc, agg_result = ctx.case.const, ctx.inputs, ctx.acc, ctx.agg_result
    _log = ctx.logger
    _scale_total_cv = scale_total_resolver(ctx)
    out: list = []

    # 1. Hard mass balance
    _log.info("Registering MassBalanceBuilder")
    out.append(MassBalanceBuilder(mask=spec.MASSBAL_MASK))

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

    # Feed coverage — the pre-imputation "did any feed observe this edge?" flag
    # (GLOSSARY.md §Coverage); the σ resolver reads the same flag.
    _cov  = np.asarray(agg_result.coverage, dtype=bool)

    _log.info("Registering SumConstraintBuilder — supply/use total-output scale totals")
    out.append(SumConstraintBuilder.from_mask_anchor(
        acc,
        mask          = supply_mask,
        anchor_values = f_agg,
        group_keys    = GROUP_KEYS,
        sigma_cv      = spec.ANCHOR_CV,
        id_prefix     = "supplytotout",
        sigma_cv_resolver = _scale_total_cv,
    ))
    out.append(SumConstraintBuilder.from_mask_anchor(
        acc,
        mask          = use_mask,
        anchor_values = f_agg,
        group_keys    = GROUP_KEYS,
        sigma_cv      = spec.ANCHOR_CV,
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

    if spec.SCRAP_LEVEL_ANCHOR:
        _log.info("Registering SumConstraintBuilder — ICSG scrap-level anchors "
                  "(smelter scrap %d edges, direct-to-refinery scrap %d edges)",
                  int(smelter_scrap_mask.sum()), int(directref_scrap_mask.sum()))
        for _mask, _prefix in ((smelter_scrap_mask,   "smelterscrap"),
                               (directref_scrap_mask, "directrefscrap")):
            out.append(SumConstraintBuilder.from_mask_anchor(
                acc,
                mask          = _mask,
                anchor_values = f_agg,
                group_keys    = SCRAP_GROUP_KEYS,
                sigma_cv      = spec.ANCHOR_CV,
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
    param_refs = ctx.param_refs
    seen_param_ids: set[str] = set()

    def _register_builder(Builder, params_id: str, **kwargs) -> None:
        csv_path = inputs.params.resolve(params_id) / "params.csv"
        _log.info("Registering %s ← params/%s v%d",
                  Builder.__name__, params_id,
                  inputs.params.entry(params_id)["version"])
        out.append(Builder.from_csv(str(csv_path), acc_struct=acc, **kwargs))
        if params_id not in seen_param_ids:
            param_refs.append(inputs.params.entry(params_id))
            seen_param_ids.add(params_id)

    # 3. Soft ratio — concentration
    _register_builder(RatioConstraintBuilder, "conc",
                      sigma_rel_floor=spec.RATIO_SIGMA_REL_FLOOR,
                      scale_abs_floor=spec.RATIO_SCALE_ABS_FLOOR)
    # 4. Soft ratio — efficiency
    _register_builder(RatioConstraintBuilder, "eff",
                      sigma_rel_floor=spec.RATIO_SIGMA_REL_FLOOR,
                      scale_abs_floor=spec.RATIO_SCALE_ABS_FLOOR)
    # 4a. Stripping ratio WR_Mass = k·Ore_Mass (ADR 0038): the only statement about
    #     the waste-rock satellite, hard by default (STRIPPING_RATIO_HARD)
    _register_builder(RatioConstraintBuilder, "stripping_ratio",
                      sigma_rel_floor=spec.RATIO_SIGMA_REL_FLOOR,
                      scale_abs_floor=spec.RATIO_SCALE_ABS_FLOOR,
                      hard=spec.STRIPPING_RATIO_HARD)
    # 3b. Soft ratio — region×year in-situ ore grades (CO/Ore edges; ADR 0015 —
    #     these replace the conc family's retired global CO/Ore rows)
    _register_builder(RatioConstraintBuilder, "ore_grade",
                      sigma_rel_floor=spec.RATIO_SIGMA_REL_FLOOR,
                      scale_abs_floor=spec.RATIO_SCALE_ABS_FLOOR)
    # 5. Hard ratio bounds — concentration
    _register_builder(RatioBoundBuilder, "conc", kind="conc")
    # 5b. Hard ratio bounds — ore grade (per-region series band; ADR 0015)
    _register_builder(RatioBoundBuilder, "ore_grade", kind="conc")
    # 6. Hard ratio bounds — efficiency
    _register_builder(RatioBoundBuilder, "eff", kind="eff")
    # 7. Net stock bounds — soft (free cone, minimized slack) or hard (nonneg cone)
    if spec.STOCK_BOUNDS_MODE == "soft":
        _register_builder(RatioConstraintBuilder, "stock_bounds",
                          sigma_rel_floor=spec.RATIO_SIGMA_REL_FLOOR,
                          scale_abs_floor=spec.STOCK_BOUNDS_SCALE_ABS_FLOOR,
                          covered=_cov if spec.STOCK_BOUNDS_SKIP_REPORTED else None)
    else:
        _register_builder(RatioBoundBuilder, "stock_bounds", kind="stock_bounds")
    # 7b. Scrap-share bands — hard (nonneg cone) by default; see SCRAP_SHARE_MODE
    if spec.SCRAP_SHARE_MODE == "hard":
        _register_builder(RatioBoundBuilder, "scrap_share", kind="scrap_share")
    elif spec.SCRAP_SHARE_MODE == "soft":
        _register_builder(RatioConstraintBuilder, "scrap_share",
                          sigma_rel_floor=spec.RATIO_SIGMA_REL_FLOOR,
                          scale_abs_floor=spec.RATIO_SCALE_ABS_FLOOR)

    # 8. Hard lower bounds from acc_struct (ADR 0039)
    _log.info("Registering FlowBoundsBuilder (enforce_lower=%s)",
              spec.ENFORCE_LOWER_BOUNDS)
    out.append(FlowBoundsBuilder(enforce_lower=spec.ENFORCE_LOWER_BOUNDS))
    return out


# =============================================================================
# The case
# =============================================================================

CASE = CaseSpec(
    id     = "copper",
    acc_id = "copper",
    imp    = CopperImp(imputers=imputers, pre_impute=pre_impute,
                       post_impute=post_impute, stage_config=imp_stage_config,
                       diag_year="1995", validate=validate_imp),
    const  = CopperConst(builders=builders, nodata_anchor_exempt=nodata_anchor_exempt,
                         stage_config=const_stage_config, diag_year="1995"),
    # Stage 6's primary valuation is the acc's content valuation unless named; copper's
    # is contained copper.
    driver_knobs = {"psut": {"PRIMARY": "CC"}},
)
