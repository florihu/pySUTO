# Import-free fans distribute domestically where BACI observes the region

**Status:** accepted — amends [ADR 0008](0008-unweighted-fans-are-not-distributed.md)
(the scope of "no weight → no distribution"). Evidence: REP
[`…roundC-transit-diagnosis`](../rep/analysis_opt_20260903_124655_4b23df2f_20260907_roundC-transit-diagnosis.md)
§1–2 and [`…roundD-fan-recovery`](../rep/analysis_opt_20260908_084500_724780b3_20260908_roundD-fan-recovery.md).
Amended 2026-09-16 by [ADR 0027](0027-re-exports-are-re-attributed-at-feed-time.md): a
`Σ imports = 0` that the re-export netting *constructed* is not an observation of domestic
sourcing, so those groups are withheld (left undistributed, ADR 0008's honest state) instead of
fanned all-domestic. The observability *set* is unchanged.
Amended 2026-10-13 (#66): the bootstrap below is now the rule — `usage` is read from the raw
ICSG rows for every group, so the trade-share weights depend only on registered inputs and one
pass (trade shares, then the ICSG feed) is exact after any ICSG change. Evidence: REP
[`…issue65-icsg-stock-parse`](../rep/analysis_opt_20261012_135741_67851aa1_20261012_issue65-icsg-stock-parse.md)
§6 (trade_shares v27 carried 47 stale weights built on the previous ICSG feed run; v28 → v29
converged to 3e-12 relative), and the rebuild on icsg v13 with baci v16
(`feed_baci_20261012_130305_1cdad15b`) reproduces v29 to 1.6e-11 relative on all 146,311 weights.
Amended 2026-10-14 (#77): BACI 490 "Other Asia, nes" maps to Taiwan, so Taiwan is observed and its
exception below no longer holds; the deferred option "Fix the BACI-490 mapping first" is taken.
Measured by #79.

## Context

[ADR 0008](0008-unweighted-fans-are-not-distributed.md) stopped the uniform fallback from
fabricating partner patterns: a fan with no weights at all is **not distributed**, and its raw
rows leave the concordance table. Its Consequences accepted the cost — "Region-years without
trade-share coverage lose their ICSG per-edge values and their contribution to the `f_agg`
scale-total anchors" — and named the revisit trigger: *"Can be revisited if dropped totals
prove material."*

They are material. The gate is [`build_trade_shares.py`](../../scripts/build_trade_shares.py)
emitting fan weights only for `covered = (imp_sum > 0) & (usage > 0)`, i.e. only where BACI
recorded **some** import of the feed commodity into that region-year. Everything else reaches
ADR 0008's drop. Measured on the shipped γ=0.5 chain:

- **469 undistributed fans, ~45 Mt CC** of ICSG production/usage never reaching any edge
  (Refining/Matte 22.4, Semi/Cathode 18.3, Smelting/Concentrate 4.8).
- The dropped region-years coincide **exactly** with Round C's "fabricated chain" class —
  exporters the model must supply from an invented domestic chain. Kazakhstan cathode is the
  clean case: fan weights exist for 2006, 2007, 2009 and 2012–24, and covered production is
  zero in precisely the complementary years, while ICSG carries a Refining/Matte row for all
  30 years (256–460 kt/yr).
- That class carries **926k of 3.78M** pooled sum-row z² (24.5 %).

The omission is expensive rather than merely absent: by [ADR 0013](0013-no-data-flat-prior-anchor.md)
the uncovered domestic legs get the flat-prior noisy zero **plus** the soft no-data anchor
(`NODATA_ANCHOR_CV = 0.05`) pulling them toward 0, and by [ADR 0006](0006-scale-total-supersedes-external-control-total.md)
the scale totals then self-anchor to that depressed `f_agg` — while BACI insists the region
exports hundreds of kt. That conflict is the tension.

## Decision

**A fan with no recorded imports is distributed onto its domestic leg when BACI observes the
consumer region; it stays undistributed when BACI is blind to that region.**

- [ADR 0002](0002-single-endpoint-rows-fan-out-not-replicate.md) option D already defines the
  domestic leg as `max(0, usage − Σ imports)` and states it *"needs no ICSG production data"*.
  At `Σ imports = 0` that formula yields `usage`. This ADR removes the `imp_sum > 0` guard that
  prevented the existing rule from applying; it introduces no new weighting policy.
- **ADR 0008's principle stands.** Its target is the *even 1/N split* — GLOSSARY's
  **Unweighted fan** is defined as one "whose legs carry no weighting information … every leg
  still at the uniform default". A domestic-weighted fan carries weights and is not an
  unweighted fan. Only the scope narrows.
- **Observability is the condition.** `Σ imports = 0` is evidence of domestic sourcing only
  where BACI actually sees the region. `baci_observed_regions()` derives the set from the BACI
  `F_root` (any cross-region Mass import, any sector, any year): **46 of 47 regions**. The sole
  exception is **Taiwan**, trade-blind because BACI 490 "Other Asia, nes" folds into ROW
  ([ADR 0014](0014-bilateral-feeds-scoped-to-cross-region-edges.md)) — and the largest single
  dropped block at 15.9 Mt. Its fans stay undistributed.
- **Bootstrap — since #66 the rule.** `usage` was the ICSG `F_root` fan total, which is 0 on
  every leg of a previously-dropped fan, so the guard could never re-open. `icsg_raw_usage()`
  reads the ICSG Use totals from the raw rows through the same sector/region maps; it
  reproduces the `F_root` fan total to a maximum relative difference of 3e-16 across all
  distributed groups (2,610 at the time; 3,113 on `feed_icsg_20261012_130637_fa562177`, 599
  of them signed stock groups). It was first applied strictly where the `F_root` total is 0,
  which left the weights depending on the previous ICSG feed run everywhere else; since #66 it
  is the only source of `usage`.

## Considered options

- **Unconditional predicate (`covered = usage > 0`).** Rejected: it asserts a 100 %-domestic
  partner pattern at the observation's confidence for Taiwan — ADR 0008's exact failure mode
  with N=1 instead of 1/N. It would cure the fabricated-chain disease in the CIS and create it
  in Taiwan, the single largest block.
- **Bundle-total soft constraint** (one soft sum row over the partner bundle; per-edge priors
  left to the imputer). Rejected for now: ADR 0008 already costed it and it requires a new
  feed→constraint path, which the domestic residual makes unnecessary for the 46 observed
  regions. Still the right shape for Taiwan-class regions if the mapping fix does not land.
- **Fix the BACI-490 mapping first** so Taiwan gains real import geography. Deferred: a
  separate change with its own blast radius (region map, ADR 0014 scope), and it does not
  block the other 46 regions. Named as the follow-up that recovers the residual 15.9 Mt.
- **Keep the status quo.** Rejected: ADR 0008's own revisit trigger has fired.

## Consequences

- 165 fan groups / **29,648 kt CC** recovered; ICSG `undistributed` falls 469 → 304 raw rows
  and `F_root` rises 2,032,772 → 2,062,420 kt. The 304 residual rows are Taiwan's 34 groups
  (15,921 kt) **by design** — the honest "total known, split unknown" state.
- Regions whose production data was dropped now carry a real production prior, so their scale
  totals ([ADR 0006](0006-scale-total-supersedes-external-control-total.md)) anchor to
  an `f_agg` that includes it, and their exports no longer require an invented chain.
- [ADR 0016](0016-provenance-aware-sigma-on-scale-totals.md)'s feed-backed/fabricated group
  classification (`FEED_SHARE_OBSERVED = 0.5`) reclassifies these groups; inert today because
  `FABRICATED_ANCHOR_CV == ANCHOR_CV`.
- A cross-feed ordering note: `build_trade_shares` reads the BACI feed run and the registered
  ICSG input, never an ICSG feed run (#66). Until then it read the ICSG `F_root`, which is built
  from the weights themselves, so after an ICSG change the documented order was exact only on a
  second pass (trade_shares v27 → v28 in REP `…issue65-icsg-stock-parse` §6). After any ICSG or
  BACI change: rebuild the trade shares, then the ICSG feed, then the Stage-2 chain — one pass.
- Glossary: see **Unweighted fan** and **Trade-share fan weighting** in
  [GLOSSARY.md](../../GLOSSARY.md).
