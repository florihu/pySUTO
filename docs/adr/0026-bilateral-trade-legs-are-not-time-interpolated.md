# Bilateral trade legs are not time-interpolated

**Status:** accepted — Round L.0 measured and shipped 2026-09-15 (run
`opt_20260915_103911_38b3fffc`, sum-row z² 2,331,632 → **2,136,925, −8.4 %**, with 92,213 kt CC of
posterior mass moved). Baseline `opt_20260914_090327_c21a5c48` (ADR 0025, K.1). Evidence: REP
[`…roundL0-no-bilateral-time-interp`](../rep/analysis_opt_20260915_103911_38b3fffc_20260915_roundL0-no-bilateral-time-interp.md). Restores the intent already stated in
[ADR 0010](0010-trade-first-seeding-of-bilateral-flow-process-inputs.md) and answers the
follow-up [ADR 0024](0024-fan-domestic-leg-bounded-by-own-production.md) left open
("whether bilateral legs should be temporally interpolated at all"). Prerequisite for
[ADR 0027](0027-re-exports-are-re-attributed-at-feed-time.md).

## Context

`TimeImputer` is registered in the Stage-2 driver with **no scope**
([`collect_aggregate_impute.py`](../../scripts/collect_aggregate_impute.py)), so it groups every
edge by `(from_id, to_id, valuation)` and linearly interpolates any year missing between two
observations — bilateral BACI legs included.

That is wrong for a bilateral trade feed, and the codebase already says so. BACI is a **complete**
matrix over (exporter, importer, HS6, year): a route that carries no row in a year is a route that
carried no trade, not a route whose value is unknown. [ADR 0010](0010-trade-first-seeding-of-bilateral-flow-process-inputs.md)
states the rule for the same population — *"A bilateral import leg with no trade observation means
'no trade on this route' — 0 is the honest value"*, and the uncovered edge *"stays an honest
flat-prior gap the solver can widen"* — and GLOSSARY's **Trade-seed** entry repeats it. The
interpolator silently overrode both.

Measured on the shipped Round-K chain (`imp_20260910_131346_e055415a`):

| population | fills | prior |
|---|---|---|
| `time_linear` total | 18,838 | |
| of which cross-region, Flow-origin | 18,430 | |
| of which **Mass** (the BACI-valued half) | **9,215** | **20,252 kt** |
| same-region (the legitimate population) | 408 | |

The posterior keeps **15,266 kt** of that invented trade (retention 0.754). The largest blocks are
exactly the region-years the model has the least business inventing: **Papua New Guinea
concentrate 2,232 kt** — a source defect the Stage-2 outlier screen flags in 2019–2022 and the
interpolator then carries into the years between — **Zambia cathode 1,027**, **Peru 1,047**,
**ROW 925**, **Zambia matte 724**, **Congo D.R. 702**.

The immediate cost is a fabricated observation. The compounding cost is that it is *covered-like*:
an interpolated leg enters `f_agg`, so it counts in ADR 0017's sector median, in the outlier
screen's export totals, and in every downstream total that reads a bilateral leg.

## Decision

**The temporal interpolator is scoped to same-region edges.** `TIME_SCOPE_SAME_REGION = True`
builds the mask `region_from == region_to` and passes it as `TimeImputer(scope_edges=…)`. The
imputer's `scope_edges` drops the excluded edges from its series entirely, so a cross-region leg
is neither a fill target **nor** an interpolation source — both halves matter, because anchoring a
fill on an invented neighbour would propagate the same defect.

A bilateral leg with no BACI observation therefore reaches Stage 3 as what it is: an uncovered
edge carrying the flat prior's noisy zero plus [ADR 0013](0013-no-data-flat-prior-anchor.md)'s
no-data anchor — the state ADR 0010 designed for it.

The scope covers `Flow→SB` cathode legs correctly without a special case: a cross-region
`Cathode.A → FU/USC/MESC.Z` leg is BACI trade and is excluded; the same-region one is domestic
consumption and still interpolates.

## Considered options

- **Scope by feed instead of by region** (`scope_feed_id="baci"`). Rejected: the flag the imputer
  derives is "this series saw ≥1 BACI observation", which is the wrong test — a route BACI *never*
  reported is exactly the one that must not be filled, and it would pass the scope untouched.
- **Drop the interpolator entirely.** Rejected: the 408 same-region fills are a legitimate
  population (a domestic series with a reporting hole in one year genuinely is unknown), and
  removing them would be a second, unmeasured change in the same commit.
- **Leave it and let the QP delete the invented legs.** Rejected on the measurement: the QP keeps
  75 % of the tonnage, and ADR 0016's lesson is that a fabricated prior is not free — the scale
  totals self-anchor to it.
- **Interpolate but inflate σ.** Rejected for the reason [ADR 0014](0014-bilateral-feeds-scoped-to-cross-region-edges.md)
  gives for the same shape: *"leaves a lying value in the prior … and adds an arbitrary inflation
  knob"*. σ changes the price, not the answer.

## Consequences

- The edge set and the total fill count are unchanged (785,927); the freed edges fall through to
  the flat prior, so this moves edges between provenance classes rather than leaving holes.
- `stage_config.time.scope_same_region` records the policy per run, so a future reader can tell
  which chain had it.
- It is a **prerequisite for ADR 0027**: the re-export netting drives a hub's export legs to zero
  in some years, and the unscoped interpolator would write them straight back from the adjacent
  partially-netted years — the mechanism that carried Round J's phantom United States legs into
  2000.
- Reverting is one constant (`TIME_SCOPE_SAME_REGION = False`), and the pre-ADR behaviour is
  pinned by a test so the regression stays visible.

## Measured (Round L.0, 2026-09-15)

Feeds pinned to the baseline's, so the change is isolated to one constant.

| | base `opt_20260914_090327_c21a5c48` | L.0 `opt_20260915_103911_38b3fffc` |
|---|---|---|
| sum-row z² | 2,331,632 | **2,136,925 (−8.4 %)** |
| cross-region `time_linear` fills | 18,430 (9,215 Mass, 20,252 kt) | **0** |
| total imputer fills | 785,927 | 785,927 |
| Σ\|Δposterior\| on CC edges | — | 92,213 kt (1.34 %) |
| world retention CO Mass / Conc CC / Cath CC | 0.785 / 0.881 / 1.014 | 0.788 / 0.882 / 1.012 |
| exporter retention Chile / Peru / Indonesia | 0.618 / 0.947 / 0.841 | 0.636 / 0.944 / 0.850 |
| Zambia→CH/SG/LU anode retention | 0.021 | 0.019 |
| solver | 30/30 Solved | 30/30 Solved |

The gain is concentrated in the region-years whose trade was interpolated: Concentrate Japan
−14,591, Concentrate Austria −14,495 (to 323 — it was almost entirely invented), Matte Italy
−13,034, Concentrate Korea −12,712, Cathode Portugal −11,357, Anode Papua New Guinea −6,577.
Tension relocates onto a handful of nodes whose balance the invented imports had been closing
(CFR/Anode Belgium +5,049/+4,624, Cathode Zambia +2,316, MF Botswana +1,710); the REP names them.

**Decided 2026-09-15: shipped.**
