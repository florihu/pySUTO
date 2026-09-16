# Grade σ is stated explicitly, not derived from the band — and year-local bands are rejected

**Status:** **mechanism accepted, policy rejected** — Round M, measured 2026-09-16. The σ split
ships (it is inert by construction and is what made the measurement attributable); the **year-local
ore-grade band and the narrowed concentrate band are rejected on measurement** and reverted. This is
the [ADR 0016](0016-provenance-aware-sigma-on-scale-totals.md) pattern. Evidence: REP
[`…roundM-grade-bands`](../rep/analysis_opt_20260916_100802_4264582e_20260916_roundM-grade-bands.md).
Amends [ADR 0015](0015-in-situ-ore-grades-on-allocated-ore-basis.md) (how the band's σ is written,
not its values) and re-affirms [ADR 0012](0012-scrap-one-mixed-grade-commodity.md)'s one-hard-band-
per-commodity invariant. Baseline `opt_20260915_131115_b8bf05a9`; runs
`opt_20260916_094322_0c932b8e` (M.1) and `opt_20260916_100802_4264582e` (M.2).

## Context

The Chile diagnosis (REP `…20260916_chile-mine-chain.md`) found the model's ore deletion executed
through two **hard ratio cones** built from the params CSVs' `value_min` / `value_max`: Chile sat on
the ore-grade *ceiling* in 26 of 30 years and 35 % of its concentrate export leg-years sat exactly
on the *floor*. The ceiling was permissive because `build_ore_grade.region_bands` applies the ±10 %
margin to each region's **30-year series extremes**, so for a declining-grade region it licenses
pricing this year's ore at the grade of decades earlier — Chile's ceiling was 1.1 × its 1995 grade,
75 % above its 2019 nominal, and its posterior ore grade sat at **1.662 %** against a measured
1.002 %.

Round M was built to answer one question: **do the bands cause the deletion, or only shape it?**

Answering it required separating a confound. `value_min`/`value_max` are read **twice** — as the
hard cone by `RatioBoundBuilder` and as the soft σ by `sigma_from_row`'s `min_max` branch, which
also feeds the `ParametricImputer` prior and the `NetworkImputer` twin map. Narrowing a band would
therefore have moved three things at once: the same confound the `a_ref` pin was invented for in
Round L.

## Decision

**1. σ is stated, not derived (accepted).** Grade rows carry `unc_type = "sd"` with an explicit
`value_sd = (value_max − value_min) / 4` — bit-for-bit the σ the `min_max` branch derived from the
same band, computed as `(round8(g_max) − round8(g_min)) / 4` so the CSV round-trip is exact. Nothing
moves: on all three Round M runs the gate reads `max|Δf_agg| = 0`, `max rel Δσ_agg ≤ 7.5e-16`,
`Δcoverage = 0`. The band's two jobs — a hard plausibility limit and a measurement uncertainty — can
now be changed independently, which is the only reason Round M's result is attributable to the cone
rather than to σ.

This does **not** retreat from ADR 0015's *"one params file now drives the imputer twin map, the
soft ratio rows, and the hard band"*. The row still drives all three; what stops is overloading
**one pair of numbers to mean two different things**.
[ADR 0011](0011-one-edge-relation-shared-across-imputation-and-constraints.md)'s target model
already lists an `EdgeRelation` as *"a sparse linear row over edges plus metadata (valuation, ε,
**σ recipe**, ownership/priority…)"* — σ is separate metadata there, so this moves toward it.

**2. Year-local bands are rejected.** The band stays the region's series envelope (ADR 0015), and
the concentrate band stays `[0.20, 0.35]`.

## Considered options

- **A. Year-local ore-grade band** — `value_min/max = g(t)·(1 ∓ margin)`, ADR 0023's wider carried
  margin preserved and re-based on the carried year. **Built, measured, rejected** (below).
- **B. A + a narrowed global concentrate band [0.24, 0.33]** — Schlesinger's 20–35 % is the range of
  an *individual* concentrate, while every model edge is a region-year average over all shipments.
  **Built, measured, rejected** (below).
- **C. Region-specific `conc` rows** — not taken. Neither constraint builder dedups by specificity
  (only `param_relation.conc_twin_map` does), so a region row *layers*: two soft rows on one pair and
  a hard band intersected to `[max(lo), min(hi)]`. ADR 0015 rejected this shape and ADR 0012's
  one-hard-band-per-commodity invariant forbids it. Regionalising means *replacing* the global row,
  which moves `build_trade_shares`, `build_baci_reexport` and every Stage-2 prior — its own round.
- **D. Touching the tailings bands** — not taken; measured not to bind (Chile 0/30 at either edge on
  both routes, world 2.4 % at cap).
- **E. A wider year-local margin (±25 %)** — the one cheap variant left in this family; it would
  still forbid the 1.66 % fabrication (ceiling 1.25 %) while leaving more room than ±10 %. Not run:
  the failure below is not a width problem, it is a relocation problem.

## Measured (Round M, 2026-09-16)

**The bands did exactly what they were built to do.** Chile's posterior in-situ ore grade
1.662 % → **1.103 %** against a measured 1.002 %; its grade-row z **+13.1 → +2.9**; its ore
retention 0.386 → **0.566**; world ore Mass 0.768 → **0.873** (+2.5 Gt); and total deviation from
the crude-ore feed Σ|posterior − prior| **7.014 → 6.176 Gt (−12 %)**.

**And the pressure simply moved, three times.** Each band is a wall; the infeasibility is a fluid
that finds the next lowest one.

| | escapes through | Chile legs on the concentrate floor | US ore retention |
|---|---|---|---|
| base | Chile's ore-grade **ceiling** | 35 % | 0.990 |
| M.1 | Chile's concentrate **floor** | 55 % | 1.307 |
| M.2 | **US ore** | **66 %** | **1.356** |

Against base, M.2 leaves Chile's exports 0.574 → **0.467**, Peru 0.903 → 0.869, Chile's mine output
0.655 → 0.579, world concentrate CC 0.877 → 0.870, legs on the concentrate floor world-wide
4.9 % → 12.5 %, and sum-row z² 1,720,192 → **1,754,038 (+2.0 %)**. Four of eleven pre-agreed
acceptance criteria fail, two of them physical. The round converts **deletion into fabrication**:
1.71 Gt less ore deleted, 0.88 Gt more invented, and the US chain — which had matched its feed at
0.990 — now overshoots by 1.0 Gt.

## Consequences

- **The bands shape the deletion; they do not cause it.** This is the direct experimental
  confirmation of the amplification the Chile diagnosis could only hypothesise: that node is
  over-determined by 3,053 kt CC and the posterior destroys 13,124 — 4.3×. Round M shows where the
  rest goes when a hiding place is removed: to the next one.
- **The next target is the discharge path**, not any band: why a contradiction between two ICSG
  figures is paid for by deleting the trade and the ore that connect them rather than by the
  anchors that disagree. Chile's export legs carry cv 0.069 after ADR 0017 damping; its scale-total
  anchors carry cv 0.017. That ratio decides who pays. Related open item: the `a_ref` pin
  (`docs/TODO.md`), which is the same family.
- **A rejected policy with a working implementation** is recorded here and in commit `f6bd5c9`; it
  is not carried as dead configuration. Re-measure, do not assume, if it is revisited after the
  discharge path is fixed.
- The σ split is inert today and stays that way until a band is deliberately changed. `ore_grade`
  **v7** is numerically identical to v4 (nominal, min and max all equal; σ to 1e-16); `conc`
  **v8** likewise to v6.
- ⚠ `data/` is gitignored and `conc/current/params.csv` is hand-maintained (ADR 0015). The only
  edit this ADR leaves in it is `unc_type` `min_max` → `sd` with `value_sd = 0.0375` on
  `concconcentrateuse` / `concconcentratesupply`, band unchanged at `[0.20, 0.35]` — re-apply that
  on a fresh checkout after regenerating.
