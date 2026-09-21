# The concentrate grade stays global — the variance is in the export mass

**Status:** **rejected on measurement** — Round N, 2026-09-17. A region×year `concentrate_grade`
params family was designed, its estimator was built and run, and the proposal is **withdrawn**: the
between-region signal is a third of the σ the single global row already carries, and half of the
deletion it was meant to relieve is arithmetically forced by missing export **mass**, which no grade
can repair. The de-risking refactor that preceded it (`bd14f04`) ships and is inert by construction.
This is the [ADR 0016](0016-provenance-aware-sigma-on-scale-totals.md) / [ADR
0028](0028-grade-sigma-is-stated-not-derived-from-the-band.md) pattern: the mechanism is kept, the
policy is rejected. Evidence: REP
[`…roundN-concentrate-grade`](../rep/analysis_opt_20260915_131115_b8bf05a9_20260917_roundN-concentrate-grade.md),
reproducible via `scripts/explo/roundN_concentrate_grade_diag.py`. Baseline
`opt_20260915_131115_b8bf05a9`. **No solver run was needed to decide it.**

## Context

[ADR 0015](0015-in-situ-ore-grades-on-allocated-ore-basis.md) replaced the conc family's global
CO/Ore placeholder with region×year in-situ grades and was one of this arc's larger wins. The
obvious next move was the same fix one step downstream: `conc_Concentrate` is a single global 0.275
for every concentrate leg from every mine to every smelter on Earth, and Chile's own books appeared
to close at 0.325 — a value *inside* the existing hard band [0.20, 0.35] that the model may use and
does not.

Four rounds had by then attacked Chile's mine-chain deletion by **re-pricing** it and all four
failed the same way ([0016](0016-provenance-aware-sigma-on-scale-totals.md),
[0019](0019-contradicted-flow-nodes-lose-anchor-authority.md), 0028 twice): closing one wall relocates the
infeasibility to the next. Round N's claim was that this one is different because it moves an
**input** — the nominal — rather than a price or a wall, and because with ICSG mine production `P`,
ICSG smelter intake `U` and the BACI bilateral mass matrix all observed, the grade is *identified*
rather than fitted:

```
g_r·X_mass(r) − Σ_A g_A·mass(A→r) = P(r) − U(r)        for every region r
```

## Decision

**Keep the single global concentrate row. Do not build the family.** Three measurements, in the
order that decided it.

**1. The 0.325 was an artifact of pooling.** Chile's implied grade year by year runs 0.24 to 0.94.
Copper concentrate assays 20-35 % Cu, so 0.94 is not a grade — it is a statement that the recorded
export mass is three times too small. Only **11 of 30** Chile-years land inside the physical band.
On those 11 — 59 % of its recorded export mass — Chile pools to **0.2783**, which differs from the
global prior 0.275 by **0.003**, under a tenth of that row's declared σ of 0.0375. Pooling a ratio
whose denominator has holes moves the ratio, not the grade.

**2. There is no region signal to resolve.** On near-pure exporters, unscreened, restricted to
physically possible observations (124 region-years, 324,519 kt):

```
between-region variance share    15.8 %        within-region (year-to-year)  84.2 %
mass-weighted sd of region means 0.0117        prior σ already carried       0.0375
```

Peru 0.256, Indonesia 0.256, Australia 0.246, Canada 0.248, Chile 0.278 — the large exporters
cluster tightly and mostly *below* the prior. The regions that look distinctive are the ones whose
in-band rate is lowest; Kazakhstan's 10th-percentile implied grade is **−0.221**, Iran's **−0.344**.
A negative grade is proof the residual is not a grade.

**3. Half the deletion is forced before the solver starts.** Copper leaves a mine node only as
domestic intake or as export mass × a grade the hard cone caps at 0.35, so `cap = 0.35·X + U` bounds
discharge at prior-level intake. On the shipped run, 144 of 965 producing region-years exceed it,
and

```
forced shortfall Σ max(0, P − cap) = 25,913 kt CC = 49.0 % of all mine-chain deletion
                        of which Chile 14,270 = 40.6 % of Chile's own deletion
```

Chile is hard-capped in **19 of 30 years**. Those 19 are exactly the years its implied grade is
impossible; the other 11 are exactly its in-band years, and they pool to 0.2783. Where Chile can
discharge its copper, its grade is ordinary; where its grade is impossible, no grade discharges it.
A higher nominal cannot help a year already evaluated at the ceiling.

## Consequences

- `conc_Concentrate` stays one global row at 0.275, σ 0.0375, band [0.20, 0.35]. ADR 0012's
  one-hard-band-per-commodity invariant is untouched. Nothing is retired, no parse skip is added,
  `TradeImputer` keeps its single `conc_csv`, and `build_ore_grade`'s tailings construction keeps
  reading the global product nominal — the coupling ADR 0015's amendment closed stays closed.
- **`bd14f04` ships.** The concentration readers in `build_trade_shares`, `build_baci_reexport` and
  the Stage-2 outlier screen are now keyed on `(sector, exporter region)` through one shared helper
  (`param_relation.conc_nominal_lookup` / `conc_for`) rather than three hand-rolled dicts keyed on
  sector alone. It is inert by construction and was verified so: BACI unbumped, `trade_shares`
  bit-identical, Stage 2 `max|Δf_agg| = 0`, `max|Δσ_agg| = 0`, `Δcoverage = 0`. It removes a real
  silent-overwrite hazard — three independent `dict` comprehensions where a region row would have
  left one arbitrary survivor with no error — and it is the plumbing any future region-resolved
  family needs. It also states the right thing in code: a traded commodity's concentration belongs
  to its **producer**, and a born re-export leg is sized at its **origin's** grade.
- The diagnostic is kept as `scripts/explo/roundN_concentrate_grade_diag.py`, including the full
  ridge-regularised bounded estimator (§5). It is the evidence, and it is what a future round would
  restart from if the export-mass gap is ever repaired.
- **`cap = 0.35·X + U` is a new detector**, and a cheap one. 144 region-years name themselves. It
  belongs in the standing diagnostic set.

## Considered and rejected

- **The per-region ratio `(P−U)/X_mass` as evidence of a grade.** Valid only where imports are
  negligible, and — the reason it misled the plan — its denominator has holes. Where `X` is
  under-recorded the ratio absorbs the missing mass as a higher grade.
- **The persistence statistic as evidence of a *grade* bias.** `|Σr| / Σ|r|` = 0.738 export-weighted,
  Chile 0.927, and the plan read that as "a constant multiplier on a region's export mass **is** a
  grade". The statistic is correct and the reading was wrong: a region whose export mass is
  systematically under-recorded carries a persistently positive residual indistinguishable from a
  persistent grade bias by that statistic alone. Persistence separates bias from noise; it does not
  separate a **mass** defect from a **grade** defect. The physical band does, and it is the test
  that should have been run first.
- **Shipping the family anyway "because it is identified".** It is identified, and the estimator
  works. It would write 2,820 params rows to move Chile by 0.003 while relocating an infeasibility
  that is half arithmetic — the Round M failure mode one level further in.
- **Widening the [0.20, 0.35] band so the hard-capped years can discharge.** This is the tempting
  repair and it is wrong twice over: it would license concentrate at 60-90 % Cu, which is metal, not
  concentrate; and ADR 0028 already measured what moving that wall does — it relocates the deletion
  (Chile's ceiling → Chile's concentrate floor → US ore 0.990 → 1.356) and converts deletion into
  fabrication. The band is not the defect; it is the instrument that *reveals* it.
- **An import-weighted grade for hub regions.** Structurally unfounded and dropped during the
  grilling before any of the above: a region's `Concentrate/Flow` node is fed only by its own
  `MF→Concentrate` edge (all 2,820 inbound edges are same-region), so there is no import mix at that
  node to weight.
- **A concentrate stock node** as the explanation for the residual. Measured and rejected at the
  planning stage on the persistence statistic — a stock effect cancels over time and only Brazil
  (0.197), the US (0.215), Iran (0.218) and Bulgaria (0.164) behave that way. That rejection stands
  and is now better explained: the residual persists because the mass gap persists.

## What this sends next

The concentrate **export-mass gap** is the round this one hands over to. It has a measured size
(25,913 kt CC forced, 49 % of mine-chain deletion), a detector (`P > 0.35·X + U`), and a named
population (144 region-years). Three checks place it in the source rather than in anything this
project built: gross BACI v5 and netted v8 are byte-identical for Chile in every year, so
[ADR 0027](0027-re-exports-are-re-attributed-at-feed-time.md)'s netting never touched it; HS 260300
itself drops while Chile's anode and cathode lines stay flat, so it is not the sector map; and world
concentrate trade excluding Chile is stable while Chile halves, with China's recorded imports *from
Chile* falling 7,720 → 2,553 kt in the same step — so the tonnage is absent from **both** sides of
the bilateral matrix, not re-attributed.

Unlike every lever tried since Round I that is an **input** defect, which is where all three of this
arc's large wins came from — [ADR 0022](0022-er-pools-its-anode-input-over-all-source-regions.md)'s `effer` flag
(−26.6 %), ADR 0027's netting (−19.5 %), ADR 0015's ore grades. It must answer, before anything is
built, what the honest repair is when the mass is genuinely unrecoverable — scale the region-year's
export legs to the copper ICSG says left, widen σ on them, or let the mine node carry a stock. Each
is a different claim about *who is wrong*, and each is measurable.

The discharge path (leg cv 0.069 against anchor cv 0.017) remains the standing target for the other
half, and is now **bounded**: it can explain at most 51 % of the mine-chain deletion.
