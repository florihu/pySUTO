# Re-exports are re-attributed to origin and destination at feed time

**Status:** proposed — **measured 2026-09-15, decision pending** (run `opt_20260915_111233_def72bc3`:
sum-row z² 2,331,632 → **1,769,941, −24.1 %**, hub-class z² −76 %, Germany/Korea phantom chains to
zero, PSC tail p95 12.2 → 9.8 — but **exporter retention fails the stop rule**: Chile 0.636 → 0.561,
Peru 0.944 → 0.901, world mining chain −3 pp). Evidence: REP
[`…roundL1-baci-reexport`](../rep/analysis_opt_20260915_111233_def72bc3_20260915_roundL1-baci-reexport.md).
Round L.1. Amends
[ADR 0002](0002-single-endpoint-rows-fan-out-not-replicate.md) option D (which BACI version the
fan weights read), [ADR 0014](0014-bilateral-feeds-scoped-to-cross-region-edges.md) (the bilateral
feed now carries inferred legs, flagged and σ-compounded),
[ADR 0017](0017-size-damped-sigma-on-bilateral-trade-legs.md) (`a_ref` may be pinned across a
measurement series) and [ADR 0018](0018-import-free-fans-distribute-domestically.md) (a
*constructed* zero import sum is not an observation). Requires
[ADR 0026](0026-bilateral-trade-legs-are-not-time-interpolated.md). Baseline
`opt_20260914_090327_c21a5c48`.

## Context

A BACI row is modelled as `Flow(A) → Process(Z)` — "A's product was consumed by Z's process"
(cathode: `Flow(A) → SB(Z)`). BACI records **gross border crossings**: a cathode bundle that sails
Chile → Rotterdam → Hamburg appears as two rows, `Chile→NL` and `NL→Germany`. The second asks the
model to find Dutch cathode *production*; there is none, so
`NetworkImputer._do_flow_balance` writes the production edge as Σ exports
([network.py:468-499](../../src/pysuto/impute/network.py)) — the phantom refinery — or the QP
deletes an observed export. The first asks Dutch semis (5 kt in 2015) to absorb 387 kt, and the
trade-share fan discards the excess at feed time, so the material vanishes from the priors *and*
must be re-invented behind the export.

The measured consequences are spread across five rounds: Round C's hub class (330 group-years) and
fabricated class (296); the feed-time clip (55,248 kt CC on the fan basis); Germany "mining" 12 kt
Cu in 1,544 kt of ore to explain 12 kt of re-exported concentrate (160k z² across Ore/MF/
Concentrate); Korea "mining" 180 kt/yr until Round J; the PSC tails at Singapore, the Netherlands
and the UAE. [ADR 0002](0002-single-endpoint-rows-fan-out-not-replicate.md) option D,
[ADR 0018](0018-import-free-fans-distribute-domestically.md) and
[ADR 0024](0024-fan-domestic-leg-bounded-by-own-production.md) are all remedies for legs that
should never have needed a domestic origin, and hub∪fabricated region-sectors have carried
30–37 % of sum-row z² in every round since.

In SUT/MRIO practice a product flow is recorded by **origin of supply**, not by customs office or
by title, and where a model lacks a re-export concept the standard remedy is re-allocation to the
final destination (UNSD IMTS 2010 §2.18 defines the re-export; Feenstra–Hanson's Hong Kong
adjustment and FIGARO's quasi-transit reallocation apply it). Round C §6 recommended exactly this
shape — *"net-trade preprocessing at the feed level … confined to Stage-1 feed semantics"* — and
recorded the structural alternative (a `Flow→Flow` edge class) as the honest end-state *if* the
manuscript needs provenance through hubs. **The owner has decided against changing the accounting
structure**, so that alternative stays recorded and unbuilt.

## Decision

**A new Stage-1 step re-attributes re-exported tonnage from `A→hub→Z` to `A→Z` and emits the
result as a new registered version of the `baci` input feed.** The source version is read
immutably and hash-checked; nothing overwrites an observation in place.

Per (product *s*, region *h*, year *t*), in contained copper (BACI mass × `conc` nominal — the
basis the fan already uses; ICSG is already CC):

    I = Σ_A BACI(A→h)     X = Σ_Z BACI(h→Z)
    U = ICSG intake       P = ICSG production          (missing row → 0)
    R = max(0, I − U)     E = max(0, X − P)
    T = min(X, I, max(R, E))

`T ≤ X` and `T ≤ I` are what make the rule safe by construction. A real producer with a thin ICSG
row — Kazakhstan cathode 2015: I = 0, X = 412, U = 57, P = 299 — has `T = 0` however large `E` is.
**Producers are never netted**, which is the property Round C demanded of any netting scheme and
the reason partner-pair netting was rejected there. `R` is the use-side reading of a single
physical fact (exports supplied from imports) and `E` the supply-side reading; the rule takes the
larger and lets the caps keep it physical. `E` encodes the **production-first convention** —
domestic output supplies exports before imports do — and is what catches the `U > I` hubs that
`R` alone is blind to: Korea, Germany, France, the phantom-production cases.

Re-allocation uses the hub's own observed geography, `a_A = BACI(A→h)/I` and `b_Z = BACI(h→Z)/X`:

    BACI'(A→h) = BACI(A→h) − T·a_A
    BACI'(h→Z) = BACI(h→Z) − T·b_Z
    BACI'(A→Z) = BACI(A→Z) + T·a_A·b_Z        (A ≠ Z)

`A = Z` is a **re-import** (a returned good) or, where A is ROW, third-country trade folded onto
one aggregate; it is dropped and reported, exactly as CONTEXT's *mapped self-trade* is dropped by
ADR 0014's scope. Hubs are processed largest-`T`-first within a (sector, year) and the sweep
repeats, because hub→hub legs exist (3.4 % of trade); a hub's cumulative netting is capped at its
own `T`, so a later pass only moves what an upstream hub's re-attribution replenished.

Three exclusions, each adopted on a measurement rather than a judgement:

- **ROW is never a hub.** It is 186 economies folded into one region: it has no warehouse, its
  `I − U` mixes 186 balances, and its largest inbound legs are a source defect (BACI books
  7.8–20.8 Mt *gross* of Papua New Guinea concentrate per year to North Korea in 2019–2022,
  against PNG's real ~1.5 Mt). Allowing it made the largest inferred legs PNG→Bulgaria/Spain/
  Switzerland/China at 50–730 kt CC — phantoms laundered through an aggregate. ROW remains an
  ordinary origin and destination for other hubs.
- **Screened origins are never re-attributed.** An origin whose sector-year exports exceed
  `3 × (P + I)` with `P > 0` — the Stage-2 plausibility screen's own rule restated at Stage 1 —
  keeps its inbound legs gross. Without it the Philippines hub (PNG is 97–99 % of its imports in
  2019/2020/2022) re-attributes that same defect onto China, Japan and Korea.
- **ICSG gap-years are skipped.** Where a region reports the product in ICSG in other years but
  has neither `U` nor `P` in year *t* — Sweden 1995–2003 sits under an ICSG "Scandinavia" row that
  folds to ROW — `T = 0` and the ledger flags `icsg_gap`. A region that *never* reports the
  product keeps `U = P = 0`: that is the honest value, not a gap.

**Crumbs.** Proportional allocation over every origin × destination pair of a hub creates tens of
thousands of sub-kilotonne legs that would enter coverage and drag ADR 0017's sector medians
toward zero. A pair with no prior direct trade is therefore created only at `MIN_NEW_LEG_CC` (1 kt
CC) or above; an origin's sub-threshold pairs are folded into its kept destinations, and an origin
left with none does not participate.

**Uncertainty.** A changed leg is part observation, part inference, so the feed carries

    σ_new = cv_pedigree · √(v_obs² + (k·Δ)²),   k = √2,
    cv_root_scale(e) = max(1, √(v_obs² + 2Δ²) / |v_new|)

through the **existing** per-edge `cv_root_scale` seam that ADR 0003's ROW penalty already uses,
multiplied with the ROW ratio. The floor at 1 is the point: without it a leg that *gained* tonnage
comes out more certain than its pedigree (plain quadrature gives 0.71–1.0), which would buy metric
for nothing. A new leg carries exactly √2 × the pedigree cv; a leg 90 % netted away carries ≈ 10×.

**Two knock-on decisions the doc check forced:**

- Where netting exhausts a hub's imports (`T = I` while use remains), the group's `Σ imports = 0`
  is *constructed*, not observed, so ADR 0018's all-domestic fan must not fire. Those groups are
  **withheld** — left undistributed, ADR 0008's honest "total known, split unknown" state —
  through the `withhold_fallback` seam ADR 0020 already added to the fan builder.
- ADR 0017's `a_ref` is the median of the *assembled* covered leg population, which netting
  shifts by +22 % to +58 % per sector. Left free it would loosen σ on every trade leg and move the
  metric without moving mass. For the L runs it is **pinned** to the values the L.0 run logs; the
  unpinned variant is reported as a sensitivity line.

## Considered options

- **Imports enter the importer's Flow node (`Flow(X)→Flow(Y)` legs).** The canonical SUT form, and
  the honest end-state: the node balance becomes production + imports = domestic use + exports +
  stock, and every workaround ADR above becomes unnecessary. **Set aside by the owner** — it
  changes the accounting structure (a new edge family, ~530k edges, `rules.ods`, the masks, the
  fan's retirement) and costs origin-specific use at the process level. Recorded here so the
  option is not silently lost; Round C's option (b) and the memory note carry the full design.
- **Hub-scoped σ or anchor exceptions.** Rejected in Round C and again here: it prices away the
  symptom, teaches the model nothing, and the hub set is dynamic (France appears, Slovakia flips).
- **σ-inflate the residual inbound legs** rather than report them. Rejected for ADR 0014's reason
  for the same shape: it leaves a lying value in the prior and adds an arbitrary knob.
- **Partner-pair netting** (`A→h` against `h→A`). Rejected in Round C: `A→h→Z` is not symmetric,
  so it nets almost none of the real transit.
- **A separate feed id** (`baci_net`) instead of a new version. Rejected: `_latest_feed_froot`
  prefix-globs `feed_baci_*` and would select it silently, and `FeedCollector` would
  inverse-variance *average* two BACI feeds covering the same edges.
- **Routing the hub residual into MESC** as exchange stock. Rejected: the Netherlands/UK residual
  is quasi-transit, not warehouse stock — LME stocks run 100–500 kt against a 6.5 Mt residual.
- **Carrying surplus into the next year's T** (warehouse lag). Not taken now: it needs ledger
  evidence of adjacent years with `resid_t > 0` and `E_{t+1} > T_{t+1}`; a measured variant later.
- **Dropping the ceiling term `E` for Cathode**, whose fan residual is unbounded until the ADR
  0024 bound is extended. Rejected: the physical statement is the same for every product, and the
  gap is a fan-side one — the ledger labels those hub-years `ceiling-relabel` until L.2 closes it.
- **Capping `T` at the imports a region "needs"**. Rejected: it kills the ceiling term for every
  `U > I` hub, Korea included — the population this round exists for.

## Consequences

- The residual the rule **cannot** repair is now *reported* instead of silently clipped: tonnage
  that arrives at a hub and never leaves (the Rotterdam effect, one-sided merchanting such as
  Zambia→Switzerland, the PNG source defect, the ROW aggregate) stays where it is, carries a
  ledger label, and is refused by the QP exactly as before. Option 1 would not have repaired it
  either — it would have moved the same contradiction onto the hub's own node.
- `data/` is gitignored and `values.csv` is untracked, so **the netted feed exists only on disk**
  (the ADR 0022 hazard). A fresh checkout reproduces it from the raw BACI parse → registry v5 →
  `build_baci_reexport.py`; the ledger header records the v5 hash it was built from.
- The three ledgers live beside `values.csv` in the feed directory, so the input registry hashes
  them into the version: the run that consumed a netted feed can always be traced to the exact
  ledger that produced it.

## Measured (Round L.1, 2026-09-15) — not yet decided

Applied 10,524 kt CC (2.14 % of gross trade) over 1,358 hub-years; 182 new legs; 467 kt of
re-imports dropped; 60,600 kt of residual labelled rather than clipped.

**What it fixed.** Ore/Germany 53,606 → 0; Concentrate/Germany 21,043 → 0; Concentrate/Korea
22,479 → 0; Cathode/Portugal 30,004 → 3,984; Anode/Netherlands 68,411 → 18,379. Phantom production
at the P = 0 hubs falls 55–81 % wherever the ledger says `pass-through`. Round C hub-class z²
−76 %; hub∪fabricated 14.7 % → 8.9 % of the total; every constraint family improves; PSC net-stock
tail |z| p95 12.23 → 9.76; 187,790 kt CC of posterior mass moved; 30/30 Solved.

**What it cost.** Exporter concentrate retention Chile 0.636 → 0.561, Peru 0.944 → 0.901; world
`MF→Concentrate` 0.893 → 0.871, `Ore→MF` 0.804 → 0.772, `CO→Ore` 0.787 → 0.761. The stop rule
required ±0.01 or up, so **L.1 does not ship on this measurement**.

**Traced.** The loss sits entirely on legs the rule moved (touched-leg retention 0.814 → 0.795;
untouched legs *improve*, 0.966 → 1.007). The fan withhold is too small to explain it (124 kt of
concentrate demand) and the σ compounding is flat on every Chilean destination but Spain. The
mechanism is the concentrate balance: Chile's Mass export prior is unchanged to the kilotonne, but
re-attribution moves its tonnage from hubs — where the clip discarded it — to consumers whose ICSG
intake cannot grow to receive it, and China's node resolves the collision by deleting upstream.
Closing the clip leak therefore *exposes* a BACI-vs-ICSG disagreement the hub had been absorbing;
it does not create it.

**L.2 measured (2026-09-15), and it settles two of the three questions.** Extending ADR 0024's
production bound to Cathode and Anode — on BACI **v8**, after correcting two defects in this
round's own code (a non-producer must carry an explicit zero, or the guard reads it as unknown and
declines to bound; and `P_Anode` must pool the scrap entering CFR, or a secondary refiner like
Belgium reads 139 kt against the 380 kt its own refinery consumes) — gives, against L.0:

| | netting, bound = Concentrate | netting + bound on Cathode/Anode |
|---|---|---|
| sum-row z² (L.0 = 2,136,925) | 1,828,508 (−14.4 %) | **1,720,192 (−19.5 %)** |
| Round C hub-class z² (L.0 = 151,887) | 67,300 | **35,136** |
| Cathode/France phantom production (L.0 = 1,848) | 1,817 | **86** |
| Chile · Peru · Indonesia retention (L.0 = 0.636 · 0.944 · 0.850) | 0.574 · 0.903 · 0.845 | 0.574 · 0.903 · 0.845 |

- **The bound belongs with the netting**: a further −5.9 %, the hub class halved again, France's
  phantom refinery down 95 %, Italy's cathode cell halved, and nothing regresses.
- **The exporter cost is inherent to the re-attribution**, not to the unbounded cathode residual:
  the three exporters are identical to three decimals in both variants.
- **The stop rule still fails** (Chile −0.062, Peru −0.041 against ±0.01), so the decision stands
  with the owner: keep the netting with the extended bound, or revert to L.0 and keep ADR 0026
  alone. ADR 0024 gains the amendment either way only if the netting is kept.
