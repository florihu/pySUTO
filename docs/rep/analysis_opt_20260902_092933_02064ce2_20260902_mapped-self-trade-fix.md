# Mapped self-trade exclusion + CP recipe relaxation — China/ROW/Japan chains restored

**Fixes:** the China smelting collapse and Japan CP hole diagnosed from
`opt_20260901_101148_39fbd1bc` (see that run's REP for the scrap-faucet context).
**Chain:** `feed_baci_20260902_085153_de9e393f` → `trade_shares v5` (byte-identical to v4) →
`feed_icsg_20260902_085421_899e69f8` → `imp_20260902_091706_32723d1b` →
`const_20260902_092222_265df056` → **`opt_20260902_092933_02064ce2`** (cold, 30/30 Solved,
pre-check neutralised as established).
**Branch:** `sensitivity/no-flat-prior-zeros` · **Date:** 2026-09-02 · **ADRs:** 0014 (new),
0010 (amended) · Full test suite 867 passed / 1 skipped.

## The problem (short form)

1. **Mapped self-trade** (new GLOSSARY.md term): region folding (HK/Macao→China; smalls→ROW) turned
   5,181 of 78,312 BACI rows into root-level self-loops whose address IS the domestic edge. The
   flagship: HK→mainland blister 0.021 kt landed as the `Matte.China→CFR.China` **Mass** prior
   (0.0276 kt, σ=0.005) next to ICSG's **CC** 5,678 kt — physically impossible, coverage blocked
   the Mass-twin repair, and the conc-ratio rows dragged China's smelting to ~14 kt Cu. World
   concentrate demand halved; the hard Concentrate balance cut mining worldwide (Chile −63 % CC).
   Same poison on China cathode (the standing `sum_usetotout_Cathode_China` z≈92) and anode, and on
   ROW matte/concentrate/cathode.
2. **CP recipe gate**: `CP→Cathode` needs ALL same-region primaries (CER+CSXEW), but CSXEW is never
   fillable in no-SX-EW regions → 12 refining regions had CP→Cathode stuck at an anchored zero,
   forced open by the solver (Japan z≈80, chain dragged −35 %).

## The fix (three parts, all shipped)

1. **BACI scoped to cross-region edges** — new built-in mask `cross_region`
   ([acc.py](../../src/pysuto/acc.py) `_build_edge_masks`), `Edge_scope_mask = cross_region` in
   BACI's meta.xlsx, explicit `_apply_edge_scope_mask` in [scripts/baci_feed.py](../../scripts/baci_feed.py)
   (8,545 of 127,756 concordance rows dropped, logged). [ADR 0014](../adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md).
2. **Report-only cross-valuation gate** — `ImpDiagnostic.cross_valuation_check` writes
   `diag/cross_valuation_check.csv` (covered pairs with CC > Mass). **New imp run: CLEAN.**
3. **Two-phase recipe fixpoint** ([network.py](../../src/pysuto/impute/network.py)) — strict
   fixpoint first; at convergence a missing pooled primary counts as 0 iff ≥1 primary is known,
   then re-converge. Guards: all-missing still blocks; a relaxed-close output is never used to
   backward-fill the primary it assumed absent. ADR 0010 amended.

**Deviations from the approved plan, noted:** (a) Steps A and B were validated in one combined run
(the network change was already in the tree when the imp stage ran); attribution stays clean because
the two mechanisms touch disjoint edge families, each verified in the imputation log. (b) Two
pre-existing breakages had to be repaired to re-run Stage 1 at all: the legacy sector-pair
`baci/weights.csv` is now expanded to the route+time key at load (old semantics preserved —
scripts/baci_feed.py `load_user_weights`), and `build_trade_shares.py` got the delimiter-agnostic
conc reader. Trade shares regenerated **byte-identical** (122,670 rows, 0 diffs) as predicted.

## Results (2015 cross-section unless noted; "old" = opt_20260901_101148_39fbd1bc)

| Metric | old | **new** |
|---|---|---|
| `Smelting.China→Matte` CC (prior 6,885) | 14 | **5,373** |
| `Matte.China→CFR.China` CC / Mass prior | 5,678 / **0.03** | 5,678 / **10,815** (twin) |
| `CP.China→Cathode` CC (prior 8,439) | 53 | **8,994** |
| `sum_usetotout_Cathode_China` z (2022) | 92.2 | **9.7** |
| World `Smelting→Matte` post/prior | 0.51 | **0.82** |
| World `CFR→Anode` post/prior | 0.52 | **1.02** |
| World `CP→Cathode` post/prior | 0.65 | **1.00** |
| Japan `CP→Cathode` (prior 0 → 1,485) | forced open, z≈80 | **1,466, z=0.2** |
| CP zero-prior-forced-open regions | 12 | **8** (see follow-up 3) |
| Boundary Cu total, 30 y (post / prior) | 0.66× | **0.91×** |
| Primary share (post, sample years) | 83–92 % (demand-starved) | **69–80 %** (realistic; prior 77–84 %) |
| Panama/Taiwan scrap | 0 | **0** (4.2 t total) |
| Scrap-share hard bands (5,640 rows) | 0 violations | **0 violations** |
| Net-stock MESC / USC max \|z\| | 3.4 / 5.1 | **3.8 / 5.3** |
| Soft \|z\| p99 / max | 20.9 / 92 | **14.7 / 66** |

SD-R-10 re-rendered into the run's `diag/`. ROW matte is consistent (Mass 615 vs CC 323 ≈ 1.9×).

## Reading the numbers

- **The severed chains are healed at the data level, not by free knobs.** China's pyromet chain,
  ROW, and Japan all sit near their (real) priors; world scrap rose to 1.16× its prior — and the
  per-region shares saturate exactly at their **data-driven band ceilings** (China 0.4017 = its
  p95+0.05 bound; Austria pinned at its 0.95 floor). The bands hold as hard rows (0 violations).
  Note: the constraints-CSV `lhs_post` renders 0.00 for these netted agg band rows — a **display
  bug** in the diag export only; the shares prove the rows bind.
- **The residual mining cut is now measurable and Chile-shaped.** Chile stays at 0.37 (Zambia
  0.41) while every other miner runs at 0.93–1.13. Mechanism: world in-situ mining priors
  (24.4 Mt CC, recovered-basis reality ≈19.1) still exceed restored downstream absorption, and the
  **cv-proportional σ makes the largest bilateral legs the cheapest tonnes to delete** — zeroing
  Chile→China (1,028 kt, z −14.6) costs the same z² as zeroing a 5-kt leg. The reconciliation
  therefore lands almost entirely on the biggest exporter's biggest legs (Chile→China/Japan = 0),
  and Zambia's anode exports lose to China's restored domestic anode. This is the measured input
  for the deferred **grade-basis / eff-nominal follow-up** (ore_grade is recovered-basis 0.75 %
  for Chile 2015; the Ore-CC edge is in-situ ≈1.0 %; effmfflotation=0.8 vs real 0.85–0.90).

## Follow-ups (documented, not chased)

1. **Grade basis / σ design** (the residual Chile cut): reconcile recovered-vs-in-situ bases
   (grade ÷ eff, or re-base Ore CC), review eff nominals, and consider whether bilateral-leg σ
   should not be purely multiplicative (the delete-the-biggest-leg economics above).
2. **Germany fabricated mining chain**: `Ore.Germany→MF` Mass prior 5,744 kt against a CO feed of
   5 kt (z −58.6 in both runs, now the top strain). An imputation artifact worth tracing.
3. **EU refiner CP holes (8 regions left)**: Austria/France/Italy/Netherlands/… have ~0 prior on
   the whole Anode→ER→CER chain (France 0.1 kt vs ~500 kt real refined output) — a refined-
   production feed-coverage gap, upstream of the recipe fix.
4. PSC soft-bound z-tail (max 37) and the feasibility pre-check fragility — unchanged, standing.
5. BACI 490 ("Other Asia, nes") → ROW while Taiwan is a model region (ADR 0014 note).
