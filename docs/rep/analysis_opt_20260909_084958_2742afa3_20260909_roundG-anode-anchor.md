# Round G — the ICSG refining anchor moves to the edge it measures (ADR 0021)

*Run analysed: `opt_20260909_084958_2742afa3` (30/30 Solved cold, 238 s). Chain:
`feed_icsg_20260909_082747_a347d543` (icsg v8, trade_shares v10) →
`imp_20260909_083439_8db22a64` → `const_20260909_083952_65abac84`. Baseline throughout is the
shipped run `opt_20260908_114321_279e7979` (Round F, main\@3c24a46). Branch
`fix/masscc-round-g`, code commit `a7a6718`. Suite 895 passed / 1 skipped.*

**Verdict up front: the structural error is fixed and the chain retention improves, but total
sum-row z² rises 4.8 %. The regression is concentrated in an identified population that this
round did not touch. Not recommended for merge as it stands — see §6.**

## 1 · What was wrong

`icsg_clean.py` derived the `Matte→CFR` anchor from ICSG **refined (cathode) production**
divided by `eff_CFR · eff_ER` — a series measured at the *end* of the chain, back-propagated to
its *middle*. That is valid only where the whole `CFR→Anode→ER→Cathode` chain is domestic. For
a region that smelts and exports half-finished anode (Zambia, Bulgaria, Congo D.R. have **no
domestic ER at all**) the refined series is small, so the matte had no anchored outlet, while
the inflow side of the same node came from a *different* series (smelter production). The Matte
balance is hard and `σ_c = ANCHOR_CV·|c|`, so the solver collapsed to the smaller side.

Measured on the raw feed: **95,151 kt CC of smelted matte across 841 region-years had no
anchored outlet** — an 18.4 % systemic shortfall (Σ produced 477,367 vs Σ anchored 389,328 kt).

**Round E's "nowhere to settle" reading covers only a third of the population.** Splitting the
36 contradicted node-years by whether the unplaced matte has any observed outlet:

| class | node-years | unplaced matte | observed outlet |
|---|---|---|---|
| A — outlet exists | 25 | 7,018 kt | 11,666 kt |
| B — nowhere to settle (Russia, Iran, Sweden) | 11 | 4,143 kt | 1,414 kt |

## 2 · The change

Refined production now anchors `Anode→ER` at ÷`eff_ER`; `Matte→CFR` carries no ICSG anchor and
is left to the Matte node's hard balance. The anchor is **total** refined (primary + secondary),
not primary alone — secondary is 19 % of world refined output (Germany 49 %, Austria 100 %,
Belgium 41 %), and primary-only would have relocated the contradiction to the Anode node.
`Scrap→CFR` and every `Smelting` series are bit-for-bit unchanged; the anchored magnitude moves
~1.3 %, only its placement changes.

Sanity checks that the placement is right, not merely different:
- Germany 1999's new `Anode→ER` anchor is **697.7 kt**, matching the model's own `CFR→Anode`
  prior to the decimal.
- **Austria**, a pure secondary refiner, gains an anchor it never had (0 → 103.2 kt).
- Stage-2 covered edges **66,483 → 68,573**; ICSG `F_root` 2,060,890 → 2,148,797 kt.

## 3 · Prior-level result — the contradiction is removed, not relocated

| detector | Round F | Round G |
|---|---|---|
| contradicted **Matte** node-years | 36 | **9** |
| contradicted **Anode** node-years | 0 | **0** |

The nine that remain are all Argentina, inflow 1–22 kt — noise scale. Every substantive case
(Zambia, Bulgaria, Russia, Germany, Congo D.R., Sweden, Iran, Brazil, Mexico, ROW) is gone.
The ADR-0019 detector now reports both sectors so relocation is visible; it did not happen.

## 4 · Posterior result

| metric | Round F (shipped) | **Round G** |
|---|---|---|
| solver | 30/30 | **30/30** cold |
| sum-row z² | 3,834,168 | **4,017,200 (+4.8 %)** |
| total soft z² | 4,010,004 | 4,164,430 |
| soft \|z\| p99 | 10.474 | 10.580 |
| world 2015 CO Mass | 0.649 | **0.666** |
| world 2015 Conc-CC | 0.681 | **0.735** |
| world 2015 Smelt-CC | 0.806 | **0.837** |
| world 2015 Cath-CC | 0.994 | 0.954 |
| Chile conc exports 2015 / 2019 | 0.545 / 0.415 | **0.566** / 0.395 |
| fabricated class | 200 group-years / 46,242 kt | **189 / 44,540 kt** |
| hub class | 294 | 292 |
| hub∪fab z² | 1,287,366 (33.6 %) | **1,203,638 (30.0 %)** |
| feed-time clip | 55,248 kt | 55,248 kt |

*Retention definitions here are Σa_post/Σa_prior over CC edges leaving the named sector in 2015
(CO in Mass), computed identically for both runs. They are **not** the edge set earlier REPs
used — that set could not be reproduced from the artifacts — so read the deltas, not the levels,
against Rounds A–F.*

At the 36 previously-contradicted node-years the smelting chain roughly doubles its survival:

| edge family | Round F | Round G |
|---|---|---|
| Concentrate→Smelting | 13,388 → 3,248 (24 %) | 13,388 → 6,318 (**47 %**) |
| Smelting→Matte | 15,817 → 3,546 (22 %) | 15,817 → 7,137 (**45 %**) |
| Anode exports | 8,197 → 332 (4 %) | 8,197 → 200 (**2 %**) |

Chile, the manuscript's bellwether, gains across the chain: Concentrate 0.433 → **0.544**,
Smelting 0.478 → **0.742**, Cathode 0.875 → **0.910**. Japan is flat. China's Smelting falls
0.026 and its Cathode retention moves 1.185 → 1.057 (toward, not away from, 1.0).

## 5 · Where the +183,032 z² comes from — and one disproved prediction

The change is strongly bimodal. It helps exactly the population it targeted and costs elsewhere:

| improved | Δz² | regressed | Δz² |
|---|---|---|---|
| Zambia | −239,189 | **Belgium** | **+222,696** |
| Bulgaria | −110,836 | **Italy** | **+142,745** |
| Chile | −35,823 | **Austria** | **+122,094** |
| Sweden | −22,671 | Russia | +62,976 |
| South Africa | −20,424 | Argentina | +37,254 |
| Mexico / ROW / Brazil | −16,846 / −14,305 / −11,776 | Slovakia / Congo / UK | +35,702 / +19,195 / +14,110 |

By sector the gains are upstream (MF −103,051, Concentrate −57,534, Smelting −49,788, Anode
−38,926) and the costs are at the refining end (Cathode +102,600, CER +77,406, ER +77,291,
CP +69,112, Matte +65,049, CFR +55,684).

**Belgium, Italy and Austria carry +487,535 of the +654,000 regression, and they are all
anode-importing or secondary-dominated refiners.** Giving `Anode→ER` an anchor newly constrains
the refining end wherever refined output is mostly scrap-derived or the anode is imported —
`Anode` also becomes a *fanned* sector for the first time, so the anchored value now distributes
across BACI anode-import legs that were previously unanchored. Belgium 2015: `Anode→ER` prior
575.8 → 382.3, posterior 432.9 → 156.7.

**Disproved prediction.** I expected the deleted anode-export trade to survive once the matte
had an outlet. It did not — retention went 4 % → **2 %**. The reason is visible in the priors:
Zambia 2022 carries a BACI anode-export prior of **838 kt CC against a total smelter output of
727 kt**. The export priors exceed the region's entire production, so the model is right to
refuse them. That is BACI over-statement / transit, an independent problem no anchor placement
can fix, and it belongs to the transit round.

*Diagnostic correction made in this round:* `roundC_transit_diag.py` derives the production
basis from **covered CFR feed**, the very edge family ADR 0021 un-anchors, so an uncorrected run
reported the fabricated class at 400 group-years. `PROC_OF["ER"]` now backs `Anode` as well as
`Cathode`, and both runs in the table above were re-measured with the corrected classifier. The
script also now reads its feed runs from the imp run's own metadata instead of module constants
— the hardcoded ICSG feed made every post-Round-B clip ledger a stale copy. Re-measured with
the correct feed, the clip total is unchanged at 55,248 kt: ADR 0018/0020 changed import-free
groups while the clip affects the opposite population, so the old number was accidentally right.

## 6 · Recommendation

The physics correction is sound and demonstrably works where it was aimed: the contradiction is
gone at the prior, does not reappear at the Anode node, the collapsed smelting chains roughly
double their survival, and — unlike Rounds D and F — the world chain retention moves **up**
rather than down. But it does not beat the baseline on the headline metric.

**Recommended: hold on `fix/masscc-round-g`, do the secondary-refiner round first.** Three
quarters of the regression sits in Belgium/Italy/Austria, and ADR 0021 already names the
candidate mechanism as its deferred option: the `Refining` **Secondary** series still anchors
`Scrap→CFR` at ÷`eff_CFR·eff_ER` while the same tonnage now also rides in the `Anode→ER` anchor.
Whether scrap enters at the smelter or the refinery is a separate, measurable question, and it
is exactly the population that regressed. Two further leads named by this round:

1. **BACI anode-export priors exceed regional production** (Zambia 838 vs 727 kt). The transit
   round, not this one.
2. **Belgium's ICSG series is split** across `Belgium` (2013–2024) and `Belgium-Luxembourg`
   (1995–2013), overlapping in 2013 — both map to root `Belgium`, so 2013 is double-counted.
   Pre-existing, but it bites harder now that the series anchors a major edge.

The alternatives are to ship as is (correct physics, +4.8 % z², retention up on three of five
measures) or to revert. That is a judgement call about whether a correct anchor placement is
worth a worse headline number, and it is the user's.
