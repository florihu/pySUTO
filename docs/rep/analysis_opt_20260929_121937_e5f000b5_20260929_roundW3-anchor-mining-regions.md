# Round W.3 — the extraction chain is anchored where no ore-grade relation sets it

**Run:** `opt_20260929_121937_e5f000b5` (label `roundW3_anchor_mining_regions`, 30/30 Solved, `prior_mode` cold)
**Against:** W.2b `opt_20260929_115343_b1205921` (predecessor), base `opt_20260916_124721_c4d1d944`
**Chain:** `imp_20260929_113529_ab914090` (W.2b's, unchanged) → `const_20260929_120611_6ef43f64` → the run
**Feeds/params:** as W.2b — icsg v12 / trade_shares v21, baci v11, crude_ore v3; acc copper v4; eff v4,
stripping_ratio v1, conc v8, ore_grade v7, stock_bounds v4, scrap_share v3
**Date:** 2026-09-29 · **Branch:** `fix/waste-rock-satellite` · **Issue:** #28 ·
**ADR:** [0013 amendment 3](../adr/0013-no-data-flat-prior-anchor.md)

**Verdict: the data-aware carve-out removes most of the free signed loops at no measurable cost.
Mass edges more than 1 kt below zero fall from 817 to 149 (their negative mass from −24,272 to
−4,434 kt), common-set sum-row z² rises 0.2 % vs W.2b (−9.5 % vs base), and every stop-rule metric
holds. It does not remove phantom extraction: the concentrate that region-years without an
ore-grade row produce stays (5,839 → 5,667 kt CC over 1995–2024), now fed by positive ore mass
instead of negative tailings. That concentrate goes to the same country's smelters (Japan 3,405 of
3,430 kt CC), which points at a gap on the smelters' input side — a data question for its own round,
not slack the anchor can price away.**

---

## 1. Mechanism being removed

ADR 0013 spared every edge touching `{CO, Mining, Ore, WR, Tailings}` from the no-data anchor
because a relation, not data, sets their magnitude. That holds only where the relation exists: the
ore-grade rows (ADR 0015) cover 887 region-years. In the other region-years — countries that mine
nothing — a no-data extraction edge kept the flat-prior σ ≈ 1e6 and cost nothing, so the QP used
the chain as signed slack: on W.2b, 817 Mass edges sat more than 1 kt below zero (MF→Tailings 375,
SXEW→Tailings 245, Ore→SXEW 191, Ore→MF 6).

## 2. Change

- `nodata_exempt_mask` (`src/pysuto/constraints/engine.py`): an edge is spared when it touches an
  excluded sector **and**, if that sector is data-aware, its `(from_region, time)` has an ore-grade
  row. `ConstraintEngine(nodata_anchor_exempt_mask=…)` replaces the sector rule with this mask.
- `scripts/constraints.py`: `NODATA_ANCHOR_DATA_AWARE = ("CO", "Mining", "Ore", "WR", "Tailings")`;
  the region-years come from the `oregradecosupply` rows of ore_grade v7. Slag and PSC keep the
  sector rule. Recorded in the sidecar (`nodata_anchor_data_aware`, `nodata_anchor_grade_region_years`).
- Effect on σ: the chain is spared in 887 region-years (23,698 edges); **5,884 more** no-data edges
  are anchored (709,304 vs 703,420). The zero pin is unchanged (703,420 edges, probe only).

## 3. Results

| | W.2b | W.3 | Δ |
|---|---|---|---|
| status | 30/30 Solved | 30/30 Solved | |
| solve time | 189 s | 197 s | +4 % |
| common-set sum-row z² (25,620 rows) | 1,551,647 | 1,554,826 | +0.2 % (−9.5 % vs base) |
| Mass edges > 1 kt below 0 | 817 (−24,272 kt) | 149 (−4,434 kt) | −82 % |
| · MF→Tailings / SXEW→Tailings | 375 / 245 | 116 / 23 | |
| · Ore→SXEW / Ore→MF | 191 / 6 | 6 / 4 | |
| world WR Mass, 1995–2024 | 99,797,432 kt | 99,724,740 kt | WR/Ore 2.000 |
| ore-grade soft rows z² | 63,693 | 64,079 | +386 |
| zero-pin exceedances (\|a*\| > 1 kt) | 3,496 edges, 17,190 kt | 3,481 edges, 17,087 kt | |

All 149 remaining negative edges lie in region-years without an ore-grade row; Japan (28 edges,
−2,295 kt) and Korea (36, −1,734 kt) hold 91 % of their mass.

Extraction chain in region-years without an ore-grade row, world 1995–2024 (posterior, kt):

| pair | W.2b | W.3 |
|---|---|---|
| CO→Mining Mass | 3,577 | 18,795 |
| Mining→WR Mass | 7,154 | 37,590 |
| MF→Tailings Mass | −13,512 | −2,371 |
| MF→Concentrate Mass | 20,943 | 20,320 |
| MF→Concentrate CC | 5,839 | 5,667 |

The phantom concentrate barely moves (−3 %). With the tailings loops priced, the MF balance is now
closed by positive ore mass, so crude ore — and with it the satellite waste rock — rises in these
region-years. Largest contributors to MF→Concentrate CC (W.3): Japan 3,418 kt (15 years),
Korea 803, Belgium 328, Germany 237, Italy 166, UAE 143. Japan's domestic concentrate goes almost
entirely to its own smelters (Concentrate→Smelting 3,405 kt CC; exports 25 kt). That points at a
gap between the smelters' intake and the recorded concentrate imports (not yet verified), and the
anchor σ (5 % of the sibling median) is too loose to outweigh that pull.

Stop rule (Round E, W.2b → W.3): (a) world retention CO Mass 0.799 → 0.798, Concentrate CC
0.939 → 0.938, Cathode CC 0.969 → 0.969; (b) small nodes (|c| < 18 kt, n 4,754) median
posterior/prior 1.001 → 1.001, p10 0.966 → 0.966, p90 1.194 → 1.193; (c) concentrate export
retention 2010–19 Chile 0.620 → 0.621, Peru 1.009 → 1.008, Indonesia 0.855 → 0.855, world
0.843 → 0.843; (d) z² by anchor band ±0.0 % in every band below 1,000 kt, +1.0 % above. Mass
moved: 7,522 kt of 6,065,792 kt CC (0.12 %); largest world CC pair change Mining→Ore CC −686 kt
(−0.1 %).

Pre-registered expectations: non-mining negative loops shrink ✓ (817 → 149); phantom concentrate
falls or becomes a visible cost — ✗ it neither falls nor carries much cost (+0.2 % z²); Round-E rule
reported ✓.

## 4. What remains

- 149 negative Mass edges, all in non-mining region-years → Round W.1 (enforced lower bounds).
- **Phantom concentrate in smelting countries without mines** (Japan, Korea): compare smelter intake
  (ICSG) with recorded concentrate imports (BACI) per year; the gap is what the chain manufactures.
  Its own round, outside #28.
- **Germany 2018/2021** (pre-existing, identical in W.2b): the extraction-chain scale totals
  (`usetotout_Ore`, `supplytotout_MF`, `usetotout_Concentrate`, c ≈ 20–28 kt CC) solve to ≈ 0.07 kt
  at z ≈ −58.5 — about 101,000 z² over Germany's extraction-chain rows with z² > 50. A data question
  for a follow-up.
- Reproduce: `python scripts/explo/roundW_waste_rock_diag.py opt_20260916_124721_c4d1d944
  opt_20260929_115343_b1205921 opt_20260929_121937_e5f000b5` and
  `python scripts/explo/roundK_sigma_floor_diag.py opt_20260929_115343_b1205921 opt_20260929_121937_e5f000b5`.
