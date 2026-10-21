# #97 / #31 / #96: SX-EW zeros where ICSG reports none ship (phantom SX-EW 3.3 Mt → 75 kt CC, z² +1.1 %); mine-output and cathode-leg zeros are rejected (z² +37.8 %)

**Runs:** `opt_20261021_111957_5c6d13bc` (label `r97_sxew_only`; shipped) and `opt_20261021_105350_b1fea245` (label
`r97_obszero`; the full build, rejected); both 30/30 Solved, `qp_variables` defects, gap 1e-8
**Against:** `opt_20261021_101810_d34c6527` (#63's reference chain), called *base* below
**Chains:** icsg v16, baci v30, trade_shares v38 unchanged; the ICSG feed re-run at `26ae732` with the knobs below →
full: `feed_icsg_20261021_103320_53c3dfd2` → `imp_20261021_103553_f78c4fa7` → `const_20261021_104024_1c500255`;
SX-EW only: `feed_icsg_20261021_105931_15cda800` → `imp_20261021_110202_6577d943` → `const_20261021_110630_1c0f3cbf`;
both with `feed_baci_20261021_095637_d0ec13f1` and `feed_crude_ore_20260603_095651_558f7369`
**Feeds/params:** the base's (icsg v16, baci v30, trade_shares v38, crude_ore v3, acc copper v4, conc v10, eff v4,
icsg_aggregates v4, ore_grade v8, scrap_share v3, stock_bounds v6, stripping_ratio v2)
**Date:** 2026-10-21 · **Branch:** `diag/observed-zero-output-measure` · **Issues:** #97, #31, #96, #78 · **ADR:** 0024
(amended 2026-10-21)

**Verdict: the SX-EW zeros ship over item 4; the mine-output and cathode-leg zeros are rejected on measurement.**
Stamping a non-producer's SX-EW output as an observed zero removes the SX-EW cathode the solver built where ICSG
reports none — 1,344 → 33 kt CC in region-years with neither mine nor SX-EW, 1,911 → 42 in mining region-years
without SX-EW — at common-set sum-row z² **+1.10 % vs `opt_20261021_101810_d34c6527`**, Round-E holding. It fails
item 4: 95 stock legs newly above |z| 5, 60 of them Russia's (−102 kt CC over 30 years), where the lost phantom
SX-EW now draws on stocks. The full build, which also stamped mine output and #96's cathode legs, raised the common
set **+37.8 %**: those phantom chains fill real gaps between ICSG usage and recorded supply, which a zero cannot
close (ADR 0019). `opt_20261021_111957_5c6d13bc` is the reference chain.

## 1. Change

ADR 0024's amendment (#96, #97, #31): a region-year ICSG covers that produces none of a product says zero on its own
output edge, at σ 0.3 kt CC (ICSG's reporting resolution), unless another input (crude ore, ore grade) reports that
output there. The full build stamps `MF→Concentrate` (456 region-years), `SXEW→CSXEW` (973) and #96's zero-weight
domestic cathode legs (253). The shipped variant keeps `PRODUCTION_EDGES = (("SXEW", "CSXEW"),)` and
`OBSERVED_ZERO_SECTORS = ("Concentrate",)` (#78's legs as shipped).

## 2. Decision rules and results

**Full build** (items posted on #97 before its result):

| Item | Rule | Result |
|---|---|---|
| 1 | Ore→SXEW, no mine and no SX-EW, ≤ 10 % of base | **Holds:** 1,344 → 36 kt CC |
| 2 | MF→Concentrate, no ICSG mine, ≤ 20 % of base | **Fails:** 1,173 → 334 (28 %; Malaysia's 112 sit in the crude-ore years the guard spares) |
| 3 | Round-E | **Fails (a):** world Cathode CC retention 1.001 → 0.990 |
| 4 | Common-set z² rises ≤ 5 % | **Fails:** 934,466 → 1,287,652 (**+37.8 %**) |

Where the cost lands, by row family: Matte use totals +170,357 (Italy +115,164, the Netherlands +9,341, Malaysia
+4,445), Cathode use totals +149,730 (the Netherlands +47,236, Switzerland +26,159, the UAE +20,808, Malaysia
+17,821), Concentrate use totals +36,100 (the UAE +16,154, Switzerland +7,079). By band: |c| < 10 kt +229,193,
10–100 kt +116,717. The phantom chains were the solver's way to meet ICSG usage that recorded imports and ICSG
production cannot supply in these hubs and non-producers.

**SX-EW zeros only** (items posted on #97 before its result):

| Item | Rule | Result |
|---|---|---|
| 1 | Ore→SXEW, no mine and no SX-EW, ≤ 10 % of base | **Holds:** 1,344 → 33 kt CC (2.5 %) |
| 2 | Common-set z² rises ≤ 5 % | **Holds:** 941,200 → 951,519 (**+1.10 %**; 27,367 rows in both runs) |
| 3 | Round-E | **Holds.** (a) crude-ore Mass 0.998, Concentrate CC 1.003, Cathode CC 1.001 → 0.999; (b) median 1.000, p90 1.124 → 1.016; (c) Chile 1.020 → 1.021, Peru 0.978, Indonesia 0.916, world 0.947; (d) mass moved 298,356 → 285,093 kt (−4.45 %) |
| 4 | No stock leg newly above \|z\| 5 | **Fails:** 307 → 402, 95 new (65 CC: Russia 60 legs, −102.3 kt CC; Kazakhstan 1, −12.3; Slovakia 4, +8.9); all CC stock legs' net posterior −567 → −821 kt CC |

**Predictions** (full build): items 1–2 below 5 % — right on SX-EW (2.7 %), wrong on concentrate (28 %); common set
0 … +3 % — wrong (+37.8 %); SX-EW in mining regions −50 % or more — right (1,911 → 41 full, → 42 SX-EW only).

## 3. What the shipped variant moves

- **SX-EW where ICSG reports none, kt CC posterior:** no mine and no SX-EW — Belgium 475 → 1, Germany 342 → 0, the UAE
  142 → 3, Japan 128 → 0, Malaysia 108 → 1; mining regions without ICSG SX-EW — Russia 700 → 4, Kazakhstan 262 → 1,
  South Africa 199 → 9, Indonesia 165 → 1, Sweden 137 → 9, Poland 133 → 0, Bulgaria 129 → 5.
- **The phantom mine chains of non-mining regions remain,** part re-routed through concentrate: CO→Mining 2,714 →
  2,028 kt CC posterior (prior 900 → 1,968), MF→Concentrate 1,173 → 1,595 (prior 629 → 1,574). A stated limitation.
- **By region:** the Netherlands −18,530, Botswana −4,102, Taiwan −3,918; Slovakia +11,257, the UAE +9,830, Bulgaria
  +6,039, South Africa +4,806, Singapore +4,523.

## Reproduce

```bash
# full build (main 26ae732): python scripts/icsg_feed.py, then Stages 2–4 with label r97_obszero
# SX-EW only (this branch's icsg_feed.py knobs):
python scripts/icsg_feed.py
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261021_105931_15cda800 \
    --feed-id feed_baci_20261021_095637_d0ec13f1 --feed-id feed_crude_ore_20260603_095651_558f7369 --label r97_sxew_only
python scripts/constraints.py --imp-run-id imp_20261021_110202_6577d943 r97_sxew_only
python scripts/run_solver.py --const-run-id const_20261021_110630_1c0f3cbf --label r97_sxew_only
python scripts/explo/issue97_observed_zero_diag.py opt_20261021_101810_d34c6527 opt_20261021_111957_5c6d13bc
python scripts/explo/issue92_refined_scope_diag.py opt_20261021_101810_d34c6527 opt_20261021_111957_5c6d13bc 30 30 28 28
```
