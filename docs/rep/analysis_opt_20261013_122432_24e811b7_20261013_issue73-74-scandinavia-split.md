# #75: "Scandinavia" split to Sweden for 1995–2003, and its 2004 duplicate dropped (#73, #74)

**Run:** `opt_20261013_122432_24e811b7` (label `r75_scandinavia_split`; 30/30 Solved; `prior_mode` cold;
1,844 s of solve)
**Against:** the reference chain `opt_20261013_095846_46ece808` (#69, REP
[`…issue67-aggregate-gap`](analysis_opt_20261013_095846_46ece808_20261013_issue67-aggregate-gap.md)),
called *base* below
**Chain:** `icsg_clean.py --to-feed` on the #65 parse output → icsg **v14** (`fb182c99`) → baci v13 netted
against it → baci **v18** (`418d945b`; `weights.csv` = v16's, `values.csv` = v17's) →
`feed_baci_20261013_113334_e136d122`; trade_shares **v32** (`d7e4276f`, one pass) →
`feed_icsg_20261013_113455_40931987`; + `feed_crude_ore_20260603_095651_558f7369` →
`imp_20261013_113740_5e11f431` → `const_20261013_114216_1ad3239e` → the run
**Feeds/params:** icsg **v14**, baci **v18**, trade_shares **v32**, icsg_aggregates **v4**, acc copper v4,
conc v10, eff v4, ore_grade v7, stock_bounds v5, stripping_ratio v2. These are the base's versions except
the ones in bold. ore_grade and stock_bounds are kept on purpose (#30, #37), so the run isolates the
split.
**Date:** 2026-10-13 · **Branch:** `diag/scandinavia-split-measure` · **Issues:** #75, #73, #74 · **ADR:**
0032 (amended by #73/#74), 0027, 0025 (stop rule)

**Verdict: passes. Every item of the rule fixed on #75 holds**, and common-set sum-row z² falls **−6.16 %
vs `opt_20261013_095846_46ece808`**, the largest fall of the 2026-10-13 rounds.

- **#74 is fixed.** Sweden's ore goes through its concentrate route again. The phantom SX-EW route
  carries 0.0 kt CC over 1996–2003, against 307.8 in the base. The 2001 MESC move is gone: z 15.7 → 0.01.
- **#73 is fixed.** ROW's 2004 anode, smelter-concentrate and mine-output totals fall by exactly
  Scandinavia's dropped rows, and ROW's 2004 sum-row z² falls 26,099 → 609.
- **The larger effect is ROW's refinery.** In the base, Sweden's share of Scandinavia's refined output
  (1995–2003) and the 2004 double count sat in ROW. That output had no outlet, so the solver deleted
  **1,897 kt CC** of ROW's refined cathode over 1995–2004. Now it deletes 5. That alone is most of ROW's
  −79,367 z².

## 1. Change

#73/#74 (merged `bc6b346`, ADR 0032 amendment) split ICSG's "Scandinavia" rows to Sweden for 1995–2003 by
the 2004 handover year's shares (refined primary 0.521, smelting primary 0.525, mining 0.837, semis
0.618, secondary 0.914–0.916), and drop Scandinavia's 2004 rows where Sweden reports the series. icsg v14
differs from v13 in exactly 114 rows: Sweden +54, Scandinavia's 54 in 1995–2003 shrunk, 6 Scandinavia
2004 rows dropped.

## 2. Pass rule

| Item (fixed on #75 before the numbers) | Result |
|---|---|
| **#74, Sweden's processing:** `Ore→SXEW` CC posterior over 1996–2003 falls below 10 % of 307.8 kt CC; Sweden's ore goes through `Ore→MF`; its Mining→Ore posterior/prior in 1996–2003 does not fall below #69's values | **holds.** `Ore→SXEW` 307.8 → **0.0** kt CC. `Ore→MF` 134.3 → 761.8 kt CC, now with a prior of 84–107 kt CC/yr at σ 8.4–10.7, where it was a flat-prior zero at σ 1e6. Mining→Ore posterior/prior 1996–2003 is 1.00–1.12 in every year, against 0.38–0.92 in the base. |
| **#74, the 2001 MESC move:** the ROW → Sweden MESC leg's \|z\| in 2001 falls below 3 | **holds.** z 15.67 → **0.01** (CC). The leg's prior is now 2.8 kt CC, because Sweden's own cathode supply takes the larger share of its stock change. |
| **#73, ROW 2004:** ROW's 2004 ICSG-backed scale totals fall by Scandinavia's dropped rows; ROW's 2004 sum-row z² does not rise | **holds.** Anode use −408.6, smelter concentrate use −357.7 and mine output (MF supply) −98.3 kt CC, each exactly the dropped row. Semis use has no ROW scale total (see below). ROW's 2004 sum-row z² **26,099 → 609**. |
| **Netting:** Sweden 1995–2003 stays `icsg-gap`; report every other hub-year whose T changes | **holds.** baci v17 → v18: no hub-year's T or T_applied changes. Sweden's cathode 1996–2003 stays `icsg-gap`. |
| **Round-E** (ADR 0025) vs `opt_20261013_095846_46ece808` | **holds** on every item; table below. |
| **Stock legs:** net per family and per region-year, with each leg's z; no year grows into a sink | **holds.** No region-year's stock net grows by more than 1.5 kt CC (United States MESC 2001). Legs above \|z\| 5: 316 → 300, and none is new. Two family-years grow in \|net\|, both explained in §5. |

**Semis use.** ICSG books a region's semis use on the cathode legs into its FU, and those legs count
toward each *exporter's* cathode use total, not ROW's. So no ROW scale total moves by the dropped 297.4
kt CC. ROW's own cathode use total *rises* in 2004 (143.3 → 417.2 kt CC) with its z² falling 2,388.5 → 0.0,
for the reason in §3.

**Round-E, `opt_20261013_095846_46ece808` → `opt_20261013_122432_24e811b7`:**

| Item | Base | New |
|---|---|---|
| (a) world retention, CO Mass / Concentrate CC / Cathode CC | 0.862 / 0.991 / 0.986 | 0.863 / 0.993 / 0.988 |
| (b) small nodes, posterior/prior median / p90 | 1.000 / 1.165 | 1.000 / 1.167 |
| (c) exporter retention, Chile / Peru / Indonesia / world | 0.957 / 1.010 / 0.910 / 0.937 | 0.968 / 1.011 / 0.916 / 0.941 |
| (d) sum-row z², anchor band \|c\| < 10 kt | 109,638 | 109,607 (−31) |
| (d) sum-row z², 10–100 kt | 531,942 | 526,701 (−5,241) |
| (d) sum-row z², 100–1,000 kt | 597,277 | 522,857 (−74,420) |
| (d) sum-row z², > 1,000 kt | 158,190 | 151,876 (−6,314) |
| (d) mass moved, all CC edges | 482,781 kt | 465,571 kt (**−3.56 %** vs `opt_20261013_095846_46ece808`) |

Every retention rises, every band falls, and the solver moves 17,210 kt less mass.

## 3. ROW's refinery keeps its output

ROW's refined cathode supply (`CP→Cathode`, CC), prior / posterior:

| Year | 1995 | 1996 | 1997 | 1998 | 1999 | 2000 | 2001 | 2002 | 2003 | 2004 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 530 / 454 | 627 / 504 | 641 / 417 | 606 / 294 | 522 / 315 | 524 / 527 | 577 / 198 | 578 / 570 | 564 / 553 | 824 / 263 |
| new | 404 / 426 | 471 / 474 | 481 / 469 | 444 / 442 | 374 / 368 | 369 / 368 | 373 / 373 | 359 / 355 | 341 / 336 | 416 / 416 |

Σ(prior − posterior) over 1995–2004: **1,897 kt CC in the base, 5 now.**

In the base the prior carried Sweden's share of Scandinavia's refined output in 1995–2003, and in 2004
Scandinavia's output on top of Finland's and Norway's own. ROW's semis legs had no room for it: its own
cathode reached its semis demand on a leg of 2–32 kt CC in 1998, 2001 and 2004. The solver resolved the
contradiction by deleting the refinery output. Now the prior is ROW's real share. ROW's semis demand,
smaller by Sweden's share and the 2004 duplicate, leaves its domestic leg to the network imputer, which
fills it from ROW's own supply: 2004 prior 306, posterior 306 kt CC.

## 4. Cost and where it sits

**Common-set sum-row z²**, 27,012 rows in both runs, built in memory: 1,397,046 → **1,311,041 (−6.16 % vs
`opt_20261013_095846_46ece808`)**. The new run also has 156 sum rows the base lacks, all Sweden's new
1995–2003 scale totals, with z² 2,099. Counting them, the fall is still 83,906.

Every year whose sum-row z² moves by more than 1,000 vs base:

| Year | 1995 | 1996 | 1997 | 1998 | 1999 | 2001 | 2004 | 2020 |
|---|---|---|---|---|---|---|---|---|
| Δz² | −5,127 | −3,820 | −6,068 | −11,867 | −8,778 | −21,307 | −25,209 | −1,644 |

| Region | Δz² vs base | Where |
|---|---|---|
| ROW | −79,367 | the refinery chain and cathode use total, 1995–2004 (§3) |
| Sweden | −7,764 | mining and cathode use totals, 1996–2003 |
| China | −1,042 | |
| Zambia | **+2,082** | cathode use totals 1999–2004 |
| Uzbekistan | **+1,107** | cathode use totals 1996–2000 |
| Kazakhstan | +402 | |

**The rises are the exporters into ROW's semis demand.** That demand shrinks by Sweden's share
(1995–2003) and the 2004 duplicate, so the cathode legs sized to it shrink too. Zambia's 2004 cathode use
total falls 313.1 → 280.7 kt CC, below what its refinery reports. The legs had been carrying demand that
was never ROW's.

## 5. Stock legs

Per family, two years grow in \|net\|:

- **MESC 2001, −27.9 → −84.4 kt CC.** Sweden alone moves −48.8 of the −56.5: its forced +49.0 (#69, §4)
  is gone, and it no longer offsets the other regions' draw-downs. The rest is spread, none by more than
  1.5 kt CC.
- **USC 1998, +3.4 → −17.1 kt CC.** Spread over 47 regions: ROW −2.6, China −1.1, Belgium and the
  Netherlands −1.0 each, and the remaining −12.9 over 41 regions at under 0.9 each. No leg crosses
  \|z\| 5.

Per region-year, no stock net grows by more than 1.5 kt CC. Stock legs above \|z\| 5 fall 316 → 300,
and none is new.

## 6. Predictions (posted on #75 before Stages 2–4 ran)

| # | Prediction | Outcome |
|---|---|---|
| 1 | `Ore→SXEW` < 10 %; ore through `Ore→MF`; mining ratio above #69's in every year | **Holds.** 0.0 kt CC; ratio 1.00–1.12. |
| 2 | ROW → Sweden MESC 2001 \|z\| < 3 | **Holds,** 0.01. |
| 3 | ROW's 2004 scale totals fall by about the dropped rows; ROW's 2004 z² falls | **Holds** for anode, concentrate and mine output, exactly. ROW's cathode use total rises instead (§3). z² 26,099 → 609. |
| 4 | Common-set z² falls, led by Sweden and ROW | **Holds,** −6.16 % vs `opt_20261013_095846_46ece808`, led by ROW. |
| 5 | Round-E holds; mass moved changes by < 1 % | **Half wrong.** Round-E holds, but mass moved falls 3.56 % vs `opt_20261013_095846_46ece808`: the refinery deletion of §3 was larger than I expected. |
| 6 | No stock sink; no new leg above \|z\| 5 | **Holds** (§5). |

## 7. Follow-ups (not part of #75)

1. **Merging makes `opt_20261013_122432_24e811b7` the reference chain** and closes #73 and #74.
2. **The exporters into ROW** (Zambia, Uzbekistan, Kazakhstan; +3,591 together vs
   `opt_20261013_095846_46ece808`) now show their cathode outlet shortfall. That is the next place to look,
   not a regression of the split.
3. **#30 and #37** rebuild ore_grade and stock_bounds. Both read ICSG, so they should run on icsg v14.
4. **The constant 2004 shares are an estimate** (ADR 0032 amendment). If a year-specific Nordic split
   becomes available, it replaces them.

## Reproduce

```
python scripts/parse/icsg_parse/icsg_clean.py --to-feed        # icsg v14
python scripts/build_baci_reexport.py --source-version 13       # baci v18
python scripts/baci_feed.py
python scripts/build_trade_shares.py                             # trade_shares v32, one pass (#66)
python scripts/icsg_feed.py
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261013_113455_40931987 \
    --feed-id feed_baci_20261013_113334_e136d122 --feed-id feed_crude_ore_20260603_095651_558f7369 \
    --label r75_scandinavia_split
python scripts/constraints.py --imp-run-id imp_20261013_113740_5e11f431 r75_scandinavia_split
python scripts/run_solver.py --const-run-id const_20261013_114216_1ad3239e --label r75_scandinavia_split
python scripts/explo/issue75_scandinavia_split_diag.py opt_20261013_095846_46ece808 \
    opt_20261013_122432_24e811b7 17 18                           # every number in this REP
```
