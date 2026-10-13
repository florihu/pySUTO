# #75: "Scandinavia" split to Sweden for 1995–2003, and its 2004 duplicate dropped (#73, #74)

**Run:** `opt_20261013_122432_24e811b7` (label `r75_scandinavia_split`; 30/30 Solved; `prior_mode` cold;
1,844 s of solve, per the run's sidecar)
**Against:** the reference chain `opt_20261013_095846_46ece808` (#69, REP
[`…issue67-aggregate-gap`](analysis_opt_20261013_095846_46ece808_20261013_issue67-aggregate-gap.md)),
called *base* below
**Chain:** `icsg_clean.py --to-feed` on the #65 parse output → icsg **v14** (`fb182c99`) → baci v13 netted
against it → baci **v18** (`418d945b`; `weights.csv` = v16's, `values.csv` = v17's) →
`feed_baci_20261013_113334_e136d122`; trade_shares **v32** (`d7e4276f`, one pass) →
`feed_icsg_20261013_113455_40931987`; + `feed_crude_ore_20260603_095651_558f7369` →
`imp_20261013_113740_5e11f431` → `const_20261013_114216_1ad3239e` → the run
**Feeds/params:** icsg **v14**, baci **v18**, trade_shares **v32**, icsg_aggregates **v4** (base: v2, the
membership unchanged), acc copper v4, conc v10, eff v4, ore_grade v7, scrap_share v3, stock_bounds v5,
stripping_ratio v2. These are the base's versions except the ones in bold. ore_grade and stock_bounds are
kept on purpose (#30, #37), so the run isolates the split.
**Date:** 2026-10-13 · **Branch:** `diag/scandinavia-split-measure` · **Issues:** #75, #73, #74; fix
ticket #76 · **ADR:** 0032 (amended by #73/#74), 0027, 0025 (stop rule)

**Verdict: fails on one clause, #73's semis use. Every other item of the rule fixed on #75 holds.** Fix
ticket **#76**.

- **#74 is fixed.** Sweden's ore goes through its concentrate route again. The phantom SX-EW route
  carries 0.0 kt CC over 1996–2003, against 307.8 in the base, and the 2001 MESC move goes from z 15.7 to
  |z| 0.01.
- **#73 holds on anode, concentrate and mine output, not on semis.** ROW's 2004 totals for those three
  fall by exactly Scandinavia's dropped rows, and ROW's 2004 sum-row z² falls 26,099 → 609. ROW's semis use
  does not fall: the ICSG fan falls by the dropped 297.4 kt CC, but the network imputer refills ROW's
  domestic semis leg, which the fan now leaves at 0. Model prior 1,396 → 1,404 kt CC, posterior 1,498 →
  1,680.
- **The headline must be read with that.** Common-set sum-row z² falls −6.16 % vs
  `opt_20261013_095846_46ece808`, most of it on ROW's refinery chain. On the base ROW's refined cathode was
  deleted (1,897 kt CC over 1995–2004). Now it is kept, and it reaches an imputed ROW semis demand that ICSG
  does not report: over 1995–2004 the model books ROW's semis use 4,500 kt CC above the ICSG fan in
  posterior, against 2,434 on the base. Part of the gain is that route, filed as #76.

## 1. Change

#73/#74 (merged `bc6b346`, ADR 0032 amendment) split ICSG's "Scandinavia" rows to Sweden for 1995–2003 by
the 2004 handover year's shares, and drop Scandinavia's 2004 rows where Sweden reports the series. The
clean step logs the shares, read off the raw series: refined primary 0.521, smelting primary 0.525,
mining 0.837, semis 0.618, refined and smelting secondary 0.916 and 0.914. icsg v14 differs from v13 in
exactly 114 rows: Sweden +54, Scandinavia's 54 in 1995–2003 shrunk, and 6 Scandinavia 2004 rows dropped.
The dropped rows, in kt CC, are refinery anode use 408.626, smelter concentrate use 357.720, semis use
297.400, mine output 98.300, smelter scrap 60.104 and refinery scrap 2.638.

## 2. Pass rule

| Item (fixed on #75 before the numbers) | Result |
|---|---|
| **#74, Sweden's processing:** `Ore→SXEW` CC posterior over 1996–2003 falls below 10 % of 307.8 kt CC; Sweden's ore goes through `Ore→MF`; its Mining→Ore posterior/prior in 1996–2003 does not fall below #69's values | **holds.** `Ore→SXEW` 307.8 → **0.0** kt CC. `Ore→MF` 134.3 → 761.8 kt CC, with a prior of 84–107 kt CC/yr at σ 8.4–10.7, where it was a flat-prior zero at σ 1e6. Mining→Ore posterior/prior 1996–2003 is 1.00–1.12 in every year, against 0.38–0.92 in the base. |
| **#74, the 2001 MESC move:** the ROW → Sweden MESC leg's \|z\| in 2001 falls below 3 | **holds.** \|z\| 15.67 → **0.01** (CC). The leg's prior is 2.8 kt CC, against 14.4 in the base. |
| **#73, ROW 2004:** ROW's 2004 ICSG-backed scale totals fall by Scandinavia's dropped rows (refined anode use −408.6, smelter concentrate use −357.7, semis use −297.4, mine output −98.3); ROW's 2004 sum-row z² does not rise | **fails on semis use; holds on the rest.** Anode use −408.6, smelter concentrate use −357.7 and mine output (MF supply) −98.3 kt CC, each exactly the dropped row; ROW's 2004 sum-row z² **26,099 → 609**. ROW has no semis-use scale total. Read on the edges, the ICSG fan into ROW's semis falls 1,396 → 1,099 kt CC, but the model's prior on ROW's semis legs goes 1,396 → 1,404 and its posterior 1,498 → 1,680 (§3). |
| **Netting:** Sweden 1995–2003 stays `icsg-gap`; report every other hub-year whose T changes | **holds.** baci v17 → v18: no hub-year's T or T_applied changes. Sweden's cathode 1996–2003 stays `icsg-gap`. |
| **Round-E** (ADR 0025) vs `opt_20261013_095846_46ece808` | **holds** on every item; table below. |
| **Stock legs:** net per family and per region-year, with each leg's z; no year grows into a sink | **holds.** No region-year's stock net grows in \|net\| by more than 1.5 kt CC (United States MESC 2001). Legs above \|z\| 5: 316 → 300, none new. The family nets move toward draw-downs, not toward a sink (§5). |

**Round-E, `opt_20261013_095846_46ece808` → `opt_20261013_122432_24e811b7`**, retention pooled over
1995–2024 as Σ posterior / Σ prior:

| Item | Base | New |
|---|---|---|
| (a) world retention: CO Mass / mine output (MF→Concentrate, CC) / refined cathode (CP→Cathode, CC) | 0.862 / 0.991 / 0.986 | 0.863 / 0.993 / 0.988 |
| (b) small nodes (\|c\| < 18 kt), posterior/prior median / p90 | 1.000 / 1.165 | 1.000 / 1.167 |
| (c) exporter retention, cross-region concentrate legs into smelters (CC): Chile / Peru / Indonesia / world | 0.957 / 1.010 / 0.910 / 0.937 | 0.968 / 1.011 / 0.916 / 0.941 |
| (d) sum-row z², anchor band \|c\| < 10 kt | 109,638 | 109,607 (−31) |
| (d) sum-row z², 10–100 kt | 531,942 | 526,701 (−5,241) |
| (d) sum-row z², 100–1,000 kt | 597,277 | 522,857 (−74,420) |
| (d) sum-row z², > 1,000 kt | 158,190 | 151,876 (−6,314) |
| (d) mass moved, all CC edges | 482,781 kt CC | 465,571 kt CC (**−3.56 %** vs `opt_20261013_095846_46ece808`) |

## 3. ROW: the refinery and the semis leg

ROW's refined cathode supply (`CP→Cathode`, CC), and its semis use: the ICSG fan, the model's prior and
posterior on all its semis legs, and its domestic leg's feed value, prior and posterior. kt CC.

| Year | base CP→Cathode | base semis: fan / prior / post | base domestic | new CP→Cathode | new semis: fan / prior / post | new domestic |
|---|---|---|---|---|---|---|
| 1995 | 530 / 454 | 847 / 847 / 1,010 | 102 / 102 / 275 | 404 / 426 | 679 / 912 / 988 | 0 / 232 / 257 |
| 1996 | 627 / 504 | 836 / 836 / 1,047 | 83 / 83 / 257 | 471 / 474 | 681 / 916 / 1,037 | 0 / 235 / 251 |
| 1997 | 641 / 417 | 890 / 890 / 1,000 | 31 / 31 / 125 | 481 / 469 | 726 / 976 / 1,107 | 0 / 249 / 237 |
| 1998 | 606 / 294 | 882 / 882 / 882 | 2 / 2 / 2 | 444 / 442 | 710 / 988 / 1,120 | 0 / 278 / 248 |
| 1999 | 522 / 315 | 952 / 952 / 1,058 | 25 / 25 / 93 | 374 / 368 | 772 / 977 / 1,157 | 0 / 204 / 199 |
| 2000 | 524 / 527 | 1,027 / 1,472 / 1,545 | 0 / 445 / 413 | 369 / 368 | 840 / 1,095 / 1,376 | 0 / 255 / 252 |
| 2001 | 577 / 198 | 1,000 / 1,000 / 977 | 2 / 2 / 2 | 373 / 373 | 818 / 1,089 / 1,239 | 0 / 271 / 270 |
| 2002 | 578 / 570 | 1,046 / 1,569 / 1,674 | 0 / 524 / 475 | 359 / 355 | 882 / 1,150 / 1,446 | 0 / 268 / 255 |
| 2003 | 564 / 553 | 1,150 / 1,686 / 1,769 | 0 / 536 / 475 | 341 / 336 | 972 / 1,231 / 1,529 | 0 / 259 / 243 |
| 2004 | 824 / 263 | 1,396 / 1,396 / 1,498 | 32 / 32 / 113 | 416 / 416 | 1,099 / 1,404 / 1,680 | 0 / 306 / 306 |

Σ over 1995–2004: refined cathode deleted (prior − posterior) **1,897 kt CC on the base, 5 now**. Semis
booked above the ICSG fan: prior 1,504 and posterior 2,434 on the base, prior **2,556** and posterior
**4,500** now.

**My reading of the table, not printed as such:**
- *The domestic semis leg decides whether ROW's refined cathode survives.* Where the ICSG fan gives the
  leg a value, it is covered at that value (2–102 kt CC on the base), and ROW's refined cathode is cut:
  posterior/prior 0.32–0.86 in 1995–1999, 2001 and 2004. Where the fan leaves it at 0, the leg is not
  covered, the network imputer fills it from ROW's own cathode balance, and the refined cathode is kept:
  0.98–1.01 in 2000, 2002 and 2003. That holds on the base already.
- *The split moves every year into the second case.* With Sweden's share out of ROW's semis use and the
  2004 duplicate dropped, BACI's imports into ROW exceed ROW's ICSG semis use in every year, so the fan's
  domestic weight is 0 everywhere and the leg is imputed. ROW's smaller refined output is then kept in
  full, but it lands on a semis demand ICSG does not report, and the import legs into ROW's semis rise in
  posterior as well.
- *Neither state is right.* The base deleted real refinery output; the new run books semis use ICSG does
  not report. #76 asks which a zero-weight fan leg should be.

## 4. Cost and where it sits

**Common-set sum-row z²**, 27,012 rows in both runs, built in memory: 1,397,046 → **1,311,041 (−6.16 % vs
`opt_20261013_095846_46ece808`)**. The new run also has 156 sum rows the base lacks, all Sweden's new
1995–2003 scale totals, with z² 2,099. Counting them, the fall is 83,906.

Every year whose sum-row z² moves by more than 1,000 vs `opt_20261013_095846_46ece808`:

| Year | 1995 | 1996 | 1997 | 1998 | 1999 | 2001 | 2004 | 2020 |
|---|---|---|---|---|---|---|---|---|
| Δz² | −5,127 | −3,820 | −6,068 | −11,867 | −8,778 | −21,307 | −25,209 | −1,644 |

By region, vs `opt_20261013_095846_46ece808`:

| Region | Δz² | Where |
|---|---|---|
| ROW | −79,367 | by sector: Cathode −19,081, ER −11,583, CER −11,583, CP −10,296, CFR −7,819, Anode −7,521, Smelting −4,406, Matte −3,706, Concentrate −2,288; the refinery chain (CP, ER, CER, CFR, Anode, Cathode) carries −67,883 |
| Sweden | −7,764 | mining and cathode use totals, 1996–2003 |
| China | −1,042 | |
| Zambia | **+2,082** | cathode use totals 1999–2004; 2004: c 313.1 → 280.7 kt CC, z² 177.4 → 425.8 |
| Uzbekistan | **+1,107** | cathode use totals 1996–2000 |
| Kazakhstan | +402 | |

*Reading:* the rises sit on exporters' cathode use totals. Their legs into ROW's semis are sized to the
ICSG fan, which is smaller now by Sweden's share and the 2004 duplicate.

## 5. Stock legs

The family nets move toward draw-downs: Σ net over 30 years, MESC +4.0 → −128.0, USC −169.5 → −225.4,
PSC 90.5 → 77.7 kt CC. In 1996–2004, the years the script prints per year, MESC grows in \|net\| in
1996, 1997, 1998, 1999, 2000, 2001 and 2003, and USC in 1998 and 2001. A draw-down supplies cathode, the
opposite of a sink. The largest growth per family, by region, with its legs' z²:

- **MESC 2001, −27.9 → −84.4 kt CC.** Sweden alone moves −48.8 of the −56.5: its forced +49.0 (#69 REP,
  §4) is gone, so it no longer offsets the other regions' draw-downs. The rest is spread, none by more
  than 1.5 kt CC. The 2001 MESC legs' Σz² falls 1,017.2 → 703.4, and legs above \|z\| 3 9 → 8.
- **USC 1998, +3.4 → −17.1 kt CC.** Spread over 47 regions: ROW −2.6, China −1.1, Belgium and the
  Netherlands −1.0 each, Canada and Brazil −0.9 each, and the rest −12.9 over 41 regions. The 1998 USC
  legs' Σz² falls 1,523.1 → 1,342.9, and legs above \|z\| 3 19 → 17.

Per region-year, no stock net grows in \|net\| by more than 1.5 kt CC. Stock legs above \|z\| 5 fall
316 → 300, and none is new.

## 6. Predictions (posted on #75 before Stages 2–4 ran)

| # | Prediction | Outcome |
|---|---|---|
| 1 | `Ore→SXEW` < 10 %; ore through `Ore→MF`; mining ratio above #69's in every year | **Holds.** 0.0 kt CC; ratio 1.00–1.12. |
| 2 | ROW → Sweden MESC 2001 \|z\| < 3 | **Holds,** 0.01. |
| 3 | ROW's 2004 scale totals fall by about the dropped rows; ROW's 2004 z² falls | **Partial.** Anode, concentrate and mine output fall exactly and z² falls 26,099 → 609, but ROW's semis use does not fall (§3), and its cathode use total rises 143.3 → 417.2 kt CC. |
| 4 | Common-set z² falls, led by Sweden and ROW | **Holds,** −6.16 % vs `opt_20261013_095846_46ece808`, led by ROW, with the caveat of §3. |
| 5 | Round-E holds; mass moved changes by < 1 % | **Half wrong.** Round-E holds, but mass moved falls 3.56 % vs `opt_20261013_095846_46ece808`. |
| 6 | No stock sink; no new leg above \|z\| 5 | **Holds** (§5). |

## 7. Follow-ups (not part of #75)

1. **#76: a zero-weight fan leg is refilled by the imputer.** It decides whether ROW's refined cathode is
   deleted or booked as unreported semis use. It is the failed clause's cause, and its fix will move this
   round's headline.
2. **Merging is the owner's call.** #74 is fixed and #73's duplicate is gone from the inputs; the
   failed clause is a pre-existing mechanism the split extends to every year. Merging makes
   `opt_20261013_122432_24e811b7` the reference chain. ADR 0043 is the precedent for shipping over a
   failed item by the maintainer's decision.
3. **#30 and #37** rebuild ore_grade and stock_bounds. Both read ICSG, so they should run on icsg v14.
4. **The constant 2004 shares are an estimate** (ADR 0032 amendment).

## Reproduce

```
python scripts/parse/icsg_parse/icsg_clean.py --to-feed        # icsg v14; logs the shares of §1
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
    opt_20261013_122432_24e811b7 17 18
```

The diagnostic prints every number in this REP except the solve time (the run's sidecar) and the shares
(the clean step's log).
