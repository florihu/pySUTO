# #69: Sweden's cathode un-netted in 1996–2003 (#67)

**Run:** `opt_20261013_095846_46ece808` (label `r69_aggregate_gap`; 30/30 Solved; `prior_mode` cold;
1,825 s of solve)
**Against:** the reference chain `opt_20261012_135741_67851aa1` (#65, REP
[`…issue65-icsg-stock-parse`](analysis_opt_20261012_135741_67851aa1_20261012_issue65-icsg-stock-parse.md))
**Chain:** baci v13 netted against icsg v13 with #67 → baci **v17** (`ae1ac629`; `weights.csv` hash =
v16's) → `feed_baci_20261013_090735_16160de7`; trade_shares **v31** (`81b8871f`, one pass, exact since
#66) → `feed_icsg_20261013_090857_576f8330`; + `feed_crude_ore_20260603_095651_558f7369` →
`imp_20261013_091151_8f1e8a4d` → `const_20261013_091634_bf0829db` → the run
**Feeds/params:** icsg v13, baci **v17**, trade_shares **v31**, icsg_aggregates **v2** (new), acc copper v4,
conc v10, eff v4, ore_grade v7, stock_bounds v5, stripping_ratio v2. These are the base's versions except
the ones in bold.
**Date:** 2026-10-13 · **Branch:** `diag/aggregate-gap-measure` · **Issues:** #69, #67 · **ADR:** 0027
(amended by #67), 0025 (stop rule), 0043

**Verdict: passes. Every item of the rule fixed on #69 holds**, and common-set sum-row z² falls
**−0.69 % vs `opt_20261012_135741_67851aa1`**. Two findings cut against the intent of the round. Both have
one cause, which this round exposes but does not create.

- **Sweden's mine output gains its outlet through an SX-EW plant Sweden does not have.** The rise in its
  mining ratio, 0.41 → 0.70 in 2003, travels entirely on `Ore→SXEW`, whose prior is 0 in every year.
  Over 1996–2003 that route carries 307.8 kt CC, against 175.0 in the base, which already had it.
- **Sweden's exchange stock absorbs ROW's cathode in 2001.** The MESC net there is +48.9 kt CC above its
  prior, on the ROW → Sweden leg. The pre-registered stock item reads the net per family and year, where
  other regions' small draws hide it.
- **The cause.** In 1995–2003 ICSG reports Sweden's smelting and refining inside "Scandinavia", which
  maps to ROW. Sweden keeps a mine (crude-ore feed) and its trade (BACI), but has no processing in those
  years. #67 restores the trade and so asks the model for Swedish cathode. The model builds it from
  Swedish ore through SX-EW, or parks ROW's cathode in a Swedish stock.

## 1. Change

#67 (merged `4c2f7d8`) makes a region-year whose ICSG flows sit inside an aggregate row a reporting gap
for the ADR 0027 netting, whatever stock rows it carries itself. The membership is declared in the new
params input `icsg_aggregates` (v2: Sweden ⊂ Scandinavia, 1995–2003). Before, Sweden's lone LME changes
gave it an intake `U`, and its cathode was netted as a pass-through hub in every year 1996–2003.

## 2. Pass rule

| Item (fixed on #69 before the numbers) | Result |
|---|---|
| **Netting:** the ledger's netted volume falls by the Sweden rows alone, 220.9 kt CC over 1996–2003, and no other hub-year's T changes | **holds.** baci v16 → v17: T changes on Sweden's 8 cathode hub-years only, Σ ΔT −220.9 kt CC, all now `icsg-gap` with T = 0. Σ T_applied falls 215.3: Sweden 2002 had applied 31.3 of its 36.4, and 25 other hub-years' T_applied move by ≤ 0.28 kt CC through hub→hub chains (United Kingdom 2002 the largest). |
| **Sweden:** Mining→Ore CC posterior/prior in 2003 rises from 0.41; sum-row z² over 1996–2003 falls | **holds.** 0.41 → 0.70. Sum-row z² 17,062 → 11,008 (−6,055), falling in every year but 2002 (+299). See §5 for what carries the rise. |
| **Singapore:** sum-row z² within ±1 % of base | **holds.** 164 rows, 1,532 → 1,530 (−0.087 %). |
| **Round-E** (ADR 0025) vs base | **holds** on all four items; table below. |
| **Stock legs:** MESC, USC and PSC net per year show no year growing into a sink | **holds as written.** Per family, the largest growth of \|net\| is +10.4 kt CC (USC 2001, −46.0 → −56.4), spread over many regions, none by more than 1.1 kt CC. Per region, Sweden's MESC 2001 grows +48.9 kt CC (§5). |

**Round-E, `opt_20261012_135741_67851aa1` → `opt_20261013_095846_46ece808`:**

| Item | Base | New |
|---|---|---|
| (a) world retention, CO Mass / Concentrate CC / Cathode CC | 0.862 / 0.991 / 0.986 | 0.862 / 0.991 / 0.986 |
| (b) small nodes, posterior/prior median / p90 | 1.000 / 1.165 | 1.000 / 1.165 |
| (c) exporter retention, Chile / Peru / Indonesia / world | 0.957 / 1.009 / 0.910 / 0.937 | 0.957 / 1.010 / 0.910 / 0.937 |
| (d) Δz² by anchor band \|c\| < 10 / 10–100 / 100–1,000 / > 1,000 kt | — | +478 / −3,360 / −3,970 / −2,911 |
| (d) mass moved, all CC edges | 482,769 kt | 482,781 kt (+0.00 % vs base) |

The < 10 kt band's rise is one row: Sweden's cathode use total in Mass, 2001, +705 z². Its anchor moved
from 3.3 to 52.7 kt because the un-netted imports now enter it. It is not a repricing.

## 3. Cost and where it sits

**Common-set sum-row z²**, 26,998 rows in both runs, built in memory: 1,406,344 → **1,396,581 (−0.69 % vs
`opt_20261012_135741_67851aa1`)**.

| Year | 1996 | 1997 | 1998 | 2000 | 2001 | 2002 | 2003 |
|---|---|---|---|---|---|---|---|
| Δz² vs base | −349 | −740 | −718 | −478 | −6,752 | +433 | −1,125 |

| Region | Δz² | Where |
|---|---|---|
| Sweden | −6,055 | mining supply totals in every year; 2001 alone −2,223 |
| ROW | −4,517 | 2001: its ER and CER totals −490 each (4 rows) and its cathode use total in CC −493; its cathode use total in Mass +951 |
| United Kingdom | +578 | 1996: its CP, ER and CER totals +90 each |
| China | +160 | mining totals in 2017 and 2019, small moves of both signs |

## 4. Stock legs

Per family, no year grows. Σ net over 30 years: MESC −15.9 → +4.0, USC −128.8 → −169.5, PSC 92.2 →
90.5 kt CC. USC's 2001 move is spread: Austria, China, France, Germany, the Netherlands and ROW each move
by about −1 kt CC, and the rest of the −10.4 kt CC is spread over regions that move by less.

Per region, one year stands out: **Sweden's MESC in 2001, net 0.1 → 49.0 kt CC.** The family net hides
it because MESC 2001 is −69.4 in the base and −27.9 in the new run. ICSG reports a +21.8 kt build for
Sweden that year.

## 5. Sweden: what carries the restored trade

Sweden 2001 and 2003, kt CC, prior / posterior:

| Leg group | 2001 base | 2001 new | 2003 base | 2003 new |
|---|---|---|---|---|
| imports into Sweden's FU | 0 / 0 | 48.5 / 50.8 | 0 / 0 | 47.7 / 47.0 |
| imports into Sweden's MESC | 0 / 0 | 21.3 / **70.2** | 0 / 0 | −18.8 / −19.2 |
| Sweden's cathode exports | 1.2 / 4.4 | 18.6 / 27.6 | 18.7 / 24.8 | 48.1 / 45.7 |
| Sweden's own cathode, CSXEW→CP | 0 / 6.6 | 0 / 26.6 | 0 / 24.2 | 0 / 41.2 |

- **The MESC sink is ROW's cathode.** The ROW → Sweden MESC leg goes from a prior of 14.4 to a posterior
  of 63.5 kt CC in 2001. In the same year ROW's ER and CER totals lose z²: Scandinavia's refined output
  is booked in ROW, and the leg gives that cathode an outlet.
- **The exports are supplied by SX-EW.** Sweden's concentrate route (`Ore→MF`) has no prior before 2004,
  because ICSG's Swedish mining starts then. `Ore→SXEW` has no prior in any year. The solver sends the
  extra ore to SX-EW: 1996–2003 posterior 175.0 → 307.8 kt CC. `Ore→MF` falls slightly, for example
  2002 23.5 → 13.7. The base already did this (40.3 kt CC in 2003, 35.1 in 2005 when Sweden reports its
  own mine), so the route predates #67. #67 enlarges it.
- **The mining ratio rises for that reason.** Mining→Ore CC posterior/prior goes up in every year
  1996–2003 (2001: 0.22 → 0.57, 2003: 0.41 → 0.70). This is the outlet #67 predicted, delivered through
  a process Sweden does not run.

## 6. Predictions (posted on #69 before Stages 2–4 ran)

| # | Prediction | Outcome |
|---|---|---|
| 1 | Sweden 2003 mining ratio rises from 0.41 to ≥ 0.69 | **Holds,** 0.70. It travels on SX-EW (§5). |
| 2 | Sweden's 1996–2003 sum-row z² falls by more than 2,636 | **Holds,** −6,055. |
| 3 | Singapore within ±0.1 % | **Holds,** −0.087 %. |
| 4 | Round-E holds, mass moved changes < 1 % | **Holds,** +0.00 %. |
| 5 | Common-set z² falls; the largest moves are Sweden and its cathode partners | **Holds,** −0.69 %. Sweden and ROW lead, and the United Kingdom is the largest rise. |
| 6 | No stock year grows into a sink | **Holds per family; wrong per region:** Sweden's MESC 2001 grows +48.9 kt CC. |

## 7. Follow-ups (not part of #69)

1. **Sweden's processing in 1995–2003 sits in ROW.** ICSG's "Scandinavia" smelting and refining rows map
   to ROW, so Sweden has a mine and trade but no smelter or refinery in those years. #67 restores the
   trade, and the model answers with a Swedish SX-EW chain and a 2001 MESC sink. The fix belongs where
   #73 sits, in how the ICSG feed treats an aggregate row whose members the registry now declares.
   Not filed.
2. **#73**, the 2004 double count of the same rows, stays open for triage.
3. **The stock item reads per family.** A regional sink can pass it, as Sweden 2001 does here. A future
   rule could read the net per region and year.

## Reproduce

```
python scripts/build_baci_reexport.py --source-version 13      # baci v17
python scripts/baci_feed.py
python scripts/build_trade_shares.py                            # trade_shares v31, one pass (#66)
python scripts/icsg_feed.py
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261013_090857_576f8330 \
    --feed-id feed_baci_20261013_090735_16160de7 --feed-id feed_crude_ore_20260603_095651_558f7369 \
    --label r69_aggregate_gap
python scripts/constraints.py --imp-run-id imp_20261013_091151_8f1e8a4d r69_aggregate_gap
python scripts/run_solver.py --const-run-id const_20261013_091634_bf0829db --label r69_aggregate_gap
python scripts/explo/issue69_aggregate_gap_diag.py opt_20261012_135741_67851aa1 \
    opt_20261013_095846_46ece808 16 17
```
