# #69: Sweden's cathode un-netted in 1996–2003 (#67)

**Run:** `opt_20261013_095846_46ece808` (label `r69_aggregate_gap`; 30/30 Solved; `prior_mode` cold;
1,825 s of solve)
**Against:** the reference chain `opt_20261012_135741_67851aa1` (#65, REP
[`…issue65-icsg-stock-parse`](analysis_opt_20261012_135741_67851aa1_20261012_issue65-icsg-stock-parse.md)),
called *base* below
**Chain:** baci v13 netted against icsg v13 with #67 → baci **v17** (`ae1ac629`; `weights.csv` hash =
v16's) → `feed_baci_20261013_090735_16160de7`; trade_shares **v31** (`81b8871f`, one pass, exact since
#66) → `feed_icsg_20261013_090857_576f8330`; + `feed_crude_ore_20260603_095651_558f7369` →
`imp_20261013_091151_8f1e8a4d` → `const_20261013_091634_bf0829db` → the run
**Feeds/params:** icsg v13, baci **v17**, trade_shares **v31**, icsg_aggregates **v2** (new), acc copper v4,
conc v10, eff v4, ore_grade v7, stock_bounds v5, stripping_ratio v2. These are the base's versions except
the ones in bold.
**Date:** 2026-10-13 · **Branch:** `diag/aggregate-gap-measure` · **Issues:** #69, #67; follow-ups #74,
#73 · **ADR:** 0027 (amended by #67), 0025 (stop rule), 0043

**Verdict: passes. Every item of the rule fixed on #69 holds**, and common-set sum-row z² falls
**−0.69 % vs `opt_20261012_135741_67851aa1`**. Two findings cut against the intent of the round. Both
have one cause, which this round exposes but does not create.

- **Sweden's mine output gains its outlet through an SX-EW plant Sweden does not have.** Its mining
  ratio rises from 0.41 to 0.70 in 2003. The rise travels on `Ore→SXEW`, a flat-prior zero at σ_a = 1e6
  and so a **sink** in GLOSSARY's sense: over 1996–2003 it carries 307.8 kt CC, against 175.0 in the
  base, which already had it. It is not a stock leg, so the stock item does not test it.
- **A forced move on one stock leg.** In 2001 the ROW → Sweden MESC leg goes from a prior of 14.4 to a
  posterior of 63.5 kt CC, at σ 3.13 and z 15.7. That is a contradiction surfacing as z rather than a
  sink. Its z² sits outside the sum-row metric.
- **The cause.** In 1995–2003 ICSG reports Sweden's smelting and refining inside "Scandinavia", which
  maps to ROW. Sweden keeps a mine (crude-ore feed) and its trade (BACI), but has no processing in those
  years. #67 restores the trade and so asks for Swedish cathode, which the model builds from Swedish ore
  through SX-EW, or takes from ROW's refined output into a Swedish stock. Filed as **#74**.

## 1. Change

#67 (merged `4c2f7d8`) makes a region-year whose ICSG flows sit inside an aggregate row a reporting gap
for the ADR 0027 netting, whatever stock rows it carries itself. The membership is declared in the new
params input `icsg_aggregates` (v2: Sweden ⊂ Scandinavia, 1995–2003). Before, Sweden's lone LME changes
gave it an intake `U`, and its cathode was netted as a pass-through hub in every year 1996–2003.

## 2. Pass rule

| Item (fixed on #69 before the numbers) | Result |
|---|---|
| **Netting:** the ledger's netted volume falls by the Sweden rows alone, 220.9 kt CC over 1996–2003, and no other hub-year's T changes | **holds.** baci v16 → v17: T changes on Sweden's 8 cathode hub-years only, Σ ΔT −220.9 kt CC, all now `icsg-gap` with T = 0. Σ T_applied falls 215.3 kt CC: Sweden 2002 had applied 31.3 of its 36.4, and 25 other hub-years' T_applied move through hub→hub chains, the largest United Kingdom 2002 by +0.277 kt CC. |
| **Sweden:** Mining→Ore CC posterior/prior in 2003 rises from 0.41; sum-row z² over 1996–2003 falls | **holds.** 0.41 → 0.70. Sum-row z² 17,062 → 11,008 (−6,055 vs base), falling in every year but 2002 (+299). §5 shows what carries the rise. |
| **Singapore:** sum-row z² within ±1 % of base | **holds.** 164 rows, 1,532 → 1,530 (−0.087 % vs `opt_20261012_135741_67851aa1`). |
| **Round-E** (ADR 0025) vs `opt_20261012_135741_67851aa1` | **holds** on every item: world retention, small nodes, exporters, z² by anchor band and mass moved. Table below. |
| **Stock legs:** MESC, USC and PSC net per year show no year growing into a sink vs base | **holds.** Per family, the largest growth of \|net\| is +10.4 kt CC (USC 2001, −46.0 → −56.4). Per region, the largest move is Sweden's MESC in 2001, net 0.1 → 49.0 kt CC. It is a penalised move at z 15.7, not a sink (§4). |

**Round-E, `opt_20261012_135741_67851aa1` → `opt_20261013_095846_46ece808`:**

| Item | Base | New |
|---|---|---|
| (a) world retention, CO Mass / Concentrate CC / Cathode CC | 0.862 / 0.991 / 0.986 | 0.862 / 0.991 / 0.986 |
| (b) small nodes, posterior/prior median / p90 | 1.000 / 1.165 | 1.000 / 1.165 |
| (c) exporter retention, Chile / Peru / Indonesia / world | 0.957 / 1.009 / 0.910 / 0.937 | 0.957 / 1.010 / 0.910 / 0.937 |
| (d) sum-row z², anchor band \|c\| < 10 kt | 110,119 | 110,597 (+478) |
| (d) sum-row z², 10–100 kt | 535,689 | 532,329 (−3,360) |
| (d) sum-row z², 100–1,000 kt | 599,436 | 595,465 (−3,970) |
| (d) sum-row z², > 1,000 kt | 161,101 | 158,190 (−2,911) |
| (d) mass moved, all CC edges | 482,769 kt | 482,781 kt (+0.00 % vs `opt_20261012_135741_67851aa1`) |

The < 10 kt band's rise is one row: Sweden's cathode use total in Mass, 2001, +705 z². Its anchor moved
from 3.3 to 52.7 kt because the un-netted imports now enter it. It is not a repricing.

## 3. Cost and where it sits

**Common-set sum-row z²**, 26,998 rows in both runs, built in memory: 1,406,344 → **1,396,581 (−0.69 % vs
`opt_20261012_135741_67851aa1`)**.

Every year whose sum-row z² moves by more than 150 vs base:

| Year | 1995 | 1996 | 1997 | 1998 | 1999 | 2000 | 2001 | 2002 | 2003 | 2014 | 2015 | 2017 | 2019 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Δz² | −185 | −349 | −740 | −718 | −199 | −478 | −6,752 | +433 | −1,125 | +169 | +246 | −372 | +228 |

The regions moving most, vs base:

| Region | Δz² | Where |
|---|---|---|
| Sweden | −6,055 | mining supply totals, largest in 2001 (CC −1,463, Mass −1,362) |
| ROW | −4,517 | 2001: its ER and CER totals −490 each (4 rows) and its cathode use total in CC −493; its cathode use total in Mass +951 |
| United Kingdom | +578 | 1996: its CP, ER and CER totals +90 each |
| China | +160 | mining totals in 2017 and 2019, small moves of both signs |
| Russia | +123 | |
| United States | −113 | |

## 4. Stock legs

Per family, no year grows into a sink. Σ net over 30 years: MESC −15.9 → +4.0, USC −128.8 → −169.5, PSC
92.2 → 90.5 kt CC. USC's 2001 move, −10.4 kt CC, is spread: Austria, China, France, Germany, the
Netherlands and ROW each move by about −1 kt CC, and the rest is spread over regions that move by less.
MESC's 2001 net moves +41.5 kt CC signed, −69.4 → −27.9, toward zero, which is why \|net\| reads as
shrinking.

Per region, one move stands out: **Sweden's MESC in 2001, net 0.1 → 49.0 kt CC.** ICSG reports a
+21.8 kt CC build for Sweden that year. Almost all of it is one leg, ROW → Sweden, prior 14.4, posterior
63.5 kt CC, σ 3.13, **z 15.7**, and the same in Mass. GLOSSARY defines a sink as an edge family whose σ
makes its penalty negligible against the mass it absorbs. This leg pays z 15.7, so it is a contradiction
surfacing as z rather than a sink. On the base the same leg had a prior of 0 and σ 0.05.

## 5. Sweden: what carries the restored trade

Sweden's cathode legs in CC, prior / posterior:

| Leg group | 2001 base | 2001 new | 2003 base | 2003 new |
|---|---|---|---|---|
| imports into Sweden's FU | 0.0 / 0.1 | 49.0 / 51.4 | 0.0 / 0.1 | 47.9 / 47.3 |
| imports into Sweden's MESC | 0.0 / 0.1 | 21.8 / **70.8** | 0.0 / 0.0 | −19.3 / −19.7 |
| imports into Sweden's USC | 0.0 / 0.1 | 0.0 / −0.6 | 0.0 / 0.0 | 0.0 / −1.0 |
| Sweden's cathode exports | 1.3 / 6.6 | 19.3 / 26.9 | 19.1 / 25.9 | 48.9 / 44.5 |
| Sweden's own cathode, CSXEW→CP | 0.0 / 6.6 | 0.0 / 26.6 | 0.0 / 24.2 | 0.0 / 41.2 |

- **The exports are supplied by SX-EW.** Sweden's `Ore→MF` and `Ore→SXEW` edges are both flat-prior
  zeros at σ_a = 1e6 in 1996–2003. `Ore→MF` gets its prior only from 2004, when ICSG's Swedish mining
  starts. `Ore→SXEW` has a prior of 0 in all 30 years. The solver sends the extra ore to SX-EW: over
  1996–2003 its posterior goes 175.0 → 307.8 kt CC while `Ore→MF` goes 149.3 → 134.3. The base already
  used the route, for example 40.3 kt CC in 2003 and 35.1 in 2005, when Sweden reports its own mine.
  So #67 does not create it, but it enlarges it.
- **The mining ratio rises for that reason.** Mining→Ore CC posterior/prior goes up in every year
  1996–2003 (2001: 0.22 → 0.57, 2003: 0.41 → 0.70). This is the outlet #67 predicted, delivered through
  a process Sweden does not run.
- **The MESC move takes ROW's cathode** (§4). In the same year ROW's ER and CER totals lose z²:
  Scandinavia's refined output is booked in ROW, and the leg gives it an outlet.

## 6. Predictions (posted on #69 before Stages 2–4 ran)

| # | Prediction | Outcome |
|---|---|---|
| 1 | Sweden 2003 mining ratio rises from 0.41 to ≥ 0.69 | **Holds,** 0.70. It travels on SX-EW (§5). |
| 2 | Sweden's 1996–2003 sum-row z² falls by more than 2,636 | **Holds,** −6,055 vs `opt_20261012_135741_67851aa1`. |
| 3 | Singapore within ±0.1 % | **Holds,** −0.087 % vs `opt_20261012_135741_67851aa1`. |
| 4 | Round-E holds, mass moved changes < 1 % | **Holds,** +0.00 % vs `opt_20261012_135741_67851aa1`. |
| 5 | Common-set z² falls; the largest moves are Sweden and its cathode partners | **Holds,** −0.69 % vs `opt_20261012_135741_67851aa1`. Sweden and ROW lead, and the United Kingdom is the largest rise. |
| 6 | No stock year grows into a sink | **Holds.** Sweden's MESC 2001 takes a penalised +49.0 kt CC at z 15.7 (§4). The sink this round leans on is the SX-EW route, which is not a stock leg (§5). |

## 7. Follow-ups (not part of #69)

1. **#74: Sweden's processing in 1995–2003 sits in ROW.** That is the cause of both findings. One fix
   is to allocate Scandinavia's smelting and refining rows to their members through the `icsg_aggregates`
   declaration instead of to ROW.
2. **#73**, the 2004 double count of the same rows, stays open for triage.
3. **The stock item reads per family.** A move on one region's stock leg can hide in the family net, as
   Sweden's 2001 MESC does here. A future rule could read the net per region and year, with the leg's z.

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
    opt_20261013_095846_46ece808 16 17                          # every number in this REP
```
