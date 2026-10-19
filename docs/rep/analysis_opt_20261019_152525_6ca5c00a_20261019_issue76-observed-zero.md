# #78: a non-miner's zero-weight domestic concentrate leg is an observed zero (#76, narrowed)

**Run:** `opt_20261019_152525_6ca5c00a` (label `r78b_observed_zero_nonproducer`; 30/30 Solved; `prior_mode` cold;
`solve_wall_s` 614 s per the run's sidecar). It is the second run; the first, `opt_20261019_145258_6bbfa0b5`
(label `r78_observed_zero`; 29 Solved, 2023 AlmostSolved; 622 s), measured #76's first build and was
rejected (§2).
**Against:** the reference chain `opt_20261019_142026_83bfe843` (#92, REP
[`…issue91-refined-scope`](analysis_opt_20261019_142026_83bfe843_20261019_issue91-refined-scope.md)), called
*base* below
**Chain (second run):**
- `icsg_feed.py` on `fix/observed-zero` 89fd55d → `feed_icsg_20261019_150334_42906d2e` (icsg v15, trade_shares
  v36; #76 does not touch the trade-share builder).
- With `feed_baci_20261019_135628_500e99ce` and `feed_crude_ore_20260603_095651_558f7369` →
  `imp_20261019_150618_993d9856` → `const_20261019_151044_cf8aab33` → the run.
- The first run: `feed_icsg_20261019_143004_d2d0ff50` → `imp_20261019_143336_405523cc` →
  `const_20261019_143808_e3bdff8a`.

**Feeds/params:** the base's versions (icsg v15, baci v27, trade_shares v36, acc copper v4, conc v10, eff v4,
icsg_aggregates v4, ore_grade v7, scrap_share v3, stock_bounds v5, stripping_ratio v2); only the ICSG feed run is
rebuilt.
**Date:** 2026-10-19 · **Branch:** `diag/observed-zero-measure` (from `fix/observed-zero`) · **Issues:** #78, #76,
#32; follow-up #96 · **ADR:** 0024 (amended by #76, narrowed by #78), 0025 (stop rule)

**Verdict: the narrowed build passes every item.**

- **Germany's phantom mine prior goes.** Germany's mine-chain sum rows carry Σz² 74,373 → 0 (Germany's common-set
  Δz² −74,269). The NetworkImputer no longer writes the residual of Germany's smelter balance onto its domestic
  concentrate leg (#32).
- **Common-set sum-row z² −7.0 % vs `opt_20261019_142026_83bfe843`** (1,242,441 → 1,155,371). Every anchor band
  falls. Korea −7,262 and Spain −6,329 also gain; the 54 rows only the base has (z² 57,455, Germany's, Korea's and
  Spain's mine rows) drop out with their phantom priors.
- **The first build, which stamped every zero-weight domestic leg, is rejected:** common-set z² +29.2 % (§2).

## 1. Change

#76 carries a weighted ICSG fan's domestic leg at trade-share weight 0 as an **observed zero**: covered at 0 with
σ = max(1 % of its fans' usage, 0.3 kt CC), so no imputer refills it (ADR 0024 am.).
- **The first build** stamped every such leg: 367 on this feed, 271 `Cathode→FU` and 96 `Concentrate→Smelting`.
- **The narrowed build** (89fd55d, a94cdcb) stamps a domestic concentrate leg only where the region's own ICSG
  concentrate production is 0 that year: `icsg_feed.non_producer_mask`, `OBSERVED_ZERO_SECTORS`. That is 82 legs,
  Germany 30, Korea 30, Japan 20, Spain 2.

## 2. The first build (`opt_20261019_145258_6bbfa0b5`): rejected on measurement

| Item | Result |
|---|---|
| 1 | Holds: no imputer writes an observed zero (base: 367 writes, 3,689 kt CC) |
| 2 | Holds |
| 3 | **Fails:** 22 stock legs newly above \|z\| 5 (Indonesia, Zambia, Sweden), 4 observed zeros above \|z\| 5 |
| 4 | **Fails:** common-set sum-row z² **+29.2 %** vs `opt_20261019_142026_83bfe843`, the (d) 10–100 kt band +316,234 |

- **The rises:** Italy +262,853, the United Kingdom +95,382, Bulgaria +32,336, South Africa +29,575, Spain +23,499.
  They sit on refinery and smelter rows (CP, ER, CER +108k each).
- **Their source:** the zeros in **producer** region-years. There the bound's 0 is the residual of imports above
  usage, not "the region supplies none of its own use": 25 of the 271 cathode zeros (Italy 13, the UK 4, South
  Africa 3, …) and 14 of the 96 concentrate zeros (Bulgaria, Spain). Pinning them left the region's own output
  nowhere to go.
- **Germany** was already fixed in that run (mine rows 74,373 → 0).

## 3. Pass rule (fixed on #78 at triage; judged on the second run)

| Item | Rule | Result |
|---|---|---|
| 1 | No imputer writes an observed zero | **Holds.** 0 writes. Base: 82 on these legs (`network_backward_clipped` 60, `network_backward_capped` 21, `network_backward` 1), 370 kt CC of prior |
| 2 | ROW's semis prior equals its ICSG fan | **Holds.** Within 0.023 kt CC in all 17 years 1995–2011; ROW untouched |
| 3 | No new sink: observed zeros' posterior and z; no stock leg newly above \|z\| 5 | **Holds.** The observed zeros carry Σ 73.7 kt CC of posterior (Japan 67.8), max \|z\| 0.7; no stock leg newly above \|z\| 5 (319 → 319) |
| 4 | Round-E, ROW's refined cathode reported not judged | **Holds** (§4) |

## 4. Round-E (ADR 0025), `opt_20261019_142026_83bfe843` → `opt_20261019_152525_6ca5c00a`

| Item | Base | New |
|---|---|---|
| (a) world retention: CO Mass / Concentrate CC / Cathode CC | 0.864 / 0.994 / 0.993 | 0.865 / 0.995 / 0.993 |
| (b) small nodes (\|c\| < 18 kt), posterior/prior median / p90 | 1.000 / 1.133 | 1.000 / 1.132 |
| (c) exporter retention: Chile / Peru / Indonesia / world | 0.966 / 1.011 / 0.914 / 0.935 | 0.973 / 1.013 / 0.917 / 0.938 |
| (d) sum-row z², \|c\| < 10 kt | 91,757 | 72,953 (−18,804) |
| (d) 10–100 kt | 393,169 | 342,335 (−50,833) |
| (d) 100–1,000 kt | 604,820 | 598,024 (−6,796) |
| (d) > 1,000 kt | 152,696 | 142,059 (−10,636) |
| (d) mass moved, all CC edges | 467,096 kt CC | 468,968 kt CC (+0.40 % vs `opt_20261019_142026_83bfe843`) |

- **(a)–(c) hold;** exporter retention rises by at most +0.007 (Chile).
- **(d) holds;** every band falls. Mass moved is reported, not judged.
- **ROW's `CP→Cathode`** (reported): 2012–2024 posterior/prior 0.94–1.05 in both runs, and ROW's common-set Δz² is
  +187. ROW is untouched; the first build had cut it to 0.85 in 2015.

## 5. Germany (#32) and the same class

| | base | new |
|---|---|---|
| mine-chain sum rows Σz² (Ore, MF, Concentrate, Mining) | 74,373 | 0 |
| by year: 2017 / 2018 / 2020 / 2021 / 2022 | 19,835 / 20,517 / 6,731 / 20,508 / 6,780 | 0 each |
| domestic concentrate into smelting, Σ prior / post | 185 / 184 | 0 / 3 |
| imported concentrate into smelting, Σ prior / post | 9,383 / 8,972 | 9,383 / 9,033 |
| smelter scrap, Σ prior / post | 7,142 / 7,225 | 7,142 / 7,273 |
| CO→Mining, Σ prior / post | 159.7 / 503.7 | 1.4 / 388.1 |
| Ore→SXEW, Σ post | 287.1 | 381.2 |

(Σ over 1995–2024, kt CC.)
- **The smelter residual** moves onto the import legs (+61 kt CC posterior) and scrap (+48). Germany's smelter-side
  rows rise by only +50 (Smelting) and +37 (Matte).
- **Germany's posterior mine chain does not vanish.** On a prior of 1.4 kt CC, the solver still carries 388 kt CC of
  German mine output, almost all of it a phantom SX-EW route (Ore→SXEW 381). That is a no-data chain the solver
  builds without any prior, not the imputer residual this ticket removes. Reported, not judged; follow-up below.
- **The same class:** Korea's mine-chain Σz² 10,773 → 1,555; Spain's 19,291 → 12,958; Slovakia's (8,991) and
  Botswana's (9,289) are unchanged. Neither of the last two has a zero-weight concentrate leg.

## 6. Common set

- **Σz² 1,242,441 → 1,155,371 (−7.008 % vs `opt_20261019_142026_83bfe843`)**, 27,458 rows in both runs.
- **Rows only one run has:** the 54 rows only in base carry z² 57,455 (Germany 26,706, Korea 19,652, Spain 11,097);
  none are new. Counting them, the total goes 1,299,896 → 1,155,371 (−11.1 %).
- **By region:**
  - falls: Germany −74,269, Korea −7,262, Spain −6,329, Chile −731;
  - rises: UAE +555, Malaysia +263, Peru +202, ROW +187.
- **By sector:** Ore −40,875, Concentrate −24,082, MF −23,911, Mining −1,510; Cathode +1,187, Smelting +1,007,
  Matte +948.
- **By year:** the gains sit in 1997 (−9,187), 2003 (−6,229) and 2016–2022 (2017 −19,653, 2018 −19,941, 2021
  −20,395). 2024 rises +2,314.

## 7. Predictions

| Prediction (posted on #78 before each run) | Outcome |
|---|---|
| First run: item 4 holds, common-set z² falls 4–8 % | **Wrong:** +29.2 %; the named risk (producer regions' refined output) occurred at scale |
| Second run: items 1–3 hold | **Right** |
| Second run: z² falls 4–7 %; Germany ≈ −74k; Germany's smelter rows rise < 5k | **Right on Germany** (−74,269) and its smelter rows (+50, +37); **wrong on size:** −7.0 %, at the edge |
| Germany's posterior mine output falls to ≈ 0 in every year | **Wrong:** 388 kt CC remains, almost all SX-EW (§5) |

## 8. Follow-ups

- **#96:** the 246 cathode zeros in non-producer region-years. Keying the test on ADR 0024's raw ICSG production
  would extend the observed zero to them.
- **Germany's phantom SX-EW chain** (Ore→SXEW 381 kt CC on a prior of 1.4): a no-data chain. The #32 triage counted
  906 kt CC of such posterior chains in regions ICSG reports no mine for. #32 closes on this REP with the chain
  noted.

## Reproduce

```
# second run, on fix/observed-zero 89fd55d (a94cdcb is behaviour-preserving)
python scripts/icsg_feed.py                                       # → feed_icsg_20261019_150334_42906d2e
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261019_150334_42906d2e \
    --feed-id feed_baci_20261019_135628_500e99ce --feed-id feed_crude_ore_20260603_095651_558f7369 \
    --label r78b_observed_zero_nonproducer
python scripts/constraints.py --imp-run-id imp_20261019_150618_993d9856 r78b_observed_zero_nonproducer
python scripts/run_solver.py --const-run-id const_20261019_151044_cf8aab33 --label r78b_observed_zero_nonproducer
python scripts/explo/issue78_observed_zero_diag.py opt_20261019_142026_83bfe843 opt_20261019_152525_6ca5c00a
# first run: the same chain on fix/observed-zero 3a9e4b0, then
python scripts/explo/issue78_observed_zero_diag.py opt_20261019_142026_83bfe843 opt_20261019_145258_6bbfa0b5
```

The diagnostic prints every number in this REP or the numbers it is summed from, except the solve times, which
come from the runs' sidecars, and the first run's cause split (25 cathode, 14 concentrate zeros in producer
region-years), which comes from the narrowed build's review (posted on #78).
