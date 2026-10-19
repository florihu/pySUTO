# #78: a non-miner's zero-weight domestic concentrate leg is an observed zero (#76, narrowed)

**Run:** `opt_20261019_152525_6ca5c00a` (label `r78b_observed_zero_nonproducer`; 30/30 Solved; `prior_mode` cold;
`solve_wall_s` 614 s per the run's sidecar). It is the **second** run, of a build narrowed **after** the first run
failed. The first, `opt_20261019_145258_6bbfa0b5` (label `r78_observed_zero`; 29 Solved, 2023 AlmostSolved;
622 s), measured #76's original build and is rejected (§2).
**Against:** the reference chain `opt_20261019_142026_83bfe843` (#92, REP
[`…issue91-refined-scope`](analysis_opt_20261019_142026_83bfe843_20261019_issue91-refined-scope.md)), called
*base* below
**Chain (second run):**
- `icsg_feed.py` on `fix/observed-zero` 89fd55d (a94cdcb renames the rule without changing the stamped set) →
  `feed_icsg_20261019_150334_42906d2e` (icsg v15, trade_shares v36; #76 does not touch the trade-share builder).
- With `feed_baci_20261019_135628_500e99ce` and `feed_crude_ore_20260603_095651_558f7369` →
  `imp_20261019_150618_993d9856` → `const_20261019_151044_cf8aab33` → the run.
- The first run: `feed_icsg_20261019_143004_d2d0ff50` (on 3a9e4b0) → `imp_20261019_143336_405523cc` →
  `const_20261019_143808_e3bdff8a`.

**Feeds/params:** the base's versions (icsg v15, baci v27, trade_shares v36, acc copper v4, conc v10, eff v4,
icsg_aggregates v4, ore_grade v7, scrap_share v3, stock_bounds v5, stripping_ratio v2); only the ICSG feed run is
rebuilt.
**Date:** 2026-10-19 · **Branch:** `diag/observed-zero-measure` (from `fix/observed-zero`) · **Issues:** #78, #76,
#32; follow-up #96 · **ADR:** 0024 (amended by #76, narrowed by #78), 0025 (stop rule)

**Verdict: the narrowed build passes items 1, 3 and 4. Item 2 fails as worded in one year, 2015, and identically in
the base.** ROW's 2015 semis prior sits 281.7 kt CC above its ICSG fan in both runs. ROW refines copper, so the
narrowed rule leaves its zero-weight 2015 leg to the imputer by design (§3). **Ships by the maintainer's decision**
(2026-10-19, delegated to the agent for the manuscript-ready week): `opt_20261019_152525_6ca5c00a` becomes the
reference chain.

- **Germany's phantom mine prior goes.** Germany's mine-chain sum rows carry Σz² **101,078 → 0** over all rows
  (74,373 → 0 on the rows both runs have). The NetworkImputer no longer writes the residual of Germany's smelter
  balance onto its domestic concentrate leg (#32).
- **Common-set sum-row z² −7.0 % vs `opt_20261019_142026_83bfe843`** (1,242,441 → 1,155,371, 27,458 rows). Every
  anchor band falls. The 54 rows only the base has (z² 57,455: Germany, Korea and Spain's mine rows, and Japan's at
  0) drop out with their phantom priors. Counting them, the total goes 1,299,896 → 1,155,371 (−11.1 %).
- **The second build is post hoc.** It was designed after the first run showed where the first build failed. Its
  criterion is ADR 0024's own rule, a non-producer's zero, not a choice of regions. The pass rule is unchanged,
  predictions were posted before its Stage 2, and both runs are reported. The scope change is recorded on #76.

## 1. Change

#76 carries a weighted ICSG fan's domestic leg at trade-share weight 0 as an **observed zero**: covered at 0 with
σ = max(1 % of its fans' usage, 0.3 kt CC), so no imputer refills it (ADR 0024 am.).
- **The first build** stamped every such leg: 367 on this feed, 271 `Cathode→FU` and 96 `Concentrate→Smelting`.
- **The narrowed build** (89fd55d, a94cdcb) stamps a domestic concentrate leg only where the region's own ICSG
  concentrate production is 0 that year (`icsg_feed.non_producer_mask`, `OBSERVED_ZERO_SECTORS`): 82 legs, Japan
  20, Germany 30, Korea 30, Spain 2.

## 2. The first build (`opt_20261019_145258_6bbfa0b5`): rejected on measurement

| Item | Result |
|---|---|
| 1 | Holds: no imputer writes an observed zero (base: 367 writes, 3,689 kt CC) |
| 2 | Holds over 1995–2011 |
| 3 | **Fails:** 22 stock legs newly above \|z\| 5 (Indonesia, Zambia, Sweden), 4 observed zeros above \|z\| 5 |
| 4 | **Fails:** common-set sum-row z² 1,228,580 → 1,586,970 (27,410 rows), **+29.2 %** vs `opt_20261019_142026_83bfe843`; the (d) 10–100 kt band +316,234 |

- **The rises:** Italy +262,853, the United Kingdom +95,382, Bulgaria +32,336, South Africa +29,575, Spain +23,499.
  They sit on refinery and smelter rows (CP, ER, CER +108k each).
- **Their source:** the zeros in **producer** region-years, read off ADR 0024's raw ICSG production. There the
  bound's 0 is the residual of imports above usage, not "the region supplies none of its own use":
  - 25 of the 271 cathode zeros: Italy 13, the United Kingdom 4, South Africa 3, Brazil 2, ROW, Slovakia and Zambia
    1 each;
  - 14 of the 96 concentrate zeros: Spain 8, Bulgaria 3, India 2, the Philippines 1.

  Pinning them left the region's own output nowhere to go.
- **Germany** was already fixed in that run: mine rows 74,373 → 0 on the common set.

## 3. Pass rule (fixed on #78 at triage; judged on the second run)

| Item | Rule | Result |
|---|---|---|
| 1 | No imputer writes an observed zero | **Holds.** 0 writes. Base: 82 on these legs (`network_backward_clipped` 60, `network_backward_capped` 21, `network_backward` 1), 370 kt CC of prior |
| 2 | ROW's semis prior within 0.5 kt CC of its ICSG fan in every year | **Fails as worded in 2015 only:** +281.7 kt CC, identical in the base. 29 of 30 years hold (1995–2011 within 0.023) |
| 3 | No new sink: every observed zero's posterior and z; no stock leg newly above \|z\| 5 | **Holds.** Observed zeros by region (legs / Σ post kt CC / max \|z\|): Japan 20/67.8/0.73, Germany 30/2.9/0.25, Korea 30/2.6/0.61, Spain 2/0.4/0.14. No stock leg newly above \|z\| 5 (319 → 319) |
| 4 | Round-E, ROW's refined cathode reported not judged | **Holds** (§4) |

- **Item 2's 2015.** ROW's domestic semis leg is fed 0 that year (it has weight 0 in the trade shares), and the
  imputer refills it. ROW produces refined copper, so under the narrowed rule that zero is a producer's residual and
  stays imputable. It is #76's original mechanism, kept on purpose. #96 would not change it either, since #96
  concerns non-producers.

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
- **ROW's refined cathode** (reported, not judged): `CP→Cathode` posterior/prior 0.95–1.20 in 1995–2011 and
  0.94–1.05 in 2012–2024, in both runs. ROW's common-set Δz² is +187, with CP, ER and CER at −0 each. ROW is
  untouched; the first build had cut it to 0.85 in 2015.

## 5. Germany (#32) and the same class

| | base | new |
|---|---|---|
| mine-chain sum rows (Ore, MF, Concentrate, Mining), all rows: Σz² | 101,078 | 0 |
| by year: 2017 / 2018 / 2020 / 2021 / 2022 | 19,835 / 20,517 / 20,002 / 20,508 / 20,214 | 0 each |
| domestic concentrate into smelting, Σ prior / post | 185 / 184 | 0 / 3 |
| imported concentrate into smelting, Σ prior / post | 9,383 / 8,972 | 9,383 / 9,033 |
| smelter scrap, Σ prior / post | 7,142 / 7,225 | 7,142 / 7,273 |
| anode exports, Σ prior / post | 348 / 295 | 348 / 295 |
| CO→Mining, Σ prior / post | 159.7 / 503.7 | 1.4 / 388.1 |
| Ore→SXEW, Σ prior / post | 0.1 / 287.1 | 0.5 / 381.2 |

(Σ over 1995–2024, kt CC.)
- **The smelter residual** moves onto the import legs (+61 kt CC posterior) and scrap (+48). Germany's smelter-side
  rows rise by +50 (Smelting) and +37 (Matte); its Anode rows by +9.
- **The change pushes posterior onto Germany's phantom SX-EW route.** Ore→SXEW rises 287.1 → 381.2 kt CC (+33 % vs
  `opt_20261019_142026_83bfe843`), on a prior of 0.5. With the imputer's concentrate residual gone, the solver
  supplies part of Germany's balance through a mine-and-SX-EW chain that has almost no prior: a no-data chain. It is
  reported here, not judged, and it is the open part of #32.
- **The same class:** Korea's mine-chain Σz² 10,773 → 1,555 and Spain's 19,291 → 12,958 (common set). Slovakia's
  (8,991) and Botswana's (9,289) are unchanged; neither has an observed zero.
- **#76's original refill class**, none of them stamped now: United Kingdom −0, Indonesia +102, Italy +0, Sweden +2
  (common-set Δz²).

## 6. Common set

- **Σz² 1,242,441 → 1,155,371 (−7.008 % vs `opt_20261019_142026_83bfe843`)**, 27,458 rows in both runs.
- **Rows only one run has:** the 54 base-only rows carry z² 57,455 (Germany 26,706, Korea 19,652, Spain 11,097; Japan
  12 rows at 0); none are new.
- **By region:**
  - falls: Germany −74,269, Korea −7,262, Spain −6,329, Chile −731;
  - rises: UAE +555, Malaysia +263, Peru +202, ROW +187.
- **By sector:** Ore −40,875, Concentrate −24,082, MF −23,911, Mining −1,510; Cathode +1,187, Smelting +1,007,
  Matte +948.
- **By year:** the gains sit in 1997 (−9,187), 2003 (−6,229), 2016–2018 (2017 −19,653, 2018 −19,941) and 2020–2022
  (2021 −20,395). 2019 rises +288, 2024 +2,314.

## 7. Reported, not judged

- **The small-weight legs** (#76's comment): 7 domestic legs fed below `min_flow_kt` are dropped and refilled, in
  both runs, carrying 268.6 kt CC of prior. They are Austria Cathode 1995, Australia Cathode 2014–2016, Brazil
  Concentrate 2015 and 2019, and Bulgaria Concentrate 2015. Unchanged, and outside #76's rule.
- **ROW's 1995–2011 surplus**, by year: moot. ROW's domestic semis leg has been fed and positive in every year since
  #79, and its 1995–2011 prior is on its fan here.
- **Where ROW's surplus would go:** ROW is untouched (§4).

## 8. Predictions

| Prediction (posted on #78 before each run) | Outcome |
|---|---|
| First run: items 1 and 2 hold | **Right** |
| First run: item 3 holds, with a risk in UK, Taiwan, Korea, Switzerland | **Wrong:** it fails, on Indonesia, Zambia and Sweden |
| First run: item 4 holds, common-set z² falls 4–8 % | **Wrong:** +29.2 % |
| First run: Germany's mine rows fall by about 100k | **Right:** 74,373 → 0 on the common set; with its base-only rows, about 101k |
| First run: Germany's smelter rows rise by about 10–20k | **Wrong:** +52 (Smelting), +39 (Matte) |
| First run: Germany's posterior mine output falls to ≈ 0 | **Wrong:** 383.9 kt CC remains |
| Second run: items 1–3 hold | **Right** for 1 and 3; item 2 fails as worded in 2015, unchanged from base |
| Second run: z² falls 4–7 %; Germany about −74k; Korea a little more; Germany's smelter rows rise < 5k | **Right:** −7.0 %; Germany −74,269 (common set); Korea −7,262; Germany's smelter rows +50 and +37 |
| Germany's posterior mine output falls to ≈ 0 in every year | **Wrong:** 388.1 kt CC remains, almost all SX-EW (§5) |

## 9. Follow-ups

- **#76** records the scope change: the first build was rejected, and the narrowed rule is what ships.
- **#96:** the 246 cathode zeros in non-producer region-years. Keying the test on ADR 0024's raw ICSG production
  would extend the observed zero to them.
- **#32:** Germany's phantom SX-EW chain, Ore→SXEW 381.2 kt CC on a prior of 0.5. It is a no-data chain, and the
  open part of #32.

## Reproduce

```
# second run, on fix/observed-zero 89fd55d (a94cdcb is behaviour-preserving)
python scripts/icsg_feed.py                                       # → feed_icsg_20261019_150334_42906d2e
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261019_150334_42906d2e \
    --feed-id feed_baci_20261019_135628_500e99ce --feed-id feed_crude_ore_20260603_095651_558f7369 \
    --label r78b_observed_zero_nonproducer
python scripts/constraints.py --imp-run-id imp_20261019_150618_993d9856 r78b_observed_zero_nonproducer
python scripts/run_solver.py --const-run-id const_20261019_151044_cf8aab33 --label r78b_observed_zero_nonproducer
python scripts/explo/issue78_observed_zero_diag.py opt_20261019_142026_83bfe843 opt_20261019_152525_6ca5c00a \
    feed_icsg_20261019_143004_d2d0ff50
# first run: the same chain on fix/observed-zero 3a9e4b0, then
python scripts/explo/issue78_observed_zero_diag.py opt_20261019_142026_83bfe843 opt_20261019_145258_6bbfa0b5
```

The diagnostic prints every number in this REP or the numbers it is summed from, except the solve times, which
come from the runs' sidecars.
