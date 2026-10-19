# #92: all refined copper enters the Cathode sector (#91), and BACI 178 is dropped in 2015–2023 (#88)

**Run:** `opt_20261019_142026_83bfe843` (label `r92_refined_scope`; 29 Solved, 2016 AlmostSolved; `prior_mode`
cold; `solve_wall_s` 612 s per the run's sidecar)
**Against:** the reference chain `opt_20261015_093646_ddd9d3d9` (#86, REP
[`…issue82-congo-sacu`](analysis_opt_20261015_093646_ddd9d3d9_20261015_issue82-congo-sacu.md)), called *base* below
**Chain:**
- `baci_parse_feed.py` re-run → baci **v26** (gross) → `build_baci_reexport.py --source-version 26` → baci **v27**.
- → `feed_baci_20261019_135628_500e99ce` → trade_shares **v36** → `feed_icsg_20261019_135738_6fb63aff`.
- With `feed_crude_ore_20260603_095651_558f7369` → `imp_20261019_140120_eb1fdded` →
  `const_20261019_140547_4ed676e5` → the run.
- baci v25 is an unused orphan, registered by a diagnostic's registry lookup while `current/` held the new sector
  map (#93).

**Feeds/params:** icsg v15, **baci v27**, **trade_shares v36**, acc copper v4, conc v10, eff v4, icsg_aggregates
v4, ore_grade v7, scrap_share v3, stock_bounds v5, stripping_ratio v2. These are the base's versions except the
ones in bold.
**Date:** 2026-10-19 · **Branch:** `diag/refined-scope-measure` (from `fix/baci-refined-scope`) · **Issues:** #92,
#91, #88, #63; fix ticket #95; #90 superseded · **ADR:** 0014 (amended by #88, #91), 0033, 0025 (stop rule), 0019

**Verdict: fails on item 4 (no new sink).** Items 1, 2, 3, 5 and 6 hold. Fix ticket **#95**. **Ships over item 4
by the maintainer's decision** (2026-10-19, delegated to the agent for the manuscript-ready week; #79 and #86
precedent): `opt_20261019_142026_83bfe843` becomes the reference chain.

- **Austria's cathode node gets its outlet.** The Σz² of its 1995–2014 cathode-use rows falls from 48,500 to 2,624
  (0.054 of base); Austria's common-set Δz² is −88,777.
- **Common-set sum-row z² −5.57 % vs `opt_20261015_093646_ddd9d3d9`.** It falls in 23 of 30 years; Iran −11,402,
  the United Kingdom −9,631 and India −4,298 also improve. Counting the rows only one run has (the new run's 20 UAE
  rows carry 46,472), the total goes 1,326,550 → 1,299,897, **−2.0 %**.
- **Two side effects.**
  - Refined exports beyond ICSG production (Kazakhstan, DR Congo, Zambia, Russia) open 25 CC stock legs
    above |z| 5. The net stock posterior over 1995–2024 goes from +51 to −1,328 kt CC.
  - Dropping 178 → UAE leaves the UAE's re-exports without supply: common-set Δz² +25,625, plus 46,472 on the
    UAE rows only the new run has.

## 1. Change

- **#91:** BACI's Cathode sector reads HS 740311 cathodes plus 740312 wire bars, 740313 billets and 740319 other
  refined copper (ADR 0014 am.). The last three were dropped before. ADR 0033's copper price stays 740311 alone.
- **#88:** `MIRROR_DUPLICATES = {178: (2015, 2023)}`. Code 178's rows are dropped in the years DR Congo's own
  declarations (180), counted over all of 7403.1x, lie within −74 … +422 kt of its ICSG production, so that 178
  would push them past it (ADR 0014 am.).

## 2. Pass rule (fixed on #92 at filing; item 1 clarified before any run)

| Item | Rule | Result |
|---|---|---|
| 1 | Against gross v21, only the build's changes | **Holds.** 39,555 rows of 740312/13/19 are added (21,164 kt), 249 code-178 rows of 2015–2023 removed, and 720 concentrate legs moved by ADR 0033's reconstruction (Σ\|Δ\| 141.1 of 142,561 kt, 0.099 %, inside the 0.5 % bound). By cause, from the sidecars: 587 legs through a changed `g_ref` (118.2 kt), 133 through the copper price alone (22.9 kt), none otherwise. Nothing else changes. |
| 2 | Austria's 1995–2014 cathode-use rows: Σz² at most half the base's | **Holds.** 48,500 → 2,624 (0.0541). |
| 3 | Round-E (ADR 0025) | **Holds** (§5). |
| 4 | No stock leg newly above \|z\| 5 | **Fails.** 54 legs, 25 of them CC (§6). |
| 5 | No code 178 in DR Congo's 2015–2023 balance | **Holds.** Gross v26 has no code-178 row in 2015–2023. 178 maps only to DR Congo, so this stands for the ledger's exports. |
| 6 | ROW's 2012–2024 ledger cathode gap ≤ +2,500 kt CC | **Holds.** +1,803 (base +1,951). |

## 3. Predictions (posted on #92 before Stage 2)

| Prediction | Outcome |
|---|---|
| Item 2 holds; Austria's 1995–2014 Σz² falls to 15–40 % of base | **Right on the pass, wrong on size:** 5.4 % |
| Round-E holds; (a)–(c) move < 0.01 | **Partly right:** Round-E holds; (a) and (c) move < 0.01, but (b)'s p90 moves −0.015 (1.148 → 1.133) |
| (d) common-set z² falls 0.5–2 %; Austria −30k to −45k; DR Congo and the UAE rise | **Wrong on size:** −5.57 %, and Austria −88,777. **Right** that the UAE rises (+25,625); DR Congo barely moves (+228) |
| Item 4 at risk; expect a DR Congo stock leg above \|z\| 5 | **Right:** 4 DR Congo legs, and Kazakhstan, Zambia, Russia and Indonesia too |
| DR Congo's CP→Cathode retention in 2017–2020 rises to 1.15–1.30 | **Wrong:** 1.120, 1.138, 1.135, 1.099 |
| DR Congo's export legs are cut | **Right:** 2019–2023 prior 8,601, posterior 8,347 kt CC |
| Chile's 1995–2010 cathode exports gain about 100–160 kt CC a year of prior | **Wrong:** its 1995–2011 export prior (the diagnostic's window, one year longer) rises 42,512 → 43,129, about +36 a year |

## 4. Austria (#63's first defect)

| | base | new |
|---|---|---|
| cathode-use Σz², 1995–2014 | 48,500 | 2,624 |
| cathode-use Σz², 2015–2024 | 6,969 | 2,475 |
| CP→Cathode, 1995–2014, prior / post | 1,638 / 1,322 | 1,638 / 1,675 |
| cathode exports, 1995–2014, prior / post | 628 / 898 | 1,562 / 1,452 |

- **The model no longer cuts Austria's refined output** to fit the missing exports: in 1995–2014 it was 0.81 of its
  prior, now 1.02.
- **Austria's Δz² by sector:** Cathode −50,371, CP −7,971, ER and CER −7,661 each, CFR −6,304, Smelting −4,869,
  Anode −3,844.
- **The scrap cluster of 2015–2024 remains:** Austria's Smelting, CFR, Anode and Matte rows carry Σz² 52,597 →
  51,720 there. That is #63's second defect, imported anode in ADR 0030's nesting, which this round does not build.

## 5. Round-E (ADR 0025), `opt_20261015_093646_ddd9d3d9` → `opt_20261019_142026_83bfe843`

| Item | Base | New |
|---|---|---|
| (a) world retention: CO Mass / Concentrate CC / Cathode CC | 0.863 / 0.993 / 0.990 | 0.864 / 0.994 / 0.993 |
| (b) small nodes (\|c\| < 18 kt), posterior/prior median / p90 | 1.000 / 1.148 | 1.000 / 1.133 |
| (c) exporter retention: Chile / Peru / Indonesia / world | 0.968 / 1.011 / 0.914 / 0.936 | 0.966 / 1.011 / 0.914 / 0.935 |
| (d) sum-row z², \|c\| < 10 kt | 110,039 | 141,532 (+31,493) |
| (d) 10–100 kt | 528,865 | 429,565 (−99,300) |
| (d) 100–1,000 kt | 533,862 | 529,891 (−3,971) |
| (d) > 1,000 kt | 152,831 | 150,737 (−2,093) |
| (d) mass moved, all CC edges | 469,971 kt CC | 467,096 kt CC (−0.61 % vs `opt_20261015_093646_ddd9d3d9`) |

- **(a)–(c) hold.** Retention rises; exporter retention moves by at most −0.002.
- **(d) holds.** The gain sits where the change aims, Austria's cathode rows in the 10–100 kt band (−99,300), and it
  comes with mass moved: Austria's export posterior over 1995–2014 rises 898 → 1,452 kt CC. One band rises beside
  three that fall, the #65 and #86 reading; #79 failed (d) because every band rose.
- **But the rising band is larger than #86's.** The < 10 kt band rises +31,493, +28.6 % of its base (#86's rising
  band: +1.1 %). Most of it is a new contradiction, the UAE's cathode-use totals (§7), plus the UK 1996 (+3,488) and
  Portugal 2004 (+3,252). The UAE's 46,472 on rows only the new run has sit outside the common set.
- **Mass moved** is reported, not judged, as in #79 and #86.
- **The Lp σ side effect** (noted on #92): a Cathode leg now summed from several HS codes gets a smaller σ under the
  Lp rule. That tightens those legs slightly; it is not separated out here.

## 6. Stock legs (item 4)

- **|z| > 5:** 289 legs in base, 319 new. 54 are newly above it, 22 of them into ROW.
- **The 25 CC legs among them:**

  | exporter | legs | Σ posterior, kt CC |
  |---|---|---|
  | Kazakhstan | 11 | −129.4 |
  | DR Congo | 4 | −113.7 |
  | Zambia | 1 | −28.5 |
  | Russia | 8 | −18.5 |
  | Indonesia | 1 | −14.5 |

  Σ|post| is 304.8 kt CC, against 74.6 in base.
- **The largest by |z|:** Indonesia → Japan MESC 1999 (z −10.6), Zambia → China USC 2014 (−9.8), Russia → Taiwan
  1997–98 (−9.7, −8.4), Kazakhstan → China/ROW in 2015–2024 (−5.9 … −8.2), DR Congo → ROW 2019 (−6.1), DR Congo →
  China 2018–2019 (−5.9).
- **The net stock posterior** of all CC stock legs over 1995–2024 goes from +51 to −1,328 kt CC (Σ|post| 16,544 →
  16,531).
- **The model draws on stocks** to supply refined exports beyond these exporters' production. With little domestic
  use, a positive ledger gap U − (P + I − X) is exports beyond production plus imports, and the new codes raise it:

  | exporter | baci v22 | baci v27 |
  |---|---|---|
  | Kazakhstan, 2012–2024 | +1,901 | +2,299 |
  | Zambia, 2012–2024 | +2,831 | +3,569 |
  | Zambia, 1995–2011 | +42 | +270 |
  | Russia, 1995–2011 | +6,894 | +7,208 |
  | Indonesia, 1995–2011 | +319 | +650 |

  Russia's gap also carries its large domestic use, so for Russia it is not only exports. #95 asks whether one
  shipment is declared 740311 by one side and 740319 by the other.

## 7. DR Congo and the UAE

- **DR Congo.** Its common-set Δz² is +228. Its v27 ledger U − (P + I − X), code 180 alone on the full scope, is
  +149, −26, +411, +419, +367, +386, +198, +141, −80 kt CC in 2015–2023.
  - CP→Cathode retention by year: 2015 1.064, 2016 1.002, 2017 1.120, 2018 1.138, 2019 1.135, 2020 1.099, 2021
    1.066, 2022 1.040, 2023 1.016, 2024 1.012 (base 2020–2023: 1.129, 1.114, 1.143, 1.080).
  - Over 2019–2023 refined output exceeds its prior by +502 kt CC (base +829), over 2020–2023 by +341 (base +777).
  - Cathode exports, prior/post by year: 2015 859/816, 2016 744/710, 2017 764/855, 2018 1,235/1,098, 2019
    1,468/1,357, 2020 1,588/1,486, 2021 1,640/1,591, 2022 1,885/1,882, 2023 2,021/2,031, 2024 2,278/2,285.
- **The UAE.** Its common-set Δz² is +25,625: cathode-use CC totals of 213, 217, 227 and 240 kt CC in 2016, 2017,
  2019 and 2022, on a base prior of 0. Its rows for the other years of 2015–2023 exist only in the new run (z²
  46,472). These are the 178 → UAE flows the window drops, 80–252 kt a year in gross v21. The UAE still
  re-exports that cathode, so its use total carries it with no import to match. Code 180 → UAE is 1–125 kt a year
  in those years. #95.
- **South Africa** becomes a cathode pass-through hub in 2017–2024: T 77, 81, 47 kt CC in 2017–2019, consistent
  with DR Congo's refined copper moving through Durban. Its Δz² is +2,649.

## 8. ADR 0033's sidecars and copper price

- **Flagged concentrate legs:** 1,000 (v21) → 993 (v26). 23 exporters' `g_ref` changes.
- **The copper price,** pooled HS 740311 unit value from the raw BACI files, without 178's 2015–2023 rows moves by at
  most 0.3 %: 2019 6,074 → 6,059 USD/t, 2021 9,163 → 9,136, 2022 8,814 → 8,829.

## 9. Trade of the largest movers (Σ prior / Σ post, kt CC)

| | 1995–2011 exports | 1995–2011 imports | 2012–2024 exports | 2012–2024 imports |
|---|---|---|---|---|
| Chile, base | 42,512 / 41,555 | 4 / 7 | 30,855 / 30,047 | 1 / 4 |
| Chile, new | 43,129 / 42,925 | 4 / 6 | 30,490 / 30,056 | 2 / 3 |
| China, base | 990 / 963 | 20,054 / 19,849 | 2,547 / 2,620 | 45,129 / 44,203 |
| China, new | 1,067 / 1,059 | 20,632 / 20,400 | 2,576 / 2,690 | 47,222 / 46,452 |
| Italy, base | 83 / 58 | 10,234 / 9,532 | 63 / 28 | 6,820 / 6,850 |
| Italy, new | 57 / 45 | 10,466 / 10,687 | 54 / 16 | 6,814 / 6,619 |

China's 2012–2024 cathode import posterior rises 44,203 → 46,452 kt CC (+5.1 % vs `opt_20261015_093646_ddd9d3d9`).

## 10. Common set

- **Σz² 1,325,597 → 1,251,725 (−5.573 % vs `opt_20261015_093646_ddd9d3d9`)**, 27,418 rows in both runs.
- **By region:**
  - falls: Austria −88,777, Iran −11,402, United Kingdom −9,631, India −4,298, Zambia −3,241, Chile −1,643;
  - rises: UAE +25,625, Indonesia +4,368, Kazakhstan +3,513, Portugal +2,957, South Africa +2,649, ROW +2,569.
- **By sector:** Cathode −21,533; ER, CER, CP −10.6k each; CFR −6,974; Smelting −5,935; Anode −5,638.
- **By year:** it falls in 23 of 30 years. It rises in 1998–1999 (+815, +1,251) and in 2016, 2017, 2019, 2021 and
  2022 (+4,265 … +7,188).
- **Rows in one run only:**
  - 94 new rows carry z² 48,172, 20 of them the UAE's (46,472);
  - 68 rows exist only in base (z² 953).

## 11. Follow-ups

- **#95:** the stock legs of §6 and the UAE of §7.
- **#63:** imported anode in ADR 0030's nesting (Austria's 2015–2024 scrap rows) and the 2006–2012 restatements.
- **#93:** diagnostics that register inputs.
- **#78** runs next, on this run as its base.
- **The manuscript:** Method's sources paragraph and Table 1 list the refined codes (#91). The counts and the trade
  numbers in Results change with the final chain.

## Reproduce

```
python scripts/parse/baci_parse/baci_parse_feed.py                 # → baci/v0
#   baci current/: set aside reexport_*.csv, copy v0/{values,quantity_recon,quantity_cv_scale}.csv in,
#   then InputRegistries.initialize(root="data/input")            # → baci v26 (gross)
python scripts/build_baci_reexport.py --source-version 26         # → baci v27
python scripts/baci_feed.py
python scripts/build_trade_shares.py                               # → trade_shares v36
python scripts/icsg_feed.py
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261019_135738_6fb63aff \
    --feed-id feed_baci_20261019_135628_500e99ce --feed-id feed_crude_ore_20260603_095651_558f7369 \
    --label r92_refined_scope
python scripts/constraints.py --imp-run-id imp_20261019_140120_eb1fdded r92_refined_scope
python scripts/run_solver.py --const-run-id const_20261019_140547_4ed676e5 --label r92_refined_scope
python scripts/explo/issue92_refined_scope_diag.py opt_20261015_093646_ddd9d3d9 \
    opt_20261019_142026_83bfe843 22 27 21 26
```

The diagnostic prints every number in this REP or the numbers it is summed from, except the solve time, which
comes from the run's sidecar.
