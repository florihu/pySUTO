# #86: DR Congo's copper leaves ROW (BACI 178, ICSG 2005 "Congo") and SACU maps to South Africa (#84, #85)

**Run:** `opt_20261015_093646_ddd9d3d9` (label `r86_congo_sacu`; 30/30 Solved; `prior_mode` cold; `n_workers` 8,
`solve_wall_s` 633 s per the run's sidecar)
**Against:** the reference chain `opt_20261014_131552_f320b5f7` (#79, REP
[`…issue77-baci-490-taiwan`](analysis_opt_20261014_131552_f320b5f7_20261014_issue77-baci-490-taiwan.md)),
called *base* below
**Chain:**
- `icsg_clean.py --to-feed` → icsg **v15**. It relabels 27 rows and drops 3; its diff from v14 touches only
  `Congo Rep.` (30 rows out) and `Congo, D.R.` (27 rows in).
- `baci_parse_feed.py` re-run. Its `values.csv` and `quantity_recon.csv` reproduce v19's (`9807e3dd…`); 38 keys
  of the ADR 0033 σ sidecar change, every one touching Congo, D.R., South Africa or ROW.
- → baci **v21**, the gross version under the new map → `build_baci_reexport.py --source-version 21` → baci
  **v22**.
- → `feed_baci_20261015_091204_f89febc2` → trade_shares **v34** → `feed_icsg_20261015_091319_e9a23515`.
- With `feed_crude_ore_20260603_095651_558f7369` → `imp_20261015_091708_965bf416` →
  `const_20261015_092141_b704b044` → the run.

**Feeds/params:** **icsg v15**, **baci v22**, **trade_shares v34**, acc copper v4, conc v10, eff v4,
icsg_aggregates v4, ore_grade v7, scrap_share v3, stock_bounds v5, stripping_ratio v2. These are the base's
versions except the ones in bold.
**Date:** 2026-10-15 · **Branch:** `diag/congo-sacu-measure` (from `fix/congo-sacu-mapping`) · **Issues:**
#86, #84, #85, #82; fix ticket #88 · **ADR:** 0014 and 0032 (amended by #84, #85), 0025 (stop rule), 0019

**Verdict: fails on two items, 5 (stock legs) and 6 (DR Congo's production), both on DR Congo.** Items 1–4
hold. Fix ticket **#88**. **Merged 2026-10-19 over the failed items by the maintainer's decision** (#75 and #79
precedent): `opt_20261015_093646_ddd9d3d9` is the reference chain, and #82 is closed.

- **ROW's cathode gap closes to the importer residual.** In the ledger, ROW's 2012–2024 cathode U − (P + I − X)
  falls from +5,829 to +1,951 kt CC. The Σz² of ROW's 2012–2024 cathode-use sum rows falls from 16,055 to 3,208
  (0.200 of base). ROW's domestic semis leg in 2012–2024 goes from a posterior of 1,024 to 3,709 kt CC.
- **South Africa enters 1995–1999 with trade.** Its 1995–1999 cathode use rows fit (Δz² −9,775 on South
  Africa), and its refined output there is no longer cut (CP→Cathode 572 → 654 posterior on a prior of 619).
- **DR Congo takes the contradiction.** Code 178's cathode lands on top of code 180's in 2020–2023, where 180
  alone already balanced. The solver raises DR Congo's refined output above its prior by 829 kt CC over
  2019–2023 (base 207), with CP→Cathode retention 1.080–1.143 in 2020–2023, and opens one stock leg at
  |z| 6.9.
- **Common-set sum-row z² −0.75 % vs `opt_20261014_131552_f320b5f7`**: ROW −14,013, South Africa −9,775, DR
  Congo +13,320.

## 1. Change

- **#84:** the BACI region map sends code 178 "Congo" to Congo, D.R. and 711 "Southern African Customs Union"
  to South Africa; both were ROW.
- **#85:** the ICSG clean step relabels the 2005 yearbook's `Congo` (parsed as `Congo Rep.` → ROW) to
  Congo, D.R. for 1995–2003 and drops its 2004 rows, which the 2014 yearbook reports under Congo, D.R.

Both come from #82's triage: country by country, most of ROW's cathode gap was trade and production that
belonged to DR Congo and South Africa.

## 2. Pass rule (fixed on #86 before any run)

| Item | Rule | Result |
|---|---|---|
| 1 | The labels land | **Holds.** The gross baci v21 and the netted v22 map 178 → Congo, D.R. and 711 → South Africa. In icsg v15, Congo, D.R. carries the 2005 yearbook's 1995–2003 (27 rows), 2004 has 2 rows under Congo, D.R. only, and `Congo Rep.` keeps 2017–2024 (8 rows). The icsg v14 → v15 diff is 57 rows, all `Congo Rep.` or `Congo, D.R.` (30 out, 27 in). |
| 2 | ROW's ledger gap, 2012–2024, ≤ +2,500 kt CC | **Holds.** +5,829 (v20) → **+1,951** (v22); by year −52 … +546. |
| 3 | ROW's 2012–2024 cathode-use sum rows: Σz² at most half the base's | **Holds.** 16,055 → 3,208, 0.1998 of base. |
| 4 | Round-E (ADR 0025) on the common set | **Holds** (§6). |
| 5 | No new stock leg above \|z\| 5; ROW's eight 2022–23 MESC/USC legs fall below 5 | **Fails on the first half.** One leg, USC 2022 DR Congo → China, in both valuations (z −6.89 Mass, −6.88 CC; base −0.05, −0.04). ROW's eight legs fall from −5.95 … −6.49 to −0.09 … +0.01. |
| 6 | DR Congo's CP→Cathode CC retention ≤ 1.05 in every year | **Fails.** 1.129, 1.114, 1.143, 1.080 in 2020–2023; base 1.016–1.034. |

Retention is posterior/prior of the CP→Cathode CC edge into DR Congo, per year. Two more years need saying:
- **2016** is 1.047, inside the bar, up from 1.000 in base.
- **2005 and 2006** have a zero prior in both runs, so retention is undefined. The rule as written fails them in
  both runs; their posteriors, 1.68 → 2.89 and 3.37 → 8.86 kt CC, are a few kt and do not decide the item.
  The base also had a zero prior in 1995–2003 with posteriors of 0.9–14.1 kt CC; #85 gives those years a
  prior, and their retention is now 1.000–1.007.

## 3. Predictions (posted on #86 before Stage 2)

| Prediction | Outcome |
|---|---|
| Item 3 passes; ROW's 2012–2024 cathode-use Σz² falls to 20–35 % of base | **Right on the pass; the size falls just below the band** (19.98 %) |
| Round-E holds; (a)–(c) within noise | **Right** |
| (d) common-set z² falls 1–3 % | **Wrong on size**: −0.75 %, because DR Congo's 2004–2008 rows rise (Δz² +22,220, all sectors) |
| ROW's eight 2022–23 legs fall below \|z\| 5; risk: a DR Congo stock leg absorbing the 2020–2023 excess | **Right**, and the named risk occurred (item 5) |
| Item 6 holds: retention ≤ 1.05, exports cut by 1,000–1,500 kt CC over 2020–2023 instead | **Wrong.** Retention 1.080–1.143; exports cut by 625 kt CC over 2020–2023 (8,231 → 7,606) |
| Tanzania and the UAE are the likeliest export legs to be cut | **Partly right.** Over 2020–2023 the UAE leg is cut by 160 kt CC (1,337 → 1,177) and the ROW leg, which holds Tanzania, by 159 (1,921 → 1,762); China's is cut most, by 274 (3,247 → 2,973) |
| DR Congo's 1995–2003 smelter or SX-EW chain cut by about half | **Wrong.** Both routes are carried whole (§5) |
| South Africa's 711 cathode exports partly cut | **Right**: 350 → 314 kt CC |

## 4. ROW (items 2 and 3)

Ledger, cathode U − (P + I − X), kt CC:

| | 1995–2011 | 2012–2024 |
|---|---|---|
| baci v20 (base) | +2,355 | +5,829 |
| baci v22 | +2,169 | **+1,951** |

- **2012–2024: ROW's cathode use is fed.** Its semis use posterior/fan rises from 0.61–0.89 to 0.72–1.01, with
  2 years below 0.85 instead of 12. The domestic leg's posterior is 3,709 kt CC (fed 4,877) against 1,024
  (fed 5,436); ROW's production prior less its export prior is 3,703, against 244.
- **By year,** the Σz² of ROW's cathode-use rows falls in every year 2012–2024; relative to each year's base,
  most in 2020–2023 (811 → 3, 1,385 → 114, 1,777 → 53, 1,624 → 1).
- **ROW's refined output** in 2020–2022 is now cut a little: CP→Cathode posterior/prior 0.94, 0.95, 0.91
  (base 0.99, 0.98, 0.99). ROW's ledger gap there is −52, +48, +3: ROW is no longer short of cathode.
- **1995–2011** moves little (ledger +2,355 → +2,169): #85 takes DR Congo's SX-EW cathode out of ROW's
  production, and #84 takes code 178's and 711's exports out of ROW's. ROW's CP→Cathode in 1995 rises to 1.22
  (base 1.08), on a prior that lost DR Congo's output.
- **ROW's Δz² vs base: −14,013**, of which Cathode −13,715 (100–1,000 kt band −9,819, > 1,000 kt −3,895).

## 5. DR Congo (items 5 and 6)

| | base | new |
|---|---|---|
| ledger U − (P + I − X), 1995–2011 | −474 | −543 |
| ledger U − (P + I − X), 2012–2024 | −2,785 | **+1,094** |
| CP→Cathode 2019–2023, Σ prior / Σ post | 8,023 / 8,230 (+207) | 8,023 / 8,852 (**+829**) |
| CP→Cathode 2020–2023, Σ prior / Σ post | 6,828 / 7,011 (+183) | 6,828 / 7,605 (+777) |
| cathode exports 2019–2023, Σ prior / Σ post | 7,505 / 7,448 | 9,502 / 8,853 |
| cathode exports 2020–2023, Σ prior / Σ post | 6,636 / 6,576 | 8,231 / 7,606 |
| common-set Δz² | | **+13,320** |

- **2020–2023.** The ledger gap by year is +441, +321, +489, +341 kt CC. In the base, code 180 alone gave +64,
  0, +104, −123 there: DR Congo's own exports already matched ICSG's production, and code 178's flows add
  321–464 kt CC a year on top. In 2012–2019 the same code fills a shortfall (base −134 … −358 a year).
- **The solver** splits the excess: refined output rises (+829 vs +207 kt CC over 2019–2023) and exports fall
  by 649 kt CC (China 3,480 → 3,202, ROW 2,241 → 2,073, UAE 1,661 → 1,495). By year, China's leg is cut in
  2020–2022 (2020: 542 → 476) and the UAE's in every year. It also opens the item-5 leg, USC 2022 DR Congo →
  China at −63 kt CC: a stock drawdown that acts as a cathode source for DR Congo.
- **Transit.** In gross baci v21, codes 178 and 180 ship 785 kt of cathode to Tanzania in 2018–2023 (228 in
  2020) and 517 kt to Singapore. Tanzania sits in ROW, which the netting never nets. Singapore's leg is barely
  cut (2020–2023: 354 → 346).
- **Neither DR Congo nor South Africa becomes a hub.** DR Congo's ledger nets 1–5 kt CC a year in 2010, 2017
  and 2024, as before; South Africa's 16–23 kt CC of cathode in 2019–2024 is unchanged.
- **2004–2010.** DR Congo's concentrate export prior rises (2008: 93 → 143 kt CC) with code 178's flows, and
  its smelter chain carries it. DR Congo's Δz² by sector, all years: Anode +8,532, Matte +7,247, Smelting
  +4,402, CFR +3,428. By year: 2004 +4,654, 2007 +4,309, **2008 +11,684**; 2004–2008 together +22,220.
- **1995–2003 is fed.** Δz² −12,923 over 1995–2002 (by sector, all years: Mining −9,387, Cathode −2,974).
  Both of the 2005 yearbook's routes are carried whole: Ore→SXEW 355/355 and Concentrate→Smelting 288/307
  kt CC (prior/post), on a mine of 734/724. In the base, DR Congo's chain had priors only on the mine (364) and the solver cut it
  to 178, building Ore→SXEW 151 and CP→Cathode 92 on zero priors. The 2005 yearbook books the low-grade
  electrowon cathode in both its smelter and its SX-EW tables; whether the mine of 734 double-counts it is not
  settled here.

## 6. Round-E (ADR 0025), `opt_20261014_131552_f320b5f7` → `opt_20261015_093646_ddd9d3d9`

| Item | Base | New |
|---|---|---|
| (a) world retention: CO Mass / Concentrate CC / Cathode CC | 0.863 / 0.993 / 0.989 | 0.863 / 0.993 / 0.990 |
| (b) small nodes (\|c\| < 18 kt), posterior/prior median / p90 | 1.000 / 1.172 | 1.000 / 1.148 |
| (c) exporter retention: Chile / Peru / Indonesia / world | 0.968 / 1.011 / 0.914 / 0.936 | unchanged |
| (d) sum-row z², \|c\| < 10 kt | 112,785 | 111,878 (−907) |
| (d) 10–100 kt | 526,695 | 532,584 (+5,889) |
| (d) 100–1,000 kt | 536,735 | 526,703 (−10,033) |
| (d) > 1,000 kt | 158,847 | 153,908 (−4,940) |
| (d) mass moved, all CC edges | 473,283 kt CC | 469,971 kt CC (−0.70 % vs `opt_20261014_131552_f320b5f7`) |

- **(a)–(c) hold.**
- **(d) holds.** The gain sits on the rows the change targets: ROW's cathode use (100–1,000 kt and > 1,000 kt)
  and South Africa's 1995–1999 cathode (10–100 kt), and it comes with mass moved, not a repricing: ROW's
  domestic semis posterior +2,685 kt CC in 2012–2024. (d)'s own mass-moved total falls 0.70 %; as in #79, it is
  reported, not judged. The 10–100 kt band rises on DR Congo. A rise in one band
  beside a located gain is the #65 precedent, where (d) was not failed.

## 7. Stock legs

- **|z| > 5:** 302 legs in base, 289 new. Two are newly above it: the USC 2022 DR Congo → China leg, in Mass
  and CC (item 5).
- **ROW's eight 2022–23 MESC/USC legs** (#82) fall below |z| 5: z −6.49/−6.47 (2022) and −5.98/−5.95 (2023) in
  base, −0.09 … +0.01 now.
- **The ten Kazakhstan → ROW and Russia → ROW legs** (#82) stay where they were. Kazakhstan, Mass/CC (the USC
  and MESC legs are identical): 1999 z 5.20/5.27 → 5.37/5.34, 2000 5.03/5.14 → 5.04/5.14. Russia 2002 CC −5.47,
  unchanged. Their years keep ROW's
  1995–2011 gap.
- **Net stock change:** USC Σ −276.4 → −461.5 kt CC, most in China 2020–2023 (2022: −40.7 → −103.0); MESC
  −188.3 → −213.1.

## 8. Common set

- **Σz² 1,335,063 → 1,325,073 (−0.748 % vs `opt_20261014_131552_f320b5f7`)**, 27,298 rows in both runs.
- **By region:** falls ROW −14,013, South Africa −9,775, China −588, Zambia −217; rises DR Congo +13,320,
  Malaysia +639, UAE +437.
- **By year:** 1995–2002 fall (1998 −5,264), 2003–2008 rise (2008 +11,160), every year 2012–2024 falls
  (−30 … −2,357).
- **Rows in one run only:** 188 new rows (z² 1,477), 176 of them DR Congo's; 24 rows only in base
  (z² 378), 16 of them DR Congo's.

## 9. South Africa, 1995–1999

- **Trade enters.** Exports prior/post: cathode 350/314, anode 214/189, concentrate 84/88, matte 5/4 kt CC.
  The base had no export prior, and its posterior carried 74 kt CC of concentrate exports. The cathode goes
  to Taiwan, Italy, Japan, Germany and Malaysia; the anode to Korea, Belgium, China, Germany and Japan.
- **Refined output** is no longer cut: CP→Cathode prior 619, posterior 572 → 654.
- **Its semis-use fan** (Cathode→FU into South Africa) was all domestic in base, and the solver raised it above
  its prior (396 → 562) to give the refined output an outlet. Now it is cut below it (392 → 342), and imports
  are 4/5.
- **Namibia's share of 711's anode** cannot be separated: 711 is one code, and the map gives all of it to South
  Africa. For scale, in gross baci v21 711's anode exports are 44, 39, 44, 43 and 26 kt in 1995–1999. From
  2000, South Africa's (710) are 31, 40, 28, 19 and 7, and Namibia's (516) 8, 21, 22, 29 and 25.
- **The ledger** shows South Africa's 1995–1999 cathode U − (P + I − X) at −3 … +56 kt CC a year.

## 10. ROW's remaining cathode gap, by country (#82)

ICSG v15 labels folded into ROW against gross baci v21 by country code, HS 740311, kt CC:

| | 1995–2011 | 2012–2024 |
|---|---|---|
| ROW gap | +2,148 | +1,920 |
| Saudi Arabia | +1,065 | +676 |
| Turkey | +204 | +646 |
| Vietnam | | +556 |
| Pakistan | +203 | +490 |
| Egypt | +493 | +189 |
| North Korea | | +268 |
| Oman | | +207 |
| Tanzania, Mozambique (transit) | | −532, −282 |
| Finland | | −573 |
| Serbia / Serbia and Montenegro | −332 / +325 | +216 |

This is the disagreement between the sources #82's triage named: ICSG's semis use in Saudi Arabia, Turkey,
Vietnam, Pakistan, Egypt, Oman and North Korea exceeds what BACI shows them importing. No mapping explains it.
It stays visible as misfit (ADR 0019).

## 11. Follow-ups

- **#88:** DR Congo's 2020–2023 cathode excess and its 2004–2010 concentrate. The triage questions: does the UAE
  report the same shipments under 178 and 180; does the transit through Tanzania (785 kt gross, 2018–2023; ROW,
  not netted) count DR Congo's cathode twice; is ICSG's DR Congo production low.
- **#82's question is answered**: its gap was DR Congo's and South Africa's trade and production, plus the
  importer disagreement of §10. #86 closes #82 only on a pass, so closing it now is the maintainer's call.

## Reproduce

```
python scripts/parse/icsg_parse/icsg_clean.py --to-feed            # → icsg current/
python scripts/parse/baci_parse/baci_parse_feed.py                 # → baci/v0; values.csv = v19's
#   baci current/: set aside reexport_*.csv, copy v0/{values,quantity_recon,quantity_cv_scale}.csv in,
#   then InputRegistries.initialize(root="data/input")            # → icsg v15, baci v21 (gross)
python scripts/build_baci_reexport.py --source-version 21         # → baci v22
python scripts/baci_feed.py
python scripts/build_trade_shares.py                               # → trade_shares v34
python scripts/icsg_feed.py
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261015_091319_e9a23515 \
    --feed-id feed_baci_20261015_091204_f89febc2 --feed-id feed_crude_ore_20260603_095651_558f7369 \
    --label r86_congo_sacu
python scripts/constraints.py --imp-run-id imp_20261015_091708_965bf416 r86_congo_sacu
python scripts/run_solver.py --const-run-id const_20261015_092141_b704b044 --label r86_congo_sacu
python scripts/explo/issue86_congo_sacu_diag.py opt_20261014_131552_f320b5f7 \
    opt_20261015_093646_ddd9d3d9 20 22 21
```

The diagnostic prints every number in this REP or the numbers it is summed from, with two exceptions: the
solve time comes from the run's sidecar, and the re-parse hashes and sidecar-key count come from building the
chain; they are posted on #86.
