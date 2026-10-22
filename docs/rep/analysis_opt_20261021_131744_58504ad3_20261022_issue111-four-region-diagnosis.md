# #111: the four regions' Σz² is two feed defects and four source contradictions, not one parse error

**Run:** `opt_20261021_131744_58504ad3` (label `r110_russia`; the final chain of manuscript ready 2)
**Against:** none (read-only diagnosis; no rerun)
**Chain:** `feed_icsg_20261021_125725_b033ebc4`, `feed_baci_20261021_125615_bc691704`,
`feed_crude_ore_20260603_095651_558f7369` → `imp_20261021_125954_b112b170` → `const_20261021_130426_c5331a7f`
**Feeds/params:** icsg v17, baci v31 (gross v28), trade_shares v39, crude_ore v3, acc copper v4, conc v10, eff v4,
ore_grade v8, scrap_share v3, stock_bounds v6, stripping_ratio v2
**Date:** 2026-10-22 · **Branch:** `diag/anode-deletion-diagnosis` · **Issue:** #111 (#31, #96 folded in) ·
**ADR:** 0019, 0018, 0027, 0025
**Verdict: two input defects (feed level, fixable) and four contradictions between ICSG and BACI, which the
solver refuses correctly.** Neither ICSG series has a cell-layout defect of the #110 kind. The eff recipe is
not involved: every conflict is between totals from two sources and runs from ×2 to ×50, far beyond the
1–3 % that an efficiency moves.

| Region | Σz² | Mechanism | Kind |
|---|---:|---|---|
| South Africa 2017–2019 | 7,255 | ICSG smelter and refinery intake **withheld** from the fan as `imports_exhausted` although gross imports were 0.03–0.39 kt | **defect F1** |
| DR Congo 2015–2022 | (in 2019: ≈ 4,100) | the 2025 yearbook revised DRC primary refining 35 → 0 kt; the parse drops exact zeros, so the 2023 yearbook's 35 kt survives | **defect F2** |
| DR Congo 2004–2013, 2019 | 66,461 | BACI books 7–96 kt CC a year of DRC anode/blister exports (HS 7402, 2005–2019); ICSG has no DRC smelter (no row 2004–2012, 0 in 2013–2018, 1.5 kt in 2019) | contradiction C1 |
| South Africa 2020–2024 | 48,753 | ICSG reports smelter and refinery at 0; BACI books 47–137 kt CC a year of SA anode exports | contradiction C2 |
| Zambia | 37,669 | BACI's Zambia→Switzerland merchanting legs: cathode 462–699 kt Mass in 2013–2017, anode from 2016 (8ac5720) | contradiction C3 |
| Uzbekistan 1995–1998 | 44,104 | refined output 85–111 kt CC against BACI exports of 4–42 kt plus 10–16 kt domestic use; 1995 mine output 40 kt against 84 kt smelted | contradiction C4 |

## 1. Where the 34.7 % sits

Scale-total Σz² 755,225 over 28,982 rows; the four regions hold 261,919 (34.7 %). Blocks are chosen so each
carries one mechanism (`scripts/explo/issue111_four_region_diag.py`, section A).

| Region | Block | Σz² | share of total |
|---|---|---:|---:|
| Congo, D.R. | 1995–2003 | 2,234 | 0.30 % |
| | 2004–2013: no ICSG smelter row (2014 yearbook) or 0 | 47,069 | 6.23 % |
| | 2014–2018: ICSG smelter 0, dropped at parse | 2,747 | 0.36 % |
| | 2019: ICSG smelter 1.5 kt | 19,392 | 2.57 % |
| | 2020–2024 | 4,728 | 0.63 % |
| South Africa | 1995–2016 | 15,467 | 2.05 % |
| | 2017–2019: ICSG usage withheld | 7,255 | 0.96 % |
| | 2020–2024: ICSG smelter and refinery 0 | 48,753 | 6.46 % |
| Zambia | cathode rows | 14,413 | 1.91 % |
| | anode / CFR rows | 23,256 | 3.08 % |
| | other rows | 21,460 | 2.84 % |
| Uzbekistan | 1995–1998 | 44,104 | 5.84 % |
| | 1999–2024 | 11,042 | 1.46 % |

## 2. DR Congo: anode exports BACI books and ICSG's smelter series does not support

**Trace, 2012** (the issue's worst row, `usetotout Anode·Congo, D.R. CC`, c 96.1 → 2.0, z −57.4). The anode
node's use side is BACI: 95.6 kt CC to China's ER. Its supply side, `CFR→Anode` 1.9 kt CC, has no feed
behind it. The NetworkImputer filled it (`network_forward_capped`) from a concentrate balance capped
(`network_balance_capped`, 3.97 kt) by what BACI leaves at home: DRC's BACI concentrate exports, 116.9 kt CC,
take almost all of its 120.9 kt CC concentrate. `supplytotout CFR` is therefore a fabricated total of 1.93 kt
with σ_c 0.03 kt. ADR 0025's 0.3 kt floor applies to feed-backed totals only, so this total pins the chain.
Building 96 kt of anode would cost z ≈ 3,000 on it; deleting the export costs z −57.

**ICSG, raw:** the 2014 yearbook's Table 4 (sheets `Smelting_1/2`) has no DRC row for 2004–2013. The 2023
and 2025 yearbooks list `Congo DR · Smelter Primary` at 0.000 for 2013–2018, 1.5 for 2019, then 70–159 kt
(the Lualaba smelter's years). The parse drops the exact zeros (`icsg_2023f.py:295`), so the feed reads
2004–2018 as missing, and the NetworkImputer refills that gap differently year by year:

| DRC, kt CC | conc exports (BACI) | conc→smelting, prior → post | CFR→anode, prior → post | anode exports (BACI), prior → post |
|---|---:|---:|---:|---:|
| 2008 | 143.3 | 17.1 → 32.6 | 106.7 → 46.0 | 86.8 → 30.2 |
| 2010 | 109.7 | 0.0 → 3.0 | 0.0 → 8.6 | 63.9 → 8.6 |
| 2012 | 116.9 | 4.0 → 3.9 | 1.9 → 2.0 | 96.1 → 2.0 |
| 2013 | 195.7 | 21.0 → 21.2 | 17.5 → 23.2 | 76.5 → 23.2 |
| 2014 | 13.4 | 115.9 → 115.3 | 108.6 → 108.7 | 62.0 → 62.1 |
| 2019 | 82.1 | 1.5 → 1.6 | 100.3 → 10.4 | 65.2 → 0.0 |
| 2021 | 156.8 | 130.7 → 138.3 | 133.9 → 129.6 | 97.8 → 93.0 |

The pattern holds in every year. When BACI's concentrate exports leave room in DRC's own concentrate, the
imputer builds a smelter that ICSG does not report, and the anode export survives (2014). When they don't,
the export is deleted (2010–2012). In 2020–2024 ICSG's smelter carries the exports and DRC is quiet (4,728).

Read together, the 2004–2019 years are a source contradiction (C1). BACI's DRC 7402 exports plus its
concentrate exports exceed what ICSG's mine and smelter series let DRC produce. They are plausibly the
blister and black copper of small Katanga furnaces, which ICSG's smelter table does not list (Bloomberg,
2008: six Lubumbashi furnaces at ≈ 100 t blister a month). This REP does not settle that.

**2019, and defect F2.** CFR's supply total, c 100.3, is BACI's 65.2 kt of exports plus 35.1 kt on
`Anode→ER` from ICSG's DRC primary refining. But the 2025 yearbook revised that series to 0.000 for
2015–2024. `icsg_comb.py:271` keeps the newest vintage per cell, and the zero never reaches it, so the 2023
yearbook's 35.0 kt survives for 2015–2022. Section V lists every cell a newer yearbook revised away: the 8
DRC refinery cells (282 kt) and 9 seam-year singletons (Thailand 2004 smelter and refinery 39.6 kt, Brazil
2004 semis 143.7 kt, five of ≈ 1 kt). 2019's `supplytotout ER` and `usetotout CER` rows (z −32.1 each, CC and
Mass) are this stale value.

## 3. South Africa

**2017–2019, defect F1.** ICSG's smelter intake (72.5 kt in 2017, 51.8 kt in 2018) and refinery intake
(70.2 kt in 2018, 15.0 kt in 2019) are in `values.csv`, but no edge receives them: `trade_shares` v39 has no
row for those groups. ADR 0027's ledger flags them `imports_exhausted` (`build_baci_reexport.py:248`:
`T > 0 and T ≈ I and U > P`), and `build_trade_shares.imports_exhausted_groups` withholds every flagged
group. The ADR's premise is that the netting *constructed* the zero import sum, but the gross imports here were
0.11, 0.03, 0.26 and 0.18 kt. A zero that small was already observed, and ADR 0018's all-domestic fan should
fire. Section W:

| imports_exhausted groups | n | ICSG usage withheld, kt CC |
|---|---:|---:|
| all | 71 | 1,567 |
| netting not applied (`T_applied` NaN, label `untouched`) | 24 | 311 |
| gross imports below the feeds' 0.3 kt resolution (includes all 24 above) | 54 | 867 |
| material imports, netted (the case ADR 0027 means) | 17 | 700 |

The 54 groups are DRC anode (16) and matte (11), Italy matte (15), Argentina matte (10), South Africa
concentrate and anode (2017–2019) and Portugal cathode. The 17 that stay withheld are the hubs ADR 0027 was
written for: UAE, Singapore, Slovakia, Portugal and South Africa's 2020–2024 cathode.

**2020–2024, contradiction C2.** ICSG's 2023 and 2025 yearbooks report SA smelter and refinery output as
zero (1e-8 to 1e-17 kt) from 2020; BACI books SA anode exports of 96.5, 137.2, 121.5, 52.5, 47.4 kt CC. The
solver keeps ICSG and deletes the exports (2021: 137.2 → 4.5, z −56.8). Its cathode use (ICSG semis,
30 kt a year) loses its supply too: with no production, the netting reads SA's cathode imports as
re-exports and withholds the group. I could not confirm from public sources whether Palabora's smelter
stopped in 2019/2020. BACI's 7402 exports could also be transit metal from Zambia or DRC, booked as SA exports.

## 4. Zambia and Uzbekistan

**Zambia (C3).** BACI books Zambian cathode to Switzerland at 507, 699, 462 and 585 kt Mass in 2013, 2014,
2015 and 2017. The CC legs carry no such tonnage, and Zambia's ICSG refined output is 426–568 kt. These are
merchanting destinations, the cathode twin of the anode trade 8ac5720 documented and left to the solver.
The posteriors are 14.2, −14.2, −8.3 and 28.0 kt. The 21,460 of "other rows" (e.g. 1995 smelting supply Mass
446.7 → 785.0, z +44.4) are not traced here.

**Uzbekistan (C4).** In 1995, ICSG has mine output at 40.0 kt CC, smelter intake at 87.0 kt and refined
output at 90 kt. BACI records 0.1 kt of concentrate imports and 4.1 kt of cathode exports, and ICSG's semis
use is 10 kt. The posterior meets in the middle: smelting 32.0, refinery 28.9, cathode use 28.9 (z +61.3 on
the Mass row). 1996–1998 repeat it with exports of 42, 25 and 20 kt against 85–111 kt refined. This is BACI's
mirror coverage of a non-reporting Central Asian exporter, plus a 1995 mine figure below the smelter's
intake. Both are in the data, not the model.

## 5. Fix options

| | Change | Expected effect | Risk |
|---|---|---|---|
| F1 | Withhold an `imports_exhausted` group only where the netting was applied and gross imports reach 0.3 kt (ADR 0025's resolution); the rest takes ADR 0018's all-domestic fan. ADR 0027 amendment. | 54 groups, 867 kt CC of ICSG usage back on edges. SA 2017–2019 (7,255) should fall by most of it. | DRC's released refinery (12–35 kt a year) and matte usage add demand on a node ICSG gives no smelter in 2004–2018, so DRC's z² can **rise**. Italy and Argentina matte move. |
| F2 | At combine, let a newer yearbook's exact zero supersede the older value, then drop it as now. Zeros stay "not reported" downstream (#97 rejected observed zeros). | DRC refining 2015–2022 35 → none; the 2019 ER/CER rows (≈ 4,100) go. Thailand 2004 and Brazil 2004 lose seam values. | Small; touches 17 cells. |
| C1–C4 | State as limitations (ADR 0019); no σ change, no row dropped. | None on z². | None. |

**Outcome (2026-10-22):** F2 ships (`opt_20261022_112855_928da10c`, common set −0.672 %). F1 is rejected on
measurement (`opt_20261022_105118_b7a659a8`, +1.67 % together with F2), because the usage it releases collides with
C1, with South Africa's 2017–2018 concentrate exports and with Portugal as a cathode non-producer. See
[the measurement REP](analysis_opt_20261022_112855_928da10c_20261022_issue111-zero-revision.md).

F1 and F2 interact on DRC and should be built and measured as one chain: ICSG parse → `icsg_comb` →
`process_icsg` → `icsg_clean --baci-version 28 --to-feed` → `build_baci_reexport --source-version 28` →
`baci_feed` → `build_trade_shares` → `icsg_feed` → Stages 2–4.

**#31 and #96.** Neither defect sits behind their gaps. No withheld group is Japan, Korea, Belgium or Germany
concentrate (#31), or France, Netherlands, Taiwan or Switzerland cathode (#96), and no stale cell touches
them. By the rule recorded on both issues, they close as not planned unless the F1/F2 measurement moves
their phantom chains.

## Reproduce

```bash
python scripts/explo/issue111_four_region_diag.py opt_20261021_131744_58504ad3
```

Artefacts: `data/proc/opt/opt_20261021_131744_58504ad3/timesteps/t_*.npz`;
`data/proc/const/const_20261021_130426_c5331a7f/` (row_meta, σ_c via `SolverStatsDiagnostic`);
`data/proc/imp/imp_20261021_125954_b112b170/{agg_result.npz, imputation_log.parquet}`;
`data/input/feed/icsg/current/values.csv`; `data/input/feed/baci/current/reexport_ledger.csv`;
`data/input/params/trade_shares/current/params.csv` (v39); `data/proc/feedprep/icsg_clean/icsg_{2005,2014,2023,2025}_flows.csv`,
`icsg_flows_comb.csv`; raw `data/input/raw/ICSG_Yearbooks/ICSG {2014,2023,2025} Statistical Yearbook.xlsx`
(sheets `Smelting_1/2`, `4.Sml`, `5.Ref`).
