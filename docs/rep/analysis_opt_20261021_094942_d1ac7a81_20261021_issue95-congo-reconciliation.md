# #95: the larger of 178's and 180's refined declarations, DR Congo's transit legs dropped — the UAE's cathode rows close, common-set z² −5.28 %

**Run:** `opt_20261021_094942_d1ac7a81` (label `r95_congo`; 30/30 Solved; `prior_mode` cold; `qp_variables` defects,
gap 1e-8; `solve_wall_s` 525 s per the run's sidecar)
**Against:** `opt_20261020_112726_2f451f31` (#33's reference chain), called *base* below
**Chain:** `baci_parse_feed.py` at `7958da3` → baci **v28** (gross) → `build_baci_reexport.py --source-version 28` →
baci **v29** → `feed_baci_20261021_092806_7795f503` → trade_shares **v37** (its `parents.json` names that run, baci
v29, icsg v15, conc v10, eff v4) → `feed_icsg_20261021_092916_aadab46c`; with
`feed_crude_ore_20260603_095651_558f7369` → `imp_20261021_093159_8b0afa00` → `const_20261021_093627_9b2ad0ec` → the
run
**Feeds/params:** icsg v15, **baci v29**, **trade_shares v37**, crude_ore v3, acc copper v4, conc v10, eff v4,
icsg_aggregates v4, ore_grade v8, scrap_share v3, stock_bounds v6, stripping_ratio v2 (bold: changed)
**Date:** 2026-10-21 · **Branch:** `diag/congo-reconciliation-measure` · **Issues:** #95, #92, #88 · **ADR:** 0014
(amended 2026-10-20, #95)

**Verdict: passes; ships.** All five items fixed on #95 before the result hold. The UAE's cathode-use rows, which
re-exported DR Congo cathode they no longer imported under #88's drop, close: Δz² −59,021 on the common set plus
12,531 on four base-only rows. Common-set sum-row z² **1,074,905 → 1,018,123 (−5.28 % vs
`opt_20261020_112726_2f451f31`)**; no stock leg newly above |z| 5. `opt_20261021_094942_d1ac7a81` is the
reference chain.

## 1. Change

Evidence and rule in ADR 0014's amendment (#95); read-only on BACI HS92 V202601 (2026-10-20):
- **178 vs 180.** In 2015–2023 the parse keeps, per year and partner, the larger of 178's ("Congo", partners'
  declarations) and 180's (DR Congo's own) refined-copper exports, pooled over HS 7403.1x, instead of dropping 178
  (#88). The smaller declaration dropped: 391.5 kt of 178's, 1,331.1 kt of 180's. 178's other rows in the window
  stay dropped as under #88 (concentrate 37.1, anode 23.1, matte 2.2 kt).
- **Transit declarations.** DR Congo's (180) refined copper into Tanzania and Mozambique is dropped: 1,553 kt and
  382 kt in 2015–2023, against 213 kt and 16 kt of those countries' own exports and no refinery. The parse logged
  302.3, 130.5, 35.6, 98.0, 264.9, 247.4, 196.3, 134.7, 143.2 kt into Tanzania (2015–2023, 0.1 in 2024) and 15.8,
  61.1, 28.3, 47.2, 103.0, 126.0 kt into Mozambique (2018–2023).
- **Checked and set aside:** HS 740311/740319 duplicates per pair (12 % of 740319 tonnage sits in pairs with a
  comparable 740311 leg); Kazakhstan's exports above its production, which the ICSG 2025 yearbook itself states
  (298–366 kt total refinery 2015–2024, parsed exactly) — an ICSG–BACI contradiction, kept as a limitation.

## 2. Decision rule and result (posted on #95 before the parse finished)

| Item | Rule | Result |
|---|---|---|
| 1 | Gross diff confined to the declared rows plus ADR 0033's copper-price knock-on | **Holds.** 911 rows differ from gross v26: 755 concentrate legs moved by ADR 0033's reconstruction (Σ\|Δ\| 86.6 of 146,857 kt, 0.059 %; 610 through a changed `g_ref`, 122 through the copper price alone), the rest DR Congo's declared rows. Copper price moves −0.15 … +0.22 %. |
| 2 | The UAE's sum-row z² falls ≥ 50 % | **Holds.** Its cathode-use rows 2012–2022 go from c 160–283 kt CC (z² 2,690–3,084 each) to c ≈ 0 (z² 0–42): the netting now re-attributes the UAE's re-exports to DR Congo. Common-set Δz² −59,021, base-only rows −12,531. |
| 3 | Fewer stock legs above \|z\| 5 | **Holds** on all stock legs: 313 → 307, none new; all CC stock legs' net posterior −873 → −603 kt CC, 1995–2024. |
| 4 | Round-E (ADR 0025) | **Holds.** (a) crude-ore Mass 0.998 / 0.998, Concentrate CC 1.003 / 1.003, Cathode CC 1.002 / 1.001; (b) small-node median 1.000 / 1.000, p90 1.129 / 1.132; (c) Chile 1.020, Peru 0.978, Indonesia 0.916, world 0.947, all unchanged; (d) mass moved 303,845 → 301,004 kt (−0.93 %). |
| 5 | Common-set sum-row z² rises ≤ 1 % | **Holds:** −5.28 %. |

**Predictions:** the UAE falls > 80 % — right (≈ 100 %); DR Congo's draws shrink in 2019–2022 — right on its
refinery (CP→Cathode 2019–2023 posterior above prior +455 → +235 kt CC); common set −0.5 … −2 % — wrong on size
(−5.28 %, the UAE alone −59,021).

## 3. What else moves

- **By region:** the UAE −59,021, Italy −776 (cathode use 2016), Austria −465, DR Congo −174; ROW **+3,288**:
  its cathode-use totals rise in 2021–2023 (c 510 → 697, 669 → 872, 663 → 859 kt CC) as DR Congo's transit
  copper leaves the Tanzania and Mozambique legs; India +457, Zambia +200.
- **By year:** −5,000 to −7,600 in each of 2012 and 2014–2022; 2023 +1,228.
- **DR Congo's cathode exports 2019–2023,** prior/post: to the UAE 1,538/988 (base 473/460), ROW 1,070/1,039 (base
  2,308/2,166), China 4,038/3,995.

## Reproduce

```bash
python scripts/parse/baci_parse/baci_parse_feed.py                 # → baci/v0
#   baci current/: set aside reexport_*.csv, copy v0/{values,quantity_recon,quantity_cv_scale}.csv in,
#   then InputRegistries.initialize(root="data/input")            # → baci v28 (gross)
python scripts/build_baci_reexport.py --source-version 28         # → baci v29
python scripts/baci_feed.py && python scripts/build_trade_shares.py && python scripts/icsg_feed.py
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261021_092916_aadab46c \
    --feed-id feed_baci_20261021_092806_7795f503 --feed-id feed_crude_ore_20260603_095651_558f7369 --label r95_congo
python scripts/constraints.py --imp-run-id imp_20261021_093159_8b0afa00 r95_congo
python scripts/run_solver.py --const-run-id const_20261021_093627_9b2ad0ec --label r95_congo
python scripts/explo/issue92_refined_scope_diag.py opt_20261020_112726_2f451f31 \
    opt_20261021_094942_d1ac7a81 27 29 26 28
```
