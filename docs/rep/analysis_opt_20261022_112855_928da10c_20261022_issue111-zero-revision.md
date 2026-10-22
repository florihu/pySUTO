# #111: a newer ICSG yearbook's revision to zero now supersedes the older value. DR Congo's stale refinery goes, and common-set Σz² falls 0.67 %; the withhold threshold (F1) is rejected

**Run:** `opt_20261022_112855_928da10c` (label `r111_f2`; 30/30 Solved; `solve_wall_s` 534.9 s)
**Against:** `opt_20261021_131744_58504ad3` (the final chain of manuscript ready 2), called *base* below. Rejected
variant: `opt_20261022_105118_b7a659a8` (F1 + F2, label `r111_f1f2`)
**Chain:** the 2014 and 2023/2025 yearbooks re-parsed → `icsg_comb.py` → `process_icsg.py` →
`icsg_clean.py --baci-version 28 --to-feed` → **icsg v18**; `build_baci_reexport.py --source-version 28` → **baci v33**
→ `feed_baci_20261022_110715_fdf3b4ac`; `build_trade_shares.py` → **trade_shares v41**;
`feed_icsg_20261022_110826_8a031806`; with `feed_crude_ore_20260603_095651_558f7369` → `imp_20261022_111102_1b49a2b3`
→ `const_20261022_111531_df61bf5c`
**Feeds/params:** **icsg v18**, **baci v33** (gross v28), **trade_shares v41**, crude_ore v3, acc copper v4, conc v10,
eff v4, icsg_aggregates v4, ore_grade v8, scrap_share v3, stock_bounds v6, stripping_ratio v2 (bold: changed)
**Date:** 2026-10-22 · **Branch:** `diag/anode-deletion-diagnosis` · **Issue:** #111 · **ADR:** 0027 (amended:
F1 rejected on measurement)
**Verdict: F2 ships; items 2–5 hold, and item 1 belonged to F1.** The parsers dropped exact zeros, so the 2025
yearbook's revision of DR Congo's primary refining to 0 for 2015–2022 lost to the 2023 yearbook's 35 kt. With zeros
kept to the vintage dedup, those 8 cells go and nothing else in the ICSG feed changes. DR Congo's scale-total Σz²
falls 76,170 → 65,581 (−10,588), and common-set scale-total Σz² falls **0.672 % vs
`opt_20261021_131744_58504ad3`**. `opt_20261022_112855_928da10c` is the reference chain.

## 1. Change

- **F2:** `icsg_2014.py` and `icsg_2023f.py` keep exact zeros. `icsg_comb.latest_vintage` keeps the newest vintage
  per cell and only then drops zeros, so downstream a zero is still "not reported" (#97 rejected observed zeros).
  `remove_total_if_specific_exists` counts nonzero specifics only, the parsers' own rule, so a kept zero cannot
  drop a printed Total.
- **Effect on the data:** every non-zero row of the three re-parsed vintages is identical; 14, 330 and 362 zero
  rows are added. After the combine, exactly 8 rows of `icsg_flows.csv`, `icsg_flows_comb.csv` and the ICSG feed's
  `values.csv` differ from icsg v17: DR Congo `Refining · Primary` 2015–2022, 35.0–36.0 kt, now absent. Stocks and
  NAS rows are unchanged.
- **Trade shares:** v41 differs from v39 by five dropped groups, DR Congo `Anode→ER` in 2017 and 2019–2022.
  2015, 2016 and 2018 were already withheld.

## 2. Decision rule and results (posted on #111 before either run)

| Item | Rule | F1 + F2 (`…105118_b7a659a8`) | F2 (`…112855_928da10c`) |
|---|---|---|---|
| 1 | South Africa's 2017–2019 scale-total Σz² falls ≥ 50 % | **Fails:** 7,255 → 12,564 | n/a (F1's item): 7,255 → 7,232 |
| 2 | DR Congo's 2019 `supplytotout ER` + `usetotout CER` rows < 100 | Holds: 4,105.6 → 0.0 | **Holds:** 4,105.6 → 0.0 |
| 3 | Common-set scale-total Σz² falls | **Fails:** 749,713 → 762,233 (+1.670 %) | **Holds:** 749,713 → 744,674 (**−0.672 %**; 28,972 rows) |
| 4 | Round-E (ADR 0025) | Holds | **Holds:** (a) crude-ore Mass 0.999, Concentrate CC 1.010, Cathode CC 0.997 → 0.998; (b) median 1.000, p90 1.015; (c) Chile 1.022, Peru 0.978, Indonesia 0.917, world 0.947; (d) mass moved 256,349 → 256,225 kt (−0.05 %) |
| 5 | No region's scale-total Σz² rises > 5,000, DR Congo included | **Fails:** South Africa +5,297 | **Holds:** largest rise Uzbekistan +22 |

**Predictions, made for F1 + F2:**
- **Common set −0.5 to −2 %.** Wrong for the pair (+1.67 %); F2 alone gives −0.67 %.
- **South Africa's 2017–2019 smelter chain within 15 % of ICSG.** Right under F1: 2017 72.5 → 61.6, 2018 51.8 → 51.2.
- **C1–C4 within ±5 %.** Wrong for C1, which falls 14.4 % under F2, because its 2019 block carried the stale
  refinery (§3). C2–C4 within 0.1 %.

## 3. What moves under F2

| DR Congo, scale-total Σz² | base | F2 | Δ |
|---|---:|---:|---:|
| 2017 | 2,032 | 1,289 | −743 |
| 2019 | 19,392 | 9,810 | −9,582 |
| 2020 | 2,150 | 2,064 | −86 |
| 2022 | 1,470 | 1,272 | −198 |
| all years | 76,170 | 65,581 | −10,588 |

**2019 in detail.** In base the NetworkImputer filled `CFR→Anode` at 100.3 kt CC from the anode's two uses:
- BACI's 65.2 kt of exports;
- the stale 35.1 kt refinery intake.

That made `supplytotout CFR` a 100.3 kt supply total against ICSG's 1.5 kt smelter (z −52.6). Without the refinery,
the imputer builds no CFR output, and the row is not built (one of 10 base rows outside the common set, z² 5,502
together). The ER and CER rows fall to 0. What is left in 2019 (9,810) is C1: 65.2 kt of BACI anode exports
against a 1.5 kt smelter. No region other than DR Congo moves by more than 22.

## 4. F1 rejected on measurement

F1 changed `imports_exhausted` to need gross imports ≥ 0.3 kt. It released 867 kt CC of ICSG usage in 54 groups to
ADR 0018's all-domestic fan, and in `opt_20261022_105118_b7a659a8` it adds ≈ +17,000 of scale-total Σz² against
F2. The released usage meets contradictions that the withhold had kept out of the prior:

| Where | Δz² | What collides |
|---|---:|---|
| DR Congo 2007, 2013 | +7,208, +6,452 | the 2014 yearbook's DRC refinery intake (22.6, 35.1 kt) against a smelter ICSG does not report: `usetotout Matte` c 9.1 → 65.8 (2007) on `supplytotout Smelting` c 9.1 (C1) |
| South Africa 2017–2018 | +1,573, +2,307 | ICSG's smelter intake (72.5, 51.8 kt) against BACI concentrate exports (67.6, 59.4 kt) equal to the whole mine output (65.5, 46.9 kt): `usetotout Concentrate` c 67.9 → 140.4 (2017) |
| Portugal 2001 | +1,848 | an all-domestic cathode fan for a region with no refinery: `usetotout Cathode·Portugal Mass` c 0.2, z² 0 → 1,751 |
| DR Congo 2005, 2006, 2009 | −436, −1,633, +451 | the same release, cheaper in those years |

The withhold therefore stays as ADR 0027 wrote it (amendment, 2026-10-22). South Africa's 2017–2018 contradiction
(ICSG smelter intake against BACI concentrate exports) joins C1–C4 as a stated limitation.

## 5. #31 and #96

Neither F2 nor the rejected F1 touches their gaps. F2 changes DR Congo's refinery only. F1's 54 groups contain no
Japan, Korea, Belgium or Germany concentrate group (#31) and no France, Netherlands, Taiwan or Switzerland
cathode group (#96). By the rule recorded on both issues, they close as not planned.

## Reproduce

```bash
python scripts/parse/icsg_parse/icsg_2014.py && python scripts/parse/icsg_parse/icsg_2023f.py \
    && python scripts/parse/icsg_parse/icsg_comb.py && python scripts/parse/icsg_parse/process_icsg.py
python scripts/parse/icsg_parse/icsg_clean.py --baci-version 28 --to-feed      # → icsg v18
python scripts/build_baci_reexport.py --source-version 28                     # → baci v33
python scripts/baci_feed.py && python scripts/build_trade_shares.py && python scripts/icsg_feed.py
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261022_110826_8a031806 \
    --feed-id feed_baci_20261022_110715_fdf3b4ac --feed-id feed_crude_ore_20260603_095651_558f7369 --label r111_f2
python scripts/constraints.py --imp-run-id imp_20261022_111102_1b49a2b3 r111_f2
python scripts/run_solver.py --const-run-id const_20261022_111531_df61bf5c --label r111_f2
python scripts/explo/issue111_f1f2_measure.py opt_20261021_131744_58504ad3 opt_20261022_112855_928da10c
```

The F1 + F2 variant is the same chain built with the threshold in `build_baci_reexport.reexport_volumes`, giving baci
v32, trade_shares v40, `imp_20261022_103317_ac4ad80d`, `const_20261022_103750_941e5722` and
`opt_20261022_105118_b7a659a8`.
