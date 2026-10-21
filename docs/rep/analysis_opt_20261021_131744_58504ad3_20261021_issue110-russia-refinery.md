# #110: the 2005 yearbook's merged refinery cells pair their values by the SX-EW table — Russia's 1995–2003 scale-total Σz² −98.3 %, common-set z² −20.7 %

**Run:** `opt_20261021_131744_58504ad3` (label `r110_russia`; 30/30 Solved, `qp_variables` defects, gap 1e-8;
`solve_wall_s` 529.3 s)
**Against:** `opt_20261021_121332_acc68135` (#98's reference chain), called *base* below
**Chain:** the 2005 yearbook re-parsed at `8cdb63a` → `icsg_comb.py` → `process_icsg.py` →
`icsg_clean.py --baci-version 28 --to-feed` → **icsg v17**; `build_baci_reexport.py --source-version 28` → **baci v31** →
`feed_baci_20261021_125615_bc691704`; `build_trade_shares.py` → **trade_shares v39**; `feed_icsg_20261021_125725_b033ebc4`;
with `feed_crude_ore_20260603_095651_558f7369` → `imp_20261021_125954_b112b170` → `const_20261021_130426_c5331a7f`
**Feeds/params:** **icsg v17**, **baci v31** (gross v28), **trade_shares v39**, crude_ore v3, acc copper v4, conc v10,
eff v4, icsg_aggregates v4, ore_grade v8, scrap_share v3, stock_bounds v6, stripping_ratio v2 (bold: changed)
**Date:** 2026-10-21 · **Branch:** `diag/icsg-2005-electrowon-measure` · **Issue:** #110 · **ADR:** 0035 (amended)

**Verdict: passes all five items; ships.** Russia's 1995–2003 refinery intake was a parse defect: the 2005
yearbook's merged cell `Electrowon Primary Secondary` holds two values, and a positional pairing booked the
primary refined output (500–662 kt a year) as electrowon, which the refinery table drops. Paired by the yearbook's
own SX-EW table, Russia's anode intake is 552–897 kt in 1995–2003 and joins the 2014 vintage's 922 kt in 2004
without a seam. Russia's 1995–2003 scale-total Σz² falls 171,895 → 2,842, its stock legs beyond |z| 5 vanish
(−1,030 kt CC → 0), its mine chain stops being deleted, and common-set sum-row z² falls **20.7 % vs
`opt_20261021_121332_acc68135`**. `opt_20261021_131744_58504ad3` is the
reference chain.

## 1. Change

`icsg_2005.clean_flows` (ADR 0035 am.): where a merged cell holds fewer values than types and the yearbook's
SX-EW table reports no output for the region-year, the cell's Electrowon type is dropped before its values are
paired. A diff of the real 2005 parse changes 41 rows — Russia 1995–2004, Iran 1995–96, Others 1999–2000 — and no
other cell; after `icsg_comb.py` (whose 2014 vintage supersedes 2004) 22 rows of icsg v17 differ from v16, Russia's
and Iran's refinery anode and scrap intake. Stocks, net additions and prices are unchanged.

| Russia, kt CC | 1995 | 1997 | 1999 | 2001 | 2003 | 2004 |
|---|---|---|---|---|---|---|
| refinery anode intake, icsg v16 | 50 | 65 | 241 | 291 | 181 | 922 |
| refinery anode intake, icsg v17 | 552 | 602 | 782 | 897 | 845 | 922 |

## 2. Decision rule and results (posted on #110 before the result)

| Item | Rule | Result |
|---|---|---|
| 1 | Russia's 1995–2003 scale-total Σz² falls ≥ 75 % | **Holds:** 171,895 → 2,842 (−98.3 %); all years 175,519 → 6,462 |
| 2 | Russia's CC stock legs beyond \|z\| 5 sum to ≥ −200 kt CC | **Holds:** 181 legs, −1,030.2 → 0 legs |
| 3 | Common-set scale-total Σz² falls | **Holds:** 952,429 → 755,122 (**−20.72 %**; 28,969 rows in both runs) |
| 4 | Round-E | **Holds.** (a) crude-ore Mass 0.998 → 0.999, Concentrate CC 1.003 → 1.010, Cathode CC 0.999 → 0.997; (b) median 1.000, p90 1.016 → 1.015; (c) Chile 1.021 → 1.022, Peru 0.978, Indonesia 0.916 → 0.917, world 0.947; (d) mass moved 285,108 → 256,349 kt (−10.09 %) |
| 5 | No other region's scale-total Σz² rises by more than 5,000 | **Holds:** largest rise Belgium +195 |

**Predictions:** common set −15 … −20 % — right in direction, short on size (−20.7 %); Russia's `CO→Mining` CC
posterior over 1995–2003 within 10 % of its prior — right (6,278 against 6,226, base 3,506); world crude-ore Mass
retention unchanged within 0.005 — right (+0.001).

## 3. What moves

- **Russia's chain, domestic CC, Σ 1995–2003 (kt CC), prior → posterior:** base `CO→Mining` 6,226 → 3,506,
  `Smelting→Matte` 6,562 → 2,233, `Scrap→Smelting` 1,458 → 197, `CP→Cathode` 1,489 → 2,022; new 6,226 → 6,278,
  6,562 → 6,628, 1,458 → 1,470, 6,574 → 6,544. The deletion was the solver's way to meet a refinery that read a
  tenth of its smelter output; the stock draws met BACI's cathode exports. The phantom SX-EW the solver built in
  Russia before #97 (700 kt CC) filled the same gap.
- **By sector, common-set Δz²:** Smelting −51,374, Matte −32,512, Cathode −25,726, Concentrate −17,075, MF −15,453.
  By year, 1995 −39,130 to 2000 −10,081; nothing after 2003 moves by more than 50.
- **Iran** −27,255: its 1995–96 refinery read 4 kt of intake against 116–122 kt CC of smelter output.
- **Elsewhere** the netted BACI moves 8,803 legs, 95.3 % of them by less than 0.1 kt (largest 20.2 kt, Σ|Δ| 428 kt),
  because re-export shares weight each origin by its production; no region outside Russia, Iran and Uzbekistan (−1,210) moves by more than 200.

## Reproduce

```bash
python scripts/parse/icsg_parse/icsg_2005.py && python scripts/parse/icsg_parse/icsg_comb.py \
    && python scripts/parse/icsg_parse/process_icsg.py
python scripts/parse/icsg_parse/icsg_clean.py --baci-version 28 --to-feed
python scripts/build_baci_reexport.py --source-version 28                 # → icsg v17, baci v31
python scripts/baci_feed.py && python scripts/build_trade_shares.py && python scripts/icsg_feed.py
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261021_125725_b033ebc4 \
    --feed-id feed_baci_20261021_125615_bc691704 --feed-id feed_crude_ore_20260603_095651_558f7369 --label r110_russia
python scripts/constraints.py --imp-run-id imp_20261021_125954_b112b170 r110_russia
python scripts/run_solver.py --const-run-id const_20261021_130426_c5331a7f --label r110_russia
python scripts/explo/issue110_russia_refinery_diag.py opt_20261021_121332_acc68135 opt_20261021_131744_58504ad3
```
