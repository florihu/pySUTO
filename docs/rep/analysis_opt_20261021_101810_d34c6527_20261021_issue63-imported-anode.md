# #63: imported anode leaves ADR 0030's direct-to-refinery route — Austria's sum-row z² −90.9 %, common-set z² −6.72 %

**Run:** `opt_20261021_101810_d34c6527` (label `r63_anode`; 30/30 Solved; `prior_mode` cold; `qp_variables`
defects, gap 1e-8)
**Against:** `opt_20261021_094942_d1ac7a81` (#95's reference chain), called *base* below
**Chain:** gross baci **v28** → `icsg_clean.py --baci-version 28 --to-feed` → icsg **v16** →
`build_baci_reexport.py --source-version 28` → baci **v30** → `feed_baci_20261021_095637_d0ec13f1` → trade_shares
**v38** → `feed_icsg_20261021_095747_f99c73a2`; with `feed_crude_ore_20260603_095651_558f7369` →
`imp_20261021_100015_ea05007a` → `const_20261021_100441_814be306` → the run
**Feeds/params:** **icsg v16**, **baci v30**, **trade_shares v38**, crude_ore v3, acc copper v4, conc v10, eff v4,
icsg_aggregates v4, ore_grade v8, scrap_share v3, stock_bounds v6, stripping_ratio v2 (bold: changed)
**Date:** 2026-10-21 · **Branch:** `diag/imported-anode-measure` · **Issues:** #63 · **ADR:** 0030 (amended
2026-10-21)

**Verdict: passes; ships.** All five items fixed on #63 before the result hold. Austria's common-set sum-row z²
**68,232 → 6,196 (−90.9 % vs `opt_20261021_094942_d1ac7a81`)**, its 2015–2024 scrap-chain rows 51,644 → 334;
Belgium −20.8 %, Italy −1.3 %; common-set sum-row z² **1,018,103 → 949,738 (−6.72 %)**; Round-E holds; no stock
leg newly above |z| 5. `opt_20261021_101810_d34c6527` is the reference chain.

## 1. Change

ICSG's secondary refined production also holds the imported blister and anode a region refines beyond its
primary output, and ADR 0030's nesting booked that copper as direct-to-refinery scrap (Austria 2019: 128.1 − 60.0 ·
0.9845 = 70.1 kt CC, against ≈ 60 kt CC of imported anode). The parse now subtracts it:
`A_imp = min(max(0, eff_ER·N − P), S_ref)`, `Scrap→CFR = max(0, S_ref − S_sm·k − A_imp)/k`, with `N` the region's
net cross-region imports of HS 740200 from the registered gross BACI parse (baci v28, gross mass × the exporter's
anode concentration 0.94, conc v10) and `P` its ICSG primary refined output. 70 region-years lose 920 kt Cu of
direct route (`data/proc/feedprep/icsg_clean/icsg_imported_anode.csv`). The ICSG input now depends on a gross BACI
version, recorded in its `parents.json`; `build_baci_reexport.py` refuses to net another one (build order in
CLAUDE.md and README).

## 2. Decision rule and result (posted on #63 before the chain ran)

| Item | Rule | Result |
|---|---|---|
| 1 | Austria's common-set sum-row z² falls ≥ 50 % | **Holds:** 68,232 → 6,196 (−90.9 %). By sector: Smelting −22,982, CFR −17,899, Anode −9,551, Matte −4,050, ER and CER −2,663 each, Cathode −1,256. |
| 2 | Belgium and Italy rise ≤ 10 % | **Holds:** Belgium 18,859 → 14,944 (−20.8 %), Italy 46,141 → 45,540 (−1.3 %). |
| 3 | Round-E (ADR 0025) | **Holds.** (a) crude-ore Mass 0.998, Concentrate CC 1.003, Cathode CC 1.001, all unchanged; (b) small-node median 1.000 / 1.000, p90 1.132 → 1.124; (c) Chile 1.020, Peru 0.978, Indonesia 0.916, world 0.947, unchanged; (d) mass moved 301,004 → 298,356 kt (−0.88 %). |
| 4 | Common-set sum-row z² rises ≤ 1 % | **Holds:** −6.72 % (27,439 rows in both runs). |
| 5 | No stock leg newly above \|z\| 5 | **Holds:** 307 → 307, none new. |

**Predictions:** Austria −60 … −85 % — wrong on size (−90.9 %); Austria's 2019 smelter-scrap posterior within 20 %
of its anchor 62.2 kt CC — right (z² 2,966.6 → 0.3); common set −2 … −6 % — wrong on size (−6.72 %).

## 3. What else moves

- **Austria 2019:** direct-refinery-scrap anchor 70.1 → 13.4 kt CC (z² 2,217.3 → 0.7). Cathode-use rows 2015–2024
  Σz² 2,014 → 666; 1995–2014 2,621 → 2,714. Cathode exports 2015–2024, prior/post: 1,132/999 → 1,132/1,070 kt CC.
- **Other regions:** Slovakia −1,722 (22,722 → 21,000), Belgium −3,915, Italy −602; the largest rise is Russia +66.
- **Left:** Austria's 2006–2012 smelter values are still the 2014 yearbook's vintage (#63's defect 3), a stated
  limitation; `production_recipes` (the netting's P) keeps ADR 0030's old split, which moved nothing measurable
  here (Austria's Matte rows −4,050).

## Reproduce

```bash
python scripts/parse/icsg_parse/icsg_clean.py --baci-version 28 --to-feed
python scripts/build_baci_reexport.py --source-version 28          # → baci v30, icsg v16 registered
python scripts/baci_feed.py && python scripts/build_trade_shares.py && python scripts/icsg_feed.py
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261021_095747_f99c73a2 \
    --feed-id feed_baci_20261021_095637_d0ec13f1 --feed-id feed_crude_ore_20260603_095651_558f7369 --label r63_anode
python scripts/constraints.py --imp-run-id imp_20261021_100015_ea05007a r63_anode
python scripts/run_solver.py --const-run-id const_20261021_100441_814be306 --label r63_anode
python scripts/explo/issue92_refined_scope_diag.py opt_20261021_094942_d1ac7a81 \
    opt_20261021_101810_d34c6527 29 30 28 28
```
