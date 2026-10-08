# Round Z: BACI concentrate legs booked in kilograms are rebuilt from their value

**Run:** `opt_20261008_092950_625a6c7e` (label `roundZ_lowside`; 28 Solved + 2018 and 2022
AlmostSolved; `prior_mode` cold)
**Against:** Z.base `opt_20261008_083810_56f905e9` (label `roundZ_base`, 30/30 Solved), which differs
only in the parse; and the reference `opt_20261007_130833_1a9b7af7` (29 Solved + 2022 AlmostSolved)
**Chain:** Z `feed_baci_20261008_084113_ebca1805` + `feed_icsg_20261008_084222_e1b5a044` →
`imp_20261008_084450_041271d3` → `const_20261008_084916_293ff701` → the run. Z.base
`feed_baci_20261008_074910_7ccef558` + `feed_icsg_20261008_075020_da834f5b` →
`imp_20261008_075249_bb1e1884` → `const_20261008_075717_9ccd0d40` → `opt_20261008_083810_56f905e9`
**Feeds/params:** Z: baci gross **v13** (new parse) → netted **v14**, trade_shares **v23**. Z.base:
baci gross v10 → netted **v12**, trade_shares **v22**. Both: icsg v12, crude_ore feed
`feed_crude_ore_20260603_095651_558f7369`, acc copper v4, conc v10, eff v4, ore_grade v7 (#30 not
refreshed in either), stock_bounds v5, scrap_share v3, stripping_ratio v1
**Date:** 2026-10-08 · **Branch:** `fix/baci-low-side-recon` · **Issue:** #48 · **ADR:**
[0033, amended 2026-10-08](../adr/0033-baci-quantity-contradicts-its-own-value.md)

**Verdict: accepted.** All four pre-registered predictions hold.

- **Prior:** the world Concentrate `Mass` prior in 2020 falls 105,492 → 65,491 kt. SD-R-09
  Concentrate `Mass` retention in 2019–2022 is now 0.950–0.962, against 0.584–0.725 before.
- **CC fan:** PNG's share of the Philippines' CC concentrate fan falls from 0.807 to 0.033 in
  2020, and its share of ROW's from 0.922 to 0.021.
- **Congo D.R. → Zambia 2020:** the z −36.5 leg sits at +0.4.
- **Null:** the gross-level null holds exactly, and Chile's forced term is unchanged.
- **z²:** common-set sum-row z² is **−2.79 % vs `opt_20261008_083810_56f905e9`**, or −3.05 % if
  2018 is left out. Z's AlmostSolved 2018 is shown below to have stopped short of its own optimum.
- **Two costs are reported, not hidden.** Congo D.R. 2020 gains 66.7 kt CC of forced deletion
  (the ambiguous case). Exporter retention ratios fall by 0.4–0.5 pp, a denominator effect: the
  corrected fan raises those exporters' priors, and their posteriors rise.

---

## 1. Defect

ADR 0033's reconstruction was one-sided, so a concentrate leg whose value buys about 1/1000 of its
recorded mass passed untouched. PNG's HS 260300 exports in 2019–2022 are kilograms booked as
tonnes. PNG → Philippines 2020 is 21,277,530 t at 76,901 kUSD, which is 3.6 USD/t against a copper
price of 6,224 USD/t, a value-implied Cu fraction of 0.0006. The diagnosis is in #48 and in the ADR
0033 amendment. In the reference run it shows up three ways:

- **Mass:** the false mass inflates the world Concentrate `Mass` prior directly.
- **CC:** through the trade-share fan, the false mass takes the Philippines', ROW's and India's CC
  intake. The solver then deletes it at z −9.2 / −6.5 / −14.6.
- **Congo D.R. → Zambia 2020:** the leg has the same shape (31 USD/t).

## 2. Change (build ticket, commits `1e04ff1`, `3248051`)

A leg ≥ 1 kt whose fraction is below 0.02 (floor / 10), in an exporter-year below 0.20 (the floor),
is rebuilt downwards. It uses the same `q = v / (P_cu · g_ref)`, is never raised above `q_obs`, and
carries its exporter's ADR 0033 `cv_scale`. The above side is unchanged. No other code changed;
every downstream artefact was rebuilt in the documented order (§9).

## 3. Pre-registered predictions

| | prediction | measured (Z) | |
|---|---|---|---|
| **Z-1** | PNG export mass at the parse 468 / 391 / 329 / 334 kt (2019–22) | gross 468.5 / 391.3 / 328.8 / 334.6; netted and Stage-2 Mass prior 468.4 / 391.3 / 328.8 / 334.0 (was 22,248 / 40,813 / 21,607 / 28,381) | **holds** |
| **Z-2** | world Concentrate Mass prior 2020 ≈ 65 Mt; SD-R-09 Mass retention ≥ 0.90 in 2019–22 | 65,491 kt; 0.962 / 0.950 / 0.957 / 0.950 | **holds** |
| **Z-3** | PNG's share of the PHL CC fan 2020 falls from 81 % toward 6 %, of ROW's from 92 %; PNG CC export prior ≤ 0.2 Mt in 2019–22 | PHL 0.807 → **0.033** (2018: 0.056), ROW 0.922 → **0.021**; 119.2 / 103.8 / 88.7 / 86.6 kt | **holds** |
| **Z-4** | Congo D.R. → Zambia 2020 ≈ 38 kt at the parse; its z (Mass −36.5, CC −10.3) relieved | 37.6 kt; Mass z −36.52 → **+0.36**, CC z −10.28 → **−2.54** | **holds** |

The 2018 PNG CC export prior is 92.8 kt (#48 quoted ≈ 96). PNG's 2020 legs in Z are small and
lightly moved. The one visible residue is PNG → Philippines CC: prior 8.49 → posterior 11.19 kt at
σ_a 0.58, z +4.6.

**SD-R-09, Concentrate (Flow → Process ∪ SB), prior / posterior kt and retention:**

| year | Mass, Z.base | **Mass, Z** | CC, Z.base | **CC, Z** |
|---|---|---|---|---|
| 2018 | 64,063 / 61,836 · 0.965 | 64,091 / 61,694 · **0.963** | 0.949 | **0.934** ¹ |
| 2019 | 85,349 / 61,208 · 0.717 | 63,716 / 61,266 · **0.962** | 0.934 | **0.944** |
| 2020 | 105,492 / 61,637 · 0.584 | 65,491 / 62,186 · **0.950** | 0.904 | **0.922** |
| 2021 | 87,446 / 63,389 · 0.725 | 66,244 / 63,368 · **0.957** | 0.914 | **0.923** |
| 2022 | 94,745 / 63,973 · 0.675 | 67,055 / 63,670 · **0.950** | 0.924 | **0.915** ¹ |
| 2023 | 71,080 / 67,628 · 0.951 | 71,085 / 67,705 · **0.952** | 0.921 | **0.922** |

¹ AlmostSolved in Z (§5). The Mass posterior barely moves (61.2–63.9 Mt in both runs): the solver
had already deleted the phantom. What changes is the prior, and with it the CC geography.

## 4. The pre-registered null (Z-h)

- **Gross values file: holds exactly.** v13 differs from v10 on the 119 below-flagged legs only
  (115,854.0 → 390.3 kt). The other 78,193 rows are byte-identical, and the 881 above-band
  reconstructions are identical. This was checked before netting.
- **Chile: holds.** Forced deletion is 295 kt CC with 5 of 30 years hard-capped in all three runs.
- **Exporter-years outside the flagged ones: holds once the prior is held fixed, fails as worded.**
  Cross-region Concentrate posteriors of exporter-years with no below-flagged leg move as follows,
  in kt over 30 years:

| Z vs Z.base | Mass | CC |
|---|---|---|
| all unflagged exporter-years | 5,174 (0.78 % of Σpost) | 1,650 (0.90 %) |
| … whose prior moved in either valuation (275 exporter-years) | 4,877 | 1,499 |
| … whose prior did not move | **297** (0.04 %) | **152** (0.08 %) |
| for scale: reference vs Z.base, prior unchanged (#30 + noise) | 64 | 84 |

The null as written did not foresee that the corrected fan hands the PNG-vacated smelter intake
(Philippines, ROW, India) to *other* exporters' priors. That reassignment is story 11's intended
effect, not a leak.

On exporter-years whose prior is untouched, the posterior moves within twice the
reference-vs-Z.base level. That gap is the scale of #30's refresh plus solver noise.

## 5. Costs

**Common-set sum-row z²** (27,028 rows present in all three runs; built in memory, no per-year CSV):

| | 30 y | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|---|
| Z.base | 1,442,713 | 48,802 | 70,423 | 83,794 | 90,651 | 59,951 | 34,776 |
| **Z** | **1,402,531** | 51,142 | 65,430 | 73,830 | 75,675 | 56,778 | 34,641 |
| Δ vs Z.base | **−2.79 %** | +4.79 % | −7.09 % | −11.89 % | −16.52 % | −5.29 % | −0.39 % |

**2018 is solver inexactness, not the round (ADR 0039 cross-evaluation).** Z.base's 2018 posterior
is feasible on Z's own 2018 QP: hard violations are 1.6e-7 on the zero cone and 7e-13 on the nonneg
cone. It scores J = 143,828.6 there, against **146,243.1** for Z's own answer, which is **2,415
lower** (AlmostSolved, gap 10.1, last step 0.00). The method reproduces the logged objective (Z 2018
−1,083,918.5 vs −1.0839e+06). Without 2018 the move is **−3.05 %**.

The method is not informative in 2020 or 2022. There the priors differ so much that each borrowed
posterior scores millions higher, so Z's AlmostSolved 2022 (gap 69.6) is judged only by its
direction. The reference was also AlmostSolved in 2022.

**Where the gain sits (ADR 0025 (d)), Δz² by |c| band:** < 10 kt −2,416; 10–100 kt −2,904;
100–1,000 kt **−17,637**; ≥ 1,000 kt **−17,227**. The gain is on the large concentrate and smelting
totals the phantom fan had distorted, not a repricing of small anchors.

**Round-E items (ADR 0025), Z.base → Z:**

- **(a) World retention:** CO Mass 0.862 → 0.860, Concentrate CC 0.990 → 0.990, Cathode CC
  0.988 → 0.987. Every move is under 0.01, so this passes.
- **(b) Small nodes (|c| < 18 kt):** posterior/prior median 1.000 → 1.000, p90 1.144 → 1.140.
  Passes.
- **(c) Exporter retention:** Chile 0.969 → **0.964**, Peru 1.014 → **1.009**, Indonesia
  0.915 → **0.911**, world 0.924 → 0.939.
  - The letter fails. The intent passes, as in ADR 0025's own (a).
  - The fall is in the denominator. The corrected fan raises these exporters' CC priors.
    Chile 2020: 3,292 → 3,467 kt, of which India 49.5 → 135.7, ROW 5.1 → 63.1 and Philippines
    0 → 5.7.
  - Their posteriors rise. Chile 2019–2021 goes 3,181 / 3,113 / 2,999 → 3,208 / 3,159 / 3,018.
  - In Z.base the solver had to invent Chile → India above its prior (49.5 → 115.5). In Z the
    prior carries it (135.7 → 122.7).
  - The one posterior fall is Chile 2018 (2,799 → 2,739), in Z's inexact year.
- **Mass moved, all CC edges:** 502,534 → 499,701 kt (−0.56 %).

**Congo D.R. 2020 gains 66.7 kt CC of forced deletion** (ADR 0029 total 2,905 → 2,974; the only
other mover is Botswana 2010, +1.8). The rebuilt leg lowers Congo's export mass X, and with it its
discharge cap `0.35·X + U`. This is the case #48 asked to keep visible, and it remains ambiguous.

- **Against a real ore lot:** 1,964 kt at 31 USD/t has a value 1/40 of the leg's 2011–2018 history
  (1,087–2,392 USD/t at 172–714 kt).
- **For a real ore lot:** both fields are anomalous, and a low-grade ore lot would carry more mass
  per dollar.

The rule rebuilds it, as it would any other leg. If later evidence shows a real ore lot, the remedy
is a sector reclassification of that leg, not the gate.

**ADR 0027 netting delta (story 27):** Concentrate T applied falls 5,590 → 5,149 kt CC (−441).

- **Philippines hub:** 668 → 477, all of it in 2019–2022 (285 → 94). PNG's false imports had
  inflated its surplus `R`.
- **India:** 55 → 30 (2019–22: 39 → 19).
- **Germany, Singapore and Korea:** unchanged at 275 / 50 / 660.
- **Bulgaria 2020:** was a partial hub (T 27.7 kt CC, I 329.9 against U 302.1) only because of
  PNG's false imports. It is now a producer (T 0), so its export legs keep their gross mass, which
  is the Bulgaria 2020 move in §4.
- **PNG:** the netting moved PNG's own netted exports by −0.1 and −0.6 kt (2019, 2022).

The largest unflagged move, ROW 2019, is not netting, because ROW is never a hub (ADR 0027). Its
imports fall from 2,607 to 467 kt Mass (PNG via North Korea). Its intake is no longer 83 % PNG in
the fan, and its export legs are retained instead of deleted: → China, Mass posterior 805 → 1,159
kt against priors of 1,245 / 1,222.

## 6. #30's share (Z.base vs the reference, both at the old parse)

The re-netting on icsg v12 and trade_shares v22 is almost inert on these metrics:

- **z²:** common-set sum-row z² **−0.07 %** (1,443,659 → 1,442,713).
- **Netting:** T applied 5,703 → 5,590 kt CC.
- **Unflagged exporter posteriors:** move 0.05 % (Mass) / 0.10 % (CC), mostly Zambia 2005 and 2007.
- **Unchanged:** SD-R-09, PNG's fan shares and the Round-E items, to the third decimal.

So none of Z's movement is #30's.

## 7. Sensitivity at the parse (story 28, no solve)

Year gate fixed at 0.20, raw BACI HS92 V202601:

| leg gate | legs | exporter-years | rebuilt mass |
|---|---|---|---|
| 0.01 | 57 | 38 | 114,842 → 342 kt |
| **0.02** | **119** | **70** | **115,854 → 390 kt** |
| 0.05 | 218 | 153 | 118,727 → 767 kt |

The flagged mass moves by 2.5 % across a fivefold range of the gate, because it sits below 0.01
(PNG alone is 111,809 kt). The conclusion does not rest on 0.02.

## 8. Follow-ups

- **Reference chain.** Z becomes the reference chain on merge (story 29); merging is the user's
  call. `docs/manuscript/method.md` (trade-defects paragraph) and `si.md` (BACI row of the
  input-correction table) still describe the one-sided rule, and their Concentrate figures come from
  the old reference.
- **Congo D.R. → Zambia 2020.** Keep it under watch for evidence that settles real ore lot versus
  unit error (§5).
- **PNG → Philippines CC 2020** has z +4.6 at σ_a 0.58. The rebuilt fan now gives it a small,
  tightly priced prior. Not acted on.

## 9. Reproduce

From the repo root on `fix/baci-low-side-recon`. Run strictly in sequence; the registries have no
locking.

```
# Z.base: re-net the existing gross version
python scripts/build_baci_reexport.py --source-version 10      # → baci v12
python scripts/baci_feed.py
python scripts/build_trade_shares.py                           # → trade_shares v22
python scripts/icsg_feed.py                                    # sidecar: trade_shares v22
python scripts/collect_aggregate_impute.py --feed-id <icsg> --feed-id <baci> \
       --feed-id feed_crude_ore_20260603_095651_558f7369 --label roundZ_base
python scripts/constraints.py --imp-run-id <imp> roundZ_base
python scripts/run_solver.py --const-run-id <const> --label roundZ_base --prior-mode cold

# Z: new parse → gross version → the same order
python scripts/parse/baci_parse/baci_parse_feed.py             # writes baci/v0/
#   current/: set aside reexport_*.csv, copy v0/{values,quantity_recon,quantity_cv_scale}.csv in,
#   then InputRegistries.initialize(root="data/input")       # → baci v13 (gross)
python scripts/build_baci_reexport.py --source-version 13      # → baci v14
#   … then baci_feed → build_trade_shares (v23) → icsg_feed → Stages 2–4, label roundZ_lowside

python scripts/explo/roundZ_png_kg_diag.py opt_20261008_083810_56f905e9:10 \
       opt_20261008_092950_625a6c7e:13 opt_20261007_130833_1a9b7af7:10
python scripts/explo/roundZ_png_kg_diag.py --crosseval opt_20261008_083810_56f905e9 \
       opt_20261008_092950_625a6c7e 2018 2020 2022
python scripts/explo/roundZ_png_kg_diag.py --sensitivity
```

No per-year tables were written (#46); the diag builds the sum-row frame in memory.
