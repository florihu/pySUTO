# #94: soft ratio rows carry their stated σ relative to r (#29)

**Run:** `opt_20261019_161937_fd301f58` (label `r94_ratio_sigma_relative`; 30/30 Solved; `prior_mode` cold;
`solve_wall_s` 668 s per the run's sidecar)
**Against:** the reference chain `opt_20261019_155415_a1d7e9a9` (#30, REP
[`…issue30-ore-grade-v8`](analysis_opt_20261019_155415_a1d7e9a9_20261019_issue30-ore-grade-v8.md)), called *base*
below
**Chain:** Stage 3 → 4 on the base's imp run `imp_20261019_153427_7b8966ff` (no Stage-2 code reads the changed σ),
with `fix/ratio-sigma-relative` 67806eb → `const_20261019_160401_ea793c6f` (base `const_20261019_153857_0a993c65`) →
the run.
**Feeds/params:** the base's (icsg v15, baci v27, trade_shares v36, acc copper v4, conc v10, eff v4, ore_grade v8,
scrap_share v3, stock_bounds v5, stripping_ratio v2).
**Date:** 2026-10-19 · **Branch:** `diag/ratio-sigma-measure` (from `fix/ratio-sigma-relative`) · **Issues:** #94,
#29 · **ADR:** 0028 (amended by #29), 0025 (stop rule)

**Verdict: fails on items 1 (as worded) and 2 (Round-E (c), Peru); item 3 holds. Neither failure is a defect, so
no fix ticket follows.**
- **Item 1** fails only on its wording. The stored σ_c also carries the engine's σ_c Lp lift (§3), which the rule
  left out. With the lift inverted, all 217,717 changed rows match the rule to rtol 1e-9 (max deviation 1.0e-15),
  and every one is a soft ratio row.
- **Item 2:** Peru's concentrate exporter retention (CC, 1995–2024) falls 1.011 → 0.986, outside the
  ±0.01-or-up reading of Round-E (c). Its upstream chain, read as Round L.1 requires before an exporter verdict
  (§4.1), shows no deletion: export mass holds (0.998 of prior); the fall is the export grade; and Peru's mine CC
  comes back toward its prior (CO→Mining CC 1.057 → 1.028 of prior).
- **Shipped by the agent** under the maintainer's delegation for the manuscript-ready week (2026-10-19). #29 makes
  the code apply the σ the parameter tables and the manuscript state. `opt_20261019_161937_fd301f58` becomes the
  reference chain.

- **The ratio rows stop being pins.** Their Σz² falls 55,213 → 8,146, but that is mostly the repricing itself and
  is not comparable across the two runs. At unchanged residuals the concentrate rows would fall 3.64² = 13.2×. They
  fall 6.6× (7,326 → 1,118), so their residuals grow about 1.4×: the solver uses the freedom it is given.
- **Mass moved:** 468,934 → 400,049 kt CC (−14.7 % vs `opt_20261019_155415_a1d7e9a9`); reported, not judged.
- **Common-set sum-row z²:** −0.48 % vs `opt_20261019_155415_a1d7e9a9`. China −2,979, Russia −800, Peru −600.
- **Exporters (CC, 1995–2024):** Chile 0.973 → 1.009, Peru 1.011 → 0.986, world 0.938 → 0.945.

## 1. Change

A soft ratio row's stated σ is in ratio units: `value_sd`, or band/4. The row read it as if it were already
relative to r, so every row with r < 1 was priced up to 1/r times too tight. #29 divides by |r| before
`RATIO_SIGMA_REL_FLOOR` (0.02), on soft rows only (ADR 0028 am.). Hard rows, bands, bounds, the net-stock rows and
the imputers are unchanged.

## 2. Pass rule (fixed on #94 at filing)

| Item | Rule | Result |
|---|---|---|
| 1 | Only soft ratio rows' σ_c change, by max(σ/\|r\|, floor)/max(σ, floor) to rtol 1e-9 | **Fails as worded; holds with the Lp lift inverted.** 217,717 rows change, every one a soft ratio row (type `ratio`, cone `free`, not hard); the rows are the same, in the same order, 1,685,939 both. On the stored σ_c the factor matches to 3.9e-5 (24 rows to 1e-9); with the lift inverted, all 217,717 match to 1e-9 (§3) |
| 2 | Round-E (ADR 0025) | **Fails on (c):** Peru's concentrate exporter retention 1.011 → 0.986 (−0.025). (a), (b) and (d) hold (§4) |
| 3 | No stock leg newly above \|z\| 5 | **Holds.** 319 → 312, none new; net CC stock posterior −1,335 → −951 kt CC |

## 3. The σ_c factors

| family | factor σ_c(new)/σ_c(base) | rows | deviation from max(σ/\|r\|, 0.02)/max(σ, 0.02), stored σ_c |
|---|---|---|---|
| concentrate (conc v10) | 3.64 | 67,680 | −1.4e-5 … −1.1e-7 |
| matte | 1.90 | 67,680 | −2.4e-6 … −4.7e-9 |
| anode | 1.06 | 67,680 | −3.0e-6 … −6.2e-9 |
| scrap | 1.43 | 2,820 | −1.1e-6 … −1.7e-8 |
| slag | 20.45 | 4,230 | −3.9e-5 … −5.7e-6 |
| eff flotation / SX-EW | 1.25 / 1.38 | 1,410 each | ≥ −3.1e-6 |
| ore grade (CO, ore use, ore supply; ore_grade v8) | 2.50–78.11 | 887 / 887 / 1,774 | −3.9e-5 … −1.4e-7 |
| tailings | 2.51–101.02 | 1,259 | −3.9e-5 … −1.7e-6 |

**The deviation is the engine's σ_c Lp lift** (`engine._lp_regularize`, step 2b). Every free row's σ_c is stored as
σ' = (σ^p + (cv_min·|c|)^p + abs_floor^p)^(1/p), with p 1.3, cv_min 0.01, abs_floor 0.001 kt, and c = 0 on ratio rows.
The lift adds the same small term to both runs' σ_c, so the factor falls slightly below the rule. The effect is
largest on the smallest σ_c: 0.02 × the 100 kt `RATIO_SCALE_ABS_FLOOR` = 2 kt, lifted by (0.001/2)^1.3/1.3 = 3.9e-5,
which is the observed maximum. With σ' inverted using each const run's sidecar knobs, all 217,717 rows match to
1e-9; the largest deviation is 1.0e-15.

Cathode, CER, CSXEW, PSC, WR, and eff smelting, CFR and ER stay on the floor and do not change.

## 4. Round-E (ADR 0025), `opt_20261019_155415_a1d7e9a9` → `opt_20261019_161937_fd301f58`

Retentions are Σ posterior / Σ prior over 1995–2024.

| Item | Base | New |
|---|---|---|
| (a) world retention: CO Mass / Concentrate CC / Cathode CC | 0.869 / 0.994 / 0.993 | 0.863 / 0.996 / 1.000 |
| (b) small nodes (\|c\| < 18 kt), posterior/prior median / p90 | 1.000 / 1.133 | 1.000 / 1.129 |
| (c) concentrate exporter retention (cross-region Concentrate→Smelting, CC): Chile / Peru / Indonesia / world | 0.973 / 1.011 / 0.917 / 0.938 | **1.009 / 0.986** / 0.917 / 0.945 |
| (d) sum-row z², \|c\| < 10 kt | 72,953 | 72,934 (−19) |
| (d) 10–100 kt | 342,068 | 342,443 (+375, +0.11 % vs the band's z² on `opt_20261019_155415_a1d7e9a9`) |
| (d) 100–1,000 kt | 597,936 | 595,029 (−2,907) |
| (d) > 1,000 kt | 142,801 | 139,801 (−3,000) |
| (d) mass moved, all CC edges | 468,934 kt CC | 400,049 kt CC (−14.69 % vs `opt_20261019_155415_a1d7e9a9`) |

- **(a) holds:** CO Mass −0.006, inside 0.01; Concentrate and Cathode CC rise.
- **(b) holds.**
- **(c) fails on Peru (−0.025).** Chile rises +0.036, the permitted direction. The established reading is "within
  ±0.01 or up" (Round L.1 REP). Peru's distance from 1 grows only from 0.011 to 0.014. Its common-set Δz² is −600.
- **(d) holds:** three bands fall; the 10–100 kt band rises +375. Mass moved is reported, not judged.

### 4.1 Peru's upstream chain

Concentrate→Smelting, 1995–2024, prior → posterior:

| | base | new |
|---|---|---|
| exports, Mass | 112,233 → 112,343 kt | 112,233 → 111,999 kt |
| exports, CC | 31,487 → 31,830 kt CC | 31,487 → 31,053 kt CC |
| exports, grade CC/Mass | 0.2806 → 0.2833 | 0.2806 → 0.2773 |
| domestic, CC | 10,205 → 9,360 kt CC | 10,205 → 9,321 kt CC |
| CO→Mining CC, posterior/prior | 1.057 | 1.028 |
| MF→Concentrate CC, posterior/prior | 1.053 | 1.032 |
| MF→Tailings CC, posterior/prior | 1.104 | 1.031 |

- **The export mass holds.** Peru's BACI export mass stays within 0.3 % of its prior in both runs.
- **The export CC falls 777 kt CC over 1995–2024 through the grade alone.** The largest falls are in 2020 (−79),
  2016 (−72) and 2018 (−67).
- **In the base, the export grade sits above its prior** (0.2833 vs 0.2806), beside a mine chain 5.7 % above its
  prior (CO→Mining CC). In the new run the mine chain is 2.8 % above it, and less CC leaves at the same mass. The
  base's 1.011 was surplus, not retention; the route from the loosened rows to Peru's mine is not traced further.
- **No deletion follows,** so the exporter rule's purpose, guarding trade-dependent mine chains from deletion (the
  Chile mine-chain diagnosis, Round L), is not engaged.

## 5. Ratio rows and the concentrate grade

| family | rows | Σz² base → new | \|z\| > 3 base → new |
|---|---|---|---|
| concentrate | 203,040 | 7,326 → 1,118 | 203 → 0 |
| matte | 203,040 | 994 → 296 | 11 → 0 |
| anode | 203,040 | 1,838 → 1,527 | 0 → 0 |
| cathode (on the floor) | 335,580 | 2,120 → 2,101 | 26 → 24 |
| slag | 12,690 | 4,066 → 105 | 73 → 0 |
| scrap | 8,460 | 929 → 429 | 35 → 13 |
| ore grade CO / ore use / ore supply | 2,661 / 2,661 / 5,322 | 8,657 / 8,657 / 14,556 → 268 / 268 / 499 | 110 / 110 / 157 → 0 / 0 / 1 |
| tailings | 3,777 | 4,992 → 479 | 159 → 0 |
| eff SX-EW / flotation / smelting / CFR | 4,230 each | 230 / 438 / 149 / 57 → 138 / 493 / 175 / 47 | 3 / 0 / 0 / 0 → 1 / 0 / 0 / 0 |
| all ratio rows | | **55,213 → 8,146** | |

These z² are priced at different σ in the two runs (§3's factors), so a fall measures the repricing as much as any
change in the residuals.

**Concentrate grade** (Σ CC / Σ Mass of MF→Concentrate, 1995–2024, prior → posterior):
- Chile: 0.2750 → 0.2756 (base) and 0.2750 → 0.2763 (new).
- Peru: 0.2750 → 0.2815 and 0.2750 → 0.2765.
- World: 0.2750 → 0.2741 and 0.2750 → 0.2731.

**The rows above 100 % stated relative σ** (#94's watch item; on ore_grade v8, #30). There are 36 params rows, all
in DR Congo (32) and Spain (4). They become 120 constraint rows: CO supply 12, ore use 12, ore supply 24, tailings 72.
- **Their z barely moves.** Σz² falls 20.66 → 0.28 and max |z| 3.34 → 0.30.
- **Neither region's mine chain moves much** (CC posterior/prior, base → new):
  - DR Congo: CO→Mining 1.006 → 1.000, MF→Concentrate 0.861 → 0.864; MF→Tailings rises 0.877 → 0.974.
  - Spain: CO→Mining 1.014 → 1.012, MF→Concentrate 1.072 → 1.074.
- **Reconciliation with #94's predictions comment:** it counted 4 rows above 100 % (DR Congo 3, Spain 1; max 156 %).
  That is the count per grade family (CO, ore use, ore supply). The 36 counts every ore_grade family, including MF
  tailings (19) and SX-EW tailings (5), as the #30 REP does.

## 6. Common set

- **Σz² 1,155,757 → 1,150,207 (−0.480 % vs `opt_20261019_155415_a1d7e9a9`)**, 27,458 rows, none in one run only.
- **By region:**
  - **Falls:** China −2,979, Russia −800, Peru −600, ROW −569, Australia −428, Indonesia −426. China's largest
    movers are its 2024 and 2020 Mining and Ore rows (−1,247 over the six largest).
  - **Rises:** United States +929, Canada +598, Mexico +378, Zambia +276. The United States' largest movers are its
    1999, 2000 and 2016 Mining and Ore Mass rows (+373 over the six largest).
- **By sector:** Concentrate −3,637, Cathode −2,029, Ore −953, Mining −948, Smelting −827; MF +2,759.
- **By year:** the largest falls are 2024 (−2,178), 2020 (−1,661) and 2023 (−805); the largest rises are 2016
  (+636), 2019 (+500) and 1996 (+404).

## 7. Predictions (posted on #94 before Stage 3)

| Prediction | Outcome |
|---|---|
| Item 1 holds | **Wrong as worded, right in substance:** right on which rows move and by what factor once the Lp lift is inverted (1e-15); wrong at rtol 1e-9 on the stored σ_c |
| (a) retention moves < 0.02 | **Right:** −0.006, +0.002, +0.007 |
| (b) and (c) move < 0.01 | **Wrong on (c):** Chile +0.036, Peru −0.025 |
| (d) common-set z² falls 2–6 % vs base | **Wrong on size:** −0.48 % vs `opt_20261019_155415_a1d7e9a9` |
| Item 3 holds | **Right** |
| Ratio-row Σz² falls by more than half | **Right** (55,213 → 8,146) |
| … concentrate rows the most, at about 1/3.6² of base | **Wrong:** concentrate fell 6.6× (7,326 → 1,118), not 13×. Ore grade fell 31× (31,870 → 1,035), slag 39× (4,066 → 105), tailings 10× (4,992 → 479) |

## 8. The manuscript

`method.md` lines 314 (the #29 sentence) and 347 (the limitation), and `si.md` line 202 (Table S4's ratio row),
describe the old rule. They change with the final chain.

## Reproduce

```
# on fix/ratio-sigma-relative 67806eb
python scripts/constraints.py --imp-run-id imp_20261019_153427_7b8966ff r94_ratio_sigma_relative
python scripts/run_solver.py --const-run-id const_20261019_160401_ea793c6f --label r94_ratio_sigma_relative
python scripts/explo/issue94_relative_ratio_sigma_diag.py opt_20261019_155415_a1d7e9a9 opt_20261019_161937_fd301f58
```

The diagnostic prints every number in this REP, or the numbers it is summed from. The exceptions are the solve
time, which comes from the run's sidecar, and the arithmetic stated in the text (13.2×, 6.6×, 1.4×, the 3.9e-5
lift).
