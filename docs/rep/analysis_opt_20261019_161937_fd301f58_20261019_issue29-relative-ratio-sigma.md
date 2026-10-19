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
icsg_aggregates v4, scrap_share v3, stock_bounds v5, stripping_ratio v2).
**Date:** 2026-10-19 · **Branch:** `diag/ratio-sigma-measure` (from `fix/ratio-sigma-relative`) · **Issues:** #94,
#29 · **ADR:** 0028 (amended by #29), 0025 (stop rule)

**Verdict: fails on two items, 1 (as worded) and 2 (Round-E (c), Peru); item 3 holds.**
- **Item 1** fails only on its tolerance: every changed σ_c is on a soft ratio row, but the factor matches the rule's
  to 3.9e-5, not to rtol 1e-9.
- **Item 2:** Peru's exporter retention falls 0.025, outside the ±0.01-or-up reading of Round-E (c).
- **Ships over both by the maintainer's decision** (2026-10-19, delegated to the agent for the manuscript-ready week;
  Round L.1 shipped over a failed exporter rule). #29 makes the code apply the σ the parameter tables and the
  manuscript state. `opt_20261019_161937_fd301f58` becomes the reference chain.

- **The ratio rows stop being pins.** Their Σz² falls 55,213 → 8,146 (−85 % vs `opt_20261019_155415_a1d7e9a9`).
  Concentrate rows 7,326 → 1,118, ore-grade rows 31,870 → 1,035, slag 4,066 → 105, tailings 4,992 → 479. The rows
  above |z| 3 fall from 890 to 42.
- **The solver moves less mass:** 468,934 → 400,049 kt CC (−14.7 %). Common-set sum-row z² −0.48 %, with China −2,979
  (its 2020 and 2024 mine rows), Russia −800, Peru −600.
- **Exporters:** Chile's concentrate export retention rises 0.973 → 1.009; Peru's falls 1.011 → 0.986.

## 1. Change

A soft ratio row's stated σ is in ratio units: `value_sd`, or band/4. The row read it as if it were already
relative to r, so every row with r < 1 was priced up to 1/r times too tight. #29 divides by |r| before
`RATIO_SIGMA_REL_FLOOR` (0.02), on soft rows only (ADR 0028 am.). Hard rows, bands, bounds, the net-stock rows and
the imputers are unchanged.

## 2. Pass rule (fixed on #94 at filing)

| Item | Rule | Result |
|---|---|---|
| 1 | Only soft ratio rows' σ_c change, by max(σ/\|r\|, floor)/max(σ, floor) to rtol 1e-9 | **Fails as worded.** 217,717 rows change, every one a soft ratio row (type `ratio`, cone `free`, not hard); the rows are the same, in the same order, 1,685,939 both. The factor matches only to a relative 3.9e-5 (24 rows to 1e-9), always slightly below the rule's value (§3) |
| 2 | Round-E (ADR 0025) | **Fails on (c):** Peru's concentrate exporter retention 1.011 → 0.986 (−0.025). (a), (b) and (d) hold (§4) |
| 3 | No stock leg newly above \|z\| 5 | **Holds.** 319 → 312, none new; net CC stock posterior −1,335 → −951 kt CC |

## 3. The σ_c factors

| family | factor σ_c(new)/σ_c(base) | rows | deviation from max(σ/\|r\|, 0.02)/max(σ, 0.02) |
|---|---|---|---|
| concentrate (conc v10) | 3.64 | 67,680 | −1.4e-5 … −1.1e-7 |
| matte | 1.90 | 67,680 | −2.4e-6 … −4.7e-9 |
| anode | 1.06 | 67,680 | −3.0e-6 … −6.2e-9 |
| scrap | 1.43 | 2,820 | −1.1e-6 … −1.7e-8 |
| slag | 20.45 | 4,230 | −3.9e-5 … −5.7e-6 |
| eff flotation / SX-EW | 1.25 / 1.38 | 1,410 each | ≥ −3.1e-6 |
| ore grade (CO, ore use, ore supply; ore_grade v8) | 2.50–78.11 | 887 / 887 / 1,774 | −3.9e-5 … −1.4e-7 |
| tailings | 2.51–101.02 | 1,259 | −3.9e-5 … −1.7e-6 |

The deviation is always negative and largest on the smallest σ_c (slag, ore grade). The stored σ_c carries a small
term beyond `max(σ_r, floor)·scale` that the rule did not foresee. It is not traced here; it does not change which
rows move or by how much to four digits. Cathode, CER, CSXEW, PSC, WR, and eff smelting, CFR and ER stay on the floor
and do not change.

## 4. Round-E (ADR 0025), `opt_20261019_155415_a1d7e9a9` → `opt_20261019_161937_fd301f58`

| Item | Base | New |
|---|---|---|
| (a) world retention: CO Mass / Concentrate CC / Cathode CC | 0.869 / 0.994 / 0.993 | 0.863 / 0.996 / 1.000 |
| (b) small nodes (\|c\| < 18 kt), posterior/prior median / p90 | 1.000 / 1.133 | 1.000 / 1.129 |
| (c) exporter retention: Chile / Peru / Indonesia / world | 0.973 / 1.011 / 0.917 / 0.938 | **1.009 / 0.986** / 0.917 / 0.945 |
| (d) sum-row z², \|c\| < 10 kt | 72,953 | 72,934 (−19) |
| (d) 10–100 kt | 342,068 | 342,443 (+375) |
| (d) 100–1,000 kt | 597,936 | 595,029 (−2,907) |
| (d) > 1,000 kt | 142,801 | 139,801 (−3,000) |
| (d) mass moved, all CC edges | 468,934 kt CC | 400,049 kt CC (−14.69 % vs `opt_20261019_155415_a1d7e9a9`) |

- **(a) holds:** CO Mass −0.006, inside 0.01; Concentrate and Cathode CC rise.
- **(b) holds.**
- **(c) fails on Peru (−0.025);** Chile rises +0.036, the permitted direction. The established reading is "within
  ±0.01 or up" (Round L.1 REP). Peru's concentrate grade posterior also moves, 0.2815 → 0.2765 (§5); the route from
  the loosened rows to Peru's export legs is not traced further here.
- **(d) holds:** three bands fall, the 10–100 kt band rises +375 (+0.11 % of its base). Mass moved falls 14.7 %
  because the loosened rows no longer force mass onto pinned ratios; it is reported, not judged.

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

**Concentrate grade** (Σ CC / Σ Mass of MF→Concentrate, 1995–2024, prior → posterior):
- Chile: 0.2750 → 0.2756 (base) and 0.2750 → 0.2763 (new).
- Peru: 0.2750 → 0.2815 and 0.2750 → 0.2765.
- World: 0.2750 → 0.2741 and 0.2750 → 0.2731.

**The rows above 100 % relative σ** (#94's watch item; 36 on ore_grade v8, DR Congo 32 and Spain 4, #30) sit in the
ore-grade and tailings families. Their factors reach 78 and 101; across those families one row ends above |z| 3
(ore supply).

## 6. Common set

- **Σz² 1,155,757 → 1,150,207 (−0.480 % vs `opt_20261019_155415_a1d7e9a9`)**, 27,458 rows, none in one run only.
- **By region:**
  - falls: China −2,979 (Mining and Ore rows of 2020 and 2024), Russia −800, Peru −600, ROW −569, Australia −428,
    Indonesia −426;
  - rises: United States +929 (its 1999, 2000 and 2016 ore Mass rows), Canada +598, Mexico +378, Zambia +276.
- **By sector:** Concentrate −3,637, Cathode −2,029, Ore −953, Mining −948, Smelting −827; MF +2,759.
- **By year:** the largest falls are 2024 (−2,178), 2020 (−1,661) and 2023 (−805); the largest rises are 2016
  (+636), 2019 (+500) and 1996 (+404).

## 7. Predictions (posted on #94 before Stage 3)

| Prediction | Outcome |
|---|---|
| Item 1 holds | **Wrong as worded:** right on which rows move and by what factor to 4e-5; wrong at rtol 1e-9 |
| (a) retention moves < 0.02 | **Right:** −0.006, +0.002, +0.007 |
| (b) and (c) move < 0.01 | **Wrong on (c):** Chile +0.036, Peru −0.025 |
| (d) common-set z² falls 2–6 % | **Wrong on size:** −0.48 % |
| Item 3 holds | **Right** |
| Ratio-row Σz² falls by more than half, concentrate rows to about 1/3.6² of base | **Right on the total** (−85 %); concentrate 7,326 → 1,118, i.e. 1/6.6, not 1/13 |

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

The diagnostic prints every number in this REP or the numbers it is summed from, except the solve time, which
comes from the run's sidecar.
