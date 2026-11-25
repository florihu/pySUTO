# #316: around the reference run, A6 fails on scrap into refineries alone, now at 15 of 64 points (max 0.050); termination still moves the answer nearly four times more

**Run:** `sweep_20261124_153029_a4bb2afb` (label `knob sweep (#108)`).
- Design: the design of `sweep_20261102_145601_8413cace` (#108): 64 points of a Latin hypercube, seed 20261022, over the same 15 knobs and ranges (SI Table S22b).
- Steps: imp → const → opt. Only the opt runs are kept (ADR 0045 am.).
- Fixed overrides: `opt.FEASIBILITY_CHECK = "none"`, `const.DIAG_PLOT = False`.
- Outcome: 64/64 points done, 0 failed, 2026-11-24 15:30 → 2026-11-25 00:14 UTC (31,455 s = 8.7 h, mean 8.2 min per point).

**Against:** the reference run `opt_20261124_141714_e2892160` of milestone *manuscript final*, the base of every point:
- const run `const_20261124_141044_e73c5952`;
- imp run `imp_20261124_140720_ad766e00`.

**Chain:** the reference's feeds `feed_icsg_20261124_133729_9d114cf6`, `feed_baci_20261124_133656_107b0feb` and `feed_crude_ore_20261124_133710_d59a325d`.
- The sweep ran from the main checkout's `data/` with the code of `73115e1` (the #314 merge; the sidecar's `environment.git_dirty` is false), with no other stage driver running.
- It was scored by `scripts/sweep_diag.py` at `73115e1` on 2026-11-25.

**Feeds/params:** the reference's (acc copper v5; icsg v19, baci v35, trade_shares v45, crude_ore v3; conc v10, eff v4, ore_grade v9, stripping_ratio v2, stock_bounds v6, scrap_share v3).

**Date:** 2026-11-25 · **Branch:** `diag/manuscript-refresh-320` · **Issue:** #316 (design #108, library #117) · **ADR:** none (no decision changes; 0045 for the sweep's storage)

**Verdict: A6 fails on one stage at 15 of 64 points, as it failed around the earlier chain; Round-E (c) fails on Indonesia; no Table 3 outcome flips and no knob changes.**

- **A6.** Scrap into refineries (Scrap→CFR CC) reaches |Δ pooled retention| 0.0496 (point 15) against the 0.02 limit. Around `opt_20261022_112855_928da10c` it reached 0.0298 at 5 of 64 points. The other eight stages stay within 0.0101 at every point (was 0.0048). The termination pair of A5 moves one stage by 0.184 on the same const run (#315), 3.7 times the sweep's largest change.
- **The breaches.** Fourteen of the 15 lie above the reference, at a no-data anchor coefficient of variation of 0.059–0.096, 1.2–1.9 times the shipped 0.05. That range holds 24 points; 10 of them meet A6. The fifteenth lies below the reference (−0.021), at 0.026, near the range's lower end.
- **The ranking.** Three σ knobs rank first on all twelve criterion outcomes:
  - the scale totals' coefficient of variation (`ANCHOR_CV`, tied) on the five primary stages from concentrate to cathode and on Chile's and Indonesia's exports;
  - `NODATA_ANCHOR_CV` on both scrap legs;
  - `LEG_SIGMA_GAMMA` on crude ore into mining, final use and Peru's exports.

  Nine of the 15 knobs are influential on none of the twelve (was ten).
- **Round-E (c).** It holds for Peru (64/64), misses for Chile at four points (60/64, was 62/64) and fails for Indonesia (16/64, was 18/64). Indonesia's concentrate export retention moves ±0.032 with the scale-total cv.

The account is robust to the knobs, except for the scrap legs and Indonesia's exports, as around the earlier chain. The scrap legs now move about twice as much.

## 1. Design

The design is §1 of `analysis_sweep_20261102_145601_8413cace_20261105_knob-study.md`, unchanged, with one exception:
- `CASCADE_ORDER` takes the shipped order `(trade, network, conc, ore_grade)` or its reverse. The eff imputer left the cascade before this chain, so the axis has four imputers, not five.

All 64 points meet the reference's hard rows to 2.6e-6 kt, against the reference's own 1.1e-6 kt (A1).

## 2. The twelve criterion outcomes

Δ is point minus the reference, pooled over 1995–2024 for the nine stages and over 2010–2019 for the exporters. "Old" is `sweep_20261102_145601_8413cace` around `opt_20261022_112855_928da10c`.

| outcome | limit | max \|Δ\| (old) | points within (old) | top knob (old) | SRRC, R² |
| --- | --- | --- | --- | --- | --- |
| CO→Mining Mass | 0.02 | 0.0003 (0.0010) | 64 (64) | leg-σ exponent (cascade order) | +0.73, 0.80 |
| MF→Concentrate CC | 0.02 | 0.0035 (0.0026) | 64 (64) | scale-total cv (leg-σ exponent) | +0.86, 0.98 |
| Concentrate→Smelting CC | 0.02 | 0.0033 (0.0025) | 64 (64) | scale-total cv (leg-σ exponent) | +0.86, 0.98 |
| Anode→ER CC | 0.02 | 0.0021 (0.0031) | 64 (64) | scale-total cv (same) | −0.84, 0.96 |
| CP→Cathode CC | 0.02 | 0.0023 (0.0034) | 64 (64) | scale-total cv (same) | −0.90, 0.94 |
| Cathode→FU CC | 0.02 | 0.0033 (0.0028) | 64 (64) | leg-σ exponent (scale-total cv) | −0.75, 0.90 |
| Scrap→Smelting CC | 0.02 | 0.0101 (0.0048) | 64 (64) | no-data anchor cv (same) | +0.92, 0.99 |
| Smelting→Matte CC | 0.02 | 0.0016 (0.0028) | 64 (64) | scale-total cv (same) | −0.66, 0.90 |
| Scrap→CFR CC | 0.02 | **0.0496** (0.0298) | **49** (59) | no-data anchor cv (same) | +0.84, 0.99 |
| Chile exports 2010–19 | 0.01 | 0.0119 (0.0103) | 60 (62) | scale-total cv (same) | −0.86, 0.96 |
| Peru exports 2010–19 | 0.01 | 0.0018 (0.0023) | 64 (64) | leg-σ exponent (scale-total cv) | +0.89, 0.90 |
| Indonesia exports 2010–19 | 0.01 | 0.0325 (0.0329) | 16 (18) | scale-total cv (same) | +0.99, 0.98 |

The six influential knobs (|SRRC| ≥ 0.2 with R² ≥ 0.5 on at least one outcome) are the scale-total cv, the no-data anchor cv, the leg-σ exponent, the soft-ratio σ floor, the Lp exponent p and the stock-seed cv. The cascade order, first on crude-ore mass around the earlier chain, is influential on none.

## 3. The breaches of A6

| point | no-data anchor cv | Δ Scrap→CFR CC | Δ Scrap→Smelting CC |
| --- | --- | --- | --- |
| p019 | 0.026 | −0.021 | −0.004 |
| p054 | 0.059 | +0.022 | +0.003 |
| p035 | 0.064 | +0.026 | +0.003 |
| p050 | 0.072 | +0.027 | +0.006 |
| p002 | 0.077 | +0.028 | +0.004 |
| p044 | 0.077 | +0.023 | +0.005 |
| p062 | 0.080 | +0.032 | +0.005 |
| p039 | 0.081 | +0.024 | +0.005 |
| p045 | 0.083 | +0.025 | +0.005 |
| p060 | 0.087 | +0.033 | +0.007 |
| p047 | 0.089 | +0.050 | +0.010 |
| p015 | 0.091 | +0.050 | +0.009 |
| p052 | 0.093 | +0.027 | +0.006 |
| p004 | 0.095 | +0.022 | +0.006 |
| p033 | 0.096 | +0.026 | +0.006 |

- At each breach the stage lies beyond 0.02 in 13 to 26 of the 30 years, so a breach shifts the series, not one year.
- Year by year, the only other stage that leaves 0.02 is scrap into smelters, in 1995 and 2016–2018 (at most 0.031).
- p015 and p047 reach twice the other breaches. Their other knob values overlap those of p060 and p062 (ore-grade-first cascade, scale-total cv 0.006–0.007), which reach 0.033 and 0.032. No single knob separates them; this REP does not test interactions.
- p027 has the largest no-data anchor cv of the sweep (0.099) and stays at +0.007. The rank regression explains R² 0.99 of the ranks, not of the values.

## 4. Why the scrap legs move more than around the earlier chain

Not measured. The two sweeps share their design and differ in their base chain: #154, #201, #233, #240 and #312 changed the prior since `opt_20261022_112855_928da10c`. The pooled retention of Scrap→CFR CC at the base moved from 1.026 to 1.037. Which change raised the leverage of `NODATA_ANCHOR_CV` stays open; it changes no outcome of Table 3 and no knob.

## 5. Other outcomes

- **Slag legs** (outside the criteria): ER→Slag CC −0.110 to +0.040 and Smelting→Slag CC −0.068 to +0.073, driven by the scale-total cv (SRRC +0.97 each); CFR→Slag CC −0.057 to +0.029, driven by the soft-ratio scale floor (−0.65). Around the earlier chain the widest was CFR→Slag CC, −0.052 to +0.039.
- **Contraction–sensitivity link** (exploratory, Results §4.2): over the 23 contained-copper categories with a retention, Spearman ρ between the largest |Δ| and the mass-weighted 1 − κ is 0.71 (p < 0.001; was 0.48, p 0.02); gross mass 0.22 (p 0.30; was 0.20, p 0.35). Scrap→CFR CC (κ 0.92) stays the exception.

## 6. What reads this sweep

`results_figs.py --sweep-id sweep_20261124_153029_a4bb2afb` (no `--sweep-base`: the sweep's base is the reference run) and `manuscript_numbers.py` group 12 read it. On branch `diag/manuscript-refresh-320` (#320) the following now read it:
- Table 3 row A6 and its caption, Figure 5d–e and its caption, and the §4.2 bullets;
- Discussion lines on A5 against A6, the two σ knobs, robustness to judgement values, the local sweep and take-home 2 ("nearly four times");
- SI S15.2, S15.4 (Tables S22b, S22c), S18b and Figure S3 (`figs/si_s15_knob_criteria.png`, a copy of `diag/sw_r_05_criterion_bands.png`).

## 7. Reproduce

```bash
python scripts/sweep_diag.py sweep_20261124_153029_a4bb2afb
python scripts/report/results_figs.py opt_20261124_141714_e2892160 \
    --psut-run-id psut_20261124_150658_8f727db2 \
    --tol-pair opt_20261124_152425_856f440b opt_20261124_152923_0bbd377c \
    --uncprop-run-id uncprop_20261124_144415_13fea7bf \
    --sweep-id sweep_20261124_153029_a4bb2afb
python scripts/report/manuscript_numbers.py opt_20261124_141714_e2892160
```
