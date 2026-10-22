# #99: a multiplier rule cannot replace Stage 5's slack cutoff. The 1e-4 kt fringe is mostly real slack, and its swing is one degenerate family

**Run:** `opt_20261021_131744_58504ad3` (the final chain of manuscript ready 2), re-solved in memory with Clarabel's
dual vector `z` read out
**Against:** Stage 5's slack rule, `binding ⇔ resid ≥ −1e-6 kt` (`DEFAULT_BIND_TOL`, `src/pysuto/uncprop.py:94`, used at l.554)
**Chain:** `const_20261021_130426_c5331a7f` → `opt_20261021_131744_58504ad3`
**Feeds/params:** as the run (icsg v17, baci v31, trade_shares v39, …)
**Date:** 2026-10-22 · **Branch:** `diag/anode-deletion-diagnosis` (REP only) · **Issue:** #99 · **ADR:** 0044
**Verdict: closed on measurement, no code.** The proposal was to read the binding set from the QP's multipliers.
At the production stop, `s·z` lies within ×0.51–1.42 of the year's median μ on 90 % of the ≈ 39,500 nonneg
rows. So any rule on one iterate's `z` is a slack cutoff at another threshold: `z > s` ⇔ `s < √μ` (3.6e-7 to 7.9e-5 kt
by year), `z > τ` ⇔ `s < μ/τ`. On lower bounds `z > s` does worse than the slack rule. #81 goes ahead on the slack
rule and reports the one degenerate family both ways.

## 1. Method

Stage 4 does not persist duals: `_solve_period` reads `sol.x` only, and the run's `t_*.npz` hold F_opt, eps_opt, c_star,
residuals, iter_history, cols_t, rows_t and a_prior. An 11-line patch (`OptimResult.duals` = Clarabel's `z` in
`rows_t` order) re-solved all 30 years in memory. **F_opt is bit-identical to the stored run in 30/30 years**, and
the suite passes (1,548 passed, 1 skipped). The patch was discarded with the decision. Five years (1998, 2002, 2011,
2016, 2019) were solved again at tolerance 1e-10, with μ 10³–10⁴× smaller. Each fringe row was then classified by
its trend between the two stops:
- **free:** s stays, z falls;
- **active:** s falls, z stays;
- **degenerate:** both fall.

## 2. Results

**Bound-active edge-years, 30 years:** slack rule 274,577; `z > s` 290,791 (+5.9 % vs the slack rule);
`z > 1e-9·max z` 263,873 (−3.9 %). `z > s` and the slack rule disagree on 16,220 lower bounds (33,670 nonneg rows).

**The fringe (lower bounds with slack in (1e-6, 1e-4] kt) is mostly free.** The slack rule already frees these
rows correctly:

| Year | fringe | free | degenerate | active |
|---|---:|---:|---:|---:|
| 2011 | 1,805 | 1,453 | 250 | 44 |
| 2016 | 1,714 | 1,235 | 368 | 109 |
| 1998 | 1,182 | 819 | 184 | 179 |
| 2002 | 1,030 | 1,026 | 0 | 4 |
| 2019 | 733 | 708 | 0 | 25 |

**`z > s` is worse on lower bounds.** It pins free bounds wrongly: 959 in 2011 (slack rule 2) and 815 in 2016
(slack rule 3). Of the small-multiplier active bounds the slack rule misses (79 and 143), it recovers only 40 and
107.

**The year-to-year swing is one degenerate family.** It is the cross-region `Cathode→FU` legs, Mass and CC in
pairs, about 250–370 a year, with the prior on the bound (0) and z* = s* = 0. At the stop they sit at s ≈ 1.4e-6 kt
(2011) and 2.1e-6 kt (2016) with z ≈ 1e-3, right at the cutoff. So they are pinned in 2002 and 2019 and free in
1998, 2011 and 2016. No single-iterate rule can sort them, because s and z fall together along the path.

## 3. Options not taken

| Option | Effect | Why not now |
|---|---|---|
| Tolerance 1e-10, slack rule kept | Pins the degenerate family consistently (slack 1e-8 to 3e-7 kt). Bound-active spread over the 5 years 1,057 → 564. +1 to +10 iterations a year, ΔJ ≤ 0.01. | A knob change: measure ticket, REP and a re-solve of every reference chain |
| Plus persisted duals and `z > s` | Differs from the slack rule at 1e-10 by 8–128 lower bounds a year | Adds little over the line above |
| Polish step (equality QP on the guessed active set) | Exact classification, degenerate rows included | The largest change, for a convention question |

The open question is a modelling convention: does a degenerate lower bound (prior on its bound, zero multiplier)
pin its edge? ADR 0044 defines binding by slack, so it does. Measured sensitivity is left to #81: its Table 3
intervals are computed with the degenerate `Cathode→FU` family pinned and free.

## Reproduce

The re-solve read `sol.z` through an uncommitted Stage-4 patch, discarded once the decision was taken, so these
numbers are not reproducible from `main`. Re-adding `duals = sol.z` to `_solve_period` and saving it beside
F_opt rebuilds them.
