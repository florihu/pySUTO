# #149: the parametric σ-units fix (#148) moves the prior's σ on 6,687 edges and the result by solver noise

**Run:** `python -m pysuto run --case copper --from imp --to opt --base opt_20261022_112855_928da10c`, then
`--from uncprop --to psut --base opt_20261112_135523_88fa69f3` → `imp_20261112_133434_22fc54b3` →
`const_20261112_134032_b6802ef3` → `opt_20261112_135523_88fa69f3` (30/30 Solved, cold, `solve_wall_s` 608) →
`uncprop_20261112_144236_f79da518`, `psut_20261112_144358_449d34b5` (label `#148 parametric σ units`, every knob at its
default)
**Against:** the final reference chain `const_20261022_111531_df61bf5c` → `opt_20261022_112855_928da10c` →
`uncprop_20261029_090754_0446e784`, `psut_20261029_095807_61a70a1a`; for the prior the post-#113 imp
`imp_20261102_094925_95744204`. Called *base* below.
**Chain:** the base's feeds `feed_icsg_20261022_110826_8a031806`, `feed_baci_20261022_110715_fdf3b4ac`,
`feed_crude_ore_20260603_095651_558f7369`, resolved by `--base`. Code main 8fe5adf (contains #148 293fa0f and the
bit-identical #11, #144, #143), run from the worktree `diag/parametric-sigma-units-measure` with `--data-dir` on the main
checkout's `data/` (the main checkout was on another session's branch); one driver at a time under the shared stage lock.
**Feeds/params:** acc copper v5 (`54bc4323`); conc v10, eff v4, ore_grade v8, stripping_ratio v2, stock_bounds v6,
scrap_share v3 — the base's; `input_status.py`: every `current/` at its latest version
**Date:** 2026-11-12 · **Branch:** `diag/parametric-sigma-units-measure` · **Issues:** #149 (measures #148; parent #140)
· **ADR:** 0025 (stop rule), 0044 (Stage 5 linearisation)
**Diagnostic:** `python scripts/explo/issue149_parametric_sigma_units_diag.py opt_20261022_112855_928da10c opt_20261112_135523_88fa69f3` (<opt_base> <opt_new>)

**Verdict: ships. Every pre-registered ship item holds (1, 2, 3a–c); `opt_20261112_135523_88fa69f3` becomes the reference
chain.** The fix corrects a dimension error — a ratio's σ in ratio units was added to flow σ in kt — and changes the
prior's σ on 6,687 of 854,460 edges. The result does not move at any precision the manuscript quotes: Round-E items are
equal to three decimals, common-set Σz² moves −0.001 %, Σ PSUT is unchanged to 1 part in 10⁷. Prediction P1 was wrong in
direction: the fix *tightens* the small-leg tail rather than loosening large legs.

## 1. Change under test

- **#148 (`5e0f6a2`, review fix `3556c91`)**: `ParametricImputer` now Lp-combines the parameter term `|f_a|·σ_r` (forward,
  `b = r·a`) and `|f_a|·σ_r/|r|` (backward, `a = b/r`) — the first-order image of the ratio's σ in the filled edge's flow
  units, the form `TradeImputer` and the network's Mass twin already used. Before, it added the bare `σ_r` (e.g. 0.0375
  for a grade of 0.28) as if it were kt.

## 2. Decision items (pre-registered on #149 before the first driver)

| # | Rule | Result | |
|---|---|---|---|
| 1 | `f_agg`, `coverage` bit-identical to the base imp; method counts change only by `_capped` | bit-identical; no method count changes; σ moves on 6,687 edges (6,585 parametric fills, 102 downstream) | holds |
| 2 | every year Solved / AlmostSolved | 30/30 Solved | holds |
| 3a | world retention falls < 0.01 in every category | CO Mass 0.999, Concentrate CC 1.010, Cathode CC 0.998 — equal to the base | holds |
| 3b | small nodes (\|c\| < 18 kt) median post/prior within 0.01 | 1.000 (p90 1.015), equal | holds |
| 3c | Chile / Peru / Indonesia exporter retention falls < 0.01 or rises | 1.022 / 0.978 / 0.917, equal (world 0.947) | holds |
| 3d | mass moved and z² by anchor band, reported | mass moved 256,225 → 256,214 kt CC (−0.004 %); band Δz² −0, +5, −10, −1 | — |
| 4 | common-set Σz² by family, reported | 744,674 → 744,669 (−0.001 % vs `opt_20261022_112855_928da10c`); largest regional moves ±3 | — |

**Predictions:** P1 (Mass conc fills' median CV 0.07 → ≈ 0.15) **wrong**: 0.0705 → 0.0685 median, p90 0.139 → 0.0685;
P2 (< 2 % capped) right: 0 capped; P3 (Σz² falls, < 5 %) right: −0.001 %; P4 (exporter retention < 0.01) right: 0.000;
P5 (all Solved) right.

## 3. What moved

- **Prior σ (Stage 2).** The bare-σ_r term dominated only where the flow is small (a 0.0375 "kt" term on a 0.1 kt leg
  is a CV of 0.375). In flow units the parameter term is scale-free (CV = σ_r/|r|), so the Mass conc fills' CV collapses
  onto 0.0685 at every size: median 0.0705 → 0.0685, p90 0.139 → 0.0685 (5,664 fills). The 898 CC conc fills barely
  move (median 0.406 → 0.404). The 23 ore_grade fills widen (median 0.104 → 0.136, p90 0.114 → 0.290): their backward
  fill divides by a grade of order 0.01.
- **Stage 3.** All 280 const arrays equal the base except `sigma_a` (the 6,687 edges, median ratio new/base 0.986) and
  `row_meta.parquet`, which differs only by #11's schema (`sense` for `cone_type`; equal after the legacy upgrade).
  `G`, `c` and `σ_c` are bit-identical.
- **Stage 4.** 30/30 Solved; every Round-E item equal to three decimals; Σz² −0.001 %.
- **Stage 6.** Σ PSUT 389,469,686 → 389,469,672 kt; 178 cells move by > 5 kt. The largest is Chile's Ore Mass split
  between MF and SX-EW (2017: −166 kt on a 507,683 kt leg, 0.03 %), mirrored on the tailings legs — a nearly flat
  direction the solver resolves at its tolerance, not a response to the changed σ.
- **Stage 5.** Σσ* +0.03 % (9,208,848 → 9,211,592 kt); Σ lower −0.0003 %, Σ upper +0.0013 %. Of the 43,523 edge-years
  with σ* > 1 kt, the 0.1–99.9 % quantiles of σ*_new/σ*_base are 0.27–66.6, and 146 move by more than 2×. **None of
  those 146 lies on an edge whose prior σ moved**: they are active-set flips (ADR 0044) — near-zero legs (e.g. Greece
  Ore→SX-EW 2002) that sit at their bound in one posterior and not in the other (σ* 0 ↔ 26.49 kt). 1,286 of 275,043
  active edge-years flip (0.47 %). The intervals of Table 3 should be re-read on the new chain; nothing here suggests
  they move at the precision quoted.

## 4. Consequences

- **Reference chain:** `opt_20261112_135523_88fa69f3` (`uncprop_20261112_144236_f79da518`,
  `psut_20261112_144358_449d34b5`), recorded in `docs/history/TODO.md`. The manuscript still reads `opt_20261022_112855_928da10c`;
  a follow-up re-points `manuscript_numbers.py`, `results_figs.py` and `table3_intervals.py` and checks that no keyed
  number changes at its quoted precision.
- **#140** (one σ rule per Relation): the measured sensitivity of copper to the conc fills' σ is nil at the level of
  every Round-E item; the remaining unification is a code-quality change, not a modelling one.
- **Lesson:** the fix tightened rather than loosened — the defect inflated σ on small legs, the opposite of the
  pre-registered expectation drawn from large legs.
