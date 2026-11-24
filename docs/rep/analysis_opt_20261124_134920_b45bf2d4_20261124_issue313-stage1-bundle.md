# #313: the Stage 1 bundle (#233, #240 and the ore_grade rebuild) moves 842 BACI σ entries and common-set sum-row Σz² by +0.047 %; no Round-E item moves

**Run:** from a frozen detached worktree at main `ae58d40` (reference code, #233/#240, no #312), with the main checkout as working directory. One driver at a time, under the stage lock, every knob at its default, label `#313 stage1 bundle`:
- `baci_parse_feed.py` → `data/input/feed/baci/v0/`. Registered by hand as the **gross baci v34** (`current/reexport_*.csv` set aside, v0's three CSVs copied in, `InputRegistries.initialize`).
- `icsg_clean.py --baci-version 34 --to-feed` → icsg v19 (`values.csv` byte-identical to v18; only `parents.json` changed: baci 34, plus `eff` 4 and `icsg_aggregates` 4).
- The Stage 1 hook, `python -m pysuto run --case scripts/copper.py --data-dir data --from feed --to feed`: netted baci **v35**, trade_shares **v45**, `feed_baci_20261124_133656_107b0feb`, `feed_icsg_20261124_133707_4c523be0` (the hook run, on ore_grade v8), `feed_crude_ore_20261124_133710_d59a325d`.
- `build_ore_grade.py` → ore_grade **v9**; then `icsg_feed.py` again → `feed_icsg_20261124_133729_9d114cf6` (on ore_grade v9, the ICSG feed measured).
- `collect_aggregate_impute.py` on the three feeds in the reference imp's order (icsg, baci, crude_ore) → `imp_20261124_133941_619d23b2`; `constraints.py` → `const_20261124_134305_8726d7f3`; `run_solver.py --prior-mode cold` → `opt_20261124_134920_b45bf2d4` (30/30 solved, `solve_wall_s` 95).

**Against:** the reference chain `feed_icsg_20261119_125015_e2e9520e`, `feed_baci_20261022_110715_fdf3b4ac`, `feed_crude_ore_20260603_095651_558f7369` → `imp_20261119_125259_f2b11cd5` → `const_20261119_125651_ff7bfc9c` → `opt_20261119_130302_b20f0dba` (*base* below).
**Chain:** new feeds and Stages 2–4 only. Stage 5 and 6 were not rerun.
**Feed+params versions:** baci v35 (gross v34), icsg v19, trade_shares v45, ore_grade v9, crude_ore v3, icsg_aggregates v4, acc copper v5; imp and const params conc v10, eff v4, stripping_ratio v2, stock_bounds v6, scrap_share v3 as in the base. `input_status.py`: every `current/` at its latest version, before and after.
**Date:** 2026-11-24 · **Branch:** `diag/stage1-bundle-measure` · **Issues:** #313 (measures #233, #240 and #311 check 8) · **ADR:** 0033 (amendment 2026-11-24), 0025 (stop rule)

**Verdict: the bundle ships on the pre-registered rules. It moves the BACI feed σ on 842 edges and nothing else upstream of the solver.**
- **#233, #240:** `F_root` of the BACI feed is identical (0 of 854,460 differ). `σ_root` differs on 842 edges: 116 on the 145 root legs whose `cv_scale` #240 lowers by more than 0.2 %, all new/old ≤ 1 (median 0.904, min 0.200); 726 elsewhere, within 1.3e-3 relative (median +1.6e-4).
- **Ore grade:** v9 differs from v8 only by its `parents.json`: `params.csv` is byte-identical. The ICSG feed on v9 equals the base ICSG feed in every array, and equals the hook run's. The rebuild moves no number; it makes the lineage readable.
- **Solver:** common-set sum-row Σz² 959,575 → 960,023 (+0.047 % vs `opt_20261119_130302_b20f0dba`). Round-E holds to three decimals.
- **Table 3:** nothing here points to a flip, but Stages 5–6 were not rerun, so it is not measured. The maintainer decides whether the bundle needs them.

## 1. Decision items (pre-registered on #313 before the first driver)

| # | Item | Rule | Result | |
|---|---|---|---|---|
| R1 | Solve | every year Solved, no guard bypassed (required) | 30/30 solved, `solve_wall_s` 95 (base 95). No guard refused a step | ✓ |
| D1 | #233 | no BACI `F_root` moves; σ moves by about 5e-4 relative on the flagged legs #240 leaves alone | `F_root` 0 of 854,460 differ. On the 726 other edges with σ changed: new/old median 1.000161, range 0.998659–1.000305, \|rel\| max 1.34e-3. `cv_scale` rises on 754 of 931 legs, max +3.05e-4 | ✓ |
| D2 | #240 | only mixed legs lower σ, new/old ≤ 1 | 163 of 931 root legs have a lower `cv_scale` (148 touch ROW). On the 116 edges of the 145 legs below 0.998: new/old σ_root median 0.904, min 0.2001, max 0.9978, share ≤ 1 = 1.000. Mass-weighted `cv_scale` over the 163 legs 1.403 → 1.104 (`q_new` weights). 754 legs rise by ≤ 3e-4, 14 are equal | ✓ |
| D3 | ore_grade (check 8) | `parents.json` names the ICSG feed run; only observed-zero and ore-grade cells move; no BACI value moves | `parents.json` names `feed_icsg_20261124_133707_4c523be0` (the hook run, built before the rebuild) and `feed_crude_ore_20261124_133710_d59a325d`, with conc 10 and eff 4. ICSG feed base → new: 0 entries differ in `F_root`, `sigma_root`, `cv_root`, `cv_src`, `row_penalized`, `observed_zero`. `params.csv` v8 = v9 | ✓ |
| D4a | World retention | falls < 0.01 | CO Mass 0.999 → 0.999, Concentrate CC 1.021 → 1.021, Cathode CC 0.994 → 0.994 | holds |
| D4b | Small nodes (\|c\| < 18 kt) | median posterior/prior moves < 0.01 | median 1.000 → 1.000, p90 1.037 → 1.037 | holds |
| D4c | Exporter retention | falls < 0.01 or rises | Chile 1.033 → 1.033, Peru 0.982 → 0.982, Indonesia 0.922 → 0.922, world 0.952 → 0.953 | holds |
| D4d | Mass moved, Δz² by band | reported | mass moved 257,746 → 257,652 kt (−0.04 %). Δz² by anchor band: < 10 kt 0; 10–100 kt +33; 100–1,000 kt +192; ≥ 1,000 kt +224 | — |
| D5 | Objective | common-set sum-row Σz², reported | 959,575 → 960,023 over 29,356 rows (+0.047 % vs `opt_20261119_130302_b20f0dba`). ROW +389 (MF and Concentrate ROW, 2021–2022); Belgium +23, Australia +12 | — |
| D6 | Table 3 | any flip | not decidable at the opt level: Stage 5/6 not rerun. Screen: 30/30 solved, D4 unchanged | open |

All percentages are vs `opt_20261119_130302_b20f0dba`. Numbers come from `scripts/explo/issue313_stage1_bundle_diag.py opt_20261119_130302_b20f0dba opt_20261124_134920_b45bf2d4 --icsg-hook feed_icsg_20261124_133707_4c523be0 --baci-cv-old <old parse's quantity_cv_scale.csv>`.

## 2. Compare (`python -m pysuto compare --data-dir data`, base opt leaf vs new opt leaf)

| Stage | Arrays differing | Where |
|---|---|---|
| feed (crude_ore) | 0 arrays; `parents.inputs` only (acc v3 → v5) | by design |
| feed (baci) | `sigma_root`, `cv_root`: 842 of 854,460 each | the BACI σ |
| feed (icsg) | 0 arrays; `parents.inputs` only (feed v18 → v19, trade_shares v41 → v45, ore_grade v8 → v9) | by design |
| imp | `sigma_agg` 1,782 of 854,460; `Sigma` 842; `sigma_after` 976 | the σ of the 842 BACI edges and what aggregates them |
| const | `sigma_a` 1,782 of 854,460, then the matrices that carry σ | σ only |
| opt | every year's `F_opt`, `eps_opt`, `c_star`: solver response to σ | 240 arrays |

No `F_root`, prior or bound array moves before the solver. The 1,782 `sigma_agg` entries are the 842 BACI edges plus the cells that pool them.

## 3. Attribution

| Part | Evidence |
|---|---|
| #233 and #240 | all of the 842 `sigma_root` changes; the only upstream array that moves |
| ore_grade rebuild | nothing: `params.csv` v8 = v9, ICSG feed equal to base |
| trade_shares v45 | nothing: the ICSG feed arrays equal base, so the weights fan the same |

## 4. Predictions (pre-registered)

| | Prediction | Result | |
|---|---|---|---|
| P1 (D1) | no BACI value moves; σ moves by about 5e-4 on legs #240 leaves alone | `F_root` 0 differ. Median +1.6e-4, max 1.3e-3 | right on the sign and order; the median is 3× lower |
| P2 (D2) | 178 of 931 root legs mixed (158 ROW); new/old median 0.944, min 0.20; mass-weighted `cv_scale` 1.76 → 1.12; 753 pure-flagged legs unchanged | 163 legs lower (148 ROW); the 145 below 0.998 have a median new/old `cv_scale` of 0.896; min 0.2001; mass-weighted 1.40 → 1.10; 768 legs equal or up by ≤ 3e-4 | partly right: min and the ≤ 1 rule hold; counts 9 % low, the new level right, the old level and the median off |
| P3 (D3) | `parents.json` exists; fewer than 5 % of the 456 stamped edges (≤ 25) differ in the ICSG feed | present; 0 differ | right, and stronger |
| P4 (D4) | Round-E holds | holds to three decimals | right |
| P5 (D5) | Σz² rises +0.2 % to +3 % | +0.047 % | wrong: the rise is 4× below the range |
| P6 (D6) | no flip | no sign of one at opt level; not measured | open |

## 5. Notes for the maintainer

- **The mixed-leg count.** L1 read 178; the registered parse gives 163. L1 read the old `values.csv` where 49 flagged legs are missing (the caveat), and weighted differently. The #240 change is as designed; the REP quotes the measured 163.
- **A stale grade stays readable now.** The ore grade's `parents.json` names the *previous* ICSG run by design (the loop in README Stage 1), so it names `…133707`, not the final `…133729`.
- **A bit-identical rebuild registers new versions.** icsg v19 and baci v34/v35 and trade_shares v45 exist only because of lineage and the netting; trade_shares v45 names `feed_baci_20261124_133656_107b0feb`.
- **This chain is the base of #314** (the maintainer decides whether it becomes the reference chain; Stages 5–6 are not run on it). `feed_icsg_20261124_133707_4c523be0`, `imp_20261124_133941_619d23b2`, `const_20261124_134305_8726d7f3` and `opt_20261124_134920_b45bf2d4` stay: trade_shares v45 names the BACI feed run, and no REP cites the first ICSG run except this one, which the maintainer may prune.
