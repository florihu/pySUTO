# Round W.1 — the structural lower bounds reach the QP, and the solver is run to its floor

**Run shipped:** `opt_20260930_092111_e3e0677d` (label `roundW1_lower_bounds_tol`; 28 years Solved,
2015 and 2016 AlmostSolved at the numerical floor; `prior_mode` cold)
**Also measured:** `opt_20260929_134712_2d3dde19` (the same constraint run at the old tolerances)
**Against:** W.3 `opt_20260929_121937_e5f000b5` (predecessor), base `opt_20260916_124721_c4d1d944`
**Chain:** `imp_20260929_113529_ab914090` → `const_20260929_122634_40ebe5c9` → the run
**Feeds/params:** as W.2b/W.3 — icsg v12 / trade_shares v21, baci v11, crude_ore v3; acc copper v4;
eff v4, stripping_ratio v1, conc v8, ore_grade v7, stock_bounds v4, scrap_share v3
**Date:** 2026-09-30 · **Branch:** `fix/waste-rock-satellite` · **Issue:** #28 ·
**ADR:** [0039](../adr/0039-flow-lower-bounds-are-enforced-in-the-qp.md)

**Verdict: no flow is negative any more (smallest value −1.6e-11 kt on 586,560 bounded edges), and
the bounds cost almost nothing once the QP is solved properly. Measuring the round exposed the
larger defect: every production run had stopped at a relative duality gap of 1e-3, thousands of
objective units short of the optimum, and the bound rows made that early stop worse (a spurious
collapse, z² +44.5 %). With the termination tightened the same constraint run gives common-set z²
1,380,735 — −11.2 % vs W.3 and −19.6 % vs the base — while mine-output deletion falls from
24,884 to 3,325 kt CC (90.6 % of it forced) and Chile's concentrate export retention rises from
0.621 to 0.979. Most of that gain is the solver's, not the rows'; the base re-solved at the same
tolerance (`base_retol`, see §5) separates the two.**

---

## 1. Mechanism being removed

Bounds never reached the QP. Since the first solver (`8c0bf3f`, 2026-04-29) `A = [G_p | S]`
carried no bound rows and `FlowBoundsBuilder` emitted nothing; the stored `[l, u]` fed only the
first-year feasibility probe. ADR 0013 amendment 1 recorded this for the zero pin and left
non-negativity undocumented while the README, the solver docstring and the May REPs described
`l ≤ a ≤ u` as part of the problem. On the shipped Round-T′ run 1,108 mass edges of the mine chain
sat more than 1 kt below zero; W.2 (waste rock as a satellite flow) and W.3 (data-aware anchor)
removed the causes and left 149, all in region-years without an ore-grade row (Japan 28 edges /
−2,295 kt, Korea 36 / −1,734 kt).

## 2. Change

- `FlowBoundsBuilder` emits one hard non-negative-cone row `−a_i ≤ 0` per edge whose structural
  lower bound `acc.bounds.l` is 0: 586,560 rows, 19,552 per year — every edge except the signed
  stock changes Cathode→USC, Cathode→MESC and CP→PSC (267,900 edges, `l = −1e6`). Row type
  `bound`, variant `lower`. The pinned `ConstraintRun.l` is never read, so the zero pin stays
  probe-only and no-data stock changes stay signed. Reproducing toy
  (`tests/test_solver.py::TestLowerBoundsEnforced`): a balance `a = b + w` with tight `a = 100`,
  `b = 120` and a loose `w` solves to `w = −20.0` without the rows and to `w = 0`, `a = b = 108.4`
  with them.
- `G` grows from 1,365,461 to 1,952,021 rows (hard 1,491,472, soft 460,549); the per-year block
  from 45,416–45,584 to 64,968–65,136 rows.
- `scripts/run_solver.py`: the Clarabel tolerances move into the USER CONFIGURATION block and are
  recorded in the sidecar (`stage_config`); values in §3.

## 3. The tolerance finding

At the old settings (gap_rel 1e-3, feas 1e-7; reduced 1e-2 / 1e-5) the run with bound rows
(`opt_20260929_134712_2d3dde19`) looked catastrophic against W.3, although only 25 edges that were
negative in W.3 sit exactly at zero in it:

| old tolerances, W.3 → W.1 | |
|---|---|
| common-set sum-row z² | 1,554,826 → 2,247,349 (+44.5 %) |
| world retention CO Mass / Concentrate CC / Cathode CC | 0.798 / 0.938 / 0.969 → 0.688 / 0.858 / 0.916 |
| Chile concentrate export retention 2010–19 | 0.621 → 0.303 |
| CC mass moved | 533,295 kt (8.8 %) |
| iterations (median) / solve time | 27 / 197 s → 68 / 626 s |

Single-year re-solves without persistence (`w1_tol.py`, feasibility probe stubbed) explain it.
The objective Clarabel minimises is `Σz² − Σ(a₀/σ)²`; in 2015 the constant is 1,170,075 against a
misfit of 113,035 (the three parts reconcile to the logged −1,057,041), so a relative gap of 1e-3
admits ≈ 1,060 objective units, and the dual cost at a stalled iterate is no lower bound: W.3 2015
stopped at a reported gap of 805 with a dual cost of −1.058e6, above the optimum found later
(−1.066e6). Both traces stop the moment the gap test passes while the cost still falls by
1,000–7,000 per iteration; the W.1 trace also stalls at iterations 24–28 (step length 0.02–0.06)
before recovering.

| 2015 | objective | Chile conc CC exports (prior 2,522) | mine CC |
|---|---|---|---|
| W.3 production (gap_rel 1e-3) | −1.057052e6 | 1,658 | 22,591 |
| W.3 tight (1e-8; AlmostSolved, 146 it.) | −1.066052e6 | 2,452 | 23,929 |
| W.1 production (1e-3) | −1.015400e6 | 899 | 20,053 |
| W.1 medium (1e-6, floor accepted at 1e-4) | −1.064244e6 | 2,506 | 23,729 |

2010 (W.3 at 1e-8, Solved in 142 iterations, 28 s): objective 5,560 below production; Chile 1,791
prior / 1,207 production / 1,756 tight. The bound rows therefore cost almost nothing (W.1 within
1,808 objective units of W.3's tight optimum in 2015); they only make the premature stop worse.
The strict 1e-8 target does not converge on the bound-row problem (InsufficientProgress at 223
iterations without acceptance), so the production setting is now gap_rel 1e-6, feas 1e-8, gap_abs
1e-2, with the numerical floor accepted as AlmostSolved at gap_rel 1e-4, feas 1e-6, gap_abs 10.

## 4. Results at the new tolerance

| | W.3 (old tol.) | W.1 (new tol.) | Δ |
|---|---|---|---|
| status | 30/30 Solved | 28 Solved, 2 AlmostSolved (2015, 2016) | |
| iterations / solve time | 25–36 / 197 s | 148–254 / 1,687 s | 41–69 s per year |
| common-set sum-row z² (25,620 rows) | 1,554,826 | 1,380,735 | −11.2 % (−19.6 % vs base) |
| Mass edges > 1 kt below 0 | 149 | 0 (smallest −1.6e-11) | |
| world retention CO Mass / Concentrate CC / Cathode CC | 0.798 / 0.938 / 0.969 | 0.862 / 0.992 / 0.994 | |
| mine output deleted, of prior 403,804 kt CC | 25,172 | 3,325 (0.8 %); 3,013 forced = 90.6 % | |
| hard-capped region-years (P > 0.35 X + U) | 132 | 132 | |
| Chile concentrate export retention 2010–19 | 0.621 | 0.979 | |
| MF→Concentrate CC outside ore-grade region-years | 5,667 kt | 1,874 kt | −67 % |
| zero-pin exceedances (\|a*\| > 1 kt) | 3,481 edges, 17,087 kt | 3,057 edges, 15,624 kt | |
| world WR Mass (= 2.000 × ore) | 99,724,740 kt | 107,659,184 kt | |
| ore-grade soft rows z² | 64,079 | 39,382 | |
| CC mass moved | | 329,527 kt (5.44 %) | |

Where the bounds bite (edge below −1e-3 in W.3, at 0 in W.1): 909 edges, all gross-mass legs of
the mine chain in non-mining region-years — MF→Tailings 447, SXEW→Tailings 400, Ore→SXEW 60,
Ore→MF 2; by region Japan 34 (−2,259 kt removed), Korea 50 (−1,469), Belgium 49, Germany 34,
Italy 63, UAE 63. With the loops closed and the solve accurate, the phantom concentrate of those
region-years falls by two thirds (5,667 → 1,874 kt CC); the remainder is the smelter-intake pull
the W.3 REP describes.

Stop rule (Round E, W.3 → W.1): (a) world retention rises — CO Mass +0.064, Concentrate CC +0.054,
Cathode CC +0.025; (b) small nodes (|c| < 18 kt, n 4,754) median posterior/prior 1.001 → 1.000,
p10 0.966 → 0.955, p90 1.193 → 1.114; (c) concentrate export retention 2010–19 Chile 0.621 → 0.979,
Peru 1.008 → 1.006, world 0.843 → 0.932, **Indonesia 0.855 → 0.834 (−0.021, outside the ±0.01
band)**; (d) sum-row z² by anchor band: < 1 kt −0.0 %, 1–10 kt +16 %, 10–100 kt +2 %, 100–1,000 kt
−2 %, > 1,000 kt −50 % (total −10.7 %). Rule (d) was written for σ changes that should act on
small anchors; a solver-accuracy change acts everywhere, and its gain sits in the largest band.
The Indonesia miss and the +16 % in the 1–10 kt band are recorded, not explained; both are
candidates for the follow-up on phantom concentrate.

Pre-registered expectations: no edge below −1e-6 ✓; all 30 years accepted ✓; few binding bounds
outside the non-mining residual edges ✓ (all 909 are such edges).

## 5. What remains

- Every earlier run — the base, Rounds A–W.3, the manuscript's previous reference chain — carries
  the termination error, so a comparison across this change mixes the model's effect with the
  solver's. The base constraint set (`const_20260916_123437_8e8e9676`) was therefore re-solved at
  the new tolerance: `opt_20260930_093914_523d57d7` (label `base_retol`; 27 Solved, 3 AlmostSolved,
  627 s). The solver alone moves the base's common-set z² from 1,718,085 to 1,533,213 (−10.8 %);
  W.1 is a further −9.9 % below that (1,380,735), so of the −19.6 % vs the original base roughly
  half is the termination fix and half the modelling rounds W.2–W.1. On the base alone the
  tolerance lifts world retention CO Mass 0.779 → 0.861 and Concentrate CC 0.873 → 0.930, Chile's
  concentrate export retention 0.574 → 0.893, and moves Indonesia's 0.845 → 0.834 — the Indonesia
  miss of §4 is the solver's, not the bound rows'. The round REPs stand as comparisons between runs
  solved alike.
- The QP's weights span 1e-12..1e12 and two years end at the floor (AlmostSolved). Conditioning
  the problem (scaling the variables by σ) is the numerical follow-up; 1e-8 should then converge.
- Stage 5 (#27): the 586,560 bound rows are hard rows; the active-set fix must drop slack bounds
  as well as slack bands.
- Phantom concentrate in smelting countries without mines (Japan, Korea; W.3 REP), Germany's
  2018/2021 mine-chain scale totals (z ≈ −58.5, unchanged here) and Indonesia's export retention
  stay open.
- Reproduce: `python scripts/explo/roundW_waste_rock_diag.py opt_20260916_124721_c4d1d944
  opt_20260929_121937_e5f000b5 opt_20260930_092111_e3e0677d`;
  `python scripts/explo/roundK_sigma_floor_diag.py opt_20260929_121937_e5f000b5 opt_20260930_092111_e3e0677d`;
  `python scripts/explo/roundN_concentrate_grade_diag.py opt_20260930_092111_e3e0677d` (section 4).
  Both diagnostics need the per-year tables (`scripts/solver_stats.py <opt> --years all`) once the
  solver driver stops writing them by default.
