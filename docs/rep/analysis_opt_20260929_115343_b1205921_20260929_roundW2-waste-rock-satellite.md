# Round W.2 — waste rock is a satellite mass flow, fixed by a hard stripping relation

**Run shipped:** `opt_20260929_115343_b1205921` (label `roundW2b_waste_rock_hard`, 30/30 Solved, `prior_mode` cold)
**Also measured:** W.2a `opt_20260929_111551_21703f80` (soft stripping row, WR imputed as k·Ore)
**Against:** shipped Round T′ `opt_20260922_121256_d4394c0a`, base `opt_20260916_124721_c4d1d944`
**Chain (W.2b):** `imp_20260929_113529_ab914090` → `const_20260929_114027_b8490fcb` → the run
**Feeds/params:** feeds unchanged (icsg v12 / trade_shares v21, baci v11, crude_ore v3); **acc copper v4**
(new `masks.csv`, edge index byte-identical to v3); **eff v4** (stripping row removed); **stripping_ratio v1**
(new family, k = 2.0, range 1–3, Nassar et al. 2022); conc v8, ore_grade v7, stock_bounds v4, scrap_share v3
**Date:** 2026-09-29 · **Branch:** `fix/waste-rock-satellite` · **Issue:** #28 · **ADR:** [0038](../adr/0038-waste-rock-is-a-satellite-mass-flow.md)

**Verdict: waste rock now carries real, calculated mass and no copper. Taking Mining→WR out of the
Mining mass balance (a satellite flow) closes the slack that booked the crude-ore/ore gap on waste
rock; a hard relation WR = k·Ore then sets its mass exactly. World waste rock over 1995–2024 goes
from −257,667 kt to +99,797,432 kt (= 2.000 × ore), no waste-rock edge is negative, crude ore equals
ore in every region-year, and copper content is untouched (largest world change 0.02 %). A soft
stripping row (W.2a) does not work: the solver cannot pin down an edge that only weak soft terms
touch, and left waste rock at ≈ 1 × ore.**

---

## 1. Mechanism being removed

On the shipped run, Mining's gross-mass balance read `CO = Ore + WR`. Crude ore (the feed) and ore
(crude-ore CC ÷ the in-situ grade) come from the same allocated ore mass, so the imputed WR was ≈ 0
(clipped residual, ADR 0015). In the QP, the CO and Ore grade rows could part inside one hard band
and the balance booked the difference on WR, which had no bound in the QP: world WR −257,667 kt,
225 edges below −1 kt (Mexico 2019: CO 0.731 %, Ore 0.592 %, WR −37,510 kt). The stripping-ratio
row `WR = 2·Ore`, called "inert" by ADR 0015, was priced (z² 3,499, max |z| 3.57).

## 2. Change

- acc v4 `masks.csv`: mask `massbal_scope` = every edge except Mining→WR (2,820 edges, both
  valuations); `MassBalanceBuilder(mask="massbal_scope")`. Mining now balances `CO = Ore`.
- `AccountingStructure.save/load` persist file-loaded masks (`masks_user.npz`); before, `load()`
  silently dropped every `masks.csv` mask (a latent bug; test added).
- New params family `stripping_ratio` (own file, k = 2.0 ± range 1–3 as σ = 0.5), removed from eff.
- W.2a: soft stripping row; imputer fills WR = k·Ore. W.2b (shipped): the row is a **hard zero-cone
  equality** (`RatioConstraintBuilder(hard=True)`); the imputer no longer writes WR (flat prior),
  so waste rock carries no prior of its own.

## 3. W.2a — why the soft row fails

| | shipped | W.2a (soft) |
|---|---|---|
| world WR, 1995–2024 | −257,667 kt | 51,657,854 kt (WR/Ore **1.036**) |
| Mexico 2019 WR vs 2·Ore | −37,510 vs 395,746 | 187,903 vs 374,982 |

Only two terms touch the Mexico 2019 WR edge: its prior (262,426 ± 93,022 kt) and the stripping row
(2·Ore = 374,982 ± 131,213 kt). Both gradients at the reported solution point up (−1.72e-5 and
−2.17e-5), so the edge is not stationary; its weighted optimum is ≈ 300,000 kt. Re-solving 2019 alone
at a 1e-6 relative gap (105 iterations, no persistence) still gives WR/Ore 1.008 (world 1.033). The
QP's prior weights span ~1e-12 to 1e12 and the stopping rules are relative to the largest terms, so a
variable held only by weak soft terms is never pinned down. The fix is a hard relation, which the
solver enforces to feasibility tolerance (primal residuals ≤ 2e-9).

## 4. W.2b — results

| | shipped | W.2b | Δ |
|---|---|---|---|
| status | 30/30 Solved | 30/30 Solved | |
| common-set sum-row z² (25,620 rows) | 1,552,866 | 1,551,647 | −0.1 % (−9.7 % vs base) |
| world WR Mass, 1995–2024 | −257,667 kt | 99,797,432 kt | WR/Ore 2.000 |
| WR edges < −1 kt | 225 | 0 | |
| region-years with \|CO − Ore\| > 1 kt | 798 | 0 | |
| Mexico 2019 CO / Ore Mass | 160,363 / 197,873 | 193,790 / 193,790 | |
| ore-grade soft rows z² | 61,065 | 63,693 | +2,628 |
| Mass edges > 1 kt below l = 0 | 1,108 | 817 | WR gone |
| solve time | 188 s | 189 s | |

Stop rule (Round E, shipped → W.2b): (a) world retention CO Mass 0.796 → 0.799, Concentrate CC
0.938 → 0.939, Cathode CC 0.969 → 0.969; (b) small nodes (|c| < 18 kt, n 4,754) median posterior/prior
1.001 → 1.001; (c) concentrate export retention 2010–19 Chile 0.619 → 0.620, Peru 1.007 → 1.009,
Indonesia 0.857 → 0.855, world 0.842 → 0.843; (d) z² by anchor band moves < 0.4 % in every band.
Mass moved: 7,860 kt of 6,065,097 kt CC (0.13 %). Largest world CC pair change: Mining→Ore CC
+125.5 kt (0.02 %).

Pre-registered expectations: WR = k × ore and positive ✓; zero negative WR edges ✓; CO grade = Ore
grade ✓; former slack shows up in grade rows ✓ (+2,628 z²); CC unchanged within tolerance ✓.

## 5. What remains

- 817 Mass edges still sit > 1 kt below 0 — the free signed loops in non-mining regions (MF→Tailings
  375, SXEW→Tailings 245, Ore→SXEW 191, Ore→MF 6). Rounds W.3 (data-aware anchor exemption) and W.1
  (enforced lower bounds) address them.
- Waste rock's uncertainty from k (±25 %) does not enter the posterior of a hard relation; report it
  as σ_WR = σ_k·Ore (0.25 × WR/2) alongside the flow.
- k is one global value; region-specific k from Climate TRACE mine types is the refinement (D4).
- Reproduce: `python scripts/explo/roundW_waste_rock_diag.py opt_20260916_124721_c4d1d944
  opt_20260922_121256_d4394c0a opt_20260929_115343_b1205921` and
  `python scripts/explo/roundK_sigma_floor_diag.py opt_20260922_121256_d4394c0a opt_20260929_115343_b1205921`.
