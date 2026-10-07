# Waste rock gets its prior: WR Mass = k·Ore at the flat σ

**Run:** `opt_20261007_130833_1a9b7af7` (label `wr_stripping_prior`; 29 Solved + 2022 AlmostSolved;
`prior_mode` cold)
**Against:** Round Y.0 reference `opt_20261006_105904_61c5efe9` (30/30 Solved)
**Chain:** `imp_20261007_122232_2b67458e` → `const_20261007_122703_50f6ff71` → the run
**Feeds/params:** feeds as the reference (`feed_icsg_20260922_115241_eb9cb9ff` / trade_shares v21,
`feed_baci_20260922_090456_cdcb4c41`, `feed_crude_ore_20260603_095651_558f7369`); acc copper v4;
conc **v10** (differs from the reference's v9 only in the comment cell of the PSC row), eff v4,
ore_grade v7, stock_bounds v5, scrap_share v3, stripping_ratio v1 — **now also a Stage-2 input**
**Date:** 2026-10-07 · **Branch:** `fix/wr-prior-stripping` · **ADR:**
[0038, amended 2026-10-07](../adr/0038-waste-rock-is-a-satellite-mass-flow.md)

**Verdict: the prior now carries waste rock and the solve is unchanged within what the solver
resolves.** Every WR Mass prior edge equals 2·Ore exactly (1,410 of 1,410; 0 → 124,325 Mt over
1995–2024). The fix shifts the QP objective by ≤ 3.5 per year, almost uniformly, so it cannot move
the optimum. The run's posterior differences — WR −68 / −67 / −70 Mt in 2008 / 2009 / 2022,
common-set sum-row z² **+0.1 % vs `opt_20261006_105904_61c5efe9`** — are solver inexactness: in those
three years the reference posterior is feasible for the new QP and scores 684 / 627 / 956 lower than
the new run's own answer, against a reported duality gap of 0.3–5. Round-E items hold.

---

## 1. Defect

On the reference chain all 1,410 Mining→WR Mass edges had `a_prior = 0`, σ_a = 10⁶ (imputation
log: `flat_prior_noisy_zero` on every one), against a posterior of 2,285–5,004 Mt/yr set by the hard
stripping row `WR = 2·Ore`. No feed covers WR, WR is copperless so the NetworkImputer's conc twin
skips it, and Round W.2b (ADR 0038) had removed the W.2a `k·Ore` fill together with the soft
stripping row. Every consumer of the prior saw no waste rock: `a_prior` in the edge tables, WR
retention (Σa*/Σa_prior undefined), and the hard stripping rows, which carried the whole waste-rock
mass as a defect at the prior.

## 2. Change

- `NetworkImputer(stripping_ratio_csv=…, stripping_sigma=…)` resolves the `stripping_ratio` rows
  with the shared `ParamRelationResolver` (A = Mining→Ore Mass, B = Mining→WR Mass, most specific row
  wins) and, after the recipe fixpoint, writes `WR_Mass = k·Ore_Mass` wherever Ore Mass is known
  (method `network_stripping`). WR is a leaf, so one pass per call suffices; the cascade re-runs it.
- σ = `FLAT_SIGMA_HIGH` (10⁶ kt), passed **past the imputer's cv cap** (`cv_cap = 1.5` would have
  turned it into 1.5·WR, an informative prior). An informative WR prior would count the Ore prior
  twice through the hard row; waste rock has no data of its own.
- `collect_aggregate_impute.py` passes the `stripping_ratio` params and records them in the imp
  run's `parents.inputs`.

## 3. The prior

| | reference | this run |
|---|---|---|
| WR Mass prior, 1995–2024 | 0 Mt | 124,325.4 Mt (= 2 × 62,162.7 Mt Ore) |
| WR prior edges equal to 2·Ore prior | — | 1,410 / 1,410 |
| network fills (both passes) | 41,216 | 42,103 (+887 `network_stripping`) |
| time-interpolation fills | 207 | 209 (+2: Botswana WR 2019–20) |
| flat-prior fills | 713,691 | 712,802 (−889) |

WR retention (Mining→WR Mass, Σa*/Σa_prior) is now defined: 0.866 over 1995–2024, from 0.816 in
2024 to 0.970 in 1997 — the Mining→Ore Mass retention, since prior and posterior are both 2·Ore.
On the reference it is undefined (prior 0).

Botswana 2019–20 has Ore Mass only from time interpolation (after the cascade); its WR now
interpolates from the neighbouring `k·Ore` years at σ ≈ 10⁶, so these two edges also leave the
no-data anchor, which pulled Botswana's ore toward zero through the hard row (WR 0 ± 1,176 kt
against Ore 849 ± 416 kt in 2020; the reference posterior gave way by only 0.3 kt).

## 4. The solve

G, `row_meta`, c and σ_c are identical between the two const runs; a₀ differs on the 889 filled
edges, σ_a on those plus 1,042 anchored zeros (§5). With `J(F) = Σ((F − a₀)/σ_a)² + Σ_soft
((GF − c)/σ_c)²` (reproduces Clarabel's `cost_primal` to the reported digits after the constant
Σa₀²/σ_a²), the fix's own change to the objective is `δ(F) = J_new(F) − J_base(F)`:

| | max over years |
|---|---|
| \|δ\| from the WR priors, at either posterior | 3.45 |
| \|δ\| from the anchor σ change, at either posterior | 0.37 |
| \|δ(F_base) − δ(F_new)\| | 0.14 |

A near-constant shift of ≤ 3.5 against J of 44,287–231,317 per year leaves the optimum where it
was. Evaluating each posterior under both QPs shows where the observed differences come from:

| year | status base / new | gap_abs base / new | J_new(F_new) | J_new(F_base) − J_new(F_new) |
|---|---|---|---|---|
| 2008 | Solved / Solved | 0.66 / 0.29 | 58,069.8 | **−683.6** |
| 2009 | Solved / Solved | 0.60 / 0.92 | 44,913.4 | **−627.4** |
| 2022 | Solved / AlmostSolved | 0.82 / 5.18 | 161,755.8 | **−956.1** |
| 2019 | Solved / Solved | 0.65 / 0.63 | 165,846.7 | +206.1 |
| 2017 | Solved / Solved | 0.64 / 0.57 | 210,817.4 | +114.0 |
| 2013 | Solved / Solved | 0.98 / 0.95 | 88,558.9 | +101.2 |

A negative entry means the reference posterior is a better point of the **new** QP than the new
run's answer; a positive one, that the new posterior beats the reference on the reference's QP
(the column mirrors J_base(F_new) − J_base(F_base) to within δ). Both runs stop hundreds of units
from the optimum in some years while reporting a gap below 1. The three negative years carry the
WR posterior drops (2008 3,299.6 → 3,231.8 Mt, 2009 3,346.9 → 3,279.9 Mt, 2022 5,004.2 →
4,934.5 Mt; Chile 2009 WR −35,698 kt, i.e. ore −17,849 kt) and the whole common-set z² increase:

| sum-row z² by anchor band (\|c\|, kt) | rows | reference | this run | Δ |
|---|---|---|---|---|
| < 1 | 1,807 | 8,453 | 8,453 | 0.0 % |
| 1–10 | 1,770 | 117,124 | 117,124 | 0.0 % |
| 10–100 | 5,240 | 569,658 | 569,658 | 0.0 % |
| 100–1000 | 13,242 | 611,039 | 611,044 | 0.0 % |
| > 1000 | 5,057 | 184,221 | 185,361 | +0.6 % |
| total (27,116 common rows) | | 1,490,497 | 1,491,642 | +0.1 % |

The > 1000 kt increase is 2008 +451, 2009 +397, 2022 +681, led by the Ore/Mining scale totals of
China 2022 (z −18.2 → −20.5 Mass, −12.9 → −15.4 CC) and Chile 2008/09 (−23.7 → −25.2,
−22.4 → −24.0).

Round-E (ADR 0025), reference → this run: world retention 1995–2024 CO Mass 0.863 → 0.862,
Concentrate CC 0.990 → 0.990, Cathode CC 0.988 → 0.988; small nodes (|c| < 18 kt, n 4,772, all
years) posterior/prior median 1.000 → 1.000, p90 1.149 → 1.149; cross-region concentrate exporter
retention 1995–2024 Chile 0.970 → 0.970, Peru 1.013 → 1.014, Indonesia 0.915 → 0.915, world 0.924 →
0.924; mass moved over CC edges 501,359 → 502,999 kt CC.
The WR posterior stays exactly 2·Ore in every year (1995–2024: 107,842.9 → 107,685.8 Mt).

## 5. Side effect: the no-data anchor's sibling median

The anchor (ADR 0013) prices a zero-prior edge at 5 % of the median |a₀| of its anchored
(from_sector, valuation) siblings. The WR priors now count as (Mining, Mass) siblings: 889 → 1,778
siblings, median 23,520.6 → 34,826.3 kt, so the 1,042 anchored zeros (Mining→Ore and Mining→WR Mass
in the 521 region-years that mine nothing) loosen from σ 1,176 to 1,741 kt. Their contribution to δ
is ≤ 0.37 per year; no measurable effect. Kept on purpose (2026-10-07): leaving flat-σ priors out
of the siblings would restore the old scale, but it is a Stage-3 change for an effect below what
the solver resolves.

## 6. Follow-ups

- **Solver resolution** (commented on #33). At ADR 0039's tolerances a `Solved` year can sit
  ≥ 684 units of J from a known feasible point (2008 here, gap_abs 0.29). Run-to-run differences of
  a few hundred z² per year — ±0.1 % of the common-set sum-row z² of
  `opt_20261006_105904_61c5efe9`, up to 17,849 kt of ore Mass in Chile 2009 — are below what a
  single solve resolves. Cross-evaluating posteriors (as in §4) separates a model effect from this
  noise; earlier rounds that moved z² by ≤ 0.1 % against their base run did not.
- `docs/manuscript/si.md` Table S15 ("waste rock is no longer filled … 889 mass edges … to the flat
  prior") describes the W.1 reference run; it changes when the manuscript moves to a newer chain.

## 7. Reproduce

The chain:

```
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20260922_115241_eb9cb9ff \
    --feed-id feed_baci_20260922_090456_cdcb4c41 --feed-id feed_crude_ore_20260603_095651_558f7369 \
    --label wr_stripping_prior
python scripts/constraints.py --imp-run-id imp_20261007_122232_2b67458e wr_stripping_prior
python scripts/run_solver.py --const-run-id const_20261007_122703_50f6ff71 --label wr_stripping_prior
```

The diagnostics read every year, so they need the run's per-year tables, written once on their own
(≈ 0.43 GB; the reference run carries them already):

```
python scripts/solver_stats.py opt_20261007_130833_1a9b7af7 --years all
python scripts/explo/wr_prior_crosseval.py opt_20261006_105904_61c5efe9 opt_20261007_130833_1a9b7af7
python scripts/explo/roundY_psc_diag.py opt_20261006_105904_61c5efe9 opt_20261007_130833_1a9b7af7
python scripts/explo/roundK_sigma_floor_diag.py opt_20261006_105904_61c5efe9 opt_20261007_130833_1a9b7af7
```

`wr_prior_crosseval.py` prints §3's WR table and retention (W1), the δ table (W2) and the
cross-evaluation (W3).
