# #87: Stage 5's intervals are too narrow in the tail and too wide in the core — Birge R = 5.21; quote the Birge-scaled interval with its tail caveat; scale totals carry a tenth of the feed-CC contraction

**Runs:** `uncprop_20261021_134829_ad7e8d62` (default) and `uncprop_20261022_081020_2176b072` (label
`r87_no_scale_totals`, `exclude_scale_totals` True: 28,982 scale-total rows left out of S(t)), both on
`opt_20261021_131744_58504ad3`
**Against:** the model's own expectation — `z/√κ ~ N(0, 1)` under the active-set linearisation (ADR 0044), `R = 1`
**Chain:** `opt_20261021_131744_58504ad3` ← `const_20261021_130426_c5331a7f` ← `imp_20261021_125954_b112b170` ←
`feed_icsg_20261021_125725_b033ebc4`, `feed_baci_20261021_125615_bc691704`, `feed_crude_ore_20260603_095651_558f7369`
**Feeds/params:** icsg v17, baci v31 (gross v28), trade_shares v39, crude_ore v3, acc copper v4, conc v10, eff v4,
icsg_aggregates v4, ore_grade v8, scrap_share v3, stock_bounds v6, stripping_ratio v2
**Date:** 2026-10-22 · **Branch:** `diag/final-chain-3` · **Issue:** #87 (items 1–3 merged in `8bf3d4c`) · **ADR:** 0044

**Verdict: the raw σ* is not a calibrated interval, and no single scale makes it one.** The run-level Birge ratio
is **R = 5.213** (Σz² 2,574,347 on 493,864 degrees of freedom), so the stated σ are too small by √R ≈ 2.28 on
average. But the shifts are not a scaled normal. The core is *narrower* than predicted: the median |z/√κ| of
feed-observed CC edges is 0.46 against N(0, 1)'s 0.67. The tail is far heavier: 13.0 % lie beyond 3 against 0.27 %,
and 4.1 % still do after Birge scaling. Leaving the scale totals out of S(t) lowers the mass-weighted contraction of
feed-observed edges by 0.10 (CC) and 0.25 (Mass) and leaves most of it in place. **Recommendation:** the manuscript
quotes `a* ± 1.96·√R·σ*` (half-width 4.47 σ*) as the interval and raw σ* as the sensitivity under the stated σ. It
states that both understate the tail where the sources contradict each other, and that the scale totals, which
reuse the prior, count some prior information twice.

## 1. Calibration of the shifts (item 1)

Edges not bound-active, κ ≥ 0.01, not flat; weight `max(|a_prior|, |a*|)`. Expected share beyond 3: 0.27 %.

| Valuation | Prior class | Edge-years | \|z/√κ\| > 3 | weighted | median \|z/√κ\| | q99 | > 3 after Birge | weighted |
|---|---|---|---|---|---|---|---|---|
| CC | from a feed | 18,854 | 13.0 % | 5.2 % | 0.46 | 15.1 | 4.1 % | 1.0 % |
| CC | imputed | 47,182 | 7.5 % | 1.2 % | 0.16 | 10.4 | 2.4 % | 0.35 % |
| CC | zero-pinned | 100,411 | 1.6 % | 39.5 % | 0.05 | 5.6 | 0.8 % | 21.6 % |
| Mass | from a feed | 17,309 | 18.3 % | 0.46 % | 0.53 | 36.8 | 8.7 % | 0.20 % |
| Mass | imputed | 47,505 | 5.7 % | 0.03 % | 0.11 | 9.0 | 1.7 % | 0.01 % |
| Mass | zero-pinned | 100,214 | 1.3 % | 24.6 % | 0.04 | 3.8 | 0.6 % | 14.6 % |

Every class has a core below N(0, 1) and a tail above it. #87 names two candidate causes for the narrow core, neither
measured here: an edge observed by several rows moves less than its own σ_a allows, and imputed priors derived from
the same feed totals move together although the diagonal prior treats them as independent. The tail sits on the
overruled edges, where the stated limitations place the contradictions between the sources (method.md §3.2.8,
#111). The zero-pinned weighted shares are large on a tiny weight (2,336 kt CC over 30 years of edge-years). QQ plots per class: `ud_r_03_shift_calibration.png` in the run's
`diag/`.

## 2. Birge ratio (item 2)

R = Σz² / dof, with dof = soft rows kept + independent hard rows kept + bound-active edges not pinned at their prior.

- **Run:** R = 5.213, Birge-scaled 95 % half-width 1.960·√R = 4.475 σ*. It was 5.950 on
  `uncprop_20261020_120031_b2c16162` and 5.864 on `uncprop_20261021_125407_12e1dea7`, before #110 removed Russia's
  1995–2003 contradiction.
- **By year:** R runs from 1.83 (2000) to 11.96 (2017). The 1995–2003 mean is 2.81 and the 2016–2021 mean is 8.67.
  The late years carry the larger misfit: total Σz² 146,146 in 2016 and 194,702 in 2017, 75 % and 90 % of it on
  edges.

## 3. Scale-total ablation (item 3)

Same point estimate. κ and σ* come from S(t) without the 28,982 scale-total rows (`supplytotout`, `usetotout`,
`smelterscrap`, `directrefscrap`).

| Valuation | Prior class | κ (mass-weighted) default → ablation | confirmed share (by weight) | median σ*/σ_a |
|---|---|---|---|---|
| CC | from a feed | 0.931 → 0.831 (−0.101) | 0.915 → 0.855 (−0.060) | 0.967 → 0.979 |
| CC | imputed | 0.978 → 0.927 (−0.051) | 0.979 → 0.963 (−0.016) | 0.898 → 0.922 |
| CC | zero-pinned | 0.496 → 0.438 (−0.058) | 0.209 → 0.159 (−0.051) | 0.702 → 0.704 |
| Mass | from a feed | 0.973 → 0.719 (−0.255) | 0.982 → 0.940 (−0.043) | 0.981 → 0.989 |
| Mass | imputed | 0.968 → 0.906 (−0.062) | 0.999 → 0.999 | 0.888 → 0.902 |

The scale totals are the contraction's second source, not its first. Feed-observed Mass edges depend on them most
(−0.25). Everywhere else most of the contraction stays when the totals go, so it comes from the node balances and
ratio rows. The ablation is not a better interval: its σ*
belongs to a fit without the totals, while a* and z come from the fit with them.

## 4. Recommendation (item 4)

1. **Quote the Birge-scaled interval** `a* ± 1.96·√R·σ*`, with R = 5.21, wherever the manuscript gives an interval.
   It corrects the average understatement the model's own residuals reveal. It stays a lower bound in the tail: 4.1 %
   of feed-observed CC edges remain beyond 3 scaled σ.
2. **Report raw σ\* as a sensitivity,** as method.md §3.1.7 already does ("the estimator's sensitivity under the
   stated σ, not a calibrated interval"), and the contraction figures (Results) as properties of the model.
3. **State the double counting:** the scale totals reuse the prior and account for 0.10 of the 0.93 weighted
   contraction of feed-observed CC edges and 0.25 of 0.97 for Mass.
4. **Not adopted:** a per-class or per-year scale. The core is over-wide and the tail over-heavy in every class, so
   any single factor fits neither. The tail belongs to the stated limitations (#111), not to σ (ADR 0019).

Out of scope, as #87 says: correlated priors from Stage 2, parameter uncertainty in the hard ratio rows, and
aggregate σ (#81). Follow-up #99 changes the bound-active set by reading it from the multipliers.

## Reproduce

```bash
python scripts/uncertainty_propagation.py opt_20261021_131744_58504ad3
python -c "import sys; sys.path.insert(0, 'scripts'); import uncertainty_propagation as U; \
U.EXCLUDE_SCALE_TOTALS = True; U.main('opt_20261021_131744_58504ad3', 'r87_no_scale_totals')"
python scripts/explo/issue87_calibration_diag.py uncprop_20261021_134829_ad7e8d62 uncprop_20261022_081020_2176b072
```
