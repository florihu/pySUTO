# Soft net-stock bounds + share-of-base Δstock priors — fix for the flat-prior-zero-pin infeasibility

**Fixes:** `analysis_const_20260625_130524_f7c42959_20260625_flat-prior-zero-pin-cold.md`
**Const (soft, Part A):** `const_20260831_102251_f0fd9c4c`  ·  **Opt (Part A validation):** `opt_20260831_121315_645bbff1`
**Imp (Part B, ShareImputer):** `imp_20260831_125708_4e84af11`
**Branch:** `sensitivity/no-flat-prior-zeros`  ·  **Date:** 2026-09-01

## What was changed

Two coordinated changes replace the *hard* net-stock-change bounds (PSC/USC/MESC) — which went infeasible
under the flat-prior-zero pin — with a *soft* belt plus better priors.

**Part A — soft net-stock bounds (belt; guarantees feasibility).**
- `STOCK_BOUNDS_MODE="soft"` ([scripts/constraints.py]) routes `param_type=="stock_bound"` rows from the hard
  `RatioBoundBuilder` (nonneg cone) to the free-cone `RatioConstraintBuilder` (soft, minimized quadratic slack).
- σ_c fix in `RatioConstraintBuilder` ([src/pysuto/constraints/ratio.py]): the net-stock band is centred at 0
  (`value_nominal=0`), so the generic soft-ratio scale `max(num_scale, |r|·den_scale, floor)` collapsed to the
  abs floor (σ_c≈2 t: quasi-hard). Now, for `stock_bound` rows, σ_r = `value_max/Z95` (Z95=1.645, read p95 as
  the empirical 95th percentile) and `scale_mode="denominator"` scales off the **throughput denominator**:
  `σ_c = (value_max/1.645)·Σ|throughput prior|`. The per-region `value_max` already carries the MESC
  domestic/hub cluster, so A and B stay cluster-correct for free.

**Part B — `ShareImputer` (root cause; cold-smooth priors).** New [src/pysuto/impute/share.py], registered
before the flat-prior fallback. Gives Δstock edges informative priors instead of `noisy_zero`:
- **MESC/USC:** zero-mean, throughput-scaled seed `Δstock~N(0,(p95/Z95·FU(i→j))²)` per co-located edge —
  the honest prior for a 47×47 source→dest net change whose signed central share ≈ 0. **26,668 fills.**
- **PSC:** per-region share-transfer `share=PSC/(CSXEW+CER)` interpolated across years (NOCB seeds the 1995
  baseline). See limitation below.

## Result: feasible, no collapse, bounds well-calibrated

`opt_20260831_121315` (cold, year 1995 — the exact block that was infeasible):

| Check | Before (hard) | After (soft) |
|---|---|---|
| **Year 1995 feasibility** | INFEASIBLE (empty MESC interval) | **Solved** (35 iters, 8.8 s) |
| **Cathode collapse** (net-stock / throughput) | ~95 % into stock sinks | **1.35 %** — chain intact |
| **MESC slack \|z\|** (was the culprit) | — | median 0.07, max 0.40 |
| **USC slack \|z\|** | — | median 0.09, max 0.94 |

The soft stock rows land in the `free` cone (4230 rows); the hard-rows-only feasibility pre-check no longer
sees them, so the empty-interval failure is gone.

**Full A+B stack confirmed.** Rebuilding const on the Part-B imp (`const_20260901_072941_aa16dd34`) pins
**fewer** edges (671,376 vs 698,044 — the 26,668 ShareImputer fills raised coverage), and its 1995 σ-weighted
solve converges cleanly (**status=Solved, 36 iters, dres=2.3e-11**). Caveat: the *feasibility pre-check* probe
(a P≈1e-10 phase-1 QP) false-alarms `InsufficientProgress` on this pin set — a numerical fragility of the probe,
not a real infeasibility (the actual solve, and the larger feasible region from fewer pins, both confirm it).
Making the pre-check accept a stalled-but-feasible iterate (as the main solve already does via reduced
tolerances) is a separate solver-diagnostic follow-up.

## Incidental fix: solver MAX_ITER 200 → 2000

The first cold re-run still halted at the 1995 feasibility pre-check with `status=MaxIterations` (not
`PrimalInfeasible`). Cause: the pinned 1.37M-row phase-1 QP (P≈1e-10·I) needs more than the default 200
Clarabel iterations to converge. Direct test: `max_iter=200` → FAIL, `max_iter=2000` → **PASS**. Raised
`MAX_ITER=2000` in [scripts/run_solver.py] (a cap — easy runs still stop early). The *actual* σ-weighted solve
is well-conditioned and converges in 35 iters; only the pure-feasibility probe needed the higher cap.

## Known limitation: PSC slack z-tail (low value, documented)

PSC slack z has a tail (median 0, p90 7.4, **max ~36** on a few regions). This is **not** a
collapse/infeasibility, and it is **not** fixed by `ShareImputer`:
- PSC `share_transfer` fires **0** — PSC coverage correlates with its base: where the CP recipe closes,
  `NetworkImputer` already fills PSC (684 edges); where it doesn't, the base (CSXEW+CER) is missing too.
- The z-tail is downstream of the pinned/cold run's ~10× throughput inflation (a separate magnitude-anchoring
  issue) combined with genuinely-no-data regions, where the σ_c denominator (CP→Cathode prior) floors. The
  σ_c-off-inputs tweak (CSXEW+CER) helps only regions where the output is uncovered but the inputs are (e.g.
  Belgium: CP→Cathode=0, CSXEW+CER=471), not truly-no-data regions (Argentina: both 0).

Left for a future magnitude-anchoring pass; it does not affect the feasibility or collapse results above.

## Sources

- Code: [scripts/constraints.py], [src/pysuto/constraints/ratio.py], [src/pysuto/impute/share.py],
  [scripts/collect_aggregate_impute.py], [scripts/run_solver.py].
- Tests: `tests/test_constraints.py::test_stock_bound_soft_sigma_c_throughput_scaled`, `tests/test_impute_share.py`.
- Runs: const_20260831_102251_f0fd9c4c, opt_20260831_121315_645bbff1, imp_20260831_125708_4e84af11.
