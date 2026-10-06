# Closing the free copper source/sink — no-data σ anchor + scrap-share bands + FU-split formula fix

**Fixes:** the primary/secondary inversion found in
`analysis_opt_20260901_080337_99aa122f_20260901_all-years-deviation-scan.md` (baseline).
**Step 1 (const-only):** `const_20260901_094743_1cb295a5` → `opt_20260901_095600_34428708`
**Step 2 (full):** `imp_20260901_095921_6bf957e8` → `const_20260901_100422_446a848d` → **`opt_20260901_101148_39fbd1bc`**
**Branch:** `sensitivity/no-flat-prior-zeros` · **Date:** 2026-09-01

## The problem

The baseline all-years solve inflated boundary **Scrap** copper ×10.7 (169 → 1,805 kt Cu) while deflating
primary **CO** ×0.81, inverting the primary input share from a realistic 75–84 % (prior) to 12–29 % (post),
and inflating total boundary copper ×2.9. Mechanism: 94 % of the scrap copper entered through
`flat_prior_noisy_zero` edges (a₀=0, σ=1e6 — e.g. Scrap/Panama→CFR and Scrap/Taiwan→CFR at 21 kt Cu each,
z≈0.02), and FU final-use outputs were equally noisy_zero. **Inflation needs a free source and a free sink —
both existed.** Root findings:

1. **`pin_flat_prior_zeros` is inert in the QP.** It sets `ConstraintRun.l/u = 0` on 671,376 edges, but l/u
   are only read by the feasibility pre-check; neither solve path hands them to Clarabel. (ADR 0013 amended.)
2. `NODATA_ANCHOR_CV = 0.0` — the soft σ anchor was disabled *because* the pin was assumed to supersede it.
3. Scrap→Process edges are in **neither** scale-total sum family → no row constrained them at all.
4. Latent bug: the cathode FD-split FormulaRelations selected `entity_to=="Process"`, but FU/USC/MESC are
   `entity=="SB"` → the formulas never fired (0 rows in every imputation log to date).

## The fix (three parts)

**1 — No-data σ anchor re-enabled, tight ([scripts/constraints.py]).** `NODATA_ANCHOR_CV = 0.05`
(σ = 0.05·median|a₀| of data-backed siblings per (from_sector, valuation) — z=20 per median-flow deviation;
soft, cannot cause infeasibility). Exclusion list aligned with the pin's 7-sector set
(`+Slag,PSC` — SEM residual outputs). This is the QP-effective mechanism the inert pin was believed to be.
671,376 edges anchored, σ 1e6 → [0.008, 116].

**2 — Per-region hard scrap-share bands (new `scrap_share` params family).**
`bl·Σ(CC inputs to P) ≤ Σ(Scrap→P) ≤ bu·Σ(CC inputs to P)` per (region, P ∈ {Smelting, CFR}), `agg=true`,
via the unmodified `RatioBoundBuilder` (numerator ∈ denominator nets 1−bu — scipy sums duplicate COO
entries). Bands from observed ICSG shares (`scripts/explo/scrap_share_study.py`, mirroring
stock_bounds_study): per-region [p5−0.05, p95+0.05] (≥3 obs), default [0, 0.25] otherwise. 94 rows / 2,820
year-expanded / 5,640 G rows. Strongly region-dependent and correct by construction: pure-secondary smelters
(Austria/Slovakia/Colombia/Italy) get [0.95, 1.0]; primary-dominated (Mexico [0, 0.10], US [0, 0.27]).
Hard-only by design — a soft centre row would fight the zero anchor on no-data regions (deliberate deviation
from the ADR 0011/0012 one-relation-drives-all pattern). `SCRAP_SHARE_MODE` knob: hard/soft/off.

**3 — FU-split formula fix ([scripts/collect_aggregate_impute.py] + [src/pysuto/impute/formula.py]).**
Predicate `entity_to` "Process"→"SB" (4 sites) **plus** a new `SourceTerm.missing="zero"` policy (default
"block" — no change to other relations): the FD trio was mutually circular (FU blocked on USC+MESC being
observed, and the export sum blocked on any missing bilateral leg), so even with correct predicates it fired
0×. Now **FU counts missing USC/MESC/export legs as 0** (GLOSSARY.md "missing counts as 0") and fires wherever
CP→Cathode production is known (**194 fills**); USC/MESC keep full block semantics so the ShareImputer's
zero-mean throughput-scaled seed (Part B) is not preempted.

## Results (cold, all 30 years, pre-check neutralised; every year's real solve **Solved**)

| Metric (CC, Σ over 30 y) | Baseline `opt_..._99aa122f` | Step 1 (anchor+bands) | **Step 2 (full)** |
|---|---|---|---|
| Boundary Scrap in | 1,805 kt (×10.7 prior) | 67 kt (×0.39) | **66 kt (×0.39)** |
| Boundary CO in | 515 kt (×0.81) | 450 kt (×0.71) | **466 kt (×0.73)** |
| Total boundary Cu | 2,320 kt (×2.9 prior) | 517 kt (×0.64) | **532 kt (×0.66)** |
| Primary share (post, 1995→2024) | 12–29 % | 82–91 % | **83–92 %** (prior: 75–84 %) |
| Panama/Taiwan scrap | ~42 kt/yr peaks | 0.0 | **0.0** |
| Scrap-band violations (of 5,640 hard rows) | — | 0 | **0** |
| Net-stock MESC / USC max \|z\| | 3.4 / 2.6 | 3.5 / 5.1 | **3.4 / 5.1** |
| Soft \|z\| p99 | 16.4 | 24.3 | **20.9** |

SD-R-10 (`diag/sd_r_10_boundary_flows.png`): the CC-inputs pane now shows **CO well above Scrap** in every
year — the inversion is gone; boundary trajectories remain smooth.

## Reading the numbers

- **The faucet is closed and the mix is right.** Scrap enters only where data supports it (edges with a real
  prior move by ratio ≈1.00); the primary share is restored slightly *above* prior — the solve now leans on
  the data-backed primary chain.
- **The failure mode moved from "inflate ×2.9 through free faucets" to "settle at the data-backed level".**
  Total boundary copper is now ~⅔ of the raw prior sum, pinned by the anchored CO chain, because most FU legs
  still have no observation and are anchored to ~0 (the FU formula fills 194 region-years — those with known
  cathode production; the rest have nothing to anchor to). Raising this further is a **data-coverage**
  question (real per-region consumption series for FU), not a solver question.
- **The z-tail (p99 20.9, max 92) sits in the `sum_usetotout_Cathode_*` scale totals** (China 2022/23,
  Kazakhstan/Uzbekistan/Iran mid-90s): the Stage-2 prior total for those regions' cathode output cannot be
  reached now that its noisy destinations are anchored — the pre-existing sum-family strain (baseline had
  z≈50–59 on China ER) made visible instead of absorbed by free edges. Same family as the magnitude-anchoring
  follow-up; not a regression of this fix.

## Files

- [scripts/constraints.py] — `NODATA_ANCHOR_CV=0.05`, `NODATA_ANCHOR_EXCLUDE` (7 sectors),
  `SCRAP_SHARE_MODE="hard"` + registration, stage_config, docstring.
- [scripts/collect_aggregate_impute.py] — FD-split entity predicates + `missing="zero"` on the FU terms.
- [src/pysuto/impute/formula.py] — `SourceTerm.missing` policy ("block" default / "zero").
- [scripts/explo/scrap_share_study.py] (NEW) → `data/input/params/scrap_share/current/params.csv` (v1, 94
  rows) + `figs/ex/scrap_share.png`.
- [docs/adr/0013-no-data-flat-prior-anchor.md] — amended (pin is pre-check-only; anchor is the QP mechanism;
  structural SEM detection must be direction-aware).
- Tests: `test_scrap_share_band_num_in_pooled_denom`, `test_scrap_share_generator_orientation`
  (test_constraints.py); `TestCathodeFdSplitSelectors` (×3), `test_missing_zero_counts_nan_source_as_zero`
  (test_imputation.py). Full suite: **860 passed, 1 skipped**.

## Follow-ups (documented, not chased)

1. **FU data coverage** — the boundary-copper level is now floored by anchored zeros on unobserved FU legs;
   a real consumption feed (e.g. ICSG usage series per region) would anchor the sink at its true level.
2. **Cathode `usetotout` scale-total strain** (z up to ~92 on China/Kazakhstan/Uzbekistan/Iran) — part of the
   standing magnitude-anchoring follow-up.
3. Feasibility pre-check probe fragility — unchanged, still bypassed for the all-year scans.
