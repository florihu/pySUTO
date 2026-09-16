# Bilateral trade-leg σ is size-damped (power law around the sector median)

**Status:** accepted. Evidence: Round-B REP (γ=0.5 and γ=0.7 chains vs Round-A baseline
`opt_20260903_105145_fa7049e3`); motivation in the root-cause REP
`analysis_opt_20260903_084528_45fdd041_20260903_masscc-tension-root-causes.md` §4.
Amended 2026-09-16 by [ADR 0027](0027-re-exports-are-re-attributed-at-feed-time.md): `a_ref`
may be **pinned** to a predecessor run's medians across a measurement series, so that a change to
the covered-leg population is not read as a change in physics. The median definition below stands
for a fresh baseline, and a pin is an obligation to release it at the next round's start.

## Context

BACI Mass legs carry the Laner et al. pedigree cv (~0.21) — a quadrature of source-quality
scores that is **size-blind by construction**. Under cv-proportional σ, deleting any leg
entirely costs z = 1/cv ≈ 4.8 whether it is 5 kt or 21 Mt; per tonne, the biggest legs are
thousands of times cheaper to delete. Measured: deleted share of prior tonnage rises
monotonically from 7 % (smallest size decile) to 37–45 % (largest); Chile→China/Japan
concentrate legs are zeroed at exactly z = −1/cv while thousands of tiny legs survive. As a
statement about data this is inverted — a multi-Mt flow recorded by two customs offices is
among the best-measured numbers in the system, while small flows carry the misclassification
noise (Papua New Guinea's 62–136× legs being the extreme, handled by the plausibility screen).

## Decision

For every **covered** bilateral **Flow-origin Mass** leg (any destination — Cathode's trade
legs terminate in `SB` final demand, not a Process; same population as the outlier screen),
Stage 2 rescales the aggregated σ before imputation:

    σ_e = cv_pedigree(e) · a_ref(sector)^(1−γ) · a_e^γ ,   γ = 0.5 (default)

- `cv_pedigree(e)` — the leg's incoming cv (σ_agg/f_agg): the pedigree stays the quality
  anchor; damping only redistributes confidence **within** a sector by size.
- `a_ref(sector)` — the sector's median covered bilateral leg |f|, pooled over all years: a
  median-sized leg keeps exactly its pedigree σ; a 100× bigger leg gets √100 = 10× lower cv
  (full deletion costs 10× the z, 100× the z²); smaller legs get looser, capped at cv ≤ 1.0.
- Order: damping first, then the outlier screen (ADR-0014-era plausibility σ-inflation, Round
  A) — a flagged leg's cv-5 inflation overrides.
- CC twins inherit automatically: `trade_seed` combines `conc·σ_Mass` with no cv cap
  ([impute/trade.py](../../src/pysuto/impute/trade.py)).
- `LEG_SIGMA_GAMMA = 1.0` (or None) disables the feature.

## Placement: Stage 2, an extension of the ADR-0003 pattern

ADR 0003 penalises *fed* edges in Stage 1 (`cv_root_scale`) and reserved Stage 2 for imputed
edges. This decision deliberately modulates fed-edge σ in **Stage 2** because the information
used only exists post-collection: the sector median is a property of the *assembled* root-leg
population across all years (Stage 1 sees one feed's source rows), and γ iteration must not
require feed reruns. The division of labour is now: **Stage-1 pedigree = pure source quality;
Stage-2 modulation = post-collection information** (size within the assembled market — this
ADR; cross-feed plausibility — the outlier screen).

## Considered options

- **Quadrature two-component** σ² = (cv∞·a)² + s0² — fits the Laner sum-in-quadrature
  philosophy, but the absolute floor s0 needs per-sector tuning (concentrate vs cathode leg
  sizes differ by orders of magnitude); the power law self-scales via a_ref.
- **Pedigree size-axis** (a magnitude score through `_cv_other`) — philosophically minimal,
  but quadrature can only *add* variance: it can loosen small legs, never tighten big legs
  below the pedigree cv. Half the lever.
- **Share-preserving (gravity) composition rows** — attacks the symptom structurally but is a
  new constraint family with its own σ design; deferred.
- **Global or per-year anchors** — a global a_ref conflates size-within-market with
  cross-market quality; per-year anchors make the same physical leg's σ drift with trade growth.

## Consequences

- Deletions redistribute according to size-encoded confidence; the world Mass–CC gap does not
  vanish — it relocates onto the scale totals of the nodes whose (weak) totals the
  now-trusted big legs push against (Cathode_ROW ≈ z 70 — pre-existing in the baseline at 67.6,
  barely moved).
- **γ evidence** (Round-B runs vs baseline `opt_20260903_105145_fa7049e3`): γ=0.5
  (`opt_20260903_124655_4b23df2f`) flattens the deletion-by-decile profile 0.07→0.38 to
  0.15→0.26 and recovers Chile (2015: CO Mass 0.22→0.31, CO CC 0.35→0.50, concentrate exports
  0.16→**0.51**) with the world chain intact (Cathode CC retention 0.95→0.99) at the cost of
  one regional tail (Cathode_China max |z| 11.7→18.1); γ=0.7 (`opt_20260903_131430_4194ccf4`)
  buys only half the effect (Chile exports 0.22, deciles still rising 0.09→0.35).
  **γ = 0.5 ships.** Totals: soft z p99 9.98→10.38, total z² unchanged (3.96M).
- The Laner framework is intentionally departed from for these legs; the departure is
  size-only, sector-local, and reversible by one constant.
