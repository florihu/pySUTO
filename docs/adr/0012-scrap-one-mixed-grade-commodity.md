# Scrap is one mixed-grade commodity — a single conc band, no grade classes

*Its invariant — **each commodity carries exactly one hard grade band** — was re-affirmed
2026-09-16 by [ADR 0028](0028-grade-sigma-is-stated-not-derived-from-the-band.md) as the reason
region-specific `conc` rows are not added: neither constraint builder dedups by specificity, so a
region row layers on the global one instead of replacing it.*

The `conc` params shipped **two** scrap supply concentrations — `concscrapclass1supply`
(`[0.90, 0.98]`, high-grade) and `concscrapclass2supply` (`[0.40, 0.60]`, low-grade) — both
*mapped onto the single `Scrap/SB` root node* (their own comments say *"Mapped from
'Scrap_class_N' → root 'Scrap/SB'"*). `RatioBoundBuilder` emits each as a **hard** grade band
on the same `(Mass, CC)` edge pair, so every scrap edge was required to satisfy
`[0.90, 0.98] ∩ [0.40, 0.60] = ∅`. The only feasible flow is 0, and the solver duly drove
**all 5,640 scrap input edges to ~0** in every year (`opt_20260624_083508_a6fdcc8c`;
[REP](../rep/analysis_opt_20260624_083508_a6fdcc8c_20260624_scrap-collapse.md)). The scrap
prior carried a real ~7–18 kt/yr signal, all destroyed.

The two "classes" are a **source taxonomy** (Schlesinger's scrap grades), not two model
commodities: the accounting structure has exactly **one `Scrap/SB/<region>/SEM` node**, and
[ADR 0005](0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md) already treats "Scrap"
as a single pooled secondary input to the Smelting/CFR recipe.

## Decision

Model **scrap as one mixed-grade commodity**. Replace the two class params with a single
`concscrapsupply` whose grade band spans both classes: `value_min = 0.40`, `value_max = 0.98`,
`value_nominal = 0.70` (assumed class-1/class-2 blend). Like every other `conc` param, this one
relation drives all three consumers together (per
[ADR 0011](0011-one-edge-relation-shared-across-imputation-and-constraints.md)): the
`ParametricImputer` prior (`CC = 0.70·Mass`), the soft `CC = conc·Mass` ratio row, and the hard
`0.40 ≤ CC/Mass ≤ 0.98` bound.

To stop this class of misconfiguration recurring, `ConstraintEngine.run` now validates that no
edge-pair carries hard ratio bounds whose intersection is empty (`_validate_ratio_bounds`), and
`RatioBoundBuilder.from_csv` rejects any single param with `value_min > value_max`. The
**invariant: each commodity carries exactly one hard grade band** is now enforced at build time.

## Considered options

- **A. One mixed-grade commodity, band `[0.40, 0.98]`. (chosen.)** Matches the single
  `Scrap/SB` root node and ADR 0005's single-pool model. The band contains the observed prior
  grade (0.50), so scrap is freed; `0.70` nominal is a neutral blend assumption, revisable when
  scrap-mix data exists.
- **B. Split scrap into `Scrap_class_1` / `Scrap_class_2` nodes**, each with its own conc and a
  smelter that consumes both. Preserves grade resolution but **contradicts the single root
  node**, is a structural accounting change, and has **no feed data** to populate two separate
  scrap flows (4,528 / 5,640 scrap edges have zero feed coverage). Rejected.
- **C. Keep two classes but make the conc bounds soft** (free cone, no hard band). Removes the
  hard contradiction but still imposes two conflicting soft pulls (toward 0.94 and 0.50) on one
  edge, and abandons the legitimate hard grade limits that every other commodity keeps. Rejected.

## Consequences

- **Scrap recovers.** The hard band `[0.40, 0.98]` contains the prior grade, so scrap is no
  longer pinned to 0; its level is set by the eff balance + priors.
- **Scrap's CC contribution to the eff pool rises ~40%** (nominal `0.50 → 0.70`), since
  Smelting/CFR pool scrap CC into `eff·Σin` (ADR 0005). Bounded — scrap is a secondary input,
  small next to concentrate — and visible in the re-solve.
- **Grade resolution is lost on purpose.** A region whose scrap is genuinely high-grade can no
  longer be distinguished; with no per-region scrap feed this was never identifiable anyway.
- **Empty hard ratio bands now fail loudly** at constraint-build time instead of silently
  collapsing flows — guarding all `conc`/`eff`/bound param classes, not just scrap.
- **Latent (out of scope):** the `ParametricImputer` resolves two equal-specificity params on
  one edge by row order; moot now that scrap has a single conc param, tracked separately.
