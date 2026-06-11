# Scale totals (self-anchored to f_agg) replace external ICSG control totals

**Status:** accepted — amends [ADR 0001](0001-replace-coherence-sum-eq-with-bilateral-anchoring.md).

[ADR 0001](0001-replace-coherence-sum-eq-with-bilateral-anchoring.md) fixed the homogeneous
scale-collapse by anchoring a handful of edge families to **externally published ICSG totals**
("control total" in the [`README`](../../README.md) glossary — a Type-2 soft-sum whose RHS `c(t)`
is an external number). That covered only 7 hand-listed families
([`scripts/constraints.py` ICSG_ANCHOR_FAMILIES](../../scripts/constraints.py)); every other process
and flow stayed unprotected from the collapse. We **replace** that block with a **scale total**: a
*self-referential* total-of-flows soft-sum that pins each process's and each flow's total-output
**scale** to the model's own Stage-2 prior, `Σ (group edges) = Σ f_agg over the group`, with the
usual tight soft `σ_c = ANCHOR_CV·|c|` (0-excluded, year-locked, per valuation). Two families,
both grouped by `(sector_from, region_from, time, valuation)`:

- **supply total output** — mask `entity_from==Process & entity_to==Flow` (process→flow useful
  output) ⇒ one node-scoped row per process (a single process→flow edge *is* the total supply);
- **use total output** — mask `entity_from==Flow` (flow→process use **+** flow→SB final demand)
  ⇒ one node-scoped row per flow, summed over all `sector_to` destinations = the total output of
  an industrial flow.

**Why `f_agg`, not external data.** The Stage-4 solver is a warm-start chain — `a₀ = f_agg` at the
first year, `a_post(t−1)` thereafter ([`README` §Prior a₀](../../README.md)). So the year-`t` solve
never *sees* the fixed Stage-2 `Σ f_agg` total as its warm-start; the scale total acts
**"kind-of-external"** to each per-year solve, pulling it back toward the imputation-aggregate scale.
And `f_agg` already embeds the Stage-1/2 external feeds (ICSG usage, BACI trade-share weighting per
[ADR 0002](0002-single-endpoint-rows-fan-out-not-replicate.md)), so anchoring to it is **not purely
circular** — it preserves the scale of a prior that *was* built from external data, now over **all**
processes/flows instead of 7 families.

## Considered options

- **A. Keep external ICSG control totals only (status quo).** Rejected: covers only 7 families;
  leaves most of the network exposed to the collapse the anchoring was meant to stop.
- **B. Coexist — external control totals + scale totals.** Rejected: the 7 ICSG sectors would be
  anchored **twice** on the same edges (an external `c` and a `Σ f_agg` `c`), an over-constraint that
  can conflict with the `ratio_bound` limits in the closed sub-network.
- **C. Replace the external block with self-referential scale totals. (chosen.)** Comprehensive
  scale-preservation; one mechanism (`from_mask_anchor` pinning a masked group-sum to a reference),
  the reference now being `f_agg` instead of `icsg_F_root`.

## Consequences

- **`scripts/constraints.py`.** `ICSG_ANCHOR_FAMILIES` + `ANCHOR_USE_SECTORS`, the A3 `icsg_F_root`
  loading, the `DataFeedRegistry` import, and the `icsg_feed_run_id` provenance entry are all removed;
  two `SumConstraintBuilder.from_mask_anchor(anchor_values=f_agg, …)` registrations take their place.
- **Amends ADR 0001.** Its Stage-3 external-anchor *registration* is removed; its Stage-2 `a₀`
  bilateral seeding is **untouched** and still flows into `f_agg`.
- **Loses external grounding at Stage 3 (the deliberate no-).** No total is pinned to a published
  number anymore; correctness now rests on the Stage-1/2 feeds embedded in `f_agg`, hard mass
  balance, and the conc/eff `ratio_bound` limits.
- **Terminology.** New canonical term **scale total** (internal RHS), distinct from **control total**
  (external RHS, now removed). The README glossary's *avoid "anchor"/"coherence soft-sum"* stands;
  the builder method `from_mask_anchor` is kept as the generic *verb* (pin a masked group-sum to a
  reference vector) shared by both mechanisms. See [CONTEXT.md](../../CONTEXT.md).
- **`sum.py`.** `from_mask_anchor` now fills `row_meta.sector` (symmetric with `region`) for
  node-scoped rows grouped on a sector key, so the new node-sum rows carry full node identity.
- **Validation.** At the prior `a0 = f_agg` we have `c = Σ f_agg`, so the scale-total rows have
  residual `G·a0 − c ≈ 0` (`z ≈ 0`) by construction in `ConstStatsDiagnostic`'s prior violation
  table — a direct wiring check. Re-run against the collapse reference and confirm total-mass
  retention near 1.0.
