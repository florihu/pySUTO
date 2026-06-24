# Pool Scrap into the eff constraint; store the stripping ratio as a process I/O ratio

Two efficiency-parameter inconsistencies surfaced by the [`NetworkImputer`](0004-process-recipe-traversal-imputer.md):

1. **Pooled inputs were not reflected in the `eff` *constraint*.** Smelting and CFR pool `Scrap`
   with their primary feed (`Matte = eff·(Concentrate + Scrap)`, `Anode = eff·(Matte + Scrap)`).
   The NetworkImputer already imputes this pooled form, but the Stage-3 `eff` ratio constraint
   related the product to the **single** `eff`-named primary only, so the imputed prior exceeded
   the soft row by `eff·Scrap` — the row was a soft upper-bound proxy, not the recipe (the
   [ADR 0004](0004-process-recipe-traversal-imputer.md) "eff ratio scope" follow-up).
2. **The waste-rock ratio was a hard-coded constant.** `WR_Mass = 2.0·Ore_Mass` lived as
   `WR_TO_ORE_RATIO = 2.0` in `scripts/collect_aggregate_impute.py`. It is the mining **stripping
   ratio** — a `Mass→Mass` ratio between two *outputs* of Mining (Ore product, WR waste) — and
   should be a versioned parameter driving both the imputer and the constraint stage.

We **pool Scrap into the `eff` constraint via schema-driven aggregation** and **store the stripping
ratio in the eff params file as its own `param_type`**.

## Decision

- **Pooled `eff` constraint via `agg` rows.** The existing `agg=true` mechanism (previously only in
  `RatioBoundBuilder`, for stock-change bounds) is ported to the soft `RatioConstraintBuilder` and
  its bucket key gains **`region_to`**: an `agg` row pools every matched a-side edge against every
  matched b-side edge per `(region_to, time, valuation)` bucket into one row `Σb − r·Σa`. The
  Smelting/CFR `eff` rows are marked `agg=true` with a broadened `a_sector_from=*` so the denominator
  is the **sum over all CC use-edges into the process** (Concentrate/Matte + Scrap, any source
  region) — the *same* set the imputer pools, so prior and constraint agree.
- **Aggregation is declared explicitly in the params file.** Whether a row is pooled is legible from
  the CSV (`agg=true` + a self-documenting `description`), never inferred from code.
- **Stripping ratio as a parameter.** A `stripping_ratio` row (`param_type=stripping_ratio`,
  `Mass`, single-pair `WR ← Mining → Ore`) replaces the constant. `NetworkImputer` reads it by
  `param_type` (with the constructor value as a fallback); the soft `RatioConstraintBuilder` emits
  `WR − stripping_ratio·Ore = 0`. It is **not** typed `efficiency`: the glossary reserves `eff` for
  a CC recovery fraction, so the eff *file* becomes a store of **process input/output ratios** of
  which recovery-`eff` and `stripping_ratio` are two families. Soft-only — no hard ratio-bound cone.

## Considered options

- **A. Schema-driven `agg` with `region_to` bucketing. (chosen.)** Declarative in the CSV, reuses
  the existing `agg` concept, and one bucket-key change serves both the soft eff constraint and the
  hard eff bound. Adding `region_to` is safe for stock-bounds (which pin `region_to` per row) and
  necessary for eff (per-region, same-region recipes).
- **B. Recipe-derived constraint builder.** Have `RatioConstraintBuilder` read `acc.process_recipes`
  and build `product − eff·Σ(inputs)` directly. Guarantees an identical input set to the imputer,
  but is not expressed in the schema, couples the builder to recipes, and only works for `eff`.
  Rejected: the user wants pooling expressed *in the params file*.
- **C. Same-region-only pooling.** Sum only inputs whose source region equals the process region.
  Rejected: diverges from `_do_recipe`'s all-region `S`, so prior and constraint would pool
  different sets.
- **Stripping ratio typed `efficiency`.** Rejected: contradicts the glossary's CC-recovery
  definition of `eff` (the value is `2.0`, a Mass ratio, not a recovery fraction).
- **Hard bounds on WR/Ore.** Rejected for now (soft equality only); no defensible empirical
  min/max yet.

## Consequences

- **Constraint↔prior consistency.** For Smelting/CFR the soft eff residual at the prior is now ≈ 0
  instead of `eff·Scrap`. The eff row is the recipe, not an upper-bound proxy.
- **Set-scoped ratio rows.** A `ratio` row may now be set-scoped (agg): `num_*`/`denom_*` row_meta
  columns are null and its id rests on `variant` + `(region, valuation)` group keys — like a
  set-`sum`. Glossary "Node-scoped vs set-scoped" and README §Stage 3 updated.
- **Single source of truth for the WR ratio.** The constant is gone; both stages read the
  `stripping_ratio` row. Its uncertainty (`unc_type`) is a placeholder pending a domain source.
- **ParametricImputer consumed-set narrowed.** The broadened `a_sector_from=*` would have made the
  per-pair fallback backfill a spurious `Scrap = Matte/eff`. `ParametricImputer` now **explicitly
  skips `agg=true` and `stripping_ratio` rows** (those are owned by `NetworkImputer` + the ratio
  builders). The systemic "every consumer declares which params it reads" redesign is deferred to a
  GitHub issue.
- **Underscore-free param_ids (incidental, required).** The `constraint-id-grammar` refactor (#10)
  made `naming._safe` reject `_` in the param_id-derived `variant`, which had silently broken the
  CSV-driven ratio path for every shipped param file. Renaming `eff_*`/`conc_*`/`net_stock_bound_*`
  to underscore-free spellings (and the `stock_bounds_study` generator) unblocks it. `float_or` was
  also made to honour its "non-numeric → default" contract so blank bounds (`*` after `fillna`) no
  longer raise.
- **Scope.** The Stage-3 `σ_a` CV floor (ADR 0004 "A1 caveat") is untouched.

> **Amendment (2026-06-24).** The shipped `stock_bounds` params were later found **a/b-inverted**
> relative to the `Σb − r·Σa` form documented above (and CONTEXT.md §"Node-scoped vs set-scoped"):
> `stock_bounds_study.py` placed the stock edge on the `a`-side (numerator) and the FU/Cathode
> denominator on the `b`-side, so `RatioBoundBuilder` built `denom ≤ p95·stock` (forcing
> `stock ≥ denom/p95`, ~20–42×) instead of `stock ≤ p95·denom`. This forced ~95% of refined copper
> into the PSC/USC/MESC stock sinks and collapsed the Cathode chain. Fixed by swapping the generator's
> CSV `a`/`b` assignment so the stock edge is the `b`-side numerator. (A separate, pre-existing bug —
> the PSC denominator referencing a non-existent `Anode` ICSG type — is tracked as a follow-up.)
