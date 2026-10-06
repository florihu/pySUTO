# Unweighted fans are not distributed; qualifier fans are restricted to their input sector's legs

**Status:** accepted — amends [ADR 0002](0002-single-endpoint-rows-fan-out-not-replicate.md)
(option A's residual uniform fallback).

[ADR 0002](0002-single-endpoint-rows-fan-out-not-replicate.md) fixed single-endpoint replication
by routing ICSG `Use` rows through a weighted fan, and option D reweighted the fan by BACI trade
shares. Two residual defects remained, both found while tracing the Stage-4 mass collapse (REP
[`…mass-collapse-ratio-bounds.md`](../rep/analysis_opt_20260611_075036_6d2e540f_20260611_mass-collapse-ratio-bounds.md)):

1. **The uniform fallback fabricates data.** Fans BACI does not cover (499 fans, median 47 legs;
   e.g. `Matte→CFR.KZ 1995`) fell back to an even 1/N split — a fabricated uniform partner
   pattern asserted per edge at the national total's confidence (cv ≈ 6.8 %), feeding both the
   per-edge prior `a₀` and (via `f_agg`) the scale-total anchors.
2. **Route-keyed weights bleed across products.** `user_weights_df` is keyed on the *root* route
   `(sector_from, sector_to, region_from→region_to, year)`, so Concentrate trade shares meant for
   `Smelting_Concentrate` fans also attached to the Concentrate legs that the `'<any>'` open side
   had wrongly pulled **into `Smelting_Scrap` fans** (657 fans) — scrap-production tonnage was
   distributed onto Concentrate trade edges while the correct domestic `SB.Scrap` leg held a
   median 0.27 % share. Symmetrically, every Concentrate/Anode fan carried a spurious domestic
   `SB.Scrap` leg (share up to 0.50). The same mechanism let `Refining_Anode → ER` (and later
   `→ CFR`, where **no Anode leg exists structurally**) build a non-empty fan out of whatever
   edges the consumer node happened to have — no guard could fire because the fan was non-empty.

## Decision

Two coordinated policies in `ConcordanceTable.from_raw` (opt-in per feed; ICSG enables both):

1. **Qualifier leg filter** (`leg_filter: {src_sector → required root supplier sector}`, derived
   from the ICSG sector_map's `icsg_type`). A Use fan whose variable names its input product keeps
   only legs of that supplier sector. A fan left with **zero legs is a hard error** — the mapping
   targets a structurally nonexistent edge family (this is what would have caught
   `Refining_Anode → CFR`). Scrap fans thereby collapse to the **domestic** `Scrap` leg without a
   special case, because `Scrap→Smelting` / `Scrap→CFR` edges are domestic-only in the acc.
2. **No weight → no distribution** (`drop_unweighted_fans`). On the filtered fan: legs that remain
   uniform inside a weighted fan are dropped (weights renormalise over the weighted legs); a fan
   with **no** weights at all (or weights summing to 0) is **not distributed** — its raw rows
   leave the concordance table and are reported as *undistributed* (distinct from *unmatched*).
   N = 1 exact matches are untouched. Dropped rows are automatically excluded from the
   mass-conservation total (the ADR 0002 guard counts only matched-and-distributed raw rows).

## Considered options

- **Keep the uniform fallback (status quo, ADR 0002 option A residual).** Rejected: fabricates a
  uniform partner pattern with high per-edge confidence; the REP showed these values feed real
  constraint tension.
- **Bundle-total soft constraint** (one soft sum row over the partner bundle, per-edge priors fall
  back to the imputer; REP §8.4). Rejected for now: preserves the national total in the constraint
  set but requires a new feed→constraint path; the scale totals (ADR 0006) already anchor to
  `f_agg` per node, so the marginal value does not justify the machinery. Can be revisited if
  dropped totals prove material.
- **Weight-mode-based policy without the leg filter.** Rejected: it would have dropped the
  *correct* `SB.Scrap` leg from `Smelting_Scrap` fans (uniform) and kept the *wrong* BACI-weighted
  Concentrate legs — the bleed makes weight modes unreliable as a per-leg correctness signal.

## Consequences

- Region-years without trade-share coverage lose their ICSG per-edge values and their contribution
  to the `f_agg` scale-total anchors: those edges fall back to the imputation cascade — the honest
  "total unknown split" state. The undistributed tonnage is logged per feed build.
- Scrap-fed consumer totals land on the domestic `Scrap` leg only (scrap is not traded in this
  model; cross-region scrap legs do not exist structurally).
- A wrong sector_map root (the `Refining_Anode → ER/CFR` class) now fails the feed build loudly
  instead of silently distributing onto foreign edge families.
- Glossary: see **Unweighted fan** and **Qualifier fan** in [GLOSSARY.md](../../GLOSSARY.md).
