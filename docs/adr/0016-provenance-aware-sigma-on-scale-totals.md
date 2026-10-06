# Provenance-aware σ on scale totals: mechanism retained, loosening rejected

**Status:** rejected as a default (mechanism kept, inert). Evidence: REPs
`analysis_opt_20260903_084528_45fdd041_20260903_masscc-tension-root-causes.md` (motivation) and
the Round-A REP (rejection runs `opt_20260903_101715_821a620b` cv=0.15,
`opt_20260903_103441_42cf1331` cv=0.05, isolation `opt_20260903_105145_fa7049e3`).

## Context

The `supplytotout`/`usetotout` **scale totals** pin each node's total output to the model's own
Stage-2 prior group-sum. Their documented purpose is scale preservation against the homogeneous
collapse, *"not to add information"* (GLOSSARY.md §Total-output). Every row ran the same tight
`ANCHOR_CV = 0.01` (effective cv ≈ 0.017 after Lp conditioning) regardless of whether the prior
behind the group was a feed observation or a twin/recipe fabrication.

The root-cause analysis of `opt_20260903_084528_45fdd041` showed the cost of that uniformity:
3.0M of 5.13M z² sat on **fabricated-prior** totals, and the drag inverted the data hierarchy —
observation-backed totals moved *more* than fabricated ones (median |move|: Concentrate CC 8.7 %
vs its twin-derived Mass 4.9 %; Smelting CC 10.2 % vs 3.3 %).

## Decision

1. **Mechanism (kept):** `SumConstraintBuilder.from_mask_anchor` accepts an optional per-group
   `sigma_cv_resolver`; `scripts/constraints.py` classifies each scale-total group by the
   feed-backed share of its Σ|f_agg| (`agg_result.coverage`, threshold
   `FEED_SHARE_OBSERVED = 0.5`) and assigns `FABRICATED_ANCHOR_CV` to fabricated groups.
2. **Policy (rejected):** loosening fabricated groups. Tried at cv 0.15 and 0.05 (~77 % of
   groups loosened); **both collapsed the world chain** instead of restoring the hierarchy:
   world CO Mass posterior retention 0.66 → 0.41 / 0.45, world Cathode CC 0.93 → 0.75 / 0.79,
   Chile CO Mass 0.26 → 0.11, and even *observed* totals were dragged through the hard balances
   (China cathode-use z ≈ −20 / −16 on its own tight anchor).
3. **Default:** `FABRICATED_ANCHOR_CV = ANCHOR_CV` — the resolver is inert until a σ design
   decouples the two roles below.

## Why loosening fails here

The fabricated scale totals do **double duty**. Besides (over-)preserving fabricated priors they
are the effective counterweight to the **nodata zero anchors**
([ADR 0013](0013-no-data-flat-prior-anchor.md), cv 0.05·median-sibling, pulling every
no-data interior edge toward 0) on the very same subgraphs. At cv 0.017 the scale totals win and
the interior holds its scale; at cv ≥ 0.05 the zero anchors win, the interior deflates, and hard
mass balances propagate the deflation into the observed checkpoints — which is exactly the
homogeneous collapse the rows exist to stop, now paid for at the observed rows' tight σ.

## Consequences / follow-ups

- The 3.0M z² phantom cost and the hierarchy inversion are **real but unfixable by uniform σ
  loosening**; any fix must separate scale preservation from information weighting. Candidate:
  propagate σ from the imputed edges' own σ into the total (`sigma_ck_mode="p_mean"` exists in
  the builder), possibly combined with rebalancing the zero-anchor strength — a next-round design
  question, to be decided on the post-Round-A baseline.
- The classification itself (fabricated vs feed-backed per group) is validated and cheap; it is
  reusable by any future policy and by diagnostics.
- Reverting to the experiment is one constant (`FABRICATED_ANCHOR_CV`).
