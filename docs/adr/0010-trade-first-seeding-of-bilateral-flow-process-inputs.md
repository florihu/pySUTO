# Trade-first seeding of bilateral Flow→Process inputs + non-blocking import tier

**Status:** proposed

The NetworkImputer's process-recipe forward fill (`product = eff·Σinputs`,
[network.py:365](../../src/pysuto/impute/network.py)) fires only when **every** primary input is
known (`_RRecipe.primary` "block forward if missing", network.py:120-121). A Process is fed by
**bilateral** Flow→Process imports from up to 47 source regions, but observed trade coverage is
sparse (e.g. Anode→ER at 4.3 %; feeds icsg/crude_ore/baci, 7.8 % overall). So ≥1 primary input is
NaN for essentially every region-year, the gate never opens, and the `agg=true` process products
(Smelting→Matte via `effsmelting`, CFR→Anode via `effcfr`) fall to `flat_prior_noisy_zero`
(f = 0, σ = 1e6). Because those products are `agg=true`, the ParametricImputer skips them
(`_parametric_consumes`, [parametric.py:48](../../src/pysuto/impute/parametric.py)) — they are
NetworkImputer-only — and the `*_domestic` FormulaImputer cascade is starved too (e.g.
`anode_domestic` needs `CFR→Anode (CC)`, which is 0). The downstream ER→CER→CP chain therefore
gets a ~0 prior; the solver rebuilds it from hard mass-balance + soft `eff` constraints (correct
solved values, but a 3–4 order "rocket" from the prior). This **defeats the warm-start prior chain**
([ADR 0009](0009-warm-start-prior-chain.md)) — there is nothing meaningful to carry forward.

Diagnosis: [analysis_opt_20260611_140957_24eeecb0_20260615_er-cer-prior-not-propagated.md](../rep/analysis_opt_20260611_140957_24eeecb0_20260615_er-cer-prior-not-propagated.md).

## Decision

Make the network recipe close from observed trade + domestic supply instead of stalling on
unobserved bilateral routes. Two parts, both required, applied generically (not hard-wired to one
sector):

- **Non-blocking `bilateral_import` input tier (load-bearing).** Add a third tier to `_RRecipe`
  for cross-region Flow→Process legs (classified in `prepare()` via `from_region != to_region`).
  In `_do_recipe`, treat it like the existing SEM `secondary` tier: pool when known, **count as 0
  when missing, and exclude from `miss_PR`** so it no longer blocks the forward gate. Domestic
  (X==Y) legs stay blocking `primary`. A bilateral import leg with no trade observation means "no
  trade on this route" — 0 is the honest value, the same trade-off
  [ADR 0004](0004-process-recipe-traversal-imputer.md) made for SEM secondaries.
- **`TradeImputer` (supporting).** A new `BaseImputer` (`src/pysuto/impute/trade.py`) registered
  first in the cascade (before `network`). For each cross-region Flow→Process **CC** edge whose
  BACI **Mass** twin is known, write `f_CC = conc·f_Mass` (method `"trade_seed"`), reusing a shared
  `conc_twin_map()` factored out of `NetworkImputer._load_twin_map`. It reads the Mass twin already
  present in the aggregated `f` — no new artifact, no feed access — and is a no-op when conc/BACI are
  absent.

Net effect: the network forward-fills the `agg=true` products from domestic + covered-import inputs
→ CFR→Anode/Smelting→Matte revive → the Flow-node balance fills the domestic Anode→ER leg →
`anode_domestic`/`effer` propagate ER→CER → CER→CP. The warm-start chain then carries a real prior.

## Considered options

- **Extend the ICSG trade-share fan ([ADR 0002](0002-single-endpoint-rows-fan-out-not-replicate.md)
  option D) to seed Anode→ER.** Rejected: `Anode` is in that fan's traded-sector list but ICSG
  carries **zero anode usage** (its Use rows are Concentrate/Scrap→Smelting, Matte/Scrap→Refining,
  Cathode→semis), so the fan is inert for Anode; it also runs at feed time, not imputation time.
- **Blanket gate relaxation (treat *all* missing primaries as 0).** Rejected: would zero genuine
  domestic legs and break processes with legitimately-missing same-region inputs (Smelting, MF).
  Only cross-region import legs are zeroed.
- **Trade-seeding alone, without the gate change.** Rejected as insufficient: the dominant
  *domestic* Anode→ER leg has no measurement anywhere (BACI is international; ICSG no anode usage),
  so seeding only covers import legs — the gate must let the network close the node from the
  upstream product.
- **A pooled-eff parametric / formula for the products.** Rejected (Simplicity-First): the products
  are inherently pooled over bilateral imports (`eff·Σinputs`), which a pairwise relation cannot sum
  — exactly why they are `agg=true`, NetworkImputer-owned today.

## Consequences

- **Imputed priors change** for every deep-chain `agg=true` product and the segments below them
  (Smelting→Matte, CFR→Anode, Anode→ER, ER→CER, CER→CP); the warm-start prior chain becomes
  effective on the refining chain. Existing imp/const/opt runs must be re-run to benefit.
- Solved values are largely unchanged (the solver already recovered them via constraints); the gain
  is a meaningful prior, faster/steadier convergence, and a non-degenerate warm-start carry.
- `_RRecipe` gains a third input tier; [ADR 0005](0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md)'s
  pooled-eff recipe now also pools the seeded import legs (the soft `eff` row continues to absorb any
  residual).
- Regions where the upstream product is also fully uncovered remain flat-prior — strictly better
  than today, and improved by the time-carry once any year closes.
- New `TradeImputer` + shared `conc_twin_map()` helper; document in CONTEXT.md (trade-seed vocab)
  and README §6 (test map). Verification per the report §8.4.
