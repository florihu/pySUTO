# Process-recipe traversal imputer for the production chain

Stage-2 imputation is a cascade of **per-edge** fillers (`ParametricImputer`,
`FormulaImputer`, `TimeImputer`, `FlatPriorImputer`); the unit of reasoning is an *edge*, not
a *node*. In the reference run [`imp_20260608_080016_0563ae84`](../../data/proc/imp/) **80.3 %
of edges (686 k) end as `flat_prior_noisy_zero`** — the smart imputers fill ~9 % and the rest
is given up on. The **SEM residual outputs** (Slag / Tailings / WR / PSC) have zero feed
[`Coverage`](../../GLOSSARY.md) and can *only* be reached by mass closure, so today they are
noisy zeros (no [`val`](../../GLOSSARY.md)). The conc-after-eff Mass-gap is the same
per-edge brittleness.

We add a **`NetworkImputer`** ([`src/pysuto/impute/network.py`](../../src/pysuto/impute/network.py))
that treats each **Process node as a mass-closing [recipe](../../GLOSSARY.md)**: it propagates
copper content (`CC`) through recovery efficiencies (`product = eff·Σ(primary inputs)`, `waste =
(1−eff)·Σ(primary inputs)`, Scrap pooled in), balances Flow nodes (`Σ supply-in = Σ uses-out`,
which joins one process's product to the next process's input and resolves the `Ore` branch
split MF-pyro vs SXEW-hydro from whichever side the data pins), and derives `Mass` twins via
`conc` in the same step. It runs **first** in the existing cascade and is the **sole writer of
process-incident, same-region** chain edges; cross-region traded imports and the
`Cathode → USC/MESC` stock changes are read as boundary terms but never written, and a same-region
`Cathode → FU` leg is written only when no feed covers it. *(Amended 2026-10-14, #76: this
sentence said final demand is never written, stale since
[ADR 0040](0040-stock-change-legs-are-held-at-their-prior.md) let the flow balance close the FU
residual; a fan's observed zero is covered, so it is never written.)* eff/conc are resolved
with the same `ParamRelationResolver` the constraint builders use. It is a new *imputer*, not a
new *engine* — it reuses the engine cascade/fixpoint, the Lp/σ machinery, `_reg_sensitivity`,
the `ImputationLog`, and the bounds-clip.

## Considered options

- **A. New `NetworkImputer` in the existing engine; sole writer of chain edges. (chosen.)**
  One ledger (`agg_result`), one σ/provenance path, one coefficient source shared with
  Stage 3. The race that produced the conc-after-eff bug is removed structurally: on its
  territory the network is the only writer and it emits CC + Mass + waste atomically.
- **B. A dedicated `ProcessNetwork` engine.** Cleaner conceptual model, but a *second* ledger
  writing into `agg_result` and re-derived σ/provenance/determinacy — a new place for the conc
  value to diverge from Stage 3. Rejected: more inconsistency surface, not less.
- **C. Keep the per-edge model, author more `FormulaRelation`s + reorder the cascade.** Lowest
  effort, but stays per-edge: it cannot traverse the network (a single upstream gap blocks
  everything below) and the SEM residual outputs stay unreachable.
- **D. Dual CC + Mass closure at every node.** Rejected: `conc` point-values make Mass
  infeasible through a concentrating process (at MF, mass-in ≠ mass-out by construction), and
  it duplicates the solver's hard mass-balance. The network closes **CC** (Cu conserved by
  `eff`) and leaves Mass mass-balance to Stage 4.
- **E. Fabricate free splits (equal / param split now).** Rejected for the `Ore` branch and the
  Concentrate-vs-Scrap split: an honest flat-prior gap the solver can widen beats an invented
  number. The network fills only what the recipe + known neighbours pin uniquely; a missing
  **SEM secondary** (Scrap) is treated as absent (0) so the primary chain still flows.

## Consequences

- **Generalises [ADR 0002](0002-single-endpoint-rows-fan-out-not-replicate.md) option D.** That
  decision made `conc_*_supply` hold *by construction* for the four traded edges; resolving
  conc with the same `ParamRelationResolver` here extends "conc holds by construction" to the
  whole chain — domestic legs, intermediates, and SEM waste.
- **Serves [ADR 0001](0001-replace-coherence-sum-eq-with-bilateral-anchoring.md) and delivers
  its follow-up A2.** The network propagates the pinned demand-side scale upstream
  Cathode → … → Mining, and replaces the `flat_prior` noisy-zero seed with derived non-zero
  values on chain + SEM edges (A2: "non-zero seed for FlatPrior edges").
- **A1 caveat (not fixed here).** The network's genuinely tight σ on high-recovery steps (ER
  `eff = 0.997`) is floored by the Stage-3 `σ_a` relative-CV floor (`SIGMA_A_CV_MIN = 0.001`,
  [`constraints/engine.py:_lp_regularize`](../../src/pysuto/constraints/engine.py)) until A1
  lands. Not a blocker; the constraint stage is intentionally untouched by this work.
- **`eff` ratio scope (follow-up — RESOLVED by [ADR 0005](0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md)).**
  As first shipped, the `eff` *ratio constraint* related the product to the single `eff`-named
  primary only; the pooled recipe (`product = eff·(primary + scrap)`) made the imputed prior exceed
  it by `eff·Scrap` for Smelting / CFR, which the soft eff row absorbed. ADR 0005 pools Scrap into
  the `eff` ratio constraint (schema-driven `agg` rows with `region_to` bucketing), so prior and
  constraint now agree.
- **Provenance shift.** Process-chain edges are now attributed to `network_*` methods; the
  eff/conc `ParametricImputer`s remain as a fallback on whatever the network leaves
  undetermined (and on non-chain edges).
- **Flow-node balance is keyed per `(node, year)` (defect fix, 2026-06-09).** The Decision above
  describes the Flow balance `Σ supply-in = Σ uses-out` as joining one process's product to the
  next process's input. As first shipped, [`_build_flow_balances`](../../src/pysuto/impute/network.py)
  grouped a Flow node's incident edges on the **time-agnostic node id** (`node_index` has no time
  axis), pooling all 30 years onto one node; with that many unknown legs the single-unknown
  determinacy guard never fired, so `network_balance` ran **0×** — the supply→use join was inert,
  the `CP` pool (reachable only through it) never closed (`CSXEW→CP`, `CER→CP`, `CP→Cathode`,
  `CP→PSC` left `flat_prior_noisy_zero`), and the `Ore` split was resolved only via recipe-backward
  from observed products. Keying the balance per `(node_id, time)` realises the documented join —
  the CP pool now closes via balance → recipe. Because recipes still run **before** balances in the
  sweep, the change is additive (it fills only what recipes leave) and introduces no new node
  conflict. A two-year regression test (`tests/test_network_imputer.py`) guards it, since a
  single-year fixture cannot exhibit the pooling.
- **`acc` accessor added.** `AccountingStructure.process_recipes` exposes the structural recipe
  skeletons (derived from `edge_meta` + supply/use masks + `entity_sub`, which survive
  `load()` — `ruleset` is `None` there).
- **In-imputer bounds-clip (2026-06-10).** `_w` clips every fill to the edge's `[lower, upper]`
  bounds *before* it propagates through the internal fixpoint, σ-widened to the clip gap and
  tagged `network_*_clipped`. Without it, a negative residual at the over-determined `Ore` split
  (`Ore→SXEW = Mining→Ore − Ore→MF`, negative in regions with little/no SXEW when feed priors
  disagree) propagated un-clipped down the SXEW branch and **dragged downstream sums below their
  true values** — `CP→Cathode = CER − |neg|` → engine-clipped to 0, losing the real CER (pyro)
  contribution. The engine's post-return clip can't prevent this (it runs after the fixpoint), so
  the floor must be applied inside. Chose **saturated-zero over NaN**: NaN would block the
  downstream recipe (missing primary input), pushing the whole SXEW branch to `flat_prior_noisy_zero`
  (worse `val_pct`) and discarding the structural fact that the branch is ~0; for the dominant
  no-SXEW case 0 is the true value. The widened σ keeps the zero honest (a floor, not a precise
  measurement). See [GLOSSARY.md](../../GLOSSARY.md) §"Bounds-clip (in-imputer)".
- **Validation.** Re-run the imputation driver; success = **val-coverage (`val_pct`) of the
  `SEM` column in `coverage_system_boundary_<year>.csv` and of chain commodities in
  `coverage_supply_use_<year>.csv` rises**, with `flat_prior_noisy_zero` share well below 80 %.
