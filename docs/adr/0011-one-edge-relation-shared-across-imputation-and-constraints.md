# One EdgeRelation representation shared across Stage-2 imputation and Stage-3 constraints

**Status:** proposed

Stage-2 imputation and Stage-3 constraint-building are **two independent enumerations of the same
edge-relation network**, and the duplication is a recurring source of drift:

- **Node balance** `Σin − Σout = 0` is enumerated by `MassBalanceBuilder.build_static`
  ([src/pysuto/constraints/mass_balance.py](../../src/pysuto/constraints/mass_balance.py), both
  valuations, emits a `G` row) **and** by `NetworkImputer._do_flow_balance`
  ([src/pysuto/impute/network.py](../../src/pysuto/impute/network.py), CC, solves a single unknown).
- **eff/conc ratio** `a − r·a = …` is enumerated by `RatioConstraintBuilder.from_csv`
  ([src/pysuto/constraints/ratio.py](../../src/pysuto/constraints/ratio.py)) **and** by
  `ParametricImputer.from_csv` ([src/pysuto/impute/parametric.py](../../src/pysuto/impute/parametric.py))
  — from the *same* params CSV, via the *same* `ParamRelationResolver`.
- **pooled-eff recipe** `Σb = r·Σa` is the `RatioConstraintBuilder` `agg` bucket **and**
  `NetworkImputer._do_recipe`.
- The `agg` true-set literal is copy-pasted (`parametric.py` ↔ `ratio.py`), and
  [ADR 0005](0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md) records a *manual*
  re-alignment of the eff pooling between the two sides — evidence the drift is a real maintenance cost.

This makes the imputation step the conceptually weakest stage: it silently re-derives the Stage-3
relation network in an ad-hoc, scattered way (recipe traversal + flow-balance + hand-written
multi-source formulas), instead of being "the same relations, solved cheaply for a prior."

## Decision

*(Deferred — this ADR records the target, not a committed implementation. It is the sanctioned
follow-up to [ADR 0010](0010-trade-first-seeding-of-bilateral-flow-process-inputs.md), which deepened
the NetworkImputer and removed the redundant chain `*_domestic` formulas as a first step.)*

Introduce a single **`EdgeRelation`** representation — a sparse linear row over edges plus metadata
(valuation, ε, σ recipe, ownership/priority, the `agg` relation-type marker). `ConstraintBlock.G_rows`
([src/pysuto/constraints/base.py](../../src/pysuto/constraints/base.py)) is already exactly this shape,
so an `EdgeRelation` is the natural common form. One declaration then feeds **both** consumers:

- a **solve-for-prior** adapter (Stage 2) that sequentially fills single unknowns to a fixpoint, and
- an **emit-constraint-row** adapter (Stage 3).

Relation *sources* — structural (recipes, flow balances derived from `acc`), param-CSV (conc/eff via the
existing resolver), and any remaining hand-written multi-source — all produce `EdgeRelation`s behind one
seam. `TimeImputer` and `FlatPriorImputer` stay separate (temporal interpolation and last-resort fallback
do not fit the linear-edge-relation mould).

## Considered options

- **Status quo (two enumerations).** Rejected long-term: the duplication above keeps drifting (ADR 0005
  was a manual re-sync); the prior and the constraint that checks it can diverge silently.
- **Share only the resolver (today's partial state).** The `ParamRelationResolver` + params CSV are already
  shared for *pairwise* relations, but node-balance and recipe relations are still enumerated independently
  in `mass_balance.py` / `network.py`. Insufficient — the deepest duplication is structural, not pairwise.
- **Imputation = a full unconstrained/MAP solve over `G`.** Conceptually cleanest endpoint, but largest
  change and risks coupling the prior to the very constraints the solver enforces; out of scope here.

## Consequences

- **Prior↔constraint consistency by construction** — the warm-start prior is computed from the relations
  the solver later enforces, so they cannot drift (no more ADR-0005-style manual re-alignments).
- **One σ model, one ownership rule, one fixpoint** — change/verify in one place; the `agg` flag and
  ownership become relation attributes rather than cross-module coordination.
- **Large blast radius** — touches `impute/` and `constraints/` (or a shared `relation/` module both import);
  both stages must be re-verified. Hence deferred and staged after ADR 0010, not bundled with the er-cer fix.
- Cf. the long-standing "express all SUT edge relations as one unit" idea (memory `edge-relation-unification-idea`).
