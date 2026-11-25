# One EdgeRelation representation shared across Stage-2 imputation and Stage-3 constraints

**Status:** accepted — 2026-09-22 (amended: sequential single-unknown fixpoint, not a MAP solve;
writer scope = permission, cascade order = precedence, scopes may overlap; N-product recipes close
by sum with one unknown; scale totals and hard bounds are emit-only relations; see
[issue #26](https://github.com/florihu/pySUTO/issues/26) §2 and the `Relation` / `Writer scope`
entries in [GLOSSARY.md](../../GLOSSARY.md)); built 2026-11-12 through #128–#136, consequences as
built below, copper's imp arrays and log unchanged (checked by #132 and #138)

Stage-2 imputation and Stage-3 constraint-building are **two independent enumerations of the same
edge-relation network**, and the duplication is a recurring source of drift:

- **Node balance** `Σin − Σout = 0` is enumerated by `MassBalanceBuilder.build_static`
  ([src/pysuto/constraints/mass_balance.py](../../src/pysuto/constraints/mass_balance.py), both
  valuations, emits a `G` row) **and** by `NetworkImputer._do_flow_balance`
  ([src/pysuto/impute/network.py](../../src/pysuto/impute/network.py), CC, solves a single unknown).
- **eff/conc ratio** `a − r·a = …` is enumerated by `RatioConstraintBuilder.from_csv`
  ([src/pysuto/constraints/ratio.py](../../src/pysuto/constraints/ratio.py)) **and** by
  `ParametricImputer.from_csv` (`src/pysuto/impute/parametric.py`, deleted in #136)
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

### As built (#128–#136, 2026-11-12)

- **One params reader, one source.** `pysuto.relation.params.read_params` is the only reader of
  a params CSV (#128); `ParamSource` resolves its per-pair or twin rows into `Relation`s
  (`pysuto.relation.sources`), the object Stage 3's ratio builders already shared.
- **A fill is a source, a rule and a scope.** `Fill(source, rule, scope, id)` and
  `FillImputer` replace the per-family imputers: the trade seed (`forward`, the cross-region
  `Flow→Process` content scope, σ rule `trade`) and the conc and ore-grade fills (`both`, σ rule
  `parametric`) are three `Fill`s of the params relation source. `TradeImputer`, `FormulaImputer`
  and `ParametricImputer` are deleted; copper's cascade no longer holds the `eff` fill, which
  wrote 0 edges in every reference run — the network's pooled recipe owns the `eff` rows.
- **The network solves Relations.** `NetworkImputer` builds its Flow and Process balances, conc
  twins and pooled-`eff` relations from `pysuto.relation.balance` and `ratio_relation`, the
  constructors `MassBalanceBuilder` and `RatioConstraintBuilder` use (#134); a recipe with one
  unknown product fills it forward, with none it derives the inputs backward, with two or more it
  fills nothing (ADR 0004 option E stays rejected, #130).
- **The case declares the cascade as data.** `ImpSection.cascade` lists `FillEntry` entries
  (a params family, a reading, a rule, a σ rule, a scope function) and `CascadeStep`s (the
  network); the knob `CASCADE_ORDER` orders them and the imp sidecar records each entry's source,
  rule and scope (#136). A case's terminal fillers (`TimeImputer`, `FlatPriorImputer`, copper's `ShareImputer`)
  stay imputers behind `ImpSection.terminal`: temporal interpolation and the last-resort fallback
  are not linear edge relations.
- **Not yet shared:** `_do_recipe`'s arithmetic still runs beside the pooled-`eff` `Relation`
  (#134), and the fill's σ rules and the network's differ; one σ rule per `Relation` and one
  engine-level fixpoint change copper's numbers and are their own measured tickets (#139, #140).
