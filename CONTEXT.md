# CONTEXT — shared language for pySUTO

This file is the project's **glossary**: the precise meanings of the terms
we use when talking about the pipeline, the model, and the constraints.
It is not a spec; implementation details live in code and in `README.md`.
Add a term here only when its meaning has been the subject of a genuine
clarification.

## The flow variable: `a`, `a₀`, `a*`, `a_agg`

* **Flow `a`** — element of the decision-variable vector in the Stage 3
  QP. One per **structural edge** (see below). Replaces the legacy
  notation `f` / `F` project-wide.
* **Prior `a₀`** — the initial estimate that the QP's anchor term
  `(a − a₀)ᵀ Σ_a⁻¹ (a − a₀)` pulls toward. Under the structural model
  `a₀` is a 1-D structural seed (`ConstraintRun.a0`); the `Prior` adapter
  (see *Prior seam*) decides which value enters the objective at each
  period.
* **Optimum `a*`** — output of the solver at one period. `OptimResult`
  carries it as `a_star`; the legacy attribute name `F_opt` is preserved
  as a read-only alias for older callers.
* **`a_agg`** — Stage 2 aggregation output. 2-D
  `(n_edges, n_timesteps)` — per-cell inverse-variance combination of the
  selected feeds, then imputed forward by the registered imputers.
* **`a_target` / `a_root`** — Stage-1 source / root flow vectors.
  Per-snapshot 1-D per feed. The collector lifts them onto the
  `(n_edges, n_timesteps)` grid.

## Time terms

* **Period** — one index along the `n_timesteps` axis. Labelled by an
  entry in `acc.time_range` (e.g. `"1995"`); identified internally by
  `t_idx`. The canonical sweep variable across the pipeline.
* **`acc.time_range`** — the ordered list of period labels. Drives the
  second axis of every per-period array. No longer carried on
  `edge_index`.
* **Period selector (`a_time` / `b_time`)** — the value of the
  `a_time` / `b_time` CSV column on a params row. `*` (wildcard) →
  broadcast to every period; a literal year (`"1997"`) → that period
  only; `"same"` on the B-side → mirror A's resolved periods.
  Consumed by both Stage-2 imputers and Stage-3 builders via
  `param_relation.resolve_period_selector`.
* **Warm-start** (active mechanism, not the legacy noun): the carry
  `a*(t-1) → a₀(t)` enforced by `WarmStartPrior`. Because edges are now
  structurally shared, this carry is meaningful — pre-refactor it was a
  no-op because the previous year's optimum lived on a different (time-
  replicated) edge identity.

## Structural model

* **Structural edge** — one row of `acc.edge_index`. Identified by
  `(from_id, to_id, valuation)`; the time dimension is **not** part of
  the edge identity. `acc.n_edges` counts structural edges.
* **`StructG`** — the constraint operator. Shared CSR sparsity
  (`indptr` + `indices`) + per-period data tape
  `data: (nnz, n_timesteps)` + `row_meta` (no `time` column) + `shape =
  (n_constraints, n_edges)`. `StructG.at(t_idx) → csr_matrix` is a
  zero-copy view; the indptr / indices arrays are shared with the
  operator.
* **`row_meta`** — DataFrame with one row per constraint. Columns:
  `constraint_id`, `type`, `is_hard`, `cone_type`, plus topology
  (`valuation`, `sector`, `region`, `entity`). No `time` column —
  per-period applicability lives on `active`.
* **Sparsity invariant** — `build_dynamic(t)` MUST preserve
  `build_static`'s nnz pattern; the engine asserts equality of
  `(indptr, indices, nnz)` per builder. Per-period coefficients (e.g.
  ratio `r(t)`) overwrite values inside the existing nnz slots — they
  never introduce new ones.

## Row activation

* **Active row** — the row participates in the per-period QP. Boolean
  `ConstraintRun.active[i, t]` is `True`.
* **Inactive row** — the row is dropped from the per-period QP entirely.
  Free rows: contribute nothing to the objective. Hard rows: relaxed
  (not enforced) at that period.
* **`active` mask** — first-class `(n_constraints, n_timesteps)` boolean
  on `ConstraintRun`. Replaces the legacy `SIGMA_C_INACTIVE` (`1e10`) /
  `BOUND_C_INACTIVE` (`1e15`) sentinel floats. The sentinel constants
  still exist in `pysuto.constraints.sum` for import-time
  backwards compatibility but are not written by any builder under the
  structural model.

## Prior seam

The Stage-3b objective takes a prior mean at each period. Which value
is the right one is a *policy* choice; pySUTO exposes two implementations:

* **`AggregationPrior(a_agg)`** — `a₀(t) = a_agg[:, t]`. No temporal
  smoothing. The validation / regression baseline against the
  pre-refactor pipeline.
* **`WarmStartPrior(a0)`** — `a₀(0) = a0`; `a₀(t > 0) = a*(t-1)`. The
  carry makes the previous year's optimum visible at the next year. This
  is a model change vs. pre-refactor — `t > 1995` results legitimately
  differ — and is the production path under the structural model.

Both implement the `Prior` protocol so the solve loop is policy-agnostic.

## Constraint typology — by what `c` represents

When we say a soft sum-style row is active at a period, we mean the slack
term contributes to the QP objective. For active rows we further
distinguish them by **what the right-hand side `c` represents** — this is
the key axis for reasoning about whether a row pins scale, propagates the
prior, or sits at zero:

* **Anchoring constraint** — `c` comes from a **reliable external
  source** (a published total, an authoritative aggregate).
  Non-homogeneous (`c ≠ 0`). Adds new information beyond `a₀` and
  structurally opposes collapse, because `a = 0` is infeasible against a
  positive `c` with tight `σ_c`. This is the RAS "marginal totals" role.
* **Coherence constraint** — `c` is **derived from `a₀` at solve-time**
  (e.g. `c = Σ a₀(rhs_ids)`). Non-homogeneous in value, but adds no
  information beyond the prior; it just keeps two linked sums in
  agreement with what the prior says. Force against collapse is bounded
  by `1/CV(σ_c)²`, same magnitude-blind ceiling that limits the prior
  penalty. The sub-entity `sum_eq` rows produced by
  `SumConstraintBuilder.from_acc_struct_subentity` are of this type when
  `sigma_mode="p_mean"`.
* **Homogeneous constraint** — `c = 0` for genuine reasons (a topology
  identity, e.g. `Σ supply − Σ use = 0`, or a ratio equality). `a = 0`
  satisfies these trivially, so they contribute no collapse deterrent and
  can bias the solver toward zero whenever their slack is cheap.

The same row family can be coherence or homogeneous depending on
`sigma_mode` and on whether `_rhs_ids` resolves at solve-time.

## Boundary topology

* **System boundary (SB) node** — a node on the open boundary of the
  modelled system. Imports, exports, final use, and stock changes
  typically terminate at or originate from SB nodes. **SB nodes carry no
  `mass_balance` row** — they absorb or produce whatever the network
  requires. Consequence: a hard equality constraint on a flow whose other
  end is an SB node cannot, by itself, create infeasibility, because the
  SB end balances freely. Infeasibility from such constraints comes only
  from conflicts with *other* hard constraints (ratio bounds, closed-
  sub-network mass balance), not from disagreement with `a₀`.
* **Closed sub-network** — the portion of the network bounded by
  `mass_balance` rows (Flow / Process nodes). Constraints that traverse a
  closed sub-network must be feasibility-consistent with `mass_balance`
  and `ratio_bound` rows there.

## Scale-preservation vocabulary

When discussing why flows collapse:

* **Collapse-penalty ceiling** — the maximum cost of driving an edge to
  zero, equal to `(a₀/σ_a)² = (1/CV_a)²`. For pySUTO's `cv_min = 0.05`
  floor this caps at ~400, independent of `a₀`'s magnitude.
* **Scale-blind weighting** — any weighting where the per-element cost of
  collapse is independent of the element's magnitude. The relative-CV
  floor produces this.
* **Magnitude-aware anchor** — `σ_a` set from genuine reliability, so
  high-magnitude high-confidence edges have correspondingly large
  `(a₀/σ_a)²`. The CRAS / KRAS approach.
* **Scale-pinning** — any mechanism (anchoring constraint,
  magnitude-aware σ_a, hard total) that makes `a = 0` either infeasible
  or expensive in absolute terms.

## Retired terms

These were load-bearing in the pre-refactor architecture and are gone
under the structural model. They are listed here so that legacy reports
and code archeology stay readable.

* **Period slice / per-`t` partition** — the operation
  `_partition_G_by_time` that turned the monolithic `G` into one block
  per period. Now structurally impossible (StructG has no time axis on
  the column side); replaced by `StructG.at(t_idx)`.
* **`G_blocks` / `edge_cols_by_t`** — the dict-keyed-by-time per-period
  blocks and their global column indices. Removed.
* **`timestep_data[t] = {"c": ..., "sigma_c": ...}`** — the per-period
  dict on `ConstraintRun`. Replaced by 2-D `ConstraintRun.c` and
  `ConstraintRun.sigma_c`.
* **Cross-year placeholder** — the artifact whereby a per-period block
  solve filled `F_opt` outside its own period columns with prior values;
  the source of REP §8.2 "year-local persistence" warning. The new
  per-period output is structurally year-local because StructG slices
  cleanly.
* **`SIGMA_C_INACTIVE` / `BOUND_C_INACTIVE`** — sentinel floats that
  marked deactivated rows. Replaced by the first-class
  `ConstraintRun.active` boolean.
* **Block path / legacy path** — the two solver code paths
  (`_solve_all_via_blocks` and `_solve_all_legacy`). Collapsed into one
  loop driven by `StructG.at(t)` + `active[:, t]`.
* **No cross-time coupling** — the runtime invariant verified per run
  pre-refactor. Now structural (StructG's column axis is the structural
  edge index; nothing can couple across periods through `G`).
