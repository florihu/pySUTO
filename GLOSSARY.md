# GLOSSARY — pySUTO glossary

The project's single glossary. Glossary only — no implementation details.

Pick the bold term; the listed `_Avoid_` aliases are ambiguous and should not be used.
Detail that does not fit an entry is in [docs/architecture.md](docs/architecture.md), under each stage's
"Glossary detail" and in its framework section.
Definitions are code-verified against [acc.py](src/pysuto/acc.py),
[conc.py](src/pysuto/conc.py), [relation/](src/pysuto/relation/), and
[constraints/](src/pysuto/constraints/).

## Domain vocabulary

**Node**:
A `(sector, entity, region)` triple — one point in the accounting structure.

**Edge** ≡ **flow**:
"Edge" and "flow" are **synonyms** throughout pySUTO — one atomic directed quantity between two nodes,
addressed by the 8-tuple
`(time, valuation, region_from, entity_from, sector_from, region_to, entity_to, sector_to)`. Its value
`a` is the magnitude in a single valuation. _Note:_ lowercase "flow" (= edge) is distinct from the
capitalised **`Flow` entity** (a commodity node — see Entity); disambiguate by capitalisation.

**Entity** (`entity_from` / `entity_to`):
The structural *kind* of a node — a class, one of `Process | Flow | SB`. `Process` + `Flow`
are the industrial system; `SB` is the system boundary. (The `Flow` entity is a
commodity/product node — not to be confused with "flow" meaning an edge.)
_Avoid_: "node type" (collides with the constraint-row `type` column).

**Sector** (`sector_from` / `sector_to`):
The named *instance* within an entity kind — the copper-chain classification (`Ore`, `Matte`,
`Cathode`, `SR`, `FU`, …). "Cathode" is a sector; its entity says whether that node is a
`Process` or a `Flow`.

**Region**:
The geography of a node. An edge is **domestic** (same region both ends) unless it is a
tradable use edge.

**Valuation**:
The measurement basis of a flow. In the copper model: `Mass` (gross tonnage) and `CC` (copper
content), both in kt; the framework also admits money / energy bases. A mass-balance row is
**per-valuation** (Mass balances with Mass, CC with CC). Bases are bridged by intensive-
property relations: `CC = concentration × Mass`.
_Avoid_: "price basis" (valuation is not monetary-only here).

**Supply–Use Table (SUT)** / industrial system:
The `Process` + `Flow` nodes, recording per (region, time, valuation) which processes
*produce* which flows (supply) and which *consume* which flows (use). Supply = Use is enforced
per node by hard mass balance.

**Supply edge**:
A `(sector_from, sector_to)` pair in `supply_pairs`; structurally **Process → Flow** (a
process produces a commodity) or **Process → SB** (a process books a residual on a
satellite, e.g. Mining → WR).

**Use edge**:
A pair in `use_pairs`; structurally **Flow → Process**, **Flow → final-demand category**,
and **SB → Process** (a satellite stock or scrap input, e.g. Deposit → Mining, Scrap →
Smelting). The `tradable` mask is what permits a use edge to cross
regions.

**SB (system boundary)**:
A user-defined boundary entity, **excluded from mass balance** (an open source/sink).
An **SB-input edge** (`entity_from == "SB"`) feeds the system, e.g. crude ore from nature → a
process. An **SB-output edge** (`entity_to == "SB"`) lets the system deliver to the boundary,
e.g. cathode → stock addition or out to another system. The built-in `sb` mask flags
SB-*output* edges only.

**Tradable**:
A mask marking which edges may be **traded across regions**; every other edge is **domestic**
(same-region). Only meaningful on use edges.

**Final-demand category**:
A destination sector where flows terminate: `FU` (Final Use), `MESC` (Metal Exchange Stock
Change), `USC` (User Stock Change).

**End-use category (IWCC)**:
A demand segment, one of 16 in six groups, to which IWCC assigns the copper content of
semi-finished products, e.g. Power utility or Automotive electrical; it includes direct-melt
scrap and lies outside the model, which does not reconcile it. A different quantity from the
`FU` (Final Use) **Final-demand category**, the cathode leaving the modelled system.
_Avoid_: "final use" when the IWCC categories are meant.

**Root** (root edge / root accounting structure):
The canonical internal **data model** of this system — the common index every external feed is
fitted onto. **Source / src edge** = an edge in a feed's own classification, *before*
concordance maps it onto root via `W` (`f_root = Wᵀ f_src`).
_Avoid_: "common structure" (ambiguous about which side is canonical).

**Control total** *(removed — superseded by **Scale total**, see [ADR 0006](docs/adr/0006-scale-total-supersedes-external-control-total.md))*:
A Type-2 **soft-sum** constraint (`Σ aᵢ = c + ε`, free cone) whose RHS `c(t)` is an externally published
total — here ICSG production + usage, per year — that a defined edge-set must sum to. Internal edge
consistency is *not* a control total — that is the job of hard mass balance. _Avoid_: "anchor" for an
external total (the live term **Anchor** below is the model's own prior), "coherence soft-sum" (removed
mechanism).

**Data-category typology** (Pauliuk et al. 2019):
The four parameter kinds, split **extensive** vs **intensive**. *Extensive* (kt magnitudes):
**flows** (individual edges) and **totals of flows** (aggregates, e.g. scale totals).
*Intensive* (per-unit ratios relating flows): **concentration** (category 3) and **efficiency**
(category 4) — these two are the pair readings of the params **relation source**.

**Prior `a₀`**:
The best-estimate flow vector the QP stays close to: the year's conditioned `f_agg` slice
(`prior_mode="cold"`, the driver default) or, under `"warm"`, the previous year's optimum.

**Prior σ (`σ_a`)**:
The standard deviation of an edge's prior: how far the QP may move the edge from `a₀` at unit cost. A
**flat** σ states that the edge carries no information of its own. _Avoid_: "σ init", "initial σ" —
nothing iterates it.

## Pipeline & process vocabulary

The stage verbs and the cross-cutting machinery that runs them.

**Datafeed** (feed):
One external data source. Stage 1 runs once per datafeed, producing a weight matrix `W`, a
root flow vector `F_root`, and `σ_root` into the `DataFeedRegistry`. The `DataFeed` class.

**Concordance**:
The classification mapping that fits a feed's **source (src)** edges onto **root** edges,
encoded as the sparse weight matrix `W` (`f_root = Wᵀ f_src`).

**Fan-in / fan-out**:
The two concordance shapes. *Fan-in* = many source edges combine into one root edge — the σ
**diversification invariant** `CV_root ≤ CV_src` lives here. *Fan-out* = one source edge is distributed
across several root edges (a mapping that classifies only one endpoint). These are the **source → root**
(Mapping 2) shapes; the upstream **raw → source** projection (Mapping 1, where raw-row *aggregation* and
*replication* live) is a separate level — see *"Two mapping levels"* (Concordance vocabulary, below).

**Aggregation**:
Stage 2 — combining multiple feeds' observations of the *same* root edge by inverse-variance
weights `Ω` into a single `f_agg` / `σ_agg`.

**Imputation**:
Stage 2 — filling root edges that **no feed covered**, via an engine of imputers run in order;
every fill is recorded in the `ImputationLog`.

**Imputer**:
One fill method, a `BaseImputer` subclass: a `FillImputer` (one **Fill**), the `NetworkImputer`,
or a terminal `Time`, `Share` or `FlatPrior` imputer. A case declares the first two as its cascade
entries and the last three as its terminal fillers. `Trade`, `Parametric` and `Formula` are
retired as classes.

**Constraint building**:
Stage 3 — compiling `f_agg` / `σ_agg` into the sparse constraint matrix `G` plus per-timestep
right-hand side `c(t)` and `σ_c(t)`.

**Builder**:
One constraint-row generator — `MassBalance`, `Sum`, `Ratio`, `RatioBound`, or `FlowBounds`.

**Solver**:
Stage 4 — the σ-weighted QP, built per timestep by the orchestrator `Solver` (σ weights, prior,
slack of the soft rows) and solved by an injected **QP backend** (`ClarabelBackend`), producing
the optimised flow vector `a*`.

**QP backend**: The numerical solver of one period's standard-form QP (variables `[a; ε]`, equality and
`≤` rows), reporting a neutral status (`solved`, `solved_inaccurate`, `failed`) beside its own raw one
(`OptimResult.backend_status`); it knows nothing of edges, rows, σ or periods. It owns its settings, its
telemetry (`convergence`, `iter_history`) and the reuse of its factorisation. Clarabel
(`ClarabelBackend`) is the production backend. _Avoid_: "the solver" (that is Stage 4 as a whole),
"solver backend".

**Posterior uncertainty propagation**:
Stage 5 — the closed-form posterior standard deviation `σ_a*` and confidence intervals; its
per-edge summary is the **posterior contraction** `κ`.

**Engine**:
The orchestrator for a stage — it holds a registered, ordered list of plugins and runs them
(`ImputationEngine`, `ConstraintEngine`).
_Avoid_: "pipeline" for one stage's engine — the **pipeline** is the ordered stages (below).

**Register** / **plugin**:
`.register(...)` appends an imputer or builder to an engine; each stage's engine runs its
plugins in order.

**Chaining**:
Running plugins in **registration order** so each consumes the previous one's output. Imputers
mutate `f` / `σ` **in place**, so order matters — e.g. `FlatPrior` is registered **last** as
the final fallback.

**Lp / power-mean combine**:
The single σ combiner used by every stage — `lp_combine({xᵢ}, p) = (Σ|xᵢ|ᵖ)^(1/p)`. `p = 2` recovers
Gaussian quadrature; the project's `P_PROP` damps fan-in σ inflation.

**Pedigree → CV**:
The Laner et al. (2016) map from five 1–4 quality scores to a coefficient of variation — the
origin of every feed's σ.

**Hard** vs **soft** (constraint):
**Hard** ⇔ `σ_c = 0` — exact, lands in the zero or nonneg cone, no slack. **Soft** ⇔
`σ_c > 0` — carries a slack `ε` in the free cone, penalised by `1/σ_c²` in the objective.

**Sense** (`eq` / `le`):
The mathematical sense of a `G` row, a `row_meta` column: `eq` (`G·a = c`) or `le` (`G·a ≤ c`;
`G` pre-flips `ge` bounds, `bound_dir` keeps the original direction). Solver-neutral and
orthogonal to `is_hard`; the (`le`, soft) cell is unrealised (`NotImplementedError`). _Avoid_:
"relation" (a different term, below), "cone type".

**Cone** (`zero` / `nonneg` / `free`):
The Clarabel cone a constraint row lands in, derived inside `pysuto.clarabel_backend` from the QP's row
groups (`pysuto.solver.row_layout` orders them from `sense` and `is_hard`; ADR 0042) and not stored in
`row_meta` — `zero` (hard `eq`, and a soft `eq` too: its slack `ε` is a free variable of the QP),
`nonneg` (hard `le`). The legacy name `free` is the soft `eq`.

**Warm start**:
Seeding the prior `a₀(t)` for a timestep with the previous year's optimum `a*(t−1)`; a second,
numerical sense (reusing Clarabel's KKT factorisation) is under "Two warm-starts" below.

**Run** / **run-id**:
One persisted execution of a stage or a sweep, keyed `{kind}_{YYYYMMDD_HHMMSS}_{uuid}`
(`kind ∈ {feed, imp, const, opt, uncprop, psut, sweep}`; a sweep is no stage).

**Registry**:
A per-stage store of runs. The **input registry** versions the input *bytes*
(feed / acc / params / mask, hardlink-deduped); each **output registry** records a run plus its
sidecar.

**Sidecar** (`metadata.json`):
A run's self-contained provenance record — `parents.inputs` (input-registry refs) and
`parents.upstream` (parent run-ids).

**Diagnostic** (diag):
A per-stage plot suite that consumes a run's output (`AggDiagnostic`, `SolverDiag`, …); never
imported into the data layer.

**Pipeline**:
The ordered stages, feed → imp → const → opt (→ uncprop → psut), each run by its driver and
consuming the previous stage's run. A pipeline may start or stop at any stage.
_Avoid_: "pipeline" for one stage's engine; "workflow".

**Chain**:
One lineage of runs through the pipeline, linked by `parents.upstream` — e.g. the **reference
chain** is the feed, imp, const and opt runs the manuscript cites.
_Avoid_: "chaining" (that is plugin order inside an engine); "the run" for a whole lineage.

**Knob**:
A tuning constant of a stage driver (`P_PROP`, `ANCHOR_CV`, `TOL_GAP_REL`, the cascade order): an
UPPER_CASE field of the case's section for Stages 2 and 3, a driver constant for Stage 4; its value is
recorded in the run's sidecar. Changing a knob is a measured change. In reader-facing text (manuscript,
abstracts) the knobs a **sweep** varies are **judgement values**. _Avoid_: "hyperparameter"; "parameter"
/ "param" (those are the `params` input tables); "setting", "config"; "judgement parameter".

**Sweep**:
A set of chains that share their inputs and differ only in knob values, run and recorded as one
`sweep` run. Two **modes**: a **random sweep** draws `n_points` points from the axes' ranges
with a seeded sampler (the default); a **grid sweep** runs every combination of the axes'
listed values.
_Avoid_: "experiment", "study" (a REP), "search" (nothing is optimised).

**Point**:
One chain of a sweep — its inputs plus one knob value. The **base point** is the chain the
other points are compared with.
_Avoid_: "scenario", "variant", "configuration".

**Axis**:
One knob (or a tied set of knobs that must share a value, such as `P_PROP` in imp and const)
with its **range**: a list of values to choose from, or an interval or distribution to draw
from. A grid sweep accepts lists only.
_Avoid_: "dimension", "factor", "parameter grid".

**Outcome**:
A named scalar that compares a point's answer with the base point's: the cross-evaluated
objective, the mass moved, or the change in pooled retention or in the posterior total over a
chosen set of edges. Outcomes are computed from the stored posteriors after the sweep, so a new
one needs no rerun, and the axes are ranked per outcome.
_Avoid_: "metric", "response", "objective" (that is the QP's J).

**Swing**:
How far an axis moves an outcome, in the outcome's own unit: the outcome's mean over the points
in the top third of the axis's values minus its mean over the bottom third (for a list axis, the
spread of the means per value; of two values, the second's mean minus the first's). It sizes an
effect that SRRC and Spearman's ρ only rank.
_Avoid_: "effect size", "sensitivity" (the ranking), "main-effect range".

**Criterion**: A limit on one or more of a sweep's outcomes, named by its item (A6) and fixed before the
sweep is read; its **verdict** is how many points meet it and which point and outcome come nearest to the
limit or exceed it. A pre-registered item of the manuscript is a criterion. _Avoid_: "threshold", "test",
"stop rule" (the Round-E rule that judges a σ or repricing change). ⚠ Round-E (c), an item of that stop
rule, is also declared as a sweep criterion: there it is judged at every point against the base chain,
not once for a proposed change.

**Expected answer**:
A row of the manuscript's ledger (A1–B3, Table 3): the answer to one sub-question of the research
questions RQ1–RQ3, with its metric and threshold written before the reference run was read; its
outcome is **met** or **not met**. A question without one is *exploratory*.
_Avoid_: "hypothesis" (the rows answer questions; they were called hypotheses until 2026-11-12),
"pass" / "fail" (a not-met answer is a finding, not a failure of the reconciliation).

**Key result**: What a reader of the paper takes away: a property of the reconciled account on the
**reference chain**, or of its sources as the account reveals them, that answers a sub-question of
RQ1–RQ3. It passes three tests: reference chain, bug-free, research question. A size, runtime or coverage
is a *stage fact* (Methods); a fixed defect, a solver-setting artefact, a delta against an earlier chain
or the ledger tally is *debugging* (Methods, SI, ADRs, REPs). _Avoid_: "result" for debugging; "finding"
for a stage fact; "headline" for anything large.

**Year profile**:
An outcome evaluated once per year of its window instead of pooled, so that the axes can be
ranked year by year.
_Avoid_: "time series" (that is the data), "trajectory" (the imputation steps).

**Flagged ambiguity — "pipeline".** Before 2026-10-21 the glossary used "pipeline" for one
stage's **engine**; README and CLAUDE.md always meant the six stages. It now means only the
latter.

## Solver / time-partition vocabulary

Project-specific terms for the time-partitioned solve.

**Period slice** (`a*(t)`):
The optimised flow values for a single period — the `cols_t` edges (those whose
`time == t`). This is what a per-t block-path file stores. Shape `[E_t]`, not `[E]`.
_Avoid_: "the per-t F_opt vector" (ambiguous — used to mean the full `[E]` vector
with placeholders), "yearly solution vector".

**Complete `a*`**:
The full flow solution over all `[E]` edges, formed by the **disjoint union** of
every period slice (each edge is solved exactly once, in its own year). Assembled
on demand by consumers (e.g. `SolverDiag`); never stored as a single per-t array.
_Avoid_: "the global F_opt", "the full vector" (collides with the legacy full-G output).

**Cross-year placeholder**:
A non-`t` edge value left at the prior `a_0` inside a per-t file. The pre-fix block
path stored these; they caused the spurious mass-balance deviation and bimodal `a`
distribution. Year-local persistence removes them entirely.
_Avoid_: "warm-start carry" (that's a separate, legitimate mechanism).

**Block path**:
`_solve_all_via_blocks`, the only way Stage 4 solves: the `ConstraintRun` carries a per-t
partition (`G_blocks`) and the output is year-local. A run without one raises (the full-`[E]`
legacy path was removed, ADR 0042); opt runs written by it still load, and their npz lack
`cols_t`.
_Avoid_: "per-t solver" / "global solver".

**No cross-time coupling** (invariant):
The property that every constraint row's non-zeros lie within a single time
bucket, so `G` partitions cleanly into `G_blocks[t]`. The Stage-3 partition raises
if this is violated.
_Avoid_: "time independence".

**Parallel solve**:
Solving the periods of a cold-prior run concurrently, each period its own QP; possible only because a
cold period's prior does not depend on another period's optimum. A warm run is a chain and is solved in
period order. _Avoid_: "multi-threaded solve" — that is the solver's threading *inside* one period's QP
(its linear algebra), which a parallel solve pins to one thread.

## Concordance vocabulary

**Two mapping levels** — the feed stage composes *two* independent raw→root mappings, each of which can
be 1:1, fan-in, or fan-out. The fan-in/fan-out terms below name the **second** level only.

- **Mapping 1 — raw rows → source edges.** Defined by the concordance **variable map** (`variable_map`
  sheet).
- **Mapping 2 — source edges → root edges.** The WeightMatrix.

**Fan-out** — *one* source edge distributed across several root edges, arising when a raw row
classifies **only one endpoint** (the other endpoint is left open as `"<any>"` and resolved by
normalised weights). Mass-preserving: the source value is *split* across the fan, weights sum to 1.

**Replication (anti-pattern, not a concordance shape)** — the failure mode where a single raw row that
classifies only one endpoint is materialised into *many distinct source edges* (one per matching value of
the open endpoint), each carrying the **full** raw value. This is **not** fan-out; it inflates total mass
by the fan cardinality. Distinguished from fan-out by counting *distinct src edges per raw row*: fan-out
= 1 (→ many roots); replication = many. See [ADR
0002](docs/adr/0002-single-endpoint-rows-fan-out-not-replicate.md).

**Unweighted fan** — a fan-out whose legs carry no weighting information (no trade shares, no user
weights — every leg still at the uniform default). Such a fan is **not distributed**: an even 1/N split
asserts a fabricated partner pattern at the observation's confidence, whereas the honest state of
knowledge is "total known, split unknown". The raw value stays undistributed (excluded from the feed's
projected mass). See [ADR 0008](docs/adr/0008-unweighted-fans-are-not-distributed.md).

**Qualifier fan** — a Use fan whose source variable names its **input product** (e.g. smelter production
*from scrap*, refinery production *from matte*). Only root legs whose supplier sector matches the named
product belong to the fan. A qualifier fan with **zero** matching legs marks a structurally impossible
mapping (the named product does not feed that consumer in the accounting structure) and is a hard error,
never a silent fan. See [ADR 0008](docs/adr/0008-unweighted-fans-are-not-distributed.md).

**Mapped self-trade** — a bilateral src row whose two endpoints fold onto the **same root region** under
the region map. Its root address is indistinguishable from the **domestic** edge, yet its value is a
tiny, customs-precise subsample of a production-scale transfer — an inconsistent observation, not a
measurement of that edge. Bilateral feeds are therefore **scoped to cross-region edges** ([ADR
0014](docs/adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md)). _Avoid_: reading a mapped
self-trade row as a domestic observation.

**Mirror duplicate** — a BACI flow under one code that repeats copper another code already declares, so
mapping both to one region counts it twice. The test is the balance, not the customs record: in the years
the true exporter's own declarations already reach about its production, so that the second code would
push them past it, the two codes' flows are duplicates and only one declaration survives ([ADR
0014](docs/adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md)). _Avoid_: "mis-coded flow" for
every year of such a code — it is a duplicate only where the balance says so.

**Transit declaration** — a BACI flow of refined copper (HS 7403.1x) into a country it only passes
through, whose final buyer declares the true exporter as origin too, so the same copper is exported
twice ([ADR 0014](docs/adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md)). _Distinct from_: a
**re-export**, which the record shows leaving the hub again. _Avoid_: "transit" alone, which the
**transit residual ledger** uses for what the re-export netting leaves behind.

**Reporting territory** — the real-world territory a feed's rows describe, which is *not* the same as the
**source region label** those rows carry. Two source labels denote one territory exactly when the region
map assigns them to the same **real-country root**. Anything computed *before* the region map is applied
must therefore group by territory, not by label ([ADR 0032](docs/adr/0032-one-territory-one-label.md)).
_Avoid_: reading a handover-year pair as two measurements.

**Vintage** — the release a feed row came from, when a source publishes the same series repeatedly. A
later vintage **restates** an earlier one, so rows sharing a key are resolved by keeping the newest. That
rule is only sound *between* vintages: two rows of **one** vintage sharing a key are two different
**streams** of the same quantity and they **sum**. _Avoid_: resolving a same-vintage collision as a
restatement ([ADR 0036](docs/adr/0036-a-renamed-type-is-a-second-stream-not-a-restatement.md)); reading a
vintage as a *time* coordinate; "edition" for a yearbook's release — say vintage.

**Boundary topology** *(referenced by [ADR 0001](docs/adr/0001-replace-coherence-sum-eq-with-bilateral-anchoring.md); to be expanded)* —
the open System-Boundary (SB) endpoints where flows terminate, so prior disagreement there does
not create feasibility conflicts.

## Accounting-structure invariants

**Sector → entity is 1:1** — every *concrete* root sector maps to exactly one entity kind (`Flow` |
`Process` | `SB`), so entity is functionally determined once the sector is known. **Caveat:** an **open
(`"<any>"`) sector does not determine an entity**. Concordance code must therefore make the entity
`"<any>"` whenever its sector is `"<any>"`. See [ADR
0002](docs/adr/0002-single-endpoint-rows-fan-out-not-replicate.md).

**System-boundary category (`entity_sub` on SB nodes)** — the sub-classification carried by every `SB`
node, naming *what kind* of boundary crossing it is. `entity_sub` is meaningful only on `SB` nodes
(`Flow` / `Process` nodes leave it empty). It is the case's **label**; the library reads the boundary
**role** bound to it (see Role). _Avoid_: conflating `entity_sub` with **entity** (only `Process` |
`Flow` | `SB`); and avoid "boundary type" as a loose synonym for it.

**Satellite flow** — an edge of the accounting structure that carries a calculated value but is excluded
from its source node's mass balance by the `massbal_scope` mask. See [ADR
0038](docs/adr/0038-waste-rock-is-a-satellite-mass-flow.md). _Avoid_: "residual output" (a residual
output closes its node's balance; a satellite never does) and "slack".

## Coverage & val vocabulary

**Value (`val`)** — *a value of an edge*: the post-imputation `f_agg` on a root edge.
Every root edge has one (imputation fills all NaN), so "having a val" alone says nothing
about data quality — see **val-coverage**.

**val-coverage** — the relative number of edges in a subset that carry a **real** val, where "real"
follows the imputation-log *method* rule (`has_val`): flat-prior / `flat_prior_noisy_zero` default fills
are **not** real; observed values — *including a genuine `0.0`* — and fan / time / parametric / formula /
**network** (including a bounds-**clipped** fill, below) imputations **are**. _Avoid_: deciding
val-coverage from `f == 0` — a flat-prior fill is `0` but so is a real observed zero.

**Bounds-clip (in-imputer) — `network_*_clipped` fill.** The NetworkImputer floors every fill to the
edge's `[lower, upper]` bounds **before** it propagates through the recipe/balance fixpoint. The clipped
value is a **saturated estimate**, not a precise bound: σ is widened to the clip gap and the fill is
tagged `network_*_clipped`. A clipped network fill still **counts as real val-coverage**.

**Coverage** (glossary, the `agg_result.coverage` boolean) — *did any feed observe this
edge?* — a **pre-imputation** notion, strictly **narrower** than val-coverage: it credits
**only** feed observations, never any imputation. _Avoid_: conflating the two — the stats
column is `val_pct` (not `coverage_pct`) precisely because val-coverage also credits
non-flat-prior imputations that feed-`Coverage` excludes.

**Observed zero** — an edge a feed reports at exactly 0 with a σ > 0 (the feed result's `observed_zero`
mask): covered, so no imputer fills it, and any flow the solver puts on it is priced. Distinct from an
uncovered edge (no feed observed it; NaN in Stage 2) and from the net-stock seed (a prior no feed
reports). _Avoid_: "fed zero"; "zero pin" (the `l = u = 0` bound on no-data edges, [ADR
0013](docs/adr/0013-no-data-flat-prior-anchor.md)).

**Non-producer** — a region-year ICSG covers whose own ICSG production of a product is 0: ICSG
lists the region in some table, in some year, but reports none of that product for it that
year. Never ROW (a sum of members), a region ICSG never lists (Luxembourg) or a region-year ICSG
reports inside one of its own aggregates: their zero is a gap, not a report. _Avoid_: "non-miner"
for a region that only lacks SX-EW; "unreported region".

**Non-processing region** — a model region with no ICSG smelter, refinery or SX-EW production
row in any year: the set on which ledger row A3 scores the edge objective of cross-region legs.
It is read from the ICSG feed, not from the posterior, which books some processing output
everywhere. It mixes trading economies (Switzerland, Singapore), fabricators (France, Portugal)
and mine-only exporters (Papua New Guinea, Panama). _Avoid_: "merchant hub" or "hub" for this
set (a hub is a region whose trade the re-export netting re-attributes); "transit region".

## Defect, residual & violation vocabulary

**a_prior (warm-start prior)** — the per-**period** prior mean the optimiser uses for year `t`. Under
`prior_mode='warm'` (the `Solver` class default) it is genuinely the *previous* year's posterior:
`a_prior(t0) = f_agg` (the cold seed), `a_prior(t) = a_post(t-1)` for `t > 0`. Under `prior_mode='cold'`
every year uses its own cold seed `f_agg(t)`. _Avoid_: reading `ConstraintRun.a0` as the per-year prior —
`a0` is the legacy global (time-replicated) cold vector; the year's prior is `a_prior`.

**Two warm-starts (not the same thing).** `prior_mode` (warm/cold) is the **modelling** prior
above. The `reuse` argument of `QPBackend.solve` is a purely **numerical** warm-start —
`ClarabelBackend` then reuses Clarabel's KKT factorisation across timesteps (its `inst.update`); it exists only in a serial solve, since a
**parallel solve** factorises each period on its own. _Avoid_: conflating the two; the local
that selects KKT reuse was historically also called `warm_start`.

**Defect** — the per-**edge** prior departure `a* − a_prior` (solved value minus the warm-start prior,
`a_prior` above). Plotted by SD-T-07 (the a-space rocket, `a_prior(t)` vs `a*`), ranked by **objective
share** in SD-R-13 / SD-R-14 / SD-R-15a (the edge-defect family), and tabulated row-level by
`SolverStatsDiagnostic` (columns `defect` / `defect_abs` / `z`). Its standardised form is
`z = (a*−a_prior)/σ_a`. _Avoid_: calling the defect a "residual" — that term is reserved (below).

**Residual** — the per-**constraint-row** delta `G·a − c` (`OptimResult.residuals`). Its feasible range
depends on the row's cone: **≈ 0 on hard *equalities*** (`zero` cone — mass balance); **`≤ 0` on hard
*bounds***, where a *negative* residual is healthy **slack** (distance below the limit), not a violation,
and only a *positive* residual is a breach; **`= ε`, the slack, on a soft (free) row**. _Avoid_: (1)
reading a large-negative residual on a `nonneg` bound as a violation — it is slack; (2) conflating with
**defect** — residual is per-constraint, defect is per-edge.

**Prior breach** — a hard row the prior does not satisfy: `|G·a_prior − c|` above the bind
tolerance on a `zero` row, `G·a_prior − c` above it on a `nonneg` row. The solver has to move
flows to close it. A soft row has no breach — its disagreement with the prior is its `z`.
_Avoid_: "prior violation"; "infeasible prior" — the prior as a whole is never feasible, a breach is
one row's.

**Structural lower bound** — the per-edge `l` of the accounting structure (`acc.bounds.l`): 0 on every
edge except the signed stock-change edges (`Cathode→USC`, `Cathode→MESC`, `CP→PSC`), which carry a
negative `l`. Since [ADR 0039](docs/adr/0039-flow-lower-bounds-are-enforced-in-the-qp.md) each `l = 0`
enters the QP as one hard `nonneg` row `−a ≤ 0`. A signed edge (`l < 0`) carries no **one-signed band**
([ADR 0041](docs/adr/0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md)). _Avoid_:
reading `ConstraintRun.l` as the structural bound.

**One-signed band** — a hard ratio band `bl·denom ≤ num ≤ bu·denom` whose lower ratio bound is
≥ 0 (or that has an upper bound alone): satisfiable for a negative pair only at zero, so on a
**stock-change leg** it would forbid the draw-down half. Never emitted on a signed edge; a band
that spans zero (`bl < 0`, the stock bands) is sign-safe and stays hard
([ADR 0041](docs/adr/0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md)).
_Avoid_: "positive band", "non-negative band".

**Forced vs discretionary deletion** — the split of a node's prior-to-posterior mass loss into the part
no optimiser could have avoided and the part it chose. The remainder is **discretionary**. The
distinction says which lever can move a number: σ, anchor placement and pricing reach only the
discretionary part, while the forced part moves only when `X`, `U` or `P` themselves change ([ADR
0029](docs/adr/0029-the-concentrate-grade-stays-global.md)). _Avoid_: reading a forced term as a defect
to be relieved.

**Constraint table (prior/post)** — the harmonised **row-level, all-rows** per-`constraint_id` table pair
(hard + soft rows; `type` / `sense` columns carry the family for filtering). **`lhs_prior` = G·a0** means
the same thing in *both* files; **`lhs_post` = c_star = G·a*** exists only post-solve. _Avoid_: reading
the solver-stage `residual` on a **hard** row as the **residual** above — `c_star = c` there by
construction, so the column is identically 0.

**Objective share** — one item's `z²` (an edge's defect or a soft row's residual, standardised) divided
by the `Σz²` of its family or of the whole run. The QP objective *is* `Σz²` over edges plus soft rows, so
the share is what that item cost the optimiser. Ranked per **identity** over all years by `SolverDiag`.
_Avoid_: "deviation" (say defect or residual), "violation" (a soft row is meant to absorb disagreement —
see Constraint table), "fit".

**Objective decomposition** — the per-year split of `Σz²` into the three families, drawn with the
max hard mass-balance `|G·a* − c|` (SD-R-11). _Avoid_: reading the Clarabel `objective` in the opt
sidecar as this quantity — it is `½xᵀPx + qᵀx` without the constant term and is negative.

**Identity (cross-year)** — the key under which one edge or one constraint row recurs across years. For
an edge: the **edge template** `(from_id, to_id, valuation)`. For a row: the tuple of its `row_meta`
columns other than `constraint_id` and `time`. Always read from columns; a row whose columns do not
single it out within its year has no identity (it still counts in every total). _Avoid_: stripping the
year off a `constraint_id` (see `constraint_id`: a key for reading, never parsing).

**Retention** — the share `Σa* / Σa_prior` computed **by category**: the posterior sum over the edges of
one category in one year divided by the prior sum over the same edges. It is a dimensionless share of the
prior mass: `1` keeps the prior scale, `< 1` is deletion, `> 1` inflation. The complement of the deletion
in **forced vs discretionary deletion**. _Avoid_: "adaptation", "recovery", "survival"; a retention
quoted without its category and year; a retention on a signed category whose prior sum is near zero.

**Category** — the pooling set of a per-category table: one sector pair in one valuation,
labelled `sector_from→sector_to valuation` (`Scrap→CFR CC`), as the sweep's category tables, the
Table S19 stages and the posterior contraction per category (Figure 5a) use it; the **retention**
entry's wider sense (a node, a region, an SB class) also applies where the category is named.
_Avoid_: "stage" for a category outside Table S19's nine; "flow" (that is one edge); "sector"
alone (a category is a pair).

**Pooled retention** — a retention whose sums run over a stated **window** of years: `Σ_t Σa* /
Σ_t Σa_prior`, one ratio of sums, so the large years weigh most. It is always quoted with its
category and window (e.g. Scrap→CFR CC, 1995–2024).
_Avoid_: a pooled retention quoted without its window; "average retention" (it is not a mean of
the yearly ratios); a pooled retention on a signed category.

**Posterior contraction (`κ`)** — the share of an edge's prior variance the constraints remove in one
year, `κ = 1 − σ_a*² / σ_a²`: `0` leaves the prior uncertainty untouched, `1` pins the edge. It depends
on `G`, `σ_a` and `σ_c` and, through the QP's active set at `a*` (which rows bind, which lower bounds pin
their edge), on `a*`; given that active set it does not depend on `c`, so it measures how informative the
constraints are, not whether they agree with the prior — that is the **defect**'s `z`. _Avoid_:
"shrinkage", "variance reduction", "information gain".

**Trace contraction** — the run-level contraction of one year, `1 − Σσ_a*² / Σσ_a²` over that
year's edges: a variance-weighted mean of `κ`, so a few flat-σ edges dominate it.
_Avoid_: "mean shrinkage".

**Active set** — the hard inequality rows (lower bounds, ratio bands) that bind at `a*` in one
year; the posterior is the QP's linearisation there. A row of the active set binds **on flow**
when at least one of its edges (its nonzero coefficients) carries `|a*|` above the bind
tolerance, and **on zero flows** otherwise.
_Avoid_: "active constraints", "binding constraints" for the mass balances — equalities always hold.

**Bound-active** — an edge whose lower bound is in the active set; its posterior is pinned at the
bound (`σ_a* = 0`).
_Avoid_: "deleted", "at zero" — a signed edge at zero is not bound-active.

**Prior class** — where an edge's prior comes from, one of four: **from a feed** (a data feed
covers the edge), **imputed** (a Stage-2 imputer wrote it), **flat** (no information of its own:
a flat prior, anchored or not, and the fabricated waste-rock satellites) and **zero-pinned** (a
no-data edge with `l = u = 0`). The four are disjoint and cover every edge.
_Avoid_: "observed" vs "modelled" — an imputed prior also derives from data.

**Provenance** (of an edge's prior) — *from a feed*, or the one imputer that wrote it (`network`,
`trade`, `conc`, …): the prior class at imputer resolution. An edge has exactly one.
_Avoid_: "raw" for the feed-covered share — a network fill also carries feed totals; "provenance"
for a run's lineage — that is the **Sidecar**'s record.

**Prior-carried / confirmed / overruled / unexplained move** — the four quadrants of an edge-year in
posterior contraction `κ` and standardised defect `z`, cut at `KAPPA_CUT` and `Z_CUT`: prior-carried
(`κ < KAPPA_CUT`, `|z| ≤ Z_CUT`), confirmed (`κ ≥ KAPPA_CUT`, `|z| ≤ Z_CUT`), overruled (`κ ≥ KAPPA_CUT`,
`|z| > Z_CUT`, a prior–data conflict) and unexplained move (`κ < KAPPA_CUT`, `|z| > Z_CUT`, moved
uninformed). _Avoid_: "unidentified" for prior-carried — the edge has a value, only no new information.

**Standardised shift (`z/√κ`)** — an edge-year's defect `z` over `√κ`: N(0, 1) under the active-set
linearisation when σ is right, read only on edge-years that are not bound-active, have
`κ ≥ CALIB_KAPPA_MIN` and are not flat. Its **tail share** is the share beyond `|z/√κ| = Z_CUT`
(`CALIB_EXPECTED_SHARE` under N(0, 1)). _Avoid_: "pull", "normalised residual".

**Flagged ambiguity — "standardised".** The **standardised defect** `z = (a* − a_prior)/σ_a` is
scaled by the prior σ and shrinks toward 0 as `κ → 0`; the **standardised shift** `z/√κ` is scaled
by the shift's own σ. Only the shift is compared with N(0, 1).

**Flagged ambiguity — "sensitivity".** Three meanings live in the manuscript and the code: (1)
the **posterior σ_a\*** is the estimator's *local sensitivity under the stated σ* (Stage 5; it is
not a calibrated interval); (2) a **knob sensitivity** is a sweep's ranking of its axes on an
outcome (SRRC / Spearman ρ, sized by the **swing**); (3) the pedigree *sensitivities* of a feed are
the CV contributions of its data-quality scores (SI S2). Say which one; "uncertainty of an edge"
is (1), "sensitivity to a knob" is (2).

**Birge ratio (`R`)** — `Σz²/dof` of a year or the run: the QP objective at `a*` over the
independent constraints of the active-set problem (soft rows kept, independent hard rows kept,
scale totals an ablation left out, bound-active edges whose prior is off the bound — a pin on its
own prior is zero by construction); `R > 1` means σ is too small by about `√R`, and the
**Birge-scaled interval** `a* ± z_α·√R·σ_a*` is quoted beside the stored `[lower, upper]`.
_Avoid_: "reduced χ²", "goodness of fit".

**Aggregate σ** — the posterior σ of a linear total `wᵀa*` (a node total, a stage's Σa*, a
region's output), including the covariance the constraints induce between its edges. The
**quadrature σ** `√Σ wᵢ²σ_a*,ᵢ²` treats the edges as independent and misstates a constrained
total: a mass balance makes a node's total better known than its split. _Avoid_: summing edge
intervals.

## Stock-change vocabulary

**Stock-change leg** — one of the three signed edges of the account: `Cathode→USC`, `Cathode→MESC` and
`CP→PSC`, each a net addition to a stock (negative = a draw-down). No one-signed hard band sits on it.
For `USC` / `MESC` the prior is a reported value or the **net-stock seed** and no imputer writes it; for
`PSC` an unreported year is still the recipe's eff-less `0` ([ADR
0041](docs/adr/0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md)). _Avoid_: "stock leg",
"stock edge", "Δstock edge".

**Exchange location** — a region in which ICSG reports exchange stocks: the LME warehouse countries by
location, COMEX as the United States, SHFE as China. It is the destination of a `Cathode(i)→MESC(j)` leg,
not a property of the origin. _Avoid_: "hub" (reserved for the merchant / re-export hubs of [ADR
0027](docs/adr/0027-re-exports-are-re-attributed-at-feed-time.md)), "warehouse region".

**Net-stock seed** — the prior of a `MESC` / `USC` leg that no feed reports: value 0, held there (priced
as a feed-backed level, [ADR 0040](docs/adr/0040-stock-change-legs-are-held-at-their-prior.md)). A `MESC`
/ `USC` leg's prior is a reported value or this seed; no imputer manufactures one. _Avoid_: reading a
seeded 0 as a reported zero.

**Supply-mix fan** — the split of a region's reported stock change over the origins of its
stock-change legs, in proportion to the region's cathode supply: own production on the domestic
leg, each partner's cathode exports to it on the import legs
([ADR 0043](docs/adr/0043-a-stock-change-is-drawn-from-its-regions-cathode-supply.md)). _Avoid_:
"domestic residual" (the flow fan's rule, ADR 0024), "imports first".

**Sink** — an edge family whose σ makes its penalty negligible against the mass it can absorb, so a
contradiction elsewhere discharges into it instead of surfacing as `z` where it belongs. The signed stock
legs are the structural candidates (they alone may be negative, ADR 0039). Relates to **forced vs
discretionary deletion**: a sink converts a forced contradiction into a discretionary move. _Avoid_:
"slack" (the soft-row residual), "buffer".

**Stock level** — the end-of-period quantity ICSG reports for a region and one stock type
(producers, consumers, merchants, government, or an exchange). It never enters the account
directly. _Avoid_: "stock" when the account's `PSC` / `USC` / `MESC` entities are meant, "stock
change" for a level.

**Net addition to stock (NAS)** — the change of one stock series between two consecutive years of
one **vintage**: the reported value a **stock-change leg** carries. `PSC` is the producers' NAS
(a region that prints only a total counts as producers), `MESC` the exchanges' NAS at one
**exchange location**, `USC` the summed consumer, merchant and government NAS. _Avoid_: "stock
change" for the quantity (ambiguous with the leg), a difference across vintages or a reporting gap.

## Feed valuation & trade-share vocabulary

**Valuation — `CC` vs `Mass`.** Every edge exists in two valuations. **`Mass`** is the gross product
mass (tonnes of the commodity as traded/handled). **`CC`** (*copper content*) is the contained-copper
mass (tonnes Cu). They are linked per sector by the **concentration coefficient**.

**Concentration coefficient (`conc`).** The Cu mass fraction of a sector's commodity. The param's
`value_nominal` drives the **soft** `CC = conc·Mass` ratio row; its `[value_min, value_max]` drives a
**hard grade band** `value_min ≤ CC/Mass ≤ value_max` (`ratio`/`nonneg`). **Invariant: each commodity
carries exactly one such band** ([ADR 0012](docs/adr/0012-scrap-one-mixed-grade-commodity.md)). _Avoid_:
reading the `conc_*_supply` z-violation as a "bug" — it fires whenever the two feeds populating the CC
and Mass sides of an edge disagree.

**Smelter scrap · direct-to-refinery scrap.** The model's two `Scrap` inflows, named by where the scrap
enters: **smelter scrap** feeds `Smelting` with the primary concentrate and leaves as matte;
**direct-to-refinery scrap** feeds `CFR` with the matte and leaves as anode. ICSG reports two *output*
classifications — **secondary smelter production** and **secondary refined production** — and the second
contains the first. _Avoid_: booking secondary refined production as a second inflow beside smelter
scrap.

**Recovery efficiency (`eff`).** The copper-recovery fraction of a Process, in **`CC`** (from the `eff`
param) — the sibling coefficient to `conc`. A process's recovered Flow product is
`eff · Σ(all CC inputs)` in CC; the `(1 − eff)` complement is the residual that leaves as the **SEM**
residual output. `eff` acts on the **sum of all CC inputs — primary plus the `Scrap` SEM-secondary**.
_Avoid_: conflating `eff` (a CC recovery fraction) with `conc` (a Mass→CC fraction) — they are different
coefficient families.

**Process recipe.** A Process node viewed as a mass-closing transformation in CC: its **pooled inputs**
(primary + `Scrap` secondary) → one **Flow product** (`eff·Σin`) + a **SEM residual output**
(`(1−eff)·Σin`). The unit of reasoning for the network-traversal imputer ([ADR
0004](docs/adr/0004-process-recipe-traversal-imputer.md)). _Avoid_: confusing the recipe's **SEM
residual-output** with a **SEM secondary input**.

**Stripping ratio (`stripping_ratio`).** The mass of waste rock (`WR`) stripped per unit of `Ore`
extracted at a Mining process — `WR_Mass = stripping_ratio · Ore_Mass`, a **Mass→Mass** ratio between
two **outputs** of Mining, held **hard**: it alone sets waste rock, a satellite
flow with no data of its own ([ADR 0038](docs/adr/0038-waste-rock-is-a-satellite-mass-flow.md)).
_Avoid_: calling it an `eff` — `eff` is strictly a CC copper-recovery fraction (above); reading
waste rock as a residual (`CO − Ore`).

**Allocated ore mass.** The `crude_ore` feed's ore Mass is S&P **price-allocated**: a
poly-metallic mine's physical ore is split across its metals by price share, so the copper-
attributed mass is *smaller than the physical ore mined*. `Ore`/`CO` Mass edges carry this
allocated mass, and everything derived per unit of it (grade, waste rock) inherits the basis.
_Avoid_: comparing it to physically mined/milled tonnages.

**Recovered vs in-situ grade.** Two grade notions per region-year: **recovered grade** = metal produced ÷
allocated ore; **in-situ grade** = copper contained in the mill feed ÷ allocated ore = recovered ÷
recovery-`eff`. The model's `CO`/`Ore` **CC edges are in-situ**, so their grade rows use the **in-situ**
value ([ADR 0015](docs/adr/0015-in-situ-ore-grades-on-allocated-ore-basis.md)). _Avoid_: applying a
recovered-basis grade to an in-situ edge.

**Concentrate grade.** The Cu mass fraction of copper concentrate — the `Concentrate` case of the
concentration coefficient, one **global** row. Unlike an **in-situ ore grade** it is a *direct assay*: no
allocation basis, no `eff` division, so it may be read against industry figures as it stands. It is
deliberately **not** region-resolved the way ore grades are ([ADR
0029](docs/adr/0029-the-concentrate-grade-stays-global.md)). _Avoid_: reading a region's implied
`(P−U)/X` as its grade.

**Grade band vs grade σ.** A `conc`-family params row's `value_min`/`value_max` are read **twice**:
as a **hard cone** by `RatioBoundBuilder` (`value_min·Mass ≤ CC ≤ value_max·Mass`, per edge per year — a
physical plausibility limit, no slack) and, when `unc_type = "min_max"`, as the **σ** of the *soft* ratio
row via `ParamRow.sigma` — which also sets the parametric fill's prior and the `NetworkImputer` twin-map
σ. Those are two different statements, and conflating them means changing a limit silently re-prices
three things. _Avoid_: calling the band "the σ" — a band is the wall, σ is the price of approaching it.

**Relative ratio σ (`σ_r`).** The σ of a soft ratio row as a fraction of its ratio, `σ_r = σ/|r|`, where
`σ` is the params row's stated σ in **ratio units** (`value_sd`, or `(value_max − value_min)/4`) and `r`
its `value_nominal`. `RATIO_SIGMA_REL_FLOOR` floors it and the row's flow scale multiplies it. _Avoid_:
calling `value_sd` a CV; quoting a ratio row's σ without saying which of the two it is.

**Grade carried past the ore feed.** The `crude_ore` feed ends before ICSG production does. For a
region-year with production but no ore mass the `ore_grade` generator carries the region's **last
observed in-situ grade** forward. **The assumption is on the grade, never on the mass**: ore mass tracks
production ([ADR 0023](docs/adr/0023-ore-grade-carried-past-the-crude-ore-feed.md)). _Avoid_: carrying
the ore Mass itself (holds tonnage flat whatever was produced, and disagrees with the grade).

**⚠ The grades in use are internal, not external.** Every `ore_grade` row the model consumes is *derived*
from ICSG production and crude_ore allocated mass (ADR 0015). The **external** S&P factor series enters
only as a fallback where ICSG production is absent; using its level would reintroduce the basis jump ADR
0015 removed. _Avoid_: describing the ore-grade series as external data.

**Feed semantics — ICSG vs BACI.** **ICSG** publishes production/usage statistics in **copper content
(`CC`)**; its `Use` rows are **single-endpoint** (consumer known, producer open) ⇒ producer **fan-out**.
**BACI** publishes **bilateral gross trade** in **`Mass`**. On a trade edge the ICSG-CC value is
therefore a *fan artifact* and the BACI-Mass value is the real observation — for a leg the **re-export**
netting did not change. _Avoid_: treating a per-edge ICSG-CC trade value as a measurement.

**Trade-share fan weighting** ([ADR 0002](docs/adr/0002-single-endpoint-rows-fan-out-not-replicate.md)
option D) — weighting ICSG's producer fan by BACI bilateral import shares instead of uniformly, so
ICSG-CC inherits BACI's geography. Since [ADR
0027](docs/adr/0027-re-exports-are-re-attributed-at-feed-time.md) those shares are read from the
**netted** BACI version. The domestic leg (X==Y) carries the residual `usage − Σ imports` (in CC units).
_Avoid_: including **Scrap** — it is an ICSG `Use` type but is **not traded**.

**Re-export (re-attributed leg)** — tonnage a region imported and shipped on unchanged. BACI records it
as **two** gross border crossings (`A→h`, `h→Z`). ADR 0027 moves it at feed time: the volume is split by
the hub's own observed origin and destination shares onto direct `A→Z` legs. The two caps `T ≤ X` and
`T ≤ I` are the safety property. _Avoid_: calling every hub leg a re-export — only the part the caps
admit is one, and the rest stays in the **transit residual ledger**.

**Re-import** — the `A→h→A` branch of a re-attribution: goods that left `A`, reached the hub and came
back. It is dropped rather than booked, for the same reason **mapped self-trade** is: its root address is
the domestic edge, which it is not a measurement of. Where `A` is `ROW` the same branch is third-country
trade folded onto one aggregate, and is dropped identically.

**Merchanting** — a title trade in a third country while the cargo sails elsewhere (BPM6 §10.41). Two
shapes, and only one is netted: **both legs recorded** is an ordinary re-export and re-attributes in
full; **inbound only** cannot be, because nothing leaves the hub to carry it. _Avoid_: reading a small
`T` at a merchanting hub as a failure of the rule — it is the rule declining to invent a destination.

**Transit residual ledger** — what the netting deliberately leaves behind, written per (product, region,
year) beside the feed with a label. Before ADR 0027 this tonnage was the fan's silent **clip**; it is the
same material, now named. _Distinct from_: the **Bounds-clip** above, which is an imputer's saturated
fill. _Avoid_: reading the residual as an error.

**ICSG aggregate row** — an ICSG source row that reports several countries together. Its members and
their years are declared in the `icsg_aggregates` params input, never inferred from names; the ICSG clean
step splits it to its member ([ADR 0032](docs/adr/0032-one-territory-one-label.md)). _Distinct from_:
`excluded-aggregate`, the ledger label for ROW, a *root* region; and from `agg=true` pooling and Stage-2
aggregation, which are unrelated. _Avoid_: "aggregate" alone.

**Handover year** — the first year ICSG publishes both an aggregate row and the members it contains, the
year after the declared member years. It states each member's share of the aggregate's series, and the
aggregate row in it is a republication of its members. _Avoid_: summing the two forms in that year.

**BACI-observed region** — a consumer region for which the BACI feed records at least one cross-region
import (any traded sector, any year). The flag is what makes `Σ imports = 0` *informative*: for an
observed region it means "sourced domestically", for a trade-blind one it means "we cannot see". Derived
in `feedprep.trade_shares.baci_observed_regions()` from the BACI `F_root`. _Avoid_: treating a zero import
sum as evidence without checking observability — that is how a 100 %-domestic pattern gets fabricated at
the observation's confidence.

**Trade-seed** — the trade `Fill` that sets the **CC** value of a *cross-region* import leg from its
already-aggregated BACI **Mass** twin. It is the imputation-time complement of the trade-share fan. A
route with no BACI Mass observation is left NaN (an honest flat-prior gap), never seeded to 0. See [ADR
0010](docs/adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md). _Avoid_: reading a
trade-seed CC as a new measurement.

**Value-implied Cu fraction** — `unit_value(product) / unit_value(cathode)` within one trade feed, one
year. It is a **consistency check between a feed's two fields**, not a property of the material: where it
leaves the physical band the *quantity* is the field in doubt ([ADR
0033](docs/adr/0033-baci-quantity-contradicts-its-own-value.md)). _Avoid_: reading it as a grade.

**Traded (realized) vs tradable; `region_crossed`.** **Tradable** is a *sector capacity* — the four
ICSG-fanned sectors (Cathode, Concentrate, Anode, Matte) whose Use edges *may* cross regions (above).
**Traded / inter-regional** is a *realized* per-edge property — the Use edge actually crossed a border
(`region_from != region_to`) — captured as the boolean **`region_crossed`**. A Sankey's **traded-flow
share** is this *realized* split. _Avoid_: reading "traded share" as the *tradable-sector* fraction — it
is the realized inter-regional fraction.

## Total-output / scale-total vocabulary

**Total output** — of a process or flow, the sum of **all its outgoing edges** (incl. waste, final
demand, SB outflows) ≙ PIOLab *throughput / gross output* `x = S·j + P·l + W·n` (Wieland et al. 2022).
_Avoid_: equating it with **useful output** (which is net of waste).

**Total input** — of a process or flow, the sum of **all its incoming edges** (incl. SB secondary and
nature inputs) ≙ PIOLab `x′ = i·U + k·O + m·E`.

**Useful output** — a process's total output **minus** the SB outflows it supplies (waste / SEM
residuals); i.e. its process→flow edges only ≙ PIOLab *useful output* (supply table `S`,
`x_new = S·j = x − P·l − W·n`). _Avoid_: "total output" — useful output is net of waste.

**Scale total** — a Type-2 **soft-sum** (`Σ aᵢ = c + ε`, free cone) whose RHS `c` is the model's **own
prior** group-sum (`c = Σ f_agg`), used for **scale-preservation** against the homogeneous collapse — not
to add information. Three families: **supply total output**, **use total output**, and the **scrap level
anchor** below. _Avoid_: **control total** ([ADR
0006](docs/adr/0006-scale-total-supersedes-external-control-total.md)); **anchor** / **coherence
soft-sum**.

**Scrap level anchor** — the third **scale total** family (`smelterscrap`, `directrefscrap`): a soft-sum
stating the **level** of *one input class* of a node — the scrap ICSG reports entering `Smelting` and
entering `CFR` — per region-year, **CC only**, and **only where a feed observed it**. It is the one `sum`
family whose *row existence is data-gated*. _Avoid_: calling it a **total input** (that is *all* incoming
edges — this is one class of them); reading the **absence** of a scrap-level row as a prohibition.

**Contradicted flow node** — a Flow node whose **inflow** prior (`Process→Flow`) and **outflow** prior
(`Flow→Process` + `Flow→SB`) cannot both hold under its own **hard** mass balance. Detected by
`pysuto.flow_node_balance.contradicted_flow_nodes`. The term survives as a **diagnostic**: a flag is a
statement about the *input data* (the source's two series disagree for that region-year), not about the
reconciliation, and the remedy is data-side. _Avoid_: reading a downstream export as an outlet for the
node ([ADR 0019](docs/adr/0019-contradicted-flow-nodes-lose-anchor-authority.md)).

**Anchor** — a Type-2 **soft-sum** row built by `SumConstraintBuilder.from_mask_anchor` that pins the group-sum
of a masked edge family to the model's **own** Stage-2 prior: the supply / use **scale totals** and the
**scrap level anchors**. Also the **no-data σ anchor** (ADR 0013), a σ rule on flat-prior edges. Neither
imports an external total, so neither is a **Control total** (removed): the word "anchor" is live in both
uses and the *Avoid* on **Control total** concerns only the external-total reading.
_Avoid_: "control total", "coherence soft-sum" for either.

**Anchor placement** — *which edge* a source series is attached to, as distinct from its value. A
production series measures throughput at one point of the chain; converting it into an edge prior by
dividing through efficiencies moves it to a *different* point, and each stage crossed adds an assumption.
A misplaced anchor is not a σ problem and not a value problem: the number is right and the edge is wrong.
_Avoid_: anchoring both sides of a hard-balanced node — the balance already determines the second side.
See [ADR 0021](docs/adr/0021-icsg-refined-production-anchors-anode-into-er.md).

## Downstream PSUT / PIOT vocabulary

**PSUT (physical supply-use table)** — the **materialised numeric** supply-use flows, derived from the
solved edge vector `a*`; one table holds every period and every valuation, and a view selects one of
each. **Shape-agnostic**: its canonical form is a **tidy edge table** (one row per directed edge,
carrying its `val`). _Avoid_: calling the **valueless** graph a "PSUT" (the bare graph is the **SUT**);
"SUT" for a numeric realisation.

**Block** — the PSUT quadrant a directed edge belongs to, read off its endpoint entity kinds. `S`
Process→Flow, `U` Flow→Process, `Y` Flow→SB·FD, `E` SB·Nature→Process, `O` SB·SEM→Process, `P`
Process→SB·SEM, `W` Process→SB·Nature. Every admissible edge lands in exactly one block. _Avoid_: "mask"
for a block — the `supply`/`use` **masks** are rule-derived edge selections that straddle blocks.

**Primary valuation** — the valuation a multi-valuation PSUT's views and plots default to (`primary`; copper:
`CC`, the `content` role, ADR 0034); `valuation=` overrides it. A PSUT carries **every** valuation of
its structure in one table, and the realized `conc` is the ratio of two of them. _Avoid_: reading the structure's
first-listed valuation (`Mass`) as the default; one PSUT per valuation.

**Aggregation (pushforward)** — region/sector aggregation of a PSUT as a *pushforward* of the tidy table:
one concordance map per dimension (`region`, `sector`) applied **symmetrically to both facets** (`from_`,
`to_`), then a sum over every key column — `period`, `valuation`, `block` and the entity kinds stay in
the key, so nothing merges across them and **time is never summed**; a sector map may only merge sectors
of one entity kind. _Avoid_: summing `period`; a one-label `sector` collapse (it would merge Process with
Flow).

**Supply-use Sankey** — the bipartite Sankey drawn *directly* from a PSUT: processes **and** commodities
are nodes, the `S`/`U` rows are links, the boundary blocks attach as links to their own `SB` nodes, and a
border-crossing `U`/`Y` flow is its own **traded** link. Needs no construct. _Avoid_: "process network"
for it (that is the PIOT's contracted view).

**PIOT (physical input-output table)** — a **symmetric** (square) physical IO table derived from a PSUT
by a chosen **construct** (below). The shipped one is the **industry-by-industry** (process×process)
PIOT, the **process network** (below): technical coefficients `A`, core flows `Z`, total requirements
`L`, for footprints. _Avoid_: conflating with the rectangular **PSUT** (a PIOT is square); reading "PIOT"
as *product-by-product only* (it is a construct **family**).

**Construct** — the SUT→PIOT transformation choice, on **two orthogonal axes**. **(1) Table shape**:
**product-by-product** (commodity×commodity) vs **industry-by-industry** (process×process), the latter
via the **market-share** rule. **(2) By-product method** — meaningful *only* on the analytic
product-by-product PIOT: **Suh** (2004) endogenises disposals-to-nature as **negative input
coefficients**; **Xu–Zhang** (2009) leaves the table intact via a modified multiplier. _Avoid_:
conflating the axes — **market-share** is a *shape*, **Suh** is a *by-product method*.

**Process network** — the **industry-by-industry** PIOT in tidy/graph form: **processes are nodes,
commodity flows are edges**; each edge `p→q` carries commodity `f` at value `D[p,f]·U[f,q]` (market-share
allocation of `q`'s use of `f` back to its producers `p`). Boundary blocks attach as edges to boundary
pseudo-nodes; **disposals-to-nature** (`P` residual outputs) are **final outputs, never inter-process
edges**. The substrate of the production-network **Sankey**. _Avoid_: calling it a "PIOT" unqualified
(defaults to product-by-product).

**gPIOT** — the full **multi-regional** PIOT over (region × flow), whose `L` traces embodied/indirect flows
across borders. _Avoid_: "global PIOT" for a single-region table.

**Technical coefficients `A`** (uppercase) — the per-unit input-requirements matrix of the PIOT
(`A = Z·x̂⁻¹` for the shipped industry-by-industry construct).

**Total requirements / Leontief inverse `L`** — `L = (I − A)⁻¹`, mapping final demand to total (direct +
indirect) output. _Avoid_: "multiplier" (a multiplier is a row/column sum of `L`, not `L` itself).

**Footprint** — a **consumption-based** indicator: the boundary input (extraction) embodied in a vector of
final demand. _Avoid_: "direct/apparent" use (a footprint is the indirect/embodied counterpart).

**Embodied bilateral trade** — the footprint resolved on both ends: rows the region where the boundary input
enters (extraction), columns the region whose final demand embodies it (`Σ_{p∈r} e_p L[p,q] Y[q,d∈s]`); row sums
are each region's extraction, column sums its **RMC**. _Avoid_: reading it as physical trade (that is the `Z`/`Y`
edges crossing a border).

**ew-MFA headline indicators** — per region and period, in the table's unit: **DE** (domestic extraction,
the `E` block into the region's processes), **imports** / **exports** (network edges crossing into / out
of the region), **PTB** `= imports − exports`, **DMI** `= DE + imports`, **DMC** `= DMI − exports`,
**RMC** (raw-material consumption, the extraction footprint of the region's final demand). Over a closed
world `Σ RMC = Σ DE`. _Avoid_: comparing DMC (territorial, gross trade) with RMC (consumption-based,
embodied) as if they measured the same thing.

**Realized `conc`** — the *ex-post* concentration `CC/Mass` of an edge in the materialised PSUT, versus the
parametric **concentration coefficient** input; the gap is solver slack. Ore grade is the Ore-sector instance.
_Avoid_: reading a realized≠parametric gap as an error (cf. the `conc_*_supply` z-violation — it is absorbed slack).

## Constraint identity vocabulary

**`constraint_id`** — the unique, human-readable **display key** of one constraint row, with the
grammar `{type}_{variant}_{specs}_{year}` (tokens joined by `_`; every token underscore-free). It is
a key for *reading*, never *parsing*: all structured fields live as `row_meta` columns and are read
from there. _Avoid_: splitting a `constraint_id` to recover a field — read the column instead.

**type** (constraint family) — the top identity slot, one of `massbal` | `sum` | `ratio` | `bound`
(the structural lower bounds, ADR 0039). Already a
reserved word for the constraint-row `type` column (see Glossary §Entity `_Avoid_`: "node type").
_Avoid_: using `type` for an entity kind (that is **entity**: `Process` | `Flow` | `SB`).

**variant** — the second identity slot: the *role* a constraint plays within its family
(`massbal`; `usefulsupply` / `subentity` / `supplytotout` / `usetotout` / `smelterscrap` /
`directrefscrap` / a user csv label for `sum`;
`ratioeq` / `bound` for `ratio`). _Avoid_: "name" (collides with **Sector**, "the named instance").

**Node-scoped vs set-scoped constraint** — a `massbal` or node-`sum` row balances **one node**, so its
specs render that node as `sector_region_valuation`. A `ratio` row is either **pair-scoped** — relating
**two edges** — or **set-scoped** when its row carries `agg=true`: it pools **many** a-side edges against
many b-side edges, so it has no single numerator/denominator edge. _Avoid_: expecting a single
`sector`/`region`, or populated `num_*`/`denom_*`, on a set-sum or **set-scoped (agg) ratio** row (they
are null there).

**Cross-valuation row** — `ratio` is the **only** constraint family whose two sides may differ in
**valuation** (a `conc` ratio pins `CC = conc·Mass`, so `num_valuation = CC`, `denom_valuation =
Mass`); `massbal` and `sum` rows are strictly single-valuation. This is why `ratio` is the family
whitelisted for mixed-valuation rows. _Avoid_: reading a ratio's two valuations as a bug.

## Framework meta-model vocabulary

**Role** — a framework-owned kind that a case's label is bound to. Two families: the **boundary role** of
an `SB` node, `nature | secondary | final_demand`, and the **valuation role**, `content | gross`
(contained-metal mass vs gross product mass). The entity kinds `Process | Flow | SB` are reserved
framework words, not roles. A case binds its labels to roles in its acc spec ([ADR
0034](docs/adr/0034-roles-are-the-frameworks-labels-are-the-cases.md)). _Avoid_: reading a label (`SEM`,
`CC`) as the concept.

**Relation** — one linear relation over edges with a σ (`Σ αᵢ aᵢ sense c ± σ`) that exists before either
stage consumes it, one object (`pysuto.relation.relation.Relation`) carrying its `sense` (`eq` / `le`,
see **Sense**): a **constraint row** is a relation *emitted* into `G`; an imputation fill is a relation
*solved* for its lone unknown. **Scale totals** and hard bounds are *emit-only* relations — never solved
for a prior. _Avoid_: "constraint row" for the pre-stage object; "recipe" for a single relation.

**Writer scope** — the set of edges a relation source is *permitted* to write. Scopes may overlap;
**cascade order** decides precedence among sources whose scope contains an edge; a write outside
scope is rejected (`PriorLedger`: an error in tests, logged and dropped in production). _Avoid_:
"ownership", "sole writer" — they implied an exclusivity the `conc` fallback never respected.

**Relation source** — what yields a case's relations: a params file (`ParamSource`, its per-pair or
conc-twin reading), the structure of the acc (recipes, Flow-node balances). A source is declared
once and consumed by both stages. _Avoid_: "relation" for the source; "parameter set".

**Fill** — one entry of a cascade: a relation source, a **rule** (`forward`: `b` from `a`;
`backward`: `a` from `b`; `both`), a **writer scope** and an imputer id
(`pysuto.relation.sources.Fill`), run by a `FillImputer` under a σ rule (`trade` or `parametric`).
A case declares it as data, a `FillEntry`; copper's are `trade`, `conc` and `ore_grade`. _Avoid_:
"imputer" for the entry (the imputer is what runs it).

**Case** — one metal's complete declaration: an accounting structure (referenced by registry id), its
feeds, param families, relation sources and cascade, anchor families and σ policies, run by the generic
stage drivers. Declared as a `CaseSpec` named `CASE` in a Python module or file that `--case` resolves: a
path, a dotted module, or a bundled alias. A case also sets the Stage 4 and 6 driver constants it departs
from (`driver_knobs`). Its identity in a run is `CaseSpec.id`. _Avoid_: "model" (the QP, the PIOT),
"study" (a REP analysis), "scenario".

**Application** — everything outside the library that one case needs to run end to end and to report on:
its case declaration, its feed specs, source parsers and feed builders, its sweeps, and its report and
exploration scripts. The library holds the method, generic for every case; an application holds what
carries its metal's name. _Avoid_: "study" (a REP analysis), "project", "scripts" (a folder, not the
concept). See [ADR 0047](docs/adr/0047-pysuto-ships-no-study-case.md).

**Flagged ambiguity — "case-specific code".** In conversation it usually means the
**application**, not the `CaseSpec`.

**Acc spec** — the declaration of one accounting structure: its sectors (each with an entity, and a
boundary role on an SB sector), regions, valuations (each with a valuation role) and times, its supply
and use adjacency, tradable flows and entity-pair accounting rows, and optionally its bound and mask
records ([ADR 0046](docs/adr/0046-an-accounting-structure-is-declared-by-a-spec.md)). _Avoid_: "case
spec" (the typed `CaseSpec`, which references an acc by registry id), "root file" (one of the files a
reader reads).

**Build record** (`build_record.json`) — the file beside a built accounting structure that names
the acc input version (id, version, code) it was built from. _Avoid_: "source record" (collides
with **Source / src edge**, the **source region label** and the params `source` column); "acc
sidecar" (a built structure is not a run).

**Acc ref** — `{id, version}` naming one acc input version, as a sidecar's `parents.inputs.acc`
records it.

**Toy case** — the synthetic **Case** of the fictional metal orium (`pysuto.tutorial.toy`), declared in
code with its own labels (`Gross`/`Content` valuations, `Raw`/`Sec`/`Demand` boundary subs, bound to
roles), generated from a **ground truth** and observed through **synthetic feeds**. _Avoid_: "toy
fixture" (the per-test structures built in the tests), "test case", "demo".

**Ground truth** — the balanced per-edge values a synthetic case is generated from, in every
valuation and year; every `Process` and `Flow` node closes. Never a property of a real case: a
real case has a prior and a posterior, and no truth to score them against. _Avoid_: "true value"
for a copper edge; "truth" for the prior or the posterior.

**Synthetic feed** — a datafeed derived from a **ground truth**: reclassified into its own
vocabulary, noisy at exactly its pedigree CV, with gaps and **planted defects**. It enters Stage 1
like any feed. _Avoid_: "mock feed", "simulated data" (the feed is not simulated, the truth behind
it is).

**Planted defect** — a deliberate error a synthetic case writes into one of its **synthetic
feeds**, beyond the pedigree noise, for the pipeline to find; the case's `reveal()` names it.
_Avoid_: "bug", "outlier" (an outlier is found in data, a planted defect is put there).

## ROW pedigree-penalty vocabulary

**ROW** — *Rest of World*, the residual region aggregate for countries not individually
represented. Canonical sentinel: the string `"ROW"` in a node's `region`.
_Avoid_: "RoW" (normalise to upper-case), "rest-of-world"; "Other regions", a figure's pool of
the regions it does not name, which holds ROW together with the named regions left out.

**`row` mask** — the named edge mask selecting edges that touch ROW on **either** endpoint
(`region_from == "ROW"` **or** `region_to == "ROW"`). A registered input mask.

**row_penalty** — a per-feed boolean (on `DataFeedMetadata`) opting a feed's ROW-touching edges
into the geographic pedigree penalty.
_Avoid_: reading the "row" here as a DataFrame row or a constraint `row_meta` row — it means **ROW**.

**Geographic pedigree penalty** — treating a ROW-touching edge as worst-quality on the
*geographical* pedigree axis (`geographical_correlation = 4`), inflating its σ. Applied as a
per-edge `cv_root_scale` multiplier in Stage 1 (fed edges) and as a quadrature-added `CV_GEO4`
term in the Stage-2 FlatPrior (imputed edges). See
[ADR 0003](docs/adr/0003-row-geographic-pedigree-penalty.md).

**Flagged ambiguity — "row".** Three distinct meanings live in this codebase: (1) **ROW** the
region / the `row` mask; (2) a DataFrame row; (3) a constraint `row_meta` row. In feed, mask and
penalty contexts, "row" always means **ROW** (Rest of World).

## Facility-level vocabulary

Vocabulary of the facility-level extension of the account (REMASS second half; agreed 2026-11-10,
nothing built yet). The country × sector account of the copper case is unchanged — the facility level
nests under it.

**Facility** — one physical site with a mass balance of its own: a mine, a smelter, a refinery. A
facility is a **region** nested in its country (Region = the geography of a node), so a facility
node is an ordinary `(sector, entity, region)` triple and a transfer between two facilities is an
ordinary use edge. _Avoid_: "company" (a facility's owner, not the site); "asset" (S&P's and
Climate TRACE's word — it also names a financial stake); "plant" (excludes mines).

**Remainder facility** — the one facility per country × sector that carries the share of the
country's flow no registered facility covers. _Avoid_: "other", "unallocated"; "residual" (a soft
row's slack).

**Nesting row** — the sum row that ties a country × sector flow to the sum of its facilities' flows,
remainder facility included; the country × sector flow is the reconciled value of the country
account, not an observation. _Avoid_: "control total" (removed, ADR 0006); "scale total" (a soft
sum whose RHS is an input).

**Owner share** — a company's percentage interest in a facility in one year; a facility's shares sum
to 100 %. _Avoid_: "ownership" (the retired writer-scope sense); "stake".

**Operator** — the company running a facility in one year, one per facility, whatever its owner
share. _Avoid_: "owner" for the operator.

**Company-level view** — an aggregation of facility flows by owner share or by operator, the way the
country level aggregates nodes by region; a derived view, never a node of the account. _Avoid_:
"company level" as if companies were nodes; "company account".

**Transport link** — one routed segment of an edge between two facilities, or between a facility and
a border crossing, with a mode (rail, road, pipeline, sea) and, where a statistic exists, a
throughput; a **route** is the ordered chain of links an edge takes. Routing is a derived view on
the reconciled edges, like the PIOT on the PSUT (ADR 0037). _Avoid_: "transport infrastructure"
for one link; "hub" (a region whose trade is re-exported — see re-export netting); "node" for a
port (a port is not a node of the account).

## Review vocabulary

The final review of the codebase: what one audit reads, what it checks and what it records.

**Review dimension**:
One quality aspect a code review checks, with its check and its pass rule: nine general (G1–G9)
and eight specific to scientific software (S1–S8).
_Avoid_: "axis" (a sweep's knob), "criterion" (a limit on a sweep's outcomes), "metric".

**Review unit**:
The code one audit reads at once: a pipeline stage with its driver, diagnostics and tests, or one
shared layer (U1a–U8).
_Avoid_: "step", "component", "module" (any piece of code with an interface).

**Review finding**:
One observation of a review: its location at the audited commit, its review dimension, the
proposed action, and whether acting on it would move a number.
_Avoid_: "issue" (a tracker ticket), "bug" (only a defect is one).

**Standards review / Spec review**:
The two halves of a `code-review` run, reported side by side and never merged: conformance to the
repository's documented conventions, and conformance to the specification.
_Avoid_: "axis" (the skill's own word, which collides with a sweep's **Axis**).

**Check chain**:
A chain rerun on unchanged inputs to show that a code change leaves every stored array identical
to the reference chain's.
_Avoid_: "regression run", "validation run".

**Reference chain**:
The chain of stage runs whose arrays the manuscript reads, and against which every check chain
and measure ticket is compared. A change that moves one of its numbers makes a new reference
chain; after the manuscript's number freeze, no such change lands before submission.
_Avoid_: "final chain" (older REPs), "base" (the base of one comparison only).
