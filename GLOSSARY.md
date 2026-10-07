# GLOSSARY — pySUTO glossary

The project's single glossary. Glossary only — no implementation details.

Pick the bold term; the listed `_Avoid_` aliases are ambiguous and should not be used.
Definitions are code-verified against [acc.py](src/pysuto/acc.py),
[conc.py](src/pysuto/conc.py), [param_relation.py](src/pysuto/param_relation.py), and
[constraints/](src/pysuto/constraints/).

## Domain vocabulary

**Node**:
A `(sector, entity, region)` triple — one point in the accounting structure.

**Edge** ≡ **flow**:
"Edge" and "flow" are **synonyms** throughout pySUTO — one atomic directed quantity between two
nodes, addressed by the 8-tuple
`(time, valuation, region_from, entity_from, sector_from, region_to, entity_to, sector_to)`.
Its value `a` is the magnitude in a single valuation.
_Note:_ lowercase "flow" (= edge) is distinct from the capitalised **`Flow` entity** (a
commodity node — see Entity); disambiguate by capitalisation.

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
process produces a commodity).

**Use edge**:
A pair in `use_pairs`; structurally **Flow → Process**, and **Flow → final-demand category**.
Only `Flow` origins can be "used"; the `tradable` mask is what permits a use edge to cross
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
A Type-2 **soft-sum** constraint (`Σ aᵢ = c + ε`, free cone) whose RHS `c(t)` is an externally
published total — here ICSG production + usage, per year — that a defined edge-set must sum to.
**ROW (Rest-Of-World) totals are not imposed** (ROW-excluded). Internal edge consistency is
*not* a control total — that is the job of hard mass balance.
_Avoid_: "anchor", "coherence soft-sum" (both removed mechanisms).

**Data-category typology** (Pauliuk et al. 2019):
The four parameter kinds, split **extensive** vs **intensive**. *Extensive* (kt magnitudes):
**flows** (individual edges) and **totals of flows** (aggregates, e.g. scale totals).
*Intensive* (per-unit ratios relating flows): **concentration** (category 3) and **efficiency**
(category 4) — these two are the `ParametricImputer` edge-pair relations.

**Prior `a₀`**:
The best-estimate flow vector the QP stays close to: `f_agg` at the first timestep, and the
previous year's optimum thereafter (warm start).

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
**diversification invariant** `CV_root ≤ CV_src` lives here. *Fan-out* = one source edge is
distributed across several root edges (a mapping that classifies only one endpoint).
These are the **source → root** (Mapping 2) shapes; the upstream **raw → source** projection
(Mapping 1, where raw-row *aggregation* and *replication* live) is a separate level — see
*"Two mapping levels"* (Concordance vocabulary, below).

**Aggregation**:
Stage 2 — combining multiple feeds' observations of the *same* root edge by inverse-variance
weights `Ω` into a single `f_agg` / `σ_agg`.

**Imputation**:
Stage 2 — filling root edges that **no feed covered**, via an ordered pipeline of imputers;
every fill is recorded in the `ImputationLog`.

**Imputer**:
One fill method — `Parametric`, `Formula`, `Time`, or `FlatPrior`; a `BaseImputer` subclass.

**Constraint building**:
Stage 3 — compiling `f_agg` / `σ_agg` into the sparse constraint matrix `G` plus per-timestep
right-hand side `c(t)` and `σ_c(t)`.

**Builder**:
One constraint-row generator — `MassBalance`, `Sum`, `Ratio`, `RatioBound`, or `FlowBounds`.

**Solver**:
Stage 4 — the σ-weighted QP solved per timestep with Clarabel (`ClarabelSolver`), producing
the optimised flow vector `a*`.

**Posterior uncertainty propagation**:
Stage 5 — the closed-form posterior standard deviation `σ_a*` and confidence intervals.

**Engine** ≡ **pipeline**:
The orchestrator for a stage — it holds a registered, ordered list of plugins and runs them
(`ImputationEngine`, `ConstraintEngine`). "Engine" and "pipeline" are used interchangeably.

**Register** / **plugin**:
`.register(...)` appends an imputer or builder to an engine; each stage is a plugin pipeline.

**Chaining**:
Running plugins in **registration order** so each consumes the previous one's output. Imputers
mutate `f` / `σ` **in place**, so order matters — e.g. `FlatPrior` is registered **last** as
the final fallback.

**Lp / power-mean combine**:
The single σ combiner used by every stage — `lp_combine({xᵢ}, p) = (Σ|xᵢ|ᵖ)^(1/p)`. `p = 2`
recovers Gaussian quadrature; the project default `p = 1.3` damps fan-in σ inflation.

**Pedigree → CV**:
The Laner et al. (2016) map from five 1–4 quality scores to a coefficient of variation — the
origin of every feed's σ.

**Hard** vs **soft** (constraint):
**Hard** ⇔ `σ_c = 0` — exact, lands in the zero or nonneg cone, no slack. **Soft** ⇔
`σ_c > 0` — carries a slack `ε` in the free cone, penalised by `1/σ_c²` in the objective.

**Cone** (`zero` / `nonneg` / `free`):
The Clarabel cone a constraint row lands in — `zero` (hard equality), `nonneg` (hard
inequality / bound), `free` (soft equality with slack `ε`).

**Warm start**:
Seeding the prior `a₀(t)` for a timestep with the previous year's optimum `a*(t−1)`.

**Run** / **run-id**:
One persisted execution of a stage, keyed `{stage}_{YYYYMMDD_HHMMSS}_{uuid}`
(`stage ∈ {feed, imp, const, opt, uncprop}`).

**Registry**:
A per-stage store of runs. The **input registry** versions the input *bytes*
(feed / acc / params, hardlink-deduped); each **output registry** records a run plus its
sidecar.

**Sidecar** (`metadata.json`):
A run's self-contained provenance record — `parents.inputs` (input-registry refs) and
`parents.upstream` (parent run-ids).

**Diagnostic** (diag):
A per-stage plot suite that consumes a run's output (`AggDiagnostic`, `SolverDiag`, …); never
imported into the data layer.

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

**Block path** vs **legacy path**:
*Block path* = `_solve_all_via_blocks`, used when the `ConstraintRun` carries a
per-t partition (`G_blocks`); produces year-local output. *Legacy path* =
`_solve_all_legacy`, used when `G_blocks` is empty; produces full-`[E]` output.
The npz `cols_t` key discriminates the two on load.
_Avoid_: "per-t solver" / "global solver" (both solve per timestep — the
distinction is *partitioned* vs *full-G*, not per-t vs not).

**No cross-time coupling** (invariant):
The property that every constraint row's non-zeros lie within a single time
bucket, so `G` partitions cleanly into `G_blocks[t]`. The Stage-3 partition raises
if this is violated.
_Avoid_: "time independence".

*Example.* "Why does the 1995 file's mass balance look enormous?" — "That's a
pre-fix block run: the file holds 1995's **period slice** plus **cross-year
placeholders** for every other year, and the residual was taken over the full
`G`. Re-solve and it stores only the period slice; the **complete `a***`** is
reassembled by the diag when it needs the whole picture."

## Concordance vocabulary

**Two mapping levels** — the feed pipeline composes *two* independent raw→root mappings, each of
which can be 1:1, fan-in, or fan-out. The fan-in/fan-out terms below name the **second** level only.

- **Mapping 1 — raw rows → source edges.** Defined by the concordance **variable map**
  (`variable_map` sheet), which is *auxiliary-conditional* (e.g. on `Supply_Use`): it projects raw
  variables onto the source 8-tuple slots (`src_sector_from/to`, `src_entity_*`, `src_region_*`,
  `src_time`, `src_valuation`), so the *same* raw variable lands in a *different* endpoint depending
  on the auxiliary category (`Process` → producer/`from` for Supply rows, → consumer/`to` for Use
  rows). This is what **imposes directed supply-use structure** onto raw data that may be
  **production-based** (ICSG) rather than already supply-use shaped (BACI). Cardinality: **1:1** or
  **many:1** (*raw-row aggregation* — several raw rows resolving to the same source edge have their
  values **summed**). A single raw row resolving to *many* full-value source edges is **replication**
  (the anti-pattern below), not a valid shape.
- **Mapping 2 — source edges → root edges.** The WeightMatrix. **Fan-out** and **fan-in** (below)
  live here. Mass-preserving iff each source edge's outgoing weights sum to 1.

**Fan-out** — *one* source edge distributed across several root edges, arising when a raw row
classifies **only one endpoint** (the other endpoint is left open as `"<any>"` and resolved by
normalised weights). Mass-preserving: the source value is *split* across the fan, weights sum to 1.

**Replication (anti-pattern, not a concordance shape)** — the failure mode where a single raw
row that classifies only one endpoint is materialised into *many distinct source edges* (one per
matching value of the open endpoint), each carrying the **full** raw value. This is **not**
fan-out; it inflates total mass by the fan cardinality. Distinguished from fan-out by counting
*distinct src edges per raw row*: fan-out = 1 (→ many roots); replication = many. See
[ADR 0002](docs/adr/0002-single-endpoint-rows-fan-out-not-replicate.md).

**Unweighted fan** — a fan-out whose legs carry no weighting information (no trade shares, no
user weights — every leg still at the uniform default). Such a fan is **not distributed**: an
even 1/N split asserts a fabricated partner pattern at the observation's confidence, whereas the
honest state of knowledge is "total known, split unknown". The raw value stays undistributed
(excluded from the feed's projected mass). A fan that degenerates to a single leg is an exact
match, not an unweighted fan. _Avoid_: reading "no imports" as "no weighting information" — a
fan weighted **all-domestic** because BACI observed the region and recorded no imports carries
weights and is not an unweighted fan ([ADR 0018](docs/adr/0018-import-free-fans-distribute-domestically.md));
the doctrine bites only where the split is genuinely uninformed. See
[ADR 0008](docs/adr/0008-unweighted-fans-are-not-distributed.md).

**Qualifier fan** — a Use fan whose source variable names its **input product** (e.g. smelter
production *from scrap*, refinery production *from matte*). Only root legs whose supplier sector
matches the named product belong to the fan; other incoming edges of the consumer node are
out-of-scope legs that the open (`"<any>"`) endpoint would otherwise wrongly absorb — and onto
which route-keyed weights of *other* products can bleed. A qualifier fan with **zero** matching
legs marks a structurally impossible mapping (the named product does not feed that consumer in
the accounting structure) and is a hard error, never a silent fan. See
[ADR 0008](docs/adr/0008-unweighted-fans-are-not-distributed.md).

**Mapped self-trade** — a bilateral src row whose two endpoints fold onto the **same root region**
under the region map (Hong Kong→mainland once both map to "China"; any small-country pair inside
ROW). Its root address is indistinguishable from the **domestic** edge, yet its value is a tiny,
customs-precise subsample of a production-scale transfer — an inconsistent observation, not a
measurement of that edge. Bilateral feeds are therefore **scoped to cross-region edges** (their
edge scope drops rows resolving to domestic addresses), see
[ADR 0014](docs/adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md). _Avoid_: reading a
mapped self-trade row as a domestic observation — the china-matte case put 0.028 kt of gross mass
next to 5,678 kt of copper content on one edge.

**Reporting territory** — the real-world territory a feed's rows describe, which is *not* the same
as the **source region label** those rows carry. A feed may rename a territory mid-series (ICSG's
`Taiwan` → `Taiwan, China` from 2013) or publish historical joint reporting under a joint label
(`Belgium-Luxembourg` to 2013), so one territory wears two labels across the series and, in the
handover year, two labels on the same cell. Two source labels denote one territory exactly when the
region map assigns them to the same **real-country root**; `ROW` is excluded, being a bucket of 48
territories rather than one. Anything computed *before* the region map is applied — a parse rule, a
param study reading `values.csv` — must therefore group by territory, not by label. _Avoid_: reading
a handover-year pair as two measurements. It is one measurement republished, so adding the two
doubles the cell, and a rule keyed on the label silently skips the territory whose two series are
labelled differently ([ADR 0032](docs/adr/0032-one-territory-one-label.md)).

**Vintage** — the release a feed row came from, when a source publishes the same series
repeatedly (ICSG's 1995, 2005, 2014, 2023 and 2025 yearbooks; BACI's annual releases). A later
vintage **restates** an earlier one, so rows sharing a key are resolved by keeping the newest.
That rule is only sound *between* vintages: two rows of **one** vintage sharing a key are two
different **streams** of the same quantity — a source category folded onto another by a rename,
such as ICSG's `Low-grade` smelter feed folded into `Primary` — and they **sum**. _Avoid_:
resolving a same-vintage collision as a restatement. One stream then silently replaces the other
rather than adding to it, and which one survives depends on row order
([ADR 0036](docs/adr/0036-a-renamed-type-is-a-second-stream-not-a-restatement.md)). _Avoid_ also:
reading a vintage as a *time* coordinate — it labels the publication, not the year observed; one
vintage carries many observation years.

**Boundary topology** *(referenced by [ADR 0001](docs/adr/0001-replace-coherence-sum-eq-with-bilateral-anchoring.md); to be expanded)* —
the open System-Boundary (SB) endpoints where flows terminate, so prior disagreement there does
not create feasibility conflicts.

## Accounting-structure invariants

**Sector → entity is 1:1** — every *concrete* root sector maps to exactly one entity kind
(`Flow` | `Process` | `SB`), so entity is functionally determined once the sector is known.
**Caveat:** an **open (`"<any>"`) sector does not determine an entity** — an unclassified endpoint
can match producers across several entities (e.g. a real `Flow` source *and* the `SB`
system-boundary/import source). Concordance code must therefore make the entity `"<any>"` whenever
its sector is `"<any>"`, not pin it to a concrete edge entity. See
[ADR 0002](docs/adr/0002-single-endpoint-rows-fan-out-not-replicate.md).

**System-boundary category (`entity_sub` on SB nodes)** — the sub-classification carried by every
`SB` node, naming *what kind* of boundary crossing it is. In the copper model: `Nature` (primary
extraction entering the system — crude ore, `CO`), `FD` (final demand leaving the system — the
final-demand categories `FU` / `MESC` / `USC`), `SEM` (secondary / residual material — `Scrap`
recycling inputs and `Slag` / `Tailings` / `WR` / `PSC` residual outputs). `entity_sub` is
meaningful only on `SB` nodes (`Flow` / `Process` nodes leave it empty). _Note:_ `FD` is
structurally an SB-*output* (`entity_to == "SB"`) yet semantically belongs to the **use** side — a
commodity consumed by final demand, with a real `Flow` commodity on its origin endpoint; only
`Nature` and `SEM` are pure raw/secondary boundary crossings with no `Flow` endpoint.
_Avoid_: conflating `entity_sub` with **entity** (only `Process` | `Flow` | `SB`); and avoid
"boundary type" as a loose synonym for it — that phrase is ambiguous between this `entity_sub` class
(Nature/SEM/FD) and the **sector** of the SB node (`CO`, `Slag`, `FU`, …, the finer per-commodity
grain), so name the grain you mean.

**Satellite flow** — an edge of the accounting structure that carries a calculated value but is
excluded from its source node's mass balance by the `massbal_scope` mask (acc `masks.csv`). In
copper: `Mining → WR` (waste rock), mass `k·Ore` from the stripping ratio, copper content zero; its
mass enters Mining from nature without an input edge (unused extraction lies outside the crude-ore
feed), so the reported PSUT does not balance Mining in gross mass by exactly the waste rock. See
[ADR 0038](docs/adr/0038-waste-rock-is-a-satellite-mass-flow.md). _Avoid_: "residual output" (a
residual output closes its node's balance; a satellite never does) and "slack".

## Coverage & val vocabulary

**Value (`val`)** — *a value of an edge*: the post-imputation `f_agg` on a root edge.
Every root edge has one (imputation fills all NaN), so "having a val" alone says nothing
about data quality — see **val-coverage**.

**val-coverage** (reported as the `val_pct` column of the stats/coverage diagnostics) —
the relative number of edges in a subset that carry a **real** val, where "real" follows
the imputation-log *method* rule (`has_val`): flat-prior / `flat_prior_noisy_zero` default
fills are **not** real; observed values — *including a genuine `0.0`* — and
fan / time / parametric / formula / **network** (including a bounds-**clipped** fill, below)
imputations **are**. _Avoid_: deciding val-coverage from `f == 0` — a flat-prior fill is `0`
but so is a real observed zero (and a clipped-to-0 network fill), and the three are
distinguished by the log method, not the magnitude.

**Bounds-clip (in-imputer) — `network_*_clipped` fill.** The NetworkImputer floors every
fill to the edge's `[lower, upper]` bounds **before** it propagates through the recipe/
balance fixpoint, so a negative residual (e.g. `Ore→SXEW = Mining→Ore − Ore→MF` in a region
with little/no SXEW) cannot cascade and drag downstream sums below their true values. The
clipped value is a **saturated estimate**, not a precise bound: σ is widened to the clip gap
and the fill is tagged `network_*_clipped`. A clipped network fill still **counts as real
val-coverage** — it is a network inference, and its low confidence lives in the widened σ,
not in coverage. _Distinct from_: (1) the engine's **post-return bounds clip**
([impute/engine.py](src/pysuto/impute/engine.py)) which warns + clips *any* imputer's
out-of-bounds fill — the in-imputer clip makes it a no-op for network fills; (2) the **σ-cap**
(`_capped`, the `cv_cap` ceiling on σ — caps the *uncertainty*, not the value).

**Coverage** (glossary, the `agg_result.coverage` boolean) — *did any feed observe this
edge?* — a **pre-imputation** notion, strictly **narrower** than val-coverage: it credits
**only** feed observations, never any imputation. _Avoid_: conflating the two — the stats
column is `val_pct` (not `coverage_pct`) precisely because val-coverage also credits
non-flat-prior imputations that feed-`Coverage` excludes.

## Defect, residual & violation vocabulary

**a_prior (warm-start prior)** — the per-**period** prior mean the optimiser uses for year `t`.
Under `prior_mode='warm'` (the `ClarabelSolver` default) it is genuinely the *previous* year's
posterior, aligned by **edge template** `(from_id, to_id, valuation)` since each year occupies
disjoint global columns: `a_prior(t0) = f_agg` (the cold seed), `a_prior(t) = a_post(t-1)` for
`t > 0`. Under `prior_mode='cold'` every year uses its own cold seed `f_agg(t)`. Stored as the
per-period cold seed on `ConstraintRun.timestep_data[t]["a_prior"]` and as the *actually-used*
prior on `OptimResult.a_prior`; `ConstraintRun.a_prior_global` reconstructs the global cold
prior. _Avoid_: reading `ConstraintRun.a0` as the per-year prior — `a0` is the legacy global
(time-replicated) cold vector; the year's prior is `a_prior`. _Historical bug_: the block path
once wrote the carry-forward into the same disjoint columns it read, so the warm chain was a
silent no-op and every year fell back to `f_agg` — see [ADR 0009](docs/adr/0009-warm-start-prior-chain.md).

**Two warm-starts (not the same thing).** `prior_mode` (warm/cold) is the **modelling** prior
above. `reuse_factorization` is a purely **numerical** warm-start — reusing Clarabel's KKT
factorisation across timesteps via `solver.update()`. _Avoid_: conflating the two; the local
that selects KKT reuse was historically also called `warm_start`.

**Defect** — the per-**edge** prior departure `a* − a_prior` (solved value minus the warm-start
prior, `a_prior` above). Plotted by SD-T-07 (the a-space rocket, `a_prior(t)` vs `a*`), ranked
by **objective share** in SD-R-13 / SD-R-14 / SD-R-15a (the edge-defect family), and tabulated
row-level by `SolverStatsDiagnostic` (`edges_<year>.csv`, columns `defect` / `defect_abs` / `z`).
Its standardised form is `z = (a*−a_prior)/σ_a`. _Avoid_: calling the defect a "residual" — that
term is reserved (below).

**Residual** — the per-**constraint-row** delta `G·a − c` (`OptimResult.residuals`; its hard
mass-balance maximum per year is the middle panel of SD-R-11). Its feasible range depends on the
row's cone: **≈ 0 on hard *equalities*** (`zero` cone — mass balance); **`≤ 0` on hard *bounds***
(`nonneg` cone — ratio/flow bounds, since feasibility is `G·a ≤ c`), where a *negative* residual
is healthy **slack** (distance below the limit), not a violation, and only a *positive* residual
is a breach; **`= ε`, the slack, on a soft (free) row** (see README §"Cone order": `Gx − ε = c`).
_Avoid_: (1) reading a large-negative residual on a `nonneg` bound as a violation — it is slack
(a misread that once produced a phantom "inverted bound" diagnosis); (2) conflating with
**defect** — residual is per-constraint, defect is per-edge.

**Structural lower bound** — the per-edge `l` of the accounting structure (`acc.bounds.l`): 0 on
every edge except the signed stock-change edges (`Cathode→USC`, `Cathode→MESC`, `CP→PSC`), whose
`l` is −1e6. Since [ADR 0039](docs/adr/0039-flow-lower-bounds-are-enforced-in-the-qp.md) each
`l = 0` enters the QP as one hard `nonneg` row `−a ≤ 0` (type `bound`, variant `lower`). _Avoid_:
reading `ConstraintRun.l` as the structural bound — Stage 3 overwrites it with the prior box and the
**zero pin** (`l = u = 0` on no-data edges, [ADR 0013](docs/adr/0013-no-data-flat-prior-anchor.md)),
and those boxes, like every upper bound `u`, reach only the feasibility pre-check. Before ADR 0039
no bound reached the QP, so `l ≤ a ≤ u` in older REPs and the README describes the probe, not the
solve.

**Forced vs discretionary deletion** — the split of a node's prior-to-posterior mass loss into the
part no optimiser could have avoided and the part it chose. Copper leaves a mine node only as
domestic smelter intake `U` or as export mass `X` times a grade the hard cone caps at 0.35, so
`cap = 0.35·X + U` bounds its discharge and `Σ max(0, P − cap)` is **forced** — arithmetic, not
optimisation. The remainder is **discretionary**: the solver could have kept it and priced the
disagreement elsewhere. The distinction says which lever can move a number: σ, anchor placement
and pricing reach only the discretionary part, while the forced part moves only when `X`, `U` or
`P` themselves change ([ADR 0029](docs/adr/0029-the-concentrate-grade-stays-global.md)). _Avoid_:
reading a forced term as a defect to be relieved. Where `P` is phantom — production the source
never had — deleting it is the *correct* answer, and the repair is to remove the input rather than
loosen the constraint
([ADR 0035](docs/adr/0035-the-yearbooks-label-is-not-on-the-row-it-owns.md)). _Avoid_ also: reading
a forced term larger than the realised deletion as a contradiction — `cap` is evaluated at
prior-level intake, and a region that substitutes scrap or imported anode escapes a ceiling the
formula cannot see.

**Constraint table (prior/post)** — the harmonised **row-level, all-rows** per-`constraint_id`
table pair (hard + soft rows; `type` / `cone_type` columns carry the family for filtering):
`constraints_prior_<year>.csv` at the constraint stage (`ConstStatsDiagnostic`) and
`constraints_<year>.csv` at the solver stage (`SolverStatsDiagnostic`). **`lhs_prior` = G·a0**
means the same thing in *both* files; **`lhs_post` = c_star = G·a*** exists only post-solve.
`residual` / `z = residual/σ_c` (ranked by **objective share** in SD-R-13 / 14 / 15b-c) are the
*stage's achieved* residual — `lhs_prior − c` at the constraint stage, `lhs_post − c` (= the slack
`eps` on a soft row) at the solver stage; `z` is NaN on hard rows (`σ_c = 0`). _Avoid_: reading a
large `|z|` as a failure on its own — a soft row is *meant* to absorb disagreement; `|z|` just
measures it in units of the row's tolerance `σ_c`. _Avoid_: reading the solver-stage `residual`
on a **hard** row as the **residual** above — `c_star = c` there by construction, so the column
is identically 0; the hard residual lives in `OptimResult.residuals` (frame column
`resid_solver`). _Avoid_: an unqualified `lhs` column — the files qualify every LHS as
`lhs_prior` / `lhs_post` so no column changes meaning between stages. _Note_: `ConstDiagnostic`
plots the same prior quantity under the opposite sign (`eps0 = c − G·a0`); the table follows
the reserved `G·a − c` sign.

**Objective share** — one item's `z²` (an edge's defect or a soft row's residual, standardised)
divided by the `Σz²` of its family or of the whole run. The QP objective *is* `Σz²` over edges plus
soft rows, so the share is what that item cost the optimiser. Ranked per **identity** over all
years by `SolverDiag` (SD-R-13 by sector key, SD-R-14 top identities, SD-R-15 z profiles) and
tabulated in `ranking_<family>.csv`; three families: edge defects, sum-row residuals (`type =
sum`, all soft), soft-ratio residuals (`type = ratio`, free cone). _Avoid_: "deviation" (say defect
or residual), "violation" (a soft row is meant to absorb disagreement — see Constraint table),
"fit".

**Objective decomposition** — the per-year split of `Σz²` into the three families, drawn with the
max hard mass-balance `|G·a* − c|` (SD-R-11). _Avoid_: reading the Clarabel `objective` in the opt
sidecar as this quantity — it is `½xᵀPx + qᵀx` without the constant term and is negative.

**Identity (cross-year)** — the key under which one edge or one constraint row recurs across
years. For an edge: the **edge template** `(from_id, to_id, valuation)`. For a row: the tuple of
its `row_meta` columns other than `constraint_id` and `time` — `(variant, sector, region,
valuation, …)` node-scoped, the `num_*`/`denom_*` endpoints pair-scoped, `(variant, region,
valuation)` set-scoped, with `cone_type`/`bound_dir` telling a pair's soft row from its two band
rows. Always read from columns; a row whose columns do not single it out within its year has no
identity (it still counts in every total). _Avoid_: stripping the year off a `constraint_id` (see
`constraint_id`: a key for reading, never parsing).

**Retention** — the share `Σa* / Σa_prior` computed **by category**: the posterior sum over the
edges of one category in one year (a sector, a node, a region, an SB class — always name the
category) divided by the prior sum over the same edges. It is a dimensionless share of the prior
mass: `1` keeps the prior scale, `< 1` is deletion, `> 1` inflation (second row of SD-R-08 / 09 /
10, per sector and year). The complement of the deletion in **forced vs discretionary deletion**.
_Avoid_: "adaptation", "recovery", "survival"; a retention quoted without its category and year;
a retention on a signed category whose prior sum is near zero (`MESC` / `USC` / `PSC`) — the ratio
is undefined there, so quote the mass moved `Σ|a* − a_prior|` beside it.

## Stock-change vocabulary

**Exchange location** — a region in which ICSG reports exchange stocks: the LME warehouse
countries by location, COMEX as the United States, SHFE as China (16 regions in the feed). It is
the destination of a `Cathode(i)→MESC(j)` leg, not a property of the origin. _Avoid_: "hub"
(reserved for the merchant / re-export hubs of [ADR 0027](docs/adr/0027-re-exports-are-re-attributed-at-feed-time.md)),
"warehouse region".

**Net-stock seed** — the prior of a `MESC` / `USC` leg that no feed reports: value 0, held there
with `σ = max(0.01·|co-located Cathode(i)→FU(j) flow|, 0.3 kt)` — the same pricing as a
feed-backed level ([ADR 0040](docs/adr/0040-stock-change-legs-are-held-at-their-prior.md)). A stock
leg's prior is a reported value or this seed; no imputer manufactures one. _Avoid_: reading a seeded
0 as a reported zero; "throughput-scaled band" (the retired p95 pricing).

**Sink** — an edge family whose σ makes its penalty negligible against the mass it can absorb, so
a contradiction elsewhere discharges into it instead of surfacing as `z` where it belongs. The
signed stock legs are the structural candidates (they alone may be negative, ADR 0039); the
exchange-location `MESC` legs at σ ≈ 553 kt were one (Round X). Relates to **forced vs
discretionary deletion**: a sink converts a forced contradiction into a discretionary move.
_Avoid_: "slack" (the soft-row residual), "buffer".

## Feed valuation & trade-share vocabulary

**Valuation — `CC` vs `Mass`.** Every edge exists in two valuations. **`Mass`** is the gross product
mass (tonnes of the commodity as traded/handled). **`CC`** (*copper content*) is the contained-copper
mass (tonnes Cu). They are linked per sector by the **concentration coefficient**.

**Concentration coefficient (`conc`).** The Cu mass fraction of a sector's commodity (e.g. Cathode
0.9975, Anode 0.94, Matte 0.525, Concentrate 0.275; from the `conc` param). The `conc_<sector>_supply`
relation enforces `CC = conc · Mass` on each node pair, so the two valuations of one edge are not
independent. The param's `value_nominal` drives the **soft** `CC = conc·Mass` ratio row; its
`[value_min, value_max]` drives a **hard grade band** `value_min ≤ CC/Mass ≤ value_max`
(`ratio`/`nonneg`). **Invariant: each commodity carries exactly one such band** — `Scrap` is one
mixed-grade commodity with a single wide band `[0.40, 0.98]`, nominal 0.70, *not* two grade classes
(two simultaneous hard bands on the one `Scrap/SB` node intersected to ∅ and collapsed all scrap;
[ADR 0012](docs/adr/0012-scrap-one-mixed-grade-commodity.md)). _Avoid_: reading the `conc_*_supply`
z-violation as a "bug" — it fires whenever the two feeds populating the CC and Mass sides of an edge
disagree.

**Smelter scrap · direct-to-refinery scrap.** The model's two `Scrap` inflows, named by where the
scrap enters: **smelter scrap** feeds `Smelting` with the primary concentrate and leaves as matte;
**direct-to-refinery scrap** feeds `CFR` with the matte and leaves as anode. ICSG reports two
*output* classifications — **secondary smelter production** (blister/anode made from scrap) and
**secondary refined production** (refined copper made from scrap by *either* route) — and the
second contains the first. Direct-to-refinery scrap is therefore secondary refined *minus* the
smelter route; in fifteen countries the two published figures are identical and the direct route
is nil. **Direct melt** — scrap fabricators use without refining — is a third quantity, outside
this model's boundary ([ADR 0030](docs/adr/0030-secondary-smelter-copper-enters-the-chain-once.md)).
Since [ADR 0031](docs/adr/0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md) both
inflows are stated to the model **twice**: as a **level** (a scrap level anchor, where ICSG reports
one) and as a **share** (the `scrap_share` band, every region). Before it only the share existed.
_Avoid_: booking secondary refined production as a second inflow beside smelter scrap (the copper
enters twice, and the anode anchor discharges the surplus by deleting concentrate); reading
"secondary" as the model's SEM input class when ICSG means an output classification.

**Recovery efficiency (`eff`).** The copper-recovery fraction of a Process, in **`CC`** (from the `eff`
param) — the sibling coefficient to `conc`. A process's recovered Flow product is `eff · Σ(all CC
inputs)` in CC; the `(1 − eff)` complement is the residual that leaves as the **SEM** residual output.
`eff` acts on the **sum of all CC inputs — primary plus the `Scrap` SEM-secondary** (Smelting/CFR
consume Scrap, so `product = eff·(primary + scrap)`). Both the imputer prior and the `eff` **ratio
constraint** now pool that *same* set, so there is no longer an `eff·Scrap` gap between them — see
[ADR 0005](docs/adr/0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md). The pooling is
**declared explicitly in the eff params file** by the `agg` flag on the row (a reader sees it in the
CSV), never inferred from code. The input-tier distinction is retained only for
**NaN-propagation** in the imputer, *not* for what gets summed. There are **three** tiers (all pooled
into `Σin`): **primary** (same-region Flow / SB-Nature inputs — a missing one *blocks* the forward
fill while derivation is still possible; once the traversal has converged, a missing primary counts
as 0 *provided ≥1 primary is known* — parallel technology routes such as CP's CER+CSXEW, where an
underivable leg means the region lacks that route; a recipe with **no** known primary stays blocked,
[ADR 0010 amendment](docs/adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md));
**SEM-secondary** (`Scrap` — missing counts as 0); and **bilateral-import** (cross-region
`Flow→Process` legs — missing counts as 0, non-blocking, [ADR 0010](docs/adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md)).
The same cross-region-as-0 rule applies to the Flow-node balance (a missing cross-region leg is
counted as 0, never written). _Avoid_: conflating `eff` (a CC recovery fraction) with `conc` (a
Mass→CC fraction) — they are different coefficient families.

**Process recipe.** A Process node viewed as a mass-closing transformation in CC: its **pooled inputs**
(primary + `Scrap` secondary) → one **Flow product** (`eff·Σin`) + a **SEM residual output**
(`(1−eff)·Σin`). The unit of reasoning for the network-traversal imputer
([ADR 0004](docs/adr/0004-process-recipe-traversal-imputer.md)). **Mining is the exception**: it has
*no* `eff` row (eff defaults to 1.0, so CC is conserved CO→Ore) and its `WR` residual follows the
**stripping ratio** (below) on Mass (`WR_Mass = stripping_ratio·Ore_Mass`), not the generic
`(1−eff)·Σin` SEM-residual rule. _Avoid_: confusing the recipe's **SEM residual-output** (Slag / Tailings / WR / PSC,
reached via a *supply* edge) with a **SEM secondary input** (Scrap, reached via a *use* edge) — both
carry `entity_sub == "SEM"`, and only the edge direction distinguishes the two roles.

**Stripping ratio (`stripping_ratio`).** The mass of waste rock (`WR`) stripped per unit of `Ore`
extracted at a Mining process — `WR_Mass = stripping_ratio · Ore_Mass`, a **Mass→Mass** ratio between
two **outputs** of Mining (default 2.0). It is *not* a recovery efficiency: not in `CC`, not a
recovery fraction, and `> 1`. It lives in the **eff parameter file** as its own `param_type`
(`stripping_ratio`, distinct from `efficiency`), making that file a store of **process input/output
ratios** of which recovery-`eff` and `stripping_ratio` are two families. It survives **only as the
soft `ratio` constraint** `WR − stripping_ratio·Ore = 0` (kept free/inert): the NetworkImputer fills
WR as the **mass-balance residual** `WR_Mass = max(0, CO_Mass − Ore_Mass)` instead — and on the
allocated-ore basis (below) that residual is ≈ 0 by construction, as allocated data carries no
separate waste-rock mass. _Avoid_: calling it an `eff` — `eff` is strictly a CC copper-recovery
fraction (above); the two only share a file. _Avoid_: reading the imputer's WR fill as
stripping-ratio-driven — that rule was replaced 2026-06.

**Allocated ore mass.** The `crude_ore` feed's ore Mass is S&P **price-allocated**: a
poly-metallic mine's physical ore is split across its metals by price share, so the copper-
attributed mass is *smaller than the physical ore mined*. `Ore`/`CO` Mass edges carry this
allocated mass, and everything derived per unit of it (grade, WR residual) inherits the basis.
_Avoid_: comparing it to physically mined/milled tonnages.

**Recovered vs in-situ grade.** Two grade notions per region-year: **recovered grade** = metal
produced ÷ allocated ore (`1/factor`, the raw S&P number); **in-situ grade** = copper contained in
the mill feed ÷ allocated ore = recovered ÷ recovery-`eff`. The model's `CO`/`Ore` **CC edges are
in-situ** (production ÷ eff, matching the recipe's `(1−eff)` Tailings copper), so their grade
rows use the **in-situ** value — region×year-resolved, owned by the `ore_grade` params, not the
global `conc` family
([ADR 0015](docs/adr/0015-in-situ-ore-grades-on-allocated-ore-basis.md)). The same artifact also
owns the region×year×route **tailings** conc (`(1−eff)` recipe copper over the process
Mass-balance residual, per MF/SXEW route — ADR 0015 amendment); the global `conctailingsuse`
row is retired. On the allocated basis
these are *not* physical mill-head grades (Australia ~2–3 % vs ~0.8 % head grade). _Avoid_:
applying a recovered-basis grade to an in-situ edge (a systematic 1/eff error), or citing these
grades against literature head grades without converting.

**Concentrate grade.** The Cu mass fraction of copper concentrate — the `Concentrate` case of the
concentration coefficient, one **global** row (0.275, band [0.20, 0.35]). Unlike an **in-situ ore
grade** it is a *direct assay*: no allocation basis, no `eff` division, so it may be read against
industry figures (20–35 % Cu) as it stands. It is deliberately **not** region-resolved the way ore
grades are: measured over the region-years whose books are physically possible, the between-region
spread is 0.0117 against the 0.0375 σ the single row already declares, and the residual that looks
like a regional grade is a gap in the recorded export **mass**
([ADR 0029](docs/adr/0029-the-concentrate-grade-stays-global.md)). _Avoid_: reading a region's
implied `(P−U)/X` as its grade — where export mass is under-recorded that ratio absorbs the missing
tonnage and returns values (0.6–0.9, and negatives elsewhere) that are not grades at all.

**Grade band vs grade σ.** A `conc`-family params row's `value_min`/`value_max` are read **twice**:
as a **hard cone** by `RatioBoundBuilder` (`value_min·Mass ≤ CC ≤ value_max·Mass`, per edge per
year — a physical plausibility limit, no slack) and, when `unc_type = "min_max"`, as the **σ** of the
*soft* ratio row via `sigma_from_row` — which also sets the `ParametricImputer` prior and the
`NetworkImputer` twin-map σ. Those are two different statements, and conflating them means changing
a limit silently re-prices three things. The `ore_grade` grade rows therefore carry
`unc_type = "sd"` with an explicit `value_sd` (ADR 0028); tailings rows still use `min_max`. The
other `unc_type` mode is `pedigree` (σ = nominal · Laner total CV). _Avoid_: calling the band "the
σ" — a band is the wall, σ is the price of approaching it; and narrowing a `min_max` band when you
only meant to move the wall.

**Grade carried past the ore feed.** The `crude_ore` feed ends before ICSG production does
(2022 vs 2024). For a region-year with production but no ore mass the `ore_grade` generator
carries the region's **last observed in-situ grade** forward (`src=carry`, band ±20 % instead
of ±10 %); the NetworkImputer then derives the whole `CO→Mining→Ore→…` Mass chain from that
year's production through the grade, and the scale totals and grade rows follow on their own.
**The assumption is on the grade, never on the mass**: ore mass tracks production
([ADR 0023](docs/adr/0023-ore-grade-carried-past-the-crude-ore-feed.md)). _Avoid_: carrying
the ore Mass itself (holds tonnage flat whatever was produced, and disagrees with the grade), or
running any Mass extrapolator *after* the network cascade (fills the boundary leg only, leaves
the chain behind it at flat-prior zero and drops its anchor rows).

**⚠ The grades in use are internal, not external.** Every `ore_grade` row the model consumes is
*derived* from ICSG production and crude_ore allocated mass (ADR 0015). The **external** S&P
factor series (`data/input/params/ore_grade/current/all_integrated_est_fac_2024-07-04.csv`, the
basis of `ore_grade` **v1** from 2026-05) enters only as a fallback where ICSG production is
absent (129 rows) — even though it carries real 2023 values for every region the carry-forward
touches. Measured 2026-09-10: its *level* differs from the internal grade by up to ±8 % at the
2022/2023 seam (Mexico +7.6 %, Russia +6.0 %, Chile −2.7 %), its year-on-year *trend* by
< 0.5 %. Using that trend for carried years (`g(t) = g(t−1)·factor(t−1)/factor(t)`) is an open,
sub-percent option recorded in ADR 0023; using its level would reintroduce the basis jump
ADR 0015 removed. _Avoid_: describing the ore-grade series as external data, or comparing it to
the S&P factor without converting bases (`1/factor` is recovered-basis, the model is in-situ).

**Feed semantics — ICSG vs BACI.** **ICSG** publishes production/usage statistics in **copper content
(`CC`)**; its `Use` rows are **single-endpoint** (consumer known, producer open) ⇒ producer **fan-out**.
**BACI** publishes **bilateral gross trade** in **`Mass`** (fully classified X→Y; no self-trade rows *at
source* — but region folding creates **mapped self-trade** at root, so the feed is scoped to
cross-region edges and never informs a domestic leg,
[ADR 0014](docs/adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md)). On a trade edge the ICSG-CC
value is therefore a *fan artifact* and the BACI-Mass value is the real observation — for a leg the
**re-export** netting did not change; a re-attributed leg is part observation, part inference, and carries
the compounded σ ADR 0027 defines. _Avoid_: treating a
per-edge ICSG-CC trade value as a measurement — it is the national usage split across producers by the
fan weights.

**Trade-share fan weighting** ([ADR 0002](docs/adr/0002-single-endpoint-rows-fan-out-not-replicate.md)
option D) — weighting ICSG's producer fan by BACI bilateral import shares instead of uniformly, so ICSG-CC
inherits BACI's geography. Since [ADR 0027](docs/adr/0027-re-exports-are-re-attributed-at-feed-time.md)
those shares are read from the **netted** BACI version, and a group whose imports the netting exhausted is
**withheld** (left undistributed) rather than fanned all-domestic: its `Σ imports = 0` is constructed, not
observed, so ADR 0018's condition does not hold. The domestic leg (X==Y) carries the residual `usage − Σ imports` (in CC units);
the split is computed in copper content via `conc`. Applies only to the four **traded** ICSG-fanned
sectors — Cathode, Concentrate, Anode, Matte. The residual also governs the **import-free** case: where
BACI *observes* the consumer region (46 of 47 — all but **Taiwan**, trade-blind because BACI 490 "Other
Asia, nes" folds into ROW, [ADR 0014](docs/adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md)),
`Σ imports = 0` is a real observation of domestic sourcing and the residual `usage − 0 = usage` gives an
all-domestic fan ([ADR 0018](docs/adr/0018-import-free-fans-distribute-domestically.md)); where BACI is
blind, zero imports carries no information and the fan stays undistributed. **The domestic leg is
bounded by the region's own production** of the product for the sectors in
`BOUND_DOMESTIC_SECTORS` (Concentrate; [ADR 0024](docs/adr/0024-fan-domestic-leg-bounded-by-own-production.md)):
where the residual exceeds what the region produced, the excess is spread over the import legs in
BACI proportion (usage conserved) — a region cannot supply itself more than it makes, and the hard
Flow-node balance enforces exactly that on the posterior. Bounded only where the recorded imports
can carry the remainder (`Σ imports ≥ residual − production`, legs at most doubled); import-free
fans keep the ADR 0018 form and near-self-sufficient smelters (US 1997–2003) keep the residual,
because their gap is ICSG-internal and spreading it fabricates imports instead.
_Avoid_: including **Scrap** — it is an ICSG `Use` type but is **not traded**; and reading a large
domestic leg as evidence of domestic mining — before ADR 0024 it was the BACI-vs-ICSG shortfall.

**Re-export (re-attributed leg)** — tonnage a region imported and shipped on unchanged. BACI records
it as **two** gross border crossings (`A→h`, `h→Z`), which the accounting structure would read as two
statements about *supply origin* — the second one false, since `h` produced nothing. ADR 0027 moves it
at feed time: `T = min(X, I, max(R, E))` per (product, region, year) in `CC`, where `R = max(0, I−U)` is
the use-side reading (imports above ICSG intake) and `E = max(0, X−P)` the supply-side one (exports above
ICSG production); the volume is split by the hub's own observed origin and destination shares onto direct
`A→Z` legs. The two caps `T ≤ X` and `T ≤ I` are the safety property: a **producer** with thin ICSG
coverage (Kazakhstan cathode — no imports) has `T = 0` however large `E` is and is never netted. See
[ADR 0027](docs/adr/0027-re-exports-are-re-attributed-at-feed-time.md). _Avoid_: calling every hub leg a
re-export — only the part the caps admit is one, and the rest stays in the **transit residual ledger**.

**Re-import** — the `A→h→A` branch of a re-attribution: goods that left `A`, reached the hub and came
back. It is dropped rather than booked, for the same reason **mapped self-trade** is: its root address is
the domestic edge, which it is not a measurement of. Where `A` is `ROW` the same branch is third-country
trade folded onto one aggregate, and is dropped identically. 467 kt CC over 1995–2024.

**Merchanting** — a title trade in a third country while the cargo sails elsewhere (BPM6 §10.41). Two
shapes, and only one is netted: **both legs recorded** (Switzerland concentrate: `I ≈ X`, no process) is
an ordinary re-export and re-attributes in full; **inbound only** (Zambia→Switzerland/Singapore/
Luxembourg anode: `X ≈ 0`) cannot be, because nothing leaves the hub to carry it — those legs stay gross
and the QP keeps refusing them at the exporter's capacity ([`zambia-anode-phantom-trade`](docs/rep/analysis_20260910_zambia-anode-phantom-trade.md)).
_Avoid_: reading a small `T` at a merchanting hub as a failure of the rule — it is the rule declining to
invent a destination.

**Transit residual ledger** — what the netting deliberately leaves behind, written per (product, region,
year) beside the feed (`reexport_ledger.csv`) with a label: `pass-through`, `partial`, `ceiling-relabel`,
`inbound-only[/no-process]`, `producer`, `icsg-gap`, `excluded-aggregate` (ROW is never a hub — an
aggregate of 186 economies has no warehouse), `untouched`. Before ADR 0027 this tonnage was the fan's
silent **clip** (`max(0, Σ imports − usage)`); it is the same material, now named. _Distinct from_:
the **Bounds-clip** above, which is an imputer's saturated fill. _Avoid_: reading the residual as an
error — the Rotterdam effect (imports a hub never re-exports on the record) genuinely cannot be
re-attributed from BACI alone.

**BACI-observed region** — a consumer region for which the BACI feed records at least one cross-region
import (any traded sector, any year). The flag is what makes `Σ imports = 0` *informative*: for an
observed region it means "sourced domestically", for a trade-blind one it means "we cannot see". Derived
in `build_trade_shares.baci_observed_regions()` from the BACI `F_root`. _Avoid_: treating a zero import
sum as evidence without checking observability — that is how a 100 %-domestic pattern gets fabricated at
the observation's confidence.

**Trade-seed** — the `TradeImputer` fill (method `trade_seed`) that sets the **CC** value of a
*cross-region* `Flow→Process` import leg from its already-aggregated BACI **Mass** twin
(`f_CC = conc·f_Mass`), reusing the shared `conc_twin_map`. It is the imputation-time complement of the
trade-share fan: the fan distributes ICSG-CC by BACI geography at *feed* time, whereas trade-seed reads
the BACI-Mass observation directly at *imputation* time so the network recipe gate can pool the covered
import legs. Trade-seed's **writer scope** (below) is the cross-region `Flow→Process` CC edges; the
`NetworkImputer`'s is the same-region edges. A route with no BACI
Mass observation is left NaN (an honest flat-prior gap), never seeded to 0. See
[ADR 0010](docs/adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md). _Avoid_: reading a
trade-seed CC as a new measurement — it is the conc twin of the BACI Mass observation.
**A bilateral leg is never interpolated in time** ([ADR 0026](docs/adr/0026-bilateral-trade-legs-are-not-time-interpolated.md)):
BACI is a *complete* matrix, so a route carrying no row in a year carried no trade, and the
`TimeImputer` is scoped to same-region edges — a cross-region leg is neither a fill target nor an
interpolation anchor. _Avoid_: reading an absent bilateral cell as a gap; before ADR 0026 the
unscoped interpolator invented 9,215 Mass legs / 20,252 kt of trade, 75 % of which survived into
the posterior.

**Value-implied Cu fraction** — `unit_value(product) / unit_value(cathode)` within one trade
feed, one year. Refined cathode is ~99.99 % Cu, so its unit value *is* the copper price on that
feed's own terms, and the ratio is what the product's recorded value implies about its recorded
mass. It is a **consistency check between a feed's two fields**, not a property of the material:
where it leaves the physical band the *quantity* is the field in doubt, because the value can be
checked independently against production statistics. _Avoid_: reading it as a grade — the same
_Avoid_ the **Concentrate grade** entry already carries for `(P−U)/X`, and for the same reason. A
fraction of 0.6–0.9 is not a 60–90 % concentrate; it is a statement that the recorded mass is two to
three times too small ([ADR 0033](docs/adr/0033-baci-quantity-contradicts-its-own-value.md)).
_Avoid_ also: flagging a single leg inside a year whose *aggregate* fraction is in band — the
aggregate is the level at which production statistics can license the claim, and lifting a leg
below it fabricates mass.

**Traded (realized) vs tradable; `region_crossed`.** **Tradable** is a *sector capacity* — the four ICSG-fanned
sectors (Cathode, Concentrate, Anode, Matte) whose Use edges *may* cross regions (above). **Traded / inter-regional**
is a *realized* per-edge property — the Use edge actually crossed a border (`region_from != region_to`) — captured
as the boolean **`region_crossed`**, stamped before a region collapse (`keep_crossings`) so it survives aggregation.
Supply edges are always same-region; **Use** edges *and* **final-use (`Y`) edges** can be `region_crossed` — in
copper 6,486 of 6,627 `Y` edges cross regions in 2020 (cathode into foreign final use), as many as the `U`
edges do. A Sankey's **traded-flow share** is this *realized* split. _Avoid_: reading "traded share" as the
*tradable-sector* fraction — it is the realized inter-regional fraction.

## Total-output / scale-total vocabulary

**Total output** — of a process or flow, the sum of **all its outgoing edges** (incl. waste, final
demand, SB outflows) ≙ PIOLab *throughput / gross output* `x = S·j + P·l + W·n` (Wieland et al. 2022).
_Avoid_: equating it with **useful output** (which is net of waste).

**Total input** — of a process or flow, the sum of **all its incoming edges** (incl. SB secondary and
nature inputs) ≙ PIOLab `x′ = i·U + k·O + m·E`.

**Useful output** — a process's total output **minus** the SB outflows it supplies (waste / SEM
residuals); i.e. its process→flow edges only ≙ PIOLab *useful output* (supply table `S`,
`x_new = S·j = x − P·l − W·n`). _Avoid_: "total output" — useful output is net of waste.

**Scale total** — a Type-2 **soft-sum** (`Σ aᵢ = c + ε`, free cone) whose RHS `c` is the model's
**own prior** group-sum (`c = Σ f_agg`), used for **scale-preservation** against the homogeneous
collapse — not to add information. "Kind-of-external" because the warm-start solve (`a₀ = f_agg` at
t₀, `a_post(t−1)` after) never sees this fixed Stage-2 total. Three families: **supply total output**
(per process — its process→flow useful output), **use total output** (per flow — Σ over all
destinations incl. final demand), and the **scrap level anchor** below. On a **degenerate (one-leg)
fan** the prior group-sum coincides with the feed's own published number, and such a row then *does*
add information — this is a consequence of the fan, not the ADR 0006 mechanism (an external totals
file reconciled in a separate pass). Groups are classified **fabricated vs feed-backed** (per
`agg_result.coverage`) — but both keep the same tight cv: loosening fabricated groups was tried
and rejected, because these rows double as the counterweight to the nodata zero anchors and
loosening them collapses the interior chain
([ADR 0016](docs/adr/0016-provenance-aware-sigma-on-scale-totals.md)). σ_c carries an
**absolute floor at the feeds' reporting resolution** (0.3 kt, the rounding σ of integer-kt
ICSG/crude_ore values, Lp-combined with the cv term — [ADR 0025](docs/adr/0025-scale-totals-carry-the-feeds-reporting-resolution-as-a-sigma-floor.md));
it reaches totals below ~18 kt only, so a sub-kilotonne total is never priced to tens of tonnes.
_Avoid_: **control total** (its RHS is an *external* published
number — see README; that mechanism was removed by
[ADR 0006](docs/adr/0006-scale-total-supersedes-external-control-total.md)); **anchor** /
**coherence soft-sum** (deprecated terms — see **Control total** _Avoid_ above);
reading a scale total as only a prior-preservation device — on no-data subgraphs it is also
what holds scale against the zero anchors (ADR 0016).

**Scrap level anchor** — the third **scale total** family (`smelterscrap`, `directrefscrap`): a
soft-sum stating the **level** of *one input class* of a node — the scrap ICSG reports entering
`Smelting` and entering `CFR` — per region-year, **CC only**, and **only where a feed observed it**.
It is the one `sum` family whose *row existence is data-gated*, which is what makes ADR 0025's σ
floor unconditional for it. Needed because `Scrap` is an `SB` node: `Scrap→Process` edges are in
neither output-total family, so before [ADR 0031](docs/adr/0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md)
the only statement about them was the `scrap_share` **band**, and a share band over a node with no
throughput is satisfied trivially. _Avoid_: calling it a **total input** (that is *all* incoming
edges — this is one class of them); reading the **absence** of a scrap-level row as a prohibition —
an absent row is an *unstated* level, and the edge keeps exactly the freedom it had.

**Contradicted flow node** — a Flow node whose **inflow** prior (`Process→Flow`) and **outflow**
prior (`Flow→Process` + `Flow→SB`) cannot both hold under its own **hard** mass balance. Because
`σ_c = ANCHOR_CV·|c|` scales with the total, the *smaller* side is the cheaper anchor to satisfy,
so the solver collapses the node to the smaller prior even when that side is the worse evidenced
one (Russia Matte 1995: inflow 545 kt CC from ICSG concentrate use vs outflow 50.8 from an
under-reported ICSG refining series ⇒ the smelting chain collapsed). Detected by
`pysuto.flow_node_balance.contradicted_flow_nodes` (outflow < `K_CLOSE`·inflow) and treated by
stripping the **outflow** side's authority — σ on its covered edges and on its `usetotout` anchor
— was tried and **rejected**: σ reprices a data contradiction without resolving it (ADR 0019).
The term survives as a **diagnostic**: a flag is a statement about the *input data* (the
source's two series disagree for that region-year), not about the reconciliation, and the
remedy is data-side.
_Avoid_: reading a downstream export as an outlet for the node — anode exports are produced *by*
CFR *from* matte, so they raise the matte requirement rather than relieving it. See
[ADR 0019](docs/adr/0019-contradicted-flow-nodes-lose-anchor-authority.md).

**Anchor placement** — *which edge* a source series is attached to, as distinct from its value.
A production series measures throughput at one point of the chain; converting it into an edge
prior by dividing through efficiencies moves it to a *different* point, and each stage crossed
adds an assumption. ICSG smelter production ÷ `eff_Smelting` is the concentrate entering
`Smelting` (one stage); ICSG refined production ÷ `eff_ER` is the anode entering `ER` (one
stage). Dividing that same refined series by `eff_CFR·eff_ER` instead placed it two stages back,
on the matte entering `CFR` — and thereby assumed every anode a region makes is refined at home,
which is false for anode/blister exporters. A misplaced anchor is not a σ problem and not a
value problem: the number is right and the edge is wrong.
_Avoid_: anchoring both sides of a hard-balanced node — the balance already determines the
second side, and double anchoring is the over-constraint
[ADR 0006](docs/adr/0006-scale-total-supersedes-external-control-total.md) rejected. See
[ADR 0021](docs/adr/0021-icsg-refined-production-anchors-anode-into-er.md).

## Downstream PSUT / PIOT vocabulary

**PSUT (physical supply-use table)** — the **materialised numeric** supply-use flows for *one period and
one valuation*, derived from the solved edge vector `a*`. **Shape-agnostic**: its canonical form is a **tidy
edge table** (one row per directed edge, carrying its `val`), and the block matrices — supply `S` (useful
output), use `U`, final use `Y`, plus boundary blocks nature-input `E`, SEM-input `O`, SEM-output `P` — are
**derived pivot views** of that same PSUT. The valueless Process+Flow **SUT** is the *graph* (the
`AccountingStructure` topology); a PSUT is that graph **carrying solved values**. _Avoid_: calling the
**valueless** graph a "PSUT" (the bare graph is the **SUT**); "SUT" for a numeric realisation (long or matrix);
reading "PSUT" as *matrices only* (the matrices are one view; the tidy table is the canonical other).

**Block** — the PSUT quadrant a directed edge belongs to, read off its endpoint entity kinds (and, for
the boundary blocks, the `SB` node's boundary role): `S` Process→Flow, `U` Flow→Process, `Y` Flow→SB·FD,
`E` SB·Nature→Process, `O` SB·SEM→Process, `P` Process→SB·SEM. Every admissible edge lands in exactly one
block; the tidy table carries it as the `block` column and each matrix view is one block's pivot. _Avoid_:
"mask" for a block — the `supply`/`use` **masks** are rule-derived edge selections that straddle blocks (a
loaded structure's `supply` mask also holds `P`, its `use` mask `E` and `O`).

**Primary valuation** — the valuation a multi-valuation PSUT's views and plots default to (`primary`; copper:
`CC`, the `content` role once ADR 0034 lands); `valuation=` overrides it. A PSUT carries **every** valuation of
its structure in one table, and the realized `conc` is the ratio of two of them. _Avoid_: reading the structure's
first-listed valuation (`Mass`) as the default; one PSUT per valuation.

**Aggregation (pushforward)** — region/sector aggregation of a PSUT as a *pushforward* of the tidy table: one
concordance map per dimension (`region`, `sector`) applied **symmetrically to both facets** (`from_`, `to_`),
then a sum over every key column — `period`, `valuation`, `block` and the entity kinds stay in the key, so
nothing merges across them and **time is never summed**; a sector map may only merge sectors of one entity kind.
`region_crossed` is stamped **before** the relabel, at the finest resolution ever seen (a two-step collapse keeps
the country-level stamp), so the realized traded share survives a World collapse. `edge_id` does not survive.
_Avoid_: summing `period`; reading `region_crossed` at the aggregated resolution (it is the root-resolution
crossing); a one-label `sector` collapse (it would merge Process with Flow).

**Supply-use Sankey** — the bipartite Sankey drawn *directly* from a PSUT: processes **and** commodities are
nodes, the `S`/`U` rows are links, the boundary blocks attach as links to their own `SB` nodes (inputs `E`/`O`
on the left, outputs `P`/`W`/`Y` on the right), and a border-crossing `U`/`Y` flow is its own **traded** link
(from `region_crossed`, or from distinct regions). Needs no construct. In copper it carries the same numbers as
the process-network Sankey — `D` is the identity (ADR 0037 §1) — with the commodity nodes shown rather than
contracted into edge labels. _Avoid_: "process network" for it (that is the PIOT's contracted view);
reading a node's total as throughput of anything but that (period, valuation).

**PIOT (physical input-output table)** — a **symmetric** (square) physical IO table derived from a PSUT by a
chosen **construct** (below). The **product-by-product** PIOT (commodity×commodity) is the analytic primary —
technical coefficients `A`, core flows `Z`, total requirements `L`, for footprints; the **industry-by-industry**
(process×process) PIOT is the **process network** (below). _Avoid_: conflating with the rectangular **PSUT** (a
PIOT is square); reading "PIOT" as *product-by-product only* (it is a construct **family**).

**Construct** — the SUT→PIOT transformation choice, on **two orthogonal axes**. **(1) Table shape**
(Miller & Blair ch. 5 / Eurostat): **product-by-product** (commodity×commodity) vs **industry-by-industry**
(process×process), the latter via the **market-share** rule (`D = S·q̂⁻¹`, fixed-product-sales). **(2) By-product
method** — meaningful *only* on the analytic product-by-product PIOT (Altimiras-Martin 2014): **Suh** (2004)
endogenises disposals-to-nature as **negative input coefficients**, *altering* `A`/`L` (the PIOLab/Wieland choice,
project primary); **Xu–Zhang** (2009) leaves the table intact via a modified multiplier. _Avoid_: conflating the
axes — **market-share** is a *shape*, **Suh** is a *by-product method*.

**Process network** — the **industry-by-industry** PIOT in tidy/graph form: **processes are nodes, commodity
flows are edges**; each edge `p→q` carries commodity `f` at value `D[p,f]·U[f,q]` (market-share allocation of
`q`'s use of `f` back to its producers `p`). Boundary blocks attach as edges to boundary pseudo-nodes (`Nature`,
`FD`, SEM-in, SEM-out); **disposals-to-nature** (`P` residual outputs) are **final outputs, never inter-process
edges** — allocation does not apply to them (Altimiras-Martin 2014, fn. 3). The substrate of the production-network
**Sankey**. Built by `PSUT.to_piot('market_share', level='Z')` as a tidy edge table (`block ∈ {Z, Y, E, O, P, W}`,
`Z` = inter-process; each edge names its `commodity`; `region_crossed` travels with the row it came from), with
`Z()` the process-by-process pivot and `D()` the shares. _Avoid_: calling it a "PIOT" unqualified (defaults to
product-by-product); drawing the supply legs `S` as their own edges (they only attribute producers).

**gPIOT** — the full **multi-regional** PIOT over (region × flow), whose `L` traces embodied/indirect flows
across borders. _Avoid_: "global PIOT" for a single-region table.

**Technical coefficients `A`** (uppercase) — the per-unit input-requirements matrix of the PIOT (`A = B·D`).
_Avoid_: confusing with the lowercase **edge vector `a`** (the decision variable) — case distinguishes them.

**Total requirements / Leontief inverse `L`** — `L = (I − A)⁻¹`, mapping final demand to total (direct +
indirect) output. _Avoid_: "multiplier" (a multiplier is a row/column sum of `L`, not `L` itself).

**Footprint** — a **consumption-based** indicator: the boundary input (extraction) embodied in a vector of
final demand. _Avoid_: "direct/apparent" use (a footprint is the indirect/embodied counterpart).

**Embodied bilateral trade** — the footprint resolved on both ends: rows the region where the boundary input
enters (extraction), columns the region whose final demand embodies it (`Σ_{p∈r} e_p L[p,q] Y[q,d∈s]`); row sums
are each region's extraction, column sums its **RMC**. _Avoid_: reading it as physical trade (that is the `Z`/`Y`
edges crossing a border).

**ew-MFA headline indicators** — per region and period, in the table's unit: **DE** (domestic extraction, the `E`
block into the region's processes), **imports** / **exports** (network edges crossing into / out of the region;
`region_crossed` rows of a collapsed table count as both), **PTB** `= imports − exports`, **DMI** `= DE + imports`,
**DMC** `= DMI − exports`, **RMC** (raw-material consumption, the extraction footprint of the region's final demand).
Over a closed world `Σ RMC = Σ DE`. _Avoid_: comparing DMC (territorial, gross trade) with RMC (consumption-based,
embodied) as if they measured the same thing.

**Realized `conc`** — the *ex-post* concentration `CC/Mass` of an edge in the materialised PSUT, versus the
parametric **concentration coefficient** input; the gap is solver slack. Ore grade is the Ore-sector instance.
_Avoid_: reading a realized≠parametric gap as an error (cf. the `conc_*_supply` z-violation — it is absorbed slack).

## Constraint identity vocabulary

**`constraint_id`** — the unique, human-readable **display key** of one constraint row, with the
grammar `{type}_{variant}_{specs}_{year}` (tokens joined by `_`; every token underscore-free). It is
a key for *reading*, never *parsing*: all structured fields live as `row_meta` columns and are read
from there. _Avoid_: splitting a `constraint_id` to recover a field — read the column instead.

**type** (constraint family) — the top identity slot, one of `massbal` | `sum` | `ratio`. Already a
reserved word for the constraint-row `type` column (see Glossary §Entity `_Avoid_`: "node type").
_Avoid_: using `type` for an entity kind (that is **entity**: `Process` | `Flow` | `SB`).

**variant** — the second identity slot: the *role* a constraint plays within its family
(`massbal`; `usefulsupply` / `subentity` / `supplytotout` / `usetotout` / `smelterscrap` /
`directrefscrap` / a user csv label for `sum`;
`ratioeq` / `bound` for `ratio`). _Avoid_: "name" (collides with **Sector**, "the named instance").

**Node-scoped vs set-scoped constraint** — a `massbal` or node-`sum` row balances **one node**, so its
specs render that node as `sector_region_valuation`; the collapse of a node to `sector_region` is
sound because constraint rows are emitted over **concrete** nodes only (the `sector→entity` 1:1
invariant holds; open `<any>` sectors never reach a row). A `ratio` row is either **pair-scoped** —
relating **two edges**, each rendered full `from→to`, numerator over denominator, with `num_*`/`denom_*`
populated — or **set-scoped** when its row carries `agg=true`: it pools **many** a-side edges against
many b-side edges per `(region_to, time, valuation)` bucket (`Σb − r·Σa`), so it has no single
numerator/denominator edge and its identity rests on the `variant` plus `region`/`valuation` group-key
specs (`num_*`/`denom_*` are null, exactly like a set-`sum`). A set-`sum` (a csv constraint grouped
across nodes) spans **many** edges with no single node — its identity rests on the `variant` plus
group-key specs; by contrast the `supplytotout` / `usetotout` **scale totals** — and the
`smelterscrap` / `directrefscrap` **scrap level anchors**, which carry the *consuming* node — are
node-`sum` rows and so carry populated `sector`/`region`. _Avoid_:
expecting a single `sector`/`region`, or populated `num_*`/`denom_*`, on a set-sum or **set-scoped
(agg) ratio** row (they are null there).

**Cross-valuation row** — `ratio` is the **only** constraint family whose two sides may differ in
**valuation** (a `conc` ratio pins `CC = conc·Mass`, so `num_valuation = CC`, `denom_valuation =
Mass`); `massbal` and `sum` rows are strictly single-valuation. This is why `ratio` is the family
whitelisted for mixed-valuation rows. _Avoid_: reading a ratio's two valuations as a bug.

## Framework meta-model vocabulary

**Role** — a framework-owned kind that a case's label is bound to. Two families: the **boundary
role** of an `SB` node, `nature | secondary | final_demand` (the three system-boundary categories
above; a `secondary` node is an input when reached by a use edge and a residual output when reached
by a supply edge), and the **valuation role**, `content | gross` (contained-metal mass vs gross
product mass). The entity kinds `Process | Flow | SB` are reserved framework words, not roles.
_Avoid_: reading a label (`SEM`, `CC`) as the concept — labels are the case's, roles are the
framework's; "entity kind" for a role.

**Relation** — one linear equality over edges with a σ (`Σ αᵢ aᵢ = c ± σ`) that exists before
either stage consumes it: a **constraint row** is a relation *emitted* into `G`; an imputation
fill is a relation *solved* for its lone unknown. A **process recipe**, a Flow-node balance and a
`conc`/`eff` params row are relation *sources*, not relations (a recipe yields the node balance
and the pooled `eff` ratio; the Mass twin is the per-edge `conc` ratio). **Scale totals** and hard
bounds are *emit-only* relations — never solved for a prior. _Avoid_: "constraint row" for the
pre-stage object; "recipe" for a single relation.

**Writer scope** — the set of edges a relation source is *permitted* to write. Scopes may overlap;
**cascade order** decides precedence among sources whose scope contains an edge; a write outside
scope is rejected. _Avoid_: "ownership" — it implied an exclusivity the `conc`/`eff` fallback
imputers never respected.

**Case** — one metal's complete declaration: an accounting structure (referenced by registry id),
its feeds, param families, relation sources and cascade, anchor families and σ policies, run by the
generic stage drivers. The copper case is the first. _Avoid_: "model" (the QP, the PIOT),
"study" (a REP analysis), "scenario".

## ROW pedigree-penalty vocabulary

**ROW** — *Rest of World*, the residual region aggregate for countries not individually
represented. Canonical sentinel: the string `"ROW"` in a node's `region`.
_Avoid_: "RoW" (normalise to upper-case), "rest-of-world".

**`row` mask** — the named edge mask selecting edges that touch ROW on **either** endpoint
(`region_from == "ROW"` **or** `region_to == "ROW"`). A registered input mask
(`data/input/mask/row/`).

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
