# CONTEXT — pySUTO glossary

The canonical glossary lives in [README.md](README.md) §"Glossary / vocabulary". This file
captures terms sharpened during design/grilling sessions that the README does not yet state
precisely. Glossary only — no implementation details.

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
prior, `a_prior` above). Plotted per period by SD-T-05 (prior vs posterior magnitude), SD-T-07
(prior-vs-posterior rocket) and SD-T-10 (per-year boxplot of `log10|defect|`), by SD-T-09
(`|a*−a_prior|` by from/to sector/region), and tabulated row-level by `SolverStatsDiagnostic`
(`edges_<year>.csv`, columns `defect` / `defect_abs` / `z`). Its standardised form is
`z = (a*−a_prior)/σ_a` (SD-T-03). _Avoid_: calling the defect a "residual" — that term is
reserved (below).

**Residual** — the per-**constraint-row** delta `G·a − c` (`OptimResult.residuals`; SD-T-01
"Constraint-residual digest", SD-R-02 mass-balance leakage). Its feasible range depends on the
row's cone: **≈ 0 on hard *equalities*** (`zero` cone — mass balance); **`≤ 0` on hard *bounds***
(`nonneg` cone — ratio/flow bounds, since feasibility is `G·a ≤ c`), where a *negative* residual
is healthy **slack** (distance below the limit), not a violation, and only a *positive* residual
is a breach; **`= ε`, the slack, on a soft (free) row** (see README §"Cone order": `Gx − ε = c`).
_Avoid_: (1) reading a large-negative residual on a `nonneg` bound as a violation — it is slack
(a misread that once produced a phantom "inverted bound" diagnosis); (2) conflating with
**defect** — residual is per-constraint, defect is per-edge.

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
`residual` / `z = residual/σ_c` (SD-T-02) are the *stage's achieved* residual — `lhs_prior − c`
at the constraint stage, `lhs_post − c` (= the slack `eps` on a soft row) at the solver stage;
`z` is NaN on hard rows (`σ_c = 0`). _Avoid_: reading a large `|z|` as a failure on its own — a
soft row is *meant* to absorb disagreement; `|z|` just measures it in units of the row's
tolerance `σ_c`. _Avoid_: an unqualified `lhs` column — the files qualify every LHS as
`lhs_prior` / `lhs_post` so no column changes meaning between stages. _Note_: `ConstDiagnostic`
plots the same prior quantity under the opposite sign (`eps0 = c − G·a0`); the table follows
the reserved `G·a − c` sign.

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
import legs. Ownership is a clean structural partition: trade-seed writes **only** cross-region
`Flow→Process` CC edges; the `NetworkImputer` writes **only** same-region edges. A route with no BACI
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
**coherence soft-sum** (deprecated terms — see README §Glossary "control total" _Avoid_);
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

**PSUT (physical supply-use table)** — the **materialised numeric** supply/use matrices for *one period and
one valuation*, derived from the solved edge vector `a*`: supply `S` (useful output), use `U`, final use `Y`,
plus boundary blocks nature-input `E`, SEM-input `O`, SEM-output `P`. The Process+Flow **SUT** is the *graph*;
the **PSUT** is its matrix realisation. _Avoid_: calling the edge graph a "PSUT" (the graph is the **SUT**);
"SUT" for the numeric matrices.

**PIOT (physical input-output table)** — the **symmetric** product-by-product table derived from a PSUT by
endogenising by-products as negative inputs (Suh 2004): technical coefficients `A`, core flows `Z`, total
requirements `L`. _Avoid_: conflating with the rectangular **PSUT** (a PIOT is square).

**gPIOT** — the full **multi-regional** PIOT over (region × flow), whose `L` traces embodied/indirect flows
across borders. _Avoid_: "global PIOT" for a single-region table.

**Technical coefficients `A`** (uppercase) — the per-unit input-requirements matrix of the PIOT (`A = B·D`).
_Avoid_: confusing with the lowercase **edge vector `a`** (the decision variable) — case distinguishes them.

**Total requirements / Leontief inverse `L`** — `L = (I − A)⁻¹`, mapping final demand to total (direct +
indirect) output. _Avoid_: "multiplier" (a multiplier is a row/column sum of `L`, not `L` itself).

**Footprint** — a **consumption-based** indicator: the boundary input (extraction) embodied in a vector of
final demand. _Avoid_: "direct/apparent" use (a footprint is the indirect/embodied counterpart).

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
