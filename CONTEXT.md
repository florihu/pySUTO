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
_Avoid_: conflating `entity_sub` with **entity** (only `Process` | `Flow` | `SB`).

## Coverage & val vocabulary

**Value (`val`)** — *a value of an edge*: the post-imputation `f_agg` on a root edge.
Every root edge has one (imputation fills all NaN), so "having a val" alone says nothing
about data quality — see **val-coverage**.

**val-coverage** (reported as the `val_pct` column of the stats/coverage diagnostics) —
the relative number of edges in a subset that carry a **real** val, where "real" follows
the imputation-log *method* rule (`has_val`): flat-prior / `flat_prior_noisy_zero` default
fills are **not** real; observed values — *including a genuine `0.0`* — and
fan / time / parametric / formula imputations **are**. _Avoid_: deciding val-coverage from
`f == 0` — a flat-prior fill is `0` but so is a real observed zero, and the two are
distinguished by the log method, not the magnitude.

**Coverage** (glossary, the `agg_result.coverage` boolean) — *did any feed observe this
edge?* — a **pre-imputation** notion, strictly **narrower** than val-coverage: it credits
**only** feed observations, never any imputation. _Avoid_: conflating the two — the stats
column is `val_pct` (not `coverage_pct`) precisely because val-coverage also credits
non-flat-prior imputations that feed-`Coverage` excludes.

## Defect, residual & violation vocabulary

**Defect** — the per-**edge** prior departure `a* − a₀` (solved value minus the warm-start
prior). Plotted by SD-T-09 (`|a*−a₀|` by from/to sector/region) and tabulated by
`SolverStatsDiagnostic` (`defect_from/to_<year>.csv`), keyed by `(sector, region, valuation)`.
Its standardised form is `z = (a*−a₀)/σ_a` (SD-T-03). _Avoid_: calling the defect a
"residual" — that term is reserved (below).

**Residual** — the per-**constraint-row** delta `G·a − c` (`OptimResult.residuals`; SD-T-01
"Constraint-residual digest", SD-R-02 mass-balance leakage). ≈ 0 on hard rows; on a **soft
(free) row** it equals the slack `ε` (see README §"Cone order": `Gx − ε = c`). _Avoid_:
conflating with **defect** — residual is per-constraint, defect is per-edge.

**Soft-constraint violation table** — the harmonised per-soft-`constraint_id` table emitted at
*both* stages with one schema `[type, valuation, c, lhs, residual, sigma_c, z]`, where
`lhs = G·a` (the achieved left-hand side), `residual = lhs − c` and `z = residual/σ_c`
(SD-T-02). The only difference is which `a` feeds `lhs`: the **prior** `a0` at the constraint
stage (`ConstStatsDiagnostic` → `constraint_violations_prior_<year>.csv`, `lhs = G·a0`) vs the
**solved** `a*` at the solver stage (`SolverStatsDiagnostic` →
`constraint_violations_<year>.csv`, `lhs = c_star = G·a*`). _Avoid_: reading a large `|z|` as a
failure on its own — a soft row is *meant* to absorb disagreement; `|z|` just measures it in
units of the row's tolerance `σ_c`. _Note_: `ConstDiagnostic` plots the same prior quantity
under the opposite sign (`eps0 = c − G·a0`); the table follows the reserved `G·a − c` sign.

## Feed valuation & trade-share vocabulary

**Valuation — `CC` vs `Mass`.** Every edge exists in two valuations. **`Mass`** is the gross product
mass (tonnes of the commodity as traded/handled). **`CC`** (*copper content*) is the contained-copper
mass (tonnes Cu). They are linked per sector by the **concentration coefficient**.

**Concentration coefficient (`conc`).** The Cu mass fraction of a sector's commodity (e.g. Cathode
0.9975, Anode 0.94, Matte 0.525, Concentrate 0.275; from the `conc` param). The `conc_<sector>_supply`
relation enforces `CC = conc · Mass` on each node pair, so the two valuations of one edge are not
independent. _Avoid_: reading the `conc_*_supply` z-violation as a "bug" — it fires whenever the two
feeds populating the CC and Mass sides of an edge disagree.

**Feed semantics — ICSG vs BACI.** **ICSG** publishes production/usage statistics in **copper content
(`CC`)**; its `Use` rows are **single-endpoint** (consumer known, producer open) ⇒ producer **fan-out**.
**BACI** publishes **bilateral gross trade** in **`Mass`** (fully classified X→Y; no self-trade rows, so
no domestic leg). On a trade edge the ICSG-CC value is therefore a *fan artifact* and the BACI-Mass value
is the real observation. _Avoid_: treating a per-edge ICSG-CC trade value as a measurement — it is the
national usage split across producers by the fan weights.

**Trade-share fan weighting** ([ADR 0002](docs/adr/0002-single-endpoint-rows-fan-out-not-replicate.md)
option D) — weighting ICSG's producer fan by BACI bilateral import shares instead of uniformly, so ICSG-CC
inherits BACI's geography. The domestic leg (X==Y) carries the residual `usage − Σ imports` (in CC units);
the split is computed in copper content via `conc`. Applies only to the four **traded** ICSG-fanned
sectors — Cathode, Concentrate, Anode, Matte. _Avoid_: including **Scrap** — it is an ICSG `Use` type but
is **not traded**.

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
