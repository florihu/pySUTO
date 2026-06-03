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
