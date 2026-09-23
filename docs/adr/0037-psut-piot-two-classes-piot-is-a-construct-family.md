# PSUT (supply-use) and PIOT (input-output) are two classes; PIOT is a construct family built as a transform

**Status:** proposed — refines [ADR 0007](0007-downstream-psut-piot-follows-piolab-suh-no-external-io-lib.md).

Stage 6 splits along the glossary's SUT/PIOT line. **`PSUT`** owns the supply-use world: materialise `a*` into a
tidy edge table (one period, one valuation), expose the `S/U/Y/E/O/P` block views, and aggregate along
region/sector. **`PIOT`** owns the input-output world: built on demand via `PSUT.to_piot(construct, level)`, it
owns the symmetric tables (`Z`/`A`/`L`), footprint calculations, and the IO plots — including the
process-network **Sankey**. A constructed PIOT is a *derived view* of a PSUT, not a separately-persisted stage.

`to_piot` is parameterised on **two orthogonal axes**: **construct** (table shape — *industry-by-industry /
market-share* vs *product-by-product*) and **method** (by-product calculus — *Suh* vs *Xu–Zhang*, meaningful only
on the analytic product-by-product PIOT), with a **level** (`Z` | `A` | `L`) gating how much is computed.

The first deliverable builds the **market-share / industry-by-industry** construct at `level='Z'` — the Sankey
substrate — **not Suh**. Although Suh product-by-product is the more important *analytic* method, the Sankey needs
**processes as nodes**, which is industry-by-industry; Suh, `A`/`L`, and footprints are product-by-product and
defer to #20.

## Considered options

- **Single `PSUT` class hosting everything.** Rejected: footprints/`A`/`L` and the Sankey are input-output work on
  a square table, conceptually distinct from the supply-use PSUT; folding them in blurs the SUT/PIOT boundary the
  glossary already draws.
- **`PIOT` as a separate persisted stage** (own registry / `from_disk` / handoff, like the solver). Rejected: a
  PIOT is cheap to derive and ephemeral; persisting it duplicates the PSUT it came from. It is a transform, not a
  stage.
- **Two classes, `PIOT` derived via `to_piot`. (chosen.)**

## Consequences

- The process-network Sankey lives on `PIOT`, not `PSUT`; any supply-use-shaped plots (block heatmaps, bipartite
  Process↔Flow view) live on `PSUT`.
- `region_crossed` is stamped by `PSUT.aggregate(..., keep_crossings=True)` and carried onto the PIOT edges, so the
  Sankey's domestic/traded split survives the region collapse.
- Adding the Suh product-by-product analytic PIOT and footprints (#20–22) is a new `construct`/`method` on the same
  `to_piot` surface — no new class, no redesign.
