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

## Decisions from the 2026-09-23 grilling (Slice 1 shipped)

1. **The first Sankey is the bipartite supply-use view, on `PSUT`.** In the copper structure every Flow node
   has exactly one producer (329 of 329 in 2020) and, after a World collapse, every flow sector still has one
   process sector, so the market-share matrix `D = S·q̂⁻¹` is the identity and the process network is the
   bipartite view with its commodity nodes contracted into edge labels. The `PSUT` Sankey therefore needs no
   construct — only the boundary blocks (#19) and the World collapse with `region_crossed` (#24) — and ships
   before #25. The process-network Sankey stays on `PIOT` for #25, where a sector map that merges processes
   with different products (e.g. ER + SXEW) makes `D` non-trivial. Both use one Plotly renderer in
   `diag/psut.py`, reached through thin lazily-importing methods `PSUT.sankey(...)` / `PIOT.sankey(...)`;
   the data classes import no plotting library at module level.
2. **Boundary blocks are classified by `entity_sub` labels, in one place.** `classify_blocks` reads `Nature`,
   `SEM` and `FD` to split `E`/`O`/`P`/`Y` (and the empty `W`); it is the single site to switch to `acc.roles`
   when issue #26 §1 (ADR 0034) lands. The non-copper-label fixture belongs to #26, not to Stage 6.
3. **One table, every valuation, a recorded primary.** A `PSUT` carries every valuation of its structure;
   `primary` (copper: `CC`, the future `content` role) is what views and plots default to, `valuation=`
   overrides, and `realized_conc()` is the ratio of two valuations from the same table. The structure's
   first-listed valuation (`Mass`) is *not* the default.
4. **Order:** #19 → #24 → bipartite Sankey on `PSUT` → #25; one commit per slice, the shipped run
   re-materialised after #19 and #24.

Finding recorded in [GLOSSARY.md](../../GLOSSARY.md): final-use (`Y`) edges are bilateral in copper (6,486 of
6,627 cross regions in 2020 — cathode into foreign final use), so `region_crossed` applies to `U` and `Y`,
not to `U` alone.
