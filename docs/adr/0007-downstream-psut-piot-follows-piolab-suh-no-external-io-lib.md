# Downstream PSUT/PIOT construction follows PIOLab + Suh, with no external IO-library dependency

**Status:** proposed — introduces **Stage 6** (materialise `a*` → PSUT → PIOT + IO analytics).

After the Stage-4 solve, pySUTO holds the solved edge vector `a*` (`OptimResult.F_opt`). To answer
*downstream-dependency* questions we add **Stage 6**: per period and per **valuation** (`CC`, `Mass`),
materialise `a*` as the PIOLab physical supply-use blocks — supply `S` (useful output, `Process→Flow`),
use `U` (`Flow→Process`), final use `Y` (`Flow→SB·FD`), nature input `E` (`SB·Nature→Process`), SEM input
`O` (`SB·SEM→Process`), SEM output `P` (`Process→SB·SEM`) — then build the symmetric **PIOT** and run
classical IO analytics over a **full MRIO** (region × flow). The block decomposition is already latent in
`acc._build_edge_masks()` + `edge_meta.entity_sub`; the `W` (discard-to-nature) block is **empty** in the
copper model (all residuals — Slag/Tailings/WR/PSC — are SEM-outputs), so `x_new = x − P·l`.

The squaring follows **Suh (2004)** as used by the **PIOLab** (Wieland et al. 2022, `docs/ref/`): by-products
are endogenised as negative inputs and the core supply table holds **useful output** only; then, commodity-by-
industry (Miller & Blair ch. 5), `B = U·x̂_new⁻¹`, `D = S·q̂⁻¹`, `A = B·D`, `Z = A·q̂`, `L = (I−A)⁻¹`.
This matches the PIOLab vocabulary CONTEXT.md already adopted (*total/useful output*, throughput).

**No runtime dependency on `pySUT` or `pymrio`.** The construction step (edge vector → physical SUT) exists in
neither; both are downstream *consumers* of an already-built table. The Suh/PIOLab calculus is a few sparse
linear-algebra lines, so we implement it directly (`piot.py`) rather than take on a library.

## Considered options

- **A. IndEcol/pySUT.** Rejected as a dependency: it *manipulates* pre-built single-unit (monetary) SUTs given
  as numpy `V/U` arrays — it neither builds the SUT from edge data nor handles the physical/by-product case as
  PIOLab does — and is stale (PyPI v1.1, 2015). Its construct catalogue is a useful *reference*, not a runtime dep.
- **B. pymrio.** Rejected as the foundation: it *consumes* prebuilt MRIO databases with a **monetary core +
  physical satellite** model and offers no SUT→IO construction. Its IO math is marginal value once `A`/`L` exist.
  We may ship a thin optional exporter to a pymrio `IOSystem` (assign `A`/`Z`/`Y`) for users who want its
  utilities — not imported by the pipeline.
- **C. Build our own, following PIOLab/Suh. (chosen.)** Honours the AISHA/PIOLab lineage pySUTO already sits in,
  keeps the physical/by-product treatment correct, and adds no stale or monetary-first dependency.

## Consequences

- **New modules** `src/pysuto/psut.py`, `piot.py`, `psut_indicators.py`, `registry/psut.py`, `scripts/psut.py`;
  artifacts under `data/proc/psut/<run_id>/` with `parents.upstream = opt_run_id`.
- **Point estimate first (the deliberate no-).** v1 builds from `a*` only; Stage-5 posterior σ is **not**
  propagated yet — the API is shaped so a later Monte-Carlo layer (sample edges → rebuild `L` per draw) can add it.
- **Terminology.** New canonical terms **PSUT / PIOT / gPIOT / `A` / `L` / footprint / realized `conc`** recorded
  in [CONTEXT.md](../../CONTEXT.md), with `G` kept for the constraint matrix and lowercase `a` for the edge vector.
