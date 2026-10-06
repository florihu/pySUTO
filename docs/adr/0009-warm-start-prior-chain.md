# Genuine per-period warm-start prior chain (a_prior(t) = a_post(t-1))

**Status:** accepted

The solver was always documented as warm-starting the prior across time
(`a_0(0) = f_agg`, `a_0(t) = a*(t-1)`), but in the **block path** the carry-forward
was a silent no-op. After each year the solver wrote `a_0_t_global[cols_t] = F_opt_t`
into the **same** global columns it had just read — and the per-year `cols_t` tile
the global edge vector **disjointly** (30 × 28482 = 854460 = E, consecutive-year
intersection 0). So a later year never read the write: every year optimised against
the conditioned Stage-3 `f_agg`, which is itself time-replicated
(`a0[1995] == a0[2024]` exactly). The temporal smoothing the model claimed to do was
not happening. It surfaced while fixing two diagnostics: SD-T-05 pooled the whole
854 460-entry vector (`n=854460`, identical across years) and SD-T-07's
prior-vs-posterior columns for non-first years collapsed onto the diagonal because the
"prior" (the reassembled posterior) equalled the posterior.

## Decision

Make the warm-start real and per-period, and store it explicitly:

- `a_prior(t0) = f_agg` (cold seed); `a_prior(t>0) = a_post(t-1)` **aligned by edge
  template** `(from_id, to_id, valuation)`, which lives on `AccountingStructure`
  (`edge_template_code` / `warm_start_gather`). All years share the identical 28482
  templates, so the gather is a clean 1:1 map (positional in practice).
- The per-period cold seed is stored on `ConstraintRun.timestep_data[t]["a_prior"]`;
  the *actually-used* `a_prior(t)` is stored on `OptimResult.a_prior`. Diagnostics read
  the prior directly instead of reconstructing it.
- `ClarabelSolver(prior_mode='warm'|'cold')` chooses the regime at init; default `warm`.
  σ_a is still conditioned once from the cold prior — the mean drifts, the precision
  does not.
- The legacy KKT-factorisation-reuse local `warm_start` is renamed `reuse_factorization`
  to end the collision with the *prior* warm-start.

## Considered options

- **Stage-3 prior `a0` sliced per year (cold, honest to the old solve).** Rejected: it
  documents the no-op as intended rather than fixing it, and discards the temporal
  smoothing the model is built around.
- **Reconstruct `a*(t-1)` only in the diagnostics, leave the solver alone.** Rejected:
  the plot would show a prior the optimisation never used — misleading.

## Consequences

- **Solved results change.** Every warm year now genuinely depends on the previous
  year's posterior; existing opt runs must be re-solved. Objective values look very
  different (larger `|a_prior|` ⇒ large negative reported objective — the omitted QP
  constant — not a convergence problem).
- The legacy full-vector solve path has no per-year `cols` and cannot replicate the
  year-local chain; path-equivalence with the block path is only defined under
  `prior_mode='cold'`.
- `ConstraintRun.a0` is now vestigial (the global time-replicated cold vector);
  `a_prior_global` is the accessor for the global cold prior. See GLOSSARY.md
  *a_prior (warm-start prior)*.
