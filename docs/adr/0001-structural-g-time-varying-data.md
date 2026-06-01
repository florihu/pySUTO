# ADR 0001 — Structural G, time-varying data tape (+ warm-start prior)

* **Status:** Accepted (2026-05-31).
* **Reverses:** the prior "temp-aware G / `G_blocks`" decision recorded in
  commit `8bd3783` ("revised the G init to temp aware G").
* **Supersedes:** the README narrative around "G is time-invariant"
  realised by a `(n_edges × n_timesteps)`-replicated edge set with row
  activation sentinels.

## Context

Pre-refactor, the constraint graph carried the time axis directly:
`acc.edge_index` had a `time` column, the constraint matrix was sized
`(M_total, E_total) = (n_constraints × n_timesteps, n_edges × n_timesteps)`,
and the solver assembled per-period blocks via
`_partition_G_by_time` + `G_blocks`. Per-period applicability of a row was
encoded by writing the sentinel σ_c = `SIGMA_C_INACTIVE` (free rows) or
c = `BOUND_C_INACTIVE` (nonneg rows) into rows that didn't belong to that
period.

This worked but had four recurring failure modes:

1. **Replication bloat.** Every structural edge / constraint paid an
   `n_timesteps` storage tax even when admissibility, sparsity and topology
   never varied across `t`.
2. **Sentinel arithmetic.** `SIGMA_C_INACTIVE = 1e10` is fine when squared
   in the objective; it is *not* fine when read by a downstream consumer
   that doesn't know it's a sentinel (this bit the Stage-5 trace-prior
   diagnostic).
3. **Partition invariant under runtime check.** "No row of `G` may touch
   columns outside its own time bucket" had to be re-verified every run via
   `_partition_G_by_time`; a builder bug that violated it surfaced as a
   runtime error instead of a structural impossibility.
4. **Dual solver / uncprop paths.** Block path + legacy fallback path each
   needed their own warm-start, year-local persistence, cone permutation,
   feasibility check.

The realisation that triggered this reversal: the **structure** (sparsity
pattern + row identities + edge identities) really is time-invariant. The
**numeric data** in `G` is *not* — Argentina ore grade declines from 0.00878
in 1995 to 0.00703 in 2001, and 918 of those `r(t)` entries live inside
`G` because the ratio constraint `a_i = r(t)·a_j` puts the coefficient on a
decision variable. So time-invariance of the matrix as previously claimed
was a half-truth: the *pattern* is invariant; the *coefficients* are not.

## Decision

Move the time axis off `G`'s row/column identity and onto a per-period
data tape:

```text
StructG:
    indptr  : (n_constraints + 1,)
    indices : (nnz,)
    data    : (nnz, n_timesteps)
    shape   : (n_constraints, n_edges)
    row_meta: DataFrame, no `time` column
    .at(t_idx) → csr_matrix view (zero-copy on indptr / indices)
```

* `acc.edge_index` becomes structural — one row per `(from_id, to_id,
  valuation)`.
* `acc.time_range` remains the canonical period axis.
* Per-period applicability moves to a first-class
  `ConstraintRun.active: (n_constraints, n_timesteps)` boolean mask;
  inactive rows are dropped from the per-period QP entirely.
* Per-period `c`, `sigma_c`, `sigma_a` become 2-D `(•, n_timesteps)`.
* `a0` is a 1-D warm-start seed `= a_agg[:, 0]` (1995 anchor).
* Aggregation moves to `(N_f, n_edges, n_timesteps)` feed tensors with
  2-D `(n_edges, n_timesteps)` outputs (`a_agg`, `sigma_agg`).
* Solver collapses to a single per-period loop driven by `StructG.at(t)`
  and `active[:, t]`; the dual block / legacy paths and KKT factorisation
  reuse heuristics are removed.
* The "prior" is a first-class adapter:
  * `AggregationPrior(a_agg)` — `a_0(t) = a_agg[:, t]` (no smoothing).
  * `WarmStartPrior(a0)` — `a_0(0) = a0`; `a_0(t>0) = a*(t-1)`.

## Trade-offs

What we **give up**:

* The advertised "matrix `A` is fixed across all `t`" property. Each period
  rebuilds Clarabel from scratch (data tape varies and the active row
  subset may change), so the KKT factorisation is *not* reused. This is a
  real cost on the legacy "full-`E`" path. It is *not* a regression on the
  preferred block path — which already rebuilt per period — so the
  end-to-end pipeline's wall-clock budget is comparable.
* Embarrassingly-parallel per-year solves under `WarmStartPrior`. The warm
  start serialises the period loop (`t` needs `a*(t-1)`).
  `AggregationPrior` stays parallelisable if ever needed.

What we **gain**:

* `n_timesteps`× less storage on `G`, `row_meta`, and the per-period
  payload arrays.
* The no-cross-time-coupling invariant is structurally impossible to
  violate: `StructG` has no time axis on the column side. The runtime
  partition check is gone.
* One solver loop, one uncprop loop. ~1500 lines deleted across
  `constraints/engine.py`, `solver.py`, `uncprop.py` and their tests.
* Activation becomes a first-class boolean. Downstream readers no longer
  need to know which float value means "inactive".
* The Stage-2 ↔ Stage-3 seam is cleaner: `a_agg[:, t]` slices directly into
  the per-period inputs each builder sees.

## A new behavioural change worth flagging

`WarmStartPrior` is **not** a pure re-index — it is a deliberate model
change. Pre-refactor, each period was seeded by the original `a_agg`
(roughly equivalent to `AggregationPrior` today). Under
`WarmStartPrior`, `a*(t-1)` flows into `a_0(t)` through the *shared*
edge identity that StructG enforces, which makes the warm-start carry
meaningful across periods. Results for `t > 1995` legitimately differ
from the pre-refactor baseline; year-1995 still matches exactly (no
prior year to carry).

## Validation strategy

* **Re-index anchor (`AggregationPrior`)**: full pipeline `a*` / `c*` /
  posterior σ must match the Phase-0 baseline within `rtol ≈ 1e-6` for
  every year.
* **Warm-start model (`WarmStartPrior`)**: year 1995 must still match the
  baseline exactly; `t > 1995` validated by sanity invariants (mass
  balance closes on hard rows, posterior σ within `[l, u]`, no NaN
  regressions) — not by equivalence (results legitimately differ).

## Consequences for downstream readers

* Anything that loaded a pre-refactor `ConstraintRun` from
  `data/proc/const/` cannot reload it; the registry persistence schema
  changed (`struct_g.npz` + 2-D `c.npy` / `sigma_c.npy` / `active.npy` +
  `sigma_a.npy` 2-D + `timesteps.json`). Old runs are stale and will be
  regenerated by re-running the pipeline.
* `OptimResult` field rename `F_opt → a_star`. A read-only `F_opt`
  property survives as an alias for older callers; on persist the
  canonical `a_star` is written.
* `SIGMA_C_INACTIVE` / `BOUND_C_INACTIVE` symbols still exist in
  `pysuto.constraints.sum` for backwards-compatible imports but are not
  written by any builder under the new model — `active = False` is the
  replacement.
* The diagnostics layer (`pysuto.diag.imp`, `pysuto.diag.agg`,
  `pysuto.diag.solver`) is the last subsystem still on the pre-refactor
  schema. Its rewire is mechanical (replace `edge_index["time"]` with
  `acc.time_range`-based slicing; consume 2-D arrays) and will follow.
