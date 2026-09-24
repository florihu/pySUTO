# Persistence map — where each artefact lives

Quick reference for the on-disk layout of a pySUTO run. Re-read before loading any artefact for the first time in a session.

## Stage 1 — DataFeed

```
data/proc/feed/registry/
├── registry.json            # index of all feeds + versions
├── metadata.json            # per-feed metadata (pedigree, source, etc.)
├── <feed_id>.pkl            # full DataFeed object (incl. W, F_src, pedigree)
└── <feed_id>.result.npz     # arrays: F_root, sigma_root, W (sparse)
```

Useful arrays in `result.npz`:
- `F_root` — `(N_edges,)` — feed contribution on the root edge index.
- `sigma_root` — `(N_edges,)` — propagated σ on the root edge index.

## Stage 2 — Imputation / Aggregation

```
data/proc/imp/<imp_run_id>/
├── metadata.json            # sidecar (see Sidecar schema below)
├── agg_result.npz           # main array bundle
├── imputation_log.parquet   # per-fill provenance
└── diag/                    # plots (optional, written when diag_plot=True)
```

`agg_result.npz` keys:
- `f_agg` — `(N_edges,)` — aggregated flow vector.
- `sigma_agg` — `(N_edges,)` — aggregated σ.
- `F` — `(N_feeds, N_edges)` — per-feed observation matrix (NaN where uncovered).
- `Sigma` — `(N_feeds, N_edges)` — per-feed σ matrix.
- `Omega` — `(N_feeds, N_edges)` — inverse-variance weight matrix (columns sum to 1).
- `coverage` — `(N_edges,)` — bool / count of feeds observing each edge.
- `feed_ids` — `(N_feeds,)` — string array of feed IDs aligned with `F` rows.

`imputation_log.parquet` columns:
- `edge_id`, `step`, `method`, `source_edges`, `coefficients`, `f_before`, `f_after`, `sigma_before`, `sigma_after`, `description`, `param_id`.

## Stage 3 — Constraints

```
data/proc/const/<const_run_id>/
├── metadata.json
├── G.npz                    # scipy.sparse CSR; shape (M_rows, N_edges)
├── a0.npy                   # (N_edges,) prior flow vector
├── sigma_a.npy              # (N_edges,) prior σ (post-regularisation)
├── u.npy                    # (N_edges,) upper bounds
├── l.npy                    # (N_edges,) lower bounds
├── row_meta.parquet         # one row per constraint row
└── timesteps/
    └── t_<year>.npz         # per-t: c (M,), sigma_c (M,)
```

`row_meta.parquet` columns:
- `constraint_id` — string ID built from `(builder, scope, …)`.
- `type` — `mass_balance | sum_eq | ratio_eq | ratio_bound | flow_bound`.
- `is_hard` — bool.
- `cone_type` — `zero | nonneg | free`.
- `time` — year string for time-bound rows, `None` for time-agnostic.

Loading:
```python
from scipy import sparse
import numpy as np, pandas as pd
G = sparse.load_npz(const_dir / "G.npz")
a0 = np.load(const_dir / "a0.npy")
sigma_a = np.load(const_dir / "sigma_a.npy")
row_meta = pd.read_parquet(const_dir / "row_meta.parquet")
t_data = np.load(const_dir / "timesteps" / f"t_{year}.npz")
c, sigma_c = t_data["c"], t_data["sigma_c"]
```

## Stage 4 — Solver

```
data/proc/opt/<opt_run_id>/
├── metadata.json
├── timesteps/
│   └── t_<year>.npz         # per-t: F_opt, a_prior, cols_t, rows_t, eps_opt, c_star, residuals, iter_history
└── diag/                    # SolverDiag figures sd_*.png, ranking_<family>.csv, provenance.md,
                             # edges_<year>.csv / constraints_<year>.csv (SolverStatsDiagnostic)
```

`t_<year>.npz` keys — **block path** (every run since the year-local fix; `cols_t` is the
schema discriminator):
- `F_opt` — `(E_t,)` — optimised flow over the year's edges `cols_t` — a **year-local slice**,
  not `(N_edges,)`. Global edge `e` is `F_opt[np.searchsorted(cols_t, e)]`, or scatter
  `F = zeros(E); F[cols_t] = F_opt` before any product with `G`.
- `a_prior` — `(E_t,)` — the prior the objective used for the year (cold: `f_agg(t)`; warm: `a_post(t−1)`).
- `cols_t` / `rows_t` — `(E_t,)` / `(M_t,)` int64 — global edge / row indices of the slices.
- `eps_opt` — `(n_free_t,)` — slack on the year's free rows.
- `c_star` — `(M_t,)` — `c + ε` over `rows_t`: equals `G @ F_opt` on soft rows and **`c` itself on
  hard rows** (so a `c_star − c` "residual" is identically 0 there).
- `residuals` — `(M_t,)` — `G @ F_opt - c` over `rows_t` (sign-aware; the true hard-row residual).
- `iter_history` — structured array (`iter, cost_primal, cost_dual, gap_abs, res_primal, res_dual,
  ktratio, mu, step_length`), one record per Clarabel iteration.

Legacy full-vector runs (empty `G_blocks`) store `(N_edges,)` / `(M,)` vectors with `cols_t = rows_t = None`.

Solver telemetry per timestep lives in the sidecar `stage_summary.per_timestep[year]`:
`objective, iterations, solve_time_s, solver_status, convergence` (Clarabel dict).

## Stage 5 — Uncertainty propagation

```
data/proc/uncprop/<uncprop_run_id>/
├── metadata.json
└── timesteps/
    └── t_<year>.npz         # per-t: sigma_a_star, lower, upper
```

`a*` is NOT stored — join on the upstream `opt_run_id` to get F_opt.

## Sidecar schema (`metadata.json`)

Every stage writes the same shape:

```json
{
  "run_id": "opt_20260521_100428_9d33a0b9",
  "stage": "opt",
  "label": "",
  "created_at": "2026-05-21T10:04:28Z",
  "parents": {
    "inputs":   { "acc": {...}, "params": [...] },
    "upstream": { "const_run_id": "const_20260521_091641_8d374993",
                  "imp_run_id":   "imp_20260521_074525_b1921faf",
                  "feed_run_ids": ["..."] }
  },
  "stage_config": { ... },          // resolved knobs (p, q, alpha_min, ...)
  "stage_summary": { ... }          // per-timestep convergence, counts, etc.
}
```

`parents.upstream` only contains the keys that apply to that stage. Walk recursively to get the full chain.

## Edge / node indices

The `AccountingStructure` (Stage 0) writes its node & edge tables to:

```
data/proc/acc/<acc_id>/
├── node_index.csv
└── edge_index.csv
```

`edge_index.csv` columns include `edge_id`, `sector_from`, `sector_to`, `entity_from`, `entity_to`, `region_from`, `region_to`, `valuation`, `time`. Load with `pandas.read_csv` and join on `edge_id` to translate column indices into human-readable identifiers — essential for Step 3 stratification and Step 4 traces.

The acc_id is in the sidecar `parents.inputs.acc.id`.
