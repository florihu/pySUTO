# Imputation Capability Revision

## Context

Three issues with the current Stage 2c imputation pipeline:

1. **No registry** — imputation runs are persisted to a manually-named folder
   (`copper_agg_imp_v0`) by `AggregationResult.save()`. No JSON index, no run-id,
   no time tag, no provenance link to feeds used.
2. **Pathological σ values** — after `FlatPriorImputer`, ~54 % of edges carry
   σ ≈ 4×10¹¹. The cause is two stacked failures (see § Sigma investigation).
3. **Diagnostic plots** — the per-step σ histogram looks identical to the per-step
   F histogram (artefact of σ ∝ |f| through Stage 1 propagation), so it adds no
   information. There is no per-method view of σ for debugging outliers.

This plan locks in the design decisions reached during the grill session
(branch `main`, conversation 2026-04-30).

---

## Decisions made

### A — Registry (mirrors `ConstraintRunRegistry`)

| # | Decision |
|---|---|
| A1 | New `ImputationRunRegistry` at `data/proc/imp/registry.json`. Auto-generated `run_id = imp_<label>_<YYYYMMDD_HHMMSS>_<shorthash>`. Manual folder convention dropped. |
| A2 | Two-tier persistence: thin `registry.json` index + per-run `metadata.json` sidecar holding step records, feeds list, imputer config snapshots. |
| A3 | Imputation log persisted as `imputation_log.parquet` (was CSV). Native list-column support, smaller, faster. |
| A4 | File IO moves from `AggregationResult.save()` into `ImputationRunRegistry.register()`. `AggregationResult.save/load` removed. `ImputationEngine.imputers` becomes a public read-only property. `register()` raises if `agg_result.imputation_log is None`. |

### B — Diagnostic redesign

| # | Decision |
|---|---|
| B1 | Replace per-step σ-histogram with **CV (σ/\|f\|)-histogram per step**. Removes redundancy with F panel. |
| B2 | New **σ-by-method log boxplot**, grouped by `log["method"]` column (~7 boxes incl. `parametric_capped` / `formula_capped`), outliers shown (1.5·IQR red dots). Includes a leftmost "covered" baseline box (Stage 2a observed σ). |
| B3 | Boxplot gets its own figure (`{tag}_imputation_sigma_box.png`). Layout becomes: `imputation_steps.png` (2×2: F, CV, coverage, edges/step), `imputation_sigma_box.png` (1×1), `imputation_timeseries.png` (unchanged). |

### C — Sigma fix

| # | Decision |
|---|---|
| C1 | `FlatPriorImputer.sigma_floor` strategy → **anchor floor to `max(\|f\|)` not max σ**. Default `sigma_floor_factor = 100`. New formula: `floor = sigma_floor_factor × max(\|f_finite\|)`. Decouples flat-prior σ from upstream σ contamination. |
| C2 | Add **CV cap** to `ParametricImputer` and `FormulaImputer`. New parameter `cv_cap = 100` (default). When `σ_j > cv_cap × \|f_j\|`, clip to `cv_cap × \|f_j\|` and log with method label `parametric_capped` / `formula_capped`. Triggers on top ~4 % of fills given current data. |
| C3 | The `f=1.0` bound clip in `conc_wr_use` (and similar) is a separate root-cause issue. Out of scope for this work. Tracked in `plans/issues/conc_wr_use_clip.md` for follow-up. |

### Cross-stage impact

| # | Decision |
|---|---|
| X1 | When `ConstraintEngine.run(...)` is called, the resulting `ConstraintRun.agg_id` field should be set to the imputation `run_id` (currently a free-form string). Closes const → imp → feed provenance chain. |

---

## File-by-file change list

### `src/pysuto/impute.py`

- `ImputationEngine`: expose `self._imputers` as `imputers` property.
- `FlatPriorImputer`:
  - Replace floor calc with `floor = sigma_floor_factor × max(|f|)` over finite `f`.
  - Default `sigma_floor_factor` 1e6 → 100.
  - Update docstring to reflect the new strategy (and warn that the old strategy
    tracked max σ, which contaminates with parametric outliers).
- `ParametricImputer`:
  - New `__init__` parameter `cv_cap: float = 100.0`.
  - In every fill path (CSV pairs and lambda relations), after computing
    `σ_j`: if `σ_j > cv_cap * abs(f_j)` and `f_j != 0`, set
    `σ_j = cv_cap * abs(f_j)` and emit log entry with `method="parametric_capped"`.
- `FormulaImputer`:
  - Same `cv_cap` mechanism; method label `"formula_capped"`. Note: existing
    `"formula_clipped"` label means *value* clipped to bound — keep it distinct.
- New module-level `BaseImputer.config()` → returns a JSON-able dict of the
  imputer's config (used by registry to snapshot per-step config).
  Default impl: `{k: v for k, v in self.__dict__.items() if json.dumps-able}`.

### `src/pysuto/agg.py`

- Delete `AggregationResult.save()` and `AggregationResult.load()` classmethods.
- Keep the dataclass definition exactly as-is (still updated in-place by engine).

### `src/pysuto/registry.py` (or new `src/pysuto/imp_registry.py`)

- New class `ImputationRunRegistry`, mirroring `ConstraintRunRegistry`:

  ```python
  class ImputationRunRegistry:
      REGISTRY_FILENAME = "registry.json"

      def __init__(self, storage_dir: str = "data/proc/imp"): ...
      def register(
          self,
          agg_result: AggregationResult,
          *,
          label: str,
          feeds: list[FeedRef],         # populated by FeedCollector.select()
          imputers: list[BaseImputer],  # from engine.imputers
          acc_id: str,
      ) -> str:                          # returns run_id
          ...
      def load(self, run_id: str) -> AggregationResult: ...
      def summary_df(self) -> pd.DataFrame: ...
  ```

- Per-run directory layout:

  ```
  data/proc/imp/
  ├── registry.json
  └── imp_copper_20260430_101455_a1b2c3d4/
      ├── metadata.json          ← run-level + step + feeds + imputer configs
      ├── agg_result.npz         ← arrays only (f_agg, sigma_agg, Omega, F, Σ, coverage, feed_ids)
      └── imputation_log.parquet ← per-edge fill log
  ```

- `registry.json` row schema:

  ```json
  {
    "run_id":      "imp_copper_20260430_101455_a1b2c3d4",
    "label":       "copper_agg_imp_v0",
    "acc_id":      "copper",
    "created_at":  "2026-04-30T10:14:55Z",
    "n_edges":     854460,
    "n_filled":    637235,
    "n_steps":     6,
    "feed_ids":    ["baci", "crude_ore", "icsg"]
  }
  ```

- `metadata.json` schema:

  ```json
  {
    "run_id":     "...",
    "label":      "...",
    "acc_id":     "...",
    "created_at": "...",
    "feeds": [
      {"feed_id": "baci",      "version": 1, "registry_path": "data/proc/feed/registry/baci/"},
      {"feed_id": "crude_ore", "version": 0, "registry_path": "..."},
      {"feed_id": "icsg",      "version": 0, "registry_path": "..."}
    ],
    "imputers": [
      {
        "step": 0,
        "type": "ParametricImputer",
        "config": {"cv_cap": 100.0, "_source_path": ".../ore_grade_params.csv"},
        "input_paths": ["data/input/params/ore_grade_params.csv"],
        "elapsed_seconds": 12.4,
        "edges_filled": 1234,
        "edges_missing_after": 5678,
        "pct_additionally_filled": 1.4
      },
      ...
    ],
    "totals": {
      "n_total":          854460,
      "n_covered_before": 217225,
      "n_covered_after":  854460,
      "n_imputed":        637235
    }
  }
  ```

- `register()` raises `ValueError` if `agg_result.imputation_log is None`.

### `src/pysuto/collector.py`

- `FeedCollector.select()`: store the resolved `(feed_id, version, registry_path)`
  triples on `self.selected` so the registry can introspect them at register time.
  (Currently `selected` only holds feed-id strings.)

### `src/pysuto/diag.py` — `ImpDiagnostic`

- Replace `_panel_sigma_dist` with `_panel_cv_dist` (computes `σ/|f|`, plots
  log₁₀(CV) histogram per step). Same color palette, same step labels.
- New method `_fig2_sigma_box()` returning a single-panel figure with a
  log boxplot grouped by `method` (covered baseline + ~7 imputer methods).
  Outliers shown as red dots (matplotlib default fliers, `flierprops` red).
  `n=…` annotation above each box.
- Re-number existing `_fig2_timeseries` → `_fig3_timeseries`.
- `plot()` writes 2–3 files: `_imputation_steps.png`, `_imputation_sigma_box.png`,
  `_imputation_timeseries.png`.
- `from_saved(run_dir, ...)` loads via `ImputationRunRegistry.load(run_id)` instead
  of `AggregationResult.load()`.

### `src/pysuto/constraints.py`

- `ConstraintEngine.run(agg_result, ..., agg_id=None)`: if `agg_id is None` and
  `agg_result` carries an `_imp_run_id` attribute (set by registry on register),
  use it. Otherwise use the existing fallback string.
- `AggregationResult` gains an optional `_imp_run_id: str | None = None` attribute
  set by `ImputationRunRegistry.register()` for downstream provenance.

### `scripts/init_collect_aggregate_impute.py`

- Replace `agg_result.save(OUTPUT_DIR)` with:

  ```python
  registry = ImputationRunRegistry(storage_dir="data/proc/imp")
  run_id = registry.register(
      agg_result,
      label=IMP_ID,
      feeds=collector.selected,
      imputers=engine.imputers,
      acc_id=acc.id,
  )
  ```

- Replace `ImpDiagnostic(...).plot(odir=os.path.join(OUTPUT_DIR, "diag"))` with
  the registry-derived path: `os.path.join(registry.storage_dir, run_id, "diag")`.

### Tests

- `tests/test_imp_registry.py` (new): round-trip `register` → `load`, raises
  on missing `imputation_log`, summary_df shape, run-id uniqueness.
- `tests/test_imputation.py`: extend with cv_cap fire test (parametric &
  formula), flat-prior floor anchored to max|f| not max σ.
- `tests/test_diag.py` (if exists, else add coverage in test_imputation):
  CV panel reads `σ/|f|`, boxplot has correct number of boxes incl. covered
  baseline.

---

## Sigma investigation — findings

User hypothesis: `sigma_floor_factor` is too high → σ ≈ 10¹².
**Confirmed, with two stacked failures:**

```
final sigma_agg statistics (post-pipeline):
  P50–P100 all ≈ 3.879e+11   ← 54 % of edges are flat-prior fills
  raw max  = 3.879e+11

floor calc check:
  floor = sigma_floor_factor × max(finite σ before flat-prior runs)
        = 1e6 × 3.879e+05
        = 3.879e+11   ✅ exact match

what produced the 3.879e+05 max upstream σ:
  param_id    = "conc_wr_use"
  description = "Cu concentration in WR (SB) — use edges (to-node=WR/SB)"
  f_after     = 1.0  (suspicious — bound-clipped)
  sigma_after = 3.88e+05  →  CV ≈ 380,000

parametric CV percentiles (all 165,854 parametric fills):
  P50:    5.9
  P75:   16.9
  P90:   39
  P95:   62
  P99:   2,012
  P99.9: 73,230
  max:   387,915
```

**Diagnosis:**

1. `conc_wr_use` is producing σ ≈ 4×10⁵ on near-unity values — itself broken.
   Likely: a low-concentration coefficient (e.g. 1/0.999 inversion of a 99.9 %-Cu
   cathode → waste-residue concentration) inflates σ via Gaussian propagation.
   Combined with the `f=1.0` bound clip, this leaves the σ unbounded.
2. `FlatPriorImputer` then takes `max(σ)` and multiplies by 1e6, applying the
   contaminated upstream σ to ~460 k edges (54 % of the model).

**Fixes** (both applied):

- **C1** changes the floor reference from `max(σ)` to `max(|f|)`, breaking the
  contamination chain at the flat-prior layer.
- **C2** caps σ at the imputer level, breaking it at the source.

After both fixes, with `cv_cap=100` and `sigma_floor_factor=100`:
- `conc_wr_use` σ caps at 100 × 1.0 = 100 (was 388,000) → method label changes
  to `parametric_capped`.
- Flat-prior floor anchors to max|f| ≈ 1e5 kt (largest national flow) →
  floor ≈ 100 × 1e5 = 1e7. Large enough to be non-informative, small enough
  to keep `Σ_a⁻¹` well-conditioned.

---

## Migration / risk

- **Existing run** at `data/proc/imp/copper_agg_imp_v0/` not touched. Users can
  delete after rerunning under new registry, or leave alongside.
- **Stage 3 callers** that depend on `AggregationResult.load(odir)` will break
  (the method is gone). The only known caller is the diagnostic `from_saved`
  classmethod — updated as part of this change. `grep` for any others before
  merging.
- **CV-cap semantics**: cap-fires use the *clipped* σ in subsequent imputers
  that source from this edge. This is intentional — a sane σ propagates better
  than a 10⁵ outlier. Original σ is *not* preserved (unlike `formula_clipped`'s
  pre-clip σ retention). If preservation is needed later, add a `sigma_pre_cap`
  field to the log entry.
- **Backward compat**: `imputation_log.csv` no longer written. Anyone reading
  it from disk needs to switch to `imputation_log.parquet` or run the
  one-liner export.

---

## Followups (out of scope, but tracked)

- `plans/issues/conc_wr_use_clip.md` — investigate the `f=1.0` bound clip and
  the upstream σ inflation in `conc_wr_use`.
