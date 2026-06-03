# ROW edges carry the worst geographical pedigree

A flow assigned to **ROW** (Rest of World) is a geographic *aggregate* of countries that are not individually represented, so its data quality on the geographical axis is the worst possible — yet pedigree scores are otherwise **dataset-level** (one [`PedigreeScores`](../../src/pysuto/unc.py) per feed), treating ROW rows like fully-resolved ones. We now let any feed opt in via a `row_penalty` boolean ([`DataFeedMetadata`](../../src/pysuto/meta.py)): when set, every ROW-touching edge (`region_from`/`region_to == "ROW"`, see [`CONTEXT.md`](../../CONTEXT.md)) is propagated as if its `geographical_correlation` were the worst score (**4**), inflating its σ. baci, icsg and crude_ore all opt in.

Mechanically this is a per-root-edge `cv_root_scale` multiplier on the existing power-mean propagation ([`unc.py` `apply_with_uncertainty`](../../src/pysuto/unc.py)); scaling ROW edges by `ratio = cv(geo=4)/cv(geo=baseline)` is **exactly** equivalent to re-propagating those rows at `geographical_correlation=4` (deterministic region mapping ⇒ a ROW root edge only receives fan-in from ROW source rows). The same penalty reaches **imputed** ROW edges by quadrature-adding `CV_GEO4` to the Stage-2 FlatPrior CVs, replacing the previous ad-hoc `FLAT_ROW_INFLATION = ×1.5`.

## Considered options

- **A. Keep the Stage-2 `FLAT_ROW_INFLATION ×1.5` and add nothing in Stage 1.** Rejected: an opaque magic factor that only touches imputed edges; fed ROW edges stay un-penalised, and ×1.5 has no pedigree grounding.
- **B. Penalise the *temporal* dimension.** Rejected during grilling: ROW is a *geographic* aggregate; geographical correlation is the honest axis. (The original request said "temporal"; the design moved to geographical.)
- **C. Carry a full per-source-row pedigree vector through a re-derived power-mean.** Rejected: mathematically *identical* to the per-edge `cv_root_scale` here (see above), so strictly more code for zero numerical difference.
- **D. Per-edge geographical penalty via `cv_root_scale` (Stage 1) + quadrature-added `CV_GEO4` (Stage-2 FlatPrior). (chosen.)**

## Consequences

- **One deliberate exception to dataset-level pedigree.** Pedigree is feed-level everywhere *except* this ROW carve-out, which is per-edge. A future reader seeing per-edge σ inflation should find it here, not assume a bug.
- **Provenance.** A `row_penalized` boolean mask is added to [`PropagationResult`](../../src/pysuto/unc.py) and persisted in the feed run's `result.npz`; the run's `metadata.json` `stage_summary` records counts, value share and `cv_norm→cv_pen`.
- **Two stages, disjoint edge sets, one notion.** Stage 1 penalises **fed** ROW edges; Stage-2 FlatPrior penalises **imputed** ROW edges. The Stage-2 path is inert under the default `FLAT_MODE="noisy_zero"` (active only in `magnitude_aware`).
- **Known redundancy (deferred).** ROW detection now exists in three places — the registered `row` input mask (`data/input/mask/row/`), the Stage-2 driver's inline `expand_edge_index`, and feedpipe's inline `acc`-based mask. feedpipe computes it inline (not via the `row` mask, not via `param_relation`) to respect the one-way import direction (README §"Import direction"). Unifying onto `acc.masks["row"]` is left for a later cleanup.
- **Bound.** Quadrature-added FlatPrior `cv_hi` stays ≤ 1.0 (row_use peaks at ≈0.54), so the FlatPrior `cv_hi ≤ 1.0` assertion holds without a clamp.
