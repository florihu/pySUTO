# Investigate `conc_wr_use` σ inflation and `f=1.0` bound clip

## Title
ParametricImputer: `conc_wr_use` produces σ ≈ 4×10⁵ on f=1.0 — likely bound-clip masking unbounded σ propagation

## Labels
bug, imputation, stage-2

## Body

Discovered while debugging Stage 2 σ pathology (see `plans/imputation_revision.md`).
This is the upstream source of the σ ≈ 10¹¹ values observed across ~54 % of
edges after `FlatPriorImputer`. The flat-prior layer is being addressed
separately (C1+C2 in the imputation revision plan); this issue tracks the
root cause one layer up.

### Symptom

In imputation run `data/proc/imp/copper_agg_imp_v0/`, the top 10 σ_after values
in the parametric step are all from `param_id = "conc_wr_use"`:

```
edge_id  f_after  sigma_after  CV          description
3838     1.0      387915.33    387915.33   Cu concentration in WR (SB) — use edges
3830     1.0      386364.12    386364.12   Cu concentration in WR (SB) — use edges
3826     1.0      385521.75    385521.75   Cu concentration in WR (SB) — use edges
...                                         (all 10 entries: f_after = 1.0 exactly)
```

The `f_after = 1.0` value is suspicious — it is exactly 1.0 (no decimals),
suggesting an upper-bound clip on either the source flow or the fill itself.
The σ of ~3.9×10⁵ is propagated through Gaussian quadrature from the source
σ × coefficient, but f is clipped, so CV explodes (σ = 388k × f).

### Suspected mechanism

`conc_wr_use` is a "concentration" parameter (data category 3): links the same
`(from_id, to_id, time)` pair across two valuations (Mass ↔ CC), with the
parametric edge-pair semantics described in
[README.md § Concentration](../../README.md#parameter-types-and-the-edge-pair-schema).

For a waste-residue (WR) use edge in CC valuation:

```
f_CC = coeff × f_Mass     (when Mass observed, CC missing)
f_Mass = f_CC / coeff     (when CC observed, Mass missing)
```

If the concentration coefficient is small (e.g. 0.001 for a low-concentration
slag) and we are computing `f_Mass = f_CC / coeff`, then `1/coeff = 1000` and
σ propagates as `σ_Mass = (1/coeff) × σ_CC = 1000 × σ_CC`. With moderate σ_CC
(say 400) → σ_Mass = 400,000.

Then either:
- a bound on `f_Mass ≤ 1.0` is hit (the WR/SB flow is small by definition)
  and the value is clipped to 1.0 but σ is *not* clipped, leaving a
  CV of 400,000;
- or the value happens to land at 1.0 by coincidence (unlikely given 10
  edges all read exactly 1.0).

### What to investigate

1. **Confirm the bound clip path.** Check `ImputationEngine.run()` post-step
   bounds enforcement at [impute.py:320–344](../../src/pysuto/impute.py#L320-L344).
   The clip currently rewrites `f` but leaves `σ` untouched. Confirm whether
   `conc_wr_use` edges are hitting this clip (search the warning logs from the
   `copper_agg_imp_v0` run).
2. **Check the `conc_wr_use` rows in the params CSV.** Look at
   `data/input/params/copper_conc_params.csv` for rows with `param_id = "conc_wr_use"`.
   Examine `value_nominal`, `unc_type`, and direction. Is the coefficient ≪ 1?
   What does `bound_upper` (if any) say?
3. **Decide on σ-on-clip semantics.** When the engine clips a fill to a bound,
   should it also clip σ? Three options:
   - keep σ as-is (current behaviour) — preserves "honest" propagation but
     produces nonsensical CVs;
   - clip σ proportionally — `σ_new = σ × (f_clipped / f_pre_clip)`;
   - apply the new `cv_cap` (from imputation revision C2) at the engine level
     in addition to the imputer level — would catch this case automatically.
4. **Audit other concentration parameters** for similar patterns
   (`conc_*` params with small coefficients, especially when used in the
   "divide" direction).

### Acceptance criteria

- Root cause identified (which params, which direction, which clip).
- σ-on-clip policy decided and implemented.
- After fix: no parametric fills with CV > 1000 unless explicitly justified
  (e.g. an intentionally non-informative prior with `cv_cap` opt-out).
- Test added in `tests/test_imputation.py` covering the bound-clip + σ
  scenario.

### Out of scope

- The `FlatPriorImputer` floor strategy and `cv_cap` mechanism — covered by
  `plans/imputation_revision.md`. This issue addresses the *upstream* parametric
  σ inflation that caused the flat-prior contamination, not the contamination
  itself.

### How to file

```bash
gh issue create \
  --title "ParametricImputer: conc_wr_use produces σ≈4×10⁵ on f=1.0 — likely bound-clip masking σ propagation" \
  --label "bug,imputation,stage-2" \
  --body-file plans/issues/conc_wr_use_clip.md
```
