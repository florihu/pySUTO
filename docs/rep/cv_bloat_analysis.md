# CV bloat analysis — Stage 1 / Stage 2 / imputation

Run under analysis: `data/proc/imp/imp_copper_agg_imp_v0_20260430_061428_05d76679`.
Symptom: a long tail of root-edge CVs being capped at 100 (cv_cap=100.0
in every imputer's config).

## Conclusion up front

The CV bloat is **not** produced by Stage 1 (concordance propagation) and
**not** by Stage 2a (inverse-variance aggregation). It is produced
entirely by the imputation pipeline.

The earlier hypothesis — "outliers are already in the pre-impute data"
— came from reading the σ-by-method boxplot, which showed large σ in
the `covered` baseline. Large σ ≠ large CV. The covered baseline's σ
distribution is dominated by the magnitude of `f_agg`; once divided
through, CV is bounded by 0.23 (= max CV_tot across feeds).

## Theory: where can CV bloat come from?

### Stage 1 — concordance

Each feed propagates a single dataset-level CV through

$$\sigma_\text{root}[j] = \mathrm{CV}_\text{tot} \sqrt{\sum_i W_{ij}^2 f_i^2}$$

so

$$\mathrm{CV}_\text{root}[j] = \frac{\sigma_\text{root}[j]}{|F_\text{root}[j]|}
   = \mathrm{CV}_\text{tot} \cdot
     \frac{\sqrt{\sum_i W_{ij}^2 f_i^2}}{\left|\sum_i W_{ij} f_i\right|}.$$

Two regimes inflate CV beyond CV_tot:

- **Fan-in dilution.** Many similar-magnitude positive contributions in
  one column. The numerator sums in quadrature, the denominator sums
  linearly, so the ratio grows toward √n. Expected, bounded.
- **Sign cancellation.** Positive and negative `f_src` map onto the same
  root edge. Numerator stays bounded, denominator collapses → CV→∞.

Both are **mathematical, not bugs.** They are the price of mixing
heterogeneous source contributions into one root edge. If they show up
empirically, the fix is on the data side (split signs, use a different
concordance), not on the propagation code.

### Stage 2a — aggregation

$$f_\text{agg}[j] = \frac{\sum_k f_k[j] / \sigma_k^2[j]}{\sum_k 1/\sigma_k^2[j]},
\qquad \sigma_\text{agg}[j] = \frac{1}{\sqrt{\sum_k 1/\sigma_k^2[j]}}.$$

Edge-level CV after aggregation can exceed any per-feed CV when feeds
disagree near zero (e.g. one feed reports +x, another -x with similar
σ; `f_agg` ≈ 0, `σ_agg` ≈ smallest σ; CV → ∞). Otherwise σ_agg ≤ min σ_k
and CV_agg is bounded by max(CV_k).

### Stage 2c — imputation

Each imputer fills a missing edge as a linear combination of source
edges. The README's "Mathematical foundations" section gives the
relevant Gaussian propagation:

$$f_j = \sum_l \alpha_l f_{i_l},
\qquad \sigma_j^2 = \sum_l \alpha_l^2 \sigma_{i_l}^2 + (\varepsilon f_j)^2.$$

CV bloat sources here are concrete and many:

1. **Sources with high CV propagate.** If a source has CV=0.2 and the
   coefficient α=0.12 (e.g. scrap return), then
   `σ_j² ≈ (0.12 σ_i)² + (ε f_j)²` while `f_j = 0.12 f_i`. CV of the
   target equals √(CV_i² + ε²) — same order as source.
2. **Tiny `f_j` with non-trivial `ε`.** `(ε f_j)²` is harmless when f_j
   is small in absolute terms — but CV is σ/|f_j|, so
   ε contributes ε itself. ε defaults are 0.05–0.25 in the codebase;
   that's a CV floor, not a bloat source on its own.
3. **`FormulaImputer` with sign cancellation (clip to 0).** If
   `domestic = supply − exports` with supply ≈ exports, the clipped
   value goes to 0 but `sigma_after` carries pre-clip σ
   (README: *"sigma_after carries the pre-clip σ as a conservative
   uncertainty estimate"*). f → 0 with σ ~ O(supply) → CV → ∞. This is
   the cleanest known mechanism for the high-CV tail and is the most
   likely culprit here.
4. **`TimeImputer` with `gap_inflation` and large gaps.** σ inflated by
   `γ_0^g`. With γ_0=1.05 and g=10, that's 1.63× — modest.
5. **`FlatPriorImputer`.** Default `sigma_floor_factor=100` × max(|f|).
   `f_j = 0`. CV is undefined (σ/0 = inf or NaN); this is what the cap
   silently maps to 100.

Of these, (3) and (5) produce CV→∞ and are bounded only by the
`cv_cap=100` setting on each imputer.

## Empirical results

Run via `scripts/explo/cv_bloat_diag.py`.

### Layer 1 — per feed (Stage 1 propagation)

| feed       | cv_src   | n_root_with_F | cv_root p50 / p90 / p99 / max  | fan-in p50 / p99 / max |
|------------|----------|---------------|--------------------------------|------------------------|
| baci       | 0.0392   | 72,830        | 0.0392 / 0.0392 / 0.0392 / 0.0392 | 1 / 15 / 90        |
| crude_ore  | 0.230    | 865           | 0.230  / 0.230  / 0.230  / 0.230  | 1 / 26 / 28        |
| icsg       | 0.0226   | 175,279       | 0.0226 / 0.0226 / 0.0226 / 0.0226 | 1 / 16 / 26        |

**Every per-feed CV_root equals CV_tot at every percentile.** Despite
fan-in up to 90 sources per column and 7,179 negative `f_src` values in
icsg, no edge bloats. Reason: in this data each column of W has one
dominant source per root edge (the per-feed `dominant_share_p10` was
not even reached — no edges hit CV>1 to compute the statistic).

So the concordance propagation formula is **not** producing bloat for
any of the three current feeds. The math is fine; the data does not
trigger either the fan-in or sign-cancellation regime at the per-feed
level.

### Layer 2 — pre-imputation aggregation

Recomputed directly from per-feed `F` and `Sigma` matrices stored in
`agg_result.npz` (the on-disk `f_agg` / `sigma_agg` are
post-imputation):

| metric                       | value    |
|------------------------------|----------|
| n_covered_pre_impute         | 217,225  |
| n_multi_feed_edges           | **0**    |
| cv_agg_pre p50 / p90 / p99   | 0.0226 / 0.039 / 0.039 |
| cv_agg_pre p999 / max        | 0.230 / 0.230 |
| pct above CV=1 / 10 / 100    | 0 / 0 / 0 |

The three feeds do **not overlap** — every covered root edge is covered
by exactly one feed, so aggregation is a pass-through and CV_agg = the
covering feed's CV_tot. Maximum CV in pre-impute aggregated data is
0.23 (from crude_ore).

### Layer 3 — post-imputation (from saved metadata of the run)

From the diagnostic image of the affected run:

- p50 CV ≈ 0.04 (carried over from baci-covered edges)
- p90 CV ≈ 17.7
- p99 CV = 100 (cap)
- max CV = 1326 (one outlier escaping the cap, presumably FlatPrior)
- 30.9% of edges with CV > 1
- 15.4% of edges with CV > 10
- ~0.02% of edges with CV > 100

**All of this is introduced by imputation.**

## What's the dominant imputation mechanism?

The metadata.json fill counts:

| step | imputer          | edges filled | % of total |
|------|------------------|--------------|-----------|
| 0    | ParametricImputer (ore_grade)        | 820     | 0.10 |
| 1    | ParametricImputer (conc)             | 162,426 | 19.0 |
| 2    | ParametricImputer (efficiency)       | 2,608   | 0.31 |
| 3    | FormulaImputer (domestic balances)   | 395     | 0.05 |
| 4    | TimeImputer                          | 10,563  | 1.24 |
| 5    | **FlatPriorImputer**                 | **460,423** | **53.9** |

53.9% of edges are filled by the flat prior with `f=0` and `σ = 100 ×
max(|f|)`. **For every flat-prior edge, CV is undefined** (σ/0). The
`cv_cap=100` is what produces the visible cap-to-100 tail; without it
those edges would carry CV=inf or NaN.

**The per-step CV histogram (see imputation_steps.png panel (1,0))
should confirm this:** step 5 should be a tall narrow peak at the cap,
distinct from steps 0–4. The current diagnostic miscolors it as step 4
(see plans/diag fixes), which has been masking exactly this signal.

## Strategies

Listed in order of decreasing impact on the CV tail.

1. **Stop using FlatPriorImputer with `f=0, σ=100·max|f|`.** This is the
   single largest CV-bloat source (54% of edges). Options:
   - **Drop it.** Leave residual NaN edges as NaN; let the optimiser's
     bounds + balance constraints determine them. Σ_a is a diagonal
     prior weight; an uninformative prior is *infinite* σ, i.e. zero
     weight. The current "100×max|f|" is a finite but huge σ that the
     optimiser still pulls toward 0 with weight `1/σ²` ≈ 1e-22 per kt²
     — practically equivalent to no prior, but visually a CV outlier.
     If the optimisation is run with `f=NaN` allowed, this is cleanest.
   - **Replace `f=0` with a sector-median magnitude** (per
     valuation/region) so CV is bounded. Loses the "missing"
     interpretation but produces sane diagnostics.
   - **Keep `f=0` but encode "no information" via a NaN CV** rather
     than the cap. Means no plot bloat, but downstream code must
     tolerate NaN CV.

   Recommendation: **drop FlatPriorImputer** and let the optimiser fill
   from constraints. If the optimiser cannot run without a prior on
   every edge, fall back to the median-magnitude option.

2. **Audit `FormulaImputer` clip-to-zero behaviour.** 395 edges only
   (0.05%), but each is a candidate for the σ→large, f→0, CV→∞
   pattern. Look at the `formula_clipped` rows in the imputation log:
   if their median CV is much higher than the `formula` median, they
   are contributing to the tail beyond their count. Two responses:
   - **Drop clipped edges entirely** (leave NaN, let later steps or
     the optimiser handle).
   - **Reset σ along with f when clipping** — if you decide the value
     is zero by physical bound, you also decide σ is small (you have
     evidence of a bound, not just a measurement). σ_after = 0 or
     ε·median_neighbour rather than the pre-clip σ.

3. **Don't `cv_cap` silently.** The cap is currently an emergency
   bandaid hiding pathology. Either:
   - Log every capped row separately so the count is visible, or
   - Replace the cap with a hard error (`raise` if any imputer
     produces CV>cap), forcing each pathological case to be addressed.

4. **No action on Stage 1.** The empirical evidence shows Stage 1
   propagation is well-behaved on this data. Keep the formula as is.
   If a future feed *does* trigger fan-in or sign-cancellation, the
   per-feed CV histogram in the feed diagnostic will surface it
   (after the linear-axis fix).

5. **No action on Stage 2a.** With non-overlapping feeds, aggregation
   is a pass-through. Bloat from sign-disagreement aggregation cannot
   occur until two feeds cover the same edge.

## Next concrete step

Run the per-step CV-by-method boxplot (Problem 3.2 — to be implemented)
once Problem 3 changes are in. That will quantitatively confirm
strategy (1) is the right priority by isolating the `flat_prior`
group's CV distribution.
