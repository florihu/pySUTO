# pySUTO

**Python Supply–Use Table Optimiser**

A modular Python framework for constructing, reconciling, and propagating uncertainty through material flow accounts derived from heterogeneous data sources. pySUTO takes raw statistical data feeds, maps them onto a common accounting structure, aggregates across sources with inverse-variance weighting, imputes missing values, and solves a supply–use constrained linear optimisation problem — all while tracking uncertainty end-to-end from raw data quality scores to final edge estimates.

---

## Table of contents

1. [Overview](#overview)
2. [Three-stage architecture](#three-stage-architecture)
3. [Module reference](#module-reference)
4. [Mathematical foundations](#mathematical-foundations)
5. [Workflow](#workflow)
6. [File structure](#file-structure)
7. [Testing strategy](#testing-strategy)
8. [Installation](#installation)

---

## Overview

Supply–use tables and material flow accounts are assembled from multiple data sources that differ in classification, temporal resolution, regional scope, and data quality. pySUTO addresses three interrelated problems:

- **Heterogeneous classifications** — raw data uses different sector taxonomies, regional aggregations, and time resolutions. A concordance layer maps every source edge onto a common root accounting structure via a sparse weight matrix.
- **Multi-source uncertainty** — each data source carries a pedigree quality score (Laner et al. 2016) that translates into a coefficient of variation. Uncertainty is propagated through every transformation step using Gaussian error propagation.
- **Physical consistency** — supply must equal use for every node and every material parameter efficiency must be respected. A linear optimiser with σ-weighted objective enforces these constraints while staying as close as possible to the aggregated data, with full uncertainty quantification on the optimised values.

---

## Three-stage architecture

The package is organised into three sequential stages. Each stage has a clear input, a clear output, and its own persistence layer.

### Stage diagram

![Full package architecture](figs/architecture.svg)

### Data flow

![Stage-to-stage data flow](figs/dataflow.svg)

### Stage 1 — Feed pipeline

Runs once per data source. Reads metadata (including pedigree quality scores) from an Excel sidecar, builds a concordance table that maps source edges to root edges, assembles a sparse weight matrix, applies it to the source flow vector, and propagates uncertainty. Results are persisted in the `DataFeedRegistry`.

**Inputs:** raw flow data, classification mapping tables, Excel metadata file  
**Outputs per feed:** weight matrix `W`, root flow vector `F_root`, standard deviation `σ_root`, metadata sidecar

### Stage 2 — Aggregation and imputation

Runs once across a selected set of registered feeds. `FeedCollector` loads `F_root` and `σ_root` for each selected feed from the registry and aligns them onto the common edge index. `AggregationMath` stacks the feed observations into a data matrix **F** and computes the inverse-variance weighted mean using a normalised weight matrix **Ω**, yielding `f_agg` as a single matrix–vector product. `ImputationEngine` then fills missing edges (root edges not covered by any feed) using parametric relations or time-series methods, logging every fill.

**Inputs:** feed IDs, registry  
**Outputs:** `AggregationResult` with `f_agg [N_r]`, `σ_agg [N_r]`, weight matrix **Ω**, imputation log

### Stage 3 — Linear optimisation

Runs as a loop over T timesteps. `ConstraintBuilder` assembles the constraint matrix `G` and right-hand-side vector `c` from the accounting structure (supply = use balances) plus user-defined parametric relations. `LinearOptimiser` solves the σ-weighted least-squares problem with equality constraints. Each timestep is warm-started from the previous solution. `OptUncertainty` propagates input uncertainty into the optimised solution via the KKT sensitivity Jacobian. All results are collected into an `OptimRun` and persisted.

**Inputs:** `AggregationResult`, accounting structure, parametric relations  
**Outputs per timestep:** `OptimResult` with `F_opt [E]`, `σ_opt [E]`, residuals, solver status  
**Outputs per run:** `OptimRun` with full provenance trace

---

## Module reference

![File structure overview](figs/file_structure.svg)

### Existing modules

| File | Classes | Responsibility |
|---|---|---|
| `acc.py` | `AccountingStructure` | Node/edge index, supply–use topology, parametric relations; uses `scipy.sparse` COO/CSR |
| `conc.py` | `ConcordanceTable`, `WeightMatrix` | src→root edge mapping, sparse W matrix (scipy.sparse CSR) |
| `feed.py` | `DataFeed` (ABC) | Base class; owns `weight_matrix` (CSR), pedigree, `F_src`, persistence |
| `feedpipe.py` | step functions, `run_feed_pipeline` | Orchestrates steps A→D for a single feed; delegates to `DataFeedRegistry` |
| `unc.py` | `PedigreeScores`, `UncertaintyMixin`, `PropagationResult` | Laner et al. pedigree → CV; Gaussian error propagation through W (scipy.sparse) |
| `meta.py` | `DataFeedMetadata` | Parses Excel sidecar; stores identification, pedigree scores, provenance, computed outputs |
| `registry.py` | `DataFeedRegistry`, `_RegistryEntry` | Persistent JSON index + per-feed metadata.json, pickle, result.npz |
| `utils.py` | `_NpEncoder`, io helpers | JSON serialisation, numpy-safe dump/load |
| `logging.py` | — | Structured run logging |

### New modules

| File | Classes | Responsibility |
|---|---|---|
| `collector.py` | `FeedCollector`, `AggregationResult` | Load and align selected feeds; stack into **F** and **Σ** matrices; delegate to `agg.py` |
| `agg.py` | `AggregationMath` | Build weight matrix **Ω**; compute `f_agg = 1ᵀ(Ω ⊙ F)` and `σ_agg` |
| `impute.py` | `ImputationEngine`, `ParametricImputer`, `TimeSeriesImputer` | Fill missing edges; log every imputation |
| `constraints.py` | `ConstraintBuilder` | Assemble G and c from supply–use balances and parametric relations |
| `optim.py` | `LinearOptimiser`, `OptimResult` | σ-weighted equality-constrained least squares; warm-start |
| `optunc.py` | `OptUncertainty` | KKT sensitivity-based uncertainty propagation for constrained optimum |
| `optpipe.py` | `OptimPipeline`, `OptimRun` | T-step loop with warm-start; collect and persist results |

---

## Mathematical foundations

### Pedigree → coefficient of variation

Following Laner et al. (2016). Each data source carries five quality indicator scores s ∈ {1,2,3,4}.

**Reliability** (non-zero even at the best score, since even perfectly documented data carries residual uncertainty):

$$\mathrm{CV}_\mathrm{rel} = a \cdot e^{\,b\, s_\mathrm{rel}}, \qquad a = 0.75,\quad b = 1.105$$

**All other indicators** (zero at score 1, i.e. no additional uncertainty when the indicator is fully met):

$$\mathrm{CV}_i = a_i \cdot e^{\,b\,(s_i - 1)}, \qquad b = 1.105$$

Sensitivity parameter $a_i$: not sensitive $= 0.375$, medium sensitive $= 0.75$, highly sensitive $= 1.5$.

**Total CV** (quadrature sum — assumes independence across indicators):

$$\mathrm{CV}_\mathrm{tot} = \sqrt{\mathrm{CV}_\mathrm{rel}^2 + \mathrm{CV}_\mathrm{comp}^2 + \mathrm{CV}_\mathrm{temp}^2 + \mathrm{CV}_\mathrm{geo}^2 + \mathrm{CV}_\mathrm{other}^2}$$

### Stage 1 — Concordance error propagation

The weight matrix $\mathbf{W} \in \mathbb{R}^{N_\mathrm{src} \times N_\mathrm{root}}$ maps source edges to root edges, assembled as a `scipy.sparse.csr_matrix`. For a dataset with uniform $\mathrm{CV}_\mathrm{tot}$ across all source edges:

$$\boldsymbol{\Sigma}_\mathrm{src} = \mathrm{CV}_\mathrm{tot}^2 \cdot \mathrm{diag}(\mathbf{f}_\mathrm{src}^2)$$

$$\mathbf{f}_\mathrm{root} = \mathbf{W}^\top \mathbf{f}_\mathrm{src}$$

$$\boldsymbol{\Sigma}_\mathrm{root} = \mathbf{W}^\top \boldsymbol{\Sigma}_\mathrm{src}\, \mathbf{W} = \mathrm{CV}_\mathrm{tot}^2 \cdot \mathbf{W}^\top \mathrm{diag}(\mathbf{f}_\mathrm{src}^2)\, \mathbf{W}$$

$$\boldsymbol{\sigma}_\mathrm{root} = \mathrm{CV}_\mathrm{tot} \cdot \sqrt{(\mathbf{W} \odot \mathbf{W})^\top \mathbf{f}_\mathrm{src}^2} \qquad \text{(element-wise } \mathbf{W}^2\text{)}$$

In scipy.sparse code, $\mathbf{W} \odot \mathbf{W}$ is formed by squaring the `.data` array of a copy of the CSR matrix (no densification). The diversification property guarantees $\mathrm{CV}_\mathrm{root} \leq \mathrm{CV}_\mathrm{src}$ for every root edge.

### Stage 2 — Aggregation across N_f feeds as a matrix product

After Stage 1, each feed k has produced a root-edge observation vector **f**_k ∈ ℝ^(N_r) and a standard deviation vector **σ**_k ∈ ℝ^(N_r). Entries are NaN for root edges not covered by feed k.

**Data matrix and variance matrix**

Stack feed observations and variances row-wise:

$$\mathbf{F} \in \mathbb{R}^{N_f \times N_r}, \qquad F_{kj} = f_{\mathrm{root},k}[j] \quad (\text{NaN if feed } k \text{ has no observation at } j)$$

$$\mathbf{V} \in \mathbb{R}^{N_f \times N_r}, \qquad V_{kj} = \sigma_k^2[j] \quad (\text{NaN where } \mathbf{F} \text{ is NaN})$$

**Unnormalised inverse-variance weights**

$$P_{kj} = \frac{1}{V_{kj}} = \frac{1}{\sigma_k^2[j]} \qquad (0 \text{ where } V_{kj} \text{ is NaN or zero})$$

**Normalised weight matrix $\boldsymbol{\Omega} \in \mathbb{R}^{N_f \times N_r}$**

Each column $j$ sums to 1 over the feeds that observe edge $j$:

$$\Omega_{kj} = \frac{P_{kj}}{\displaystyle\sum_{k} P_{kj}} = \frac{1/\sigma_k^2[j]}{\displaystyle\sum_{k} 1/\sigma_k^2[j]}$$

**Aggregated flow vector as a matrix product**

Writing $\mathbf{1} \in \mathbb{R}^{N_f}$ for a column vector of ones, the inverse-variance weighted mean over all root edges simultaneously is:

$$\boxed{\mathbf{f}_\mathrm{agg} = \mathbf{1}^\top (\boldsymbol{\Omega} \odot \mathbf{F}) \;\in\; \mathbb{R}^{N_r}}$$

For each root edge $j$ this reduces to $f_\mathrm{agg}[j] = \sum_k \Omega_{kj} \cdot F_{kj}$. Edges covered by only one feed receive that feed's value unchanged ($\boldsymbol{\Omega}_{:,j}$ has a single non-zero entry of 1). Edges covered by no feed remain NaN and are passed to the imputation step.

In numpy: `f_agg = np.nansum(Omega * F, axis=0)` (equivalent to the matrix product when NaN entries in $\boldsymbol{\Omega}$ are replaced with 0 before summing).

**Aggregated uncertainty**

$$\sigma_\mathrm{agg}[j] = \frac{1}{\displaystyle\sqrt{\sum_k P_{kj}}} = \frac{1}{\displaystyle\sqrt{\sum_k 1/\sigma_k^2[j]}}$$

$\sigma_\mathrm{agg}[j] \leq \min_k \sigma_k[j]$: combining feeds never increases uncertainty.

**NaN handling (coverage)**

- A root edge $j$ is **covered** if at least one feed has a non-NaN observation there.
- Covered edges: $f_\mathrm{agg}[j]$ and $\sigma_\mathrm{agg}[j]$ are finite; column $j$ of $\boldsymbol{\Omega}$ sums to 1.
- Uncovered edges: $f_\mathrm{agg}[j] = \mathrm{NaN}$; passed to `ImputationEngine`.

The matrix $\boldsymbol{\Omega}$ is a full provenance record: row $k$ gives feed $k$'s fractional contribution to every root edge.

### Stage 3 — Constrained optimisation objective

Minimise the $\sigma$-weighted distance from the aggregated estimate subject to physical balance constraints:

$$\min_{\mathbf{f}} \; (\mathbf{f} - \mathbf{f}_\mathrm{agg})^\top \boldsymbol{\Sigma}_\mathrm{agg}^{-1} (\mathbf{f} - \mathbf{f}_\mathrm{agg}) \qquad \text{subject to} \quad \mathbf{G}\mathbf{f} = \mathbf{c}$$

where $\boldsymbol{\Sigma}_\mathrm{agg} = \mathrm{diag}(\boldsymbol{\sigma}_\mathrm{agg}^2)$, $\mathbf{G}$ encodes supply–use balances and parametric efficiency constraints, and $\mathbf{c}$ holds the corresponding totals.

### Optimised uncertainty via KKT sensitivity

The closed-form optimum for equality-constrained WLS:

$$\mathbf{f}_\mathrm{opt} = \mathbf{f}_\mathrm{agg} + \boldsymbol{\Sigma}_\mathrm{agg}\mathbf{G}^\top \!\left(\mathbf{G}\boldsymbol{\Sigma}_\mathrm{agg}\mathbf{G}^\top\right)^{-1} (\mathbf{c} - \mathbf{G}\mathbf{f}_\mathrm{agg})$$

Jacobian of $\mathbf{f}_\mathrm{opt}$ with respect to $\mathbf{c}$:

$$\mathbf{J} = \frac{\partial \mathbf{f}_\mathrm{opt}}{\partial \mathbf{c}} = \boldsymbol{\Sigma}_\mathrm{agg}\mathbf{G}^\top \!\left(\mathbf{G}\boldsymbol{\Sigma}_\mathrm{agg}\mathbf{G}^\top\right)^{-1}$$

Propagated uncertainty on the optimised solution:

$$\boldsymbol{\Sigma}_\mathrm{opt} = \mathbf{J}\,\boldsymbol{\Sigma}_c\,\mathbf{J}^\top, \qquad \boldsymbol{\sigma}_\mathrm{opt} = \sqrt{\mathrm{diag}(\boldsymbol{\Sigma}_\mathrm{opt})}$$

where $\boldsymbol{\Sigma}_c = \mathrm{diag}(\boldsymbol{\sigma}_\mathrm{agg}^2)$ is the uncertainty in the constraint right-hand sides inherited from the aggregation step.

---

## Workflow

```python
# ── Stage 1: one feed pipeline run per data source ─────────────────────────

registry = DataFeedRegistry(odir="data/proc/feeds")
registry.load_index()

for feed_id, excel_path, raw_df, F_src in feed_configs:
    wm, meta, result = run_feed_pipeline(
        excel_path=excel_path,
        raw_df=raw_df,
        acc_struct=acc,
        sector_map_df=...,
        region_map_df=...,
        time_map_df=...,
        valuation_map_df=...,
        var_map_df=...,
        F_src=F_src,
        registry=registry,
        feed_id=feed_id,
        aux_col="Supply_Use",
        allow_unmatched_edges=True,
    )
    # registry.json, metadata.json, <feed_id>.pkl, result.npz written automatically

# ── Stage 2: aggregate selected feeds ───────────────────────────────────────

collector = FeedCollector(registry=registry, acc_struct=acc)
collector.select(feed_ids=["icsg", "bgs", "usgs"])

# Stacks into F [N_f × N_r] and V [N_f × N_r], builds Ω, computes f_agg
agg_result = collector.aggregate()          # f_agg, σ_agg, Ω

engine = ImputationEngine(acc_struct=acc)
engine.register(ParametricImputer(relations=acc.parametric_relations))
engine.register(TimeSeriesImputer(method="linear"))
agg_result = engine.run(agg_result)         # fill NaN edges, log every fill

# ── Stage 3: optimise over T timesteps ──────────────────────────────────────

cb  = ConstraintBuilder(acc_struct=acc,
                        parametric_relations=acc.parametric_relations)
opt = LinearOptimiser(acc_struct=acc)
unc = OptUncertainty()

pipeline = OptimPipeline(
    acc_struct=acc,
    registry=registry,
    constraint_builder=cb,
    optimiser=opt,
    unc=unc,
)

run = pipeline.run(
    agg_result=agg_result,
    timesteps=list(range(2010, 2024)),
)

# run.results[t] → OptimResult with F_opt, σ_opt, residuals, solver_status
# run is persisted automatically to the registry
```

---

## File structure

```
pysuto/
├── __init__.py            public re-exports only
│
├── acc.py                 AccountingStructure  (scipy.sparse COO/CSR)
├── conc.py                ConcordanceTable · WeightMatrix  (scipy.sparse CSR)
├── feed.py                DataFeed (ABC)  ← weight_matrix typed as csr_matrix
├── feedpipe.py            step_A/B/C/D functions · run_feed_pipeline  ── Stage 1
├── unc.py                 PedigreeScores · UncertaintyMixin · PropagationResult
├── meta.py                DataFeedMetadata  (Excel sidecar + computed outputs)
├── registry.py            DataFeedRegistry · _RegistryEntry  (JSON index)
├── utils.py               _NpEncoder · io helpers
├── logging.py             structured run logging
│
├── collector.py           FeedCollector · AggregationResult  ── Stage 2
├── agg.py                 AggregationMath  (F matrix · Ω matrix · f_agg product)
├── impute.py              ImputationEngine · ParametricImputer · TimeSeriesImputer
│
├── constraints.py         ConstraintBuilder  ── Stage 3
├── optim.py               LinearOptimiser · OptimResult
├── optunc.py              OptUncertainty
└── optpipe.py             OptimPipeline · OptimRun

init_icsg_feed.py          Driver script for the ICSG copper feed (calls feedpipe)

tests/
├── conftest.py            shared fixtures: mock acc, W_dense, F_src, metadata
├── test_pedigree.py
├── test_concordance.py
├── test_weightmatrix.py
├── test_aggregation.py
├── test_imputation.py
├── test_constraints.py
├── test_optimiser.py
├── test_optunc.py
├── test_registry.py
└── test_workflow.py       integration test: full three-stage run
```

**Import direction is strictly one-way:**

```
acc  →  conc  →  feed  →  feedpipe
unc  ─────────────────────────────┘ (mixin, no upward imports)
meta → registry
collector → agg → impute → constraints → optim → optunc → optpipe
utils / logging ← imported by any layer, import nothing from pysuto
```

Nothing in the data layer (`acc`, `conc`, `feed`) imports from the registry or pipeline layers. `utils.py` and `logging.py` have no pysuto-internal imports. This makes every module independently testable.

---

## Testing strategy

Each layer has a dedicated unit test file that can be run in isolation using the shared fixtures in `conftest.py`.

| Test file | Scope | Key fixture |
|---|---|---|
| `test_pedigree.py` | CV formulas, score validation, invariant CV_root ≤ CV_src | none — pure numpy |
| `test_concordance.py` | Weight imputation, sum-to-one, src index, CSR shape | `mock_acc_struct` |
| `test_weightmatrix.py` | W construction, apply(), scipy.sparse round-trip pickle | `mock_acc_struct`, `W_dense` |
| `test_aggregation.py` | Ω construction, `f_agg = 1ᵀ(Ω⊙F)`, σ_agg, NaN coverage | `F_src`, `sigma_src` |
| `test_imputation.py` | Parametric fill, LOCF, imputation log | `agg_result_partial` |
| `test_constraints.py` | G assembly, supply=use rows, parametric rows | `mock_acc_struct` |
| `test_optimiser.py` | Feasibility, residuals near zero, warm-start delta | `G`, `c`, `F_agg` |
| `test_optunc.py` | σ_opt ≤ σ_agg invariant, Jacobian shape | `G`, `Sigma_diag` |
| `test_registry.py` | Round-trip save/load, summary_df, overwrite guard | temp directory |
| `test_workflow.py` | Full three-stage integration run, provenance trace | all fixtures |

---

## Installation

```bash
# clone
git clone https://github.com/your-org/pySUTO.git
cd pySUTO

# install in editable mode with dev dependencies
pip install -e ".[dev]"

# run tests
pytest tests/ -v
```

**Core dependencies:** `numpy`, `pandas`, `scipy`, `sparse`

**Optional:** `matplotlib` for diagnostic plots of feed uncertainty distributions

---

## References

Laner, D., Rechberger, H., & Astrup, T. (2016). A novel approach to characterize data uncertainty in material flow analysis and its application to plastics flows in Austria. *Journal of Industrial Ecology*, 20(5), 1050–1063.

Lenzen, M., Wood, R., & Wiedmann, T. (2009). Matrix balancing under conflicting information. *Economic Systems Research*, 22(1), 1–23.
