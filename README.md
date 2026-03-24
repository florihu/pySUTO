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

Runs once across a selected set of registered feeds. `FeedCollector` loads `F_root` and `σ_root` for each selected feed from the registry and aligns them onto the common edge index. `AggregationMath` computes the inverse-variance weighted mean and propagates uncertainty. `ImputationEngine` then fills missing edges using parametric relations or time-series methods, logging every fill.

**Inputs:** feed IDs, registry
**Outputs:** `AggregationResult` with `F_agg [E]`, `σ_agg [E]`, source weights, imputation log

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
| `acc.py` | `AccountingStructure` | Node/edge index, supply–use topology, parametric relations |
| `conc.py` | `ConcordanceTable`, `WeightMatrix` | src→root edge mapping, sparse W matrix |
| `feed.py` | `DataFeed` (ABC) | Base class, inherits `UncertaintyMixin`; owns pedigree, `F_src`, persistence |
| `feedpipe.py` | `FeedPipeline` | Orchestrates steps A→D for a single feed; registers result |
| `unc.py` | `PedigreeScores`, `UncertaintyMixin`, `PropagationResult` | Laner et al. pedigree → CV; Gaussian error propagation through W |
| `meta.py` | `DataFeedMetadata` | Parses Excel sidecar; stores identification, pedigree scores, provenance |
| `registry.py` | `DataFeedRegistry`, `_RegistryEntry` | Persistent JSON index + per-feed metadata and result arrays |
| `utils.py` | `_NpEncoder`, io helpers | JSON serialisation, numpy-safe dump/load |
| `logging.py` | — | Structured run logging |

### New modules

| File | Classes | Responsibility |
|---|---|---|
| `collector.py` | `FeedCollector`, `AggregationResult` | Load and align selected feeds; delegate to `agg.py`; return aggregated arrays |
| `agg.py` | `AggregationMath` | σ-weighted mean and uncertainty propagation across N feeds |
| `impute.py` | `ImputationEngine`, `ParametricImputer`, `TimeSeriesImputer` | Fill missing edges; log every imputation |
| `constraints.py` | `ConstraintBuilder` | Assemble G and c from supply–use balances and parametric relations |
| `optim.py` | `LinearOptimiser`, `OptimResult` | σ-weighted equality-constrained least squares; warm-start |
| `optunc.py` | `OptUncertainty` | KKT sensitivity-based uncertainty propagation for constrained optimum |
| `optpipe.py` | `OptimPipeline`, `OptimRun` | T-step loop with warm-start; collect and persist results |

---

## Mathematical foundations

### Pedigree → coefficient of variation

Following Laner et al. (2016). Each data source carries five quality indicator scores s ∈ {1,2,3,4}.

**Reliability** (non-zero even at best score):

```
CV_rel = a · exp(b · s_rel)          a = 0.75,  b = 1.105
```

**All other indicators** (zero at score 1):

```
CV_i = a_i · exp(b · (s_i − 1))     a depends on sensitivity level
```

Sensitivity parameter a: not sensitive = 0.375, medium = 0.75, highly sensitive = 1.5.

**Total CV** (sum in quadrature):

```
CV_tot = sqrt( CV_rel² + CV_comp² + CV_temp² + CV_geo² + CV_other² )
```

### Concordance error propagation

The weight matrix W ∈ ℝ^(N_src × N_root) maps source edges to root edges. For a dataset with uniform CV_tot across all source edges:

```
Σ_src = CV_tot² · diag(F_src²)

F_root  = Wᵀ F_src
Σ_root  = Wᵀ Σ_src W  =  CV_tot² · Wᵀ diag(F_src²) W

σ_root  = CV_tot · sqrt( (W ⊙ W)ᵀ F_src² )       elementwise
```

The diversification property guarantees CV_root ≤ CV_src for every root edge.

### Aggregation across N feeds

Inverse-variance weights for feed k at edge j:

```
w_kj = 1 / σ²_kj

F_agg_j    = Σ_k ( w_kj · F_kj ) / Σ_k w_kj
σ_agg_j    = 1 / sqrt( Σ_k w_kj )
```

### Constrained optimisation objective

Minimise the σ-weighted distance from the aggregated estimate subject to physical balance constraints:

```
minimise    (f − F_agg)ᵀ Σ_agg⁻¹ (f − F_agg)
subject to  G f = c
```

where Σ_agg = diag(σ_agg²), G encodes supply–use balances and parametric efficiency constraints, and c holds the corresponding totals.

### Optimised uncertainty via KKT sensitivity

The closed-form optimum for equality-constrained WLS:

```
F_opt = F_agg + Σ_agg Gᵀ (G Σ_agg Gᵀ)⁻¹ (c − G F_agg)
```

Jacobian of F_opt with respect to c:

```
J = ∂F_opt/∂c = Σ_agg Gᵀ (G Σ_agg Gᵀ)⁻¹
```

Propagated uncertainty on the optimised solution:

```
Σ_opt = J · Σ_c · Jᵀ
σ_opt = sqrt( diag(Σ_opt) )
```

where Σ_c = diag(σ_agg²) is the uncertainty in the constraint right-hand sides inherited from the aggregation step.

---

## Workflow

```
# ── Stage 1: one FeedPipeline per data source ──────────────────────────────

registry = DataFeedRegistry(odir="data/proc/feeds")
registry.load_index()

for feed_id, excel_path, raw_df, F_src in feed_configs:
    pipe = FeedPipeline(
        acc_struct=acc,
        registry=registry,
        feed_id=feed_id,
    )
    wm, meta, result = pipe.run(
        excel_path=excel_path,
        raw_df=raw_df,
        sector_map_df=...,
        region_map_df=...,
        time_map_df=...,
        valuation_map_df=...,
        var_map_df=...,
        F_src=F_src,
    )

# ── Stage 2: aggregate selected feeds ───────────────────────────────────────

collector = FeedCollector(registry=registry, acc_struct=acc)
collector.select(feed_ids=["icsg", "bgs", "usgs"])

engine = ImputationEngine(acc_struct=acc)
engine.register(ParametricImputer(relations=acc.parametric_relations))
engine.register(TimeSeriesImputer(method="linear"))

agg_result = collector.aggregate()          # σ-weighted mean + σ_agg
agg_result = engine.run(agg_result)         # fill missing edges

# ── Stage 3: optimise over T timesteps ──────────────────────────────────────

cb     = ConstraintBuilder(acc_struct=acc,
                           parametric_relations=acc.parametric_relations)
opt    = LinearOptimiser(acc_struct=acc)
unc    = OptUncertainty()

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
├── acc.py                 AccountingStructure
├── conc.py                ConcordanceTable · WeightMatrix
├── feed.py                DataFeed (ABC) + UncertaintyMixin
├── feedpipe.py            FeedPipeline  ── Stage 1 orchestrator
├── unc.py                 PedigreeScores · UncertaintyMixin · PropagationResult
├── meta.py                DataFeedMetadata
├── registry.py            DataFeedRegistry · _RegistryEntry
├── utils.py               _NpEncoder · io helpers
├── logging.py             structured run logging
│
├── collector.py           FeedCollector · AggregationResult  ── Stage 2
├── agg.py                 AggregationMath
├── impute.py              ImputationEngine · ParametricImputer · TimeSeriesImputer
│
├── constraints.py         ConstraintBuilder  ── Stage 3
├── optim.py               LinearOptimiser · OptimResult
├── optunc.py              OptUncertainty
└── optpipe.py             OptimPipeline · OptimRun

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
meta → collector → agg → impute → constraints → optim → optunc → optpipe
utils / logging ← imported by any layer, import nothing from pysuto
```

Nothing in the data layer (`acc`, `conc`, `feed`) imports from the registry or pipeline layers. `utils.py` and `logging.py` have no pysuto-internal imports. This makes every module independently testable.

---

## Testing strategy

Each layer has a dedicated unit test file that can be run in isolation using the shared fixtures in `conftest.py`.

| Test file | Scope | Key fixture |
|---|---|---|
| `test_pedigree.py` | CV formulas, score validation, invariant CV_root ≤ CV_src | none — pure numpy |
| `test_concordance.py` | Weight imputation, sum-to-one validation, src index | `mock_acc_struct` |
| `test_weightmatrix.py` | W construction, apply(), round-trip pickle | `mock_acc_struct`, `W_dense` |
| `test_aggregation.py` | σ-weighted mean, edge cases (single feed, zero σ) | `F_src`, `sigma_src` |
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

**Core dependencies:** `numpy`, `pandas`, `scipy`, `sparse`,

**Optional:** `matplotlib` for diagnostic plots of uncertainty distributions

---

## References

Laner, D., Rechberger, H., & Astrup, T. (2016). A novel approach to characterize data uncertainty in material flow analysis and its application to plastics flows in Austria. *Journal of Industrial Ecology*, 20(5), 1050–1063.

Lenzen, M., Wood, R., & Wiedmann, T. (2009). Matrix balancing under conflicting information. *Economic Systems Research*, 22(1), 1–23.
