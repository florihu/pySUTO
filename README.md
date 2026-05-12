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

### Stage 2 — Collection, aggregation, and imputation

Runs once across a selected set of registered feeds. Three sub-steps:

#### 2a — Collection (`FeedCollector`)

`FeedCollector` selects feeds from the registry by feed ID (or by specific feed instance) and aligns each feed's `F_root` and `σ_root` onto the common edge index. Feeds that do not cover a given root edge contribute NaN at that position. The result is a pair of stacked matrices:

- **F** ∈ ℝ^(N_f × N_r) — raw per-feed observations, NaN where not covered
- **Σ** ∈ ℝ^(N_f × N_r) — per-feed standard deviations, NaN where not covered

Selection is by feed ID string (latest registered version is used by default). A specific version or instance can be selected by passing a `(feed_id, version)` tuple.

#### 2b — Aggregation (`AggregationMath`)

`AggregationMath` builds the normalised inverse-variance weight matrix **Ω** and computes:

```
f_agg[j] = Σ_k  Ω[k,j] · F[k,j]      (inverse-variance weighted mean)
σ_agg[j] = 1 / sqrt(Σ_k  1/σ[k,j]²)  (combined uncertainty, always ≤ min σ_k)
```

Both operations are vectorised as a single matrix product over all root edges simultaneously (see [Mathematical foundations](#mathematical-foundations)). **Ω** itself is a complete provenance record: column j gives the fractional contribution of every feed to root edge j.

Edges covered by no feed remain NaN in `f_agg` and are passed to the imputation step.

#### 2c — Imputation pipeline (`ImputationEngine`)

`ImputationEngine` manages an ordered pipeline of imputers. Each imputer receives the current state `(f_agg, σ_agg, missing_mask)`, attempts to fill remaining NaN entries, propagates uncertainty through its fill formula, and appends to the shared `ImputationLog`. Multiple imputers chain: the second imputer only sees edges still NaN after the first, and so on.

When an imputer is registered via `register()`, `ImputationEngine` calls its `prepare(acc_struct)` method (if present), moving expensive one-time setup — such as building the shared edge-lookup index or pre-computing target/source masks — to registration time rather than the first `impute()` call. Imputers that lack `prepare()` are unaffected.

#### Parameter types and the edge-pair schema

All parametric imputation rests on a bidirectional linear constraint between two edges, called **edge A** and **edge B**:

```
f_b = coeff × f_a    (when A is observed and B is missing)
f_a = f_b / coeff    (when B is observed and A is missing)
```

The imputer resolves which direction applies at runtime from the missing mask. Both edges are identified by the 8-dimensional accounting-structure selector — all values are root-system coordinates (names from the Sector_Entity, Region, Valuation, and Time sheets):

```
(sector_from, entity_from, region_from,
 sector_to,   entity_to,   region_to,
 time, valuation)
```

A-selector fields accept a specific value or `*` (wildcard, matches any admissible value). B-selector fields additionally accept `same` (binds to the value resolved for the corresponding A field in each matched edge pair).

Two named types standardise the most common patterns (Pauliuk et al. 2019):

**Concentration** (data category 3 — intensive object property): links the same `(from_id, to_id, time)` pair across two valuations. All B fields are `same` except `b_valuation`. Because a commodity sector (e.g. `Concentrate/Flow`) appears in both supply edges (as from-node) and use edges (as to-node), two rows are needed — one with the sector in `a_sector_from`, one in `a_sector_to` — sharing the same value and uncertainty.

**Efficiency** (data category 4 — intensive process property / yield factor): links a USE edge (input into a process) to a SUPPLY edge (output from the same process). The process node appears as `(a_sector_to, a_entity_to)` in the USE edge and as `(b_sector_from, b_entity_from)` in the SUPPLY edge. Region and time are `same`. If `a_region_from = *`, the same efficiency applies globally.

**Uncertainty** is specified in one of three modes:

| Mode | Fields | σ |
|---|---|---|
| `sd` | `value_sd` | used directly |
| `min_max` | `value_min`, `value_max` | (max − min) / 4 |
| `pedigree` | five Laner et al. scores 1–4 (`reliability`, `completeness`, `temporal_correlation`, `geographical_correlation`, `other_correlation`) | `PedigreeScores(...).total_cv() × value_nominal` |

The `epsilon` field adds relative model uncertainty in quadrature: σ²_j += (ε · f_j)². Physical bounds `bound_lower` / `bound_upper` clip filled values.

The parameter CSV template with full column reference and example rows for all types is in `data/input/params/params_template.csv`.

**Built-in imputers (registered in order):**

| Imputer | Method | Uncertainty |
|---|---|---|
| `ParametricImputer` | Bidirectional edge-pair constraints; CSV loading or lambda relations | Lp combine; bare coeff in forward (A→B) and lambda; regularized `s(1/coeff)` only in backward (B→A): `σ_j = ( Σ \|coeff_i·σ_i\|^p + \|σ_param\|^p + \|ε·f_j\|^p )^(1/p)` |
| `FormulaImputer` | Directed multi-source balance equations; region/time-correlated source selection; physical clip | Lp: `σ_j = ( Σ_i |coeff_i·σ_i|^p + |ε·f_j|^p )^(1/p)`; pre-clip σ retained on clipped fills |
| `TimeImputer` | Window-based temporal interpolation / extrapolation | Lp: `σ_j = ( Σ_i |w_i·σ_i|^p )^(1/p) · γ(gap)` |
| `FlatPriorImputer` | Final fallback — `f=0`, `σ = CV[mask] · scale[group]` (CV by accounting mask, scale = group-mean of \|f_agg\|) | `σ = CV·scale` (no Lp combine; one source) |

The exponent `p ∈ [1.1, 1.5]` (default `1.3`) is a single global power-mean parameter — see [Power-mean uncertainty propagation](#power-mean-uncertainty-propagation). Setting `p=2` recovers Gaussian quadrature.

`s(α) = |α| / (|α|^q + α_min^q)^(1/q)` is the regularized sensitivity used in the **backward branch only** of `ParametricImputer` (CSV B→A direction, where the multiplier is `1/coeff`). The forward branch (A→B) and lambda relations use bare α — see [Regularized sensitivity (ParametricImputer)](#regularized-sensitivity-parametricimputer).

Additional imputers can be created by subclassing `BaseImputer` and implementing a single `impute()` method.

##### ParametricImputer

Resolves parameter definitions into bidirectional edge-pair constraints and fills missing edges. Two instantiation modes are supported:

**Load from CSV** (recommended for concentration and efficiency parameters):

```python
imputer = ParametricImputer.from_csv(
    "data/input/params/copper_params.csv",
    acc_struct=acc,
)
```

The loader resolves each row's A- and B-selector fields against `acc_struct.edge_index`, expands wildcards and `same` bindings into concrete `(edge_a_id, edge_b_id, coeff, σ)` quadruples, and converts uncertainty to σ:
- `min_max`: σ = (max − min) / 4
- `pedigree`: σ = `PedigreeScores(...).total_cv() × value_nominal`

**Lambda relations** (for custom or cross-type relations not covered by the CSV schema):

```python
ParametricRelation(
    description="Scrap return ≈ 0.12 × primary production (same region, same time)",
    target=lambda ei: (ei.sector_from == "SR") & (ei.sector_to == "FU"),
    sources=lambda ei: (ei.sector_from == "Cathode") & (ei.sector_to == "FU"),
    coefficients=[0.12],
    epsilon=0.25,
)
```

Each `ParametricRelation` specifies:
- **target** — a predicate over `edge_index` identifying which edges to fill
- **sources** — list of root edge IDs (or predicates) whose values drive the fill
- **coefficients** — the linear weights α_i (sum need not equal 1)
- **epsilon** — optional relative residual uncertainty (fraction of `f_j`), added in quadrature

Uncertainty propagation:

`σ_j = ( Σ_i |α_i·σ_i|^p + |σ_param|^p + |ε·f_j|^p )^(1/p)`

with the **forward CSV branch (A→B)** and **lambda relations** using bare α directly. The **backward CSV branch (B→A)** swaps `1/coeff` for the regularized sensitivity `s(1/coeff) = |1/coeff| / (|1/coeff|^q + α_min^q)^(1/q)` (defaults `q=3`, `α_min=0.05`) — this caps σ amplification when `1/coeff` is large (e.g. ore-grade inverse ≈ 100s) so that `s(1/coeff)·σ_src ≤ σ_src` rather than growing unbounded. The fill itself always uses bare α: `f_j = α · f_i + …`. Regularization applies to σ propagation in the backward direction only — physical balance equations are not perturbed. See [Regularized sensitivity (ParametricImputer)](#regularized-sensitivity-parametricimputer).

All fills are logged to `ImputationLog` with the `param_id` from the CSV row (None for lambda relations).

##### FormulaImputer

Fills missing edges using **directed multi-source balance equations**. Where `ParametricImputer` handles pairwise bidirectional constraints, `FormulaImputer` handles cases where a target edge is computed from several source edges with region-correlated selection, wildcard summation, and a physical clip.

Three building blocks:

**`ParamRef`** — references a value from a loaded params CSV:

```python
ParamRef(param_id="conc_ore_supply", transform="inverse")
# resolves to 1 / value_nominal of the named parameter at runtime
```

`transform="identity"` (default) uses the value directly; `transform="inverse"` takes its reciprocal. Useful when a supply edge is in CC valuation and must be converted to Mass via a concentration parameter before the balance is applied.

**`SourceTerm`** — one additive term in the formula:

```python
SourceTerm(
    selector=lambda ei: ...,          # predicate over edge_index → bool mask
    coefficient=1.0,                   # float or ParamRef; negative = subtraction
    region_binding="from_matches_target_from",  # see below
    time_binding=True,                 # if True: source.time must equal target.time
    aggregate="sum",                   # "single" (expect one match) or "sum" (sum all)
)
```

`region_binding` constrains which region the source edge must be in relative to the matched target edge:

| Value | Constraint |
|---|---|
| `"none"` | no region constraint |
| `"from_matches_target_from"` | `source.from_region == target.from_region` |
| `"from_matches_target_to"` | `source.from_region == target.to_region` |
| `"to_matches_target_from"` | `source.to_region == target.from_region` |
| `"to_matches_target_to"` | `source.to_region == target.to_region` |

**`prepare(acc_struct)`** — pre-builds the shared edge-lookup index and pre-computes target and source masks for all relations in this imputer. Called automatically by `ImputationEngine.register()`; can also be called explicitly before standalone use.

**`FormulaRelation`** — directed multi-source balance equation:

```python
FormulaRelation(
    description="...",
    target=lambda ei: ...,    # predicate identifying the target edges to fill
    terms=[...],              # list[SourceTerm]; f_target = Σ coeff_i × f_source_i
    clip_lower=0.0,           # physical lower bound (None = no clip)
    clip_upper=None,          # physical upper bound
    epsilon=0.0,              # relative model uncertainty added in quadrature
)
```

**Domestic use example** — for every region and every traded intermediate flow, domestic use = total supply minus all exports, clipped at zero:

```python
domestic_matte = FormulaRelation(
    description="Domestic Matte use: Smelting output minus exports, clipped at zero",
    target=lambda ei: (
        (ei.sector_from == "Matte") & (ei.entity_from == "Flow") &
        (ei.sector_to   == "CFR")   & (ei.entity_to   == "Process") &
        (ei.valuation   == "Mass")
    ),
    terms=[
        # Total Matte produced by Smelting in the same region
        SourceTerm(
            selector=lambda ei: (
                (ei.sector_from == "Smelting") & (ei.entity_from == "Process") &
                (ei.sector_to   == "Matte")    & (ei.entity_to   == "Flow") &
                (ei.valuation   == "Mass")
            ),
            coefficient=1.0,
            region_binding="to_matches_target_from",  # supply's to_region == target's from_region
            aggregate="single",
        ),
        # All Matte exports from the same region (summed over all destinations)
        SourceTerm(
            selector=lambda ei: (
                (ei.sector_from == "Matte") & (ei.entity_from == "Flow") &
                (ei.entity_to   == "Flow")  &   # Flow→Flow = cross-region trade edge
                (ei.valuation   == "Mass")
            ),
            coefficient=-1.0,
            region_binding="from_matches_target_from",  # export origin == target's from_region
            aggregate="sum",
        ),
    ],
    clip_lower=0.0,
    epsilon=0.05,
)

engine.register(FormulaImputer(relations=[domestic_matte]))
```

The same pattern generalises to any tradable intermediate flow (Concentrate, CSXEW, etc.) by swapping sector names.

**Uncertainty propagation:** for a linear formula with terms `(coeff_i, f_i, σ_i)`:

```
σ_target = ( Σ_i |coeff_i · σ_i|^p + |ε · f_target|^p )^(1/p)
```

If the pre-clip value is negative and the result is clipped to `clip_lower`, the fill is logged with `method="formula_clipped"` and `sigma_after` carries the pre-clip σ as a conservative uncertainty estimate.

For `aggregate="sum"` terms, zero matching source edges is treated as a contribution of 0 (not a failure). This handles the common case where a country has no exports for a given flow in a given year.

##### TimeImputer

Fills NaN edges by temporal interpolation or extrapolation using a configurable sliding window. Parameters:

- **window** — number of time periods to look back and forward (symmetric by default; `window_before` / `window_after` can differ)
- **method** — `"linear"` (weighted linear interpolation, weights ∝ 1/gap), `"locf"` (last-observation-carried-forward), `"nocb"` (next-observation-carried-backward), `"mean"` (unweighted average within window)
- **min_obs** — minimum number of non-NaN observations required within the window to attempt a fill (default 1)
- **gap_inflation** — σ multiplier per time-step gap; accounts for the fact that extrapolation uncertainty grows with the size of the gap (default 1.0 = no inflation; `sqrt(gap)` is a common choice)

For linear interpolation between two observations at gap g: the interpolated value at fractional position t is `f_t = (1-t)·f_a + t·f_b`, and the propagated uncertainty is `σ_t = ( |(1-t)·σ_a|^p + |t·σ_b|^p )^(1/p) · γ(gap)` where `γ(gap) = gap_inflation^gap`.

##### FlatPriorImputer

Final-fallback imputer. Register **last** in the pipeline; every residual NaN edge is filled with a flat prior:

```
f_j  = 0
σ_j  = CV[mask(j)] · scale(j)
```

- `mask(j)` is the **first matching** mask name (priority by row order in the CV CSV / dict). Supported masks: `supply`, `use`, `tradable`, `sb` — drawn directly from `AccountingStructure.{supply,use,tradable,sb}_edge_mask`. Edges that match no listed mask fall through to `default_cv` (a required constructor kwarg).
- `scale(j) = mean(|f_agg[i]|)` over edges `i` sharing the same `(sector_from, entity_from, region_from, valuation)` group as `j` and finite `f_agg[i]` at the moment FlatPrior runs (i.e. after every earlier imputer has filled what it can).
- An empty group (no finite `f_agg` shares the key) yields `scale = 0` → `σ_j = 0`. This is the intended behavior when the region/sector has no flow data at all.

The CV table lives in [data/input/params/flat_prior_cv.csv](data/input/params/flat_prior_cv.csv) (columns `mask_name,cv,description`; row order = priority). The seed file ships with two rows: `supply,0.001` (well-covered by feeds) and `use,0.3` (sparsely covered).

```python
engine.register(FlatPriorImputer.from_csv(
    "data/input/params/flat_prior_cv.csv",
    acc_struct=acc,
    default_cv=0.3,
    p=1.3,
))
```

Or programmatically: `FlatPriorImputer(cv_by_mask={"supply": 0.001, "use": 0.3}, default_cv=0.3, acc_struct=acc)`.

##### ImputationLog

Every fill is recorded in the `ImputationLog`, a per-edge provenance table with columns:

| Column | Type | Description |
|---|---|---|
| `edge_id` | int | Root edge index |
| `step` | int | Pipeline step index (0 = first registered imputer) |
| `method` | str | `"parametric"`, `"formula"`, `"formula_clipped"` (clipped to bound), `"time_linear"`, `"locf"`, etc. |
| `source_edges` | list[int] | Edge IDs used as inputs |
| `coefficients` | list[float] | Weights applied to source values |
| `f_before` | float | NaN (edge was missing before fill) |
| `f_after` | float | Filled value |
| `sigma_before` | float | NaN (edge was missing before fill) |
| `sigma_after` | float | Propagated σ |
| `description` | str | Human-readable description from the relation or method |
| `param_id` | str \| None | Key from the parameter CSV row; None for lambda-based relations |

The log is attached to `AggregationResult` and persisted alongside it. Combined with **Ω** from the aggregation step, this gives a complete per-data-point audit trail: for every root edge, you know exactly which feeds contributed (and with what weight) or which imputation rule filled it and from which sources.

**Inputs:** feed IDs, registry  
**Outputs:** `AggregationResult` with `f_agg [N_r]`, `σ_agg [N_r]`, weight matrix **Ω**, `ImputationLog`

#### Diagnostics (`AggDiagnostic`, `ImpDiagnostic`)

`AggDiagnostic` runs **immediately after `collector.aggregate()`** (and BEFORE imputation, snapshotting `f_agg` / `σ_agg` since the imputation engine mutates them in place). It writes one 2×2 figure that exposes whether the dominant CV-bloat is upstream or inside the imputers:

- **[0,0] log10 |f_agg|** — magnitude distribution of covered edges
- **[0,1] log10 σ_agg** — uncertainty distribution
- **[1,0] log10 CV** (`σ_agg/|f_agg|`) with reference lines at 0.1, 1, 10, 100
- **[1,1] per-feed scatter** of `|f_root|` vs `σ_root` (one colour per feed) with diagonals at CV=1 and CV=10 — directly attributes any high-CV outliers to a Stage-1 feed

`ImpDiagnostic` runs after `engine.run()` and produces the per-step trajectory figures (F_root and σ histograms, coverage and fill counts, σ- and CV-by-method boxplots, and a TimeImputer timeseries panel when `acc_struct` is provided).

```python
agg_diag = AggDiagnostic(agg_result, label="copper")
agg_diag.summary()                                            # CV percentiles + top σ outliers

agg_result = engine.run(agg_result)                            # mutates f_agg / σ_agg in place

agg_diag.plot(odir="data/proc/imp/copper/diag", show=False)    # writes copper_agg_distributions.png
ImpDiagnostic(agg_result, acc_struct=acc, label="copper").plot(odir="data/proc/imp/copper/diag")
```

### Stage 3 — Constraint building and linear optimisation

Stage 3 has two sequential sub-steps.

#### 3a — Constraint building (`ConstraintEngine`)

`ConstraintEngine` follows the same plugin pattern as `ImputationEngine`: register `BaseConstraintBuilder` subclasses in order, then call `run()`. The engine calls `build_static()` on every registered builder once to assemble the time-invariant constraint matrix **G**, then calls `build_dynamic()` per timestep to produce `c(t)` and `σ_c(t)`. Results are persisted in a `ConstraintRunRegistry` at `data/proc/const/`.

**Five built-in builders** (registered in any order):

| Builder | Constraint | Cone | σ_c |
|---|---|---|---|
| `MassBalanceBuilder` | Hard: Σ incoming_edges − Σ outgoing_edges = 0 per (Flow\|Process) node × time | Zero | 0 |
| `SumConstraintBuilder` | Soft: Σ coeff_i · a_i = c_k + ε_k (from `acc_struct` topology or CSV) | Free | `fraction_of_flow` or `sd` |
| `RatioConstraintBuilder` | Soft: a_i − r · a_j = ε_k (from params CSV) | Free | from params CSV unc_type |
| `RatioBoundBuilder` | Hard: a_i − r_u · a_j ≤ 0 and −a_i + r_l · a_j ≤ 0 (from params CSV) | Non-neg | 0 |
| `FlowBoundsBuilder` | Hard: l_i ≤ a_i ≤ u_i (from `acc_struct.bounds`) | Non-neg | 0 |

`MassBalanceBuilder` derives balance rows automatically from four boolean edge masks that `AccountingStructure` computes at construction time (`supply_edge_mask`, `use_edge_mask`, `tradable_edge_mask`, `sb_edge_mask`). SB nodes are open-boundary endpoints and are excluded from balance rows.

**Sum constraint CSV** (long format — one row per term; `constraint_id` groups terms):

| Column | Description |
|---|---|
| `constraint_id` | Groups all terms of one constraint |
| `description` | Human-readable label (first row of each group) |
| `sector_from`, `entity_from`, `region_from` | Edge from-node selector (`*` = wildcard) |
| `sector_to`, `entity_to`, `region_to` | Edge to-node selector |
| `time`, `valuation` | Optional selectors (`*` = all time periods / all valuations) |
| `coefficient` | Float; sign encodes direction (+1 supply, −1 waste output) |
| `c_k` | RHS target value (default 0 — balance equation) |
| `sigma_ck_mode` | `sd` (absolute) / `fraction_of_flow` (fraction of node flow magnitude) |
| `sigma_ck_value` | Absolute σ_c (mode = `sd`) |
| `sigma_ck_fraction` | Fraction of node flow magnitude from `f_agg` (mode = `fraction_of_flow`) |

Ratio and bound constraints reuse the existing params CSV schema from `ParametricImputer` (`value_nominal`, `unc_type`, `bound_lower`, `bound_upper`).

`ConstraintRun` output structure:
```
data/proc/const/{run_id}/
├── registry.json        ← acc_id, agg_id, timestamp
├── G.npz                ← sparse (M × E), time-invariant
├── row_meta.parquet     ← constraint_id, type, is_hard, cone_type per row
├── a0.npy               ← f_agg at t=0 (prior for optimiser)
├── sigma_a.npy          ← sigma_agg
├── u.npy / l.npy        ← flow bounds from acc_struct
└── timesteps/t_{year}.npz  ← c(t), sigma_c(t)
```

**Inputs:** `AggregationResult`, accounting structure, optional CSV files  
**Outputs:** `ConstraintRun` with **G** [M×E], **c(t)** [M], **σ_c(t)** [M], **u** [E], **l** [E], `row_meta`

#### 3b — Linear optimisation (`LinearOptimiser`)

`LinearOptimiser` solves the σ-weighted least-squares problem with equality and inequality constraints. Each timestep is warm-started from the previous solution (`a_0(t) = a*(t−1)` for `t > 0`; the constraint engine stores `a_0` only for `t = 0`). `OptUncertainty` propagates input uncertainty into the optimised solution via the KKT sensitivity Jacobian. All results are collected into an `OptimRun` and persisted.

**Inputs:** `ConstraintRun`, `AggregationResult`  
**Outputs per timestep:** `OptimResult` with `F_opt [E]`, `σ_opt [E]`, residuals, solver status  
**Outputs per run:** `OptimRun` with full provenance trace

---

## Module reference

![File structure overview](figs/file_structure.svg)

### Stage 1 modules

| File | Classes | Responsibility |
|---|---|---|
| `acc.py` | `AccountingStructure` | Node/edge index, supply–use topology, per-edge bounds; structural edge masks (`supply_edge_mask`, `use_edge_mask`, `sb_edge_mask`, `tradable_edge_mask`) used by `ConstraintEngine`; uses `scipy.sparse` COO/CSR |
| `conc.py` | `ConcordanceTable`, `WeightMatrix` | src→root edge mapping, sparse W matrix (scipy.sparse CSR) |
| `feed.py` | `DataFeed` (ABC) | Base class; owns `weight_matrix` (CSR), pedigree, `F_src`, persistence |
| `feedpipe.py` | step functions, `run_feed_pipeline` | Orchestrates steps A→D for a single feed; delegates to `DataFeedRegistry` |
| `unc.py` | `PedigreeScores`, `UncertaintyMixin`, `PropagationResult` | Laner et al. pedigree → CV; Gaussian error propagation through W (scipy.sparse) |
| `meta.py` | `DataFeedMetadata` | Parses Excel sidecar; stores identification, pedigree scores, provenance, computed outputs |
| `registry.py` | `DataFeedRegistry`, `_RegistryEntry` | Persistent JSON index + per-feed metadata.json, pickle, result.npz |
| `utils.py` | `_NpEncoder`, io helpers | JSON serialisation, numpy-safe dump/load |
| `logging.py` | — | Structured run logging |

### Stage 2 modules

| File | Classes | Responsibility |
|---|---|---|
| `collector.py` | `FeedCollector`, `AggregationResult` | Select and load feeds; align F and Σ matrices onto common edge index |
| `agg.py` | `AggregationMath` | Build **Ω**; compute `f_agg = 1ᵀ(Ω ⊙ F)` and `σ_agg`; coverage mask |
| `impute.py` | `ImputationEngine`, `BaseImputer`, `ParametricImputer`, `FormulaImputer`, `TimeImputer`, `ParametricRelation`, `FormulaRelation`, `SourceTerm`, `ParamRef`, `ImputationLog` | Ordered imputation pipeline; per-edge provenance logging; uncertainty propagation per fill |
| `diag.py` | `AggDiagnostic`, `ImpDiagnostic` (Stage 2); `SolverDiag` (Stage 3c — listed here too because `diag.py` is a single cross-stage module) | Stage-2a aggregation distributions (`AggDiagnostic`), Stage-2c per-step imputation trajectories (`ImpDiagnostic`), and Stage-3c 13-figure solver portfolio (`SolverDiag`) |

### Stage 3 modules

| File | Classes | Responsibility |
|---|---|---|
| `constraints.py` | `ConstraintEngine`, `BaseConstraintBuilder`, `MassBalanceBuilder`, `SumConstraintBuilder`, `RatioConstraintBuilder`, `RatioBoundBuilder`, `FlowBoundsBuilder`, `ConstraintBlock`, `ConstraintRun`, `ConstraintRunRegistry` | Plugin constraint pipeline; builds **G** (once), **c(t)**, **σ_c(t)**; persists to JSON registry |
| `optim.py` | `LinearOptimiser`, `OptimResult` | σ-weighted equality-constrained least squares; warm-start |
| `optunc.py` | `OptUncertainty` | KKT sensitivity-based uncertainty propagation for constrained optimum |
| `optpipe.py` | `OptimPipeline`, `OptimRun` | T-step loop with warm-start; collect and persist results |
| `diag.py` | `SolverDiag` (cross-stage; the same module also hosts `AggDiagnostic` / `ImpDiagnostic` for Stage 2) | Stage-3c 13-figure solver diagnostic portfolio (`SD-R-01..07` run-scope + `SD-T-01..06` per-timestep cross-sections); `from_opt_id` loader and `compare()` overlay across runs |

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

### Power-mean uncertainty propagation

All Stage 1 / Stage 2a / Stage 2b σ propagation steps share a single global power-mean (Lp) combiner with exponent $p \in [1.1, 1.5]$, default $p = 1.3$. For a list of contributions $\{x_i\}$:

$$\mathrm{Lp}(\{x_i\};\, p) \;=\; \left(\sum_i |x_i|^p\right)^{1/p}.$$

`p = 2` recovers the legacy Gaussian-quadrature formulas exactly. Smaller `p` weights large terms relatively less, which suppresses σ inflation in fan-in regimes where the per-source independence assumption (one dataset-level CV applied across many edges) is structurally violated. The exponent is a constructor kwarg on every consumer (`UncertaintyMixin.apply_with_uncertainty`, `AggregationMath.run`, every `BaseImputer`); init scripts pass the same value to all of them.

### Regularized sensitivity (ParametricImputer)

The **backward CSV branch (B→A)** of `ParametricImputer` — and only that branch — passes the inverse multiplier $1/\alpha$ through a bounded sensitivity function before the Lp combine:

$$s(\alpha;\, q,\,\alpha_{\min}) \;=\; \frac{|\alpha|}{\bigl(|\alpha|^q + \alpha_{\min}^q\bigr)^{1/q}}.$$

Defaults: $q = 3$, $\alpha_{\min} = 0.05$. Behaviour at the backward multiplier $\beta = 1/\alpha$:
- $|\beta| \gg \alpha_{\min}$ (small original $\alpha$, e.g. ore-grade inverse): $s(\beta) \to 1$. The term $s(\beta)\cdot\sigma_b$ cannot grow beyond $\sigma_b$, capping the amplification that would otherwise occur from the bare $1/\alpha$ multiplier.
- $|\beta| \ll \alpha_{\min}$: $s(\beta) \to |\beta|/\alpha_{\min}$ — a relative floor of $1/\alpha_{\min}$ on the effective sensitivity.
- $|\beta| = \alpha_{\min}$: $s = 2^{-1/q}$. Larger $q$ gives a sharper knee; $q \to \infty$ recovers a hard clip at 1, $q \to 0$ recovers bare $|\beta|$.

The **forward CSV branch (A→B)** and **lambda relations** use bare $\alpha$ in σ propagation — those directions correspond to author-chosen, naturally oriented coefficients (typically $|\alpha| \leq 1$) and don't suffer from inverse-direction amplification. The fill formula $f_j = \sum_l \alpha_l f_{i_l}$ always uses bare α — physical balance equations are not perturbed. Other imputers (`FormulaImputer`, `TimeImputer`) keep bare-coefficient Lp combines.

### Stage 1 — Concordance error propagation

The weight matrix $\mathbf{W} \in \mathbb{R}^{N_\mathrm{src} \times N_\mathrm{root}}$ maps source edges to root edges, assembled as a `scipy.sparse.csr_matrix`. The point estimate is

$$\mathbf{f}_\mathrm{root} = \mathbf{W}^\top \mathbf{f}_\mathrm{src}.$$

Uncertainty is propagated in **CV-invariant** form. For each root edge $j$, define the normalized fan-in shares from the per-source absolute contributions

$$A_{ij} = \bigl|W_{ij}\, f_{\mathrm{src},i}\bigr|, \qquad \omega_{ij} = \frac{A_{ij}}{\sum_k A_{kj}} \quad\Bigl(\sum_i \omega_{ij} = 1\Bigr).$$

Then

$$\mathrm{CV}_\mathrm{root}[j] \;=\; \mathrm{CV}_\mathrm{src} \cdot \left(\sum_i \omega_{ij}^p\right)^{1/p}, \qquad \boldsymbol{\sigma}_\mathrm{root}[j] \;=\; \mathrm{CV}_\mathrm{root}[j] \cdot |\mathbf{f}_\mathrm{root}[j]|.$$

For $p \geq 1$ the power-mean of normalized weights is at most 1, so the **diversification invariant** $\mathrm{CV}_\mathrm{root}[j] \leq \mathrm{CV}_\mathrm{src}$ holds for every root edge. For $\mathbf{W} \geq 0$ and $\mathbf{f}_\mathrm{src} \geq 0$ this matches the legacy formula $\mathrm{CV}_\mathrm{tot} \cdot (\sum_i |W_{ij} f_i|^p)^{1/p}$ exactly, but when sources partially **sign-cancel** in $\mathbf{f}_\mathrm{root}$ the new form keeps σ tied to the signed sum (so a near-zero $\mathbf{f}_\mathrm{root}$ produces a near-zero σ instead of an inflated absolute σ from the absolute Lp accumulation). Sparse implementation uses one Hadamard pass over the CSR `.data` array (no densification).

### Stage 2a — Aggregation across N_f feeds

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

**Aggregated uncertainty (Lp / power-mean)**

The point estimate uses inverse-variance (BLUE) weights, but σ is combined via the Lp norm of the inverse-variance-weighted per-feed contributions:

$$\sigma_\mathrm{agg}[j] \;=\; \left( \sum_k |\Omega_{kj}\, \sigma_k[j]|^p \right)^{1/p}.$$

For single-cover edges ($\Omega_{kj} = 1$ for one $k$) this equals the covering feed's σ unchanged. At $p = 2$ it reduces to $1/\sqrt{\sum_k 1/\sigma_k^2[j]}$ — the legacy combined-uncertainty formula. For $p \in [1, 2]$, $\sigma_\mathrm{agg}[j] \leq \max_k \sigma_k[j]$ always holds; the strict $\sigma_\mathrm{agg} \leq \min_k \sigma_k$ invariant is exact at $p = 2$ and held tightly for $p \approx 1.3$ in practice.

**NaN handling (coverage)**

- A root edge $j$ is **covered** if at least one feed has a non-NaN observation there.
- Covered edges: $f_\mathrm{agg}[j]$ and $\sigma_\mathrm{agg}[j]$ are finite; column $j$ of $\boldsymbol{\Omega}$ sums to 1.
- Uncovered edges: $f_\mathrm{agg}[j] = \mathrm{NaN}$; passed to `ImputationEngine`.

The matrix $\boldsymbol{\Omega}$ is a full provenance record: row $k$ gives feed $k$'s fractional contribution to every root edge.

### Stage 2b — Imputation uncertainty propagation

Each imputation step propagates uncertainty through the fill via the same Lp combiner used in Stages 1 and 2a. The key invariant: every filled edge has a well-defined $\sigma_\mathrm{agg}[j]$ after each step, enabling downstream stages to treat imputed and observed edges uniformly.

#### Parametric imputation

A relation fills target edge $j$ as a linear combination of source edges $\{i_1, \dots, i_m\}$:

$$f_j = \sum_{l=1}^{m} \alpha_l \cdot f_{i_l}$$

Lp propagation across sources, the per-relation parameter σ, and the structural-uncertainty term:

$$\sigma_j \;=\; \left( \sum_{l=1}^{m} |\alpha_l \cdot \sigma_{i_l}|^p \;+\; |\sigma_\mathrm{param}|^p \;+\; |\varepsilon \cdot f_j|^p \right)^{1/p}$$

with bare coefficients $\alpha_l$ in the **forward CSV branch (A→B)** and **lambda relations**. The **backward CSV branch (B→A)** is the only place where the multiplier $1/\alpha$ is replaced by the regularized sensitivity $s(1/\alpha) = |1/\alpha| / (|1/\alpha|^q + \alpha_{\min}^q)^{1/q}$ (defaults $q=3$, $\alpha_{\min}=0.05$) — this caps σ amplification when $1/\alpha \gg 1$ (the ore-grade inverse case). The fill formula always uses bare $\alpha_l$. See [Regularized sensitivity (ParametricImputer)](#regularized-sensitivity-parametricimputer).

The $\varepsilon$ term ($\varepsilon$ = `epsilon`, a user-supplied fraction) encodes the structural uncertainty in the relation itself — the fact that the parametric formula is an approximation. The $\sigma_\mathrm{param}$ term is the parameter-row σ from the CSV (`sd`, `min_max`, or `pedigree` mode) and is omitted (treated as 0) for lambda-based relations. Both inflate $\sigma_j$ even when source uncertainties are small.

#### Time imputation — linear interpolation

Between two observations $(f_a, \sigma_a)$ at time $t_a$ and $(f_b, \sigma_b)$ at time $t_b$, with a gap of $g = t_b - t_a$ time periods, the interpolated value at position $t_a + s \cdot \Delta t$ (fractional position $\lambda = s/g \in [0,1]$) is:

$$f_j = (1-\lambda)\,f_a + \lambda\,f_b$$

$$\sigma_j \;=\; \bigl( |(1-\lambda)\,\sigma_a|^p + |\lambda\,\sigma_b|^p \bigr)^{1/p} \cdot \gamma_0^{\,g}$$

The multiplicative **gap inflation** factor $\gamma_0^{\,g}$ accounts for model uncertainty growing with gap size; $\gamma_0 = $ `gap_inflation` (default 1.0 = no inflation).

#### Time imputation — extrapolation (LOCF / NOCB)

Last-observation-carried-forward at gap $g$:

$$f_j = f_a, \qquad \sigma_j = \sigma_a \cdot \gamma_0^g$$

The gap inflation grows $\sigma$ exponentially with the extrapolation distance, reflecting increasing uncertainty the further the estimate is from the last observation.

#### Pipeline chaining

Steps are applied in registration order. After step $s$, let $\mathcal{M}_s$ denote the set of still-missing edges. Step $s+1$ only touches $\mathcal{M}_s$; it reads the current $(f, \sigma)$ values of already-filled edges (including those filled by earlier steps) as valid inputs. This means a `ParametricImputer` registered after a `TimeImputer` can use time-filled values as sources.

### Stage 3a — Constraint types

All five constraint types produce rows in the joint constraint matrix **G** ∈ ℝ^(M×E) with associated right-hand side **c** ∈ ℝ^M and per-row uncertainty **σ_c** ∈ ℝ^M.

#### Summary

| # | Type | Form | Cone | σ_c |
|---|---|---|---|---|
| 1 | Hard mass balance | Σ incoming_i − Σ outgoing_i = 0 | Zero | 0 |
| 2 | Soft sum | Σ coeff_i · a_i = c_k + ε_k | Free | user-specified |
| 3 | Soft ratio | a_i − r · a_j = ε_k | Free | from params CSV |
| 4 | Hard ratio bound | a_i − r_u · a_j ≤ 0 and −a_i + r_l · a_j ≤ 0 | Non-neg | 0 |
| 5 | Hard flow bound | a_i ≤ u_i and −a_i ≤ −l_i | Non-neg | 0 |

#### Type 1 — Hard mass balance (zero cone)

For every (Flow | Process) node $n$ at every timestep $t$ that has at least one incoming and one outgoing admissible edge:

$$\sum_{e \,:\, \mathrm{to}(e)=n,\, t(e)=t} a_e \;-\; \sum_{e \,:\, \mathrm{from}(e)=n,\, t(e)=t} a_e \;=\; 0$$

One **G** row per balanced (node, time) pair. SB nodes are open-boundary sinks/sources and are excluded. σ_c = 0 for all mass-balance rows.

#### Type 2 — Soft sum constraint (free cone)

A user-defined linear combination of edges with target value $c_k$ and slack $\varepsilon_k$:

$$\sum_{i \in S} \alpha_i \cdot a_i \;=\; c_k + \varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}(0,\, \sigma_{c,k}^2)$$

where $S$ is the set of matched edges (from CSV selectors or auto-derived from `acc_struct`) and $\alpha_i$ are the signed coefficients (+1 for supply, −1 for waste).

**σ_c modes:**
- `sd`: $\sigma_{c,k}$ supplied directly
- `fraction_of_flow`: $\sigma_{c,k} = \phi \cdot \left|\sum_{i \in S,\, \alpha_i>0} f_\mathrm{agg}[i]\right|$ — scales with the magnitude of the supply-side flows at each timestep

#### Type 3 — Soft ratio constraint (free cone)

Links two edges $i$ (target) and $j$ (source) through a scalar coefficient $r$:

$$a_i - r \cdot a_j \;=\; \varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}(0,\, \sigma_{c,k}^2)$$

$r$ is the `value_nominal` from the params CSV. $\sigma_{c,k}$ is derived from the same three uncertainty modes as `ParametricImputer`:

| Mode | $\sigma_{c,k}$ |
|---|---|
| `sd` | `value_sd` directly |
| `min_max` | `(value_max − value_min) / 4` |
| `pedigree` | `PedigreeScores(...).total_cv() × value_nominal` |

Same params CSV as `ParametricImputer.from_csv()` — one file serves both imputation and constraint building.

#### Type 4 — Hard ratio bound (non-neg cone)

For each param row with finite `bound_lower` $r_l$ or `bound_upper` $r_u$, two inequality rows:

$$a_i - r_u \cdot a_j \leq 0 \qquad \text{(upper bound)}$$

$$-a_i + r_l \cdot a_j \leq 0 \qquad \text{(lower bound)}$$

Both rows have $\sigma_c = 0$ and `cone_type = "nonneg"`.

#### Type 5 — Hard flow bound (non-neg cone)

For each edge with non-default bounds (from `acc_struct.bounds`):

$$a_i \leq u_i \;\Rightarrow\; a_i - u_i \leq 0$$

$$a_i \geq l_i \;\Rightarrow\; -a_i + l_i \leq 0$$

Default bounds ($l_i = 0$, $u_i = 10^9$) produce no G rows. Only exceptions in the bounds spreadsheet generate rows.



`ConstraintEngine.run()` produces solver-neutral outputs. A thin transform script maps them to Clarabel's cone format. Partition **G** and **c** by `cone_type`:

$$\mathbf{G} = \begin{bmatrix} \mathbf{G}_\text{free} \\ \mathbf{G}_\text{zero} \\ \mathbf{G}_\text{nonneg} \end{bmatrix}, \qquad \mathbf{c} = \begin{bmatrix} \mathbf{c}_\text{free} \\ \mathbf{c}_\text{zero} \\ \mathbf{c}_\text{nonneg} \end{bmatrix}$$



### Stage 3b — Constrained optimisation objective

#### The (P\*) problem

$$\min_{\mathbf{a},\,\boldsymbol{\varepsilon}} \quad (\mathbf{a} - \mathbf{a}_0)^\top \Sigma_a^{-1} (\mathbf{a} - \mathbf{a}_0) \;+\; \boldsymbol{\varepsilon}^\top \Sigma_c^{-1} \boldsymbol{\varepsilon}$$

$$\text{s.t.} \quad \mathbf{G}\,\mathbf{a} - \boldsymbol{\varepsilon} = \mathbf{c}, \qquad \mathbf{l} \leq \mathbf{a} \leq \mathbf{u}$$

| Symbol | Meaning in pySUTO |
|---|---|
| $\mathbf{a}$ | Vector of all flow values (854,460 edges for copper) |
| $\mathbf{a}_0$ | Aggregated prior from Stage 2 (inverse-variance weighted mean) |
| $\Sigma_a$ | Diagonal matrix of pedigree-derived flow variances |
| $\boldsymbol{\varepsilon}$ | Slack on balance/constraint equations |
| $\Sigma_c$ | Diagonal matrix of constraint uncertainties |
| $\mathbf{G}$ | Constraint matrix — encodes all linear equations on $\mathbf{a}$ |
| $\mathbf{c}$ | Right-hand side of constraints |
| $\mathbf{l}, \mathbf{u}$ | Physical lower/upper bounds (e.g. $l_i = 0$ for non-negative flows) |

#### Solver


##### Clarable


Clarabel solves:

$$\min_x \quad \tfrac{1}{2} x^\top P x + q^\top x \qquad \text{s.t.} \quad Ax + s = b, \quad s \in \mathcal{K}$$

where $\mathcal{K} = \mathcal{K}_1 \times \mathcal{K}_2 \times \cdots$ is a product of cones.

Stack the decision variable:

$$x = \begin{bmatrix} \mathbf{a} \\ \boldsymbol{\varepsilon} \end{bmatrix} \in \mathbb{R}^{n_a + n_c}$$

Objective in Clarabel's $\tfrac{1}{2}x^\top P x + q^\top x$ form (note the factor $\tfrac{1}{2}$,
so $P$ absorbs a factor of 2):

$$P = 2\begin{bmatrix} \Sigma_a^{-1} & 0 \\ 0 & \Sigma_c^{-1} \end{bmatrix} \quad \text{(upper triangle only, CSC format)}$$

$$q = \begin{bmatrix} -2\,\Sigma_a^{-1}\mathbf{a}_0 \\ \mathbf{0} \end{bmatrix}$$

The Clarabel constraint matrix and RHS are then:

$$A = \begin{bmatrix} \mathbf{G}_\text{free} & -\mathbf{I}_\text{free} \\ \mathbf{G}_\text{zero} & \mathbf{0} \\ \mathbf{G}_\text{nonneg} & \mathbf{0} \end{bmatrix}, \qquad b = \begin{bmatrix} \mathbf{c}_\text{free} \\ \mathbf{c}_\text{zero} \\ \mathbf{c}_\text{nonneg} \end{bmatrix}$$

Cone order: `ZeroConeT(n_free)` for the soft-equality residual rows (slack ε is a free decision variable inside `x`; the row residual `Gx - ε = c` is itself an exact equality), `ZeroConeT(n_zero)` for hard balance rows, `NonnegativeConeT(n_nonneg)` for bound rows. **P** and **q** follow the existing (P\*) formulation below.

$$ b = c $$

### Stage 3c — Solver diagnostics

`SolverDiag` ([src/pysuto/diag.py](src/pysuto/diag.py)) renders a 13-figure portfolio that turns raw `OptimResult` arrays into actionable views of convergence behaviour, constraint satisfaction, and prior–posterior drift. It is triggered either directly with a live `results` dict + `ConstraintRun` (typically right after `solver.solve_all()`) or by `opt_run_id` against the persisted registry. Output PNGs land under `{output_dir}/{opt_run_id}/diag/`.

```python
from pysuto.diag import SolverDiag

# (A) Direct from a finished solver
diag = SolverDiag(results, constraint_run, label="copper")
diag.summary()
diag.plot()                                           # default odir resolves to {run_dir}/diag

# (B) Reconstruct from a persisted run id
diag = SolverDiag.from_opt_id("optim_<…>_20260511_…", output_dir="data/proc/opt")
diag.plot()                                           # → data/proc/opt/<opt_run_id>/diag/

# (C) Overlay multiple runs (run-scope plots only)
SolverDiag.compare({"sf005": d1, "sf01": d2}, odir="data/proc/opt/compare")
```

#### Plot portfolio

Two scopes:

* **Scope 1 — run-scope** (`SD-R-NN`): x-axis is timestep, one line per run. The five hueable plots accept `SolverDiag.compare()` for overlay; `SD-R-06` and `SD-R-07` are single-run only.
* **Scope 2 — per-timestep cross-section** (`SD-T-NN`): three columns (`t_min`, `t_worst`, `t_max`), every cell is a histogram so the columns are directly visually comparable. `t_worst = argmax_t ‖ε(t)‖₂` by default; pass `timesteps=(...)` to override.

**Scale-transformation note:** wherever a y- or colour-axis is log₁₀- or symlog-transformed, the axis label and/or colorbar label contains the annotation `(log₁₀-transformed)` or `symlog scale`.

| ID | Title | Scope | Description | Interpretation |
|---|---|---|---|---|
| **SD-R-01** | Solver health | run | 4 stacked panels: objective, iterations, `solve_time_s`, final KKT `(res_primal, res_dual, gap_abs)` semilogy with tolerance band | Warm-start working ⇒ iterations and time drop after `t=0`; KKT spikes mark non-converged timesteps. Hueable. |
| **SD-R-02** | Mass-balance leakage | run | `max\|residuals\|` and `‖residuals‖₂` over [Type-1](#type-1--hard-mass-balance-zero-cone) rows (semilogy) + solver tolerance band | Type-1 rows must sit at solver tolerance; persistent breaches localise to node-scaling pathology. Hueable. |
| **SD-R-03** | Soft-slack stress | run | Panel A: `‖ε‖∞`, `‖ε‖₂`, median `\|ε\|`, max `\|ε\|` per timestep (semilogy). Panel B: fraction of soft rows with `\|ε_k\| > 3·σ_{c,k}`, split by [Type-2](#type-2--soft-sum-constraint-free-cone) / [Type-3](#type-3--soft-ratio-constraint-free-cone) | Sustained high panel-B fraction ⇒ soft constraints inconsistent with the data. Hueable (drops type-split when hued). |
| **SD-R-04** | Hard-inequality activity | run | Panel A: count of active [Type-4](#type-4--hard-ratio-bound-non-neg-cone) ratio bounds. Panel B: count of edges at `l` / `u` (Type-5 flow bounds; computed from `F_opt` because flow bounds live in `u`/`l`, not in `G`) | A flow pegged at its bound for every `t` is a candidate for bound review. Hueable. |
| **SD-R-05** | Warm-start delta | run | Consecutive `‖a*(t) − a*(t−1)‖` in L∞, L₂, and relative L₂ (semilogy) | Jitter here correlates with iteration spikes in SD-R-01: warm-start not amortising. Hueable. |
| **SD-R-06** | `F*(t)` evolution | run | Panel A: symlog heatmap (symlog-transformed colour) of `F_opt` for the top-200 most-variable edges. Panel B: line plot of the top-10 most-variable edges on a symlog y-axis (symlog scale annotated) | Sharp jumps between adjacent `t` flag warm-start failure or genuine structural discontinuity. Not hueable. |
| **SD-R-07** | Time-hued distributions | run | 3 stacked panels, each a per-timestep step-histogram hued by viridis time gradient: (A) log₁₀\|a*\| (log₁₀-transformed); (B) log₁₀\|c*\| on soft rows (log₁₀-transformed); (C) log₁₀\|ε_opt\| (log₁₀-transformed) | Shifts in the histogram bulk across years reveal structural regime changes; panel C shows how ε_opt evolves independently of c*. Not hueable. |
| **SD-T-01** | Constraint-residual digest | timestep | 5 rows: (1) [Type-1](#type-1--hard-mass-balance-zero-cone) `Gx`; (2) [Type-2](#type-2--soft-sum-constraint-free-cone) ε; (3) [Type-3](#type-3--soft-ratio-constraint-free-cone) ε; (4) [Type-4](#type-4--hard-ratio-bound-non-neg-cone) slack; (5) [Type-5](#type-5--hard-flow-bound-non-neg-cone) slack `(u−F, F−l)`. Footer: `n / median / p95 / max` | One-page snapshot of where the solver succeeded vs. compromised, broken down by constraint type. |
| **SD-T-02** | Soft-slack z-scores | timestep | 2 rows: (1) Type-2 `ε_k/σ_{c,k}`; (2) Type-3 `ε_k/σ_{c,k}` with reference lines at ±1, ±3. Y-axis symlog-transformed; x-axis thinned to ≤5 major ticks. | Bars outside ±3 are constraints the solver couldn't honour within their stated σ_c. |
| **SD-T-03** | Solution vs prior | timestep | 2–3 rows: (1) `(a*−a₀)/σ_a` z-score; (2) `(a*−a₀)/\|a₀\|` relative change (symlog x); (3) **optional** `σ_opt/σ_a` ratio when `sigma_opt` is provided. Y-axis symlog-transformed on all rows; x-axis thinned to ≤5 major ticks. | Row-1 tails mark constraint-dominated edges; row 3 below 1 confirms `OptUncertainty` is tightening σ. |
| **SD-T-04** | Per-type binding rate | timestep | 2 rows: (1) [Type-4](#type-4--hard-ratio-bound-non-neg-cone) expression `Gx − c` (symlog) with shading on the `≤0` slack side; (2) [Type-5](#type-5--hard-flow-bound-non-neg-cone) `F−u` and `l−F` distribution | Negative ⇒ slack, zero ⇒ active, positive ⇒ violation (should not happen if solver converged). |
| **SD-T-05** | Distribution of a | timestep | 3 rows: (1) log₁₀\|a*\| histogram (log₁₀-transformed) with p50/p95/p99 markers; (2) a* signed symlog histogram (symlog scale); (3) log₁₀\|a₀(t)\| histogram (log₁₀-transformed; prior is the warm-start path). Shared bin edges within each row across the three columns. | Rows 1–2 together show both magnitude and sign structure; row 3 reveals how the prior shifts with time. |
| **SD-T-06** | `c*` vs `c` | timestep | 2 rows for soft rows only: (1) `c* − c = ε` (symlog x, symlog y); (2) `(c* − c)/(\|c\|+δ)` relative residual (symlog y) with ±1 reference lines. X-axis thinned to ≤5 major ticks. | Asymmetric tails ⇒ systematic bias on soft rows. A broad row-2 distribution ⇒ the solver is fighting data magnitude rather than noise level. |
| **SD-T-07** | Rocket scatter | timestep | 3 rows: (1) log₁₀\|c\| vs log₁₀\|c*\| on sum_eq only, colour = \|ε\|/σ_c; (2) log₁₀\|ε₀\| vs log₁₀\|ε_opt\| on free-cone rows, colour = timestep — shows how optimisation reduces the prior soft residual; (3) log₁₀\|a₀(t)\| vs log₁₀\|a*\|, colour = \|a*−a₀\|/σ_a. All axes log₁₀-transformed. Diagonal = perfect agreement. | Row 2 points above the diagonal ⇒ solver amplified the soft residual (unusual); far below ⇒ data and prior are inconsistent. |
| **SD-T-08a** | ratio_eq ε heatmap | — | 1 row × 2 cols: (left) log₁₀\|ε\| per (constraint\_id × timestep); (right) log₁₀\|ε\|/σ_c. Top-50 constraint_ids by mean\|ε\| across all timesteps (log₁₀-transformed colour). | Persistent hot rows identify ratio constraints that the solver routinely cannot satisfy. |
| **SD-T-08b** | sum_eq ε heatmap | — | 1 row × 2 cols: (left) log₁₀\|ε\| per (node × timestep); (right) log₁₀\|ε\|/σ_c. Top-50 nodes by mean\|ε\| across all timesteps (log₁₀-transformed colour). | Persistent hot rows identify mass-balance nodes with the largest systematic imbalance. |

#### Layout conventions

- Per-constraint-type colour palette is shared across every figure (`_TYPE_COLORS`): mass_balance = navy, sum_eq = blue, ratio_eq = purple, ratio_bound = orange, flow_bound = green.
- Modern style: top/right spines hidden, `grid.alpha = 0.25`, `stepfilled` histograms at `alpha = 0.55`, dashed grey reference lines.
- Filenames: `{label}_{plot_id_lower}_{slug}.png` (e.g. `copper_sd_r_01_solver_health.png`).
- The `plot()` method returns a list of saved paths and prints `[ID] → [path]  ✓` to stdout for each figure, matching `AggDiagnostic` / `ImpDiagnostic`.

## Workflow

```python
# ── Stage 1: one feed pipeline run per data source ─────────────────────────

registry = DataFeedRegistry(odir="data/proc/feed/registry")
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

# ── Stage 2a: select and align feeds ───────────────────────────────────────

collector = FeedCollector(registry=registry, acc_struct=acc)

# Select by ID — latest registered version used automatically
collector.select(["baci", "crude_ore"])

# Or select a specific version:
# collector.select([("baci", 1), "crude_ore"])

# Power-mean exponent shared across Stage 1, 2a, and every imputer.
P_PROP = 1.3
# ParametricImputer α-regularization (caps σ amplification for large |α|, floors small |α|).
Q_REG = 3.0
ALPHA_MIN_REG = 0.05

agg_result = collector.aggregate(p=P_PROP)
# agg_result.f_agg    [N_r]       inverse-variance weighted mean
# agg_result.sigma_agg[N_r]       combined uncertainty (Lp norm of Ω·σ)
# agg_result.Omega    [N_f × N_r] provenance weight matrix
# agg_result.coverage [N_r] bool  True where at least one feed has data

# Snapshot post-aggregation distributions BEFORE imputation mutates them.
agg_diag = AggDiagnostic(agg_result, label="copper")
agg_diag.summary()

# ── Stage 2b: imputation pipeline ──────────────────────────────────────────

# Define parametric relations
scrap_return = ParametricRelation(
    description="Scrap return ≈ 12 % of cathode→FU flow (same region, same time)",
    target=lambda ei: (ei.sector_from == "SR") & (ei.sector_to == "FU"),
    sources=lambda ei: (ei.sector_from == "Cathode") & (ei.sector_to == "FU"),
    coefficients=[0.12],
    epsilon=0.25,
)

engine = ImputationEngine(acc_struct=acc)

# Concentration and efficiency parameters from CSV
engine.register(ParametricImputer.from_csv(
    "data/input/params/copper_conc_params.csv", acc_struct=acc,
    p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG))
engine.register(ParametricImputer.from_csv(
    "data/input/params/copper_efficiency_params.csv", acc_struct=acc,
    p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG))

# Lambda-based custom relation
engine.register(ParametricImputer(
    relations=[scrap_return], p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG))

# Balance equations for domestic use of traded flows
engine.register(FormulaImputer(relations=[domestic_matte], p=P_PROP))

# Temporal gap fill
engine.register(TimeImputer(window=5, method="linear", gap_inflation=1.05, p=P_PROP))

# Final fallback — flat prior sized by mask CV × group scale (register LAST)
engine.register(FlatPriorImputer.from_csv(
    "data/input/params/flat_prior_cv.csv",
    acc_struct=acc,
    default_cv=0.3,
    p=P_PROP,
))

agg_result = engine.run(agg_result)
# agg_result.f_agg updated in-place for imputed edges
# agg_result.imputation_log: complete per-edge fill record

# Diagnostics: pre-imputation aggregation distributions + per-step imputation trajectories
agg_diag.plot(odir="data/proc/imp/copper/diag", show=False)
ImpDiagnostic(agg_result, acc_struct=acc, label="copper").plot(odir="data/proc/imp/copper/diag")

# Inspect provenance for a specific edge:
log = agg_result.imputation_log
print(log.for_edge(edge_id=42))
# → method="parametric", source_edges=[1305], coefficients=[0.12], sigma_after=0.031

# ── Stage 3a: build constraint matrix ───────────────────────────────────────

const_engine = ConstraintEngine(acc_struct=acc)

# Hard mass balance: Σ incoming − Σ outgoing = 0 per (Flow|Process) node × time
const_engine.register(MassBalanceBuilder())

# Soft useful-supply balance auto-derived from acc_struct topology
# (excludes SB/waste edges; σ_c = 5 % of node flow magnitude)
const_engine.register(SumConstraintBuilder.from_acc_struct(acc, sigma_fraction=0.05))

# Custom sum constraint from CSV (long format, one row per term)
const_engine.register(SumConstraintBuilder.from_csv(
    "data/input/params/sum_constraints.csv", acc_struct=acc))

# Soft ratio constraints + hard ratio bounds from the same params CSV
# (reuses the ParametricImputer schema: value_nominal, unc_type, bound_lower, bound_upper)
const_engine.register(RatioConstraintBuilder.from_csv(
    "data/input/params/copper_params.csv", acc_struct=acc))
const_engine.register(RatioBoundBuilder.from_csv(
    "data/input/params/copper_params.csv", acc_struct=acc))

# Hard flow bounds from acc_struct.bounds (only non-default edges produce rows)
const_engine.register(FlowBoundsBuilder())

constraint_run = const_engine.run(
    agg_result=agg_result,
    timesteps=[str(t) for t in range(2010, 2024)],
    agg_id="copper_agg_v0",
)
# constraint_run.G          sparse (M × E), time-invariant
# constraint_run.row_meta   DataFrame — constraint_id, type, is_hard, cone_type per row
# constraint_run.a0         f_agg at t=0 (prior for optimiser)
# constraint_run.timestep_data["2010"] → {"c": ..., "sigma_c": ...}
# persisted automatically to data/proc/const/

# ── Stage 3b: optimise over T timesteps ─────────────────────────────────────

opt = LinearOptimiser(acc_struct=acc)
unc = OptUncertainty()

pipeline = OptimPipeline(
    acc_struct=acc,
    constraint_run=constraint_run,
    optimiser=opt,
    unc=unc,
)

run = pipeline.run(timesteps=list(range(2010, 2024)))

# run.results[t] → OptimResult with F_opt, σ_opt, residuals, solver_status
# Warm start: a_0(t) = F_opt(t−1) for t > 0; constraint_run.a0 used for t = 0
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
├── collector.py           FeedCollector · AggregationResult             ─┐
├── agg.py                 AggregationMath  (Ω · f_agg · σ_agg)          │ Stage 2
├── impute.py              ImputationEngine · BaseImputer                 │
│                          ParametricImputer · FormulaImputer             │
│                          TimeImputer                                     │
│                          ParametricRelation · FormulaRelation            │
│                          SourceTerm · ParamRef · ImputationLog         ─┘
│
├── constraints.py         ConstraintEngine · BaseConstraintBuilder       ─┐
│                          MassBalanceBuilder · SumConstraintBuilder       │
│                          RatioConstraintBuilder · RatioBoundBuilder      │ Stage 3
│                          FlowBoundsBuilder                               │
│                          ConstraintBlock · ConstraintRun                 │
│                          ConstraintRunRegistry                           │
├── optim.py               LinearOptimiser · OptimResult                  │
├── optunc.py              OptUncertainty                                  │
├── optpipe.py             OptimPipeline · OptimRun                       │
└── diag.py                AggDiagnostic (Stage 2a) · ImpDiagnostic        │
                           (Stage 2c) · SolverDiag (Stage 3c)             ─┘
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
| `test_collector.py` | Feed alignment, NaN coverage mask, Ω column sums to 1 | `mock_registry`, `mock_acc` |
| `test_aggregation.py` | Ω construction, `f_agg = 1ᵀ(Ω⊙F)`, σ_agg ≤ min σ_k, NaN coverage | `F_src`, `sigma_src` |
| `test_imputation.py` | ParametricImputer fill, TimeImputer linear/locf, log completeness, σ propagation, pipeline chaining | `agg_result_partial` |
| `test_constraints.py` | MassBalance rows (balanced nodes, SB exclusion, cone type), SumConstraint (acc_struct auto-derive, CSV load, σ_c scaling), RatioConstraint (G structure, all unc modes), RatioBound (two rows per param, inequality direction), FlowBounds (non-default edges), ConstraintEngine (shapes, G time-invariance, row_meta), ConstraintRunRegistry round-trip | `toy_acc` (5-node inline fixture) |
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

**Core dependencies:** `numpy`, `pandas`, `scipy`

**Optional:** `matplotlib` for diagnostic plots of feed uncertainty distributions

---

## References

Laner, D., Rechberger, H., & Astrup, T. (2016). A novel approach to characterize data uncertainty in material flow analysis and its application to plastics flows in Austria. *Journal of Industrial Ecology*, 20(5), 1050–1063.

Lenzen, M., Wood, R., & Wiedmann, T. (2009). Matrix balancing under conflicting information. *Economic Systems Research*, 22(1), 1–23.
