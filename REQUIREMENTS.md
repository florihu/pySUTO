# pySUTO — Project Requirements

This file is the single source of truth for all decisions that constrain
how the codebase is built, tested, and extended. It covers modelling
assumptions, data conventions, numerical contracts, and open questions.

Engineers touching any part of the codebase should check this file first
and update it when a decision changes.

---

## Table of contents

1. [Accounting structure](#1-accounting-structure)
2. [Edge bounds](#2-edge-bounds)
3. [Data feeds and concordance](#3-data-feeds-and-concordance)
4. [Uncertainty model](#4-uncertainty-model)
5. [Aggregation](#5-aggregation)
6. [Imputation](#6-imputation)
7. [Optimisation](#7-optimisation)
8. [Persistence and registry](#8-persistence-and-registry)
9. [Numerical conventions](#9-numerical-conventions)
10. [Testing contracts](#10-testing-contracts)
11. [Open questions](#11-open-questions)

---

## 1. Accounting structure

| # | Requirement | Rationale |
|---|---|---|
| A1 | The `AccountingStructure` is the single owner of the node/edge space. All other classes receive it by dependency injection; none may build their own node or edge index. | Guarantees one consistent index throughout a run. |
| A2 | Nodes are identified by the triple `(sector, entity, region)`. Edge keys are `(from_node, to_node, time, valuation)`. | Mirrors standard MRIO/SUT conventions. |
| A3 | `time_range` and `valuation` must be deduplicated on construction. Duplicate entries are silently removed with a warning. | Prevents silent index misalignment. |
| A4 | Rules are pure functions with no side-effects. Node rules have signature `(ni, nj) -> bool`; edge rules have signature `(ni, nj, time, valuation) -> bool`. Rule type is auto-detected by argument count. | Keeps the rule language simple and testable. |
| A5 | Rules short-circuit: the first `False` stops evaluation. Node-level rules are tested before edge-level rules. | Performance: avoids redundant computation on dense node spaces. |
| A6 | `edge_index` and `edge_index_all` are built once and treated as immutable after construction. Downstream code must never modify them in place. | Prevents silent index corruption. |

---

## 2. Edge bounds

| # | Requirement | Rationale |
|---|---|---|
| B1 | Default lower bound: `l = 0`. Default upper bound: `u = +inf`. These apply to every admissible edge unless overridden. | Physical flows are non-negative unless explicitly modelled as signed. |
| B2 | Exceptions are stored in an Excel file (`bounds.xlsx`, sheet `bounds`). The file is optional; if absent, all-default bounds are used without error. | Keeps the common case (no exceptions) frictionless. |
| B3 | Exception rules match on `(sector_from, sector_to, valuation)`. `region_from`, `region_to`, and `time` are optional filters. `*` matches any value in any column. | Most exceptions are sector/valuation-based, not region-specific. |
| B4 | Rules are applied top-to-bottom. A later rule that matches the same edge overrides an earlier rule. | Allows broad exception + narrow refinement in the same file. |
| B5 | A rule that matches zero edges produces a logged warning but is not an error. | A rule may be written in anticipation of sectors that are added later. |
| B6 | A rule referencing a `sector_from` or `sector_to` that does not exist in the node index produces a logged warning but is not an error. | Same reasoning as B5. |
| B7 | Bound vectors `l` and `u` are `np.ndarray` of dtype `float64`, shape `(n_edges,)`, aligned to `edge_index` row order. | Allows direct use in `scipy.optimize` without further transformation. |
| B8 | `acc.l` and `acc.u` are read-only properties on `AccountingStructure`. They delegate to `acc.bounds.l` and `acc.bounds.u`. | Clean access pattern for `ConstraintBuilder` and `OptimPipeline`. |
| B9 | `EdgeBounds.validate(F, edge_index)` must be called by `OptimPipeline` after every timestep. The pipeline logs all violations and raises if `raise_on_violation=True` (default). | Optimiser may technically satisfy `G f = c` while violating bounds if bounds are not hard constraints in the solver. |
| B10 | Bounds are persisted alongside the structure as `bounds_l.npy` / `bounds_u.npy`. `AccountingStructure.load()` restores them from these files, not by re-parsing the Excel. | Reproducibility: the saved state matches exactly what was used at construction. |

**Sectors that require signed bounds (known exceptions at project start):**

| Sector | Reason | Default exception |
|---|---|---|
| `PSC` | Producer stock change — draw-downs are negative | `l = -1e9, u = 1e9` |
| `CSC` | Consumer stock change — same as PSC | `l = -1e9, u = 1e9` |
| `VA`  | Value added — can be negative (e.g. subsidies exceed output) | `l = -1e9, u = 1e9` |
| `NX`  | Net exports — net importer produces negative value | `l = -1e9, u = 1e9` |

These are starting assumptions. Tighten or relax per material and region as data allows.

---

## 3. Data feeds and concordance

| # | Requirement | Rationale |
|---|---|---|
| C1 | Every data feed must have a unique `feed_id` string. The registry raises on duplicate registration unless `overwrite=True`. | Prevents silent data replacement. |
| C2 | A `ConcordanceTable` may only reference nodes and edges that exist in the `AccountingStructure` it was built against. Validation runs at construction. | Catches classification errors early. |
| C3 | Weight rows in a `ConcordanceTable` must sum to 1.0 per source edge (within `numpy.allclose` tolerance `1e-6`). | Preserves mass balance through the weight matrix. |
| C4 | Three weight modes are supported: `user` (explicit values provided), `uniform` (equal split), `manufacturing_shares` (from a shares DataFrame). No others. | Limits the complexity surface. |
| C5 | Source edges that match no root edge produce a warning if `allow_unmatched_edges=True`, or raise if `False`. Default: `True`. | Most workflows have some unmatched source edges (e.g. aggregates). |
| C6 | `F_src` is a 1-D `np.ndarray` of length `n_src_edges`, aligned to the `src_edges` index of the `ConcordanceTable`. | Unambiguous alignment. |

---

## 4. Uncertainty model

The pedigree model follows **Laner et al. (2016)**.

| # | Requirement | Rationale |
|---|---|---|
| U1 | Five quality indicators: `reliability`, `completeness`, `temporal_correlation`, `geographical_correlation`, `other_correlation`. Scores ∈ {1, 2, 3, 4}. | Matches Laner et al. Table 1. |
| U2 | CV formula for reliability: `CV = 0.75 · exp(1.105 · score) / 100`. Non-zero at score 1. | Reliability always contributes baseline uncertainty. |
| U3 | CV formula for all other indicators: `CV = a · exp(1.105 · (score − 1)) / 100`. Zero at score 1. | Zero uncertainty at best quality for these indicators. |
| U4 | Sensitivity parameter `a`: `not_sensitive = 0.375`, `medium_sensitive = 0.75`, `highly_sensitive = 1.5`. | Laner et al. Table 2. |
| U5 | Total CV: `CV_tot = sqrt(CV_rel² + CV_comp² + CV_temp² + CV_geo² + CV_other²)`. | Independent error sources, Gaussian sum-in-quadrature. |
| U6 | Pedigree CV is uniform across all source edges of a single feed. A feed has one set of quality scores. | One pedigree per dataset, not per observation. |
| U7 | After classification: `σ_root = CV_tot · sqrt((W ⊙ W)ᵀ F_src²)`. | Gaussian error propagation through the linear weight map. |
| U8 | Invariant (tested): `CV_root[j] ≤ CV_tot` for all root edges `j`. Aggregation cannot amplify proportional uncertainty. | Diversification property. |

---

## 5. Aggregation

| # | Requirement | Rationale |
|---|---|---|
| AG1 | Aggregation uses inverse-variance weighting: `w_kj = 1 / σ²_kj`. | BLUE estimator under independent Gaussian errors. |
| AG2 | `F_agg_j = Σ_k(w_kj · F_kj) / Σ_k(w_kj)`. | Weighted mean. |
| AG3 | `σ_agg_j = 1 / sqrt(Σ_k(w_kj))`. | Propagated uncertainty for the weighted mean. |
| AG4 | If a single feed has `σ_kj = 0` for edge `j` (perfectly certain), that feed dominates with weight `+inf`. Handle by setting `σ_agg_j = 0` and `F_agg_j = F_kj` for that feed. | Avoids division by zero; perfectly certain data should be trusted. |
| AG5 | Edges with no feed coverage (all weights zero / all NaN) are flagged in `AggregationResult.imputation_log` and passed to `ImputationEngine`. | Explicit tracking of data gaps. |

---

## 6. Imputation

| # | Requirement | Rationale |
|---|---|---|
| IM1 | Imputers are registered in order and run sequentially. Each imputer receives the current `(F, sigma, missing_mask)` and returns an updated triple plus a log list. | Composable pipeline; easy to add or remove strategies. |
| IM2 | `ParametricImputer` fills edge `b` from edge `a` via `f_b = α · f_a`, `σ_b = α · σ_a`. The relation dict comes from `acc_struct` or a user-supplied mapping. | Preserves known physical relationships (efficiencies, splits). |
| IM3 | `TimeSeriesImputer` operates only if the `AggregationResult` covers multiple timesteps or if prior-timestep data is available in the registry. Methods: `linear`, `locf`, `nocb`. | Gap-fill for isolated missing years. |
| IM4 | Every filled edge must appear in `imputation_log` with: `edge_id`, `method`, `source_edge_id` (for parametric), `filled_value`, `filled_sigma`. | Full audit trail. |
| IM5 | Imputed edges are flagged with `is_imputed=True` in `AggregationResult`. The optimiser treats imputed edges with their imputed `σ` (not zero). | Imputed values are less certain than observed ones. |

---

## 7. Optimisation

| # | Requirement | Rationale |
|---|---|---|
| O1 | Objective: minimise `(f − F_agg)ᵀ Σ_agg⁻¹ (f − F_agg)` subject to `G f = c` and `l ≤ f ≤ u`. | σ-weighted least squares; Lenzen et al. (2009) formulation. |
| O2 | `G` encodes two constraint types: (i) supply = use balance rows from `AccountingStructure`; (ii) parametric efficiency rows from `ParametricImputer` relations. | Physical accounting identities and known material ratios. |
| O3 | Warm-start: for `t > 0`, the initial point `f0` is `F_opt[t-1]`. For `t = 0`, `f0 = F_agg`. | Exploits temporal continuity; improves solver speed. |
| O4 | Bound constraints `l ≤ f ≤ u` are hard constraints passed directly to the solver (`scipy.optimize.lsq_linear` or equivalent QP). | Bounds must be satisfied exactly, not penalised. |
| O5 | `OptUncertainty` computes `σ_opt` via KKT sensitivity: `J = Σ_agg Gᵀ (G Σ_agg Gᵀ)⁻¹`, `Σ_opt = J Σ_c Jᵀ`, `σ_opt = sqrt(diag(Σ_opt))`. | Lenzen et al. (2009) / van der Ploeg (1982, 1984, 1988). |
| O6 | `OptimResult` stores: `F_opt`, `σ_opt`, `residuals (G·F_opt − c)`, `objective`, `solver_status`, `G`, `c`, `t`. | Full reproducibility of any single-timestep result. |
| O7 | `OptimPipeline` calls `acc.bounds.validate(F_opt, acc.edge_index)` after each timestep. Violations are recorded in `OptimResult.bound_violations`. | Explicit bounds audit per timestep. |
| O8 | If the solver returns a non-zero residual exceeding `tolerance = 1e-6`, log a warning with the maximum absolute residual. Do not raise; supply–use imbalance in input data is expected. | Distinguishes solver failure (raise) from data inconsistency (warn). |

---

## 8. Persistence and registry

| # | Requirement | Rationale |
|---|---|---|
| R1 | The registry index (`registry.json`) is a human-readable JSON file. Heavy arrays go in `.pkl` or `.npz` files. | JSON is grep-friendly; binary is efficient for arrays. |
| R2 | Every registered feed has a `metadata.json` sidecar with all scalar fields from `DataFeedMetadata`. | Enables audit without loading binary files. |
| R3 | `OptimRun` stores `registry_ids` — the exact list of feed IDs used — so any run can be reproduced from the registry. | Full provenance trace. |
| R4 | `AccountingStructure.save()` must persist `bounds_l.npy` and `bounds_u.npy` alongside the structure. `load()` restores from these files. | Loaded structures must be bit-identical to the constructed ones. |
| R5 | All timestamps use UTC ISO-8601 format with second resolution: `2025-06-11T14:32:00Z`. | Unambiguous across timezones. |

---

## 9. Numerical conventions

| Convention | Value | Applies to |
|---|---|---|
| Default lower bound | `0.0` | All edges unless overridden in bounds file |
| Default upper bound | `+numpy.inf` | All edges unless overridden in bounds file |
| Weight sum tolerance | `1e-6` (via `numpy.allclose`) | `ConcordanceTable._validate_weight_sums` |
| CV units | Fraction (not percent) | All CV values throughout the codebase |
| Optimiser residual warning threshold | `1e-6` (absolute) | `OptimPipeline` post-solve check |
| JSON float precision | Full float64 | `_NpEncoder` — no rounding |
| Infinity representation in JSON | `"Infinity"` string | `_NpEncoder` must handle `numpy.inf` |

---

## 10. Testing contracts

Every module has a dedicated unit test. The following invariants must be asserted in tests:

| Invariant | Location |
|---|---|
| `CV_root[j] ≤ CV_src` for all j | `test_weightmatrix.py` |
| Weight rows sum to 1.0 | `test_concordance.py` |
| `acc.l.shape == (acc.n_edges,)` | `test_acc.py` |
| `acc.u.shape == (acc.n_edges,)` | `test_acc.py` |
| Default bounds: `acc.l.min() == 0` and `all(numpy.isinf(acc.u))` when no bounds file | `test_acc.py` |
| `EdgeBounds.validate()` returns empty DataFrame on a valid `F_opt` | `test_acc.py` |
| `EdgeBounds.validate()` raises on any `F < l` or `F > u` when `raise_on_violation=True` | `test_acc.py` |
| Aggregation invariant: `σ_agg[j] ≤ min_k(σ_kj)` | `test_aggregation.py` |
| Optimiser residual `‖G F_opt − c‖ ≤ tolerance` for feasible problems | `test_optimiser.py` |
| Bounds satisfied after optimisation: `(F_opt ≥ l).all()` and `(F_opt ≤ u).all()` | `test_optpipe.py` |
| Round-trip save/load of `AccountingStructure` preserves `l` and `u` exactly | `test_acc.py` |

---

## 11. Open questions

Items that are not yet decided. Each should be resolved before the relevant
module is implemented and the decision recorded in the appropriate section above.

| # | Question | Owner | Target |
|---|---|---|---|
| Q1 | Should `ParametricImputer` relations live in `AccountingStructure` (as part of the topology) or in a separate config file passed to `ImputationEngine`? | — | Before `impute.py` |
| Q2 | What solver backend for the bounded QP? `scipy.optimize.lsq_linear` handles linear bounds efficiently but requires the problem to be cast as a least-squares problem. A full QP solver (e.g. `quadprog`, `osqp`) may be needed if parametric constraints produce non-diagonal `G Σ G ᵀ`. | — | Before `optim.py` |
| Q3 | Should `TimeSeriesImputer` access the registry directly (to load prior-timestep results), or should the caller pre-load and pass prior values explicitly? Direct registry access couples imputation to persistence. | — | Before `impute.py` |
| Q4 | Bound violations after optimisation: currently logged as warning. Should severe violations (e.g. 10× bound) be treated as hard errors that abort the run? | — | Before `optpipe.py` |
| Q5 | How should conflicting bounds rules be handled — e.g. two rules that both fully match the same edge and specify different `lower` values? Currently last-row wins. Should there be a strict-mode that raises on overlap? | — | Before bounds file is in production use |
| Q6 | Should `σ_opt` from `OptUncertainty` respect the bound constraints (i.e. truncate the Gaussian at the bounds)? The KKT formula gives unconstrained propagation. For edges at an active bound, the true uncertainty is zero in that direction. | — | Before `optunc.py` |

---

*Last updated: see git log*
*Maintainer: add your name when updating a section*
