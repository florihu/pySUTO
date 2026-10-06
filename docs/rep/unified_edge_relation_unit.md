# The Unified `EdgeRelation` Unit

> **Status:** idea-stage design note (not an architecture requirement, not a committed
> refactor). A *lens*, plus a schema sketch and a non-binding roadmap.
> **Date:** 2026-06-02
> **Scope:** the whole pySUTO pipeline, Stage 1 (datafeed/concordance) → Stage 5
> (posterior uncertainty).
> **Anchor reference:** Geschke, Lenzen, Kanemoto & Moran, *AISHA: A tool to construct a
> series of contingency tables* — `docs/ref/Geschke et al. - AISHA A tool to construct a
> series of contingency tables.pdf`.

---

## 1. TL;DR — the claim

**The minimal orthogonal unit is a single declarative record we call an `EdgeRelation`:**

```
EdgeRelation = ⟨ selector, reduction, coefficients, value, sigma, role ⟩
```

over the **8-dimensional edge address**
`(time, valuation, region_from, entity_from, sector_from, region_to, entity_to, sector_to)`.

Every numerical relation pySUTO builds between edges — concordance `W`, aggregation `Ω`,
parametric/formula imputation, mass balance, sum constraints, ratios, bounds, even the
solver prior `a₀` and the posterior — is **one or more `EdgeRelation`s being compiled**
into that stage's matrix/vector. The stages are *consumers/compilers*; the QP and the
posterior are pure functions of the compiled forms.

The piece that makes the four primitives the user already uses — **mask, group,
coefficient, value** — collapse into one unit is this:

> **A mask and a group are the same object: a selector with a per-dimension *reduction*
> operator.** A *mask* sums/keeps-all over a dimension; a *group* loops over it (one
> instance per value). Add a coefficient, a value, and a sigma, tag it with a *role*,
> and you have spanned every computation in the codebase.

AISHA's "meta-command sheet" is the existence proof: a whole multi-region IO model is
expressed as a list of exactly such records (AISHA §3.6–3.7).

---

## 2. The 8-D edge address — the substrate

pySUTO's atomic flow ("edge") is addressed by an 8-tuple. From
[acc.py](../../src/pysuto/acc.py) (`_build_edge_index`, `edge_meta`) and its expanded
form `expand_edge_index` in [param_relation.py](../../src/pysuto/param_relation.py):

```
(time, valuation,
 region_from, entity_from, sector_from,
 region_to,   entity_to,   sector_to)
```

This is **identical** to AISHA's "8-dimensional hierarchy for addressing MRIO table
entries" (AISHA §1.2, Table 3):

| # | AISHA term            | AISHA alt. terminology         | pySUTO column |
|---|-----------------------|--------------------------------|---------------|
| 1 | Stacks                | Time                           | `time`        |
| 2 | Sheets                | Valuation                      | `valuation`   |
| 3 | Row Parents           | Region of flow origin          | `region_from` |
| 4 | Row Children          | Entity of flow origin          | `entity_from` |
| 5 | Row Grandchildren     | Sector of flow origin          | `sector_from` |
| 6 | Column Parents        | Region of flow destination     | `region_to`   |
| 7 | Column Children       | Entity of flow destination     | `entity_to`   |
| 8 | Column Grandchildren  | Sector of flow destination     | `sector_to`   |

Structurally, **pySUTO is a modern AISHA**: a tool that assembles heterogeneous feeds
into one vectorised contingency table `τ` (here `a`/`f_agg`), imposes hard + soft
constraints `Gτ = c`, and solves a reliability-weighted reconciliation
(AISHA Eq. 23 ≡ pySUTO's Stage-4 QP). Every relation any stage forms is a statement
*about a selected region of this 8-D space*.

---

## 3. The `EdgeRelation` unit

```
EdgeRelation = ⟨ selector, reduction, coefficients, value, sigma, role ⟩
```

| Field          | What it is | Already in the code as… |
|----------------|------------|--------------------------|
| **selector**   | A predicate over the 8-D address that picks edge sets. Either the A/B-field CSV schema or a Python callable. | `A_FIELDS`/`B_FIELDS` + `ParamRelationResolver` ([param_relation.py](../../src/pysuto/param_relation.py)); `SourceTerm.selector`/`FormulaRelation.target` callables ([impute/formula.py](../../src/pysuto/impute/formula.py)); `MaskRuleSet` ([masks.py](../../src/pysuto/masks.py)) |
| **reduction**  | Per-dimension operator: *loop* (one relation instance per value → group-by), *sum* (collapse → mask/aggregate), or *bind* (tie two dimensions → diagonal). | `_build_group_ids` ([impute/flat_prior.py](../../src/pysuto/impute/flat_prior.py)); `groupby(group_keys)` in `from_group_sums` ([constraints/sum.py](../../src/pysuto/constraints/sum.py)); `region_binding`/`time_binding` ([impute/formula.py](../../src/pysuto/impute/formula.py)) |
| **coefficients** | One linear weight per selected (term) edge. | `ParametricRelation.coefficients`, `SourceTerm.coefficient`, sum/ratio specs |
| **value**      | RHS / prior value: a constant, a data reference (`param_id` / file), or computed. | `value_nominal` (params CSV); `c_k` (sum specs); `0` (balance/ratio) |
| **sigma**      | Uncertainty of the value/relation. `σ = 0` ⇒ **hard**, `σ ≠ 0` ⇒ **soft**. Derived once, combined once. | `sigma_from_row` (`sd`/`min_max`/`pedigree`) + `lp_combine` ([lp.py](../../src/pysuto/lp.py)) |
| **role**       | What the relation *does*, hence which stage consumes it and which cone it lands in: `prior \| imputation \| equality \| bound`. | implicit today: which builder/imputer you instantiate |

**Orthogonality** (no field is derivable from another):
- `selector` (which edges) is independent of `reduction` (how to collapse/expand them):
  the same edge set can be summed into one row *or* looped into many.
- `coefficients` (the linear form) is independent of `value` (the RHS) and of `sigma`
  (its reliability) — AISHA's hard/soft distinction (Def. 2.13) is exactly "is `sigma`
  zero," orthogonal to the coefficients.
- `role` is independent of all of the above: the *same* linear form `Σ a_in − Σ a_out`
  is a hard `equality` (mass balance) or, with a nonzero `sigma` and a data `value`,
  a soft `equality` (an anchored total).

**Completeness:** §7 shows every existing stage computation is a set of `EdgeRelation`s.

---

## 4. Why mask == group (the reduction argument)

The user's four candidate primitives — *mask, group, coefficient, value* — are not four
things. **Mask and group are one mechanism** (a selector) differing only in the
*reduction operator* applied per dimension. AISHA encoded this directly in its
meta-script addressing operators (AISHA §3.6.1–3.6.2):

| AISHA operator | Meaning | pySUTO concept | Effect on rows |
|----------------|---------|----------------|----------------|
| `a:b`          | **loop** indices a…b | **group-by** a dimension | one relation per value |
| `a-b`          | **sum** over a…b | **mask / aggregate** a dimension | collapse to one row |
| `[a;b;c]`      | loop over disjoint set | group-by an explicit set | one per listed value |
| `[a,b,c]`      | sum over disjoint set | mask over an explicit set | collapse listed values |
| `~`            | **correlation** (tie two dims) | `region_binding` / `time_binding` (diagonal) | restrict to the diagonal |

So:
- [masks.py](../../src/pysuto/masks.py) `MaskRuleSet.resolve()` is the *sum/keep-all*
  projection of a selector — a boolean `(n_edges,)` array.
- `_build_group_ids(keys)` in [impute/flat_prior.py](../../src/pysuto/impute/flat_prior.py)
  and `groupby(group_keys)` in [constraints/sum.py](../../src/pysuto/constraints/sum.py)
  are the *loop* projection of a selector — dense group ids.
- `SourceTerm.region_binding`/`time_binding`
  ([impute/formula.py](../../src/pysuto/impute/formula.py) lines 60–68) is *exactly*
  AISHA's `~` correlation operator (e.g. `from_matches_target_from` ≡ tie source
  `region_from` to target `region_from`).

These are three faces of one object. Today they live in three modules with three data
shapes; an `EdgeRelation` selector with a per-dimension `reduction` map is their common
generalisation.

---

## 5. AISHA as prior art

AISHA's **meta-command** (AISHA §3.7, Tables 5–7) is a single declarative row:

```
label | on/off | #parts | role(I/…) | pre-map | post-map | S.E.-formula | coef | <8-D address>
```

Map to `EdgeRelation`:
- **`<8-D address>`** (cols 12–19, using `:` `-` `;` `,` `~`) → `selector` + `reduction`
- **`coef`** → `coefficients`
- **value / data filename** → `value`
- **`S.E.`-formula** (`CNx`, `LNx`, `MXxMNy`, …; AISHA §3.6.4) → `sigma`
- **role label** (`I` = initial estimate, balance, bound, …) → `role`
- **`#parts`** — a *compound* constraint overlays several parts addressing the same
  number of elements → this is precisely how a **ratio** (`a_i − r·a_j = 0`) is two
  coefficient-terms over two selectors, i.e. a multi-term `EdgeRelation`.

AISHA's **hard/soft** rule (Def. 2.13: `gᵀ` is *hard* iff `σ_c = 0`, else *soft*) is
pySUTO's exact rule — hard rows go to the Zero cone, soft rows carry `ε` with
`Σ_c⁻¹` penalty ([constraints/](../../src/pysuto/constraints/),
[solver.py](../../src/pysuto/solver.py)).

pySUTO already has AISHA's σ-formula meta-language, narrowed to three modes, in
`sigma_from_row` ([param_relation.py:92](../../src/pysuto/param_relation.py)): `sd`,
`min_max` (= `(max−min)/4`), `pedigree` (Laner et al. → CV). That function *is* the
`sigma` compiler for the unit.

The upshot: AISHA already factored a complete IO-table model into a list of these
records. pySUTO has the same factors but spread across five declaration formats (§3
table). The unit re-unifies them.

---

## 6. Proposed schema sketch

Illustrative only — names/shape to be refined. The point is it **reuses existing
resolution machinery** rather than inventing any.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Literal, Union

Role = Literal["prior", "imputation", "equality", "bound"]
Reduce = Literal["loop", "sum", "bind"]   # group-by | mask/aggregate | correlation(~)

@dataclass
class ValueSpec:
    """Where the RHS/prior value comes from."""
    kind: Literal["const", "param", "data", "computed"]
    const: float | None = None
    param_id: str | None = None          # resolved against a loaded params CSV

@dataclass
class SigmaSpec:
    """Reliability of the relation. Compiled via param_relation.sigma_from_row + lp.lp_combine."""
    unc_type: Literal["sd", "min_max", "pedigree", "zero"] = "zero"  # zero ⇒ hard
    # carries value_sd / value_min,max / pedigree_* exactly as the params CSV does

@dataclass
class EdgeRelation:
    description: str
    role: Role
    # selector: either the A/B-field dict (CSV schema) OR a callable over ei_exp.
    selector: Union[dict, Callable]
    # reduction: per-dimension operator over the 8 address dims; default "sum" (mask).
    reduction: dict[str, Reduce] = field(default_factory=dict)
    # multi-term linear form: one coefficient per (sub)selector term.
    coefficients: list[float] = field(default_factory=lambda: [1.0])
    value: ValueSpec = field(default_factory=lambda: ValueSpec(kind="const", const=0.0))
    sigma: SigmaSpec = field(default_factory=SigmaSpec)
```

**Reuse, don't reinvent:**
- selector resolution → `param_relation.ParamRelationResolver` /
  `expand_edge_index` / `get_edge_cache`, and `masks.coerce_mask`.
- `bind` reduction → the existing `region_binding`/`time_binding` logic in
  `FormulaImputer`.
- `sigma` compilation → `param_relation.sigma_from_row` + `lp.lp_combine`.
- `value` `param` lookup → the same params CSV (`value_nominal`, `param_id`).

A stage "compiler" is then a pure function `list[EdgeRelation] → (matrix, c, σ, l, u)`
for that stage (`W`, `Ω`, or `G`+`c`+`σ_c`+bounds), reusing the sparse-assembly already
in each builder.

---

## 7. Stage-by-stage compile table (the central evidence)

Each existing computation, expressed as `EdgeRelation`(s). "—" = field unused for that
form.

| Stage / artifact | code | selector | reduction | coefficients | value | sigma | role |
|---|---|---|---|---|---|---|---|
| Concordance `W` (`f_root = Wᵀ f_src`) | [conc.py](../../src/pysuto/conc.py) | src↔root edge pairs | sum (fan-in) | weights | from `f_src` | CV-invariant fan-in (Lp) | prior-map |
| Aggregation `Ω` (`f_agg = Σ Ω·F`) | [agg.py](../../src/pysuto/agg.py) | per-edge × feeds | — (per edge) | `Ω = (1/σ²)` normalised | `F` (feed flows) | `lp_combine` | prior |
| Parametric imputation (`f_t = Σ α·f_s`) | [impute/parametric.py](../../src/pysuto/impute/parametric.py) | A/B-field pairs | sum | `α` (coeffs) | data / `0` | `sigma_from_row` + Lp | imputation |
| Formula imputation (`f_t = clip(Σ α·f_s)`) | [impute/formula.py](../../src/pysuto/impute/formula.py) | `target`+`terms` callables | sum + **bind** (`~`) | `α`/`ParamRef` | data / `0` | Lp (+`epsilon`) | imputation |
| Time imputation (interp/extrap) | [impute/time.py](../../src/pysuto/impute/time.py) | edge × time axis | loop over `time` | **computed** (interp weights) | neighbours | gap-inflation | imputation |
| Flat prior fallback | [impute/flat_prior.py](../../src/pysuto/impute/flat_prior.py) | per mask/group | loop (group) | — | `0` | `CV·scale` by coverage | prior |
| Mass balance (`Σ_in − Σ_out = 0`) | [constraints/mass_balance.py](../../src/pysuto/constraints/mass_balance.py) | node in/out edges | loop (node,t,val) + sum | `±1` | `0` | `0` (**hard**) | equality |
| Sum constraint (`Σ α·a = c + ε`) | [constraints/sum.py](../../src/pysuto/constraints/sum.py) | group/mask | loop (group) + sum | `α` | `c` (data) | user (**soft**) | equality |
| Ratio (`a_i − r·a_j = ε`) | [constraints/ratio.py](../../src/pysuto/constraints/ratio.py) | A,B pair (2 parts) | — | `1, −r` | `0` | params (**soft**) | equality |
| Ratio bound (`l·a_j ≤ a_i ≤ u·a_j`) | [constraints/ratio.py](../../src/pysuto/constraints/ratio.py) | A,B pair | — | `1, −l` / `1, −u` | `0` | `0` | bound |
| Flow bounds (`l ≤ a ≤ u`) | [constraints/flow_bound.py](../../src/pysuto/constraints/flow_bound.py) | per edge | — | — | `l, u` | — | bound |
| Solver QP | [solver.py](../../src/pysuto/solver.py) | (consumes `G`, `Σ_a`, `Σ_c`) | — | — | `a₀`, `c` | `Σ_a`, `Σ_c` | — (consumer) |
| Posterior | [uncprop.py](../../src/pysuto/uncprop.py) | (consumes `G`, `Σ_a`, `Σ_c`) | — | — | — | `Σ_{a*}` | — (consumer) |

Reading the table: **declaration layer** rows (concordance → bounds) are `EdgeRelation`
sets; **consumer layer** rows (solver, posterior) are pure functions of the compiled
`G`/`Σ`/`a₀`. Two rows to watch (see §10): `Ω` and `TimeImputer` have *computed*
coefficients (BLUE weights; interpolation weights) rather than declared ones — they fit
as `EdgeRelation`s whose `coefficients` are produced by a resolver, not authored.

---

## 8. The provenance-layer architecture (the user's idea, made precise)

The user's framing: *"everything must go through a standardized schema; we implicitly
log in class initialization all modelling choices and do not have to have strict
external data stored."*

Realised as a **provenance layer**, not a storage purge:

- **Source of truth = raw feeds + the relation log.** Every modelling choice — each
  imputer relation, each constraint spec, each mask, each parameter — is one
  `EdgeRelation`, **logged at class init** into a relation log that sits *alongside* the
  existing run registries ([registry/](../../src/pysuto/registry/)).
- **Computed intermediates (`agg.npz`, `G`, `c`, `σ`, `F_opt`, …) become a regenerable
  cache,** reproducible by replaying the relation log against the feeds + the `acc` tree
  structure. They are kept for speed and diffing — **this note does not propose deleting
  them** (explicit non-goal).
- **Input-side schema (the AISHA contract):** the relation log + the `acc`
  AccountingStructure (the "tree structure") together *are* the strict input schema —
  the direct analog of AISHA's *tree-structure file* + *meta-command sheet*. That is the
  "strict external data schema on the input side" the user is after: instead of each
  stage inventing its own declaration format (masks CSV vs params CSV vs builder kwargs
  vs imputer dataclasses), one schema feeds them all.

Why this is attractive here specifically:
- pySUTO **already logs** `source_edges` + `coefficients` per filled edge in
  `ImputationLog` ([impute/engine.py](../../src/pysuto/impute/engine.py)) and persists
  harmonised `parents`/`stage_config`/`stage_summary` sidecars per stage. The relation
  log generalises that already-present habit into one schema across all stages.
- It makes a run **fully auditable**: "why is edge X what it is?" resolves to the set of
  `EdgeRelation`s that selected it, in order, with their values and sigmas — across
  stages, not just within imputation.

---

## 9. Glossary

- **edge / 8-D address** — an atomic SUT flow, addressed by
  `(time, valuation, region_from, entity_from, sector_from, region_to, entity_to, sector_to)`.
- **selector** — a predicate (CSV A/B-fields or a callable) choosing an edge set.
- **reduction** — per-dimension operator: `loop` (group-by → many rows), `sum`
  (mask/aggregate → one row), `bind` (correlate two dimensions → diagonal). Subsumes
  *mask* and *group*.
- **coefficient** — a linear weight on a selected (term) edge.
- **value** — the relation's RHS/prior magnitude (const / param / data / computed).
- **sigma** — the relation's uncertainty; `σ = 0` ⇒ **hard**, `σ ≠ 0` ⇒ **soft**.
- **role** — `prior | imputation | equality | bound`: what the relation does and which
  stage/cone consumes it.
- **hard / soft** — AISHA Def. 2.13: a constraint is hard iff its σ is zero.
- **relation log** — the ordered list of `EdgeRelation`s logged at class init; the
  canonical record of modelling choices.
- **EdgeRelation** — the unit: `⟨selector, reduction, coefficients, value, sigma, role⟩`.

---

## 10. Open questions, risks, non-goals

- **Selector expressiveness: predicates vs CSV.** Callables (`FormulaRelation`) are more
  expressive than the A/B-field CSV; the unit must allow both. Open: is the A/B schema a
  complete *serialisation* of every callable currently in use, or only most?
- **Computed-coefficient relations.** Aggregation `Ω` (BLUE inverse-variance) and
  `TimeImputer` (interpolation weights) have coefficients that are *computed from data*,
  not authored. They fit as `EdgeRelation`s with a coefficient-*resolver*; the schema
  should mark "declared" vs "resolved" coefficients so the relation log stays honest.
- **Concordance `W`.** It maps a *different* index (source edges) into root edges, so its
  selector spans two spaces. Likely a distinct `prior-map` role rather than forcing it
  into the single-space mould.
- **Recompute cost of the cache stance.** Treating intermediates as regenerable assumes
  replay is cheap enough; for large `G`/solver runs it is not free. The provenance layer
  must coexist with kept caches (which is the chosen stance — see §8).
- **Non-goals.** No deletion of stored data; no new ADR or `GLOSSARY.md`; this is a lens +
  sketch, *not* a committed refactor.

---

## 11. Adoption roadmap (phased, non-binding)

- **Phase 0 — lens only (this doc).** Use `EdgeRelation` vocabulary in design
  discussion and code review; no code change.
- **Phase 1 — one declaration sheet.** Unify [masks.py](../../src/pysuto/masks.py)
  rules + the params CSV into a single relation sheet feeding *both* imputation and
  constraints (they already share `param_relation` resolution).
- **Phase 2 — imputers speak `EdgeRelation`.** Imputers emit/consume `EdgeRelation`s;
  low risk because they already log `source_edges` + `coefficients`.
- **Phase 3 — constraint builders compile from the sheet.** Builders become thin
  compilers `list[EdgeRelation] → ConstraintBlock`.
- **Phase 4 — relation log as canonical provenance** alongside the run registries; runs
  become replayable from feeds + log + `acc`.

---

### Provenance of this note

Grounded in: a full read of `src/pysuto/` (Stages 1–5), the existing
mask/group/coefficient/param/registry patterns, and the AISHA paper
(`docs/ref/Geschke et al. - AISHA …pdf`), in particular §1.2 + Table 3 (8-D hierarchy),
§3.6 (meta-script addressing operators), §3.6.4 (σ-formula language), §3.7 (meta-command
sheet), and Def. 2.13 (hard/soft constraints).
