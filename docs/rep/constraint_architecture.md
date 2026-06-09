# Constraint Library — Architecture Comparison

## Context

pySUTO needs a constraint engine that produces the inputs for the linear optimiser:
- **G** (M × E, time-invariant sparse matrix)
- **c(t)**, **σ_c(t)** per timestep
- **u**, **l** bounds vectors
- **row_meta** audit log (constraint type, hard/soft, cone type per G row)

Two architectures were designed and compared. **Architecture A was selected.**

---

## Five Constraint Types

| # | Name | Form | Cone | σ_c |
|---|---|---|---|---|
| 1 | Soft sum | Σ coeff_i · a_i = c_k + ε | Free | fraction of node flow magnitude |
| 2 | Hard mass balance | Σ supply_i − Σ use_i = 0 | Zero | — |
| 3 | Soft ratio | a_i − r · a_j = ε | Free | from params CSV uncertainty modes |
| 4 | Hard ratio bound | a_i − r_u · a_j ≤ 0 and −a_i + r_l · a_j ≤ 0 | Non-neg | — |
| 5 | Flow bounds | l_i ≤ a_i ≤ u_i | Non-neg | from acc_struct.bounds |

---

## Architecture A — Plugin Engine ✅ SELECTED

Mirrors the existing `ImputationEngine` pattern exactly.

```python
class BaseConstraintBuilder(ABC):
    def build_static(self, acc_struct) -> ConstraintBlock: ...   # called once
    def build_dynamic(self, acc_struct, f_agg, sigma_agg, t) -> ConstraintBlock: ...  # per timestep

class ConstraintEngine:
    def register(self, builder: BaseConstraintBuilder): ...
    def run(self, agg_result, timesteps) -> ConstraintRun: ...

# Usage:
engine = ConstraintEngine(acc_struct)
engine.register(MassBalanceBuilder())
engine.register(SumConstraintBuilder.from_acc_struct(acc_struct))
engine.register(SumConstraintBuilder.from_csv("sum_constraints.csv", acc_struct))
engine.register(RatioConstraintBuilder.from_csv("params.csv", acc_struct))
engine.register(RatioBoundBuilder.from_csv("params.csv", acc_struct))
engine.register(FlowBoundsBuilder())
run = engine.run(agg_result, timesteps=list(range(2010, 2024)))
```

**Pros:** mirrors existing patterns, each builder independently testable, static/dynamic G split enforced at the interface, new constraint types = new class with no engine changes.

**Cons:** more boilerplate than Architecture B.

---

## Architecture B — Declarative Spec Registry ❌ NOT SELECTED

```python
class ConstraintRegistry:
    specs: list[ConstraintSpec]

    @classmethod
    def from_acc_struct(cls, acc_struct) -> "ConstraintRegistry": ...
    @classmethod
    def from_csv(cls, path, acc_struct) -> "ConstraintRegistry": ...
    def build(self, f_agg, sigma_agg, t) -> ConstraintBlock: ...
```

**Pros:** fewer classes, easy to serialize/inspect specs.

**Cons:** fixed type enum, all logic in one `build()` method (hard to test per type), doesn't mirror existing patterns, static/dynamic separation must be managed manually.

---

## Comparison

| Criterion | A: Plugin Engine | B: Declarative Registry |
|---|---|---|
| Consistency with existing code | ✅ Mirrors ImputationEngine | ❌ New pattern |
| Flexibility (new constraint types) | ✅ New class, no engine changes | ⚠️ Enum + switch change |
| Testability per type | ✅ Each builder unit-testable | ⚠️ All in one build() |
| Static/dynamic G separation | ✅ Enforced by interface | ⚠️ Manual inside build() |
| Code simplicity | ⚠️ More classes | ✅ Fewer classes |
| Inspectability | ⚠️ Need row_meta | ✅ Specs transparent |

---

## Sum Constraint CSV Schema (long format)

One row per term; `constraint_id` groups terms belonging to the same constraint.
The CSV `constraint_id` becomes the grammar **`variant`** slot and must be
**underscore-free** (`_` separates id slots) — a label with `_` raises
`ConstraintIdError`. The emitted row id is `sum_{variant}_{valuation}_{year}`.

| Column | Description |
|---|---|
| `constraint_id` | Groups all terms of one constraint; underscore-free (becomes the id `variant`) |
| `description` | Human-readable label (first row of each group) |
| `sector_from`, `entity_from`, `region_from` | Edge from-node selector (`*` = wildcard) |
| `sector_to`, `entity_to`, `region_to` | Edge to-node selector |
| `time`, `valuation` | Optional selectors (`*` = all) |
| `coefficient` | Float; sign encodes direction (+1 supply, −1 waste) |
| `c_k` | RHS target (default 0) |
| `sigma_ck_mode` | `sd` / `fraction_of_flow` / `pedigree` |
| `sigma_ck_value` | Absolute σ (mode=`sd`) |
| `sigma_ck_fraction` | Fraction of node flow magnitude (mode=`fraction_of_flow`) |

Ratio/bounds constraints reuse the existing params CSV schema from `ParametricImputer`.
A supplied `param_id` becomes the id `variant` (must be underscore-free); when absent
the variant defaults to `ratioeq` (soft ratio) or `bound` (ratio bound). Ratio ids
render both edges full from→to: `ratio_{variant}_{numEdge}_{denomEdge}_{numVal}_{denomVal}[_{dir}]_{year}`.

---

## Persistence

```
data/proc/const/{run_id}/
├── registry.json        ← acc_id, agg_id, params_hash, timestamp
├── G.npz                ← sparse (M × E), time-invariant
├── row_meta.parquet     ← uniform schema (naming.ROW_META_COLS): constraint_id,
│                          type ∈ {massbal,sum,ratio}, variant, is_hard, cone_type,
│                          time, valuation, sector/region/entity, ratio num_*/denom_*
├── a0.npy               ← f_agg at t=0
├── sigma_a.npy          ← sigma_agg
├── u.npy / l.npy        ← flow bounds
└── timesteps/t_{year}.npz ← c, sigma_c per timestep
```

**`constraint_id` grammar.** `{type}_{variant}_{specs}_{year}` (tokens joined by
`_`, every token underscore-free). It is a unique human-readable **display key** —
read structured fields from the `row_meta` columns, never split the id. The old
node/edge-id ids and the legacy `type` values (`sum_eq`/`ratio_eq`/`ratio_bound`/
`mass_balance`) are gone; **old const runs are not forward-compatible** with the
new diagnostics and must be re-run through Stage 3.
