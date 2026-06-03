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

| Column | Description |
|---|---|
| `constraint_id` | Groups all terms of one constraint |
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

---

## Persistence

```
data/proc/const/{run_id}/
├── registry.json        ← acc_id, agg_id, params_hash, timestamp
├── G.npz                ← sparse (M × E), time-invariant
├── row_meta.parquet     ← constraint_id, type, is_hard, cone_type per row
├── a0.npy               ← f_agg at t=0
├── sigma_a.npy          ← sigma_agg
├── u.npy / l.npy        ← flow bounds
└── timesteps/t_{year}.npz ← c, sigma_c per timestep
```
