# Stage 4 owns the model; a QP backend solves a standard-form QP

**Status:** proposed (2026-10-08; triage of the modularity audit #49, follows #11). Accepted when the
copper re-solve of `const_20261008_075717_9ccd0d40` through the new layer reproduces
`opt_20261008_083810_56f905e9` within solver noise.

Stage 4 called Clarabel inline. Cone construction, settings, `solve`/`update`, the cold-restart retry and
the iteration telemetry were interleaved with the model's own mathematics: the σ weights and the
slack `ε` that realises a soft row. No other solver could take the problem, and diag read Clarabel's
console columns as if they were the stage's output (audit #49, Stage 4). We decide that the Stage-4
orchestrator owns the model. Per period it builds one standard-form QP over `x = [a; ε]`,
`min ½xᵀPx + qᵀx` subject to equality rows and `≤` rows, putting the σ weights into `P` and `q` and
realising every soft row as `G a − ε = c` in this one place. A **QP backend** solves that problem and
reports a neutral status: `solved`, `solved_inaccurate` or `failed`. The orchestrator accepts the first
two and raises on the third, quoting the backend's raw status. The backend owns what is specific to it:
- its settings, because tolerances do not transfer between solvers (ADR 0039 measured Clarabel's);
- its telemetry: the iteration history and the info snapshot;
- reuse of its factorisation across periods;
- the cold-restart retry that its own reuse makes necessary.

The driver injects the backend, so no data-layer module imports Clarabel, and reading a persisted opt
run no longer needs it installed. The feasibility probe becomes a per-period zero-objective QP solved
by the same backend, and the full-G legacy path is deleted. Rejected: handing the backend the semantic
period problem (`G`, `c`, `σ_c`, relation, hardness), which would make every backend re-implement the
model; and Clarabel's conic form, whose cone vocabulary #11 removes from the constraint contract.
Clarabel stays the only production backend, and a second backend is a separate, measured change.
