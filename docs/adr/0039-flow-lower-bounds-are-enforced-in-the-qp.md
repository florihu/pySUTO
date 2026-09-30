# Flow lower bounds are enforced in the QP, and each year is solved to its numerical floor

**Status:** accepted — Round W.1, 2026-09-30 (run `opt_20260930_092111_e3e0677d`; REP
[`…roundW1-lower-bounds`](../rep/analysis_opt_20260930_092111_e3e0677d_20260930_roundW1-lower-bounds.md)).
Amends [ADR 0013](0013-no-data-flat-prior-anchor.md) amendment 1: the pin's boxes still reach only
the probe, the structural lower bounds no longer do. Issue #28.

Since the first solver (`8c0bf3f`) the per-edge bounds `[l, u]` of the accounting structure reached
only the feasibility probe — `FlowBoundsBuilder` emitted no rows — while the README, the solver
docstring and every REP described `l ≤ a ≤ u` as part of the QP, and residual mass edges served
as signed slack (1,108 mass edges of the mine chain more than 1 kt below zero on the shipped run,
149 after Rounds W.2 and W.3). We now emit one hard non-negative-cone row `−a_i ≤ 0` for every
edge whose *structural* lower bound is 0 (586,560 rows, 19,552 per year: every edge but the
signed stock changes Cathode→USC, Cathode→MESC and CP→PSC), read from `acc.bounds.l` and never
from the pinned `ConstraintRun.l`, so the zero pin stays probe-only and no-data stock changes stay
signed; upper bounds emit nothing. Measuring the round exposed a second defect the rows amplified:
every production run had stopped at Clarabel's relative gap of 1e-3, which is no accuracy on this
objective — its magnitude is the prior-only constant Σ(a₀/σ)² (1.17e6 in 2015 against a misfit
Σz² of 1.13e5), so the rule admits ≈ 1,000 objective units, and the dual cost at a stalled iterate
lies above the true optimum. Re-solving 2015 of the W.3 problem at 1e-8 lowered the objective by
9,052 (8 % of the misfit) and moved Chile's concentrate exports from 1,658 to 2,452 kt CC against
a prior of 2,522; with the bound rows the old setting stopped 48,800 short and produced a spurious
collapse (common-set z² +44.5 %, Chile export retention 0.303; `opt_20260929_134712_2d3dde19`).
The driver now runs each year to its numerical floor (gap_rel 1e-6, feas 1e-8, gap_abs 1e-2; the
floor accepted as AlmostSolved at 1e-4 / 1e-6 / 10, since the strict 1e-8 target does not converge
on the bound-row problem) and records the tolerances in the sidecar: 28 years Solved, 2
AlmostSolved, 41–69 s per year, common-set z² −11.2 % vs W.3 and mine deletion 24,884 → 3,325 kt
CC, 90.6 % of it forced. Rejected: keeping the bounds probe-only (a negative flow is not a
result); reading the pinned `l` (would forbid negative no-data stock changes); the old tolerance
(measured above). Consequences: every earlier run's retention and deletion figures carry the
termination error, so a comparison across this change mixes the model's effect with the solver's
(the base re-solved at the new tolerance, `opt_20260930_093914_523d57d7`, sits at −10.8 % on
the solver alone and W.1 a further −9.9 % below it); Stage 5's active-set fix (#27)
must drop slack bound rows as well as slack bands; conditioning the QP so that 1e-8 converges is
the open numerical item.
