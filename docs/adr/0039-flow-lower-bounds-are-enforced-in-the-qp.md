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
signed (in the bound rows — the hard concentration band still forbade their negative half until
[ADR 0041](0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md)); upper bounds emit
nothing. Measuring the round exposed a second defect the rows amplified:
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

**Amended 2026-10-20 (#72, measured):**
1. **The numerical floor depends on the linear solver; faer is kept.** qdldl, Clarabel's other
   built-in LDLᵀ, solved the final chain's 30 period QPs 4.5× faster than faer (233.6 → 51.8 s on
   8 threads) and stopped at a different point near the floor: scored on the same QPs, Σ_t J
   +1,337.5 (+0.043 % vs `opt_20261019_171746_3272682b`), lower in 13 of 30 years, per-year ΔJ
   mean +44.6 and sd 134.9, so indistinguishable from faer at these tolerances. Rejected on
   measurement (run `opt_20261020_090246_99da4a39`; REP
   [`…issue72-qdldl`](../rep/analysis_opt_20261020_090246_99da4a39_20261020_issue72-qdldl.md)):
   no quality gain, Stage 4 only 745 → 563 s behind the 444-s probe, and every number of the
   reference chain would move. A comparison across a linear-solver change, like one across the
   tolerance change above, mixes the solver's floor into the model's effect.

**Amended 2026-10-21 (#33, measured):**
1. **The period QP is solved for the defects at 1e-8; the floor is gone.** Solved for the flows,
   Clarabel's cost is J − Σ(a_prior/σ_a)², a constant ten times the misfit, so its relative gap
   certified nothing: a read-only probe bounded each answer from below with the Lagrange dual and
   found the flows posteriors 945–4,363 above the optimum per year. Solved for d = a − a_prior
   (same P and A, q = 0, b = c − G·a_prior, cost = J) at feas = gap_rel = gap_abs = 1e-8, every
   year reaches Solved in 64–81 iterations within 0.2 of that bound. On the final chain (run
   `opt_20261020_112726_2f451f31`; REP
   [`…issue33-defects`](../rep/analysis_opt_20261020_112726_2f451f31_20261020_issue33-defects.md)):
   Σ_t J −66,702 (−2.13 % vs `opt_20261019_171746_3272682b`), lower in 30/30 years; common-set
   sum-row z² −5.43 %; world crude-ore Mass retention 0.864 → 0.998; Stage 4 745 → 604 s. Shipped
   over two items that fail at the numerical floor: equality-row residuals of 1.05–1.62e-6 kt in
   3 of 30 years, and Stage 5's bound-active count below 90 % of the base's in 2011 and 2012 — a
   fringe within 1e-4 kt of its bounds that the 1e-6 kt cutoff sorts either way (#99). The
   reduced tolerances stay as the AlmostSolved acceptance; no year needs them.
