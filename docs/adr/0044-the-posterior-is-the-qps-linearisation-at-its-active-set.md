# The posterior is the QP's linearisation at its active set: a binding lower bound pins its edge, and dependent hard rows are dropped, not softened

**Status:** proposed — 2026-10-14 (#27; built on `fix/uncprop-active-set`). Prototyped read-only on
2020 of the reference chain `opt_20261013_122432_24e811b7` (const `const_20261013_114216_1ad3239e`):
9,022 of 28,482 edges bound-active, 16,722 rows kept of 56,222, 107 of 5,918 kept hard rows
dependent, `S_t` 16,615² factored in 0.8 s and 19,460 columns solved in 72 s, every `σ_a* ≤ σ_a`.
The copper run that measures it is pending.

Stage 5's closed form `Σ* = Σ − Σ Gᵀ S⁻¹ G Σ` assumes equality constraints. Until #27 it gave every
hard row σ_c = 0, so the ADR 0039 lower bounds (19,552 a year) and the ratio bands (19,972) entered
as equalities. That made `S_t` singular on every copper chain, and wherever it did not fail it forced
`σ_a* = 0` on every bounded edge, binding or not. We linearise the QP where it actually stands
instead. A hard inequality row binds when `G·a* − c ≥ −bind_tol` (1e-6 kt). A binding lower bound
**pins** its edge: its prior variance goes to 0 in `S_t`, giving `σ_a* = 0` and a zero-width interval
at 0. This is the local sensitivity of the estimator that produced `a*`: under a small change in the
data, an edge held at its bound stays there. A binding band is a hard equality; slack rows drop out.
Intervals clip at the structural zero only, never at the `ConstraintRun`'s prior box or zero pin.

**Rejected: dropping the bounds altogether.** It would give an unconstrained `σ_a*` clipped to
`[0, z·σ_a*]` on deleted edges. That is more conservative, but it describes an estimator the QP is
not.

**Redundant hard equalities are dropped, not softened.** Closed balance loops and a band's two rows
at `0 = 0` are the cases. After a unit-diagonal scaling, a pivoted Cholesky of the hard block
(`dpstrf`, `rank_tol` 1e-10) drops the rows past the numerical rank, and the rest keep exactly zero
σ_c. **Rejected: a relative σ_c on hard rows.** The kept hard rows' prior variance spans 8.5e-11 to
1.2e12 in 2020, so one ε either gives the 135 flat-edge rows a σ_c of order 10 kt (at ε = 1e-10) or
wrecks the conditioning of `S_t`.
