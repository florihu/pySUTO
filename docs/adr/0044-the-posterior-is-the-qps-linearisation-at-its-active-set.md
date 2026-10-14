# The posterior is the QP's linearisation at its active set: a binding lower bound pins its edge, and dependent hard rows are dropped, not softened

**Status:** proposed — 2026-10-14 (#27, `fix/uncprop-active-set`). It is accepted when a copper
uncprop run completes every year with `σ_a* ≤ σ_a` and a REP reports how the pinned set moves with
`bind_tol`. Read-only checks on the reference chain `opt_20261013_122432_24e811b7` (const
`const_20261013_114216_1ad3239e`):
- 2020: 9,022 of 28,482 edges bound-active, 16,640 of 56,222 rows kept, 82 hard rows dependent;
- 1995: 9,445 edges bound-active, 119 hard rows dependent;
- both counts are the same at `rank_tol` 1e-10 and 1e-12;
- a 2020 prototype factored `S_t` in 0.8 s and solved every column in 72 s.

Stage 5's closed form `Σ* = Σ − Σ Gᵀ S⁻¹ G Σ` assumes equality constraints. Until #27 it gave every hard row σ_c = 0, so the ADR 0039 lower bounds and the ratio bands entered as equalities. That made `S_t` singular on every copper chain and pinned every bounded edge, binding or not. We linearise the QP where it stands instead. A hard inequality row binds when `G·a* − c ≥ −bind_tol` (1e-6 kt). A binding lower bound pins its edge: its prior variance goes to 0, so `σ_a* = 0`. That is the local sensitivity of the estimator that produced `a*`. A binding band is a hard equality, and slack rows drop out. Intervals clip at the bound row's own `l`, not at the prior box or the zero pin. Hard rows linearly dependent on the others are found by a pivoted Cholesky of the unit-diagonal Gram of the rows over the live edges and dropped; the rest keep exactly zero σ_c. The Gram is unweighted, because a σ²-weighted one drops independent rows that differ only on low-σ edges (107 dropped in 2020 instead of 82, varying with the tolerance). Two alternatives were rejected. Dropping the bounds altogether gives an unconstrained `σ_a*` clipped to `[0, z·σ_a*]`, which describes an estimator the QP is not. A relative σ_c on hard rows fails because the kept rows' prior variance spans 8.5e-11 to 1.2e12: any single ε either gives the flat-edge rows a σ_c of order 10 kt or wrecks the conditioning of `S_t`.
