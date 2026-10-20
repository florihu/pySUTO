# The posterior is the QP's linearisation at its active set: a binding lower bound pins its edge, and dependent hard rows are dropped, not softened

**Status:** accepted — 2026-10-20 on `uncprop_20261020_082104_b26ec19c` (final chain
`opt_20261019_171746_3272682b`; REP
[`…issue27-stage5-active-set`](../rep/analysis_uncprop_20261020_082104_b26ec19c_20261020_issue27-stage5-active-set.md)):
all 30 years complete with `σ_a* ≤ σ_a` on every edge-year; 245,943 edge-years bound-active at `bind_tol`
1e-6 kt (160,087 at 1e-8, 306,201 at 1e-4), carrying 10,417 kt; an LU of `S` on the same chain still puts κ
outside [0, 1] on 365 edges of 1995 and 279 of 2016. Proposed 2026-10-14 (#27, `fix/uncprop-active-set`) with the
condition that a copper uncprop run completes every year with `σ_a* ≤ σ_a` and a REP reports how the pinned set
moves with `bind_tol`. Read-only checks on the reference chain `opt_20261013_122432_24e811b7` (const
`const_20261013_114216_1ad3239e`):
- 2020: 9,022 of 28,482 edges bound-active, 16,640 of 56,222 rows kept, 82 hard rows dependent;
- 1995: 9,445 edges bound-active, 119 hard rows dependent;
- both counts are the same at `rank_tol` 1e-10 and 1e-12;
- on `uncprop_20261014_140949_fcf53674` an LU of `S_t` put κ outside [0, 1] on 331 edges of 1995 and 368 of 2016, and gave `σ_a* > σ_a` on 87 edge-years, all with `σ_a ≤ 0.065` kt; the QR that replaced it (d54cc2f) takes 65 s for 1995 and 119 s for 2007, the largest year.

Stage 5's closed form `Σ* = Σ − Σ Gᵀ S⁻¹ G Σ` assumes equality constraints. Until #27 it gave every hard row σ_c = 0, so the ADR 0039 lower bounds and the ratio bands entered as equalities. That made `S_t` singular on every copper chain and pinned every bounded edge, binding or not. We linearise the QP where it stands instead. A hard inequality row binds when `G·a* − c ≥ −bind_tol` (1e-6 kt). A binding lower bound pins its edge: its prior variance goes to 0, so `σ_a* = 0`. That is the local sensitivity of the estimator that produced `a*`. A binding band is a hard equality, and slack rows drop out. Intervals clip at the bound row's own `l`, not at the prior box or the zero pin. Hard rows linearly dependent on the others are found by a pivoted Cholesky of the unit-diagonal Gram of the rows over the live edges and dropped; the rest keep exactly zero σ_c. The Gram is unweighted, because a σ²-weighted one drops independent rows that differ only on low-σ edges (107 dropped in 2020 instead of 82, varying with the tolerance). Those rows are still parallel in `S` to working precision, so the contraction is read off a QR of the whitened system `[G·diag(σ_a), diag(σ_c)]`, never an LU of `S`, which squares its condition number. Two alternatives were rejected. Dropping the bounds altogether gives an unconstrained `σ_a*` clipped to `[0, z·σ_a*]`, which describes an estimator the QP is not. A relative σ_c on hard rows fails because the kept rows' prior variance spans 8.5e-11 to 1.2e12: any single ε either gives the flat-edge rows a σ_c of order 10 kt or wrecks the conditioning of `S_t`.
