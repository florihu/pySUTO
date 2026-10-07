# Waste rock is a satellite mass flow, fixed by a hard stripping relation

**Status:** accepted — Round W.2, 2026-09-29 (run `opt_20260929_115343_b1205921`; REP
[`…roundW2-waste-rock-satellite`](../rep/analysis_opt_20260929_115343_b1205921_20260929_roundW2-waste-rock-satellite.md)).
Amends [ADR 0005](0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md) and
[ADR 0015](0015-in-situ-ore-grades-on-allocated-ore-basis.md). Issue #28.

**Amended 2026-10-07 (run `opt_20261007_130833_1a9b7af7`; REP
[`…wr-prior-stripping`](../rep/analysis_opt_20261007_130833_1a9b7af7_20261007_wr-prior-stripping.md)):**
the prior carries waste rock; "the imputer leaves it to the flat prior" below no longer holds.
Left to the noisy zero, every prior-side view showed none (124,325 Mt
over 1995–2024 missing against 2,285–5,004 Mt/yr posterior), so the imputer now writes
`WR_Mass = k·Ore_Mass` from the same `stripping_ratio` row as the hard relation, at the flat σ
(10⁶ kt, exempt from the imputer's cv cap). The σ stays flat because waste rock has no data of its
own: under the hard row an informative WR prior would count the Ore prior twice. The solve does not
move beyond what the solver resolves — the objective shifts by ≤ 3.5 per year, almost uniformly —
and the WR priors now enter the no-data anchor's (Mining, Mass) sibling median, loosening the zero
anchor on 1,042 no-mine Ore/WR edges from σ 1,176 to 1,741 kt. k's band is an input (`value_min`/
`value_max` of the `stripping_ratio` row): v2 narrows it to 1.5–2.5, so "range 1–3" and "±25 %"
below hold for v1 only. WR touches no row but the stripping relation and its own bound, so
`WR*(k) = k·Ore*` exactly and the band is reported as a range of the flow (SI Table S12), never as
a σ (decided 2026-10-07; a regional k is #47).

The crude-ore feed measures ore (used extraction), so crude-ore mass equals ore mass by construction
and waste rock, unused extraction, lies outside every feed. With Mining→WR inside the Mining mass
balance the edge became the slack that absorbed the crude-ore/ore grade gap — negative in 225
region-years, world −257,667 kt on the shipped run — while ADR 0015 declared waste rock massless.
We take Mining→WR out of the Mining balance (acc `masks.csv` mask `massbal_scope`, so Mining balances
`CO = Ore`) and set its mass by a **hard** zero-cone relation `WR_Mass = k·Ore_Mass`, k from a new
params family `stripping_ratio` (global 2.0, range 1–3, Nassar et al. 2022); its copper content
stays zero, the imputer leaves it to the flat prior, and the reported PSUT shows Mining's gross-mass
outputs exceeding its input by exactly the waste rock (a satellite flow). A soft stripping row was
measured and rejected (W.2a, `opt_20260929_111551_21703f80`): an edge that only weak soft terms touch
is never pinned down by the interior-point stopping rules and settled at ≈ 1 × ore instead of 2 × ore,
even at a 1e-6 gap. Rejected alternatives: an explicit unused-extraction input (same result, a new
boundary sector and a Stage-6 split of domestic extraction); removing WR (loses the flow); computing
WR after the solve (a new derived-edge mechanism). k's ±25 % is reported beside the flow, not in the
posterior.
