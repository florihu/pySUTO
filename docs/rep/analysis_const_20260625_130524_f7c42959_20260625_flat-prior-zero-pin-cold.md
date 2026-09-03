# Cold sensitivity run — pin flat-prior zeros: infeasible at the net-stock hard bounds

**Run:** `const_20260625_130524_f7c42959` ("sensitivity: pin flat-prior zeros (cold)")
**Upstream imp:** `imp_20260625_063036_8491ea42` (scrap-merge-conc-0.7)
**Branch:** `sensitivity/no-flat-prior-zeros`  ·  **Date:** 2026-06-25

## What was run

Hard-pinned every genuine no-data ("flat-prior") leaf edge to `l = u = 0` in the flow
bounds — `pin_flat_prior_zeros=True`, exclusion set
`{CO, Mining, Ore, WR, Tailings, Slag, PSC}` (extraction chain + all four SEM residual
outputs) — and solved cold (`--prior-mode cold`, feasibility-check all 30 years). The soft
no-data σ anchor was disabled (`NODATA_ANCHOR_CV=0`) since the hard pin supersedes it.

**698,044 / 854,460 edges (82 %) pinned to [0,0].** Sanity-verified: all pinned edges
persist as `l == u == 0`; **zero** pinned edges touch the protected sectors; all four SEM
residual outputs have 0 pinned. The const build passed all checks (`u ≥ l` for all edges).

## Result: infeasible at the feasibility pre-check (1995 block)

The cold solve **halts at the first block** — `solve_all` feasibility pre-check, status
`MaxIterations`, treated as infeasible. The top hard-row offenders are **net-stock-change
ratio bounds** (`ratio_netstockboundmesc*`, MESC stock sinks — Netherlands, Singapore, …),
`|G·a − c| ≈ 1e15` (degenerate/empty interval). Mass balance is **never reached** — the
hard stock bounds break first.

## Root cause

The `stock_bounds` rows are **set-scoped hard ratio bounds** `Σ stock − r·Σ denom ≤ 0`
(`nonneg` cone), pooling a per-`(region, time, valuation)` bucket of stock edges against a
final-demand denominator (MESC/FU/Cathode). Sampled bucket
`netstockboundmescargentina_…_CC_upper_1995`: **89 of 94 edges pinned to [0,0]**, including
the bulk of both the stock (`b`-side) and the MESC denominator (`a`-side); the only survivor
with magnitude is one data-backed denom edge (`a₀ ≈ 23.8`). Pinning the no-data denominator
collapses `Σ denom`, shrinking the allowed stock-change band to a near-point; combined with
the paired lower bound (and whatever the few surviving data-backed edges demand), the hard
interval goes empty → infeasible.

So under the pin, the **first obstruction is not mass balance but the hard net-stock-change
bounds**, whose denominators the pin zeroes. This is consistent with these bounds' known
fragility (recent commits: "stock-bound a/b swap", "cathode collapse", ADR 0005 amendment).

## Notes

- **Diagnostic reporting quirk (separate, minor):** the `RuntimeError` from
  `_raise_infeasible` was raised for timestep `'1995'` but listed offender `constraint_id`s
  from 2004/2016. Under the block path the per-block solve and the global `row_meta` row
  indices appear misaligned in the offender lookup, so the *years* in the dump are
  unreliable (the *family* — net-stock MESC bounds — is correct). Worth a follow-up fix to
  `_raise_infeasible`'s row→cid mapping under `_solve_all_via_blocks`.

## Interpretation / next step

The probe answered a sharper question than "does the network balance": **the hard
net-stock-change bounds are load-bearing on the no-data slack** — they only stay feasible
because the no-data edges are free to populate their denominators. To test whether the
*core* data-backed network (mass balance + conc/eff ratios) is self-consistent under the
pin, the natural next iteration is to treat the fragile hard `stock_bounds` as part of the
slack being removed — drop/relax them for this sensitivity run and re-solve. That is an
experiment-scope decision (pending user direction), not applied here.
