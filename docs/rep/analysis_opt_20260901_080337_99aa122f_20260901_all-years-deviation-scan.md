# All-years cold solve on the soft-stock A+B const — most-deviating edges & constraints

**Opt:** `opt_20260901_080337_99aa122f` (cold, all 30 years 1995–2024) · **Const:** `const_20260901_072941_aa16dd34`
(A+B: `stock_bounds_mode=soft`, `pin_flat_prior_zeros=true`) · **Branch:** `sensitivity/no-flat-prior-zeros` · **Date:** 2026-09-01

## Headline

**All 30 years `Solved`** (cold, no warm drift; the fragile P≈0 feasibility pre-check was neutralised — every
year's real σ-weighted solve converged, dres ~1e-11). Deviations are concentrated and structural; the
soft-stock change we shipped is healthy — **MESC/USC net-stock rows never exceed |z|≈3.4**, only the
already-documented PSC tail pokes through. The dominant deviations are **pre-existing data-reconciliation
strains** (China ER supply/use balance, concentrate/ore ratios, ROW-anode anchoring, Chile mining scale), not
introduced by the stock work.

Scale: of 465,332 soft (free-cone) constraint rows, **98.3 % have |z|<10**; of 854,460 edges, **99.8 % have
|z|<10**. The tails:

## Most-deviating CONSTRAINTS (residual / σ_c = z)

| Family | variant | count \|z\|>10 | max \|z\| | where |
|---|---|---|---|---|
| **Total-out balances** | `sum_usetotout` / `sum_supplytotout` | **5,950** | ~59 | **China ER/CER** (552 rows \|z\|>50), Korea (Ore/Concentrate), ROW, Germany |
| **Concentrate/ore recipe ratios** | `conctailingsuse` (800), `concoresupply` (443), `concoreuse` (419) | ~1,660 | — | thin-ore regions |
| **ER efficiency ratio** | `effer` | 63 | **77.8** | Taiwan 2023 (single outlier; σ_c floored to 2.0) |
| **Net-stock PSC (soft)** | `netstockboundpsc*` | ~62 | **38.3** | France/Italy/Austria/UAE/Singapore |
| Net-stock MESC / USC (soft) | — | 0 | 3.4 / 2.6 | clean |

- **The single largest constraint strain is the total supply-out = use-out identity for China ER→CER**
  (residual ≈ −13 kt in 2024, and negative *every* year): China electrolytic refining is persistently
  ~1.7 % of throughput over-supplied vs downstream use, absorbed as soft slack. The z saturates near ~50–59
  because σ_c is a fixed fraction of throughput — so these are throughput-scaling **saturation**, not
  blow-ups. Korea (ore/concentrate) shows the same pattern.
- **Net-stock bounds behave as designed:** median |z|=0.064, p90=1.24. Only **PSC** has a tail
  (median 0.001, p90 9.97, max 38.3) on France/Italy/Austria/UAE/Singapore — all at the floored σ_c=3.06.
  This is exactly the documented PSC z-tail (no-data regions where the σ_c denominator floors + the cold
  run's throughput inflation); MESC/USC are clean. Not a regression.

## Most-deviating EDGES (posterior move / σ_a = z)

| Pattern | max \|z\| | reading |
|---|---|---|
| **Anode → ER, region ROW, Mass** (every year) | **82** | Solver multiplies a tiny prior (~2–17 t) by ~15× (to 25–240 t) against a tiny σ_a — ROW is a residual region with a near-zero, badly-anchored prior. Magnitude-anchoring issue, not a data conflict. |
| **CO → Mining / Mining → Ore, Portugal 2024** | 46 | small-producer scale correction |
| Concentrate / Scrap / Matte / Cathode source edges | 11–15 | recipe-ratio reconciliation, ~564/245/155/149 rows \|z\|>10 |

**By absolute mass (not z), the Chile mining chain dominates:** `CO→Mining` and `Mining→Ore` for **Chile**
are cut by **560–660 kt/yr** (a_prior ~730–780 kt → a_post ~120–170 kt, ≈−75 %), consistently across all
years — the largest single reconciliation in the system by tonnage. Worth a look on its own (relates to the
mining mass-collapse thread), but it is a scale correction the solver applies smoothly, not an infeasibility.

## System-boundary in/outflows (SD-R-10)

`diag/sd_r_10_boundary_flows.png` — Σa* over system-boundary edges, 2×2 (rows: Mass / CC; columns: inputs
*from SB* / outputs *to SB* incl. final demand); dashed = a₀ 1995 prior baseline. Rendered post-hoc (not in
`run_solver.py`'s default plot subset). Reading:

- **Boundary trajectories are smooth and physically ordered** — no spikes, no 1995 discontinuity. The cold run
  gives stable boundary flows across all 30 years.
- **Inputs:** CO (concentrate/ore feed) rises ~1.1→2.4 Mt; Scrap volatile 30–150 kt — both above their 1995
  baseline (the throughput inflation the deviation tails trace back to).
- **Outputs (Mass):** Tailings dominates (~1 Mt, rising), then WR, then FU/Slag; **MESC/USC (the net-stock
  indicators) are small (~10²–10³ t) and stable, no collapse or spike** — the soft-stock priors are behaving.
- **CC pane:** FU carries the copper (~10⁵ t Cu); WR sits at ~10⁻¹¹ (waste rock is barren of Cu — correct).

Consistent with the deviation scan: throughput sits above the 1995 prior (magnitude-anchoring), while the
net-stock outflows we reworked are calm.

## Takeaways / follow-ups

1. **Soft-stock work confirmed good across all years** — feasible, MESC/USC well-calibrated, no collapse.
2. **China ER→CER (and Korea ore) supply/use imbalance** is the top structural strain — a persistent ~1.7 %
   throughput gap the soft `sum` rows absorb. A data/recipe question, not a solver one.
3. **ROW Anode→ER anchoring** and **Chile mining scale** are the biggest edge moves — both magnitude-anchoring
   (poor priors on residual/large regions), same root as the documented PSC tail.

## Sources

- Diag CSVs: `data/proc/opt/opt_20260901_080337_99aa122f/diag/{edges,constraints}_YYYY.csv` (all 30 years).
- Scan scripts: scratchpad `run_all_years.py`, `analyze_dev.py`.
