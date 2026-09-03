# Round B: size-damped bilateral-leg σ (γ=0.5 ships) — Chile export deletion resolved

**Baseline:** `opt_20260903_105145_fa7049e3` (Round A). **Round-B chains** (ADR 0017,
`σ_e = cv_pedigree·a_ref(sector)^(1−γ)·a_e^γ`, per-sector median anchor, cv cap 1.0, damping
before the outlier screen):

| run | γ | chain |
|---|---|---|
| **`opt_20260903_124655_4b23df2f`** | **0.5 (ships)** | imp_20260903_122719_f13ea10b → const_…123225 (pruned, regenerable) |
| `opt_20260903_131430_4194ccf4` | 0.7 (sensitivity) | imp_…130041-era → const_…125716 (pruned, regenerable) |

Both 30/30 Solved, cold, 30 years. 40,844 legs damped (Anode 5,587 / Cathode 24,288 /
Concentrate 7,511 / Matte 3,458; a_ref medians 0.2 / 0.3 / 10.3 / 0.2 kt; pedigree cv median
0.209 → damped cv range 0.003–1.0). CC twins verified to inherit: trade_seed cv spread p10 0.054
/ p90 1.06 (was ~0.30 flat).

## Headline results (γ=0.5 vs Round-A baseline)

| metric | baseline | γ=0.5 | γ=0.7 |
|---|---|---|---|
| deleted share, smallest→largest size decile | 0.07→**0.38** | 0.15→**0.26** (flat) | 0.09→0.35 |
| Chile 2015: CO Mass / CO CC / conc exports | 0.22 / 0.35 / 0.16 | **0.31 / 0.50 / 0.51** | 0.24 / 0.39 / 0.22 |
| Chile 2019: same | 0.20 / 0.34 / 0.19 | 0.27 / 0.47 / 0.42 | 0.20 / 0.35 / 0.19 |
| world 2015 retention CO/Conc-CC/Smelt-CC/Cath-CC | 0.63/0.69/0.81/0.95 | 0.65/0.70/0.82/**0.99** | 0.64/0.70/0.81/0.96 |
| soft z p99 | 9.98 | 10.38 | 10.22 |
| total soft z² | 3.96M | 3.96M | 4.01M |
| Cathode_China max \|z\| | 11.7 | **18.1** | 12.4 |
| solver | 30/30 | 30/30 | 30/30 |

The design goal is met at γ=0.5: the deletion-by-size gradient is gone (the largest decile is no
longer the preferred dumping ground), Chile's export legs survive at scale for the first time in
the whole reconciliation series (0.51/0.42 vs 0.16–0.31 in every prior round), and the mining
chain follows (CO CC 0.50 — best to date; grade-basis era peaked at 0.42). γ=0.7 buys roughly
half of each effect; γ=0.5 ships.

## Where the tension relocated (as predicted)

Total z² is unchanged — the Mass–CC gap moved from *deleted legs* onto *scale totals*: sum-family
z² rose 1.68M→3.78M while edge-deviation and per-leg costs fell. The worst rows —
`sum_usetotout_Cathode_ROW_CC` ≈ z 70 and `Ore_Korea, Rep.` ≈ z −59 — **pre-exist in the
baseline** (67.6 / 58.7): they are fabricated/weak totals that trusted big legs push against, the
double-anchoring story of ADR 0016 surfacing at specific nodes, not a Round-B regression. The one
genuine cost is Cathode_China (11.7→18.1): China's observed cathode totals vs its now-trusted
cathode trade. Follow-up candidates: the ROW cathode usage total (residual-region ICSG usage is
weak — a targeted σ exception may be honest) and the standing lever-1 redesign (propagated σ).

## Also in this round

- Population fix discovered mid-run: Cathode's bilateral legs terminate in SB final demand, not a
  Process — the damping mask uses Flow-origin (any destination), matching the screen (24,288 of
  the 40,844 legs are Cathode).
- PNG screened legs still die at ~zero z; tailings family stays ~8.5k; no scrap-band violations.
- Disk housekeeping: `data/proc` reached 100 % of the root disk mid-round (const 27G + opt 12G
  accumulated since April). Pruned (registry-cleanly) only this session's regenerable
  intermediates: the rejected σ-policy const runs, the two Round-B const runs, and two
  diag-truncated opt dirs. **The shipped runs' const dirs are regenerable** from the committed
  scripts + their imp runs (`constraints.py --imp-run-id imp_20260903_122719_f13ea10b`). A
  retention policy for the ~39G of historical const/opt runs is a user decision.
