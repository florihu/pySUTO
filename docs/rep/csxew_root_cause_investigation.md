# CSXEW infeasibility — pipeline trace and root cause

Follow-up to [analysis_optim_const_copper_20260429_070511_db39a7d4_20260511_084052.md](analysis_optim_const_copper_20260429_070511_db39a7d4_20260511_084052.md), which flagged the CSXEW `sum_eq` rows as the worst constraint family in the solve (mean |z| ≈ 30–43σ, ROW row max |z| ≈ 62σ).

Three candidate explanations were on the table:

- **A.** Raw data not properly read into `a₀`.
- **B.** Imputation gap → wrong / missing `f_agg` for CSXEW.
- **C.** Sum constraint built incorrectly (wrong `c` / `σ_c`).

After tracing one offending row (`sub_entity_node798_t2024`, i.e. **ROW · CSXEW · 2024**) all the way from the raw feed to `F_opt`, the conclusion is:

> **None of A, B, or C in isolation.** The root cause is a **structural valuation asymmetry**: the only feed that knows about CSXEW (`icsg`) reports **CC (Cu-content) only** — never **Mass**. The constraint engine builds the `sum_eq` correctly from those CC values, but the *Mass* side of the same physical flow has **zero feed coverage** across the entire SX/EW upstream chain. The tight Mass↔CC `ratio_eq` couples the two, and the *hard* mass-balance further up the chain forces the model to collapse both legs to ~1 % of the icsg-reported value.

This is in essence a **B-like data sparsity in one valuation**, surfacing as an A-like "prior=0" artefact, but the constraint and aggregation code are doing exactly what they were designed to do.

---

## 1 — Pipeline state for the worst-z CSXEW row

Worst CSXEW rows at t=2024 (all icsg-fed):

| region | c | σ_c | F_sum (solver) | z | upstream_cov |
|---|---|---|---|---|---|
| **ROW** | 22.86 | 0.365 | 0.21 | **−62.1** | 1 / 2 edges |
| Russia | 2.39 | 0.045 | 0.034 | −52.8 | 1 / 2 |
| Spain | 0.24 | 0.004 | 0.096 | −35.7 | 1 / 2 |
| Mongolia | 8.01 | 0.225 | 0.074 | −35.3 | 1 / 2 |
| Indonesia | 14.90 | 0.446 | 0.136 | −33.1 | 1 / 2 |
| Australia | 22.25 | 0.681 | 0.203 | −32.4 | 1 / 2 |
| Iran | 22.85 | 0.701 | 0.208 | −32.3 | 1 / 2 |
| Kazakhstan | 35.14 | 1.095 | 0.320 | −31.8 | 1 / 2 |
| China | 66.00 | 2.084 | 0.599 | −31.4 | 1 / 2 |
| Peru | 92.96 | 2.949 | 0.778 | −31.3 | 1 / 2 |
| Zambia | 111.91 | 3.556 | 0.873 | −31.2 | 1 / 2 |
| Mexico | 144.91 | 4.614 | 1.057 | −31.2 | 1 / 2 |
| **United States** | 460.40 | 14.73 | 3.05 | −31.0 | 1 / 2 |
| **Chile** | 1224.30 | 39.22 | 8.03 | −31.0 | 1 / 2 |
| **DR Congo** | 2559.03 | 82.02 | 16.59 | −31.0 | 1 / 2 |
| countries w/o icsg row (Sweden, Austria, …) | 0 | 5.49e11 | small | **0** | 0 / 2 |

Two observations:
1. **Every covered country has F_sum ≈ 0.6–3 % of c**, regardless of magnitude. The system collapses production by ~30× across the board → constant z ≈ −31 (because σ_c is also a fixed 3.2 % fraction of c).
2. **Uncovered countries have z = 0** — they sit on a "wide-prior" trivial sum_eq (`c=0`, `σ_c=5.5e11`).

---

## 2 — Tracing the ROW · 2024 case through every stage

### 2.1 Feed (raw) — icsg reports only one number

```
agg_result.npz  →  Sigma[icsg, edge=307379]  > 0        ← only on the CC edge
                   Sigma[baci, …] = Sigma[crude_ore, …] = 0
                   F[icsg, edge=307379] = 21.859        ← icsg production figure
```

That single number lands in the CC valuation of the **SXEW→CSXEW** edge (id=307379, from `SXEW · Process · ROW` → `CSXEW · Flow · ROW`). The Mass twin (id=307378) is *not* covered by any feed.

### 2.2 Aggregation — pre-pipeline coverage by sector × valuation

| sector | Mass cov % | CC cov % | Mass edges | CC edges |
|---|---|---|---|---|
| Mining | 20.4 % | 40.0 % | 4 230 | 4 230 |
| **Ore** | **0 %** | 20.0 % | 4 230 | 4 230 |
| Concentrate | 11.2 % | 59.4 % | 67 680 | 67 680 |
| **SXEW** | **0 %** | 18.4 % | 4 230 | 4 230 |
| **CSXEW** | **0 %** | 13.8 % | 2 820 | 2 820 |
| Smelting | 10.7 % | 58.2 % | 70 500 | 70 500 |
| Matte | 5.2 % | 54.2 % | 67 680 | 67 680 |
| Anode | 8.3 % | 40.3 % | 67 680 | 67 680 |
| ER | 8.2 % | 39.5 % | 69 090 | 69 090 |
| **CP** | **0 %** | 6.6 % | 5 640 | 5 640 |
| Cathode | 12.2 % | 33.4 % | 200 220 | 200 220 |

**The entire SX/EW chain (Ore → SXEW → CSXEW → CP) has 0 % Mass coverage.** This is *not* a bug — it's a feed-design fact: ICSG publishes copper-content tonnages, not gross-ore tonnages, and no other feed picks up the slack on the SX/EW route.

### 2.3 Prior (`a₀`, `σ_a`) — uncovered edges get a flat fallback

For uncovered edges (637 235 of them, 75 % of the network):
- `a₀ = 0` (median) — flat zero prior.
- `σ_a = 3.879 × 10¹¹` (essentially uninformative).

For the four edges of the ROW · 2024 CSXEW Flow node:

| col | from → to | val | a₀ | σ_a |
|---|---|---|---|---|
| 307378 | SXEW · ROW → CSXEW · ROW (Flow) | Mass | **0** | 3.88e11 |
| 307379 | SXEW · ROW → CSXEW · ROW (Flow) | CC | **21.859** | small (informative) |
| 451198 | CSXEW · ROW (Flow) → CP · ROW | Mass | **0** | 3.88e11 |
| 451199 | CSXEW · ROW (Flow) → CP · ROW | CC | **0** | 3.88e11 |

So **the data DID come through** (col 307379), but only on the single inflow CC edge; the prior is zero on the four sister edges.

### 2.4 Constraint engine — `c` and `σ_c` built correctly

`SumConstraintBuilder.build_dynamic` runs in Gaussian mode (`constraints.py:592–596`):

```python
elif mode == "gaussian":
    rhs = spec.get("_rhs_ids", [])
    if rhs:
        c_out[i]   = float(np.nansum(f_agg[rhs]))
        sigma_c[i] = float(np.sqrt(np.nansum(sigma_agg[rhs] ** 2)))
```

For the ROW · 2024 row, the **RHS supply ids** are the two inflow edges (cols 307378, 307379). The CC edge supplies 21.859 with a small σ; the Mass edge is uncovered (NaN treated as 0). Result:

- `c = 22.86`  (≈ 21.86 from CC + Mass fallback piece)
- `σ_c = 0.36`  (Gaussian-propagated from the icsg σ_agg)

**This is correct behaviour.** Option C is rejected.

### 2.5 The seven constraints that pin those four columns

For the ROW CSXEW Flow node at t=2024 the four columns `307378, 307379, 451198, 451199` appear in:

| row | type | constraint | coefficients | c | σ_c |
|---|---|---|---|---|---|
| 19 645 | **mass_balance (HARD)** | node 798 (CSXEW Flow ROW) | +307378 +307379 −451198 −451199 = 0 | 0 | 0 |
| 19 692 | **mass_balance (HARD)** | node 845 (CP Process ROW) | +448378 +448379 +451198 +451199 − 456778 − 456779 − 456838 − 456839 | 0 | 0 |
| 29 609 | sum_eq (soft) | node 845 sub-entity | … | 0 | 5.49e11 (no data) |
| **39 432** | **sum_eq (soft)** | node 798 (the offender) | +451198 +451199 = 22.86 | 22.86 | **0.365** |
| 265 079 | **ratio_eq (soft, tight)** | `conc_csxew_supply` | −0.9975·451198 + 1.0·451199 = 0 | 0 | **0.00125** |
| 926 368 | ratio_bound (hard) | upper | … | 0 | 0 |
| 926 369 | ratio_bound (hard) | lower | … | 0 | 0 |

The mechanics of the failure follow directly from this:

1. **HARD mass-balance @ node 798**: `(307378 + 307379) = (451198 + 451199)`.
2. **HARD mass-balance @ node 610** (SXEW Process upstream): forces Ore→SXEW Mass + Ore→SXEW CC inflow to equal the SXEW outflow.
3. **TIGHT ratio_eq** (`σ=1.25e-3`): `CC ≈ 0.9975 · Mass` on every CSXEW edge — locks the two valuations together.
4. **SOFT sum_eq** (`σ=0.365`): asks for `(451198+451199) = 22.86`.

The hard balance from (1) lets the solver push the outflow up if-and-only-if the inflow can be pushed up too. The inflow consists of one informed edge (307379, `a₀=21.86`, low σ) and one uninformed edge (307378, `a₀=0`, σ huge). The tight ratio (3) says the two inflow legs must be near-equal. **But upstream Mass is also entirely uncovered**: edges 14098, 14099 (Ore→SXEW for ROW · 2024) carry `a₀=0` and `a₀=1` with σ ≈ 3.88e11. The chain has nowhere to draw Mass from. So the *hard* upstream mass-balance pulls F_Mass to a small number, the *tight* ratio_eq drags F_CC down with it, and the *soft* sum_eq absorbs the leftover 22-tonne shortfall as a 62 σ violation.

In the solution we actually observe:
```
F_opt[14098] (Ore→SXEW Mass, a₀=0):    11.53     ← solver pushed it WAY up
F_opt[14099] (Ore→SXEW CC,   a₀=1):     0.105
F_opt[307378] (SXEW→CSXEW Mass):        0.105
F_opt[307379] (SXEW→CSXEW CC, a₀=21.86): 0.105   ← collapsed 200× below prior
F_opt[451198] (CSXEW→CP Mass):          0.105
F_opt[451199] (CSXEW→CP CC):            0.105
```

The solver split the discrepancy: the **upstream Ore→SXEW Mass** was inflated to ~11.5 t, but every downstream edge stayed near 0.1 t. The two are inconsistent with the *hard* mass-balance at node 610, which means there must be another outlet (likely a `Smelting/Anode` route from the SXEW process pulling the Mass out into a different chain that *does* have data). The icsg-reported value gets stranded.

---

## 3 — Verdict on the three candidates

| candidate | status | why |
|---|---|---|
| **A.** Raw data not properly read in | **Rejected** | The icsg value 21.859 appears as `a₀[307379]` *and* propagates into `c=22.86` of the sum_eq row. |
| **B.** Imputation issue | **Half true** | Aggregation is fine *per se*, but the **icsg feed is single-valuation** (CC only). The Mass twin gets the flat-prior fallback (`a₀=0`, σ uninformative) for every Ore / SXEW / CSXEW / CP edge. |
| **C.** Sum constraint not properly initialised | **Rejected** | Builder runs in Gaussian mode and produces `c = Σ f_agg[rhs]` and `σ_c = √Σ σ_agg²[rhs]`, exactly as designed. |

The actual root cause is structural:

> **Valuation asymmetry × hard-mass-balance × tight CC↔Mass coupling.**
> icsg feeds the CC leg only → the Mass leg gets a flat-zero prior → the tight `ratio_eq` couples them → the *hard* upstream mass-balance has no Mass to give → the *soft* sum_eq is the only outlet for the discrepancy.

The 1 / 2 "upstream_cov" pattern in §1 is the smoking-gun signature: every icsg country shows the same 1-of-2 coverage profile, and every one collapses to the same ~1 % F_sum / c.

---

## 4 — What to change

Three concrete fixes, in order of expected impact.

### 4.1 (Highest leverage) — Mass-side imputation for the SX/EW chain

For every CSXEW / SXEW / Ore / CP edge whose CC twin is covered by icsg, **synthesise a Mass prior from CC × (1 / Cu-content_ratio)** at the aggregation stage:

```
if Sigma[icsg, CC_edge] > 0 and Sigma[*, Mass_twin] == 0:
    F[Mass_twin]    = F[CC_edge] / 0.9975          # SXEW cathode is ~99.75% Cu
    Sigma[Mass_twin] = Sigma[CC_edge] / 0.9975 + 1% σ from ratio uncertainty
```

This breaks the asymmetry: when the solver enters the optimisation, both valuations carry the same observed flow (within the ratio uncertainty), and the hard mass-balance has the Mass it needs to honour the icsg target.

This is essentially the inverse of the `ratio_eq conc_csxew_supply` constraint, but applied at the imputation stage so the *prior* is consistent, not just the constraint.

### 4.2 — Make the icsg-fed `sum_eq` honestly less certain

The Gaussian propagation of `σ_c` reflects only the *measurement* uncertainty on the supply value, not the *structural* uncertainty (how much of the supply ICSG missed, how the country aggregates SX/EW vs. concentrate, etc.). For a feed of this kind the realistic σ is probably 10–20 % of `c`, not 3 %. Options:

- Add a floor: `σ_c = max(σ_propagated, 0.10 × |c|)`.
- Switch the SumConstraintBuilder for icsg-fed rows from `sigma_mode="gaussian"` to `"fraction_of_flow"` with `sigma_fraction=0.10` (the machinery already exists in `constraints.py:588–591`).

This alone would drop ROW-2024 z from 62 to ~6 — still binding but not pathological.

### 4.3 — Treat the all-zero-prior Mass column as warm-startable

The 75 % of edges with `a₀=0, σ_a=3.88e11` (the "fallback flat prior") are essentially `nan` in disguise. Two improvements:

- After the first cold-start solve, **substitute `a*(t=1995)` back into `a₀` for *all* timesteps**, not just t≥1996. Re-solving with this informed prior across 1995–2024 collapses the 1995 cold-start outlier (already 10× the 1996 distance, per [sd_r_05_prior_posterior.png](../data/proc/opt/optim_const_copper_20260429_070511_db39a7d4_20260511_084052/diag/sd_r_05_prior_posterior.png)).
- Add a regularisation term that penalises *Mass* edges from drifting from their corresponding *CC* edges scaled by the ratio constant — same logical content as `ratio_eq conc_csxew_supply`, but as a soft regulariser on `a` rather than a hard row.

---

## 5 — Suggested verification steps

1. **Confirm the diagnosis on one more region.** Run the same trace for `Russia · CSXEW · 2024` (z=−52.8) and `DR Congo · CSXEW · 2024` (z=−31.0). The expected pattern: `a₀` populated only on the CC inflow edge, Mass twins zero, upstream Ore Mass zero, F_opt collapsed.
2. **Spot-check `Smelting`, which has 58 % CC / 11 % Mass coverage.** It's the next-worst valuation gap. If it shows similar (smaller) z violations on its `sum_eq` rows, the same Mass-synthesis fix should apply.
3. **Try the σ_c floor (§4.2) first** — it's a one-line change in the SumConstraintBuilder and a no-cost test of whether structural noise alone explains the violations.

---

*Generated 2026-05-12. Trace files used:*
- *raw / agg: [data/proc/imp/copper_agg_imp_v0/agg_result.npz](../data/proc/imp/copper_agg_imp_v0/agg_result.npz)*
- *constraint priors: [data/proc/const/const_copper_20260429_070511_db39a7d4/](../data/proc/const/const_copper_20260429_070511_db39a7d4/) (`a0.npy`, `sigma_a.npy`, `row_meta.parquet`, `G.npz`, `timesteps/t_2024.npz`)*
- *solver output: [data/proc/opt/optim_const_copper_20260429_070511_db39a7d4_20260511_084052/timesteps/t_2024.npz](../data/proc/opt/optim_const_copper_20260429_070511_db39a7d4_20260511_084052/timesteps/t_2024.npz)*
- *index / metadata: [data/proc/acc/copper/node_index.csv](../data/proc/acc/copper/node_index.csv), [data/proc/acc/copper/edge_index.csv](../data/proc/acc/copper/edge_index.csv)*
- *builder code: [src/pysuto/constraints.py:340–438, 560–603](../src/pysuto/constraints.py)*
