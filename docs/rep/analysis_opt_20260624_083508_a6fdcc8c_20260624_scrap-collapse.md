# Scrap input collapse — `opt_20260624_083508_a6fdcc8c`

**Date:** 2026-06-24
**Start path:** `data/proc/opt/opt_20260624_083508_a6fdcc8c`
**Observation:** the scrap input total (≈7.2 kt in 1995, growing to ≈17.8 kt by 2024 in the prior `a0`) collapses to ~0 in the optimised `F*` across every year. The user asked whether adding scrap as a **soft sum constraint** in const init would fix it.

---

## 0. Executive summary

Scrap is not weakly anchored — it is **hard-infeasible at any positive value**, because a single
`Scrap/SB` edge is constrained by **two mutually-exclusive hard grade bands**:
`concscrapclass1supply` `[0.90, 0.98]` **and** `concscrapclass2supply` `[0.40, 0.60]`, both emitted
on the same `(Mass, CC)` pair. Their intersection is `∅`, so the only feasible flow is 0 — the
solver duly drives every one of the 5,640 scrap input edges to ~0 in every year. The prior carries a
real ~7–18 kt/yr scrap signal, all destroyed. The per-edge prior wants scrap at `a0` with a pull of
`(a0/σ_a)² ≈ 48`; that is irrelevant — **no soft term (prior, or the user's proposed soft sum) can
relax a hard infeasibility.** The fix is in the hard ratio-bound spec, not the constraint init (§5).

> **Correction (2026-06-24).** An earlier draft of this report claimed a *second*, systemic defect: an
> inverted `RatioBoundBuilder._emit` (commit `4fd60ef`) making every `conc`/`eff` band empty. **That was
> wrong** — it assumed the feasibility sign backwards. Hard `nonneg` rows are feasible at **`G·a ≤ c`**
> (Clarabel `NonnegativeConeT`, [solver.py:313](../../src/pysuto/solver.py#L313)); on disk **100 % of
> hard nonneg rows are satisfied** (max residual ≈ +2e-9), and the large *negative* residuals are slack,
> not violations (see CONTEXT.md §"Residual"). `_emit`, `eff`, and `stock_bounds` are correct — the
> `5abf8eb` net-stock fix at the generator/CSV layer (ADR 0005 amendment) confirms the convention. A
> scan of all 637,320 bounded edge-pairs finds the empty-band defect is **scrap and only scrap** (2,820
> pairs); no other `conc`/`eff`/`trade_shares`/`stock_bounds` band is contradictory.

---

## 1. Run chain

| stage | run id |
|---|---|
| opt | `opt_20260624_083508_a6fdcc8c` |
| const | `const_20260624_082019_faf4c868` |
| imp | `imp_20260624_055120_ad6b028d` |
| acc | `copper` |

`G` shape `(1,383,242 × 854,460)`; 5,640 scrap input edges (`Scrap/SB → Smelting|CFR`, Mass + CC valuations; 2,820 Mass + 2,820 CC).

---

## 2. The collapse (Step 1)

Per-year totals over the **2,820 scrap-Mass edges** (94 regions/yr). `a0` is real and growing; `F*` is ~0 in every year — **all 94 edges collapse, every year**.

| year | Σ a0 (t) | Σ F* (t) | n collapsed |
|---|---|---|---|
| 1995 | 7,246 | 1.3e-5 | 94/94 |
| 2000 | 7,135 | 1.4e-7 | 94/94 |
| 2010 | 12,216 | 1.5e-6 | 94/94 |
| 2020 | 14,791 | 2.5e-9 | 94/94 |
| 2024 | 17,762 | 1.1e-6 | 94/94 |

The prior run `opt_20260616_102226_a28a854e` shows the same edge at `F*≈3.4e-7` — the collapse is **longstanding** (as long as both scrap classes have shipped), not new to this run. "Previously 10⁴" = the `a0`/feed signal, which the optimiser has never respected.

---

## 3. Worked trace — edge 291598 (Step 2–4)

`Scrap/SB/China → Smelting/China`, valuation Mass, year 2024. CC twin = 291599.

| | a0 | σ_a | CV | F* |
|---|---|---|---|---|
| Mass (291598) | 4997.08 | 721.65 | **0.144 (tight)** | 1.15e-8 |
| CC (291599) | 2498.54 | 171.36 | 0.069 | 8.37e-9 |

`a0` grade = `CC/Mass = 0.500`. The seven rows touching the Mass edge:

| row | constraint | type | cone | hard | Mass coeff | CC coeff | σ_c |
|---|---|---|---|---|---|---|---|
| 38361 | `massbal_Smelting_China_Mass_2024` | massbal | zero | ✅ | +1.00 | — | 0 |
| 803460 | `concscrapclass2supply…upper` | ratio | nonneg | ✅ | −0.60 | +1.00 | 0 |
| 803461 | `concscrapclass2supply…lower` | ratio | nonneg | ✅ | +0.40 | −1.00 | 0 |
| 809100 | `concscrapclass1supply…upper` | ratio | nonneg | ✅ | −0.98 | +1.00 | 0 |
| 809101 | `concscrapclass1supply…lower` | ratio | nonneg | ✅ | +0.90 | −1.00 | 0 |
| 216331 | `concscrapclass2supply…` | ratio | free | ❌ | −0.50 | +1.00 | 125 |
| 219151 | `concscrapclass1supply…` | ratio | free | ❌ | −0.94 | +1.00 | 94 |

Decoding the four **hard** `nonneg` rows (feasible at `coeff·F ≤ 0`, the Clarabel convention) into
bounds on grade `g = CC/Mass`. The `bound_dir` labels are **correct**:

| row | `bound_dir` | inequality (`≤ 0`) | implied bound on g |
|---|---|---|---|
| 809100 | upper | `−0.98·Mass + CC ≤ 0` | `g ≤ 0.98` |
| 809101 | lower | `+0.90·Mass − CC ≤ 0` | `g ≥ 0.90` |
| 803460 | upper | `−0.60·Mass + CC ≤ 0` | `g ≤ 0.60` |
| 803461 | lower | `+0.40·Mass − CC ≤ 0` | `g ≥ 0.40` |

So class-1 gives `g ∈ [0.90, 0.98]` and class-2 gives `g ∈ [0.40, 0.60]`, both **hard** on the one
edge. Combined feasible set:

```
g ≥ max(0.90, 0.40) = 0.90   AND   g ≤ min(0.98, 0.60) = 0.60
→  0.90 ≤ g ≤ 0.60  →  ∅   →  only Mass = CC = 0 is feasible.
```

**Counterfactual** (set the edge back to `a0` ⇒ grade 0.50, hold others fixed): class-2 `[0.40,0.60]`
is satisfied (0.50 ✓), but the class-1 lower bound `g ≥ 0.90` is violated (row 809101 residual
`0.90·Mass − CC = +1999 > 0`). The two classes cannot both hold for any positive flow, so the solver's
only feasible move is scrap → 0. Prior penalty *saved* by `a0` = `(a0/σ_a)² ≈ 48`; irrelevant against a
hard infeasibility. **Verdict: dual-class hard-band contradiction forces the collapse**, not a weak
prior and not an inverted builder.

Mass balance (row 38361) is *not* the driver — with scrap free, other inputs rebalance. The **two
conc-supply hard bands on one edge are the killers.**

---

## 4. Root cause (Step 5)

### 4a. Two grade classes mapped onto one node

The two scrap conc params in [data/input/params/conc/current/params.csv](../../data/input/params/conc/current/params.csv) both target the **same** `Scrap/SB` root node (their comments say *"Mapped from 'Scrap_class_N' → root 'Scrap/SB'"*):

| param | value_min | value_max | source |
|---|---|---|---|
| `concscrapclass2supply` | 0.4 | 0.6 | Schlesinger et al. (2022, Ch.18) — mixed low-grade |
| `concscrapclass1supply` | 0.9 | 0.98 | high-grade |

`RatioBoundBuilder` emits each as a **hard** grade band on the one `(Mass, CC)` pair. Two disjoint hard bands on one edge ⇒ empty feasible set ⇒ flow pinned to 0. This is systemic: all **2,820** scrap-Mass edges (94 regions × Smelting+CFR × 30 yr) carry the same `class-1 ∩ class-2 = ∅`.

### 4b. The builder is correct — there is no inversion

An earlier draft blamed an inverted `RatioBoundBuilder._emit` (commit `4fd60ef`). **That was a misread of the cone sign.** Hard `nonneg` rows are feasible at `G·a ≤ c` (Clarabel `NonnegativeConeT`, [solver.py:313](../../src/pysuto/solver.py#L313)); decoded that way the `bound_dir` labels and bands are correct (§3). Empirics confirm it:

- **100 % of hard nonneg rows feasible** — max residual ≈ +2e-9 on disk; the 5,399 large-*negative* residuals are slack, not violations.
- The `eff` bounds are **correct**: `effsmelting` 2024 holds `0.95·Σin ≤ Matte ≤ 0.98·Σin`, and the solved `Matte/Σin ≈ 0.965` sits inside it.
- A scan of **all 637,320 bounded edge-pairs** finds an empty band on **exactly 2,820 — every one the scrap conflict**. No other `conc`/`eff`/`trade_shares`/`stock_bounds` band is contradictory.

So the cause is **solely** the dual-class mapping; `_emit`, `eff`, and `stock_bounds` (the latter fixed at the generator layer in `5abf8eb`, ADR 0005 amendment) are all sound.

---

## 5. Fix (implemented)

Scrap is **one** mixed-grade commodity (one `Scrap/SB` node; ADR 0005 single pool), so the two
classes are merged into a single `concscrapsupply` band `[0.40, 0.98]`, `value_nominal = 0.70`
(blend). The band contains the prior grade (0.50→now 0.70), so scrap is freed; the level is then set
by the eff balance + priors. Decision recorded in
[ADR 0012](../adr/0012-scrap-one-mixed-grade-commodity.md).

To prevent recurrence across **all** param classes, `ConstraintEngine.run` now rejects any edge-pair
whose hard ratio bounds intersect to ∅ (`_validate_ratio_bounds`), and `RatioBoundBuilder.from_csv`
rejects `value_min > value_max`. Glossary sharpened (CONTEXT.md §"Residual", §"Concentration
coefficient"). Alternatives considered and rejected: **split** scrap into two nodes (contradicts the
root node, no feed data); **soft-only** conc bounds (keeps two conflicting pulls); a **soft sum** on
scrap (the original idea — cannot relax a hard infeasibility, circular target, pins only the total).

**Verified (re-solve `opt_20260625_064556_c98db9b0`, conc `v4`, nominal 0.70):**

- Empty-band scan on the new const → **0 pairs** (was 2,820); the build-time guard passes on the
  merged params.
- Scrap is un-collapsed. The **anchored** scrap-Mass edges (a0 ≥ 0.1) land at their prior —
  **median `F*/a0 = 0.96`** (2024). Edge 291598 (China Smelting): `F* = 6,238 t`, grade 0.846 ∈
  `[0.4,0.98]`.
- Mass-balance residual `max|Gx−c| = 3.9e-8`; `pytest tests/test_constraints.py` 76 passed.
- Boundary totals (`sd_r_10_boundary_flows.png`): Scrap returns as a live SB *input* (~10⁴–10⁵ t/yr)
  vs ~0 before.

**Follow-up (separate, pre-existing — not this fix).** The **flat-prior** scrap edges (a0 = 0,
σ = 1e6; 57/94 in 2024) overshoot: they sum to ~82 kt, dominated by **Taiwan CFR ≈ 40 kt** and
**Panama CFR ≈ 40 kt** — regions with no scrap prior where scrap floats as a slack variable for the
CFR/Smelting eff balance (the band constrains *grade*, not *magnitude*). This is the "weakly-anchored
scrap" condition (§ coverage: 4,528/5,640 edges have no feed); previously masked by the empty-band
collapse. Candidate next step: a magnitude anchor on scrap (scale-total or tighter flat-prior σ),
tracked separately.

---

## 6. Evidence captured

- Per-year scrap totals (a0 vs F*, all 94 edges collapse every year) — ad-hoc numpy over the `t_*.npz` / `a0.npy` arrays below.
- Edge-291598 row decode + counterfactual: class-1 and class-2 hard bands disjoint ⇒ flow=0; prior pull ≈ 48 ≪ hard infeasibility.
- Cone convention confirmed from disk: 100 % of hard nonneg rows feasible (`G·a ≤ c`, max residual ≈ +2e-9); eff band `[0.95,0.98]` satisfied at 0.965.
- Full scan of 637,320 bounded edge-pairs ⇒ empty band on exactly 2,820, all scrap; no other param class affected.

## Sources

- `data/proc/opt/opt_20260624_083508_a6fdcc8c/timesteps/t_*.npz` — `F_opt`, `cols_t`, `residuals`
- `data/proc/const/const_20260624_082019_faf4c868/{G.npz, a0.npy, sigma_a.npy, row_meta.parquet, timesteps/t_*.npz}`
- `data/proc/acc/copper/{edge_index.csv, node_index.csv}`
- `data/input/params/conc/current/params.csv`
- `src/pysuto/constraints/ratio.py` — `RatioBoundBuilder.from_csv` / `build_static._emit`; `src/pysuto/solver.py:313` (cone)
- `src/pysuto/constraints/engine.py` — `_validate_ratio_bounds`
