# Scrap input collapse — `opt_20260624_083508_a6fdcc8c`

**Date:** 2026-06-24
**Start path:** `data/proc/opt/opt_20260624_083508_a6fdcc8c`
**Observation:** the scrap input total (≈7.2 kt in 1995, growing to ≈17.8 kt by 2024 in the prior `a0`) collapses to ~0 in the optimised `F*` across every year. The user asked whether adding scrap as a **soft sum constraint** in const init would fix it.

---

## 0. Executive summary

Scrap is not weakly anchored — it is **hard-infeasible at any positive value**. Two independent defects each pin every one of the 5,640 scrap input edges to exactly zero:

1. **Inverted ratio-bound direction (systemic).** Commit `4fd60ef` (2026-06-11, *"fixed icsg conc err + fan out"*) flipped the upper/lower sign convention in `RatioBoundBuilder._emit` ([ratio.py:550–553](../../src/pysuto/constraints/ratio.py#L550-L553)). Every hard `conc`/`eff` ratio bound now enforces the **empty band `[hi, lo]`** instead of `[lo, hi]`. For scrap-supply `conc` rows this means `grade ≥ conc_max` **and** `grade ≤ conc_min` simultaneously — satisfiable only at flow = 0.

2. **Dual-class over-constraint (scrap-specific).** Both `concscrapclass1supply` (grade band 0.90–0.98) **and** `concscrapclass2supply` (0.40–0.60) are emitted as hard bounds on the **same** `Scrap/SB` root edge. Even with defect #1 corrected, `[0.90,0.98] ∩ [0.40,0.60] = ∅`, so any positive scrap flow is still infeasible.

The per-edge prior wants scrap at `a0` with a pull of only `(a0/σ_a)² ≈ 48`; that is irrelevant — **no soft term (prior or a proposed soft sum) can relax a hard infeasibility.** The user's soft-sum proposal therefore **cannot** fix this. The fix must be in the hard ratio-bound construction (§5).

This is the same defect *family* as `5abf8eb` ("correct inverted net-stock-change ratio bounds (cathode collapse)"), which fixed the inversion in `stock_bounds_study.py`. The `RatioBoundBuilder` inversion governing `eff` + `conc` is still live and is consistent with the paused "mining mass-collapse" work.

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

The prior run `opt_20260616_102226_a28a854e` shows the same edge at `F*≈3.4e-7` — the collapse is **longstanding** (post-`4fd60ef`, 2026-06-11), not new to this run. "Previously 10⁴" = the `a0`/feed signal, which the optimiser has never respected.

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

Decoding the four **hard** `nonneg` rows (`coeff·F ≥ 0`) into bounds on grade `g = CC/Mass`:

| row | `bound_dir` | inequality | implied bound on g |
|---|---|---|---|
| 809100 | **upper** | `−0.98·Mass + CC ≥ 0` | `g ≥ 0.98`  ← a *lower* bound |
| 809101 | **lower** | `+0.90·Mass − CC ≥ 0` | `g ≤ 0.90`  ← an *upper* bound |
| 803460 | **upper** | `−0.60·Mass + CC ≥ 0` | `g ≥ 0.60` |
| 803461 | **lower** | `+0.40·Mass − CC ≥ 0` | `g ≤ 0.40` |

The `bound_dir` label is **inverted** relative to the inequality it produces. Combined feasible set:

```
g ≥ max(0.98, 0.60) = 0.98   AND   g ≤ min(0.90, 0.40) = 0.40
→  0.98 ≤ g ≤ 0.40  →  ∅   →  only Mass = CC = 0 is feasible.
```

**Counterfactual** (set the edge back to `a0`, hold others fixed): rows 38361, 809100, 803460, 803461 flip infeasible (residuals +4997, −4897, −2998, +1499). Prior penalty *saved* = 48; hard infeasibility *incurred* = unbounded. The solver's only feasible move is scrap → 0. **Verdict: hard-constraint forced collapse**, not a weak prior.

Mass balance (row 38361) is *not* the driver — with scrap free, other inputs rebalance. The **conc-supply hard bounds are the killers.**

---

## 4. Root cause (Step 5)

### 4a. Inversion source — `RatioBoundBuilder._emit`

`4fd60ef` changed both where `r_coeff` is applied **and** the upper/lower signs:

```python
# current (post-4fd60ef) — ratio.py:550-553
if not np.isnan(bu): _emit("upper", ei_sign= 1.0, ej_sign=-1.0, r_coeff=bu)
if not np.isnan(bl): _emit("lower", ei_sign=-1.0, ej_sign= 1.0, r_coeff=bl)
```

With `ej = a = Mass` (numerator), `ei = b = CC` (denominator) and conc value = grade = `CC/Mass = b/a`, this emits `g ≥ bu` (upper) and `g ≤ bl` (lower) — the band `[bu, bl]`, which is empty whenever `bl < bu` (always, by construction). The conc params at [data/input/params/conc/v1/params.csv](../../data/input/params/conc/v1/params.csv):

| param | value_min | value_max | source |
|---|---|---|---|
| `concscrapclass2supply` | 0.4 | 0.6 | Schlesinger et al. (2022, Ch.18) — mixed low-grade |
| `concscrapclass1supply` | 0.9 | 0.98 | high-grade |

### 4b. The inversion is systemic (eff too)

11,280 hard `conc`-scrap-supply rows (5,640 upper + 5,640 lower) follow the identical inverted pattern across all 94 regions. The **`eff` bounds are inverted identically**: `effsmelting` 2024 emits `Matte ≥ 0.98·Σinputs` (upper) **and** `Matte ≤ 0.95·Σinputs` (lower) → empty band `[0.98, 0.95]`. Difference in outcome:

- **eff outputs** (Matte, CSXEW, …) are pinned **nonzero** by downstream demand + the soft eff equation, so the contradictory hard bound is merely **violated** (e.g. Matte 386.8 vs 0.95·Σin ≈ 382 → residual ≈ +5). These are the hundreds-magnitude residual / huge-z hard rows seen in the first-pass scan.
- **scrap-supply conc** edges have **no downstream pull** (scrap is a secondary input, missing→0 per ADR 0004/0005), so the optimiser satisfies the contradiction the only feasible way: drive scrap to **exactly 0**.

So one inversion explains both the broad eff/stock "violations" and the total scrap collapse.

### 4c. Why the sign-revert alone is insufficient for scrap (defect #2)

Reverting to `upper:(ei=-1, ej=+1, r=bu)`, `lower:(ei=+1, ej=-1, r=bl)` (old signs, new `r_coeff` placement) yields the **correct** bands:

| class | current (empty) | fixed |
|---|---|---|
| class1 | `g ≥ 0.98 ∧ g ≤ 0.90` | `g ∈ [0.90, 0.98]` |
| class2 | `g ≥ 0.60 ∧ g ≤ 0.40` | `g ∈ [0.40, 0.60]` |

But both classes still bind the **same** edge as hard constraints → `[0.90,0.98] ∩ [0.40,0.60] = ∅`. `a0` grade 0.50 fits class2 only. **Scrap stays pinned to 0 until defect #2 is also resolved.** The params note both are *"Mapped … → root 'Scrap/SB'"* — two physical grade classes collapsed onto one root edge.

---

## 5. Fix strategies (Step 6) — ranked

> Investigation-only per the user's instruction; no code changed. The user picks.

| # | fix | P(fixes it) | complexity | blast radius |
|---|---|---|---|---|
| **1** | **Revert the `_emit` sign inversion** in `RatioBoundBuilder` ([ratio.py:550–553](../../src/pysuto/constraints/ratio.py#L550-L553)) to `upper:(ei=-1, ej=+1, r=bu)`, `lower:(ei=+1, ej=-1, r=bl)`. | High — necessary | 1-line | **Large but correct**: fixes *all* `eff`+`conc` hard bounds; should also relieve the eff/stock violations and the broader "mining mass-collapse". Re-solve required. |
| **2** | **De-conflict the scrap dual-class bound** (needed *in addition* to #1 for scrap to recover). Options, simplest first: (a) emit a single **union** band `[min(value_min), max(value_max)] = [0.40, 0.98]` per shared `Scrap/SB` edge; (b) make `conc…supply` bounds **soft** (free cone, σ from `min_max`) so they pull but don't pin; (c) **split** scrap into `class_1`/`class_2` edges. | High (with #1) | (a) small / (b) small / (c) large | (a)/(b) scrap edges only; (c) touches accounting structure. |
| 3 | **Soft sum constraint on scrap (user's original idea).** | **~0 alone** | medium | Cannot relax a hard infeasibility; target `c` is circular (4,528/5,640 edges have no feed); a sum pins only the total, not members. Documented for completeness — **not recommended.** |

**Recommended:** #1 + #2(a) or #2(b). Verify by re-solving and checking (i) Σ scrap-Mass `F*` returns to the `a0` order (~7–18 kt), (ii) `eff`/`conc` hard-row residuals drop from hundreds to ~0, (iii) no new collapses elsewhere.

---

## 6. Verification done

- Per-year scrap totals (a0 vs F*, all 94 edges collapse every year) — ad-hoc numpy over the `t_*.npz` / `a0.npy` arrays below.
- Edge-291598 row decode + counterfactual: 4 hard rows force flow=0; prior pull = 48 ≪ hard infeasibility.
- Systemic: 11,280 hard conc-supply rows, identical inversion; eff bounds inverted identically; eff outputs violated-not-collapsed (187/235 nonzero).
- Sign-revert algebra verified to restore `[lo,hi]`; dual-class intersection confirmed empty post-revert.

## Sources

- `data/proc/opt/opt_20260624_083508_a6fdcc8c/timesteps/t_*.npz` — `F_opt`, `cols_t`, `residuals`
- `data/proc/const/const_20260624_082019_faf4c868/{G.npz, a0.npy, sigma_a.npy, row_meta.parquet, timesteps/t_*.npz}`
- `data/proc/acc/copper/{edge_index.csv, node_index.csv}`
- `data/input/params/conc/v1/params.csv`
- `src/pysuto/constraints/ratio.py` — `RatioBoundBuilder.from_csv` (393–478), `build_static._emit` (524–553)
- commits `4fd60ef` (inversion introduced), `5abf8eb` (sibling fix in `stock_bounds_study.py`)
