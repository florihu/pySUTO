# Mass collapse via inverted hard ratio bounds — opt_20260611_075036_6d2e540f

Standalone investigation triggered by user observations on
[data/proc/opt/opt_20260611_075036_6d2e540f/](../../data/proc/opt/opt_20260611_075036_6d2e540f/)
(label: `fix imp + sup use tot`).

Upstream chain:
- const run: [const_20260611_073743_f1912321](../../data/proc/const/const_20260611_073743_f1912321/)
- imp run:   [imp_20260610_111400_d0c5b535](../../data/proc/imp/imp_20260610_111400_d0c5b535/)
- feeds:     icsg, crude_ore, baci

Config knobs of note:
- sum rows are *scale totals* (c = Σ f_agg per group, σ_c ≈ 1.7%·|c| in this run; `ANCHOR_CV` in `scripts/constraints.py`)
- all 30 timesteps solver_status = "Solved", gap_rel ≤ 1e-8 — the solver is healthy; the problem is what it was asked to solve

---

## 0 — Executive summary

The run solves cleanly but every one of the 22,146 soft `sum` scale-total rows sits at
z ≈ **−58.67 = −c/σ_c** in every year 1995–2024: the optimiser leaves `G·F* ≈ 0` against
totals of 10²–10⁶ t. The Mass layer of the network collapses to ~0 despite healthy priors
(the worst-row prior a₀ literally equals its total c). The cause is not prior
initialisation, imputation, or input data: **`RatioBoundBuilder` emitted every hard ratio
bound with numerator and denominator swapped**, so the conc* upper bounds read
`Mass ≤ CC` — ore/concentrate/matte tonnage constrained *below its own copper content*.
Combined with the quasi-hard soft grade ratios (CC ≈ 0.009·Mass for ore), the only
feasible point is Mass = CC = 0, and hard mass balances propagate the collapse
network-wide. Two further latent issues were characterised on the way: ratio-row σ_c is
dimensionally unscaled (ratio-space σ on tonnage rows → soft ratios are quasi-hard, and
the top ratio "violations" of z ≈ 42–44 are tonnage dust of 0.2–2 t), and the ICSG feed
broadcasts national totals as an even 1/N split across all 47 partner edges with the
total's cv stamped per edge (a fabricated uniform trade pattern asserted at cv 6.8%).

> **Verdict.** Constraint construction (user's class A): `RatioBoundBuilder._emit`
> ([ratio.py:475–502](../../src/pysuto/constraints/ratio.py)) attached the ±1 to the
> denominator edge and the bound coefficient to the numerator — inverted hard bounds
> (`Mass ≤ CC`) admit only the zero flow, which the solver dutifully returned.

Fixed in this session (coefficient swap + per-column regression test). **The re-solve
shows the fix is necessary but not sufficient (§7.2):** with corrected bounds the
collapse persists, because 276 quasi-hard soft ratio rows per year are mutually
inconsistent at the prior (|z| at a₀ up to 4.6e6) and F = 0 satisfies all of them
exactly — the unscaled ratio σ_c (H2) independently makes collapse *optimal*. H2 is
therefore upgraded to co-primary; the run cannot recover until §8.2 lands.

---

## 1 — Observations as given

1. "filter constraint and edges that have the highest z vals" — quantified in §3.
2. "is it an issue of A const initialization, B imputation, C basic data mismatch?" — §7.1.
3. "to what extent can scaling the ratio const help (log scale or is this only problem
   shift)?" — §8.2 / §8.5.
4. (interim, mid-investigation) "fan out issue in the smelting and CFR edges, for
   instance Kazakhstan and South Africa, where the prior a is equal all the time for all
   concentrate edges that supply the process — this is the issue behind the highest abs
   z scores" — confirmed as a real Stage-1 artefact but **rejected as the driver of the
   top z's** (§7, H4): the collapse is time-invariant and present in fan-out-free years.

Observations 1–3 are one system: the top z's, their cause, and the proposed remedy.

---

## 2 — Pipeline trace setup

| observation | owning stage | first place to look |
|---|---|---|
| sum rows at z=−58.67, all years | Stage 3 (constraint construction) | `G.npz` bound rows + solver cone convention |
| ratio rows at z≈42–44 | Stage 3 (σ_c units) | `row_meta` + `t_<year>.npz` σ_c |
| equal a₀ across partner edges (KZ/ZA) | Stage 1 (ICSG concordance W) | `agg_result.npz` per-feed F |

Loaded: `const_20260611_073743_f1912321/{G.npz, a0.npy, sigma_a.npy, row_meta.parquet,
timesteps/t_*.npz}`, `opt_20260611_075036_6d2e540f/timesteps/t_*.npz`,
`imp_20260610_111400_d0c5b535/{agg_result.npz, imputation_log.parquet}`,
`data/proc/acc/copper/{edge_index.csv, node_index.csv}`.

Note for future investigations: this opt run persists per-timestep *slices*
(`rows_t`, `cols_t`, `residuals` of length `rows_t`); the skill helper
`compute_z.py` assumes the older full-width layout and needs the `rows_t` join:
`z = residuals / sigma_c[rows_t]`.

---

## 3 — Data characterisation

### 3.1 |z| by constraint type (soft rows, all 30 years)

| type | n rows·yr | median | p90 | p99 | max |
|---|---:|---:|---:|---:|---:|
| ratio | 437,100 | 2.7e-8 | 0.35 | 14.7 | 44.4 |
| sum   | 22,146  | 58.45  | 58.67 | 58.67 | **58.673023** |

Every sum row is pinned at z = −c/σ_c (residual = −c exactly, i.e. G·F* ≈ 0). The max
58.673023 repeats to 6 decimals across years and regions — a ceiling, not a tail.

### 3.2 Ratio variants with mass violations (selected; n per variant over 30 y)

| variant | n | median \|z\| | max \|z\| | n(\|z\|>5) |
|---|---:|---:|---:|---:|
| concscrapclass2supply | 2,820 | 16.6 | 42.9 | 2,589 |
| effcfr                | 1,410 | 9.7  | 44.1 | 1,058 |
| concanodeuse          | 1,410 | 9.8  | 44.4 | 1,063 |
| effsmelting           | 1,410 | 8.2  | 19.0 | 1,384 |
| concmatteuse          | 1,410 | 7.8  | 18.2 | 1,381 |
| effer                 | 1,410 | 6.0  | 26.4 | 820  |
| conccathodeuse        | 1,410 | 4.4  | 19.5 | 623  |

### 3.3 Worst-N

Top 25 by |z| are **all** sum rows at z = −58.673, spanning 1996–2024 and Chile, Congo
D.R., United States, Peru — e.g. `sum_supplytotout_Mining_Chile_Mass_2017`
(c = 787,165.2, σ_c = 13,416.1, residual = −787,165.2). The worst *ratio* rows
(z ≈ 42–44) are `concanodeuse`/`concanodesupply` (Congo D.R. 2016/2018, Uzbekistan 2002,
Russia 1995), `effcfr` (same region-years), `concscrapclass2supply` (Malaysia, Singapore,
Congo D.R., Italy, France, Switzerland 2012/2016) — with residuals of only
**0.18–2.15 t** against σ_c of 0.004–0.05 (see §5 H2).

---

## 4 — Pipeline trace of one specific case

Worst case: `sum_supplytotout_Mining_Chile_Mass_2017` (row 43875) → its single column,
edge **3884** = Mining.Chile→Ore.Chile [Mass] 2017.

### 4.1–4.2 Feed / aggregation
a₀ = 787,165.23 = c exactly (the sum row is a scale total built from Σ f_agg — consistent
by construction). σ_a = 175,201 (cv 22%). The prior is healthy.

### 4.3 Constraints touching edge 3884

| row | constraint_id | type | cone | c | σ_c |
|---:|---|---|---|---:|---:|
| 28960 | massbal_massbal_Mining_Chile_Mass_2017 | massbal | zero (hard) | 0 | 0 |
| 29007 | massbal_massbal_Ore_Chile_Mass_2017 | massbal | zero (hard) | 0 | 0 |
| 43875 | sum_supplytotout_Mining_Chile_Mass_2017 | sum | free | 787,165 | 13,416 |
| 64708 | ratio_concoreuse_…_CC_Mass_2017 | ratio | free | 0 | **0.00214** |
| 497578 | ratio_strippingratio_…_Mass_Mass_2017 | ratio | free | 0 | 0.50 |
| 504890 | ratio_concoreuse_…_CC_Mass_**upper**_2017 | ratio | nonneg (hard) | 0 | 0 |
| 504891 | ratio_concoreuse_…_CC_Mass_**lower**_2017 | ratio | nonneg (hard) | 0 | 0 |

Row 504890 coefficients as persisted: **(+1.0 · Mass edge 3884, −1.0 · CC edge 3885)**.
Solver nonneg convention is `G·x ≤ c` ([solver.py:383–384](../../src/pysuto/solver.py)):
the row reads `Mass − CC ≤ 0`, i.e. **Mass ≤ CC** — ore tonnage at most its contained
copper. Row 504891 reads `−Mass ≤ 0` (the lower bound bl = 0 landed on the wrong edge;
redundant). The intended semantics per the builder's own docstring
([ratio.py:339–340](../../src/pysuto/constraints/ratio.py)) are
`CC ≤ bu·Mass`, `CC ≥ bl·Mass`.

### 4.4 Solver
F*₃₈₈₄ = 5.9e-08, F*₃₈₈₅ (CC) = 0.0017. With hard `Mass ≤ CC` and quasi-hard
`CC = 0.009·Mass ± 0.00214` (row 64708), Mass ≤ CC ≈ 0.009·Mass forces Mass ≈ 0.
Residuals: sum row −787,165 (z = −58.67); both hard rows feasible at the zero point.
The prior pull (z_prior ≈ −4.5) and the sum row (z = −58.67) are the prices the solver
pays; per-year objective ≈ 2.09e6 ≈ ½ Σ z² over the ~730 dropped sum rows — the
objective is dominated by abandoned scale totals.

The same inverted pattern holds for **all 29,328 bound rows per year** (every conc*
variant, both valuations' pairings); `git log -L` shows the swap predates the
bd2992b refactor (inverted since inception — the original loop already attached +1 to
the denominator list).

---

## 5 — Hypotheses (as ranked before testing)

### H1 — Hard ratio bounds are emitted inverted (constraint construction)
**Mechanism.** `RatioBoundBuilder._emit` (ratio.py:475–502) appends `ej_sign` (±1) to
the **denominator** list and `ei_sign·r_coeff` to the **numerator** — swapped vs the
docstring. Under `G·x ≤ 0` this pins `denom ≤ bu·num`.
**Predicted signature.** Collapse on every Mass edge that carries a conc* bound, in
every year; sum rows at exactly −1/cv; priors healthy.
**Falsified by.** Bound rows with the coefficient on the denominator column, or a
nonzero Mass solution coexisting with the bounds.

### H2 — Soft ratio σ_c is ratio-space, unscaled by the denominator
**Mechanism.** `RatioConstraintBuilder` writes σ_c = σ(ratio) (e.g. 0.00214) onto a
linearised row in tonnes (`num − r·denom = 0`); effective weight is ~10⁵–10⁶× too
strong; all ratio rows are homogeneous (c = 0).
**Predicted signature.** Top ratio |z| arise from sub-tonne residuals; z magnitude
tracks 1/σ_r, not physical violation size.
**Falsified by.** Top ratio rows having residuals comparable to flow magnitudes.

### H3 — Imputation produced broken priors (user's class B)
**Falsified by.** a₀ = c on the worst edge; imputation_log shows only parametric
Mass-twin derivation; collapse present where priors are perfect.

### H4 — ICSG fan-out broadcast drives the top z's (user's interim observation)
**Predicted signature if true.** Top z's concentrated in broadcast region-years
(KZ/ZA-like, pre-resolution years) and absent elsewhere.

---

## 6 — Test strategy

| H | experiment | expected if true | expected if false |
|---|---|---|---|
| H1 | read persisted bound-row coefficients + solver cone code | +1 on Mass, −bu on CC | coefficient on denom |
| H1 | fix swap, rebuild const, re-solve, recompute z | sum z leaves −58.67 ceiling | collapse persists |
| H2 | tabulate σ_c and residuals of top ratio rows | σ_c ∈ ratio units, residuals ≪ flows | σ_c ∝ flow scale |
| H3 | trace a₀/σ_a/imputation_log on worst edge | broken a₀ | a₀ = c (healthy) |
| H4 | top-z year/region distribution; per-feed F on broadcast edges | top z only in broadcast years | top z uniform across years |

---

## 7 — Test results

| H | result | evidence |
|---|---|---|
| H1 | **confirmed** | row 504890 = (+1·Mass, −1·CC), `G·x ≤ c` convention; collapse in all 30 years; **re-solve after fix: see §7.2** |
| H2 | **confirmed** | σ_c = 0.00214 (concoreuse), 0.0043 (effcfr), 0.020 (concanode\*), 0.050 (concscrapclass2supply), 0.50 (strippingratio); top ratio residuals 0.18–2.15 t on flows of 10²–10⁵ t |
| H3 | **rejected** | a₀ = 787,165.23 = c; σ_a cv 22%; imputation_log: only `parametric` Mass-twin rows (e.g. edge 159060 ← 1.9047·CC edge 159061, `concmattesupply`); 678k edges are `flat_prior_noisy_zero` by design |
| H4 | **rejected as primary, confirmed as artefact** | all 47 Matte→CFR.KZ edges carry identical ICSG values (5.341 in 1995; 47·5.341 ≈ 251 matches resolved-year totals → even 1/N split, total-conserving) at cv 6.8% per edge; but top sum-z's are uniform 1995–2024 incl. resolved years, and top ratio-z residuals are tonnage dust (H2), not fan-out scale |

Supporting facts for H4: BACI is not era-silent (≈1,200–1,500 observed edges/yr,
1995–2024, incl. 80–160 Matte→CFR edges/yr) but has **0 edges on exactly the broadcast
region-years**, so ICSG's national total is the only anchor there. ICSG switches between
partner-resolved (1–3 edges, e.g. KZ 2006/2007/2009, 2012+) and broadcast (47 edges,
even split) per region-year.

### 7.1 Verdict

The user's **class A** — with the precision that it is constraint *construction*, not
a₀/σ_a initialisation: H1 (inverted hard bounds) made the collapse the only *feasible*
point; H2 (unscaled ratio σ_c) makes it the *optimal* point even with correct bounds
(established by the §7.2 re-solve — H2 upgraded from secondary to co-primary). Class B
is rejected. Class C is real but tertiary (H4, a Stage-1 concordance artefact:
fabricated uniform partner pattern at unwarranted confidence). The defects are
independent; H1 *and* H2 are each fatal on their own, so both fixes are required.

### 7.2 Fix verification (re-solve)

Fix applied in this session: `RatioBoundBuilder._emit` now attaches `ej_sign·r_coeff`
to the denominator list and the bare `ei_sign` to the numerator, with call-site signs
flipped (`upper: ei_sign=+1, ej_sign=−1`; `lower: ei_sign=−1, ej_sign=+1`), restoring
`bl·denom ≤ num ≤ bu·denom`. The membership-only assertion in
`test_ratio_bound_inequality_direction` (tests/test_constraints.py) — which passed for
both the correct and the swapped placement because the value *sets* coincide — was
strengthened to per-column checks. Rebuilt: const_20260611_101432_4f78bbf0 →
re-solved (same configs).

Re-solve: opt_20260611_102234_3d776ba4 (all 30 timesteps `Solved`, obj ≈ 1.8–2.2e6/yr).
Before/after, soft rows over all 30 years (z recomputed via the rows_t join, per-year
`c`/`sigma_c` from `const/.../timesteps/t_<year>.npz`):

| metric | broken bounds (opt_…075036) | fixed bounds (opt_…102234) | pass criterion | result |
|---|---|---|---|---|
| sum rows median \|z\| | 58.457 | 58.615 | O(1–5) | **FAIL** |
| sum rows max \|z\| | 58.673 (= 1/cv ceiling) | 58.673 | leaves ceiling | **FAIL** |
| sum rows n at z ≤ −58 | 13,221 | 16,206 | ~0 | **FAIL** |
| ratio rows mean \|z\| | 0.518 | 0.473 | — | — |
| ratio rows max \|z\| | 44.367 | 58.525 | — | — |
| F\* edge 3884 (Mining.Chile→Ore.Chile Mass 2017) | 5.9e-08 | 3.8e-09 | ~787k scale | **FAIL** |
| solver status | Solved ×30 | Solved ×30 | Solved ×30 | pass |

**The collapse persists with corrected bounds.** The bound rows themselves were verified
correct on disk (Chile-2017 upper row now `+1·CC −1·Mass ≤ 0`, i.e. grade ≤ 100%), so
H1's mechanism is gone — collapse is no longer the *only feasible* point. It is still
the *optimal* point, and the reason is H2 acting alone. Objective decomposition for the
2017 slice (z² summed per term):

| term | at prior a₀ | at F\* (collapsed) |
|---|---|---|
| soft ratio rows | **2.54e13** | 7.6e4 |
| soft sum rows | 0 (c built from same f_agg → consistent) | 2.21e6 |
| prior term (F−a₀)/σ_a | 0 | 7.0e5 |

276 soft ratio rows (of 14,570 in the 2017 slice) are *mutually inconsistent at the
healthy prior*: real cross-feed residuals of 4e2–9e3 t divided by ratio-space σ_c ≈
10⁻³–10⁻² give \|z\| up to **4.59e6** (worst: `conccathodesupply Cathode.China→FU.China`,
residual 8,816 t / σ_c 0.00192; then `effcfr China` z = 1.5e6, `effsmelting China`
z = 9.9e5, `concwruse Mining.Chile` z = 3.7e5, `conccosupply` US/China/Congo D.R.).
Because every ratio row is homogeneous (c = 0), F = 0 satisfies all of them *exactly*;
at 10⁵–10⁶× the weight of sum/prior rows, the solver trades 22,146 sum rows at z = −58.67
(cost ≈ 2.5e6/yr) plus the prior term (0.7e6) to zero out a 2.5e13 ratio penalty. That is
the observed objective (~2.2e6/yr) in both runs — the tension never moved, the weights
never allowed the network to leave zero.

**Verification verdict: 8.1 is necessary but not sufficient.** H2 is upgraded from
secondary (z-cosmetics) to **co-primary**: the unscaled ratio σ_c independently forces
the collapse. The run cannot recover until 8.2 (σ_c scaling) — or 8.3 (hard ratios,
which would surface those 276 inconsistent rows as infeasibility, not relief) — lands.

Artefacts: `/tmp/all_z_fixed.parquet` (recomputed z, fixed run), diag at
`data/proc/opt/opt_20260611_102234_3d776ba4/diag/`.

### 7.3 Fix verification round 2 — 8.2 implemented (Run A) and ICSG fan policy (Run B)

Implemented this session (2026-06-11, second half): **8.2** symmetric σ_c scaling
(`RatioConstraintBuilder.build_dynamic`, σ_c = max(σ_r, 0.02)·max(Σ\|a₀,num\|,
\|r\|·Σ\|a₀,denom\|, 100 t)) and the **ICSG fan policy** (ADR 0008: qualifier leg
filter + no-1/N; plus the `Refining_Anode→Matte` values-chain fix, the `'Srap'` typo,
and the two-stage CFR·ER efficiency divisor). Verification was staged:

- **Run A** — 8.1+8.2 on the *old* imp run `imp_20260610_111400_d0c5b535`
  (isolates the σ scaling): const `const_20260611_125202_2b3c73cd`,
  opt `opt_20260611_140040_0ec71443`.
- **Run B** — 8.1+8.2 + new ICSG feed (`feed_icsg_20260611_131027_a7c03949`,
  trade_shares v4): imp `imp_20260611_131319_928c86ee`,
  const `const_20260611_131746_485f4d5f`.

| metric (active soft rows, 30 yr) | 8.1 only (§7.2) | Run A (8.1+8.2) | Run B (+ICSG) |
|---|---|---|---|
| sum rows: median \|z\| | 58.62 | **6.79** | **3.38** |
| sum rows: n at z ≤ −58 | 16,206 / 22,146 | **2,646** / 22,146 | **1,001** / 17,480 |
| ratio rows: p99 \|z\| | 10.6 | 13.6 | 8.2 |
| ratio rows: max \|z\| | 58.5 (σ-artifact) | **44.1** (≤ 1/σ_r = 50 ceiling) | 47.9 |
| F\*[edge 3884] @2017 (a₀ = 787k) | 3.8e-09 | **1.5e4** | 1.5e4 |
| solver | 30/30 Solved | 29 Solved + 1 AlmostSolved | 28 Solved + 2 AlmostSolved |

(Run B has fewer sum rows because the undistributed fans no longer fabricate
`f_agg` anchors — `from_mask_anchor` drops the \|c\|≈0 groups.)

**Run A verdict: the global collapse is broken.** The sum-row median leaves the
−58.67 ceiling (→ −3.6), 84 % of the pinned rows release, objectives drop from
1.8–2.2e6/yr (collapse cost) to 0.3–9.7e5/yr, and ratio rows respect the designed
z-ceiling. **A sub-population stays suppressed**: the 3,464 rows still at z ≤ −55
concentrate in the refined-cathode chain in CC valuation (Cathode 1,236, CP 432,
CER/ER 758, CSXEW/SXEW 662) — precisely the families fed by the ICSG cathode fans
and the (now-fixed) `Refining_*` mappings, i.e. the data Run B replaces. Edge 3884
(Mining.Chile→Ore.Chile) moves off zero but only to 1.5e4 of 787k — upstream of the
same chain.

**Run B verdict: the feed fixes halve the residual again** (pinned z ≤ −58:
2,646 → 1,001; median \|z\| 6.79 → 3.38; ratio p99 13.6 → 8.2). What remains is a
*third, distinct* signal, untouched by either fix: full-period (all-30-year) pins of
the **Cathode supply/use totals in the major refining countries** (Russia, Chile,
US, Germany, Japan: 54–60 rows each, CC valuation) plus the Chile
Mining→Ore→MF chain that holds edge 3884 at 2 % of its prior. This is no longer a
weighting or fan-out artifact — it points at a real structural/data conflict in the
refined-cathode balance (candidates: CP pooling closure, `conccathodesupply`
twins vs ICSG refined-production totals, CSXEW/SXEW split). **Next investigation,
out of this session's scope.**

**Numerical note.** The scaled-σ_c QP is harder than the collapsed one (weight
spread ~6e17 across soft rows): with Clarabel's 1e-8 defaults several years end
`InsufficientProgress` with the iterate at rel gap 1e-4–5e-3 and dual residual
1e-8–7e-7 (μ at the 1e-15 floor). Three driver/solver changes (this session):
`tol_feas=1e-7, tol_gap_rel=1e-3` main tolerances, reduced-accuracy acceptance
(`reduced_tol_feas=1e-5, reduced_tol_gap_rel=1e-2` → floored iterates return
AlmostSolved), and a cold-restart retry in `_solve_all_via_blocks` (the reused
warm-start instance keeps the first year's equilibration; per-t σ_c now varies with
f_agg). Also raised `RATIO_SCALE_ABS_FLOOR` 1→100 t: at 1 t, the ~77 % of soft ratio
rows that are zero-prior parametric ties got σ_c = 0.02 t and the Hessian stalled the
solver outright (first attempt died at t=2005).

Artefacts: `/tmp/all_z_runA.parquet`, `/tmp/all_z_runB.parquet`.

---

## 8 — Proposed fixes

### 8.1 (Done this session) — Un-invert the hard ratio bounds
**What.** Coefficient swap in `RatioBoundBuilder._emit` (ratio.py) + per-column
regression test.
**Expected effect.** Sum rows leave the −58.67 ceiling; Mass layer recovers to control-
total scale.
**Risk.** Corrected bounds are *active constraints the model never actually felt* —
region-years whose data violate `bl ≤ num/denom ≤ bu` will now surface as new tension
(that is the intended behaviour).

### 8.2 — Scale soft-ratio σ_c **symmetrically** (answers observation 3)
**What.** In `RatioConstraintBuilder`,
σ_c(row) = σ_r · max(Σ|a₀(num edges)|, Σ|r·a₀(denom edges)|, floor), floor as a config
knob. The symmetric `max` (not denominator-only — see assessment below) is load-bearing.
**Expected effect.** Demotes ratio rows from quasi-hard (weight ~10⁵–10⁶× overstated)
to honest soft constraints. Quantified on the 2017 slice (fixed-bounds const run):
worst-row |z| at prior drops 4.6e6 → 1.2e3; all remaining large z's are capped at
1/σ_r (a zero-side row violates its ratio by 100% relative). Staying *exactly* at the
prior would still cost more than collapsing (Σz² = 5.4e7 vs 2.9e6), so decollapse
relies on the solver locally relieving each of the ~280 conflicting rows by moving the
zero/small side (cheap where that side is a wide-σ flat prior) — confirm by re-solve.
**Companion knob.** Several variants carry σ_r cv ≈ 0.1–0.2% (`conccathodesupply`
0.00192, `concwruse` 0.00105) — a relative-σ floor (e.g. 2% → z ceiling 50) belongs in
the same change.
**Risk / problem-shift caveat.** Scaling by a₀ inherits prior defects — on fan-out
edges (8.4) per-edge splits are fabricated, but bundle *totals* are right, so the `max`
over the row support is usable before 8.4 lands.

**Log-scale assessment (user question: soft rows as log xᵢ − log xⱼ = log r).**
The log-residual penalty ((log xᵢ − log xⱼ − log r)/σ_log)² is the *right* error model —
dimensionless, scale-invariant (collapse neither relieves nor tightens it, so the F = 0
degeneracy of homogeneous linear rows disappears), σ_log = σ_r/r is the native cv. But:
1. **Not Clarabel-expressible.** The squared log-residual is nonconvex in (xᵢ, xⱼ);
   substituting y = log x makes the hard mass balances log-sum-exp *equalities*
   (nonconvex); exponential-cone tricks fail on the two-sided soft penalty (the
   epigraph needs xᵢ ≤ e^{log r + t}·xⱼ — bilinear in (t, xⱼ)). Exact log needs an NLP
   solver (IPOPT) or sequential convex programming.
2. **Undefined on zero-side rows.** 139 of the 276 collapse-driving 2017 rows have
   numerator a₀ = 0 (e.g. `effcfr China`: num 0, denom 6.5 kt) — log gives an
   infinite/undefined residual exactly where the data conflicts live.
3. **Its Gauss–Newton linearization at the prior is this fix.** First order:
   (log xᵢ − log xⱼ − log r)/σ_log ≈ (xᵢ − r·xⱼ)/(σ_r·a₀_denom-scale) — i.e. the
   scaled-σ linear row *is* the log formulation, evaluated at a₀. Iterating the
   linearization point (re-solve 2–3× with σ_c rebuilt from F\* of the previous solve)
   converges toward the true log-ratio solution while staying inside the Clarabel QP.
So: adopt the log error model, implement it as symmetrically-scaled linear rows
(+ optional relinearization); do not attempt literal log rows. No σ rescaling of any
kind touches hard rows, so neither variant could ever have fixed H1.

### 8.3 (Experiment arm) — Hard ratio rows, no bounds
**What.** Make the soft conc/eff ratio rows hard (zero cone) at nominal r and drop the
bound rows entirely.
**Expected effect.** Removes the σ_c question; totals + priors pick the scale on the
ratio-consistent subspace.
**Risk.** Asserts r exactly where the params CSV gives ±10–25%: real ratio variation
is pushed into sum/prior z's (problem shift by construction), and any inconsistent
ratio cycle becomes solver *infeasibility* instead of a diagnosable z. Run as an
experiment, not the default.

### 8.3b (Rejected) — Box the solution at l = a₀ − σ_a
**What.** Hard per-edge box `a₀ − σ_a ≤ F ≤ a₀ + σ_a` (l/u arrays already exist in the
const persistence; currently l = 0 ∨ −1e6, u = 1e6–1e9). Would forbid F = 0 on the
150,611 edges with a₀ − σ_a > 0 (97.6 % of nonzero-prior edges, median half-width ±20 %).
**Why rejected — provably infeasible.** The priors do not satisfy the hard mass
balances within ±1σ: in 2017 alone, **79 of 1,316 hard massbal rows have
|G·a₀ − c| exceeding the total 1σ box slack Σ|g_ij|·σ_a(j)** (worst row needs 5.5×
the available slack; residuals up to 1.62e6 t) — and that slack already includes the
σ_a = 1e6 flat-prior edges. Clarabel would return primal-infeasible immediately.
**Why rejected even with a wider box.** (i) A box never removes the 2.5e13 ratio
gradient — the solver slides every conflicted edge onto the lower face, so the
collapse just relocates from F = 0 to F = a₀ − κσ_a: still constraint-geometry, not
data. (ii) A 1σ box asserts the truth lies within 1σ of all 150k boxed priors
simultaneously (~⅓ should genuinely lie outside even with perfect priors). (iii)
Bound-active solutions break the z diagnostics and the Stage-5 uncprop posterior
(no interior Gaussian). (iv) It duplicates information the σ-weighted prior term
already encodes — the only reason it "helps" is that the ratio rows are 10⁵–10⁶×
over-weighted, which is exactly what 8.2 repairs at the source.
**Legitimate residual use.** A *wide* box (κ ≈ 3–5, l = max(0, a₀ − κσ_a)) is
principled as a **trust region for the 8.2 relinearization iterations** (keeps each
Gauss–Newton re-solve where the linearization is valid; re-centred and released as
iterations converge), not as a permanent statistical constraint.

### 8.4 — ICSG fan-out: bundle total as a soft constraint
**What.** On region-years without partner resolution, stop writing per-edge values;
emit one soft sum row over the 47-edge partner bundle (total, σ = total's cv ≈ 6.8%);
per-edge priors fall back to flat prior / BACI.
**Expected effect.** Keeps the national total at its native resolution; removes the
fabricated uniform partner pattern asserted at cv 6.8%/edge.
**Risk.** Touches the feed→constraint path (new SumConstraintBuilder source); partner
split becomes solver-determined — underdetermined where BACI is silent (acceptable:
that is the honest state of knowledge).

### 8.5 — Fix order
8.1 (done) → **8.2 (done — §7.3 Run A)** → ICSG fan policy (done — ADR 0008,
superseding 8.4's bundle-total design per user decision; §7.3 Run B), with 8.3 still
available as an experiment arm. Status after this session: collapse broken, pinned
sum rows 16,206 → 1,001; the residual refined-cathode pins (§7.3 Run B verdict) are
the next investigation.

---

## Artefact sources

- opt: `data/proc/opt/opt_20260611_075036_6d2e540f/timesteps/t_*.npz` (`F_opt`, `residuals`, `rows_t`, `cols_t`)
- const: `data/proc/const/const_20260611_073743_f1912321/{G.npz, a0.npy, sigma_a.npy, row_meta.parquet, timesteps/t_*.npz}`
- imp: `data/proc/imp/imp_20260610_111400_d0c5b535/{agg_result.npz, imputation_log.parquet}`
- acc: `data/proc/acc/copper/{edge_index.csv, node_index.csv}`
- z table (all rows × 30 y): rebuildable via `z = residuals / sigma_c[rows_t]` join; session scratch at `/tmp/all_z.parquet`
- fixed const: `data/proc/const/const_20260611_101432_4f78bbf0` (label `fix ratio bound inversion`)
- fixed-bounds opt: `data/proc/opt/opt_20260611_102234_3d776ba4`; recomputed z at `/tmp/all_z_fixed.parquet`
- Run A (8.1+8.2): const `const_20260611_125202_2b3c73cd`, opt `opt_20260611_140040_0ec71443`; z at `/tmp/all_z_runA.parquet`
- Run B (+ICSG fan policy): feed `feed_icsg_20260611_131027_a7c03949` (trade_shares v4), imp `imp_20260611_131319_928c86ee`, const `const_20260611_131746_485f4d5f`, opt `opt_20260611_140957_24eeecb0`; z at `/tmp/all_z_runB.parquet`
- code: `src/pysuto/constraints/ratio.py` (`RatioBoundBuilder._emit`, `RatioConstraintBuilder.build_dynamic` σ scaling), `src/pysuto/conc.py` (fan policy, ADR 0008), `scripts/icsg_feed.py` (leg filter wiring), `scripts/parse/icsg_parse/icsg_clean.py` (Matte rename, 2-stage eff), `src/pysuto/solver.py` (cold-restart retry, reduced tolerances), `tests/test_constraints.py`, `tests/test_conc.py::TestFanPolicy`
