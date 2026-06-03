# Flow collapse from ICSG-CC ↔ BACI-Mass feed-scale incoherence — `opt_20260602_133908_46bc514e`

Investigation triggered by two user observations on
[data/proc/opt/opt_20260602_133908_46bc514e/](../../data/proc/opt/opt_20260602_133908_46bc514e/)
(label *"new soft constraints init fixed dfeed imp chains"*):
1. The control totals look **static across time** — suspected encoding bug.
2. The solver pulls **almost all data points toward very small values**.

The user also proposed a "tight-fitting" solver init (box `a ∈ [a₀−σ_a, a₀+σ_a]`,
drop the quadratic prior, box `ε ∈ [−σ_c, +σ_c]`) and asked whether it is consistent
with RAS / Lenzen 2009. §8.3 addresses that directly.

Upstream chain (from sidecars):
- const run: [const_20260602_131610_a37706d3](../../data/proc/const/const_20260602_131610_a37706d3/) (label *"feed debug, imp debug, new control tot"*)
- imp run:   [imp_20260602_124241_77ee0ad4](../../data/proc/imp/imp_20260602_124241_77ee0ad4/)
- feeds:     `crude_ore`, `baci`, `icsg`
- acc:       `copper` v3 (`0342a990`)

Config knobs of note (const `stage_config`):
- `sigma_a_cv_min: 0.001`, `sigma_a_abs_floor: 1e-6`
- `sigma_c_cv_min: 0.01`, `sigma_c_abs_floor: 0.001`
- `p: 1.3`

This report builds on the prior collapse lineage and does **not** re-derive the QP-collapse
mechanism — see [`…_20260527_flow-collapse.md`](analysis_opt_20260526_165810_3fd6d527_20260527_flow-collapse.md),
[`scale_preservation_recommendations_20260528.md`](scale_preservation_recommendations_20260528.md),
[`…_20260526_scaling.md`](analysis_opt_20260521_100428_9d33a0b9_20260526_scaling.md), and the ICSG fan-out
guard [`…_conc-both-endpoints.md`](analysis_feed_icsg_20260602_073443_06a5e349_20260602_conc-both-endpoints.md).
Its new contribution is to show that the Mass/CC incoherence those REPs assumed-fixed is **still live**
in this run and is the proximate cause of the collapse here.

---

## 0 — Executive summary

The "static control totals" is a non-issue: `sum_eq` anchors **do** vary by year (≥2 distinct
values per row across 30 years) and are correctly per-year activated. The flatness in the diag plot
is the *collapse symptom* — every flow ≈ 0 every year — not a c-encoding bug.

The collapse is real and severe: **96.3%** of edges with `a₀≠0` are pulled below 10% of prior;
**91%** of solved edges land at `|F_opt| < 1e-6`; **60%** sit exactly on the lower bound `l` (=0 for
69% of edges). The driver is that the prior `a₀` is **dimensionally incoherent** and therefore lies
~6 orders of magnitude off the constraint manifold, while the soft constraints outweigh the prior
~1.4 million-to-one — so the QP's cheapest feasible point is the trivial collapse to `l`.

The incoherence is specific and traceable: the `conc_*_supply` ratio constraints
(`+1·edge[CC] − 0.9975·edge[Mass] = 0`) pair an **ICSG-sourced CC (copper-content) edge** against a
**BACI-sourced Mass edge** on the *same* physical node pair, and the two feeds populate them at scales
that differ by 144× (median) to ~3·10⁵× (worst). At the worst, ICSG fans an identical production figure
`7393.1` onto 47 CC edges across regions while BACI gives the paired Mass edge `0.0248`. **100% of the
877 ratio rows with `z > 1e3` at the prior are CC-vs-Mass pairs.**

> **Verdict.** Not a control-total bug. The prior `a₀` is dimensionally incoherent — ICSG (CC valuation,
> partly fanned-out) and BACI (Mass valuation) populate the two sides of every `conc_*_supply` ratio at
> incompatible scales (≈3·10⁵× worst) — so `a₀` grossly violates the ratios and 68% of hard mass-balance
> nodes. With soft ratios weighted ~1.4M× the prior, the global optimum collapses the connected network
> to the lower bound `0`.

---

## 1 — Observations as given

1. *"Control totals are static across time … bug in how the control totals are encoded."*
2. *"The solver still pulls almost all data points towards very small values."*

These are **not** independent. (1) is the user's hypothesis; the trace below **rejects** it and shows
the flatness is the visual symptom of (2). The real question is (2).

---

## 2 — Pipeline trace setup

| Observation | Owning stage | First place to look |
|---|---|---|
| Control totals "static" | Stage 3 (constraint RHS) | `const/timesteps/t_*.npz` `c`, stratified by `row_meta.type` |
| Flows collapse to ≈0 | Stage 4 (solver) | `F_opt` per year vs `a₀`, objective decomposition |
| Why `a₀` off-manifold | Stage 2 → Stage 1 | `agg_result.npz` `F`/`Sigma`; feed `result.npz` `F_root` |

Sidecar peculiarity: the opt run's `stage_config` and `parents.inputs` are **empty** — the resolved
solver knobs are not recorded. All config values above are read from the **const** sidecar and the live
arrays on disk, not the opt sidecar.

---

## 3 — Data characterisation

### 3.1 Control totals are not static (observation 1 rejected)

`c(t)` loaded for all 30 timesteps, stratified by `row_meta.type` (N_rows = 1,358,620):

| type | rows | c varies across t? | c≠0? | note |
|---|---:|---|---|---|
| `sum_eq` (control-total anchors) | 3,610 | **yes (100%)** | yes | per-year activated; **~112–125 active/year**, rest σ_c=1e10 sentinel |
| `ratio_bound` (hard) | 879,840 | yes | yes | |
| `ratio_eq` (soft process ratios) | 435,690 | no (conc/eff params) | mostly **c=0** | time-invariant — correct |
| `mass_balance` (hard) | 39,480 | n/a | **c=0** | homogeneous — correct |

`sum_eq` distinct-c-per-row across 30 years: min 2, median 2 (≥2 ⇒ varies). Example
`sxew_prod_anchor_Australia_1995_CC = 36.1`, active only in 1995 (σ_c sentinel elsewhere). **No
encoding bug.** The diag-plot flatness is observation 2.

### 3.2 Magnitude of the collapse

`F_opt` reassembled across block timesteps (year-local `cols_t`) vs prior `a₀` (N=854,460 edges, 46.7%
nonzero, median `|a₀|` = 148):

| metric | value |
|---|---|
| edges with `a₀≠0` pulled below 10% of prior | **96.3%** |
| …below 1% of prior | 95.4% |
| solved edges with `\|F_opt\| < 1e-6` | 91.0% |
| solved edges exactly at lower bound `l` | 60.6% (`l`=0 for 68.6% of edges) |
| `F_opt/a₀` percentiles (p25 / p50 / p75) | 2e-14 / 1.6e-13 / 3e-12 |

### 3.3 Objective decomposition (year 2010, representative)

| evaluated at | soft (ratio+sum) cost | prior cost | total | max \|hard residual\| |
|---|---:|---:|---:|---:|
| `a = a₀` (prior) | **4.95e14** | 0 | 4.95e14 | 7.83e5 |
| `a = F*` (solved) | 4.45e5 | 3.34e7 | 3.39e7 | 2.25 |

- Active soft rows in 2010: **14,523 homogeneous (c=0)** ratio equalities + **125 anchored** control totals.
- σ_c weight on active soft rows: median `1/σ_c² ≈ 15,955`. Prior weight: median `1/σ_a² ≈ 0.011`.
  **Soft constraints outweigh the prior ~1.4 million-to-one.**
- Hard mass-balance at `a₀`: **67.6%** of nodes have `|residual| > 1` (max 8.3e5).

`a = 0` satisfies every homogeneous row and the bounds (`l`=0); the prior is too weak (by ~10⁶) to oppose
it. Hence the collapse. This is the mechanism already documented in the May-27/May-28 REPs.

### 3.4 Worst-N: where the ratio violation lives

Active `ratio_eq` rows with `|z| > 1e3` at `a₀` (year 2010): **877 rows, 100% of them CC-vs-Mass
valuation pairs.**

| constraint family | rows (z>1e3) |
|---|---:|
| `conc_cathode_supply` | 375 |
| `conc_concentrate_supply` | 229 |
| `conc_anode_supply` | 116 |
| `conc_matte_supply` | 102 |
| `conc_scrap_class_*` | 55 |

Scale gap `max(\|a₀\|)/min(\|a₀\|)` on those rows: median **144×**, p90 **~7000×**, worst **~3·10⁵×**.

---

## 4 — Pipeline trace of one specific case

Worst single row (year 2010): `conc_cathode_supply`, `row 429715`, structure
`+1·edge[770431] − 0.9975·edge[770430] = 0`, `σ_c = 0.00192`, `z = 3.85e6`.
Both edges are node pair **883 → 902** in two valuations.

### 4.1 Feed layer
- ICSG `result.npz` `F_root[770431] = 7393.1`; the value `7393.1` appears on **47 CC edges** (all CC).
  ICSG is **entirely CC** (173,856 CC edges, 0 Mass). The identical value across 47 region edges is a
  **single-endpoint fan-out** signature (cf. [`…_conc-both-endpoints.md`](analysis_feed_icsg_20260602_073443_06a5e349_20260602_conc-both-endpoints.md)) — a national/aggregate
  figure replicated onto every region's CC edge without weight-splitting.
- BACI `F[baci, 770430] = 0.0248` (σ=0.00187) — the Mass side.

### 4.2 Aggregation
- `f_agg[770431] = 7393.1`, `f_agg[770430] = 0.0248`. **Neither edge appears in
  `imputation_log.parquet`** (0 rows) → both are direct feed observations via `f_agg`, **not**
  FlatPrior fills. So this is genuine feed data at incompatible scales, not an imputation artefact.

### 4.3 Constraints
- `a₀[770431] = 7393.1`, `a₀[770430] = 0.0248`; `σ_a` at cv ≈ 0.069 (so σ_a[770431] ≈ 505).
- Row 429715 ties `CC ≈ 0.9975·Mass` ⇒ requires `7393.1 ≈ 0.0247` — violated by ~3·10⁵×.

### 4.4 Solver
- `F_opt` drives both edges toward 0 (`0 ≈ 0.9975·0` satisfies the ratio at σ_c-precision, paying only
  the negligible prior cost). The collapse propagates to mass-balance neighbours.

---

## 5 — Hypotheses

### H1 — Control-total RHS `c(t)` is mis-encoded as time-constant
**Mechanism.** `SumConstraintBuilder` per-year activation writes the same `c` for every `t`.
**Signature if true.** `sum_eq` rows show 1 distinct c across 30 years.
**Falsifier.** ≥2 distinct values per row. → **REJECTED** (§3.1: min 2 distinct).

### H2 — Weight imbalance alone collapses the system
**Mechanism.** σ_c floor (`1e-3`/cv `0.01`) weights soft rows ~10⁶× the prior; `a=0` is feasible for
homogeneous rows ([`engine.py` σ floors]; documented May-28 REP).
**Signature if true.** Collapse persists even when `a₀` is consistent with the constraints.
**Falsifier.** If `a₀` were near-manifold and collapse vanished, H2 is the *whole* story.
→ **PARTIALLY CONFIRMED** — necessary but not sufficient; see H3.

### H3 — `a₀` is dimensionally incoherent (ICSG-CC vs BACI-Mass), so it is far off-manifold
**Mechanism.** `conc_*_supply` ratios pair ICSG-CC edges (partly fanned-out) against BACI-Mass edges;
the two feeds are not scale-reconciled, so `a₀` violates the ratios by 144×–3·10⁵×.
**Signature if true.** The severe-violation rows are exclusively CC-vs-Mass pairs; the offending CC
edges trace to ICSG, the Mass edges to BACI, neither imputed.
**Falsifier.** If the z>1e3 rows were a mix of valuation pairs, or the values were FlatPrior fills.
→ **CONFIRMED** (§3.4: 100% CC-vs-Mass; §4: ICSG vs BACI, 0 imputation-log rows).

### 7.1 / Verdict
H1 rejected. The root cause is **H3 compounded by H2**: the feed-scale incoherence (H3) places `a₀`
~6 orders of magnitude off-manifold, and the σ_c/σ_a weight imbalance (H2) makes collapse to `l` the
cheapest feasible response. Neither alone explains the run: with a coherent `a₀`, the weight imbalance
would still bias downward but not by 10⁶×; with a balanced weight, the incoherent `a₀` would still be
infeasible on the hard rows. The fan-out component (identical 7393.1 on 47 CC edges) shows the CC side
also carries an *unsplit-aggregate* error on top of the valuation-scale gap.

---

## 6 — Test strategy

| H | experiment | expected if true | result |
|---|---|---|---|
| H1 | distinct `c` per `sum_eq` row across 30y | =1 | **2** → rejected |
| H2 | median `1/σ_c²` vs `1/σ_a²` on active soft rows | ratio ≫ 1 | **~1.4e6** → confirmed |
| H3 | valuation-pair composition of `z>1e3` ratio rows | all CC-vs-Mass | **877/877 = 100%** |
| H3 | feed source + imputation-log of the worst edge pair | ICSG-CC vs BACI-Mass, no imp fill | **confirmed** |

No re-solve or code change was needed to reach the verdict; all tests are read-only.

---

## 7 — Test results

See the table in §6 and verdict in §5/7.1. All four tests executed; H1 rejected, H2 partial, H3 confirmed.

---

## 8 — Proposed fixes

> **Resolution (2026-06-03).** Defect (a) below — the ICSG fan-out replication — is **fixed and
> verified**. Root cause was deeper than "not splitting": it took **three coordinated changes**
> (symmetric `<any>` sector/region fallback; entity follows `<any>`; `build_F_src` de-dups the
> fan-out before summing), each surfaced by a new hard-error **mass-conservation guard** at feed
> read-in. ICSG mass inflation `36.08× → 1.0000×` (ΣF_src 74.7M → 2.07M = raw total). See
> [ADR 0002](../adr/0002-single-endpoint-rows-fan-out-not-replicate.md). Defect (b), the CC↔Mass
> valuation-scale reconciliation, remains open.

### 8.1 (Highest leverage, real cure) — reconcile ICSG-CC ↔ BACI-Mass scale upstream
**What.** Make the two sides of every `conc_*_supply` ratio dimensionally coherent **before** Stage 3.
Two distinct defects to separate: (a) **fan-out** — ICSG replicates an aggregate (`7393.1`) onto 47 CC
edges without splitting (same class as the conc-both-endpoints guard, evidently not covering CC supply);
(b) **valuation scale** — CC (copper-content) vs Mass (gross) live at different magnitudes per feed.
Entry points: ICSG `result.npz` `F_root` for the 47 fan-out edges and the feed concordance/parser;
BACI Mass scale.
**Expected effect.** Removes the 4.95e14 soft cost at `a₀`; ratio z drops from 3.85e6 toward O(1).
**Risk.** Touches feed parsing/concordance; needs a feed-level regression (re-run feed → imp → const).

### 8.2 — rebalance σ_c vs σ_a (KRAS-faithful, stops collapse without boxes)
**What.** Raise `sigma_c_cv_min`/`abs_floor` (or lower σ_a floor) so soft rows no longer outweigh the
prior 10⁶×. One-line config in the const `stage_config`.
**Expected effect.** Reduces the downward bias on coherent edges; **will not** fix the incoherent ratios
(those are genuinely infeasible at `a₀`), so collapse persists wherever H3 holds. Symptomatic only.
**Risk.** Loosening soft constraints weakens legitimate ratios elsewhere; blast radius = all soft rows.

### 8.3 — "tight-fitting" box guardrail (the user's idea, scoped) — see [plan §5](../../../.claude/plans/pls-analyse-run-data-proc-opt-opt-202606-zippy-wadler.md)
**What.** Box `l = max(l, a₀−k·σ_a)`, `u = min(u, a₀+k·σ_a)` in Stage 3 (writing `l.npy`/`u.npy`, which
already feed both the solver and `feasibility_check` [solver.py:361](../../src/pysuto/solver.py#L361)),
behind a config flag, **keeping the quadratic prior** (do not box ε — Stage-5 posterior depends on it).
**Consistency with RAS / Lenzen 2009.** This grafts RAS's scale floor — *"the balanced estimate A\* can
never go more than one σ below A₀"* (Lenzen 2009, per
[`scale_preservation_recommendations_20260528.md`](scale_preservation_recommendations_20260528.md) §3.1)
— onto the QP. So it **is** consistent with RAS's scale-preservation property and is the literature-backed
fix for the *collapse*. Caveats from this run's data:
1. On dimensionally-coherent edges it stops the collapse (a=0 violates `l=a₀−σ_a` for cv<1). ✓
2. On the incoherent edges it becomes **infeasible by construction** — CC box `[6887, 7898]` vs required
   `[0.023, 0.027]`; `ε∈[±0.0019]` cannot bridge 7393 → Clarabel returns `PrimalInfeasible`, and
   `_raise_infeasible` ([solver.py:411](../../src/pysuto/solver.py#L411)) will name the `conc_*_supply`
   rows. This matches the user's stated goal *"surface as infeasible"* and makes the box a **detector**
   for the 8.1 defect — but it yields no table until 8.1 lands.
3. Hard boxes that go infeasible are *anti*-KRAS (KRAS never goes infeasible under conflict — it violates
   constraints ∝ unreliability). So the box satisfies Lenzen's *scale-floor* property but not KRAS's
   *conflict-resolution* property; the two goals are in tension and the box chooses surfacing over
   reconciling. **Recommended ordering: 8.1 first (cure), 8.3 as a permanent guardrail+detector, 8.2 only
   if residual downward bias remains.**

---

## 9 — Suggested verification

1. **Spot-check the pattern.** Confirm the ICSG-CC/BACI-Mass split across all 47 fan-out edges and that
   the `z>1e3` set stays 100% CC-vs-Mass after any feed fix (should shrink toward 0).
2. **Regression on the feed fix (8.1).** Re-run feed → imp → const; check `conc_*_supply` ratio residuals
   at the new `a₀` drop below O(σ_c), and hard mass-balance `|residual|>1` share falls well below 67.6%.
3. **Box smoke test (8.3).** Re-solve year 2010 with the box at `k=2`: (a) before 8.1 — expect
   `PrimalInfeasible` naming `conc_*_supply` (detector works); (b) after 8.1 — expect `Solved` with
   retention `Σ|F*|/Σ|a₀|` → O(1) and no mass-balance residual above tolerance.

---

## Source artefacts

- Opt: [data/proc/opt/opt_20260602_133908_46bc514e/timesteps/](../../data/proc/opt/opt_20260602_133908_46bc514e/timesteps/) (`F_opt`, `cols_t`, `rows_t`)
- Const: [data/proc/const/const_20260602_131610_a37706d3/](../../data/proc/const/const_20260602_131610_a37706d3/) (`a0.npy`, `sigma_a.npy`, `l.npy`, `u.npy`, `G.npz`, `row_meta.parquet`, `timesteps/t_*.npz`)
- Imp: [data/proc/imp/imp_20260602_124241_77ee0ad4/agg_result.npz](../../data/proc/imp/imp_20260602_124241_77ee0ad4/agg_result.npz) (`F`, `Sigma`, `feed_ids`), `imputation_log.parquet`
- Feed: [data/proc/feed/registry/feed_icsg_20260602_121317_59beab7c/result.npz](../../data/proc/feed/registry/feed_icsg_20260602_121317_59beab7c/result.npz) (`F_root`)
- Edge index: [data/proc/acc/copper/edge_index.csv](../../data/proc/acc/copper/edge_index.csv)
- Plan: [.claude/plans/pls-analyse-run-data-proc-opt-opt-202606-zippy-wadler.md](../../../.claude/plans/pls-analyse-run-data-proc-opt-opt-202606-zippy-wadler.md)
