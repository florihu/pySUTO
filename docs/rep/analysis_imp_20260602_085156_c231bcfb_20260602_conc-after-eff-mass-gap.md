# Mass-valuation supply edges stuck at zero — `imp_20260602_085156_c231bcfb`

Standalone investigation triggered by a user observation on
[data/proc/imp/imp_20260602_085156_c231bcfb/](../../data/proc/imp/imp_20260602_085156_c231bcfb/).

Upstream chain:
- imp run: [imp_20260602_085156_c231bcfb](../../data/proc/imp/imp_20260602_085156_c231bcfb/)
- feeds:   `feed_crude_ore_20260602_074032_20f17887`, `feed_baci_20260602_073955_eddd55bf`, `feed_icsg_20260602_073443_06a5e349`
- acc:     `copper` v3 (`854460` edges, valuations `[Mass, CC]`, 47 regions × 30 years 1995–2024)

Config knobs of note (`metadata.json` → `stage_config`):
- `flat_mode: noisy_zero`, `flat_sigma_high: 1e6` — residual NaN edges are filled with **f = 0**, σ = 1e6.
- `p: 1.3`, `q: 3.0`, `alpha_min: 0.05`.

---

## 0 — Executive summary

The user noticed that process-output supply edges **from (Smelting, CFR, ER, …)
to (Matte, Anode, CER)** carry no value **in the Mass valuation only**, while the
CC valuation of the same edges is partly populated. This is **not** a
feed-coverage / pre-imputation gap: those CC supply edges were never observed by
any feed (`observed = 0`) — they exist only because the **efficiency** imputer
(`eff`) produced them. The Mass side is missing because of **imputer ordering**:
the bidirectional Mass↔CC converter (`conc`) runs *first* (log step 0), before
`eff` (step 2) and the `FormulaImputer` (step 3) populate the CC supply edges.
`conc` is single-pass and only fires when exactly one side of a Mass↔CC pair is
known, so the CC values `eff` creates arrive too late to ever be converted to
Mass. Those Mass edges stay NaN and are swept to **f = 0 by the `flat_prior`
`noisy_zero` fallback** (step 6). All **855** `Smelting→Matte CC` edges that
`eff` filled have a Mass twin stuck at `flat_prior_noisy_zero = 0`.

> **Verdict.** The Mass↔CC concentration converter (`conc`) runs once, before
> the efficiency/formula passes that create the CC supply edges, so eff-produced
> CC values are never converted to Mass — the Mass supply edges fall through to
> the flat-prior zero. A second `conc` pass after `eff`/`FormulaImputer`
> recovers **2,316** Mass supply edges.

---

## 1 — Observations as given

> "why do certain mass sectors like from (Smelting, CFR, ORE, ER) and to matte
> anode cer have no vals but only for mass — was this pre imputation? pls check
> whats going on here"

> hypothesis: "i have to initiate after the concentr imput and eff imput again
> the conc impute to propagate the eff imputed changes?"

Two interdependent claims: (a) Mass-valuation edges into Matte / Anode / CER are
zero while their CC twins are not, and (b) re-running `conc` after `eff` would
fix it. They are the same phenomenon viewed from two sides.

---

## 2 — Pipeline trace setup

Stage 2 (aggregation + imputation) owns this entirely; the run never reached
Stage 3+. Artefacts loaded:

- `agg_result.npz` → `f_agg`, `coverage`.
- `imputation_log.parquet` → per-edge final fill `method` / `step`.
- Expanded edge index via `expand_edge_index(acc.edge_index, acc.node_index)`.

The 7 imputers actually ran in this order (`metadata.json → imputers`, log step
is 0-indexed; the script's inline comments number them 1–7):

| log step | imputer | params | edges filled |
|---:|---|---|---:|
| 0 | ParametricImputer | `conc` | 163,306 |
| 1 | ParametricImputer | `ore_grade` | 0 |
| 2 | ParametricImputer | `eff` | 2,608 |
| 3 | FormulaImputer | `conc` (6 relations) | 395 |
| 4 | TimeImputer (interpolate) | — | 10,587 |
| 5 | TimeImputer (crude_ore LOCF) | — | 62 |
| 6 | FlatPriorImputer (`noisy_zero`) | — | **460,191** |

`n_total = 854,460`, `n_covered_before = 217,311`, `n_imputed = 637,149`. More
than half of all edges end as flat-prior zeros.

---

## 3 — Data characterisation

### 3.1 Where the zeros are

460,299 edges have `f_agg == 0`; 460,191 of them are `flat_prior_noisy_zero`
(the rest are genuine observed/derived zeros). The flat-prior fills are the
"missing" edges the user is seeing.

### 3.2 Stratification — the named sectors

**CC supply edges (process → flow), Mass vs CC.** `n = 1410` per pair
(47 regions × 30 years). These are the edges the user pointed at:

| edge (process→flow) | Mass: flat-prior 0 | CC: parametric (eff) | CC: flat-prior 0 | CC: observed |
|---|---:|---:|---:|---:|
| Smelting→Matte | **1410 / 1410** | 855 | 554 | 0 |
| CFR→Anode | **1410 / 1410** | 782 | 628 | 0 |
| ER→CER | **1410 / 1410** | 582 | 820 | 0 |
| MF→Concentrate | **1410 / 1410** | 0 | 1410 | 0 |
| CP→Cathode | 1193 / 1410 | (187 obs, 30 time) | 1193 | 187 |

The Mass valuation of every Smelting→Matte / CFR→Anode / ER→CER / MF→Concentrate
supply edge is **100 % flat-prior zero**, while the CC valuation is partly
eff-filled. `observed = 0` everywhere — no feed ever saw these CC supplies.

**Downstream domestic-use Mass edges** (region_from = region_to; targets of the
`FormulaImputer`), `n = 66,270` (47 × 47 × 30):

| target (flow→process) Mass | parametric | observed | time | formula | flat-prior 0 |
|---|---:|---:|---:|---:|---:|
| Concentrate→Smelting | 33,756 | 7,568 | 479 | 0 | 24,467 |
| Matte→CFR | 33,833 | 3,507 | 328 | 90 | 28,512 |
| Anode→ER | 23,292 | 5,644 | 900 | 305 | 36,129 |

(The `Cathode→FU/USC/MESC` formula relations matched 0 edges — those
flow→process Mass targets do not exist in this structure; the 90 + 305 = **395**
formula fills are entirely Matte→CFR and Anode→ER.)

### 3.3 Worst case

`Smelting→Matte` Mass: 1410/1410 zero is the cleanest worst case — 855 of those
zeros have a CC twin that `eff` *did* fill, so the value exists in the system
and is simply never converted.

---

## 4 — Pipeline trace of one specific case

Edge pair for `Smelting→Matte`, one region/year (from the `conc_matte_use`
resolver):

- **edge 152281** — `Smelting/Process → Matte/Flow`, valuation **CC**: filled by
  `eff` at **log step 2** (`method = parametric`, relation `eff_smelting`:
  `Matte CC = 0.965 × Concentrate→Smelting CC`).
- **edge 152280** — same edge, valuation **Mass**: final `method =
  flat_prior_noisy_zero`, `f = 0` (log step 6).

The conc Mass↔CC pair for this edge **exists** — `conc_matte_use` resolves
(152280 Mass ↔ 152281 CC), value 0.525, so `Mass = CC / 0.525` is available in
principle. But `conc` (ParametricImputer, [parametric.py:221-256](../../src/pysuto/impute/parametric.py#L221-L256))
ran at **step 0**, when *both* sides were NaN, and `impute` only writes where the
target is NaN ([parametric.py:222-224](../../src/pysuto/impute/parametric.py#L222-L224)).
By step 2, when `eff` writes the CC side, `conc` is finished and never revisits
the pair. The Mass side therefore survives to the flat-prior fallback.

Aggregating over all 855 eff-filled `Smelting→Matte CC` edges: **every one** has
a Mass twin whose final method is `flat_prior_noisy_zero`.

---

## 5 — Hypotheses

### H1 — `conc` runs before `eff`, so eff-produced CC is never converted to Mass (the user's hypothesis)

**Mechanism.** Registration order in
[collect_aggregate_impute.py:359-377](../../scripts/collect_aggregate_impute.py#L359-L377):
`conc` (step 0) → `ore_grade` (1) → `eff` (2) → `FormulaImputer` (3). `eff`
params are all `a_valuation = CC, b_valuation = CC` (e.g. `eff_smelting`,
`eff_cfr`, `eff_er` in `data/input/params/eff/v1/params.csv`), so `eff` only ever
writes CC edges. The Mass↔CC bridge is `conc`, which ran first and is single-pass
([parametric.py:206-256](../../src/pysuto/impute/parametric.py#L206-L256)).

**Predicted signature.** Every CC supply edge that `eff` fills has a Mass twin
ending as flat-prior zero; and a re-run of `conc` after `eff` would fill exactly
those Mass twins.

**Falsifier.** If eff-filled CC edges had non-zero Mass twins, H1 is wrong.

### H2 — Feed-coverage gap (the "pre-imputation" reading)

**Mechanism.** The Mass supply edges were never observed, so nothing could fill
them. **Rejected:** the CC twins were *also* never observed (`observed = 0`) yet
got filled by `eff`; the Mass side could have been filled from the CC side. The
gap is created inside imputation, not upstream.

### H3 — `FormulaImputer` gating leaves Matte/Anode domestic use empty

**Mechanism.** The matte/anode formula needs the CC supply edge as
`aggregate="single"`, non-NaN ([formula.py:241-246](../../src/pysuto/impute/formula.py#L241-L246)),
so where the CC supply is still NaN at step 3 the formula skips. Real but
**secondary**: the formula filled 90 + 305 edges; it is not the source of the
100 %-zero Mass *supply* edges (which are not formula targets at all).

---

## 6 — Test strategy

| H | experiment | expected if true | expected if false |
|---|---|---|---|
| H1 | for `Smelting→Matte CC` edges with `method=parametric` (eff), check Mass twin's final method | all `flat_prior_noisy_zero` | some non-zero |
| H1 | simulate a 2nd `conc` pass: count edges with `f==0` whose Mass↔CC partner is known | thousands, concentrated on the named supply pairs | ≈ 0 |
| H2 | `observed` count on the CC supply edges | 0 | > 0 |

No re-solve or code change was needed to *test* (all read-only), so this stays
inline — no plan file.

---

## 7 — Test results

| H | result | evidence |
|---|---|---|
| H1 | **confirmed** | 855/855 eff-filled `Smelting→Matte CC` edges have a `flat_prior_noisy_zero` Mass twin; a simulated 2nd conc pass fills **3,166** edges, **2,316** of them CC→Mass supply edges (Smelting→Matte 856, CFR→Anode 782, ER→CER 590, CP→Cathode 88) |
| H2 | **rejected** | `observed = 0` on all CC supply edges — they came from `eff`, not feeds |
| H3 | **secondary** | formula filled only 90 (Matte→CFR) + 305 (Anode→ER); not the cause of the zero *supply* edges |

### 7.1 Verdict

The root cause is **imputer ordering, exactly as the user hypothesised**. `conc`
is a single-pass bidirectional Mass↔CC converter registered *before* the
efficiency (`eff`, step 2) and formula (step 3) passes that fabricate the
CC-valuation process-output supply edges (Smelting→Matte, CFR→Anode, ER→CER,
CP→Cathode). Because `conc` only fires when exactly one side of a pair is known
and never re-runs, the CC values `eff` creates are never carried over to Mass;
those Mass edges remain NaN and the `flat_prior` `noisy_zero` fallback sets them
to 0. The "Mass-only" character of the gap is the direct signature: the CC side
was filled, the Mass side was orphaned.

---

## 8 — Proposed fixes

> **Resolution (2026-06-02).** 8.1 was applied first as a stop-gap, then
> **reverted and superseded by 8.2** (the full fixpoint cascade), which was
> implemented in `ImputationEngine.register_cascade` + a fixpoint `run()` loop.
> A re-run (`imp_20260602_100507_06266cd6`) converged in **4 passes** and
> recovered **3,972** Mass edges (vs 2,316 for the one-shot 8.1): Smelting→Matte
> 855, CFR→Anode 872, ER→CER 933 parametric — the cascade reaching further
> down-chain than 8.1 could. `MF→Concentrate` stays 1410/1410 zero, confirming
> it is a genuine data gap (`Ore→MF CC` unobserved), not late-arrival
> starvation. The change also added imputer-`id` + `pass` provenance to the log
> and sidecar. Implemented in `ImputationEngine.register_cascade` + the fixpoint
> `run()` loop ([engine.py](../../src/pysuto/impute/engine.py)).

### 8.1 (Stop-gap — applied then reverted) Second `conc` pass after `eff` / `FormulaImputer`

**What.** Register one more `ParametricImputer.from_csv("conc", …)` after the
`FormulaImputer` and before the `TimeImputer` in
[collect_aggregate_impute.py:379](../../scripts/collect_aggregate_impute.py#L379)
(added as step "4b").

**Expected effect.** Recovers **2,316** Mass supply edges — Smelting→Matte 856,
CFR→Anode 782, ER→CER 590, CP→Cathode 88 — plus ~850 other Mass↔CC edges
(≈ 3,166 total), converting them from flat-prior zero to real metal-content-based
values.

**Risk.** Effectively none. `ParametricImputer.impute` writes only where the
target is NaN ([parametric.py:222-224](../../src/pysuto/impute/parametric.py#L222-L224)),
and the pass runs before `flat_prior`, so it can only *add* fills to edges that
would otherwise become 0 — no existing value changes.

### 8.2 (Implemented) Full `[conc, eff, formula]` fixpoint loop

**What.** `ImputationEngine.register_cascade([conc, ore_grade, eff, conc_formula],
max_iter=10)` iterates the group until a full sweep fills no new edges; the
terminal imputers (`time_interp`, `time_crude`, `flat_prior`) run once after.

**Actual effect (measured on `imp_20260602_100507_06266cd6`).** Converged in 4
passes; flat-prior fills dropped 460,191 → 456,219 (**3,972** more real fills).
The `MF→Concentrate` / `ORE` branch stays zero — `eff` fills 0 of
`MF→Concentrate CC` because its upstream `Ore→MF CC` is unobserved, so this is a
true data gap the loop correctly does not fabricate.

**Risk — handled.** Imputers only write `NaN` cells, so fills are monotone (loop
always terminates) and no value is overwritten *within* a run. Across runs, 14
edges shifted `time_linear → parametric` — the cascade now reaches them before
temporal interpolation, a deliberate priority improvement, not a regression.

### 8.3 (Rejected) Reorder `eff` before `conc`

`eff` consumes CC inputs that `conc` produces from observed Mass; running `eff`
first would starve it. The dependency is genuinely cyclic, so a single linear
order cannot satisfy both — which is exactly why 8.1/8.2 (a second pass /
iteration) is the right shape.

---

## 9 — Suggested verification

1. Re-run Stage 2 with the same feeds and confirm the new conc step appears in
   the run's `metadata.json → imputers` filling ~2–3 k edges:
   ```
   python scripts/collect_aggregate_impute.py \
     --feed-id feed_crude_ore_20260602_074032_20f17887 \
     --feed-id feed_baci_20260602_073955_eddd55bf \
     --feed-id feed_icsg_20260602_073443_06a5e349 \
     --label "conc-after-eff fix"
   ```
2. On the new run, confirm `Smelting→Matte` / `CFR→Anode` / `ER→CER` **Mass**
   supply edges now carry `parametric` fills instead of `flat_prior_noisy_zero`.
3. Regression: total `n_covered_after` still 854,460, and no edge that was
   non-zero before changed value (the pass only fills NaN).
4. Spot-check that `MF→Concentrate` Mass is still zero — expected, and the
   trigger for considering the §8.2 fixpoint if that branch matters downstream.

---

*Generated 2026-06-02. Trace files used:*
- *imp output: [data/proc/imp/imp_20260602_085156_c231bcfb/agg_result.npz](../../data/proc/imp/imp_20260602_085156_c231bcfb/agg_result.npz), [imputation_log.parquet](../../data/proc/imp/imp_20260602_085156_c231bcfb/imputation_log.parquet), [metadata.json](../../data/proc/imp/imp_20260602_085156_c231bcfb/metadata.json)*
- *params: [data/input/params/conc/v1/params.csv](../../data/input/params/conc/v1/params.csv), [data/input/params/eff/v1/params.csv](../../data/input/params/eff/v1/params.csv)*
- *imputer code: [parametric.py:206-319](../../src/pysuto/impute/parametric.py#L206-L319), [formula.py:171-293](../../src/pysuto/impute/formula.py#L171-L293)*
- *driver / fix: [collect_aggregate_impute.py:359-405](../../scripts/collect_aggregate_impute.py#L359-L405)*
