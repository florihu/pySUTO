# ER→CER→CP prior not propagated by the NetworkImputer — opt_20260611_140957_24eeecb0

Standalone investigation triggered by user observation on
[data/proc/opt/opt_20260611_140957_24eeecb0/](../../data/proc/opt/opt_20260611_140957_24eeecb0/)
(label: `runB-final`): *"the flow output of CER and the process output of ER … the defect should
be orders of magnitude … my hypothesis is that the imputer does NOT propagate the ER output
(CER)."*

Upstream chain:
- const run: [const_20260611_131746_485f4d5f](../../data/proc/const/const_20260611_131746_485f4d5f/) (label `runB-fan-policy-sigma-scaled`)
- imp run:   [imp_20260611_131319_928c86ee](../../data/proc/imp/imp_20260611_131319_928c86ee/) (label `runB-fan-policy`)
- feeds:     icsg, crude_ore, baci · acc: copper v3 (code `0342a990`) · params: conc v2, eff v2

**Lineage.** This is the *"Run B"* of
[analysis_opt_20260611_075036_6d2e540f](analysis_opt_20260611_075036_6d2e540f_20260611_mass-collapse-ratio-bounds.md).
That report fixed the global Mass collapse (inverted ratio bounds + unscaled ratio σ_c + ICSG fan
policy) and closed by flagging a *"third, distinct signal"* still pinned in Run B — full-period
pins in the **refined-cathode chain** (Cathode/CP/**CER/ER**/CSXEW), *"a real structural/data
conflict … next investigation, out of this session's scope."* **This report is that next
investigation**, taken from the imputation side.

---

## 0 — Executive summary

The user reads an orders-of-magnitude gap between the ER process output and the CER flow output.
The gap is **real but lives in the PRIOR, not the solved values**: the solver recovers a
mass-balanced `ER→CER == CER→CP == 13,551 t` (CC, 2020), while the Stage-3 prior `a₀` for both is
**≈ 1.8 t** — a ~7,400× "rocket" the solver climbs each year from a floor seed. The seed is
`flat_prior_noisy_zero` (f = 0, σ = 1e6): the imputer never produced a value, so the warm-start
prior chain (this branch, [ADR 0009](../adr/0009-warm-start-prior-chain.md)) has **nothing
meaningful to carry forward** on this chain.

The hypothesis is **confirmed and sharpened**. The NetworkImputer's process-recipe forward fill
(`product = eff·Σinputs`) fires only when *every* primary input is known
([network.py:365](../../src/pysuto/impute/network.py)). ER|Process is fed by **bilateral**
Anode|Flow imports from up to 47 source regions; with that feed at 4.3 % coverage, ≥1 input is NaN
for essentially every region-year, so the gate never opens — ER→CER is `network_forward` **0/1410**
times. The CER flow-node balance then has no inflow to propagate, so CER→CP is flat-prior too.

The same gate blocks the **second, intended fill path** — the metallurgical formula cascade — and
this is the direct answer to the user's follow-up ("why isn't the domestic CC Anode→ER leg computed
by the formula calculator?"): the `anode_domestic` formula needs `CFR→Anode (CC)`, which is itself
an `agg=true` product owned solely by the NetworkImputer and blocked the same way one stage up.
Every process *product* in the chain (Smelting→Matte, CFR→Anode) is `agg=true` →
NetworkImputer-only → flat-prior-zero, starving both downstream paths.

> **Verdict.** User's hypothesis confirmed (imputation, Stage 2). Mechanism: the all-or-nothing
> primary-input determinacy gate ([network.py:120-121, 365](../../src/pysuto/impute/network.py))
> under sparse bilateral trade coverage leaves every `agg=true` process product at
> `flat_prior_noisy_zero`. The solver reconstructs the segment from hard mass-balance + soft `eff`
> constraints (hence the correct solved values), but the prior is a floor seed → the rocket, and
> the warm-start chain is defeated on this chain.

**No code changed this session** (per approved scope: report + fix design). The fix design is §8.

---

## 1 — Observation as given

1. "flow output of CER vs process output of ER — defect should be orders of magnitude." → §3.1.
2. "identify the upstream issue." → §4, §7.
3. (follow-up) "why is the domestic CC leg Anode→ER not calculated in the formula calculator?" → §4.2, §7 H4.

---

## 2 — Pipeline trace setup

| observation | owning stage | first place to look |
|---|---|---|
| ER process output ≫ CER flow output (in the prior) | Stage 2 (imputation) | `imputation_log.parquet`, const `a0.npy` |
| ER→CER prior ≈ 0 | Stage 2 | imputer method + feed `coverage` |
| solved ER→CER == CER→CP | Stage 4 (solver) | opt `F_opt` (consistency check, not the defect) |

Loaded: `opt_…24eeecb0/timesteps/t_*.npz` (`F_opt`, `cols_t`),
`const_…485f4d5f/{a0.npy, sigma_a.npy}`, `imp_…928c86ee/{agg_result.npz, imputation_log.parquet}`,
`data/proc/acc/copper/{edge_index.csv, node_index.csv}`. Edge id → (sector, entity, region,
valuation) via `node_index` row-index = node_id; column index → global edge_id via `cols_t`.

Chain topology (verified from `edge_index`/`node_index`):
```
Smelting|Process ─► Matte|Flow ─► CFR|Process ─► Anode|Flow ─► ER|Process ─► CER|Flow ─► CP|Process
                                                              └► Slag|SB (waste)
```
CER|Flow is supplied **only** by ER|Process and feeds **only** CP|Process; Anode|Flow is supplied
**only** by CFR|Process (no Anode→Anode flow→flow trade layer). So bilateral anode trade lands
directly on the `Anode|Flow(X) → ER|Process(Y)` flow→process edges (66,270 per valuation = 47×1410).

---

## 3 — Data characterisation

### 3.1 The symptom — prior vs solved (year 2020, CC, summed over 47 regions)

| quantity | solved `F_opt` | prior `a₀` | ratio |
|---|---:|---:|---:|
| ER process output (ER→CER + ER→Slag) | 14,057.6 | 1.84 | 7,640× |
| ER→CER (product) | 13,551.5 | 1.83 | 7,404× |
| CER→CP (CER flow output) | 13,551.5 | 1.83 | 7,404× |

Solved `ER→CER == CER→CP` to the displayed precision — the flow node *is* balanced in the solution.
The "orders of magnitude" is the **prior-to-solved rocket**, visible as the `a₀(1995)` dashed
floors in [diag/sd_r_08_process_outputs.png](../../data/proc/opt/opt_20260611_140957_24eeecb0/diag/sd_r_08_process_outputs.png)
(ER dashed ≈ 4, solid ≈ 10⁴) and [sd_r_09_flow_outputs.png](../../data/proc/opt/opt_20260611_140957_24eeecb0/diag/sd_r_09_flow_outputs.png).

### 3.2 Imputation method — ER→CER product (all 30 years, CC, 1410 edges)

| method | n | meaning |
|---|---:|---|
| `flat_prior_noisy_zero` | 1349 | f = 0, σ = 1e6 — last-resort fallback (step 12) |
| `parametric` (`effer`) | 57 | pairwise `effer × known Anode→ER` |
| `time_linear` | 4 | temporal interpolation of the few non-zero |
| **`network_forward`** | **0** | the recipe forward-fill **never fires for ER** |

For comparison, process *outputs* by sector (CC, final method): Smelting and CFR are **100 %
`flat_prior_noisy_zero`**; only CP gets any `network_forward` (101). Globally
`flat_prior_noisy_zero` is **728,382 / 854,460 edges (85 %)**.

### 3.3 Observed feed coverage (`agg_result.coverage`; feeds icsg, crude_ore, baci) — overall 7.8 %

| edge group | coverage |
|---|---:|
| ER→CER (product) | **0.0 %** |
| CER→CP (flow output) | **0.0 %** |
| ER→Slag (waste) | **0.0 %** |
| Anode→ER (ER inputs, bilateral) | **4.3 %** (Mass 8.6 % import legs; domestic ≈0 % CC) |
| Cathode outputs (downstream) | 9.3 % (the highest in the chain) |

BACI (Mass only) covers part of the **import** legs; the **domestic** Anode→ER leg (X==Y, the
dominant term) has effectively no measurement (CC 0 %, Mass ~4 %, `f_agg`≈0).

### 3.4 The eff params that decide who may fill each product (`data/input/params/eff/current/params.csv`)

| param | `agg` | product edge | fillable by |
|---|---|---|---|
| `effsmelting` | **true** (pools Concentrate+Scrap) | Smelting→Matte | NetworkImputer only |
| `effcfr` | **true** (pools Matte+Scrap) | CFR→Anode | NetworkImputer only |
| `effer` | *(blank)* | ER→CER | ParametricImputer (pairwise) ✓ |

`_parametric_consumes` ([parametric.py:48](../../src/pysuto/impute/parametric.py)) **skips
`agg=true` rows** — they are explicitly "owned by NetworkImputer."

---

## 4 — Pipeline trace of one case (Germany, 2020, CC domestic legs)

### 4.1 The chain dies at the `agg=true` products

| chain edge (CC, domestic) | `f_agg` | final method | status |
|---|---:|---|---|
| Concentrate→Smelting | 79.17 | time_linear | ✅ alive |
| **Smelting→Matte** (product, `effsmelting` agg) | **0** | flat_prior_noisy_zero | ❌ break #1 |
| Matte→CFR | 387.96 | (observed/BACI, covered) | ✅ alive |
| **CFR→Anode** (product, `effcfr` agg) | **0** | flat_prior_noisy_zero | ❌ break #2 |
| Anode→ER (Mass; formula target) | 0 | flat_prior_noisy_zero | ❌ dead |
| Anode→ER (CC) | 0 | flat_prior_noisy_zero | ❌ dead |
| ER→CER (product, `effer`) | 0 | flat_prior_noisy_zero | ❌ dead |

The breaks are exactly the two `agg=true` products. Matte→CFR is *covered* (BACI) but its consumer
product CFR→Anode is not, because CFR's forward gate is blocked by CFR's own sparse bilateral Matte
imports.

### 4.2 Why the formula calculator does not fill the domestic CC Anode→ER leg

`anode_domestic` ([collect_aggregate_impute.py:193-223](../../scripts/collect_aggregate_impute.py)):
- **Targets `valuation == "Mass"`** — by design the formula never computes the CC leg; the CC twin
  is meant to come from the `concanode` parametric.
- Its only live source term is `CFR→Anode (CC) / concanodesupply` (the export term sums over
  Anode flow→flow edges, of which there are **none** → 0). A `single` source that is NaN aborts the
  relation ([formula.py:243-248](../../src/pysuto/impute/formula.py)).
- `CFR→Anode (CC)` is flat-prior-zero (§4.1) → the formula has no live source → **`anode_domestic`
  fired 0 times** this run. The only `*_domestic` formula that fired was `concentrate_domestic`
  (240 rows; the only chain point with a live upstream CC product, MF→Concentrate, in some regions).

So the formula calculator is behaving exactly as specified — it is **starved** because the
`agg=true` product it depends on is never filled.

---

## 5 — Hypotheses (as ranked before testing)

### H1 — NetworkImputer forward gate blocked by sparse bilateral inputs (user's hypothesis)
**Mechanism.** `_RRecipe.primary` blocks the forward fill if any primary input is missing
([network.py:120-121](../../src/pysuto/impute/network.py)); the gate `if PR and not miss_PR`
([network.py:365](../../src/pysuto/impute/network.py)) needs all of ER's ~47 bilateral Anode inputs
known; coverage is 4.3 % → never all-known.
**Predicted signature.** ER→CER `network_forward` = 0; ER→CER prior ≈ 0; same for any deep-chain
process fed by sparse bilateral trade. **Falsified by** any `network_forward` on ER→CER, or a
non-zero ER→CER prior.

### H2 — Defect is in the solved values, not the prior
**Predicted signature if true.** Solved ER→CER ≠ CER→CP by orders of magnitude. **Falsified by**
solved ER→CER == CER→CP.

### H3 — `agg=true` products are network-only, so the parametric/eff cascade cannot rescue them
**Mechanism.** `_parametric_consumes` skips `agg=true` ([parametric.py:48]); only `effer` (non-agg)
is pairwise-fillable. **Predicted.** Smelting→Matte/CFR→Anode 100 % flat-prior; ER→CER gets a few
`parametric` where its CC input happens to be alive.

### H4 — Formula cascade (`anode_domestic`) is the intended path and is starved upstream
**Mechanism.** `anode_domestic` needs `CFR→Anode (CC)` (an `agg=true` product); when it is 0 the
formula aborts. **Predicted.** `anode_domestic` fires 0×; the live chain dies at the products.

---

## 6 — Test strategy

| H | experiment | expected if true |
|---|---|---|
| H1 | ER→CER method counts + `coverage` on Anode→ER | `network_forward`=0; coverage ≪ 100 % |
| H2 | compare solved `F_opt` ER→CER vs CER→CP | equal (defect not in solution) |
| H3 | `agg` flag on eff params + process-output method counts | Smelting/CFR 100 % flat-prior |
| H4 | `anode_domestic` fire count + domestic-chain trace | 0 fires; chain dead at products |

---

## 7 — Test results

| H | result | evidence |
|---|---|---|
| H1 | **confirmed** | ER→CER `network_forward` 0/1410; flat-prior 1349/1410; Anode→ER coverage 4.3 % (§3.2–3.3) |
| H2 | **rejected** (reframes the symptom) | solved ER→CER = CER→CP = 13,551 (2020 CC); the gap is prior (1.83) vs solved (§3.1) |
| H3 | **confirmed** | `effsmelting`/`effcfr` `agg=true`, skipped by parametric; Smelting/CFR outputs 100 % flat-prior (§3.4, §4.1) |
| H4 | **confirmed** | `anode_domestic` targets Mass, needs `CFR→Anode (CC)`=0 → 0 fires; Germany trace breaks at the products (§4) |

### 7.1 Verdict

User's hypothesis (Stage 2, imputation) **confirmed**, with the precision that the imputer *can*
propagate (CP shows `network_forward`); it is the **all-or-nothing primary-input determinacy gate
under sparse bilateral trade coverage** that blocks every `agg=true` process product. Those dead
products starve **both** intended fill paths — the network flow-balance and the
formula→conc→eff cascade — so ER→CER→CP (and Smelting→Matte, CFR→Anode) receive a
`flat_prior_noisy_zero` seed. The solver then rebuilds the segment from constraints (correct solved
values), but the prior is a floor → the rocket, and warm-start has nothing to carry. This is the
imputation-side origin of the refined-cathode pins flagged as the "next investigation" in the
[mass-collapse report](analysis_opt_20260611_075036_6d2e540f_20260611_mass-collapse-ratio-bounds.md).

---

## 8 — Proposed fix (design only — no code this session)

**Goal.** Give the `agg=true` products and ER→CER→CP a meaningful, non-zero prior so the warm-start
chain ([ADR 0009](../adr/0009-warm-start-prior-chain.md)) carries real values instead of `f=0,σ=1e6`.
Two parts, both required, written **generically** (not hard-wired to Anode→ER — also unblocks
Smelting→Matte and CFR→Anode).

### 8.1 Recipe-gate reclassification (load-bearing)
In [network.py](../../src/pysuto/impute/network.py): add a third input tier `bilateral_import` to
`_RRecipe` for cross-region Flow→Process legs (classify in `prepare()` via
`from_region != to_region`). In `_do_recipe` treat it like the existing SEM `secondary` tier — pool
when known, **count as 0 when missing, and exclude from `miss_PR`** so it does not block the forward
gate ([network.py:355,365](../../src/pysuto/impute/network.py)). Domestic (X==Y) legs stay blocking
`primary`. **Rationale:** a bilateral import leg with no BACI trade row genuinely means "no trade on
this route" → 0 is the honest value (the same trade-off [ADR 0004](../adr/0004-process-recipe-traversal-imputer.md)
made for SEM secondaries; [ADR 0005](../adr/0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md)
pooling precedent). This lets the network forward-fill the `agg=true` products from
domestic + covered-import inputs → CFR→Anode/Smelting→Matte revive → the Anode|Flow node balance
fills the domestic Anode→ER leg → `anode_domestic`/`effer` then propagate ER→CER → CER→CP.

### 8.2 `TradeImputer` (supporting)
New `src/pysuto/impute/trade.py` (`BaseImputer`), registered **first** in the cascade
([collect_aggregate_impute.py ~404](../../scripts/collect_aggregate_impute.py), before `network`).
For each cross-region Flow→Process **CC** edge whose BACI **Mass** twin is known in `f`, write
`f_CC = conc·f_Mass`, σ mirroring `network._emit_twins` ([network.py:438-441]); method
`"trade_seed"`. The Mass twin is already in `f_agg` on covered legs → **no new artifact, no BACI
feed access**; no-op when conc/BACI absent (backward-compatible). Factor
`NetworkImputer._load_twin_map` into a shared `conc_twin_map()` in `param_relation.py` for reuse.

**Why seeding alone is insufficient (key finding).** The dominant *domestic* Anode→ER leg has no
measurement anywhere (BACI is international; ICSG carries no anode usage — see below), so the gate
change (§8.1) is what actually lets the network close the node from the upstream product; §8.2 only
supplies the covered import legs. **Rejected:** extending the ICSG trade-share fan
([ADR 0002](../adr/0002-single-endpoint-rows-fan-out-not-replicate.md) option D) — `Anode` is in its
traded-sector list but ICSG carries **zero anode usage** (its only Use rows are Concentrate/Scrap→
Smelting, Matte/Scrap→Refining, Cathode→semis), so the fan is inert for Anode and runs at the wrong
stage (feed, not imputation). A blanket gate relaxation (all missing primaries→0) is **rejected** —
it would zero genuine domestic legs and break Smelting/MF; only cross-region import legs are zeroed.

### 8.3 Tiny commits
1. Extract shared `conc_twin_map()` (pure refactor; existing `test_network_imputer.py` green).
2. Add `bilateral_import` tier + `prepare()` classification (no behaviour change yet).
3. `_do_recipe` treats `bilateral_import` like SEM (the behaviour change).
4. Add `TradeImputer` + export.
5. Register `TradeImputer` first in the cascade.
6. ADR 0010 + README §6 test-map line + GLOSSARY.md vocab line.

### 8.4 Verification
- **Unit** (`tests/test_trade_imputer.py` or extend `tests/test_network_imputer.py`): 2–3 region
  `MockAcc` with `CFR→Anode→ER→CER→CP` + domestic `CFR→Anode`; seed BACI Mass on some import legs +
  Matte→CFR domestic, leave products NaN; assert `TradeImputer` writes the CC import legs and after
  `NetworkImputer` `ER→CER`/`CER→CP` are filled by `network_forward`/`network_balance` (not
  flat-prior) with `ER→CER ≈ eff·Σinputs`; negative test: conc absent → no-op.
- **End-to-end** (re-run imp→const→opt on the reference feeds): assert `ER→CER [CC]` now has
  substantial `network_forward` (currently 0) and the flat-prior share drops well below 1349/1410;
  assert const `a0` for ER→CER/CER→CP is within O(1) of solved `F_opt` (~13,551), not ~1.8; confirm
  CFR→Anode/Smelting→Matte likewise revive.

### 8.5 ADR
[ADR 0010](../adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md) records this
decision. Touches/extends ADR 0002 (Anode inert in the fan), ADR 0004 (new `bilateral_import` tier;
closes its rejected "single upstream gap blocks everything" case), ADR 0005 (pooled-eff recipe now
also pools seeded import legs), ADR 0009 (the enabler — warm-start now has a real ER→CER prior).

### 8.6 Risks
- Zeroing no-BACI import legs zeros genuine-but-unobserved trade (mitigation: domestic stays
  blocking; BACI has broad country coverage; solver mass-balance + soft `eff` can still widen).
- Regions where Matte→CFR is also fully uncovered stay flat-prior — strictly better than today,
  and improved further by the warm-start time carry once any year closes.

---

## Artefact sources

- opt: `data/proc/opt/opt_20260611_140957_24eeecb0/timesteps/t_*.npz` (`F_opt`, `cols_t`); diag PNGs `sd_r_08`/`sd_r_09`
- const: `data/proc/const/const_20260611_131746_485f4d5f/{a0.npy, sigma_a.npy}`
- imp: `data/proc/imp/imp_20260611_131319_928c86ee/{agg_result.npz (coverage, f_agg, feed_ids), imputation_log.parquet}`
- acc: `data/proc/acc/copper/{edge_index.csv, node_index.csv}`
- params: `data/input/params/eff/current/params.csv` (agg flags), `data/input/params/conc/current/params.csv`
- code: `src/pysuto/impute/network.py` (`_RRecipe`, `derive_process_recipes`, `_do_recipe` gate, `_do_flow_balance`, `_emit_twins`), `src/pysuto/impute/parametric.py` (`_parametric_consumes`), `src/pysuto/impute/formula.py` (`FormulaImputer`), `scripts/collect_aggregate_impute.py` (cascade + `*_domestic` relations)
- reproduction: all numbers via read-only `numpy`/`pandas` loads of the above (no scratch artifacts written)
- predecessor: `docs/rep/analysis_opt_20260611_075036_6d2e540f_20260611_mass-collapse-ratio-bounds.md` (this run = its "Run B")
