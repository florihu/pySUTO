# Round R — BACI's quantity contradicts its own value, and the forced wall comes down

**Run measured:** `opt_20260922_092530_f21bdbe7` (label `roundR_quantity_reconstruction`, 30/30 Solved, `prior_mode` cold)
**Against:** Q `opt_20260921_114721_de4c5cc5`, P.1 `opt_20260921_101621_cd0c0ae2`, O `opt_20260917_121759_c57f252b`, base `opt_20260916_124721_c4d1d944`
**Chain:** `imp_20260922_090803_6da0f169` → `const_20260922_091301_c6edc72d` → the run
**Feeds/params:** baci **v11** (was v9) — gross **v10** reconstructed, then ADR 0027 re-netted on it; icsg v10, crude_ore v3, conc v8, ore_grade v7, scrap_share v3 unchanged
**Date:** 2026-09-22 · **ADR:** [0033](../adr/0033-baci-quantity-contradicts-its-own-value.md)

**Verdict: R delivers the criterion five rounds could not touch. ADR 0029's forced deletion falls
26,959 → 12,315 kt CC (−54 %) and its share of all deletion 68.4 % → 33.5 %; Chile's forced term
goes 14,270 → 295, from 40.6 % of its own deletion to 1.0 %, and its hard-capped years 19 → 5. The
pre-registered null held exactly: ROW and Zambia, which the detector does not flag, did not move.**

**But read the result precisely. The round did not mainly *reduce* deletion — it *reclassified* it.**

---

## 1. What the round actually did

| run | deleted | forced | discretionary | forced share |
|---|---|---|---|---|
| base | 52,871 | 25,913 | 26,958 | 49.0 % |
| O | 44,197 | 26,421 | 17,776 | 59.8 % |
| P.1 | 38,934 | 26,411 | 12,523 | 67.8 % |
| Q | 39,394 | 26,959 | 12,435 | 68.4 % |
| **R** | **36,807** | **12,315** | **24,492** | **33.5 %** |

Total deletion falls only 6.6 %. What moves is **~14,600 kt CC from forced to discretionary** —
Chile alone 44.6 % → 1.0 % forced, with its deletion nearly flat (32,016 → 29,410 kt).

That is the point, and it should not be dressed up as more. The forced term is *arithmetic*:
`cap = 0.35·X + U` bounds what a mine node can discharge, and `Σ max(0, P − cap)` is what no
optimiser can reach. Five rounds cut the discretionary term and left it standing. Round R removes
the wall by correcting `X`; what is left is a **fit** problem, of the kind this project's levers
(σ, anchor placement, pricing) actually address.

**The mechanism is mechanical, and the diagnostic proves it.** §5 recomputes every run's forced term
against the *new* feed and gives Q **12,339** — i.e. Q's own posterior, scored against the corrected
export mass, is already almost where R lands. The correction raises the ceiling; it does not need
the solver's cooperation to do so. What the run then demonstrates is that the model **uses** the
headroom: mine deletion 39,394 → 36,807, Chile 32,016 → 29,410, Chile's retention 0.574 → **0.617**.

## 2. Acceptance

| # | criterion | Q | R | |
|---|---|---|---|---|
| **R-a** | ADR 0029 forced deletion | 26,959 | **12,315** | ✓ −54 % |
| R-b | Chile forced shortfall | 14,270 | **295** | ✓ |
| R-c | Chile hard-capped years | 19 of 30 | **5 of 30** | ✓ |
| R-d | Chile MF→Concentrate retention | 0.574 | **0.617** | ✓ +4.3 pp |
| R-e | common-set sum-row z² | 1,698,662 | 1,705,881 | ✓ +0.4 %, still −0.7 % vs base |
| R-f | world MF→Concentrate retention | 0.905 | **0.912** | ✓ |
| R-g | intake retention > 1.05 | ROW, Kazakhstan | ROW, Kazakhstan | ✓ did not grow |
| **R-h** | ROW forced share (**pre-registered null**) | 7,576 | **7,589** | ✓ unmoved |
| R-i | status | 30/30 | 30/30 | ✓ |

**R-h is the check that matters most.** The detector flags neither ROW nor Zambia, so both were
predicted *not* to move before the run: ROW 7,576 → 7,589, Zambia 1,418 → 1,452. A correction that
merely shuffled mass would not respect a boundary it was never told about.

**R-e is the honest cost.** z² rises 0.4 % — the price of asserting 82,320 kt of mass the feed did
not record. By sector on the common set, base → R: Anode −38,812, CFR −36,856, Cathode −14,781,
Matte −8,901 against Ore +41,447, Concentrate +29,605, MF +25,056. Trade-dependent intake retention
rises 0.743 → **0.761**, China 0.669 → 0.694; self-sufficient slips 0.925 → 0.918.

## 3. The gates, and the design they corrected

**Two gates, not one — the gate caught this, not the plan.** Flagging per-leg alone reaches into
years whose *aggregate* fraction is in band: Chile came out flagged in all 30 years with 121,630 kt
recovered. But the ICSG cross-check licenses the **aggregate**, not the leg — in a clean year the
recorded mass already matches the value, so lifting a leg there **fabricates** mass, which is
exactly what R-g exists to catch. Adding the exporter-year gate gives **18 years and 82,320 kt**,
within **2 %** of the independent cross-tab (Chile 58,517 vs 57,944; `g_ref` 0.274 vs 0.273).

**`g_ref` is calibrated on exporter-years.** A per-leg median is dragged low by small legs (0.257 vs
0.274) and would inflate mass, since `q = v/(P_cu·g_ref)`.

**The threshold does not carry the result.** Relief measured at 0.35 / 0.40 / 0.45 / 0.50 / 0.60 is
**61 / 61 / 59 / 57 / 54 %**, and Chile's forced term closes at every one. 0.35 is kept because it
is ADR 0028's own cone ceiling — a leg above it is one the model cannot represent anyway.

**ADR 0027's σ formula was rejected on measurement**, not on taste: `√(q_obs² + (k·Δ)²)/q_new` gives
raw scales 0.839 / 1.028 / 0.960 here, all clipped to ~1.0 and *non-monotone*, because the
denominator outruns the numerator when a leg is multiplied by 2–3×. Replaced by the `conc` row's
declared σ (cv 0.1364) widened per region by its own in-band spread: Chile 1.194 at the floor,
Belgium 1.88, Italy 2.18, Austria 2.56.

Pipeline gates all passed: flagging in every five-year block (126/187/111/172/96/189 legs — the
"2021-onward" reading was wrong); both scale files composed (26,037 + 814 root legs → 40,352 edges,
product capped at 5.00, so ADR 0027's σ survives); netting re-derived on corrected mass
(78,312 → 62,946 rows, 1,145,803 → 1,118,619 kt); feed 100 % matched.

## 4. What is open

1. **ROW's 7,589 kt CC, now 62 % of the whole forced term** and 98.9 % of ROW's own deletion. Not
   this defect — predicted and confirmed. [ADR 0014](../adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md)'s
   folded internal trade is the remaining candidate, and it is now the largest single item.
2. **Belgium appears hard-capped at 1,029 kt CC forced against 300 deleted (343 %)** — new, and
   Belgium has no mines. This is the fabricated domestic `Concentrate→Smelting` edge Round Q
   identified (505 kt, zero feed-observed) surfacing in the detector.
3. **Chile's deletion is now 29,410 kt CC and 99 % discretionary.** The wall is gone; the fit
   problem is not. This is the first time that sentence can be written.
4. **`stock_bounds_study.py`** still carries the ADR 0032 source-vs-root bug, untouched on purpose.
5. The parse drops rows with a value but no quantity — **0.307 % of value** — now logged, not silent.
