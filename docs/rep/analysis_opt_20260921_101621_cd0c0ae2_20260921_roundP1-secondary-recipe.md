# Round P.1 — the pure-secondary recipe closes, and the stack becomes mergeable

**Run measured:** `opt_20260921_101621_cd0c0ae2` (label `roundP1_secondary_recipe`, 30/30 Solved, `prior_mode` cold)
**Against:** P.0 `opt_20260921_090618_1ceb5eed`, Round O `opt_20260917_121759_c57f252b`, base `opt_20260916_124721_c4d1d944`
**Chain:** `imp_20260921_095700_8285cb6c` → `const_20260921_100253_fe0ef3ad` → the run
**Feeds/params:** unchanged — icsg v9, baci v9, crude_ore v3, scrap_share v2
**Date:** 2026-09-21 · **ADR:** [0010 amendment](../adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md) · [0031](../adr/0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md)

**Verdict: P.1 delivers. Slovakia 0.10 → 0.96, its anode 7.81 → 58.53 kt, and the common-set z² falls
+12.4 % → +7.4 % — below Round O's own +10.9 %. Every acceptance criterion except P-g now passes.
The O + P.0 + P.1 stack is, on the governing criterion, better than Round O alone.**

---

## 1. What changed

Round P.0's diagnosis was right that Slovakia was strangled and wrong about where. The wall was not
the `scrap_share` band (its Smelting ceiling is 1.000) and not ADR 0019's contradicted-node rule
(its detector needs `inflow > 1 kt` and fires on **1 node-year in the whole model**). It was Stage 2.

`NetworkImputer._do_recipe`'s gate — *"a recipe with NO known primary keeps the gate closed (no
fabricated zeros)"* — is right about a data gap, but a **100 %-secondary smelter has no primary by
construction**, and its input is not missing: it is measured, on the SEM scrap leg. So Slovakia's
recipe never fired, its chain carried nothing, and Stage 3 built `usetotout Matte/Slovakia` from that
zero at `c = 0.011 kt, σ = 0.001` — a near-*hard* zero that no feed-backed anchor could cross.

The [ADR 0010 amendment](../adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md) lets the
relaxed phase close on `S = Σ(known secondaries) + Σ(known imports)` when no primary is known but a
secondary is. Three guards keep the original intent: relaxed phase only; `sec_sum > 0`, so no known
input of any kind still blocks (CP has no SEM leg and is untouched); and `relaxed_close` still bars
backward-filling the primary it assumed absent.

## 2. The Stage-2 gate — blast radius

**445 of 854,460 edges moved (0.052 %)**, in two regions.

**Slovakia**, 21 years, exactly as designed — the chain now closes:
`Smelting→Matte` 0 → **714 kt CC**, `Matte→CFR` 0 → **714**, `CFR→Anode` 0 → **712**,
`Smelting→Slag` 0 → 26, and the downstream `ER→CER→CP→Cathode` 0 → ~100.

**Italy**, 5 years (2012, 2014, 2016-18) — *not* predicted, and it looked like a regression in the
prior: `Mining→Ore` 16.59 → 3.92, `Matte→CFR` 11.95 → 0, `CP→Cathode` 9.49 → 0, and a new
`Cathode→USC` of −34.16. Italy's CFR band is `[0.950, 1.000]` — one of exactly two regions pushed
there by the `.clip(upper=1.0)` defect — so it is *recorded* as ~100 % scrap-fed and the new close
fires on it, displacing the primary route its chain had been running on.

**At the posterior this is not a regression: Italy's common-set z² falls by 2,320.** The prior change
looks odd and the fit improves. Worth knowing, not worth blocking on — but it does mean the two
halves of P.1 are less independent than planned: the clip defect is now load-bearing for Italy.

## 3. Acceptance

| # | criterion | Round O | P.0 | **P.1** | |
|---|---|---|---|---|---|
| P-a | pairs with posterior < 5 % of reported | 0 of 44 | 0 | **0** | **PASS** |
| P-b | Slovakia/Smelting ratio > 0.50 | 0.089 | 0.10 | **0.96** | **PASS** |
| P-c | world posterior ÷ reported → 1.00 ± 0.05 | 1.191 | 1.027 | **1.032** | **PASS** |
| P-d | **common-set z² must fall** | +10.9 % | +12.4 % | **+7.4 %** | **PASS** |
| P-e | new-family z² reported | — | 206,531 | **154,513** | **PASS** |
| P-f | Slovakia anode > 40 kt | 7.86 | 7.81 | **58.53** | **PASS** |
| P-g | world secondary share → ≈ 0.17 | 0.257 | 0.220 | **0.223** | **PARTIAL** |
| P-h | discretionary deletion must not rise | 17,776 kt | 12,455 | **12,523 kt** | **PASS** (flat) |
| P-i | China off the 0.402 ceiling | 0.398 | 0.341 | **0.38 share** | **PASS** |
| P-j | 30/30 Solved | 30/30 | 30/30 | **30/30** | **PASS** |

Slovakia's own rows, 2024 — the whole round in one table:

| row | c | posterior | z at P.0 | **z at P.1** |
|---|---|---|---|---|
| `smelterscrap` Smelting | 46.634 | 46.366 | −58.7 | **−0.34** |
| `usetotout` Anode | 57.190 | 58.529 | −50.7 | **+1.37** |
| `usetotout` Matte | 45.002 *(was 0.011)* | 47.287 | +0.9 | **+2.98** |

Common-set z² by region, P.0 → P.1: **Slovakia −122,346**, Austria −16,989, Italy −2,320, total
**−142,967**; the largest regression anywhere is China +1,371.

## 4. The stack, base → O → P.0 → P.1

| | base | O | P.0 | **P.1** |
|---|---|---|---|---|
| **common-set sum-row z²** | 1,718,085 | 1,905,519 | 1,931,717 | **1,844,644** (+7.4 %) |
| posterior ÷ ICSG-reported scrap | 1.318 | 1.191 | 1.027 | **1.032** |
| (route, region) pairs below 0.50 | 5 | 2 | 2 | **1** |
| region-years reported > 1 kt, posterior < 5 % | 79 | 46 | 28 | **15** |
| invented direct-to-refinery scrap | 7,652 kt | 164 | 185 | **198 kt** |
| world secondary share (prior 0.164) | 0.300 | 0.257 | 0.220 | **0.223** |
| mine-chain deletion | 52,871 kt | 44,197 | 38,876 | **38,934 kt** |
| — forced (ADR 0029) | 25,913 | 26,421 | 26,421 | **26,411** |
| — **discretionary** | 26,958 | 17,776 | 12,455 | **12,523 kt** |
| MF→Concentrate retention, 30 y | 0.873 | 0.894 | 0.906 | **0.906** |
| Chile mine retention | 0.557 | 0.561 | 0.578 | **0.571** |
| Peru | 0.884 | 0.923 | 0.953 | **0.945** |
| new nodes above 1.05 intake retention | — | Kazakhstan | ROW, Kazakhstan | **ROW** |

The z² by sector on the common set (base → P.1): Cathode −13,824, Matte −1,462, CP −1,276, Smelting
+184, Concentrate +21,943, MF +29,737, Ore +41,107. **Anode has left the table entirely** — it was
+87,980 at P.0. What remains is Ore / MF / Concentrate: the mine chain, i.e. the export-mass gap.

## 5. Recommendation

**Merge the stack — Round O + P.0 + P.1 together.** It is now better than Round O alone on the
criterion that governs (+7.4 % vs +10.9 % common-set z²), and it carries: 13.9 Mt less mine-chain
deletion, **half** the discretionary deletion, world scrap fidelity within 3 % of ICSG, the phantom
direct-to-refinery scrap gone (7,652 → 198 kt), one remaining collapsed route instead of five, and
Kazakhstan's fabricated node gone.

It is still +7.4 % against base. That residual is honest and is not the scrap arc's to pay: it sits
in Ore / MF / Concentrate, which is [ADR 0029](../adr/0029-the-concentrate-grade-is-not-the-lever.md)'s
export-mass gap — **67.8 % of all remaining deletion, and a term no round in this arc has moved by a
single tonne** (25,913 → 26,421 → 26,421 → 26,411).

Two items explicitly deferred, both narrow and both measured:

1. **Belgium/CFR**, the one remaining pair below 0.50 (0.17). Band-blocked, not level-blocked:
   headroom `(node input × ceiling 0.195) ÷ reported = 0.216`, 22 of 29 years. Predicted before P.0
   and unchanged by both rounds, exactly as stated.
2. **The `scrap_share` derivation defects** — `.clip(upper=1.0)` forcing Argentina and Italy to a hard
   `[0.95, 1.00]`, and the absence of any band-width floor. Italy's §2 behaviour shows this is now
   load-bearing, so it should be done before the next feed change, not after.

Then the export-mass gap, which is the only thing left that can move Chile.

## 6. Reproduction

```bash
python scripts/explo/roundP1_imp_gate.py imp_20260917_114838_eebe842b imp_20260921_095700_8285cb6c
python scripts/explo/roundP_gate.py const_20260921_100253_fe0ef3ad
python scripts/explo/roundO_secondary_diag.py opt_20260916_124721_c4d1d944 \
    opt_20260917_121759_c57f252b opt_20260921_090618_1ceb5eed opt_20260921_101621_cd0c0ae2
```
