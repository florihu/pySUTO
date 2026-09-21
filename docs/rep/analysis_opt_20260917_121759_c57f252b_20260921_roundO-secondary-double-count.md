# Round O — the secondary double count is real, and removing it moves the cost, not the total

**Run measured:** `opt_20260917_121759_c57f252b` (label `roundO_secondary`, 30/30 Solved)
**Base:** `opt_20260916_124721_c4d1d944` (label `roundM_revert_verify`, sum-row z² 1,719,806)
**Chain:** `imp_20260917_114838_eebe842b` → `const_20260917_120521_71351e80` → the run
**Feeds:** icsg **v9** (was v8), baci v9 (netted), crude_ore v3 · **Params:** scrap_share **v2** (was v1)
**Date:** 2026-09-21 · **Branch:** `feat/region-resolved-concentrate-grade` · **ADR:** [0030](../adr/0030-secondary-smelter-copper-enters-the-chain-once.md)

**Verdict: the parse fix is correct and buys back 8.7 Mt of mine-chain deletion, but the fit gets
13.3 % worse and the cause is a structural gap the fix exposed — ICSG's smelter-scrap *level* is
never a scale anchor, only a share band. Do not merge as-is. Round P is named in §6.**

---

## 1. The feed gate — passed, cleanly

Run before any solver, on the feed alone:

| Check | v8 (base) | v9 | |
|---|---|---|---|
| rows changed | — | 611 of 8,113, **all** `Refining/Scrap/Use` | PASS |
| `Refining/Scrap/Use` world | 86,475 kt | 18,740 kt (−78.3 %) | as designed |
| Austria, both-booked in ÷ out, 2024 | 1.480 (30-y 1.45–1.99) | **1.000** (30-y 1.00–1.00) | PASS |
| world concentrate gap, 2024 | +3,450 kt | **−464 kt** | PASS (plan said ≈ −0.46 Mt) |
| `P_Cathode`/(refined+SXEW), 2024 | 1.000 | 1.000 | PASS |
| excess region-years (`S_sm·k > S_ref`) | — | 107, 1,945 kt, listed | PASS |

Top excess: Japan 19 y 693 kt, Bulgaria 29 y 588, Iran 23 y 352, Sweden 3 y 117, South Korea 11 y 105.
The nesting assumption holds everywhere else.

**Nothing in §§2–6 disputes this section.** The double count was real, the arithmetic that removes it
is right, and Austria — the motivating case — closes exactly.

## 2. What the fix bought

| | base | Round O | |
|---|---|---|---|
| world MF→Concentrate retention, 30 y | 0.873 | **0.894** | +2.1 pp |
| … 2020-24 | 0.836 | 0.853 | +1.7 pp |
| mine-chain deletion (ADR 0029 detector) | 52,871 kt | **44,197 kt** | **−16.4 %** |
| — of which *forced* by missing BACI export mass | 25,913 kt (49.0 %) | 26,421 kt (59.8 %) | +2.0 % |
| — of which **discretionary** (the model's choice) | 26,958 kt | **17,776 kt** | **−34.1 %** |
| Peru mine retention, 2020-24 | 0.884 | **0.923** | +3.9 pp |
| trade-dependent intake retained | 0.700 | 0.721 | +2.1 pp |

The headline is the third row split in two. Round O barely touches the deletion that ADR 0029 showed
is **forced arithmetic** — it cannot, that is a BACI mass problem — and removes **a third of the
deletion the model was choosing freely**. That is exactly what a true double count should do, and it
is the strongest evidence that the diagnosis was right.

The forced share rising to 59.8 % is not a regression: the numerator barely moved (+508 kt) while the
denominator fell. The export-mass gap is now **the** remaining term, as the plan predicted.

## 3. What it cost

| | base | Round O | |
|---|---|---|---|
| sum-row z² (soft scale totals, ADR 0006 family) | 1,719,806 | **1,949,396** | **+13.3 %** |
| solver status | 30/30 Solved | 30/30 Solved | — |

By sector: Anode **+87,939**, MF +45,013, Concentrate +40,714, CSXEW +333; Cathode −34,192,
Matte −2,121, Smelting −2,049.

**This is not a σ artifact.** On the 21,729 sum rows whose anchor value is identical in both runs, σ
is identical on **every one** (shrank on 0, grew on 0). The rise is residual, i.e. a genuine loss of
fit. Decomposed:

| rows | z² base | z² O | Δ |
|---|---|---|---|
| anchor unchanged (21,729) | 1,341,843 | 1,458,211 | **+116,368** |
| anchor changed (3,953) | 377,963 | 447,374 | +69,411 |
| **present only in Round O (146)** | — | 43,811 | **+43,811** |

The changed-anchor rows are the `Refining/Scrap/Use` family the fix intended to move; a worse fit
there is expected and not interesting. The other two blocks are.

### 3.1 The unchanged-anchor rise is three countries

Top (sector, region) movers on unchanged anchors:

| | base | O | Δ |
|---|---|---|---|
| **Anode / Slovakia** | 6,156 | **70,786** | **+64,630** |
| Anode / Belgium | 23,420 | 31,885 | +8,465 |
| ER + CER / Austria | 8,972 ×2 | 22,748 ×2 | +13,776 ×2 |
| CP / Austria | 9,118 | 16,438 | +7,321 |
| CFR / Austria | 2,663 | 9,608 | +6,945 |
| Anode / Austria | 3,864 | 10,190 | +6,326 |
| Anode / Belgium, CFR / Belgium, ER / Belgium | | | +14,573 |

Slovakia alone is **56 % of the unchanged-anchor rise**. Its anode anchor never moved
(c ≈ 57 kt, σ ≈ 0.97 kt) and its posterior anode production **collapses from ~43 kt to ~6.5 kt in
every one of the 30 years** — |z| from ~14 to ~52.

### 3.2 The 146 new rows are phantom mines in refinery countries

Rows that exist only in Round O, because Stage 2 now imputes non-zero activity where it imputed none:

| sector / region | n | z² | mean anchor |
|---|---|---|---|
| Concentrate + MF / Germany | 8 + 8 | 13,352 ×2 | 31.8 kt |
| Concentrate + MF / Spain | 2 + 2 | 5,549 ×2 | 24.9 kt |
| Smelting + Matte / Austria | 20 + 20 | 2,991 ×2 | 39.0 kt |
| Concentrate + MF / Belgium, Smelting + Matte / Slovakia | 18+18 | small | |

Germany's eight new concentrate/MF rows sit at |z| ≈ 41. Germany has no copper mine to speak of.
The model, deprived of the phantom scrap, went looking for the copper and put a small phantom **mine**
in its place.

## 4. The mechanism — the smelter route cannot pick up what the parse moved to it

ADR 0030 reassigns the overlap from the refinery (`Scrap→CFR`) to the smelter route. For that to be
mass-neutral, the smelter has to actually run. Scrap routing, kt CC pooled 2020-24:

| | Scrap→Smelting | Scrap→CFR | Matte→CFR | **CFR→Anode** |
|---|---|---|---|---|
| China base → O | 17,684 → 19,033 | 9,097 → 3,254 | 45,239 → 46,530 | 53,250 → **48,788** (−8 %) |
| Japan base → O | 1,430 → 2,261 | 1,766 → 134 | 6,005 → 7,489 | 7,662 → **7,505** (−2 %) |
| Germany base → O | 545 → 1,191 | 1,239 → 423 | 1,643 → 2,473 | 2,845 → **2,859** (+0 %) |
| Belgium base → O | 533 → 750 | 537 → 107 | 620 → 807 | 1,143 → **903** (−21 %) |
| Austria base → O | 8 → 98 | 374 → 163 | 8 → 115 | 377 → **275** (−27 %) |
| **Slovakia base → O** | **0.3 → 0.3** | 161 → 10 | 1 → 25 | 161 → **34** (−79 %) |

Where the smelter node has real throughput (DE, JP, CN) the substitution works and the anode output
is preserved — **the fix is sound there**. Where it does not, the copper is simply lost.

Slovakia is the clean case. ICSG reports its smelter consuming **~28 kt/yr of scrap** (810.8 kt over
29 years). The model routes **0.06 kt/yr — in both runs.** That level is never enforced, because
ICSG's `Smelting/Scrap` enters the model **only as a share band**, `0.950 ≤ Σ(Scrap→Smelting) /
Σ(all→Smelting) ≤ 1.000`. A share band over a node with ~zero throughput is satisfied trivially.
Nothing makes Slovakia's smelter run.

Meanwhile `scrap_share` **v2 — regenerated from the post-fix series, as the plan required** — flipped
Slovakia's refinery band:

```
v1   Scrap share of CFR inputs, Slovakia:  0.950 ≤ · ≤ 1.000   (own series)
v2   Scrap share of CFR inputs, Slovakia:  0.000 ≤ · ≤ 0.280   (own series)
```

Slovakia's ICSG refinery-scrap was cut 89 % by the ADR 0030 rule (its smelter route consumed almost
all of it), so the *share* derived from it collapsed, so the band now **forbids** the refinery from
running on scrap. Slovakia's refinery is a secondary refinery. Both routes are closed and the anode
anchor is missed by 50 σ.

This is one defect with two faces, and it is not in the parse rule:

> **The parse moves copper from a route the model anchors by level to a route the model anchors only
> by share.** The band regeneration then narrows the vacated route. Regions whose smelter node is
> small or idle in the model have nowhere left to put the copper.

Band occupancy confirms the pressure is general, not only Slovakian — the big secondary smelters are
all pinned against their ceilings in Round O: China 0.398 of [0.115, **0.402**], Austria 0.988 of
[0.950, **1.000**], Japan 0.291 of [0.046, **0.306**].

## 5. Acceptance table

| # | Criterion (plan of 2026-09-17) | base | Round O | |
|---|---|---|---|---|
| 0 | feed gate: one row class, Austria 1.00, gap ≈ −0.46 Mt, P ratio 1.000 | — | all four | **PASS** |
| a | mine-chain deletion ≪ 12.7 % | 12.7 % | 10.6 % | **PARTIAL** — fell, not "≪" |
| a′ | discretionary deletion (not in the plan; the sharper test) | 26,958 kt | 17,776 kt | **PASS** (−34 %) |
| b | posterior secondary share → ≈ 0.17 | 0.300 | 0.257 | **FAIL** (prior is 0.164; model still adds +9.3 pp) |
| c | China's smelters off the 0.40 ceiling | 0.379 | 0.398 | **FAIL** — harder against it |
| d | Chile retention up, no cone change | 0.557, capped 19/30 | 0.561, capped 19/30 | **FAIL** — unchanged |
| d′ | Peru | 0.884 | 0.923 | **PASS** |
| e | no new node above 1.05 intake retention | — | Kazakhstan | **FAIL** (one) |
| f | 30/30 solved | 30/30 | 30/30 | **PASS** |
| g | sum-row z² honest | 1,719,806 | 1,949,396 | **FAIL** (+13.3 %, diagnosed §3-4) |
| h | excess region-years listed, not silently dropped | — | 107 listed | **PASS** |

Secondary shares, refinery-side input, 2020-24 pooled — prior → posterior:

| | base | Round O |
|---|---|---|
| world | 0.258 → 0.300 | **0.164 → 0.257** |
| China | 0.34 → 0.48 | 0.23 → 0.44 |
| EU-14 | 0.49 → 0.46 | 0.37 → 0.40 |
| JP/KR/TW | 0.37 → 0.40 | 0.24 → 0.29 |
| United States | 0.06 → 0.07 | 0.05 → 0.05 |

The feed now says 0.164, close to ICSG's ~0.17. **The model still overrides it to 0.257.** Round O
fixed the prior; it did not fix the posterior's appetite for secondary copper, because the thing that
creates that appetite — the refinery anchors it cannot otherwise meet — is untouched.

Other ledgers, for the record: cathode netting `T_applied` 4,635 → 4,716 kt CC (+81, the ADR 0027
cathode term); MESC cumulative 15.8 → **14.3 Mt** (prior 5.8 → 6.1) — better, still 2.3× prior;
USC 3.1 → 3.0; PSC −2.0 → −2.2; hubs all fall (Singapore 2.1 → 1.9, UK 2.6 → 2.2, NL 2.0 → 1.7).

## 6. Recommendation — Round P, then re-measure

**Do not merge Round O as it stands.** Not because it is wrong — §1 and §2 say it is right — but
because it ships a 13.3 % fit regression whose cause is known, local, and fixable, and because
criteria b, c and d all fail for the *same* reason: the copper the fix freed has nowhere to go.

The fix is **not** to soften the parse rule. It is to close the gap the parse rule exposed:

1. **Anchor the smelter-scrap level.** ICSG `Smelting/Scrap` should become a soft scale total on
   `Σ(Scrap→Smelting)` per region-year — the same treatment `Refining/Scrap` already gets — in
   addition to the share band. This is the single change that would let Slovakia's, Austria's and
   Belgium's smelters carry what ADR 0030 assigned to them. Expect the Anode/MF/Concentrate z² to
   come back and the secondary share to fall toward 0.17 on its own.
2. **Do not derive a band from a series the same commit rewrote.** `scrap_share` v2's CFR bands are
   computed from the post-fix direct-route series; for regions where that series was clipped to near
   zero the band becomes a prohibition. Either floor the band width, or derive the CFR band from
   *total* secondary refinery input and let the level anchors split the routes.
3. Re-run the chain and re-measure with this same script. Kazakhstan's >1.05 intake retention and
   Chile's unchanged hard cap (19/30) should be re-checked then, not now.

Chile did **not** recover (0.557 → 0.561) and remains hard-capped in 19 of 30 years. The Round O
thesis that the secondary double count was Chile's mechanism is **not** supported: Chile's deletion
is the forced, BACI-export-mass kind, and §2 shows that term is the one Round O cannot touch.

## 7. Reproduction

```bash
python scripts/explo/roundO_secondary_diag.py \
    opt_20260916_124721_c4d1d944 opt_20260917_121759_c57f252b     # §1,2,5 and the ledgers
```
§3 and §4 were produced by three throwaway drills (z² split by anchor movement; σ decomposition;
scrap routing by region) over `data/proc/opt/<run>/diag/constraints_<year>.csv` joined to the const
run's `row_meta.parquet`, and `data/input/params/scrap_share/{v1,v2}/params.csv`.
