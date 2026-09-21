# Round P.0 — the scrap level is now stated, and the world honours it; two regions cannot

**Run measured:** `opt_20260921_090618_1ceb5eed` (label `roundP_scrap_level`, 30/30 Solved, `prior_mode` cold)
**Against:** `opt_20260917_121759_c57f252b` (Round O) and `opt_20260916_124721_c4d1d944` (base)
**Chain:** `imp_20260917_114838_eebe842b` (reused) → `const_20260921_085252_511ae089` → the run
**Feeds/params:** unchanged from Round O — icsg v9, baci v9, crude_ore v3, scrap_share v2
**Date:** 2026-09-21 · **Branch:** `feat/region-resolved-concentrate-grade` · **ADR:** [0031](../adr/0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md)

**Verdict: P.0 does what it was built to do at world scale — scrap fidelity 1.191 → 1.027, mine-chain
deletion −5.3 Mt, Chile's first real movement in the whole arc — but it fails its governing criterion
(common-set z² rose +10.8 % → +12.3 %) and does NOT fix the two regions that motivated it. Both are
now diagnosed, and neither is blocked by the thing P.0 fixed. Hold with Round O; P.1 has two precise,
disjoint targets.**

---

## 1. The gate (no solve)

All six checks passed on the first build and reproduced exactly on the rerun:

| check | result |
|---|---|
| G1 rows by variant | `smelterscrap` **531**, `directrefscrap` **483** |
| G2 CC only | ✓ (0 Mass rows) |
| G3 population re-derived from `f_agg` + coverage | f_agg says 1,014, row_meta has **1,014** |
| G5 row exists ⇔ feed observed a level | **0** mismatches either direction |
| G4 ADR 0025 σ floor, all 30 y | min active σ_c = **0.3002** |
| G6 no row spans both routes | 0 |

G5 is the structural form of the requirement *"Smelting & Refining are still modelled with scrap
inflow"*: rows exist in exactly the region-years a feed observed and nowhere else, so no region-year
was silently forbidden scrap. 24 of 47 regions carry no scrap anchor at all and are untouched.

## 2. What P.0 bought

| | base | Round O | **P.0** |
|---|---|---|---|
| world posterior ÷ ICSG-reported scrap | 1.318 | 1.191 | **1.027** |
| (route, region) pairs below 0.50 | 5 | 2 | **2** |
| pairs below 0.05 (route closed) | 2 | 0 | **0** |
| region-years reported > 1 kt, posterior < 5 % | 79 | 46 | **28** |
| world secondary share, posterior (prior 0.164) | 0.300 | 0.257 | **0.220** |
| China smelter scrap share vs its 0.402 ceiling | 0.379 | **0.398** (on it) | **0.341** (off it) |
| mine-chain deletion | 52,871 kt | 44,197 kt | **38,876 kt** |
| — forced by missing BACI export mass | 25,913 | 26,421 | **26,421** (flat) |
| — **discretionary** | 26,958 | 17,776 | **12,455 kt** |
| world MF→Concentrate retention, 30 y | 0.873 | 0.894 | **0.906** |
| **Chile mine retention, 2020-24** | 0.557 | 0.561 | **0.578** |
| Peru | 0.884 | 0.923 | **0.953** |
| trade-dependent intake retained | 0.700 | 0.721 | **0.746** |

Two of these deserve emphasis.

**The world now honours ICSG's scrap levels almost exactly** — 1.027 against a target of 1.00 ± 0.05,
from 1.318 at base. The model had been running 19–32 % more scrap than the feed reports, and simply
stating the level removed that. China's smelter, which ran 54 % above its reported scrap, now sits at
1.24 and has come **off** the band ceiling it was pinned against in Round O (0.398 of 0.402 → 0.341).

**Chile moved.** 0.557 → 0.561 → **0.578**. Every previous round left Chile flat; the mine-chain arc
has been trying to move it since Round L. It is still hard-capped in 19 of 30 years and its deletion
is still 32,081 kt, so this is a first crack, not a fix — but it is the first crack.

The discretionary deletion — total minus the ADR 0029 forced term — is now **12,455 kt**, less than
half its base value, while the forced term has not moved a tonne (25,913 → 26,421 → 26,421). The
forced share is 68.0 %. **The export-mass gap is now unambiguously the whole remaining problem.**

## 3. What it cost — and why the governing criterion fails

| | common-set z² | vs base |
|---|---|---|
| base | 1,719,806 | — |
| Round O | 1,905,585 | +10.8 % |
| **P.0** | **1,931,784** | **+12.3 %** |

**P-d fails.** The common row set (25,682 ids present in all three runs) is the only comparable basis
— P.0 adds 1,014 rows and Round O added 146, so the totals 1,719,806 / 1,949,396 / 2,183,809 answer a
different question and are reported only as bookkeeping. Incidentally this corrects Round O's own
headline: its regression was **+10.8 %**, not the +13.3 % its REP reported, the rest being its own new
rows.

By sector on the common set (vs base): Anode **+87,980**, Ore +41,179, MF +29,412, Cathode −25,363.
Against Round O the Anode figure is essentially unchanged (+87,939 → +87,980): **P.0 did not make
Anode worse, it simply failed to fix it.** The +26,199 deterioration is diffuse — Cathode +8,829,
Matte +5,033, MF +3,310, Ore +789 — not one broken thing.

The new rows themselves cost **252,025** z²: `smelterscrap` 139,104 over 531 rows (mean |z| ≈ 16),
`directrefscrap` 67,427 over 483. The model is not honouring the anchors quietly; it is paying a large
penalty on them. §4 is why.

## 4. The two regions that failed, and their two different walls

`P-b` required Slovakia/Smelting past 0.50. It went **0.089 → 0.10**. Its anode is unchanged
(7.86 → 7.81 kt against a 57.19 kt anchor). Belgium/CFR went 0.18 → 0.17.

### 4.1 Slovakia — strangled by a fabricated zero on an intermediate node

Slovakia 2024, every CC scale-total row on the chain:

| row | `c` | posterior | σ_c | z |
|---|---|---|---|---|
| `smelterscrap` Smelting | 46.634 | 0.012 | 0.795 | **−58.7** |
| `usetotout` Anode | 57.190 | 7.806 | 0.975 | **−50.7** |
| **`usetotout` Matte** | **0.011** | 0.012 | **0.001** | +0.9 ✓ |

The chain is `Scrap→Smelting → Matte → CFR → Anode`. Slovakia's **Matte node total output is pinned
at 0.011 kt with σ 0.001**. No copper can cross it, so both feed-backed anchors — the new scrap level
and the BACI-derived anode export — are violated at |z| ≈ 50-59 while the near-zero pin is satisfied
to within 0.9σ. Violating the Matte pin by even 0.05 kt would cost z ≈ 50 on its own.

That pin is a **fabricated** group: no feed covers Slovakia's matte, so under ADR 0016 it keeps the
tight cv and **ADR 0025's 0.3 kt floor deliberately does not apply** (Round K measured that a global
floor lets fabricated chains grow, and rejected it). This is the cost of that decision, visible for
the first time: a fabricated near-zero scale total on an intermediate flow node is a near-hard zero,
three orders of magnitude tighter in absolute terms than the feed-backed anchors either side of it.

**How general?** 889 of 26,842 active sum rows (3.3 %) are near-hard zero pins (|c| < 1 kt, σ_c < 0.01;
median |c| 0.032 kt, median σ 0.00133), concentrated in Matte (171), MF (115), Concentrate (77).
But only **11 region-years** actually strangle a scrap anchor this way — **639 kt of the 153,415 kt
anchored, 0.4 %** — all of them Slovakia (9) and Belgium (2). So it is a narrow defect with an
outsized effect on exactly the two regions P.0 targeted.

### 4.2 Belgium — blocked by the band ceiling, as predicted before the run

Belgium/CFR headroom `(node input × ceiling) ÷ reported` = **0.216** (ceiling 0.195, blocked in 22 of
29 years). This was stated before the run and excluded from acceptance for that reason: honouring
Belgium's 123 kt/yr of direct-to-refinery scrap needs its fire refinery to grow 4.5×, which no level
anchor can do. Unchanged at 0.216 after P.0, exactly as predicted.

By contrast Slovakia's Smelting ceiling is **1.000** — the band never blocked it, which is why §4.1
had to look elsewhere.

## 5. Acceptance

| # | criterion | Round O | P.0 | |
|---|---|---|---|---|
| P-a | pairs with posterior < 5 % of reported | 0 of 44 | **0** | **PASS** |
| P-b | Slovakia/Smelting ratio > 0.50 | 0.089 | **0.10** | **FAIL** — §4.1 |
| P-c | world posterior ÷ reported → 1.00 ± 0.05 | 1.191 | **1.027** | **PASS** |
| P-d | **common-set z² must fall** | +10.8 % | **+12.3 %** | **FAIL** — governs the merge |
| P-e | new-family z² reported | — | 139,104 + 67,427 | **PASS** (reported) |
| P-f | Slovakia anode > 40 kt | 7.86 | **7.81** | **FAIL** — §4.1 |
| P-g | world secondary share → ≈ 0.17 | 0.257 | **0.220** | **PARTIAL** |
| P-h | discretionary deletion must not rise | 17,776 kt | **12,455 kt** | **PASS** (−30 %) |
| P-i | China off the 0.402 ceiling | 0.398 | **0.341** | **PASS** |
| P-j | 30/30 Solved | 30/30 | **30/30** | **PASS** |

Regressions to record: **ROW** joins Kazakhstan above 1.05 intake retention (a new fabricated node),
and Cathode CC retention slips 0.984 → 0.966.

## 6. Recommendation

**Hold P.0 with Round O; do not merge yet.** P-d governs, and it failed. But the pre-stated priority
rule anticipated a P-b ✓ / P-d ✗ split, and what happened is *not* that: the world-level fidelity
target passed handsomely while the two motivating regions failed for reasons now precisely known and
disjoint from each other and from the level.

The honest reading is that P.0's thesis was **right but incomplete**. Stating the input level is
necessary and it demonstrably works — 42 of 44 (route, region) pairs now track ICSG, the world sits at
1.027, and the discretionary deletion halved. It is not *sufficient* when the chain the level feeds is
blocked further down.

Two disjoint P.1 targets, both measured and both narrow:

1. **The fabricated-zero pin** (Slovakia, 9 region-years; Belgium, 2). A fabricated scale total whose
   `c` is near zero becomes a near-hard zero because ADR 0016 gives fabricated groups the tight cv and
   ADR 0025's floor is feed-backed-only. The narrow fix: when a fabricated group's own node sits on a
   chain between two feed-backed anchors that contradict it, it should lose authority — the ADR 0019
   *contradicted node* machinery already exists for exactly this shape and is currently inert
   (`UNCLOSED_ANCHOR_CV = None`). This is a σ question, not a new constraint.
2. **The `scrap_share` band ceiling** (Belgium/CFR, 22 region-years), plus the two known derivation
   defects — `.clip(upper=1.0)` forcing Argentina and Italy to a hard `[0.95, 1.00]`, and the absence
   of any band-width floor.

Do **not** widen the new anchors to make the z² look better: at 1.027 world fidelity they are stating
the feed correctly, and the penalty they carry is the system telling us where it disagrees.

## 7. Reproduction

```bash
python scripts/explo/roundP_gate.py const_20260921_085252_511ae089     # §1, no solve
python scripts/explo/roundO_secondary_diag.py \
    opt_20260916_124721_c4d1d944 opt_20260917_121759_c57f252b opt_20260921_090618_1ceb5eed
```
§4.1's row table is `data/proc/opt/<run>/diag/constraints_2024.csv` joined to the const run's
`row_meta.parquet`; the 889-pin and 11-strangled-region-year counts are over all 30 years of the same.
