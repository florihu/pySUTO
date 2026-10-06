# Round A (Mass–CC tension): regional tailings conc + outlier-leg screen shipped; σ policy rejected

**Baseline:** `opt_20260903_084528_45fdd041` (grade-basis run). **Round-A chain:**
`imp_20260903_100104_cf897869` (screen + ore_grade v3 incl. tailings, conc v6) →
three const/solve configurations of the fabricated-anchor cv:

| run | FABRICATED_ANCHOR_CV | verdict |
|---|---|---|
| `opt_20260903_101715_821a620b` | 0.15 | rejected — world-chain collapse |
| `opt_20260903_103441_42cf1331` | 0.05 | rejected — collapse barely reduced |
| **`opt_20260903_105145_fa7049e3`** | 0.01 (= policy off) | **shipped configuration** |

All runs 30/30 Solved, cold, full 30 years. Follows the root-cause REP
`analysis_opt_20260903_084528_45fdd041_20260903_masscc-tension-root-causes.md`; plan approved
2026-09-03.

## 1. Shipped: regional tailings conc (ADR 0015 amendment) — WORKS

`ore_grade` v3 adds 1,146 region×year×route rows (787 MF + 359 SXEW) built so the tailings Mass
twin closes the hard MF/SXEW Mass balance at prior (`g_tail = ProdCC·(1−eff)/eff ÷
[(ProdCC/eff)/g_insitu − ProdCC/conc_product]`); the global `conctailingsuse` row (0.55 %) is
retired (conc v6). Generated values: median 0.25 % (the global nominal was ~2× too high nearly
everywhere), Chile MF 0.209 % (verified digit-for-digit against the run's prior construction),
Panama ≈ 0.105 % — right where its old z −84 pointed.

**Result:** tailings family z² **800k → 8k**; the Panama/US/Mexico tails are gone. Side-effect
(honest): consistent tailings priors are ~2.6× *larger* in Mass, so their cv-σ is larger in
tonnes and the mining chain is somewhat cheaper to shrink — part of Chile's small regression
below.

## 2. Shipped: bilateral-leg plausibility screen — WORKS

Stage 2 σ-inflation (`OUTLIER_LEG_K = 3`, cv → 5.0) flagged 19 exporter-sector-years / 104 legs.
The target class is caught exactly: Papua New Guinea concentrate exports at **62–136×** its
covered production (2019–22, 14–21 Mt/yr legs). Those legs now die at **z ≈ −0.2** instead of
−3.4…−4.8 each; report CSV at `imp_…cf897869/diag/outlier_leg_screen.csv`.

Caveat: Japan (1996–2004) and Spain (2011–14) concentrate cases are **re-export false
positives** (legitimate exports > own production; tens of kt, low impact). Follow-up: an
import-adjusted reference (production + covered imports) before raising K.

## 3. Rejected: loosening fabricated scale totals — NEGATIVE RESULT (the big lesson)

The motivation was real (3.0M z² phantom cost on fabricated totals; observed totals moving more
than fabricated ones). But at cv 0.15 the QP **shrank the whole world chain instead of
reconciling** — world 2015 posterior retention CO Mass 0.66 → 0.41, Concentrate CC 0.67 → 0.56,
Cathode CC 0.93 → 0.75; even *observed* anchors were dragged (China cathode-use z ≈ −20). cv
0.05 recovered only marginally (CO Mass 0.45).

**Mechanism:** fabricated scale totals do double duty — they are also the counterweight to the
ADR-0013 nodata **zero anchors** on the same interior subgraphs. At cv 0.017 the scale totals
win; loosened, the zero anchors win, the interior deflates, and the hard balances pull the
observed checkpoints down with it. Uniform σ loosening cannot remove the phantom cost without
disabling collapse protection; a future design must decouple the two roles (candidate:
propagated edge σ via the builder's `p_mean` mode, and/or rebalancing zero-anchor strength).
Mechanism (resolver + group classification) is merged but inert (`FABRICATED_ANCHOR_CV =
ANCHOR_CV`); ADR 0016 records the rejection.

## 4. Shipped configuration vs baseline (`fa7049e3` vs `45fdd041`, all years)

| metric | baseline | Round A |
|---|---|---|
| total soft z² | 5.13M | **3.96M** |
| soft z p99 | 12.65 | **9.98** |
| tailings family z² | 800k | **8k** |
| world 2015 retention CO/Conc-CC/Smelt-CC/Cath-CC | 0.66/0.67/0.78/0.93 | 0.63/0.69/0.81/**0.95** |
| PNG outlier legs | deleted at z 3.4–4.8 | deleted at **z ≈ 0.2** |
| Chile 2015: CO Mass / CO CC / conc exports | 0.26 / 0.42 / 0.31 | 0.22 / 0.35 / **0.16** |
| Cathode_China max |z| | 9.7 | 11.7 |
| solver | 30/30 | 30/30 |

Chile regressed mildly: with the fake PNG supply neutralised and larger consistent tailings σ,
the unchanged **leg-σ economics** (cv ≈ 0.21 uniform ⇒ full deletion costs z = 4.78 flat) now
deletes the Chile→Japan 2015 leg fully (was 27 % retained). This isolates lever #3a/b — the
bilateral-leg σ design — as the next round, exactly as the root-cause REP ranked it; the Round-A
baseline (collapse-free, tailings noise gone, outliers de-authorised) is the right ground to
measure it on.

## 5. Suite / artifacts

Tests 878 passed (new: resolver, tailings generator ×3, screen ×2). ADR 0016 (new), ADR 0015
amendment, GLOSSARY.md scale-total + grade entries. ⚠ gitignored data regenerated: `ore_grade`
v3 (rerun `scripts/build_ore_grade.py`), `conc` v6 (conctailingsuse row removed from the
hand-maintained live CSV — do not blind-regenerate).
