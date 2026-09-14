# Round K — scale totals carry the feeds' reporting resolution as a σ floor (ADR 0025), two variants

- **K.1** (recommended): `opt_20260914_090327_c21a5c48` ← Stage 3 on the shipped imp run
  `imp_20260910_131346_e055415a` (no feed / Stage 2 rebuild); floor via ADR 0016's resolver on
  **feed-backed** groups only, `SCALE_TOTAL_ABS_FLOOR = 0.3` kt; engine `SIGMA_C_ABS_FLOOR` 1e-3
- **K** (global): `opt_20260914_083811_b86cef9c` ← `const_20260914_082530_549d17fc`; engine
  `SIGMA_C_ABS_FLOOR = 0.3` on every free-cone row — **measured and rejected**
- baseline: `opt_20260910_133212_45f034f1` (shipped Round J.1), sum-row z² 2,745,311
- reproducer: `python scripts/explo/roundK_sigma_floor_diag.py <base> <new>`
- branch `fix/sigma-c-floor`, **K.1 merged to main 2026-09-14**. ADR
  [0025](../adr/0025-scale-totals-carry-the-feeds-reporting-resolution-as-a-sigma-floor.md).

## 1. Why

Every scale total was priced at σ_c = 0.017·|c| with an absolute floor of 1 t. A fifth of the
27,242 sum rows had σ_c < 0.03 kt; the largest cell on the shipped run, Portugal Cathode
(90,160 z²), is a node with no refinery and a 0.3–1.7 kt anchor. Eighteen percent of the
headline sat on anchors below 10 kt. ICSG reports at 0.1–1 kt; integer-kt rounding alone is
σ = 0.29 kt.

## 2. The two variants

| | base | K global floor | **K.1 feed-backed floor** |
|---|---|---|---|
| sum-row z², 30 y | 2,745,311 | 1,804,818 (−34.3 %) | **2,331,632 (−15.1 %)** |
| < 1 kt band | 136,001 | 5,639 | 64,292 |
| 1–10 kt band | 353,082 | 51,737 | 206,207 |
| 10–100 kt band | 856,955 | 471,820 | 787,693 |
| 100–1,000 kt band | 815,791 | 745,744 | 813,688 |
| > 1,000 kt band | 583,481 | 529,878 | 459,752 |
| rows lifted ≥ 5 % | — | 10,430 | 1,879 |
| world CO Mass retention | 0.753 | 0.757 | **0.785** |
| world Concentrate CC / Cathode CC retention | 0.856 / 1.009 | 0.860 / 1.011 | 0.881 / 1.014 |
| small nodes (|c| < 18 kt) post/prior median · p10 · p90 | 1.001 · 0.901 · 1.063 | 1.029 · 0.826 · **2.332** | 1.001 · 0.814 · 1.114 |
| Chile / Peru / Indonesia concentrate export retention 2010–19 | 0.601 / 0.918 / 0.838 | **0.523** / 0.927 / 0.857 | 0.618 / 0.947 / 0.841 |
| Chile mine output posterior 2010–19 (prior 38.0 Mt) | 25.3 Mt | **23.3 Mt** | — |
| CC mass moved vs base | — | 3.5 % | 2.5 % |
| 30/30 Solved | ✓ | ✓ | ✓ |

**K (global) is rejected.** The noise went (−96 % / −85 % on the tiny bands) but so did the
counterweight: every tiny anchor loosened, so small fabricated chains grew — Matte Portugal
0.01 → 2.95 kt, Anode Switzerland 0.01 → 1.09 kt, 10 % of small nodes more than doubled in
*both* ADR 0016 classes — and the concentrate market re-equilibrated with Chile's exports to
Japan −542, China −446, India −246 kt CC and a further 2 Mt off its posterior mine output.
ADR 0016's "double duty" failing in the inflation direction. Also: the global engine floor
reached the 462k soft ratio rows too, which was never the intent.

**K.1 is the ADR's own derivation applied faithfully.** A total a feed *reported* cannot be known
better than the feed reports it; a total the model built from twins and recipes keeps the
tightness that is its only job. Implemented through the ADR 0016 resolver — its first live use in
the direction that ADR asked for.

## 3. K.1 against the stop rule

- **(a)** world retention moves by +0.032 / +0.025 / +0.005 — beyond the "< 0.01" letter, but
  *upward*: the rule exists to catch ADR 0016's collapse (retention falling). Intent passes.
- **(b)** small nodes: median unchanged (1.001); p90 1.11 (not 2.33); nodes below half their
  prior 242 → 293, above 1.5× 36 → 142. Mild, and the deflators are nodes already at ~0 in base.
- **(c)** every exporter up.
- **(d)** the gain is in the < 10 kt bands (−219k) and, unexpectedly, in the > 1,000 kt band
  (−124k): China MF / Concentrate / Ore / Mining −19k each, Chile −5.7k ×3, US Ore −5.6k. Those
  rows' σ did not change; their posteriors moved because the mining collapse eased (world CO
  Mass retention +3.2 pp, 130 Mt of the 152 Mt moved sits on > 1,000 kt nodes). Why tiny
  feed-backed anchors were holding the big mining chains down is **not traced here** — it is a
  lead for the mining-collapse thread, and a real one.

Target cells: Portugal Cathode 90,160 → 30,004; Germany Concentrate 53,109 → 21,043; PNG Anode
13,148 → 7,097; Korea Concentrate 27,174 → 22,479; Netherlands Anode 80,120 → 68,411; Chile Ore
48,740 → 43,103. Unchanged: Italy Cathode 70,935, Argentina Smelting 33,536, India Concentrate
23,261 (large or fabricated anchors). Every sector family falls (Concentrate −95k, Cathode −86k,
Anode −85k, MF/Mining/Ore −29k each).

## 4. What stays, and why

Of the 659k z² below 18 kt on the base run, 362k sits on **fabricated** groups (twin/recipe
priors with no feed behind them). K.1 leaves them tight by design; K showed what loosening them
does. Their cure is structural — the re-export round (Round L) removes the phantom stubs they
price — not a σ.

## 5. Decision

**K.1 shipped (2026-09-14).** The `>1,000 kt` improvement is welcome but unexplained; record it as the
first lead of the mining-collapse thread rather than count it as this round's design.
