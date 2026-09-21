# Round Q — one territory, one label, and the scrap arc goes below base

**Run measured:** `opt_20260921_114721_de4c5cc5` (label `roundQ_one_territory`, 30/30 Solved, `prior_mode` cold)
**Against:** P.1 `opt_20260921_101621_cd0c0ae2`, Round O `opt_20260917_121759_c57f252b`, base `opt_20260916_124721_c4d1d944`
**Chain:** `imp_20260921_112949_8ac29fdd` → `const_20260921_113447_9c8ccc38` → the run
**Feeds/params:** icsg **v10** (was v9), scrap_share **v3** (was v2); baci v9, crude_ore v3, conc v8, ore_grade v7 unchanged
**Date:** 2026-09-21 · **ADR:** [0032](../adr/0032-one-territory-one-label.md)

**Verdict: Q delivers, and it closes the arc. Common-set z² 1,844,644 → 1,698,662 — a −1.1 % improvement
on base, so the whole O + P.0 + P.1 + Q stack is now better than the model it started from. Round O's
+10.9 % regression is fully retired. Round P.2's premise is withdrawn: two of its three items are not
defects, and the third was a feed error, not a band error.**

---

## 1. What changed, and why it was invisible

`data/input/feed/<id>/v<n>/values.csv` carries the **source** region labels a feed publishes;
`conc.xlsx::region_map` maps them onto the model's **root** regions, and that map is applied
downstream, inside `feedpipe`. Everything that runs *before* the feed pipeline sees source labels.

ICSG does not use one label per territory for all 30 years. It renames a territory mid-series
(`Taiwan` → `Taiwan, China` from 2013) and publishes historical joint reporting under a joint label
(`Belgium-Luxembourg` to 2013). Three consequences, one cause.

**Q-a — the ADR 0030 subtraction was skipped for 19 of Belgium's 30 years.**
`icsg_clean.py::_smelter_route` groups by `['Region','Year']` and `apply_input_divisors` does
`.reindex(keys).fillna(0.0)`. Belgium's `Refining/Secondary` rows are labelled `Belgium-Luxembourg`
for 1995–2013; its `Smelting/Secondary` rows are all labelled `Belgium`. The lookup missed,
`nested = 0`, and the correction did nothing.

The published series jumped from ~160 kt/yr to ~18 kt/yr at the handover — a discontinuity with no
physical cause. After the fix it runs 13.3 · 11.8 · 46.9 … 9.9 into 18.4 · 13.9 · 7.3 · 39.1:
**continuous**. And the corrected 2013 value is **9.9 kt**, exactly what ICSG publishes under the
`Belgium` label in that same year. The arithmetic reproduces their own post-handover series.

| | 1995–2013 | 2014–2024 | 30 y |
|---|---|---|---|
| before | 3,331.1 | 247.9 | 3,579.0 |
| after | **897.4** | 247.9 | **1,145.3** |
| removed | 2,433.8 | **0.0** | 2,433.8 |

Egypt, Ukraine and Romania are **not** instances — they have no smelter series of their own, so
`nested = 0` is correct for them. An earlier cut of this measurement joined on the root region and
cross-multiplied ROW's 48 members into 1,087 false positives; keying on the source label, the way
the code does, gives 79 rows and one real territory.

**Q-b — the handover year was published twice and summed.** Belgium 2013 anode into ER
**388.2 → 776.3**, in a series otherwise flat at ~390; Belgium 2013 cathode 211 → 422; Taiwan 2013
cathode **437.5 → 875.0**. Four cells feed-wide. At the raw level the two labels agree to within
0.03 kt, so these are republications, not measurements. ROW's 226 multi-source cells are a genuine
aggregate and stay summed.

**Q-c — the band study never mapped at all.** `scrap_share_study.py` grouped raw `values.csv` by
source region and looked the result up with `acc.node_index["region"]`. **172,174 kt of series
values were never seen.** Five of 94 bands move:

| process | region | v2 | v3 |
|---|---|---|---|
| Smelting | Korea, Rep. | `[0, 0.250]` *default* | **`[0.032, 0.296]`** n=30 |
| Smelting | ROW | `[0, 0.250]` *default* | **`[0.010, 0.229]`** n=30 |
| CFR | Belgium | `[0, 0.195]` n=12 | **`[0, 0.243]`** n=30 |
| CFR | Korea, Rep. | `[0, 0.250]` *default* | **`[0, 0.109]`** n=30 |
| CFR | ROW | `[0, 0.250]` *default* | **`[0, 0.137]`** n=30 |

Both of the actively-binding ceilings are in that list: Korea/Smelting (7 region-years, 1,061 kt)
and Belgium/CFR (22 of 30 years). Taiwan correctly does not move — it has no ICSG scrap series, so
its default band is right.

## 2. What Round P.2 turned out not to be

P.2 was scoped as three band items. Two are withdrawn on measurement.

- **`.clip(upper=1.0)`** was recorded as turning a data inconsistency into a share of 1. It clips by
  ~1e-5 relative — **6 grams on 123 kt** — on 68 CFR observations in three regions, two of which
  (Egypt, Ukraine) are not model regions. It is float-noise cleanup on `scrap == anode ÷ eff`.
- **The six `[0.95, 1.00]` bands** come from the documented `fillna(0.0)` "missing counts as 0"
  convention — none of those six regions has a primary series at all — not from the clip. They block
  1,023 kt of primary input that is **100 % fabricated**: zero feed-observed edges, and 505 kt of it
  a *domestic* `Concentrate→Smelting` edge in Belgium, a country with no mines. The bands are doing
  their designed faucet-stop job and must not be widened.
- The minimum band width 0.05 is not an absent floor but the `max(0,·)`/`min(1,·)` truncation
  halving the ±0.05 margin at the two ends — the ends where the bands do their most useful work.

And **item 1 was real but misattributed.** Belgium/CFR's posterior ÷ reported of 0.17 was never a
band that is too tight. The *reported level* was ~5× too high for 19 years, ADR 0031 anchored it
there, and the hard ceiling 0.195 was the only thing refusing it. The band was right; the feed was
wrong.

## 3. The gates, before the solve

Both pre-solver gates were run and both passed, so no compute was spent on a build that could not work.

**Stage 2 — 331 of 854,460 edges moved (0.039 %), zero coverage flips.** The correction changed
values, not what a feed observed. `Scrap→CFR` CC moved in **Belgium and nowhere else**, by
**−2,433.8 kt across exactly 19 edges** — the number, the years and the region all as predicted. Its
Mass twin moved −3,476.8 = −2,433.8 / 0.70, the `conc` nominal to the decimal.

The rest is Belgium's chain re-deriving the long way round now the direct-scrap shortcut is gone:
`Smelting→Matte` +2,006 CC, `Matte→CFR` +1,986 CC, `Concentrate→Smelting` +1,963 Mass,
`MF→Concentrate` +1,963 Mass. That is the physical route ADR 0030 says the copper takes. Collateral
elsewhere is small: Germany −200, Bulgaria −107, Chile −43, Spain −34, Italy −26, ROW −26.

**Stage 3 — scrap_share v3 consumed, +36 rows, 0 removed.** Anchor counts unchanged at 531
`smelterscrap` / 483 `directrefscrap`, exactly as coverage-invariance predicts. The 36 added rows are
`supplytotout MF_Belgium` (18 years × 2 valuations), the group that was below `min_abs_c` and now
is not — consistent with the Stage-2 `MF→Concentrate` move.

## 4. Results

### 4.1 The governing criterion

| | base | O | P.1 | **Q** |
|---|---|---|---|---|
| **common-set sum-row z²** | 1,718,085 | 1,905,519 (+10.9 %) | 1,844,644 (+7.4 %) | **1,698,662 (−1.1 %)** |
| total z² (not comparable) | 1,719,806 | 1,949,396 | 2,044,660 | 1,831,522 |
| only-here z² | 1,721 | 43,877 | 200,015 | **132,860** |
| `directrefscrap` z² | — | — | 69,200 | **21,833** (−68 %) |
| `smelterscrap` z² | — | — | 85,313 | **57,783** (−32 %) |

The anchor family became far cheaper to satisfy because the level it asserts is now correct. That is
the cleanest possible confirmation that ADR 0031 was anchoring a wrong number for Belgium, not
over-constraining the model.

By sector on the common set, base → Q: **Anode −38,703, CFR −36,722, Cathode −14,657, Matte −8,662**
against **Ore +40,678, MF +28,113, Concentrate +20,548**. The fit moved onto the refinery chain and
off the mine chain — Belgium's smelter now needs concentrate the trade matrix cannot supply.

### 4.2 Levels and bands

| | base | O | P.1 | Q |
|---|---|---|---|---|
| Belgium/CFR posterior ÷ reported* | 1.89 | 0.56 | 0.55 | **0.82** |
| Belgium/CFR band headroom | — | 0.22 | 0.22 | **1.236** |
| Belgium/CFR years blocked by the ceiling | — | 22 | 22 | **9** |
| Korea/Smelting ceiling | — | — | 7 y, 4,246 kt | **1 y, 604 kt** |
| region-years on a band ceiling | 129 | 170 | 118 | **99** |
| reported > 1 kt but posterior < 5 % of it | 75 | 44 | 13 | **2** |
| pairs below 0.50 of 44 | 5 | 1 | 0 | **0** |
| invented direct scrap (kt) | 7,652 | 164 | 198 | **179** |

\* every column uses **Q's corrected feed** as the reference, so the four are like-for-like. The
**0.17** recorded for Belgium/CFR in the earlier TODO was P.1's posterior against P.1's own inflated
level; against the corrected level P.1 was already at 0.55.

**The headroom prediction was pre-registered and held.** Before the run I projected Belgium/CFR
headroom `(node × 0.243) / 38.2 kt/yr ≈ 1.45` and predicted it would clear 1 for the first time.
Measured: **1.236** — direction and threshold right, magnitude over-estimated because the CFR node
grew less than the Stage-2 `Matte→CFR` delta implied.

Still below 1, all with a ceiling of **1.000** — not band-blocked, merely needing their node to grow:
Smelting/Austria 0.596 (30 years), UK 0.946, Slovakia 0.974, CFR/Argentina 0.999. **Austria is the
one substantial case left.**

### 4.3 Costs, stated plainly

- **Mine-chain deletion rose 38,934 → 39,394 kt.** All of it is the **forced** term (26,411 → 26,959):
  Belgium's chain now demands concentrate the BACI matrix cannot source — ADR 0029's open item, not a
  Q effect. The **discretionary** term, which is what the model chooses, **fell 12,523 → 12,435**.

  | run | deleted | forced | discretionary |
  |---|---|---|---|
  | base | 52,871 | 25,913 | 26,958 |
  | O | 44,197 | 26,421 | 17,776 |
  | P.1 | 38,934 | 26,411 | 12,523 |
  | **Q** | 39,394 | **26,959** | **12,435** |

- **World scrap fidelity slipped 1.057 → 1.076**, and **Belgium/Smelting overshoots at 1.17**
  (4,721 vs 4,048 kt reported). Same cause: copper removed from the direct route reappears on the
  smelter route, and ICSG's smelter series is the only thing sized to hold it.
- **`Kazakhstan` rejoins `ROW`** above 1.05 intake retention, having cleared in P.1.
- **World secondary share unchanged at 0.223** against a 0.164 prior — P-g still unmet.
- Mine retention essentially flat: world MF→Concentrate 0.906 → 0.905, Chile 0.571 → **0.574**,
  Peru 0.945 → **0.952**, trade-dependent intake 0.742 → 0.743.

## 5. What this says about method

Three things worth carrying forward.

**The discontinuity was the signal.** Nothing in the model complained. The tell was a published series
that stepped an order of magnitude at a label handover with no physical cause — and the proof was
that correcting it reproduced ICSG's own post-handover figure (9.9 kt) to the decimal.

**A rule that keys on a label silently skips whatever is labelled differently.** `.fillna(0.0)` after
a `.reindex` turns "no match" into "nothing to subtract", which is indistinguishable from a correct
zero. ADR 0030 shipped, its tests passed, its feed gate passed, and it did nothing for 19 of Belgium's
30 years. A gate that had compared the corrected series for **continuity** would have caught it.

**Two of P.2's three items were wrong, and measuring them first cost nothing.** The `.clip(upper=1.0)`
was recorded as masking a data inconsistency; it clips 6 grams on 123 kt. The `[0.95, 1.00]` bands
were recorded as a defect; they block 1,023 kt of input that is 100 % fabricated. Both would have been
"fixed" into regressions.

## 6. What is open

1. **Smelting/Austria, headroom 0.596, blocked in 30 of 30 years** — the last substantial
   level-vs-node mismatch. Its ceiling is 1.000, so the band is not the wall; the node is.
2. **Belgium/Smelting at 1.17 of its ICSG level** — the smelter route absorbing what the direct route
   gave up. Worth checking whether ADR 0030's nesting is now too aggressive for Belgium specifically.
3. **`stock_bounds_study.py` has the same source-vs-root bug** (see TODO) — deliberately untouched
   here so Q's lever stays attributable.
4. **The concentrate export-mass gap** — now **68.4 %** of all deletion and, again, unmoved. Every
   round in this arc has cut the discretionary term and left the forced one alone. It is the only
   thing left that can move Chile.
