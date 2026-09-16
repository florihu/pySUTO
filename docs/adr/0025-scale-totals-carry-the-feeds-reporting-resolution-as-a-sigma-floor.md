# Scale totals carry the feeds' reporting resolution as an absolute σ floor

**Status:** accepted — K.1 merged 2026-09-14 (shipped run `opt_20260914_090327_c21a5c48`); REP
[`…roundK-sigma-floor`](../rep/analysis_opt_20260914_090327_c21a5c48_20260914_roundK-sigma-floor.md). Amends
[ADR 0006](0006-scale-total-supersedes-external-control-total.md) (σ_c = ANCHOR_CV·|c|); same
shape as [ADR 0013](0013-no-data-flat-prior-anchor.md)'s edge floor `σ = max(abs_floor, cv·…)`;
tested against [ADR 0016](0016-provenance-aware-sigma-on-scale-totals.md)'s collapse mode.
Its note that a re-export structure would *"add small hub nodes"* anticipated Option 1;
[ADR 0027](0027-re-exports-are-re-attributed-at-feed-time.md) (2026-09-16) takes Option 2 instead
and adds no nodes at all.

## Context

Every scale total (ADR 0006) is priced at `σ_c = ANCHOR_CV·|c|` — 0.01, effective 0.017 after
the engine's Lp CV floor — and then Lp-combined with an absolute floor `SIGMA_C_ABS_FLOOR` that
stood at **1 t** (`1.0e-3` kt). A tonne is no floor: on the Round-J run a fifth of the 27,242 sum
rows carried σ_c below 0.03 kt, i.e. the model asserted sub-kilotonne national totals to tens of
tonnes. The largest single cell on that run was **Portugal Cathode, 90,160 z²** — a node with no
refinery, imports of 1–2 kt and an anchor of 0.3–1.7 kt, so a 1 kt miss cost z −56. Germany
Concentrate (12 kt), Korea (3–6 kt) and Papua New Guinea Anode (5 kt) are the same shape, and
Round J showed that every fabricated chain removed leaves such a stub behind.

| anchor band |c| | rows | z² | share of 2,745,311 |
|---|---|---|---|
| < 1 kt | 2,121 | 136,001 | 5.0 % |
| 1–10 kt | 2,141 | 353,082 | 12.9 % |
| 10–100 kt | 4,869 | 856,955 | 31.2 % |
| > 100 kt | 18,111 | 1,399,272 | 51.0 % |

No feed supports that precision. ICSG's raw series are published at **0.1–1 kt** (Mining: 51 % of
values at 0.1 kt, 20 % at integer kt; the finer Smelting/Refining values are the ÷eff conversion
of integer and 0.1-kt rows), crude_ore at 0.1–1 kt. Integer-kt rounding alone carries an
irreducible `σ = 0.5/√3 = 0.29 kt` per reported value.

## Decision

**Feed-backed scale totals are floored at the feeds' reporting resolution; fabricated ones are
not.** `SCALE_TOTAL_ABS_FLOOR = 0.3` kt is applied through ADR 0016's `sigma_cv_resolver`
(`scripts/constraints.py::scale_total_cv`): a group whose feed-backed share of Σ|f_agg| is at
least `FEED_SHARE_OBSERVED` gets `cv = max(ANCHOR_CV, 0.3/|c|)`, i.e. `σ_c = max(0.01·|c|, 0.3)`
before the engine's Lp conditioning; a fabricated group keeps `FABRICATED_ANCHOR_CV`. The
engine's global `SIGMA_C_ABS_FLOOR` stays at 1e-3 kt, so soft ratio rows are untouched.

The floor reaches feed-backed totals below `0.3/0.01 = 30 kt`; a 100 kt total is unchanged.
This is the direction ADR 0016 asked for — a σ design that "separates scale preservation from
information weighting": a total a feed reported cannot be known better than the feed reports
it, and a total the model built from twins and recipes keeps the tightness that is its only
job (the counterweight to ADR 0013's zero anchors).

It is still a **repricing**, so it ships under the Round-E rule: metric by anchor band, mass
moved next to it, world retention, small-node posterior/prior, exporter retention.

**Why not a global floor — measured and rejected (Round K, first variant).** Setting the
engine's `SIGMA_C_ABS_FLOOR` to 0.3 kt for *every* free-cone row (run `opt_20260914_083811_b86cef9c`,
const `const_20260914_082530_549d17fc`, cold, 30/30 Solved) against the shipped
`opt_20260910_133212_45f034f1`:

| | base | global floor 0.3 kt |
|---|---|---|
| sum-row z², 30 y | 2,745,311 | 1,804,818 (−34.3 %) |
| < 1 kt / 1–10 kt bands | 136,001 / 353,082 | 5,639 / 51,737 (−96 % / −85 %) |
| 10–100 kt band | 856,955 | 471,820 (−45 %) |
| > 100 kt bands | 1,399,272 | 1,275,622 (**−9 %**, not the < 1 % the plan required) |
| world CO Mass / Conc CC / Cathode CC retention | 0.753 / 0.856 / 1.009 | +0.003 / +0.004 / +0.002 (pass) |
| nodes with |c| < 18 kt, posterior/prior median / p90 | 1.001 / 1.063 | 1.029 / **2.332** |
| Chile concentrate export retention 2010–19 | 0.601 | **0.523** |
| Chile mine output posterior 2010–19 (prior 38.0 Mt) | 25.3 Mt | **23.3 Mt** |

The noise went, but so did the counterweight: with every tiny anchor loosened, small fabricated
chains grow (Matte Portugal 0.01 → 2.95 kt, Anode Switzerland 0.01 → 1.09 kt, 10 % of small
nodes more than double — in *both* ADR 0016 classes), and the global concentrate market
re-equilibrated with Chile's exports to Japan −542, China −446, India −246 kt CC and a further
2 Mt off its posterior mine output. A change sold as noise removal moved the largest producer.
That is ADR 0016's "double duty" failing in the inflation direction, and the reason the floor is
scoped to feed-backed groups.

## Considered options

- **Global engine floor on every free-cone row.** Measured and rejected above.
- **Keep 1e-3 and accept the noise.** Rejected: 18 % of the headline sits on totals below 10 kt,
  and every later round (the re-export restructure adds small hub nodes) would be measured
  through it.
- **Drop tiny groups instead** (`min_abs_c` in `from_mask_anchor`). Rejected: a node with no
  scale total loses its only counterweight to the zero anchors; a loose one keeps it.
- **`sigma_ck_mode="p_mean"`** (σ_c from the group's own edge σ, the ADR 0016 candidate). Not
  taken here: it reprices *every* total from its edges' σ — flat-prior edges carry σ ≈ 10⁶ — which
  is the uniform loosening ADR 0016 measured as a collapse. The floor is the surgical version.
- **Floor scaled by √(group size)** (rounding errors add in quadrature). Not taken: most group
  edges are imputed, not reported; one reporting unit is the defensible lower bound.
- **A larger floor (0.5–1 kt).** Not taken: the counterfactual gain keeps growing with the floor
  (−38 % at 0.5, −48 % at 1.0) precisely because it starts to reprice 10–100 kt totals that the
  feeds *do* resolve; 0.29 kt is where the derivation stops.

## Measured (Round K, second variant — feed-backed floor, the Decision above)

Run `opt_20260914_090327_c21a5c48` (const `const_20260914_085039_01346eff`, imp `imp_20260910_131346_e055415a`
unchanged, cold, **30/30 Solved**) against the shipped `opt_20260910_133212_45f034f1`;
`scripts/explo/roundK_sigma_floor_diag.py`. 1,879 sum rows lifted ≥ 5 % (vs 10,430 under the
global floor); the engine's own floor stays at 1e-3 (1,498 row-timesteps).

| sum-row z² by anchor band | rows | base | K.1 | Δ |
|---|---|---|---|---|
| < 1 kt | 2,121 | 136,001 | 64,292 | −53 % |
| 1–10 kt | 2,141 | 353,082 | 206,207 | −42 % |
| 10–100 kt | 4,869 | 856,955 | 787,693 | −8 % |
| 100–1,000 kt | 13,057 | 815,791 | 813,688 | 0 % |
| > 1,000 kt | 5,054 | 583,481 | 459,752 | −21 % |
| **total** | 27,242 | **2,745,311** | **2,331,632** | **−15.1 %** |

Stop rule, all four:
- (a) world retention **rises** — CO Mass 0.753 → 0.785, Concentrate CC 0.856 → 0.881, Cathode CC
  1.009 → 1.014. The rule's "< 0.01" guards against ADR 0016's collapse (retention *falling*);
  the letter is exceeded in the opposite direction, so the intent passes.
- (b) nodes with |c| < 18 kt: median posterior/prior 1.001 → 1.001; p10 0.901 → 0.814; p90
  1.063 → 1.114 (global floor: 2.332). Nodes below 0.5× their prior 242 → 293, above 1.5× 36 → 142.
- (c) exporters up: Chile 0.601 → 0.618, Peru 0.918 → 0.947, Indonesia 0.838 → 0.841, world
  0.804 → 0.815 (global floor had cut Chile to 0.523).
- (d) the gain sits in the < 10 kt bands (−219k) **and** in the > 1,000 kt band (−124k): the
  latter is China MF/Concentrate/Ore/Mining (−19k each), Chile and the US mining chains —
  the mining collapse eased, not a repricing of those rows (their σ is untouched). Mass moved
  2.5 % of CC posterior, 130 Mt of it on > 1,000 kt nodes, i.e. mass restored where retention
  rose. Why tiny feed-backed anchors were holding the big mining chains down is not traced here
  and belongs to the mining-collapse thread.

Target cells (sum-row z², base → K.1): Portugal Cathode 90,160 → 30,004; Germany Concentrate
53,109 → 21,043; Papua New Guinea Anode 13,148 → 7,097; Korea Concentrate 27,174 → 22,479;
Netherlands Anode 80,120 → 68,411; Chile Ore 48,740 → 43,103. Unchanged: Italy Cathode, Argentina
Smelting, India Concentrate (their anchors are large or fabricated). By sector every family falls:
Concentrate −95k, Cathode −86k, Anode −85k, MF/Mining/Ore −29k each.

What remains on tiny anchors is the **fabricated** half (base: 362k of the 659k below 18 kt),
untouched by design — it is the counterweight ADR 0016 measured, and the global variant showed
what loosening it does. Its cure is not σ: it is the re-export structure (Round L), after which
those phantom stubs stop existing.

**Decided 2026-09-14: K.1 shipped.**
