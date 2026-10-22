# One territory, one label

**Status:** accepted — Round Q, 2026-09-21. Completes
[ADR 0030](0030-secondary-smelter-copper-enters-the-chain-once.md), whose nesting subtraction was
silently skipped for 19 of Belgium's 30 years. Corrects the Round P.2 scope recorded in
[TODO](../TODO.md). **Amended 2026-10-13 (#73, #74):** an ICSG aggregate row is split to its declared
member by the handover year's shares, and its handover-year duplicate is dropped; see the amendment
at the end. Built, not yet measured. **Amended 2026-10-15 (#85):** a vintage that publishes a
territory under another territory's label is relabelled, and its rows are dropped in the years the
territory reports itself; see the second amendment. Built and measured by #86.

## Context

`data/input/feed/<id>/v<n>/values.csv` carries the **source** region labels a feed publishes. The
concordance `conc.xlsx::region_map` maps them onto the model's **root** regions, and that map is
applied downstream inside `feedpipe`. Everything that runs *before* the feed pipeline — the parse
in `scripts/parse/icsg_parse/`, and any study reading `values.csv` directly — therefore sees source
labels, not root regions.

ICSG does not use one label per territory for all 30 years. It **renames** a territory mid-series
(`Taiwan` → `Taiwan, China` from 2013) and publishes **historical joint reporting** under a joint
label (`Belgium-Luxembourg` to 2013, `Belgium` from 2013). In the handover year both labels appear,
carrying the same numbers.

Three things went wrong, all from the same cause.

**The nesting subtraction was skipped.** `icsg_clean.py::_smelter_route` groups by `['Region','Year']`
and `apply_input_divisors` does `.reindex(keys).fillna(0.0)`. Belgium's `Refining/Secondary` rows are
labelled `Belgium-Luxembourg` for 1995–2013 while its `Smelting/Secondary` rows are all labelled
`Belgium`, so the lookup missed, `nested = 0`, and ADR 0030's correction did nothing:

| root | source | n | years | published | corrected | overstated |
|---|---|---|---|---|---|---|
| Belgium | Belgium-Luxembourg | 19 | 1995–2013 | 3,331.1 | 897.4 | **2,433.8 kt** |

The published series jumps from ~160 kt/yr to ~18 kt/yr at the 2013 handover — a discontinuity with
no physical cause. After the fix it runs 13.3 · 11.8 · 46.9 … 9.9 into 18.4 · 13.9 · 7.3 · 39.1:
**continuous**. The corrected 2013 value is 9.9 kt, exactly what ICSG publishes under the `Belgium`
label in that same year.

Egypt, Ukraine and Romania are **not** instances. They have no smelter series of their own, so
`nested = 0` is correct for them; their whole refined-secondary series is the direct route.

**The handover year was summed.** Both labels map to one root, so `feedpipe` added them: Belgium 2013
anode into ER **388.2 → 776.3** in a series otherwise flat at ~390, Belgium 2013 cathode 211 → 422,
Taiwan 2013 cathode **437.5 → 875.0**. Four cells feed-wide. At the raw level the two labels agree to
within 0.03 kt, so these are republications, not measurements.

**The band study never mapped at all.** `scripts/explo/scrap_share_study.py` groups raw `values.csv`
by source region and looks the result up with `acc.node_index["region"]`. **172,174 kt of series
values were never seen.** `Korea, Rep.` (source `South Korea`) and `ROW` (48 sources) fell back to
`DEFAULT_BAND (0.0, 0.25)` — Korea/Smelting's ceiling is active in 7 region-years, 1,061 kt — and
`Belgium` and `Taiwan` got bands built from a partial series.

## Decision

**Two source labels that denote the same reporting territory are merged to one canonical label
before anything keys on `Region`.** The concordance already states which pairs those are: both
members are assigned to the same **non-ROW** root. That is the test.

- `territory_aliases()` reads `region_map`, drops `ROW`, and returns the source→root renames for
  every root with more than one source. It asserts the root name is itself one of the sources, so a
  label is never invented.
- `merge_territories()` applies the renames, then drops the handover-year duplicate — asserting the
  two labels agree to within 0.5 kt first, so a genuine conflict raises instead of being silently
  resolved by `drop_duplicates`.
- It is called in `cl()` **before** `secondary_excess` and `apply_input_divisors`.
- `scrap_share_study.py` maps to root regions when it reads the feed, and asserts every source
  region has a map entry.

**ROW is excluded, deliberately.** It is the catch-all bucket for 48 countries, not a territory.
Merging its members would nest Serbia's smelter route against Egypt's refinery. Nesting happens per
reporting territory; aggregation into ROW happens afterwards, in the feed pipeline, where it belongs.

## Consequences

- Belgium's direct-to-refinery scrap falls 3,579 → 1,145 kt over 30 years. 2014–2024 is byte-identical;
  the entire change is 1995–2013.
- **Round P.2's premise is withdrawn.** Belgium/CFR's posterior ÷ reported of 0.17 was never a band
  that is too tight — the *reported level* was ~5× too high for 19 years, [ADR 0031](0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md)
  anchored it there, and the hard ceiling 0.195 was the only thing refusing it. The band was right;
  the feed was wrong.
- Belgium now appears in `icsg_secondary_excess.csv` for one region-year (16 kt), clipped to a zero
  direct route and listed — the ADR 0030 reporting path, working as designed.
- `scrap_share` bands move on **both** processes, so the v1→v2 guard asserting "Smelting is untouched"
  no longer holds and is replaced by a diff report on both.

### Considered and rejected

- **Merging on `mapping_type`.** `conc.xlsx` types `Taiwan, China` as `alias` but `Belgium-Luxembourg`
  as `aggregate` (with `mapping_confidence: medium`), the same type as the 47 ROW entries. The type
  column does not separate "same territory" from "bucket member"; the non-ROW root test does.
- **Merging at root level, i.e. nesting on the mapped region.** That would nest across ROW's 48
  members and is wrong for exactly the reason Egypt/Ukraine/Romania are not instances.
- **Summing the handover year, or averaging it.** The two labels agree to 0.03 kt; there is one
  measurement, published twice.
- **Fixing only the band study (Q-c alone).** The bands would then be re-derived from the same
  inflated Belgium series. The feed fix has to land first.
- **Widening the Belgium/CFR band instead.** [ADR 0028](0028-grade-sigma-is-stated-not-derived-from-the-band.md)
  measured what moving a hard wall does: it relocates the deletion. The band was carrying a real
  defect; removing the defect is the fix.

### Measured

Run `opt_20260921_114721_de4c5cc5` (30/30 Solved, `prior_mode` cold), chain
`imp_20260921_112949_8ac29fdd` → `const_20260921_113447_9c8ccc38`, icsg **v10**, scrap_share **v3**.
Against P.1 `opt_20260921_101621_cd0c0ae2`, Round O `opt_20260917_121759_c57f252b`, base
`opt_20260916_124721_c4d1d944`.

**The arc goes below base.** Common-set sum-row z²:

| | base | O | P.1 | **Q** |
|---|---|---|---|---|
| common-set z² | 1,718,085 | 1,905,519 (+10.9 %) | 1,844,644 (+7.4 %) | **1,698,662 (−1.1 %)** |
| `directrefscrap` z² | — | — | 69,200 | **21,833** (−68 %) |
| `smelterscrap` z² | — | — | 85,313 | **57,783** (−32 %) |

The anchor family got cheaper to satisfy because the level it asserts is now right. By sector on the
common set, base → Q: Anode −38,703, CFR −36,722, Cathode −14,657, Matte −8,662 against
Ore +40,678, MF +28,113, Concentrate +20,548 — the fit moved onto the refinery chain and off the
mine chain.

**The feed lineage, as each run's own feed states Belgium's direct-refinery scrap:**

| run | icsg | 1995–2013 | 2014–2024 | anode 2013 |
|---|---|---|---|---|
| base | v8 | 3,481.7 | 1,713.5 | 776.3 |
| O, P.1 | v9 | 3,331.1 | 247.9 | 776.3 |
| **Q** | **v10** | **897.4** | 247.9 | **388.2** |

ADR 0030 (v9) fixed 2014–2024 and left 1995–2013 almost untouched. ADR 0032 (v10) fixed the rest.

**Stage 2 was surgical:** 331 of 854,460 edges moved (0.039 %), **zero coverage flips**;
`Scrap→CFR` CC moved in Belgium and nowhere else, by −2,433.8 kt across exactly 19 edges. Its Mass
twin moved −3,476.8 = −2,433.8 / 0.70, the `conc` nominal to the decimal. Stage 3 added 36 rows and
removed none: `supplytotout MF_Belgium`, 18 years × 2 valuations, previously below `min_abs_c`.

**Bands and levels:**

| | base | O | P.1 | Q |
|---|---|---|---|---|
| Belgium/CFR posterior ÷ reported* | 1.89 | 0.56 | 0.55 | **0.82** |
| Belgium/CFR band headroom | — | 0.22 | 0.22 | **1.236** |
| Belgium/CFR years blocked by the ceiling | — | 22 | 22 | **9** |
| Korea/Smelting ceiling | — | — | 7 y, 4,246 kt | **1 y, 604 kt** |
| region-years on a band ceiling | 129 | 170 | 118 | **99** |
| reported > 1 kt but posterior < 5 % of it | 75 | 44 | 13 | **2** |
| pairs below 0.50 of 44 | 5 | 1 | 0 | **0** |

*all four columns use Q's corrected feed as the reference, so they are like-for-like; the 0.17
recorded for P.1 in the earlier TODO was P.1's posterior against P.1's own inflated level.

**Costs, stated plainly.** Mine-chain deletion rose 38,934 → 39,394 kt, but entirely in the **forced**
term (26,411 → 26,959): Belgium's chain now demands concentrate the BACI matrix cannot source, which
is ADR 0029's open item, not a Q effect. The **discretionary** term — what the model chooses — fell
12,523 → **12,435**. World scrap fidelity slipped 1.057 → 1.076 and Belgium/Smelting overshoots its
ICSG level at 1.17, both because the copper removed from the direct route reappears on the smelter
route. `Kazakhstan` rejoins `ROW` above 1.05 intake retention, having cleared in P.1. World secondary
share is unchanged at 0.223 against a 0.164 prior.

## Amendment (#73, #74, 2026-10-13): an aggregate row is split to its declared member

ICSG reports some countries only inside an **aggregate row** until a handover year, then the countries
themselves. "Scandinavia" carries Sweden's, Finland's and Norway's mining, smelting, refining and semis
for 1995–2004, loaded from the 2005 yearbook. The three countries report themselves from 2004, loaded
from the 2014 yearbook. The region map sends Scandinavia to ROW. That did two kinds of damage.

- **#74, 1995–2003.** Sweden had a mine (crude-ore feed) and its trade (BACI) but no smelter or refinery.
  Once #67 stopped netting its cathode trade away, the model supplied it through a phantom SX-EW chain
  (`Ore→SXEW`, a flat-prior zero, 175.0 → 307.8 kt CC over 1996–2003) and a forced ROW → Sweden MESC
  move in 2001 (z 15.7), on `opt_20261013_095846_46ece808`.
- **#73, 2004.** Both forms are published and the feed added them, so the Nordic flows entered twice,
  once via ROW: refined anode use 408.6 kt CC each way.

This is ADR 0032's handover, with an aggregate in place of a renamed territory. `split_aggregates()`
runs in `cl()` right after `merge_territories()`, before anything keys on `Region`. It reads the
hand-authored params input `icsg_aggregates` (v4: Sweden ⊂ Scandinavia, 1995–2003), the same
declaration #67's netting reads. The **handover year** is the year after the member years, 2004.

- **Member years.** The declared member's share of each series is its value over the aggregate's in the
  handover year. Each aggregate row in the member years is split: the member gets that share, the
  aggregate keeps the rest, and the total is unchanged. Shares on icsg v13: Mining 0.837, Smelting
  primary 0.525, Refining primary 0.521, Semis 0.618, Smelting secondary 0.914, Refining secondary
  0.916. Sweden gains 54 rows for 1995–2003; over those years it now carries 1,558.2 kt CC of refined
  anode use, 1,296.0 of smelter concentrate use, 681.3 of mine output and 1,548.3 of semis use. Each
  production and use series keeps its total. One knock-on: after ADR 0030's nesting, refinery scrap use
  rises 0.40 kt CC over 1995–2003, because in 9 years Scandinavia's remainder of the smelter route
  exceeds its remainder of secondary refined output. Those 9 region-years (0.02–0.06 kt CC each) are
  clipped and listed in `icsg_secondary_excess.csv`, the ADR 0030 path.
- **Handover year.** The aggregate republishes its members, so its row is dropped wherever the member
  reports the series: 6 rows. In 2004 Scandinavia's refined primary output equals Sweden + Finland +
  Norway exactly (347.7 kt CC); mining differs by 0.5 kt CC and semis by 4.6, yearbook revisions.
- **Left alone.** A series the member does not report in the handover year has no share and stays with
  the aggregate, here Scandinavia's producer stock change. Finland and Norway are not declared: they map
  to ROW, so their part stays where it was.
- **Raises.** A second member declared for one aggregate, an aggregate row outside its member years and
  handover year, a handover year without both forms, a series whose share cannot be read or lies
  outside [0, 1], and a member row the split would duplicate.

**The assumption.** Each share is held at its handover-year value for every member year. ICSG publishes
no Swedish split before 2004, and Nordic capacity moved over the decade, so the 1995–2003 split is an
estimate, not an observation. That is why #67's netting keeps treating Sweden 1995–2003 as a reporting
gap: its allocated intake and production are not observations of a hub's balance.

**Considered and rejected.**
- *Keep Scandinavia in ROW.* That is the #74 defect: Sweden's trade without its processing.
- *Give the whole aggregate to Sweden.* It would hand Sweden Finland's smelter and refinery, about half
  of Scandinavia's primary output in 2004.
- *Drop only the 2004 duplicate.* That fixes #73 and leaves #74.
- *Year-specific shares from another source.* None is in the repo.

## Amendment (#85, 2026-10-15): one vintage's label for another territory

The 2005 yearbook publishes DR Congo as `Congo`. Its mining footnote 3/ reads "SXEW production counted
in Congo only at the mine level, as the ore is exported to Zambia for treatment", which is Katanga, not
Brazzaville. The parse maps `Congo` to `Congo Rep.`, the label every later yearbook uses for Brazzaville,
and the region map sends that to ROW. So DR Congo had no ICSG rows in 1995–2003, and ROW carried its
mined concentrate, smelter intake and SX-EW cathode: 303.2, 266.9 and 257.6 kt CC over those nine
years in icsg v14.

**Decision.** `icsg_clean.relabel_by_vintage` maps `(label, vintage)` to the territory's own label,
before anything keys on `Region`; `('Congo Rep.', 2005) → 'Congo, D.R.'` is the one declaration. A
territory merge (above) cannot do it, because the label is right in other vintages.

**The overlap year is dropped whole.** In 2004 both yearbooks publish DR Congo, and they type its
refined output differently: SX-EW cathode in the 2005 yearbook, refinery output in the 2014 one. The
per-series dedup cannot see that overlap, so a plain rename would book the 20 kt twice (ADR 0036's
merge-or-replace question). The relabelled rows are therefore dropped in every year the territory
reports under its own label in another vintage, here 2004's three rows.

**Raises** on a declaration that matches no rows, and on a vintage carrying both labels.

**Left as found.** The 2005 yearbook books DR Congo's low-grade electrowon cathode in both its smelter
and its SX-EW tables (33.4, 43.8, … kt). The parse keeps both, as it did under ROW; #86 reports how DR
Congo's 1995–2003 chain fares.
