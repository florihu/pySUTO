
# Harmonize registry. done
* registry harmonization for feed, imp, const solve registry
    * pattern is always stage id (feed, imp, const, opt)+ datet time + random code !! label lives only in registry

* input registry for data feeds, accs min id + version + date + random code see (const imp feed) label only in jason -> these are initialized when the init script is initilized and contain only minimal information.... extend with label var if not in meta 
    * for feeds lives in data/input/feed
    * for acc lives in data/input/acc
    * params lives in data/input/params

# Major change 
* Rework stock addition modelling
* Introduce as constant constraint



# Mask as an input

1. conceptualization
2. implementation
3. versioning


# Unc propagation

1. Conceptualization - > What methods are feasilbe.
2. Implementation
3. Analysis


# Improve solver diag

What do i want to analyse

1. General convergence analysis of solver
2. Aanlysis of a*(t)-a(t-1) and z scores -> which were the most changed - outliers
3. c*(t)-c_0(t) which constraints were tha
# OPEN (2026-09-16, out of Round L)

## ADR 0017's reference leg size drifts with the population it prices

`a_ref` is the **median of the covered bilateral legs**, so cleaning the leg
population moves it. The ADR 0027 netting removes the small hub-side legs and the
median rises (Concentrate 10.26 → 12.0 kt, Anode 0.21 → ~0.3); a higher `a_ref`
raises the damped cv of every surviving leg, looser legs are cheaper for the QP to
delete, and deleting them makes the independently anchored scale-total rows worse.
So **improving the trade matrix silently loosens σ on what survives** — the rule
penalises the cleaner population.

Measured (`opt_20260916_072151_ac748919`, the shipped chain with the pin off):
sum-row z² 1,748,645 vs 1,720,192 (**+1.7 %**), Chile concentrate retention
0.574 → **0.550**, world CO Mass 0.779 → 0.767, Concentrate CC 0.872 → 0.864.
30/30 Solved. REP `…roundL1-baci-reexport` Addendum 3.

`LEG_SIGMA_AREF` in `collect_aggregate_impute.py` currently pins the L.0 medians.
**Do not just restore the `None` default** — that is the worse configuration, and
it is the measurement above that says so. Decide instead: a per-sector constant, a
quantile of the *gross* matrix, or a physically motivated size — something that does
not move when the matrix it prices gets better.

## ROUND O — THE SECONDARY DOUBLE COUNT (measured 2026-09-21; part of the mergeable O+P stack)

Found while writing the copper-metabolism insights REP
(`docs/rep/analysis_opt_20260916_124721_c4d1d944_20260917_copper-metabolism-insights.md`).

`scripts/parse/icsg_parse/icsg_clean.py:18-22` converts BOTH ICSG secondary
production series into inputs — `Smelter Second.` → Scrap→Smelting and
`Refinery Second.` → Scrap→CFR — but ICSG's secondary refined production already
INCLUDES the copper that came through secondary smelters (blister → anode → ER).
The two series overlap by the smelter route, per region-year
`min(S_sm·k, S_ref)`: **3,718 kt in 2024, 70.0 Mt over 1995-2024**, against a
30-y world mine-chain deletion of 52.9 Mt (ratio 0.75, corr 0.67). In 119
region-years (Japan, Bulgaria, Iran …, 1.99 Mt total) the smelter route exceeds
secondary refined: clip to zero, list them. **Second instance:** the cathode
term in `icsg_intake_production` adds `Refining/Scrap·k` on top of an
`anode_in` that is already total refined — ~4.7 Mt (17 %) too much cathode
production in the P that ADR 0027/0024 use.

Proof: Austria (Brixlegg, 100 % secondary) — ICSG 60 kt secondary smelter inside
123 kt secondary refined = 123 kt total; the feed books 62 + 125 = 187 kt of
scrap against a 123 kt anode anchor; the posterior deletes the smelter scrap to
0.1 kt. With both booked, the anode anchor can absorb 14,758 kt of concentrate
in 2024 against 18,207 mined (gap 3,450); nested, the gap is −652.

**The fix (parse level + cathode term + scrap_share v2, then full chain —
plan grilled 2026-09-17):**
```
Scrap→CFR      = max(0, S_ref − S_sm · eff_CFR · eff_ER) / (eff_CFR · eff_ER)   # direct-to-refinery only
Scrap→Smelting = S_sm / eff_Sm                                                     # unchanged
P_Cathode      = anode_in · eff_ER + SXEW                                          # no second scrap term
```
`scrap_share` MUST be regenerated with it (v1 CFR floors — China 0.160,
Germany 0.342, Austria 0.95 — sit above the post-fix direct-route shares
0.040 / 0.126 / 0.503 and would force the scrap back in).
Acceptance, before any solver run: Austria's both-booked share → 1.0; the
world `conc_absorbable` gap → ≈ −0.46 Mt (`t1a_secondary_world.csv`, nested column).
On the chain: mine-chain deletion ≪ 12.8 %; posterior secondary share → ≈ 0.17
(ICSG); China's smelters off their scrap-band ceiling (0.40); Chile's mine
retention up with NO change to the grade cones. List, do not clip, the
region-years where `S_sm·k > S_ref` — they test the nesting assumption.
Amend ADR 0021's "not a double count" sentence. Own ADR, own plan.

**The export-mass gap below is the regional routing of this surplus** — the
world needs 3-4 Mt/yr less concentrate than it mines and drops it where a hard
cone already makes it free. Re-measure it after O.

**MEASURED 2026-09-21** — run `opt_20260917_121759_c57f252b`, 30/30 Solved, against base
`opt_20260916_124721_c4d1d944`.
[REP](rep/analysis_opt_20260917_121759_c57f252b_20260921_roundO-secondary-double-count.md) ·
[ADR 0030](adr/0030-secondary-smelter-copper-enters-the-chain-once.md).

Feed gate: all four checks PASS (Austria 1.480 → 1.000, world gap +3,450 → −464 kt, one row class,
107 excess region-years listed). Mine-chain deletion 52,871 → 44,197 kt; the **forced** term is flat
(25,913 → 26,421) and the **discretionary** term falls 26,958 → **17,776 kt (−34.1 %)**. The double
count was real.

But sum-row z² **+13.3 %** (1,719,806 → 1,949,396), σ identical on all 21,729 unchanged-anchor rows,
so it is a genuine fit loss: Slovakia's anode +64,630 alone (anchor 57 kt unchanged, posterior
43 → 6.5 kt in all 30 y), plus Austria/Belgium, plus 146 new phantom concentrate/MF rows in Germany
and Spain. Secondary share 0.300 → 0.257 against a prior now at 0.164; China *harder* on its 0.402
ceiling; **Chile unchanged** (0.557 → 0.561, capped 19/30) — the Chile thesis is not supported.

## ROUND P.0 + P.1 — THE SCRAP ARC (measured 2026-09-21; READY TO MERGE — user's call)

Round O moved copper from a route the model anchors **by level** (`Refining/Scrap`) to one it anchors
**only by share** (`Smelting/Scrap`: `0.95 ≤ Σ(Scrap→Smelting)/Σ(all→Smelting) ≤ 1.00`). A share band
over an idle node is free. Correcting the earlier note: ICSG has Slovakia's smelter eating
**46.6 kt/yr** (the ~28 figure was 2008), the prior carries it feed-backed at σ 6.8 %, and the model
routes **0.012 kt/yr in both runs**. It is *not* prior decay — both runs are `--prior-mode cold`, so
`f_agg` is re-asserted annually and the solver walks away annually. **A per-edge prior penalty is
cheaper to pay than to satisfy; only a scale total prices the group.**

**Built** ([ADR 0031](adr/0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md)): two
soft `sum` families, `smelterscrap` and `directrefscrap`, CC only, covered edges only, grouped on
`(sector_to, region_to, time, valuation)`, σ via the unchanged `_scale_total_cv`. Registered
**separately** so neither route can absorb the other. A region-year ICSG does not report gets no row
and keeps its freedom — the family can only assert a scrap inflow, never forbid one.

**MEASURED** — run `opt_20260921_090618_1ceb5eed`, 30/30 Solved.
[REP](rep/analysis_opt_20260921_090618_1ceb5eed_20260921_roundP-scrap-level.md) ·
[ADR 0031](adr/0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md).

Gate passed all six checks twice. **The family works at world scale**: posterior ÷ ICSG-reported
scrap 1.318 → 1.191 → **1.027**; secondary share 0.300 → 0.257 → **0.220** (prior 0.164); China
**off** its 0.402 ceiling (0.398 → 0.341); mine-chain deletion 44,197 → **38,876 kt** with the forced
term flat, so **discretionary deletion 17,776 → 12,455 kt**; retention 0.894 → 0.906; Peru → 0.953;
and **Chile finally moved, 0.561 → 0.578**.

**But P-d fails**: common-set z² +10.8 % → **+12.3 %** (this also corrects Round O's headline — its
regression was +10.8 %, not +13.3 %; the rest was its own new rows). The two motivating regions did
not recover, for two *different* reasons, neither of them the missing level — see P.1.

⚠ Also: ROW joins Kazakhstan above 1.05 intake retention; Cathode CC retention 0.984 → 0.966.

## ROUND P.1 — DONE: the pure-secondary recipe closes (2026-09-21)

Run `opt_20260921_101621_cd0c0ae2`, 30/30.
[REP](rep/analysis_opt_20260921_101621_cd0c0ae2_20260921_roundP1-secondary-recipe.md) ·
[ADR 0010 amendment](adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md).

P.0's diagnosis was right that Slovakia was strangled, wrong about where. Not the band (its Smelting
ceiling is 1.000) and **not ADR 0019** — that detector needs `inflow > 1 kt` and fires on **one
node-year in the whole model**; it is not a usable lever as written. It was **Stage 2**: the
`NetworkImputer` gate "no known primary ⇒ stay closed" is right about a data gap but wrong for a
100 %-secondary smelter, which has no primary *by construction* and whose input is measured on the
scrap leg. Slovakia's chain therefore carried nothing and Stage 3 pinned `usetotout Matte` at
`c = 0.011 kt, σ = 0.001`.

Result: **Slovakia 0.10 → 0.96**, anode 7.81 → **58.53 kt** (z −50.7 → +1.37), scrap level z −58.7 →
**−0.34**. Common-set z² **+12.4 % → +7.4 %**, below Round O's own +10.9 %. Stage-2 gate: 445 of
854,460 edges moved (0.052 %), two regions. Every criterion passes except P-g (secondary share 0.223
vs ≈0.17).

**The stack O + P.0 + P.1 vs base:** deletion 52,871 → 38,934 kt, discretionary 26,958 → **12,523**,
scrap fidelity 1.318 → **1.032**, invented direct scrap 7,652 → **198 kt**, pairs below 0.50 five →
**one**, Kazakhstan's fabricated node gone. **Ready to merge — the user's call on these numbers.**

## ROUND Q — SHIPPED 2026-09-21: one territory, one label (ADR 0032)

Run `opt_20260921_114721_de4c5cc5`, 30/30 Solved, icsg **v10**, scrap_share **v3**.
**Common-set z² 1,844,644 → 1,698,662 — `−1.1 %` vs base.** The whole O + P.0 + P.1 + Q stack is now
better than the model it started from; Round O's +10.9 % regression is retired. `directrefscrap` z²
−68 %, Belgium/CFR 0.55 → 0.82 and its headroom 0.22 → 1.236, band ceilings 118 → 99, dead
region-years 13 → **2**. Costs: total deletion +460 kt but **entirely forced** (discretionary
12,523 → **12,435**); world scrap fidelity 1.057 → 1.076; Belgium/Smelting overshoots at 1.17;
Kazakhstan rejoins ROW above 1.05. REP `…roundQ-one-territory`.

Still open from Q: **Smelting/Austria headroom 0.596, blocked 30/30** (ceiling 1.000 — the node, not
the band); **Belgium/Smelting at 1.17** of its ICSG level; and the `stock_bounds_study.py` bug below.

## ROUND P.2 — WITHDRAWN 2026-09-21: the bands were not the defect. The FEED was. (→ Round Q)

P.2 was scoped as three band items. Measurement refutes two of them and traces the third to a
single root cause one stage upstream: **code that runs before the feed pipeline applies
`conc.xlsx`'s `region_map` matches on the SOURCE region label where the ROOT region is meant.**
`values.csv` holds raw source names (`Belgium-Luxembourg`, `South Korea`, `Taiwan, China`); the map
to root names (`Belgium`, `Korea, Rep.`, `Taiwan`) is applied downstream inside `feedpipe`.

### Q-a — ADR 0030's nesting subtraction skipped Belgium 1995-2013  ⚠ 2,424 kt

`scripts/parse/icsg_parse/icsg_clean.py::_smelter_route` groups by `['Region','Year']`;
`apply_input_divisors` then does `.reindex(keys).fillna(0.0)`. Belgium's `Refining/Secondary` rows
are labelled `Belgium-Luxembourg` for 1995-2013 while its `Smelting/Secondary` rows are all labelled
`Belgium`, so the lookup misses, `nested = 0`, and **the subtraction silently does nothing for 19 of
Belgium's 30 years**:

```
root     source              n   years       published  corrected  overstated
Belgium  Belgium-Luxembourg  19  1995-2013     3,321.3      897.4     2,423.9 kt
```

Validation: the corrected 2013 value is **9.9 kt** — exactly what ICSG publishes under the `Belgium`
label in that same handover year. Egypt / Ukraine / Romania are **not** instances: they have no
smelter series of their own, so `nested = 0` is correct for them. Nesting must happen per real
reporting territory and never at ROW level, which would subtract Serbia's smelter from Egypt's
refinery.

**This is item 1.** Belgium/CFR's 0.17 is not a band that is too tight — it is a **reported level
~5× too high for 19 years**, which P.0 now anchors and which the hard ceiling 0.195 is the only
thing refusing. The band was right; the feed was wrong.

### Q-b — the handover year is published under both labels and summed  (4 cells)

Belgium 2013 anode 388.2 → **776.3** (the series is otherwise flat at ~390), Belgium 2013 cathode
211 → 422, Taiwan 2013 cathode 437.5 → **875.0**. ROW's 226 multi-source cells are a genuine
aggregate and correctly summed — `conc.xlsx` already carries the distinction in `mapping_type`
(`aggregate` 47 / `direct` 45 / `alias` 2 / `typo` 2).

### Q-c — the band study never applies the region map

`scripts/explo/scrap_share_study.py` groups raw `values.csv` by source region, then looks the result
up with `acc.node_index["region"]` (root names). **172,174 kt of series values are never seen.**
**Korea, Rep.** (source `South Korea`) and **ROW** therefore get `DEFAULT_BAND (0.0, 0.25)` on both
processes — Korea/Smelting's ceiling is active in 7 years, 1,061 kt. **Belgium** and **Taiwan** get
bands computed from a partial series.

### Items 2 and 3 are NOT defects — measured, closed

- **`.clip(upper=1.0)`** clips by ~1e-5 relative (**6 grams on 123 kt**) on 68 CFR observations in
  three regions, two of which (Egypt, Ukraine) are not model regions. It is float-noise cleanup on
  `scrap == anode ÷ eff`, not a masked data inconsistency. The TODO's earlier characterisation was
  wrong.
- The six **`[0.95, 1.00]`** bands come from the documented `fillna(0.0)` "missing counts as 0"
  convention (none of those six regions has a primary series at all), not from the clip. They block
  1,023 kt of primary input that is **100 % fabricated** — zero feed-observed edges, 505 kt of it a
  *domestic* `Concentrate→Smelting` edge in Belgium, a country with no mines. The bands are doing
  their designed faucet-stop job. **Do not widen them.**
- The minimum band width 0.05 is not an absent floor but the `max(0,·)` / `min(1,·)` truncation
  halving the intended ±0.05 margin exactly at the two ends. Measurement says those narrow ends are
  where the bands do their most useful work.

### Sequencing

Q-a and Q-b are **feed** fixes (full Stage 1-4 rebuild); Q-c is a **param** fix (scrap_share v3,
Stage 3-4 only). Q-a changes the level P.0 anchors, so Q-c's bands should be rebuilt *after* it —
otherwise Belgium's band is re-derived from the same inflated series.

Also open: **ROW** remains above 1.05 intake retention (Kazakhstan cleared in P.1).

### Carried out of Round Q, not fixed — the same bug in `stock_bounds_study.py`

`scripts/explo/stock_bounds_study.py` is the other param study that reads a feed's raw
`values.csv`. It clusters regions from the **source** labels and then looks each one up with
`acc.node_index["region"]` (**root** names), so `Korea, Rep.`, `ROW`, `Belgium` and `Taiwan` fall
through to `default_cluster` — whose comment reads "catch-all for regions absent from ICSG", and
they are not absent, only spelled differently. Same fix as ADR 0032's Q-c: map to root regions
when the feed is read. Deliberately left alone in Round Q so the scrap lever stays attributable;
it changes the `stock_bounds` params and so needs its own measured round.

Every other script reading a feed `values.csv` was checked and is region-map aware.


## THE CONCENTRATE EXPORT-MASS GAP (re-measure after Round O)

Round N went looking for a region-resolved concentrate grade and found this
instead. **Half of the mine-chain deletion is arithmetic, not optimisation.**

Copper leaves a mine node only as domestic smelter intake or as export mass
times a grade the hard cone caps at 0.35. So `cap = 0.35·X_mass + U` bounds what
a region can discharge at prior-level intake — and on the shipped run

```
HARD-CAPPED region-years                      144 of 965 with production
forced shortfall Σ max(0, P − cap)        25,913 kt CC = 49.0 % of all deletion
   Chile   14,270 (40.6 % of its own)   ROW 7,840 (92.6 %)   Zambia 1,460 (76.5 %)
```

Chile is hard-capped in **19 of 30 years**. Its BACI concentrate export mass
swings 12,401 (2020) → 4,723 (2021) → 4,088 (2022) → 10,269 (2023) → 4,951
(2024) while its mine output stays flat at ~4,200 kt CC and its anode and
cathode export lines do not move. Three checks put the defect in the **source**:
gross v5 ≡ netted v8 for Chile every year (ADR 0027 never touched it); HS 260300
itself drops, so it is not the sector map; and world concentrate trade excluding
Chile is stable while Chile halves, with China's recorded imports *from Chile*
falling 7,720 → 2,553 kt in the same step. The tonnage is absent from **both**
sides of the bilateral matrix.

This is an **input** defect, which is where all three large wins in this arc came
from (ADR 0022 −26.6 %, ADR 0027 −19.5 %, ADR 0015). Before anything is built it
has to answer:

1. Is the hole Chile's non-reporting, and does BACI's mirror reconstruction fail
   for HS 260300 specifically? Check the raw release for a reporter flag on
   Chile in 2021-22 and 2024, and on Australia→China 2021 (809 → 11 kt).
2. If the mass is unrecoverable, what is the honest repair — scale the
   region-year's export legs to the copper ICSG says left, widen σ on them, or
   let the mine node carry a stock? Each is a different claim about *who is
   wrong*, and each is measurable.
3. Does ROW's 92.6 % forced share have this cause or is it the aggregate's
   folded internal trade (ADR 0014)? Separate it before it contaminates a repair.

**Do not widen the [0.20, 0.35] band to relieve this.** It would license
concentrate at 60-90 % Cu (that is metal), and ADR 0028 already measured what
moving that wall does.

Detector and full evidence: `scripts/explo/roundN_concentrate_grade_diag.py`,
REP `…roundN-concentrate-grade`, ADR 0029.

## THE DISCHARGE PATH — why the softest thing pays (the other half)

Round M settled what the Chile diagnosis could only hypothesise. The grade bands
do **not** cause the ore deletion; they **shape** it. Closing them relocates the
same contradiction and, on the copper side, relocates it worse:

| | escapes through | Chile legs on the conc floor | US ore retention |
|---|---|---|---|
| base | Chile's ore-grade **ceiling** | 35 % | 0.990 |
| + year-local ore band | Chile's concentrate **floor** | 55 % | 1.307 |
| + narrowed conc band | **US ore** | **66 %** | **1.356** |

**Now bounded at 51 %** of the mine-chain deletion: the other 49 % is the
export-mass gap above, which no re-pricing can reach.

So the question is not where the walls are, it is **why the fluid exists and
who pays**. Chile's concentrate node is over-determined by 3,053 kt CC and the
posterior destroys **13,124 — 4.3x**. A disagreement between two ICSG figures
(a mine total and a smelter total) is settled by deleting the *trade and ore
that connect them*, because that is the softest thing in the neighbourhood:
**leg cv 0.069** after ADR 0017 damping against **scale-total anchor cv 0.017**.

Leads, in the ADR 0016 / 0017 / 0025 family:
- ADR 0016 already measured that *loosening* fabricated anchors collapses the
  chain (CO Mass 0.66 -> 0.41). So the fix is not "loosen the anchors".
- The `a_ref` item above is the same family and has its own measurement.
- A contradiction between two anchors should probably be *visible as a residual
  on those anchors*, not discharged through a third quantity neither of them
  measures. ADR 0019 (contradicted nodes lose anchor authority) and ADR 0020
  (withhold) are the existing precedents for that shape.

Rejected, with a working implementation in commit `f6bd5c9` and the numbers in
ADR 0028: year-local grade bands, and a narrowed concentrate band. Do not
re-propose either until the discharge path is understood; then re-measure
rather than assume.

## Chile's mine chain — the diagnosis that led here

**The instrument is a pair of HARD bands, both pinned.** `value_min`/`value_max` are
not prior shaping — `RatioBoundBuilder` makes them hard cones. Chile sits on the
concentrate *floor* (0.20) in 35 % of its export leg-years, 100 % of the Chile→Japan
lane, and on the ore-grade *ceiling* in 10 years out of 10. The ceiling is permissive
because ADR 0015 made the grade nominal region×year but left the band at the region's
30-year extremes: Chile's cap is 1.1× its **1995** grade, 75 % above its 2019 nominal.
World-wide only 5.9 % of region-years are at that cap; of the 12 biggest ore producers
only Chile (10/10) and China (7/10) are — and they are exactly the two whose ore mass
collapses (0.386, 0.555) while the US, Peru, Zambia and Indonesia hold at ~1.0.

Which gives the loss an exact decomposition:
**ore Mass retention 0.42 = 0.69 (copper deleted) × 0.61 (rock discarded at the cap).**

**Both band fixes were built, measured and REJECTED** (Round M, ADR 0028): the
ore band moved Chile's retention 0.386 → 0.566 and removed its impossible
1.662 % posterior grade, but the deletion relocated to the concentrate floor and
then to US ore. See the discharge-path section above, which is where this went.
