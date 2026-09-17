
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

## THE CONCENTRATE EXPORT-MASS GAP (the new active target, 2026-09-17)

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
