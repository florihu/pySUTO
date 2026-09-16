
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

## THE DISCHARGE PATH — why the softest thing pays (the active target)

Round M settled what the Chile diagnosis could only hypothesise. The grade bands
do **not** cause the ore deletion; they **shape** it. Closing them relocates the
same contradiction and, on the copper side, relocates it worse:

| | escapes through | Chile legs on the conc floor | US ore retention |
|---|---|---|---|
| base | Chile's ore-grade **ceiling** | 35 % | 0.990 |
| + year-local ore band | Chile's concentrate **floor** | 55 % | 1.307 |
| + narrowed conc band | **US ore** | **66 %** | **1.356** |

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
