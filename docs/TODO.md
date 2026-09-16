
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

## Chile's mine chain — DIAGNOSED 2026-09-16, experiment pending

REP `analysis_opt_20260915_131115_b8bf05a9_20260916_chile-mine-chain.md`.

Not Chile's defect: **trade's**. Self-sufficient nodes hold their anchors (US 1.000,
Russia 1.024, Zambia 1.014); trade-dependent ones lose a fifth to a third (Korea
0.717, Germany 0.737, China 0.748, Japan 0.789, Chile's mine 0.655). Chile is 69 % of
the world shortfall only because it is the largest number attached to a
trade-dependent node. Nothing downstream backs up (anode 0.994, cathode 1.002) — the
hole is healed by scrap backfill and by implied efficiency drift (smelting
0.917 → 0.963, refining 0.881 → 0.983), both silent.

**The lead:** `conc_Concentrate` is a single global constant — `a_region = *`,
nominal 0.275, band [0.20, 0.35] — so every leg from every mine to every smelter is
priced at 27.5 % Cu. The grades the nodes need are not uniform and not 0.275: Chile
0.325, China 0.292, Korea 0.315, Germany 0.320, Japan 0.260, Spain 0.371, Brazil
0.384, India 0.466. Chile's 0.325 is *inside the parameter's own band*, and the
solver went the other way to 0.210. This is exactly the defect ADR 0015 fixed for
**ore** grades and never applied to **concentrate** grade.

**Next:** region×year concentrate grade, then measure. Caveat before spending on it:
Chile's node is over-determined by 3,053 kt CC and the posterior destroys 13,124 —
a **4.3× amplification** that the grade alone does not explain. If closing the node
does not stop the deletion, the fault is in how infeasibility is discharged (leg
cv 0.069 vs anchor cv 0.017) — the same family as the `a_ref` item above.
