
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

## ROUND O — THE SECONDARY DOUBLE COUNT (built + measured 2026-09-21; HELD, not merged)

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

## ROUND P — ANCHOR THE SMELTER-SCRAP LEVEL (the active target, from Round O's measurement)

Round O moved copper from a route the model anchors **by level** (`Refining/Scrap`) to one it anchors
**only by share** (`Smelting/Scrap`: `0.95 ≤ Σ(Scrap→Smelting)/Σ(all→Smelting) ≤ 1.00`). A share band
over an idle node is free: ICSG has Slovakia's smelter eating ~28 kt/yr of scrap, the model routes
**0.06 kt/yr in both runs**. Where the smelter node has throughput the substitution works (Germany
+0 %, Japan −2 %, China −8 % anode); where it does not the copper is lost (Belgium −21 %, Austria
−27 %, Slovakia −79 %).

1. Make ICSG `Smelting/Scrap` a **soft scale total** on `Σ(Scrap→Smelting)` per region-year — the
   treatment `Refining/Scrap` already gets — *in addition to* the share band.
2. Stop deriving a band from a series the same commit rewrote: `scrap_share` v2 flipped Slovakia's
   CFR band from [0.950, 1.000] to [0.000, 0.280] because the ADR 0030 rule had just clipped the
   series it is computed from, closing the second route too. Floor the band width, or derive the CFR
   band from *total* secondary refinery input and let the level anchors split the routes.
3. Re-run the chain and re-measure with `scripts/explo/roundO_secondary_diag.py`. Expect the
   Anode/MF/Concentrate z² back and the secondary share to fall toward 0.17 on its own. Only then is
   Round O mergeable; recheck Kazakhstan (>1.05 intake retention) and Chile's 19/30 cap there.

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
