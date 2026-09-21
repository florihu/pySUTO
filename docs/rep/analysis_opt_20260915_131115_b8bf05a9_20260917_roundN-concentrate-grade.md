# Round N — the concentrate grade is not the lever

**Run measured:** `opt_20260915_131115_b8bf05a9` (the shipped Round L run, sum-row z² 1,720,192)
**Feeds:** baci v8 (netted), icsg v8, crude_ore v3 · **Params:** conc v8, ore_grade v7
**Date:** 2026-09-17 · **Branch:** `feat/region-resolved-concentrate-grade`
**Verdict: N.1 not built. The thesis is refuted on measurement; N.0 stands on its own.**

---

## 1. What Round N proposed

`conc_Concentrate` is a single global 0.275 for every concentrate leg from every mine to every
smelter. The plan's evidence was Chile's own books over 2010-19:

```
ICSG mine            38,000 kt CC
− domestic smelting  13,428
= must leave by sea  24,573
÷ BACI export mass   75,549
= 0.325
```

0.325 sits inside the existing hard band [0.20, 0.35], so the model *may* use it and does not. The
round would have derived one grade per region per 5-year window from the coupled node identity

```
g_r·X_mass(r) − Σ_A g_A·mass(A→r) = P(r) − U(r)
```

by ridge-regularised, box-bounded least squares, and retired the two global rows.

The estimator was built and run. **The estimator works. The premise does not.**

## 2. The pooled 0.325 is an artifact of pooling

Chile's implied grade `(P−U)/X`, year by year, on the netted BACI matrix:

| | 1995 | 1996 | 1997 | 1998 | 1999 | 2000 | 2001 | 2002 | 2003 |
|---|---|---|---|---|---|---|---|---|---|
| g | 0.582 | 0.944 | **0.327** | 0.902 | 0.938 | 0.912 | 0.357 | **0.294** | 0.941 |

| | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| g | **0.318** | **0.291** | **0.278** | **0.282** | **0.275** | **0.246** | **0.243** | 0.627 | 0.676 | **0.273** | 0.657 |

Copper concentrate assays 20-35 % Cu. A value of 0.94 is not a grade; it is a statement that the
recorded export **mass** is about three times too small. Only 11 of 30 Chile-years land inside the
band at all.

**On those 11 credible years — 94,706 kt, 59 % of Chile's recorded export mass — Chile pools to
`g = 0.2783`.** Against the global prior 0.275 that is a difference of **0.003**, less than a tenth
of the σ the model already carries on that row (0.0375).

The plan's 0.325 came from averaging the credible years with the mass-deficient ones. Pooling a
ratio whose denominator has holes moves the ratio, not the grade.

## 3. There is no region signal worth a params family

Restricting to near-pure exporters (imports < 10 % of exports), excluding screened origins, and
keeping only observations inside the physical band — 124 region-years, 324,519 kt of export mass:

```
world mass-weighted mean g                    0.2640
between-region variance share                  15.8 %
within-region (year-to-year) variance share    84.2 %
mass-weighted sd of the region means           0.0117   (prior σ = 0.0375)
```

**The dispersion between regions is a third of the uncertainty the single global row already
declares.** The four biggest credible exporters cluster tightly and *below* the prior:

| region | n | export mass kt | pooled g | in-band years |
|---|---|---|---|---|
| Chile | 30 | 159,411 | 0.397 (0.278 on credible years) | 37 % |
| Peru | 23 | 109,158 | 0.256 | 100 % |
| Indonesia | 26 | 52,149 | 0.256 | 77 % |
| Australia | 29 | 45,046 | 0.246 | 97 % |
| Canada | 12 | 14,422 | 0.248 | 75 % |
| United States | 16 | 13,469 | 0.266 | 50 % |
| Kazakhstan | 23 | 12,844 | 0.193 | 35 % |
| Congo, D.R. | 6 | 5,099 | 0.387 | 0 % |
| South Africa | 13 | 3,586 | 0.070 | 15 % |
| Botswana | 14 | 830 | 0.113 | 29 % |

The regions that *look* like they carry a distinctive grade are exactly the regions whose in-band
rate is low. Kazakhstan's 10th percentile implied grade is **−0.221**, Iran's **−0.344**, South
Africa's **−0.027**. A negative grade is not a measurement; it is proof that the residual those
regions carry is not a grade at all.

## 4. What the residual actually is

Chile's export mass in the raw BACI source, HS 260300, against its other copper exports:

| year | 260300 conc | 740200 anode | 740311 cathode | ICSG mine CC |
|---|---|---|---|---|
| 2019 | 12,852 | 254 | 2,218 | 4,207 |
| 2020 | 12,401 | 296 | 2,289 | 4,266 |
| 2021 | **4,723** | 276 | 2,196 | 4,211 |
| 2022 | **4,088** | 322 | 2,130 | 3,915 |
| 2023 | 10,269 | 251 | 2,002 | 3,916 |
| 2024 | **4,951** | 265 | 1,895 | 4,283 |

Chilean mine output is flat. Cathode and anode exports are flat. Only the concentrate line halves
and doubles. Three checks place the defect in the source:

- **Not the netting.** Gross v5 and netted v8 are byte-identical for Chile in every year — ADR
  0027 never touched it.
- **Not the sector map.** HS 260300 itself drops; no other copper code absorbs it.
- **Not a re-attribution.** World concentrate trade excluding Chile (and the screened PNG/Italy
  origins) is stable at 25,800-29,700 kt across 2019-22 while Chile falls 12,401 → 4,723. The
  importers lost it too: China's recorded imports **from Chile** go 7,720 → 2,553 and its total
  concentrate imports 21,539 → 17,040. Australia→China goes 809 → 11 in the same step.

The tonnage is absent from both sides of the bilateral matrix. This is a hole in BACI's concentrate
mass series, not a property of Chilean ore.

## 5. Why that hole *is* the deletion — and why no grade can close it

The model is handed ICSG's mine production in CC and BACI's export tonnage in Mass, joined by a
grade that may not leave [0.20, 0.35]. Copper can leave a mine node only as domestic smelter intake
or as export mass × grade. So at the band **ceiling**, region-year discharge capacity is

```
cap = 0.35 · X_mass + U        (prior-level intake; the solver can raise U, at its own cost)
```

Chile, shipped run:

```
HARD-CAPPED (prior mine P > cap) in 19 of 30 years
forced shortfall Σ max(0, P − cap)          14,270 kt CC
total Chile mine deletion                   35,117 kt CC
                                            → 40.6 % of Chile's deletion is arithmetic
```

World, all regions:

```
mine prior      414,781 kt CC        posterior 361,885        deleted 52,896 (12.8 %)
HARD-CAPPED region-years                    144 of 965 with production
forced shortfall Σ max(0, P − cap)       25,913 kt CC = 49.0 % of ALL mine-chain deletion
```

| region | forced kt CC | of its deletion |
|---|---|---|
| Chile | 14,270 | 40.6 % |
| ROW | 7,840 | 92.6 % |
| Zambia | 1,460 | 76.5 % |
| Congo, D.R. | 1,164 | 165.7 % |
| Russia | 175 | 5.3 % |

*(Shares above 100 % mean the solver discharged that region's surplus by raising intake or stock
instead of deleting the mine edge — `cap` is a prior-level diagnostic, not a theorem.)*

**Half of the deletion this arc has been chasing since Round I is forced before the solver starts.**
And it is forced *at the ceiling*: raising a region's grade from 0.275 to 0.325 cannot help a year
already evaluated at 0.35.

The two halves of the diagnosis are the same 11 years. Chile's headroom years —
1997, 2002, 2009, 2014-20, 2023 — are **exactly** its in-band years from section 2, and they pool to
`g = 0.2783`, needing 0.243-0.327 individually. So: where Chile can physically discharge its copper,
its implied grade is ordinary and sits at the prior; where its implied grade is impossible, no grade
can discharge it. Round N's lever pushes in the wrong direction in the years it could act, and
cannot act at all in the years that matter.

## 6. Why the estimator confirmed this rather than hiding it

The coupled solve was built and run over centred 5-year windows — `scripts/explo/
roundN_concentrate_grade_diag.py` §5, which reproduces every number in this REP. Three gates:

| gate (window 2018-2022) | eqs | at bound | export mass at bound | ‖residual‖ |
|---|---|---|---|---|
| raw | 46 | 33/46 | 57.6 % | 25,883 |
| + require both P and U observed | 24 | 35/46 | 61.3 % | 13,150 |
| + drop screened origins (PNG, Italy) | 24 | 28/46 | 47.2 % | 1,303 |
| + drop ROW's node equation (an aggregate; ADR 0014 folds its internal trade) | 23 | 23/46 | 51.9 % | 1,082 |

Each gate is defensible on its own terms and each was measured rather than assumed — screening the
PNG units defect alone takes the residual from 13,150 to 1,303. But even fully gated, **23-31 of 46
regions land on a bound in every window, carrying 20-58 % of export mass**, and Chile clips at the
0.35 ceiling in 4 of 6 windows. That is the estimator reporting the
same thing section 5 does: for much of the record no grade inside the physical band closes these
books. The plan's acceptance criterion (e) — *"the clipped population is confined to the known
defects, not to a major producer"* — fails on its own terms.

## 7. Decision

**N.1 is not built.** Building it would write 2,820 params rows to move Chile by 0.003 while
relocating an infeasibility that is 49 % arithmetic — the Round M failure mode exactly, one level
further in. The Round M rule stands: *the bands shape the deletion, they do not cause it*, and this
round adds *neither does the nominal*.

**N.0 stands.** The region-keyed concentration readers (`bd14f04`) are inert by construction and
verified so — no artifact bumped, Stage 2 exact to 0. They remove a real silent-overwrite hazard
and they are what a future region-resolved family would need. Keep them; they cost nothing.

**Superseded by this measurement**, as recorded in ADR 0029:

- the per-region ratio `(P−U)/X_mass` as evidence of a grade — its denominator has holes;
- the plan's persistence statistic (`|Σr|/Σ|r|` = 0.738, Chile 0.927) as evidence of a *grade*
  bias. The statistic is right and the reading was wrong: a region whose export mass is
  systematically under-recorded carries a persistently positive residual that is indistinguishable
  from a persistent grade bias by that statistic alone. Persistence separates bias from noise; it
  does not separate a mass defect from a grade defect. The band test does.

## 8. What this sends next

**The concentrate export-mass gap is the round.** It is a Stage-1 feed defect with a measured size
(25,913 kt CC forced, 49 % of mine-chain deletion), a clean detector (`P > 0.35·X + U` at prior
level), and a population of 144 region-years that names itself. Unlike every lever tried since
Round I it is an *input* fix, which is where every large win in this arc has come from — ADR 0022's
`effer` flag (−26.6 %), ADR 0027's netting (−19.5 %), ADR 0015's ore grades.

Open questions it has to answer before anything is built:

1. Is the hole Chile's non-reporting, and does BACI's mirror-flow reconstruction fail for HS 260300
   specifically? Check the raw BACI release for a reporter flag on Chile in 2021-22 and 2024.
2. If the mass is genuinely unrecoverable, what is the honest repair — scale the region-year's
   export legs to the copper ICSG says left, widen σ on them, or let the mine node carry a stock?
   Each is a different claim about *who is wrong*, and the choice is measurable.
3. Does ROW's 92.6 % forced share have the same cause or is it the aggregate's folded internal
   trade (ADR 0014)? ROW needs separating before it contaminates a repair.

The discharge path (leg cv 0.069 against anchor cv 0.017) remains the standing target for the other
half of the deletion, and is now bounded: it can explain at most 51 % of it.
