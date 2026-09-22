# BACI's quantity contradicts its own value; the quantity is the broken field

**Status:** accepted — Round R, 2026-09-22. Amends
[ADR 0029](0029-the-concentrate-grade-stays-global.md), whose **forced** term is now largely
explained and whose ROW question is answered negatively. Reuses the
[ADR 0027](0027-re-exports-are-re-attributed-at-feed-time.md) pattern (correct the value *and*
carry the inference in σ) while **rejecting** its σ formula for this population.

## Context

Five rounds cut the **discretionary** mine-chain deletion 26,958 → 12,435 kt and left the **forced**
term at 26,959 kt CC — 68.4 % of what remains. ADR 0029 defines it: copper leaves a mine node only
as domestic smelter intake `U` or export mass `X` times a grade the hard cone caps at 0.35, so
`cap = 0.35·X + U` and `Σ max(0, P − cap)` is arithmetic, not optimisation. Chile is hard-capped in
**19 of 30** years and carries 14,270 kt CC of it.

ADR 0029 suspected the export mass was under-recorded — *"0.94 is not a grade, it is a statement
that the recorded export mass is three times too small"* — but could only see it through
`(P−U)/X`, which mixes BACI's mass with ICSG's production and is a grade only where imports are
negligible. **It can be seen from inside BACI alone.**

HS 740311 is refined cathode (~99.99 % Cu), so its unit value *is* the copper price in the same
release, same year, same valuation convention. The **value-implied Cu fraction**

```
cu_frac = unit_value(HS 260300) / unit_value(HS 740311)
```

must then lie inside ADR 0028's physical band `[0.20, 0.35]`. Chile leaves it in **18 of 30** years,
reaching 0.65–0.68 — blister territory, not concentrate.

**The value field is intact; the quantity field is not.** Against ICSG production from the model's
own feed (`avail` = mine supply − domestic smelter intake; `from_v` = value ÷ copper price, which
uses **no grade at all**; `from_q` = quantity × 0.27):

| yr | avail | from_v | from_q | v/avail | q/avail |
|---|---|---|---|---|---|
| 2018 | 2965 | 2739 | 2909 | 0.924 | 0.981 |
| 2019 | 3159 | 2929 | 3470 | 0.927 | 1.098 |
| 2020 | 3016 | 3275 | 3348 | 1.086 | 1.110 |
| **2021** | 2960 | 3212 | **1275** | 1.085 | **0.431** |
| **2022** | 2763 | 2670 | **1104** | 0.966 | **0.399** |
| 2023 | 2803 | 2833 | 2773 | 1.011 | 0.989 |
| **2024** | 3252 | 3298 | **1337** | 1.014 | **0.411** |

BACI's value agrees with ICSG in **every** year; its quantity agrees in four and is 60 % out in
three — the same three the fraction flags, from data that never touches production.

**Alternatives refuted.** *Reclassification*: Chile's matte, anode, cathode and scrap exports are
flat or falling in those years (matte 101 → 69 → 10 → 6), so the mass reappears in no other HS line.
*Real collapse*: ICSG has Chile's mine output and smelter intake flat and its producer stocks at
45 kt, so 2,200 kt CC of copper would have nowhere to be. *A bad release*: both vintages agree
(V202501 4,247 kt vs V202601 4,723 kt).

**Cross-tabulated against ADR 0029's own detector**, 30 years, all exporters, root regions:
on Chile the two agree **18 of 19** years; globally they agree only 45 %, because `(P−U)/X` flags
385 of 639 region-years (60 %) against the fraction's 124 (19 %). The fraction is the more specific
instrument, exactly as ADR 0029's import caveat predicts.

## Decision

**Rebuild the quantity from the value, at the parse, on legs that contradict themselves — and give
the rebuilt leg the grade's declared σ.**

```
flagged  ⇔  Sector = 260300 ∧ q ≥ 1 kt ∧ cu_frac(leg) > 0.35 ∧ cu_frac(exporter-year) > 0.35
q_new    =  v / (P_cu · g_ref)          , never below q_obs
```

**Two gates, and both are load-bearing.** The ICSG cross-check licenses the *aggregate* value-to-
copper relationship per exporter-year, so that is where intervening is justified: in a year whose
own fraction is in band the recorded mass already matches the value, and lifting individual legs
there would **fabricate** mass. Within a flagged year the per-leg fraction decides *which* legs
moved, because the defect is a gradient across partners (Chile 2021: Brazil 0.19, Bulgaria 0.29,
Finland 0.39, Japan 0.46, **China 0.86** at 69 % of the export value), not a uniform per-country
offset. Measured: the leg gate alone flags Chile in all 30 years and recovers 121,630 kt; both gates
give **18 years and 82,320 kt**, within 2 % of the independent cross-tab.

**`g_ref` is measured, on exporter-years.** The median of that region's own in-band years — the
statistic ADR 0029 pooled, and the level at which the ICSG evidence lives. Chile's is **0.274**
against the global prior 0.275 and ADR 0029's 0.2783: three independent routes to the same number.
A per-leg median is dragged low by small legs (0.257), which would inflate mass, since
`q = v/(P_cu·g_ref)`.

**Threshold 0.35 is ADR 0028's own cone ceiling**, not a new constant: a leg above it is one the
model cannot represent anyway, because the hard cone would clip its CC/Mass. The conclusion does not
rest on it — relief measured across thresholds 0.35 / 0.40 / 0.45 / 0.50 / 0.60 is
**61 / 61 / 59 / 57 / 54 %**, and Chile's forced term closes at every one.

**σ: not ADR 0027's formula.** Measured, `√(q_obs² + (k·Δ)²)/q_new` **degenerates** here — raw
scales 0.839 / 1.028 / 0.960, all clipped to ~1.0, and *non-monotone*, because the denominator
outruns the numerator when a leg is multiplied by 2–3×. That form assumes a mass-conserving
re-attribution between observed legs. Here `v` and `P_cu` are well measured and **`g_ref` is the
uncertain input**, so the σ is the one the model already declares — the `conc` row's
`value_sd 0.0375` on `0.275`, cv **0.1364** — widened per region by its own in-band spread:

```
cv_gref(r)  = max( sd(cu_frac | in-band years) / g_ref(r) , 0.1364 )
cv_scale(r) = clip( √(1 + (cv_gref(r)/0.209)²), 1, 5 )
```

Chile earns **1.194** (a tight history, at the declared floor, as do Canada, the US and Australia);
Belgium 1.88, Italy 2.18, UK 2.25, Austria 2.56 — re-exporters whose fraction wanders earn real
doubt. A region with fewer than two in-band years has no measurable spread, which is ignorance, so
it goes to the **cap**, never the floor.

**The scales compose.** 3,593 of 6,706 Concentrate legs already carry an ADR 0027 re-export scale,
so `baci_feed.load_cv_scales` **multiplies** every scale file onto one array and caps the *product*
at 5.0. The previous loader assigned, which would have silently undone ADR 0027 on every overlap.

## Consequences

- Concentrate export mass **+82,320 kt** over 30 years (Chile +58,517), on 881 of 18,141 legs in
  every five-year block — **not** a recent defect.
- `LEG_SIGMA_AREF` stays pinned at `Concentrate: 10.26 kt`. Chile's leg masses roughly double, which
  would move an unpinned `a_ref` and re-price σ on *every* concentrate leg through ADR 0017's
  damping. **The pin is what makes this round attributable.**
- ADR 0027's netting re-derives on the corrected mass (a consequence, not a second lever) and is
  reported separately.
- The parse drops rows carrying a value but no quantity — **0.307 % of value** — now logged rather
  than silent.
- **40 % of the forced term is untouched.** ROW 7,576 kt CC is relieved 0 %, Zambia 1,418 kt 4 %:
  they are **not** this defect. That answers ADR 0029's open question — *"does ROW's forced share
  have this cause or is it the aggregate's folded internal trade?"* — negatively, leaving
  [ADR 0014](0014-bilateral-feeds-scoped-to-cross-region-edges.md) the candidate.

### Considered and rejected

- **σ-widening alone**, the first choice. [ADR 0019](0019-contradicted-flow-nodes-lose-anchor-authority.md)
  measured that remedy (0.46 % of mass moved, targeted collapses untouched) and CONTEXT.md records
  the verdict: *"σ reprices a data contradiction without resolving it."* The forced term is
  `cap = 0.35·X + U`; σ only *permits* the solver to raise `X`, while reconstruction raises it in
  the prior. σ encodes "we do not know" when the evidence says we do.
- **ADR 0027's `cv_scale` form** — degenerate here, as measured above.
- **A flat per-exporter factor** — the per-partner gradient refutes it.
- **Reconstructing matte / anode / cathode too.** Only concentrate mass enters `cap`, so it cannot
  move the forced term and would add confounds. The detector takes a sector table and the gate
  reports what the others would flag, as R.1's evidence.
- **Trusting BACI over ICSG.** Defensible only if ICSG's Chilean production were wrong, which no
  other evidence suggests and which the model already relies on everywhere else.

### Measured

Run `opt_20260922_092530_f21bdbe7` (30/30 Solved, cold), chain
`imp_20260922_090803_6da0f169` → `const_20260922_091301_c6edc72d`, baci **v11**.

| | Q | **R** |
|---|---|---|
| ADR 0029 forced deletion | 26,959 kt CC | **12,315** (−54 %) |
| forced share of all deletion | 68.4 % | **33.5 %** |
| Chile forced | 14,270 (40.6 % of its deletion) | **295** (**1.0 %**) |
| Chile hard-capped years | 19 of 30 | **5 of 30** |
| Chile MF→Concentrate retention | 0.574 | **0.617** |
| world MF→Concentrate retention | 0.905 | **0.912** |
| common-set sum-row z² | 1,698,662 | 1,705,881 (+0.4 %, still −0.7 % vs base) |
| **ROW forced (pre-registered null)** | 7,576 | **7,589 — unmoved** |

**The round reclassifies more than it reduces.** Total deletion falls only 39,394 → 36,807 kt
(−6.6 %); what moves is **~14,600 kt CC from forced to discretionary** (12,435 → 24,492). That is
the intended effect: the forced term is arithmetic on `cap = 0.35·X + U`, so correcting `X` removes
the wall and leaves a *fit* problem. The diagnostic confirms the mechanism is mechanical — scoring
**Q's own posterior** against the corrected feed already gives 12,339, so the ceiling rises without
the solver's cooperation. What the run shows is that the model then **uses** the headroom.

**R-h, the pre-registered null, is the strongest check**: neither ROW nor Zambia is value-flagged,
both were predicted not to move, and neither did. A correction that merely shuffled mass would not
respect a boundary it was never given.

**Costs:** z² +0.4 % (the price of asserting 82,320 kt the feed did not record); self-sufficient
intake retention 0.925 → 0.918; Belgium newly appears hard-capped at 1,029 kt CC forced against 300
deleted, which is Round Q's fabricated domestic `Concentrate→Smelting` edge surfacing in the
detector. **ROW is now 62 % of the remaining forced term** and the largest single open item.
