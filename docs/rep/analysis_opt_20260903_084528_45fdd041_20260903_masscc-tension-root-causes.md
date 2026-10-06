# Mass–CC tension: root-cause analysis (Lever 0)

**Run analysed:** `opt_20260903_084528_45fdd041` (all 30 years, 30/30 Solved) — the grade-basis
run — plus per-edge provenance from `imp_20260903_083103_4a7e63fd/imputation_log.parquet`
(an edge absent from the log is feed-covered; each imputed edge carries exactly one fill record).
**Question:** the residual Mass-vs-CC reconciliation tension (Chile export deletion, band-pinned
conc ratios) looked structural — quantify the three hypothesised root causes and rank them.

## 1. The reconciliation budget

Total soft penalty ≈ 6.0M z² (constraint rows 5.13M + edge-prior deviations 0.88M):

| Block | z² | share |
|---|---|---|
| **Scale totals (`sum` rows)** | **4.06M** | **68 %** |
| — on totals whose prior is *fabricated* (twin/recipe, no feed behind it) | 3.00M | 50 % |
| — on feed-backed totals | 1.06M | 18 % |
| `conctailingsuse` (global nominal) | 0.80M | 13 % |
| Edge deviations (0.54M on bilateral legs) | 0.88M | 15 % |
| `oregrade` 94k, `netstockbound` 71k, `strippingratio` 61k, rest | 0.25M | 4 % |

Per-edge conc rows on trade legs are near-zero z² (Cathode supply: 198,810 rows, 21 z² total) —
on any single leg the two valuations move together; **the ledger tension expresses only in the
totals**. Top sum offenders mix the genuine Mass–CC tension (Concentrate 710k, Anode 525k,
Cathode 506k, Chile Ore+Mining 149k) with fabricated-chain noise (Slovakia Anode 134k, India
Concentrate 113k, Portugal Cathode 93k — BACI trade exists, ICSG production doesn't).

## 2. Cause 1 — global conc nominals one level down (CONFIRMED, one-directional)

Implied conc = CC-total ÷ Mass-total per (supply sector, region, year):

| sector | nominal | posterior median | p10–p90 | at band floor | at ceiling |
|---|---|---|---|---|---|
| Concentrate | 0.275 | 0.264 | 0.210–0.277 | 6 % | 0 |
| Matte | 0.525 | 0.515 | 0.417–0.538 | 3 % | 0 |
| Anode | 0.940 | **0.902** | 0.900–0.939 | **49 %** | 0 |
| Cathode | 0.9975 | 0.995 | 0.995–1.000 | **58 %** | 16 % |
| Scrap | 0.700 | 0.722 | 0.462–0.945 | 9 % | 8 % |

Uniform downward direction = BACI gross Mass is rich relative to ICSG-CC ÷ nominal
(moisture/impurities/gross weights). World posterior concentrate conc drifts −10 % below nominal
(prior is consistent by twin construction; the gap opens at posterior). **Cathode is measurable
at prior level**: world implied cathode conc 0.956 vs 0.9975 (−4.2 % BACI-vs-ICSG gap) squeezed
through the [0.995, 1.0] band → forced into the totals (cathode sum 506k z²). `conctailingsuse`
(800k z², Panama z −84) is the same disease at the residual output — and internally derivable
from the ADR-0015 quantities.

## 3. Cause 2 — double anchoring of scale totals (CONFIRMED, largest block, degrades real data)

Feed-backed share of supply-prior tonnage — every sector has **at most one** observed valuation:

- Observed: CO Mass 91 % (S&P); Concentrate CC 96 %, Matte CC 92 %, Cathode CC 92 %, Scrap CC
  100 %, MF CC 78 %, SXEW CC 73 % (ICSG).
- Fabricated: Mining/Ore/MF/SXEW/Smelting/CFR/ER/CER/CP **Mass 100 % recipe/twin**;
  Smelting/CFR/ER/CER/CP CC also recipe-derived.

Yet **every scale total runs cv = 1.7 % identically** (ANCHOR_CV 0.01 + conditioning floor) — a
twin-derived total carries the same authority as an ICSG observation. GLOSSARY.md documents scale
totals as scale preservation, *"not to add information"* — the uniform tight cv over-preserves
fabricated priors. Result: 3.0M z² of phantom cost, and the drag **inverts the data hierarchy** —
observed totals move *more* than fabricated ones (median |move|: Concentrate CC 8.7 % vs Mass
4.9 %; Smelting CC 10.2 % vs Mass 3.3 %).

## 4. Cause 3 — cv-proportional σ on bilateral legs (CONFIRMED, the *shape* problem)

cv ≈ 0.21 nearly uniform on bilateral Mass legs → full deletion costs |z| = 1/cv ≈ 4.8
(z² ≈ 23) for a 5 kt leg and a 21 Mt leg alike. Deleted share of prior tonnage rises
monotonically with size decile: 7 % (smallest) → **37 % (largest)**. Chile 2015 concentrate
exports cut 69 % (8.8 → 2.8 Mt), each large leg zeroed at exactly z = −4.78; Chile CO Mass
retention 0.23–0.37 across years. Direct z² is modest (~0.5M) — the damage is *where*
reconciliation lands. **Bonus find:** the largest deleted legs are Papua New
Guinea→Philippines/ROW "Concentrate" at 14–21 Mt/yr — >20× PNG's production, a BACI
misclassification/gross-weight outlier class that should lose its authority at feed level
instead of being paid for at z ≈ 3.4–4.7 per leg-year.

## 5. Cause 4 (surfaced) — fabricated chains / feed coverage holes

Slovakia/Netherlands Anode ~224k, Germany Ore+MF ~110k, Belgium MF 54k, Portugal CP/CER 47k:
BACI shows flows, ICSG shows no production (EU-refiner-hole class, standing follow-up).
~0.5–0.9M z². Data acquisition, not σ design.

## 6. Ranked attack list

1. **Provenance-aware σ on scale totals** (3.0M z², 50 %): loosen cv on fabricated-prior groups
   (chosen); alternatives — drop those rows (drift risk) or propagate edge σ (later refinement).
2. **Conc recalibration, cheapest first** (~1.5–2.0M attributable): regionalise tailings conc
   internally (chosen quick win); then empirical cathode/concentrate recalibration; regional
   concentrate-conc data acquisition (external only — route-implied conc is circular, ICSG trade
   CC ≡ 0.275·BACI Mass).
3. **Bilateral-leg σ design** (the isolated Chile lever) — decide **after** #1 with re-measured
   deletion economics; regardless, screen the PNG-class outliers now via σ inflation (chosen).
4. **Feed coverage** (EU refiners, Germany ore chain): data acquisition, standing.

**Round A (branch `fix/masscc-round-a`)**: provenance-aware scale-total σ + regional tailings
conc (ADR 0015 amendment) + outlier-leg σ screen; validation vs this run per the approved plan.
