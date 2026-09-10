# India Concentrate — the fan's domestic residual books copper the region never mined

- run analysed: `opt_20260910_114200_7f4040d9` (shipped Round I) ← `const_20260910_112841_87020ec3`
  ← `imp_20260910_112302_4040ad37`; raw feeds icsg `20260909_082747_a347d543`, baci `20260902_085153_de9e393f`
- reproducer: `python scripts/explo/india_concentrate_diag.py` (reads run artifacts + raw ICSG/BACI; changes nothing)
- status: **diagnosis only — no code change, no re-solve.** Remedy options ranked in §5, decision pending.

## 1. The cell

`usetotout_Concentrate_India` is the largest single region-sector cell on the run: **112,955 z²**
pooled over 30 years (CC 56,764 + Mass 56,190), against 13,202 for India's next sector (Cathode).
The row pays z ≈ −40 to −55 in 24 of 30 years:

| year | anchor c (CC kt) | posterior Σ | z |
|---|---|---|---|
| 2003 | 219.3 | 32.0 | −50.1 |
| 2012 | 300.9 | 32.2 | −52.4 |
| 2015 | 430.4 | 31.8 | −54.3 |
| 2017 | 425.3 | 34.1 | −54.0 |
| 2020 | 22.8 | 23.2 | +1.0 |
| 2024 | 180.2 | 29.8 | −49.0 |

The anchor is `Σ f_agg` over `Concentrate.India → *` (ADR 0006 scale total), i.e. the fanned
domestic leg plus exports; the posterior is what the **hard Concentrate-node balance** allows,
which is India's own mine production (23–48 kt CC/yr, ICSG-covered). 2020 is the control: the
anchor happens to equal production and z ≈ 0.

## 2. Mechanism — a construction, not raw data

ADR 0002 option D fans each ICSG single-endpoint usage row over producer regions with

    domestic (X==Y) weight = max(0, usage_CC − Σ_X conc · BACI_Mass(X→Y))

and states the domestic leg *"needs no ICSG production data"*. For India the residual is
100–456 kt/yr against 23–48 kt mined. Edge families around the node, CC kt, 2015:

| family | prior | posterior |
|---|---|---|
| MF.India → Concentrate.India (mine production, ICSG) | 30 | 32 |
| Concentrate.India → Smelting.India (domestic residual) | **417** | **27** |
| Concentrate.X → Smelting.India (BACI imports · 0.275) | 414 | 657 |
| Concentrate.India → Smelting.X (exports) | 13 | 5 |
| Scrap → Smelting.India | 3 | 4 |

The solver does the physically sensible thing — domestic ≈ production, imports scaled up
+59 %, smelter input 688 vs ICSG 821 — and pays z −54 on an anchor built from a residual.

## 3. Which side is short for India

ICSG smelter use vs BACI concentrate imports (HS 2603) at the 0.275 nominal:

| year | BACI gross imports | ×0.275 | ICSG smelter use | mine | residual → domestic | grade imports would need |
|---|---|---|---|---|---|---|
| 2005 | 1,058 | 291 | 500 | 26 | 209 | 0.45 |
| 2010 | 1,774 | 488 | 678 | 28 | 190 | 0.37 |
| 2015 | 1,470 | 404 | 821 | 30 | 417 | 0.54 |
| 2017 | 1,603 | 441 | 843 | 32 | 402 | 0.51 |
| 2020 | 1,693 | 465 | 252 | 23 | 0 | 0.14 |
| 2024 | 879 | 242 | 416 | 27 | 174 | 0.44 |

Over 2003–2018 the imports would have to carry **36–59 % Cu**; concentrate is 20–35 %. So the
pair is inconsistent by ~40 %: either BACI understates India's gross concentrate imports or ICSG
overstates its smelter production. 2015 sources (gross kt): Chile 391, Indonesia 339,
Australia 222, Peru 153, Brazil 147. Not decidable from the repo; what *is* decidable is that
booking the gap as domestic mining is wrong under any reading.

## 4. The class

Fanned domestic-leg prior > own production prior, CC, 30 y pooled, on the imp run:

| flow sector | region-years | excess (kt) | usetotout z² on those rows | z² on the other rows |
|---|---|---|---|---|
| Concentrate | **210** | **21,707** | **348,142** (430 rows) | 285,045 (2,132 rows) |
| Matte | 33 | 1,270 | 30,092 | 125,433 |
| Anode | 0 | 0 | — | — |

The Concentrate excess rows are **12.0 % of all sum-row z²** (2,893,508). Anode carries none
since ADR 0021/0022 — its usage anchor is total refined production, which the anode inflow covers.

Per region, the grade BACI imports would need over that region's excess years splits the class
into three kinds:

| region | excess years | needed grade | BACI gross kt/y | ICSG use | mine | kind |
|---|---|---|---|---|---|---|
| India | 24 (1999–2024) | **0.44** | 1,042 | 499 | 30 | importer, pair inconsistent |
| Japan | 10 (1995–2004) | **0.40** | 3,195 | 1,309 | 1 | importer, pair inconsistent |
| Brazil | 9 (1995–2003) | **0.42** | 322 | 176 | 31 | importer, pair inconsistent |
| Philippines | 20 (1995–2024) | 0.34 | 414 | 211 | 46 | marginal — a regional import grade would close it |
| China | 27 (1995–2024) | 0.32 | 4,144 | 2,542 | 1,045 | marginal (9.5 Mt excess, 32 % of the class by tonnage) |
| Spain | 8 (1995–2005) | 0.31 | 813 | 269 | 16 | marginal |
| United States | 6 (1997–2003) | 1.24 | 96 | 996 | 886 | self-sufficient: use exceeds mine by 5–12 %, imports negligible |
| Poland | 23 | 0.62 | 44 | 477 | 427 | self-sufficient |
| Uzbekistan | 25 | 0.90 | 22 | 104 | 80 | self-sufficient |

For the self-sufficient kind the residual is not BACI's doing: ICSG smelter input (production ÷
`eff_smelting`) exceeds ICSG mine output with nothing to import. That is an ICSG-internal or
eff-nominal inconsistency (or stock draw / secondary feed) and a cap cannot be spread onto
imports there.

A crude counterfactual — cap the anchor at production + exports, hold the posterior — puts
India at **112k → 12k**, Philippines 54k → 27k, Spain 29k → 11k, Brazil 16k → 3k, Japan
68k → 47k, China 30k → 20k. (Not valid for Matte/Congo, where the capped anchor is ≈ 0.)

## 5. Doc check and options

- **ADR 0002** defines the residual and its "no production data" property — the thing at issue.
- **ADR 0018** widened it to import-free fans (residual = usage) and named the "fabricated chain"
  class it cures; the self-sufficient kind above is that class's Concentrate face.
- **ADR 0019** detects exactly this contradiction (outflow anchor ≫ inflow prior) but its scope is
  `("Matte","Anode")` — Concentrate was the *inflow evidence* in that ADR, never a monitored node —
  and its σ action is inert (`UNCLOSED_ANCHOR_CV = None`) after Round E showed σ reprices without
  resolving. CONTEXT: "the remedy is data-side".
- **ADR 0021** precedent: a derivation placed where it cannot hold is fixed in the derivation.

Options, ranked:

1. **Bound the domestic weight by the region's own ICSG mine production** in
   `build_trade_shares._raw_weights`; spread the remainder of the usage over the import legs in
   BACI proportions (usage conserved). Where there are no import legs to carry it (self-sufficient
   kind, 67 region-years, 2.8 Mt) keep the present behaviour and report via the ADR 0019 detector
   extended to Concentrate (detector only). Amends ADR 0002's "needs no production data" to
   "bounded by production" — which the hard node balance enforces on the posterior anyway, so the
   prior is only being made consistent with the model's own physics (ADR 0015's principle). Tension
   moves from a fabricated domestic anchor to the import legs, i.e. to the real BACI-vs-ICSG
   scale disagreement, where ADR 0017's size-damped σ lives. Blast radius: every exporter's
   `usetotout_Concentrate` anchor sees scaled-up import legs — Chile/Peru/Indonesia retention must
   be watched. **A measurement round (J).**
2. **Extend ADR 0019 to Concentrate and set `UNCLOSED_ANCHOR_CV`.** Two constants. Round E's
   lesson: the posterior is already at production, so this removes the fine without changing the
   answer (~100k), and it hides a construction defect behind a σ. Not recommended alone.
3. **Regional concentrate import grade** (ADR 0015 pattern). Closes the marginal kind (China,
   Spain, Philippines: 0.31–0.34 needed) but not India/Japan/Brazil, and has no data source in the
   repo — it would be tuned to the residual it is meant to remove.

**Recommendation:** Round J = option 1, Concentrate only (Matte's 33 region-years as follow-up
once the rule is measured), with the detector extension as the reporting instrument and the
Round-E stop rule: ship only if mass moves with the metric and exporter retention does not
regress. India's expected landing is ~12k on its cell; the class ~350k → measured, not assumed.
