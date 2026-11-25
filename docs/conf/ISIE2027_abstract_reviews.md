# ISIE 2027 abstract: reviews, responses and evidence

**Current version: v27, `docs/conf/ISIE2027_abstract.tex` (2026-11-25), on the manuscript's reference chain `opt_20261124_141714_e2892160`; see Round 7 and "Claim → evidence (v27)" below. The .docx (v20) and everything from here to Round 7 describe the earlier chain `opt_20261022_112855_928da10c` and are history.**

Abstract (v1–v20): `docs/conf/ISIE2027_abstract.docx`, rewritten from `ISIESEM_abstract_finis.docx` (ISIE-SEM, Cambridge).
Call: https://www.ntnu.edu/isie2027/call-for-abstracts. Deadline 30 November 2026 (extended from 15 November).
Reference chain: `opt_20261022_112855_928da10c` (manuscript-ready-3). Stage 5: `uncprop_20261029_090754_0446e784`. Knob sweep: `sweep_20261102_145601_8413cace`.
Drafted and reviewed 2026-11-09. Two review rounds, each with two independent reviewer agents that had not seen earlier drafts or reviews: round 1 on draft v2 — (A) committee and positioning, (B) evidence and model fidelity; round 2 on draft v5, after the author's re-brief (methods in the foreground, modularity, traceability, open source, AISHA and Pauliuk 2024 references, Spaceship Earth reduced to one sentence at each end) — (C) content and positioning, (D) a scorer of AI-typical prose. v6 closed round 2. Round 3 (2026-11-10, no reviewer agents): the author asked for a Results section about the reconciled copper system — scale, distribution, uncertainty and sensitivity — instead of the diagnostics after the hub sentence (v10), then that every indicator in the Results be briefly explained (v13). Round 4 (2026-11-10): a readability and flow pass (v15), then two fresh reviewers that had seen no earlier draft or review — (E) peer review with repository access, verifying every number and gloss; (F) readability, flow and AI-prose markers, text only. v17 closed round 4. Round 5 (2026-11-10): the author removed the inline citations; the lineage is named instead (AISHA, IELab) and the freed words went to the clarity items of round 4 that budget had refused. v18 closed round 5. Round 6 (2026-11-10): the author asked for a two-sentence insight on the secondary share, contrasting China, the EU and Japan/South Korea, as temporal development; it replaced the Herfindahl sentence, the four-country output-total sentence, the Chile–Peru embodied ratio and the unnamed gap sentence. v20 is the final version.

## Before submitting (the author's decisions)

**Status for v24 (2026-11-25):** item 1 (co-authors) is still open; item 2 is resolved (MIT `LICENSE`, `pyproject.toml`); item 3: oral, 478 words (484 with the headings); item 6 is moot (termination left the text on the author's request); item 7 stands ("Stage components are plug-ins" names no solver). New: the call says titles have "a maximum of 20 words (no abbreviations)"; the author keeps "pySUTO" in the title (19 words) as a software name, defined at first use in the body. The keyword line's MFA and IO/EE-MRIO are the official keyword list's own labels.

1. **Authors.** The call requires every co-author in order, with the presenting author marked. The .docx has a placeholder.
2. **Licence.** The repository has no LICENSE file, `pyproject.toml` declares none, and GitHub reports none, so "open-source" and the title's "open" are not yet true. Add a licence before 30 November or change the wording. method.md §3.4 still reads "[DOI and licence to be inserted]".
3. **Format.** The text is written for oral (≤ 500 words). A poster version (≤ 250 words) would be a separate cut.
4. **Affiliation.** The ISIE-SEM bio read "Vienna University of Business and Economics". WU's English name is Vienna University of Economics and Business, which the .docx uses. Confirm the group name.
5. **References in the body.** None since v18: the author removed the five author–year citations (Lenzen et al. 2009; Geschke et al. 2011; Pauliuk and Heeren 2020; Wieland et al. 2022; Pauliuk et al. 2024). The lineage is named without references — AISHA (expanded) and the Industrial Ecology Virtual Laboratory (IELab) in place of Lenzen et al. (2009); the frameworks and laboratories of the gap sentence are unnamed. The `refs.bib` note on the AISHA entry ("verify: booktitle/year") now matters only for the paper.
6. **Solver-termination sentence (Reviewer E, round 4).** The Results no longer say that tightening solver tolerance alone moved one retention by 0.22 (A5), cut in round 3 for the system results. Reviewer E calls it the paper's headline on stability and the one thing a methods audience would remember, and notes the Objective promises to test whether the account is "as stable as claimed". Reinstating costs nine words ("solver termination alone moves one by 0.22" after the sweep sentence), paid by "(0.108–0.114)" and "chiefly hubs such as Switzerland and Singapore". Author's call.
7. **Solver claim dropped.** The brief asked to indicate that the solver can be exchanged. `src/pysuto/solver.py` is one concrete `ClarabelSolver` with no abstract base, unlike the four real seams (`DataFeed`, `BaseImputer`, `BaseConstraintBuilder`, `BaseUncProp`). The persisted constraint run makes a replacement possible, but a reader would hear "plug-in solver", so the abstract names only the four seams. Add a solver base class if the claim should go in.

## Submission fields

- **Title** (17 words, no abbreviations): When the gauges of Spaceship Earth disagree: an open, traceable framework for reconciling global physical supply–use tables
- **Presentation format:** oral
- **Main topic:** Cross-Cutting Methods and Emerging Topics. The alternative is Input-Output and Economic Systems Analysis, which puts the talk in front of the Eora/PIOLab compilers.
- **Keywords (5):** Material flow analysis (MFA) · Input-output analysis (IO/EE-MRIO) · Supply chains · Footprinting · Energy transition
- **Word count:** body 493 (98 / 105 / 231 / 59); 499 if a counter includes the four section headings. A counter that splits hyphenated words gives more, and the call does not say how it counts. If the portal rejects it, cut "(0.108–0.114)" (−1), "(sum of squared regional shares)" (−5) or "chiefly hubs such as Switzerland and Singapore" (−6).
- **Alternative titles** (Reviewer C): "Reading the gauges of Spaceship Earth together: an open, traceable framework for global physical supply–use tables" (17); "Tracing what moves through Spaceship Earth: open, traceable reconciliation of global physical supply–use tables, demonstrated on copper" (17). "Gauges" is our word, not Fuller's; do not put it in quotation marks.

## What changed against the Cambridge abstract

| Cambridge abstract | Current model (method.md, results.md) |
| --- | --- |
| Linear Programming, disaggregation by LP | One σ-weighted QP per year (Clarabel), hard and soft rows; no disaggregation step |
| 80 regions, 50 sectors | 47 regions (46 named + rest of world), 23 sectors, gross mass and contained copper |
| DMC, RMC, stock accumulation | The boundary ends at cathode into semis fabrication; there is no in-use stock model. Stage 6 derives PSUT and PIOT views (footprints, embodied trade, ew-MFA) |
| No uncertainty | Pedigree σ from each source, power-mean σ rule, Stage-5 posterior, Birge check |
| ~600 words, unstructured, special-session line | ≤ 500 words in the four required sections; theme and keywords from the official lists |

## Round 2 — Reviewer C (content and positioning): verdict "revise, then oral"

| Issue | Response in v6 |
| --- | --- |
| "Since AISHA, compilers treat every data point as an uncertainty-weighted constraint" misattributes the idea: Byron 1978, van der Ploeg 1982 and Lenzen et al. 2009 precede it; AISHA is the KRAS-based tool behind Eora, not a data model; AISHA is an acronym and must be expanded (paper footnote 19) | Accepted. "Since Lenzen et al. (2009) and AISHA (An Automated Integration System for Harmonised Accounts; Geschke et al., 2011), compilers treat …". |
| "The solver is replaceable at its stage boundary" overclaims: one concrete class, no base | Accepted; sentence removed (item 6 above). |
| "Every number traces to raw data (Pauliuk et al., 2024)": steps 2–5 of Box 1 are met, but the sidecar records no package versions (step 4a) and the raw yearbooks are licensed PDFs parsed outside the registry (step 1) | Accepted. "Every flow traces to its sources and fills (Pauliuk et al., 2024)"; Conclusion: "audited back to its sources and fills". |
| "Plug-in stages apply beyond copper, to other materials, regions and constraint sets" is untested; `impute/share.py` hard-codes copper node names | Accepted. "By design, copper enters only through the accounting structure, feeds and parameter tables; a second material will test that." Parametrising `share.py` would make the "by design" true in code. |
| Positioning names only AISHA; PIOLab, ODYM, STAN and Bayesian MFA are absent | Partly accepted. Added "a gap between system-definition frameworks (Pauliuk and Heeren, 2020) and balancing laboratories (Wieland et al., 2022)". STAN and Bayesian MFA did not fit the budget; name them in the talk. |
| pySUTO used before its expansion; BACI undefined | Accepted. Expanded at first use in the Objective; "BACI (Base pour l'Analyse du Commerce International)". |
| 1.010 is labelled "mine output in contained copper" in results.md, not concentrate output | Accepted. "Refined and mine output". |
| "Three of nine expectations pass" lands unexplained; merge with the robustness sentences | Accepted. "Of nine pre-registered expectations, three pass; the failures are informative. Tightening solver tolerance alone moved … 0.22. A 64-point sweep … moved none beyond 0.03." A6 fails its 0.02 threshold; the 0.03 is stated, not called a pass. |
| "Several units and several levels" overstates two valuations, both in kt | Accepted. "Readable in gross mass or copper content at any aggregation". |
| Keywords: Footprinting for the embodied result; Energy transition only as a fifth | Accepted. |
| Optional: "(1969)" after Fuller; "widened" for "scaled" | Accepted. "Constrained" before "least-squares" was added and then cut for length. |

## Round 2 — Reviewer D (AI-typical prose): ~50 % of v5 "reads machine", ~30 % after its edits

Markers found: tricolons in 8 of 19 sentences, semicolons in 7, mirrored gerund antitheses ("Tightening … moved; sweeping … moved"), abstract agents ("the diagnostics exposed", "flows show what no single source can"), the inversion "Harder is to …", and the alliterative "harmonises heterogeneous". Numeric tokens before and after its rewrite were identical.

| Edit | Taken into v6 |
| --- | --- |
| Split "rebuild the whole; since AISHA …" into two sentences | Yes. |
| "Harder is to keep …" → "Keeping … traceable is harder, as is testing …" | Yes. |
| "The demonstration reconciles …" → "We reconcile …" | Yes. |
| "the diagnostics exposed … both fixed in the input" → "the diagnostics caught …; we fixed both in the input" | Yes. |
| Break the mirrored pair: "Tightening … 0.22. When we swept …, no retention moved beyond 0.03." | Yes, shortened to "A 64-point sweep of 15 judgement parameters moved none beyond 0.03." |
| "embodied flows show what no single source can" → a plainer claim | Yes: "The embodied view adds what no source reports". |
| Conclusion tricolon → two sentences; "harmonises heterogeneous" → "brings heterogeneous sources into" | Yes. |
| Left as is: the Objective's two factual triples (sources; axes of disagreement) and the two mandated Spaceship Earth sentences | Kept. Reviewer D expects the first sentence still to read as machine-written; the triples are the three real sources and the three real axes. |

## Round 1 — Reviewer A (committee and positioning, on v2): verdict "revise, then accept as oral"

| Issue | Response |
| --- | --- |
| "Balancing … usually erases the sources' quality" is wrong: STAN, Bayesian MFA (Lupton & Allwood 2018; Dong et al. 2023; Wang et al. 2024), KRAS/AISHA and Eora's standard-deviation matrices all carry uncertainty | Accepted in v3 and kept: the Objective concedes that compilers "treat every data point as an uncertainty-weighted constraint"; the claimed advance is traceability of quality and corrections, and testing the result. |
| "Finite stores cannot be resupplied" is the Wikipedia summary, not Fuller; his ship is "continually refueled radiationally by the Sun"; the closed finite ship is Boulding's (1966) | Accepted. The remaining quote, "must be comprehended and serviced in total", was checked verbatim against the 1969 text (archive.org); its sentence is "We have not been seeing our Spaceship Earth as an integrally-designed machine which to be persistently successful must be comprehended and serviced in total." The other two quotes were dropped when the metaphor was cut to one sentence at each end. |
| Undefined terms (BACI, retention, "Gaussian posterior linearised at the binding constraints", "hard and soft", "quadratic program") | Accepted; all defined or replaced in plain words. |
| "Consumption" implies final demand; the account ends at semis fabrication | Accepted: "refined copper use". |
| Generality claimed from one material | Accepted; see Reviewer C row on `share.py`. |
| Theme link to emerging economies | Accepted: the four-region clause (33.8 % of the scale-total objective, §3.2.8, #111). |
| "Solver termination is a first-order choice" is overgeneral | Accepted; v6 reports the measurement without the generalisation. |
| Optional: Fuller's "melted up about every twenty-five years … metals come back into recirculation"; BACI kilograms-as-tonnes (112 Mt PNG); positioning against EXIOBASE (NTNU) | Not used, for lack of words. Keep for the talk. |

## Round 1 — Reviewer B (evidence and model fidelity, on v2): all numbers match the records

| Issue | Response |
| --- | --- |
| "Six stages that never drop a flow's standard deviation" is wrong: Stage 6 carries point estimates only (§3.1.8) | Accepted; v6 lists the six stages and names "posterior standard deviations" as a stage, with no claim about Stage 6. |
| The audit sentence credited both defects to one misparsed yearbook found by ranked residuals; ADR 0035 was found through the forced-deletion ledger, ADR 0030 by comparing the posterior secondary share (30 %) with ICSG's (17 %), and the double count came from two nested series | Accepted; the two defects are separate and no mechanism of discovery is claimed beyond "the diagnostics caught". |
| The 0.22 tolerance shift is on the base constraint set (`const_20260916_123437_8e8e9676`), not the reference run | Accepted in v4 ("on the study's base model"); v6 says "alone", i.e. with nothing else changed. State the base set in the talk. |
| The Birge factor is an average with a heavy tail | Accepted: "2.3 times too small on average". The lower-bound clause was cut for length. |
| Scale totals are the prior's own node sums, not reported totals | Accepted: "output totals". |
| The 13 regions *report* no smelting or refining (PNG, Panama, Mauritania among them) | Accepted: "regions reporting no smelting or refining". |
| The embodied ratio should carry its interval | Accepted: 1.49 (95 % interval 1.42–1.55). |
| No LICENSE, so "open-source" is unsupported | Open, for the author (item 2 above). |
| Add "7 % observed" | Accepted. |

## Round 3 — the Results section rewritten (v10, 2026-11-10)

Kept: mass balance, retention, the hub sentence. Dropped: the data-audit sentence (ADR 0030, 0035), "three of nine expectations", the tolerance shift of 0.22 (a property of the solve, not of the system; it stays in the manuscript and the talk). Added, all from the reference chain's own records: boundary input 15.0 → 34.7 Mt CC, four fifths to users; concentrate smelted abroad 26 → 60 %; China's final-use share 10 → 58 %; Herfindahl 0.111 vs 0.352 with Birge-scaled intervals; 56 % of 2024 extraction embodied in China's use; the Birge factor; reconciliation invariance of the concentration indices (≤ 0.003); the knob sweep (≤ 0.03). Words were paid for in the Objective (−6), Methods (−3) and Conclusion (−3).

**Gloss pass (v13).** Each indicator is explained in a few words at first use: mass balances "close within a kilogram"; retention = "reconciled total over prior"; misfit = "the squared adjustment in standard deviations" (Σz²); system input = "ore and scrap entering the system", output = "reaching fabricators as cathode" (the boundary ends at cathode into semis fabrication); Herfindahl index = "sum of squared regional shares"; embodied = "traced through the process network … embodied in China's final use"; the Birge factor = "the adjustments' root mean square is 2.3 standard deviations" (√R = 2.28); the knobs = "judgement-set parameters (uncertainty rules and imputation settings)" — Table S22b: eight σ-rule knobs, six imputer settings, the L_p exponent. Paid for by dropping "Reconciliation moves no concentration index by over 0.003" (B2, stays in the manuscript), the Conclusion's audit sentence (folded into "auditable to its sources", since Methods already says every flow traces to its sources and fills), "(1969)", "versioned", "crude-", "refined" and a few articles.

## Round 4 — readability pass, Reviewer E (peer review) and Reviewer F (readability and AI-prose), on v15

**Readability pass (v15, before the reviewers).** Subject and verb brought together in the opening sentence; the Lenzen/AISHA clause split from the "rebuild the whole" sentence; the Methods list made parallel (all outputs: feeds, priors, constraints, solution, standard deviations, tables); the scope phrases of the copper sentence moved ahead of the verb; the hub sentence split at the semicolon; "Herfindahl index … for mining against … for use"; the Birge sentence tied to the intervals; the Conclusion's tricolon made a relative clause.

### Reviewer E — peer review with repository access: "minor revision"

Numbers check: 28 quantities traced to numbers.json, table3_intervals.csv, text.md, results.md, si.md and the code; 24 match to the stated precision, the four exceptions are the first four rows below. Fuller's quotation and the keyword list could not be checked offline (both verified in round 1).

| Comment | Response |
| --- | --- |
| M1 "open-source" unsupported: no LICENSE, method.md §3.4 "[DOI and licence to be inserted]" | Unchanged — item 2 above, the author's decision. |
| M2 "Mass balances close within a kilogram" contradicts Table 3: the residual is 1.09 kg and A1 (< 10⁻⁶ kt) is recorded FAIL | Accepted: "close within 1.1 kilograms". |
| M3 "moves no retention beyond 0.03" is false for Indonesia's 2010–19 concentrate-export retention (0.0329, Table S22c); A6 at 0.02 fails at 5/64 points; the termination finding (0.22) is absent though the Objective promises a stability test | Accepted in scope: "changes no 30-year retention by over 0.03" (the nine stage retentions pooled 1995–2024, max 0.0298; Indonesia's is a 2010–19 series). The 0.22 sentence is item 6 above. |
| M4 "Copper enters only through …" overstated: `impute/share.py` L58–69 and `psut.py` L82 hard-code copper names | Accepted: "By design, copper specifics sit in the accounting structure, feeds and parameter tables" — intent, not a verified property. |
| M5 "built for both" has two readings; the Birge sentence states a procedure, not the finding | Accepted: "Two tasks remain hard: … They fall between … built for both"; "Adjustments average 2.3 stated standard deviations (root mean square), so the 95% intervals below are widened 2.3-fold", placed before the first interval. |
| M6 Lenzen 2009 and AISHA are monetary MRIO/contingency-table methods; the physical step is the PIOLab | Accepted in part: the physical-tables sentence and the constraint-paradigm sentence are now separate, so the lineage is dated for the paradigm, not for physical tables; PIOLab is not named (its expansion costs four words). |
| M7 "weighted least squares" omits the bounds (598,994 bands, 586,560 non-negativity rows; 8,492–9,836 edges per year at a bound) | Accepted: "one bounded weighted least-squares solution per year". |
| Minor: "keeping source quality, imputations and decisions traceable"; "concentrate" for "mine output"; drop "over 1995–2024"; both Herfindahl intervals or neither; "shipments to China"; "constraint rows" | First four accepted. "to China" applied, then cut for budget — "embodied in China's final use" sets the comparator. "constraint rows" declined for budget. |
| Minor: "chiefly Switzerland and Singapore" (−2); "trace" → "show" in the last sentence | Declined: the author's phrasing of the hub finding and of the closing narrative. |

### Reviewer F — readability, flow and AI-prose markers, text only

Flow ratings on v15: Objective 2/5, Methods 3/5, Results 3/5, Conclusion 3/5. AI-prose estimate: about 6 of 18 sentences, concentrated in the Objective and Conclusion; the Methods and Results tells are compression artefacts (telegraphic triples, stacked glosses). What a committee member remembers after one reading: the Python tool, the Spaceship Earth hook, China's 58 % and 56 %; the methods claims stick less than the copper numbers.

| Comment | Response |
| --- | --- |
| "its gauges" has the wrong nearest antecedent; the em-dash list separates subject from verb | Accepted: "the ship's gauges (yearbooks, customs records and extraction accounts) disagree". |
| "rebuild the whole … on one structure": undefined referents; 17 words of citation between "since" and its clause | Accepted: "reconcile them into one account. Since Lenzen et al. (2009) and AISHA (…), every data point enters as an uncertainty-weighted constraint." |
| "Harder is to …": fronted comparative without comparand; non-parallel triple; "as claimed" by whom | Accepted: "Two tasks remain hard: keeping source quality, imputations and decisions traceable, and testing whether the account is as certain and stable as claimed." |
| Proposed "Neither system-definition frameworks nor balancing laboratories do both" | Declined — a stronger claim about ODYM and the PIOLab than the evidence supports; "They fall between …" keeps the gap without the negative. |
| "each reading only its predecessor's output" overlaps "upstream runs" → "in sequence" (−4) | Accepted. |
| "fill", "posterior propagators" are in-house words | Accepted: "imputation(s)", "uncertainty propagators". |
| "(Pauliuk et al., 2024)" reads as if Pauliuk did the tracing | Accepted: the citation now closes the recording clause. |
| "ore extraction" is a quantity, not a source; scope phrases stacked before the subject | Accepted: "ore extraction accounts"; scope after the verb. "from ore to refined cathode" declined for budget — the Results carry the chain's ends. |
| Retention pairing needs an implied "respectively" | Accepted: "0.998 for refined output and 1.010 for concentrate". |
| Hub sentence: colon plus semicolon; split | Accepted. |
| Sweep sentence sits last and reads as covering the copper shares; "moves … beyond" ambiguous | Accepted: moved to follow the misfit sentences; "changes … by over 0.03". |
| "Ore and scrap inputs" for "entering the system" | Declined: the boundary gloss was asked for in round 3. |
| Verb-less Herfindahl clause; one interval of two | Accepted: "Mining stayed dispersed: its Herfindahl index (…) is 0.111 (0.108–0.114) against 0.352 (0.344–0.361) for use." |
| "Intervals are scaled by 2.3" is an orphan methods note; "widened" | Accepted in substance; the sentence now states the finding and precedes the intervals. |
| Stacked appositives in the embodied sentence | Accepted: semicolon and "comes from". The participle stays. |
| Conclusion: drop "consistent", add "and"; replace the closing sentence with "reconcile the gauges of Spaceship Earth into one reading" | First accepted. Replacement declined: the closing sentence is the author's narrative ("trace the distribution and scale"); "let us trace" → "trace". |
| Content note: the Chile–Peru sentence could give way to a sentence on what traceability buys | Not applied; recorded with item 6 above for the author. |

## Round 7 — rebuilt from the manuscript (v21–v24, 2026-11-25, `.tex`)

Grilled with the author (grill-with-docs, Q1–Q18). Decisions: the abstract becomes a 500-word version of the paper (Introduction's gap, Method's contributions, Results' answers); Fuller kept at both ends; numbers only from the manuscript's number files on the reference chain (the text.md trajectories of v20 dropped); Results as a story with few headline numbers; four support rules (each claim maps to a manuscript sentence and, for a number, to a JSON key; no claim of absence about the literature; nothing marked `verify:`/`check:`; interpretations only where the Discussion makes them, the closing Fuller sentence excepted as the author's vision); output as `.tex`; pySUTO in the title (the author's title, a comma added). Later in the round the author swapped content: the termination result out (not interesting to the author), then the source defects, China's embodied Chile/Peru copper (B3) and the ledger tally out (the author's own debugging), and in their place the trade hubs (the surprise), the objective by sector pair and node, and China's rising concentration in refining and cathode use. The closing sentence became the author's case for metabolic planning, with Fuller's definition of wealth.

**Reviewers.** v21: (G) committee and evidence, with repository access; (H) readability and AI-prose, text only. v22: (I) a fresh evidence check. All three had seen no earlier draft or review.

| Finding | Response |
| --- | --- |
| G: AISHA expansion wrong; the title of the paper is "A tool to construct a series of contingency tables" | Declined: footnote 19, p. 27 of the IIOA 19th-conference PDF reads "The AISHA-acronym stands for An Automated Integration System for Harmonised Accounts." The author's citation says 2015; the PDF sits in the 2011 (19th conference) folder — matters only for `refs.bib`. |
| G: "Two tasks remain hard" is a claim of absence (Intro marks the gap `verify:`); I: "addresses two tasks" set against AISHA implies the same | Accepted: "pySUTO … carries each source's uncertainty to every reconciled flow." |
| G: copper as an EU strategic raw material is not in the repo | Kept, external source: the Commission's critical-raw-materials page and Regulation (EU) 2024/1252 list copper as strategic (not critical). |
| G, H: "stage" means pipeline stage and chain step at once | Accepted: "nine main chain steps"; "stage" only for the pipeline. |
| G, I: "closes every mass balance" conflicts with A1 "not met" | Accepted: "every exact balance to 1.1 kg". |
| G, I: "4 % from its sources": retention is against the prior, which holds imputations, pooled over 30 years | Accepted: "from the gap-filled sources over 30 years". |
| G: "only the joined account shows" is a claim of absence | Accepted, then moot (B3 left the text). |
| G: "In 2020–24 … region groups" | Accepted. |
| H: "termination" and "stopping rule" used as synonyms; "2.2-fold too small" has no base; "bands" | Accepted: one term; "2.2-fold smaller than the realised disagreement"; "bounds". |
| H: "preregistered hypotheses", "settings" | Declined: "expected answers" is the author's word (2026-11-12; GLOSSARY *Expected answer*); "setting" is an _Avoid_ term (GLOSSARY *Knob*). Moot after the tally left. |
| H: replace the closing Fuller sentence | Declined at the time (the author's sentence); the author later replaced it. |
| I: hub sentence says "copper" and "trade hubs"; the evidence is gross mass on two regions, and in contained copper the Netherlands ends above its prior | Accepted: "far more copper by gross mass into two trade hubs"; "net of re-exports" (the prior is BACI after the re-export netting); Taiwan and France 99–100 % (Malaysia, 0.981, made it 98). |
| I: "Stages are plug-ins": the six stages are fixed | Accepted: "Stage components are plug-ins". |
| I: "a second metal enters as data": a case is a Python module | Accepted: "enters as a new case". |
| I: Fuller wrote "we are physically prepared to sustain" | Accepted. |
| I: "IELab … by source reliability" is half supported (IELab names KRAS, not reliability weighting) | Declined: introduction.md:71 places the IELab in the reliability-weighted lineage; the author chose the name in round 5. |

**New measurement for the hub sentence.** `scripts/report/manuscript_checks.py` gained two keys (`discussion:L88.hub_mass_imports_by_region`, `discussion:L88.hub_cathode_mass_trade_hubs`), regenerated read-only on the reference chain (39 old keys identical), and `discussion.md` §5.2 A3 gained one bullet citing them.

**v25 (2026-11-25).** The author asked the Objective to state the goal — distilling advances in the data model and in the reconciliation problem formulation into an open-source, modular framework — and to name ODYM. ODYM is named for the data-model line (it structures biophysical data; the Introduction says such frameworks do not reconcile), AISHA and the IELab for the reconciliation line; Methods is regrouped into data model, reconciliation and framework. Paid for by the licence moving into the Objective and by dropping two glosses (interval widening, Herfindahl definition).

**v26 (2026-11-25).** The author asked whether the abstract says pySUTO serves other metals (steel) and substances (carbon); it did not. Q19 (b): the claim went into introduction.md:90 first, then into the Conclusion as a design target next to the synthetic case. "Most surprising" became "Surprisingly" to pay for it.

**v27 (2026-11-25).** The author's rule: every key result carries its implication (`docs/agents/key-results.md` step 4). v26 paired two of six results with one, both in the Conclusion. Q20 (a): discussion.md §5.4 "For copper" gained two implications by the author's decision, for the hubs (check hub imports against production and use before they enter footprints) and for concentration (it sits downstream of the mine, so supply-risk assessment must look past mining); the Results now give each of the six results its §5.4 implication, and the Conclusion keeps the transfer and Fuller sentences. Paid for by dropping "The result is one traceable physical account …" (Objective and Methods say it), "Surprisingly", "in open software", "(IELab)" (not used again), "year after year" (now "chronically", the author's word) and "already". Body 490 words, 496 with the headings.

**Found along the way (not changed).** discussion.md take-home 5 lists "regional secondary shares" among what the sources cannot show alone, but the ICSG feed's own rows give them (the reconciliation moves no group by more than 0.015). method.md "Practical significance" says four regions carry a third of the scale-total objective, discussion.md §5.5.2 says five carry two fifths.

## Claim → evidence (v27)

Reference run `opt_20261124_141714_e2892160`; N = `docs/manuscript/figs/numbers.json`, C = `manuscript_checks.json`.

| Claim in v27 | Value | Source |
| --- | --- | --- |
| "must be comprehended and serviced in total" | verbatim | Fuller (1969), archive.org OCR full text, §4 |
| copper, an EU strategic raw material | strategic, not critical | Regulation (EU) 2024/1252; Commission CRM page |
| four traded goods; yearbooks, customs records and extraction accounts disagree | — | introduction.md §1; discussion.md §5.2 (A2) |
| AISHA = An Automated Integration System for Harmonised Accounts; constrained balancing by source reliability | footnote 19, p. 27 | Geschke et al., IIOA 19th conference PDF; introduction.md:71 |
| ODYM = Open Dynamic Material Systems Model; structures biophysical data | — | Pauliuk and Heeren (2020), JIE 24(3); github.com/IndEcol/ODYM ("modeling biophysical stock-flow relations"); introduction.md:68 (structures data, does not reconcile — so ODYM is named for the data model, not for reconciliation) |
| goal: distil advances in the data model and the formulation of the reconciliation problem into one open-source (MIT), modular framework | — | method.md N1–N3 ("What is new"), N7, §3.1.9; `LICENSE`. "advances", not "new": N3 carries a `verify:` on novelty |
| data model: every flow in gross mass and contained copper, with a pedigree-based σ; reconciliation: one σ-weighted least-squares problem per year with exact balances and bounds; posterior σ per flow (Stage 5) | — | method.md §3.1.1 (Rule), N1–N3, §3.1.6–3.1.7 |
| six stages; exact balances and bounds; pedigree σ → posterior σ per flow; two valuations | — | method.md §3.1.1, N1–N3 |
| stage components are plug-ins; case definition; traceable runs | — | method.md §3.1.4–3.1.5, N7, §3.1.9; `LICENSE`, `pyproject.toml` |
| 47 regions, 23 sectors, 1995–2024; about 850,000 flows; 1.7 million constraints; under two minutes | 854,460 edges; 1,687,613 rows; 95.4 s solve, 98 s with probe | method.md §3.1.1, N2, §3.1.6; opt sidecar `solve_wall_s` |
| every exact balance to 1.1 kg | 1.15 × 10⁻⁶ kt | N `max_hard_residual_kt`; Table 3 A1 |
| none of nine main chain steps more than 4 % from the gap-filled sources over 30 years | 0.973–1.037 | N `retention_30y`; Table S19 |
| the sources disagree mostly on totals, which trade absorbs | edges 0.4–1.5 % beyond 3σ, scale totals 12.9–24.4 %; cross-region legs 92 % of the edge objective | N `share_gt3`, `hub_share.cross_region`; discussion.md §5.2 |
| cathode to users and anode to refineries carry 87 % of the flow adjustment; matte, cathode, concentrate totals the largest residuals | 0.746 + 0.126; 0.238, 0.210, 0.175 | N `edge_share_top`, `sum_share_top`; results.md §4.1 |
| customs records chronically book far more gross mass into two trade hubs than the account can place | retention below 0.8 in 26 (CH) and 30 (NL) of 30 years | C `discussion:L88.hub_mass_imports_by_region`; discussion.md:89 |
| keeps 45 % (CH) and 32 % (NL) of recorded imports net of re-exports, 1995–2024; Taiwan and France 99–100 % | 6,485/14,567; 2,703/8,499 kt; 0.995, 1.004 | C same key; the prior on these legs is BACI after the re-export netting (method.md §3.2.4, ADR 0027) |
| stated uncertainties on average 2.2-fold smaller than the realised disagreement | R = 4.86, √R = 2.21 | N `stage5.birge_ratio`; method.md §3.1.7; Table 3 caption |
| a single 2020–24 world scrap share hides region groups from none to over a third | world 0.171; EU-14 0.375; Chile+Peru 0.000 | N `secondary_pooled_2020_24`; results.md §4.3 (B1) |
| since 1995 the Herfindahl index of refining more than doubled, of cathode use nearly quadrupled, mining stayed dispersed; 2024 China 45 % of refining, its fabricators 57 % of cathode use | 0.085 → 0.223; 0.091 → 0.341; mining 0.117 → 0.111; 0.447; 0.574 | N `hhi_1995_post`, `hhi_2024`; results.md §4.3 (B2) |
| *Implications (Q20; one per result):* | | |
| production totals can be quoted near the yearbooks; bilateral flows need intervals | — | discussion.md §5.4 "For copper", "The trade legs are the least certain part" |
| customs-based footprints should check hub imports against production and use | — | discussion.md §5.4 "For copper", "Hub imports need checking" (added 2026-11-25, the author's decision) |
| intervals must be widened | factor 2.21 | discussion.md §5.4 "For reconciliation practice", "The posterior σ is a sensitivity" |
| regional recycling indicators need regional accounts | — | discussion.md §5.4 "For copper", "A world secondary share hides the regions" |
| supply-risk assessment must look past mining | — | discussion.md §5.4 "For copper", "Concentration sits downstream of the mine" (added 2026-11-25, the author's decision) |
| by design, other metals and substances, such as iron in steel or carbon, enter as new cases, as a synthetic one does | — | introduction.md:90 (added 2026-11-25: content role, ADR 0034; "holds by design; besides copper, only a synthetic case runs"); method.md N7, §3.2 "Transfer to other materials"; untested for carbon, whose gross-mass balance needs oxygen and CO2 booked at the nature boundary |
| "account wealth" … "liberating level of metabolic and metaphysical regeneration" | verbatim | Fuller (1969), §6 "Synergy"; page number not visible in the OCR copy (the author cites p. 93) |
| semi-automated physical accounts are the basis for such metabolic planning | — | the author's vision (rule 4 exception) |

## Claim → evidence (v20)

| Claim in v20 | Value | Source |
| --- | --- | --- |
| 47 regions, 23 sectors, ore to refined cathode, 1995–2024, two valuations | — | method.md §3.2.1 |
| 854,460 flows, 7 % observed | 61,218 edges, 7.16 % | method.md §3.1.1, §3.1.4 |
| 1.69 million constraints | 1,687,230 | numbers.json `n_rows` |
| Mass balances close within 1.1 kilograms | 1.09 × 10⁻⁶ kt = 1.09 kg | numbers.json `max_hard_residual_kt`; Table 3 A1 (< 10⁻⁶ kt) is a narrow FAIL, hence 1.1 and not "a kilogram" |
| 30-year retention (reconciled over prior total), 0.998 for refined output and 1.010 for concentrate | 0.998 (CP→Cathode CC), 1.010 (MF→Concentrate CC, "mine output in contained copper"), 30 years | numbers.json `retention_30y`; results.md §4.1 |
| Misfit = squared adjustment in standard deviations; 79 % of flow-level misfit on imports into regions reporting no smelting or refining | Σz² = Σ((a* − a_prior)/σ)²; 0.787 of edge Σz²; 13 regions | numbers.json `hub_share.hub`, `hub_regions` |
| Ore and scrap entering the system 15.0 → 34.7 Mt CC a year, four fifths reaching fabricators as cathode | boundary input 15.0 (1995), 34.7 (2024) Mt CC; final use 78.6 % and 78.8 % of input | `scripts/report/insights_docx/text.md` §1 (f02, f03), run `opt_20261022_112855_928da10c` |
| Concentrate smelted outside its mining region 26 → 60 % | 2.3 of 9.0 Mt CC (1995), 11.3 of 18.8 (2024) | text.md §2 (f04) |
| China's share of final use 10 → 58 % | 0.584 posterior 2024 | text.md §4 (f08); numbers.json `hhi_2024.use_post.top_share` |
| World scrap per unit of primary feed 0.21 (1995) → 0.22 (2024); 14 EU states 0.83 → 0.62; China 0.84 → 0.34 with scrap feed ×6.5 to 62 % of the world's; Japan 0.12 → 0.30; South Korea 0.17 → 0.40 | scrap-to-primary ratio (scrap to smelters and fire refineries over concentrate to smelters plus SX-EW cathode), world trough 0.13 in 2003; China 0.5 → 3.1 Mt CC, 22 % → 62 % of world scrap feed | `scripts/report/insights_docx/text.md` §3 (f05, f07), run `opt_20261022_112855_928da10c` |
| Secondary share 2020–24: 0.37 (0.367–0.374) in those 14 EU states, 0.24 (0.234–0.245) in China, 0.25 in Japan, South Korea and Taiwan | input-side share pooled 2020–24: EU-14 0.370 [0.367, 0.374], China 0.240 [0.234, 0.245], JP/KR/TW 0.249 (no interval row), world 0.172; EU-14 = DE IT FR ES PL BE NL AT SE BG GR PT SK LU | numbers.json `secondary_pooled_2020_24`; `table3_intervals.csv` B1; results.md "Secondary copper"; si.md S15 (B1 definition) |
| Traced through the process network, 56 % of 2024's mined copper is embodied in China's final use | 16,727 / 29,620 kt CC | numbers.json `china_footprint_kt`, `world_extraction_kt`; results.md "Trade relations" |
| Adjustments average 2.3 stated standard deviations (root mean square); 95 % intervals widened 2.3-fold | Birge R = 5.18 (mean z² of feed-observed CC edges), √R = 2.28 | results.md §4.1; method.md §3.1.7 (#87) |
| 64-point sweep of 15 judgement parameters (uncertainty rules, imputation settings) changes no 30-year retention by over 0.03 | max 0.0298 over the nine stage retentions pooled 1995–2024 (Scrap→CFR CC); Indonesia's 2010–19 export retention moves 0.0329 and is outside the 30-year scope; knobs: 8 σ rules, 6 imputer settings, L_p exponent | results.md Table 3 A6; si.md S15.4, Table S22b |
| Six stages in sequence; one bounded weighted least-squares solution per year (a QP with bands and non-negativity, method.md Eq. 2); four plug-in seams; fills logged with method, sources, coefficients; sidecar with input versions, upstream runs, settings | — | method.md §3.1.1, §3.1.4, §3.1.9; `feed.py:41`, `impute/base.py:62`, `constraints/base.py:60`, `uncprop.py:194` |
| AISHA expansion | "An Automated Integration System for Harmonised Accounts" | Geschke et al., footnote 19 |
| IELab expansion | "Industrial Ecology Virtual Laboratory" | Lenzen et al. (2014), Compiling and using input–output frameworks through collaborative virtual laboratories, Sci. Total Environ. 485–486 — the author's choice of name for the lineage (round 5) |
| Fuller quote | verbatim | Fuller (1969), *Operating Manual for Spaceship Earth*, archive.org full text |
| By design, copper specifics sit in the accounting structure, feeds and parameter tables | `impute/share.py` L58–69 and `psut.py` L82 still hard-code copper names — hence "by design", not "only" | Reviewer E, round 4 |

## Found in the manuscript along the way (not changed)

- method.md, Figure 1: the Mermaid node (line 80) says Stage 5 is "not run on copper" and the caption (line 101) says it "has not been executed". Stage 5 has been run (`uncprop_20261029_090754_0446e784`, §3.1.7).
- method.md §3.2 says "SI S11 lists all 39" ADRs, the Table 2 caption says 44, and `docs/adr/` holds 45.
- The header comment of method.md says the AISHA lineage "does not carry per-source data quality through", but Eora publishes standard-deviation matrices. Reword it before it reaches the Introduction.
- The header comment cites Lupton & Allwood as 2017; Reviewer A's search gives *Journal of Industrial Ecology* 2018.
- `src/pysuto/impute/share.py` hard-codes copper node names (`"Cathode"`, `"MESC"`, `"USC"`, …), so the accounting structure is not the only copper-specific input; relevant to the metal-agnostic milestone (#26, #58).
- `registry/_sidecar.py` records no package versions or environment (Pauliuk et al. 2024, step 4a); only the sweep manifest stores a git commit and scipy version.
