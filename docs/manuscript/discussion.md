<!--
Discussion section of the pySUTO model paper. Numbered Section 5, after the Results
(results.md). Drafted as bullet points on 2026-11-12, like the Method and the Results:
one point per bullet; prose follows once the content is agreed.

Structure (the seven elements of a discussion, in order):
  5.1 answers to the research questions   summary, against RQ1–RQ3
  5.2 why the account came out this way   mechanisms behind the ledger outcomes
  5.3 agreement with previous work        AUTHOR — scaffold only, see "compare:" below
  5.4 significance and implications       for reconciliation practice, for copper
  5.5 limitations                         of the method and of the application
  5.6 take-home messages and outlook      five messages, four research directions

Conventions:
  → Ak / Bk / Nk   the bullet rests on ledger row Ak/Bk of Table 3 or contribution Nk
                   of the Method's contribution block
  (§x.y)           where the number is reported; no number appears here that is not
                   already in the Method, the Results or the SI (no new results)
  *Compare:*       a visible placeholder for a comparison with external work, written
                   by the author: what to compare, our number, candidate sources (none
                   is checked; refs.bib keys where present, "verify:" otherwise)
  check:           inline HTML comments marking an explanation that rests on reasoning
                   about the model, not on a measurement in the paper; each needs a
                   measurement or a softer wording before it goes into prose

Research questions: RQ1–RQ3 and their sub-questions are defined in the Introduction
(introduction.md) and answered in Results 4.1–4.3; 5.1 answers them in the same order. The
ledger rows A1–B3 are their expected answers, "met" or "not met"; a departure from an
expected answer is a finding, explained in 5.2.

Limitations: 5.5 holds the limitations of the method (new) and those of the copper
application, moved here from method.md §3.2.8 on 2026-11-12 with their wording and
numbers unchanged (numbers: manuscript_numbers.py group 11, `python scripts/explo/final_limitations_diag.py
<opt_run_id> [region ...]` for the mine-chain, stock-leg, PSC and region-share items, and the REPs
cited in the method.md header).

Word budget of the later prose: <= 1,500 words (5.1 200 / 5.2 400 / 5.3 300 / 5.4 250 /
5.5 250 / 5.6 150), headings excluded.

The interpretations parked in the results.md header on 2026-11-12 (A2, A3, B2) are
used in 5.2 and 5.3.
-->

<a id="sec-discussion"></a>

## 5 Discussion

- Section [5.1](#sec-disc-answers) answers the three research questions; Section [5.2](#sec-disc-why) explains the ledger outcomes; Section [5.3](#sec-disc-literature) sets the method and the three copper insights against previous work.
- Section [5.4](#sec-disc-implications) draws the implications, Section [5.5](#sec-disc-limits) the limitations, Section [5.6](#sec-disc-outlook) the take-home messages and the research they call for.

<a id="sec-disc-answers"></a>

### 5.1 Answers to the research questions

**RQ1: the account is consistent, and its adjustment is small and falls on trade.**

- The reconciliation closes every hard balance to the solver's numerical floor ($1.1\times10^{-6}$ kt, narrowly above the expected $10^{-6}$ kt; A1) and moves the sources little: all nine stages retain 0.972–1.037 of their prior over 1995–2024, and the adjustment does not grow upstream, contrary to the expectation (A4 not met; §4.1).
- It adjusts the trade legs, not production: 95% of the edge objective lies on cross-region legs, 79% on legs into the 13 non-processing regions, as expected (A3).
- Single flows are compatible with their stated $\sigma$; scale totals are not: between one in eight and one in four lie beyond $3\sigma$ in any year, but fewer than the expected one in five in the 21 years before 2019 (A2 not met).

**RQ2: the account is certain where the sources agree, and depends on its termination far more than on its judgement values.**

- The constraints contract the stated uncertainty most from ore to cathode and least on the slag legs; the stated $\sigma$ are too small on aggregate (Birge ratio 4.83) and the misfit is concentrated in a heavy tail (§4.2; exploratory).
- The termination rule alone moves one stage by 0.223, the whole knob sweep moves one stage by at most 0.030; neither stays within the expected 0.02 (A5, A6). → N4

**RQ3: the account shows heterogeneity and indirect trade that no single source shows.**

- The world secondary share is a mean over regions that range from 0 to 0.374 (B1, met).
- Concentration rises downstream, and the reconciliation leaves it unchanged within 0.004 at the point estimates (the final-use interval reaches −0.011); refining's 0.223 stays below the expected 0.25 (B2, not met).
- China's final use embodies 1.49 times the Chilean and Peruvian copper it imports directly as concentrate; 1,327 kt CC of it reaches China by no direct import from the two countries (B3, met).

**Across the ledger.** Three of the nine expected answers are met. The six that are not are answers as well: four of them say the account is closer to its sources (A2, A4) or less concentrated (B2) than expected, or misses its bound at the numerical floor (A1); two say the answer depends on termination and, at its margin, on a judgement value (A5, A6). Section [5.2](#sec-disc-why) explains each.

<a id="sec-disc-why"></a>

### 5.2 Why the account came out this way

**The sources disagree on totals, not on single flows** (A2).

- An edge carries the wide $\sigma$ of a single observation, a scale total the narrow $\sigma$ of a yearbook figure; where ICSG, BACI and the ore allocation cannot all hold, the contradiction surfaces on the total that two of them share.
- Five regions carry two fifths of the scale-total objective (41.4%): Italy's share is the cost of holding a mine ICSG does not report at zero (issue #201), and the other four are documented contradictions between ICSG and BACI (Section [5.5.2](#sec-disc-limits-copper)). The heavy tail of the standardised shifts is not confined to them: 5.1% of the feed-observed edges remain beyond three Birge-scaled units (§4.2), and 17.4% of those tail edge-years have an endpoint in the four regions of the contradictions, 1.5 times their 11.4% share of the sample, while four fifths lie elsewhere (Italy: 2.2% of the tail against 5.1% of the sample). <!-- keys: discussion:L81.tail_share, discussion:L81.tail_in_four_either, discussion:L81.tail_in_italy_either; opt_20261124_141714_e2892160 with uncprop_20261124_144415_13fea7bf; SI Table S22d; refresh with #320 -->

- A2 is not met by its direction: fewer totals lie beyond $3\sigma$ than expected (0.129–0.198 in 16 years against $> 0.20$). On the chain the expectation was written on (`opt_20260930_092111_e3e0677d`) the share was 0.191–0.318 and above 0.20 in 27 of 30 years; on the reference run it is lower in every year (mean 1995–2018 from 0.235 to 0.182). The changes between the two chains, the input corrections among them (#65, #84/#85, #110, #111), lowered it; the comparison does not isolate which one. <!-- keys: discussion:L83.a2_share_by_year_old, discussion:L83.a2_share_by_year_new, discussion:L83.a2_mean_to_2018_old, discussion:L83.a2_mean_to_2018_new, discussion:L83.a2_years_above_old; opt_20261124_141714_e2892160 against opt_20260930_092111_e3e0677d; SI Table S22d; refresh with #320 -->

**Trade pays for a disagreement, and it pays into regions that cannot place it** (A3).

- A trade leg is priced at a larger coefficient of variation than an anchor (0.069 at the median leg against 0.017), so the solver resolves an anchor conflict on the legs (Section [5.5.2](#sec-disc-limits-copper)).
- The non-processing regions report no smelter, refinery or SX-EW output. Their legs carry 69.6% of the edge objective (A3), but 96% of that sits on cathode legs into final use, mostly in gross mass (Cathode→FU: 63.5% of the whole objective); concentrate, matte and anode legs carry 3.7% of it. The disagreement into these regions is therefore on cathode, which can enter final use, and not on intermediates with no process to enter; why the cathode legs absorb it is not tested here. <!-- keys: discussion:L88.hub_share, discussion:L88.hub_share_by_good_total, discussion:L88.hub_share_conc_matte_anode_of_hub, discussion:L88.hub_top_pairs; opt_20261124_141714_e2892160; SI Table S22d; refresh with #320 -->

**The upstream deletion the ledger expected was in the inputs and in the solver, not in the method** (A4).

- A4 was written when crude-ore mass retained 0.864. That deletion was a termination artefact: solved for the defects at a gap of $10^{-8}$, the same QPs retain 0.998 (issue #33; §4.1).
- The large mine-chain deletions of earlier chains traced to source defects that were fixed at the input — phantom mine output (11,558 kt CC), concentrate exports booked in kilograms as tonnes, and scrap counted twice (N5; §3.2.3–3.2.4). → N5
- What remains is small and partly forced: 6,543 kt CC of mine output is deleted over 30 years, 35.1% of it by export mass that no grade can carry (§3.3).

**Termination moves the loosely weighted flows** (A5).

- The relative duality gap of the flow formulation is measured against the prior-only constant, an order of magnitude larger than the misfit; a loose gap leaves the flows with the widest $\sigma$ furthest from the optimum (§3.1.6). → N4
- Scrap is such a flow: in the sweep the no-data anchor's coefficient of variation, not a feed's, is the top-ranked knob on both scrap legs (§4.2), and Scrap→Smelting CC moves 0.223 between the two gaps. <!-- check: the A5 pair is the base chain of 2026-09-16, which still booked the secondary double count (ADR 0030); part of the 0.223 may be the tight solve deleting that phantom scrap, not loose weighting alone -->

**The judgement values move what the data constrain least** (A6).

- Across the 23 contained-copper categories the sweep's largest change and the lack of posterior contraction rank together (Spearman $\rho = 0.49$): the slag legs, least constrained, move most (§4.2).
- Scrap→CFR CC breaks A6 despite a contraction of 0.93. The contraction does not come from the no-data edges: feed-observed edges carry 96% of the category's weight, and their contraction falls from 0.94 to 0.17 when the scale totals and the ICSG scrap level anchors are left out of the posterior (the Stage-5 ablation); the no-data (zero-pinned) edges carry 3%. The rows that contract it are the ICSG scrap level anchors on the feed-observed edges. The no-data anchor's coefficient of variation moves the retention by another route, which the contraction does not show. <!-- keys: discussion:L104.scrap_cfr_kappa_default, discussion:L104.scrap_cfr_kappa_ablation, discussion:L104.scrap_cfr_rows_touching; uncprop_20261124_144415_13fea7bf and its ablation uncprop_20261124_150634_1ebdc2db; the route of the knob is untested; SI Table S22d; refresh with #320 -->
- The two influential $\sigma$ knobs are the two that price contradictions: the scale totals' coefficient of variation decides how much of a disagreement the totals absorb, and it moves Chile and Indonesia in opposite directions (SRRC −0.95 against +0.86).

**The stated $\sigma$ are too small on average and too wide at the core** (§4.2).

- A Birge ratio of 4.83 says the residuals are $\sqrt{R} = 2.20$ times larger than the stated $\sigma$ predict.
- The narrower core (median 0.16 for imputed edges against 0.67 under $N(0,1)$) and the heavier tail say the misfit is not spread evenly: most edges are over-covered, a few contradictions carry the excess. Pinned and flat edges do not make the core: edges whose $a^*$ lies within $10^{-4}$ kt of zero are 0.1% of the feed-observed and 4.3% of the imputed sample, and the imputed median is 0.18 without them. Zero-seed edges ($a_{\mathrm{prior}} = 0$) are 62% of the imputed sample, and the median without them is 0.27: the imputed core is narrow in part because of them. A uniform Birge scale widens both. <!-- keys: discussion:L110.qq_pinned_flat; uncprop_20261124_144415_13fea7bf; SI Table S22d; refresh with #320 -->

**What the reconciliation leaves alone** (B1, B2).

- The world secondary share stays within 0.007 of the ICSG-implied value because ICSG is an input; B1 tests fidelity, not validation (§4.3).
- Concentration indices move by at most 0.004 because the reconciliation adjusts trade legs and keeps each step's production at its prior scale (A3, A4); the producer shares that set the HHI are the yearbook's.

**Why embodied exceeds direct** (B3).

- Chilean and Peruvian copper also reaches China as matte, anode and cathode and through processing in third countries; the Leontief inverse of the market-share PIOT follows it through every process and region, a bilateral trade table follows one border crossing.
- The 1,327 kt CC that no direct import matches is the part the market-share construct allocates through third regions; it depends on that construct (Section [5.5](#sec-disc-limits)).

<a id="sec-disc-literature"></a>

### 5.3 Agreement with previous work

<!-- AUTHOR: this section is the author's. The bullets below fix what is compared and with which of our numbers; the comparison itself, its direction (agrees / differs) and the reason are left to write. Candidate sources are from the results.md header; none is checked. -->

**Method.**

- *Compare:* constrained balancing at the scale of a multi-regional account — what pySUTO adds and what it lacks against AISHA and the PIOLab (`geschke2011aisha`, `geschke2019parallel`, `lenzen2009matrix`, `wieland2022piolab`): per-source $\sigma$ carried from pedigree to posterior (N1), inequality bands in the reconciliation (N2), two valuations in one account (N3). Our numbers: 854,460 edges, 1,687,614 rows, 535 s wall-clock (§3.1.6).
- *Compare:* data reconciliation with error propagation in MFA (`cencic2008stan`, `laner2016`): our posterior is the equality-constrained GLS posterior of `byron1978` / `vanderploeg1982` linearised at the active set (ADR 0044); our calibration result ($R = 4.83$, heavy tail) against what MFA studies report about stated versus realised uncertainty.
- *Compare:* whether published balancing studies report their solver's termination criterion, and whether any tests it (A5: 0.223; N4). verify: candidate sources.

**Insight 1: secondary share** (B1).

- *Compare:* world 0.171 (input side, 2020–2024) and the regional range 0–0.374 against the ICSG secondary-refined share (`icsg2023yearbook`, `icsg2025yearbook`; 0.158–0.176 on the output basis, SI Table S18) and against dynamic copper-cycle estimates. verify: Glöser, Soulier and Espinoza 2013; Soulier et al. 2018 (China's copper cycle).
- State the basis difference before the numbers: input-side share of refinery intake versus end-of-life recycling rate versus recycled-content share.

**Insight 2: concentration** (B2).

- *Compare:* mining 0.111 and refining 0.223 (2024) against producer shares in USGS Mineral Commodity Summaries and against concentration measures of copper trade-network studies. verify: USGS MCS; Dong et al. 2020.
- *Compare:* refining's 0.223 against the 2023 U.S. Merger Guidelines' threshold of 0.18 for "highly concentrated" (`dojftc2023mg`; verify: the threshold), under which refining is highly concentrated; the pre-registered threshold stays 0.25.

**Insight 3: embodied and direct trade** (B3).

- *Compare:* China's footprint of 16,431 kt CC (55% of world extraction, 2024) and the ratio 1.49 against embodied-flow and footprint estimates for China. verify: `wieland2022piolab`; a copper RMC or footprint study for China.
- State the construct difference before the numbers: market-share PIOT in physical units versus a monetary MRIO footprint.

<a id="sec-disc-implications"></a>

### 5.4 Significance and implications

**For reconciliation practice.**

- **Report and test the termination.** An interior-point solve stopped at a gap of $10^{-3}$ read Chile's concentrate export retention as 0.66 instead of 0.97 (N4). Any reconciliation that reports a balanced table without its gap, and without a tighter re-solve, carries an error of unknown size in exactly the flows it weights least. → N4
- **A residual is a finding about a source.** The defects found in copper were in the inputs, and the fix belongs there, not in a looser $\sigma$ (N5; Table 2). For the data providers this is a list of specific, located errors (§3.2.3–3.2.4; Section [5.5](#sec-disc-limits)). → N5
- **The posterior $\sigma$ is a sensitivity, not an interval.** With $R = 4.83$, unscaled Stage-5 intervals would be too narrow by a factor of 2.20 on average; Birge-scaled ones still understate the tail where sources contradict (§3.1.7). Users of the account should read the intervals as lower bounds there.
- **Robustness to judgement values is cheap to show and should be shown.** A 64-point sweep over 15 knobs bounds the judgement's effect on the nine stages' retention at 0.030; single exporters and the slag legs move more (up to 0.033 and 0.052; §4.2). That bound is a stronger statement than one reference run. → N6

**For copper.**

- **A world secondary share hides the regions.** At 0.171 it averages EU-14 at 0.374 and the United States at 0.046, a value the missing direct melt partly explains (B1; §4.3, Section [5.5.1](#sec-disc-limits)); a recycling indicator for a region has to come from that region's account.
- **The trade legs are the least certain part of the account.** They absorb the sources' disagreements (A3; Section [5.5](#sec-disc-limits)); bilateral flows from the account should be quoted with their intervals, production totals can be quoted near their yearbook values.
- **Attribution through processing changes who consumes the mined copper.** Chile's and Peru's extraction attributed to China's final use exceeds China's direct concentrate imports from them by half (B3); consumption-based accounts of mine-site impacts depend on this step.

<a id="sec-disc-limits"></a>

### 5.5 Limitations

Section 5.5.1 bounds the method's conclusions; Section 5.5.2 lists the limitations of the copper application, each with the issue that records it.

#### 5.5.1 Limitations of the method

**Uncertainty.**

- The prior $\sigma$ are pedigree judgements; the Birge ratio shows they understate the disagreement, and a single scale factor corrects their mean, not their distribution.
- The posterior is linearised at the active set: it is local to the optimum and treats a binding band as an equality (ADR 0044).
- Stage 6 carries point estimates only; the footprints and embodied trade of B3 have intervals only where the delta method or the posterior draws were run (Table 3).

**Robustness tests.**

- The termination pair (A5) is on the base chain of 2026-09-16, not on the reference chain; it shows the size of the effect, not the reference run's own termination error. A $10^{-6}$ against $10^{-8}$ pair on the reference chain would put Figure 5d's diamonds and boxes on one QP.
- The knob sweep is local: a factor two either side of the shipped value, around one chain (SI S15.4). All five A6 breaches sit at 1.6–1.9 times the shipped no-data anchor coefficient of variation, inside a range capped at two; a wider range would find more.
- The structural modes, the calibrated values and the solver's settings are compared in their decision records, not swept.

**System boundary and constructs.**

- The account has no fabrication process, and scrap enters only smelters and fire refineries, not direct melt; secondary shares of regions that melt scrap directly, the United States among them, are understated (§4.3).
- The embodied flows of B3 rest on the market-share construct, which gives every output of a region's process the same input mix (§3.1.8, ADR 0037).
- A third of the scale-total objective lies in four unresolved source contradictions (Section [5.5.2](#sec-disc-limits-copper)); the account there reflects the $\sigma$ the sources were given, not a resolution.

**Transfer and reproducibility.**

- One material is reconciled; the framework's transfer is shown on a synthetic case only (N7).
- The ICSG yearbooks and the ore-allocation file are licensed; the feeds can be rebuilt only by holders of those sources (§3.4).

<a id="sec-disc-limits-copper"></a>

#### 5.5.2 Limitations of the copper application

<!-- Moved from method.md §3.2.8 on 2026-11-12, unchanged except its last line ("Uncertainty scope. Stage 6 carries point estimates only."), which 5.5.1 now states. -->

**Mine output where ICSG reports none** (issues #31, #183, #200, #201; ADR 0024).

- A region-year that ICSG covers and lists no mine for is held at zero mine output, as an observed zero at ICSG's reporting resolution (456 region-years in 20 regions), except where the crude-ore series or an ore grade reports a mine.
- 335 kt CC of concentrate is still produced over the period where ICSG reports no mine output, 99.5% of it exported: Malaysia's 112 kt CC, most of it in 1995–1999, when the crude-ore series reports its mine, then the United Arab Emirates, Switzerland, Luxembourg (which ICSG never lists), Singapore and Taiwan, each 19–39 kt CC.
- The zero costs fit: it raised the common-set scale-total objective by 23.3% against the chain before it, three fifths of the rise in Italy, whose smelter and fire refinery the prior had fed from an imputed domestic mine; the rest lies on the hubs' concentrate and matte exports, which recorded production cannot supply.

**Who pays for a disagreement.**

- Where two ICSG anchors disagree, the solver pays with trade and ore, because a trade leg's CV (0.069 at the median leg) exceeds an anchor's (0.017).

**Five regions carry two fifths of the objective** (41.4% of the scale-total objective; issues #111, #201).

- Italy (16.2%): its matte and anode use totals lose the supply of the imputed mine that #201 holds at zero (2012: 8.7 kt CC of anode use placed at 2.1).

- South Africa (7.5%): ICSG reports no smelter or refinery output from 2020, while BACI books 47–135 kt CC a year of anode exports, which the solver deletes (2021: 137.2 kt CC of anode use placed at 4.5).
- D.R. Congo (6.8%): BACI books 6–91 kt CC a year of anode and blister exports over 2005–2019, when ICSG reports at most 1.5 kt CC of smelter output (2012: 96.1 kt CC of anode use placed at 2.0).
- Zambia (5.2%): BACI cathode exports, booked largely to Swiss trading destinations, exceed refined output by up to 863 kt CC (2014: 1,363 against 499 kt CC).
- Uzbekistan (5.7%): BACI's mirror data miss most of its refinery output in 1995–1998, the years of 80% of its share.

**Usage the netting leaves undistributed** (ADR 0027).

- 1,320 kt CC of ICSG usage in 64 product-region-years that recorded production and imports cannot supply; 360 kt CC of it is South Africa's, its 2017–2018 smelter intake among it, against BACI concentrate exports equal to its whole mine output.
- Releasing the 867 kt CC of it whose recorded imports lie below the feeds' 0.3 kt resolution raised the scale-total objective by 2.4% against the reference run.

**Re-export anchors** (issue #183; ADR 0026, 0027).

- BACI books anode and concentrate exports of the Netherlands, Malaysia, Taiwan and the United Arab Emirates with no ICSG production behind them (829 kt CC of anode, 1,426 kt CC of concentrate over 1995–2024); their scale totals carry 10.8% of the scale-total objective, and a redesign of the netting is left out (SI S9.1).

**Choices that no measurement fixes.**

- The 0.3 kt and 100 kt $\sigma$ floors apply in kt to Mass rows and to kt CC rows alike, because ADR 0025 names no valuation (issue #241).
- The concentrate relation is priced by three $\sigma$ rules (trade, network twin, parametric); a measurement (issue #149) found the items of the stop rule (SI S13.1) equal to three decimals against `opt_20261022_112855_928da10c`, so copper's result does not depend on the choice (issue #140).

**Stocks.**

- Kazakhstan's refined exports exceed the refinery output of ICSG's 2025 yearbook and are booked as stock draws beyond $5\sigma$, $-348.3$ kt CC over 1995–2024 (issue #95).
- Producer stock changes follow ICSG where it reports them; the world posterior departs from ICSG's by 101.5 kt CC summed over 1996–2024, in territories ICSG does not report (issue #98).
- A cathode pool without a recovery efficiency holds its producer stock change at zero (issue #36).

**Input vintage.** Austria's 2006–2012 smelter values are a superseded yearbook vintage (issue #63).

**Numerical floor** (issues #99, #81).

- At a gap of $10^{-8}$, hard rows still close only to $1.1\times10^{-6}$ kt.
- 2,168–3,186 zero-prior cross-region cathode legs a year end within $10^{-4}$ kt of their lower bound, where Stage 5's slack threshold sorts them by rounding; pinning or freeing all of them moves no interval bound of Table 3 by more than 0.001.

<a id="sec-disc-outlook"></a>

### 5.6 Take-home messages and outlook

**Take-home messages.**

1. A $\sigma$-weighted reconciliation of a global, two-valuation copper account over 30 years solves on a workstation in minutes and closes every balance to the solver's floor; it moves the sources little and puts its adjustment on trade.
2. The solver's termination is part of the estimate: on the base chain it moved one stage by 0.223, seven times the largest change the whole knob sweep made to the nine stages on the reference chain (0.030).
3. The large residuals were source defects, fixed at the input; the residuals that remain point at four named contradictions.
4. Stated $\sigma$ understate the realised disagreement by a factor of about 2.3 on average and much more in the tail; intervals must say so.
5. The reconciled account adds what the sources cannot show alone: regional secondary shares, concentration with intervals, and attribution of mined copper through processing.

**Research directions.**

- **Calibrated uncertainty.** Estimate the prior $\sigma$ per source and family from the residuals, instead of one Birge scale, and propagate the posterior into Stage 6 so that footprints carry intervals.
- **A wider boundary.** Add fabrication and direct melt, so that the secondary share and final use are measured where scrap is used.
- **Other materials.** Bind a second metal to the framework's roles, the test of N7 on real data. <!-- check: name the metal once chosen (metal-agnostic milestone) -->
- **Finer resolution.** Disaggregate to company level and to transport infrastructure, so that the trade legs, the least certain part of the account, are constrained by more than two statistics. <!-- check: matches the REMASS outlook (docs/remass/); keep only if the paper wants it -->
