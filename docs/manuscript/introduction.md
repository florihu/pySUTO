<!--
Introduction of the pySUTO model paper. Numbered Section 1. Drafted as bullet points on
2026-11-12, like the Method, the Results and the Discussion: one point per bullet; prose
follows once the content is agreed. Section 2 is not assigned: the Method keeps its
number 3 "as in the paper" (method.md header); decide whether a background section
fills it or the Method becomes Section 2.

Research questions. The three research questions RQ1–RQ3 and their sub-questions below are
the single list the paper answers. Each sub-question carries the id of its ledger row
(Table 3, SI S15): the row is the answer expected before the numbers were read, and the
Results report whether it was met. The rows were written as hypotheses ("the adjustment
grows upstream"); they are presented as expected answers to questions since 2026-11-12,
with their metrics, thresholds and outcomes unchanged (SI S15.1). Questions without a
pre-registered answer are marked exploratory. The same RQ numbering is used in
method.md §3.3, results.md §4 and Table 3, and discussion.md §5.1.

Conventions:
  *Cite (author):*  a visible placeholder for a statement that needs an external source,
                    written by the author; candidate refs.bib keys where present,
                    "verify:" otherwise
  → Nk             the bullet previews contribution Nk of the Method's contribution block
  no number appears here that the Method or the Results do not report

Positioning: the state-of-the-art paragraph parked in the method.md header comment
("Positioning paragraph ... for the Introduction") is the source of the gap bullets.
Word budget of the later prose: <= 900 words, headings excluded.
Motivation, literature strands, gaps and practicality expanded on 2026-11-12: one bullet
per strand, each ending with what it leaves open; references outside refs.bib are
candidates to verify (none is in refs.bib yet). The strands may not fit the budget; see
the comment under "What existing approaches do and leave open".
-->

<a id="sec-intro"></a>

## 1 Introduction

**Why a physical account of copper needs reconciling.**

- Copper's supply chain runs from mine to final use through concentrate, matte, anode and cathode, and crosses borders at each of these four products.
  - *Cite (author):* copper's role in electrification and the energy transition, and the demand outlook. verify: IEA (2021) *The Role of Critical Minerals in Clean Energy Transitions*; Elshkaki et al. 2016; Schipper et al. 2018.
- Policy now asks for the chain stage by stage and region by region: the EU Critical Raw Materials Act sets benchmarks for extraction, processing and recycling and caps the dependence on any single third country at each stage.
  - *Cite (author):* Regulation (EU) 2024/1252. verify: the article numbers and the benchmark values before quoting any of them.
- Questions of secondary supply, supply concentration and consumption-based responsibility need the whole chain in one account, by region and year, in physical units. They are the three questions of RQ3 (B1, B2, B3).
  - *Cite (author):* copper MFA and footprint studies that need this account. verify: candidate sources; `wieland2022piolab`.
- The account has to be physical: footprints from monetary multi-regional input–output tables allocate by price, which for a metal changes by orders of magnitude along the chain, from ore to cathode. <!-- verify: the claim and its size; the price ratio is not a number of this paper -->
  - *Cite (author):* physical versus monetary footprints of metals. verify: Merciai and Schmidt 2018 (EXIOBASE 3 hybrid); Hawkins et al. 2007; Schoer et al. 2013; Wieland et al. 2018.
- No single source describes that chain:
  - industry statistics report production and use per region in contained copper ([ICSG, 2025](#ref-icsg2025yearbook));
  - trade statistics report bilateral flows in gross mass ([Gaulier and Zignago, 2010](#ref-gaulier2010baci));
  - crude-ore extraction comes from a third source, a price-allocated ore series ([Kroiss, 2024](#ref-kroiss2024crudeore)).
- Joined, the sources do not close: they disagree on the same flows, in different valuations, and each has its own reporting errors. An account built from them has to be reconciled, and the reconciliation has to say how uncertain its result is.
- The errors are not only noise: quantities booked in the wrong unit, misread yearbook blocks, nested series counted twice, and mirror and re-export discrepancies in trade. A reconciliation that only spreads the disagreement hides them; one that locates them audits the sources. → N5
  - *Cite (author):* data-quality problems of trade statistics (mirror flows, quantity and unit-value outliers). verify: `gaulier2010baci`; mirror-statistics studies.
- Without an uncertainty per flow, a regional difference or a concentration index above a threshold cannot be told apart from the error of the sources.
  - *Cite (author):* uncertainty in MFA and MRIO results. verify: Laner et al. 2014 (systematic review of MFA uncertainty); Lenzen et al. 2010 (MRIO uncertainty); Moran and Wood 2014 (footprints disagree across MRIO databases).

**What existing approaches do and leave open.**

<!-- One bullet per strand, each ending with what it leaves open. If the strands do not
fit the 900-word budget, a short Background section could take them as Section 2, and
the Introduction keeps the motivation, the gaps and the aim. Decide with Section 2. -->

- *Copper MFA* traces copper's stocks and flows from mine to waste, globally and by region, but mostly treats trade apart from production, reconciles little across sources and propagates little uncertainty. → RQ3
  - *Cite (author):* verify: Graedel et al. 2004 (multilevel copper cycle); Glöser et al. 2013; Soulier et al. 2018; Ciacci et al. 2020.
- *Copper trade-network studies* map who trades with whom at each product, from trade statistics alone, in gross mass, without closing the trade against production. → N3, RQ3 (B2, B3)
  - *Cite (author):* verify: Dong et al. 2020; further network studies of copper ore and concentrate trade.
- *System-definition frameworks* structure material-system data but do not reconcile it ([Pauliuk and Heeren, 2020](#ref-pauliuk2020odym); `pauliuk2019datamodel`).
- *Data reconciliation in MFA* propagates errors through a balanced system ([Cencic and Rechberger, 2008](#ref-cencic2008stan)), at the size of a single system and mostly with equality constraints; data uncertainty can be characterised from a source's pedigree ([Laner et al., 2016](#ref-laner2016)). <!-- verify: as in Method N2 --> → N2
  - Bayesian MFA estimates flows and their uncertainty jointly from data and priors, again at the size of a single system. It is the closest approach to carrying uncertainty from source to posterior and has to be named here. → N1
  - *Cite (author):* verify: Gottschalk et al. 2010; Lupton and Allwood 2018.
- *Constrained balancing* reconciles large multi-regional accounts — AISHA, large-scale matrix balancing, the IELab and the PIOLab (`geschke2011aisha`, `geschke2019parallel`; [Lenzen et al., 2009](#ref-lenzen2009matrix); `lenzen2014virtuallab`; [Wieland et al., 2022](#ref-wieland2022piolab)) — and weights the adjustment by source reliability at the balancing step. → N1, N2
  - The gap is narrower than "no uncertainty": Eora publishes standard deviations of its tables and MRIO uncertainty has been studied by Monte Carlo. What is not done is to carry each source's pedigree $\sigma$ through concordance and imputation to an analytic posterior $\sigma$ per flow under inequality constraints. <!-- verify: as in Method N1; Lenzen et al. 2010, 2013 (Eora) — read what Eora's standard deviations are (prior or posterior) before stating the gap -->
- *Physical and hybrid supply–use tables* exist as accounting standards and as databases; the nearest existing product to pySUTO's output is the multi-regional hybrid supply–use table of EXIOBASE 3. pySUTO differs in resolving one element's chain product by product, in two valuations, with a posterior $\sigma$, from an open tool. → N1, N3
  - *Cite (author):* verify: [Suh, 2004](#ref-suh2004pio); `altimirasmartin2014piot`; Hoekstra and van den Bergh 2006; SEEA Central Framework (UN 2014); Merciai and Schmidt 2018; Nakamura et al. 2007 (WIO-MFA).
- *Transparency and reproducibility* of industrial-ecology models is an open demand (`pauliuk2015transparency`, `pauliuk2024traceability`, `wilkinson2016fair`, `stodden2014best`), but the large balancing tools run inside laboratory infrastructures; no open reconciliation tool for physical accounts runs at the scale of a multi-regional account on a workstation. → N6, N7 <!-- verify: how open AISHA, the IELab and the PIOLab code are -->
- Balancing and reconciliation studies rarely report their solver's termination criterion, and none we know tests it. → N4 <!-- verify: candidate sources; soften to "rarely report" if none is found (Discussion 5.3) -->
- Six things are therefore open:
  - carrying each source's uncertainty from its pedigree to a posterior per flow, in an account of multi-regional size with inequality constraints; → N1, N2
  - reconciling gross mass and contained copper in one account, so trade and industry statistics constrain the same flows; → N3 <!-- verify: as in Method N3 -->
  - using the reconciliation to locate the defects of the sources, not only to spread their disagreement; → N5
  - knowing whether the reconciled answer depends on how the problem is solved and on the judgement values set in building it; → N4, N6
  - a framework that a new material enters through its data and its accounting structure, not through new code; → N7
  - knowing what such an account shows that its sources, read separately, cannot. → RQ3

**Why it is practical.**

- It runs on a workstation, on an open stack (NumPy, SciPy, pandas and the open-source solver Clarabel; `goulart2024clarabel`), not on laboratory infrastructure (Section [3.1.9](#sec-software)).
- One reconciled account yields many views: the physical input–output table, footprints, embodied trade and economy-wide MFA indicators are derived from it on demand (Section [3.1.8](#sec-stage6)).
- It is re-run when its sources update — the industry yearbook and the trade database are published every year — and every number traces to a run identifier and its exact input versions.
- It transfers to the other metals the Critical Raw Materials Act lists, such as nickel, cobalt and lithium, through their feeds, accounting structure and parameter tables. → N7
- The code is open; licensed inputs such as the industry yearbooks are not redistributed, and what is shared is the code and the derived, attributed results (Section [3.4](#sec-availability)). <!-- verify: the outcome of the ICSG licence request -->

**Aim.**

- This paper presents pySUTO, a six-stage framework that compiles, reconciles and propagates the uncertainty of physical supply–use tables, and applies it to global copper: 23 sectors in each of 47 regions, 1995–2024, in gross mass and in contained copper (Section [3](#sec-method)).

**Research questions.** Each sub-question names the row of Table [3](#tab-ledger) that holds its expected answer, written before the numbers were read (SI S15).

- **RQ1 How consistent is the reconciled account, and how is its adjustment of the sources distributed?** (Section [4.1](#sec-results-reconciliation))
  - Does the account close every hard identity? (A1)
  - How compatible is the adjustment with the uncertainty stated for each source, and does this differ between single flows and reported totals? (A2)
  - Where in the account does the adjustment fall? (A3)
  - How large is the adjustment along the chain, from ore to cathode? (A4)
- **RQ2 How certain is the reconciled account, and how far does it depend on how it is solved and parametrised?** (Section [4.2](#sec-results-uncertainty))
  - How much do the constraints contract the stated uncertainty, and is the stated uncertainty calibrated against the residuals? (exploratory)
  - How far does the solver's termination rule move the answer? (A5)
  - How far do the judgement values of the framework move the answer? (A6)
- **RQ3 What does the reconciled account show about the copper system that the separate sources cannot?** (Section [4.3](#sec-results-system))
  - How does the share of secondary copper in refinery input differ between regions? (B1)
  - How does the concentration of producers change along the chain, and does the reconciliation change it? (B2)
  - How does the copper mined in Chile and Peru that China's final use embodies compare with China's direct imports from them? (B3)
- RQ1 and RQ2 ask about the method on a real case; RQ3 asks what the method makes visible.

**Structure.**

- Section [3](#sec-method) presents the framework and its copper application, with the contributions N1–N7 and the evaluation protocol that answers RQ1–RQ3.
- Section [4](#sec-results) answers the research questions; Section [5](#sec-discussion) discusses why the answers came out as they did, how they compare with previous work, and the limitations.
