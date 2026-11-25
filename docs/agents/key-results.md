# Key results

A **key result** is a property of the reconciled account on the reference chain, or of its
sources as the account reveals them, that answers a sub-question of the manuscript's research
questions RQ1–RQ3 (`docs/manuscript/introduction.md`). It is what a reader of the paper takes
away. The results part of an abstract, a talk, a REP verdict or the manuscript carries key results
only.

## The three tests

A result is a key result when all three hold:

1. **Reference chain**: it holds on the chain the manuscript reads (GLOSSARY.md "Reference
   chain"), not on an earlier run.
2. **Bug-free**: it would hold had the pipeline been correct from the start. A defect found and
   fixed, an artefact of a solver setting or of an earlier chain, and a delta against a previous
   chain describe how the account was built: they are **debugging**.
3. **Research question**: it answers an RQ sub-question about the account, its sources or the
   copper system, and results.md §4 or discussion.md §5.1–5.4 states it.

"It is in the manuscript", "it is large or surprising" and "it transfers beyond copper" test
nothing: the manuscript itself still carries debugging among its contributions (see *Open*).

## Where a result belongs

| Level | Belongs in | Examples (reference chain `opt_20261124_141714_e2892160`) |
| --- | --- | --- |
| **Whole model**, all three tests pass | the results part | where the adjustment falls (cathode→users and anode→refineries carry 87 % of the edge objective; the trade hubs Switzerland and the Netherlands keep 45 % and 32 % of their recorded import mass); certainty (Birge √R 2.2); the spread of regional secondary shares; the concentration of refining and cathode use on China |
| **Stage fact** | the methods part; the results part only when it answers an RQ (Stage 5's Birge ratio answers RQ2) | 854,460 edges, 1,687,613 constraint rows, 98 s for Stage 4, imputation coverage |
| **Debugging** | Methods, SI, ADRs, REPs | the source defects fixed at the input (N5), termination artefacts and the tolerance pair (A5, ADR 0039), the ledger tally of Table 3, any delta against a previous chain |

The ledger of expected answers (Table 3) is the author's protocol: it shows how the paper checked
itself, and its met / not-met tally is bookkeeping.

## Steps: choosing results for a text

1. **Collect** candidates from results.md §4, discussion.md §5.1–5.4 and the number files
   `docs/manuscript/figs/*.json`. *Done when* each candidate carries its manuscript line and its
   JSON key.
2. **Test** each candidate against the three tests and against the user's recorded rules for that
   text (the memory note of the deliverable). *Done when* every candidate is labelled key result,
   stage fact or debugging.
3. **Select** key results for the results part and stage facts for the methods part. A key result
   the manuscript lacks enters the manuscript first (a `scripts/report/manuscript_checks.py` key
   and a Discussion line), then the text. *Done when* every results sentence of the text is a
   labelled key result with its source.
4. **Imply**: write each key result with its **implication**, the "so what" for a named reader
   (modellers, data providers, policy, footprint users), in the same sentence or the next. Take it
   from discussion.md §5.4 "Significance and implications"; an implication §5.4 lacks enters §5.4
   first, as the author's decision. *Done when* every key result in the text has its implication
   and each implication has its §5.4 line.

## Relevance check (reviewer brief)

Paste into the brief of every reviewer of a results text, beside the evidence check:

> Label each results sentence key result, stage fact or debugging, by the three tests of
> `docs/agents/key-results.md`. Flag every stage fact or debugging item in the results part,
> every sentence about an earlier run, and every key result without an implication from
> discussion.md §5.4. External facts already verified are listed in the text's
> claim→evidence table (for the ISIE 2027 abstract: `docs/conf/ISIE2027_abstract_reviews.md`);
> dispute one only with a new primary source.

## Open

The manuscript frames N4 (termination) and N5 (source defects) as contributions and as
take-homes 2–3 of discussion.md §5.6, and results.md §4.1 closes A4 with an earlier chain. Their
placement is the author's decision, pending an audit (retro 2026-11-25); derived texts treat them
as debugging until then.
