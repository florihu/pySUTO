# A restored production total is withheld where it contradicts its own chain

**Status:** accepted — amends [ADR 0018](0018-import-free-fans-distribute-domestically.md)
(bounds what the domestic fallback may restore) and closes the remedy question
[ADR 0019](0019-contradicted-flow-nodes-lose-anchor-authority.md) left open. Evidence: REP
[`…roundF-withhold`](../rep/analysis_opt_20260908_114321_279e7979_20260908_roundF-withhold.md).

## Context

[ADR 0018](0018-import-free-fans-distribute-domestically.md) recovered 29,648 kt CC of ICSG
production data the pipeline had been discarding. Most of it was a clean gain, but total
sum-row z² rose 4.3 % and world Conc-CC/Smelt-CC retention fell ~3 pp.

[ADR 0019](0019-contradicted-flow-nodes-lose-anchor-authority.md) found why: a Flow node's mass
balance is **hard**, and two tight scale totals sit across it — inflow from ICSG's concentrate
use, outflow from the restored refining feed. Because `σ_c` scales with `|c|`, the smaller side
is the cheaper anchor to satisfy, so the solver collapses the node to it regardless of which
side is better evidenced. In a minority of region-years ICSG's two series simply disagree:
Russia 1995 reports refining 50.8 kt CC against smelting 544 while exporting 501.8 kt Mass of
*cathode*; Iran's 1995–96 refining values of 4.1/3.8 jump to 100.6 in 1997.

ADR 0019 tried to price the contradiction away with σ and **failed on measurement**: z² returned
to baseline while only 0.46 % of the model's mass moved and retention did not recover. σ cannot
resolve a contradiction in the input data — it can only stop charging for it.

## Decision

**Where a region-year's ICSG refining feed contradicts its own smelting series, the ADR-0018
domestic fallback is withheld for that `Matte→CFR` group** — the fan stays undistributed exactly
as it was before ADR 0018, and the NetworkImputer fills the leg from the smelting side as it
always did. The test (`build_trade_shares.chain_inconsistent_refining`) is

    refining_feed  <  k · eff_smelting · (smelting concentrate + smelting scrap)

with `k = 0.5`; `eff_smelting` is read from the live `eff` params and the inputs are pooled per
[ADR 0005](0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md). It reads the **raw ICSG
rows only** — no model state — so it can gate the very feed build whose output the model-side
detector would need.

Scope is deliberately narrow:
- **Only the ADR-0018 fallback is gated.** Groups BACI covers keep their trade-share weights
  regardless; withholding never removes data that was already distributed.
- **Only `Matte→CFR`.** The contradiction is between the refining and smelting series; the
  `Concentrate→Smelting` and `Cathode→FU` restorations in the same region-years are not the
  trigger and are kept.
- Effect: **14 of 165 fallback groups withheld (1,530 kt), 151 kept (28,118 kt) — 95 % of the
  recovery retained.** Withheld: Zambia 3, Russia 3, Iran 4, Bulgaria 3, Congo D.R. 1.

## Considered options

- **σ-loosen the contradicted side** ([ADR 0019](0019-contradicted-flow-nodes-lose-anchor-authority.md)).
  Rejected on measurement: repriced the contradiction without resolving it (0.46 % of mass
  moved, retention unchanged) — the "prices away the symptom" failure Round C rejected for hub
  σ exceptions.
- **Merge ADR 0018 unchanged** and accept +4.3 % z². Rejected: it ships 13 newly collapsed
  node-years and −3 pp retention for data that is demonstrably self-contradictory.
- **Revert ADR 0018.** Rejected: it would discard 28,118 kt of sound recovered data to avoid
  1,530 kt of unsound data.
- **Widen the gate to every contradicted node-year** (139 by this test, 48 by the model-side
  detector). Rejected for now: **35 of those are pre-existing** — already collapsing in the
  shipped model, unrelated to ADR 0018 — so treating them changes the shipped baseline and
  belongs to its own round with its own evidence.
- **Correct the ICSG values** (e.g. treat Russia 1995–98 refining as missing rather than small).
  Deferred: the right long-run answer is a data-source question, and withholding is the same
  outcome achieved without editing a source series.

## Consequences

- The fallback restores data only where the source's own chain supports it; the rest stays in
  the honest "total known, split unknown" state ([ADR 0008](0008-unweighted-fans-are-not-distributed.md)).
- A withheld group is **reported, not silently dropped** — the build prints the count and the
  `contradicted_node_screen.csv` from ADR 0019's detector remains the list to take to the data
  source.
- The 35 pre-existing contradicted node-years are now named and measurable; they are the
  candidate for a later round.
- Glossary: **contradicted flow node** in [GLOSSARY.md](../../GLOSSARY.md).
