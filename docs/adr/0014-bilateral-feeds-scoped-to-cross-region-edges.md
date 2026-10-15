# Bilateral feeds are scoped to cross-region edges (mapped self-trade exclusion)

**Status:** accepted. Evidence run: `opt_20260901_101148_39fbd1bc`; REP
`analysis_opt_20260901_101148_39fbd1bc_*` (scrap-faucet fix) and the follow-up self-trade REP.
Amended 2026-09-16 by [ADR 0027](0027-re-exports-are-re-attributed-at-feed-time.md): the
bilateral feed now carries **inferred** legs as well as observed ones (re-attributed re-exports,
flagged in the ledger and σ-compounded), and a re-attributed pair whose two ends fold to the same
root region is dropped as mapped self-trade exactly as a source row would be.
Amended 2026-10-14 (#77): BACI 490 "Other Asia, nes" maps to **Taiwan**, not ROW — the known quirk
below is resolved; to be measured by #79.

## Context

The region map folds several source reporters into one root region (Hong Kong 344 and Macao 446
→ "China"; ~150 small reporters → "ROW"). A bilateral trade row whose two endpoints fold onto the
same root region becomes **mapped self-trade** (GLOSSARY.md term): its root 8-tuple address is
byte-identical to the **domestic** edge, so the feed writes a tiny, customs-precise trade value
onto a production-scale transfer. 5,181 of 78,312 BACI rows (~6.6 %) are mapped self-trade.

The flagship failure: HK→mainland blister (HS 740110, 0.021 kt in 2015) landed as the
`Matte.China→CFR.China` **Mass** prior (0.0276 kt, σ=0.005) next to the ICSG **CC** prior of
5,678 kt Cu — physically impossible (content ≫ gross mass). Because the Mass edge counted as
covered, the network Mass-twin repair was blocked; the conc-ratio rows then dragged the CC side to
~0, deleting China's smelting (6,885 → 14 kt Cu), halving world concentrate demand, and cutting
Chile's mining chain by ~80 % in Mass. The same mechanism poisoned China's cathode edge (740311
self-trade, the `sum_usetotout_Cathode_China` z≈92 tail), China's anode edge, and ROW's
matte/concentrate/cathode edges (intra-ROW trade).

GLOSSARY.md already documented the intended invariant — "BACI … no self-trade rows, so no domestic
leg" — true at source level, silently broken at root level by the fold. Likewise
`build_trade_shares.py` already forces the domestic fan leg to the residual (domestic `imp_cc=0`):
the codebase's intent that trade data never informs domestic edges predates this ADR.

## Decision

**A bilateral-trade feed is scoped to cross-region edges.** Implemented with existing mechanisms,
no new seam:

- a built-in structural mask `cross_region` (`region_from != region_to`) joins
  `supply/use/tradable/sb` in `AccountingStructure._build_edge_masks` (the CSV mask rules cannot
  express a cross-column comparison);
- BACI's `meta.xlsx` declares `Edge_scope_mask = cross_region` (⚠ `data/` is gitignored — on a
  fresh checkout re-add the row `Scope | Edge_scope_mask | cross_region` to
  `data/input/feed/baci/current/meta.xlsx`, else the scope silently does not apply; the baci_feed
  log's "edge_scope_mask … dropped" line is the check), and
  `_apply_edge_scope_mask` (feedpipe) drops concordance rows resolving outside the mask —
  `scripts/baci_feed.py` now applies it explicitly since it builds its ct without
  `run_feed_pipeline`.

The dropped rows are real trade, but at this aggregation level they are neither a boundary flow
nor the domestic transfer — there is no valuation in which "0.021 kt crossed the HK border" is a
valid observation of an edge whose true magnitude is 10–500,000× larger. With the rows gone the
domestic edges are uncovered and the NetworkImputer closes them (Mass = CC/conc), exactly as in
every unaffected region.

## Considered options

- **Keep but σ-inflate** — leaves a lying value in the prior, keeps the edge "covered" (twin
  repair stays blocked), and adds an arbitrary inflation knob.
- **Keep but mark uncovered** — needs a partial-coverage concept Stage 2 does not have.
- **Provenance-based row drop** (drop only rows whose *src* regions differed) — equivalent for
  BACI (raw trade has no self-loops), but weaker as a statement: the invariant is about the feed's
  scope, not row provenance, and the scope mechanism already existed.

## Consequences

- A report-only Stage-2 gate (`ImpDiagnostic.cross_valuation_check`,
  `diag/cross_valuation_check.csv`) now flags any covered CC/Mass pair with CC > Mass, so a future
  inconsistent observation of this class fails loudly instead of silently reshaping the solution.
- Known quirk left open: BACI 490 ("Other Asia, nes") maps to ROW while Taiwan is a model region —
  Taiwan therefore has no trade data (the origin of the Taiwan noisy-zero scrap faucet). Revisit
  as a data decision. **Resolved 2026-10-14 (#77):** 490 is Taiwan's trade in UN Comtrade and BACI,
  and folding it into ROW gave ROW Taiwan's cathode imports, which zeroed ROW's domestic semis share
  (#76). It maps to Taiwan; to be measured by #79.
