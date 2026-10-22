# Bilateral feeds are scoped to cross-region edges (mapped self-trade exclusion)

**Status:** accepted. Evidence run: `opt_20260901_101148_39fbd1bc`; REP
`analysis_opt_20260901_101148_39fbd1bc_*` (scrap-faucet fix) and the follow-up self-trade REP.
Amended 2026-09-16 by [ADR 0027](0027-re-exports-are-re-attributed-at-feed-time.md): the
bilateral feed now carries **inferred** legs as well as observed ones (re-attributed re-exports,
flagged in the ledger and σ-compounded), and a re-attributed pair whose two ends fold to the same
root region is dropped as mapped self-trade exactly as a source row would be.
Amended 2026-10-14 (#77): BACI 490 "Other Asia, nes" maps to **Taiwan**, not ROW — the known quirk
below is resolved; measured by #79 and shipped (`opt_20261014_131552_f320b5f7`: Taiwan's undistributed ICSG usage
15,566 kt CC → 0, common-set z² +1.41 %).
Amended 2026-10-15 (#84): BACI 178 "Congo" maps to **Congo, D.R.** (it carries DR Congo's cathode:
3,890 kt exported in 2012–2024, gross baci v19, against Brazzaville's 79 kt of ICSG production) and 711 "Southern African
Customs Union" to **South Africa** (its trade until BACI splits it in 2000), not ROW. Neither
creates mapped self-trade: BACI has no 178 ↔ 180 or 711 ↔ 710 rows. Measured by #86 and shipped over its items 5 and 6 (`opt_20261015_093646_ddd9d3d9`: ROW's 2012–2024
cathode ledger gap +5,829 → +1,951 kt CC, common-set z² −0.75 %).
Amended 2026-10-19 (#88, with the HS scope of the next amendment): in 2015–2023, 178's rows are **mirror
duplicates** and the BACI parse drops them (`MIRROR_DUPLICATES`; 3,163.8 kt of refined copper, 70.5 of
concentrate, 23.1 of anode, 2.2 of matte, gross). DR Congo's own declarations (180), counted over all of HS
7403.1x, lie within −74 … +422 kt (gross mass) of ICSG's DR Congo production in 2015–2023, and 178's 226–465 kt
a year would push them past it. In every year through 2014 they fall 5–323 kt short, and 476 kt in 2024; 178
fills that and stays DR Congo's. On `opt_20261015_093646_ddd9d3d9` (740311 only,
no window), 178 on top of 180 raised DR Congo's refined output 777 kt CC above its prior over 2020–2023 (183 on
`opt_20261014_131552_f320b5f7`). A first window of 2020–2023, derived on 740311 alone, is superseded. Which
mechanism repeats the copper (an importer declaring 178 for DR Congo, or transit) is not settled; the
balance is the evidence. 180's own 2015–2022 excess is a disagreement between ICSG and BACI that no mapping
explains (ADR 0019). 178's 2004–2010 concentrate stays open in #88.
Amended 2026-10-19 (#91): the Cathode sector reads all refined copper, HS 740311 cathodes, 740312 wire bars,
740313 billets and 740319 other refined, unwrought; the last three were dropped before. Their cross-region
trade is 0.41–1.35 Mt a year, 4.5–15.1 % of 740311's in the same year (1995–2024, gross; Austria's 2003 refined
exports: 77.7 %). ADR 0033's copper price stays HS 740311 alone. Measured with #88 by #92 and shipped (`opt_20261019_142026_83bfe843`: Austria's 1995–2014
cathode-use z² 48,500 → 2,624, common-set z² −5.57 %).
Amended 2026-10-20 (#95): in 2015–2023 the BACI parse keeps, per year and partner, the **larger** of 178's
and 180's refined-copper exports (HS 7403.1x pooled; `MIRROR_DUPLICATES = {178: (180, 2015, 2023)}`), and it drops
180's refined copper into Tanzania and Mozambique in every year as **transit declarations** (`TRANSIT_DECLARATIONS
= {180: (834, 508)}`). This supersedes two sentences of the #88 amendment: its drop of 178's 3,163.8 kt of refined
copper, and its "which mechanism repeats the copper … is not settled". For refined exports both act: the UAE and
DR Congo declare one shipment under two codes, and the legs into Tanzania and Mozambique are transit. Every other
178 row in the window (70.5 kt of concentrate, 23.1 kt of anode, 2.2 kt of matte, gross v21, and 178 as importer)
is still dropped as #88 decided, because the evidence covers only refined exports. Dropping 178 outright left the
UAE re-exporting cathode it no longer imported (common-set Δz² +25,625 vs `opt_20261015_093646_ddd9d3d9`, on
`opt_20261019_142026_83bfe843`): 178 → UAE is 80–252 kt a year in 2015–2023 against 180 → UAE 53–142 kt (gross,
all of HS 7403.1x), and the UAE keeps the 80–252 kt. The refined codes are pooled because DR Congo declares
52–71 kt a year of its UAE copper as 740319 in 2015–2017 where the UAE declares 740311 under 178; the smaller
declarations dropped are 391.5 kt of 178's and 1,331.1 kt of 180's refined exports. 180 → Tanzania is 302.3,
130.5, 35.6, 98.0, 264.9, 247.4, 196.3, 134.7, 143.2 kt in 2015–2023, and 180 → Mozambique 15.8, 61.1, 28.3,
47.2, 103.0, 126.0 kt in 2018–2023 (0.1 kt each in 2024 and 2015, none before; 178 sends neither anything).
Tanzania exported 213 kt of refined copper in 2015–2023 and Mozambique 16 kt; neither refines copper, and the
final buyer declares DR Congo as origin too. DR Congo's refined exports less its ICSG production go from +152,
−24, +415, +422, +371, +391, +203, +146, −74 kt under #88's rule to −41, +60, +530, +432, +232, +293, +74, +63,
−134 kt in 2015–2023 (gross mass, BACI HS92 V202601). 2017–2018 stay high through South Africa, a model region
with a refinery that the re-export netting already treats as a hub. Both rules run before ADR 0033's
reconstruction, so its copper price (the pooled HS 740311 unit value) moves −0.15 … +0.22 % vs the #88 rule
(2019 +0.21, 2021 +0.22, 2023 −0.15): the winning declaration's HS split feeds the 740311 pool (UAE 2015: 80 kt
of 740311 instead of 8 kt of 740311 and 52 kt of 740319). DR Congo's concentrate into the two ports was not
analysed. Measured 2026-10-21: ships (#95's REP, `opt_20261021_094942_d1ac7a81`: the UAE's cathode-use rows close,
common-set sum-row z² −5.28 % vs `opt_20261020_112726_2f451f31`).

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
  (#76). It maps to Taiwan; measured by #79 and shipped (`opt_20261014_131552_f320b5f7`).
