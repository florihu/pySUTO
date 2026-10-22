# The fan's domestic leg is bounded by the region's own production

**Status:** accepted — J.1 merged 2026-09-14 (shipped run `opt_20260910_133212_45f034f1`);
**amended 2026-09-16 by [ADR 0027](0027-re-exports-are-re-attributed-at-feed-time.md)** — the
bound now covers Cathode and Anode as well as Concentrate (see *Amendment* below). Amends
[ADR 0002](0002-single-endpoint-rows-fan-out-not-replicate.md) option D and
[ADR 0018](0018-import-free-fans-distribute-domestically.md). Evidence: REP
[`…india-concentrate-residual-domestic`](../rep/analysis_20260910_india-concentrate-residual-domestic.md).
**Amended 2026-10-14 (#76), narrowed 2026-10-19 (#78)**: a domestic concentrate leg the bound
sets to 0, in a region that mines no concentrate, is an *observed zero* (see *Amendment (#76)*
below); measured by #78. **Amended 2026-10-21 (#96, #97, #31)**: a region-year with no ICSG
SX-EW output says zero on its own SX-EW output; the mine-output and cathode-leg zeros were measured
and rejected (see *Amendment (#96, #97, #31)* below).

## Context

ADR 0002 option D fans each ICSG single-endpoint usage row over producer regions with BACI
import shares and gives the domestic leg the residual

    domestic (X==Y) = max(0, usage_CC − Σ_X conc · BACI_Mass(X→Y))

adding that it *"needs no ICSG production data"*. That property is the defect. Wherever BACI
imports in copper content fall short of what the consumer's process used, the shortfall is booked
as **domestic supply of a product the region never produced**. The hard Flow-node balance then
forces the posterior domestic leg back to production, and the usage-side scale total
(`usetotout`, ADR 0006, built from the same residual) pays for it every year.

India Concentrate is the textbook case and the model's largest single cell: **112,955 z²** pooled,
z −40 to −55 in 24 of 30 years. Its smelters consume 250–840 kt CC a year, BACI imports at the
0.275 nominal supply about half, and the fan books the rest — 100–456 kt/yr — as domestic
concentrate against 23–48 kt mined. The solver's posterior is physically sensible (domestic ≈
mine output, imports +59 %, smelter input at 84 % of ICSG); the fine is for a prior we built.
BACI's gross imports into India would need 36–59 % Cu (2003–18) to feed the ICSG figure, so the
raw pair disagrees by ~40 % and the repo cannot say which side is wrong — but booking the gap as
domestic mining is wrong under either reading.

The class, on `opt_20260910_114200_7f4040d9` (domestic fan-leg prior above own production prior,
CC, 30 y pooled): **Concentrate 210 region-years, 21.7 Mt, 348,142 z²** — 12.0 % of all sum-row
z²; Matte 33 region-years, 1.3 Mt, 30,092 z²; Anode none since ADR 0021/0022. Three kinds by the
grade BACI's imports would need over each region's excess years: importers whose pair is
inconsistent (India 0.44, Japan 1995–2004 0.40, Brazil 1995–2003 0.42), marginal cases a
regional import grade would close (China 0.32 — 9.5 Mt of the tonnage — Spain 0.31, Philippines
0.34), and near-self-sufficient smelters with negligible imports (United States 1997–2003,
Poland, Uzbekistan: use exceeds mine output by 5–12 %, an ICSG-internal or eff-nominal question).

## Decision

**A region cannot supply itself more of a product than it produces.** For sectors in
`BOUND_DOMESTIC_SECTORS` (Concentrate first) `build_trade_shares.compute_trade_share_weights`
bounds the domestic raw weight by the consumer region's own ICSG production of the product —
read off the producing-process → product edge of the ICSG `F_root` (`MF→Concentrate`) — and
spreads the remainder of the usage over the import legs in BACI proportion, so usage is still
conserved:

    dom_raw   = min(max(0, usage − Σ imp_cc), production_CC(Y, t))
    imp_raw_X = imp_cc_X · (usage − dom_raw) / Σ imp_cc

Bounded only where the recorded imports can carry the remainder, `Σ imp_cc ≥ residual −
production` — the import legs are at most doubled. An import-free fan has no leg to carry it and
keeps ADR 0018's all-domestic form. A near-self-sufficient smelter (United States 1997–2003:
usage 996, mine 886, imports 26 kt CC) has legs that would need to grow fourfold; that gap is
ICSG-internal, not BACI's, and spreading it fabricates imports instead of domestic supply. The
first build (Round J) bounded on "≥ 1 recorded import" and did exactly that: the US legs were
scaled ~4× in 1998–99 and 2001–03, `time_interp` carried the inflated legs into the uncovered
2000 legs (Indonesia→US 7 → 34, Peru→US 5 → 22 kt, σ ≈ 2), and the solver traded US mine output
for them (posterior 95 % → 66 % of anchor in 2000, 10–15 pp lower in 2001–03). The guard removes
the self-sufficient kind from the bounded set; it stays visible in the builder's diagnostic.

This is the ADR 0015 principle applied to the fan: make the prior obey the identity the hard
balance enforces on the posterior anyway. The tension moves from a fabricated domestic anchor to
the import legs — the real BACI-vs-ICSG scale disagreement — where ADR 0017's size-damped σ lives.

Builder diagnostic on the first build (trade_shares v11): **265 domestic legs bounded, 27,069 kt CC
moved from the domestic residual onto import legs.** India 2015: domestic 30, imports rescaled to
791 in BACI proportion (Chile 211, Indonesia 183, Australia 120, Peru 82, Brazil 79, ROW 58), fan
total 821.3 = the ICSG usage.

## Considered options

- **Extend the ADR 0019 detector to Concentrate and loosen the usage anchor** (`UNCLOSED_ANCHOR_CV`).
  Two constants — but the detector is one-sided (it flags outflow *below* inflow, the Matte
  disease; here outflow exceeds inflow), and Round E's lesson stands: the posterior is already at
  production, so a σ removes the fine without changing the answer and hides a construction defect.
- **Bound the residual and leave the remainder undistributed** (usage not conserved). Shrinks the
  whole domestic chain prior to BACI's scale; ADR 0021's refined-production anchor then contradicts
  it one node down — the contradicted-node class relocated, not removed.
- **Regional concentrate import grade** (ADR 0015 pattern). Closes only the marginal kind and has no
  data source in the repo; it would be tuned to the residual it is meant to remove.
- **Bound Matte and Anode too.** Anode has no excess; Matte's 1.3 Mt / 33 region-years is the
  follow-up once the rule is measured on Concentrate (its production source is `Smelting→Matte`,
  itself derived from the smelting usage series rather than observed directly).

## Consequences

- ADR 0002's "needs no ICSG production data" no longer holds for bounded sectors; ADR 0018's
  import-free case is unchanged.
- The rule needs the ICSG `F_root` production edges at build time — already what the builder reads.
- `trade_shares` v11; the ICSG feed, Stage 2, Stage 3 and the solver are rebuilt in that order.

## Measured (Round J, two variants)

Both cold, **30/30 Solved**, against the shipped Round-I run `opt_20260910_114200_7f4040d9`;
`scripts/explo/roundJ_fan_bound_diag.py`. **J** = bound on "≥ 1 recorded import" (trade_shares
v11 → `opt_20260910_125244_4d50b04a`); **J.1** = bound only where the imports can carry the
remainder (v12 → `opt_20260910_133212_45f034f1`), the rule as written in the Decision above.

| sum-row z² | base | J | J.1 |
|---|---|---|---|
| total, 30 y | 2,893,508 | 2,760,269 (−4.6 %) | **2,745,311 (−5.1 %)** |
| `usetotout_Concentrate_India` | 112,955 | **12,353** | 23,261 |
| the 210 excess region-years (fixed population) | 348,126 | 155,930 | 178,148 |
| Argentina Smelting | 61,933 | 33,536 | 33,536 |
| United States Mining + Ore | 17,634 | 26,641 | 23,327 |
| Congo D.R., all sectors | 108,666 | 132,874 | 110,698 |
| Korea Concentrate / PNG Anode (tiny anchors) | 13,779 / 737 | 27,174 / 13,148 | 27,174 / 13,148 |
| Σ increases, of which on anchors < 10 kt | — | 192,430 / 46,146 | 144,178 / 35,614 |
| domestic legs bounded / kt moved (builder) | — | 265 / 27,069 | 197 / 24,194 |

Common to both: Concentrate −140 to −155k and Smelting −33 to −35k; Spain −25k, Philippines
−20k, Japan −19k, Brazil −10 to −13k; contradicted Matte node-years 9 → 5; exporters' posterior
trade volumes up (Chile +0.2 %, Peru +0.1 %, Indonesia +1.6 %, Australia +2.0 %, Brazil +3.6 %,
2010–19; the retention *ratios* fall only because the priors on their legs grew); India's 2015
smelter feed 685 → 718–720 against ICSG 821, with domestic at mine output and imports 691–694.

**What the guard buys and costs.** J.1 removes the Congo side effect entirely (Congo was in the
self-sufficient kind) and most of the United States one — US mine output 2000 posterior/anchor
0.95 (base) → 0.66 (J) → 0.71 (J.1); 2002: 0.89 → 0.72 → 0.87 — but it un-bounds India's two worst
years (2014, 2016: the imports would have to grow 2.1–2.2×), which return to z −55: India's cell
12k → 23k. The guard cannot tell India 2014 (imports are 45 % of usage; BACI plausibly short)
from the United States 2000 (imports 3 % of usage; ICSG-internal) by the ratio alone. That is a
data-plausibility judgment, not a physical one — a strictly physical bound (grade ≤ 0.35 ⇒ legs
at most ×1.27) would never bound India — and it is the owner's call.

**Two effects to read correctly, in both variants.**
- *Tiny anchors.* 36–46k of the increases sit on rows whose anchor is below 10 kt. Korea's
  Concentrate node had been back-filled by the network with 180 kt/yr of "mine production" Korea
  does not have (its residual demanded it); the imputed-production class measure missed such cases,
  the builder's 197–265 legs is the true count. With the bound that chain is gone and a 3–6 kt
  export anchor is priced at σ ≈ 0.06 kt, so a 2 kt miss is z −40. Same for Papua New Guinea's
  anode chain. σ_c ∝ |c| with no floor (ADR 0016/0017 territory), not a physical regression.
- *United States 2000 residual.* Its own fan is never bounded in 2000 (and under J.1 not in 1998–99
  either); the cut comes through `time_interp`, which fills the uncovered Indonesia→US and Peru→US
  legs of 2000 from neighbouring years where the fan *was* bounded (7 → 34, 5 → 22 kt at σ ≈ 2).
  Whether bilateral legs should be temporally interpolated at all is a pre-existing question
  (ADR 0010 treats an unrecorded route as "no trade"): follow-up.

**Recommendation:** ship **J.1** — it never inflates a leg more than twofold, removes the Congo
effect and halves the US one, and carries the larger headline gain — and record India 2014/2016
and the interpolation of bilateral legs as the next two items. **Decided 2026-09-14: J.1 shipped.**

## Amendment (Round L.2, 2026-09-16): the bound covers Cathode and Anode

`BOUND_DOMESTIC_SECTORS = ["Concentrate", "Cathode", "Anode"]`. ADR 0027's re-attribution exposed
why Concentrate alone was not enough. Where a hub's ICSG intake exceeds its BACI imports and it
produces nothing (France, UAE and Malaysia cathode), netting removes the export legs the phantom
refinery had been built to supply — but with the cathode residual unbounded the same phantom
simply moved to the use side: France 2015 booked 49.4 kt of *domestic* cathode where it had
booked 38.3 + 11.1 of exports, the same node inflow under another name. The ledger labelled those
hub-years `ceiling-relabel` precisely so the gap would not pass unnoticed.

Cathode and Anode cannot read production off the ICSG `F_root` the way Concentrate reads
`MF→Concentrate` — there is no ICSG-anchored `CP→Cathode` edge — so `icsg_raw_production` derives
it from the raw ICSG rows, with the same definitions ADR 0027 uses for `P`
(Cathode = Refining/Anode use · eff_ER + Refining/Scrap use · eff_CFR · eff_ER + SXEW supply;
Anode = (smelter intake · eff_Smelting + Refining/Scrap use) · eff_CFR). One helper, shared by
`build_trade_shares.py` and `build_baci_reexport.py`, so the two never drift apart.

Measured against the netting alone: sum-row z² 1,828,508 → **1,720,192** (a further −5.9 %), Round
C hub-class z² 67,300 → 35,136, Cathode/France phantom production 1,817 → 86 (−95 %), Italy's
cathode cell halved, and nothing regresses; the three exporters are identical to three decimals in
both variants, which is what established that ADR 0027's exporter cost is inherent to the
re-attribution and not to the unbounded cathode residual.

**Two defects in the extension's own first cut, both fixed before the measurement above** (they
are the reason this paragraph exists):

- *A non-producer must say "zero", not "nothing".* `icsg_raw_production` returned no row for
  regions that do not mine or refine at all, and the guard reads a missing row as *unknown* and
  declines to bound — silently un-bounding exactly the phantom-mine regions the bound exists for
  (Concentrate/Germany 0 → 38,731 z²; Korea's phantom mine back at 2,173 kt). The helper now spans
  the full fan-group grid with an explicit 0.0.
- *`P_Anode` omitted the scrap entering CFR* (ADR 0021), so Belgium 2015 read 139 kt of anode
  against the 380 kt its own refinery consumes and its residual was capped far too low
  (CFR/Anode Belgium 44,349 → 75,656 z²). Corrected to `(matte + scrap) · eff_CFR` → 292 kt; the
  cell returned to 43,884. `U_Matte` carries the same correction.

## Amendment (#76, 2026-10-14): a zero domestic leg is an observed zero

When the residual or the bound gives a weighted fan's domestic leg weight 0, that 0 is the
statement "the region supplies none of its own use", not a missing value. Until #76 it was lost:
the ICSG feed's σ is proportional to the weighted value, so the leg left Stage 1 at σ 0, the
collector read it as uncovered, and the imputers closed it. On `imp_20261013_113740_5e11f431` the
`NetworkImputer`'s flow balance wrote 64 domestic `Cathode→FU` legs at 4,905 kt CC (ROW 4,496,
189–318 kt CC a year in 1995–2011) as the region's supply minus its other uses, the time
interpolator 4 more at 307 kt CC, and the smelter recipe, run backward, and the balance 98 domestic
`Concentrate→Smelting` legs at 1,000 kt CC: use ICSG does not report, booked so that the solver
never saw the contradiction. The feed now carries every such leg as covered at 0 with
`σ = max(0.01·|U|, 0.3 kt CC)`, where `U` is the usage its fans distribute (an aggregate region
fans each member's row onto the same leg) — the net-stock seed's pricing
([ADR 0040](0040-stock-change-legs-are-held-at-their-prior.md)). Stock-change legs keep their own
seed and supply-mix fan (ADR 0040, 0043); zero-weight import legs stay uncovered (no trade on that
route is not the same statement). Where the region's own production then has nowhere to go, the
solver must settle that conflict in the open. ROW's 1995–2011 surplus traces to an input error,
BACI 490 folded into ROW (#77, measured by #79). This amendment is measured by #78.

**Narrowed on measurement (#78, 2026-10-19).** The first build stamped every zero-weight domestic
leg: 367 on icsg v15 + trade_shares v36. The run (`opt_20261019_145258_6bbfa0b5`) raised common-set
sum-row z² 29.2 % vs `opt_20261019_142026_83bfe843`, with Italy +262,853, the United Kingdom +95,382,
South Africa +29,575, Bulgaria +32,336 and Spain +23,499, on their refinery and smelter rows. Those are
the zeros in region-years that produce the product: 25 of the 271 `Cathode→FU` zeros (Italy 13, the
UK 4, South Africa 3, …) and 14 of the 96 `Concentrate→Smelting` zeros (Bulgaria, Spain). There the
bound's 0 is the residual where imports exceed usage, and pinning it left the region's own output
nowhere to go. That is not the statement "the region supplies none of its own use". The observed
zero now covers a domestic concentrate leg in a region whose own ICSG concentrate production
(MF→Concentrate in the feed) is 0 that year, "a non-producer must say zero, not nothing"
(`icsg_feed.non_producer_mask`, `OBSERVED_ZERO_SECTORS`). On the same feed that is 82 legs, Germany's
30 among them (#32). The 246 cathode zeros in non-producers (France, the Netherlands, Taiwan,
Switzerland, …) would qualify by the same test, but the feed carries no cathode production edge to
read it from; keying the test on the bound's raw ICSG production is #96. Every other zero-weight
domestic leg stays open to the imputer, as before #76.

## Amendment (#96, #97, #31, 2026-10-21): a non-producer says zero on its own output

**A region-year ICSG covers that produces none of a product now carries that zero on the edge its
own output would leave on, not only on a zero-weight domestic leg.** On the reference chain
`opt_20261020_112726_2f451f31` (`const_20261019_170151_13460e94`), summed over the region-years
with no ICSG mine production, the solver built mine and SX-EW output where ICSG reports none,
prior → posterior in kt CC: Ore→SXEW 3 → 1,667 (Belgium 474, Germany 341, the UAE 341),
CO→Mining 895 → 3,030, MF→Concentrate 625 → 1,168 (#97, #31). Belgium's 479 kt CC of CO→Mining
prior was written by the imputers, not reported. And #78's test read production off the feed's
`F_root`, which carries no cathode production edge, so 246 zero-weight domestic `Cathode→FU` legs
of non-refiners stayed imputable (#96).
- **Production** is the bound's own: raw ICSG rows by ADR 0027's definitions
  (`build_trade_shares.icsg_raw_production`), now also for SX-EW cathode (the SXEW/Cathode supply).
- **Covered:** a region any ICSG row maps to, in any year, except ROW (a sum of members), a region
  no row maps to (Luxembourg), and a region-year ICSG reports inside its own aggregate (#67).
- **Stamped:** `MF→Concentrate` and `SXEW→CSXEW` in CC, the edges the feed books ICSG's mine and
  SX-EW supply on, at σ = 0.3 kt CC (#76's rule with no fan usage: ICSG's reporting resolution,
  ADR 0025). The hard efficiency bands and mass balances carry the zero to Ore, Mining, CO,
  the tailings and CSXEW→CP, so none of those is stamped twice. `OBSERVED_ZERO_SECTORS` adds Cathode.
- **Not stamped where another input reports the output.** That is crude ore > 0 in the crude-ore
  feed, or an ore-grade row starting at CO, for the mine; an ore-grade row starting at SXEW, for
  SX-EW. There ICSG has folded a small producer into "Others": its missing row is missing data,
  and a zero would contradict the other input (ADR 0019).

Dry count on icsg v16, trade_shares v38, crude_ore v3 and ore_grade v8: 1,429 output zeros and 253
cathode legs (France, Greece, the Netherlands, Switzerland and Taiwan 30 each).
- `MF→Concentrate`: 456 region-years in 20 regions. The guard spares the 60 region-years where the
  crude-ore feed reports ore: Slovakia 28, Korea 12, Germany 2015–22, Malaysia 1995–99 (1.9–3.9 Mt
  of ore a year), France 1995–97, Japan 2005–06 and Spain 2002–03. Their ore-grade rows add none.
- `SXEW→CSXEW`: 973 region-years in 38 regions. 457 of them are in region-years with ICSG mine
  output but no ICSG SX-EW output, so the zero there reads "this mine has no SX-EW plant"; the REP
  reports them apart from the 516 with neither. No ore-grade SX-EW row falls in any of the 973.
- The 82 concentrate legs are those of `feed_icsg_20261021_095747_f99c73a2`, leg for leg. 24 of
  them sit in region-years the guard spares, and they stay stamped as #78 shipped them.

To be measured. The solver must now settle in the open what a phantom mine supplied: 451 kt CC of
concentrate export priors leave the stamped regions' Concentrate nodes (the UAE 132, Switzerland
59), and 556 kt CC of smelter-recipe concentrate intake sits on their domestic legs (Belgium 371,
Italy 160).

**Measured 2026-10-21: SX-EW zeros ship; mine-output and cathode-leg zeros are rejected.** REP
[`…issue97-observed-zero-output`](../rep/analysis_opt_20261021_111957_5c6d13bc_20261021_issue97-observed-zero-output.md).
The full build (`opt_20261021_105350_b1fea245`, against `opt_20261021_101810_d34c6527`) cut the
phantom SX-EW but raised common-set sum-row z² **+37.8 %**: the phantom chains had been filling
gaps between ICSG usage and recorded supply — Matte use totals +170,357 (Italy +115,164), Cathode
+149,730 (the Netherlands, Switzerland, the UAE, Malaysia), Concentrate +36,100 — and world Cathode
CC retention fell 1.001 → 0.990. The SX-EW zeros alone (`PRODUCTION_EDGES = (("SXEW", "CSXEW"),)`,
`OBSERVED_ZERO_SECTORS = ("Concentrate",)` as #78 shipped; `opt_20261021_111957_5c6d13bc`) cut
Ore→SXEW where neither mine nor SX-EW is reported 1,344 → 33 kt CC and in mining region-years
without ICSG SX-EW 1,911 → 42, at common-set z² **+1.10 %**, Round-E holding; they ship over 95
stock legs newly above |z| 5 (65 CC, 60 of them Russia's, −102 kt CC over 30 years), where
Russia's lost phantom SX-EW now draws on stocks. The phantom mine chains of non-mining regions
remain (CO→Mining 2,028 kt CC posterior over 1995–2024, part re-routed through concentrate) and so
do #96's cathode legs: closing them is a data question (ICSG usage above BACI imports plus ICSG
production in hubs), stated as a limitation with the +37.8 % above as its measured cost.
