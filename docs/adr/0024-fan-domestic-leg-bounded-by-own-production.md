# The fan's domestic leg is bounded by the region's own production

**Status:** proposed — measured in two variants (Round J), merge decision pending. Amends
[ADR 0002](0002-single-endpoint-rows-fan-out-not-replicate.md) option D and
[ADR 0018](0018-import-free-fans-distribute-domestically.md). Evidence: REP
[`…india-concentrate-residual-domestic`](../rep/analysis_20260910_india-concentrate-residual-domestic.md).

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
and the interpolation of bilateral legs as the next two items. Decision pending the owner.
