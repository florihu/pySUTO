# Round J — the fan's domestic leg bounded by own production (ADR 0024), two variants

- **J.1** (recommended): `opt_20260910_133212_45f034f1` ← `const_20260910_131906_b13e8f95` ←
  `imp_20260910_131346_e055415a` ← `feed_icsg_20260910_131054_1fffbce3` on trade_shares **v12**
- **J**: `opt_20260910_125244_4d50b04a` ← `const_20260910_123954_ddbf31cf` ← `imp_20260910_123440_88d6cd1c`
  ← `feed_icsg_20260910_123148_a4f334c6` on trade_shares **v11**
- baseline: `opt_20260910_114200_7f4040d9` (shipped Round I); baci `20260902_085153_de9e393f`,
  crude_ore `20260603_095651_558f7369`, ore_grade v4, conc v6, eff v3 unchanged
- reproducers: `python scripts/explo/roundJ_fan_bound_diag.py <base> <new>`;
  `python scripts/explo/india_concentrate_diag.py <opt>` for the class
- branch `fix/fan-domestic-bounded-by-production`, **J.1 merged to main 2026-09-14**. ADR [0024](../adr/0024-fan-domestic-leg-bounded-by-own-production.md);
  diagnosis REP [`…india-concentrate-residual-domestic`](analysis_20260910_india-concentrate-residual-domestic.md).

## 1. Change

`build_trade_shares.compute_trade_share_weights` bounds the ADR 0002 domestic residual by the
consumer region's own ICSG production of the product (Concentrate only, `BOUND_DOMESTIC_SECTORS`)
and spreads the remainder of the usage over the import legs in BACI proportion, usage conserved.
J bounded every group with ≥ 1 recorded import; J.1 only where `Σ imp_cc ≥ residual − production`
(legs at most doubled). Import-free fans keep the ADR 0018 form in both.

Builder: J 265 legs bounded / 27,069 kt CC moved; J.1 197 / 24,194 — the 68 legs and 2.9 Mt left
alone by J.1 are the self-sufficient kind (US 1997–2003, Poland, Uzbekistan, Congo…).

## 2. Measured

| sum-row z² | base | J | J.1 |
|---|---|---|---|
| total, 30 y | 2,893,508 | 2,760,269 (−4.6 %) | **2,745,311 (−5.1 %)** |
| all soft | 3,032,263 | 2,903,134 | 2,887,576 |
| `usetotout_Concentrate_India` | 112,955 | **12,353** | 23,261 |
| 210 excess region-years (fixed population) | 348,126 | 155,930 | 178,148 |
| other `usetotout_Concentrate` rows | 285,061 | 322,462 | 314,792 |
| Argentina Smelting | 61,933 | 33,536 | 33,536 |
| United States Mining + Ore | 17,634 | 26,641 | 23,327 |
| Congo D.R., all | 108,666 | 132,874 | 110,698 |
| Korea Concentrate | 13,779 | 27,174 | 27,174 |
| Papua New Guinea Anode | 737 | 13,148 | 13,148 |
| Mongolia Concentrate | 3,055 | 7,485 | 6,682 |
| Σ increases / on anchors < 10 kt | — | 192,430 / 46,146 | 144,178 / 35,614 |
| Σ decreases | — | −325,669 | −292,375 |
| contradicted Matte node-years (ADR 0019) | 9 | 5 | 5 |

By sector (J.1 − base): Concentrate −140k, Smelting −35k, Ore −4k, MF −3k, CFR −3k; Anode +20k,
Cathode +14k, Mining +7k. By region: India −90k, Argentina Smelting −28k, Spain −25k / −10k / −10k
(Concentrate / Ore / MF), Philippines −20k, Japan −19k, Brazil −10k; Korea +13k, PNG Anode +12k,
Indonesia Cathode +10k, South Africa Anode +4k, Mongolia +4k, China MF +3k, US Ore/Mining +3k each.

Prior-level check on the imp run: Concentrate domestic-above-production region-years 210 → 10
(J; import-free fans only), excess 21.7 Mt → 145 kt. India 2015 feed F_root: domestic 417 → 30,
imports 404 → 791, total 821 unchanged.

Exporters (posterior cross-region concentrate CC, 2010–19): world 65,714 → 66,702 (+1.5 %); Chile
+0.2 %, Peru +0.1 %, Indonesia +1.6 %, Australia +2.0 %, Brazil +3.6 %, Canada +1.9 %, Mexico +0.8 %,
Kazakhstan +2.1 %. No exporter loses volume; their retention ratios fall only because the priors on
their legs grew (Chile's 2000 anchor 2,042 → 2,274).

India 2015, CC: domestic 417 → 27 (base) vs 30 → 27; imports 414 → 657 vs 801 → 691; smelter feed
posterior 685 → 718 against ICSG 821.

## 3. Side effects, traced

**United States 1998–2003.** Mine output posterior/anchor: 2000 0.95 (base) → 0.66 (J) → 0.71
(J.1); 2002 0.89 → 0.72 → 0.87; 2003 0.97 → 0.83 → 0.89. The US fan is byte-identical in 2000
across v10/v11/v12 (residual 840 < production 910). Under J the US was bounded in 1998–99 and
2001–03 (imports 26–77 kt asked to carry 75–223 kt) and `time_interp` carried the inflated legs
into the uncovered 2000 legs (Indonesia→US 7.1 → 33.6, Peru→US 5.3 → 22.3 kt, σ ≈ 2); with those
tight phantom imports the QP cut US mine output and fed the smelter more scrap (148 → 234). J.1's
guard leaves 1998–99 unbounded; 2002 remains bounded at the guard's edge (76.5 vs 74.7) and the
interpolation window still reaches 2000, hence the residual. The interpolation of *bilateral* legs
from neighbouring years is pre-existing (base already carried 7.1 and 5.3 there) and is the real
follow-up: ADR 0010 treats an unrecorded route as "no trade on this route".

**India 2014 / 2016 under J.1.** z −11 / −20 (J) → −55 / −54 (J.1): the imports would have to grow
2.1–2.2× to carry the remainder, so the guard leaves the residual on the domestic leg. India's
imports are 45 % of its usage, so "BACI is short" is plausible there in a way it is not for the US
(3 %); the guard cannot see that. A share-based guard would be a second threshold.

**Tiny anchors.** Korea (27k) and PNG Anode (13k) are unchanged between J and J.1: fabricated
chains removed (Korea's network-back-filled 180 kt/yr of "mining"), leftover 3–6 kt anchors priced
at σ ≈ 0.06. Not physical; the σ-floor question of ADR 0016/0017.

**Congo.** +24k under J (self-sufficient kind: Congo's imports of Zambian concentrate asked to
carry the gap), gone under J.1.

## 4. Stop-rule check

Mass moves with the metric — 24–27 Mt of prior moved off fabricated domestic legs, India's feed
+5 % posterior, world concentrate trade +1.5 %; no exporter regresses in volume; no constant was
tuned to the metric (the guard was designed from the US trace, and it *costs* India). Both variants
pass. The one physical regression (US 2000) is traced to a pre-existing mechanism and halved.

## 5. Decision

**J.1 shipped (2026-09-14).** The alternatives considered: (a) J.1 as is; (b) J (India fully fixed, US/Congo side
effects); (c) J.1 plus a share-based guard for India-class importers; (d) either, plus excluding
bilateral legs from `time_interp` (removes the US mechanism at the root — a separate ADR).
Follow-ups regardless: σ floor on tiny scale totals; Matte's 33 region-years / 1.3 Mt.
