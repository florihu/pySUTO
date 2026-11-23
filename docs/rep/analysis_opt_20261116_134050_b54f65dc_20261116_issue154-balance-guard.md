# #154: ADR 0010's guard held for one cascade pass only; closing it removes 1,093 kt CC of phantom mine output and costs +5.95 % sum-row Σz²

**Run:** `python -m pysuto run --case copper --from imp --to opt --base opt_20261112_135523_88fa69f3 --label "#153 balance
guard"`, then `--from uncprop --to psut --base opt_20261116_134050_b54f65dc` → `imp_20261116_133053_ad1191d2` →
`const_20261116_133444_28ef32f2` → `opt_20261116_134050_b54f65dc` (30/30 solved, cold, `solve_wall_s` 93) →
`uncprop_20261116_141543_6be9fa25`, `psut_20261116_141702_37174685` (label `#153 balance guard`, every knob at its default)
**Against:** the reference chain `imp_20261112_133434_22fc54b3` → `const_20261112_134032_b6802ef3` →
`opt_20261112_135523_88fa69f3` (*base* below); the assumed-absent set is read from `imp_20261112_154659_f4ec08ae`, which
is bit-identical to the base imp and carries #150's `relaxed` / `assumed_absent` tags.
**Chain:** the base's feeds `feed_icsg_20261022_110826_8a031806`, `feed_baci_20261022_110715_fdf3b4ac`,
`feed_crude_ore_20260603_095651_558f7369`, resolved by `--base`. Code main `de8a592` (#153: `fbef5d0`, `170aced`), run
from the main checkout, one driver at a time under the stage lock.
**Feeds/params:** acc copper v5 (`54bc4323`); conc v10, eff v4, ore_grade v8, stripping_ratio v2, stock_bounds v6,
scrap_share v3 — the base's; `input_status.py`: every `current/` at its latest version
**Date:** 2026-11-16 · **Branch:** `diag/balance-guard-measure` · **Issues:** #154 (measures #153; follow-up of #132,
#139) · **ADR:** 0010 (the guard), 0025 (stop rule)
**Diagnostic:** `python scripts/explo/issue154_balance_guard_diag.py opt_20261112_135523_88fa69f3 opt_20261116_134050_b54f65dc` (<opt_base> <opt_new>)

**Verdict: ships on the pre-registered rule (items 2 and 3a–c hold); `opt_20261116_134050_b54f65dc` becomes the reference
chain.** Predictions P2 and P3 were wrong, and P1 partly. The common-set sum-row Σz² *rises* 5.95 % vs
`opt_20261112_135523_88fa69f3`. The rise is the fit the base had bought with phantom mine chains. The base's second cascade
pass re-derived the primaries the first pass's relaxed closes had counted absent, as explicit zeros. From those zeros
the network then built mine → concentrator → smelter → converter chains: 1,093 kt CC of CO→Mining over 185
region-years in 13 regions. Seven of them have no mine prior in any year (France, Greece, Netherlands, Singapore,
Switzerland, Taiwan, United Arab Emirates: 892.5 kt CC). The chains supplied anode and concentrate exports that
BACI books from re-export hubs (Netherlands → Belgium anode 45.6 kt CC in 2006). Without the chains those export anchors
are violated: a data contradiction the prior had been hiding, which σ cannot repair (ADR 0019, 0035).

## 1. Change under test

- **#153 (`fbef5d0`, review fix `170aced`)**:
  - A Flow balance no longer fills a primary that a relaxed close counted as 0. The primaries are recorded when the
    close's write lands.
  - Both guard sets (`_relaxed_fills`, `_assumed_absent`) are reset in `prepare()`, not in every `impute()`, so they hold
    across the engine's cascade passes.
  - The second point widened the ticket, by the maintainer's decision of 2026-11-16. On the base, every one of the 359
    assumed-absent primaries was re-derived:
    - 23 by a balance in pass 1 (13 nonzero, 98.6 kt CC). #132's "46 / 197 kt" counted each close's product fill and
      waste fill, so it is doubled.
    - 336 by `network_backward` in pass 2, as an explicit 0. Its σ has median 0.51 kt and maximum 88 kt; 4 capped fills
      have σ ≈ 0.

## 2. Decision items (pre-registered on #154 before the first driver)

| # | Rule | Result | |
|---|---|---|---|
| 1a | the 359 primaries: left NaN by the network, and who fills them, reported | the network writes none of them; 352 go to the flat prior (0 at σ 10⁶) and 7 to `time_interp` (32.97 kt CC, nonzero) | — |
| 1b | the edges whose `f_agg` / `σ_agg` move, reported | `f_agg` moves on 3,274 edges (24 primaries, their 24 Mass twins, 3,226 others). In CC: 2,007 edges, 1,915 of them to 0, holding 6,676 kt CC in the base. `σ_agg` moves on 8,376 edges. Coverage is bit-identical | — |
| 1c | network log rows per pass, reported | pass 1: 45,360 → 45,314; pass 2: 8,536 → 297 | — |
| 2 | every year Solved | 30/30 solved | holds |
| 3a | world retention falls < 0.01 in every category | CO Mass 0.999 → 0.999, Concentrate CC 1.010 → 1.011, Cathode CC 0.998 → 0.997 | holds |
| 3b | small nodes (\|c\| < 18 kt) median post/prior within 0.01 | median 1.000 → 1.000; p90 1.015 → 1.125 | holds |
| 3c | Chile / Peru / Indonesia exporter retention falls < 0.01 or rises | 1.022 / 0.978 / 0.917, equal (world 0.947) | holds |
| 3d | mass moved and z² by anchor band, reported | mass moved 256,214 → 259,241 kt CC (+1.18 % vs base); band Δz²: < 10 kt +12,201, 10–100 kt +31,431, 100–1,000 kt +178, ≥ 1,000 kt +6 | — |
| 4 | common-set Σz² by family and region, reported | sum rows 735,986 → 779,802 (+5.953 % vs `opt_20261112_135523_88fa69f3`, 27,376 rows); ratio families move by < 14 each (`concanodesupply` +13.7, +1.5 %) | — |

**Predictions**
- **P1** (all 359 go to the flat prior): **partly wrong**. 352 do; `time_interp` fills 7, nonzero, carrying a neighbouring
  year's derived route: Italy Matte→CFR 2012/2014, CER→CP in Botswana 2019/20, Italy 2012/14 and Slovakia 2000. On those 7
  the prior again holds both the close's 0 and a value, now from another imputer (#139's exposure, 7 edges, 33 kt CC).
- **P2** (`f_agg` moves only on the 23 balance legs and what lies downstream of them): **wrong**. The pass-2 zeros were
  not leaves: the network derived whole upstream chains from them (§3).
- **P3** (Σz² falls, by < 1 %): **wrong in sign**, +5.95 %.
- **P4** (30/30 solved, 3a–c hold): **right**.

## 3. What moved

- **The pass-2 chains.**
  - In the base, pass 2 wrote 8,536 network rows.
  - Its nonzero CC fills by sector pair (in kt CC):

    | Fill | kt CC | Region-years |
    |---|---|---|
    | Mining→Ore | 2,341 | 208 |
    | Ore→MF | 1,204 | |
    | CO→Mining | 1,093 | 185 |
    | MF→Concentrate | 963 | |
    | Smelting→Matte | 727 | |
    | CFR→Anode | 489 | |
    | Matte→CFR | 484 | |
    | Concentrate→Smelting | 474 | |

  - The CO→Mining fills cover 185 region-years in 13 regions:
    - Seven have no mine prior in any year of the new run. They hold 892.5 kt CC: United Arab Emirates 242.4,
      Netherlands 201.0, Taiwan 189.6, Singapore 96.8, Switzerland 93.0, France 62.7, Greece 7.0.
    - The other six are miners whose gap years pass 2 filled: Malaysia 95.8, United Kingdom 76.7, Panama 10.6,
      Slovakia 8.3, Botswana 5.1, Italy 3.9.
  - The mechanism, in Netherlands 2006:
    - BACI books 45.6 kt CC of anode from Netherlands to Belgium.
    - The Anode balance then derives CFR→Anode = 45.9 kt.
    - With the converter's and smelter's absent-route zeros now explicit, the backward fills walk the chain up to a
      Netherlands mine at 52.4 kt CC.
  - The base posterior keeps the phantom chain: CO→Mining 52.41 kt CC, and the export at 45.47 kt.
  - In the new run the prior has no Netherlands chain, the export leg falls to 2.68 kt and the solver opens 0.62 kt of
    mine output.
- **Mine output by region, Σ 1995–2024 (kt CC, CO→Mining posterior, base → new):**

  | Region | Base | New |
  |---|---|---|
  | Netherlands | 204.3 | 42.5 |
  | Taiwan | 194.1 | 53.1 |
  | United Arab Emirates | 248.7 | 158.1 |
  | Malaysia | 210.3 | 142.2 |
  | Singapore | 101.8 | 61.4 |
  | United Kingdom | 105.2 | 69.6 |
  | France | 66.1 | 30.7 |
  | Switzerland | 104.0 | 78.0 |

  - World posterior mine output is 640,015 → 639,488 kt CC (−527).
  - The real miners barely move: Chile +13.8, Botswana +11.0, Panama +8.2, Italy +14.0.
  - Even the new posterior opens 31–158 kt CC of mine output in each of these hubs. That output comes from a flat prior
    (0 at σ 10⁶) where the region has no mine: no observed zero holds it.
- **Where Σz² rises.**
  - The rise is concentrated in six regions: Netherlands +17,930, Slovakia +6,926, Malaysia +5,695, Taiwan +3,892,
    Botswana +2,842 and United Arab Emirates +2,121. Mostly these are Anode and Concentrate anchors.
  - The largest single rows are `usetotout Anode·Netherlands` CC/Mass 2006 and 2010 (z² 0 → ≈ 2,900–3,000 each) and
    Malaysia's Anode use 2020–23. These are re-export legs that the phantom chain had made look domestic.
  - The lowest band, < 10 kt (+12,201), is led by Botswana's 2019 MF supply total and by Slovakia's concentrate and MF
    rows in 2021. Botswana's prior MF→Concentrate moves 0.19 → 18.64 kt Mass, and the prior-derived scale total `c`
    moves with it (0.2 → 18.6).
- **Stage 3–4.**
  - Every year solved.
  - Retention and the exporters are equal to three decimals.
  - The small-node p90 widens from 1.015 to 1.125: the hubs' small mining and smelting nodes, now flat-prior, are
    opened by the solver.

## 4. Follow-ups

- **Hub mining has no observed zero.** A region with no mine output in ICSG still carries a free CO→Mining edge: flat
  prior, σ 10⁶. The phantom mines here first lived in the prior; now they live in the posterior, at a fifth to two thirds
  of their former size.
- **The 7 `time_interp` fills.** These are assumed-absent primaries filled nonzero from a neighbouring year, so #139's
  exposure is no longer zero.
- **The re-export anchors** (Netherlands, Malaysia, Taiwan, UAE anode/concentrate exports with no domestic production)
  are the data contradiction this change exposes. They belong to the netting (ADR 0026/0027), not to σ.
- **Manuscript re-point (like #152).** The common-set Σz² and every number on this chain move.
