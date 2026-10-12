# #65: the ICSG feed rebuilt from the repaired stock parse (#35)

**Run:** `opt_20261012_135741_67851aa1` (label `r65_icsg_stock_parse`; 30/30 Solved; `prior_mode` cold)
**Against:** #40.1 `opt_20261012_122124_40b1b370` (30/30 Solved; REP
[`…issue40-1-supply-mix-fans`](analysis_opt_20261012_122124_40b1b370_20261012_issue40-1-supply-mix-fans.md), ADR 0043)
**Chain:** icsg **v13** → baci **v16** (ADR 0027 netting of baci v13 re-run against icsg v13) →
`feed_baci_20261012_130305_1cdad15b`; trade_shares **v28** → `feed_icsg_20261012_130637_fa562177`;
\+ `feed_crude_ore_20260603_095651_558f7369` → `imp_20261012_131053_5d4fa75e` →
`const_20261012_131519_a86e9d3d` → the run
**Feeds/params:** icsg **v13**, baci **v16** (`weights.csv` hash = v15's, cathode 100/0/0), trade_shares
**v28**, acc copper v4, conc v10, eff v4, ore_grade v7, stock_bounds v5, stripping_ratio v2. These are
the base's versions except the three in bold.
**Date:** 2026-10-12 · **Branch:** `diag/icsg-stock-parse-measure` · **Issues:** #65, #35 · **ADR:**
0024, 0025, 0027, 0040, 0043 (none amended)

**Verdict: passes. Every item of the pass rule fixed on #65 holds, and the round is accepted.**

- **Stock series:** MESC now covers 29 years and USC and PSC 28 each.
- **U.S. producer stocks:** the 2005–2012 changes stay within ±19.9 kt.
- **Round-E rule:** it holds, and mass moved falls by 5,945 kt CC.
- **Stock legs:** both signs survive and no leg grows into a sink.
- **Stable sort:** not one row outside the stock families moves.
- **One prediction failed.** Common-set sum-row z² **rises, +0.16 % vs `opt_20261012_122124_40b1b370`**, where
  I predicted a fall on the U.S. cathode rows. Those rows fall, but only by 337 z². The base had already
  deleted the U.S. phantom almost for free, at the cost of 6,080 kt CC of gross PSC adjustment, which
  is now 207 kt.
- **Where the rise sits:** in Sweden (+2,636), whose cathode trade the ADR 0027 netting now nets in
  2003 as well, on a lone MESC row. Singapore falls by 2,924.

## 1. Change

The parse repair of #35 (`fec80ef`, `93e206b`), run from the raw yearbooks. The regenerated files are
byte-identical to those of 2026-10-08.

- **2014 yearbook:** its exchange block parses, so LME, COMEX and SHFE levels for 2004–2013 are in.
- **Net additions to stock:** each vintage differences its own consecutive years, and the newest
  vintage wins.
- **USC** is the sum of consumers, merchants and government.
- **U.S. producers row:** in the 2014 yearbook it is derived as U.S. Total − U.S. Consumers.
- **Lone totals:** a region that prints only a total counts as producer stock (PSC).
  - **Basis:** ICSG's own summary tables. The 2005 and 2014 yearbooks add every lone total to
    "Producers", and the 2023 and 2025 yearbooks still do for the estimated totals (China, Germany,
    Korea, the Philippines, Bulgaria).
  - **Japan since 2021** is the one total ICSG leaves unallocated ("detail by type not specified").
    Producers held 82–87 % of Japan's stocks in 2016–2020, its last years with types.
  - **Share:** lone totals carry **8.2 %** of all producer-stock changes on the regenerated parse
    (Σ|·|): China 189.2 kt over 28 years, Japan 37.2 kt over 3, Korea 13.0 kt and the Philippines
    5.6 kt. Germany's and Bulgaria's estimated totals are constant, so they book no change.
  - **Japan 2021:** the year its total takes over from its types books no change.

**Rebuilt, in the ADR 0024/0027 order:** icsg v12 → **v13** → `build_baci_reexport.py --source-version 13`
→ baci **v16** → `baci_feed.py` → `build_trade_shares.py` → `icsg_feed.py`. The trade shares needed a
second pass (§6). stock_bounds v5 and ore_grade v7 are unchanged; regenerating them is #37 and #30.

**baci v15 → v16.** v16 netted 10,833 kt CC of re-exports, against 10,473 in v15 (+359 kt).

- **Which hub-years:** 24 of them move from `icsg-gap` to netted, and two netted ones change their
  amount.
  - Singapore: 11 hub-years, +294.3 kt.
  - Sweden 2003: +47.9 kt.
  - UAE: 10 hub-years, +10.8 kt.
  - Slovakia: +3.1 kt.
- **Why:** each of these region-years now has an ICSG intake U, made only of the newly recovered MESC
  or USC rows.
- **Unchanged files:** `weights.csv`, `conc.xlsx`, `meta.xlsx`, `quantity_cv_scale.csv` and
  `quantity_recon.csv` hash as in v15.

## 2. Pass rule

| Item (fixed on #65 before the numbers) | Result |
|---|---|
| MESC 1996–2024 (29 y); USC and PSC 1996–2012 + 2014–2024 (28 y each), 2013 absent; MESC and USC carry both signs | **holds.** MESC: 426 rows over 29 years, 164 negative and 150 positive. USC: 173 rows over 28 years, 53 negative and 55 positive. PSC: 524 rows over 28 years. v12 had 225 rows over 18 years (MESC), 160 over 25 (USC) and 421 over 26 (PSC). |
| United States PSC \|Δ\| < 100 kt in 2005–2012 | **holds.** The feed is −2.2 … +19.9 kt, against −1,313 … +2,000 kt in v12. The posterior follows within 0.3 kt (2006: 19.9 → 19.6). |
| Round-E stop rule vs `opt_20261012_122124_40b1b370` (ADR 0025) | **holds** on (a) world retention, (b) small nodes and (c) exporters. On (d), z² falls in the bands under 100 kt and rises above, while mass moved falls 1.2 %. See the table below. |
| MESC and USC posteriors show both signs; Σ\|a* − a_prior\| on MESC, USC and PSC vs the base; none grows into a sink (read net per year) | **holds.** Both signs survive, and gross adjustment falls on all three families. The net is unchanged in 1995, the #37 sink; its other moves follow larger, correct priors (§4). |
| Rows outside the stock families that move because of the stable sort | **none.** Not one row outside MESC/USC/PSC differs between icsg v12 and v13. Round S's 21 flips were all on colliding USC keys, which the summed USC replaces outright. |

**Round-E, `opt_20261012_122124_40b1b370` → `opt_20261012_135741_67851aa1`:**

| Item | Base | New |
|---|---|---|
| (a) world retention, CO Mass / Concentrate CC / Cathode CC | 0.863 / 0.991 / 0.987 | 0.862 / 0.991 / 0.986 |
| (b) small nodes, posterior/prior median / p90 | 1.000 / 1.155 | 1.000 / 1.165 |
| (c) exporter retention, Chile / Peru / Indonesia / world | 0.955 / 1.009 / 0.910 / 0.937 | 0.957 / 1.009 / 0.910 / 0.937 |
| (d) Δz² by anchor band \|c\| < 10 / 10–100 / 100–1,000 / > 1,000 kt | — | −73 / −1,500 / +1,658 / +2,158 |
| (d) mass moved, all CC edges | 488,714 kt | **482,769 kt** (−1.2 % vs base) |

## 3. Cost and where it sits

**Common-set sum-row z²**, 26,996 rows in both runs and built in memory with no per-year CSV: 1,404,040 →
**1,406,284 (+0.16 % vs `opt_20261012_122124_40b1b370`)**.

Per year the change is under ±0.5 % except:

| Year | 1998 | 1999 | 2000 | 2003 | 2004 | 2005 | 2006 | 2007 | 2008 | 2017 |
|---|---|---|---|---|---|---|---|---|---|---|
| Δz² vs base | +0.86 % | +1.11 % | +1.30 % | +2.08 % | +1.45 % | −0.74 % | −4.37 % | −3.88 % | −1.35 % | +1.08 % |

By region:

| Region | Δz² | Where |
|---|---|---|
| Singapore | **−2,924** | 2006 −1,613, 2007 −879. The netting removes its pass-through, and its cathode use total's anchor falls from 75.5 to 0.3 kt (Mass, 2006). |
| Sweden | **+2,636** | 2003 +1,083; 1996–2000 +1,552 |
| China | +459 | 2017 +333, 2012 +140 |
| Russia | +410 | 2001–2003 +357 (CP 2003) |
| Zambia | +337 | spread |
| Kazakhstan | +307 | spread |
| United States | +244 | 2003–2004 +487, 2005–2012 −337 |

- **Sweden 2003 is a netting effect.** Until 2003 ICSG reports Sweden's smelting, refining and semis
  under "Scandinavia", so Sweden's only own row in 2003 is the newly recovered LME change of −19.3 kt.
  - v15 flagged that hub-year `icsg-gap` and left it alone. v16 reads U = −19.3 kt, P missing, and
    nets 47.9 kt of Sweden's cathode imports against its exports: ROW → Sweden FU 40.7 → 0, Sweden →
    United Kingdom 34.6 → 14.3 kt CC.
  - Sweden's mine output then has less outlet, and its posterior falls from 68.3 to 40.4 kt CC against
    a prior of 98.7. Mining z² rises 1,409 (CC + Mass).
  - The same mechanism already netted Sweden in 2001 and 2002 in v15 (49.1 and 36.4 kt), through MESC
    rows v12 had. #35 recovers the 2003 row and so extends it by one year (§7).
- **Sweden 1996–2000 is not an input change.** No prior touching Sweden moves in those years. Across
  all edges, Σ|Δ prior| is 15–92 kt CC per year in 1996–1999, almost all of it on MESC, USC and PSC
  legs elsewhere (1998: 15 kt). Sweden's mine chain in those years has zero priors downstream of the
  mine, and that is where the changed QP rebalances. The path from the moved priors to Sweden is not
  traced here.
- **The moves are real optima, not solver noise.** Each run's posterior was scored under the new QP
  (ADR 0039, `--crosseval`). The base posterior scores worse than the new run's own answer by
  **+7,626 J** in 1999 and +97k to +5.7M J in 1996, 1998, 2000, 2003, 2004 and 2006. That is far above
  the ≥ 684 J that a 'Solved' year can sit from the optimum at ADR 0039's tolerances (WR-prior REP,
  `opt_20261007_130833_1a9b7af7`). The reverse direction is not
  informative: the new posterior puts flow on stock legs the base QP pins at zero, so its J explodes.

## 4. Stock legs

Gross adjustment Σ|a* − a_prior|, 1995–2024, kt CC:

| | MESC | USC | PSC |
|---|---|---|---|
| base | 2,624.5 | 2,883.9 | **6,080.2** |
| new | 2,178.4 (−17 %) | 2,828.7 (−1.9 %) | **207.3** (−96.6 %) |

- **PSC.** The base deleted the U.S. phantom: in 2006 the prior was +1,999.9 kt and the posterior
  +36.6, and in 2010 the prior was −1,313.4 and the posterior −74.6. This round's PSC net per year is
  at most 38.2 kt (2016). Its base peak was 1,953.6 kt (2006).
- **MESC + USC, net per year** (Σ a* − Σ a_prior, kt CC):
  - 1995: −163.5 in the base, −162.8 now. This is the #37 sink, unchanged as predicted.
  - All other years stay within ±115 kt. The base reached ±91 kt.
  - The two years that exceed the base are 2001 (−115.4 vs −90.8) and 2004 (+92.9 vs +30.2).
- **Those two priors were wrong in the base.** In 2004 world exchange stocks fell from 806.0 to
  124.3 kt, a change of −681.7 kt on ICSG's levels. v13 carries −681.7 kt; v12 had −357.1 kt, because
  it kept one exchange series per location. The U.S. row alone (COMEX plus the LME's U.S. warehouses)
  goes −211.1 → −511.3 kt in 2004 and +185.1 → +597.2 kt in 2001.
- **Relative to Σ|prior|, the net shrinks.** In 2001 it is −0.31 in the base and −0.16 now; in 2003
  −0.20 and +0.01. Only 2004 rises, from 0.08 to 0.13. A sink would draw on zero-prior legs, as 1995
  does; these are proportionate adjustments to larger, correct priors.
- **Signs.** Posterior edge-years by sign, CC (negative / positive): MESC 27,806 / 38,464 → 27,159 /
  39,111; USC 27,867 / 38,403 → 27,346 / 38,924.
- **The stock legs now add up to ICSG's own total.** The feed's stock changes, all types over
  1996–2024, sum to **+539.9 kt** in v13 against −877.2 kt in v12. ICSG's summary tables put the rise
  of world refined stocks over the same years at +456 kt (#40.1 REP §3). The gap includes the 2013
  PMC change, which no vintage holds, and the years in which a series goes blank or a total takes
  over from its types; both book no change.
  - Priors in the run, kt CC: MESC +127.7, USC −23.8, PSC +451.8.
  - Posteriors: MESC + USC −40.7, PSC +544.0.
  - The base's priors summed to −788 kt.
- **Singapore** (prediction 5). Its cathode legs from partners, FU and stock legs together, fall in
  prior by 13–70 kt CC a year in 2005–2012: 2006 108 → 52, 2007 56 → −14, 2010 60 → 29 kt CC.
  - The FU legs fall by the netting: 2006 107.7 → 32.5, 2007 56.1 → 4.8 kt CC.
  - The new MESC legs carry ICSG's change, drawn from partners' supply: 2006 +19.5, 2007 −18.9,
    2010 −14.2 kt CC.
  - The posteriors follow within 6.5 kt.

## 5. #35 cross-checked

1. **2014 exchange block → 0 rows.** Fixed. The 2014 vintage now carries exchange levels for
   2004–2013, and MESC covers 29 years (v12: 18). #35 expected 28; the 2025 yearbook adds 2024.
2. **An edition boundary loses a NAS year.** Fixed.
   - 2004 and 2015 now have changes for every family.
   - PMC 2013 stays absent because no vintage holds both of its levels: the 2014 yearbook ends at
     2012 and the 2023 yearbook starts at 2013.
   - A blank year books no change for itself or for the year after it.
3. **USC types collide.** Fixed. USC is the sum of the three types per vintage, and every sort before
   a dedup is stable. USC: 160 → 173 rows over 25 → 28 years (#35 expected +2 years; 2024 is the
   third).
4. **U.S. producer stocks ±2 Mt/yr.** Fixed. The 2005–2012 changes are −2.2 … +19.9 kt, and the
   posterior follows them within 0.3 kt.

## 6. Build order: the trade shares need a second pass after an ICSG change

`build_trade_shares.py` takes a fan group's usage from the latest ICSG **feed run**'s `F_root` total.
It falls back to the raw ICSG rows only where that total is 0. After an ICSG change, the documented
order (trade shares, then the ICSG feed) therefore builds the weights on the previous ICSG data.

| Pass | Built on | Result |
|---|---|---|
| v27 | v12's feed run (`feed_icsg_20261012_113121_e7271610`) | 47 stale weights for one group: Korea, Cathode → USC, 2007. v12 had a change there, v13 has none. |
| v28 | `feed_icsg_20261012_130505_196d3b63`, built from v27 | drops the 47 weights |
| v29 | `feed_icsg_20261012_130637_fa562177`, built from v28 | equals v28 to 3e-12 relative (947 weights differ in summation order only). v29 is registered and unused. |

- **No effect here.** The feed runs built from v27 and v28 have identical coverage (35,728 covered
  edges) and differ by at most 1.8e-12 in `F_root`. The extra pass changes nothing in this
  measurement.
- **Why so little:** no non-stock row changed, and a stock leg's supply-mix weight reads only the sign
  of its usage (ADR 0043).
- **Where it would bite:** an ICSG change to a flow row would size the domestic weights on the old
  total.

## 7. Follow-ups (not part of #65)

1. **A stock-only row lifts a reporting gap out of the ADR 0027 exclusion.**
   - The netting's `icsg_gap` test asks for "neither U nor P", and U includes the MESC and USC rows.
     Sweden, whose flows sit under "Scandinavia" until 2003, is netted on a lone LME change in 2001,
     2002 and now 2003.
   - Singapore is the other face of the same rule. It reports semis only in 1995–2001, and netting
     its 2005–2015 hub trade on its MESC rows lowers z² by 2,924.
   - Telling the two apart is a decision for the re-export builder, with its own measurement.
2. **The trade-share build order (§6).** Either `build_trade_shares.py` takes usage from the raw ICSG
   rows everywhere, or the documented order becomes ICSG feed → trade shares → ICSG feed after any
   ICSG change. The docstring of `compute_trade_share_weights` already states that the raw usage
   reproduces `F_root` to 3e-16 where a fan is distributed.
3. **#37 and #30 are unblocked:** regenerate stock_bounds (plus the 1995 sink, which this round leaves
   at −162.8 kt) and ore_grade on icsg v13.

## Predictions (posted on #65 before Stages 2–4 ran)

1. Common-set z² falls, mostly on U.S. 2005–2012 rows. **Wrong:** z² rose 0.16 % vs base, and the U.S.
   rows fell only 337 z², because the base had already deleted the phantom (§4).
2. Round-E items within 1 pp, mass moved falls. **Holds.** Small-node p90 +0.010 was predicted to
   move only in the United States and Singapore, but it is not attributed by region here.
3. MESC and USC show both signs, and no sink. **Holds.** Gross MESC was predicted to rise and fell
   17 %; USC is flat and PSC falls 96.6 %, as predicted.
4. 1995 unchanged. **Holds:** −163.5 → −162.8 kt.
5. Singapore cathode import legs fall by roughly the added netting. **Holds** (§4).

## Reproduce

```
python scripts/parse/icsg_parse/{icsg_2005,icsg_2014,icsg_2023f,icsg_cap,icsg_comb,process_icsg}.py
python scripts/parse/icsg_parse/icsg_clean.py --to-feed
python scripts/build_baci_reexport.py --source-version 13 && python scripts/baci_feed.py
python scripts/build_trade_shares.py && python scripts/icsg_feed.py      # twice, §6
python scripts/explo/issue65_icsg_stock_parse_diag.py opt_20261012_122124_40b1b370 opt_20261012_135741_67851aa1
python scripts/explo/issue65_icsg_stock_parse_diag.py --crosseval opt_20261012_122124_40b1b370 opt_20261012_135741_67851aa1 1996 1998 1999 2000 2003 2004 2006
```
