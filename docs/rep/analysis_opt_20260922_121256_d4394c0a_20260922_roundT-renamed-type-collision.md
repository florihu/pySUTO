# Round T — a renamed type is a second stream; and the pipeline was being run out of order

**Run shipped:** `opt_20260922_121256_d4394c0a` (label `roundTp_correct_trade_shares`, 30/30 Solved, `prior_mode` cold)
**Also measured:** T `opt_20260922_114116_88941ebc` (the code fix alone, on the stale weights)
**Against:** S `opt_20260922_104430_60dbadb2`, Q `opt_20260921_114721_de4c5cc5`, base `opt_20260916_124721_c4d1d944`
**Chain (shipped):** `imp_20260922_115534_896bfd10` → `const_20260922_120030_24cf1571` → the run
**Feeds/params:** icsg **v12** (the `REN_TYPE` collision summed), **trade_shares v21** (was v20, built 2026-09-17); baci v11, crude_ore v3, conc v8, ore_grade v7, scrap_share v3 unchanged
**Date:** 2026-09-22 · **ADR:** [0036](../adr/0036-a-renamed-type-is-a-second-stream-not-a-restatement.md)

**Verdict: two levers, separately attributable, and the second is the larger. The `REN_TYPE`
collision fix takes ADR 0029's forced term 4,723 → 3,670 kt CC (−22 %) and Zambia's 1,452 → 399
(−73 %). Rebuilding the stale trade-share weights — no code, no feed values, just running the
pipeline in its documented order — then takes the forced term 3,670 → 3,013 and common-set z²
from −3.4 % to −9.6 % vs base. The second is worth more than Rounds S and T combined.**

---

## 1. The two levers, isolated

They are cleanly separable and are reported separately on purpose: `S → T` holds the weights
constant and changes only the code; `T → T′` holds the code and feed values constant and changes
only the weights.

| | | forced (kt CC) | mine deleted | common-set z² | vs base |
|---|---|---|---|---|---|
| S | | 4,723 | 26,742 | 1,661,006 | −3.3 % |
| **T** | `REN_TYPE` collision summed | **3,670** (−22 %) | 26,580 | 1,659,137 | −3.4 % |
| **T′** | trade shares rebuilt v20 → v21 | **3,013** (−18 %) | **24,884** (−6.4 %) | **1,552,144** | **−9.6 %** |

Both 30/30 Solved. Concentrate CC retention 0.934 → 0.938; mine prior 404,591 → 403,804;
hard-capped region-years 138 → 132.

## 2. Lever one — a renamed type is a second stream

The detector said Zambia's mine node could not discharge because its smelter intake had collapsed:

| year | mine supply | smelter intake | refinery anode use |
|---|---|---|---|
| 2003 | 269.0 | 253.9 | 241.5 |
| **2004** | **319.5** | **20.7** | **286.5** |
| **2007** | **341.0** | **1.9** | **229.4** |
| 2008 | 372.0 | 279.8 | 267.8 |

No refinery consumes 287 kt of anode fed by a smelter taking 10 kt of concentrate.

**The yearbook is right and so is the 2014 parse.** Zambia's smelter table has two rows — `Primary`
280.1 and `Low-grade / Electrowon` 20.0 — and `icsg_2014_flows.csv` carries both. `icsg_comb.py`
then folds them together (`REN_TYPE[('Low-grade','Smelting')] = 'Primary'`), leaving two rows of the
**same vintage** sharing a key, and the cross-vintage dedup — which exists to keep the newest
*restatement* — cannot tell a restatement from a second stream. It kept the 20.0 and discarded the
280.1: **1,038 kt of Zambian primary smelting over four years.** Which row survived was also
order-dependent, since `sort_values` defaults to an unstable quicksort.

Rows folded onto one key **within one vintage** now sum, before the dedup runs. Grouping includes
`Year_loaded_from_file`, so a genuine restatement is still left for the dedup — a test pins that,
because summing across vintages would double-count every restatement. Zambia 2004 becomes 300.1, and
its 2004 shortfall of 267 kt becomes 23 kt of headroom.

Six keys collide feed-wide: Zambia's four and two `Thailand 6/` `Semi/Supply` rows that are inert
(`Thailand 6/` is a footnote suffix escaping `REGION_RENAME`, so it maps to no root region, and
`cl()` drops `Semi/Supply` anyway). Feed diff: **6 values changed, 0 rows added or removed.**

## 3. Lever two — the pipeline was being run out of order

`README` requires `build_trade_shares.py` to run **before** `icsg_feed.py`: `icsg_feed` consumes the
weights to fan ICSG's single-endpoint rows, and `build_trade_shares` reads the ICSG feed's own
production for the [ADR 0024](../adr/0024-fan-domestic-leg-bounded-by-own-production.md) domestic
bound. **`trade_shares` stood at v20, built 2026-09-17**, while the ICSG feed was rewritten by Round
Q (09-21) and Rounds S and T (09-22). The bound was therefore applied against each round's
*previous* production.

Rebuilding to v21: **6,242 weights changed**, 47 dropped, **21,709 kt CC moved from domestic
residual onto import legs**; Belgium's domestic anode leg 3,219.7 → 2,250.5 (**−30 %**), with
Bulgaria 755.9 → 891.7 and Chile 529.6 → 650.2 taking it up.

**This is an operator error, not a model defect, and it is stated here rather than buried.** It cost
6.2 percentage points of fit across four rounds and very nearly cost a wasted round of work (§5).

**What it does *not* invalidate.** Every run from O onward used v20, so the arc is internally
consistent and the rounds' relative results stand — Round S's ROW 7,589 → 0, Round R's −54 %,
Round T's −22 % are all measured against like weights. Only `base` sits on v18, a boundary that
predates this session. What was stale is the bound's view of production, and the defects themselves
(Azerbaijan's 16 phantom rows, Zambia's 20-against-280) are data facts verified against the source
workbook, not against any run.

## 4. Where the forced term now stands

```
Congo, D.R.    880   of deleted    496  (177.6 %)
Zambia         399   of deleted  1,368  ( 29.1 %)
Belgium        371   of deleted     56  (664.2 %)   <- was 1,029 on v20
Chile          295   of deleted 26,117  (  1.1 %)
Iran           240 | Russia 174
```

Total 3,013 kt CC = **12.1 % of all mine-chain deletion**, its lowest across the arc (base 21.4 %,
Q 31.3 %, S 17.7 %). Congo D.R. is now the largest single item and is **not tractable by the ADR
0033 method**: its 2014 export hole (2013: 702 kt → 44 → 202) is low in *value* as well as quantity,
so BACI is internally consistent there and offers no contradiction to exploit. Congo is
mirror-reported; this is missing coverage, not a corrupted field.

## 5. The fabricated mine face, re-measured

MF→Concentrate priors in regions with **no ICSG mine production at all**:

| | region-years | prior | posterior |
|---|---|---|---|
| S / T (v20) | 82 | 1,778 | 1,386 kt CC |
| **T′ (v21)** | **76** | **992** | **845** |

The bound, once recomputed, removed **39 %** of it with no fix written — Belgium 720 → **309**,
Germany 261 → **127**. But the residue is not evenly spread: **Italy (163), Japan (114), Korea (60)
and Austria (35) did not move at all.** The bound caps a domestic leg at the region's own ICSG
production, so it cannot reach a fabrication that is upstream of what it measures — Italy has a
refinery and no smelter whatever; Japan and Korea have large concentrate smelters whose own
production the bound reads as legitimate.

**845 kt CC of phantom mine output survives in five countries that mine no copper.** That is the
next lever, now scoped on clean evidence rather than on a number this round's own ordering error had
inflated.

## 6. What is open

1. **Nothing enforces the build order.** `icsg_feed.py` loads whatever `trade_shares` version is
   registered and prints it, with no check that it post-dates the `values.csv` being fanned. The
   candidate guard is in [TODO](../TODO.md); on this evidence it is earned.
2. **845 kt CC of fabricated mine face**, targets Italy / Japan / Korea / Austria.
3. **Congo D.R., 880 kt CC** — coverage gap, parked.
4. **Q and R's published figures** were measured on v20 built before their own feed changes. The
   rounds stand; the numbers deserve a re-measure on v21.
5. **`process_icsg.py` collapses three PMC stock series onto one `USC` key** across 53 keys and
   breaks the tie with an unstable sort — the same shape as this round's defect, unfixed by choice.
6. **`icsg_2005.py`, `icsg_2023f.py`, `icsg_cap.py`** still carry the dead import and stale path.
