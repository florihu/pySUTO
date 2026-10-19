
# Harmonize registry. done
* registry harmonization for feed, imp, const solve registry
    * pattern is always stage id (feed, imp, const, opt)+ datet time + random code !! label lives only in registry

* input registry for data feeds, accs min id + version + date + random code see (const imp feed) label only in jason -> these are initialized when the init script is initilized and contain only minimal information.... extend with label var if not in meta 
    * for feeds lives in data/input/feed
    * for acc lives in data/input/acc
    * params lives in data/input/params

# Major change 
* Rework stock addition modelling
* Introduce as constant constraint



# Mask as an input

1. conceptualization
2. implementation
3. versioning


# Unc propagation

1. Conceptualization - > What methods are feasilbe.
2. Implementation
3. Analysis


# Improve solver diag — DONE 2026-09-23 (SolverDiag portfolio revision, branch diag/opt-portfolio-revision)

1. General convergence analysis of solver        → SD-R-01 (health), SD-R-12 (per-iteration traces), SD-R-11 (objective decomposition)
2. Most-changed a, z outliers                    → SD-R-13 / SD-R-14 / SD-R-15a (edge defects by objective share), SD-T-07 a-space
3. c*(t)−c₀(t): which constraints moved          → SD-R-13 / SD-R-14 / SD-R-15b-c (sum-row and soft-ratio residuals); `ranking_<family>.csv`

# ROUND Z — BACI KILOGRAMS AS TONNES: THE LOW SIDE OF ADR 0033 (MERGED 2026-10-08; issue #48)

**Defect:** PNG's 2019–2022 HS 260300 legs book kilograms under a tonne header (PNG → Philippines
2020: 21,277,530 t at 3.6 USD/t, value-implied Cu fraction 0.0006). ADR 0033 was one-sided, so
the world Concentrate `Mass` prior rose 64 → 105 Mt (SD-R-09 Mass retention 0.584 in 2020 on
`opt_20261007_130833_1a9b7af7`), and through the trade shares PNG took 81 % of the Philippines'
and 92 % of ROW's CC concentrate fan in 2020.

**Build (`fix/baci-low-side-recon`):** a leg < 0.02 in an exporter-year < 0.20 is rebuilt
downwards at its exporter's own `g_ref` and `cv_scale`; the above side is unchanged. At the parse
(V202601, no solve): 119 legs / 70 exporter-years / 27 exporters, 115,854 → 390 kt; PNG 2019–22
→ 468.5 / 391.3 / 328.8 / 334.6 kt; Congo D.R. → Zambia 2020 1,964 → 38 kt; the 881 above-band legs
are identical. ADR 0033 amended; GLOSSARY, README and the parse docstri**Measured:** `opt_20261008_092950_625a6c7e` (Z: gross v13 → netted v14, trade_shares v23;
chain `imp_20261008_084450_041271d3` → `const_20261008_084916_293ff701`; 28 Solved + 2018, 2022
AlmostSolved) against Z.base `opt_20261008_083810_56f905e9` (gross v10 re-netted → v12,
trade_shares v22; `imp_20261008_075249_bb1e1884` → `const_20261008_075717_9ccd0d40`; 30/30 Solved).
Z-1…Z-4 all hold: world Concentrate Mass prior 2020 105,492 → 65,491 kt, SD-R-09 Mass retention
2019–22 0.962 / 0.950 / 0.957 / 0.950, PNG's share of the PHL / ROW CC fan 2020 0.807 / 0.922 →
0.033 / 0.021, Congo D.R. → Zambia 2020 z −36.5 → +0.4. Common-set sum-row z² −2.79 % vs Z.base
(−3.05 % without Z's 2018, which a cross-evaluation shows stopped 2,415 short of its own optimum).
Gross null exact; Chile forced unchanged. Costs: exporter retention −0.4…−0.5 pp (a denominator
effect: their priors rise with the corrected fan, and their posteriors rise too); Congo D.R. 2020
+66.7 kt CC forced (the ambiguous leg). #30's re-netting alone: z² −0.07 %. REP
`docs/rep/analysis_opt_20261008_092950_625a6c7e_20261008_roundZ-png-kg-as-t.md`;
`scripts/explo/roundZ_png_kg_diag.py`.

**Verdict:** accepted and merged to main 2026-10-08; Z (`opt_20261008_092950_625a6c7e`) is the
reference chain. The manuscript's Method and SI describe the two-sided rule (BACI v14); its
Results still read the W.1 chain, as they have since Round X.t:** pending.

# OPEN (2026-09-16, out of Round L)

## #92 MEASURED 2026-10-19 — FAILS ON ITEM 4 (no new sink), fix ticket #95; items 1–3, 5, 6 hold

`diag/refined-scope-measure`, run `opt_20261019_142026_83bfe843` vs `opt_20261015_093646_ddd9d3d9`, REP
`docs/rep/analysis_opt_20261019_142026_83bfe843_20261019_issue91-refined-scope.md`;
`scripts/explo/issue92_refined_scope_diag.py`. Chain: baci v26 → v27, trade_shares v36 (v25 an unused orphan, #93).
- **Gains.** Austria's 1995–2014 cathode-use Σz² 48,500 → 2,624; Austria −88,777. Common-set sum-row z² −5.57 % vs
  `opt_20261015_093646_ddd9d3d9`, −2.0 % counting the rows only one run has (the UAE's new rows carry 46,472).
  Round-E (a)–(d) hold; the < 10 kt band rises +28.6 % of its base, mostly the UAE.
- **Item 4 fails.** 25 CC stock legs newly above |z| 5 (Kazakhstan, DR Congo, Zambia, Russia, Indonesia; Σ|post|
  304.8 kt CC), and the net stock posterior goes from +51 to −1,328 kt CC over 1995–2024. Refined exports exceed
  ICSG production. The UAE's re-exports lose 178's supply: UAE +25,625. Both are in #95.
- **Shipped over item 4** under the maintainer's delegation (2026-10-19; #79 and #86 precedent):
  `opt_20261019_142026_83bfe843` becomes the reference chain.

## #91 + #88 BUILT 2026-10-19 — NOT MEASURED (#92): all refined copper enters the Cathode sector, and BACI 178 repeats DR Congo's declarations in 2015–2023

`fix/baci-refined-scope` (on `fix/baci-178-mirror-duplicates`).
- **#91, the HS scope.** The BACI sector map and the parse now read HS 740312 wire bars, 740313 billets and 740319
  other refined copper as Cathode, beside 740311 (ADR 0014 am.). They are 0.41–1.35 Mt a year of cross-region
  trade, 4.5–15.1 % of 740311's (gross). Austria's cathode node lacked 77.7 % of its 2003 refined exports (#63:
  its cathode-use rows carry z² 48,500 in 1995–2014 on `opt_20261015_093646_ddd9d3d9`). Chile gains 101–162 kt a
  year of exports in 1995–2010, DR Congo 245–772 kt a year in 2015–2019. China's and Italy's imports rise. ADR
  0033's copper price stays 740311.
- **#88, the mirror duplicate.** `baci_parse_feed.drop_mirror_duplicates` drops a declared code's rows, either side,
  in its duplicate years: `MIRROR_DUPLICATES = {178: (2015, 2023)}`, which is 3,163.8 kt of refined copper, 70.5
  of concentrate, 23.1 of anode and 2.2 of matte (gross).
  - On the full scope, code 180 alone lies within −74 … +422 kt (gross) of ICSG's DR Congo production in
    2015–2023, and 178's 226–465 kt a year would push it past; it falls 5–323 kt short in every year through
    2014 and 476 kt in 2024, where 178 stays.
  - The first window, 2020–2023, was derived on 740311 alone; #90 measured it and is superseded.
  - The drop runs before ADR 0033's reconstruction, so the copper price `P_cu` and the reconstructed concentrate
    legs move slightly.
- **Side effects for #92 to report:** ADR 0033's reconstruction moves concentrate legs through the copper price
  and through a region's reference fraction `g_ref` (178's own, once its 2015–2023 years go). A Cathode leg now
  summed from several HS codes gets a smaller σ under the Lp rule (p = 1.3; Austria 2003 toy: cv 0.209 → 0.169).
- **Still open:** 180's own 2015–2022 excess over ICSG (ADR 0019 unless explained), and 178's 2004–2010
  concentrate (#88).
- **The manuscript.** Method's sources paragraph and Table 1 now list the refined codes; Table 1's counts change with
  the final chain.

## #86 MEASURED 2026-10-15 — FAILS ON ITEMS 5 AND 6 (DR Congo), fix ticket #88; items 1–4 hold

`diag/congo-sacu-measure`, run `opt_20261015_093646_ddd9d3d9` vs `opt_20261014_131552_f320b5f7`, REP
`docs/rep/analysis_opt_20261015_093646_ddd9d3d9_20261015_issue82-congo-sacu.md`;
`scripts/explo/issue86_congo_sacu_diag.py`. Chain: icsg v15, baci v21 (gross) → v22, trade_shares v34.
- **ROW (#82):** ledger cathode gap 2012–2024 +5,829 → +1,951 kt CC; its cathode-use rows' Σz² 16,055 → 3,208;
  domestic semis posterior 1,024 → 3,709. What remains is the importer disagreement (Saudi Arabia, Turkey,
  Vietnam, Pakistan, Egypt, Oman, North Korea), left visible (ADR 0019).
- **South Africa 1995–1999** gains its trade: Δz² −9,775.
- **DR Congo fails items 5 and 6:** code 178's cathode lands on top of a balanced code 180 in 2020–2023 (ledger
  +441, +321, +489, +341). The solver raises refined output above its prior by 829 kt CC over 2019–2023 (base
  207), CP→Cathode retention 1.080–1.143 in 2020–2023, and opens USC 2022 DR Congo → China at −63 kt CC
  (|z| 6.9). Code 178's 2004–2010 concentrate loads DR Congo's smelter chain (DR Congo's 2004–2008 Δz²
  +22,220, all sectors).
- **Overall:** common-set sum-row z² −0.75 % vs `opt_20261014_131552_f320b5f7`; Round-E (a)–(d) hold.
- **Merged 2026-10-19 over the failed items by the maintainer's decision** (#75 and #79 precedent):
  `opt_20261015_093646_ddd9d3d9` is the reference chain; #82 closed, #88 next. The si.md S8 concordance row
  changes with the chain that ships it.

## #84 + #85 BUILT 2026-10-15 — NOT MEASURED (#86): DR Congo's copper leaves ROW, SACU maps to South Africa

`fix/congo-sacu-mapping`. Two input labels gave ROW another region's copper (#82's triage):
- **#84, BACI.** The region map (`current/conc.xlsx`, not yet registered) sends 178 "Congo" to Congo, D.R.
  and 711 "Southern African Customs Union" to South Africa (ADR 0014 amended). 178 exported 3,890 kt of
  cathode in 2012–2024 (gross baci v19) against Brazzaville's 79 kt of ICSG production; 711 is South Africa's trade, with
  Namibia's and Botswana's, until BACI splits it in 2000.
- **#85, ICSG.** `icsg_clean.relabel_by_vintage` gives the 2005 yearbook's "Congo" (parsed as `Congo Rep.`,
  → ROW) its own label, Congo, D.R., for 1995–2003, and drops its 2004 rows, which the 2014 yearbook
  reports under Congo, D.R. with the refined output typed differently (ADR 0032 amended). On the current
  combined file: 27 rows relabelled, 3 dropped.
- **Expected** (gross values, no run): ROW's cathode U − (P + I − X) +2,335 / +5,808 → +2,148 / +1,920
  kt CC (1995–2011 / 2012–2024); DR Congo +1,136 in 2012–2024 (785 kt of its cathode goes to Tanzania,
  in ROW and not netted); South Africa +330 in 1995–1999.
- **#86's chain.** `icsg_clean.py --to-feed`, register (the icsg diff touches only the Congo rows) → BACI as
  in #79 (re-parse to `baci/v0/`, set aside `current/reexport_*.csv`, copy, register the gross version) →
  `build_baci_reexport.py --source-version <it>` → `baci_feed.py` → `build_trade_shares.py` →
  `icsg_feed.py` → Stages 2–4 against `opt_20261014_131552_f320b5f7`. Until then `current/` pairs the new
  BACI map with values netted under the old one: run nothing BACI-side but that chain. The si.md S8
  concordance row changes with the chain that ships it.

## #79 MEASURED 2026-10-14 — FAILS ON ROUND-E (d) AND THE STOCK-LEG ITEM, fix ticket #82; items 1–3 hold

`diag/baci-490-taiwan-measure`, run `opt_20261014_131552_f320b5f7` vs `opt_20261013_122432_24e811b7`, REP
`docs/rep/analysis_opt_20261014_131552_f320b5f7_20261014_issue77-baci-490-taiwan.md`;
`scripts/explo/issue79_baci490_taiwan_diag.py`.
- **Chain.** The BACI re-parse reproduces v13's `values.csv` and gives baci v19 (gross, new map); then
  baci v20 (netted), trade_shares v33 (47/47 regions observed), icsg v14.
- **Taiwan enters the account.** Its undistributed ICSG rows go 41 (15,566 kt CC) → 0, and its semis use
  3 → 14,820 kt CC against ICSG's 15,460.
- **ROW's domestic semis leg** is fed in 17 of 17 years 1995–2011, with no imputer write. Its semis `f_agg`
  equals the ICSG fan within 0.011 kt CC.
- **But ROW's hidden gap shows.** In the ledger, U − (P + I − X) goes from −6,886 to +2,355 kt CC over
  1995–2011 and from +353 to +5,829 over 2012–2024. In 1995–2011 the domestic leg's posterior barely moves
  (4,320 → 4,324 kt CC), but its prior is now ICSG's 6,194, so ROW's cathode use totals carry Δz² +18,029.
  ROW's semis use is 0.61–0.96 of its fan, and its refined cathode after 2011 is no longer cut (posterior/prior
  0.65–1.07 → 0.98–1.06).
- **Overall vs `opt_20261013_122432_24e811b7`:** common-set z² +1.41 %, rising in every anchor band; mass
  moved +1.66 %. 18 stock legs newly above |z| 5, all into ROW.
- **Merged 2026-10-15 over the failed items by the maintainer's decision** (#75 precedent);
  `opt_20261014_131552_f320b5f7` is the reference chain. #82 is next, then #78 against this run.

## #77 BUILT 2026-10-14 — NOT MEASURED (#79): BACI 490 "Other Asia, nes" maps to Taiwan, not ROW

`fix/baci-490-taiwan`. The BACI region map (`current/conc.xlsx`, not yet registered) sends code 490 to
Taiwan (ADR 0014 and 0018 amended). Folded into ROW, it gave ROW Taiwan's cathode imports (baci v18,
HS 740311: 375–658 kt a year in 1995–2011), which zeroed ROW's domestic semis share (#76), and left Taiwan
trade-blind.
`build_baci_reexport.check_region_map()` now raises when the source version's region map differs from
`current/`'s: the netting would net under one map and `baci_feed.py` apply the other. **#79's chain**
therefore starts at the parse: `baci_parse_feed.py` (→ `baci/v0/`, its ADR 0033 σ sidecar re-keyed with
the new map) → set aside `current/reexport_*.csv`, copy `v0/{values,quantity_recon,quantity_cv_scale}.csv`
into `current/`, register (gross version) → `build_baci_reexport.py --source-version <it>` → `baci_feed.py`
→ `build_trade_shares.py` → `icsg_feed.py` → Stages 2–4 against `opt_20261013_122432_24e811b7`. Until then
`current/` pairs the new map with values netted under the old one: run nothing BACI-side but that chain. Once
#79 passes and becomes the reference chain, the manuscript's Taiwan limitation (method.md §3.2.8) and its
concordance row in si.md S8 change with it.

## #30 MEASURED 2026-10-19 — PASSES: ore_grade rebuilt on icsg v15 (v7 → v8)

`diag/ore-grade-rebuild`, run `opt_20261019_155415_a1d7e9a9` vs `opt_20261019_152525_6ca5c00a`, REP
`docs/rep/analysis_opt_20261019_155415_a1d7e9a9_20261019_issue30-ore-grade-v8.md`; `scripts/explo/issue30_ore_grade_diag.py`.
- **The grades.** ROW's 2004–2013 in-situ grade goes from 3.04–8.50 % on v7 (the ADR 0035 phantom window) to
  1.04–1.48 %, and ROW's 2004–06 ore posterior goes from about 14 % of its prior to about 100 % (CO Mass retention
  0.824 → 1.077 over 1995–2024). DR Congo's new 2003 grade is 7.99 %, an outlier of its own series. Rows with
  relative σ above 100 % fall from 104 to 36 over all ore_grade families (#94).
- **z²:** common-set sum-row z² +0.033 % vs `opt_20261019_152525_6ca5c00a`, neutral; Round-E holds; no new sink.
- `opt_20261019_155415_a1d7e9a9` becomes the reference chain.

## #78 MEASURED 2026-10-19 — PASSES items 1, 3, 4 (narrowed build; item 2 fails as worded in 2015, as in base); the first build was rejected (+29.2 %)

`diag/observed-zero-measure`, runs `opt_20261019_145258_6bbfa0b5` (first build, rejected) and
`opt_20261019_152525_6ca5c00a` (narrowed, 30/30 Solved) vs `opt_20261019_142026_83bfe843`, REP
`docs/rep/analysis_opt_20261019_152525_6ca5c00a_20261019_issue76-observed-zero.md`;
`scripts/explo/issue78_observed_zero_diag.py`.
- **The narrowed build** stamps 82 observed zeros on non-miners' domestic concentrate legs. Germany's mine rows fall
  101,078 → 0 over all rows (#32). Common-set sum-row z² −7.0 % vs `opt_20261019_142026_83bfe843` (−11.1 % with the
  54 base-only rows). Every Round-E band falls, and no new stock leg rises above |z| 5.
- **Item 2** fails as worded in 2015 only: ROW's semis prior is +281.7 kt CC above its fan, identical in base. ROW
  is a producer, so its zero stays imputable by design.
- **The first build** stamped all 367 zero-weight domestic legs; producer region-years made it +29.2 %. The second
  build is post hoc, and its scope change is recorded on #76.
- **Open:** #96 (the 246 cathode non-producers) and Germany's phantom SX-EW chain (Ore→SXEW 381.2 kt CC posterior on
  a prior of 0.5; +33 % vs base).
- **Shipped by the maintainer's decision** (delegated): `opt_20261019_152525_6ca5c00a` becomes the reference chain.

## #76 NARROWED 2026-10-19 after #78's first run — NOT MEASURED: only a non-miner's domestic concentrate leg is an observed zero

`fix/observed-zero`. #78's first run (`opt_20261019_145258_6bbfa0b5`, all 367 zero-weight domestic legs stamped)
raised common-set sum-row z² 29.2 % vs `opt_20261019_142026_83bfe843`: Italy, the UK, South Africa, Bulgaria and
Spain, whose zeros came from imports above usage while they produce the product. Germany's mine rows fell 74,373 → 0.
`icsg_feed.non_producer_mask` now limits the stamp to domestic concentrate legs in regions with zero ICSG concentrate
production that year (`OBSERVED_ZERO_SECTORS`; ADR 0024 am.): 82 legs on icsg v15, Germany's 30 among them. The first
run's damage came from the 25 cathode and 14 concentrate zeros in producer region-years; the 246 cathode zeros in
non-producers would qualify but need the bound's raw-production key (#96). #78 re-runs.

## #76 BUILT 2026-10-14 — NOT MEASURED (#78): a weighted fan's zero domestic leg is an observed zero

`fix/observed-zero`. The ICSG feed carries a domestic leg its trade shares weigh at 0 as covered at 0,
`σ = max(0.01·|U|, 0.3 kt CC)` on the usage its fans distribute (ADR 0024 amendment; `observed_zero` mask in
the feed's `result.npz`, honoured by `FeedCollector` and exempt from `min_flow_kt`), so the `NetworkImputer`
no longer refills it (64 + 4 `Cathode→FU` and 98 `Concentrate→Smelting` refills, 6,212 kt CC, on
`imp_20261013_113740_5e11f431`). In memory on icsg v14 + trade_shares v32, nothing registered: the feed is unchanged
elsewhere (`F_root`, `sigma_root` equal `feed_icsg_20261013_113455_40931987` before the stamp), 355 observed
zeros (257 `Cathode→FU`, 98 `Concentrate→Smelting`), ROW's semis leg in 1995–2011, 2015, 2016 and 2020–23 at
σ 6.8–17.6 kt CC. **Measurement = #78, after #79** (#77's BACI 490 remap).

## #75 MEASURED 2026-10-13 — FAILS ON ONE CLAUSE (#73 semis use), fix ticket #76; every other item holds

`diag/scandinavia-split-measure`, run `opt_20261013_122432_24e811b7` vs `opt_20261013_095846_46ece808`, REP
`docs/rep/analysis_opt_20261013_122432_24e811b7_20261013_issue73-74-scandinavia-split.md`;
`scripts/explo/issue75_scandinavia_split_diag.py`. icsg v14, baci v18, trade_shares v32. Vs
`opt_20261013_095846_46ece808`: #74 fixed (Sweden's `Ore→SXEW` 307.8 → 0.0 kt CC over 1996–2003, ROW →
Sweden MESC 2001 |z| 15.7 → 0.01); ROW 2004 anode/concentrate/mine totals fall by exactly the dropped
rows and ROW 2004 sum-row z² 26,099 → 609; **but ROW's semis use does not fall**: the network imputer
refills ROW's domestic semis leg, which the fan now leaves at 0 (2004 model prior 1,396 → 1,404, posterior
1,498 → 1,680 kt CC; 1995–2004 posterior above the ICSG fan 2,434 → 4,500 kt CC). Common-set sum-row z²
−6.16 % and mass moved −3.56 % vs `opt_20261013_095846_46ece808`, mostly ROW's refinery chain (−67,883):
its refined cathode, deleted on the base (1,897 kt CC over 1995–2004), now lands on that imputed leg.
Δz² rises vs `opt_20261013_095846_46ece808`: Zambia +2,082, Uzbekistan +1,107. **Merged 2026-10-14 over
the failed clause by the maintainer's decision** (ADR 0043 precedent); `opt_20261013_122432_24e811b7` is the
reference chain. #76 decides the zero-weight fan leg.

## #73 + #74 BUILT 2026-10-13 — NOT MEASURED (#75): an ICSG aggregate row is split to its declared member

`fix/scandinavia-aggregate-allocation`. `icsg_clean.split_aggregates()` (ADR 0032 amendment) splits
Scandinavia's 1995–2003 rows to Sweden by the 2004 handover year's shares (refined primary 0.521, mining
0.837, …) and drops Scandinavia's 2004 rows where Sweden reports the series; params `icsg_aggregates` v4
(handover year = the year after `year_to`). On the #65 parse output, run to scratch: the old clean step
reproduces icsg v13 exactly; the new one adds 54 Sweden rows (1995–2003: refined anode use 1,558.2 kt CC,
smelter concentrate use 1,296.0, mine output 681.3, semis use 1,548.3), keeps every production and use
series total in 1995–2003 (refinery scrap use +0.40 kt CC through the ADR 0030 nesting clip, 9 new
excess region-years), and lowers 2004 by Scandinavia's rows only (refined anode use −408.6 kt CC).
**Measurement = #75** (needs-triage: its pass rule is a proposal), from `icsg_clean.py --to-feed`
(icsg v14) through Stages 2–4 against `opt_20261013_095846_46ece808`.

## #69 MEASURED 2026-10-13 — PASSES, two findings against intent: Sweden's cathode un-netted (#67)

`diag/aggregate-gap-measure`, run `opt_20261013_095846_46ece808` vs `opt_20261012_135741_67851aa1`, REP
`docs/rep/analysis_opt_20261013_095846_46ece808_20261013_issue67-aggregate-gap.md`;
`scripts/explo/issue69_aggregate_gap_diag.py`. baci v17, trade_shares v31, icsg_aggregates v2. Every
pre-registered item holds vs `opt_20261012_135741_67851aa1`: only Sweden's T changes (−220.9 kt CC),
Sweden 2003 mining ratio 0.41 → 0.70, Sweden 1996–2003 sum-row z² −6,055, Singapore −0.087 %, Round-E
unchanged, mass moved +0.00 %. Common-set sum-row z² **−0.69 % vs `opt_20261012_135741_67851aa1`**.
Against intent, both because 1995–2003 "Scandinavia" smelting/refining maps to ROW (**#74**): the restored
trade is supplied by a Swedish SX-EW chain, a flat-prior sink at σ 1e6 (`Ore→SXEW` 1996–2003 posterior
175.0 → 307.8 kt CC), and by a forced ROW → Sweden MESC move in 2001 (+49.1 kt CC at z 15.7).
**Reference chain changes to `opt_20261013_095846_46ece808` only if merged — owner's call.**

## #68 MEASURED 2026-10-13 — PASSES: the raw-usage trade shares (#66) reproduce v29

`diag/raw-usage-trade-shares-measure`, REP
`docs/rep/analysis_opt_20261012_135741_67851aa1_20261013_issue66-raw-usage-trade-shares.md`;
`scripts/explo/issue68_trade_shares_identity.py`. trade_shares **v30** (`387c5d68`), rebuilt on icsg v13
and baci v16, has exactly v29's 146,311 weights, each within 1.6e-11 relative (rule: 1e-9). No ICSG feed
or Stage 2–4 run; **the reference chain stays `opt_20261012_135741_67851aa1`.** Follow-up, not filed: the
builder's ADR 0024 line prints −304,203 "kt CC moved" because it sums the stock legs' supply-mix weights
(ADR 0043). The 442 bounded flow legs moved +20,536 kt CC. The Round T guard below is still open.

## #65 MEASURED 2026-10-12 — ACCEPTED: the ICSG feed rebuilt from the repaired stock parse (#35)

`diag/icsg-stock-parse-measure`, run `opt_20261012_135741_67851aa1` vs #40.1 `opt_20261012_122124_40b1b370`,
REP `docs/rep/analysis_opt_20261012_135741_67851aa1_20261012_issue65-icsg-stock-parse.md`. icsg v13, baci v16
(netted against v13; weights = v15's), trade_shares v28. Every pass-rule item holds:
- MESC 29 y, USC/PSC 28 y.
- US PSC ≤ 19.9 kt.
- Round-E holds and mass moved −5,945 kt.
- Gross stock-leg adjustment falls on all three families (PSC 6,080 → 207 kt CC).
- No non-stock row moves.

Sum-row z² is **+0.16 % vs base**: Sweden +2,636, Singapore −2,924. The feed's stock changes now sum to
+540 kt over 1996–2024 (ICSG: +456; v12: −877). **New reference chain once merged:
`opt_20261012_135741_67851aa1`.** Follow-ups (REP §7), filed 2026-10-13 in milestone "manuscript ready": item 1 = #67 → #69, item 2 = #66 → #68:
1. A lone MESC/USC row lifts a reporting gap out of the ADR 0027 `icsg_gap` exclusion. Sweden, folded
   into "Scandinavia" until 2003, is netted in every year 1996–2003 (220.9 kt CC; 2003 new with #35);
   Singapore is the case where that is right. **Built, #67 (2026-10-13, `fix/icsg-aggregate-gap`):
   declared membership in params `icsg_aggregates` v2; in memory on baci v13 × icsg v13 only Sweden's
   T changes (Σ T −220.9 kt CC). Σ T_applied falls 215.3 kt CC: Sweden 2002 applied 31.3 of its 36.4,
   and hub→hub chains move 25 other hub-years' T_applied, all by ≤ 0.28 kt CC (UK 2002). #69 should quote
   both. Measurement = #69.** Found while building, filed as #73 (needs-triage): ICSG's 2004 "Scandinavia" rows (loaded
   from the 2005 yearbook) duplicate Sweden's, Finland's and Norway's own 2004 rows (2014 yearbook;
   refined anode use 408.6 kt CC both ways), so the Nordic 2004 flows enter twice, once via ROW.
2. `build_trade_shares.py` takes usage from the latest ICSG feed run, so after an ICSG change it needs
   a second pass (v27 → v28 here). Either read raw ICSG usage everywhere or document
   ICSG feed → trade shares → ICSG feed. **Done, #66 (2026-10-13): raw usage everywhere; the rebuild
   on icsg v13 reproduces trade_shares v29 to 1.6e-11 relative, one pass suffices.**
3. #37 and #30 are next on icsg v13.

## #40 MEASURED 2026-10-12 — NOT ACCEPTED: negative stock changes reach the account, two fan defects surface

`fix/signed-stock-fans` (9e83a93, unmerged), run `opt_20261012_084619_3c11d751` vs Z
`opt_20261008_092950_625a6c7e`, REP `docs/rep/analysis_opt_20261012_084619_3c11d751_20261012_issue40-signed-stock-fans.md`.
The fan of a signed stock change is now computed on its magnitude, so all non-Taiwan negatives are
distributed; Round-E holds. Pre-registered item 4 fails in 2003 (+121 kt on MESC/USC) and common-set
sum-row z² rises **+9.22 %**, 97.2 % of it on 48 cathode use totals of six exchange hubs (NL, SG, MY,
FR, UAE, UK): the fan books their draw-downs on cathode they never produced. 1,250 stock leg-years
also carry a negative CC prior beside BACI's positive Mass prior (#36 item 1).

**#40.1 MEASURED 2026-10-12** (`568d62f`, run `opt_20261012_122124_40b1b370`, REP
`docs/rep/analysis_opt_20261012_122124_40b1b370_20261012_issue40-1-supply-mix-fans.md`): stock changes
drawn from the region's cathode supply mix + BACI cathode fan 100/0/0. Hub z² 160,444 → 841, mixed-sign
legs 1,250 → 0, z² +0.13 % vs Z; **item 3 fails** (gross stock-leg adjustment > +100 kt vs Z in 10 years;
net within ±91 kt except 1995, −163.5 kt with no stock data). **SHIPPED over item 3 by the maintainer's decision
(ADR 0043); the 1995 sink and the gross shuffling go to #37. #65 is next.** Registered latest: baci v15, trade_shares v26
(v25 discarded). **New reference chain: `opt_20261012_122124_40b1b370`.**

## ADR 0017's reference leg size drifts with the population it prices

`a_ref` is the **median of the covered bilateral legs**, so cleaning the leg
population moves it. The ADR 0027 netting removes the small hub-side legs and the
median rises (Concentrate 10.26 → 12.0 kt, Anode 0.21 → ~0.3); a higher `a_ref`
raises the damped cv of every surviving leg, looser legs are cheaper for the QP to
delete, and deleting them makes the independently anchored scale-total rows worse.
So **improving the trade matrix silently loosens σ on what survives** — the rule
penalises the cleaner population.

Measured (`opt_20260916_072151_ac748919`, the shipped chain with the pin off):
sum-row z² 1,748,645 vs 1,720,192 (**+1.7 %**), Chile concentrate retention
0.574 → **0.550**, world CO Mass 0.779 → 0.767, Concentrate CC 0.872 → 0.864.
30/30 Solved. REP `…roundL1-baci-reexport` Addendum 3.

`LEG_SIGMA_AREF` in `collect_aggregate_impute.py` currently pins the L.0 medians.
**Do not just restore the `None` default** — that is the worse configuration, and
it is the measurement above that says so. Decide instead: a per-sector constant, a
quantile of the *gross* matrix, or a physically motivated size — something that does
not move when the matrix it prices gets better.

## ROUND O — THE SECONDARY DOUBLE COUNT (measured 2026-09-21; part of the mergeable O+P stack)

Found while writing the copper-metabolism insights REP
(`docs/rep/analysis_opt_20260916_124721_c4d1d944_20260917_copper-metabolism-insights.md`).

`scripts/parse/icsg_parse/icsg_clean.py:18-22` converts BOTH ICSG secondary
production series into inputs — `Smelter Second.` → Scrap→Smelting and
`Refinery Second.` → Scrap→CFR — but ICSG's secondary refined production already
INCLUDES the copper that came through secondary smelters (blister → anode → ER).
The two series overlap by the smelter route, per region-year
`min(S_sm·k, S_ref)`: **3,718 kt in 2024, 70.0 Mt over 1995-2024**, against a
30-y world mine-chain deletion of 52.9 Mt (ratio 0.75, corr 0.67). In 119
region-years (Japan, Bulgaria, Iran …, 1.99 Mt total) the smelter route exceeds
secondary refined: clip to zero, list them. **Second instance:** the cathode
term in `icsg_intake_production` adds `Refining/Scrap·k` on top of an
`anode_in` that is already total refined — ~4.7 Mt (17 %) too much cathode
production in the P that ADR 0027/0024 use.

Proof: Austria (Brixlegg, 100 % secondary) — ICSG 60 kt secondary smelter inside
123 kt secondary refined = 123 kt total; the feed books 62 + 125 = 187 kt of
scrap against a 123 kt anode anchor; the posterior deletes the smelter scrap to
0.1 kt. With both booked, the anode anchor can absorb 14,758 kt of concentrate
in 2024 against 18,207 mined (gap 3,450); nested, the gap is −652.

**The fix (parse level + cathode term + scrap_share v2, then full chain —
plan grilled 2026-09-17):**
```
Scrap→CFR      = max(0, S_ref − S_sm · eff_CFR · eff_ER) / (eff_CFR · eff_ER)   # direct-to-refinery only
Scrap→Smelting = S_sm / eff_Sm                                                     # unchanged
P_Cathode      = anode_in · eff_ER + SXEW                                          # no second scrap term
```
`scrap_share` MUST be regenerated with it (v1 CFR floors — China 0.160,
Germany 0.342, Austria 0.95 — sit above the post-fix direct-route shares
0.040 / 0.126 / 0.503 and would force the scrap back in).
Acceptance, before any solver run: Austria's both-booked share → 1.0; the
world `conc_absorbable` gap → ≈ −0.46 Mt (`t1a_secondary_world.csv`, nested column).
On the chain: mine-chain deletion ≪ 12.8 %; posterior secondary share → ≈ 0.17
(ICSG); China's smelters off their scrap-band ceiling (0.40); Chile's mine
retention up with NO change to the grade cones. List, do not clip, the
region-years where `S_sm·k > S_ref` — they test the nesting assumption.
Amend ADR 0021's "not a double count" sentence. Own ADR, own plan.

**The export-mass gap below is the regional routing of this surplus** — the
world needs 3-4 Mt/yr less concentrate than it mines and drops it where a hard
cone already makes it free. Re-measure it after O.

**MEASURED 2026-09-21** — run `opt_20260917_121759_c57f252b`, 30/30 Solved, against base
`opt_20260916_124721_c4d1d944`.
[REP](docs/rep/analysis_opt_20260917_121759_c57f252b_20260921_roundO-secondary-double-count.md) ·
[ADR 0030](docs/adr/0030-secondary-smelter-copper-enters-the-chain-once.md).

Feed gate: all four checks PASS (Austria 1.480 → 1.000, world gap +3,450 → −464 kt, one row class,
107 excess region-years listed). Mine-chain deletion 52,871 → 44,197 kt; the **forced** term is flat
(25,913 → 26,421) and the **discretionary** term falls 26,958 → **17,776 kt (−34.1 %)**. The double
count was real.

But sum-row z² **+13.3 %** (1,719,806 → 1,949,396), σ identical on all 21,729 unchanged-anchor rows,
so it is a genuine fit loss: Slovakia's anode +64,630 alone (anchor 57 kt unchanged, posterior
43 → 6.5 kt in all 30 y), plus Austria/Belgium, plus 146 new phantom concentrate/MF rows in Germany
and Spain. Secondary share 0.300 → 0.257 against a prior now at 0.164; China *harder* on its 0.402
ceiling; **Chile unchanged** (0.557 → 0.561, capped 19/30) — the Chile thesis is not supported.

## ROUND P.0 + P.1 — THE SCRAP ARC (measured 2026-09-21; READY TO MERGE — user's call)

Round O moved copper from a route the model anchors **by level** (`Refining/Scrap`) to one it anchors
**only by share** (`Smelting/Scrap`: `0.95 ≤ Σ(Scrap→Smelting)/Σ(all→Smelting) ≤ 1.00`). A share band
over an idle node is free. Correcting the earlier note: ICSG has Slovakia's smelter eating
**46.6 kt/yr** (the ~28 figure was 2008), the prior carries it feed-backed at σ 6.8 %, and the model
routes **0.012 kt/yr in both runs**. It is *not* prior decay — both runs are `--prior-mode cold`, so
`f_agg` is re-asserted annually and the solver walks away annually. **A per-edge prior penalty is
cheaper to pay than to satisfy; only a scale total prices the group.**

**Built** ([ADR 0031](docs/adr/0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md)): two
soft `sum` families, `smelterscrap` and `directrefscrap`, CC only, covered edges only, grouped on
`(sector_to, region_to, time, valuation)`, σ via the unchanged `_scale_total_cv`. Registered
**separately** so neither route can absorb the other. A region-year ICSG does not report gets no row
and keeps its freedom — the family can only assert a scrap inflow, never forbid one.

**MEASURED** — run `opt_20260921_090618_1ceb5eed`, 30/30 Solved.
[REP](docs/rep/analysis_opt_20260921_090618_1ceb5eed_20260921_roundP-scrap-level.md) ·
[ADR 0031](docs/adr/0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md).

Gate passed all six checks twice. **The family works at world scale**: posterior ÷ ICSG-reported
scrap 1.318 → 1.191 → **1.027**; secondary share 0.300 → 0.257 → **0.220** (prior 0.164); China
**off** its 0.402 ceiling (0.398 → 0.341); mine-chain deletion 44,197 → **38,876 kt** with the forced
term flat, so **discretionary deletion 17,776 → 12,455 kt**; retention 0.894 → 0.906; Peru → 0.953;
and **Chile finally moved, 0.561 → 0.578**.

**But P-d fails**: common-set z² +10.8 % → **+12.3 %** (this also corrects Round O's headline — its
regression was +10.8 %, not +13.3 %; the rest was its own new rows). The two motivating regions did
not recover, for two *different* reasons, neither of them the missing level — see P.1.

⚠ Also: ROW joins Kazakhstan above 1.05 intake retention; Cathode CC retention 0.984 → 0.966.

## ROUND P.1 — DONE: the pure-secondary recipe closes (2026-09-21)

Run `opt_20260921_101621_cd0c0ae2`, 30/30.
[REP](docs/rep/analysis_opt_20260921_101621_cd0c0ae2_20260921_roundP1-secondary-recipe.md) ·
[ADR 0010 amendment](docs/adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md).

P.0's diagnosis was right that Slovakia was strangled, wrong about where. Not the band (its Smelting
ceiling is 1.000) and **not ADR 0019** — that detector needs `inflow > 1 kt` and fires on **one
node-year in the whole model**; it is not a usable lever as written. It was **Stage 2**: the
`NetworkImputer` gate "no known primary ⇒ stay closed" is right about a data gap but wrong for a
100 %-secondary smelter, which has no primary *by construction* and whose input is measured on the
scrap leg. Slovakia's chain therefore carried nothing and Stage 3 pinned `usetotout Matte` at
`c = 0.011 kt, σ = 0.001`.

Result: **Slovakia 0.10 → 0.96**, anode 7.81 → **58.53 kt** (z −50.7 → +1.37), scrap level z −58.7 →
**−0.34**. Common-set z² **+12.4 % → +7.4 %**, below Round O's own +10.9 %. Stage-2 gate: 445 of
854,460 edges moved (0.052 %), two regions. Every criterion passes except P-g (secondary share 0.223
vs ≈0.17).

**The stack O + P.0 + P.1 vs base:** deletion 52,871 → 38,934 kt, discretionary 26,958 → **12,523**,
scrap fidelity 1.318 → **1.032**, invented direct scrap 7,652 → **198 kt**, pairs below 0.50 five →
**one**, Kazakhstan's fabricated node gone. **Ready to merge — the user's call on these numbers.**

## ROUND S — SHIPPED 2026-09-22: ROW's forced deletion was phantom production (ADR 0035)

Run `opt_20260922_104430_60dbadb2`, 30/30 Solved, icsg **v11**. **ROW forced 7,589 → exactly 0**,
hard-capped years 10 → 0; ADR 0029's forced term **12,315 → 4,723 kt CC (−62 %)**, share 33.5 % →
**17.7 %**; mine deletion 36,807 → **26,742 (−27 %)**; common-set z² **1,661,859 = −3.3 % vs base**,
the best of the whole arc. World MF→Concentrate retention 0.912 → **0.934**. REP
`…roundS-mine-total-misattribution`.

**Cause:** the ICSG 2014 yearbook types a country's name on the *middle* row of its three-row block,
so `bfill` carried each block's `Total` row onto the *next* country. Australia's mine total became
`Azerbeijan`'s (899 kt/yr against the real 0.3–2.6), Russia's became `Saudi Arabia`'s, Kazakhstan's
`Laos`, Congo D.R.'s `Cyprus`. **16 rows, 11,558 kt CC of production that never existed**, all inside
ROW, all 2004–2013 — precisely the window ROW is forced in.

**The null held to the kilotonne:** Chile 295, Zambia 1,452, Congo D.R. 880 all unchanged. The whole
−7,592 kt is ROW.

**The reframing this settles.** The prior falls 11,559 kt and deletion falls 10,065 — **87 % of the
phantom mass was going straight into the deletion ledger.** `Σ max(0, P − cap)` is partly the model
*correctly refusing phantom input*; where `P` is phantom, deleting it is the right answer and the
fix is to remove the input, not relieve the constraint. R recovered mass genuinely missing; S removes
mass that was never there.

**[ADR 0014](docs/adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md)'s folded-internal-trade
hypothesis is REFUTED**, not merely unused: 5,034 kt available against 21,683 needed (23 %), and
inside an aggregate a member's export is another's import, so dropping both is correct.

Now the largest forced items: **Zambia 1,452** (79.6 % of its own deletion), **Belgium 1,029** against
299 deleted (344 %, and Belgium has no mines — Round Q's fabricated domestic `Concentrate→Smelting`
edge), **Congo, D.R. 880** against 605 (146 %).

## ROUND T — SHIPPED 2026-09-22: a renamed type is a second stream (ADR 0036) + the build order

Run `opt_20260922_121256_d4394c0a`, 30/30 Solved, icsg **v12**, **trade_shares v21**. Two levers,
separately measured — see REP `…roundT-renamed-type-collision`.

**T (the code fix, weights held at v20):** `REN_TYPE` folds `Low-grade` smelter feed into `Primary`,
leaving two rows of one vintage sharing a key; the cross-vintage dedup kept the 20.0 and discarded
the 280.1, losing **1,038 kt of Zambian primary smelting** 2004-2007. Rows folded onto one key
*within* a vintage now sum. **Zambia forced 1,452 → 399 (−73 %)**, model forced **4,723 → 3,670
(−22 %)**, z² −3.3 % → −3.4 % vs base.

**T′ (the weights, code held constant):** `build_trade_shares.py` had not run since 2026-09-17, so
Rounds Q, R, S and T fanned their ICSG rows with weights built before their own feed corrections.
Rebuilding v20 → v21 moved **21,709 kt CC** onto import legs and gave **forced 3,670 → 3,013**,
**z² −3.4 % → −9.6 % vs base** — more than Rounds S and T combined, from running the pipeline in its
documented order. Operator error, not a model defect. The arc stays internally consistent (every run
from O onward used v20), so the rounds' relative results stand.

**Forced term now 3,013 kt CC = 12.1 % of deletion**, the lowest of the arc. Largest items:
Congo D.R. 880 (coverage gap, not tractable by ADR 0033 — its 2014 hole is low in value too),
Zambia 399, Belgium 371 (was 1,029 on v20), Chile 295.

**The fabricated mine face, re-measured:** 1,386 → **845 kt CC** posterior. The ADR 0024 bound took
39 % of it once recomputed (Belgium 720 → 309, Germany 261 → 127) but **Italy 163, Japan 114,
Korea 60, Austria 35 did not move at all** — the bound cannot reach a fabrication upstream of the
production it measures. **That residue is the next lever.**

## OPEN, out of Round T — nothing enforces the documented build order

`README` §"Two preprocessing scripts run before the feed pipelines" requires
`scripts/build_trade_shares.py` to run **before** `scripts/icsg_feed.py`: `icsg_feed` consumes the
weights (`TRADE_SHARES_ID`, `load_trade_shares`) to fan ICSG's single-endpoint rows, and
`build_trade_shares` in turn reads the ICSG feed's own production through `icsg_intake_production`
for the [ADR 0024](docs/adr/0024-fan-domestic-leg-bounded-by-own-production.md) domestic bound. Change
the ICSG feed without rebuilding and the bound is applied against the *previous* production.

**It happened**, for four rounds (Q, R, S, T — all on `trade_shares` v20, built 2026-09-17), and
cost **6.2 percentage points of fit**. Rebuilding to v21 changed **6,242 weights**, dropped 47, and
moved **21,709 kt CC** from domestic residual onto import legs; Belgium's domestic anode leg fell
3,219.7 → 2,250.5 (**−30 %**). Nothing detects it: `icsg_feed.py` loads whatever version is
registered and prints it, with no check that it post-dates the `values.csv` being fanned.

**Candidate guard:** compare the registered `trade_shares` build time against the version/mtime of
`data/input/feed/icsg/current/values.csv` and refuse (or loudly warn) when the weights are older.
Cheap, and it converts a silent ordering error into a stop. Not built — it was not Round T's lever,
but on this evidence it is earned.

Related: the same ordering applies to `build_baci_reexport.py` → `baci_feed.py`, and
`build_trade_shares` also reads the **netted BACI** feed, so a BACI change (Round R) should trigger
a rebuild too.

## FIXED by #35 (`fix/icsg-stock-parse`), measured by #65 — two defects found in the ICSG parse in Round S

Both are fixed in code: `USC` is the sum of its three types (a net addition per type and vintage, every sort
before a dedup stable), and all five parsers write to `data/proc/feedprep/icsg_clean` and run from the raw
yearbooks. #65 rebuilt the feed (icsg v13, run `opt_20261012_135741_67851aa1`). The record below is kept as found.


**1. `process_icsg.py` collapses three stock series onto one key, and picks the survivor at
random.** `RENAME_DICT` maps `PMC_Consumers`, `PMC_Merchants` and `PMC_Government` all to `USC`, so
they share the key `(Region, Cathode, Year, USC, Use)`. The collision is resolved by
`drop_duplicates(keep='last')` after `combined.sort_values("Year_loaded_from_file")` — pandas'
default sort is **quicksort, which is not stable**, so which of the three survives depends on row
order and changes whenever the row count does. **53 keys collide** (51 doubles, 2 triples).

Two things are wrong and they are separable:
- *The non-determinism.* One word, `kind="mergesort"`. Round S removed 16 rows and that reshuffled
  **7** Japan USC values (≤ 14 kt each, listed in the REP). Stabilising the sort was tested and
  rejected for Round S because it flips **21** values instead of 7 — it fixes future attribution at
  the cost of a larger one-off baseline move, which belongs in its own round.
- *The collapse itself.* Keeping one of Consumers / Merchants / Government and discarding the other
  two is almost certainly wrong — `USC` should be their **sum**. That is a modelling decision, not a
  parse repair, so it is not Round S's to make.

**2. Three more files in the ICSG parse stage are still dead.** Round S revived `icsg_2014.py`,
`icsg_comb.py` and `process_icsg.py` because the chain had to run. `icsg_2005.py`, `icsg_2023f.py`
and `icsg_cap.py` carried the identical two breakages; #50 fixed the import (`pysuto.util` →
`pysuto.utils`, now under test) and `out_folder = 'data/proc/datafeed/icsg_clean'` (renamed to
`data/proc/feedprep/`) remains. Their outputs are checked in and unaffected, so nothing is blocked
today, but the stage cannot be rebuilt from the raw yearbooks until it is repaired. Mechanical.

## ROUND R — SHIPPED 2026-09-22: BACI's quantity contradicts its own value (ADR 0033)

Run `opt_20260922_092530_f21bdbe7`, 30/30 Solved, baci **v11** (gross v10 reconstructed, then
ADR 0027 re-netted on it). **ADR 0029's forced deletion 26,959 → 12,315 kt CC (−54 %)**, its share
68.4 % → **33.5 %**; **Chile forced 14,270 → 295** (40.6 % of its deletion → **1.0 %**), hard-capped
19 → **5** years, retention 0.574 → **0.617**. z² +0.4 % (still −0.7 % vs base). REP
`…roundR-quantity-reconstruction`.

**Read it precisely: the round reclassifies more than it reduces.** Deletion falls only
39,394 → 36,807 kt (−6.6 %); ~14,600 kt CC moves **forced → discretionary** (12,435 → 24,492). The
arithmetic wall is gone; a fit problem is left in its place, which is the kind this project's levers
can address. Scoring **Q's own posterior** against the corrected feed already gives 12,339, so the
ceiling rises mechanically — the run's contribution is showing the model *uses* the headroom.

The **pre-registered null held**: ROW 7,576 → 7,589 and Zambia 1,418 → 1,452, both predicted not to
move because neither is value-flagged.

### ~~Now the largest open item: ROW, 7,589 kt CC = 62 % of the remaining forced term~~ — CLOSED by Round S

98.9 % of ROW's own deletion is forced, and ADR 0033 does **not** explain it — that was predicted
and confirmed. The ADR 0014 folded-internal-trade candidate was **tested and refuted**; the cause is
phantom mine production from an ICSG 2014 parse bug. See Round S / [ADR 0035](docs/adr/0035-the-yearbooks-label-is-not-on-the-row-it-owns.md).

Also open from R:
1. **Belgium newly hard-capped**, 1,029 kt CC forced against 300 deleted (343 %) — and Belgium has
   no mines. This is Round Q's fabricated domestic `Concentrate→Smelting` edge (505 kt, zero
   feed-observed) surfacing in ADR 0029's detector.
2. **Chile's deletion is now 29,410 kt and 99 % discretionary** — the first time that can be said.
3. **R.1**: the same value/quantity test on matte, anode and cathode. The detector already takes a
   sector table; only concentrate mass enters `cap`, so the others could not have moved R-a.

## ROUND Q — SHIPPED 2026-09-21: one territory, one label (ADR 0032)

Run `opt_20260921_114721_de4c5cc5`, 30/30 Solved, icsg **v10**, scrap_share **v3**.
**Common-set z² 1,844,644 → 1,698,662 — `−1.1 %` vs base.** The whole O + P.0 + P.1 + Q stack is now
better than the model it started from; Round O's +10.9 % regression is retired. `directrefscrap` z²
−68 %, Belgium/CFR 0.55 → 0.82 and its headroom 0.22 → 1.236, band ceilings 118 → 99, dead
region-years 13 → **2**. Costs: total deletion +460 kt but **entirely forced** (discretionary
12,523 → **12,435**); world scrap fidelity 1.057 → 1.076; Belgium/Smelting overshoots at 1.17;
Kazakhstan rejoins ROW above 1.05. REP `…roundQ-one-territory`.

Still open from Q: **Smelting/Austria headroom 0.596, blocked 30/30** (ceiling 1.000 — the node, not
the band); **Belgium/Smelting at 1.17** of its ICSG level; and the `stock_bounds_study.py` bug below.

## ROUND P.2 — WITHDRAWN 2026-09-21: the bands were not the defect. The FEED was. (→ Round Q)

P.2 was scoped as three band items. Measurement refutes two of them and traces the third to a
single root cause one stage upstream: **code that runs before the feed pipeline applies
`conc.xlsx`'s `region_map` matches on the SOURCE region label where the ROOT region is meant.**
`values.csv` holds raw source names (`Belgium-Luxembourg`, `South Korea`, `Taiwan, China`); the map
to root names (`Belgium`, `Korea, Rep.`, `Taiwan`) is applied downstream inside `feedpipe`.

### Q-a — ADR 0030's nesting subtraction skipped Belgium 1995-2013  ⚠ 2,424 kt

`scripts/parse/icsg_parse/icsg_clean.py::_smelter_route` groups by `['Region','Year']`;
`apply_input_divisors` then does `.reindex(keys).fillna(0.0)`. Belgium's `Refining/Secondary` rows
are labelled `Belgium-Luxembourg` for 1995-2013 while its `Smelting/Secondary` rows are all labelled
`Belgium`, so the lookup misses, `nested = 0`, and **the subtraction silently does nothing for 19 of
Belgium's 30 years**:

```
root     source              n   years       published  corrected  overstated
Belgium  Belgium-Luxembourg  19  1995-2013     3,321.3      897.4     2,423.9 kt
```

Validation: the corrected 2013 value is **9.9 kt** — exactly what ICSG publishes under the `Belgium`
label in that same handover year. Egypt / Ukraine / Romania are **not** instances: they have no
smelter series of their own, so `nested = 0` is correct for them. Nesting must happen per real
reporting territory and never at ROW level, which would subtract Serbia's smelter from Egypt's
refinery.

**This is item 1.** Belgium/CFR's 0.17 is not a band that is too tight — it is a **reported level
~5× too high for 19 years**, which P.0 now anchors and which the hard ceiling 0.195 is the only
thing refusing. The band was right; the feed was wrong.

### Q-b — the handover year is published under both labels and summed  (4 cells)

Belgium 2013 anode 388.2 → **776.3** (the series is otherwise flat at ~390), Belgium 2013 cathode
211 → 422, Taiwan 2013 cathode 437.5 → **875.0**. ROW's 226 multi-source cells are a genuine
aggregate and correctly summed — `conc.xlsx` already carries the distinction in `mapping_type`
(`aggregate` 47 / `direct` 45 / `alias` 2 / `typo` 2).

### Q-c — the band study never applies the region map

`scripts/explo/scrap_share_study.py` groups raw `values.csv` by source region, then looks the result
up with `acc.node_index["region"]` (root names). **172,174 kt of series values are never seen.**
**Korea, Rep.** (source `South Korea`) and **ROW** therefore get `DEFAULT_BAND (0.0, 0.25)` on both
processes — Korea/Smelting's ceiling is active in 7 years, 1,061 kt. **Belgium** and **Taiwan** get
bands computed from a partial series.

### Items 2 and 3 are NOT defects — measured, closed

- **`.clip(upper=1.0)`** clips by ~1e-5 relative (**6 grams on 123 kt**) on 68 CFR observations in
  three regions, two of which (Egypt, Ukraine) are not model regions. It is float-noise cleanup on
  `scrap == anode ÷ eff`, not a masked data inconsistency. The TODO's earlier characterisation was
  wrong.
- The six **`[0.95, 1.00]`** bands come from the documented `fillna(0.0)` "missing counts as 0"
  convention (none of those six regions has a primary series at all), not from the clip. They block
  1,023 kt of primary input that is **100 % fabricated** — zero feed-observed edges, 505 kt of it a
  *domestic* `Concentrate→Smelting` edge in Belgium, a country with no mines. The bands are doing
  their designed faucet-stop job. **Do not widen them.**
- The minimum band width 0.05 is not an absent floor but the `max(0,·)` / `min(1,·)` truncation
  halving the intended ±0.05 margin exactly at the two ends. Measurement says those narrow ends are
  where the bands do their most useful work.

### Sequencing

Q-a and Q-b are **feed** fixes (full Stage 1-4 rebuild); Q-c is a **param** fix (scrap_share v3,
Stage 3-4 only). Q-a changes the level P.0 anchors, so Q-c's bands should be rebuilt *after* it —
otherwise Belgium's band is re-derived from the same inflated series.

Also open: **ROW** remains above 1.05 intake retention (Kazakhstan cleared in P.1).

### Carried out of Round Q, not fixed — the same bug in `stock_bounds_study.py`

`scripts/explo/stock_bounds_study.py` is the other param study that reads a feed's raw
`values.csv`. It clusters regions from the **source** labels and then looks each one up with
`acc.node_index["region"]` (**root** names), so `Korea, Rep.`, `ROW`, `Belgium` and `Taiwan` fall
through to `default_cluster` — whose comment reads "catch-all for regions absent from ICSG", and
they are not absent, only spelled differently. Same fix as ADR 0032's Q-c: map to root regions
when the feed is read. Deliberately left alone in Round Q so the scrap lever stays attributable;
it changes the `stock_bounds` params and so needs its own measured round.

Every other script reading a feed `values.csv` was checked and is region-map aware.


## THE CONCENTRATE EXPORT-MASS GAP (re-measure after Round O)

Round N went looking for a region-resolved concentrate grade and found this
instead. **Half of the mine-chain deletion is arithmetic, not optimisation.**

Copper leaves a mine node only as domestic smelter intake or as export mass
times a grade the hard cone caps at 0.35. So `cap = 0.35·X_mass + U` bounds what
a region can discharge at prior-level intake — and on the shipped run

```
HARD-CAPPED region-years                      144 of 965 with production
forced shortfall Σ max(0, P − cap)        25,913 kt CC = 49.0 % of all deletion
   Chile   14,270 (40.6 % of its own)   ROW 7,840 (92.6 %)   Zambia 1,460 (76.5 %)
```

Chile is hard-capped in **19 of 30 years**. Its BACI concentrate export mass
swings 12,401 (2020) → 4,723 (2021) → 4,088 (2022) → 10,269 (2023) → 4,951
(2024) while its mine output stays flat at ~4,200 kt CC and its anode and
cathode export lines do not move. Three checks put the defect in the **source**:
gross v5 ≡ netted v8 for Chile every year (ADR 0027 never touched it); HS 260300
itself drops, so it is not the sector map; and world concentrate trade excluding
Chile is stable while Chile halves, with China's recorded imports *from Chile*
falling 7,720 → 2,553 kt in the same step. The tonnage is absent from **both**
sides of the bilateral matrix.

This is an **input** defect, which is where all three large wins in this arc came
from (ADR 0022 −26.6 %, ADR 0027 −19.5 %, ADR 0015). Before anything is built it
has to answer:

1. ~~Is the hole Chile's non-reporting…~~ **ANSWERED 2026-09-22, ADR 0033.** Neither: BACI's
   *value* field is intact and agrees with ICSG in every year; its *quantity* field is the broken
   one, and the mass is recoverable from the value. Both vintages agree, so it is not a release
   artifact. ⚠ **Australia→China is NOT a defect** — 808 kt (2020) → 11.4 (2021) → 0.0 (2022) →
   325 (2024), identical in V202501 and V202601. That is China's 2020 halt on Australian copper
   concentrate and the 2024 thaw: real trade, wrongly listed here as a suspicious signature.
2. If the mass is unrecoverable, what is the honest repair — scale the
   region-year's export legs to the copper ICSG says left, widen σ on them, or
   let the mine node carry a stock? Each is a different claim about *who is
   wrong*, and each is measurable.
3. Does ROW's 92.6 % forced share have this cause or is it the aggregate's
   folded internal trade (ADR 0014)? Separate it before it contaminates a repair.

**Do not widen the [0.20, 0.35] band to relieve this.** It would license
concentrate at 60-90 % Cu (that is metal), and ADR 0028 already measured what
moving that wall does.

Detector and full evidence: `scripts/explo/roundN_concentrate_grade_diag.py`,
REP `…roundN-concentrate-grade`, ADR 0029.

## THE DISCHARGE PATH — why the softest thing pays (the other half)

Round M settled what the Chile diagnosis could only hypothesise. The grade bands
do **not** cause the ore deletion; they **shape** it. Closing them relocates the
same contradiction and, on the copper side, relocates it worse:

| | escapes through | Chile legs on the conc floor | US ore retention |
|---|---|---|---|
| base | Chile's ore-grade **ceiling** | 35 % | 0.990 |
| + year-local ore band | Chile's concentrate **floor** | 55 % | 1.307 |
| + narrowed conc band | **US ore** | **66 %** | **1.356** |

**Now bounded at 51 %** of the mine-chain deletion: the other 49 % is the
export-mass gap above, which no re-pricing can reach.

So the question is not where the walls are, it is **why the fluid exists and
who pays**. Chile's concentrate node is over-determined by 3,053 kt CC and the
posterior destroys **13,124 — 4.3x**. A disagreement between two ICSG figures
(a mine total and a smelter total) is settled by deleting the *trade and ore
that connect them*, because that is the softest thing in the neighbourhood:
**leg cv 0.069** after ADR 0017 damping against **scale-total anchor cv 0.017**.

Leads, in the ADR 0016 / 0017 / 0025 family:
- ADR 0016 already measured that *loosening* fabricated anchors collapses the
  chain (CO Mass 0.66 -> 0.41). So the fix is not "loosen the anchors".
- The `a_ref` item above is the same family and has its own measurement.
- A contradiction between two anchors should probably be *visible as a residual
  on those anchors*, not discharged through a third quantity neither of them
  measures. ADR 0019 (contradicted nodes lose anchor authority) and ADR 0020
  (withhold) are the existing precedents for that shape.

Rejected, with a working implementation in commit `f6bd5c9` and the numbers in
ADR 0028: year-local grade bands, and a narrowed concentrate band. Do not
re-propose either until the discharge path is understood; then re-measure
rather than assume.

## Chile's mine chain — the diagnosis that led here

**The instrument is a pair of HARD bands, both pinned.** `value_min`/`value_max` are
not prior shaping — `RatioBoundBuilder` makes them hard cones. Chile sits on the
concentrate *floor* (0.20) in 35 % of its export leg-years, 100 % of the Chile→Japan
lane, and on the ore-grade *ceiling* in 10 years out of 10. The ceiling is permissive
because ADR 0015 made the grade nominal region×year but left the band at the region's
30-year extremes: Chile's cap is 1.1× its **1995** grade, 75 % above its 2019 nominal.
World-wide only 5.9 % of region-years are at that cap; of the 12 biggest ore producers
only Chile (10/10) and China (7/10) are — and they are exactly the two whose ore mass
collapses (0.386, 0.555) while the US, Peru, Zambia and Indonesia hold at ~1.0.

Which gives the loss an exact decomposition:
**ore Mass retention 0.42 = 0.69 (copper deleted) × 0.61 (rock discarded at the cap).**

**Both band fixes were built, measured and REJECTED** (Round M, ADR 0028): the
ore band moved Chile's retention 0.386 → 0.566 and removed its impossible
1.662 % posterior grade, but the deletion relocated to the concentrate floor and
then to US ore. See the discharge-path section above, which is where this went.
