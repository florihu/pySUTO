# #201: a covered region-year with no ICSG mine says zero on its mine output; phantom mine output falls 1,324 → 244 kt CC, at +23.3 % common-set sum-row Σz²

**Run:** the ICSG feed, then Stages 2–4, from a frozen copy of main `e268365`, with the main checkout as the working
directory. One driver at a time, under the stage lock:
- `icsg_feed.py` → `feed_icsg_20261119_125015_e2e9520e`.
- `collect_aggregate_impute.py` with that feed, the base's BACI and crude-ore feeds, and
  `--pin-inputs-from imp_20261116_133053_ad1191d2` → `imp_20261119_125259_f2b11cd5`.
- `constraints.py` → `const_20261119_125651_ff7bfc9c`.
- `run_solver.py` → `opt_20261119_130302_b20f0dba`: 30/30 solved, cold, `solve_wall_s` 95.
- On ship, `python -m pysuto run --from uncprop --to psut --base opt_20261119_130302_b20f0dba` → `uncprop_20261119_133344_08dfa15b`, `psut_20261119_133504_4c783a88`.
- Label `#201 mine zeros`; every knob at its default except the one under test.

**Against:** the reference chain `imp_20261116_133053_ad1191d2` → `const_20261116_133444_28ef32f2` →
`opt_20261116_134050_b54f65dc` (*base* below).
**Chain:** the base's `feed_baci_20261022_110715_fdf3b4ac` and `feed_crude_ore_20260603_095651_558f7369`, with a new ICSG
feed in place of `feed_icsg_20261022_110826_8a031806`. That base ICSG feed was built on acc copper v4; the new one was
built on v5, which the base imp already used. Item 1 shows the rebuild moves nothing but the stamped edges.
**Feeds/params:** icsg v18, trade_shares v41, icsg_aggregates v4, crude_ore v3, ore_grade v8, acc copper v5. The imp's
and const's `parents.inputs` equal the base's (conc v10, eff v4, ore_grade v8, stripping_ratio v2, stock_bounds v6,
scrap_share v3). `input_status.py`: every `current/` at its latest version.
**Date:** 2026-11-19 · **Branch:** `diag/mine-zeros-measure` · **Issues:** #201 (measures #200; follow-up of #183 §1)
· **ADR:** 0024 (the observed zeros, amendment #200), 0025 (stop rule), 0019

**Verdict: ships on the pre-registered rule. Required items 1 (stamp) and 2 (all years Solved) hold.**
`opt_20261119_130302_b20f0dba` becomes the reference chain.
- **The decision:** the maintainer decided on 2026-11-19 that a region-year ICSG covers and lists no mine for mines
  nothing. Round-E and Σz² were pre-registered as reported, not gating. That is the gate under which #97 rejected this
  zero in 2026-10.
- **Mine output:** CO→Mining posterior in the 456 stamped region-years falls 1,323.9 → 243.5 kt CC (−81.6 % vs
  `opt_20261116_134050_b54f65dc`). Belgium falls 434.6 → 8.6 and Italy 222.3 → 14.6.
- **Round-E holds anyway.** World Concentrate CC retention rises 1.011 → 1.021, because real miners supply what the
  phantoms did.
- **The cost:** common-set sum-row Σz² rises 774,318 → 954,993 (**+23.3 %** vs `opt_20261116_134050_b54f65dc`).
  - Italy accounts for +108,563 of it, on its Matte use totals.
  - Most of the rest sits on the use-total outputs of the hubs' Concentrate and Matte nodes: UAE, Singapore, UK,
    Taiwan, Switzerland and France. Their `c` is the sum of the hubs' BACI export legs.
  - These rows ask the hubs to pass on flows that no recorded supply now covers. That is the data contradiction of
    #183 §3, which σ cannot repair (ADR 0019).
- **The soft zero is not airtight.** At σ 0.3 kt CC it leaves 200.9 kt CC on the stamped MF→Concentrate edges, with
  53 region-years above 1 kt (max |z| 15.6). Where a hub's use total outweighs the zero, the solver still buys a
  little mine output: the UAE keeps 48.0 kt CC over 30 years, Switzerland 23.5.

## 1. Change under test

- **#200 (`be29910`, merged `e268365`):** `PRODUCTION_EDGES` = `(("MF", "Concentrate"), ("SXEW", "CSXEW"))`.
  - Before, the tuple held SX-EW only (#97).
  - A covered region-year with no raw ICSG mine production now carries an observed zero on its MF→Concentrate CC edge,
    at σ 0.3 kt CC (ADR 0025).
  - The zero is not stamped where the crude-ore feed reports ore, or where an ore-grade row starts at CO (ADR 0019).
  - The efficiency bands and the mass balances carry the zero to Ore, Mining and CO→Mining.
- **History:** #97 measured the same edge on 2026-10-21 and rejected it: `opt_20261021_105350_b1fea245`, +37.8 %
  common-set Σz² together with #96's cathode legs. That was before #153 removed the phantom chains from the prior.
- **Dry count (posted on #201 before the run):** 456 region-years in 20 regions; 60 spared by the guard; no producer
  stamped.

## 2. Decision items (pre-registered on #201 before the first driver)

| # | Item | Rule | Result | |
|---|---|---|---|---|
| 1 | Stamp | the feed differs from the base feed only on the stamped edges; no producer, guarded or uncovered region-year stamped (required) | `observed_zero` and `sigma_root` differ on 456 entries, all of them newly stamped MF→Concentrate CC edges (F_root 0, σ 0.3). `F_root`, `cv_root`, `cv_src` and `row_penalized` are identical. Stamped producers 0, guarded 0, uncovered 0 | ✓ |
| 2 | Solve | every year Solved (required) | 30/30 solved, `solve_wall_s` 95 (base 93) | ✓ |
| 3 | Mine output | reported | CO→Mining CC, stamped region-years: prior 796.4 → 0.0, posterior 1,323.9 → 243.5 kt CC (−81.6 %). MF→Concentrate CC: posterior 1,077.1 → 200.9 (−81.3 %). World CO→Mining posterior 639,488 → 640,905 (+0.22 %) | — |
| 4a | World retention | reported (#149 threshold: falls < 0.01) | CO Mass 0.999 → 0.999, Concentrate CC 1.011 → 1.021, Cathode CC 0.997 → 0.994 | holds |
| 4b | Small nodes (|c| < 18 kt) | reported (moves < 0.01) | median posterior/prior 1.000 → 1.000 (p90 1.125 → 1.037) | holds |
| 4c | Exporter retention | reported (falls < 0.01, or rises) | Chile 1.022 → 1.033, Peru 0.978 → 0.982, Indonesia 0.917 → 0.922 | holds |
| 4d | Mass moved, z² by anchor band | reported | mass moved 259,241 → 257,746 kt (−0.58 %). Δz² by band: < 10 kt +100,697; 10–100 kt +83,522; 100–1,000 kt +2,869; ≥ 1,000 kt −6,414 | — |
| 5 | Objective | reported | common-set sum-row Σz² 774,318 → 954,993 over 26,918 rows (+23.3 %). Ratio rows 4,957 → 4,929 (−0.6 %) | — |
| 6 | Stock legs | reported | above \|z\| 5: 100 → 88. Newly above: 2 (ROW MESC CC and USC CC, −0.2 kt CC each). No longer above: 14 | — |

All percentages are vs `opt_20261116_134050_b54f65dc`. Numbers come from
`scripts/explo/issue201_mine_zeros_diag.py opt_20261116_134050_b54f65dc opt_20261119_130302_b20f0dba`.

## 3. What moved

**Mine output by region** (CO→Mining CC, Σ 1995–2024, kt CC, base → new posterior):

| Region | Prior (base) | Posterior base → new |
|---|---|---|
| Belgium | 479.3 | 434.6 → 8.6 |
| Italy | 221.7 | 222.3 → 14.6 |
| United Arab Emirates | 0.0 | 158.1 → 48.0 |
| Switzerland | 0.0 | 78.0 → 23.5 |
| United Kingdom | 24.9 | 69.6 → 18.2 |
| Singapore | 0.0 | 61.4 → 22.2 |
| Austria | 48.4 | 56.1 → 17.2 |
| Taiwan | 0.0 | 53.1 → 20.9 |
| Netherlands | 0.0 | 42.5 → 15.3 |
| France | 0.0 | 29.1 → 16.4 |
| Malaysia | 0.0 | 30.7 → 16.4 |
| 9 others | 22.1 | 88.5 → 22.2 |

- **Where the prior had a mine:** Belgium, Italy, Austria, the UK and Botswana had imputed mine priors, and the zero
  removes them. Belgium's and Italy's mine output nearly vanishes.
- **Hubs with no mine prior:** the UAE, Switzerland, Singapore, Taiwan and the Netherlands keep 30–39 % of their
  phantom output, and France 56 %. Their Concentrate and Matte outflows are export legs only. For example, the UAE's
  Concentrate outflow prior is 132.1 kt CC of exports and 0 domestic. Those use totals outweigh a zero at σ 0.3 kt.

**Where the Σz² lands** (common-set sum rows, Δz² by region): Italy +108,563, UAE +16,195, Singapore +14,898,
UK +14,176, Taiwan +13,593, Switzerland +8,010, France +7,089, Belgium +6,379, Netherlands +5,962. Zambia (−9,563),
Malaysia (−5,676) and Slovakia (−2,261) fall.
- **Italy:** 58 Matte use-total rows (Mass and CC) carry +114,574.
  - Its smelter took 170.7 of its 183.3 kt CC prior concentrate intake from its own imputed mine.
  - Its matte went to CFR: 182.9 kt CC prior.
  - With the mine at zero, that matte has no supply.
- **The hubs:** Concentrate use totals (UAE 2024: c 27.2 kt CC, z² 2,224) and Matte use totals (Singapore, Taiwan).
- **These rows are scale totals** (ADR 0006, 0016). Their `c` is the model's own prior group-sum, and that sum includes
  the BACI export legs of hubs with no recorded domestic supply. The zero exposes the contradiction; it does not
  create it.

**The zero's own price:** on the 456 stamped edges, the posterior is 200.9 kt CC. Of them, 53 are above 1 kt and 44
have |z| > 5 (max 15.6); the prior term there is Σz² 5,994.

## 4. Predictions (pre-registered)

| | Prediction | Result | |
|---|---|---|---|
| P1 | the new ICSG feed differs from the base feed on exactly the 456 MF→Concentrate CC edges | exactly those 456, on `observed_zero` and `sigma_root` | right |
| P2 | all 30 years Solved | 30/30 | right |
| P3 | CO→Mining in the stamped region-years falls by more than 90 %, and each of #183's six hubs below 10 kt CC | −81.6 %. The hubs keep 15.3–48.0 kt CC | wrong |
| P4 | common-set Σz² rises +5 % to +30 %, mostly on Matte use totals (Italy) and the hubs' re-export anchors (Netherlands, UAE, Switzerland, Taiwan, Malaysia) | +23.3 %. Right on Italy's Matte use totals and on the UAE, Switzerland and Taiwan, whose use totals are their export legs. Singapore and the UK were not predicted. The Netherlands is small (+5,962), and Malaysia falls (−5,676) | partly right |
| P5 | Round-E 3a–c hold | they hold | right |
| P6 | fewer stock legs newly above \|z\| 5 than #97's 95 | 2 | right |

P3 was wrong because the zero is soft: at σ 0.3 kt CC, a hub whose Concentrate use total is anchored in BACI exports
keeps one or two kt CC a year of mine output.

## 5. Follow-ups

- **#183 §3, the re-export anchors.** The hubs' use totals now carry the contradiction in the open: UAE, Singapore,
  Taiwan, Switzerland and France, +60,000 Σz² in all. Closing it is a netting question (ADR 0026, 0027), not a σ one,
  and it waits until after v0.1.0.
- **Italy's Matte use totals** (+108,563 for Italy in all) are the largest single block. Italy's smelter and CFR chain
  ran on concentrate from its own imputed mine. Whether the inputs book Italian smelting and refining that its
  concentrate imports cannot supply is a data check for #183 §3.
- **#184:** re-point the manuscript to this chain. Keyed numbers move: Σz² +23.3 % and mine output by region. The
  phantom-deletion prose must be re-checked.
- **#156:** pin this chain as the final review's reference chain (#155, amendment 4).
