# Manuscript handover: display items and number files

This file lists every figure, table and number file of the paper, its source and the runs
it comes from. Use it to move the Method, Results and Discussion into the journal
version (Overleaf). Numbers are frozen on the reference chain below (milestone
*manuscript final*, #13).

## 1 The reference chain

| Stage | Run id |
| --- | --- |
| 1 feeds | `feed_icsg_20261124_133729_9d114cf6`, `feed_baci_20261124_133656_107b0feb`, `feed_crude_ore_20261124_133710_d59a325d` |
| 2 imputation | `imp_20261124_140720_ad766e00` |
| 3 constraints | `const_20261124_141044_e73c5952` |
| 4 solver (reference run) | `opt_20261124_141714_e2892160` |
| 5 posterior | `uncprop_20261124_144415_13fea7bf`; ablation `uncprop_20261124_150634_1ebdc2db` |
| 6 tables | `psut_20261124_150658_8f727db2` |
| A5 tolerance pair | `opt_20261124_152425_856f440b` (gap 1e-3), `opt_20261124_152923_0bbd377c` (gap 1e-6), both on the reference constraint run (#315) |
| A6 knob sweep | `sweep_20261124_153029_a4bb2afb`, 64 points around the reference run (#316) |

Inputs: acc copper v5, icsg v19, baci v35 (gross v34), trade_shares v45, crude_ore v3,
conc v10, eff v4, stripping_ratio v2, ore_grade v9, stock_bounds v6, scrap_share v3. SI
Table S17 gives the same chain.

## 2 Figures

All paths are relative to `docs/manuscript/`. Use the PDF in Overleaf: it is vector, with
TrueType fonts embedded. The PNG (300 dpi) is for preview and for the Markdown files.

| Item | File | Shows | Runs | Made by |
| --- | --- | --- | --- | --- |
| Figure 1 | none: Mermaid block in `method.md` §3.1 | the six-stage pipeline | none | draw by hand for the journal |
| Figure 2 | none: Mermaid block in `method.md` §3.2 | the copper system definition, one region | none (nominal parameters) | draw by hand for the journal |
| Figure 3 | `figs/fig3_problem.pdf` | the 2020 QP: edge provenance, chain profile, constraint families, prior breach and active set, China's slice | opt, its const and imp | `results_figs.py` |
| Figure 4 | `figs/fig4_reconciliation.pdf` | Σz² per family, objective shares, retention per stage and year, prior against posterior | opt | `results_figs.py` |
| Figure 5 | `figs/fig5_uncertainty.pdf` | (a–c) contraction, calibration, Birge ratio; (d) termination pair and knob sweep per stage; (e) knob SRRC | uncprop, A5 pair, sweep | `results_figs.py` |
| Figure 6 | `figs/fig6_copper_system.pdf` | secondary share, HHI, Chile/Peru/China trade, embodied extraction | opt, psut | `results_figs.py` |
| Figure S1 | `figs/si_s6_shift_calibration.png` | QQ calibration of the standardised shifts | uncprop | copy of the uncprop run's `diag/ud_r_03_shift_calibration.png` |
| Figure S2 | `figs/si_s6_contraction_shift.png` | contraction against the standardised defect | uncprop | copy of the uncprop run's `diag/ud_r_01_contraction_shift.png` |
| Figure S3 | `figs/si_s15_knob_criteria.png` | the twelve sweep outcomes per point | sweep | copy of the sweep's `diag/sw_r_05_criterion_bands.png` |

Figures S1–S3 exist as PNG only, at 150 dpi: the diagnostic portfolios write no PDF.

## 3 Tables

| Item | Where | Values from |
| --- | --- | --- |
| Table 1 | `method.md` (hand-kept) | versions and counts keyed in `figs/manuscript_numbers.json` (`method:L12`–`L15`, `L119`, `L125`, `L272`–`L308`) |
| Table 2 | `method.md` (hand-kept) | the cited decision records' own measurements; not chain-dependent |
| Table 3 | `results.md` | `figs/table3_ledger.csv` (values, met / not met); brackets from `figs/table3_intervals.csv`; A6 from the sweep's `diag/verdicts.csv` |
| Tables S1–S17 | `si.md` (hand-kept) | keyed in `figs/manuscript_numbers.json` (`si:S*` keys) |
| Tables S18b–S22, S24–S27 | `si.md` | `figs/si_s15_*.csv`, `si_s16_*.csv`, `si_s17_legs.csv` (all rendered in `figs/si_tables.md`) |
| Tables S22b, S22c | `si.md` | the sweep's diag tables, keyed in `manuscript_numbers.json` group 12 |
| Table S22d | `si.md` | `figs/manuscript_checks.json` |

## 4 Number files

| File | Holds | Made by |
| --- | --- | --- |
| `figs/numbers.json` | every number of the Results and Figures 3–6 | `scripts/report/results_figs.py` |
| `figs/manuscript_numbers.json` | the Method and SI numbers; each key names its location | `scripts/report/manuscript_numbers.py` |
| `figs/manuscript_checks.json` | the Discussion's check measurements (SI S15.5) | `scripts/report/manuscript_checks.py` |
| `figs/table3_intervals.csv`, `.run.json` | Birge-scaled 95% intervals of Table 3 and Table S19, with the runs they came from | `scripts/report/table3_intervals.py` |

## 5 Refresh onto a later chain

Run the scripts from the repository root, in this order. Then compare each JSON file
with git HEAD, key by key, and edit the text at each changed key's location.

1. `python scripts/report/table3_intervals.py <opt> <uncprop>`
2. `python scripts/report/results_figs.py <opt> --psut-run-id <psut> --tol-pair <opt 1e-3> <opt 1e-6> --uncprop-run-id <uncprop> --sweep-id <sweep>`
3. `python scripts/report/manuscript_numbers.py <opt>`
4. `python scripts/report/manuscript_checks.py --opt <opt> --uncprop <uncprop> --uncprop-ablation <ablation> --old-opt opt_20260930_092111_e3e0677d --json docs/manuscript/figs/manuscript_checks.json`
5. Copy Figures S1–S3 from the new runs' `diag/` folders (Section 2).

## 6 Number freeze

The numbers above are final for the submission. Do not re-run the chain for a defect
that changes no outcome of Table 3. If a defect flips a Table 3 outcome, the author
decides. After the freeze, a change to the Results means new figures (Section 2) and
the edited text; the Method and SI numbers follow from step 3.

## 7 Left to the author

- The prose: the Method, Results and Discussion are bullet points.
- The `verify:` markers: literature checks for the *Compare:* items.
- The two `check:` markers of the outlook (`discussion.md`, Section 5.6).
- Figures 1 and 2 in the journal's style.
