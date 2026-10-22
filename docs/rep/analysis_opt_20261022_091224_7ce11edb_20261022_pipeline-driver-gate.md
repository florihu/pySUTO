# #106: the knob surface, the stage lock and the pipeline move nothing — every array of both chains and F_opt in 30/30 years bit-identical; items 1–2 fail as worded only on the imputation log's row order, a pre-existing hash-order effect in NetworkImputer

**Run:** `opt_20261022_091224_7ce11edb` (chain 2, label `r106 zero overrides`; imp `imp_20261022_085416_ba317c64`,
const `const_20261022_085845_1da3435c`; `solve_wall_s` 547.2 s) and `opt_20261022_093350_82d7c31d` (chain 3,
`r106 explicit defaults`; imp `imp_20261022_091559_5a5d3ef7` and const `const_20261022_092029_717c7396` written to the
scratch store `/dev/shm/pysuto-gate`; 529.9 s)
**Against:** `opt_20261021_131744_58504ad3`, the final reference chain (#110; imp `imp_20261021_125954_b112b170`, const
`const_20261021_130426_c5331a7f`; 529.3 s)
**Chain:** the reference's feeds `feed_icsg_20261021_125725_b033ebc4`, `feed_baci_20261021_125615_bc691704`,
`feed_crude_ore_20260603_095651_558f7369`; both chains run by `pysuto.pipeline.Pipeline` in the main checkout at
`e668821` (`feat/pipeline-sweep` merged into main), one after another, no other driver running. From 09:01:22 to
09:13:22 another session held the checkout on its own branch at the same commit (git reflog). Its two commits there
touched docs, `scripts/report/manuscript_numbers.py`, tests and a commit-msg hook, none of the files the drivers read
(`src/`, the three drivers)
**Feeds/params:** the reference's (acc copper v4; conc v10, eff v4, ore_grade v8, stripping_ratio v2, stock_bounds v6,
scrap_share v3)
**Date:** 2026-10-22 · **Branch:** `diag/pipeline-gate-measure` · **Issues:** #106 (gates #100, #101, #103, #104 and
#108's driver flags) · **ADR:** 0045 (its acceptance condition)

**Verdict: holds for every array; items 1 and 2 fail as worded on one file.** Both chains reproduce the reference bit for
bit in 527 of their 528 stored arrays. The 528 are every npz key, npy array and parquet frame of the imp, const and opt
runs, `F_opt` in all 30 years included. Items 3 and 4 hold. The 528th is `imputation_log.parquet`. It holds the same
793,237 rows as the reference's, identical once sorted, but 15,665 (chain 2) and 15,651 (chain 3) rows sit in another
order. The order comes from a set iteration in `NetworkImputer` that has been there since `e9f3874` (2026-06-09) and
follows each process's string-hash seed. Chain 2, which sets nothing, differs from the reference as much as chain 3
does, and chains 2 and 3 differ from each other. So the effect is not one of the build tickets'. ADR 0045's condition,
that the zero-override and explicit-defaults chains reproduce the reference opt run bit for bit, is met.

## 1. Change under test

- **#100:** each Stage 2–4 driver lists its `KNOBS`, takes `--set KNOB=VALUE` (a type-checked literal) and
  `--run-id-file`, and records the pairs in `stage_config["overrides"]`.
- **#104:** `CASCADE_ORDER` and the `FEASIBILITY_CHECK` mode replace the hard-coded cascade and the first-year bool.
- **#101:** every stage driver takes `data/proc/.stage.lock`.
- **#103:** `pysuto.pipeline` runs the drivers as subprocess steps.
- **#108:** `--imp-dir` and `--const-dir` let a sweep keep its imp and const runs in a scratch store.

Chain 3 passes all 66 knobs at their defaults, read from each driver's `KNOBS`: imp 28, const 26, opt 12. This includes
`EFFLESS_STOCK_SIGMA_KT` and `STOCK_BOUNDS_SKIP_REPORTED`, which joined `KNOBS` at the merge. Chain 3's imp and const
runs go through the scratch store, the path a sweep that keeps only Stage 4 takes. Neither chain skips the probe:
`FEASIBILITY_CHECK` stays `"first"`.

## 2. Decision rule and result

The rule was fixed in the docstring of `scripts/explo/issue106_driver_gate.py` before either chain ran. Chain 2 started
at 08:51:49, chain 3 ended at 09:33:50 (opt run ids), and script and result share one commit. The comparison walks every
file of the reference's run directories except `metadata.json` and `diag/`. It checks npz and npy arrays with
`np.array_equal` (NaN equal to NaN; structured and object arrays field by field) and parquet frames with
`DataFrame.equals`. Run against itself, the reference is identical in all 528 arrays, and the previous chain's opt
(`opt_20261021_121332_acc68135`) shows the comparison catches a real difference: `F_opt` differs in 30 of 30 years.

| Item | Rule | Result |
|---|---|---|
| 1 | Chain 2: every stored array equals the reference's | **Fails as worded.** imp 7/8 (`agg_result.npz` all 7 keys identical, `imputation_log.parquet` not), const 280/280, opt 240/240; `F_opt` identical in 30/30 years |
| 2 | Chain 3: the same | **Fails as worded.** The same pattern: imp 7/8, const 280/280, opt 240/240, `F_opt` 30/30 |
| 3 | `overrides` `{}` on chain 2, the full default dict on chain 3; `feasibility_check` `"first"` | **Holds.** Chain 2: `{}` on all three runs; chain 3: 28, 26 and 12 pairs, equal to the drivers' defaults as JSON; both opt runs `"first"` |
| 4 | No stage lock left after either chain | **Holds.** None after chain 2, none after chain 3 |

The per-year table is `docs/rep/figs/issue106/f_opt_equality_opt_20261022_091224_7ce11edb_opt_20261022_093350_82d7c31d.csv`.
The opt runs' `iter_history`, residuals, `eps_opt` and `c_star` are identical too, so every period solve took the same
iterations to the same point.

## 3. The one difference: the order of the NetworkImputer's log rows

All three logs have 793,237 rows and the same 13 columns and dtypes. Sorted by every column, chain 2's, chain 3's and the
reference's are identical. The rows out of place all lie between index 23,733 and 65,274, and all are rows that
`network` wrote in steps 1 and 6, passes 1–2: 15,665 of them in chain 2 and 15,651 in chain 3, so the two chains differ
from each other as well.

`NetworkImputer._flow_balances` (`src/pysuto/impute/network.py:342`) builds its list of flow balances from
`by_to.keys() & by_from.keys()`, a set of `(node_id, time)` tuples whose `time` is a string. A set's iteration order
follows the hashes of its members, and Python seeds string hashes per process, so every run visits the balances in its
own order and logs their fills in that order. In these three orders, two of them drawn today and one on 2026-10-21, the
values do not depend on the order: `f_agg`, `sigma_agg` and every array downstream agree bit for bit. Nothing here shows
whether some input could make the order change a value, for instance two balances that could fill the same edge first.

## 4. Consequence

- **What the gate certifies:** the build tickets change no number the solver, Stage 5 or Stage 6 reads.
- **#106 closes on this reading** (the maintainer's decision, 2026-10-22): items 1 and 2 fail on the log's row order
  alone, an effect older than the build tickets.
- **Follow-up #113:** visit the balances in sorted order (`sorted(by_to.keys() & by_from.keys())`), so that the
  imputation log, and any value the order could ever touch, is reproducible across processes. That is a change to an
  imputer, so it needs its own check that every array stays bit-identical to this reference.
- **What the gate wrote:**
  - Chain 2 wrote 1.0 GB to `data/proc`: imp 80 MB, const 851 MB, opt 74 MB. Its imp and const runs duplicate the
    reference's arrays; pruning them is the maintainer's call.
  - Chain 3 wrote its 74 MB opt run to `data/proc`. Its imp and const runs stayed in the scratch store.
  - 11 GB were free after the gate.
- **Next for the pipeline work:** the #108 sweep (64 points, ≈ 11 h), which waits for the maintainer.
