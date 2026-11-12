# #122: roles (#119), the CaseSpec (#120) and the case algorithms (#121) reproduce the final chain bit for bit

**Run:** `imp_20261112_090241_476e64a7` → `const_20261112_090715_91b9df0b` → `opt_20261112_092114_c8cb8b74`
(label `#122 roles+casespec measure`, cold, every knob at its default; Stage 4 solve wall 563 s)
**Against:** the final reference chain `imp_20261022_111102_1b49a2b3` → `const_20261022_111531_df61bf5c` →
`opt_20261022_112855_928da10c`; for the imputation log's row order also the post-#113 runs
`imp_20261102_094925_95744204` and `imp_20261102_095209_86454f46`
**Chain:** the reference's feeds `feed_icsg_20261022_110826_8a031806`, `feed_baci_20261022_110715_fdf3b4ac`,
`feed_crude_ore_20260603_095651_558f7369`, through `Pipeline(("imp", "const", "opt"), base=Chain.from_opt_run(…))`
in the main checkout at `566e2b7`, no other driver running
**Feeds/params:** acc copper **v5** (v4 plus the `role` column of `root.ods`, build record `54bc4323`); conc v10, eff v4,
ore_grade v8, stripping_ratio v2, stock_bounds v6 — the reference's params; every `current/` equals its latest version
**Date:** 2026-11-12 · **Branch:** `diag/roles-casespec-measure` · **Issues:** #122 (measures #119, #120, #121) ·
**ADR:** 0034 → accepted

**Verdict: all four pre-registered items hold; #122 closes and ADR 0034 is accepted.** Binding copper's labels to roles,
moving the copper model into a `CaseSpec` and its algorithms into `pysuto.cases.copper` change no number: every array
stored by the imp, const and opt runs equals the reference's bit for bit, `F_opt` in all 30 years included. The only
stored difference is the imputation log's row order (15,649 `network` rows), which is #113's documented effect on the
reference — the new log equals the post-#113 runs' logs exactly, row order included.

## 1. Change under test

- **#119 (`d07e54a`)**: a case binds its labels to framework roles in the acc spec; the library reads roles, not labels;
  `load()` returns the masks the build made; stages compare acc versions by spec. Copper's acc rebuilt at v5 right after
  the merge: every file of `data/proc/acc/copper` equals the v4 build by sha256 except `metadata.pkl` (roles and rule
  tables added); the final chain's PSUT loads under v5.
- **#120 (`f73b5ec`)**: `pysuto.case.CaseSpec`; copper's Stage 2–3 knobs, cascade and builders in `pysuto.cases.copper`.
- **#121 (`525d1e4`)**: the Stage 2–3 script algorithms moved unchanged behind the case's hooks.

## 2. Decision items (pre-registered on #122 before the run)

| # | Rule (against the final chain) | Result |
|---|---|---|
| 1 | imp: the 7 arrays of `agg_result.npz` bit-identical | **holds** — `F`, `Omega`, `Sigma`, `coverage`, `f_agg`, `feed_ids`, `sigma_agg` equal (NaN = NaN). The imputation log holds the same 793,242 rows (multiset-equal, method counts equal); 15,649 rows sit in another order, all `network_*` methods — exactly the #113 REP's count for the reference, written before `0d49c45` in its process's hash order. Against both post-#113 runs every stored file is equal, log order included |
| 2 | const: `G`, `c`, `σ_c`, `σ_a`, `row_meta` bit-identical | **holds** — `G.npz`, `a0`, `l`, `u`, `sigma_a`, `row_meta.parquet`, all 30 `blocks/g_<year>` and `blocks/edge_cols`, all 30 `timesteps/t_<year>` (`a_prior`, `c`, `sigma_c`) equal |
| 3 | opt: `F_opt` bit-identical per year, Σz² equal | **holds** — `F_opt`, `eps_opt`, `c_star`, `residuals`, `a_prior`, `rows_t`, `cols_t`, `iter_history` equal in all 30 years; Σz² is therefore equal |
| 4 | sidecars keep every `stage_config` key, plus `case` | **holds** — imp adds `case` and `input_pins` (#53), const adds `case`, opt unchanged; no key missing |

**Prediction** (all four pass with exact equality): **right**.

Comparison: `scripts/explo/issue106_driver_gate.py`'s `compare_dirs` and `log_as_multiset` (every stored array; npz/npy by
`np.array_equal` with NaN equal to NaN, parquet by `DataFrame.equals`).

## 3. Consequences

- The copper reference chain stays `opt_20261022_112855_928da10c`: the new runs are copies of it under acc v5.
- `metal-agnostic`'s remaining copper check is #126 (the generic drivers and `pysuto run --case copper`).
