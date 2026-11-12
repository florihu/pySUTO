# #126: the generic drivers and `python -m pysuto run --case copper` reproduce the final chain through Stage 6

**Run:** `python -m pysuto run --case copper --from imp --to psut --base opt_20261022_112855_928da10c` →
`imp_20261112_100315_c7ad16f3` → `const_20261112_100750_8314bb02` → `opt_20261112_102147_8adfea97` →
`uncprop_20261112_105106_f006729c`, `psut_20261112_105225_8585e456` (label `#126 generic drivers measure`, cold, every
knob at its default; ≈ 52 min wall, of which Stage 5 ≈ 29 min)
**Against:** the final reference chain `imp_20261022_111102_1b49a2b3` → `const_20261022_111531_df61bf5c` →
`opt_20261022_112855_928da10c` → `uncprop_20261029_090754_0446e784`, `psut_20261029_095807_61a70a1a`; for the imputation
log's row order the post-#113 run `imp_20261102_094925_95744204`
**Chain:** the reference's feeds `feed_icsg_20261022_110826_8a031806`, `feed_baci_20261022_110715_fdf3b4ac`,
`feed_crude_ore_20260603_095651_558f7369`, resolved by `--base` through the sidecar parents; main checkout at `9c41c56`,
no other driver running
**Feeds/params:** acc copper v5 (build record `54bc4323`); conc v10, eff v4, ore_grade v8, stripping_ratio v2,
stock_bounds v6 — the reference's params, every `current/` at its latest version
**Date:** 2026-11-12 · **Branch:** `diag/generic-drivers-measure` · **Issues:** #126 (measures #124, #125; closes the
`metal-agnostic` milestone's copper half) · **ADR:** 0034 (accepted in #122), 0046

**Verdict: all four pre-registered items hold; #126 closes and `metal-agnostic`'s done criterion is met.** Run through the
generic `--case` drivers and the one-command chain, copper reproduces the final chain: imp, const and opt bit for bit,
uncprop and psut within rtol 1e-12. With #125's toy case running all six stages under its own labels
(`tests/test_pysuto_run.py`), a second case with its own labels runs through the same drivers without a library fork.

## 1. Change under test

- **#124 (`24a37e9`)**: Stage 2–6 drivers take `--case`; the Stage 2–3 bodies are `pysuto.stages.run_imp` / `run_const`
  over a `CaseSpec`; the library's copper defaults (no-data-anchor exclusions, `"1995"` diagnostic years, Stage 6 primary
  `"CC"`) are gone and copper passes its values explicitly (`driver_knobs`, recorded as `case_knobs`).
- **#125 (`9c41c56`)**: `python -m pysuto run` chains a case from its Stage 1 hook (none for copper yet, #109) to Stage 6,
  passing `--case` and `--data-dir` to every step and the uncprop and psut drivers their defaults.

## 2. Decision items (pre-registered on #126 before the run)

| # | Rule (against the final chain) | Result |
|---|---|---|
| 1 | imp, const, opt: every stored array bit-identical; the imputation log against the post-#113 run, order included | **holds** — const and opt equal in every file (all 30 years' `G` blocks, `c`, `σ_c`, `F_opt`, `eps_opt`, `residuals`, `iter_history`); imp's 7 arrays equal; its log equals `imp_20261102_094925_95744204`'s as stored (against the pre-#113 reference it differs only in #113's documented row order, as in #122) |
| 2 | uncprop: every stored array within rtol 1e-12 (NaN = NaN) | **holds** — no array outside the rule |
| 3 | psut: every stored table within rtol 1e-12, `nodes.parquet` on the reference's columns | **holds** — no table outside the rule (the new `nodes.parquet` adds #119's `role` column) |
| 4 | sidecars keep every `stage_config` key of the reference runs | **holds** — none missing, none changed; added: `case` (all five), `input_pins` (imp), `case_knobs` (opt, psut) |

**Prediction** (all four pass): **right**.

Comparison: `scripts/explo/issue106_driver_gate.compare_dirs` for items 1; for items 2–3 the same walk with
`np.allclose(rtol=1e-12, atol=0, equal_nan=True)` on float arrays and columns and exact equality elsewhere.

## 3. Consequences

- The copper reference chain stays `opt_20261022_112855_928da10c`; this chain is its copy through the generic drivers.
- `metal-agnostic` closes. Its remainder — #26 §2 (relation solver), the solver track #11 → #55 → #56, #52, #112, and the
  §3 remainder (library generalisation of the case algorithms, merged feed drivers, copper's feed step #109) — is in
  `metal-agnostic 2`.
