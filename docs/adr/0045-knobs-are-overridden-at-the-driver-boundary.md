# Knobs are overridden at the driver boundary; the pipeline composes drivers as subprocesses

**Status:** accepted (2026-10-21; #100, #103); its condition is measured and holds (2026-10-22, #106).
`opt_20261022_091224_7ce11edb` (no overrides) and `opt_20261022_093350_82d7c31d` (all 66 knobs explicit at their
defaults) reproduce `opt_20261021_131744_58504ad3` in every stored array, `F_opt` in 30/30 years, except one. The
imputation log holds the same rows in another order. That order is drawn per process by a NetworkImputer set
iteration older than this decision (#113). Since `0d49c45` the balances are visited in sorted order: two runs under
different hash seeds reproduce `imp_20261022_111102_1b49a2b3` in every array and write one log (2026-11-02, #113,
`imp_20261102_094925_95744204`).

A sensitivity study varies the stage drivers' tuning constants (**knobs**) across many chains. The
knobs live as UPPER_CASE constants in each driver's `# USER CONFIGURATION` block, read inside `main()`,
with the library layer below taking every one as a kwarg. We decide that a knob is overridden at the
driver boundary — the driver lists its `KNOBS`, accepts `--set KNOB=VALUE` as a type-checked Python
literal and records the pairs in `stage_config["overrides"]` — and that `pysuto.pipeline` runs each
driver as a subprocess through that protocol rather than importing it. The alternative was to move
≈ 65 constants into one config dataclass per stage, passed to an importable `main()`: typed and
testable in-process, but the larger refactor, and the pipeline would then live in `scripts/` (the
data layer never imports it), inherit the drivers' import-time side effects (`load_config`, log
handlers) and hold Stage 3's 0.85 GB and Stage 4's eight solver threads in one process across a sweep.
The boundary keeps every knob's single home in its driver, keeps per-stage logs, frees each stage's
memory between steps and costs ≈ 20 lines per driver; its price is string parsing, a `KNOBS` list kept
by hand (a test checks it against the module), and function defaults bound at import that must be
passed explicitly (`scale_total_cv` in Stage 3). Knobs that must agree across stages are declared in
`pipeline.TIED` and refused when set apart; derived constants are not knobs.

## Amendment (#108, 2026-10-22): a sweep keeps only Stage 4 through a scratch store

The knob study keeps only Stage 4 of each point, which reopened the in-process option. Imp and
const held in memory would not take the ≈ 0.93 GB per point (`imp_20261021_125954_b112b170`
80 MB, `const_20261021_130426_c5331a7f` 851 MB) that limits a full-chain sweep to ≈ 9 points,
with 11.6 GB free and the 3 GB reserve on 2026-10-22. It is rejected again, for the reasons
above. Instead a sweep points its imp and const steps at a RAM-backed scratch store through two
new driver flags (`--imp-dir`, `--const-dir`, defaulting to today's `data/proc` paths), and
deletes a run once no remaining point reuses it. A point then costs the opt run's 0.08 GB, and
#106's gate still covers the drivers.

## Amendment (#186, 2026-11-17): the subprocess half stands, the children run library entry points

The decision stands in both halves, with one change of what the pipeline spawns. Stages 4–6 became
library functions (`pysuto.stages.run_opt`, `run_uncprop`, `run_psut`) beside `run_imp` and
`run_const`, so a step no longer needs a `scripts/` driver: each pipeline step runs in a fresh child,
`python -m pysuto._step <spec.json>`, which resolves the case reference, parses the overrides as
`KNOB=VALUE` literals through the knob protocol (same names, type checks and precedence: defaults,
then the case's `driver_knobs`, then the overrides) and calls the stage function. `python -m pysuto
run` needs no `scripts/` directory. The Stage 4–6 constants moved with their values to
`pysuto.stages`; the drivers list them as `KNOBS`.

**The one-process design of #185 Q4 is rejected on measurement.** It called every step in the
pipeline's own process. Two sweep-like points of `Pipeline(("imp","const","opt"))` on
`opt_20261116_134050_b54f65dc` in one process (RSS, with the peak in brackets, GB; imp and const in
`/dev/shm`) held on to memory:

| step | point 1 | point 2 |
|---|---|---|
| imp   | 11.5 (11.9) | 32.5 (42.9) |
| const | 23.5 (23.8) | 42.9 (43.2) |
| opt   | 33.1 (36.8) | ≥ 46.0 sampled, stopped by hand in the diagnostics at `MemAvailable` < 4.5 GB of 62 GB |

(runs `opt_20261117_134648_3231ec7e` and `opt_20261117_140050_847fbe73`, the second without `diag/`).
The process kept 33.1 GB after one point and peaked at least 9 GB (25 %) higher in the second, so a
sweep could not run in it.

**With a child per step the parent stays at 0.18 GB and the peak is the child's.** Two points
through the new pipeline (parent RSS after every step 183 MB; child peak sampled from `/proc`, GB;
`MemAvailable` never under 41.8 GB):

| step | point 1 child peak | point 2 child peak |
|---|---|---|
| imp   | 11.9 | 11.9 |
| const | 14.8 | 14.8 |
| opt   | 14.0 | 14.3 |

Point 2's peaks equal point 1's within 0.3 GB, against ≥ 9 GB more in one process; the largest step
is const at 14.8 GB. Runs `imp_20261117_141716_e4268743`, `const_20261117_142115_4083005e`,
`opt_20261117_142739_4151e66d` (point 1) and `imp_20261117_143143_1458a219`,
`const_20261117_143538_a7586afb`, `opt_20261117_144150_64744a5d` (point 2). Both opt runs reproduce
`opt_20261116_134050_b54f65dc` in all 240 stored arrays and tables and in `parents.inputs`. The
child design also carries Stages 5 and 6: `python -m pysuto run --case copper --from uncprop --to
psut --base opt_20261116_134050_b54f65dc` made `uncprop_20261117_152125_bcdaa2d8` and
`psut_20261117_152247_1252cb8e`, which reproduce `uncprop_20261116_141543_6be9fa25` (150 arrays and
tables) and `psut_20261116_141702_37174685` (31) with equal `parents.inputs`. With the two opt runs
above that covers Stages 4–6 through children, and the toy chain reproduces Stages 2–6 on the child
design. Copper's Stages 2–3 through children are checked only through their opt runs, which equal the
reference in every array; the imp and const runs of points 1 and 2 sat in a scratch store that is
deleted, and the stored ones (`imp_20261117_115422_d9357d2d`, `const_20261117_115808_9142e955`)
come from the one-process design.
