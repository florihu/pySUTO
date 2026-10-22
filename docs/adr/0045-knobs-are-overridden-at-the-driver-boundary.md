# Knobs are overridden at the driver boundary; the pipeline composes drivers as subprocesses

**Status:** accepted (2026-10-21; #100, #103). Accepted on the condition that #106's zero-override and
explicit-defaults chains reproduce the reference opt run bit for bit; a difference reopens it.

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

**Amendment (2026-10-22, #108).** The knob study keeps only Stage 4 of each point, which
reopened the in-process option: imp and const held in memory would not take the ≈ 0.93 GB per
point that limits a full-chain sweep to ≈ 9 points on 12 GB of free disk. It is rejected again,
for the reasons above. Instead a sweep points its imp and const steps at a RAM-backed scratch
store through two new driver flags (`--imp-dir`, `--const-dir`, defaulting to today's
`data/proc` paths), and deletes a run once no remaining point reuses it. A point then costs the
opt run's 0.08 GB, and #106's gate still covers the drivers.
