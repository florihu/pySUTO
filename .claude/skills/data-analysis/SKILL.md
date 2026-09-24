---
name: data-analysis
description: Run a deep, evidence-driven root-cause investigation on a pySUTO pipeline run (opt / const / imp output directory) given user observations about what looks wrong. Use this skill whenever the user shares observations about a pySUTO run — flow values collapsing, constraint violations, suspicious z-scores, flat time series, missing coverage, posterior σ surprises — and asks "why", "trace this back", "root cause", "diagnose", "investigate". Also trigger when the user gives a data/proc/{opt,const,imp}/<run_id> path with anything resembling a complaint, even if they don't use the word "analyse". The skill loads the run + upstream parents, computes evidence (z-scores, coverage, edge traces), ranks hypotheses, tests them with Python, and writes a final report under docs/rep/.
---

# pySUTO data analysis

A structured 9-step routine for tracing observations back through the pySUTO pipeline and producing a publishable analysis under `docs/rep/`. The aim is to match (not just gesture at) the depth of existing REPs like [`docs/rep/csxew_root_cause_investigation.md`](../../../docs/rep/csxew_root_cause_investigation.md) and [`docs/rep/analysis_opt_20260521_100428_9d33a0b9_20260526_scaling.md`](../../../docs/rep/analysis_opt_20260521_100428_9d33a0b9_20260526_scaling.md). Read one of those before drafting if you need to recalibrate.

## Inputs

The user provides two things:

1. **Start path** — a run directory under `data/proc/{opt,const,imp,uncprop}/<run_id>/`. The skill auto-detects the stage (opt / const / imp / uncprop) and walks upstream parents via each run's `metadata.json` sidecar.
2. **Observations** — free-form natural language. Multiple observations may be interdependent (e.g. "F* collapses" *and* "ratio rows show z≈−60"). Read them as one system, not three independent complaints.

If the user gave only one of these, ask once for the missing piece — don't guess.

## Project context

pySUTO is a 5-stage pipeline: **feed → agg+imp → const → solver → uncprop**. Architecture, math, and module map are in [`README.md`](../../../README.md) §2–§3. Before drafting hypotheses, skim README §2 for the stage that owns each observation (flow-collapse → Stage 4; coverage → Stage 2; constraint-row weirdness → Stage 3). Re-read these every invocation, because the math evolves and memory of it won't be load-bearing.

## The 9-step routine

This is the spine. Don't skip steps, and don't compress steps 3–7 into a single "I think it's X" — the value of these reports is the *evidence trail*, not the verdict.

### Step 1 — Get the model architecture in your head

Read [`README.md`](../../../README.md) §2 (the stage you're in plus the immediately upstream/downstream one). Identify:
- Which stage's output the start path lives in.
- What `a₀`, `σ_a`, `G`, `c(t)`, `σ_c(t)`, `F*` mean in this run.
- Which knobs in [`config/config.yaml`](../../../config/config.yaml) the run was built with (the sidecar `stage_config` records them — see also Step 2).

### Step 2 — Map persistence

Run [`scripts/load_run.py`](scripts/load_run.py) on the start path. It:
- Detects the stage kind from the directory prefix.
- Loads `metadata.json` (the sidecar).
- Recursively resolves `parents.upstream` to find every upstream run dir.

Output: a dict like `{"opt": Path, "const": Path, "imp": Path, "feeds": [Paths]}`. Use this as the address book for all subsequent loads — never hard-code paths.

For the per-stage on-disk schema (which file holds which array, which dtype, what shape), see [`references/persistence_map.md`](references/persistence_map.md). Re-read it before loading any artefact for the first time — getting `G.npz` vs `a0.npy` wrong wastes a turn.

### Step 3 — Analyse the data itself

Before forming hypotheses, characterise *what* you're looking at. Quantitative beats qualitative. Useful first-pass questions:

- **Distribution shape.** Histograms (log scale) of `|a₀|`, `|F*|`, `|F* − a₀|`, `σ_a/|a₀|`, residuals `G·F* − c`, z-scores. Look for piles at the regularisation floor, the σ_a ceiling, or zero.
- **Stratify.** Group by `(sector_from, valuation)`, by year, by constraint type (`row_meta.type`), by feed coverage. Cross-tab where it helps.
- **Identify the worst N.** Top 10–20 worst-z rows / largest collapses / widest σ_a* — this is where signal concentrates.
- **Coverage.** How many edges have ≥1 feed observation vs. flat-prior fallback? The split per `(sector, valuation)` is often the punchline (see csxew_root_cause_investigation §2.2).

Always table-format these numbers in the report. The user reads the tables first.

### Step 4 — Trace upstream dependencies

Pick the worst case from Step 3 (e.g., one edge at one timestep). Trace it stage by stage backwards:

1. `F_opt` in `data/proc/opt/<run>/timesteps/t_<year>.npz` — year-local on block runs: the value of global edge `edge` is `F_opt[np.searchsorted(cols_t, edge)]` (legacy full-vector runs: `F_opt[edge]`).
2. `a₀[edge]`, `σ_a[edge]` in `data/proc/const/<const_run>/a0.npy`, `sigma_a.npy`.
3. Which rows of `G` reference that edge column? Filter `row_meta.parquet` by joining on the nonzero indices of `G[:, edge]`. List `(constraint_id, type, is_hard, c(t), σ_c(t))` for those rows.
4. Was `a₀` populated from `f_agg` (a real observation) or from `FlatPriorImputer` (the cascade fallback)? Check `imputation_log.parquet` for `edge_id == edge`. If the log has no row, it came in via `f_agg` itself — look at `agg_result.npz` `F[:, edge]` to see which feed contributed.
5. If the upstream contributor is a feed, check the feed's pkl / result.npz: was the source value plausible? was the concordance weight `W` reasonable?

This is the heart of every good REP. Be specific: name the edge_id, the constraint_id, the feed_id, the year. Vague traces are useless.

### Step 5 — Rank hypotheses

Combine the observations (possibly interdependent) and the trace evidence into a ranked list. Each hypothesis must:
- Name a specific mechanism (which line of which module, or which input column).
- Predict an observable signature (so it's testable).
- Acknowledge what evidence would falsify it.

Order by P(this is the cause | evidence). 3 to 5 entries is plenty. If two hypotheses are entangled, say so explicitly rather than ranking them separately.

### Step 6 — Develop a test strategy

For each hypothesis, write the smallest experiment that distinguishes "true" from "false". Examples:

- "If H1 is true, the same z-collapse should appear on every region with `feed_cov[CC]>0 ∧ feed_cov[Mass]=0`." → run that filter.
- "If H2 is true, raising `sigma_a_cv_min` from 0.05 to 0.50 should weaken (but not eliminate) the collapse." → re-solve with the bumped config.

**When to spin out a plan file.** Write a separate `docs/rep/plans/<topic>_plan.md` (using [`references/plan_template.md`](references/plan_template.md)) when *any* of these hold:
- ≥3 hypotheses need testing.
- A test requires code changes (e.g. a new constraint mode, a new imputer step).
- A test requires a re-solve (>5 min wall-clock) or new input data.

Otherwise keep the test strategy inline in the report. Reason: REPs should be readable end-to-end, and short investigations don't benefit from being split.

### Step 7 — Test the hypotheses

Execute the experiments. For each one, write down:
- What you ran (one-liner or notebook cell).
- What you observed (table or single number).
- The verdict: **confirmed / partially confirmed / rejected**.

Be honest about "partially confirmed" — most pipeline bugs are compositions of two things, neither sufficient alone. The csxew REP is the canonical example: A and C rejected, B half-true, root cause was structural asymmetry not in the original list.

Conclude Step 7 with a one-paragraph **verdict** that names the actual root cause (which may differ from your top-ranked hypothesis).

### Step 8 — Develop fix strategies

Propose 2–4 fixes ranked by:

1. **P(it actually fixes the issue)** — based on the verdict and what the experiments showed.
2. **Minimisation of complexity** — one-line config bump > new constraint type > new imputer.
3. **Minimisation of blast radius** — fixes that only affect the failing edges > fixes that change defaults everywhere.

For each fix, write: the change, the expected effect (quantified if possible), and the risk (what else it might break). The user picks; don't apply fixes from the skill itself unless they explicitly ask.

### Step 9 — Write the final report

Always write a report file, even if the verdict is "inconclusive — need more data". Path:

```
docs/rep/analysis_<run_id>_<YYYYMMDD>_<topic-slug>.md
```

- `<run_id>` is the start-path run id (e.g. `opt_20260521_100428_9d33a0b9`).
- `<YYYYMMDD>` is today's date.
- `<topic-slug>` is a short kebab-case description of the observation (e.g. `flat-fstar`, `csxew-collapse`, `posterior-blowup`).

Use [`references/report_template.md`](references/report_template.md) as the skeleton. Match the existing REPs' style: tables for numbers, prose for reasoning, file:line refs for code claims, links back to npz/parquet sources at the bottom.

If you wrote a plan file in Step 6, link it from §6 of the report.

## Helper scripts

These live in [`scripts/`](scripts/) and exist because every past REP reimplemented them:

- [`scripts/load_run.py`](scripts/load_run.py) — `detect_kind(path)`, `load_sidecar(path)`, `walk_parents(path) → {kind: Path}`, `project_root(path)`. Bootstraps Step 2.
- [`scripts/compute_z.py`](scripts/compute_z.py) — `compute_z(const_dir, opt_dir, t) → DataFrame[constraint_id, type, c, sigma_c, Ga, residual, z]`. The first thing to run when the observation is "constraint violations". Joins on `row_meta.parquet` so you get `type` / `is_hard` / `time` columns for free.

Both are pure Python with numpy / pandas / scipy.sparse — no pysuto imports, so they work even mid-refactor.

Anything not covered here is fair game for ad-hoc code in a `python -c '…'` invocation. Don't add new bundled helpers unless the same code shows up in ≥2 REPs.

## Style guide for the report

- **Tables before prose.** The reader looks at the numbers first; the surrounding paragraphs explain *why those numbers matter*.
- **Name specifics.** Use the actual `edge_id`, `constraint_id`, `feed_id`, year. "An edge in the Mining sector" is not specific.
- **Reference code with line numbers.** `constraints.py:592–596` is verifiable; "the constraint builder" is not.
- **Link artefacts.** Every claim about disk state needs a path to the npz/parquet it came from, listed in the report footer.
- **Quantify fixes.** "Drops z from 62 to ~6" beats "should help".
- **Don't hide partial knowledge.** If a hypothesis is half-confirmed or you ran out of compute, say so. The next investigator needs to know where the trail ended.

## Failure modes to avoid

- **Skipping Step 4 (the trace).** A REP without a worked example of one specific bad case is just a histogram. Always trace at least one.
- **Hypothesising before measuring.** Step 3 has to produce numbers before Step 5 can rank anything. If you find yourself writing "I suspect…" in the first paragraph, go back and run a script.
- **Confusing σ_a regularisation floors with bugs.** Many surprises are the `sigma_a_cv_min: 0.05` or `sigma_a_abs_floor: 1e-6` floors firing exactly as designed. Check the config before crying foul.
- **Over-trusting `metadata.json`.** Sidecars record the *intended* config; they don't prove the binaries on disk were built with it. If something looks off, also `np.load` the array and inspect.
- **Writing the report before the verdict.** Don't pre-commit to a conclusion in §0 then write §1–§5 backwards from it. Write §1–§7 in order, then go back and add the executive summary.
