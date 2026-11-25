# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.


## 5. Explain for Understanding

**The human must understand your output. Pick the format that makes it easiest to understand.**

As LLMs do more of the legwork, the human's work moves up to oversight and understanding.

Each format below is better than the one before it:
1. **Writing** - Write in ASD-STE100, a controlled language from aerospace maintenance documentation. The spec is stringent, so aim for about 80% of the way to >
2. **Diagram** - Instead of writing, make a diagram. A diagram is easier to process, parse, and understand.
3. **Web page** - Output HTML to make a beautiful, interactive web page, with animations where they help.
4. **Explainer video** - Make a custom explainer video, for example in 3Blue1Brown style. For narration, use the user's TTS API key (e.g. ElevenLabs), or find a>

Intelligence and code are now abundant. So large, custom, discardable artifacts, such as web apps and explainer videos, now make sense. Push the boundaries.

These artifacts do not conflict with Simplicity First and Surgical Changes. Those principles govern the code you ship. Keep artifacts out of the codebase unless>

Writing rules (from ASD-STE100):
- Procedural sentences: at most 20 words. Descriptive sentences: at most 25 words.
- One instruction in each sentence, except for simultaneous actions.
- At most 6 sentences in each paragraph. Write one topic in each paragraph.
- Noun clusters: at most 3 words.
- Verbs: use the imperative and the simple present, past, and future. Do not use the progressive ("is closing") or the perfect ("has closed").
- Use the active voice in procedures.
- Each word has one meaning and one part of speech. Use the same word for the same thing every time.
- Use approved words: "use", "start", "make sure", "before", "about", "to" - not "utilize", "commence", "ensure", "prior to", "approximately", "in order to".
- Do not leave out words like "the", "a", and "this".
- Use vertical lists for complex text.
- Safety instructions: first a clear, simple command, then the risk. WARNING means a risk of injury. CAUTION means a risk of damage.
- Technical names, such as code identifiers, are allowed. Write them exactly.

The test: Does the user understand your output after one read?


---

# pySUTO — project-specific facts

Supply–use-table reconciliation (copper application): a σ-weighted QP solved in Clarabel at the end of a six-stage pipeline (numpy · scipy.sparse · pandas). Architecture and module map are in [docs/architecture.md](docs/architecture.md), the run workflow in [docs/workflow.md](docs/workflow.md), vocabulary in [GLOSSARY.md](GLOSSARY.md), decisions in `docs/adr/`, open work in GitHub issues and milestones. The code is of three kinds: the **library** `src/pysuto` holds the method, generic for every case; the **application** `scripts/` holds everything that carries copper's name (GLOSSARY.md "Application"); repo tooling goes in `tools/` (from #303 on). Re-measure, do not assume (ADR 0028): a claim about a run is backed by a run id and a number.

## Pipeline

- Pipeline, one driver per stage: `scripts/acc.py --id copper` → parse (`scripts/parse/`) → feeds `scripts/*_feed.py` (1) → `collect_aggregate_impute.py --feed-id …` (2) → `constraints.py --imp-run-id <id> <label>` (3; label is positional) → `run_solver.py --const-run-id <id>` (4; also renders diag) → `uncertainty_propagation.py <opt_run_id>` (5; also renders diag; re-render with `uncprop_diag.py <uncprop_run_id>`) → `psut.py <opt_run_id> [--sankey 2020]` (6); `python -m pysuto run --case <alias|module|path> --data-dir <dir> --from <stage> --to <stage> [--base <opt_run_id>]` (copper: `--case scripts/copper.py`, which the `scripts/` drivers default to) chains them (`--case` is required: a bundled alias, a dotted module or a file path, resolved by `pysuto.case.resolve_case`; sidecars record `case_source` and `case_sha256`, #187; `case_sha256` is the identity, `case_source` is informative: `path:<abspath>` by design, so an ephemeral path outside the repo is no defect), each step in a fresh child process (the Stage 2–6 bodies are `pysuto.stages.run_imp … run_psut`, the scripts thin wrappers; copper's Stage 1 hook is `stage1` in `scripts/copper.py`, which calls `copper_feed.run_stage1`: it runs the feed builds in the load-bearing order, while the BACI parse, the gross registration and `icsg_clean.py --to-feed` stay manual).
- Run ids are `{feed|imp|const|opt|uncprop|psut|sweep}_{YYYYMMDD_HHMMSS}_{8hex}`; feed runs embed the feed name (`feed_<feed_id>_…`).
- Per-stage `registry.json` is a trimmed index. Lineage lives in each run's `metadata.json` (`parents.inputs` = input versions, `parents.upstream` = parent run ids) — read the sidecar for provenance.
- Feed build order is load-bearing (ADR 0024, 0027, 0030 am.): BACI parse → register gross baci vN → `icsg_clean.py --baci-version N --to-feed` → `build_baci_reexport.py --source-version N` → `baci_feed.py` → `build_trade_shares.py` → `icsg_feed.py`; `build_ore_grade.py` (hand, after the feeds) feeds back into `icsg_feed.py`'s observed zeros and records its parents (`docs/architecture.md` Stage 1). Hand runs of the builders and `icsg_clean.py` take the stage lock. The netting refuses an ICSG input parsed against another N (#63); `icsg_feed.py` refuses trade-share weights whose `parents.json` names another BACI feed run or input version (#53; stale ones once cost 6.2 pp of z²).
- `run_solver.py --prior-mode` and `stages.run_opt` default to `cold`; `run_solver.main(prior_mode=)` and the `Solver` class default to `warm`. Trust the sidecar's `prior_mode`.

## Environment & commands

- Interpreter: conda env `pysuto`, Python 3.9 (`environment.yml` builds a development env on any platform); `environment-reference.yml` pins its every build and is the reference environment, where bit-identity is claimed (#194). Write 3.9-safe code: no `match`, `zip(strict=)` or `itertools.pairwise`; `X | None` only in annotations under `from __future__ import annotations`.
- Packaging is hatchling + hatch-vcs: the version comes from the latest `v*` git tag (`0.0.0+unknown` without git metadata), and `pip install -e .` gives the `pysuto` command. The `pysuto` env's editable install points at the main checkout: before you start lanes, and in a git worktree, read [docs/agents/parallel-worktrees.md](docs/agents/parallel-worktrees.md) first. The feed builders put only `scripts/` on `sys.path` and would import the main checkout's `pysuto` silently.
- Run the `scripts/` drivers from the repo root: the feed drivers (`icsg_feed.py`, `baci_feed.py`, `crude_ore_feed.py`) call `load_config("config/config.yaml")` in `main()` and use `data/`; the Stage 2–6 drivers read `--data-dir` (default `data`, their stage lock is `<data-dir>/proc/.stage.lock`) and Stages 2–3 `--config` (default `config/config.yaml`), relative to CWD. Importing `pysuto` writes nothing (#190): `get_logger()` has no file handler until a driver calls `pysuto.logging.start_run_log(data_dir, name)`; drivers with `--data-dir` (Stages 2–6, `python -m pysuto`) log under `<data-dir>/logs/`, the scripts without it (feed, parse, diag, sweep) under `data/logs/` relative to the repo root. `python -m pysuto run` requires `--data-dir`, runs from any directory and starts every step in a fresh child (`python -m pysuto._step`, #186) that calls `start_run_log(<data-dir>, "step_<step>")` itself, its stdout going to `<data-dir>/logs/pipeline/<slug>_<step>.log`; `run_feed` requires `acc_id`.
- Tests: `python -m pytest tests/` — always pass `tests/`, bare `pytest` also collects the gitignored `leg/`. One test: `python -m pytest tests/test_solver.py::TestX::test_y -q`. The suite takes about 5 min. Lint: `ruff check`; ruff comes with `.[dev]`, but the `pysuto` reference env carries none, so lint runs in CI or in a dev env.
- The tutorial notebooks ship in the wheel with their executed outputs: `python tools/refresh_tutorial.py` (from an env with `pip install -e ".[tutorial]"`, Chrome for kaleido via `plotly_get_chrome -y`) is the one way to refresh them. It runs the eleven chapters from a fresh `python -m pysuto tutorial` copy, rewrites machine paths to relative forms and fails on any `/home/` left; `tests/test_tutorial_notebooks.py` checks the shipped JSON statically; never save a notebook from Jupyter into `src/pysuto/tutorial/`.
- Check chains: `python -m pysuto compare --data-dir data --base <ref leaves> --new <check leaves>` compares two chains array by array (#157; `docs/workflow.md`). A check chain from Stage 1 is judged on its arrays: its feeds' `parents.inputs` differ from the reference's by design (a new `trade_shares` version naming the new BACI feed run, #53; the current acc version on the rebuilt BACI and crude-ore feeds; #161).
- Knob sweeps: `scripts/sweep.py --dry-run` first, then run it in the background from the main checkout (`--resume <sweep_id>` continues one); run ids `sweep_{YYYYMMDD_HHMMSS}_{8hex}`, points labelled `<sweep_id> pNNN` (#105); score it with `scripts/sweep_diag.py <sweep_id>` (#107). With `KEEP_INTERMEDIATE = False` only the opt runs persist: imp and const pass through a scratch store (`/dev/shm/pysuto-sweep`, else under the temp dir) and are deleted once no remaining point reuses them (#108).
- Before a full chain run `df -h .` and keep ≥ 3 GB free: const ≈ 0.85 GB, opt ≈ 0.08 GB per run (≈ 0.5 GB with the per-year tables). A Stage-3 `zipfile` `OSError` means the disk is full, not a bug. Deleting runs is the user's call; afterwards prune their ids from `data/proc/<stage>/registry.json`.
- `run_solver.py` writes no per-year `diag/edges_<year>.csv` / `constraints_<year>.csv` (427 of the 504 MB of `opt_20261006_105904_61c5efe9`). Write them only for the years a diagnostic reads — `scripts/solver_stats.py <opt_run_id> --years …` — and never append `--years all` to a script that runs the stage drivers back to back (#46).
- Maintenance scripts (`tools/prune_runs.py`) run with `--dry-run` first. `ruff check` (pyflakes rules `F` only, `[tool.ruff]` in `pyproject.toml`; `leg/`, `scripts/explo/` and notebooks are excluded) lints in CI (`.github/workflows/ci.yml`: suite on 3.9/3.12/3.13, ruff, build + `twine check`, PyPI trusted publishing on a `v*` tag; a manual `workflow_dispatch` run rehearses it on TestPyPI); `.githooks/pre-commit` runs `ruff check` when ruff is on PATH (the `pysuto` env has none) and `python -m pytest tests/` on commits that touch Python (the README tests alone on a commit that touches only README.md, `docs/architecture.md`, `docs/workflow.md` or `scripts/README.md`) and `.githooks/commit-msg` refuses issue-closing keywords (enable per clone: `git config core.hooksPath .githooks`).

## Numerical & performance conventions

- Stack: numpy + scipy.sparse + clarabel + pandas/pyarrow. No numba, cython, cvxpy or networkx in `src/`; matplotlib only under `diag/`; Plotly is imported inside functions, never at module level.
- dtypes: values `float64`, indices `int64` (scipy's CSR index arrays, `G`'s among them, are `int32`), masks `bool`, no `float32`. Time labels are strings, never ints. Stage 2 arrays are dense `float64` of shape `N_f × N_r` (≈ 850k edges); NaN means "not covered", never 0. Vectorise — no Python loops over edges.
- Sparse: assemble COO triplets → `csr_matrix((v, (r, c)))`; keep `G` CSR; convert to CSC only at the Clarabel boundary. `_partition_G_by_time` raises on any cross-year nonzero — keep it strict.
- Solver: one Clarabel QP per year (block path), ≈ 18–22 s each at the ADR 0039 tolerances (defects form); cold years are solved concurrently on `N_WORKERS = os.cpu_count()` threads (#70), so the solve takes ≈ 95 s wall on copper with 8 cores, plus the ≈ 15 s probe (`opt_20261119_130302_b20f0dba`: `solve_time_s` 17.7–22.4 s, Σ 591 s, `solve_wall_s` 94.9 s; per-year times and `solve_wall_s` are in the run sidecar) — run it in the background. `Solver` builds a neutral QP per year and the injected `ClarabelBackend` solves it (ADR 0042); any status outside `{solved, solved_inaccurate}` raises `RuntimeError` — never add a silent fallback. The feasibility pre-check probes every year in parallel (`FEASIBILITY_CHECK = "all"`, ≈ 15 s); `"none"` skips it (sweep points, #104).
- "Warm start" means two unrelated things: the modelling prior `a_prior(t) = a*(t−1)` (`--prior-mode warm`) and Clarabel's KKT-factorisation reuse. Say which one you mean.
- σ conditioning has one source of truth, `src/pysuto/_conditioning.py`: `Solver` imports its defaults, Stage 4 records the floors it used in the opt sidecar's `stage_config`, and Stage 5 applies the recorded ones (an opt run without them falls back to the defaults with a logged warning).
- Lp exponent `p = 1.3` everywhere; `src/` uses no RNG (run ids and the 8-hex codes of input versions in `input_registry` are uuid4) except the seeded Latin hypercube of a random sweep (`pysuto.pipeline.Sweep`, #105), Stage 5's posterior draws, which take the caller's generator (#81), and the toy case's ground truth and synthetic feeds (`pysuto.tutorial.toy`, one fixed seed, #59); tests seed `np.random.default_rng`.
- Persistence is `np.savez` (uncompressed) and `sp.save_npz` for `G`; no chunking.
- Build-time validation raises (`ValueError`, `KeyError`, `ConstraintIdError`, `ConstraintUnitMixingError`); it never clips, coerces or fills.
- Stage 2–3 knobs are UPPER_CASE fields of the case's sections (copper's: `scripts/copper.py`); solver tolerances and every other Stage 4–6 knob are UPPER_CASE constants in the `# USER CONFIGURATION` block of `src/pysuto/stages.py` (#186; the drivers list them as `KNOBS`; `config/config.yaml` holds only `pipeline.acc_id`). Changing a knob is a measured change (REP + run id), not a fix.

## Code conventions

- Module header `"""file.py\n====`; numpy-style docstrings; `from __future__ import annotations` in every new module; ~90-column lines; double quotes; Greek spelled out (`sigma_a`, `sigma_c`, `eps_opt`).
- Formatting is hand-aligned (`=` and kwargs in columns, `# ====` banners): never run black or ruff-format.
- Records are `@dataclass`; no pydantic. Asserts are rare — validate explicitly and raise.
- Params CSVs mix `,` and `;`: read with `sep=None, engine="python", dtype=str`; wildcards are `*` and `same`.
- Logging: `get_logger(name)` per module; `print` only inside diag `.summary()`, and in the Stage 1 builders and `icsg_clean.py`, whose stdout the hook captures as their log (`copper_feed._run_script`).
- Diagnostics contract: construct with data → `.plot(odir, show=False)` returns paths → `.summary()`. `diag/` is a leaf: the data layer never imports it at module level (driver-level code — the scripts, `pysuto.stages` — may).
- Module-level imports flow one way (`docs/architecture.md` *Import direction*); `logging` imports nothing from `pysuto`. A function-local import is allowed only to break a cycle and carries a comment saying so.
- Comments and docstrings state the invariant or the reason, in the present tense, and may cite an ADR; `git blame` leads to the history. Issue numbers, round letters and run ids go in commit messages, REPs and issue comments. Measured results go in the REP that measured them. A workaround for an open issue cites that issue. The rule covers the library and the application; the frozen `scripts/explo/` is exempt.

## Testing

- `tests/` holds the library's tests and those of `tools/`; `tests/copper/` holds every test that imports or runs `scripts/`, reads its files or checks docs that link into it (the tests of the generic drivers too, until #244). CI job `library` deletes `scripts/` and `tests/copper/` and runs `tests/`. Both flat, no `conftest.py`, no markers, no golden files; a copper test imports its helpers from `tests/` by a `sys.path` line, a library test imports nothing from `tests/copper/`. Fixtures are toy objects built in the test module; use `tmp_path` and `monkeypatch`, never `mocker`.
- Tolerances: `np.allclose` with `rtol=1e-12`/`1e-10` for exact algebraic identities, `1e-6`/`1e-7` for solver outputs; `pytest.approx` (default or `rel=1e-6`) for scalars; `assert_array_equal` for indices and masks. Never loosen a tolerance to make a test pass.
- A suite run writes no `logs/` (#190): `get_logger` has no file handler and `run_feed` attaches its root DEBUG handler only when given `run_log`. `tests/test_import_no_writes.py` imports every module from an empty directory.
- A bug fix ships with the test that reproduces it; a change to a σ rule, builder, imputer or feed ships with a REP measuring it.

## Data, registries & artefacts

- `data/` is gitignored. Only LICENSE-DATA's CC BY 4.0 tables (`acc/copper`, `params/{conc,eff}`, `mask/row` under `data/input/**/current/`) are tracked, via `git add -f`; the feed concordances and the other params tables have no licence and live only on disk (untracked 2026-11-24, #297); registries, `vN/` snapshots and script-written `current/` files are untracked but load-bearing — never delete them as clean-up. `data/input/raw` (17 GB) is not reproducible: read-only.
- Input registries (`feed|acc|params|mask`) are sha256-bumped and hardlinked: edit only `current/`, never a `vN/` directory (ADR 0023).
- A parameter's value and its provenance live in its row of `data/input/params/<id>/current/params.csv` (`value_*`, `source`, `comment`), and the user edits it between turns. Read the row before quoting it, cite it as `<id> vN`, and keep values out of GLOSSARY.md and code comments; `python tools/input_status.py` lists `current/` edits not yet registered.
- Registries have no locking of their own: never run two stage drivers concurrently in one checkout — index entries get lost. Every stage driver, and every step of `python -m pysuto run`, takes `data/proc/.stage.lock` and refuses to start while a live driver holds it (#101); a lock of a dead process is taken over with a warning. `scripts/sweep.py` does not take it: its steps do, one after another.
- Runs cited by a REP, an ADR, an issue's closing comment or `docs/history/TODO.md` are permanent, except the const runs of superseded chains, which the user may prune (2026-10-20: all but the final and previous reference chains'). An opt whose const is gone can no longer be loaded by `SolverDiag`; its `diag/*_<year>.csv` tables, where written, are then its only row-level record — never delete them. Figures go to `<run>/diag/`, `figs/psut/`, `figs/report/`, `docs/rep/figs/`; `docs/rep/` is git-tracked.

## Invariants no test enforces (the ADR is the source of truth)

- Never parse a `constraint_id`; read `row_meta` columns. Param ids contain no underscore (ADR 0005).
- Hard row ⇔ `σ_c = 0`. In solver CSVs `residual` is identically 0 on hard rows; use `resid_solver` for them (GLOSSARY.md "Defect, residual & violation").
- Results are year-local: scatter `F_opt` through `cols_t` before any `G @ F` (GLOSSARY.md "Period slice").
- σ cannot repair a data contradiction: fix the input, never loosen σ or drop the row (ADR 0019, 0035).
- The 0.3 kt σ floor applies to feed-backed scale totals only, never globally (ADR 0025).
- Pooling happens only via `agg=true` in the params CSV, never inferred (ADR 0005, 0022); scale totals are soft and are not control totals (ADR 0006).
- Carry the grade, never the mass (ADR 0023); no pySUT/pymrio dependency (ADR 0007); PIOT is a derived view of a PSUT and is never persisted (ADR 0037).
- A σ or repricing change is judged by the Round-E stop rule — anchor band, mass moved, world retention, small-node posterior/prior, exporter retention (ADR 0025).
- Before changing a builder, imputer or feed: `grep -l <ClassOrScriptName> docs/adr/*.md` and read the hits.

## Documentation & units

- Results for an abstract, talk, REP verdict or the manuscript are **key results**: read [docs/agents/key-results.md](docs/agents/key-results.md) before choosing them, and paste its relevance check into every reviewer brief.
- Analyses of `opt`/`const`/`imp` runs are REPs at `docs/rep/analysis_<run_id>_<YYYYMMDD>_<slug>.md`, opening with Run / Against / Chain / Feed+params versions / Date / Branch / ADR / Verdict.
- Masses are `kt` — the code's unit in logs, axis labels and the Sankey default. Name the valuation when both exist: `kt` (gross mass) vs `kt CC` (contained copper); `Mt` for world-scale totals; never an unlabelled `t`.
- Every relative number names its base and the run it is relative to: `Σz² −9.6 % vs opt_<id>`, never `−9.6 %` alone. A difference of two percentages is in `pp`. A retention (`Σa*/Σa_prior`, GLOSSARY.md "Retention") is always quoted with its category and year.
- ADRs: `docs/adr/NNNN-slug.md`, one paragraph plus a bold `**Status:**` line citing evidence run ids; write one only if hard to reverse AND surprising AND a real trade-off (`domain-modeling` skill, `ADR-FORMAT.md`). "Rejected on measurement" is a valid status.
- `GLOSSARY.md` is a glossary only: bold term, tight definition, `_Avoid_` aliases, flagged ambiguities (`domain-modeling` skill, `GLOSSARY-FORMAT.md`). Add a term whenever vocabulary had to be pinned down mid-task.
- Modelling choices and architecture go in `docs/architecture.md` (stages, modules), `docs/workflow.md` (run, knobs) or `scripts/README.md` (application), in the same branch as the code they describe (`CODING_STANDARDS.md`; `tests/test_readme_names.py`). Open work and round bookkeeping live only in GitHub issues and milestones, measurements in closing comments and REPs; `docs/history/TODO.md` is a frozen history. No third place.

## Git workflow

- Branches `feat/…`, `fix/…`, `diag/…`; merge with `--no-ff`, titled `Merge branch '<b>': <result> (ADR N, Round X)`.
- Commits: `type(scope): finding as a sentence`; types feat/fix/docs/measure/chore/test/refactor; scope is a module or artefact (`feed`, `impute`, `constraints`, `diag`, `psut`, `piot`, `rep`, `adr`, `glossary`, `params`); issues bare as `(#NN)`. Pass long messages with `-F <file>` written to the scratchpad; `git merge -F -` reads no stdin. Commit only when asked. Never stage `data/` (except the CC BY tables' `current/` files via `git add -f`), `logs/` or `leg/`.

## Agent skills

- **Source** — Matt Pocock's skills come from the read-only `mattpocock-skills` plugin, enabled at project scope in `.claude/settings.json`; the rules below adapt them to pySUTO. `.claude/skills/` holds only pySUTO's own skills. Domain docs are single-context: `GLOSSARY.md` + `docs/adr/` (`docs/agents/domain.md`).
- **Issue tracker** — GitHub issues via `gh`; rounds are milestones, issues close only through `gh issue close --comment`. See `docs/agents/issue-tracker.md`, and its *Due dates* before any time estimate or due date; `tools/issue_audit.py` checks it. Triage labels `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`: see `docs/agents/triage-labels.md`.
- **`/implement`** — commits to the current branch: start it on a `feat/`, `fix/` or `diag/` branch, never `main`. With two or more tickets, write the lane plan before the first edit (`docs/agents/parallel-worktrees.md` *When to start lanes*). There is no type checker; where it says typecheck, run the single test file. Commit first, then review the range since the branch point with `mattpocock-skills:code-review` (not the built-in `/code-review`); review fixes go in a second commit.
- **Measure tickets** — `to-tickets` gives every σ-rule, builder, imputer, feed or knob change its own measure ticket, blocked by the build ticket and closed by a `measure(...)` commit plus its REP; the steps are in `docs/agents/measure-tickets.md`. `/implement` and `/implement-spec` (code-only specs) never run the pipeline or take a measure ticket: measure tickets run afterwards in the main checkout, one at a time (worktrees have no `data/proc`). A ticketless run whose change needs a REP says so at close-out.
