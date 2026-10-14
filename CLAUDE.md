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

---

# pySUTO — project-specific facts

Supply–use-table reconciliation (copper application): a σ-weighted QP solved in Clarabel at the end of a six-stage pipeline (numpy · scipy.sparse · pandas). Architecture and module map are in [README.md](README.md) §2–§6, vocabulary in [GLOSSARY.md](GLOSSARY.md), decisions in `docs/adr/`, open rounds in `TODO.md`. Re-measure, do not assume (ADR 0028): a claim about a run is backed by a run id and a number.

## Pipeline

- Six stages, one driver each: `scripts/*_feed.py` (1) → `collect_aggregate_impute.py` (2) → `constraints.py` (3) → `run_solver.py` (4) → `uncertainty_propagation.py` (5) → `psut.py` (6). README prose still says "five"; six is correct.
- Run ids are `{feed|imp|const|opt|uncprop|psut}_{YYYYMMDD_HHMMSS}_{8hex}`; feed runs embed the feed name (`feed_<feed_id>_…`).
- Per-stage `registry.json` is a trimmed index. Lineage lives in each run's `metadata.json` (`parents.inputs` = input versions, `parents.upstream` = parent run ids) — read the sidecar for provenance.
- Feed build order is load-bearing (ADR 0024, 0027): `build_baci_reexport.py → baci_feed.py`, then `build_trade_shares.py → icsg_feed.py`. Any ICSG or BACI change invalidates the trade-share weights and nothing detects it (it once cost 6.2 pp of z²): rebuild them, then confirm `trade_shares.version` in the feed's `metadata.json`.
- `run_solver.py --prior-mode` defaults to `cold` in code although its help text, docstring and GLOSSARY.md say warm — cold is what runs; trust the sidecar's `prior_mode`.

## Environment & commands

- Interpreter: conda env `pysuto`, Python 3.9. Write 3.9-safe code: no `match`, `zip(strict=)` or `itertools.pairwise`; `X | None` only in annotations under `from __future__ import annotations`.
- `pyproject.toml` is gitignored (`*.toml`), so a fresh clone cannot `pip install -e .`; the env's editable install points at the main checkout.
- In a git worktree prefix every command with `PYTHONPATH=src`; worktrees have no `data/proc`, so run stage drivers from the main checkout or pass `psut.py --data-dir <main-checkout>/data`. The feed builders put only `scripts/` on `sys.path` and would import the main checkout's `pysuto` silently.
- Run everything from the repo root: some drivers call `load_config("config/config.yaml")` at import time and `get_logger()` writes `logs/<name>.log` relative to CWD.
- Tests: `python -m pytest tests/` — always pass `tests/`, bare `pytest` also collects the gitignored `leg/`. One test: `python -m pytest tests/test_solver.py::TestX::test_y -q`. The suite runs in under a minute.
- Pipeline: `scripts/acc.py --id copper` → parse (`scripts/parse/`) → feeds → `collect_aggregate_impute.py --feed-id …` → `constraints.py --imp-run-id <id> <label>` (label is positional) → `run_solver.py --const-run-id <id>` (also renders diag) → `uncertainty_propagation.py <opt_run_id>` (also renders diag; re-render with `uncprop_diag.py <uncprop_run_id>`) → `psut.py <opt_run_id> [--sankey 2020]`.
- Before a full chain run `df -h .` and keep ≥ 3 GB free: const ≈ 0.7 GB, opt ≈ 0.08 GB per run (≈ 0.5 GB with the per-year tables). A Stage-3 `zipfile` `OSError` means the disk is full, not a bug. Deleting runs is the user's call; afterwards prune their ids from `data/proc/<stage>/registry.json`.
- `run_solver.py` writes no per-year `diag/edges_<year>.csv` / `constraints_<year>.csv` (427 of the 504 MB of `opt_20261006_105904_61c5efe9`). Write them only for the years a diagnostic reads — `scripts/solver_stats.py <opt_run_id> --years …` — and never append `--years all` to a script that runs the stage drivers back to back (#46).
- Maintenance scripts (`scripts/migrate_registries.py`, `scripts/explo/prune_runs.py`) run with `--dry-run` first. No lint, formatter, type-checker, CI or Makefile exist; `.githooks/pre-commit` runs `python -m pytest tests/` on commits that touch Python (enable per clone: `git config core.hooksPath .githooks`).

## Numerical & performance conventions

- Stack: numpy + scipy.sparse + clarabel + pandas/pyarrow. No numba, cython, cvxpy or networkx in `src/`; matplotlib only under `diag/`; Plotly is imported inside functions, never at module level.
- dtypes: values `float64`, indices `int64`, masks `bool`, no `float32`. Time labels are strings, never ints.
- Stage 2 arrays are dense `float64` of shape `N_f × N_r` (≈ 850k edges); NaN means "not covered", never 0. Vectorise — no Python loops over edges.
- Sparse: assemble COO triplets → `csr_matrix((v, (r, c)))`; keep `G` CSR; convert to CSC only at the Clarabel boundary. `_partition_G_by_time` raises on any cross-year nonzero — keep it strict.
- Solver: one Clarabel QP per year (block path), ≈ 60 s/year at the ADR 0039 tolerances (≈ 30 min for 30 years; per-year times are in the run sidecar) — run it in the background. Any status outside `{Solved, AlmostSolved}` raises `RuntimeError` — never add a silent fallback. The feasibility pre-check covers the first year only (`FEASIBILITY_CHECK_FIRST_ONLY`); a full scan takes ≈ 350 s.
- "Warm start" means two unrelated things: the modelling prior `a_prior(t) = a*(t−1)` (`--prior-mode warm`) and Clarabel's KKT-factorisation reuse. Say which one you mean.
- σ conditioning has one source of truth, `src/pysuto/_conditioning.py`; the `DEFAULT_*` constants in `src/pysuto/uncprop.py` mirror it by hand — change both or neither.
- Lp exponent `p = 1.3` everywhere; `src/` uses no RNG (run ids are uuid4); tests seed `np.random.default_rng`.
- Persistence is `np.savez` (uncompressed) and `sp.save_npz` for `G`; no chunking.
- Build-time validation raises (`ValueError`, `KeyError`, `ConstraintIdError`, `ConstraintUnitMixingError`); it never clips, coerces or fills.
- Solver tolerances and every other knob are UPPER_CASE constants in the driver's `# USER CONFIGURATION` block (`config/config.yaml` holds only `pipeline.acc_id`). Changing a knob is a measured change (REP + run id), not a fix.

## Code conventions

- Module header `"""file.py\n====`; numpy-style docstrings; `from __future__ import annotations` in every new module; ~90-column lines; double quotes; Greek spelled out (`sigma_a`, `sigma_c`, `eps_opt`).
- Formatting is hand-aligned (`=` and kwargs in columns, `# ====` banners): never run black or ruff-format.
- Records are `@dataclass`; no pydantic. Asserts are rare — validate explicitly and raise.
- Params CSVs mix `,` and `;`: read with `sep=None, engine="python", dtype=str`; wildcards are `*` and `same`.
- Logging: `get_logger(name)` per module; `print` only inside diag `.summary()`.
- Diagnostics contract: construct with data → `.plot(odir, show=False)` returns paths → `.summary()`. `diag/` is a leaf: the data layer never imports it at module level.
- Module-level imports flow one way (README §5 "Import direction is strictly one-way"); `utils`/`logging` import nothing from `pysuto`. A function-local import is allowed only to break a cycle and carries a comment saying so.

## Testing

- Flat `tests/`, no `conftest.py`, no markers, no golden files. Fixtures are toy objects built in the test module; use `tmp_path` and `monkeypatch`, never `mocker`.
- Tolerances: `np.allclose` with `rtol=1e-12`/`1e-10` for exact algebraic identities, `1e-6`/`1e-7` for solver outputs; `pytest.approx` (default or `rel=1e-6`) for scalars; `assert_array_equal` for indices and masks. Never loosen a tolerance to make a test pass.
- One test is skipped because scikit-sparse is absent (`uncprop` uses the `splu` fallback) — not a defect.
- Importing `baci_feed` attaches a root DEBUG handler, so every suite run appends ~17k lines under `logs/` (gitignored).
- A bug fix ships with the test that reproduces it; a change to a σ rule, builder, imputer or feed ships with a REP measuring it.

## Data, registries & artefacts

- `data/` is gitignored. Hand-authored inputs under `data/input/**/current/` are tracked via `git add -f`; registries, `vN/` snapshots and script-written `current/` files are untracked but load-bearing — never delete them as clean-up. `data/input/raw` (17 GB) is not reproducible: read-only.
- Input registries (`feed|acc|params|mask`) are sha256-bumped and hardlinked: edit only `current/`, never a `vN/` directory (ADR 0023).
- A parameter's value and its provenance live in its row of `data/input/params/<id>/current/params.csv` (`value_*`, `source`, `comment`), and the user edits it between turns. Read the row before quoting it, cite it as `<id> vN`, and keep values out of GLOSSARY.md and code comments; `python scripts/input_status.py` lists `current/` edits not yet registered.
- Registries have no locking: never run two stage drivers concurrently in one checkout — index entries get lost.
- Runs cited by a REP, ADR or `TODO.md` are permanent. Figures go to `<run>/diag/`, `figs/psut/`, `figs/report/`, `docs/rep/figs/`; `docs/rep/` is git-tracked.

## Invariants no test enforces (the ADR is the source of truth)

- Never parse a `constraint_id`; read `row_meta` columns. Param ids contain no underscore (ADR 0005).
- Hard row ⇔ `σ_c = 0`. In solver CSVs `residual` is identically 0 on hard rows; use `resid_solver` for them (GLOSSARY.md "Defect, residual & violation").
- Results are year-local: scatter `F_opt` through `cols_t` before any `G @ F` (README §8 "period slice").
- σ cannot repair a data contradiction: fix the input, never loosen σ or drop the row (ADR 0019, 0035).
- The 0.3 kt σ floor applies to feed-backed scale totals only, never globally (ADR 0025).
- Pooling happens only via `agg=true` in the params CSV, never inferred (ADR 0005, 0022); scale totals are soft and are not control totals (ADR 0006).
- Carry the grade, never the mass (ADR 0023); no pySUT/pymrio dependency (ADR 0007); PIOT is a derived view of a PSUT and is never persisted (ADR 0037).
- A σ or repricing change is judged by the Round-E stop rule — anchor band, mass moved, world retention, small-node posterior/prior, exporter retention (ADR 0025).
- Before changing a builder, imputer or feed: `grep -l <ClassOrScriptName> docs/adr/*.md` and read the hits.

## Documentation & units

- Analyses of `opt`/`const`/`imp` runs are REPs at `docs/rep/analysis_<run_id>_<YYYYMMDD>_<slug>.md`, opening with Run / Against / Chain / Feed+params versions / Date / Branch / ADR / Verdict.
- Masses are `kt` — the code's unit in logs, axis labels and the Sankey default. Name the valuation when both exist: `kt` (gross mass) vs `kt CC` (contained copper); `Mt` for world-scale totals; never an unlabelled `t`.
- Every relative number names its base and the run it is relative to: `Σz² −9.6 % vs opt_<id>`, never `−9.6 %` alone. A difference of two percentages is in `pp`.
- Retention is `Σa*/Σa_prior` by category and is always quoted with its category and year (GLOSSARY.md "Retention").
- ADRs: `docs/adr/NNNN-slug.md`, one paragraph plus a bold `**Status:**` line citing evidence run ids; write one only if hard to reverse AND surprising AND a real trade-off (`domain-modeling` skill, `ADR-FORMAT.md`). "Rejected on measurement" is a valid status.
- `GLOSSARY.md` is a glossary only: bold term, tight definition, `_Avoid_` aliases, flagged ambiguities (`domain-modeling` skill, `GLOSSARY-FORMAT.md`). Add a term whenever vocabulary had to be pinned down mid-task.
- Modelling choices and architecture go in `README.md`; round bookkeeping in `TODO.md`. No third place.

## Git workflow

- Branches `feat/…`, `fix/…`, `diag/…`; merge with `--no-ff`, titled `Merge branch '<b>': <result> (ADR N, Round X)`.
- Commits: `type(scope): finding as a sentence`; types feat/fix/docs/measure/chore/test/refactor; scope is a module or artefact (`feed`, `impute`, `constraints`, `diag`, `psut`, `piot`, `rep`, `adr`, `glossary`, `params`); issues as `#NN`.
- Commit only when asked. Never stage `data/` (except `current/` inputs via `git add -f`), `logs/` or `leg/`.

## Agent skills

- **Source** — Matt Pocock's skills come from the read-only `mattpocock-skills` plugin, enabled at project scope in `.claude/settings.json`; the rules below adapt them to pySUTO. `.claude/skills/` holds only pySUTO's own skills.
- **Issue tracker** — GitHub issues via `gh`. See `docs/agents/issue-tracker.md`.
- **Triage labels** — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.
- **Domain docs** — single-context: `GLOSSARY.md` (glossary) + `docs/adr/`. See `docs/agents/domain.md`.
- **`/implement`** — commits to the current branch: start it on a `feat/`, `fix/` or `diag/` branch, never `main`. There is no type checker; where it says typecheck, run the single test file. Commit first, then review the range since the branch point with `mattpocock-skills:code-review` (not the built-in `/code-review`); review fixes go in a second commit.
- **Measure tickets** — `to-tickets` gives every σ-rule, builder, imputer, feed or knob change its own measure ticket, blocked by the build ticket and closed by a `measure(...)` commit plus its REP. `/implement` never runs the pipeline; a ticketless run whose change needs a REP says so at close-out.
- **`/implement-spec`** — code-only specs; it never takes a measure ticket. Those run afterwards in the main checkout, one at a time (worktrees have no `data/proc`; registries have no locking).
