# How modular is pySUTO? — a stage-by-stage assessment

> **Status:** research note (read-only code audit; nothing was run, no code changed).
> **Date:** 2026-10-08 · **Branch:** `fix/baci-low-side-recon` · **Commit:** `3248051`
> **Question:** to what extent can one argue that pySUTO is a *modular* framework — as
> `README.md:5` ("A modular Python framework …") and `docs/manuscript/method.md:47,127,133`
> ("the generic framework", "plugin imputers", "builder plugins") claim?
> **Sources:** the code under `src/pysuto/` and `scripts/`, the tests under `tests/`, ADRs,
> README and GLOSSARY, plus run sidecars in `data/proc/` (inspected, not executed). Every
> claim carries a `file:line`. Claims marked **[inferred]** come from reading the code and
> were not run.

---

## 1. Verdict

pySUTO is **modular by stage, partly modular within stages, and not yet generic across
commodities.** I would make the first argument without hedging, the second with named
limits, and not make the third until ADR 0034 lands.

| Sense of "modular" | Holds? | Why, in one line |
|---|---|---|
| **A. Stage decomposition**: separable stages with persisted, narrow interfaces | **Yes** | Each stage re-runs from one upstream run id and writes a lineage-stamped run; the 2→3, 3→4 and 4→5/6 interfaces are a handful of arrays and columns. |
| **B. Plug-in extensibility within a stage** | **Partly** | There are real ABC seams for imputers (Stage 2), constraint builders (Stage 3) and uncertainty propagators (Stage 5). The feeds, the solver backend and the PIOT constructs have none, and the order and wiring of all plug-ins live in driver code. |
| **C. Domain genericity**: framework vs. case | **No (not yet)** | Copper labels are hard-coded in `src/` (`CC`, `Mass`, `SEM`, `Nature/SEM/FD`, the copper stock-change topology). ADR 0034 itself says a non-`CC` case "breaks before any imputer runs". |
| **D. Hidden contracts between modules** | **Several** | `P_PROP` is copied in two drivers; the σ floors are copied in three places; Stage 3 recognises flat-prior edges from Stage 2's σ value; the trade-share build order is enforced nowhere; Stage 5 reads Stage-3 row semantics wrongly. |

**Wording I would defend in the manuscript:**

> "pySUTO decomposes compilation and reconciliation into six stages that exchange persisted,
> versioned and lineage-stamped artefacts; each stage can be re-run from its predecessor's
> run alone. Imputation and constraint building are open to extension through plug-in base
> classes. The accounting structure, concordances, pedigree scores and parameters are data;
> the copper case is a parametrisation of the framework, although some framework code still
> names copper's labels (ADR 0034)."

**Wording I would not defend yet:** "generic framework" (`method.md:47`) without that
qualification, "any metal" (ADR 0034:5 states it as the aim, not as the state), and any
claim that the QP solver or the PIOT construct is interchangeable.

Per-stage summary (detail in §3):

| Stage | Interface (A) | Extension (B) | Genericity (C) | Hidden contracts (D) | Tests in isolation | Overall |
|---|---|---|---|---|---|---|
| 1 Datafeed | Partial | Data-driven, not code-driven | Library good, ROW literal | Builder chain unenforced, import side effects | Library yes, drivers no | **Partial** |
| 2 Aggregation & imputation | Output strong, input weak | `BaseImputer` strong | Weakest in `src/` | `P_PROP`, flat-prior σ fingerprint | Strong | **Moderate–strong** |
| 3 Constraints | Strong | `BaseConstraintBuilder` strong | Partial | Engine knows builders; time=None rows | Strong | **Strong** |
| 4 Solver | Strong | None (Clarabel woven in) | Generic | σ_c reimplemented; prior-mode drift | Good | **Strong interface, monolithic backend** |
| 5 Uncertainty | Strong | `BaseUncProp` (ABC, no in-memory API) | Generic | Misreads hard inequalities; σ floors copied | Modest | **Contract modular, semantics coupled; never run** |
| 6 PSUT / PIOT | Strong (Protocol seam) | `aggregate()` data-driven; constructs closed | Boundary labels hard-coded | No acc version check | Good | **Strong consumer, bound to the case** |

---

## 2. What "modular" is measured against

There are four criteria. They follow Parnas's information-hiding criterion (a module hides
a decision that may change behind an interface that does not) and the project's own rules.
The rules are: the one-way import direction (`README.md:132-153, 1150-1162`); never parsing
a `constraint_id` (`constraints/naming.py:22-24`); and "roles are the framework's, labels are
the case's" (ADR 0034).

1. **Interface (contract / seam):** what a stage reads and writes, and whether it can be
   re-run or replaced from those artefacts alone.
2. **Extension points:** how a new feed, imputer, constraint family, solver or construct is
   added, and how many places change.
3. **Domain genericity:** whether copper knowledge lives in `src/pysuto/` or is confined to
   `scripts/`, `data/input/` (ACC, concordances, params) and ADRs.
4. **Coupling / hidden contracts:** ordering dependencies, duplicated constants,
   import-time side effects, and knowledge of another stage's internals.

Isolated testability, meaning toy-fixture tests without `data/proc`, is reported per stage
as supporting evidence.

---

## 3. Stage by stage

### Stage 1 — Datafeed (`scripts/*_feed.py`, `scripts/build_*.py`, `scripts/parse/`, `feed.py`, `feedpipe.py`, `conc.py`, `unc.py`, `meta.py`)

**For**
- **Output contract.** Each run writes `feed.pkl`, `result.npz` (`F_root`, `sigma_root`,
  `cv_root`, `cv_src`) and a sidecar whose `parents` field is required
  (`registry/feed.py:34-41, 118-119, 128-162`). The latest ICSG sidecar records
  `inputs = {acc v4, feed icsg v12, trade_shares v22}`.
- **The concordance is data, not code.** It is built from the `sector_map`, `region_map`,
  `time_map`, `valuation_map` and `variable_map` sheets (`conc.py:209-219`). Pedigree comes
  from `meta.xlsx` (`meta.py:83-145`). Scope mask and ROW penalty are metadata switches
  (`meta.py:61-70`). Per-edge σ multipliers go through a hook (`feedpipe.py:402-427`).
- **Stage 2 needs no code change for a new feed.** Feeds are repeated `--feed-id` flags
  (`collect_aggregate_impute.py:720-722`).
- **The library is domain-free.** `unc.py`, `meta.py`, `input_registry.py` and `registry/*`
  contain no copper literals. The propagation math is tested in isolation (`test_unc.py`,
  122 tests; `test_conc.py`, 39; `test_feedpipe.py`, 37).

**Against**
- **The `DataFeed` ABC is not the extension seam.** Only `ConcordanceTable` and
  `WeightMatrix` subclass it (`conc.py:20, 659`). The library pipeline `run_feed_pipeline`
  (`feedpipe.py:444-510`) is called by no script or test. In practice each feed is a copy
  of a ~390-line driver (`crude_ore_feed.py:61` differs from `baci_feed.py` by 167 diff
  lines).
- **Behaviour drifts between those copies.** Only `baci_feed.py:485` applies
  `_apply_edge_scope_mask`, so the ICSG and crude-ore drivers ignore a scope mask set in
  `meta.xlsx`. This is inert today, because their sidecars have `edge_scope_mask=''`.
- **The builder chain is ordered only by documentation and blind to lineage.**
  - `build_trade_shares.py:238-244` globs the *latest* `feed_icsg_*` and `feed_baci_*` runs
    and records them only in a `print`.
  - The `trade_shares` input entry stores no upstream run ids (`input_registry.py:171-177`).
  - So ICSG feed run *n* depends on ICSG feed run *n−1*, and nothing checks the order. The
    measured cost of breaking it was 6.2 pp of z² (`README.md:178-185`).
  - The two builders import each other function-locally to dodge a cycle
    (`build_trade_shares.py:455` ↔ `build_baci_reexport.py:498`).
- **Import-time side effects.**
  - `baci_feed.py:89-92` and `crude_ore_feed.py:86-89` set the root logger to DEBUG.
  - `load_config()` runs at import in all three drivers (e.g. `baci_feed.py:114`).
  - Module globals are mutated via `global` (`baci_feed.py:432-439`).
- **Dead parsers.** `scripts/parse/icsg_parse/icsg_2005.py:7`, `icsg_2023f.py:7` and
  `icsg_cap.py:8` import `pysuto.util`, which no longer exists (only `utils.py` does).
- **Genericity leak in `src/`.** The rest-of-world label must literally be `ROW`
  (`feedpipe.py:388`, `meta.py:67`).

**Assessment: partial.** The *library* half of Stage 1 is modular and data-driven. The
*orchestration* half (drivers and builders) is a set of copy-pasted, order-dependent
scripts. Adding a feed is cheap for Stage 2 but not clean in Stage 1.

### Stage 2 — Aggregation & imputation (`collect_aggregate_impute.py`, `collector.py`, `agg.py`, `impute/`, `param_relation.py`, `masks.py`)

**For**
- **Narrow output contract.** Stage 3 consumes only `f_agg`, `sigma_agg`, `coverage` and
  the imp run id (grep over `scripts/constraints.py` and `constraints/`). The sidecar records
  `inputs` (acc, five params) and `upstream.feed_run_ids` (`collect_aggregate_impute.py:655-696`).
- **A real plug-in seam.** `BaseImputer` has one abstract method, `impute(f, sigma, missing,
  step, acc_struct)` (`impute/base.py:62-112`). `ImputationEngine` runs a cascade to a
  fixpoint and then terminal steps (`impute/engine.py:103-202`). A new imputer is one module
  (`TradeImputer` is 99 lines) plus ~5–10 driver lines.
- **Relations are params-driven.** `param_relation.py` resolves `a_*/b_*` selectors from
  the params CSV (`param_relation.py:52-60`) and is shared with Stage 3.
- **Aggregation is domain-free and vectorised** (`agg.py:98-169`). Toy-fixture tests cover
  it: `test_imputation.py` (99), `test_network_imputer.py` (27), `test_collector.py` (22),
  `test_agg.py` (19).

**Against**
- **This stage has the most copper code in `src/`.**
  - `impute/share.py:57-70` bakes copper's stock-change topology into the module
    (`CP→PSC`, `Cathode→MESC/USC`, base `CSXEW+CER`, `Cathode→FU`).
  - It also parses `param_id` substrings `bound{mesc,usc,psc}` (`:116-118`).
  - The content valuation `"CC"` and the sub-entity `"SEM"` are literals in the
    network and trade imputers (`impute/network.py:85, 98, 253-254, 328`;
    `impute/trade.py:72`).
- **Order is semantics, and order is code.** TradeImputer must run first and FlatPrior
  last (`README.md:411, 417`). Which imputer may write which edges is a convention, checked
  only after the run and only with a warning (`collect_aggregate_impute.py:647-650`).
- **σ steps sit outside the seam.** Leg-σ damping, the outlier screen and the
  contradicted-node screen are private driver functions that mutate
  `agg_result.sigma_agg` (`collect_aggregate_impute.py:265-432`). Tests reach them only by
  importing the script (`test_outlier_screen.py:8`).
- **Inputs are weakly pinned.**
  - Params are not pinned, so a re-run takes the *latest* params
    (`collect_aggregate_impute.py:448`).
  - Unlike Stage 3, there is no acc-version check on the feeds. The live run
    `imp_20261008_075249_bb1e1884` (acc v4) aggregates a crude-ore feed built on acc v3.
    This is benign today, since only the `masks.csv` hashes changed.
  - `collector.py:131-135` swallows `TypeError`/`AttributeError`/`KeyError`, which can
    silently drop a feed run id from the lineage.
- **Hidden cross-stage contracts.**
  - `P_PROP` "must match" between the Stage-2 and Stage-3 drivers
    (`collect_aggregate_impute.py:82`, `constraints.py:126-129`); nothing checks it.
  - Stage 3 finds flat-prior edges by the fingerprint `|a0|<1e-9 & σ≥1e5`
    (`constraints/engine.py:273, 361`). That depends on Stage 2's `FLAT_SIGMA_HIGH = 1e6`
    (`collect_aggregate_impute.py:138`), not on the imputation log.
- **The shared relation unit is deferred.** ADR 0011's `EdgeRelation` is deferred
  (`0011:30-32`), and the `_AGG_TRUE` duplicate it names is still there
  (`impute/parametric.py:36`, `constraints/ratio.py:38`).

**Assessment: moderate–strong.** The plug-in architecture is real and well tested. Genericity
is the weakest of all stages, and reproducibility from the sidecar alone is not guaranteed.

### Stage 3 — Constraints (`scripts/constraints.py`, `constraints/`)

**For**
- **Self-contained re-run.** The stage needs only `--imp-run-id` and a label
  (`scripts/constraints.py:631-639`), and it refuses an acc-version mismatch with its imp run
  (`:331-343`).
  - Writes: `G.npz`, `row_meta.parquet`, `a0`, `sigma_a`, `u`, `l`, per-year `{c, sigma_c,
    a_prior}`, `G_blocks`.
  - Sidecar: `parents.upstream.imp_run_id` (`registry/constraint.py:82-115`).
- **`row_meta` is a real narrow interface.**
  - 23 uniform columns (`constraints/naming.py:87-93`).
  - `constraint_id` is a display key, and no code in `src/` parses it (grep empty).
  - Stage 4 reads only `cone_type`, `is_hard` and `time` (plus the id, for error text)
    (`solver.py:266, 401, 459, 660, 706`). Stage 5 reads only `is_hard`, `cone_type` and
    `time` (`uncprop.py:204, 210, 293-295`).
- **A plug-in seam.** The `BaseConstraintBuilder` ABC requires `build_static`;
  `build_dynamic` is optional (`constraints/base.py:60-85`). Builders are added with
  `engine.register` (`engine.py:428-429`). A new builder of an existing row type and cone
  needs no change to Stage 4 or 5.
- **No upstream imports.** `constraints/` imports no Stage-2 module and duck-types
  `agg_result` (`engine.py:607-612`).
- **Tested in isolation.** `test_constraints.py` (101 tests) runs on a toy acc built via
  `__new__` (`:39-118`).

**Against**
- **A new row *type* is not purely additive.** It needs a `TYPE_*` constant
  (`naming.py:31-34`), branches in the engine's `_check` (`engine.py:456-532`, which knows
  `MassBalance`, `ratio_eq` and `ratio_bound`), and literals in three diag modules
  (`diag/const.py:61-63`, `diag/solver.py:360-362`, `diag/stats/objective.py:80, 237`). A new
  *cone* means editing the solver (`solver.py:266-314, 401-428, 739-784`).
- **Copper defaults in the library.**
  - `nodata_anchor_exclude_sectors=("CO","Mining","Ore","WR","Tailings")` and
    `pin_exclude_sectors=(…,"Slag","PSC")` (`engine.py:272, 276`). The second is never
    passed by the driver, so the library's copper default is what runs.
  - `"stripping_ratio"` is skipped inside the generic `RatioBoundBuilder`
    (`ratio.py:458-462`).
  - `"concentration"`/`"Mass"` appear in `param_relation.py:160, 163`.
  - The year `"1995"` is wired into `engine.run` (`engine.py:807`).
- **The copper policy is in the driver (correct), but as logic, not values.** The driver's
  own header calls it `init_copper_constraints.py` (`scripts/constraints.py:2`). Masks,
  `param_id` prefixes and the param families are wired by name
  (`scripts/constraints.py:349, 493-497, 537-574`).
- **The partition is less strict than documented [inferred].** `_partition_G_by_time` checks
  only rows whose `time` equals a requested year (`engine.py:120-139`). Rows with
  `time=None` (wildcard CSV sums, `sum.py:384-391`) fall into no block and are silently left
  out of the block solve (`solver.py:731`). README:525 says they "stay active for every t".
  This is latent for copper: the driver uses no `from_csv` sums.
- **Small drifts.**
  - `lp.py:6-7` claims the engine imports it, but the engine re-implements the formula
    (`engine.py:46-75`, `sum.py:659`).
  - `ANCHOR_CV` and `SCALE_TOTAL_ABS_FLOOR` are not written to `stage_config`
    (`scripts/constraints.py:596-619`).

**Assessment: strong.** This is the best example of modular design in the code base:
plug-in builders behind a column-based interface that downstream stages read without
knowing any constraint family. The limits are an engine that knows its builders and some
copper defaults in the library.

### Stage 4 — Solver (`scripts/run_solver.py`, `solver.py`, `_conditioning.py`)

**For**
- **Agnostic to how the constraints were built.** The solver consumes G, c, σ_c, a_prior,
  bounds and three `row_meta` columns (`solver.py:190-197`). It writes `F_opt`, `eps_opt`,
  `c_star`, `residuals` and `cols_t` per year, with `parents.upstream.const_run_id`
  (`registry/optim.py:94-124`; `solver.py:941-946`).
- **Re-runnable from `--const-run-id` alone** (`scripts/run_solver.py:153-170`).
- **No copper strings.** Apart from one docstring mention, `solver.py` and
  `_conditioning.py` are generic.

**Against**
- **The backend is not swappable.**
  - `import clarabel` is at module level (`solver.py:37`).
  - The CSC conversions, the `ZeroConeT`/`NonnegativeConeT` cones and the
    settings/solve/update calls sit inline in both solve paths (`solver.py:286-314, 423-428,
    767-842`).
  - Clarabel's telemetry is part of `OptimResult` (`ITER_DTYPE`, `solver.py:51-63`) and is
    read by diag.
  - There is no solver ABC or neutral `(P, q, A, b, cones)` object, so changing the solver
    means rewriting `solver.py`. Persistence and Stage 5 would survive unchanged.
- **σ_c conditioning is duplicated.** `_conditioning.py` is the stated single source, but
  the solver re-implements the σ_c NaN→floor step inline three times
  (`solver.py:300-302, 343-344, 771-772`).
- **Hidden dependency in warm mode.** It reloads the acc from the CWD-relative
  `data/proc/acc` (`solver.py:714-722`). If that fails, it falls back silently to positional
  carry-forward, and the opt sidecar records no acc.
- **Configuration drift on a modelling choice.**
  - The `--prior-mode` default is `cold` (`scripts/run_solver.py:164`), while its help text,
    `main()`, the library and GLOSSARY say warm.
  - ADR 0006's rationale for self-referential scale totals assumes the warm chain
    (`0006:22-25`), which is not what runs.

**Assessment: strong at the interface, monolithic in the backend.** The QP is defined only
by generic arrays and columns, but those arrays are fed into Clarabel without a seam.

### Stage 5 — Uncertainty propagation (`scripts/uncertainty_propagation.py`, `uncprop.py`)

**For**
- **Solver-agnostic data seam.**
  - It discovers its const run from the opt sidecar (`uncprop.py:173-188`).
  - From the opt run it uses only `F_opt` and `cols_t`; it never imports `solver.py`.
  - It writes `sigma_a_star`, `lower` and `upper` with both parent ids
    (`uncprop.py:417-442`).
- **ABC.** `BaseUncProp.compute(opt_run_id, alpha)` (`uncprop.py:97-110`). The Cholesky
  backend (CHOLMOD or `splu`) is chosen at runtime (`uncprop.py:387-403`).
- **No copper strings.**

**Against**
- **It misreads Stage-3 row semantics.**
  - `hard_mask = row_meta["is_hard"]` includes the nonneg-cone *inequalities*: flow bounds
    and ratio bands. `condition_sigma_c` sets their σ_c to 0 (`uncprop.py:293, 311-314`;
    `_conditioning.py:103-105`).
  - Two lines later, the same function builds its residual diagnostic from zero-cone rows
    only, "because nonneg-cone hard rows are inequalities" (`uncprop.py:205-210`).
  - **[inferred]** Under the GLS update every edge with a structural lower-bound row is
    pinned (σ_a* → 0), and redundant hard rows threaten the factorisation.
  - ADR 0039 records the fix as open: Stage 5 "must drop slack bound rows as well as slack
    bands" (#27, `0039:35-36`).
- **The σ floors exist in three copies.** They are in `_conditioning.py:37-39`, in the
  `ClarabelSolver` defaults (`solver.py:147-150`) and in `DEFAULT_*` in `uncprop.py:56-63`.
  Tests hard-code them a fourth time, and no test asserts the copies are equal. Stage 5
  does not read the solver's floors from the opt sidecar, and Stage 4 does not record them.
- **No in-memory API.** `compute` takes a run id; tests bypass the constructor via `__new__`
  (`tests/test_uncprop.py:127-133`). It conditions on `const_run.a0`, not the solver's
  `a_prior`, and ignores `prior_mode` (`uncprop.py:195, 286`).
- **Never exercised in production.** `data/proc/uncprop/` is empty in this checkout.

**Assessment: the contract is modular, the semantics are coupled.** The interface would
allow Stage 5 to sit behind any solver. But it interprets a Stage-3 flag (`is_hard`) in a
way that Stage 3's newer row families (ADR 0039) broke, and no run has caught it.

### Stage 6 — PSUT / PIOT (`scripts/psut.py`, `psut.py`, `piot.py`)

**For**
- **Two input seams.**
  - `PSUT.from_results(results, acc, primary)` accepts any object that satisfies the
    `OptimResultsLike` Protocol (`psut.py:67-78, 280-345`), touching only `timestep`, `F_opt`
    and `cols_t`.
  - `from_opt_run` follows the provenance chain opt sidecar → const sidecar → acc id
    (`psut.py:347-399`) and never loads the constraint arrays.
- **Imports go one way.** Module-level imports are only `acc`, `logging` and `registry`
  (`psut.py:60-62`); `piot` imports only `psut`. Nothing from Stage 2–5.
- **Data-driven aggregation.** `aggregate(region, sector, keep_crossings)` takes
  concordance dicts and rejects maps that merge entity kinds (`psut.py:429-498`). Its steps
  are recorded in `stage_config`.
- **Tested in isolation.** `test_psut.py` (32) and `test_piot.py` (13) run on a toy acc.

**Against**
- **The block taxonomy is the copper case's.** `_BLOCK_RULES` and `_ROW_COL` hard-code the
  boundary `entity_sub` labels `Nature`, `SEM` and `FD` (`psut.py:83-105`), and the comment
  says so (`:82`). Any edge outside the taxonomy raises (`psut.py:305-309`). A case with
  other boundary labels cannot be built. The docstring names this as the site for ADR 0034
  roles (`psut.py:25-27`).
- **Constructs are a closed if-chain.**
  - `CONSTRUCTS = ("market_share", "suh_product")` (`piot.py:47`), and `suh_product` raises
    `NotImplementedError` (`piot.py:205-208`).
  - ADR 0037's second axis (Suh vs Xu–Zhang) is not in the API.
  - Unlike Stages 2–3, there is no registration hook.
- **Coupling at runtime.**
  - `from_opt_run` loads the acc by id only, with no version check. Stage 3 enforces one;
    the psut sidecar records `{id}` only (`scripts/psut.py:85-88`).
  - Reading an opt run lazily imports `pysuto.solver`, which needs Clarabel installed
    (`registry/optim.py:145` → `solver.py:37`).
  - **[inferred]** `psut → registry → meta → utils` imports `matplotlib.pyplot` at module
    level (`utils.py:11`), against README:1221.
- **Uncertainty is not consumed.** Stage 6 is point-estimate only (README:773, ADR 0007:39-40).

**Assessment: a strong consumer, bound to the case.** In code terms it is the cleanest
downstream module. In domain terms it is the most label-bound one.

---

## 4. Cross-cutting infrastructure

**Supports the modularity claim**
- **One sidecar schema for every stage.** It is `{run_id, stage, label, created_at,
  parents:{inputs, upstream}, stage_config, stage_summary}` (`registry/_sidecar.py:191-205`),
  and inputs are content-hashed and versioned (`input_registry.py:202-237`). The hand-off is
  file-based at the driver level, and no stage driver imports another.
- **The module-level import graph of `src/pysuto` is acyclic** (AST analysis). `utils` and
  `logging` import nothing from the package, and no `src/` module imports from `scripts/`.
- **The ACC is declarative data.** It is built from `root.ods` (sectors, regions,
  valuations, time), `rules.ods` (supply/use/trade/accounting), `bounds.ods` and
  `masks.csv` (`acc.py:19-36, 642-670`), selected by `pipeline.acc_id`.

**Limits the modularity claim**
- **The registry reconstructs each stage's result class**, so it depends on the stage
  modules. The resulting cycles are broken by five commented lazy imports
  (`registry/constraint.py:136`, `optim.py:145`, `psut.py:118`, `uncprop.py:110`;
  `psut.py:642`).
- **The README layer diagram is violated in places.**
  - `conc` → `feed` at module level (`conc.py:14`).
  - `acc` → `impute.network` (`acc.py:1218`), a domain module importing a pipeline module.
  - `registry` → `meta` at module level (`registry/__init__.py:27`).
  - Data-layer functions import `diag` lazily (`feedpipe.py:496`,
    `constraints/engine.py:804`).
  - About a dozen function-local imports break no cycle and carry no comment, against
    CLAUDE.md's rule.
- **A loaded ACC loses its rule-derived masks.** `AccountingStructure.load()` sets
  `ruleset=None` and rebuilds the supply/use masks from the literals `Process`/`Flow`/`SB`
  (`acc.py:1297, 1324` → `795-800`). The `tradable` mask comes back all-False, but no code
  reads it (`acc.py:806` is its only occurrence). ADR 0034 records this.
- **Configuration is spread across drivers.** `config.yaml` is parsed for `pipeline.acc_id`
  only (`config.py:50-52`). Every other knob is an UPPER_CASE constant in a driver, some
  knobs are not written to `stage_config` (Stage 3 anchors, Stage 4 σ floors), and
  README:836, 857-860 still describe a `config.yaml` with `p`, `q` and `alpha_min`.

**Copper tokens in `src/pysuto/`, by kind** (case-insensitive grep; most hits are
docstrings). Hard-coded logic or defaults:

| Site | Literal | Effect |
|---|---|---|
| `impute/share.py:57-70, 116-118` | CP, PSC, Cathode, MESC, USC, CSXEW, CER, FU | Copper's stock-change topology is the module. |
| `psut.py:83-105` | Nature, SEM, FD | Another case's boundary raises. |
| `impute/network.py:85, 98, 253-254, 328`; `impute/trade.py:72` | CC, SEM | Content valuation must be named `CC`. |
| `param_relation.py:160, 163` | concentration, Mass | Gross valuation must be named `Mass`. |
| `constraints/engine.py:272, 276, 807` | CO/Mining/Ore/WR/Tailings/Slag/PSC; "1995" | Copper defaults in the library. |
| `constraints/ratio.py:208, 461`; `impute/parametric.py:54` | stock_bound, stripping_ratio | Domain names in generic builders. |
| `feedpipe.py:388`, `meta.py:67` | ROW | Rest-of-world must be named `ROW`. |
| `acc.py:795-800`, `constraints/mass_balance.py:32`, `sum.py:98`, `engine.py:468` | Process/Flow/SB | Framework words per ADR 0034, but repeated as literals. |
| `diag/*` | "1995", CC, Mass, FD, Nature/SEM/FD | Copper defaults in the diagnostics. |

ADR 0034, the decision that would move these to a `Role` enum bound by the case, has the
status **proposed** (`0034:3`). `node_index.csv` has no `role` column.

---

## 5. How the argument could be strengthened (not done; listed for triage)

Ordered by how much each item would change the verdict:

1. **Land ADR 0034 (#26 §1).** This is the single change that turns "C: no" into "C: yes".
   It touches `impute/share.py`, the `CC`/`Mass`/`SEM` literals, `psut.py:83-105` and
   `acc.load()`. Add a non-copper toy fixture (ADR 0037:51-52 defers this to #26).
2. **Fix Stage 5's reading of hard inequalities (#27).** This is a correctness item more
   than a modularity one. It is also what keeps Stage 5 unvalidated.
3. **Make cross-stage constants single-sourced or checked.**
   - `P_PROP`: record it in the imp sidecar, then read or assert it in Stage 3.
   - σ floors: Stage 5 reads them from the opt sidecar.
   - Flat-prior detection: use the imputation log, not `σ ≥ 1e5`.
4. **Record builder lineage.** `build_trade_shares.py` should write the feed run ids it read
   into the `trade_shares` params version. `icsg_feed.py` should refuse weights built from
   other runs than the latest ICSG/BACI.
5. **Add a solver abstraction layer (decided 2026-10-08).**
   - **What:** a neutral QP object `(P, q, A, b, cones)` built from the Stage-3 run, plus a
     backend interface behind which Clarabel is the first implementation.
   - **What it replaces:** `_problem_skeleton`, `_solve_all_via_blocks`, `_solve_all_legacy`
     and `feasibility_check`, which today call Clarabel inline (`solver.py:286-314, 423-428,
     767-842`).
   - **What it must also cover:**
     - Clarabel's telemetry, which is part of `OptimResult` (`ITER_DTYPE`, status strings,
       `_info_to_dict`, `solver.py:51-63, 358-381`) and is read by diag
       (`diag/solver.py:73, 157, 1046, 1141-1142`). It needs a neutral form.
     - The accepted-status set `{Solved, AlmostSolved}`, which must map to neutral statuses.
       Any other status still raises.
     - KKT-factorisation reuse across years, a backend capability that the interface should
       expose without requiring it.
   - **What does not change:** persistence (`OptimRunRegistry`) and Stage 5.
6. **Optional seam**, only if a second construct is actually wanted: a construct registry in
   `piot.py`. Without that demand it would be speculative abstraction.

Documentation drift found along the way, for a separate clean-up:
- README "five stages", the `config.yaml` knobs, `conftest.py` (README:1170) and the
  "feedpipe end-to-end" tests (README:1177).
- The `--prior-mode` help text.
- The `lp.py:6-7` docstring.
- `psut.py:637-641` (A/L "not implemented").
- `registry/__init__.py:2, 18` ("Stages 1–4").
- ADR 0007:37 (`psut_indicators.py` was never created).
- README:525: hard rows "always active" (but `ratio.py:626-642` deactivates them).
