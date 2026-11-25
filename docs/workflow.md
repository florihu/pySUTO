# Workflow

How a run is made and recorded: the driver walk-through, the run ids, the sidecars, the input registries, the knobs and the sweep. The stages themselves are in [architecture.md](architecture.md); the README holds the quickstarts.

> Contributors / LLM assistants: see [CLAUDE.md](../CLAUDE.md) for behavioural rules.

> **Tuning knobs are UPPER_CASE names**: the Stage 2 and 3 knobs (`P_PROP`, `Q_REG`, `ALPHA_MIN_REG`, `FLAT_DEFAULT_CV`, time-imputer settings, Stage 3 σ floors) are fields of the copper case spec ([scripts/copper.py](../scripts/copper.py), #120), the others constants in the `# USER CONFIGURATION` block of [stages.py](../src/pysuto/stages.py) (solver tolerances, #186). [`config/config.yaml`](../config/config.yaml) holds only `pipeline.acc_id`; `load_config()` parses nothing else. Modules keep their existing kwarg signatures, so library use (tests, REPL) still works with module defaults.

```python
from pysuto.acc import AccountingStructure
from pysuto.collector import FeedCollector
from pysuto.config import load_config
from pysuto.feedpipe import run_feed
from pysuto.registry import DataFeedRegistry
from pysuto.input_registry import InputRegistries
from pysuto.impute import (
    ImputationEngine, FillImputer, NetworkImputer,
    TimeImputer, ShareImputer, FlatPriorImputer,
)
from pysuto.relation.sources import Fill, ParamSource, trade_scope
from pysuto.constraints import (
    ConstraintEngine, MassBalanceBuilder, SumConstraintBuilder,
    RatioConstraintBuilder, RatioBoundBuilder, FlowBoundsBuilder,
)
from pysuto.clarabel_backend import ClarabelBackend
from pysuto.solver import Solver
from pysuto.uncprop import KalmanUncProp
from pysuto.diag import AggDiagnostic, ImpDiagnostic

cfg           = load_config()           # config/config.yaml: pipeline.acc_id only
ACC_ID        = cfg.pipeline.acc_id     # "copper"
P_PROP        = 1.3                     # global Lp exponent — a driver constant
Q_REG         = 3.0                     # parametric fill's backward-branch knee sharpness
ALPHA_MIN_REG = 0.05                    # parametric fill's inverse-multiplier floor

# ── Stage 0: initialise the input registries (scan-and-bump) ─────────────
# Walks data/input/{feed,acc,params,mask}/<id>/current/ and bumps any id whose
# load-bearing files have changed since the last registered version.
inputs = InputRegistries.initialize(root="data/input")

# ── Stage 1: one feed build per data source ───────────────────────────────
# A FeedSpec names a feed's loaders, concordance policy and hooks (copper's three
# are ICSG_SPEC, BACI_FEED and CRUDE_ORE_FEED in scripts/copper_feed.py); run_feed
# reads the registered inputs (values.csv, conc.xlsx, meta.xlsx) of data_dir,
# builds, propagates and registers the feed run.
registry = DataFeedRegistry(odir="data/proc/feed/registry")
registry.load_index()

for spec in feed_specs:
    entry = run_feed(spec, acc_id=ACC_ID, data_dir="data")   # entry.run_id, .run_dir

# ── Stage 2: aggregation + imputation ─────────────────────────────────────
collector = FeedCollector(registry=registry, acc_struct=acc)
collector.select(["baci", "crude_ore"])           # input feed-ids → latest run wins
agg_result = collector.aggregate(p=P_PROP)

agg_diag = AggDiagnostic(agg_result, acc_struct=acc, label="copper")
agg_diag.summary()

# Every params.csv resolves through the input registry, so the run's sidecar
# records the exact (param_id, version) that fed each fill.
def csv(param_id):
    return str(inputs.params.resolve(param_id) / "params.csv")

engine = ImputationEngine(acc_struct=acc)
# Cascade, iterated to a fixpoint: trade seeds → network closure → parametric
# fallbacks on whatever the network leaves undetermined.
engine.register_cascade(
    [
        FillImputer(Fill(ParamSource.from_params(csv("conc"), acc, twin=True),
                         "forward", trade_scope(acc), id="trade"), "trade", p=P_PROP),
        NetworkImputer(eff_csv=csv("eff"), conc_csv=csv("conc"),
                       ore_grade_csv=csv("ore_grade"),
                       stripping_ratio_csv=csv("stripping_ratio"),
                       p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG,
                       cv_cap=NETWORK_CV_CAP,
                       stock_sectors=("MESC", "USC"), id="network"),
        *[FillImputer(Fill(ParamSource.from_params(csv(pid), acc), "both", id=pid),
                      "parametric", p=P_PROP, q=Q_REG, alpha_min=ALPHA_MIN_REG)
          for pid in ("conc", "ore_grade")],
    ],
    max_iter=10,
)
# Terminal imputers, in order.
engine.register(TimeImputer(window=3, method="linear", min_obs=2, gap_inflation=1.05,
                            interpolate_only=True, p=P_PROP))
engine.register(ShareImputer.from_stock_bounds_csv(
    csv("stock_bounds"), p=P_PROP, scale_abs_floor=100.0,
    seed_cv=STOCK_SEED_CV, seed_abs_floor=STOCK_SEED_ABS_FLOOR_KT))
engine.register(FlatPriorImputer(acc_struct=acc, mode="noisy_zero", sigma_high=1.0e6,
                                 p=P_PROP))
param_refs = [inputs.params.entry(pid) for pid in
              ("conc", "ore_grade", "eff", "stripping_ratio", "stock_bounds")]

agg_result = engine.run(agg_result)
agg_diag.plot(odir="data/proc/imp/copper/diag", show=False)
ImpDiagnostic(agg_result, acc_struct=acc, label="copper").plot(odir="data/proc/imp/copper/diag")

from pysuto.registry import ImputationRunRegistry
imp_registry = ImputationRunRegistry(storage_dir="data/proc/imp")
imp_run_id = imp_registry.register(
    agg_result,
    label    = "copper baseline",
    parents  = {
        "inputs": {"acc": inputs.acc.entry(ACC_ID), "params": param_refs},
        "upstream": {"feed_run_ids": [s["feed_run_id"] for s in collector.selected]},
    },
    imputers = engine.imputers,
)

# ── Stage 3: build the constraint system ──────────────────────────────────
const_engine = ConstraintEngine(
    acc_struct=acc, p=P_PROP,
    sigma_a_cv_min=SIGMA_A_CV_MIN, sigma_a_abs_floor=SIGMA_A_ABS_FLOOR,
    sigma_c_cv_min=SIGMA_C_CV_MIN, sigma_c_abs_floor=SIGMA_C_ABS_FLOOR,
    nodata_anchor_cv=NODATA_ANCHOR_CV)
const_engine.register(MassBalanceBuilder())
const_engine.register(SumConstraintBuilder.from_mask_anchor(
    acc, mask="supply", anchor_values=agg_result.f_agg, sigma_cv=0.05, p=P_PROP))

# Two builders consume the same param_id ("conc", "eff").
RATIO = dict(sigma_rel_floor=RATIO_SIGMA_REL_FLOOR, scale_abs_floor=RATIO_SCALE_ABS_FLOOR)
CONST_BINDINGS = [
    ("conc",         RatioConstraintBuilder, RATIO),
    ("eff",          RatioConstraintBuilder, RATIO),
    ("conc",         RatioBoundBuilder,      {}),
    ("eff",          RatioBoundBuilder,      {}),
    ("stock_bounds", RatioBoundBuilder,      {}),   # agg=true rows; |Σ b| ≤ k·Σ a  (b=stock numerator, a=Cathode denom; PSC only since ADR 0040; the Σb−r·Σa form, see GLOSSARY.md §"Node-scoped vs set-scoped")
]
for param_id, Builder, kwargs in CONST_BINDINGS:
    const_engine.register(Builder.from_csv(
        str(inputs.params.resolve(param_id) / "params.csv"),
        acc_struct=acc, **kwargs))
const_engine.register(FlowBoundsBuilder())

constraint_run = const_engine.run(
    agg_result = agg_result,
    timesteps  = [str(t) for t in range(2010, 2024)],
    label      = "copper",
    parents    = {
        "inputs": {
            "acc":    inputs.acc.entry(ACC_ID),
            "params": [inputs.params.entry("conc"), inputs.params.entry("eff"),
                       inputs.params.entry("stock_bounds")],
        },
        "upstream": {"imp_run_id": imp_run_id},
    },
)
# Persists to data/proc/const/<const_run_id>/ where const_run_id is
# const_<YYYYMMDD_HHMMSS>_<8-char-uuid>.

# ── Stage 4: solve per timestep ───────────────────────────────────────────
solver = Solver(ClarabelBackend(), const_dir="data/proc/const")
results = solver.solve_all(
    run_id=constraint_run.run_id,
    timesteps=[str(t) for t in range(2010, 2024)],
    feasibility_check="all",
)
optim_run_id = solver.persist(
    run_id=constraint_run.run_id,
    results=results,
    output_dir="data/proc/opt",
    label="copper",
)
# results[t].F_opt   → optimised flow vector (block path: year-local over cols_t)
# results[t].eps_opt → slack on free-cone rows
# results[t].residuals = G_t @ F_opt - c_t   (≈0 on hard rows, ≈ε on free rows)
# results[t].cols_t / rows_t → global indices of the year-local slices (None on legacy)

# ── Stage 5: posterior uncertainty propagation ───────────────────────────
prop = KalmanUncProp(
    opt_dir     = "data/proc/opt",
    const_dir   = "data/proc/const",
    uncprop_dir = "data/proc/uncprop",
)
uncprop_result = prop.compute(optim_run_id, alpha=0.05)
uncprop_run_id = prop.persist(uncprop_result, optim_run_id, label="copper")
# uncprop_result.per_timestep[t].sigma_a_star  → (E,) posterior std
# uncprop_result.per_timestep[t].lower / upper → (E,) Gaussian CI, clipped to [l, u]

# ── Stage 6: PSUT / PIOT (reads the opt run, not Stage 5) ─────────────────
from pysuto.psut import PSUT
from pysuto.registry import PsutRunRegistry

psut = PSUT.from_opt_run(optim_run_id)             # acc via opt → const sidecars
psut_run_id = PsutRunRegistry("data/proc/psut").register(
    psut,
    parents = {
        "inputs":   {"acc": {"id": psut.acc_id}},
        "upstream": {"opt_run_id": optim_run_id, "const_run_id": psut.const_run_id},
    },
    label   = "copper",
)
piot = psut.aggregate(region="World", keep_crossings=True).to_piot("market_share", level="L")
# psut.table        → one row per edge × period × valuation, in blocks S/U/Y/E/O/P/W
# piot.footprint("E", "2020"), piot.ew_mfa("2020") → derived views, never persisted
```

### Output run-id format

All six stages mint run-ids using one harmonized pattern:

```
{stage}_{YYYYMMDD_HHMMSS}_{8-char-uuid4-hex}
```

`stage ∈ {feed, imp, const, opt, uncprop, psut, sweep}`; a feed run embeds its feed id
(`feed_<feed_id>_…`, e.g. `feed_baci_20260915_105105_187857d6`).  No label, acc-id,
or parent run-id is embedded — the label lives only in the registry, and parent
references live in the per-run sidecar `metadata.json`.

### Per-run sidecar `metadata.json`

Every run writes a self-contained sidecar at `<run_dir>/metadata.json`:

```jsonc
{
  "run_id":     "imp_20260518_143055_b7e2f3a1",
  "stage":      "imp",
  "label":      "copper baseline",
  "created_at": "2026-05-18T14:30:55Z",
  "parents": {
    "inputs": {                                     // input-registry refs
      "acc":    {"id": "copper", "version": 2, "date": "...", "code": "...", "label": "..."},
      "params": [{"id": "conc", "version": 2, ...}, ...]
    },
    "upstream": {                                   // upstream output runs
      "feed_run_ids": ["feed_20260518_141055_aaaa1111", ...]
    }
  },
  "stage_config":  { ... },
  "stage_summary": { ... },
  "environment": {                                  // what produced the run (#210)
    "pysuto_version": "0.1.0rc1", "git_commit": "84572e4...", "git_dirty": false,
    "python": "3.9.19", "numpy": "...", "scipy": "...", "pandas": "...",
    "pyarrow": "...", "clarabel": "..."
  },
  "footprint": {"wall_s": 41.7, "peak_rss_mb": 912.4}
}
```

`environment` and `footprint` are written by `build_sidecar` for every stage and sit
outside `parents`, so `python -m pysuto compare` (which reads only `parents.inputs` of a
sidecar) is blind to them. `git_commit` and `git_dirty` (tracked files only) come from
the repository of the package directory and are `null` without git metadata; a package
that is not installed is `null`. `wall_s` is the stage body's wall time
(`registry.stage_timer`, set around `run_feed` and `stages.run_imp` ... `run_psut`;
`null` outside a timer). `peak_rss_mb` is the *process* peak resident memory so far
(`getrusage`; `null` on Windows), so a driver that runs several stages in one process
reports the maximum of them all; `python -m pysuto run` starts a fresh child per step.
A feed run's `stage_config` records the Lp exponent `p` it propagated with, the spec's
policy fields, the qualified name of each hook and `knobs` (the module constants the
hooks read).

The top-level `registry.json` per stage is a trimmed index:
`{run_id, stage, label, created_at}` per entry.  Everything richer is in
the sidecar.

### Input registry (`data/input/`)

Four kinds (`feed`, `acc`, `params`, `mask`), each with its own per-kind
`registry.json`:

```
data/input/<kind>/
├── registry.json
└── <id>/
    ├── current/      ← live working copy; user edits here
    ├── v1/           ← snapshot; unchanged files = hardlinks to the prior snapshot
    └── v2/
```

When `InputRegistries.initialize()` runs at the top of any init script,
the registry hashes the *load-bearing* files in each `<id>/current/`
and compares against the latest registered version.  If anything
changed, `v_{N+1}/` is materialised — with **hardlinks** for files
unchanged from the prior version, fresh copies for files that changed.
Registering is a pipeline step's call. A lookup or a diagnostic uses
`InputRegistries.load()`, which resolves against what is registered and
never versions a `current/` (#93).

Load-bearing files per kind:

| Kind | Load-bearing files |
|---|---|
| `params/<id>/` | `params.csv`, plus `parents.json` where a builder writes one (`trade_shares`, #53) |
| `acc/<id>/` | `root.ods`, `bounds.ods`, `rules.ods`, `masks.csv` (optional; an absent file hashes to empty), plus `acc.json` where an id declares its acc spec in one file (#57, [ADR 0046](adr/0046-an-accounting-structure-is-declared-by-a-spec.md)); `scripts/acc.py` builds from `acc.json` when the version holds it, refuses a version that also holds the ODS/CSV files, and writes `build_record.json` beside the structure it builds (#53) |
| `mask/<id>/` | `rules.csv` |
| `feed/<id>/` | per-feed declared list; default = "every file in `current/` not matching `*.bib` / `*.md`" |

Downstream runs record `(id, version)` in their sidecar
`parents.inputs.*` so any output run-id can be traced back to the exact
input-bytes that produced it.

### Pipeline and knobs

A **knob** is an UPPER_CASE name that a driver lists in its `KNOBS` tuple: for Stages 2
and 3 a field of the case spec's section, which `--set` replaces
([case.py](../src/pysuto/case.py), #120), for Stage 4 a constant of the `# USER CONFIGURATION` block of
[stages.py](../src/pysuto/stages.py) (`OPT_KNOBS`, #186). Every Stage 2–4 driver takes `--set KNOB=VALUE`
(repeatable; VALUE a Python literal, type-checked against the constant before `main()`
loads anything) and `--run-id-file PATH`; the applied pairs land in the sidecar's
`stage_config["overrides"]` beside the effective values, which the existing keys keep
recording ([knobs.py](../src/pysuto/knobs.py), #100):

```bash
python scripts/run_solver.py --const-run-id <id> --set TOL_GAP_REL=1e-6 --set QP_VARIABLES=flows
```

The Stage 2–6 drivers share `--case`, `--data-dir` and `--run-id-file`; what differs, and why (#219):

| Option | Stage 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| upstream run | `--feed-id` (repeatable) | `--imp-run-id` | `--const-run-id` | positional `opt_run_id` | positional `opt_run_id` |
| `--set` (knobs) | yes | yes | yes | no: Stage 5 has no knobs (constants of `stages`) | no: its one knob is `--primary` |
| `--config` | yes | yes | no: Stage 4 reads the acc id from the const sidecar | no | no |
| `--imp-dir` / `--const-dir` | `--imp-dir` (output) | `--imp-dir` (input), `--const-dir` (output) | `--const-dir` (input) | no | no |

The upstream id is an option where a stage reads an input directory a sweep redirects (`--imp-dir`,
`--const-dir`, #108) and positional where the stage resolves its parents from the sidecar and
only one id exists. `scripts/solver_stats.py` takes its `opt_run_id` as a required argument.

[pipeline.py](../src/pysuto/pipeline.py) composes the stages as steps (#103). It runs each
step in a fresh child process (`python -m pysuto._step`) that calls the `pysuto.stages.run_*`
function with the same overrides (same names, types and precedence as `--set`; the child's memory
returns to the system, ADR 0045), never importing `scripts/`, and threads the run ids into a `Chain` ([ADR 0045](adr/0045-knobs-are-overridden-at-the-driver-boundary.md)):

```python
from pysuto.pipeline import Chain, Pipeline

base  = Chain.from_opt_run("opt_20261020_112726_2f451f31")     # resolves const, imp, feeds
chain = Pipeline(steps=("imp", "const", "opt"), base=base, case="scripts/copper.py").run(
    {"imp.P_PROP": 1.5, "const.P_PROP": 1.5}, label="p = 1.5")  # <stage>.<KNOB> keys
```

`steps` is any contiguous slice of `ALL_STEPS` (`feed, imp, const, opt, uncprop, psut`; a sweep varies the knobs of `STEPS`, `imp, const, opt`); the first step reads its
upstream from `base`. Knobs listed in `pipeline.TIED` (the cross-stage `P_PROP`) must be
set together and to one value. A driver exiting non-zero raises `StepFailed` with the tail
of its output (`<data-dir>/logs/pipeline/<slug>_<step>.log`, the label with every run of characters outside `[\w.-]` replaced by `_`). Every registry-writing driver takes
`data/proc/.stage.lock` (#101), so a pipeline and a hand-run driver cannot interleave.

A **sweep** runs a set of chains that differ only in knob values and records them as one
`sweep` run (#105), after scikit-learn's `ParameterSampler` / `ParameterGrid`. An `Axis`
is one knob, or a tied set sharing one value, with a list of values or a scipy
distribution (`Axis.interval(keys, lo, hi, log=False)`). A **random** sweep (the default)
draws `n_points` from a seeded, scrambled Latin hypercube, so every list level gets the
same share of points; a **grid** sweep runs every combination of list values:

```python
from pysuto.pipeline import Axis, Sweep

sweep = Sweep(Pipeline(base=base, case="scripts/copper.py"), [
    Axis.interval(("imp.P_PROP", "const.P_PROP"), 1.1, 1.5),           # tied
    Axis.interval("const.SIGMA_C_ABS_FLOOR", 1e-6, 1e-3, log=True),
    Axis("imp.CASCADE_ORDER", [("trade", "network", "conc", "ore_grade"),
                               ("network", "trade", "conc", "ore_grade")]),
], n_points=30, seed=20261021)
sweep.plan()       # distinct runs per step: a sweep over solver knobs runs imp and const once
sweep_id = sweep.run(label="knob sweep")
```

Points are drawn at declaration and run one after another; within a sweep a step whose
upstream runs and overrides an earlier point already ran reuses that run. A point whose
driver fails is recorded `failed` with its output tail and the sweep goes on. Unless set,
sweep points pass `opt.FEASIBILITY_CHECK = "none"` (#104). Before the first point the
sweep refuses to start when the planned runs (`STEP_GB` per step run) plus a 3 GB reserve
exceed the free disk; it records the git commit and refuses to `resume` at another one.
`scripts/sweep.py` declares a sweep in its USER CONFIGURATION block (`--dry-run`,
`--resume <sweep_id>`).

With `keep_intermediate=False` (`KEEP_INTERMEDIATE` in `scripts/sweep.py`, #108) only the
last step's runs reach `data/proc`. The pipeline writes `imp_dir` / `const_dir` into each step's spec, so each
point's imp and const runs (≈ 0.93 GB) land in `<scratch_dir>/<sweep_id>/`, a RAM-backed
tmpfs (`/dev/shm/pysuto-sweep`), and deletes a run as soon as no remaining point reuses
it. A point then costs the opt run's 0.08 GB of disk, and the manifest still names its imp
and const runs, so the point can be rebuilt from its overrides. Those opt runs cannot be
loaded by `SolverDiag` afterwards (their const is gone); the solve renders its diag while
the const still exists. A reboot empties the store, and `resume` reruns the dropped steps of
the points not done.

```
data/proc/sweep/sweep_<YYYYMMDD_HHMMSS>_<8hex>/
    metadata.json     parents.upstream = the base chain; stage_config = steps, mode,
                      n_points, seed, axes, fixed, commit, dirty, scipy
    manifest.json     per point: overrides, status, run ids, reused steps, times, error
    diag/             scripts/sweep_diag.py: points.csv, outcomes.csv, categories.csv,
                      retention_by_year.csv, sensitivity.csv, year_profiles.csv,
                      sensitivity_by_year.csv, verdicts.csv, SW-R-01, 02, 04–07
                      figures (SW-R-03 retired, #117)
```

`scripts/sweep_diag.py <sweep_run_id>` scores every done point against the base chain
(#107), reading only the base's const run and the points' opt runs. Every point gets the
cross-evaluated objective `J_base_at_point`, which is each point's posterior scored under
the base QP by `Solver.objective_at` and split by family. On
`opt_20261019_171746_3272682b` against `opt_20261020_112726_2f451f31` it gives the 66,702
of #33. Every point also gets the mass moved by valuation, and by category with the
**pooled retention** by category and year. A category with a signed edge (structural
lower bound below 0) or a zero prior has no retention, only its mass moved. Further
**outcomes** are declared (`Outcome(name, measure, select, window)`, #108, #117). The
measure is `mass_moved`, `retention` (the change in pooled retention against the base,
refused over a signed edge) or `total` (the relative change of the posterior, which also
sees a knob that moves prior and posterior together), pooled over the outcome's window,
by default the base's full span. The selector is a dict over the edge table's columns or
a callable mask that must match at least one edge. `MaxAbs` takes the worst of several
and is NaN when any part is; it is scored and judged, never ranked, because its |·| folds
a one-way effect into a V. `EveryCategory(valuation)` declares one retention outcome per
category of a valuation that has a retention. A **criterion**, `Criterion(item, outcomes,
limit)`, is met at a point when every |outcome| is below its limit; its verdict counts
the done points that meet it and names the worst point, outcome and value. The script
declares every CC category, Table S16's nine stages both ways, their maxima, Round-E
(c)'s exporter retention over 2010–2019, and #108's criteria A6 (the nine stages, 0.02)
and Round-E (c) (the exporters, 0.01). Through the same edge table it reproduces the
#110 REP's Round-E numbers for `opt_20261021_121332_acc68135` →
`opt_20261021_131744_58504ad3`: Chile's Concentrate→Smelting CC export retention over
1995–2024 is 1.0205 → 1.0222, and the CC mass moved is 285,108 → 256,349 kt CC. The axes
are ranked and sized per outcome. A random sweep gets SRRC and Spearman ρ for numeric
axes and two-level list axes (in their declared order), plus the rank regression's R²;
where R² < 0.5 the ranking is read off ρ. An axis is influential when |SRRC| ≥ 0.2 with
R² ≥ 0.5 (#108's rule). Every axis gets a **swing**, the outcome's mean over the top
third of the axis's values minus its mean over the bottom third (for a list axis the
spread of its level means, signed as its SRRC, second level minus first, when it has
two), which sizes what SRRC and ρ only rank; a full grid also gets
the two-way interaction range. Every retention outcome gets a **year profile**, ranked
year by year by the same rule. The figures are SW-R-01 (sensitivity map), SW-R-02 (each
criterion outcome against its top axis), SW-R-04 (CC retention boxes), SW-R-05 (criterion
bands), SW-R-06 (year envelope) and SW-R-07 (per-year sensitivity); SW-R-03 is retired.
Gross-mass totals are dominated by the waste-rock satellite, so `mass_moved_CC` is the
scale-free one.

**Comparing two chains (#157).** `python -m pysuto compare --data-dir <dir> --base <leaf> [<leaf> ...] --new <leaf> [<leaf> ...]` is the acceptance test of a refactor or a fix: it walks each chain from its leaf run id(s) up `parents.upstream`, pairs the stages both chains have (feeds by feed name), and compares every stored array of every `.npy`, `.npz` and `.parquet` file (dtype, shape, NaN positions, values, no tolerance; the imputation log included) and each run's `parents.inputs`. Stage 2 stores its per-feed rows (`agg_result.npz`: `F`, `Sigma`, `Omega`) in `--feed-id` order: the same feeds in another order are aligned by `feed_ids` before the comparison and noted as "feed order differs", not counted as a difference (#211). A check chain from Stage 1 is judged on its arrays: its feeds' `parents.inputs` differ from the reference's by design, because every Stage 1 registers a new `trade_shares` version (its `params.csv` byte-identical and hardlinked, its `parents.json` naming the new BACI feed run, #53's guard) and the rebuilt BACI and crude-ore feeds record the current acc version (#161). It prints one line per paired run, names every differing array as `<file>::<key>` with its first differing index, lists stages only one chain has as skipped (no difference), writes nothing under `<dir>` (no log, no lock) and exits 0 when identical, 1 on a difference, 2 on a usage error. A leaf names itself and its ancestors; psut and uncprop both hang off the opt, so a full chain takes both leaves (`--base psut_<a> uncprop_<b>`). `feed.pkl`, `diag/` and the sidecar's `stage_config` are not compared. The reference copper chain against itself (`--base`/`--new` both `psut_20261116_141702_37174685 uncprop_20261116_141543_6be9fa25`, `--data-dir data`): eight paired runs, 1,102 arrays, identical, 1 min 30 s.
