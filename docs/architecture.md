# Architecture and module reference

How pySUTO is built: the six stages, their mathematics, the modules and the import direction. The README is the entry point; [workflow.md](workflow.md) holds the run ids, the sidecars, the registries and the knobs; the vocabulary is in [GLOSSARY.md](../GLOSSARY.md).

> Contributors / LLM assistants: see [CLAUDE.md](../CLAUDE.md) for behavioural rules.

## Table of contents

1. [Overview](#overview)
2. [Architecture](#architecture)
   - [Stage 1 — Datafeed](#stage-1--datafeed)
   - [Stage 2 — Aggregation & Imputation](#stage-2--aggregation--imputation)
   - [Stage 3 — Constraint building](#stage-3--constraint-building)
   - [Stage 4 — Solver](#stage-4--solver)
   - [Stage 5 — Posterior uncertainty propagation](#stage-5--posterior-uncertainty-propagation)
   - [Stage 6 — PSUT/PIOT builder](#stage-6--psutpiot-builder)
   - [Glossary detail — framework and cross-stage](#glossary-detail--framework-and-cross-stage)
3. [Module reference](#module-reference)
4. [Import direction](#import-direction)
5. [References](#references)

---

## Overview

The problems pySUTO addresses, the six stages and the twelve commitments that no case can change (*What the framework assumes*) are in the [README](../README.md#how-it-works).

---

## Architecture

The data-flow diagram of the six stages, with the sweep beside Stages 2–4, is in the [README](../README.md#how-it-works).

Stage 6 branches off Stage 4 beside Stage 5 rather than after it: v1 materialises the point estimate `a*` and does not read the posterior σ ([ADR 0007](adr/0007-downstream-psut-piot-follows-piolab-suh-no-external-io-lib.md)). The sweep is no stage: it re-runs a slice of Stages 2–4 once per knob point, takes its feed runs from the base chain, and is scored from the points' opt runs ([Pipeline and knobs](workflow.md#pipeline-and-knobs)).

**Orthogonal architecture** — cross-cutting concerns (top) crossing every stage, over the layered one-way import stack (bottom):

```
ORTHOGONAL CROSS-CUTTING CONCERNS   (one service across the stages;
                                     a split cell: Stages 1–3 differ from 4–6)

               Stage 1    Stage 2    Stage 3    Stage 4    Stage 5    Stage 6
               Datafeed   Agg+Imp    Constr.     Solver   Posterior  PSUT/PIOT
             ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
Registry     │   feed   │   imp    │  const   │   opt    │ uncprop  │   psut   │
(run store)  │ registry │ registry │ registry │ registry │ registry │ registry │
             ├──────────┴──────────┴──────────┴──────────┴──────────┴──────────┤
Provenance   │ sidecar metadata.json · parents.inputs (versioned) ¹            │
(lineage)    │ + parents.upstream  (run_id chain, re-discovered)               │
             ├─────────────────────────────────────────────────────────────────┤
Logging      │ get_logger(name) per module; a driver calls start_run_log() —   │
             │ one file per module + one per run; loggers funnel to root ²     │
             ├────────────────────────────────┬────────────────────────────────┤
Config /     │ load_config(config.yaml) ·     │ no registry read: the inputs   │
inputs       │ InputRegistries.initialize()   │ come with the upstream run id; │
             │ acc/feed/params/mask, hash-    │ the built acc's build record   │
             │ bumped — read every run        │ is checked vs. the acc ref ³   │
             └────────────────────────────────┴────────────────────────────────┘
¹ Stages 1–3 fill parents.inputs; Stage 4 records the const run's acc ref there only
  when warm (the warm prior reads the acc), Stage 5 parents.upstream alone, and
  Stage 6 the const run's acc ref (id and version).
² A driver calls start_run_log(data_dir, name) (#190); importing pysuto writes nothing.
  A Stage 2–6 sidecar's stage_config.case names its case: case_sha256 is the
  identity (sha256 of the case module), case_source is informative (`path:<abspath>`
  by design, so it may be an ephemeral path) (#187).
³ scripts/acc.py writes build_record.json ({id, version, code} of the acc input version
  it built from) beside the built acc. The drivers of Stages 2, 3, 4 and 6 check it
  against their acc ref before they load the acc (#53), and Stage 2 checks each feed
  run's acc ref against its own. Two versions are compared by their specs, read from
  the registry's vN/ (#119), on the keys the consumer reads: a feed run on the
  edge-universe keys (sectors, regions, valuations, times, supply, use, tradable,
  accounting); Stages 4 (warm) and 6 against the built acc on those plus the roles;
  Stages 2 and 3 against the built acc on every key (bounds, masks, roles). A version
  that declares no roles (acc/copper v4 and older) is compared through the copper
  binding; any other differing key is logged at info level, and the ODS-or-JSON format
  does not count. A structure without the record warns (raises under --pin-inputs-from).

modules:  Registry=registry/ (BaseOutputRegistry · _sidecar · make_run_id)
          a sweep, no stage, keeps its runs in its own sweep registry
          Logging=logging.py   Config=config.py + input_registry.py

LAYERED MODULE STACK   (imports flow downward only — never upward)

┌────────────────────────────────────────────────────────────────┐
│  tutorial (top, beside the stage drivers) — tutorial/          │
│  toy: a case declared in code, acc spec, synthetic feeds,      │
│  params, the six stages wired into a run tree; imported by     │
│  nothing in pysuto                                             │
│  pipeline.py  ·  Pipeline · Sweep — calls pysuto.stages for    │
│  Stages 2–6, each step in a child process that resolves the    │
│  case reference; never imports scripts/ or the tutorial        │
│  scripts/                                                      │
└──────┬─────────────────────────────────────────────────────────┘
       │
       ▼  imports
┌────────────────────────────────────────────────────────────────┐
│  domain  ·  acc → conc → unc (mixin) · masks · relation        │
│          ·  flow_node_balance (St2–St3 prior check)            │
└──────┬─────────────────────────────────────────────────────────┘
       │
       ▼  imported by
┌────────────────────────────────────────────────────────────────┐
│  pipeline (the 6 stages)                                       │
│  feed → feedpipe → collector → agg → impute/ → constraints/    │
│       → solver → uncprop                                       │
│  psut → piot   (St6 — reads persisted runs; imports acc,       │
│                 logging, registry; never solver or uncprop)    │
└──────┬─────────────────────────────────────────────────────────┘
       │
       ▼  consumed by (never imports back up)
┌────────────────────────────────────────────────────────────────┐
│  diagnostics (leaf)  ·  diag/  — emits PNGs and the Plotly     │
│  Sankeys; never imported into the data layer · diag/truth.py   │
│  scores a synthetic case's estimates against its ground truth  │
└────────────────────────────────────────────────────────────────┘

cross-cutting (import nothing from the data layer):
  any layer    : logging · config · lp (lp_combine)
  all stages   : registry/  (BaseOutputRegistry + _sidecar provenance)
  St4–St5 only : _conditioning (condition_prior · condition_sigma_c)
  plugin seam  : BaseImputer→ImputationEngine (St2) ;
                 BaseConstraintBuilder→ConstraintEngine (St3) ;
                 OptimResultsLike protocol: solver result → PSUT (St4→St6) ;
                 CONSTRUCTS × LEVELS in PSUT.to_piot (St6; suh_product = #20)
```

| Stage | Module(s) | Input | Output |
|---|---|---|---|
| 1 — Datafeed | `feedpipe.py`, `feed.py`, `conc.py`, `unc.py`, `meta.py`, `registry/` | raw flow data + Excel sidecar | `DataFeedRegistry` entries: `W`, `F_root`, `σ_root` |
| 2 — Aggregation & Imputation | `collector.py`, `agg.py`, `impute/` | selected feed IDs | `AggregationResult`: `f_agg`, `σ_agg`, `Ω`, `ImputationLog` |
| 3 — Constraint building | `constraints/` | `AggregationResult` | `ConstraintRun`: `G`, `c(t)`, `σ_c(t)`, bounds |
| 4 — Solver | `solver.py` | `ConstraintRun` | `OptimResult` per timestep |
| 5 — Posterior uncertainty | `uncprop.py` | `OptimRun` + upstream `ConstraintRun` | `UncPropResult`: `σ_a*(t)`, `lower(t)`, `upper(t)` |
| 6 — PSUT/PIOT builder | `psut.py`, `piot.py`, `diag/psut.py` | `OptimRun` (+ `ConstraintRun` → `acc` through provenance) | `PsutRun`: tidy PSUT, one parquet per period; `PIOT` views and Sankeys derived on demand, never persisted |
| Sweep (no stage) | `pipeline.py`, `diag/sweep.py` | base chain + axes | `sweep` run: per point its overrides, status and run ids; scored by `scripts/sweep_diag.py` |

**One command runs the chain (#125).** `python -m pysuto run --case <id> [--from <stage>] [--to <stage>] [--data-dir D] [--base <opt_run_id>] [--label L] [--config C]` runs the six stages one after another, each in a fresh child process (#186, `python -m pysuto._step`), `--from`/`--to` any of `feed, imp, const, opt, uncprop, psut` (default the whole chain): each child resolves the `--case` reference, calls its `pysuto.stages.run_*` function with `--data-dir`, and `--config` reaches Stages 2–3 only (the only stages that read one); the `scripts/` drivers are thin argparse wrappers over the same functions and the command needs no `scripts/` directory. Each step's stdout goes to `<data-dir>/logs/pipeline/<slug>_<step>.log` (the slugified label) under the stage lock. `--from feed` is the case's Stage 1 hook (`CaseSpec.stage1`): the toy writes its synthetic feeds; copper's (#109, `scripts/copper_feed.py` `run_stage1`) runs `build_baci_reexport.py --source-version N` (the registered gross BACI snapshot, never the raw data) → `baci_feed` → `build_trade_shares.py` → `icsg_feed` → `crude_ore_feed` in that load-bearing order and returns their three feed run ids; the BACI parse, the gross registration, `icsg_clean --to-feed` and `build_ore_grade.py` stay manual. A file-path case may import the modules beside it, as a script run by Python does: `resolve_case` puts the file's folder first on `sys.path` (if absent) before it executes the file. Any later `--from` takes its upstream from `--base <opt_run_id>`, the base chain's parents as `Chain.from_opt_run` resolves them. `--data-dir` is required and the command runs from any directory, its file logs under `<data-dir>/logs` (copper's Stage 1 hook still runs its two feed-prep scripts from `scripts/`); the toy's six stages take about 25 s under their own labels (`python -m pysuto run --case toy --data-dir /tmp/toy`).

---

### Stage 1 — Datafeed

**What this stage does.** Runs once per data source. Reads metadata (including pedigree quality scores) from an Excel sidecar, builds a concordance table mapping source edges to root edges, assembles a sparse weight matrix, applies it to the source flow vector, and propagates pedigree-derived uncertainty through the concordance map. Results land in the `DataFeedRegistry`.

**Inputs:** raw flow data, classification mapping tables, Excel metadata file.
**Outputs per feed:** weight matrix `W`, root flow vector `F_root`, standard deviation `σ_root`, metadata sidecar.

**Two preprocessing scripts run before the feed pipelines and write into the input registry, so their order matters:** `scripts/build_baci_reexport.py` reads the registered gross BACI version and writes a **netted** one — re-exported tonnage re-attributed from the hub to its true origin and destination, with a residual ledger beside it ([ADR 0027](adr/0027-re-exports-are-re-attributed-at-feed-time.md)) — and must run before `scripts/baci_feed.py`; `scripts/build_trade_shares.py` then builds the ICSG fan weights from that netted matrix ([ADR 0002](adr/0002-single-endpoint-rows-fan-out-not-replicate.md) option D, [ADR 0024](adr/0024-fan-domestic-leg-bounded-by-own-production.md)) and must run before `scripts/icsg_feed.py`.

**Stock changes fan by supply, not by residual.** For the signed stock legs (MESC, USC) the trade-share builder draws a reported change from the region's cathode supply in proportion: its own production on the domestic leg, and each partner's cathode exports to it, read from the Cathode→FU leg, on the import legs. A region with neither leaves the change to the net-stock seed. BACI's own cathode fan books all of its trade on FU, so stock legs carry no BACI prior ([ADR 0043](adr/0043-a-stock-change-is-drawn-from-its-regions-cathode-supply.md)).

**A non-producer's zero is reported, not missing.** Where a region ICSG covers produces none of a product that year, by its raw ICSG production, the ICSG feed says so with an **observed zero**: covered at 0 and never filled by an imputer. It sits on the region's domestic concentrate leg where the trade shares weigh it at 0, priced by the feed's own knobs `OBSERVED_ZERO_CV` and `OBSERVED_ZERO_FLOOR_KT` (`scripts/copper_feed.py`) as `σ = max(OBSERVED_ZERO_CV·|U|, OBSERVED_ZERO_FLOOR_KT)` on the usage `U` its fans distribute (the same form as the net-stock seed, but a separate pair of constants that Stage 1 sets and no sweep varies), and on the edge its own mine (MF→Concentrate) or SX-EW (SXEW→CSXEW) output would leave on, priced at `OBSERVED_ZERO_FLOOR_KT`, ICSG's reporting resolution, unless another input (crude ore, an ore grade) reports that output there. The hard balances carry the zero up the mine chain. The solver then has to settle the region's smelter intake and exports against its imports in the open. The cathode legs (#96) were measured and rejected ([ADR 0024](adr/0024-fan-domestic-leg-bounded-by-own-production.md) amendments, #76, #96, #97, #200).

> **⚠ Any change to the ICSG *or* BACI feed invalidates the trade-share weights — rebuild them.**
> `build_trade_shares.py` reads both: the netted BACI matrix *and* the ICSG feed's own production
> (`icsg_intake_production`), which is what the [ADR 0024](adr/0024-fan-domestic-leg-bounded-by-own-production.md)
> domestic bound is computed against. Fan ICSG with stale weights and that bound is applied to the
> *previous* release's production, silently. Nothing detected it — `icsg_feed.py` loaded whatever
> version was registered and printed it, with no check that it post-dated the `values.csv` it was
> fanning. This happened for four consecutive rounds and cost 6.2 percentage points of sum-row z²;
> rebuilding alone moved 21,709 kt CC from domestic residual onto import legs. Since #53 the builder
> writes `parents.json` beside `params.csv` — the BACI feed run and the `baci`, `icsg`, `conc` and
> `eff` versions it read — and `icsg_feed.py` refuses weights whose parents differ from what is in
> force. The full order after a feed change is:
>
> ```
> BACI parse → register gross baci vN → icsg_clean.py --baci-version N --to-feed (→ icsg)
>   → build_baci_reexport.py --source-version N → baci_feed.py
>   → build_trade_shares.py → icsg_feed.py → Stage 2
> ```
>
> Since #63 the ICSG parse reads BACI too: `icsg_clean.py --baci-version N` nets imported anode
> off the direct-to-refinery route using the registered **gross** BACI parse vN and records N in
> the ICSG input's `parents.json`. `build_baci_reexport.py --source-version N` refuses to net when
> that record names another version or none, so a BACI parse change re-runs the ICSG parse before
> the netting.
>
> Check which weights a run used: `data/proc/feed/registry/<run_id>/metadata.json` →
> `parents.inputs.trade_shares.version`. `build_trade_shares.py` also refuses to build when the
> newest BACI feed run was built from an older `baci` input than the one in force (rerun
> `baci_feed.py` first).
>
> **One edge of the order is a loop, and the hook does not run it.** `build_ore_grade.py` reads the
> newest ICSG and crude-ore feed runs' `F_root` (ore mass, mine output) and the ICSG feed reads its
> `ore_grade` params for the observed zeros, so the grades are built from the *previous* ICSG run.
> The builder records those runs and the `eff` / `conc` versions in `parents.json` beside
> `params.csv` (the form of the trade-share record), which makes a stale grade readable in the sidecar; no
> guard refuses it, because after any hook run the newest ICSG run is by construction newer than the
> grades. Rebuild it by hand (`python scripts/build_ore_grade.py`) after the feeds when the ICSG or
> crude-ore feed changed, then rerun `icsg_feed.py`.
>
> **Registering the gross BACI version** (the step between the parse and `icsg_clean.py`) is by
> hand: `baci_parse_feed.py` writes `data/input/feed/baci/v0/`; set aside any
> `current/reexport_*.csv` (a leftover `reexport_ledger.csv` makes a gross version look netted, and
> `icsg_clean.py` refuses it), copy `v0/{values,quantity_recon,quantity_cv_scale}.csv` into
> `current/`, and register (`InputRegistries.initialize`; every build script does it on start).
>
> **Hand runs take the stage lock.** `build_baci_reexport.py`, `build_trade_shares.py`,
> `build_ore_grade.py` and `icsg_clean.py` write registries, which have no locking of their own,
> so each takes `<data-dir>/proc/.stage.lock` like a stage driver; under the Stage 1 hook, whose
> caller holds it, the two it runs join that hold (the pipeline names itself the holder in the child's `PYSUTO_STAGE_LOCK_PID`).

#### The parse layer (`scripts/parse/`)

Upstream of the feed pipelines, one reader per source turns a published release into the tidy
`values.csv` a feed consumes. It is worth naming explicitly: a source-format quirk that survives the
parse is indistinguishable from real data by the time it reaches the solver, and three of the last
four correctness rounds were defects here rather than in the model.

| | chain | writes |
|---|---|---|
| **ICSG** | `icsg_{2005,2014,2023f}.py` (one per yearbook layout; `2023f` loads the 2023 and 2025 yearbooks, so four vintages are combined) → `icsg_comb.py` (combine vintages) → `process_icsg.py` (stock levels → NAS per vintage + merge) → `icsg_clean.py --baci-version N --to-feed` (reads the registered gross BACI parse vN for imported anode, #63) | `data/input/feed/icsg/current/values.csv` + `parents.json` |
| **BACI** | `baci_parse/baci_parse_feed.py` | `data/input/feed/baci/v0/values.csv` + `quantity_recon.csv`, `quantity_cv_scale.csv` |
| **Crude ore** | `crude_ore_parse_feed.py` (the Kroiss workbook; the only source of the crude-ore feed) | `data/input/feed/crude_ore/v0/values.csv` |
| **Ore grade** | `build_ore_grade.py` (not a parse: it reads the finished ICSG and crude-ore feed runs; see the loop above) | `data/input/params/ore_grade/current/params.csv` + `parents.json` |

ICSG values are kt Cu: the parse stamps `Unit = 'kt'` and the ICSG feed's `valuation_map` reads it as
contained copper (CC), while BACI's and crude ore's `'kt'` is gross mass.

Three behaviours a reader should not have to discover from the code:

- **A row's [vintage](../GLOSSARY.md) decides precedence, but only across releases.** ICSG publishes the
  same series in successive yearbooks; a later vintage restates an earlier one, so rows sharing a key
  are resolved newest-wins. Two rows of *one* vintage sharing a key are two different streams folded
  together by a rename and are **summed**, not deduplicated
  ([ADR 0036](adr/0036-a-renamed-type-is-a-second-stream-not-a-restatement.md)).
- **A stock change is computed from levels, never read.** The yearbook parsers return every
  [stock level](../GLOSSARY.md) ICSG prints; `process_icsg.py` differences consecutive levels of one
  series *within one vintage* into the [net addition to stock](../GLOSSARY.md) — never across
  vintages, and a blank year yields no change for itself or the year after. `USC` sums the
  consumer, merchant and government series and `MESC` every exchange at one location. A region
  that prints only a total — or a total its types contradict by more than `TOTAL_TOLERANCE_KT` —
  counts as producer stock (`PSC`), as ICSG's own summary tables count it, and a total books no
  change in the year it takes over from its types.
- **The BACI parse overrides its source's published quantity on flagged legs.** HS 740311 is
  ~99.99 % Cu, so its unit value *is* the copper price in the same release; a concentrate leg whose
  value-implied Cu fraction leaves the physical band is reconstructed from its intact value
  (`q = v / (P_cu · g_ref)`), and the inference is carried in σ through a sidecar. The
  reconstruction is two-sided: above the ceiling it only raises the mass, and far below the floor
  (leg < 0.02 in an exporter-year < 0.20, e.g. PNG's kilograms booked as tonnes) it only lowers it.
  Every flagged leg ships in `quantity_recon.csv` with its `side` so the override is auditable leg by
  leg ([ADR 0033](adr/0033-baci-quantity-contradicts-its-own-value.md)).

#### Math

##### Pedigree → coefficient of variation

Following Laner et al. (2016). Each data source carries five quality indicator scores $s \in \{1,2,3,4\}$.

**Reliability** (non-zero even at the best score):

$$\mathrm{CV}_\mathrm{rel} = \frac{a \cdot e^{\,b\, s_\mathrm{rel}}}{100}, \qquad a = 0.75,\quad b = 1.105$$

**All other indicators** (zero at score 1):

$$\mathrm{CV}_i = \frac{a_i \cdot e^{\,b\,(s_i - 1)}}{100}, \qquad b = 1.105$$

The coefficients $a$ and $a_i$ are in percent (Laner et al.); every CV in the code and the sidecars is a fraction, hence the division by 100. Sensitivity parameter $a_i$: not sensitive $= 0.375$, medium sensitive $= 0.75$, highly sensitive $= 1.5$; a `<Indicator>_sensitivity` row of the metadata sidecar sets it per indicator, and a feed without one is medium sensitive.

**Total CV** (quadrature sum — assumes independence across indicators):

$$\mathrm{CV}_\mathrm{tot} = \sqrt{\mathrm{CV}_\mathrm{rel}^2 + \mathrm{CV}_\mathrm{comp}^2 + \mathrm{CV}_\mathrm{temp}^2 + \mathrm{CV}_\mathrm{geo}^2 + \mathrm{CV}_\mathrm{other}^2}$$

##### Power-mean (Lp) uncertainty propagation

All Stage 1 / Stage 2 σ propagation steps share a single global power-mean (Lp) combiner with exponent $p \in [1.1, 1.5]$, default $p = 1.3$ (a Stage 1 propagation with $p < 1$ raises). For a list of contributions $\{x_i\}$:

$$\mathrm{Lp}(\{x_i\};\, p) \;=\; \left(\sum_i |x_i|^p\right)^{1/p}.$$

$p = 2$ recovers the legacy Gaussian-quadrature formulas exactly. Smaller $p$ weights large terms relatively less, which suppresses σ inflation in fan-in regimes where the per-source independence assumption is structurally violated. The exponent is a constructor kwarg on every consumer (`AggregationMath`, every `BaseImputer`, `ConstraintEngine`); init scripts pass the same value to all of them.

##### Concordance error propagation

The weight matrix $\mathbf{W} \in \mathbb{R}^{N_\mathrm{src} \times N_\mathrm{root}}$ maps source edges to root edges as a `scipy.sparse.csr_matrix`. The point estimate is

$$\mathbf{f}_\mathrm{root} = \mathbf{W}^\top \mathbf{f}_\mathrm{src}.$$

Uncertainty is propagated in **CV-invariant** form. For each root edge $j$, the normalized fan-in shares from per-source absolute contributions are

$$A_{ij} = \bigl|W_{ij}\, f_{\mathrm{src},i}\bigr|, \qquad \omega_{ij} = \frac{A_{ij}}{\sum_k A_{kj}} \quad\Bigl(\sum_i \omega_{ij} = 1\Bigr),$$

and

$$\mathrm{CV}_\mathrm{root}[j] \;=\; \mathrm{CV}_\mathrm{src} \cdot \left(\sum_i \omega_{ij}^p\right)^{1/p}, \qquad \boldsymbol{\sigma}_\mathrm{root}[j] \;=\; \mathrm{CV}_\mathrm{root}[j] \cdot |\mathbf{f}_\mathrm{root}[j]|.$$

For $p \geq 1$ the power-mean of normalized weights is at most 1, so the **diversification invariant** $\mathrm{CV}_\mathrm{root}[j] \leq \mathrm{CV}_\mathrm{src}$ holds for every root edge.

##### ROW pedigree penalty (optional, per feed)

When a feed sets `Row_penalty: TRUE` in its metadata sidecar, every **ROW-touching** root edge (`region_from` or `region_to` $=$ the caller's `row_region`, copper `"ROW"`) is propagated as if its `geographical_correlation` score were the worst value (4). This is a per-edge multiplier $\mathrm{cv\_root\_scale}[j] = \mathrm{CV}_\mathrm{tot}(\text{geo}{=}4)/\mathrm{CV}_\mathrm{tot}(\text{geo}{=}\text{baseline})$ applied to $\mathrm{CV}_\mathrm{root}[j]$, which is exactly equivalent to re-propagating those rows at `geographical_correlation = 4` (deterministic region mapping means a ROW root edge only receives fan-in from ROW source rows). Penalised edges are recorded in a `row_penalized` provenance mask (persisted in `result.npz`) and summarised in the run's `metadata.json`. Stage 2 applies the same geographic penalty — quadrature-added — to *imputed* ROW edges in the FlatPrior fallback. See [ADR 0003](adr/0003-row-geographic-pedigree-penalty.md).

#### Classes

- **`AccountingStructure`** ([acc.py](../src/pysuto/acc.py)) — root node/edge index, supply–use topology, per-edge bounds, structural edge masks (`acc.masks["supply" | "use" | "tradable" | "sb" | "cross_region"]`; `cross_region` is the edge scope of bilateral feeds, ADR 0014) consumed by `MassBalanceBuilder` / `FlatPriorImputer`. It binds the case's labels to the framework's **roles** ([ADR 0034](adr/0034-roles-are-the-frameworks-labels-are-the-cases.md)): every `SB` sector to a boundary role (`nature` / `secondary` / `final_demand`) and the two valuations to `content` / `gross`, declared by a `role` key in the acc spec (a `role` column on `root.ods`'s Sector_Entity and Valuation sheets). The library reads `acc.boundary_roles` (SB sector → `BoundaryRole`) and `acc.valuation_roles` (`.content` / `.gross`, the bound labels), never a label; `acc_store.save`/`acc_store.load` keep the roles and the rule tables, so a loaded structure carries the masks `__init__` built, `tradable` included.
- **`ConcordanceTable` / `WeightMatrix`** ([conc.py](../src/pysuto/conc.py)) — src→root edge mapping and the sparse CSR weight matrix.
- **`DataFeed`** ([feed.py](../src/pysuto/feed.py)) — base class owning a `weight_matrix`, pedigree scores and `F_src`; `WeightMatrix` is the persisted subclass (`save` / `load`).
- **`PedigreeScores` / `UncertaintyMixin` / `PropagationResult`** ([unc.py](../src/pysuto/unc.py)) — pedigree→CV computation and Lp / power-mean error propagation through `W`.
- **`DataFeedMetadata`** ([meta.py](../src/pysuto/meta.py)) — parses the Excel sidecar.
- **`DataFeedRegistry`** ([registry/feed.py](../src/pysuto/registry/feed.py)) — JSON index + per-feed metadata, pickle, and result.npz on disk.
- **`FeedSpec` / `run_feed` / `step_A..D`** ([feedpipe.py](../src/pysuto/feedpipe.py)) — `run_feed(spec, acc_id=…, data_dir=…)` builds one feed run from a `FeedSpec` (loaders, concordance policy, hooks; copper's three are in [copper_feed.py](../scripts/copper_feed.py)): read the registered inputs and the metadata sidecar → build concordance → assemble W → build `F_src` → apply + propagate → register + diagnostics. The `step_A..D` functions are its pieces; `FeedBuild` is what a spec's hooks see.

#### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `p` (power-mean) | `1.3` | Tail-weight in σ propagation; `p=2` ≡ Gaussian quadrature |
| pedigree scores | 1–4 per indicator | Map to CV via Laner et al. exponential |
| `allow_unmatched_edges` | `True` | Whether `feedpipe` tolerates source edges not in the concordance table |

#### Use case

Run Stage 1 on the toy case in [tutorial/01_datafeed.ipynb](../src/pysuto/tutorial/01_datafeed.ipynb).

#### Glossary detail, Stage 1

The glossary entries below keep their definition and their aliases to avoid. This section holds the full text they had before; each block names its entry.

**Two mapping levels**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Two mapping levels** — the feed stage composes *two* independent raw→root mappings, each of
> which can be 1:1, fan-in, or fan-out. The fan-in/fan-out terms below name the **second** level only.
>
> - **Mapping 1 — raw rows → source edges.** Defined by the concordance **variable map**
>   (`variable_map` sheet), which is *auxiliary-conditional* (e.g. on `Supply_Use`): it projects raw
>   variables onto the source 8-tuple slots (`src_sector_from/to`, `src_entity_*`, `src_region_*`,
>   `src_time`, `src_valuation`), so the *same* raw variable lands in a *different* endpoint depending
>   on the auxiliary category (`Process` → producer/`from` for Supply rows, → consumer/`to` for Use
>   rows). This is what **imposes directed supply-use structure** onto raw data that may be
>   **production-based** (ICSG) rather than already supply-use shaped (BACI). Cardinality: **1:1** or
>   **many:1** (*raw-row aggregation* — several raw rows resolving to the same source edge have their
>   values **summed**). A single raw row resolving to *many* full-value source edges is **replication**
>   (the anti-pattern below), not a valid shape.
> - **Mapping 2 — source edges → root edges.** The WeightMatrix. **Fan-out** and **fan-in** (below)
>   live here. Mass-preserving iff each source edge's outgoing weights sum to 1.

**Unweighted fan**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Unweighted fan** — a fan-out whose legs carry no weighting information (no trade shares, no
> user weights — every leg still at the uniform default). Such a fan is **not distributed**: an
> even 1/N split asserts a fabricated partner pattern at the observation's confidence, whereas the
> honest state of knowledge is "total known, split unknown". The raw value stays undistributed
> (excluded from the feed's projected mass). A fan that degenerates to a single leg is an exact
> match, not an unweighted fan. _Avoid_: reading "no imports" as "no weighting information" — a
> fan weighted **all-domestic** because BACI observed the region and recorded no imports carries
> weights and is not an unweighted fan ([ADR 0018](adr/0018-import-free-fans-distribute-domestically.md));
> the doctrine bites only where the split is genuinely uninformed. See
> [ADR 0008](adr/0008-unweighted-fans-are-not-distributed.md).

**Qualifier fan**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Qualifier fan** — a Use fan whose source variable names its **input product** (e.g. smelter
> production *from scrap*, refinery production *from matte*). Only root legs whose supplier sector
> matches the named product belong to the fan; other incoming edges of the consumer node are
> out-of-scope legs that the open (`"<any>"`) endpoint would otherwise wrongly absorb — and onto
> which route-keyed weights of *other* products can bleed. A qualifier fan with **zero** matching
> legs marks a structurally impossible mapping (the named product does not feed that consumer in
> the accounting structure) and is a hard error, never a silent fan. See
> [ADR 0008](adr/0008-unweighted-fans-are-not-distributed.md).

**Mapped self-trade**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Mapped self-trade** — a bilateral src row whose two endpoints fold onto the **same root region**
> under the region map (Hong Kong→mainland once both map to "China"; any small-country pair inside
> ROW). Its root address is indistinguishable from the **domestic** edge, yet its value is a tiny,
> customs-precise subsample of a production-scale transfer — an inconsistent observation, not a
> measurement of that edge. Bilateral feeds are therefore **scoped to cross-region edges** (their
> edge scope drops rows resolving to domestic addresses), see
> [ADR 0014](adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md). _Avoid_: reading a
> mapped self-trade row as a domestic observation — the china-matte case put 0.028 kt of gross mass
> next to 5,678 kt of copper content on one edge.

**Mirror duplicate**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Mirror duplicate** — a BACI flow under one code that repeats copper another code already declares,
> so mapping both to one region counts it twice. The test is the balance, not the customs record: in
> the years the true exporter's own declarations already reach about its production, so that the
> second code would push them past it, the two codes' flows are duplicates and only one declaration
> survives; in the years they fall clearly short, the second code fills the gap and is kept
> ([ADR 0014](adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md)). _Avoid_:
> "mis-coded flow" for every year of such a code — it is a duplicate only where the balance says so.

**Reporting territory**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Reporting territory** — the real-world territory a feed's rows describe, which is *not* the same
> as the **source region label** those rows carry. A feed may rename a territory mid-series (ICSG's
> `Taiwan` → `Taiwan, China` from 2013) or publish historical joint reporting under a joint label
> (`Belgium-Luxembourg` to 2013), so one territory wears two labels across the series and, in the
> handover year, two labels on the same cell. Two source labels denote one territory exactly when the
> region map assigns them to the same **real-country root**; `ROW` is excluded, being a bucket of 48
> territories rather than one. The region map cannot express a label that means another territory in
> one **vintage** only: the 2005 yearbook's `Congo` (parsed as `Congo Rep.`, Brazzaville in every later
> vintage) is DR Congo, so the ICSG clean step relabels it per vintage before the map applies
> ([ADR 0032](adr/0032-one-territory-one-label.md), #85). Anything computed *before* the region map is applied — a parse rule, a
> param study reading `values.csv` — must therefore group by territory, not by label. _Avoid_: reading
> a handover-year pair as two measurements. It is one measurement republished, so adding the two
> doubles the cell, and a rule keyed on the label silently skips the territory whose two series are
> labelled differently ([ADR 0032](adr/0032-one-territory-one-label.md)).

**Vintage**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Vintage** — the release a feed row came from, when a source publishes the same series
> repeatedly (ICSG's 1995, 2005, 2014, 2023 and 2025 yearbooks; BACI's annual releases). A later
> vintage **restates** an earlier one, so rows sharing a key are resolved by keeping the newest.
> That rule is only sound *between* vintages: two rows of **one** vintage sharing a key are two
> different **streams** of the same quantity — a source category folded onto another by a rename,
> such as ICSG's `Low-grade` smelter feed folded into `Primary` — and they **sum**. _Avoid_:
> resolving a same-vintage collision as a restatement. One stream then silently replaces the other
> rather than adding to it, and which one survives depends on row order
> ([ADR 0036](adr/0036-a-renamed-type-is-a-second-stream-not-a-restatement.md)). _Avoid_ also:
> reading a vintage as a *time* coordinate — it labels the publication, not the year observed; one
> vintage carries many observation years. _Avoid_ also: "edition" for a yearbook's release — say
> vintage. The ICSG flow tables persist a vintage as the column `Year_loaded_from_file` (the
> yearbook's publication year, 2005, 2014, 2023 or 2025); it is the vintage, not a load date.

**Sector → entity is 1:1**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Sector → entity is 1:1** — every *concrete* root sector maps to exactly one entity kind
> (`Flow` | `Process` | `SB`), so entity is functionally determined once the sector is known.
> **Caveat:** an **open (`"<any>"`) sector does not determine an entity** — an unclassified endpoint
> can match producers across several entities (e.g. a real `Flow` source *and* the `SB`
> system-boundary/import source). Concordance code must therefore make the entity `"<any>"` whenever
> its sector is `"<any>"`, not pin it to a concrete edge entity. See
> [ADR 0002](adr/0002-single-endpoint-rows-fan-out-not-replicate.md).

**System-boundary category (`entity_sub` on SB nodes)**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **System-boundary category (`entity_sub` on SB nodes)** — the sub-classification carried by every
> `SB` node, naming *what kind* of boundary crossing it is. In the copper model: `Nature` (primary
> extraction entering the system — crude ore, `CO`), `FD` (final demand leaving the system — the
> final-demand categories `FU` / `MESC` / `USC`), `SEM` (secondary / residual material — `Scrap`
> recycling inputs, the `Slag` / `Tailings` / `WR` residual outputs, and `PSC`, the producer
> stock-change leg). `entity_sub` is
> meaningful only on `SB` nodes (`Flow` / `Process` nodes leave it empty). It is the case's
> **label**; the library reads the boundary **role** bound to it (see Role). _Note:_ `FD` is
> structurally an SB-*output* (`entity_to == "SB"`) yet semantically belongs to the **use** side — a
> commodity consumed by final demand, with a real `Flow` commodity on its origin endpoint; only
> `Nature` and `SEM` are pure raw/secondary boundary crossings with no `Flow` endpoint.
> _Avoid_: conflating `entity_sub` with **entity** (only `Process` | `Flow` | `SB`); and avoid
> "boundary type" as a loose synonym for it — that phrase is ambiguous between this `entity_sub` class
> (Nature/SEM/FD) and the **sector** of the SB node (`CO`, `Slag`, `FU`, …, the finer per-commodity
> grain), so name the grain you mean.

**Observed zero**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Observed zero** — an edge a feed reports at exactly 0 with a σ > 0 (the feed result's
> `observed_zero` mask): covered, so no imputer fills it, and any flow the solver puts on it is
> priced. The ICSG feed gives one to a **non-producer**'s domestic concentrate leg at
> trade-share weight 0, priced like the **net-stock seed** on the usage its fans distribute, and to
> its own SX-EW output where ICSG reports none and no other input reports SX-EW there, priced at
> ICSG's reporting resolution
> ([ADR 0024](adr/0024-fan-domestic-leg-bounded-by-own-production.md) amendments). A zero weight
> in a producer, where imports exceed usage, is a residual, not an observation, and stays imputable. Distinct from
> an uncovered edge (no feed observed it; NaN in Stage 2) and from the net-stock seed (a prior no
> feed reports). _Avoid_: "fed zero"; "zero pin" (the `l = u = 0` bound on no-data edges,
> [ADR 0013](adr/0013-no-data-flat-prior-anchor.md)).

**Concentration coefficient (`conc`).**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Concentration coefficient (`conc`).** The Cu mass fraction of a sector's commodity (e.g. Cathode
> 0.9975, Anode 0.94, Matte 0.525, Concentrate 0.275; from the `conc` param). The `conc_<sector>_supply`
> relation enforces `CC = conc · Mass` on each node pair, so the two valuations of one edge are not
> independent. The param's `value_nominal` drives the **soft** `CC = conc·Mass` ratio row; its
> `[value_min, value_max]` drives a **hard grade band** `value_min ≤ CC/Mass ≤ value_max`
> (`ratio`/`nonneg`). **Invariant: each commodity carries exactly one such band** — `Scrap` is one
> mixed-grade commodity with a single wide band `[0.40, 0.98]`, nominal 0.70, *not* two grade classes
> (two simultaneous hard bands on the one `Scrap/SB` node intersected to ∅ and collapsed all scrap;
> [ADR 0012](adr/0012-scrap-one-mixed-grade-commodity.md)). _Avoid_: reading the `conc_*_supply`
> z-violation as a "bug" — it fires whenever the two feeds populating the CC and Mass sides of an edge
> disagree.

**Smelter scrap · direct-to-refinery scrap.**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Smelter scrap · direct-to-refinery scrap.** The model's two `Scrap` inflows, named by where the
> scrap enters: **smelter scrap** feeds `Smelting` with the primary concentrate and leaves as matte;
> **direct-to-refinery scrap** feeds `CFR` with the matte and leaves as anode. ICSG reports two
> *output* classifications — **secondary smelter production** (blister/anode made from scrap) and
> **secondary refined production** (refined copper made from scrap by *either* route) — and the
> second contains the first. Direct-to-refinery scrap is therefore secondary refined *minus* the
> smelter route; in fifteen countries the two published figures are identical and the direct route
> is nil. It is also minus any imported anode the region refines beyond its primary output, which
> enters as anode trade (Austria, #63). **Direct melt** — scrap fabricators use without refining — is a third quantity, outside
> this model's boundary ([ADR 0030](adr/0030-secondary-smelter-copper-enters-the-chain-once.md)).
> Since [ADR 0031](adr/0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md) both
> inflows are stated to the model **twice**: as a **level** (a scrap level anchor, where ICSG reports
> one) and as a **share** (the `scrap_share` band, every region). Before it only the share existed.
> _Avoid_: booking secondary refined production as a second inflow beside smelter scrap (the copper
> enters twice, and the anode anchor discharges the surplus by deleting concentrate); reading
> "secondary" as the model's SEM input class when ICSG means an output classification.

**Recovered vs in-situ grade.**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Recovered vs in-situ grade.** Two grade notions per region-year: **recovered grade** = metal
> produced ÷ allocated ore (`1/factor`, the raw S&P number); **in-situ grade** = copper contained in
> the mill feed ÷ allocated ore = recovered ÷ recovery-`eff`. The model's `CO`/`Ore` **CC edges are
> in-situ** (production ÷ eff, matching the recipe's `(1−eff)` Tailings copper), so their grade
> rows use the **in-situ** value — region×year-resolved, owned by the `ore_grade` params, not the
> global `conc` family
> ([ADR 0015](adr/0015-in-situ-ore-grades-on-allocated-ore-basis.md)). The same artifact also
> owns the region×year×route **tailings** conc (`(1−eff)` recipe copper over the process
> Mass-balance residual, per MF/SXEW route — ADR 0015 amendment); the global `conctailingsuse`
> row is retired. On the allocated basis
> these are *not* physical mill-head grades (Australia ~2–3 % vs ~0.8 % head grade). _Avoid_:
> applying a recovered-basis grade to an in-situ edge (a systematic 1/eff error), or citing these
> grades against literature head grades without converting.

**Concentrate grade.**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Concentrate grade.** The Cu mass fraction of copper concentrate — the `Concentrate` case of the
> concentration coefficient, one **global** row (0.275, band [0.20, 0.35]). Unlike an **in-situ ore
> grade** it is a *direct assay*: no allocation basis, no `eff` division, so it may be read against
> industry figures (20–35 % Cu) as it stands. It is deliberately **not** region-resolved the way ore
> grades are: measured over the region-years whose books are physically possible, the between-region
> spread is 0.0117 against the 0.0375 σ the single row already declares, and the residual that looks
> like a regional grade is a gap in the recorded export **mass**
> ([ADR 0029](adr/0029-the-concentrate-grade-stays-global.md)). _Avoid_: reading a region's
> implied `(P−U)/X` as its grade — where export mass is under-recorded that ratio absorbs the missing
> tonnage and returns values (0.6–0.9, and negatives elsewhere) that are not grades at all.

**Grade band vs grade σ.**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Grade band vs grade σ.** A `conc`-family params row's `value_min`/`value_max` are read **twice**:
> as a **hard cone** by `RatioBoundBuilder` (`value_min·Mass ≤ CC ≤ value_max·Mass`, per edge per
> year — a physical plausibility limit, no slack) and, when `unc_type = "min_max"`, as the **σ** of the
> *soft* ratio row via `ParamRow.sigma` — which also sets the parametric fill's prior and the
> `NetworkImputer` twin-map σ. Those are two different statements, and conflating them means changing
> a limit silently re-prices three things. The `ore_grade` grade rows therefore carry
> `unc_type = "sd"` with an explicit `value_sd` (ADR 0028); tailings rows still use `min_max`. The
> other `unc_type` mode is `pedigree` (σ = nominal · Laner total CV). _Avoid_: calling the band "the
> σ" — a band is the wall, σ is the price of approaching it; and narrowing a `min_max` band when you
> only meant to move the wall.

**Grade carried past the ore feed.**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Grade carried past the ore feed.** The `crude_ore` feed ends before ICSG production does
> (2022 vs 2024). For a region-year with production but no ore mass the `ore_grade` generator
> carries the region's **last observed in-situ grade** forward (`src=carry`, band ±20 % instead
> of ±10 %); the NetworkImputer then derives the whole `CO→Mining→Ore→…` Mass chain from that
> year's production through the grade, and the scale totals and grade rows follow on their own.
> **The assumption is on the grade, never on the mass**: ore mass tracks production
> ([ADR 0023](adr/0023-ore-grade-carried-past-the-crude-ore-feed.md)). _Avoid_: carrying
> the ore Mass itself (holds tonnage flat whatever was produced, and disagrees with the grade), or
> running any Mass extrapolator *after* the network cascade (fills the boundary leg only, leaves
> the chain behind it at flat-prior zero and drops its anchor rows).

**⚠ The grades in use are internal, not external.**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **⚠ The grades in use are internal, not external.** Every `ore_grade` row the model consumes is
> *derived* from ICSG production and crude_ore allocated mass (ADR 0015). The **external** S&P
> factor series (`data/input/params/ore_grade/current/all_integrated_est_fac_2024-07-04.csv`, the
> basis of `ore_grade` **v1** from 2026-05) enters only as a fallback where ICSG production is
> absent (129 rows) — even though it carries real 2023 values for every region the carry-forward
> touches. Measured 2026-09-10: its *level* differs from the internal grade by up to ±8 % at the
> 2022/2023 seam (Mexico +7.6 %, Russia +6.0 %, Chile −2.7 %), its year-on-year *trend* by
> < 0.5 %. Using that trend for carried years (`g(t) = g(t−1)·factor(t−1)/factor(t)`) is an open,
> sub-percent option recorded in ADR 0023; using its level would reintroduce the basis jump
> ADR 0015 removed. _Avoid_: describing the ore-grade series as external data, or comparing it to
> the S&P factor without converting bases (`1/factor` is recovered-basis, the model is in-situ).

**Feed semantics — ICSG vs BACI.**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Feed semantics — ICSG vs BACI.** **ICSG** publishes production/usage statistics in **copper content
> (`CC`)**; its `Use` rows are **single-endpoint** (consumer known, producer open) ⇒ producer **fan-out**.
> **BACI** publishes **bilateral gross trade** in **`Mass`** (fully classified X→Y; no self-trade rows *at
> source* — but region folding creates **mapped self-trade** at root, so the feed is scoped to
> cross-region edges and never informs a domestic leg,
> [ADR 0014](adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md)). On a trade edge the ICSG-CC
> value is therefore a *fan artifact* and the BACI-Mass value is the real observation — for a leg the
> **re-export** netting did not change; a re-attributed leg is part observation, part inference, and carries
> the compounded σ ADR 0027 defines. _Avoid_: treating a
> per-edge ICSG-CC trade value as a measurement — it is the national usage split across producers by the
> fan weights.

**Trade-share fan weighting**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Trade-share fan weighting** ([ADR 0002](adr/0002-single-endpoint-rows-fan-out-not-replicate.md)
> option D) — weighting ICSG's producer fan by BACI bilateral import shares instead of uniformly, so ICSG-CC
> inherits BACI's geography. Since [ADR 0027](adr/0027-re-exports-are-re-attributed-at-feed-time.md)
> those shares are read from the **netted** BACI version, and a group whose imports the netting exhausted is
> **withheld** (left undistributed) rather than fanned all-domestic: its `Σ imports = 0` is constructed, not
> observed, so ADR 0018's condition does not hold. The domestic leg (X==Y) carries the residual `usage − Σ imports` (in CC units);
> the split is computed in copper content via `conc`. Applies only to the four **traded** ICSG-fanned
> sectors — Cathode, Concentrate, Anode, Matte. The residual also governs the **import-free** case: where
> BACI *observes* the consumer region (records any cross-region import into it; BACI 490 "Other Asia, nes"
> counts as Taiwan, [ADR 0014](adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md)),
> `Σ imports = 0` is a real observation of domestic sourcing and the residual `usage − 0 = usage` gives an
> all-domestic fan ([ADR 0018](adr/0018-import-free-fans-distribute-domestically.md)); where BACI is
> blind, zero imports carries no information and the fan stays undistributed. **The domestic leg is
> bounded by the region's own production** of the product for the sectors in
> `BOUND_DOMESTIC_SECTORS` (Concentrate; [ADR 0024](adr/0024-fan-domestic-leg-bounded-by-own-production.md)):
> where the residual exceeds what the region produced, the excess is spread over the import legs in
> BACI proportion (usage conserved) — a region cannot supply itself more than it makes, and the hard
> Flow-node balance enforces exactly that on the posterior. Bounded only where the recorded imports
> can carry the remainder (`Σ imports ≥ residual − production`, legs at most doubled); import-free
> fans keep the ADR 0018 form and near-self-sufficient smelters (US 1997–2003) keep the residual,
> because their gap is ICSG-internal and spreading it fabricates imports instead.
> _Avoid_: including **Scrap** — it is an ICSG `Use` type but is **not traded**; and reading a large
> domestic leg as evidence of domestic mining — before ADR 0024 it was the BACI-vs-ICSG shortfall.

**Re-export (re-attributed leg)**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Re-export (re-attributed leg)** — tonnage a region imported and shipped on unchanged. BACI records
> it as **two** gross border crossings (`A→h`, `h→Z`), which the accounting structure would read as two
> statements about *supply origin* — the second one false, since `h` produced nothing. ADR 0027 moves it
> at feed time: `T = min(X, I, max(R, E))` per (product, region, year) in `CC`, where `R = max(0, I−U)` is
> the use-side reading (imports above ICSG intake) and `E = max(0, X−P)` the supply-side one (exports above
> ICSG production); the volume is split by the hub's own observed origin and destination shares onto direct
> `A→Z` legs. The two caps `T ≤ X` and `T ≤ I` are the safety property: a **producer** with thin ICSG
> coverage (Kazakhstan cathode — no imports) has `T = 0` however large `E` is and is never netted. See
> [ADR 0027](adr/0027-re-exports-are-re-attributed-at-feed-time.md). _Avoid_: calling every hub leg a
> re-export — only the part the caps admit is one, and the rest stays in the **transit residual ledger**.

**Merchanting**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Merchanting** — a title trade in a third country while the cargo sails elsewhere (BPM6 §10.41). Two
> shapes, and only one is netted: **both legs recorded** (Switzerland concentrate: `I ≈ X`, no process) is
> an ordinary re-export and re-attributes in full; **inbound only** (Zambia→Switzerland/Singapore/
> Luxembourg anode: `X ≈ 0`) cannot be, because nothing leaves the hub to carry it — those legs stay gross
> and the QP keeps refusing them at the exporter's capacity ([`zambia-anode-phantom-trade`](rep/analysis_20260910_zambia-anode-phantom-trade.md)).
> _Avoid_: reading a small `T` at a merchanting hub as a failure of the rule — it is the rule declining to
> invent a destination.

**Transit residual ledger**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Transit residual ledger** — what the netting deliberately leaves behind, written per (product, region,
> year) beside the feed (`reexport_ledger.csv`) with a label: `pass-through`, `partial`, `ceiling-relabel`,
> `inbound-only[/no-process]`, `producer`, `icsg-gap`, `excluded-aggregate` (ROW is never a hub — an
> aggregate of 186 economies has no warehouse), `untouched`. Before ADR 0027 this tonnage was the fan's
> silent **clip** (`max(0, Σ imports − usage)`); it is the same material, now named. _Distinct from_:
> the **Bounds-clip** above, which is an imputer's saturated fill. _Avoid_: reading the residual as an
> error — the Rotterdam effect (imports a hub never re-exports on the record) genuinely cannot be
> re-attributed from BACI alone.

**Value-implied Cu fraction**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Value-implied Cu fraction** — `unit_value(product) / unit_value(cathode)` within one trade
> feed, one year. Refined cathode is ~99.99 % Cu, so its unit value *is* the copper price on that
> feed's own terms, and the ratio is what the product's recorded value implies about its recorded
> mass. It is a **consistency check between a feed's two fields**, not a property of the material:
> where it leaves the physical band the *quantity* is the field in doubt, because the value can be
> checked independently against production statistics. _Avoid_: reading it as a grade — the same
> _Avoid_ the **Concentrate grade** entry already carries for `(P−U)/X`, and for the same reason. A
> fraction of 0.6–0.9 is not a 60–90 % concentrate; it is a statement that the recorded mass is two to
> three times too small ([ADR 0033](adr/0033-baci-quantity-contradicts-its-own-value.md)).
> _Avoid_ also: flagging a single leg inside a year whose *aggregate* fraction is in band — the
> aggregate is the level at which production statistics can license the claim, and lifting a leg
> below it fabricates mass. A leg an order of magnitude **below** the floor, in an exporter-year
> itself below the floor, is the opposite statement: the recorded mass is far more than the value
> buys, because the quantity is in the wrong unit or the leg is not concentrate (ADR 0033, Round Z
> amendment). _Avoid_: reading 0.15–0.20 as a defect. A value is net of payables and treatment
> charges, so real concentrate sits there. _Avoid_ also: reading a fraction ≪ 0.20 as a low grade.

**Exchange location**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Exchange location** — a region in which ICSG reports exchange stocks: the LME warehouse
> countries by location, COMEX as the United States, SHFE as China (16 regions in the feed). It is
> the destination of a `Cathode(i)→MESC(j)` leg, not a property of the origin. _Avoid_: "hub"
> (reserved for the merchant / re-export hubs of [ADR 0027](adr/0027-re-exports-are-re-attributed-at-feed-time.md)),
> "warehouse region".

**Re-import**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Re-import** — the `A→h→A` branch of a re-attribution: goods that left `A`, reached the hub and came
> back. It is dropped rather than booked, for the same reason **mapped self-trade** is: its root address is
> the domestic edge, which it is not a measurement of. Where `A` is `ROW` the same branch is third-country
> trade folded onto one aggregate, and is dropped identically. 467 kt CC over 1995–2024.

**ICSG aggregate row**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **ICSG aggregate row** — an ICSG source row that reports several countries together, such as
> "Scandinavia" for Sweden, Finland and Norway until 2003. Its members and their years are declared in
> the `icsg_aggregates` params input, never inferred from names; the ICSG clean step splits it to its
> member ([ADR 0032](adr/0032-one-territory-one-label.md)).
> _Distinct from_: `excluded-aggregate`, the ledger label for ROW, a *root* region; and from `agg=true`
> pooling and Stage-2 aggregation, which are unrelated. _Avoid_: "aggregate" alone.

**Handover year**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Handover year** — the first year ICSG publishes both an aggregate row and the members it contains,
> the year after the declared member years (2004 for Scandinavia). It states each member's share of the
> aggregate's series, and the aggregate row in it is a republication of its members.
> _Avoid_: summing the two forms in that year.

**`row` mask**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **`row` mask** — the named edge mask selecting edges that touch ROW on **either** endpoint
> (`region_from == "ROW"` **or** `region_to == "ROW"`). A registered input mask
> (`data/input/mask/row/`).

**Lp / power-mean combine**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Lp / power-mean combine**:
> The single σ combiner used by every stage — `lp_combine({xᵢ}, p) = (Σ|xᵢ|ᵖ)^(1/p)`. `p = 2`
> recovers Gaussian quadrature; the project default `p = 1.3` damps fan-in σ inflation.


---

### Stage 2 — Aggregation & Imputation

**What this stage does.** Selects a set of registered feeds, aligns each feed's `F_root` and `σ_root` onto the common edge index, builds the inverse-variance weight matrix `Ω`, aggregates across feeds, and runs an ordered imputation pipeline to fill any edges still uncovered. Every fill is recorded in an `ImputationLog`.

**Inputs:** feed IDs (or `(feed_id, version)` tuples), registry.
**Outputs:** `AggregationResult` with `f_agg [N_r]`, `σ_agg [N_r]`, `Ω [N_f × N_r]`, `coverage [N_r]`, `ImputationLog`.

> Uses [power-mean (Lp) uncertainty propagation](#power-mean-lp-uncertainty-propagation) and [pedigree→CV](#pedigree--coefficient-of-variation) from Stage 1.

#### Math

##### Aggregation across $N_f$ feeds

After Stage 1, each feed $k$ has produced a root-edge observation vector $\mathbf{f}_k \in \mathbb{R}^{N_r}$ and a standard-deviation vector $\boldsymbol{\sigma}_k \in \mathbb{R}^{N_r}$. Entries are NaN for root edges not covered by feed $k$.

Stack:

$$\mathbf{F} \in \mathbb{R}^{N_f \times N_r}, \qquad F_{kj} = f_{\mathrm{root},k}[j] \quad (\text{NaN if not observed})$$

$$V_{kj} = \sigma_k^2[j], \qquad P_{kj} = 1/V_{kj} \quad (0 \text{ where } V \text{ is NaN or zero})$$

Normalised weight matrix (each column sums to 1):

$$\Omega_{kj} = \frac{P_{kj}}{\sum_{k} P_{kj}} = \frac{1/\sigma_k^2[j]}{\sum_{k} 1/\sigma_k^2[j]}$$

Aggregated flow vector as a single matrix operation:

$$\boxed{\mathbf{f}_\mathrm{agg} = \mathbf{1}^\top (\boldsymbol{\Omega} \odot \mathbf{F}) \;\in\; \mathbb{R}^{N_r}}$$

For each root edge $j$ this reduces to $f_\mathrm{agg}[j] = \sum_k \Omega_{kj} F_{kj}$. Aggregated uncertainty uses the Lp combine over the inverse-variance-weighted per-feed contributions:

$$\sigma_\mathrm{agg}[j] \;=\; \left( \sum_k |\Omega_{kj}\, \sigma_k[j]|^p \right)^{1/p}.$$

At $p = 2$ this reduces to $1/\sqrt{\sum_k 1/\sigma_k^2[j]}$. For $p \in [1, 2]$, $\sigma_\mathrm{agg}[j] \leq \max_k \sigma_k[j]$ always holds; the $\sigma_\mathrm{agg} \leq \min_k \sigma_k$ bound is exact at $p = 2$ only; at $p = 1.3$ it can be exceeded slightly ($\sigma = 1, 100$ gives $\sigma_\mathrm{agg} = 1.0018$). Edges covered by no feed remain NaN and are passed to the imputation step.

##### Imputation — parametric

A parametric `Fill` fills target edge $j$ from a source relation (a pair `Relation`, $m = 1$) as a linear combination of source edges $\{i_1, \dots, i_m\}$ with coefficients $\alpha_l$:

$$f_j = \sum_{l=1}^{m} \alpha_l \cdot f_{i_l}$$

The fill always uses bare $\alpha_l$ in the value. For a pair relation $b = r\,a$ (σ_param in ratio units, $\varepsilon$ the relative structural term), the σ of the filled edge is, since #148:

- **forward** (A→B, $f_b = r f_a$): $\sigma_b = \bigl( |r\,\sigma_a|^p + |f_a\,\sigma_\mathrm{param}|^p + |\varepsilon f_b|^p \bigr)^{1/p}$;
- **backward** (B→A, $f_a = f_b / r$): $\sigma_a = \bigl( |s(1/r)\,\sigma_b|^p + |f_a\,\sigma_\mathrm{param} / r|^p + |\varepsilon f_a|^p \bigr)^{1/p}$, with $s$ the [regularized sensitivity](#regularized-sensitivity) below, which caps σ amplification when $1/r \gg 1$ (the ore-grade-inverse case);
- **trade** seed (conc forward, Mass→CC, $r$ = conc): $\sigma_b = \bigl( |r\,\sigma_a|^p + |f_b\,\sigma_\mathrm{param} / r|^p \bigr)^{1/p}$, with no $\varepsilon$ term.

The parameter term $|f\,\sigma_\mathrm{param}|$ is the first-order image of the ratio's σ in flow units (a ratio-unit σ added to flow-unit terms would be dimensionally wrong). $\sigma_\mathrm{param}$ is the per-row σ from the params file (`meta["sigma_param"]`; the relation's own `sigma` is the relative $\sigma_\mathrm{param}/|r|$ that Stage 3 uses); $\varepsilon$ encodes structural uncertainty in the relation itself.

##### Regularized sensitivity

The backward branch passes the inverse multiplier $\beta = 1/\alpha$ through:

$$s(\beta;\, q,\,\alpha_{\min}) \;=\; \frac{|\beta|}{\bigl(|\beta|^q + \alpha_{\min}^q\bigr)^{1/q}}.$$

Defaults: $q = 3$, $\alpha_{\min} = 0.05$. Behaviour:
- $|\beta| \gg \alpha_{\min}$: $s(\beta) \to 1$, so $s(\beta)\cdot\sigma_b$ cannot grow beyond $\sigma_b$.
- $|\beta| \ll \alpha_{\min}$: $s(\beta) \to |\beta|/\alpha_{\min}$ — a relative floor on the effective sensitivity.

##### Imputation — network recipe

The `NetworkImputer` closes a Process node's content (CC) balance with efficiency $\eta$ (σ_η, relative structural term $\varepsilon$): products $= \eta\,S$, waste $= (1-\eta)\,S$, $S$ the total input. $S$ and $\sigma_S$ come from the first branch that applies:

- **inputs** (every primary known, or the relaxed close of [ADR 0010](adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md)): $S = \sum$ known primary + secondary + import legs, $\sigma_S = \mathrm{Lp}(\sigma_i)$;
- **product** (a product known): $S = f_P/\eta$, $\sigma_S = \bigl( |s(1/\eta)\,\sigma_P|^p + |S\,\sigma_\eta/\eta|^p \bigr)^{1/p}$;
- **waste** (the lone waste leg known): $S = f_W/(1-\eta)$, $\sigma_S = \bigl( |s(\tfrac{1}{1-\eta})\,\sigma_W|^p + |S\,\sigma_\eta/(1-\eta)|^p \bigr)^{1/p}$.

A missing product is then $f_P = \eta S$ with $\sigma_P = \bigl( |\eta\,\sigma_S|^p + |\sigma_\eta S|^p + |\varepsilon f_P|^p \bigr)^{1/p}$ (the $\sigma_\eta$ term only on the inputs branch); the waste leg is the same with $1-\eta$; a lone missing primary is the remainder $f_i = S - \sum_{j \ne i} f_j$ with $\sigma_i = \mathrm{Lp}(\sigma_S, \sigma_j, \varepsilon f_i)$. An eff-less stock-change leg gets $\sigma \ge$ `effless_stock_sigma`. A fill outside the structural bounds is clipped, and the network widens σ to the clip gap; the engine's own post-step clip (every other imputer) writes the clipped value to the log row and array but leaves σ unchanged (the σ policy is open, #238).

##### Imputation — time

Linear interpolation between observations at $(t_a, f_a, \sigma_a)$ and $(t_b, f_b, \sigma_b)$ with gap $g = t_b - t_a$ and fractional position $\lambda \in [0,1]$:

$$f_j = (1-\lambda)\,f_a + \lambda\,f_b, \qquad \sigma_j = \bigl( |(1-\lambda)\sigma_a|^p + |\lambda\sigma_b|^p \bigr)^{1/p} \cdot \gamma_0^{\,g}$$

LOCF/NOCB at gap $g$: $f_j = f_a$, $\sigma_j = \sigma_a \cdot \gamma_0^{\,g}$. The multiplicative `gap_inflation` factor $\gamma_0$ grows σ with extrapolation distance. With an observation on one side only, `linear` carries the nearest one (logged as `time_linear`); `interpolate_only` forbids that carry. `method="mean"` is the unweighted mean of the window, $\sigma_j = \mathrm{Lp}(\sigma_i / n)$, with no `gap_inflation`; `min_obs` (the least observations the window must hold) is read by `mean` only.

##### Imputation — flat prior (final fallback)

For every residual NaN edge: $f_j = 0$, $\sigma_j = \mathrm{CV}(j) \cdot \mathrm{scale}(j)$.

**CV — coverage-aware per-mask interpolation.** The mask is the **first matching** of `supply` / `use` / `tradable` / `sb` (drawn from `AccountingStructure` edge masks); each mask carries a pair $(\mathrm{cv}_\text{lo}, \mathrm{cv}_\text{hi})$ in `flat_prior_cv.csv`, and the CV is interpolated by the **coverage density** of the edge's $(\text{sector\_from}, \text{region\_from})$ neighbourhood:

$$\rho(j) = \frac{|\{i \in (\text{sector\_from}, \text{region\_from})(j) \,:\, \text{coverage}[i]\}|}{|\{i \in (\text{sector\_from}, \text{region\_from})(j)\}|}, \qquad \mathrm{CV}(j) = \mathrm{cv}_\text{lo}[\mathrm{mask}(j)] + (\mathrm{cv}_\text{hi} - \mathrm{cv}_\text{lo})\,(1 - \rho(j))$$

Data-rich neighbourhoods (ρ ≈ 1) get the tight $\mathrm{cv}_\text{lo}$; data-poor ones get the loose $\mathrm{cv}_\text{hi}$. Unmatched edges fall through to `default_cv` with no density modulation. Legacy CSVs with a single `cv` column are still accepted (treated as $\mathrm{cv}_\text{lo}=\mathrm{cv}_\text{hi}$).

**scale — 3-level cascade.** Cascading fallback so an edge whose narrow group has no observed flow still receives a meaningful scale:

1. mean $|f_\mathrm{agg}|$ over $(\text{sector\_from}, \text{entity\_from}, \text{region\_from}, \text{valuation})$ — the original narrow group;
2. if (1) is empty, mean $|f_\mathrm{agg}|$ over $(\text{sector\_from}, \text{valuation})$ — broader fallback;
3. if (2) is also empty, the global median of finite, non-zero $|f_\mathrm{agg}|$.

Only finite, non-zero values count at every level, so upstream-imputed zeros never pull the scale to zero. The previous single-level rule collapsed σ to 0 on ~40 k copper edges, which then pinned them at $\sigma_a = 10^{-6}$ via the constraints-side floor (see [analysis_optim_const_copper_…_20260513_112419.md](rep/analysis_optim_const_copper_20260513_091054_65b9133e_20260513_112419.md)).

##### Stage 2 σ screens

Three functions of [impute/screens.py](../src/pysuto/impute/screens.py) run in the case's hooks around the cascade and mutate `sigma_agg` in place; the case supplies the sectors and thresholds. **Size damping** ([ADR 0017](adr/0017-size-damped-sigma-on-bilateral-trade-legs.md)): on covered bilateral Flow-origin Mass legs, $\sigma_e = \mathrm{cv}_e\, a_\mathrm{ref}^{1-\gamma}\, a_e^{\gamma}$ with $a_\mathrm{ref}$ the sector's median leg $|f|$ and the resulting cv capped at `cv_cap`. **Outlier-leg screen**: per (flow sector, exporter region, year), implied CC exports over covered production plus implied imports; a ratio above $k$ sets $\sigma = \mathrm{cv}\,|f|$ on the group's bilateral legs. **Contradicted-node screen** ([ADR 0019](adr/0019-contradicted-flow-nodes-lose-anchor-authority.md), after imputation): a Flow node whose outflow prior is below `K_CLOSE` $\times$ its inflow prior (`flow_node_balance.py`, shared with Stage 3's anchors) gets $\sigma = \mathrm{cv}\,|f|$ on its outflow edges.

#### Classes

- **`FeedCollector`** ([collector.py](../src/pysuto/collector.py)) — `.select([feed_id, ...])`, `.aggregate(p=…)`; aligns each feed onto the common edge index and returns an `AggregationResult`.
  The collector treats an edge with $0 < |F_\mathrm{root}| <$ `min_flow_kt` as not covered, except an observed zero (#76); the driver passes `min_flow_kt = 0.01` for both valuations, read in the edge's own unit (kt for Mass, kt CC for content).
- **`AggregationMath`** ([agg.py](../src/pysuto/agg.py)) — vectorised $\Omega$, `f_agg`, `σ_agg` over the full `[N_f × N_r]` matrices.
- **`ImputationEngine`** ([impute/](../src/pysuto/impute/)) — orders an `.register()`-list of imputers; each imputer's `.prepare(acc_struct)` is called at registration so shared lookup structures only build once.
- **`BaseImputer`** ([impute/](../src/pysuto/impute/)) — ABC; subclass and implement `.writer_scope(acc)` (the edges it may write) and `.impute(ledger, step, acc)` to add a new fill method; `.prepare(acc)` and `.attach(agg_result)` are optional hooks.
- **`PriorLedger`** ([impute/ledger.py](../src/pysuto/impute/ledger.py)) — the one write path of the prior: every imputer writes through it, and it enforces no-overwrite and the imputer's declared writer scope at write time, stamping imputer id and pass. `ImputationEngine(strict_scope=True)` raises `ScopeError` on an out-of-scope write (the toy runs so); the default logs it at WARNING and drops it. A primary a relaxed recipe close counted as 0 (`assumed_absent`, ADR 0010) is refused silently to every later imputer except `FlatPriorImputer`, so `time_interp` and the other fills never write it (#312): the prior would hold both the close's 0 and the later value. The edge stays NaN until the flat prior writes its noisy zero.
- **`FillImputer`** ([impute/fill.py](../src/pysuto/impute/fill.py)) — one `Fill(source, rule, scope, id)` of the params relation source ([relation/sources.py](../src/pysuto/relation/sources.py): a params file read as pair `Relation`s `b − r·a = 0`; `rule` ∈ forward / backward / both) with its own σ rule. The **trade** fill is the conc source forward (Mass→CC) in the cross-region `Flow→Process` content scope: it seeds those **CC** import legs from their BACI **Mass** twin (`f_CC = conc·f_Mass`, method `trade_seed`); it runs **first** in the cascade so the network recipe gate can pool the covered legs (see [ADR 0010](adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md)).
- **`NetworkImputer`** ([impute/network.py](../src/pysuto/impute/network.py)) — process-recipe traversal: closes each Process node's CC mass balance (`product = eff·Σinputs`, SEM waste, Mass twins) plus Flow-node balances; the permitted writer of same-region chain edges (its writer scope). A missing same-region primary blocks the forward fill; missing `Scrap` and missing cross-region imports count as 0 (see [ADR 0004](adr/0004-process-recipe-traversal-imputer.md), [ADR 0010](adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md)).
- **`FillEntry` / `CascadeStep`** ([case.py](../src/pysuto/case.py)) — how a case declares its cascade as data (#136): `ImpSection.cascade` lists the entries in declaration order, each a `FillEntry(id, params_id, rule, sigma_rule, twin, scope)` (one `Fill` of a params relation source, built from the registered params file) or a `CascadeStep(id, build)` (the network); the knob `CASCADE_ORDER` is the order they run in and `ImpSection.terminal(ctx)` returns the terminal imputers. The imp sidecar records each entry's params id, param types, rule, σ rule and scope name (`stage_config["case"]["imp"]["cascade"]`), and each fill step's `source_path` (its step record). Writer scopes are enforced by the ledger, not by the entries being exclusive ([ADR 0004](adr/0004-process-recipe-traversal-imputer.md) amendment, [ADR 0011](adr/0011-one-edge-relation-shared-across-imputation-and-constraints.md)).
- **`TimeImputer`** — sliding-window temporal interpolation/extrapolation; `method ∈ {"linear","locf","nocb","mean"}`. Scoped to **same-region** edges in the copper driver: a bilateral trade leg absent in a year is a recorded zero, not a gap ([ADR 0026](adr/0026-bilateral-trade-legs-are-not-time-interpolated.md)); never a stock-change leg.
- **`ShareImputer`** ([impute/share.py](../src/pysuto/impute/share.py)) — the net-stock seed: a `Cathode→MESC` / `Cathode→USC` leg no feed reports is held at 0 with `σ = max(0.01·|co-located Cathode→FU flow|, 0.3 kt)`; terminal, before the flat prior ([ADR 0040](adr/0040-stock-change-legs-are-held-at-their-prior.md)). Its PSC share-transfer mode is pre-empted by the recipe and fires 0×; a PSC seed at 1 % of `CP→Cathode` was rejected on measurement ([ADR 0041](adr/0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md)).
- **`FlatPriorImputer`** — final-fallback flat prior keyed by `AccountingStructure` edge masks; register **last**.
- **`ImputationLog`** — per-edge provenance table: `edge_id`, `step`, `imputer_id`, `pass`, `method`, `source_edges`, `coefficients`, `f_before/after`, `sigma_before/after`, `description`, `param_id`, `relaxed`, `assumed_absent` (`f_before` / `sigma_before` are always NaN: a write only fills a NaN cell).
- **`AggregationResult`** ([agg.py](../src/pysuto/agg.py)) — dataclass holding `f_agg`, `sigma_agg`, `Omega`, `F`, `Sigma`, `feed_ids`, `coverage`, plus `imputation_log`, `imputation_metadata` and `flat_prior` (the edges the last-resort imputer filled) once imputation has run.

#### Key parameters

| Parameter | Default | Where | Effect |
|---|---|---|---|
| `p` | `1.3` | every imputer + `FeedCollector.aggregate` | Lp exponent |
| `q` | `3.0` | `FillImputer` (parametric σ rule) | Sharpness of backward-branch sensitivity knee |
| `alpha_min` | `0.05` | `FillImputer` (parametric σ rule) | Effective-sensitivity floor for inverse multipliers |
| `cv_cap` | `inf` | `FillImputer` | Optional hard cap on per-fill CV |
| `epsilon` | per params row | the source's `Relation` (`meta["epsilon"]`) | Relative residual uncertainty added in quadrature |
| `window`, `gap_inflation` | `5`, `1.0` | `TimeImputer` | Look-back/forward range; per-step σ multiplier |
| `default_cv` | required | `FlatPriorImputer` | CV applied to edges that match no listed mask (no density modulation) |
| `cv_lo`, `cv_hi` (per mask) | from `flat_prior_cv.csv` | `FlatPriorImputer` | Tight / loose endpoints of the coverage-density interpolation `cv_lo + (cv_hi − cv_lo)·(1 − ρ)` |

#### Use case

Run Stage 2 on the toy case in [tutorial/02_aggregate_impute.ipynb](../src/pysuto/tutorial/02_aggregate_impute.ipynb).

#### Diagnostics

`AggDiagnostic` (post-aggregation, pre-imputation) and `ImpDiagnostic` (post-imputation per-step trajectories) live in [diag/](../src/pysuto/diag/). Both expose `.summary()` and `.plot(odir=...)`. `AggDiagnostic` snapshots copies of the aggregation at construction, so build it *before* `engine.run(...)` (imputation mutates `f_agg`/`sigma_agg` in place); one built afterwards would report the imputed arrays.

Both accept an optional `acc_struct=` argument so that diag panels can stratify per-edge quantities by year:

- `AggDiagnostic(agg_result, acc_struct=acc).plot(odir=...)` emits an additional `agg_root_by_year.png` — two panels (log10|f_root|, log10 σ_root) with viridis-coloured step-outline histograms per timestep and a year colorbar. Pools over all feeds.
- `ImpDiagnostic(agg_result, acc_struct=acc).plot(odir=...)` overlays the fully-observed year 1995's raw distribution as a black dashed reference curve on the F-root and CV panels of `imputation_steps.png` — the target shape that imputed steps should approximate. The CV panel uses a log10 axis with vertical reference lines at CV = 0.1, 1, 10, 100. The `imputation_timeseries.png` figure only renders an imputed × marker when the (from, to, valuation) series has ≥1 raw observation in [t-3, t-1] AND ≥1 raw observation in [t+1, t+3] — a defensive guard so non-bilateral fills (e.g. legacy locf/nocb) never appear.

Diag output filenames are plain plot names (`imputation_steps.png`, `agg_distributions.png`, …); the run id is already encoded in the parent folder (`data/proc/imp/<run_id>/diag/`).

#### Glossary detail, Stage 2

The glossary entries below keep their definition and their aliases to avoid. This section holds the full text they had before; each block names its entry.

**val-coverage**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **val-coverage** (reported as the `val_pct` column of the stats/coverage diagnostics) —
> the relative number of edges in a subset that carry a **real** val, where "real" follows
> the imputation-log *method* rule (`has_val`): flat-prior / `flat_prior_noisy_zero` default
> fills are **not** real; observed values — *including a genuine `0.0`* — and
> fan / time / parametric / formula / **network** (including a bounds-**clipped** fill, below)
> imputations **are**. _Avoid_: deciding val-coverage from `f == 0` — a flat-prior fill is `0`
> but so is a real observed zero (and a clipped-to-0 network fill), and the three are
> distinguished by the log method, not the magnitude.

**Bounds-clip (in-imputer) — `network_*_clipped` fill.**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Bounds-clip (in-imputer) — `network_*_clipped` fill.** The NetworkImputer floors every
> fill to the edge's `[lower, upper]` bounds **before** it propagates through the recipe/
> balance fixpoint, so a negative residual (e.g. `Ore→SXEW = Mining→Ore − Ore→MF` in a region
> with little/no SXEW) cannot cascade and drag downstream sums below their true values. The
> clipped value is a **saturated estimate**, not a precise bound: σ is widened to the clip gap
> and the fill is tagged `network_*_clipped`. A clipped network fill still **counts as real
> val-coverage** — it is a network inference, and its low confidence lives in the widened σ,
> not in coverage. _Distinct from_: (1) the engine's **post-return bounds clip**
> ([impute/engine.py](../src/pysuto/impute/engine.py)) which warns + clips *any* imputer's
> out-of-bounds fill — the in-imputer clip makes it a no-op for network fills; (2) the **σ-cap**
> (`_capped`, the `cv_cap` ceiling on σ — caps the *uncertainty*, not the value).

**Recovery efficiency (`eff`).**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Recovery efficiency (`eff`).** The copper-recovery fraction of a Process, in **`CC`** (from the `eff`
> param) — the sibling coefficient to `conc`. A process's recovered Flow product is `eff · Σ(all CC
> inputs)` in CC; the `(1 − eff)` complement is the residual that leaves as the **SEM** residual output.
> `eff` acts on the **sum of all CC inputs — primary plus the `Scrap` SEM-secondary** (Smelting/CFR
> consume Scrap, so `product = eff·(primary + scrap)`). Both the imputer prior and the `eff` **ratio
> constraint** now pool that *same* set, so there is no longer an `eff·Scrap` gap between them — see
> [ADR 0005](adr/0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md). The pooling is
> **declared explicitly in the eff params file** by the `agg` flag on the row (a reader sees it in the
> CSV), never inferred from code. The input-tier distinction is retained only for
> **NaN-propagation** in the imputer, *not* for what gets summed. There are **three** tiers (all pooled
> into `Σin`): **primary** (same-region Flow / SB-Nature inputs — a missing one *blocks* the forward
> fill while derivation is still possible; once the traversal has converged, a missing primary counts
> as 0 *provided ≥1 primary is known* — parallel technology routes such as CP's CER+CSXEW, where an
> underivable leg means the region lacks that route; a recipe with **no** known primary stays blocked,
> [ADR 0010 amendment](adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md));
> **SEM-secondary** (`Scrap` — missing counts as 0); and **bilateral-import** (cross-region
> `Flow→Process` legs — missing counts as 0, non-blocking, [ADR 0010](adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md)).
> The same cross-region-as-0 rule applies to the Flow-node balance (a missing cross-region leg is
> counted as 0, never written). _Avoid_: conflating `eff` (a CC recovery fraction) with `conc` (a
> Mass→CC fraction) — they are different coefficient families.

**Process recipe.**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Process recipe.** A Process node viewed as a mass-closing transformation in CC: its **pooled inputs**
> (primary + `Scrap` secondary) → one **Flow product** (`eff·Σin`) + a **SEM residual output**
> (`(1−eff)·Σin`). The unit of reasoning for the network-traversal imputer
> ([ADR 0004](adr/0004-process-recipe-traversal-imputer.md)). **Mining and `CP` are the
> exceptions**: Mining has *no* `eff` row (eff defaults to 1.0, so CC is conserved CO→Ore) and its
> `WR` satellite output follows the **stripping ratio** (below) on Mass
> (`WR_Mass = stripping_ratio·Ore_Mass`), not the generic `(1−eff)·Σin` SEM-residual rule; `CP` has no `eff` row either, so the recipe writes
> its SEM output `PSC` — a **stock-change leg**, not a residual — as `0` on every unreported
> region-year, at `σ 0.3 kt` rather than the hidden hard pin `0 ± 0` it was until #36 (to be measured;
> [ADR 0041](adr/0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md)).
> _Avoid_: confusing the recipe's **SEM residual-output** (Slag / Tailings / WR, reached via a
> *supply* edge; `PSC` shares the `entity_sub` but is a stock-change leg) with a **SEM secondary
> input** (Scrap, reached via a *use* edge) — both carry `entity_sub == "SEM"`, and only the edge
> direction distinguishes the two roles.

**Trade-seed**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Trade-seed** — the trade `Fill` (a forward fill of the conc relation source in the cross-region
> `Flow→Process` content scope; method `trade_seed`) that sets the **CC** value of a
> *cross-region* import leg from its already-aggregated BACI **Mass** twin
> (`f_CC = conc·f_Mass`), through the source's twin reading. It is the imputation-time complement of the
> trade-share fan: the fan distributes ICSG-CC by BACI geography at *feed* time, whereas trade-seed reads
> the BACI-Mass observation directly at *imputation* time so the network recipe gate can pool the covered
> import legs. Trade-seed's **writer scope** (below) is the cross-region `Flow→Process` CC edges; the
> `NetworkImputer`'s is the same-region edges. A route with no BACI
> Mass observation is left NaN (an honest flat-prior gap), never seeded to 0. See
> [ADR 0010](adr/0010-trade-first-seeding-of-bilateral-flow-process-inputs.md). _Avoid_: reading a
> trade-seed CC as a new measurement — it is the conc twin of the BACI Mass observation.
> **A bilateral leg is never interpolated in time** ([ADR 0026](adr/0026-bilateral-trade-legs-are-not-time-interpolated.md)):
> BACI is a *complete* matrix, so a route carrying no row in a year carried no trade, and the
> `TimeImputer` is scoped to same-region edges — a cross-region leg is neither a fill target nor an
> interpolation anchor. _Avoid_: reading an absent bilateral cell as a gap; before ADR 0026 the
> unscoped interpolator invented 9,215 Mass legs / 20,252 kt of trade, 75 % of which survived into
> the posterior.

**Prior σ (`σ_a`)**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Prior σ (`σ_a`)**:
> The standard deviation of an edge's prior: how far the QP may move the edge from `a₀` at unit
> cost. A **flat** σ (10⁶ kt) states that the edge carries no information of its own.
> _Avoid_: "σ init", "initial σ" — nothing iterates it.


---

### Stage 3 — Constraint building

**What this stage does.** Translates the aggregated `f_agg` + `sigma_agg` into a sparse constraint matrix `G ∈ ℝ^(M×E)` with per-row right-hand side `c(t)` and uncertainty `σ_c(t)`, partitioned by `sense` (`eq` / `le`) and hardness. `G` is time-invariant; `c(t)` and `σ_c(t)` are assembled per timestep.

**Inputs:** `AggregationResult`, `AccountingStructure`, optional CSV files (sum / ratio / bound parameters).
**Outputs:** `ConstraintRun` with `G [M×E]`, `c(t) [M]`, `σ_c(t) [M]`, flow bounds `u`/`l [E]`, and a `row_meta` DataFrame describing each row.

> Uses [power-mean (Lp) uncertainty propagation](#power-mean-lp-uncertainty-propagation) from Stage 1.

#### Math

Five built-in constraint types populate rows of `G`:

| # | Type | Form | Cone | σ_c |
|---|---|---|---|---|
| 1 | Hard mass balance | $\sum_\text{in} a_i - \sum_\text{out} a_i = 0$ | Zero | 0 |
| 2 | Soft sum | $\sum_i \alpha_i a_i = c_k + \varepsilon_k$ | Free | user-specified |
| 3 | Soft ratio | $a_i - r \cdot a_j = \varepsilon_k$ | Free | from params CSV |
| 4 | Hard ratio bound | $a_i - r_u a_j \le 0$, $-a_i + r_l a_j \le 0$ | Non-neg | 0 |
| 5 | Hard lower bound | $-a_i \le -l_i$ for every edge with structural $l_i = 0$ | Non-neg | 0 |

**Type 1 (mass balance).** One row per balanced `(Flow|Process)` node × time pair, with `SB` (open-boundary) nodes excluded. Derived automatically from the `AccountingStructure`'s edge index and roles, restricted to one scope mask when the case names it (`MassBalanceBuilder(mask=…)`; copper's `massbal_scope`).

**Type 2 (soft sum).** σ_c modes: `sd` (absolute), `pedigree` (the caller's pre-computed value, static like `sd`), `fraction_of_flow` (fraction of node-flow magnitude from `f_agg`), `p_mean` (Lp combine over `_rhs_ids` σ_agg using the global Lp exponent). An unknown mode, or `p_mean` on a spec without `_rhs_ids`, raises. A case builds Type 2 through `SumConstraintBuilder.from_mask_anchor` only (mode `sd`; see below); `from_acc_struct`, `from_acc_struct_subentity` and `from_csv` are tests-only API, and `p_mean` / `fraction_of_flow` / `pedigree` are reachable only through them.

**Scale totals, anchors and the no-data σ anchor.** Copper's Type 2 rows are *anchors*: soft sums that pin the group-sum of a masked edge family to the model's own Stage-2 prior (`c = Σ f_agg`), built by `from_mask_anchor` with `σ_c = cv·|c|` and the families of [constraints/anchors.py](../src/pysuto/constraints/anchors.py): the supply and use **scale totals** per process / flow ([ADR 0006](adr/0006-scale-total-supersedes-external-control-total.md); cv by provenance, [ADR 0016](adr/0016-provenance-aware-sigma-on-scale-totals.md), loosened on contradicted nodes, [ADR 0019](adr/0019-contradicted-flow-nodes-lose-anchor-authority.md), with a reporting-resolution σ floor on feed-backed totals only, [ADR 0025](adr/0025-scale-totals-carry-the-feeds-reporting-resolution-as-a-sigma-floor.md)) and the **scrap level anchors** ([ADR 0031](adr/0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md)). Two further Stage 3 mechanisms act on the edges, not on rows: the **no-data σ anchor** replaces the global-median σ_a of flat-prior edges by a sibling-median scale, sparing the structurally determined sectors ([ADR 0013](adr/0013-no-data-flat-prior-anchor.md)), and the **flat-prior pin** (`pin_flat_prior_zeros`) sets `l = u = 0` on `ConstraintRun` for the feasibility probe; an optional prior box (`prior_box_k`, off in copper) tightens bounds to `a₀ ± kσ_a`. Their knobs are the UPPER_CASE fields of the case's Stage 3 section (copper: `scripts/copper.py`; values live there, not here).

**Type 3 (soft ratio).** $r$ is `value_nominal` from the params CSV; σ_c follows the three uncertainty modes (`sd`, `min_max`, `pedigree`) shared with the parametric fill. The same CSV file thus serves both stages. The mode states σ in ratio units; the builder carries it as the relative `σ_r = σ/|r|`, floors it at `sigma_rel_floor` and scales it to flow space by `max(Σ|a₀,num|, |r|·Σ|a₀,denom|, scale_abs_floor)` (a net-stock row, `stock_bound`, reads its half-band `value_max` as p95 ≈ 1.645 σ and scales by the throughput `Σ|a₀,denom|` alone; it needs a `value_max` and one valuation on both sides, else it raises) — see [GLOSSARY.md](../GLOSSARY.md) "Relative ratio σ". A row with `agg=true` is **set-scoped**: `RatioConstraintBuilder` pools every matched a-side edge against every matched b-side edge per `(region_to, time, valuation)` bucket into one soft row $\sum b - r\sum a = \varepsilon$ — used for the pooled recovery recipe `product = eff·(Concentrate/Matte + Scrap)` at Smelting/CFR (see [ADR 0005](adr/0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md)).

**Build-time checks.** After assembly the engine raises `ValueError` on an impossible state: a `G` whose width is not the edge count or that references a column outside it, a hard row with σ_c ≠ 0 or a soft row with σ_c ≤ 0 (or non-finite) in any year, a flow with `u < l`, a builder whose per-year block has another row count than its static block; and `ConstraintUnitMixingError` on a row of a non-ratio type that mixes valuations. The mass-balance row count and the per-year uniformity of the ratio rows only log (`CHECK WARN`): a ratio template legitimately expands to a different row count in a year whose feed covers fewer edges. `FlowBoundsBuilder` raises on a structural `l > 0`, which no row would enforce.

**Type 4 / Type 5.** Hard inequalities, σ_c = 0. When a ratio-bound row sets `agg=true`, `RatioBoundBuilder` sums every matched a-side edge into one G row and every matched b-side edge into another, scoped per `(region_to, time, valuation)` bucket — used both for stock-change bounds (numerator/denominator summed over source regions per destination region; see `data/input/params/stock_bounds/current/params.csv`) and for the pooled eff bounds at Smelting/CFR. Bucketing on `region_to` keeps each process region's recipe separate. Since Round X the stock-change rows cover the producer stock change (`CP→PSC`) only: the user and exchange stock-change legs (`Cathode→USC`, `Cathode→MESC`) carry no row and are held at their prior by the Stage-2 net-stock seed σ, `max(0.01·|co-located FU flow|, 0.3 kt)`, because a signed leg with a loose σ is a sink for every contradiction of the cathode balance and a zero-centred row pulled the reported stock changes toward zero ([ADR 0040](adr/0040-stock-change-legs-are-held-at-their-prior.md)). Since Round Y the producer stock change is tied to its Mass twin by the cathode concentration row, and `RatioBoundBuilder` emits no one-signed hard band on any signed edge — such a band is satisfiable for a negative pair only at zero, which had kept every USC/MESC leg non-negative; seeding PSC like USC/MESC was measured and rejected ([ADR 0041](adr/0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md)).

**Prior regularisation.** Before persisting the `ConstraintRun`, the engine applies an Lp regularisation to both edge-level σ_a and free-cone row-level σ_c:

$$\sigma_a'[i] = \left( |\sigma_a[i]|^p + |\mathrm{CV}_a^\mathrm{min} \cdot a_0[i]|^p + |\sigma_a^\mathrm{floor}|^p \right)^{1/p}$$

$$\sigma_c'[k] = \left( |\sigma_c[k]|^p + |\mathrm{CV}_c^\mathrm{min} \cdot c[k]|^p + |\sigma_c^\mathrm{floor}|^p \right)^{1/p} \quad (\text{free rows only})$$

The four σ arguments (`sigma_a_cv_min`, `sigma_a_abs_floor`, `sigma_c_cv_min`, `sigma_c_abs_floor`) and `nodata_anchor_cv` are required of `ConstraintEngine`: the case owns them (copper's `SIGMA_*` and `NODATA_ANCHOR_CV` in `scripts/copper.py`); only $p$ has a library default, `pysuto.lp.P_DEFAULT`. Hard rows keep σ_c = 0 unchanged. The $\mathrm{CV}^\mathrm{min}$ term *is* the relative floor (a misreading of an earlier REP claimed it was missing); the absolute floor is the numerical safety net.

**Ordering.** σ_a regularisation runs *before* `build_dynamic` propagates σ_a → σ_c via the Lp combine over `_rhs_ids` (the `p_mean` mode, tests-only API; `tests/test_constraint_checks.py` pins the order), so the regularised σ_a flows into σ_c. This is deliberate: the relative term $\mathrm{CV}_a^\mathrm{min} \cdot |a_0|$ is a modelling claim about how tightly we believe σ_a, not a numerical guard, so it *should* propagate to σ_c. The absolute floor $\sigma_a^\mathrm{floor}$ is the actual safety net — once the FlatPriorImputer's 3-level scale cascade is in place (see Stage 2) it should essentially never fire on imputer-filled edges, so its presence does not distort downstream uncertainty propagation.

**Per-timestep deactivation of soft sum rows.** Each `SumConstraintBuilder` spec carries a `time` field (the year baked into `constraint_id`); `from_csv` specs with wildcard or mixed `time` are stored as `time=None` and **rejected** when the engine partitions `G` by year (`_partition_G_by_time` raises `ValueError`): such a row would sum edges across years and couple the per-year QPs. When `ConstraintEngine.run` calls `build_dynamic(t)` for a given timestep, rows whose `spec["time"]` does not match `t` are *deactivated* by setting `c=0` and `σ_c = SIGMA_C_INACTIVE` (default `1e10`). Mass-balance and flow-bound rows are active in every timestep; `RatioBoundBuilder.build_dynamic` deactivates hard band rows of other years by setting `c = BOUND_C_INACTIVE`, which makes the inequality trivially satisfied. The σ_c sentinel makes the row's objective contribution $(1/\sigma_c)^2 \approx 10^{-20}$ — effectively unenforced — without touching `G` or the cone structure. `Solver.solve_all` rebuilds the QP Hessian `P` per timestep so the per-t σ_c entries actually take effect (Clarabel's `solver.update(P=…, q=…, b=…)`).

#### Classes

- **`ConstraintEngine`** ([constraints/](../src/pysuto/constraints/)) — `.register(builder)` then `.run(agg_result, timesteps=…)`; persists to a `ConstraintRunRegistry`.
- **`BaseConstraintBuilder`** — ABC; subclass and implement `build_static(...)` / `build_dynamic(...)`.
- **`MassBalanceBuilder`** — type 1 (auto-derived from `acc_struct`).
- **`SumConstraintBuilder`** — type 2; `.from_mask_anchor(...)` is what a case builds with; `.from_acc_struct(acc, sigma_fraction=…)` (auto-derived), `.from_acc_struct_subentity(...)` and `.from_csv(...)` (hand-written rows) are tests-only API.
- **`RatioConstraintBuilder`** — type 3 (`.from_csv(...)`).
- **`RatioBoundBuilder`** — type 4 (`.from_csv(...)`).
- **`FlowBoundsBuilder`** — type 5: one row `−a_i ≤ 0` per edge whose structural lower bound `acc.bounds.l` is 0 — every edge but the signed stock edges ([ADR 0039](adr/0039-flow-lower-bounds-are-enforced-in-the-qp.md)). Upper bounds and the Stage-3 zero pin produce no rows; they stay on `ConstraintRun.u/.l` for the feasibility pre-check.
- **`ConstraintBlock`** — internal container for one builder's rows.
- **`ConstraintRun`** — bundle of `G`, `row_meta`, `a0`, `sigma_a`, `u`, `l`, `timestep_data[t] = {"c", "sigma_c"}`. `row_meta` carries a uniform column set across all builders (`pysuto.constraints.naming.ROW_META_COLS`): `[constraint_id, type, variant, is_hard, sense, time, valuation, sector, region, entity, entity_sub, ratio_kind, bound_dir, num_*/denom_* (ratio numerator/denominator decode)]`. Every `constraint_id` follows the grammar **`{type}_{variant}_{specs}_{year}`** where `type ∈ {massbal, sum, ratio, bound}` and `variant` is the role within the family (`usefulsupply`/`subentity`/`supplytotout`/`usetotout`/`smelterscrap`/`directrefscrap`/csv-label for sum, `ratioeq`/`bound` for ratio, `lower` for bound, or a user `param_id`). The id is a unique **display key — never parse it; read the columns** (see [GLOSSARY.md](../GLOSSARY.md) §"Constraint identity"). The `time` column is the year the row belongs to (the final id token; set at spec construction by each builder); a wildcard-time CSV constraint gets `None` (id ending in `_na`) and is rejected by the partition. The engine stores a per-t partition `G_blocks[t]` / `edge_cols_by_t[t]` (persisted under `blocks/`) and raises if a row's non-zeros leave its own time bucket (the **no cross-time coupling** invariant) or its `time` is no year of the accounting structure, so no row is silently left out of every block; Stage 4's block solver consumes it to solve one ~30× smaller QP per year.
- **`ConstraintRunRegistry`** — disk layout under `data/proc/const/{run_id}/` (G.npz, row_meta.parquet, a0.npy, sigma_a.npy, u.npy, l.npy, `timesteps/t_{year}.npz`).

#### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `sigma_a_cv_min` | `0.05` | Relative CV floor in the σ_a Lp regularisation (`0.05·|a₀|`) |
| `sigma_a_abs_floor` | `1e-6` | Absolute numerical floor for σ_a (safety net; the FlatPrior scale cascade should keep this dormant) |
| `sigma_c_cv_min` | `0.05` | Relative CV floor for free-row σ_c (`0.05·|c|`) |
| `sigma_c_abs_floor` | `1e-3` | Absolute numerical floor for free-row σ_c |
| `p` | `1.3` | Lp exponent for σ_a / σ_c regularisation |
| `storage_dir` | `"data/proc/const"` | Registry root |

#### Use case

Run Stage 3 on the toy case in [tutorial/03_constraints.ipynb](../src/pysuto/tutorial/03_constraints.ipynb).

A `ConstDiagnostic` plot suite (in [diag/](../src/pysuto/diag/)) can be triggered via `ConstraintEngine.run(..., diag_plot=True)`.

#### Glossary detail, Stage 3

The glossary entries below keep their definition and their aliases to avoid. This section holds the full text they had before; each block names its entry.

**Satellite flow**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Satellite flow** — an edge of the accounting structure that carries a calculated value but is
> excluded from its source node's mass balance by the `massbal_scope` mask (acc `masks.csv`). In
> copper: `Mining → WR` (waste rock), mass `k·Ore` from the stripping ratio, copper content zero; its
> mass enters Mining from nature without an input edge (unused extraction lies outside the crude-ore
> feed), so the reported PSUT does not balance Mining in gross mass by exactly the waste rock. See
> [ADR 0038](adr/0038-waste-rock-is-a-satellite-mass-flow.md). _Avoid_: "residual output" (a
> residual output closes its node's balance; a satellite never does) and "slack".

**Structural lower bound**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Structural lower bound** — the per-edge `l` of the accounting structure (`acc.bounds.l`): 0 on
> every edge except the signed stock-change edges (`Cathode→USC`, `Cathode→MESC`, `CP→PSC`), whose
> `l` is −1e6. Since [ADR 0039](adr/0039-flow-lower-bounds-are-enforced-in-the-qp.md) each
> `l = 0` enters the QP as one hard `nonneg` row `−a ≤ 0` (type `bound`, variant `lower`). A signed
> edge (`l < 0`) carries no **one-signed band**
> ([ADR 0041](adr/0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md)). _Avoid_:
> reading `ConstraintRun.l` as the structural bound — Stage 3 overwrites it with the prior box and the
> **zero pin** (`l = u = 0` on no-data edges, [ADR 0013](adr/0013-no-data-flat-prior-anchor.md)),
> and those boxes, like every upper bound `u`, reach only the feasibility pre-check. Before ADR 0039
> no bound reached the QP, so `l ≤ a ≤ u` in older REPs and the README describes the probe, not the
> solve.

**Stock-change leg**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Stock-change leg** — one of the three signed edges of the account: `Cathode→USC`,
> `Cathode→MESC` and `CP→PSC`, each a net addition to a stock (negative = a draw-down), structural
> lower bound −1e6. No one-signed hard band sits on it. For `USC` / `MESC` the prior is a reported
> value or the **net-stock seed** and no imputer writes it; for `PSC` an unreported year is still the
> recipe's eff-less `0`, priced at `± 0.3 kt` since #36 (to be measured; a seed at 1 % of `CP→Cathode`
> was rejected on measurement,
> [ADR 0041](adr/0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md)).
> _Avoid_: "stock leg", "stock edge", "Δstock edge".

**Net-stock seed**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Net-stock seed** — the prior of a `MESC` / `USC` leg that no feed reports: value 0, held there
> with `σ = max(0.01·|co-located Cathode(i)→FU(j) flow|, 0.3 kt)` — the same pricing as a
> feed-backed level ([ADR 0040](adr/0040-stock-change-legs-are-held-at-their-prior.md)). A
> `MESC` / `USC` leg's prior is a reported value or this seed; no imputer manufactures one. The same
> seed on `PSC`, priced at 1 % of the region's `CP→Cathode`, was a sink (272 kt/yr absorbed) and is
> rejected on measurement ([ADR 0041](adr/0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md)).
> _Avoid_: reading a seeded 0 as a reported zero; "throughput-scaled band" (the retired p95 pricing).

**Relative ratio σ (`σ_r`).**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Relative ratio σ (`σ_r`).** The σ of a soft ratio row as a fraction of its ratio, `σ_r = σ/|r|`,
> where `σ` is the params row's stated σ in **ratio units** (`value_sd`, or `(value_max − value_min)/4`)
> and `r` its `value_nominal`. `RATIO_SIGMA_REL_FLOOR` floors it and the row's flow scale multiplies
> it, `σ_c = max(σ_r, floor)·max(Σ|a₀,num|, |r|·Σ|a₀,denom|, RATIO_SCALE_ABS_FLOOR)`, which at a prior
> on the ratio is `σ·Σ|a₀,denom|`. The imputers read `σ` itself, and net-stock rows (`r = 0`) scale
> `σ` off the throughput alone. _Avoid_: calling `value_sd` a CV; quoting a ratio row's σ without
> saying which of the two it is.

**Scale total**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Scale total** — a Type-2 **soft-sum** (`Σ aᵢ = c + ε`, free cone) whose RHS `c` is the model's
> **own prior** group-sum (`c = Σ f_agg`), used for **scale-preservation** against the homogeneous
> collapse — not to add information. "Kind-of-external" because the warm-start solve (`a₀ = f_agg` at
> t₀, `a_post(t−1)` after) never sees this fixed Stage-2 total. Three families: **supply total output**
> (per process — its process→flow useful output), **use total output** (per flow — Σ over all
> destinations incl. final demand), and the **scrap level anchor** below. On a **degenerate (one-leg)
> fan** the prior group-sum coincides with the feed's own published number, and such a row then *does*
> add information — this is a consequence of the fan, not the ADR 0006 mechanism (an external totals
> file reconciled in a separate pass). Groups are classified **fabricated vs feed-backed** (per
> `agg_result.coverage`) — but both keep the same tight cv: loosening fabricated groups was tried
> and rejected, because these rows double as the counterweight to the nodata zero anchors and
> loosening them collapses the interior chain
> ([ADR 0016](adr/0016-provenance-aware-sigma-on-scale-totals.md)). σ_c carries an
> **absolute floor at the feeds' reporting resolution** (0.3 kt, the rounding σ of integer-kt
> ICSG/crude_ore values, Lp-combined with the cv term — [ADR 0025](adr/0025-scale-totals-carry-the-feeds-reporting-resolution-as-a-sigma-floor.md));
> it reaches totals below ~18 kt only, so a sub-kilotonne total is never priced to tens of tonnes.
> _Avoid_: **control total** (its RHS is an *external* published
> number — see README; that mechanism was removed by
> [ADR 0006](adr/0006-scale-total-supersedes-external-control-total.md)); **anchor** /
> **coherence soft-sum** (deprecated terms — see **Control total** _Avoid_ above);
> reading a scale total as only a prior-preservation device — on no-data subgraphs it is also
> what holds scale against the zero anchors (ADR 0016).

**Scrap level anchor**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Scrap level anchor** — the third **scale total** family (`smelterscrap`, `directrefscrap`): a
> soft-sum stating the **level** of *one input class* of a node — the scrap ICSG reports entering
> `Smelting` and entering `CFR` — per region-year, **CC only**, and **only where a feed observed it**.
> It is the one `sum` family whose *row existence is data-gated*, which is what makes ADR 0025's σ
> floor unconditional for it. Needed because `Scrap` is an `SB` node: `Scrap→Process` edges are in
> neither output-total family, so before [ADR 0031](adr/0031-icsg-scrap-series-anchor-their-level-not-only-their-share.md)
> the only statement about them was the `scrap_share` **band**, and a share band over a node with no
> throughput is satisfied trivially. _Avoid_: calling it a **total input** (that is *all* incoming
> edges — this is one class of them); reading the **absence** of a scrap-level row as a prohibition —
> an absent row is an *unstated* level, and the edge keeps exactly the freedom it had.

**Contradicted flow node**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Contradicted flow node** — a Flow node whose **inflow** prior (`Process→Flow`) and **outflow**
> prior (`Flow→Process` + `Flow→SB`) cannot both hold under its own **hard** mass balance. Because
> `σ_c = ANCHOR_CV·|c|` scales with the total, the *smaller* side is the cheaper anchor to satisfy,
> so the solver collapses the node to the smaller prior even when that side is the worse evidenced
> one (Russia Matte 1995: inflow 545 kt CC from ICSG concentrate use vs outflow 50.8 from an
> under-reported ICSG refining series ⇒ the smelting chain collapsed). Detected by
> `pysuto.flow_node_balance.contradicted_flow_nodes` (outflow < `K_CLOSE`·inflow) and treated by
> stripping the **outflow** side's authority — σ on its covered edges and on its `usetotout` anchor
> — was tried and **rejected**: σ reprices a data contradiction without resolving it (ADR 0019).
> The term survives as a **diagnostic**: a flag is a statement about the *input data* (the
> source's two series disagree for that region-year), not about the reconciliation, and the
> remedy is data-side.
> _Avoid_: reading a downstream export as an outlet for the node — anode exports are produced *by*
> CFR *from* matte, so they raise the matte requirement rather than relieving it. See
> [ADR 0019](adr/0019-contradicted-flow-nodes-lose-anchor-authority.md).

**Anchor placement**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Anchor placement** — *which edge* a source series is attached to, as distinct from its value.
> A production series measures throughput at one point of the chain; converting it into an edge
> prior by dividing through efficiencies moves it to a *different* point, and each stage crossed
> adds an assumption. ICSG smelter production ÷ `eff_Smelting` is the concentrate entering
> `Smelting` (one stage); ICSG refined production ÷ `eff_ER` is the anode entering `ER` (one
> stage). Dividing that same refined series by `eff_CFR·eff_ER` instead placed it two stages back,
> on the matte entering `CFR` — and thereby assumed every anode a region makes is refined at home,
> which is false for anode/blister exporters. A misplaced anchor is not a σ problem and not a
> value problem: the number is right and the edge is wrong.
> _Avoid_: anchoring both sides of a hard-balanced node — the balance already determines the
> second side, and double anchoring is the over-constraint
> [ADR 0006](adr/0006-scale-total-supersedes-external-control-total.md) rejected. See
> [ADR 0021](adr/0021-icsg-refined-production-anchors-anode-into-er.md).

**Node-scoped vs set-scoped constraint**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Node-scoped vs set-scoped constraint** — a `massbal` or node-`sum` row balances **one node**, so its
> specs render that node as sector_region_valuation; the collapse of a node to sector_region is
> sound because constraint rows are emitted over **concrete** nodes only (the `sector→entity` 1:1
> invariant holds; open `<any>` sectors never reach a row). A `ratio` row is either **pair-scoped** —
> relating **two edges**, each rendered full `from→to`, numerator over denominator, with `num_*`/`denom_*`
> populated — or **set-scoped** when its row carries `agg=true`: it pools **many** a-side edges against
> many b-side edges per `(region_to, time, valuation)` bucket (`Σb − r·Σa`), so it has no single
> numerator/denominator edge and its identity rests on the `variant` plus `region`/`valuation` group-key
> specs (`num_*`/`denom_*` are null, exactly like a set-`sum`). A set-`sum` (a csv constraint grouped
> across nodes) spans **many** edges with no single node — its identity rests on the `variant` plus
> group-key specs; by contrast the `supplytotout` / `usetotout` **scale totals** — and the
> `smelterscrap` / `directrefscrap` **scrap level anchors**, which carry the *consuming* node — are
> node-`sum` rows and so carry populated `sector`/`region`. _Avoid_:
> expecting a single `sector`/`region`, or populated `num_*`/`denom_*`, on a set-sum or **set-scoped
> (agg) ratio** row (they are null there).


---

### Stage 4 — Solver

**What this stage does.** Loads a persisted `ConstraintRun`, runs a per-period feasibility probe on the hard rows, and solves a σ-weighted QP per timestep. The work is split in two ([ADR 0042](adr/0042-stage-4-owns-the-model-a-qp-backend-solves-a-standard-form-qp.md)): the orchestrator `Solver` owns the model — it conditions σ, chooses the prior, and builds one neutral standard-form QP per period (`pysuto.qp.QPProblem`: `P`, `q`, equality rows, `≤` rows over `x = [a; ε]`, with every soft row realised as `G a − ε = c` there) — and an injected **QP backend** (`ClarabelBackend` in [clarabel_backend.py](../src/pysuto/clarabel_backend.py), the only module that imports `clarabel`) solves it and reports a neutral status, `solved`, `solved_inaccurate` or `failed`, beside its raw one. The orchestrator raises `RuntimeError` on `failed`, quoting the raw status; there is no fallback. The per-year prior is the conditioned `f_agg` slice (`prior_mode="cold"`, what `run_solver.py` runs by default) or the previous year's solution `a_0(t) = F_opt(t-1)` (`prior_mode="warm"`, the `Solver` class default). A serial run asks the backend to `reuse` its factorisation: `ClarabelBackend` then calls `solver.update(P=…, q=…, b=…)` on the previous period's instance when that period's `A` and the sparsity of `P` are identical (`P` itself is rebuilt each year because σ_c on free rows can vary), and otherwise builds a fresh one; a stalled solve on a reused instance is retried cold inside the backend.

**Inputs:** `ConstraintRun` run ID (loaded from `data/proc/const/`).
**Outputs per timestep:** `OptimResult` plus solver telemetry (`objective`, `iterations`, `solve_time_s`, `solver_status` — the neutral status — with `backend` and `backend_status`, and the backend's `convergence` dict and optional `iter_history`, whose columns the backend owns). The arrays are **year-local** slices; see [Per-t block solve & year-local persistence](#per-t-block-solve--year-local-persistence). Opt runs written before the split load with `solver_status` mapped (`Solved` → `solved`, `AlmostSolved` → `solved_inaccurate`) and `backend = "clarabel"`.

#### Math

##### The (P*) problem

$$\min_{\mathbf{a},\,\boldsymbol{\varepsilon}} \quad (\mathbf{a} - \mathbf{a}_0(t))^\top \Sigma_a^{-1} (\mathbf{a} - \mathbf{a}_0(t)) \;+\; \boldsymbol{\varepsilon}^\top \Sigma_c(t)^{-1} \boldsymbol{\varepsilon}$$

$$\text{s.t.} \quad \mathbf{G}\,\mathbf{a} - \boldsymbol{\varepsilon} = \mathbf{c}(t), \qquad a_i \geq 0 \;\; \forall i \in \mathcal{L}$$

| Symbol | Meaning |
|---|---|
| $\mathbf{a}$ | All flow values (one per root edge × time slice) |
| $\mathbf{a}_0$ | Prior — the conditioned `f_agg` slice of year $t$ (`prior_mode="cold"`, the driver default); under `"warm"` (the `Solver` class default) `f_agg` at $t=0$ and `F_opt(t-1)` for $t > 0$ |
| $\Sigma_a$ | Diagonal of σ_a² (regularised in Stage 3) |
| $\boldsymbol{\varepsilon}$ | Slack on free-cone rows; zero by construction on hard rows |
| $\Sigma_c(t)$ | Diagonal of σ_c²(t) on free rows; **may vary per timestep** (e.g. `SumConstraintBuilder` per-year row activation via `SIGMA_C_INACTIVE`; `fraction_of_flow` mode) |
| $\mathbf{G}$ | Constraint matrix from Stage 3 (time-invariant) |
| $\mathbf{c}(t)$ | Per-timestep RHS from `timestep_data[t]["c"]` |
| $\mathcal{L}$ | Edges whose structural lower bound `acc.bounds.l` is 0 — all but the signed stock edges; enforced as Type-5 rows of $\mathbf{G}$ ([ADR 0039](adr/0039-flow-lower-bounds-are-enforced-in-the-qp.md)). The upper bounds $\mathbf{u}$ and the zero pin reach only the feasibility pre-check |

##### Standard form

The orchestrator hands the backend $\min_x \tfrac12 x^\top P x + q^\top x$ s.t. $A_{eq}\,x = b_{eq}$, $A_{le}\,x \le b_{le}$ over $x = [\mathbf{a};\, \boldsymbol{\varepsilon}]$; `ClarabelBackend` maps the equality rows to `ZeroConeT` and the `≤` rows to `NonnegativeConeT` ($A x + s = b$, $s \in \mathcal{K}$). The transform:

$$P(t) = 2\begin{bmatrix} \Sigma_a^{-1} & 0 \\ 0 & \Sigma_c(t)^{-1} \end{bmatrix} \quad \text{(upper triangle only, CSC format)}$$

$$q(t) = \begin{bmatrix} -2\,\Sigma_a^{-1}\mathbf{a}_0(t) \\ \mathbf{0} \end{bmatrix}, \qquad A = \begin{bmatrix} \mathbf{G}_\text{free} & -\mathbf{I} \\ \mathbf{G}_\text{zero} & \mathbf{0} \\ \mathbf{G}_\text{nonneg} & \mathbf{0} \end{bmatrix}, \qquad b(t) = \begin{bmatrix} \mathbf{c}_\text{free}(t) \\ \mathbf{c}_\text{zero}(t) \\ \mathbf{c}_\text{nonneg}(t) \end{bmatrix}$$

Cone order: `ZeroConeT(n_free)` for the soft-equality residual rows (ε is a free decision variable; the row `Gx - ε = c` is itself an exact equality), `ZeroConeT(n_zero)` for hard balance rows, `NonnegativeConeT(n_nonneg)` for bound rows. `A` is time-invariant; `P(t)`, `q(t)`, and `b(t)` change per timestep — `P(t)` because $\Sigma_c$ on free rows may vary (e.g. `SumConstraintBuilder` deactivates rows of other years via `SIGMA_C_INACTIVE`), `q(t)` from the per-year prior $\mathbf{a}_0(t)$, `b(t)` from $\mathbf{c}(t)$. `solver.update(P=…, q=…, b=…)` refreshes them between timesteps while Clarabel reuses its solver state for $A$.

**Defects form** (`qp_variables="defects"`, driver knob `QP_VARIABLES`, which ships `"defects"`, as did the reference chain (`opt_20261119_130302_b20f0dba`); the class default is `"flows"`, the form above; #33). The same QP solved for the defect $\mathbf{d} = \mathbf{a} - \mathbf{a}_0(t)$: $x = [\mathbf{d};\, \boldsymbol{\varepsilon}]$, the same $P(t)$ and $A$, $q = \mathbf{0}$ and $b(t) = \mathbf{c}(t) - \mathbf{G}\,\mathbf{a}_0(t)$, so Clarabel's cost is $J$ itself instead of $J - \sum_i (a_{0,i}/\sigma_{a,i})^2$ (2015 of `const_20261019_170151_13460e94`: a constant of 1,288,065 against an optimal $J$ of 117,282). `F_opt` comes back as $\mathbf{a}_0 + \mathbf{d}$ and `objective` in the flows convention under both forms; Clarabel's relative gap is measured against its own cost, so the same `tol_gap_rel` is ≈ 10× tighter in absolute terms under the defects form. At feas/gap 1e-8 the defects form reached Solved in six probed years within 0.2 of a certified lower bound on $J$, where the flows form stopped 516–4,349 above it (`scripts/explo/issue33_conditioning_probe.py`, `docs/rep/figs/issue33/probe_const_20261019_170151_13460e94.csv`). Block path only; the feasibility pre-check has no σ-weighted cost.

##### Σ_a conditioning

To bound the condition number of $\Sigma_a^{-1}$, `Solver._condition_prior` floors and ceilings σ_a:

$$\text{floor} = \max(\sigma_a^\mathrm{floor,abs},\ \sigma_a^\mathrm{floor,rel} \cdot |a_0|), \qquad \text{ceiling} = \sigma_a^\mathrm{ceiling\_factor} \cdot \max_\mathrm{finite}\sigma_a$$

Defaults give ~12 orders of magnitude — fine for Clarabel with default static + dynamic regularisation.

##### Feasibility pre-check

Before solving, every requested period's hard subsystem (its hard equalities and `≤` rows plus its edges' bounds $l \le a \le u$) is solved as a pure feasibility QP (`P = εI`, `q = 0`) through the same backend, on the same thread pool as the solves. If the backend reports `failed`, `Solver` raises with a diagnostic that lists the top hard-row offenders of that period by residual. `FEASIBILITY_CHECK = "none"` skips the probe (a sweep point, #104; a failed period still raises in its own solve). The earlier first-year-only probe over the full multi-year `G` (≈ 7 min on copper whatever the linear solver or thread count) is gone.

##### Per-t block solve & year-local persistence

When the `ConstraintRun` carries a per-t partition (`G_blocks`, `edge_cols_by_t` — built in Stage 3 whenever constraints have no cross-time coupling), `solve_all` dispatches to `_solve_all_via_blocks`: each year is solved as an independent QP over only that year's edges (`cols_t`) and rows (`rows_t`), ~30× smaller than the global QP. Because the edge set partitions **disjointly by time**, every edge is solved exactly once — in its own year — and the complete `a*` is the disjoint union of the per-period slices. Under `prior_mode="warm"` the previous year's `F_opt` is carried into the next year's prior by edge template (`AccountingStructure.warm_start_gather`, so the const run must name an acc that loads: otherwise the run raises, there is no positional fallback; the chain follows the order of the requested `timesteps`, the first requested period taking the cold seed, so a subset or an unordered list chains as given); under `"cold"` every year starts from its own `f_agg` slice.

Each per-t `OptimResult` therefore persists **only that period's solved values**: `F_opt` over `cols_t` `[E_t]`, `residuals` / `c_star` over `rows_t` `[M_t]`, plus the `cols_t` / `rows_t` global-index arrays — with **no cross-year prior placeholders**. (Earlier block runs stored the full `[E]` vector with non-`t` edges frozen at the prior `a_0`; this surfaced the prior mass-balance infeasibility on every *other* year's rows and a spurious bimodal `a` distribution — see [`docs/rep/analysis_opt_20260527_095157_40846da4_20260528_block-solver-massbal-artifact.md`](rep/analysis_opt_20260527_095157_40846da4_20260528_block-solver-massbal-artifact.md).) The legacy full-G path (empty `G_blocks`) was removed with ADR 0042: a constraint run without a per-t partition raises. Opt runs it wrote keep full-length `[E]` / `[M]` vectors with `cols_t = rows_t = None`; the npz `cols_t` key is the schema discriminator on load. `SolverDiag` reassembles the disjoint per-period slices into one complete `a*` / `residuals` at load time, so every figure (and the mass-balance residual) reads the true, placeholder-free solution. Loading a pre-fix block run (const has `G_blocks` but the opt npz lacks `cols_t`) emits a `warn_if_prefix_block_run` warning.

##### Parallel solve

Under `prior_mode="cold"` no period reads another period's optimum, so with `n_workers > 1` the periods are solved concurrently on a thread pool (`min(n_workers, periods)` threads), each period in its own backend call (so its own Clarabel instance); the results come back in period order and are bit-identical to solving each period on its own. A warm run is a chain, so `n_workers > 1` with `prior_mode="warm"` raises. `ClarabelBackend` pins Clarabel's linear solver (`direct_solve_method`, default `"faer"`) to one thread for every solve, probe or period, serial or parallel: on the copper period QPs multi-threaded faer was slower than one thread and gave the same answer, so the cores go to the periods instead (#70). Reusing the KKT factorisation is a serial-only feature; a parallel solve factorises each period on its own. The first failed period cancels the periods not yet started and raises naming every period that failed. The per-period probes run before any period solve. The sidecar records `n_workers`, `qp_variables`, `backend` and `backend_settings` (and, under their earlier names, `max_threads` and `direct_solve_method`) in `stage_config` and the wall-clock of the probe plus the period solves as `stage_summary.solve_wall_s`. On copper (30 periods, 8 cores) Stage 4 took ≈ 12 min before the per-period probe: the first-period full-`G` probe ≈ 7 min, the period solves ≈ 3.3 min, and persisting the run plus rendering its diagnostics ≈ 1 min (the per-period probe has yet to be timed on copper, #56). The pre-#70 serial solve, with multi-threaded faer, spent ≈ 31 min on the period solves; with one faer thread the periods sum to ≈ 24 min, so the thread pinning and the concurrency share the gain. Every period is bit-identical to the pre-#70 solve (#71: `opt_20261014_132742_1635b787` against `opt_20261012_122124_40b1b370`).

#### Classes

- **`Solver`** ([solver.py](../src/pysuto/solver.py)) — `Solver(backend, …)`: `.load_run(run_id)`, `.feasibility_check(run, t_label)`, `.solve_all(run_id, timesteps=…, feasibility_check=…)`, `.persist(run_id, results, output_dir, …)`. It imports no numerical solver.
- **`QPProblem` / `QPResult` / `QPBackend`** ([qp.py](../src/pysuto/qp.py)) — the neutral period QP, a backend's answer (`x`, neutral `status`, `raw_status`, `iterations`, `solve_time_s`, `objective`, optional `info` and `iter_history`) and the protocol `solve(problem, reuse, verbose)`, `name`, `settings`.
- **`ClarabelBackend`** ([clarabel_backend.py](../src/pysuto/clarabel_backend.py)) — Clarabel behind that protocol: tolerances, reduced tolerances, `max_iter`, `direct_solve_method`, `max_threads`, factorisation reuse, cold retry, iteration history (`ITER_DTYPE`).
- **`OptimResult`** ([solver.py](../src/pysuto/solver.py)) — per-timestep dataclass: `F_opt`, `eps_opt`, `c_star`, `residuals`, `objective`, `iterations`, `solve_time_s`, `solver_status`, `convergence`, optional `iter_history`, `backend` / `backend_status`, plus `cols_t` / `rows_t` (the global indices the year-local `F_opt` / `residuals` slices live on; `None` on runs written by the removed legacy path).

Solver diagnostics live in [diag/](../src/pysuto/diag/) (`SolverDiag`, 12 figures: solver health and per-iteration convergence traces, the objective decomposition Σz² per family, objective share by sector key and by identity with z-profiles, sector totals with retention, and the prior-vs-posterior rocket — the module docstring is the catalogue; `plot(regions=[…])` restricts a portfolio to one region). `plot()` also writes `ranking_<family>.csv` (identity rankings by objective share) and `provenance.md`; `SolverStatsDiagnostic` writes the per-year `edges_<year>.csv` / `constraints_<year>.csv` on demand (`scripts/solver_stats.py <opt_run_id> --years …`; the solver drivers write only the rankings) from the same all-years frames the figures read, so tables and figures agree by construction. Posterior uncertainty (`σ_a*`, confidence intervals) is produced by [Stage 5](#stage-5--posterior-uncertainty-propagation) (`KalmanUncProp` in [uncprop.py](../src/pysuto/uncprop.py)).

#### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `backend` | required | The QP backend that solves each period (`ClarabelBackend`; the driver builds it from its knobs) |
| `const_dir` | `"data/proc/const"` | Where `Solver` looks up `ConstraintRun`s |
| `opt_dir` | `"data/proc/opt"` | Where `persist` registers the opt run |
| `acc_dir`, `input_dir` | `"data/proc/acc"`, `"data/input"` | A warm run loads the const run's acc from `acc_dir` and checks its build against the acc registry under `input_dir` |
| `prior_mode` | `"warm"` | `"warm"` chains `a_prior(t) = a*(t−1)`; `"cold"` seeds every period from its own `f_agg` slice. The driver's `--prior-mode` defaults to `cold` |
| `sigma_a_floor_abs` | `1e-8` | Absolute σ_a floor (Σ_a conditioning) |
| `sigma_a_floor_rel` | `1e-5` | Relative σ_a floor (× $|a_0|$) |
| `sigma_a_ceiling_factor` | `1e5` | Ceiling = factor × max finite σ_a |
| `sigma_c_floor` | `1e-9` | σ_c floor for free rows |
| `solve_all(feasibility_check=)` | `"all"` | Argument of `solve_all`, not a constructor parameter. Feasibility pre-check scope: `"all"` (one probe per period) or `"none"` (sweep points; a failed period still raises); `"first"` and the old bool `feasibility_check_first_only` are gone (ADR 0042) |
| `verbose_first_solve` | `True` | Ask the backend for its own log of the first period's solve (serial solve) |
| `n_workers` | `1` | Periods probed and solved concurrently (cold only); the driver sets `N_WORKERS = os.cpu_count()` |
| `qp_variables` | `"flows"` | Decision variables of the period QP: the flows, or their defects `a − a_0` (same QP, `q = 0`); the driver's `QP_VARIABLES` is `"defects"` |

`ClarabelBackend` parameters: `tol_feas` / `tol_gap_abs` / `tol_gap_rel` (`1e-8`), `reduced_tol_*` (Clarabel's defaults unless set), `max_iter` (`200`; the driver's `MAX_ITER` is `2000`), `direct_solve_method` (`"faer"`), `max_threads` (`1`), `capture_iter_history` (`True`).

#### Use case

Run Stage 4 on the toy case in [tutorial/04_solver.ipynb](../src/pysuto/tutorial/04_solver.ipynb).

#### Glossary detail, Stage 4

The glossary entries below keep their definition and their aliases to avoid. This section holds the full text they had before; each block names its entry.

**Parallel solve**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Parallel solve**:
> Solving the periods of a cold-prior run concurrently, each period its own QP; possible only
> because a cold period's prior does not depend on another period's optimum. A warm run is a
> chain and is solved in period order.
> _Avoid_: "multi-threaded solve" — that is the solver's threading *inside* one period's QP (its
> linear algebra), which a parallel solve pins to one thread.
>
> *Example.* "Why does the 1995 file's mass balance look enormous?" — "That's a
> pre-fix block run: the file holds 1995's **period slice** plus **cross-year
> placeholders** for every other year, and the residual was taken over the full
> `G`. Re-solve and it stores only the period slice; the **complete `a***`** is
> reassembled by the diag when it needs the whole picture."

**a_prior (warm-start prior)**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **a_prior (warm-start prior)** — the per-**period** prior mean the optimiser uses for year `t`.
> Under `prior_mode='warm'` (the `Solver` class default) it is genuinely the *previous* year's
> posterior, aligned by **edge template** `(from_id, to_id, valuation)` since each year occupies
> disjoint global columns: `a_prior(t0) = f_agg` (the cold seed), `a_prior(t) = a_post(t-1)` for
> `t > 0`. Under `prior_mode='cold'` every year uses its own cold seed `f_agg(t)`. Stored as the
> per-period cold seed on `ConstraintRun.timestep_data[t]["a_prior"]` and as the *actually-used*
> prior on `OptimResult.a_prior`; `ConstraintRun.a_prior_global` reconstructs the global cold
> prior. _Avoid_: reading `ConstraintRun.a0` as the per-year prior — `a0` is the legacy global
> (time-replicated) cold vector; the year's prior is `a_prior`. _Historical bug_: the block path
> once wrote the carry-forward into the same disjoint columns it read, so the warm chain was a
> silent no-op and every year fell back to `f_agg` — see [ADR 0009](adr/0009-warm-start-prior-chain.md).

**Residual**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Residual** — the per-**constraint-row** delta `G·a − c` (`OptimResult.residuals`; its hard
> mass-balance maximum per year is the middle panel of SD-R-11). Its feasible range depends on the
> row's cone: **≈ 0 on hard *equalities*** (`zero` cone — mass balance); **`≤ 0` on hard *bounds***
> (`nonneg` cone — ratio/flow bounds, since feasibility is `G·a ≤ c`), where a *negative* residual
> is healthy **slack** (distance below the limit), not a violation, and only a *positive* residual
> is a breach; **`= ε`, the slack, on a soft (free) row** (see `docs/architecture.md` Stage 4, "Standard form" → Cone order: `Gx − ε = c`).
> _Avoid_: (1) reading a large-negative residual on a `nonneg` bound as a violation — it is slack
> (a misread that once produced a phantom "inverted bound" diagnosis); (2) conflating with
> **defect** — residual is per-constraint, defect is per-edge.

**Forced vs discretionary deletion**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Forced vs discretionary deletion** — the split of a node's prior-to-posterior mass loss into the
> part no optimiser could have avoided and the part it chose. Copper leaves a mine node only as
> domestic smelter intake `U` or as export mass `X` times a grade the hard cone caps at 0.35, so
> `cap = 0.35·X + U` bounds its discharge and `Σ max(0, P − cap)` is **forced** — arithmetic, not
> optimisation. The remainder is **discretionary**: the solver could have kept it and priced the
> disagreement elsewhere. The distinction says which lever can move a number: σ, anchor placement
> and pricing reach only the discretionary part, while the forced part moves only when `X`, `U` or
> `P` themselves change ([ADR 0029](adr/0029-the-concentrate-grade-stays-global.md)). _Avoid_:
> reading a forced term as a defect to be relieved. Where `P` is phantom — production the source
> never had — deleting it is the *correct* answer, and the repair is to remove the input rather than
> loosen the constraint
> ([ADR 0035](adr/0035-the-yearbooks-label-is-not-on-the-row-it-owns.md)). _Avoid_ also: reading
> a forced term larger than the realised deletion as a contradiction — `cap` is evaluated at
> prior-level intake, and a region that substitutes scrap or imported anode escapes a ceiling the
> formula cannot see.

**Constraint table (prior/post)**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Constraint table (prior/post)** — the harmonised **row-level, all-rows** per-`constraint_id`
> table pair (hard + soft rows; `type` / `sense` columns carry the family for filtering):
> `constraints_prior_<year>.csv` at the constraint stage (`ConstStatsDiagnostic`) and
> `constraints_<year>.csv` at the solver stage (`SolverStatsDiagnostic`). **`lhs_prior` = G·a0**
> means the same thing in *both* files; **`lhs_post` = c_star = G·a*** exists only post-solve.
> `residual` / `z = residual/σ_c` (ranked by **objective share** in SD-R-13 / 14 / 15b-c) are the
> *stage's achieved* residual — `lhs_prior − c` at the constraint stage, `lhs_post − c` (= the slack
> `eps` on a soft row) at the solver stage; `z` is NaN on hard rows (`σ_c = 0`). _Avoid_: reading a
> large `|z|` as a failure on its own — a soft row is *meant* to absorb disagreement; `|z|` just
> measures it in units of the row's tolerance `σ_c`. _Avoid_: reading the solver-stage `residual`
> on a **hard** row as the **residual** above — `c_star = c` there by construction, so the column
> is identically 0; the hard residual lives in `OptimResult.residuals` (frame column
> `resid_solver`). _Avoid_: an unqualified `lhs` column — the files qualify every LHS as
> `lhs_prior` / `lhs_post` so no column changes meaning between stages. _Note_: `ConstDiagnostic`
> plots the same prior quantity under the opposite sign (`eps0 = c − G·a0`); the table follows
> the reserved `G·a − c` sign.

**Objective share**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Objective share** — one item's `z²` (an edge's defect or a soft row's residual, standardised)
> divided by the `Σz²` of its family or of the whole run. The QP objective *is* `Σz²` over edges plus
> soft rows, so the share is what that item cost the optimiser. Ranked per **identity** over all
> years by `SolverDiag` (SD-R-13 by sector key, SD-R-14 top identities, SD-R-15 z profiles) and
> tabulated in `ranking_<family>.csv`; three families: edge defects, sum-row residuals (`type =
> sum`, all soft), soft-ratio residuals (`type = ratio`, free cone). _Avoid_: "deviation" (say defect
> or residual), "violation" (a soft row is meant to absorb disagreement — see Constraint table),
> "fit". _Region share_: a region's share of one family's `Σz²`, tabulated per region and per
> region-year in `objective_share_by_region.csv`; an edge belongs to its region when both ends sit
> in it, a sum row to its `region`, a pair-scoped soft ratio to its numerator's region, and
> anything spanning regions (bilateral legs) or naming none (world totals) to the bucket `multi`.

**Identity (cross-year)**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Identity (cross-year)** — the key under which one edge or one constraint row recurs across
> years. For an edge: the **edge template** `(from_id, to_id, valuation)`. For a row: the tuple of
> its `row_meta` columns other than `constraint_id` and `time` — `(variant, sector, region,
> valuation, …)` node-scoped, the `num_*`/`denom_*` endpoints pair-scoped, `(variant, region,
> valuation)` set-scoped, with `sense`/`bound_dir` telling a pair's soft row from its two band
> rows. Always read from columns; a row whose columns do not single it out within its year has no
> identity (it still counts in every total). _Avoid_: stripping the year off a `constraint_id` (see
> `constraint_id`: a key for reading, never parsing).

**Retention**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Retention** — the share `Σa* / Σa_prior` computed **by category**: the posterior sum over the
> edges of one category in one year (a sector, a node, a region, an SB class — always name the
> category) divided by the prior sum over the same edges. It is a dimensionless share of the prior
> mass: `1` keeps the prior scale, `< 1` is deletion, `> 1` inflation (second row of SD-R-08 / 09 /
> 10, per sector and year). The complement of the deletion in **forced vs discretionary deletion**.
> _Avoid_: "adaptation", "recovery", "survival"; a retention quoted without its category and year;
> a retention on a signed category whose prior sum is near zero (`MESC` / `USC` / `PSC`) — the ratio
> is undefined there, so quote the mass moved `Σ|a* − a_prior|` beside it.

**Cone**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Cone** (`zero` / `nonneg` / `free`):
> The Clarabel cone a constraint row lands in, derived inside `pysuto.clarabel_backend` from the
> QP's row groups (`pysuto.solver.row_layout` orders them from `sense` and `is_hard`; ADR 0042)
> and not stored in `row_meta` — `zero` (hard `eq`, and a soft `eq` too: its slack `ε` is a free
> variable of the QP), `nonneg` (hard `le`). The legacy name `free` is the soft `eq`. Const runs persisted before #11 stored it as `cone_type`; the loader maps
> it to `sense`.

**Sink**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Sink** — an edge family whose σ makes its penalty negligible against the mass it can absorb, so
> a contradiction elsewhere discharges into it instead of surfacing as `z` where it belongs. The
> signed stock legs are the structural candidates (they alone may be negative, ADR 0039); the
> exchange-location `MESC` legs at σ ≈ 553 kt were one (Round X). Relates to **forced vs
> discretionary deletion**: a sink converts a forced contradiction into a discretionary move.
> _Avoid_: "slack" (the soft-row residual), "buffer".

**Defect**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Defect** — the per-**edge** prior departure `a* − a_prior` (solved value minus the warm-start
> prior, `a_prior` above). Plotted by SD-T-07 (the a-space rocket, `a_prior(t)` vs `a*`), ranked
> by **objective share** in SD-R-13 / SD-R-14 / SD-R-15a (the edge-defect family), and tabulated
> row-level by `SolverStatsDiagnostic` (`edges_<year>.csv`, columns `defect` / `defect_abs` / `z`).
> Its standardised form is `z = (a*−a_prior)/σ_a`. _Avoid_: calling the defect a "residual" — that
> term is reserved (below).


---

### Stage 5 — Posterior uncertainty propagation

**What this stage does.** Loads a persisted Stage-4 `OptimRun` plus its upstream `ConstraintRun`, then computes the marginal posterior standard deviation `σ_a*(t)` and a Gaussian `(1-α)` confidence interval per edge × timestep. Each year of the `ConstraintRun`'s per-period partition (`G_blocks`; a run without it raises) is reduced to the QP's active set at `a*` and the contraction is read off a QR of that year's whitened system (`_compute_via_blocks`, `_leverage`; [Marginal diagonal](#marginal-diagonal)); `S = G·diag(σ²_a)·Gᵀ + diag(σ²_c(t))` is never formed.

**Cold opt runs only.** The years are separate QPs and their posteriors add without covarying. An opt run solved with `prior_mode = warm` chains them (`a_prior(t) = a*(t−1)`), which Stage 5 does not model: it linearises each year with the prior fixed. It logs a warning on such a run and records `prior_mode` in the sidecar's `stage_config`; the reference chain is cold.

**Inputs:** `opt_run_id` (Stage 4 output; the upstream `const_run_id` is discovered from its sidecar).
**Outputs:** `UncPropResult` with `per_timestep[t] = {sigma_a_star, lower, upper, diagnostics}`. These arrays are **year-local** (over `cols_t`, matching the Stage-4 `OptimResult`) and carry `cols_t`.

> Uses the same σ-conditioning rules the solver applied (see [`_conditioning.py`](../src/pysuto/_conditioning.py)) so that the posterior is mathematically consistent with the `a*` returned by Stage 4.

#### Math

##### Closed-form posterior covariance (Approach A)

For the reconciliation QP with prior $a_0 \sim \mathcal{N}(\bar a_0, \Sigma_a)$, soft RHS $c_s \sim \mathcal{N}(\bar c_s, \Sigma_c)$, hard equalities $G_h a = c_h$, and slack ε on the soft block:

$$\min_{a,\varepsilon}\ \tfrac{1}{2}(a - a_0)^\top \Sigma_a^{-1}(a - a_0) + \tfrac{1}{2}\varepsilon^\top \Sigma_c^{-1}\varepsilon \quad \text{s.t.}\quad G_h a = c_h,\ G_s a + \varepsilon = c_s.$$

Stacking $G = [G_h;\, G_s]$ and $\Sigma_c^{\mathrm{full}} = \mathrm{blkdiag}(\mathbf{0}_{m_h},\, \Sigma_c)$, the interior KKT system collapses to

$$\boxed{\;\Sigma_{a^*} = \Sigma_a - \Sigma_a G^\top S^{-1} G \Sigma_a, \qquad S = G\Sigma_a G^\top + \Sigma_c^{\mathrm{full}}.\;}$$

##### Marginal diagonal

For diagonal $\Sigma_a = \mathrm{diag}(\sigma^2_a)$, the per-edge variance reduces to

$$\boxed{\;\sigma^2_{a^*,i}(t) = \sigma^2_{a,i}\,\big(1 - \sigma^2_{a,i}\,\|L(t)^{-1} g_i\|_2^2\big), \quad S(t) = L(t)L(t)^\top, \quad g_i = \text{col}_i(G).\;}$$

$S(t)$ is never formed. With $M = [\,G_t\,\mathrm{diag}(\sigma_a),\ \mathrm{diag}(\sigma_c)\,]$, $S = MM^\top$, and the contraction $\kappa_i = \sigma^2_{a,i}\, g_i^\top S^{-1} g_i$ is the leverage of edge $i$: the squared norm of row $i$ of $Q$ in $M^\top = QR$ (dense Householder QR, rows sorted by norm). So $\sigma^2_{a^*,i} = \sigma^2_{a,i}(1 - \kappa_i)$ with $\kappa_i \in [0, 1]$ to rounding. Forming $S$ squares $M$'s condition number: with σ_a spanning 1e-8 … 1e6 kt, two independent hard rows that differ only on low-σ edges are parallel in $S$ to working precision, and an LU of $S$ put κ outside [0, 1] on 331 edges of 1995 and 368 of 2016 in `uncprop_20261014_140949_fcf53674` (#27). The QR is dense, (live edges + soft rows) × kept rows per year: on copper 2007, the largest year (21,131 live edges, 18,480 rows), it takes ≈ 2 min on 8 cores and ≈ 7 GB on top of the loaded runs (`const_20261013_114216_1ad3239e`). $\sigma^2_{a^*} = \sigma^2_a(1-\kappa)$ resolves a posterior variance only down to ≈ $\sigma^2_a\,\varepsilon$, so a nearly determined edge reads $\sigma_{a^*} \approx 0$ to within ≈ $10^{-8}\,\sigma_a$.

Posterior std and confidence interval:

$$\sigma_{a^*,i}(t) = \sqrt{\max(\sigma^2_{a^*,i}(t), 0)}, \qquad [\text{lower}_i,\, \text{upper}_i] = \big[a^*_i(t) - z\,\sigma_{a^*,i}(t),\; a^*_i(t) + z\,\sigma_{a^*,i}(t)\big], \qquad z = \Phi^{-1}(1 - \alpha/2),$$

with `lower` clipped at the bound row's own `l` on edges that carry an ADR 0039 lower-bound row and nowhere else (block path): the `ConstraintRun`'s `l` / `u` hold the prior box and the zero pin, which reach the feasibility probe, not the QP.

##### Hard rows and the `Σ_c^full` block structure

σ_c on hard rows is forced to **exact zero** before the per-t computation (see [`condition_sigma_c(hard_mask=...)`](../src/pysuto/_conditioning.py)); soft rows receive a numerical floor only. This preserves the $\Sigma_c^{\mathrm{full}} = \mathrm{blkdiag}(\mathbf{0}_h,\, \Sigma_{c,s})$ structure the derivation relies on. Hard-balanced edges then produce **zero-width** CIs (`lower = a* = upper`) — this is the correct, expected behaviour.

##### The active set (block path, #27)

The formula holds for equality constraints; the QP also has hard **inequalities** (`nonneg` cone: the ADR 0039 lower bounds and the ratio bands). The posterior is the QP's linearisation at its active set at `a*(t)` ([ADR 0044](adr/0044-the-posterior-is-the-qps-linearisation-at-its-active-set.md)):

- a hard inequality row **binds** when `G·a* − c ≥ −bind_tol` (`1e-6` kt), read from `OptimResult.residuals`;
- a binding lower bound **pins** its edge (**bound-active**): `σ²_a → 0` in `S(t)` and `σ_a* = 0` (a zero-width interval at the bound), instead of adding a row; the mask is persisted as `bound_active`;
- a binding band enters as a hard equality; slack bounds and bands drop out;
- rows left without a live (unpinned) edge are dropped, and hard rows linearly dependent on the others over the live edges (closed balance loops, a band's two rows at `0 = 0`) are dropped after a pivoted Cholesky of the unit-diagonal Gram `G_h·G_hᵀ` (`rank_tol`; unweighted, since a σ²-weighted Gram makes rows that differ only on low-σ edges look parallel), so `S(t)` stays non-singular with every kept hard row at exactly zero σ_c.

Per year the sidecar records `n_bound_active`, `n_bands_binding`, `n_hard_dependent` and `n_rows_kept`.

##### Totals and ratios (#81)

The marginals do not give the σ of a total: the constraints correlate the edges, so adding the σ_a* of a node's legs in quadrature misstates the node total. The exact variance of an aggregate $w^\top a^*$ comes from the same QR. With $u = \mathrm{diag}(\sigma_a)\,w$ over the live edges, padded with zeros to $v$ over $M^\top$'s soft rows,

$$\mathrm{var}(w^\top a^*) = w^\top\Sigma_a w - (G\Sigma_a w)^\top S^{-1} (G\Sigma_a w) = \lVert u\rVert^2 - \lVert Q^\top v\rVert^2 .$$

`compute(..., aggregates=AggregateSpec(names, W))` returns the K×K prior and posterior covariance of K totals, their value at `a*` and the quadrature σ, in memory. `W` carries no valuation: each column keeps to one valuation, and its value and σ are in that valuation's unit (kt CC for `CC`); a column that mixes Mass and CC edges returns a σ in no unit, and nothing checks it. The years are separate QPs, so a total over several years sums the years' contributions, and totals of different years never covary. A ratio or index of totals gets its σ by the delta method on that covariance. `draws={year: k}` returns posterior deviations $\mathrm{diag}(\sigma_a)\,(I - QQ^\top)\,\xi$ from the caller's generator. They satisfy every kept hard row and pin exactly, so a function through the Leontief inverse gets its delta-method σ by finite differences along them. `force_pinned` / `force_free` override the slack rule edge by edge, for the degenerate bounds #99 found at `bind_tol`. [`scripts/report/table3_intervals.py`](../scripts/report/table3_intervals.py) puts Birge-scaled intervals beside Table 3 and SI S18b.

##### Sanity invariants (checked in tests)

- **Monotonicity:** $\sigma^2_{a^*,i} \leq \sigma^2_{a,i}$ for every $i$.
- **PSD-ness:** $\sigma^2_{a^*,i} \geq 0$ (clipped after the variance update).
- **Hard residual:** $\|G_h a^*\|_\infty < 10^{-6}$ — verified via the `max_resid_hard` diagnostic.
- **Byron (1978) limit:** as $\Sigma_c \to 0$, the formula collapses to the classical hard-only reduction.

#### Classes

- **`BaseUncProp`** ([uncprop.py](../src/pysuto/uncprop.py)) — ABC with a single abstract method `compute(opt_run_id, alpha) → UncPropResult`. Future concretes (`MCUncProp`, `BoundedKalmanUncProp`, `KKTSensitivityUncProp`) slot in as siblings.
- **`KalmanUncProp`** ([uncprop.py](../src/pysuto/uncprop.py)) — closed-form GLS concrete: a QR of the whitened system per year on the block path. Cross-stage discovery (`opt_run_id` → `const_run_id` via the opt sidecar) happens inside `.compute()`; persistence via `.persist(result, opt_run_id, label)`.
- **`UncPropResult` / `UncPropTimestep`** ([uncprop.py](../src/pysuto/uncprop.py)) — return dataclasses. `UncPropTimestep` holds `(sigma_a_star, lower, upper, diagnostics, cols_t)` per t (year-local over `cols_t`); `a*` is **not** stored — readers join on the upstream `opt_run_id`.
- **`UncPropRunRegistry`** ([registry/uncprop.py](../src/pysuto/registry/uncprop.py)) — Stage-5 output registry; harmonized run-id (`uncprop_<ts>_<code>`), per-t `.npz`, diagnostics in sidecar.
- **`condition_prior` / `condition_sigma_c`** ([_conditioning.py](../src/pysuto/_conditioning.py)) — shared helpers also used by `Solver`, guaranteeing that uncprop applies the same σ_a floor / ceiling rules the solver applied. Their default values live there only: Stage 4 records the floors it used in the opt sidecar's `stage_config`, Stage 5 applies the recorded ones (an older run falls back to the defaults with a logged warning).

#### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `alpha` (passed to `.compute()`) | `0.05` | CI level — Gaussian quantile $z = \Phi^{-1}(1-\alpha/2)$ |
| `sigma_a_floor_abs` | `1e-8` | Same as `Solver` (matched for prior consistency) |
| `sigma_a_floor_rel` | `1e-5` | Same as `Solver` |
| `sigma_a_ceiling_factor` | `1e5` | Same as `Solver` |
| `sigma_c_floor` | `1e-9` | Floor on soft-row σ_c; hard rows are forced to exact zero regardless |
| `bind_tol` | `1e-6` | kt — a hard inequality row binds at `a*` when `G·a* − c ≥ −bind_tol` (`stages.BIND_TOL`; Stage 5 has no `--set` or case knobs) |
| `rank_tol` | `1e-10` | Pivot tolerance of the scaled hard-block pivoted Cholesky that drops dependent hard rows (driver knob `RANK_TOL`) |
| `exclude_scale_totals` | `False` | Ablation (#87): leaves the scale totals and scrap level anchors (`row_meta.type == "sum"`, `variant` in `SCALE_TOTAL_VARIANTS`) out of `S(t)`; `a*` is untouched and the count lands in `n_scale_totals_left_out` (driver knob `EXCLUDE_SCALE_TOTALS`, recorded in `stage_config`) |

#### Use case

Run Stage 5 on the toy case in [tutorial/05_posterior.ipynb](../src/pysuto/tutorial/05_posterior.ipynb).

#### Diagnostics

`UncPropDiag` ([diag/uncprop.py](../src/pysuto/diag/uncprop.py)) renders the Stage-5 portfolio from the all-years edge frame of `UncPropStatsDiagnostic`. It is built live (`UncPropDiag(result, opt_results, const_run, acc=…, imp_result=…)`) or from disk (`UncPropDiag.from_uncprop_id(uncprop_run_id)`, which follows the sidecar to the opt, const and imputation runs). The driver renders it after `persist()`; `scripts/uncprop_diag.py <uncprop_run_id> [--regions …]` re-renders a persisted run (`--data-dir`, default `data`) into `<data-dir>/proc/uncprop/<id>/diag/`. Vocabulary is GLOSSARY.md's: **posterior contraction** `κ`, **prior class**, and the κ–z quadrants.

- **UD-R-01** — κ against the signed defect `z`, hexbin weighted by `max(|a_prior|, |a*|)` (kt), one panel per prior class × valuation, bound-active edges circled.
- **UD-R-02** — prior `a_prior ± z_α·σ_a` against the posterior `[lower, upper]` over the years, for the top-12 identities per valuation.
- **UD-R-03** — QQ plot of the **standardised shift** `z/√κ` against N(0, 1), one panel per prior class × valuation, over the edge-years that are not bound-active, have `κ ≥ 0.01` and are not flat; the N(0, R) line uses the run's **Birge ratio**.
- `contraction_summary.csv` — edge count and Σ weight per year × valuation × prior class × quadrant.
- `calibration.csv` — per valuation × prior class: the share with `|z/√κ| > 3` (N(0, 1) expects 0.27 %), unweighted and weighted by `max(|a_prior|, |a*|)`, the quantiles of `|z/√κ|`, and the same shares beyond `3·√R`.
- `birge.csv` — the Birge ratio `R = Σz²/dof` per year and over the run: `Σz²` is the QP objective at `a*` (SolverDiag's objective decomposition), `dof` = Stage 5's `n_rows_kept + n_scale_totals_left_out + n_bound_active` minus `n_bound_active_at_prior` (pins whose prior already sits on the bound, within `bind_tol`: zero by construction, no degree of freedom); it carries the half-widths `z_α` and `z_α·√R` of the raw and the Birge-scaled interval. `UncPropStatsDiagnostic.birge_interval(frame)` gives the scaled interval per edge-year; the stored `lower` / `upper` are never rescaled.
- A scale-total ablation run (`exclude_scale_totals`) gets no UD-R-03, no `calibration.csv` and no Birge interval: its `z` comes from the full fit, its `κ` and `σ_a*` from the reduced `S(t)`. Its Birge ratio is the full fit's.

Before the active set (#27) no copper chain could run Stage 5: every hard inequality row entered `S_t` as an equality and made it singular.

#### Caveats

The closed form is the **Laplace approximation** of the true posterior — acceptable for tightly-distributed flows but increasingly approximate for wide log-normal pedigree tails.

#### Glossary detail, Stage 5

The glossary entries below keep their definition and their aliases to avoid. This section holds the full text they had before; each block names its entry.

**Posterior contraction (`κ`)**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Posterior contraction (`κ`)** — the share of an edge's prior variance the constraints remove in
> one year, `κ = 1 − σ_a*² / σ_a²`: `0` leaves the prior uncertainty untouched, `1` pins the edge. It
> depends on `G`, `σ_a` and `σ_c` and, through the QP's active set at `a*` (which rows bind, which
> lower bounds pin their edge), on `a*`; given that active set it does not depend on `c`, so it measures how informative the
> constraints are, not whether they agree with the prior — that is the **defect**'s `z`.
> _Avoid_: "shrinkage" (in statistics, the pull of an estimate toward the prior mean — our defect),
> "variance reduction", "information gain".

**Prior-carried / confirmed / overruled / unexplained move**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Prior-carried / confirmed / overruled / unexplained move** — the four quadrants of an edge-year
> in posterior contraction `κ` and standardised defect `z`, cut at `κ = 0.5` and `|z| = 3`:
> prior-carried (`κ < 0.5`, `|z| ≤ 3`), confirmed (`κ ≥ 0.5`, `|z| ≤ 3`), overruled (`κ ≥ 0.5`,
> `|z| > 3`, a prior–data conflict) and unexplained move (`κ < 0.5`, `|z| > 3`, moved uninformed).
> _Avoid_: "unidentified" for prior-carried — the edge has a value, only no new information.

**Standardised shift (`z/√κ`)**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Standardised shift (`z/√κ`)** — an edge-year's defect `z` over `√κ`: N(0, 1) under the
> active-set linearisation when σ is right, read only on edge-years that are not bound-active, have
> `κ ≥ 0.01` and are not flat. Its **tail share** is the share beyond `|z/√κ| = 3` (0.27 % under
> N(0, 1)).
> _Avoid_: "pull", "normalised residual".


### Stage 6 — PSUT/PIOT builder

Downstream of the solver: **materialise** the solved edge vector `a*` (`OptimResult.F_opt`) as numeric **physical supply-use tables (PSUT)** — supply `S`, use `U`, final use `Y`, plus boundary blocks `E`/`O`/`P` — per period and per valuation (`CC`, `Mass`), reusing the existing supply/use masks and the SB nodes' boundary roles. From each PSUT it builds the symmetric **physical input-output table (PIOT)**: the market-share (industry-by-industry) construct, with `Z`, `A = Z·x̂⁻¹` and `L = (I−A)⁻¹` over the region × process network (with one producer per commodity, as in copper, it coincides with the product-by-product one); the PIOLab / Suh (2004) product-by-product construct, by-products endogenised as negative inputs with `A = B·D`, is #20. It ships the classical analytics: footprints / embodied flows, embodied bilateral trade, ew-MFA headline indicators, and multipliers / SDA. v1 is point-estimate only (Stage-5 σ propagation deferred). Rationale and the deliberate *no pySUT / pymrio dependency* are recorded in [ADR 0007](adr/0007-downstream-psut-piot-follows-piolab-suh-no-external-io-lib.md).

**What exists — Slices 1–6 ([#18](https://github.com/florihu/pySUTO/issues/18), [#19](https://github.com/florihu/pySUTO/issues/19), [#24](https://github.com/florihu/pySUTO/issues/24), the supply-use Sankey of [ADR 0037 §1](adr/0037-psut-piot-two-classes-piot-is-a-construct-family.md), [#25](https://github.com/florihu/pySUTO/issues/25), [#21](https://github.com/florihu/pySUTO/issues/21) and [#22](https://github.com/florihu/pySUTO/issues/22)).** [psut.py](../src/pysuto/psut.py) ships the **tidy-canonical `PSUT`** ([ADR 0037](adr/0037-psut-piot-two-classes-piot-is-a-construct-family.md)): one row per directed edge per (period, valuation), every row classified into its **block** by the SB node's boundary role — `S` Process→Flow, `U` Flow→Process, `Y` Flow→SB·final_demand, `E` SB·nature→Process, `O` SB·secondary→Process, `P` Process→SB·secondary, and `W` Process→SB·nature (empty in copper); an edge outside that taxonomy raises — with the block matrices as pivot *views* labelled from the accounting structure's node index (which carries each SB node's `role`), identical in every period. The table carries **every** valuation; `primary` (default the content valuation, copper `CC`) is the one the views default to. The product-by-product construct (#20) is the one open member of the family; the decisions behind the order are in ADR 0037.

- **`PSUT.from_opt_run(opt_run_id, timesteps=None, primary=None)`** — loads every timestep and every valuation of a persisted run, resolves the `AccountingStructure` through provenance (opt sidecar → `const_run_id` → const sidecar → `acc` id) and places the year-local `F_opt` on global edge ids via `cols_t`.
- **`PSUT.from_results(results, acc, primary=None)`** — in-memory handoff; `results` is the solver's `dict[timestep, OptimResult]`, typed by the `OptimResultsLike` protocol (`timestep`, `F_opt`, `cols_t`), so the PSUT never depends on `OptimResult` internals.
- **Views and selection** — `S`, `U`, `Y`, `E`, `O`, `P`, `W` (or `block(name)`), each taking `(period=None, valuation=None)` and defaulting to the latest period and the primary valuation; boundary levels are named by role (`nature`, `secondary`, `final_demand`); `realized_conc()` is the per-edge `CC/Mass` ratio joined on the edge template; `periods()`, `latest()`, `at(period)`. Time is a column of the table and is never aggregated.
- **Aggregation** — `aggregate(region=<map|'World'>, sector=<map>, keep_crossings=False)` is a pushforward on the table: the maps apply to both facets, rows are summed over every key column (`period`, `valuation`, `block` and the entity kinds never merge; a sector map may only merge one entity kind), and `keep_crossings=True` stamps `region_crossed` before the relabel so the realized traded share survives a World collapse. Chainable; the steps are recorded in the sidecar.
- **Supply-use Sankey** — `psut.sankey(period=None, valuation=None, **spec)` returns a Plotly figure: processes and commodities as nodes, the S/U rows as links, boundary blocks as their own nodes, border-crossing use / final-use flows as separate traded links; `min_val`, `split_traded`, `title`, `unit`, `layout` fill a `SankeySpec`; nodes are placed by longest-path depth so residual and input nodes sit beside their process. The plotting code lives in [diag/psut.py](../src/pysuto/diag/psut.py) (`sankey_tables` is the pure-pandas half), imported lazily — the data class never imports Plotly at module level.
- **PIOT** — `psut.to_piot('market_share', level='Z')` returns a [`PIOT`](../src/pysuto/piot.py), a derived view that is never persisted: the **process network** under the market-share construct (`D = S·q̂⁻¹`; every use row `f → q` becomes edges `p → q` carrying `f` at `D[p,f]·U[f,q]`, every final-use row `p → d` likewise, boundary rows carried as they are, `region_crossed` travelling along), with `Z()` (process × process), `D()` (the shares) and the process-network `sankey()` through the same renderer. `suh_product` raises `NotImplementedError` until #20.
- **Levels `A` and `L`, footprints, ew-MFA** — `to_piot(level='L')` unlocks the industry-by-industry calculus on that network: `x()` (`Z·1 + y`), `y()`, `A()` (`Z·x̂⁻¹`), `L()` (`(I−A)⁻¹`, so `L·y = x`, except for output delivered to a process with no useful output), `intensity(extension)` for the `E` (extraction) or `O` (secondary) inflow per unit of output, `footprint(extension, by='fd'|'region'|None)` (`e·L·ŷ`, summing to the extension's total), `embodied_trade(extension)` (extraction region × consuming region), `ew_mfa()` per region (`DE`, `imports`, `exports`, `PTB`, `DMI`, `DMC`, `RMC`), `multipliers()` (column sums of `L`) and `sda(period_a, period_b)` (two-factor polar decomposition of `Δ(L·y)`, which is `Δx` where `L·y = x`). With one producer per commodity, as in copper, these equal what the Suh construct would give.
- **Persistence** — `PsutRunRegistry` writes `data/proc/psut/<psut_run_id>/timesteps/t_<period>.parquet` + `nodes.parquet` with `parents.upstream.opt_run_id`, and records `valuations`, `primary` and the blocks in the sidecar; `scripts/psut.py <opt_run_id> [--primary CC] [--sankey 2020 2024] [--figs-dir figs/psut] [--no-png]` is the one-command CLI, the flag writing both Sankeys per period — `sankey_supply_use_World_<period>_<valuation>.html` and `sankey_process_network_…` — plus a PNG when kaleido can render, under `figs/psut/`.

Run Stage 6 on the toy case in [tutorial/06_psut_piot.ipynb](../src/pysuto/tutorial/06_psut_piot.ipynb); [tutorial/07_scoreboard.ipynb](../src/pysuto/tutorial/07_scoreboard.ipynb) scores the whole chain.

#### Glossary detail, Stage 6

The glossary entries below keep their definition and their aliases to avoid. This section holds the full text they had before; each block names its entry.

**Traded (realized) vs tradable; `region_crossed`.**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Traded (realized) vs tradable; `region_crossed`.** **Tradable** is a *sector capacity* — the four ICSG-fanned
> sectors (Cathode, Concentrate, Anode, Matte) whose Use edges *may* cross regions (above). **Traded / inter-regional**
> is a *realized* per-edge property — the Use edge actually crossed a border (`region_from != region_to`) — captured
> as the boolean **`region_crossed`**, stamped before a region collapse (`keep_crossings`) so it survives aggregation.
> Supply edges are always same-region; **Use** edges *and* **final-use (`Y`) edges** can be `region_crossed` — in
> copper 6,486 of 6,627 `Y` edges cross regions in 2020 (cathode into foreign final use), as many as the `U`
> edges do. A Sankey's **traded-flow share** is this *realized* split. _Avoid_: reading "traded share" as the
> *tradable-sector* fraction — it is the realized inter-regional fraction.

**PSUT (physical supply-use table)**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **PSUT (physical supply-use table)** — the **materialised numeric** supply-use flows, derived from the solved edge vector `a*`; one table
> holds every period and every valuation, and a view selects one of each. **Shape-agnostic**: its canonical form is a **tidy
> edge table** (one row per directed edge, carrying its `val`), and the block matrices — supply `S` (useful
> output), use `U`, final use `Y`, plus boundary blocks nature-input `E`, SEM-input `O`, SEM-output `P`, nature-output `W` — are
> **derived pivot views** of that same PSUT. The valueless Process+Flow **SUT** is the *graph* (the
> `AccountingStructure` topology); a PSUT is that graph **carrying solved values**. _Avoid_: calling the
> **valueless** graph a "PSUT" (the bare graph is the **SUT**); "SUT" for a numeric realisation (long or matrix);
> reading "PSUT" as *matrices only* (the matrices are one view; the tidy table is the canonical other).

**Block**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Block** — the PSUT quadrant a directed edge belongs to, read off its endpoint entity kinds (and, for
> the boundary blocks, the `SB` node's boundary role): `S` Process→Flow, `U` Flow→Process, `Y` Flow→SB·FD,
> `E` SB·Nature→Process, `O` SB·SEM→Process, `P` Process→SB·SEM, `W` Process→SB·Nature (empty in copper). Every admissible edge lands in exactly one
> block; the tidy table carries it as the `block` column and each matrix view is one block's pivot. _Avoid_:
> "mask" for a block — the `supply`/`use` **masks** are rule-derived edge selections that straddle blocks (a
> loaded structure's `supply` mask also holds `P`, its `use` mask `E` and `O`).

**Aggregation (pushforward)**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Aggregation (pushforward)** — region/sector aggregation of a PSUT as a *pushforward* of the tidy table: one
> concordance map per dimension (`region`, `sector`) applied **symmetrically to both facets** (`from_`, `to_`),
> then a sum over every key column — `period`, `valuation`, `block` and the entity kinds stay in the key, so
> nothing merges across them and **time is never summed**; a sector map may only merge sectors of one entity kind.
> `region_crossed` is stamped **before** the relabel, at the finest resolution ever seen (a two-step collapse keeps
> the country-level stamp), so the realized traded share survives a World collapse. `edge_id` does not survive.
> _Avoid_: summing `period`; reading `region_crossed` at the aggregated resolution (it is the root-resolution
> crossing); a one-label `sector` collapse (it would merge Process with Flow).

**Supply-use Sankey**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Supply-use Sankey** — the bipartite Sankey drawn *directly* from a PSUT: processes **and** commodities are
> nodes, the `S`/`U` rows are links, the boundary blocks attach as links to their own `SB` nodes (inputs `E`/`O`
> on the left, outputs `P`/`W`/`Y` on the right), and a border-crossing `U`/`Y` flow is its own **traded** link
> (from `region_crossed`, or from distinct regions). Needs no construct. In copper it carries the same numbers as
> the process-network Sankey — `D` is the identity (ADR 0037 §1) — with the commodity nodes shown rather than
> contracted into edge labels. _Avoid_: "process network" for it (that is the PIOT's contracted view);
> reading a node's total as throughput of anything but that (period, valuation).

**Construct**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Construct** — the SUT→PIOT transformation choice, on **two orthogonal axes**. **(1) Table shape**
> (Miller & Blair ch. 5 / Eurostat): **product-by-product** (commodity×commodity) vs **industry-by-industry**
> (process×process), the latter via the **market-share** rule (`D = S·q̂⁻¹`, fixed-product-sales). **(2) By-product
> method** — meaningful *only* on the analytic product-by-product PIOT (Altimiras-Martin 2014): **Suh** (2004)
> endogenises disposals-to-nature as **negative input coefficients**, *altering* `A`/`L` (the PIOLab/Wieland choice,
> project primary); **Xu–Zhang** (2009) leaves the table intact via a modified multiplier. _Avoid_: conflating the
> axes — **market-share** is a *shape*, **Suh** is a *by-product method*.

**Process network**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Process network** — the **industry-by-industry** PIOT in tidy/graph form: **processes are nodes, commodity
> flows are edges**; each edge `p→q` carries commodity `f` at value `D[p,f]·U[f,q]` (market-share allocation of
> `q`'s use of `f` back to its producers `p`). Boundary blocks attach as edges to boundary pseudo-nodes (`Nature`,
> `FD`, SEM-in, SEM-out); **disposals-to-nature** (`P` residual outputs) are **final outputs, never inter-process
> edges** — allocation does not apply to them (Altimiras-Martin 2014, fn. 3). The substrate of the production-network
> **Sankey**. Built by `PSUT.to_piot('market_share', level='Z')` as a tidy edge table (`block ∈ {Z, Y, E, O, P, W}`,
> `Z` = inter-process; each edge names its `commodity`; `region_crossed` travels with the row it came from), with
> `Z()` the process-by-process pivot and `D()` the shares. _Avoid_: calling it a "PIOT" unqualified (defaults to
> product-by-product); drawing the supply legs `S` as their own edges (they only attribute producers).

**ew-MFA headline indicators**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **ew-MFA headline indicators** — per region and period, in the table's unit: **DE** (domestic extraction, the `E`
> block into the region's processes), **imports** / **exports** (network edges crossing into / out of the region;
> `region_crossed` rows of a collapsed table count as both), **PTB** `= imports − exports`, **DMI** `= DE + imports`,
> **DMC** `= DMI − exports`, **RMC** (raw-material consumption, the extraction footprint of the region's final demand).
> Over a closed world `Σ RMC = Σ DE`. _Avoid_: comparing DMC (territorial, gross trade) with RMC (consumption-based,
> embodied) as if they measured the same thing.

**PIOT (physical input-output table)**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **PIOT (physical input-output table)** — a **symmetric** (square) physical IO table derived from a PSUT by a
> chosen **construct** (below). The shipped one is the **industry-by-industry**
> (process×process) PIOT, the **process network** (below): technical coefficients `A`, core flows `Z`,
> total requirements `L`, for footprints. The **product-by-product** PIOT (commodity×commodity) is #20. _Avoid_: conflating with the rectangular **PSUT** (a
> PIOT is square); reading "PIOT" as *product-by-product only* (it is a construct **family**).

**Technical coefficients `A`**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Technical coefficients `A`** (uppercase) — the per-unit input-requirements matrix of the PIOT (`A = Z·x̂⁻¹` for the shipped industry-by-industry construct; `A = B·D` for the
> product-by-product one, #20).
> _Avoid_: confusing with the lowercase **edge vector `a`** (the decision variable) — case distinguishes them.


### Glossary detail — framework and cross-stage

The glossary entries below keep their definition and their aliases to avoid. This section holds the full text they had before; each block names its entry.

**Control total**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Control total** *(removed — superseded by **Scale total**, see [ADR 0006](adr/0006-scale-total-supersedes-external-control-total.md))*:
> A Type-2 **soft-sum** constraint (`Σ aᵢ = c + ε`, free cone) whose RHS `c(t)` is an externally
> published total — here ICSG production + usage, per year — that a defined edge-set must sum to.
> **ROW (Rest-Of-World) totals are not imposed** (ROW-excluded). Internal edge consistency is
> *not* a control total — that is the job of hard mass balance.
> _Avoid_: "anchor" for an external total (the live term **Anchor** below is the model's own prior), "coherence soft-sum" (removed mechanism).

**Knob**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Knob**:
> A tuning constant of a stage driver (`P_PROP`, `ANCHOR_CV`, `TOL_GAP_REL`, the cascade order):
> an UPPER_CASE field of the case's section for Stages 2 and 3 (#120), a driver constant for
> Stage 4; its value is recorded in the run's sidecar. Changing a knob is a measured change.
> In reader-facing text (manuscript, abstracts) the knobs a **sweep** varies are **judgement
> values**; the solver's stopping rule (`TOL_GAP_REL`) is a knob but is reported as
> *termination*, apart from them.
> _Avoid_: "hyperparameter"; "parameter" / "param" (those are the `params` input tables);
> "setting", "config"; "judgement parameter".

**Key result**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Key result**:
> What a reader of the paper takes away: a property of the reconciled account on the **reference
> chain**, or of its sources as the account reveals them, that answers a sub-question of RQ1–RQ3.
> It passes three tests: reference chain, bug-free, research question
> ([docs/agents/key-results.md](agents/key-results.md)). A size, runtime or coverage is a
> *stage fact* (Methods); a fixed defect, a solver-setting artefact, a delta against an earlier chain
> or the ledger tally is *debugging* (Methods, SI, ADRs, REPs).
> _Avoid_: "result" for debugging; "finding" for a stage fact; "headline" for anything large.

**Role**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Role** — a framework-owned kind that a case's label is bound to. Two families: the **boundary
> role** of an `SB` node, `nature | secondary | final_demand` (the three system-boundary categories
> above; a `secondary` node is an input when reached by a use edge and a residual output when reached
> by a supply edge), and the **valuation role**, `content | gross` (contained-metal mass vs gross
> product mass). The entity kinds `Process | Flow | SB` are reserved framework words, not roles.
> A case binds its labels to roles in its acc spec
> ([ADR 0034](adr/0034-roles-are-the-frameworks-labels-are-the-cases.md)).
> _Avoid_: reading a label (`SEM`, `CC`) as the concept — labels are the case's, roles are the
> framework's; "entity kind" for a role.

**Relation**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Relation** — one linear relation over edges with a σ (`Σ αᵢ aᵢ  sense  c ± σ`) that exists
> before either stage consumes it, one object (`pysuto.relation.relation.Relation`) carrying its
> `sense` (`eq` / `le`, see **Sense**): a **constraint row** is a relation *emitted* into `G`
> (`pysuto.constraints.emit`); an imputation fill is a relation *solved* for its lone unknown. A
> **process recipe**, a Flow-node balance and a `conc`/`eff` params row are **relation sources**
> (below), not relations (a recipe references its Process node's balance and carries the pooled
> `eff` ratio; the Mass twin is the per-edge `conc` ratio). **Scale totals** and hard bounds are *emit-only*
> relations — never solved for a prior. _Avoid_: "constraint row" for the pre-stage object;
> "recipe" for a single relation.

**Case**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Case** — one metal's complete declaration: an accounting structure (referenced by registry id),
> its feeds, param families, relation sources and cascade, anchor families and σ policies, run by the
> generic stage drivers. Declared as a `CaseSpec` named `CASE` in a Python module or file that
> `--case` resolves: a path, a dotted module, or a bundled alias (#120, #187); the copper case is the
> first, `scripts/copper.py` (ADR 0047). A case also sets the Stage 4 and 6 driver constants it departs from (`driver_knobs`). Its
> identity in a run is `CaseSpec.id`. _Avoid_: "model" (the QP, the PIOT), "study" (a REP analysis),
> "scenario".

**Acc spec**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Acc spec** — the declaration of one accounting structure: its sectors (each with an entity, and
> a boundary role on an SB sector), regions, valuations (each with a valuation role) and times, its
> supply and use adjacency, tradable flows and entity-pair accounting rows, and optionally its bound
> and mask records. An `acc.json` and the copper case's four ODS/CSV files are two formats of it, each
> with its reader (`read_acc_spec_json`, `read_acc_spec_ods`;
> [ADR 0046](adr/0046-an-accounting-structure-is-declared-by-a-spec.md)).
> _Avoid_: "case spec" (#26's typed `CaseSpec`, which references an acc by registry id), "root
> file" (one of the files a reader reads).

**Application**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Application** — everything outside the library that one case needs to run end to end and to
> report on: its case declaration, its feed specs, source parsers and feed builders, its sweeps, and
> its report and exploration scripts. The library holds the method, generic for every case; an
> application holds what carries its metal's name. Copper's application is the repository's
> `scripts/` ([ADR 0047](adr/0047-pysuto-ships-no-study-case.md)). _Avoid_: "study" (a REP
> analysis), "project", "scripts" (a folder, not the concept).

**Toy case**, full text of its [GLOSSARY.md](../GLOSSARY.md) entry before the entry was shortened to its definition:

> **Toy case** — the synthetic **Case** of the fictional metal orium (`pysuto.tutorial.toy`, #59),
> declared in code with its own labels (`Gross`/`Content` valuations, `Raw`/`Sec`/`Demand`
> boundary subs, bound to roles), generated from a **ground truth** and observed through
> **synthetic feeds**.
> _Avoid_: "toy fixture" (the per-test structures built in `tests/`), "test case", "demo".

---

## Module reference

> Contributors / LLM assistants: see [CLAUDE.md](../CLAUDE.md) for behavioural rules.

| File | Public API | Responsibility |
|---|---|---|
| [acc.py](../src/pysuto/acc.py) | `AccountingStructure`, `RuleSet`, `BoundaryRole`, `ValuationRole`, `read_acc_spec_ods`, `read_acc_spec_json`, `read_acc_spec_dir`, `require_compatible_acc`, `check_feed_accs` | Built from an acc spec (`AccountingStructure.from_spec`); the ODS/CSV files and `acc.json` are its two formats, read by `read_acc_spec_ods` / `read_acc_spec_json` (`read_acc_spec_dir` picks one). The roles a case binds its labels to: `acc.boundary_roles` (SB sector → `BoundaryRole`) and `acc.valuation_roles` (`.content`, `.gross`, the bound labels), which the library reads in place of labels (ADR 0034). Root node/edge index, supply–use topology, structural edge masks (`acc.masks["supply" | "use" | "tradable" | "sb" | "cross_region"]`; `cross_region` is the edge scope of bilateral feeds, ADR 0014), per-edge bounds. `require_compatible_acc` compares two acc versions' specs on the keys a consumer depends on (#53, #119). |
| [acc_store.py](../src/pysuto/acc_store.py) | `save`, `load`, `write_build_record`, `check_built_acc` | The persisted form of an `AccountingStructure` under `<storage_dir>/<id>/` (#243): the index tables, `bounds_l.npy` / `bounds_u.npy`, `masks_user.npz` and a JSON header `metadata.json` (axes, node ids, roles, rule tables); `load` also reads the pickled `metadata.pkl` of a directory saved before #243 and never opens its `edge_index_all.csv`, `A.npz` or `A_nodes.npz`. The build record beside it names the acc input version it came from; `check_built_acc` checks it against a run's acc ref. |
| [conc.py](../src/pysuto/conc.py) | `ConcordanceTable`, `WeightMatrix` | src→root edge mapping; sparse CSR `W`. |
| [feed.py](../src/pysuto/feed.py) | `DataFeed` | Owns `weight_matrix`, pedigree, `F_src`. |
| [feedpipe.py](../src/pysuto/feedpipe.py) | `FeedSpec`, `FeedBuild`, `run_feed`, `check_file`, `step_A_read_metadata`, `step_B_build_concordance`, `step_C_build_weight_matrix`, `step_D_apply_and_propagate`, `build_F_src`, `propagate_with_row_penalty`, `register_feed` | Stage 1 orchestration. |
| [unc.py](../src/pysuto/unc.py) | `PedigreeScores`, `UncertaintyMixin`, `PropagationResult` | Pedigree→CV (Laner et al.); Lp / power-mean propagation through `W`. |
| [meta.py](../src/pysuto/meta.py) | `DataFeedMetadata` | Parses the Excel sidecar. |
| [registry/](../src/pysuto/registry/) | `BaseOutputRegistry`, `DataFeedRegistry`, `ImputationRunRegistry`, `ConstraintRunRegistry`, `OptimRunRegistry`, `UncPropRunRegistry`, `PsutRunRegistry`, `SweepRunRegistry`, `_RegistryEntry`, `make_run_id`, `build_sidecar`, `stage_timer`, `stage_lock`, `LockHeld` | Harmonized output registries (Stages 1–6); shared run-id pattern + sidecar schema; the advisory stage lock on `data/proc` (#101). **Subpackage layout:** `base.py`, `feed.py`, `imputation.py`, `constraint.py`, `optim.py`, `uncprop.py`, `psut.py`, `sweep.py`, `lock.py`, `_sidecar.py`. |
| [input_registry.py](../src/pysuto/input_registry.py) | `InputRegistry`, `InputRegistries`, `pins_from_inputs`, `input_pins` | Per-kind versioned snapshot store for `feed`, `acc`, `params`, `mask` inputs; hardlink dedup; auto-bump on content-hash change; pins that replay a sidecar's input versions (#53). Imports nothing from `acc`; the spec-comparing checks of acc versions live in `acc.py` (#119). |
| [knobs.py](../src/pysuto/knobs.py) | `parse_overrides`, `apply_overrides`, `add_knob_args`, `overrides_from_cli`, `write_run_id` | The driver knob protocol: `--set KNOB=VALUE` against a driver's `KNOBS`, and `--run-id-file` (#100). |
| [case.py](../src/pysuto/case.py) | `CaseSpec`, `Section`, `ImpSection`, `ConstSection`, `FillEntry`, `CascadeStep`, `Cascade`, `CaseContext`, `resolve_case`, `ALIASES`, `add_case_arg`, `case_from_args` | A case as a typed declaration (#120), the `--case` resolver (#187) and the drivers' `--case` option (`add_case_arg` / `case_from_args`, which end a driver through `parser.error`): its acc id, one frozen section per stage and the `driver_knobs` it sets on the Stage 4–6 drivers (#124), whose UPPER_CASE fields are the stage's knobs and whose entries and functions (the Stage 2 `cascade` of `FillEntry` data and `terminal`, the Stage 3 `builders`, selectors) build over a `CaseContext`; `with_knobs` applies `--set`, `describe` serialises a section into a run's `stage_config["case"]`. |
| [scripts/copper.py](../scripts/copper.py) | `copper.CASE`, `CopperImp`, `CopperConst`, `CASCADE`, `terminal`, `builders`, `pre_impute`, `post_impute`, `nodata_anchor_exempt`, `stage1`, `scale_total_resolver` (over `constraints.anchors`, #145) | The copper case, which lives in the repository and not in the package (ADR 0047): a module beside the drivers, outside the import direction, that imports the library. It holds the copper Stage 2 and 3 model: every knob at its shipped value, the declared imputer cascade, the constraint builders, which call the σ algorithms of `pysuto.impute` (#121, #144); the Stage 2–6 drivers take `--case <reference>` (the `scripts/` drivers default to `scripts/copper.py`, resolved by path; `copper` imports `copper_feed` one way) and `python -m pysuto run` requires it; `pysuto.case.resolve_case` resolves a bundled alias, a dotted module or a file path to the module's `CASE` and stamps it with `case_source` (`alias:<name>`, `module:<dotted>` or `path:<abs>`) and `case_sha256` (the sha256 of that module's source file), which a run records in `stage_config["case"]` and which stay out of `CaseSpec` equality; `with_knobs` keeps them, while a case built in Python, or changed structurally with `dataclasses.replace` on a module's own `CASE`, carries neither (clear both on a resolved case before changing it; #187); the drivers parse arguments, take the stage lock and run the stage. |
| [stages.py](../src/pysuto/stages.py) | `run_imp`, `run_const`, `run_opt`, `run_uncprop`, `run_psut`, `OPT_KNOBS`, `PSUT_KNOBS` | The bodies of the Stage 2–6 drivers as library functions over a `CaseSpec`, a data dir and the upstream run id (#124, #186); the toy's `run_stage2` / `run_stage3` call the first two, and `python -m pysuto run` all five. The Stage 4–6 tuning constants (solver tolerances, `N_WORKERS`, `QP_VARIABLES`, `ALPHA`, `PRIMARY`) sit in its `# USER CONFIGURATION` block. `pysuto.case.check_case` refuses a run made by another case. |
| [compare.py](../src/pysuto/compare.py) | `compare_chains`, `lineage`, `first_difference`, `format_report`, `CompareError` | `python -m pysuto compare` (#157): two chains by their leaf run ids, paired stage by stage and compared array by array; imports nothing from `pysuto`, writes nothing. |
| [pipeline.py](../src/pysuto/pipeline.py), [\_\_main\_\_.py](../src/pysuto/__main__.py) | `Pipeline`, `Chain`, `StepFailed`, `Axis`, `Sweep`, `STEPS`, `ALL_STEPS`, `TIED`, `STEP_GB`, `split_key` | The stages composed as steps, each in a child process, Stage 1 (a case's hook) to Stage 6 (`python -m pysuto run`, #125); one chain per `run(overrides, label)` with `<stage>.<KNOB>` overrides (#103); random (Latin hypercube) or grid sweeps of chains, resumable, recorded via `SweepRunRegistry` (#105); a sweep may keep only its last step's runs, the others passing through a scratch store (#108). |
| [logging.py](../src/pysuto/logging.py) | `get_logger`, `configure_file_logs`, `start_run_log`, `add_run_handler`, `utf8_stdio_env` | Named loggers; a driver turns on one file per module under `<data-dir>/logs/` plus one timestamped file per run, all UTF-8; `start_run_log` makes stdout and stderr UTF-8, `utf8_stdio_env` does it for a child process (#293). |
| [config.py](../src/pysuto/config.py) | `load_config`, `Config`, `PipelineConfig` | Typed loader for `config/config.yaml` (`pipeline.acc_id` only). |
| [lp.py](../src/pysuto/lp.py) | `lp_combine` | The Lp / power-mean σ combiner `(Σ \|x_i\|^p)^(1/p)`, one source of truth for Stages 1–3. |
| [masks.py](../src/pysuto/masks.py) | `MaskRuleSet`, `coerce_mask`, `load_masks_csv` | Named boolean edge masks over an `AccountingStructure` (flat-prior CV ranges, builder scope, feed `edge_scope_mask`). |
| [relation/params.py](../src/pysuto/relation/params.py) | `ParamRow`, `read_params`, `WILDCARD`, `A_FIELDS`, `B_FIELDS` | The one reader of the params CSVs (#128): `read_params(path, accepts)` parses with a single policy (sniffed delimiter, `dtype=str`, `#` comments) into frozen, typed `ParamRow`s (`agg` as a bool, `sigma()` from `unc_type`, `specificity`); each consumer declares an `ACCEPTS` set of `param_type`s, other rows are filtered and counted in the log, a malformed accepted row raises, and a soft row with no σ rule raises. `constraints/sum.py` reads another schema and stays. |
| [relation/param_relation.py](../src/pysuto/relation/param_relation.py) | `ParamRelationResolver`, `ResolvedRelation`, `expand_edge_index`, `conc_twin_map`, `conc_nominal_lookup` | The A→B relation resolver over a `ParamRow` (selectors, `*` / `same` wildcards), shared by Stage 2 (`ParamSource.from_params`) and Stage 3 (`RatioConstraintBuilder` / `RatioBoundBuilder.from_csv`). |
| [relation/sources.py](../src/pysuto/relation/sources.py) | `trade_scope` | `trade_scope(acc)`: the cross-region `Flow→Process` edges of the content valuation, the trade seed's territory and the complement of the NetworkImputer's same-region one (ADR 0010); the case passes it to its imputers. |
| [relation/structure.py](../src/pysuto/relation/structure.py) | `ProcessRecipe`, `NodeBalance`, `process_recipes`, `node_balances` | The structural relation sources, derived once from the acc and cached per acc: a recipe (inputs, Flow `products`, secondary-SB `wastes` per process-region-year) and the content balance of every Flow and Process node-period. `acc` imports nothing from `impute`; the NetworkImputer reads these and raises on a recipe with more than one Flow product until co-products land (#129, #130). |
| [flow_node_balance.py](../src/pysuto/flow_node_balance.py) | `flow_node_balance`, `contradicted_flow_nodes`, `K_CLOSE` | Detects a contradicted flow node — inflow and outflow priors that disagree — for the Stage 2 and Stage 3 drivers; detection only, the policy is the caller's ([ADR 0019](adr/0019-contradicted-flow-nodes-lose-anchor-authority.md)). |
| [collector.py](../src/pysuto/collector.py) | `FeedCollector` | Select feeds, align onto common edge index, produce `AggregationResult`. |
| [agg.py](../src/pysuto/agg.py) | `AggregationMath`, `AggregationResult` | Build `Ω`, compute `f_agg`, `σ_agg`, coverage. |
| [impute/](../src/pysuto/impute/) | `ImputationEngine`, `BaseImputer`, `PriorLedger`, `FillImputer`, `NetworkImputer`, `TimeImputer`, `ShareImputer`, `StockLeg`, `FlatPriorImputer`, `ImputationLog`, `ImputationMetadata`, `time_interp_scope`, `damp_leg_sigma`, `screen_outlier_trade_legs`, `screen_contradicted_nodes` | Ordered imputation pipeline; per-edge provenance logging. **Subpackage layout:** `engine.py`, `base.py`, `ledger.py`, `fill.py`, `network.py`, `time.py`, `share.py`, `flat_prior.py`. |
| [constraints/](../src/pysuto/constraints/) | `ConstraintEngine`, `BaseConstraintBuilder`, `MassBalanceBuilder`, `SumConstraintBuilder`, `RatioConstraintBuilder`, `RatioBoundBuilder`, `FlowBoundsBuilder`, `ConstraintBlock`, `ConstraintRun`, `anchors`, `naming.ROW_META_COLS`, `naming.build_constraint_id` | Plugin constraint pipeline; persists to `data/proc/const/` via `ConstraintRunRegistry` (lives in [registry/](../src/pysuto/registry/)). **Subpackage layout:** `engine.py`, `base.py`, `mass_balance.py`, `sum.py`, `ratio.py`, `flow_bound.py`, `run.py`. |
| [solver.py](../src/pysuto/solver.py) | `Solver`, `OptimResult` | Stage-4 orchestrator: builds the σ-weighted period QP (warm or cold prior) and hands it to an injected QP backend; imports no numerical solver. `Solver.objective_at` scores any year-local posterior under a const run's QP, split by family (#102). |
| [qp.py](../src/pysuto/qp.py) | `QPProblem`, `QPResult`, `QPBackend` | The neutral standard-form period QP and the backend protocol (ADR 0042). |
| [clarabel_backend.py](../src/pysuto/clarabel_backend.py) | `ClarabelBackend`, `ITER_DTYPE` | The production QP backend; the only module that imports `clarabel`. |
| [uncprop.py](../src/pysuto/uncprop.py) | `BaseUncProp`, `KalmanUncProp`, `UncPropResult`, `UncPropTimestep` | Stage-5 posterior uncertainty: active set and a QR of the whitened system per year; persists to `data/proc/uncprop/` via `UncPropRunRegistry`. |
| [psut.py](../src/pysuto/psut.py) | `PSUT`, `OptimResultsLike`, `classify_blocks`, `BLOCKS` | Stage-6 tidy-canonical PSUT: `from_opt_run` / `from_results`, block views `S`/`U`/`Y`/`E`/`O`/`P`/`W`, `realized_conc`, `aggregate` (region/sector pushforward with `region_crossed`), `sankey` (thin delegate to `diag.psut`), `to_piot`, period selection (`periods`, `latest`, `at`); every valuation in one table with a `primary`; persists to `data/proc/psut/` via `PsutRunRegistry`. |
| [piot.py](../src/pysuto/piot.py) | `PIOT`, `market_share_network`, `market_shares`, `CONSTRUCTS`, `LEVELS` | Stage-6 PIOT as a construct family: the market-share process network (`Z()`, `D()`, process-network `sankey()`), levels `A`/`L` (`x`, `y`, `A`, `L`), `intensity`, `footprint`, `embodied_trade`, `ew_mfa`, `multipliers`, `sda`; `suh_product` is a stub until #20. A derived view of a PSUT, never persisted. |
| [_conditioning.py](../src/pysuto/_conditioning.py) | `condition_prior`, `condition_sigma_c` | Shared σ_a / σ_c conditioning rules — guarantees uncprop sees the same effective prior the solver did. |
| [diag/](../src/pysuto/diag/) | `FeedDiagnostics`, `AggDiagnostic`, `ImpDiagnostic`, `ConstDiagnostic`, `SolverDiag`, `UncPropDiag`, `SweepDiag`, `sankey_figure`, `sankey_tables`, `SankeySpec`, `TruthDiag`, `truth.prior_z`; `stats/`: `CoverageDiagnostic`, `StatsDiagnostic`, `ConstStatsDiagnostic`, `SolverStatsDiagnostic`, `UncPropStatsDiagnostic`, `SweepStatsDiagnostic` | Cross-stage diagnostic plot suites (figures), row-level stats tables (CSVs), the Stage-6 supply-use Sankey (Plotly, lazy) and, for a synthetic case, every stage's estimate scored against its ground truth (`truth.py`: error by stage, calibration and coverage, residual rank). **Subpackage layout:** `feed.py`, `agg.py`, `imp.py`, `const.py`, `solver.py`, `uncprop.py`, `sweep.py`, `psut.py`, `truth.py`, `_plot.py`, `stats/{_core,coverage,val,const,solver,objective,uncprop,sweep}.py`. |
| [tutorial/](../src/pysuto/tutorial/) | `toy`, `toy_ext`, `toy_coproduct`: `ACC_SPEC`, `ACC_REF`, `build_acc`, `truth`, `world_totals`, `make_feeds`, `SyntheticFeed`, `write_params`, `write_inputs`, `CASE`, `run_stage1` … `run_stage6`, `soft_residuals`, `scoreboard`, `reveal`, `check_<chapter>_<n>`, `solution` | The tutorial: the eleven notebooks, the params template (`params_template.csv`) and the toy cases (`toy.py`, `toy_coproduct.py`, `toy_ext.py`), at the top of the import stack beside the stage drivers; `python -m pysuto tutorial DEST` copies the notebooks and the template out (#188, #197). `toy.py` is the **toy case** (#59): a three-region orium case generated from a planted ground truth, its three synthetic feeds, its params CSVs, and the six stages wired into a caller-chosen data dir (the drivers run it with `--case toy --data-dir <dir>`, #124); the tutorial's exercise hooks (#60). It declares its own labels, bound to roles (ADR 0034, #123), and registers no hard bands (#27). |

The feed-side algorithms of a case (the BACI-trade-share weights of ICSG's producer fan, the re-export netting and the ore-grade triangle; ADR 0002, 0015, 0023, 0024, 0027) are copper application code, not library: they live in `scripts/feedprep/` (ADR 0047, #302), and `scripts/build_{trade_shares,baci_reexport,ore_grade}.py` call them.


### Files inside the packages

The table above names the packages; these are the files that carry its rows' parts (moved from the former file tree).

- Top level: `_provenance.py` (`now_iso`, `sha256_file`; timestamps and hashes, #279) and `_step.py` (the child process of `python -m pysuto run`, one per step).
- `relation/`: `params.py` (`ParamRow`, `read_params`), `param_relation.py` (`ParamRelationResolver`), `structure.py` (`ProcessRecipe`, `node_balances`), `sources.py` (`ParamSource`, `Fill`), `relation.py` and `ratio_relation.py` (`Relation`, the ratio resolver), `balance.py` and `prior.py` (the node balances, the prior relation).
- `constraints/`: `engine.py`, `base.py`, `mass_balance.py`, `sum.py`, `ratio.py`, `flow_bound.py`, `anchors.py` (scale totals and the no-data σ anchor), `emit.py` (`Relation` → `G` rows), `naming.py` (constraint ids and row-meta names), `run.py` (`ConstraintBlock`, `ConstraintRun`).
- `diag/`: `feed.py`, `agg.py`, `imp.py`, `relaxed_close.py` (the relaxed-close exposure, Stage 2), `const.py`, `solver.py`, `uncprop.py`, `psut.py`, `truth.py` (`TruthDiag`), `sweep.py` (`SweepDiag`), `_plot.py`.
- `diag/stats/`: `_core.py` (`StatsDiagnostic`), `coverage.py`, `val.py`, `const.py`, `solver.py`, `objective.py` (objective share and decomposition, identity ranking; stateless builders), `problem.py` (`ProblemStatsDiagnostic`, the problem panel), `sweep.py`, `uncprop.py`.

### Public API

The **public surface** is what a second case imports: the names in the *Public API* column of the table above (listed per module, `file.name` for a file inside a package row). A case is a module beside the drivers (`scripts/copper.py`, `scripts/copper_feed.py`), `pysuto.tutorial.toy*` or the case a reader builds in chapter 10 of the tutorial; the library never imports one (§5). Packages declare it with `__all__`: `constraints/`, `impute/`, `registry/`, `diag/` and `diag/stats/` the names they re-export, `relation/` and `tutorial/` their submodules. A module without `__all__` is read by its row; anything absent from the table, and every name with a leading underscore, is internal and may change without notice. `tests/test_public_surface.py` imports every name that the copper case, the toy and the notebooks use from `pysuto` and refuses one that the table does not name.

The one private module a case reaches for is `pysuto._conditioning` (`condition_prior`, `condition_sigma_c`): the toy and chapters 4 and 5 of the tutorial rebuild the solver's conditioned QP to show it, and Stage 4, Stage 5 and the diagnostics import the same functions so that all of them see one prior. It stays private because a case never needs it to run: it declares its σ rules in its sections, and the floors a run used are in the opt sidecar's `stage_config`; a rename would add a public name that a second case has no use for, and the four import sites of the tutorial are the only ones outside the pipeline.

---

## Import direction

**Import direction is strictly one-way.** The table below is the layer table: a *unit* is a top-level module or subpackage of `pysuto` (edges inside a unit are free), and a row lists the units it imports at module level, then those it imports only under `if TYPE_CHECKING:`. `tests/test_import_direction.py` reads this table and walks `src/pysuto` with `ast`, so a module-level import that is not in its row fails the suite; change the code or the row, never one without the other (#217).

| unit | module-level imports | typing only |
| --- | --- | --- |
| `acc` | `input_registry`, `logging`, `masks` |  |
| `acc_store` | `acc`, `logging` |  |
| `agg` | `logging`, `lp` |  |
| `case` | `_provenance`, `impute`, `knobs`, `relation` | `acc`, `agg`, `input_registry` |
| `clarabel_backend` | `logging`, `qp` |  |
| `collector` | `agg`, `logging`, `lp`, `registry` |  |
| `compare` |  |  |
| `conc` | `feed` |  |
| `config` |  |  |
| `constraints` | `acc`, `flow_node_balance`, `logging`, `lp`, `registry`, `relation` |  |
| `diag` | `_conditioning`, `acc`, `acc_store`, `clarabel_backend`, `constraints`, `logging`, `registry`, `relation`, `solver`, `uncprop` | `psut` |
| `feed` | `unc` |  |
| `feedpipe` | `acc`, `acc_store`, `conc`, `input_registry`, `logging`, `lp`, `meta`, `registry`, `unc` |  |
| `flow_node_balance` | `relation` |  |
| `impute` | `acc`, `agg`, `flow_node_balance`, `logging`, `lp`, `relation` |  |
| `input_registry` | `_provenance`, `logging`, `meta` |  |
| `knobs` |  |  |
| `logging` |  |  |
| `lp` |  |  |
| `masks` | `logging` |  |
| `meta` | `unc` |  |
| `piot` | `psut` |  |
| `pipeline` | `_provenance`, `_step`, `case`, `logging`, `registry` |  |
| `psut` | `acc`, `acc_store`, `input_registry`, `logging`, `registry` |  |
| `qp` |  |  |
| `registry` | `_provenance`, `logging`, `meta` | `constraints`, `psut`, `solver`, `uncprop` |
| `relation` | `acc`, `logging`, `unc` |  |
| `solver` | `_conditioning`, `acc`, `acc_store`, `constraints`, `input_registry`, `logging`, `qp`, `registry` |  |
| `stages` | `acc`, `acc_store`, `case`, `clarabel_backend`, `collector`, `constraints`, `diag`, `impute`, `input_registry`, `knobs`, `logging`, `psut`, `registry`, `solver`, `uncprop` |  |
| `tutorial` | `*` |  |
| `unc` | `lp` |  |
| `uncprop` | `_conditioning`, `constraints`, `logging`, `registry` |  |
| `__main__` | `case`, `compare`, `logging`, `pipeline`, `tutorial` |  |
| `_conditioning` | `logging` |  |
| `_provenance` |  |  |
| `_step` |  |  |

How to read it:

- The spine runs `acc → relation → impute → constraints → solver → uncprop` with `collector → agg → impute` beside it; `input_registry`, `masks`, `unc`, `lp`, `qp`, `_conditioning`, `_provenance` (timestamps and sha256, #279) and `logging` lie below. `meta → unc`, `registry → meta`, `input_registry → meta`; `conc → feed → unc`; `flow_node_balance → relation`; `psut → acc, registry` and `piot → psut` (Stage 6, a derived view). `acc_store → acc` holds the persisted form of a structure (#243: save, load, the build record); `solver`, `stages`, `psut`, `feedpipe` and the diag load a structure through it. `relation` holds the params reader `read_params` and the A→B resolver and imports `acc`, `unc` and `logging`; `acc` imports neither `relation` nor `registry`.
- `case` (`CaseSpec`) imports `impute`, `knobs` and `relation` at module level and `acc`, `agg` and `input_registry` for typing only; `resolve_case` reaches a bundled case only through `importlib.import_module` on a name → module string, at call time, the one sanctioned reference from the library to `pysuto.tutorial` (`tests/test_case_resolution.py`). `knobs` is the `--set` / `--run-id-file` protocol of the drivers; `stages` holds the Stage 2–6 bodies over a `CaseSpec` and `pipeline` runs each step as a child of `_step`; neither imports the tutorial or `scripts/`.
- `diag` is a consumer: the data layer does not import it at module level. Four call sites do so inside a function, behind a `diag_plot` flag or on use (`constraints/engine.py`, `feedpipe.py` twice, and `psut.py`'s `sankey`), each carrying a `lazy` comment. This is the current state, not a decision: whether plotting moves to the drivers is open for the Stage 1 and Stage 3 architecture sessions (#174, #177).
- The run registries sit below the stages whose result classes they rebuild, so each `load` imports its result class (`ConstraintRun`, `OptimResult`, `UncPropResult`, `PSUT`, `AggregationResult` and `ImputationLog`) inside the function and the annotations use `TYPE_CHECKING`: a deliberate back-edge family, `registry → constraints, solver, uncprop, psut, agg, impute`, kept lazy so that importing a registry pulls in no stage.
- Every other function-local import of `pysuto` names its reason in a comment (`cycle`, `lazy` or `optional`); the test requires it. `masks → relation.param_relation` is a cycle (`relation → acc → masks`), `psut → piot` is one (`piot → psut`), and `_step` imports lazily so that `pipeline` can read `StepSpec` without the stages.
- `tutorial` is the top: it imports the library like a stage driver, and nothing but `__main__` imports it. `scripts/copper.py` is the copper case outside the package (ADR 0047): it imports the library, and the `scripts/` drivers import it. `__init__.py` defines `__version__` (read from the installed distribution) and imports nothing else.

---

## References

Laner, D., Rechberger, H., & Astrup, T. (2016). A novel approach to characterize data uncertainty in material flow analysis and its application to plastics flows in Austria. *Journal of Industrial Ecology*, 20(5), 1050–1063.

Lenzen, M., Wood, R., & Wiedmann, T. (2009). Matrix balancing under conflicting information. *Economic Systems Research*, 22(1), 1–23.

Pauliuk, S., Heeren, N., Berrill, P., Fishman, T., Nistad, A., Tu, Q., Wolfram, P., & Hertwich, E. G. (2019). Global scenarios of resource and emission savings from material efficiency in residential buildings and cars. *Nature Communications*, 12, 5097. *(cited for the data-category 3/4 intensive property typology used in the params relation source)*
