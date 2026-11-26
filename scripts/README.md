# scripts: the copper application

This folder holds the copper application of pySUTO: the case, the feed builders, the stage drivers and the reports. The library in `src/pysuto/` knows nothing about copper (ADR 0047). Run every script from the repository root.

The pipeline, the run ids and the knobs are in [docs/workflow.md](../docs/workflow.md). The stages are in [docs/architecture.md](../docs/architecture.md). The copper chain runs only in a checkout that holds the licensed source data.

## The case

| File | What it is |
|---|---|
| [copper.py](copper.py) | The copper case: `CASE`, the Stage 2 and Stage 3 model as one `CaseSpec`. Pass it as `--case scripts/copper.py`. The Stage 2 and 3 knobs are fields of its sections. |
| [copper_feed.py](copper_feed.py) | Stage 1 of the case: the three `FeedSpec`s and `run_stage1`, which runs the feed builds in their load-bearing order. |
| [acc.py](acc.py) | Builds and saves the accounting structure of an id (`--id copper`). |

## Stage 1: feeds and builders

Run order matters (ADR 0024, 0027, 0030). `run_stage1` in `copper_feed.py` runs the builders in order. The BACI parse, the gross registration and `icsg_clean.py --to-feed` stay manual.

| File | What it builds |
|---|---|
| [baci_feed.py](baci_feed.py) · [icsg_feed.py](icsg_feed.py) · [crude_ore_feed.py](crude_ore_feed.py) | The three feed runs. Each is a thin driver over `run_feed`. |
| [build_baci_reexport.py](build_baci_reexport.py) | The re-export netting of the BACI feed (ADR 0027). |
| [build_trade_shares.py](build_trade_shares.py) | The trade-share weights of ICSG's producer fan (ADR 0024). Run it before `icsg_feed.py`. |
| [build_ore_grade.py](build_ore_grade.py) | The ore-grade parameters (ADR 0015, 0023). Run it by hand after the feeds. |

## Feed preparation

[feedprep/](feedprep/) holds the library functions behind the three builders: `reexport.py`, `trade_shares.py` and `ore_grade.py`. The case supplies the labels as parameters. The wheel does not ship this folder.

## Parse layer

[parse/](parse/) turns the raw sources into feed-ready tables.

- `baci_parse/baci_parse_feed.py` parses BACI.
- `icsg_parse/` parses the ICSG yearbooks. `icsg_clean.py` combines them and writes the ICSG feed input.
- `crude_ore_parse_feed.py` parses the crude-ore dataset.

## Stage drivers

Each driver is a thin wrapper over a function of `pysuto.stages`. `python -m pysuto run` chains them.

| Stage | Driver | Diagnostics |
|---|---|---|
| 2 | [collect_aggregate_impute.py](collect_aggregate_impute.py) | |
| 3 | [constraints.py](constraints.py) | |
| 4 | [run_solver.py](run_solver.py) | [solve_diag.py](solve_diag.py), [solver_stats.py](solver_stats.py) |
| 5 | [uncertainty_propagation.py](uncertainty_propagation.py) | [uncprop_diag.py](uncprop_diag.py) |
| 6 | [psut.py](psut.py) | |

## Sweeps

[sweep.py](sweep.py) runs a set of chains that differ only in knob values. Use `--dry-run` first. [sweep_diag.py](sweep_diag.py) scores a finished sweep.

## Reports

[report/](report/) builds the figures and numbers of the manuscript from one reference run: `results_figs.py`, `manuscript_numbers.py`, `manuscript_checks.py`, `table3_intervals.py` and `copper_metabolism_insights.py`. `make_logo.py` draws the logo.

## Explorations

[explo/](explo/) holds the scripts behind the reports in `docs/rep/`. It is a frozen record: a script may not run on `main`. Read [explo/README.md](explo/README.md) before you use one.

The maintenance tools (`input_status.py`, `prune_runs.py` and others) are in [tools/](../tools/).
