# Changelog

All notable changes to pySUTO are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [0.1.0] - 2026-11-26

The reviewed release. The final review (#155) audited the code at `v0.1.0rc1` on 17 review
dimensions in 9 review units, then fixed what it found under one rule: every fix leaves the
copper reference chain and the toy chain bit-identical. The check chains and their array
counts are in `docs/rep/` (the final review's check REP) and in each fix ticket's closing
comment. No published number moves.

### Added
- Every run sidecar records what produced it: the package version, git commit and dirty
  flag, the Python and numerical-stack versions, the stage's wall time and peak memory; a
  feed run also records its Lp exponent, hooks and knobs (#210).
- `python -m pysuto compare` aligns the imp's per-feed rows by feed id, so a feed order
  alone is no difference (#211).
- `pysuto.acc_store`: the persisted form of an accounting structure in its own module,
  with a JSON header; it still reads a structure saved by 0.1.0rc1 (#243).
- `lp.P_DEFAULT`, the one definition of the Lp exponent, and `lp.lp_combine_axis`, the
  one vectorised Lp combination (#224).
- The three feed drivers answer `--help` without building anything (#219).
- Tests that guard the structure: the import direction against README §5's layer table
  (#217), the public surface a second case imports (#218), both build-order guards through
  their entry points (#221), Stage 5's three SI invariants on the block path and Eq. S7's
  σ_a branch (#222), and the unit audits' missing tests (#278–#286).

### Changed
- Loading copper's accounting structure takes under a second and 0.4 GB instead of 27 s
  and 10 GB, and building it 15 s instead of 3 min: the unused 70-million-row edge table
  is no longer built, written or read (#243).
- A case owns every σ argument of its imputers and constraint builders; the library keeps
  no σ defaults of its own (#224).
- Build-time checks raise instead of logging, at the inputs copper never has: Stage 2
  (#282), Stage 3 (#283), Stage 4's conditioning and solver statuses (#284), Stage 5
  (#285), Stage 6 (#286), the accounting structure's bounds and spec (#278).
- The copper labels left in the library moved to the copper case, and the label gate
  covers sector and region names (#220).
- The report scripts read their point counts, base runs and intervals from named runs
  (#221).
- `run_solver.main` defaults to the cold prior, like the CLI; the feasibility message
  points to the input behind a conflicting row instead of advising to drop it (#219).
- The stage lock's stale takeover is atomic, and a builder run by the Stage 1 hook joins
  the pipeline's hold (#279).
- The solver diagnostics divide by the conditioned σ the QP used (#284).
- Every ADR has one Status line and each decision's code its ADR tag (#223); the README,
  GLOSSARY.md, CLAUDE.md and the agent docs say what the code does (#225).
- The tutorial's provenance tree names its feed runs, and chapters 00–07 print no library
  log lines (#203).

### Removed
- Stage 5's full-G posterior path, `KalmanUncProp(block_size=)` and the `uncprop`
  (scikit-sparse) extra; `Solver.solve_timestep` (#214).
- `AccountingStructure`'s callable-rules constructor mode and six tests-only members
  (#215); `AccountingStructure.save`/`load` (now `acc_store.save`/`load`), `A`, `A_nodes`
  and `edge_index_all` (#243).
- `feedpipe.run_feed_pipeline`, `ConcordanceTable.save`/`load`, `WeightMatrix.apply`
  (#280); `scripts/migrate_registries.py` (#279); the uncited exploration and parse scripts
  (#213, #281); write-only fields and an ignored parameter (#216).

### Data
- The licence check of the ICSG/S&P-derived tables (`scrap_share`, `stock_bounds`,
  `ore_grade`) is recorded on #182 before the tag; a table that fails leaves `current/`
  and is listed here.

## [0.1.0rc1] - 2026-11-19

The packaged release candidate: `pyproject.toml`, the MIT licence and the CC BY 4.0 data
licence, CI, and the PyPI release (#118, #195).
