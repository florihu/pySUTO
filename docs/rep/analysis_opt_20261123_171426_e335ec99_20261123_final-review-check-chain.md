# Final review (#155, #182): the reviewed code's copper chain from Stage 1 reproduces the reference chain's 1,102 arrays bit for bit, the acc build and the ICSG parse layer reproduce byte for byte, and the toy chain its 248 arrays

**Run:**
- **The chain (wave 2):** one stage driver at a time under the stage lock, from a frozen worktree at `review/integration` `b08d860`, with the main checkout as the working directory and `PYTHONPATH=<worktree>/src`; each driver ran under `/usr/bin/time -v`. Label `final review wave w2 check chain`; every knob at its default.
  - Stage 1: `python -m pysuto run --case scripts/copper.py --data-dir data --from feed --to feed`.
  - Stage 2: `collect_aggregate_impute.py` on the three new feeds, in the reference imp's order (icsg, baci, crude_ore).
  - Stages 3–6: `constraints.py`, `run_solver.py`, `uncertainty_propagation.py`, `psut.py`.
  - Run ids: `feed_icsg_20261123_170245_50b6caae`, `feed_baci_20261123_170234_2097826c`, `feed_crude_ore_20261123_170247_ecdbbc3d` → `imp_20261123_170450_3a55f38f` → `const_20261123_170806_4991b908` → `opt_20261123_171426_e335ec99` (30/30 solved, cold) → `uncprop_20261123_174126_284c8b48`, `psut_20261123_174150_689caf93`.
- **Wave 1:** Stages 5–6 on `a483e4a`, which held #214 and #286, from the reference opt run: `uncprop_20261123_162837_82b064de`, `psut_20261123_162956_366282e0`.
- **The acc build:** `scripts/acc.py --id copper` on `b08d860`, from a copy of `data/input/acc/copper/current/` into a scratch input root and a scratch output directory. `acc_store.load` read both it and the persisted `data/proc/acc/copper`, which was written by 0.1.0rc1 and read through the old layout.
- **The ICSG parse layer:** `icsg_clean.clean_icsg_flows(raw, 28, …)` on `b08d860` into a scratch directory, and `icsg_comb.main` from the twelve vintage tables into another. Both were compared with the registered icsg input and the files under `data/proc/feedprep/icsg_clean/`.
- **Toy:** `python -m pysuto run --case toy --from feed --to psut` twice into one scratch data directory, first with main's code (`7135c54`) and then with `b08d860`. Leaves: `psut_20261123_170312_ff32591f` + `uncprop_20261123_170307_e0f3cab0` (main) and `psut_20261123_170334_4593a058` + `uncprop_20261123_170330_402e1a11` (reviewed).

**Against:** the reference chain pinned for the final review (#155, amendment 4): `feed_icsg_20261119_125015_e2e9520e`, `feed_baci_20261022_110715_fdf3b4ac`, `feed_crude_ore_20260603_095651_558f7369` → `imp_20261119_125259_f2b11cd5` → `const_20261119_125651_ff7bfc9c` → `opt_20261119_130302_b20f0dba` → `uncprop_20261119_133344_08dfa15b`, `psut_20261119_133504_4c783a88`.

**Chain:**
- **Code:** `b08d860` is `v0.1.0rc1` (84572e4) plus the 25 fix tickets of milestone `final review`, merged through `review/integration`: #203, #210–#225, #243 and #278–#286. Its full suite passed (2,404 tests).
- **Data:** the main checkout's `data/`, with no other driver running. 25 GB free before the chain, 24 GB after.

**Feeds/params:**
- icsg v18, baci v33, crude_ore v3, icsg_aggregates v4, ore_grade v8.
- trade_shares **v44**, registered by this Stage 1.
- acc copper v5.
- `input_status.py`: every `current/` matches its latest registered version.

**Date:** 2026-11-23 · **Branch:** `review/integration` · **Issues:** #182 (the final gate), #155 (the behaviour rule, amendment 6) and the fix tickets above · **ADR:** 0042 (one QP per period), 0047 (the case outside the package)

**Verdict:**
- **The reviewed code moves no number.** Every stored array from the feeds to psut is identical to the reference chain: 1,102 arrays.
- **Only by design:** the feeds' `parents.inputs` differ (§1).
- **Acc build:** every array a consumer reads is identical, and the eight files both layouts write are byte-identical.
- **ICSG parse layer:** its six outputs are byte-identical.
- **Toy chain:** identical between main's code and the reviewed code, 248 arrays.
- **Footprint (§2):** a stage's peak memory falls by 35–92 %. Stage 1 runs in a quarter of its time and Stage 6 in a sixth, because a stage now loads the accounting structure in under a second instead of 27 s and 10 GB (#243).

## 1. Comparisons (`python -m pysuto compare`)

| Comparison | Leaves | Result | Arrays |
|---|---|---|---|
| Wave 2: copper from Stage 1 vs the reference | base `psut_20261119_133504_4c783a88 uncprop_20261119_133344_08dfa15b`, new `psut_20261123_174150_689caf93 uncprop_20261123_174126_284c8b48` | every array identical; the feeds differ in `parents.inputs` | 1,102 (feeds 6 + 6 + 7, imp 24, const 303, opt 240, uncprop 150, psut 366) |
| Wave 1: Stages 5–6 vs the reference | base as above, new `psut_20261123_162956_366282e0 uncprop_20261123_162837_82b064de` | exit 0, identical | 516 new (uncprop 150, psut 366); upstream the reference's own runs |
| Toy: main's code vs reviewed code | the four toy leaves above | exit 0, identical | 248 (feeds 15, imp 24, const 78, opt 40, uncprop 25, psut 66) |

- **Feeds:** the three feeds' arrays are identical. Their `parents.inputs` differ, as every Stage-1 check chain's do (#161, S3-3):
  - the reference's BACI and crude-ore feeds were built on acc copper v4 and v3, and the rebuild records v5, the version the reference imp already reads;
  - the reference's ICSG feed names trade_shares v41, and the rebuild names v44, which this Stage 1 registered naming the new BACI feed run.
- **Imp:** built in the reference's feed order, so no per-feed array is permuted (cf. #161's S3-2 and #211's alignment).

| Acc build: item | Result |
|---|---|
| `edge_index` (854,460 edges), `node_index`, bounds `l` and `u` | identical |
| masks `cross_region`, `massbal_scope`, `sb`, `supply`, `tradable`, `use` | identical |
| `boundary_roles`, `valuation_roles` | identical |
| `edge_index.csv`, `node_index.csv`, `root_nodes.csv`, `bounds_l.npy`, `bounds_u.npy`, `masks_user.npz`, `edge_summary.csv`, `sector_pair_counts.csv` | byte-identical |

The new layout writes `metadata.json` and no `edge_index_all.csv` (1.8 GB), `A.npz`, `A_nodes.npz` or `metadata.pkl` (#243).

| ICSG parse layer: file | sha256 (first 16) | Result |
|---|---|---|
| `icsg_flows_clean.csv` = `data/input/feed/icsg/current/values.csv` | `5f0f6e9f323b65c8` | byte-identical |
| `icsg_secondary_excess.csv` | `f49ea09df2020dbe` | byte-identical |
| `icsg_imported_anode.csv` | `9541b6c490b08279` | byte-identical |
| `icsg_flows.csv`, `icsg_prices.csv`, `icsg_stocks.csv` (`icsg_comb`) | `74e97812…`, `3c8aabea…`, `71c04f57…` | byte-identical |

- **The clean step's parents record** gains `eff` v4 and `icsg_aggregates` v4 (U1b-G4-8), so a `--to-feed` would register a new icsg version with the same `values.csv`.
- **The vintage parsers** (`icsg_2005`, `icsg_2014`, `icsg_2023f`) were not rerun, because they write fixed paths under `data/`. #281 changed them only by deleting commented code, renaming, and rewriting headers and docstrings.

## 2. Footprint (`/usr/bin/time -v`, 8 cores)

Before: #210's Stage-1 check chain (2026-11-23, `5fc966d`: rc1 plus the sidecar fields), which possibly shared the cores with other work. After: wave 2 on `b08d860`, on an otherwise idle machine.

| Stage | Wall, before | Wall, after | Peak RSS, before | Peak RSS, after |
|---|---|---|---|---|
| feed (the three builds) | 2:41 | 0:37 | 16.8 GB | 9.0 GB |
| imp | 2:50 | 2:22 | 12.2 GB | 4.4 GB |
| const | 7:53 | 7:25 | 15.2 GB | 7.4 GB |
| opt (solve + diag) | 2:51 | 2:41 | 14.3 GB | 6.2 GB |
| uncprop (solve + diag) | 26:30 | 26:30 | 14.6 GB | 9.4 GB |
| psut | 0:33 | 0:05 | 10.8 GB | 0.8 GB |

- **Why memory falls:** each driver loads the copper acc. That load took 27–36 s and 10.3–10.8 GB before #243, and takes 0.8 s and 0.40 GB after it. Building the acc took 2:48 and 10.8 GB, and takes 7.7 s and 0.54 GB.
- **Stage 5:** still 60 % of the chain's wall time.

## 3. What the chain covers

| First step a fix touches | Tickets |
|---|---|
| acc build | #215, #278, #243 (and #279: `scripts/acc.py` takes the stage lock) |
| parse layer, Stage 1 | #281, #280, #222, #224 (p), #216 (`meta.py`), #279 (the lock, the knob record) |
| Stage 2 | #282, #224 (the Lp combiner), #217 (imports) |
| Stage 3 | #283, #224 (the σ arguments), #220 |
| Stage 4 | #284, #219 |
| Stage 5 | #214, #285 |
| Stage 6 | #286 |
| diagnostics, docs, tests only | #203, #213, #218, #221, #223, #225 |

The builder, imputer and feed changes this rule asks a check REP for (#280, #281, #282, #283, #224) are covered here. #283's own REP (`analysis_const_20261119_125651_ff7bfc9c_20261123_u3-build-checks.md`) holds its static evidence.

## 4. Runs for the maintainer

- **Pruned after this REP:** wave 1's and wave 2's Stage 2–6 runs were pruned under the maintainer's standing OK of 2026-11-23, once the fix tickets' closing comments recorded them:
  - `imp_20261123_170450_3a55f38f`, `const_20261123_170806_4991b908`, `opt_20261123_171426_e335ec99`, `uncprop_20261123_174126_284c8b48`, `psut_20261123_174150_689caf93`;
  - `uncprop_20261123_162837_82b064de`, `psut_20261123_162956_366282e0`.
- **Kept:** the three wave-2 feed runs (≈ 110 MB), because trade_shares v44 names `feed_baci_20261123_170234_2097826c`.
- **Not under `data/`:** the toy and scratch runs live in the session scratchpad.
