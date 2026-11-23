# #161: a copper check chain from Stage 1 reproduces the reference chain's 1,059 const-to-psut arrays bit for bit, one Stage-4 worker equals eight, and the toy reruns identically

**Run:**
- **The chain:** one stage driver at a time under the stage lock, from a frozen copy of main `6f574cd`, each driver under `/usr/bin/time -v`. Label `#161 check chain`; every knob at its default.
  - Stage 1: `python -m pysuto run --case scripts/copper.py --data-dir data --from feed --to feed`.
  - Stage 2: `collect_aggregate_impute.py` on the three new feeds.
  - Stages 3–6: `constraints.py`, `run_solver.py`, `uncertainty_propagation.py`, `psut.py`.
  - Run ids: `feed_baci_20261123_094401_2962067d`, `feed_icsg_20261123_094513_e6f2b299`, `feed_crude_ore_20261123_094544_ab880f44` → `imp_20261123_094827_92b6d103` → `const_20261123_095218_fb9afcb8` → `opt_20261123_095827_450ed9ab` (30/30 solved, cold) → `uncprop_20261123_102533_7e0cc120`, `psut_20261123_102653_b4d5796d`.
- **One Stage-4 worker:** `run_solver.py --const-run-id const_20261123_095218_fb9afcb8 --set N_WORKERS=1` → `opt_20261123_103738_8d684a55`.
- **Toy:** `python -m pysuto run --case toy` twice into one scratch data directory. Leaves `psut_20261123_103948_ff30b681` + `uncprop_20261123_103943_7baaa18b` (a) and `psut_20261123_104011_e28990c6` + `uncprop_20261123_104006_7b36d7d8` (b).

**Against:** the reference chain pinned for the final review (#155, amendment 4): `feed_icsg_20261119_125015_e2e9520e`, `feed_baci_20261022_110715_fdf3b4ac`, `feed_crude_ore_20260603_095651_558f7369` → `imp_20261119_125259_f2b11cd5` → `const_20261119_125651_ff7bfc9c` → `opt_20261119_130302_b20f0dba` → `uncprop_20261119_133344_08dfa15b`, `psut_20261119_133504_4c783a88`.

**Chain:**
- **Code:** main `6f574cd` = the audited commit `v0.1.0rc1` (84572e4), plus #200's `PRODUCTION_EDGES` (the code of the reference chain) and #157's `compare` subcommand.
- **Data:** the main checkout's `data/`, with no other driver running. `df -h`: 8.9 GB free before, 7.6 GB after.

**Feeds/params:**
- icsg v18, baci v33 (the netting rebuilt byte-identical, no bump), crude_ore v3, icsg_aggregates v4, ore_grade v8.
- trade_shares **v42**, registered by this Stage 1 (§2).
- acc copper v5; conc v10, eff v4, stripping_ratio v2, stock_bounds v6, scrap_share v3. `input_status.py`: every `current/` at its latest version.

**Date:** 2026-11-23 · **Branch:** `diag/check-chain-161` · **Issues:** #161 (S3, G9 of #155) · **ADR:** 0042 (one QP per period, threads), 0047 (the case outside the package)

**Verdict:**
- **Const through psut reproduce bit for bit:** const, opt, uncprop and psut, 1,059 arrays.
- **One Stage-4 worker equals eight:** 240 opt arrays.
- **The toy chain reruns identically:** 248 arrays.
- **Not bit-identical, as stored:**
  - The three feeds' arrays are identical, but their `parents.inputs` are not.
  - The imp's per-feed arrays are permuted, because Stage 2 stores them in the order of `--feed-id`.
  - Neither difference moves a number.
- **Footprint:** a chain takes 44 min on 8 cores. Stage 5 is 60 % of it, and every stage holds 10.8–16.8 GB of resident memory at its peak.
- **Findings:** four for S3 and one for G9 (§5). None moves a number.

## 1. Comparisons (`python -m pysuto compare`)

| Comparison | Command leaves | Result | Arrays |
|---|---|---|---|
| Copper check chain vs reference | base `psut_20261119_133504_4c783a88 uncprop_20261119_133344_08dfa15b`, new `psut_20261123_102653_b4d5796d uncprop_20261123_102533_7e0cc120` | exit 1: feeds and imp differ as stored, const–psut identical | 1,059 identical (const 303, opt 240, uncprop 150, psut 366) |
| Stage 4: 1 worker vs 8 | base `opt_20261123_095827_450ed9ab`, new `opt_20261123_103738_8d684a55` | exit 0, identical | 240 |
| Toy chain, run a vs run b | the four toy leaves above | exit 0, identical | 248 (feeds 15, imp 24, const 78, opt 40, uncprop 25, psut 66) |

- **Feeds:** BACI's 6 arrays, ICSG's 7 and crude ore's 6 are identical. `parents.inputs` differ in two ways:
  - **acc version:** the reference's BACI and crude-ore feeds were built in October and June on acc copper v4 and v3. The rebuild records v5, the version the reference imp already reads.
  - **`trade_shares`:** the reference's ICSG feed names v41, the rebuild v42. v42's `params.csv` is byte-identical to v41's; only its `parents.json` differs, because it names the new BACI feed run (finding S3-3).
- **imp:** `agg_result.npz::F`, `Sigma`, `Omega` and `feed_ids` differ at 123,348 of 2,563,380 entries.
  - The cause is order: the reference imp was built with `--feed-id icsg baci crude_ore`, the check chain with `baci crude_ore icsg`.
  - Reordered by `feed_ids`, all four arrays are equal, and the other 20 imp arrays (`f_agg`, `sigma_agg`, the imputation log, …) are identical (finding S3-2).
  - Every downstream array is identical: Stage 2 is order-invariant in what it passes on.
- **Workers:** `solve_wall_s` 94.0 s (8 threads) and 639.7 s (1 thread). Σ per-period solve times are 586.4 s and 574.2 s. The 1-thread run reruns the feasibility probe serially (63.7 s).

## 2. RNG and environment

**RNG:**
- Every random-number use in `src/` and the in-scope `scripts/` is one CLAUDE.md allows:
  - the seeded scrambled Latin hypercube of a random sweep (`pipeline.Sweep`);
  - Stage 5's draws, which take the caller's `np.random.Generator` (`uncprop.py`, raising without one);
  - the toy's `default_rng([seed, k])` for the ground truth, the synthetic feeds and `toy_ext.fourth_feed`;
  - uuid4 run ids.
- One more uuid4 use, not in CLAUDE.md's list: the 8-hex `code` of every input version (`input_registry.py:114`, `migrate_registries.py:82`). It is an identifier, not a number (finding S3-4).
- Determinism across Python's hash seed is covered by the check chain: every stage ran in a fresh process.

**Environment:**
- A sidecar records the case (`case_source`, `case_sha256`), the input versions and the knobs.
- It does not record:
  - the code version or commit;
  - the Python version;
  - the numerical stack's versions (numpy, scipy, pandas, pyarrow, clarabel).
- So a run cannot say which code produced it, and the bit-identity claim of `environment-reference.yml` (#194) cannot be tied to a run (finding S3-1).

## 3. Footprint per stage (G9)

| Stage | Wall, check chain | Wall, reference | Peak RSS | CPU | Disk per run | Time in the sidecar | Top hotspots (from the run logs) |
|---|---|---|---|---|---|---|---|
| 1 feeds | 2:45 | — | 16.8 GB | 99 % | 109 MB (BACI 46, ICSG 42, crude ore 21) | none | acc load 27 s × 3 (49 %); the two build scripts 32 s; BACI's user weights 16 s |
| 2 imp | 3:02 | 3:01 | 12.1 GB | 99 % | 81 MB | none | engine construction after the last params read, 79 s (44 %); acc load 27 s |
| 3 const | 8:02 | 8:05 | 15.2 GB | 100 % | 852 MB | none | diagnostic figures 229 s (48 %); conc params read 36 + 22 s; acc load 27 s; lower-bound rows 23 s |
| 4 opt | 2:55 | 2:57 | 14.5 GB | 428 % | 74 MB | `solve_wall_s` 94.9 → 94.0 s | G load 27 s; acc masks 20 s; 30 periods of 16–22 s each on 8 threads |
| 4 opt, 1 worker | 12:01 | — | 13.6 GB | 106 % | 74 MB | `solve_wall_s` 639.7 s | feasibility probe 64 s; 30 periods serial |
| 5 uncprop | 26:36 | 27:12 | 14.7 GB | 728 % | 29 MB | none | the period loop: 30 periods of 45–60 s each, serial, each threaded (≈ 7 cores); run load 61 s |
| 6 psut | 0:33 | 0:33 | 10.8 GB | 99 % | 11 MB | none | opt → const → acc load 27 s (83 %) |
| Toy chain | 0:23 | — | 0.4 GB | — | — | — | — |

- **What "reference" means:** the reference walls are the same drivers' runs on 2026-11-19.
  - Stages 2–4 come from #201's measurement (driver logs).
  - Stages 5–6 come from its pipeline step logs.
  - The reference's Stage 1 cannot be compared: its three feeds were built on 2026-06-03, 2026-10-22 and 2026-11-19.
- **Hotspots** are the longest gaps between consecutive log lines, charged to the line that opens each gap. They are named, not fixed (#161). A sampling profile was not taken.
  - **acc load:** `AccountingStructure.load` takes ≈ 27 s and runs eight times per chain, ≈ 3.6 min of 44.
  - **Stage 3's diagnostics** take nearly half of its wall time.
  - **Stage 5's period loop** holds 60 % of the chain.
- **Disk:** a whole chain needs ≈ 1.16 GB, three quarters of it the const run. The `trade_shares` version Stage 1 registers costs almost nothing: v42's `params.csv` is a hardlink of v41's (one inode, two links).

## 4. What was not checked

- Bit-identity on another machine, or on the pinned reference environment `environment-reference.yml`, was not checked. This run used the `pysuto` env, which is that environment plus `coverage` and `vulture` (#156).
- The Stage-1 parse steps (BACI parse, `icsg_clean --to-feed`) read `data/input/raw` and stay manual (CLAUDE.md), so the check starts at the registered inputs.

## 5. Review findings (#155's format, at `v0.1.0rc1`)

| # | File:line | Dimension | What is wrong | Proposed action | Moves a number |
|---|---|---|---|---|---|
| S3-1 | `src/pysuto/registry/_sidecar.py:115` (`build_sidecar`) | S3 | no sidecar records the code version, the Python version or the numerical stack's versions | record `pysuto.__version__` (hatch-vcs: tag, distance, sha), Python and numpy, scipy, pandas, pyarrow, clarabel in every sidecar | no |
| G9-1 | `src/pysuto/registry/_sidecar.py:115` | G9 | only opt sidecars record a time (`solve_wall_s`); feed, imp, const, uncprop and psut record none, so a run's footprint is not traceable from its sidecar | record the stage's wall time (and peak RSS where cheap) in every sidecar | no |
| S3-2 | `src/pysuto/collector.py:97` (`FeedCollector.collect`) | S3 | the stored per-feed arrays (`F`, `Sigma`, `Omega`, `feed_ids`) follow the `--feed-id` order, so the same feeds in another order persist a different imp; every downstream array is identical | `python -m pysuto compare` aligns `agg_result`'s per-feed arrays by `feed_ids` before comparing (a canonical order in Stage 2 would change the stored reference) | no |
| S3-3 | `scripts/copper_feed.py:976` (`run_stage1` → `build_trade_shares.py`) | S3 | every Stage 1 registers a new `trade_shares` version, even with a byte-identical `params.csv`, because `parents.json` names the new BACI feed run (#53's guard checks by run id); a check chain from Stage 1 can never reproduce the reference's `parents.inputs`, and each one adds an input version | document it beside the comparison command (README *Workflow*, CLAUDE.md); a content-addressed guard is a candidate for the U1b architecture session (#175) | no |
| S3-4 | `CLAUDE.md:104` (RNG list), `src/pysuto/input_registry.py:114` | S3 | the list of allowed random uses omits the uuid4 code of every input version | add it to the list | no |

No finding moves a number or questions a modelling choice recorded in an ADR.
