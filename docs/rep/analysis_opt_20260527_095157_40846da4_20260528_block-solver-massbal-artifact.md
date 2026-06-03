# Per-t block solver stores cross-year prior placeholders — opt_20260527_095157_40846da4

Investigation triggered by user observations comparing the new "temp-aware G"
run against the prior run
[opt_20260526_165810_3fd6d527](../../data/proc/opt/opt_20260526_165810_3fd6d527/):
*(1) `a` is still heavily downscaled across periods, and the distribution shifts
from a unimodal `a₀` in 1995 to bimodal where a high mode (median ~10³) collapses
to almost nothing by 2024; (2) the mass-balance deviation is enormous —
hypothesis: mass-balance constraints are not properly initialised after the new
G init.*

Upstream chain (start run):
- opt run:   [opt_20260527_095157_40846da4](../../data/proc/opt/opt_20260527_095157_40846da4/)
- const run: [const_20260527_094403_4bd9e613](../../data/proc/const/const_20260527_094403_4bd9e613/) — **has a `blocks/` dir** (per-t `G_blocks` + `edge_cols`)
- imp run:   [imp_20260527_072032_c1a26edb](../../data/proc/imp/imp_20260527_072032_c1a26edb/) (new; "time imp masking")
- feeds:     `icsg`, `crude_ore`, `baci`

Comparison run:
- opt run:   [opt_20260526_165810_3fd6d527](../../data/proc/opt/opt_20260526_165810_3fd6d527/)
- const run: [const_20260526_133303_023a9149](../../data/proc/const/const_20260526_133303_023a9149/) — **no `blocks/` dir** (legacy full-G)
- imp run:   [imp_20260521_074525_b1921faf](../../data/proc/imp/imp_20260521_074525_b1921faf/)

Config knobs (`stage_config`, identical for both const runs):
- `sigma_a_cv_min: 0.05`, `sigma_a_abs_floor: 1e-6`
- `sigma_c_cv_min: 0.05`, `sigma_c_abs_floor: 1e-3`
- `p: 1.3`

---

## 0 — Executive summary

Both reported symptoms are the **same artifact of how the new per-t block solver
stores its results**, not a constraint-initialisation failure. The "temp-aware G"
change ([`_solve_all_via_blocks`](../../src/pysuto/solver.py#L635), committed in
`8bd3783` on 2026-05-28 — *after* both runs, so run 27 ran the WIP form)
partitions the 854,460 edges into a disjoint set of **28,482 edges × 30 years**
and solves one small QP per year. But each per-t `OptimResult` stores the **full
854,460-edge vector** with the other 29/30 of edges left at their **prior `a₀`**
([`solver.py:771-772`](../../src/pysuto/solver.py#L771-L772)), and computes
residuals over the **full G** ([`solver.py:776`](../../src/pysuto/solver.py#L776)).

Consequences:
- **Mass balance is satisfied to ~1e-12 for the year actually solved.** The
  "enormous deviation" (max **16,871,723**) is *exactly* `max|G_mb · a₀|`, the
  prior mass-balance infeasibility, peaking at `mass_balance_node55_t2024_Mass` —
  it appears only on *other* years' rows because those edges are still at the
  prior in that file.
- **The "bimodal → collapse" distribution** is the full-vector histogram mixing
  the ~28k solved edges (median 0.0074) with ~826k prior placeholders (median
  ~147 ≈ the "10³ mode"); as the warm-start accumulator
  ([`solver.py:802`](../../src/pysuto/solver.py#L802)) fills in year by year, the
  placeholder mode shrinks, so by 2024 (last timestep, all edges touched) it
  looks unimodal and "collapsed".

The **genuine** downscaling the user still sees (year-local solved flows median
0.0074, max 0.13 vs priors up to 8.5e6) is **real but pre-existing** — identical
in run 26 — and is fully root-caused in
[analysis_opt_20260526_165810_3fd6d527_20260527_flow-collapse.md](analysis_opt_20260526_165810_3fd6d527_20260527_flow-collapse.md).
The temp-aware-G change neither caused nor fixed it.

> **Verdict.** The mass-balance constraints are correctly initialised (cone=zero,
> c=0, satisfied to 1e-12 for the solved year). Both symptoms are a **storage /
> diagnostic artifact** of the new block solver writing full-length vectors per
> timestep with unsolved edges pinned at the prior `a₀`, then evaluating the full
> G and the full distribution against that mostly-prior vector. The persistent
> downscaling is a separate, pre-existing structural QP degeneracy.

---

## 1 — Observations as given

1. `a` is **still** heavily downscaled across periods. *(The "still" is key — this
   is the pre-existing collapse, Finding B.)*
2. Distribution shifts from a unimodal `a₀` in 1995 to bimodal; the high mode
   (median ~10³) collapses to almost nothing by 2024.
3. The mass-balance deviation is enormous; hypothesis: mass-balance constraints
   not properly initialised after the new G init.

Observations (2) and (3) are interdependent — both are the same full-vector
storage artifact. Observation (1) is a distinct, pre-existing issue.

---

## 2 — Pipeline trace setup

| Observation | Owning stage | First place to look |
|---|---|---|
| (3) enormous mass-balance deviation | Stage 4 result **storage** | [`solver.py:771-776`](../../src/pysuto/solver.py#L771-L776) (`F_opt_global`, `residuals_global`) |
| (2) bimodal → collapse distribution | Stage 4 storage + Stage 4 diag | [`solver.py:771`](../../src/pysuto/solver.py#L771); [`diag/solver.py:304`](../../src/pysuto/diag/solver.py#L304) |
| (1) flows still downscaled | Stage 3/4 QP (structural) | prior REP — [flow-collapse](analysis_opt_20260526_165810_3fd6d527_20260527_flow-collapse.md) |

**Run-wiring facts that frame everything below.**

| | run 27 (start) | run 26 (prior) |
|---|---|---|
| solver path | `_solve_all_via_blocks` | `_solve_all_legacy` |
| const has `blocks/` | **yes** | no |
| imp parent | imp_20260527 (new) | imp_20260521 |
| per-timestep solve time | **~7 s** | ~380 s |
| iterations / t | 23–34 | 37–61 |
| solver_status (all t) | Solved | Solved |

Both const runs have **byte-identical `G.npz`** (shape `1,394,490 × 854,460`,
nnz 5,676,660; mass_balance rows = 39,480, all `cone_type=zero`, `is_hard=True`,
`c=0`, `σ_c=0`). `a0.npy` differs in only 3,508 / 854,460 entries, `sigma_a.npy`
in 27,746; `u.npy` / `l.npy` are identical. So the *constraint system* is
effectively the same — the only behavioural difference is the **solver path**,
selected by the presence of `run.G_blocks` ([`solver.py:470-475`](../../src/pysuto/solver.py#L470-L475)).

The "temp aware G" commit `8bd3783` is dated **2026-05-28**, *after* both runs
(run 27's const: 2026-05-27 09:44). Run 27 therefore executed the work-in-progress
form of that code; run 26 predates it entirely. This is why the same `solver.py`
in git history is *not* the code both runs ran.

---

## 3 — Data characterisation

### 3.1 Mass-balance residual, split same-year vs other-year (run 27)

Per per-t file, mass-balance rows split into those that **belong to the solved
year** (`row_meta.time == t`) and those that belong to other years:

| opt year | cols solved | max\|res\| **same-year** mb | max\|res\| **other-year** mb | med\|F[cols_t]\| | max\|F[cols_t]\| | nnz cols_t |
|---|---:|---:|---:|---:|---:|---:|
| 1995 | 28,482 | **1.6e-12** | 1.69e7 | 0.0074 | 0.130 | 2,507 |
| 2000 | 28,482 | **1.5e-12** | 1.69e7 | 0.0094 | 0.130 | 2,487 |
| 2010 | 28,482 | **1.5e-12** | 1.69e7 | 0.0098 | 0.130 | 2,463 |
| 2018 | 28,482 | **2.0e-12** | 1.69e7 | 0.0092 | 0.133 | 2,450 |
| 2024 | 28,482 | **3.1e-13** | 5.5e-11 | 0.0075 | 0.130 | 2,479 |

The year that is actually solved satisfies mass balance to solver tolerance. The
1.69e7 lives **only** on the other 29 years' rows. 2024 is "clean" everywhere
only because it is the **last** timestep — by then the warm-start accumulator has
overwritten every year's edges with a solved value (see §4.4).

### 3.2 The 1.69e7 *is* the prior infeasibility

For the start run, `max|G_mb · a₀|` (the pure prior mass-balance residual, `c=0`)
is **16,871,722.98**, with the peak at row `mass_balance_node55_t2024_Mass`. The
max other-year residual in the 1995 file is **16,871,722.98** — *bit-identical*.
The "enormous deviation" is the prior infeasibility (documented in the prior REP
§3.4/§4) re-surfacing on every year whose edges have not yet been solved in the
file being inspected.

The diagnostic the user looked at, SD-R-02
([`diag/solver.py:304`](../../src/pysuto/diag/solver.py#L304)), reports
`max|residuals[mass_balance]|` across **all** mass_balance rows per timestep — so
it reports 1.7e7 for every year except the last.

### 3.3 The "bimodal" distribution decomposed (1995 file, full vector)

`log10|F_opt|` over all 854,460 edges in the t_1995 file:

| bucket | count | what it is |
|---|---:|---|
| `10^-4` | 164 | solved 1995 edges |
| `10^-3` | 1,643 | solved 1995 edges |
| `10^-2` | 26,123 | **solved 1995 edges (low mode)** |
| `10^-1` | 21,864 | solved 1995 edges |
| `10^0`  | 37,995 | mixed |
| `10^1`  | 88,849 | unsolved priors |
| `10^2`  | 161,648 | **unsolved priors (high mode ≈ user's "10³")** |
| `10^3`  | 42,754 | unsolved priors |
| `10^4`  | 2,723 | unsolved priors |
| `10^5`  | 647 | unsolved priors |
| `10^6`  | 383 | unsolved priors |

Restricted to the **solved** edges (`edge_cols_by_t['1995']`) the distribution is
unimodal and small: median 0.0074, max 0.13. Restricted to the **unsolved**
edges it is exactly the prior `a₀`: median 147.3, max 17,130,904 — and
`F[unsolved] == a₀[unsolved]` holds bit-for-bit. So the bimodality is purely the
superposition of "this year's solution" and "every other year's prior".

### 3.4 Genuine year-local downscaling (Finding B — pre-existing)

The actually-solved edges are tiny relative to their priors, in **both** runs:

| run | year | prior `a₀[cols]` med / max / nnz | posterior `F[cols]` med / max / nnz |
|---|---|---|---|
| 27 (block) | 1995 | 126 / 8.46e6 / 11,769 | 0.0074 / 0.13 / 2,507 |
| 26 (legacy, full = year) | 1995 | 148 / 1.71e7 / — | 0.0069 / 0.15 / — |

post/prior ratio on solved edges has **median 0.0** — most flows collapse to
(near-)zero. This is the same collapse as run 26 and is root-caused separately.

---

## 4 — Pipeline trace of one specific case

**Picked case.** Edge **3838** = `Mining/Process/Chile → WR/SB/Chile (SEM)`,
`time=2024`, `valuation=Mass`, prior `a₀ = 17,130,904` (≈ 17.1 Mt). It is the
single largest prior in the system, and `edge_cols_by_t` assigns it to year
**2024**.

### 4.1 The giant edge is a placeholder until its own year is solved

`F_opt[3838]` read straight from each per-t file:

| opt file | F_opt[3838] | meaning |
|---|---:|---|
| t_1995 | 17,130,904.0 | placeholder = `a₀` (year 2024 not yet solved) |
| t_2010 | 17,130,904.0 | placeholder = `a₀` |
| t_2024 | 0.0 | **solved** in its own year → collapses |

So the "high mode collapsing by 2024" is, literally, the year-2024-owned edges
(of which 3838 is the extreme) sitting at their large priors in every earlier
file and only being solved — and collapsing — in the final t_2024 file. `max|F|`
in the per-t files is pinned at 1.713e7 for 1995–2023 and drops to 0.13 at 2024
for exactly this reason.

### 4.2 The same node drives the mass-balance "deviation"

Row `mass_balance_node55_t2024_Mass` (node 55 = Chile Mining/Process) is the peak
of `|G_mb · a₀|` at 1.687e7 — identical to the worst case in the prior REP §4.1
(there 1.76e7, the small difference is the new imp's slightly different `a₀`). In
run 27 this row is satisfied to ~1e-12 in the **t_2024** file (its own year) but
shows 1.687e7 in **every other** per-t file, because edge 3838 (and its
mass-balance siblings) are at the prior there. This single node accounts for the
"enormous mass balance deviation" headline number.

### 4.3 What the block solver actually computed for 2024

In t_2024, edge 3838 is in `cols_t`; the QP collapses it to 0 (the cheap
direction — see prior REP §4.4), the node-55 mass balance closes, and the stored
residual is ~1e-12. This is correct solver behaviour given the (pre-existing)
QP degeneracy; the giant value never existed in the *solution*, only in the
*prior placeholder* carried in earlier files.

### 4.4 Why 2024 is artifact-free everywhere

The warm-start accumulator `a_0_t_global[cols_t] = F_opt_t`
([`solver.py:802`](../../src/pysuto/solver.py#L802)) overwrites each year's edges
with their solved values as the loop advances. By the final timestep (2024) every
one of the 30 column-blocks has been overwritten, so the full vector stored in
t_2024 is "complete" and all 39,480 mass-balance rows evaluate to ~0. Any earlier
file is missing the not-yet-reached years.

---

## 5 — Hypotheses

### H1 — Per-t full-vector storage + full-G residual produces cross-year contamination

**Mechanism.** [`solver.py:771-776`](../../src/pysuto/solver.py#L771-L776):
`F_opt_global = a_0_t_global.copy(); F_opt_global[cols_t] = F_opt_t` then
`residuals_global = run.G @ F_opt_global - c_t_full`. Edges outside `cols_t` keep
the prior; the residual over the full G then reports the prior infeasibility on
all other years' rows. The distribution diag reads the same full vector.

**Predicted signature.** (a) same-year mass-balance residual ≈ 0; (b) other-year
residual ≈ `max|G_mb·a₀|`; (c) unsolved entries of `F_opt` bit-equal `a₀`; (d)
last timestep artifact-free.

**Falsifier.** If the same-year residual were also large, the constraints would
genuinely be mis-wired (→ H2 instead).

### H2 — (User's hypothesis) Mass-balance constraints not initialised after new G init

**Mechanism.** The new G init writes wrong cone type / RHS / σ_c for mass-balance
rows, so they are not enforced.

**Predicted signature.** mass_balance rows would have `cone_type ≠ zero`, or
`c ≠ 0`, or the *same-year* residual would be large.

**Falsifier.** Any of: cone=zero, c=0, same-year residual ≈ 0.

### H3 — The persistent downscaling is a separate pre-existing QP degeneracy

**Mechanism.** Documented in the prior REP: coverage gap at the Mining inflow
boundary + Stage-3 σ_a relative floor + ratio_eq rows active across all years +
`a=0` feasible for all homogeneous rows → degenerate optimum at `a≈0`.

**Predicted signature.** Year-local collapse present in run 26 too, unaffected by
the block change.

**Falsifier.** If run 26's year solution did **not** collapse, the block change
would be implicated.

---

## 6 — Test strategy

| H | experiment | expected if true | expected if false |
|---|---|---|---|
| H1 | split mb residual same-year vs other-year per file; compare other-year max to `max|G_mb·a₀|`; check `F[unsolved]==a₀` | same-year ≈ 1e-12; other-year == prior infeasibility; equality holds | same-year large |
| H2 | inspect `row_meta` cone/`c`/`σ_c` for mass_balance; read same-year residual | cone=zero, c=0, residual≈0 → **rejected** | cone≠zero or c≠0 |
| H3 | recompute run 26 year solution magnitude | collapse identical in run 26 | run 26 not collapsed |

All three are answerable from artefacts on disk — no re-solve required.

---

## 7 — Test results

| H | result | evidence |
|---|---|---|
| H1 | **confirmed** | §3.1 same-year max 1.6e-12 vs other-year 1.69e7; §3.2 other-year == `max|G_mb·a₀|` = 16,871,722.98 (bit-identical); §3.3 `F[unsolved]==a₀` exact; §4.1 giant edge pinned then collapses in its own year; 2024 artifact-free. |
| H2 | **rejected** | mass_balance: 39,480 rows all `cone_type=zero`, `is_hard=True`, `c=0`, `σ_c=0` in **both** const runs; G.npz byte-identical between runs; same-year residual 1e-12. Constraints are correctly initialised and enforced. |
| H3 | **confirmed (separate, pre-existing)** | §3.4: run 27 year-local med 0.0074 / max 0.13; run 26 med 0.0069 / max 0.15 — same collapse. Fully traced in [flow-collapse REP](analysis_opt_20260526_165810_3fd6d527_20260527_flow-collapse.md). |

### 7.1 Verdict

The user's hypothesis is **directionally right but mechanistically wrong**: the
problem *is* tied to the new G init, but the mass-balance constraints themselves
are correctly initialised and satisfied (1e-12 for the solved year). The
"enormous deviation" and the "bimodal collapse" are **one and the same storage
artifact**: `_solve_all_via_blocks` solves a per-year sub-QP but persists the
**full 854,460-edge vector** with all other years' edges frozen at the prior
`a₀`, and then computes residuals (and the diag plots read the distribution) over
that mostly-prior vector. Evaluating the full G against it surfaces the prior
mass-balance infeasibility (`max|G_mb·a₀| = 1.687e7`) on every unsolved year, and
the full-vector histogram superimposes the tiny solved mode on the large prior
mode. The genuinely downscaled solution (median 0.0074) is **pre-existing** and
unrelated to the block change.

---

## 8 — Proposed fixes

Two options for the artifact, laid out per request — **not picked**. Note up
front: **neither fixes the genuine downscaling (Finding B)**; that needs the
fixes in the [flow-collapse REP §8](analysis_opt_20260526_165810_3fd6d527_20260527_flow-collapse.md).

### 8.1 Option 1 — Year-scope the residual and the diagnostics (minimal)

**What.** In [`_solve_all_via_blocks`](../../src/pysuto/solver.py#L635), compute
the stored `residuals` over the **solved year's rows only** (`rows_t`) instead of
the full G — e.g. `residuals_rows_t = G_p_t @ F_opt_t - b_t` (already available in
the loop), scattered back into a `rows_t`-only array, with other rows left
`NaN`/omitted. Restrict the mass-balance diag SD-R-02
([`diag/solver.py:294-356`](../../src/pysuto/diag/solver.py#L294-L356)) and the
distribution-of-a diag SD-T-05 to `edge_cols_by_t[t]` so they never read
cross-year placeholders.

**Expected effect.** SD-R-02 max mass-balance residual drops from 1.7e7 to ~1e-12
for every year; SD-T-05 becomes the true unimodal year-local distribution
(median 0.0074). Headline symptoms disappear; no schema change.

**Risk / con.** The stored `F_opt` vector still contains prior placeholders for
non-`cols_t` edges, so any *other* naïve consumer (uncprop, external scripts) that
reads the full vector per file would still be misled. Low blast radius but does
not make the persisted object self-consistent.

### 8.2 Option 2 — Persist only the year-local solution (clean schema)

**What.** Store per-t files containing only `F_opt[cols_t]` and the `rows_t`
residuals (plus the `edge_cols`/row index needed to place them), instead of the
full 854,460-vector. Update consumers — [`uncprop.py`](../../src/pysuto/uncprop.py)
and the optim/solver diags — to assemble the cross-year picture by concatenating
the per-year blocks (the inverse of how `_solve_all_via_blocks` reads them).

**Expected effect.** Each per-t artefact is fully self-consistent (every stored
value is a solved value); residuals and distributions are correct by construction
for any consumer.

**Risk / con.** Larger blast radius — touches the OptimResult schema, the optim
registry, uncprop, and every diag that assumes a full-length `F_opt` per file.
Needs a migration note for existing runs. The legacy path
([`_solve_all_legacy`](../../src/pysuto/solver.py#L478)) still produces full
vectors, so the two paths would diverge in schema unless both are updated.

---

## 9 — Suggested verification

For the artifact fix (either option):
1. Re-render diagnostics for the **existing** run 27 (no re-solve needed for
   Option 1's diag-only part): SD-R-02 should show max mass-balance residual
   ~1e-12 for all years; SD-T-05 should be unimodal around 1e-2.
2. Spot-check: for each year `t`, `max|residuals[row_time==t & type==mass_balance]|`
   must stay ≈ 1e-12 (this already holds — it's the correctness invariant the fix
   must preserve).
3. Regression: confirm the **legacy** path output is unchanged (run 26 re-render
   identical), so the fix only affects the block path.

For the genuine downscaling (Finding B), follow the verification ladder in the
[flow-collapse REP §9](analysis_opt_20260526_165810_3fd6d527_20260527_flow-collapse.md)
(RatioConstraintBuilder dynamic deactivation → coverage extrapolation → σ_a floor
relaxation). That is out of scope for this report.

---

*Generated 2026-05-28. Trace files used:*
- *start opt run: [data/proc/opt/opt_20260527_095157_40846da4/](../../data/proc/opt/opt_20260527_095157_40846da4/) (`metadata.json`, `timesteps/t_*.npz`)*
- *start const run: [data/proc/const/const_20260527_094403_4bd9e613/](../../data/proc/const/const_20260527_094403_4bd9e613/) (`G.npz`, `a0.npy`, `sigma_a.npy`, `u.npy`, `l.npy`, `row_meta.parquet`, `timesteps/t_*.npz`, `blocks/edge_cols.npz`, `blocks/g_*.npz`)*
- *prior opt run: [data/proc/opt/opt_20260526_165810_3fd6d527/](../../data/proc/opt/opt_20260526_165810_3fd6d527/) (`metadata.json`, `timesteps/t_*.npz`)*
- *prior const run: [data/proc/const/const_20260526_133303_023a9149/](../../data/proc/const/const_20260526_133303_023a9149/) (`G.npz`, `a0.npy`, `sigma_a.npy`)*
- *acc tables: [data/proc/acc/copper/](../../data/proc/acc/copper/) (`edge_index.csv`, `node_index.csv`)*
- *solver code: [src/pysuto/solver.py:635-803](../../src/pysuto/solver.py#L635-L803) (`_solve_all_via_blocks`), [src/pysuto/solver.py:470-475](../../src/pysuto/solver.py#L470-L475) (path selection)*
- *diag code: [src/pysuto/diag/solver.py:294-356](../../src/pysuto/diag/solver.py#L294-L356) (SD-R-02 mass-balance panel)*
- *registry: [src/pysuto/registry/constraint.py](../../src/pysuto/registry/constraint.py) (blocks persistence)*
- *related: [analysis_opt_20260526_165810_3fd6d527_20260527_flow-collapse.md](analysis_opt_20260526_165810_3fd6d527_20260527_flow-collapse.md) (Finding B root cause)*
