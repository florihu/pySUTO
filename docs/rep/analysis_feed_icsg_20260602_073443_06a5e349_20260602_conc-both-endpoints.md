# `from_raw` placed concentrate on impossible edges — structural-validity guard — `feed_icsg_20260602_073443_06a5e349`

Investigation triggered by a user observation: the ICSG parser maps copper
concentrate production with producer sector `Mining`, but the real producing
sector is `MF` (`MF→Concentrate`). Question: why was this not caught at datafeed
read-in?

Stage: 1 (Datafeed). acc = `copper`. Feed input: `icsg` v2.

---

## 0 — Executive summary

`ConcordanceTable.from_raw` binds a raw row to a root edge by its **primary**
endpoint (producer for Supply rows, consumer for Use rows) and treats the
opposite endpoint as an *optional* filter. ICSG classifies only that one
endpoint, so the `Mining`-tagged concentrate tonnage fanned out across every
edge `Mining` produces — landing on **Mining→Ore (414,369 kt)** and **Mining→WR
(414,369 kt)** while the correct edge **MF→Concentrate received 0**. Because
those rows matched *real* edges, they were never "unmatched": no warning, no
error.

Per the user, **fan-out itself is legitimate** — a row classifying one endpoint
is split across that node's edges by weights supplied at initialization (e.g.
BACI trade). The error to catch is a **structurally impossible relation**: a row
that classifies *both* endpoints into a `(from→to)` sector pair with **no edge
in the acc** (e.g. `Mining→Concentrate`).

> **Verdict.** A `ValueError` now fires when a classified `(from_sector →
> to_sector)` relation has no corresponding edge in the accounting structure —
> always fatal, independent of `allow_unmatched_edges`. Fan-out from a
> single-endpoint classification is unaffected.

---

## 1 — Observation as given

> "I identified an error in the ICSG parser: `Mining` was set as the root sector
> of Concentrate, but it is actually `MF`. Why was this not triggered at read-in
> when src was mapped onto edges? … BACI fan-out is OK — the weights are given at
> initialization. Please push a `ValueError` if the mapped relation is
> structurally not possible (like `Mining→Concentrate`, see acc)."

---

## 2 — Setup

Loaded: the run's persisted concordance (`feed.pkl → concordance_tab`) and
`result.npz` (`F_root`); acc `copper` node/edge index; ICSG inputs
(`values.csv`, `conc.xlsx`). Code: [conc.py `from_raw`](../../src/pysuto/conc.py#L170-L487).

---

## 3 — Data characterisation

### 3.1 Structural facts (acc = copper)

| Sector (Process) | Produces | `…→Concentrate` edges |
|---|---|---:|
| `Mining` | Ore, WR | **0** |
| `MF` | Concentrate, Tailings | **2,820** |

`Mining→Concentrate` does not exist; `MF→Concentrate` is the real edge.

### 3.2 Where the ICSG concentrate tonnage landed (`F_root`)

| Destination edge | kt |
|---|---:|
| Mining→Ore | 414,369 |
| Mining→WR | 414,369 |
| **MF→Concentrate** | **0** |

2,882 concordance rows carry `root_sector_from='Mining'` (split 1,441 / 1,441
across Ore and WR); **zero** carry `root_sector_from='MF'`.

### 3.3 Fan-out is the norm, not the bug

Every feed classifies only one sector endpoint (icsg: producer for Supply /
consumer for Use; baci & crude_ore: from-sector only), so single-endpoint
fan-out is pervasive (icsg ~5k rows, baci ~25k rows) and **legitimate** — the
split is carried by initialization weights. The defect is not fan-out; it is
that an *impossible classified relation* was never rejected.

---

## 4 — Trace (Mining → Concentrate)

1. Raw ICSG Supply rows: `Process="Mining", Type="Concentrate"` (1,441 = mine production).
2. `_make_sector_key` ([icsg_feed.py:142-155](../../scripts/icsg_feed.py#L142-L155)) returns just `Process="Mining"` for Supply; `variable_map` sets only `src_sector_from` (no product); `sector_map` maps `Mining→Mining`.
3. `process_side(side="from")` ([conc.py:265-372](../../src/pysuto/conc.py#L265-L372)) joins on `[from_sector, from_region, time, valuation]`; `Mining` is valid → matches `Mining→Ore`, `Mining→WR`. The to-side filter (conc.py:367-370) is a no-op (`src_sector_to` is NA).
4. The row matched real edges → never "unmatched" (conc.py:398) → no warning. `_validate_table` is bypassed (`_skip_validation=True`, conc.py:483) and would not help.

Note: because the product is **not** classified, this is a fan-out — the
structural guard does **not** fire on the current ICSG data (see §8/§9).

---

## 5 — Hypotheses

- **H1 — `from_raw` enforces only the primary endpoint (user's).** Confirmed by §3.2.
- **H2 — feed-coverage gap.** Rejected: the tonnage is present (414k×2), mis-placed not missing.
- **H3 — `allow_unmatched_edges` should have caught it.** Rejected: the row matched edges; that knob only guards the 0-match case.

---

## 6 — Test strategy

| H | experiment | expected |
|---|---|---|
| H1 | `F_root` on Mining→Ore/WR vs MF→Concentrate | tonnage on Mining→Ore/WR; 0 on MF→Concentrate |
| fix | classify both endpoints to an impossible pair, build concordance | `ValueError` (structurally impossible) |
| fix | classify one endpoint (fan-out) | no error |

---

## 7 — Results & verdict

H1 confirmed (414,369 kt each on Mining→Ore/WR; 0 on MF→Concentrate). H2/H3
rejected. **Verdict:** root cause is `from_raw`'s primary-endpoint-only binding;
the appropriate guard is a structural-validity check on *classified* relations
(both endpoints), leaving legitimate single-endpoint fan-out untouched.

---

## 8 — Fixes

### 8.1 (Implemented) Structural-validity guard in `from_raw`

When a raw row classifies **both** endpoints, the resulting
`(root_sector_from, root_sector_to)` pair must exist among the acc's edge
sector-pairs; otherwise raise a `ValueError` naming the impossible pair(s)
([conc.py:393-446](../../src/pysuto/conc.py#L393-L446)). Always fatal,
independent of `allow_unmatched_edges`. Single-endpoint rows (fan-out) are not
checked. Feed drivers keep `allow_unmatched_edges=True` (fan-out + coverage
warnings unchanged).

**Limitation (by design).** The guard engages only once *both* endpoints are
classified. ICSG currently classifies only the producer, so its concentrate
rows still fan out (to Mining→Ore/WR) and the guard does **not** yet fire on
them — see §8.2.

### 8.2 (Data follow-up, user-owned) Classify the ICSG product, then fix `Mining→MF`

For the guard to protect the concentrate flow, ICSG Supply rows must classify
the product (`Type`) as `src_sector_to`. Once they do, the still-wrong producer
makes the `(Mining, Concentrate)` relation **raise** (structurally impossible);
correcting the producer to `MF` then lands the tonnage on `MF→Concentrate`.

---

## 9 — Verification performed

- `pytest tests/` → **690 passed, 1 skipped** (incl. `TestStructuralValidityGuard`: impossible classified pair raises; fatal regardless of `allow_unmatched_edges`; valid pair lands uniquely; single-endpoint fan-out does not raise).
- Re-ran all three feed drivers → **all run** (icsg, baci, crude_ore exit 0): current data classifies one endpoint, so fan-out is tolerated and no impossible classified pair exists.
- Confirmed `F_root`: Mining→Ore/WR 414,369 kt each, MF→Concentrate 0 (pre-fix run).

---

*Trace files:*
- *feed: [feed.pkl](../../data/proc/feed/registry/feed_icsg_20260602_073443_06a5e349/feed.pkl), [result.npz](../../data/proc/feed/registry/feed_icsg_20260602_073443_06a5e349/result.npz)*
- *inputs: [values.csv / conc.xlsx](../../data/input/feed/icsg/current/)*
- *code: [conc.py:265-446](../../src/pysuto/conc.py#L265-L446), [icsg_feed.py:142-155](../../scripts/icsg_feed.py#L142-L155)*
