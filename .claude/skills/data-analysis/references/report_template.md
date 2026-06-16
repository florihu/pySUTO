# Report template

Skeleton for `docs/rep/analysis_<run_id>_<YYYYMMDD>_<topic-slug>.md`. Match the style of existing REPs ([`csxew_root_cause_investigation.md`](../../../../docs/rep/csxew_root_cause_investigation.md), [`analysis_opt_…_scaling.md`](../../../../docs/rep/analysis_opt_20260521_100428_9d33a0b9_20260526_scaling.md)) — tables first, prose for reasoning, file:line refs, source-artefact links at the bottom.

The numbering here matches the 9-step routine in `SKILL.md` so you can fill in §3 from Step 3 directly. The §0 executive summary is written *last*, after §7 produces a verdict.

---

```markdown
# <Topic title> — <run_id>

Follow-up to / standalone investigation triggered by user observation(s) on
[data/proc/<stage>/<run_id>/](../../data/proc/<stage>/<run_id>/).

Upstream chain:
- const run: [const_<id>](../../data/proc/const/<const_run_id>/)
- imp run:   [imp_<id>](../../data/proc/imp/<imp_run_id>/)
- feeds:     <feed_id_1>, <feed_id_2>, …

Config knobs of note (from `metadata.json` → `stage_config`):
- `<knob>: <value>`
- ...

---

## 0 — Executive summary

<Write last.> One paragraph: what was observed, what was diagnosed, the
single sentence verdict. Lead with the smoking-gun number if there is one.

> **Verdict.** <one-sentence root cause>.

---

## 1 — Observations as given

Quote the user's observations verbatim. List 1, 2, 3 if multiple. Flag any
that are interdependent.

---

## 2 — Pipeline trace setup

Which stage owns each observation. Which artefacts you loaded. Anything
unusual in the sidecar (label, stage_config drift from defaults).

| observation | owning stage | first place to look |
|---|---|---|
| ... | ... | ... |

---

## 3 — Data characterisation

Whatever Step 3 produced. Standard panels:

### 3.1 Distribution of the flagged quantity

Histogram numbers (counts at each log-bucket) or a small table.

### 3.2 Stratification

Group by `(sector_from, valuation)`, by year, by constraint type, by
feed coverage — whichever is relevant. Tabulate.

| group | n | metric | metric | metric |
|---|---:|---:|---:|---:|
| ... | ... | ... | ... | ... |

### 3.3 Worst-N

Top 10–20 worst-case rows (highest |z|, biggest collapse, widest σ_a*).
Name region / sector / year / edge_id / constraint_id explicitly.

---

## 4 — Pipeline trace of one specific case

Pick the worst case from §3.3. Walk it back stage by stage.

### 4.1 Feed layer
Which feed(s) observed it. Raw value. Pedigree CV.

### 4.2 Aggregation
What `f_agg`, `sigma_agg` came out. Was it imputed (check
`imputation_log.parquet`) or directly observed?

### 4.3 Constraints
`a₀[edge]`, `σ_a[edge]`. Which rows of `G` reference this column —
list them with `(constraint_id, type, c, σ_c)`.

### 4.4 Solver
`F_opt[edge]` at the relevant `t`. Residual on each of the
constraint rows from §4.3. Which one binds.

### 4.5 (If relevant) Posterior
`sigma_a_star[edge]`, lower / upper bounds.

---

## 5 — Hypotheses

Ranked by P(cause | evidence). 3–5 entries. Each one:

### H1 — <one-line claim>

**Mechanism.** Which line of which module (file:line). What it does
under what input conditions.

**Predicted signature if true.** "We'd see X on every edge where Y."

**What would falsify it.** "If we re-ran with `<knob>=<value>` and X
didn't change, H1 is rejected."

### H2 — ...

---

## 6 — Test strategy

For each hypothesis, the smallest experiment that distinguishes
true / false.

| H | experiment | expected if true | expected if false |
|---|---|---|---|
| H1 | filter `…` and count | n ≈ … | n ≈ 0 |
| ... | ... | ... | ... |

If any test required code changes or a re-solve, link the plan file:
[docs/rep/plans/<topic>_plan.md](plans/<topic>_plan.md).

---

## 7 — Test results

| H | result | evidence |
|---|---|---|
| H1 | **confirmed / partial / rejected** | `n=…`, `median=…`, … |
| ... | ... | ... |

### 7.1 Verdict

One paragraph naming the actual root cause. May differ from the
top-ranked H — that's fine and worth saying explicitly.

---

## 8 — Proposed fixes

Ranked by P(fix), then by simplicity, then by blast radius.

### 8.1 (Highest leverage) — <short name>

**What.** The change (file:line or config diff).
**Expected effect.** Quantified if possible ("drops z from 62 → ~6").
**Risk.** What else might shift.

### 8.2 — ...

---

## 9 — Suggested verification

What the user should run next to confirm a fix works. Usually:
1. A spot-check on another instance of the same pattern.
2. A regression check on something the fix could break.
3. The smallest end-to-end re-solve that exercises the change.

---

*Generated <YYYY-MM-DD>. Trace files used:*
- *<role>: [<path>](<relative-path>)*
- *...*
- *builder / solver code: [<file>:<line-range>](<relative-path>)*
```
