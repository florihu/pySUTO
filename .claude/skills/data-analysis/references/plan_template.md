# Plan template

Skeleton for `docs/rep/plans/<topic>_plan.md`. Spin one out when Step 6 of the routine identifies experiments that need code changes, a re-solve, or ≥3 distinct hypothesis tests. Otherwise keep the test strategy inline in the report.

Match the style of existing plans ([`stage5_uncprop_integration_plan.md`](../../../../docs/rep/plans/stage5_uncprop_integration_plan.md), [`stage3_posterior_uncertainty_approach_A.md`](../../../../docs/rep/plans/stage3_posterior_uncertainty_approach_A.md)) — locked-decision tables, scope cuts, explicit out-of-scope, test list.

---

```markdown
# <Topic> — test plan

**Status:** draft / approved / executed.
**Linked report:** [docs/rep/analysis_<run_id>_<YYYYMMDD>_<topic>.md](../analysis_<run_id>_<YYYYMMDD>_<topic>.md)

---

## 0. What this plan is

One paragraph: which observations triggered the plan, why a separate
plan file (vs. inline), and what the user needs to approve before the
experiments run.

---

## 1. Locked decisions

| # | Question | Decision |
|---|---|---|
| 1 | <experiment design choice> | <decision + one-line reason> |
| 2 | ... | ... |

---

## 2. Experiments

### 2.1 Experiment A — <name>

**Hypothesis tested.** Hn from the linked report.
**Setup.** Inputs / config diff / code change.
**Procedure.**
1. Step.
2. Step.
3. Step.
**Pass criterion.** Quantified: "z(ROW · 2024) drops below 10."
**Fail criterion.** Quantified.
**Expected cost.** Wall-clock, RAM, requires-new-data?

### 2.2 Experiment B — ...

---

## 3. Out of scope (deliberately)

Things adjacent to the topic that this plan is *not* going to test, with
a one-line reason each. Helps the next reader avoid scope drift.

- <thing>. Reason: <why deferred>.
- ...

---

## 4. Order of operations

1. Cheapest / most-likely-to-pin-it-down experiment first.
2. ...

---

## 5. Definition of done

The plan is done when:
- Every experiment in §2 has a verdict logged in the report's §7.
- Either a fix from §8 of the report has been implemented and verified,
  or the report has been updated to say why no fix is being applied.

---

## 6. Open questions

Things the user must decide before §2 can execute (default values
proposed, but flagged):

- `<knob>` — defaulting to `<value>`; alternative `<value>`. Approve?
- ...
```
