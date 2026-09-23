# A renamed type is a second stream, not a restatement

**Status:** accepted — Round T, 2026-09-22. Continues
[ADR 0035](0035-the-yearbooks-label-is-not-on-the-row-it-owns.md), which left Zambia as the largest
forced item in [ADR 0029](0029-the-concentrate-grade-stays-global.md)'s ledger.

## Context

After Round S the ADR 0029 forced term stood at 4,723 kt CC, of which **Zambia carried 1,452 —
79.6 % of its own mine-chain deletion**, concentrated in 2004–2007.

The detector said Zambia's mine node could not discharge, because `cap = 0.35·X + U` and its
domestic smelter intake `U` had collapsed:

| year | mine supply | smelter intake `U` | refinery anode use |
|---|---|---|---|
| 2003 | 269.0 | 253.9 | 241.5 |
| **2004** | **319.5** | **20.7** | **286.5** |
| **2005** | **330.0** | **10.4** | **247.7** |
| **2006** | **350.0** | **10.4** | **240.7** |
| **2007** | **341.0** | **1.9** | **229.4** |
| 2008 | 372.0 | 279.8 | 267.8 |

**Physically impossible on its face:** no refinery consumes 287 kt of anode fed by a smelter taking
10 kt of concentrate. The intake series is wrong, not the mine series.

Unlike ADR 0035 this is **not** a defect in the yearbook or in its parse. The ICSG 2014 workbook is
right and `icsg_2014.py` reads it right — Zambia's smelter table has two rows:

```
Zambia   Primary                  280.1  244.8  289.7  224.0
Zambia   Low-grade / Electrowon    20.0   10.0   10.0    1.8
```

and `icsg_2014_flows.csv` carries both, correctly.

## The defect

`icsg_comb.py` folds source categories together before combining vintages:

```python
REN_TYPE = {('Low-grade', 'Smelting'): 'Primary', ...}
```

After that rename, Zambia has **two rows of the same vintage** sharing
`(Region, Type, Process, Supply_Use, Unit, Year)`. The next step is the cross-vintage dedup:

```python
f = (f.sort_values('Year_loaded_from_file')
       .drop_duplicates(subset=f_key, keep='last'))
```

That step exists to keep the **newest restatement** of a series, and it cannot tell a restatement
from a second stream. It kept `Low-grade` and discarded `Primary` — **1,038 kt of Zambian primary
smelting over four years.** Which of the two survives is also order-dependent, since `sort_values`
defaults to an unstable quicksort.

Six keys collide across the whole feed: Zambia's four, and two `Thailand 6/` `Semi/Supply` rows that
are inert (`Thailand 6/` is a footnote suffix that escapes `REGION_RENAME`, so it maps to no root
region, and `cl()` drops `Semi/Supply` anyway).

## Decision

**Rows that a rename folded onto one key, within one vintage, are summed before the cross-vintage
dedup runs.** Two feed streams of one process add; two vintages of one stream do not.

```python
def sum_renamed_collisions(f, f_key):
    return f.groupby(f_key + ['Year_loaded_from_file'], as_index=False,
                     dropna=False)['Value'].sum()[f.columns]
```

Grouping *includes* `Year_loaded_from_file`, so a genuine restatement across vintages is left
untouched for the dedup to resolve, exactly as before. Zambia 2004 becomes 280.1 + 20.0 = **300.1**.

### Considered and rejected

- **Drop the `('Low-grade','Smelting') → 'Primary'` rename instead.** Also puts Zambia at ~280 and is
  a smaller diff, but it discards a deliberate decision — someone judged low-grade smelter feed to
  be primary — and it would leave `Low-grade` as an unmapped type that the feed then ignores, losing
  20 kt/yr rather than mis-placing it. Summing is what the rename already meant.
- **Sum across vintages too.** Wrong, and dangerous: it would double-count every restatement. The
  vintage key is what separates the two cases, and the test pins that.
- **Stabilise the sort so `keep='last'` is at least deterministic.** Treats the symptom. A
  deterministic choice between two streams is still the wrong arithmetic — and the same unstable
  sort in `process_icsg.py` remains open in [TODO](../../TODO.md) for its own reasons.
- **Assert that no key collides within a vintage.** Would have caught this, but it fails the build
  on `Thailand 6/`, which is harmless, and offers no repair. The sum both fixes and documents.

## Consequences

Zambia's smelter intake is restored to 300.1 / 254.8 / 299.7 / 225.8 for 2004–2007, and 2004 goes
from a 267 kt shortfall to 23 kt of headroom. Measured on `opt_20260922_114116_88941ebc` (30/30
Solved): **Zambia's forced shortfall 1,452 → 399 (−73 %)**, the model's forced term **4,723 → 3,670
(−22 %)**, its share of deletion 17.7 % → 13.8 %, common-set z² −3.3 % → −3.4 % vs base.

**The class of defect is new and worth naming.** ADR 0032 was one territory under two labels; ADR
0035 was one row under the wrong label. This is **two rows correctly labelled and then made
indistinguishable by a rename**, destroyed by a deduplication rule that was right for the case it
was written for. Wherever a pipeline renames a category and later deduplicates, the rename must say
whether it merges or replaces.

**What this does not explain.** Congo D.R. (880 kt) and Belgium are untouched by this lever and are
now the largest forced items. Neither is this defect: Congo D.R.'s is single-year holes in BACI
export mass (2013: 702 kt → 2014: **44** → 2015: 202), which [ADR 0033](0033-baci-quantity-contradicts-its-own-value.md)'s
value test cannot reach because the value is low too; Belgium's is a fabricated MF→Concentrate prior
in a country with no mines and no concentrate trade (`X = 0`, `U = 0`).

**Belgium then largely resolved itself, for a reason that is not to our credit.** The
[ADR 0024](0024-fan-domestic-leg-bounded-by-own-production.md) domestic bound is computed by
`build_trade_shares.py`, which the README requires to run **before** `icsg_feed.py` and which reads
the ICSG feed's own production. It had not been re-run since 2026-09-17, so Rounds Q, R, S and T all
fanned their ICSG rows with weights built before their own feed corrections. Rebuilding (v20 → v21)
changed 6,242 weights and moved **21,709 kt CC** from domestic residual onto import legs. On an
otherwise identical chain (`opt_20260922_121256_d4394c0a`) that alone takes **Belgium's forced term
1,029 → 371 (−64 %)**, the model's forced term **3,670 → 3,013**, and common-set z² **−3.4 % → −9.6 %
vs base** — a larger gain than Rounds S and T together, from running the pipeline in its documented
order. The arc stays internally consistent (every run from O onward used v20), so the rounds'
relative results stand; what was stale is the bound's view of production. Recorded in
[TODO](../../TODO.md), together with a candidate guard.
