# The yearbook's label is not on the row it owns

**Status:** accepted — Round S, 2026-09-22. Answers the ROW question left open by
[ADR 0029](0029-the-concentrate-grade-stays-global.md) and pre-registered as the null in
[ADR 0033](0033-baci-quantity-contradicts-its-own-value.md): ROW's forced deletion is **not** a
missing-export defect and **not** [ADR 0014](0014-bilateral-feeds-scoped-to-cross-region-edges.md)'s
folded internal trade. It is phantom mine production, and the model was right to refuse it.

## Context

Five rounds cut the **discretionary** mine-chain deletion. [ADR 0029](0029-the-concentrate-grade-stays-global.md)
defines what is left: copper leaves a mine node only as domestic smelter intake `U` or export mass
`X` times a grade the hard cone caps at 0.35, so `cap = 0.35·X + U` and `Σ max(0, P − cap)` is
arithmetic, not optimisation. Round R explained Chile's share of it. **ROW's 7,589 kt CC did not
move**, exactly as pre-registered — so ROW is a different defect.

The stated hypothesis was ADR 0014: ROW aggregates 186 source regions, so trade between its members
folds onto domestic addresses and is dropped as mapped self-trade, removing export mass precisely
where the detector reports it missing.

**That hypothesis is refuted.** ROW's folded member→member concentrate trade is 5,034 kt over 30
years against the 21,683 kt needed to close its shortfall at the ceiling — 23 %. And it could not
have worked in principle: inside an aggregate one member's export is another's import, so dropping
both is arithmetically correct, not a loss.

What the evidence actually says is that ROW is forced **only in 2004–2013** and never in 2018–2024,
carrying three times the production and three times fewer exports in the early window. That points
at the source, not at the trade matrix.

## The defect

The ICSG 2014 Statistical Yearbook writes a country that reports two mine source types as a
three-row block, with the country's name typed on the **middle** row:

```
COUNTRY      Source          2004     2005     2006
(blank)      Concentrates   795.8    865.3    806.4
Australia    SX-EW           58.3     50.9     52.4
(blank)      Total          854.1    916.2    858.8
Azerbeijan   Concentrates     NaN      NaN      NaN
```

The cells are **not merged** — `openpyxl` reports no merged range in column A — so the parse filled
the blanks with `df['Region'].bfill()`. Backfilling reaches the right label from the Concentrates
row, which sits *above* the name, and carries the **Total** row onto the *next* country.

Ten years of Australia's mine total were therefore booked as Azerbaijan's:

| booked as | true owner | kt/yr | that region's own mine series |
|---|---|---|---|
| `Azerbeijan` 2004–2013 | Australia | 854–998 | `Azerbaijan` 0.3–2.6 kt/yr |
| `Saudi Arabia` 2005–2007 | Russian Fed. | 640–690 | 22.6 kt/yr |
| `Laos` 2004 | Kazakhstan | 461.8 | 60.1 kt/yr |
| `Cyprus` 2005 | Congo, D.R. | 100.5 | 1.0 kt/yr |
| `Mauritania` 2005 | (block above) | 3.0 | — |

The misspelling is the tell. ICSG's own concordance carries both `Azerbeijan` and `Azerbaijan`; the
correctly spelled series is the real one, and BACI corroborates it — Azerbaijan exported 1.3 kt of
copper across all HS lines in 2010, against 899 kt/yr of claimed mine output with no smelter series,
no refinery series and no exports.

**16 rows, 11,558 kt CC of mine production that never existed.**

A `Total` row that reaches its true owner is then removed by the parse's existing rule — a total is
redundant once its components are present — so all 16 disappear rather than move.

## Decision

**A `Total` row closes its own block, so it takes the label at or above it; every other row keeps
the backfill.** The fill also has to run **before** the aggregate rows are dropped, because the
sheet's own `WORLD`/`TOTALS` block is shaped the same way: with its labels already removed, its
grand total (15,917.8 kt, world refined production) attaches to whichever country stood above it.
Ordering the two steps the other way round trades one misattribution for a larger one.

Fill the labels, then decide what is an aggregate. You cannot classify a row before you know whose
it is.

### Considered and rejected

- **Drop the 16 rows by name.** Removes the symptom and leaves the mechanism to re-fire on the next
  vintage or the next sheet. It also would not have found the `WORLD`/`TOTALS` case.
- **A near-duplicate-spelling contradiction check** (`Azerbeijan` vs `Azerbaijan` differing 446×).
  It works — edit distance over the feed's labels gives 3 pairs and exactly 1 real hit — but it is a
  detector for a symptom of a parse bug now fixed at its cause, and the other two pairs
  (Australia/Austria, North/South Korea) are genuinely different countries. Not worth the machinery.
- **A "mines but has no smelter or refinery" screen.** Fails: Mongolia (5,779 kt), Papua New Guinea
  (4,165 kt), Portugal and Panama all legitimately mine and export concentrate.
- **ADR 0029's `P/(0.35·X+U)` as a source-level input gate.** Fails for aggregate members: `X` is
  known only at root level, so `Azerbeijan` discharges against all of ROW's exports and scores 1.25,
  indistinguishable from Congo D.R.'s 1.12.
- **Relieving the constraint instead.** Rejected on principle — see below.

## Consequences

**Part of ADR 0029's forced term is not a defect.** `Σ max(0, P − cap)` counts copper the model
deletes because it cannot leave the mine node. Where `P` is phantom, deleting it is the *correct*
answer, and the forced term is the model refusing input it should refuse. The fix is to remove the
input, not to loosen the constraint: same posterior, and 11,558 kt CC of spurious "deletion" leaves
the books honestly. This distinguishes Round S from Round R in kind — R recovered mass that was
genuinely missing, S removes mass that was never there.

**The ICSG parse stage was dead and had to be revived.** All three vintage parsers imported
`pysuto.util`, renamed to `pysuto.utils`; `icsg_2014.py`, `icsg_comb.py` and `process_icsg.py`
still wrote to `data/proc/datafeed/`, renamed to `data/proc/feedprep/`; `process_icsg.py` carried
absolute `/mnt/nfs_fineprint/` paths. The stage had not run since March. The repairs are mechanical
renames and are verified inert: re-running the unrepaired-logic chain reproduces
`icsg_flows_comb.csv` with **zero** value differences and zero row differences.

**Seven Japan stock values move, and they are not ours.** `process_icsg.py` collapses
`PMC_Consumers`, `PMC_Merchants` and `PMC_Government` onto one `USC` key and resolves the collision
with `drop_duplicates(keep='last')` after an **unstable** `sort_values`. 53 keys collide that way
(51 doubles, 2 triples), so which series survives depends on row order; removing 16 rows reshuffles
7 of them (Japan USC, ≤ 14 kt each). Stabilising the sort was tested and **rejected**: it makes the
round attributable but flips 21 values instead of 7, contaminating the baseline more, and the
collapse itself is a separate defect — USC should almost certainly be the sum of the three series,
which is a modelling decision, not a parse repair. Recorded in [TODO](../../TODO.md).

**The `Mining/Total` route is now unused for the 2014 vintage.** `RENAME_TOT`'s
`('Mining','Total') → 'Concentrate'` mapping stays: it is still correct for a country that genuinely
reports only a total, and the parse test covers that case.
