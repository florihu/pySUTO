# Round S — ROW's forced deletion was phantom mine production, and the model was right to refuse it

**Run measured:** `opt_20260922_104430_60dbadb2` (label `roundS_mine_total_misattribution`, 30/30 Solved, `prior_mode` cold)
**Against:** R `opt_20260922_092530_f21bdbe7`, Q `opt_20260921_114721_de4c5cc5`, P.1 `opt_20260921_101621_cd0c0ae2`, O `opt_20260917_121759_c57f252b`, base `opt_20260916_124721_c4d1d944`
**Chain:** `imp_20260922_102707_c85e54cd` → `const_20260922_103203_f3c7d822` → the run
**Feeds/params:** icsg **v11** (was v10) — the 2014-vintage mine-total misattribution corrected at the parse; baci v11, crude_ore v3, conc v8, ore_grade v7, scrap_share v3 unchanged
**Date:** 2026-09-22 · **ADR:** [0035](../adr/0035-the-yearbooks-label-is-not-on-the-row-it-owns.md)

**Verdict: ROW's forced deletion goes 7,589 → exactly 0, and every other region's moves by less than
1 kt. ADR 0029's forced term falls 12,315 → 4,723 kt CC (−62 %); total mine deletion 36,807 → 26,742
(−27 %); common-set z² reaches 1,661,859, −3.3 % vs base — the best fit of the whole arc, better than
Q's −1.1 %. Removing input the model was refusing improves the fit and shrinks the ledger at once.**

---

## 1. The finding

ADR 0033 pre-registered ROW as a null and it held: R left it at 7,589 kt CC, 62 % of the remaining
forced term. The standing hypothesis was [ADR 0014](../adr/0014-bilateral-feeds-scoped-to-cross-region-edges.md)'s
folded internal trade. **It is refuted** — ROW's folded member→member concentrate trade is 5,034 kt
against the 21,683 kt needed (23 %), and inside an aggregate one member's export is another's
import, so dropping both is correct rather than lossy.

The real cause is a parse bug. The ICSG 2014 yearbook types a country's name on the **middle** row
of its three-row block, cells unmerged:

```
(blank)      Concentrates   795.8
Australia    SX-EW           58.3
(blank)      Total          854.1
Azerbeijan   Concentrates     NaN
```

`df['Region'].bfill()` reaches the right label from the Concentrates row *above* the name and
carries the **Total** row onto the *next* country. Ten years of Australia's mine total were booked
as `Azerbeijan`'s — 899 kt/yr against the 0.3–2.6 kt/yr the correctly spelled `Azerbaijan` series
reports, with no smelter series, no refinery series and 1.3 kt of copper exports in 2010 across all
HS lines. Russia's total went to `Saudi Arabia`, Kazakhstan's to `Laos`, Congo D.R.'s to `Cyprus`.

**16 rows, 11,558 kt CC of mine production that never existed**, all inside ROW, all in 2004–2013 —
which is exactly the window in which ROW is forced and never outside it.

## 2. Acceptance

| # | criterion | R | S | |
|---|---|---|---|---|
| **S-a** | **ROW forced shortfall** | 7,589 | **0.0** | ✓ closes exactly |
| S-b | ADR 0029 forced deletion | 12,315 | **4,723** | ✓ −62 %, share 33.5 % → 17.7 % |
| S-c | ROW hard-capped years | 10 of 30 | **0** | ✓ |
| S-d | common-set sum-row z² | 1,705,881 | **1,661,859** | ✓ −2.6 % vs R, **−3.3 % vs base** |
| S-e | world MF→Concentrate retention | 0.912 | **0.934** | ✓ +2.2 pp |
| S-f | intake retention > 1.05 | ROW, Kazakhstan | ROW, Kazakhstan | ✓ did not grow |
| **S-g** | Chile / Zambia / Congo D.R. forced (**null**) | 295 / 1,452 / 880 | **295 / 1,452 / 880** | ✓ unmoved to the kt |
| S-h | status | 30/30 | 30/30 | ✓ |

**S-g is the check that matters.** The correction touches five source labels, none of which map to
Chile, Zambia or Congo D.R. All three are identical to the kilotonne, and so are Iran, Russia,
Italy, South Africa, Mongolia, the United States and Poland. The entire −7,592 kt of the forced term
is ROW.

## 3. The reframing: forced deletion is partly the model being right

| | Q | R | **S** |
|---|---|---|---|
| mine prior | 416,173 | 416,150 | **404,591** |
| mine deleted | 39,394 | 36,807 | **26,742** |
| forced | 12,339 | 12,315 | **4,723** |
| forced share | 31.3 % | 33.5 % | **17.7 %** |

The prior falls 11,559 kt — the phantom production — and deletion falls 10,065. **87 % of the
phantom mass was going straight into the deletion ledger.** ROW's own line says it plainly: prior
25,249 → 13,693, deleted 7,675 → **−1,152** (a small net addition), hard-capped years 10 → 0.

`Σ max(0, P − cap)` counts copper the model deletes because it cannot leave the mine node. Where `P`
is phantom, **deleting it is the correct answer**, and the forced term is the model refusing input it
should refuse. The remedy is to remove the input, not to loosen the constraint — same posterior,
and 11,558 kt CC of spurious "deletion" leaves the books honestly.

This makes S different in kind from R. R recovered mass that was genuinely missing; S removes mass
that was never there. Both reduce the forced term; only one of them adds copper to the world.

**S-d is the corroboration.** A correction that merely relabelled tonnage would not improve the fit.
z² falls to −3.3 % vs base, the best of base → O → P → Q → R → S, and Concentrate CC retention rises
0.912 → 0.934. By sector on the common set, base → S: Anode −38,504, CFR −36,538, Cathode −15,549,
Matte −8,922 against Ore +32,243, Concentrate +25,747, MF +2,566.

## 4. The fix, and what it nearly broke

**A `Total` row closes its own block, so it takes the label at or above it; every other row keeps
the backfill.** The fill also has to run **before** the aggregate rows are dropped. The first
version did not, and the sheet's own `WORLD`/`TOTALS` block — shaped identically — handed its grand
total (15,917.8 kt, world refined production) to `Others`, trading one misattribution for a larger
one. You cannot classify a row as an aggregate before you know whose it is.

**Verified surgical.** The unrepaired-logic chain first reproduced `icsg_flows_comb.csv` with **zero**
value and zero row differences, establishing a byte-exact baseline; the patch then measured **16 rows
removed, 0 added, 0 other value changes.** The totals vanish rather than move because the parse's
existing rule drops a total once its components are present. `RENAME_TOT`'s
`('Mining','Total') → 'Concentrate'` stays — still correct for a country that genuinely reports only
a total, and covered by a test.

**Rejected alternatives**, all measured: dropping the 16 rows by name (leaves the mechanism to
re-fire); a near-duplicate-spelling contradiction check (works — 3 pairs, exactly 1 real hit — but
detects a symptom of a cause now fixed); "mines but has no smelter or refinery" (fails: Mongolia
5,779 kt, Papua New Guinea 4,165 kt, Portugal and Panama are legitimate); ADR 0029's `P/(0.35·X+U)`
as a source-level input gate (fails for aggregate members — `X` is known only at root level, so
`Azerbeijan` scores 1.25 against Congo D.R.'s 1.12).

## 5. What the round carried that was not the lever

1. **The ICSG parse stage was dead and had to be revived.** All three vintage parsers imported
   `pysuto.util`, renamed to `pysuto.utils`; `icsg_2014.py`, `icsg_comb.py` and `process_icsg.py`
   still wrote to `data/proc/datafeed/`, renamed to `feedprep/`; `process_icsg.py` carried absolute
   `/mnt/nfs_fineprint/` paths. The stage had not run since March. Only the three files the chain
   needs were repaired; `icsg_2005.py`, `icsg_2023f.py` and `icsg_cap.py` carry the same two
   breakages and are left alone, recorded in [TODO](../../TODO.md).
2. **Seven Japan stock values move and they are not ours.** `process_icsg.py` collapses
   `PMC_Consumers`, `PMC_Merchants` and `PMC_Government` onto one `USC` key and breaks the tie with
   `drop_duplicates(keep='last')` after an **unstable** `sort_values`; 53 keys collide. Removing 16
   rows reshuffles 7 of them (Japan USC, ≤ 14 kt each). Stabilising the sort was tested and
   **rejected**: it flips 21 values instead of 7. The collapse itself — `USC` should almost
   certainly be the *sum* — is a modelling decision and is left to its owner.

## 6. What is open

1. **Zambia, 1,452 kt CC, is now the largest forced item** (79.6 % of its own deletion), followed by
   **Belgium 1,029** against 299 deleted (344 %) and **Congo, D.R. 880** against 605 (146 %). Belgium
   has no mines: this is Round Q's fabricated domestic `Concentrate→Smelting` edge surfacing in the
   detector, and it is not a missing-export defect either.
2. **Chile's deletion is 29,133 kt CC and 99 % discretionary.** Unchanged by S, as predicted.
3. **R.1** — the same value/quantity test on matte, anode and cathode.
4. **`stock_bounds_study.py`** still carries the ADR 0032 source-vs-root bug.
5. The remaining forced term is 4,723 kt = 17.7 % of deletion, its lowest across the arc. The
   export-mass gap that opened ADR 0029 is, after R and S, largely accounted for.
