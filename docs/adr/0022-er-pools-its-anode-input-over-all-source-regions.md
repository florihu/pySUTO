# ER pools its anode input over all source regions

**Status:** accepted — completes [ADR 0005](0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md)
on the one recipe it did not reach. Evidence: REP
[`…roundH-anode-trade`](../rep/analysis_opt_20260909_094009_49eab1dc_20260909_roundH-anode-trade.md).

## Context

`effer` was the only process-efficiency row without `agg = true`. `effsmelting` and `effcfr`
carry it; `effsxew` and `effmfflotation` do not, but their input (`Ore`) is never traded, so it
never mattered. `effer`'s input is `Anode`, which **is** traded — and the row was therefore
emitted per endpoint pair, resolving to the same-region leg only:

```
ratio_effcfr_Belgium_CC_2015                                        ← aggregate, pools every input leg
ratio_effer_ER.Belgium-CER.Belgium_Anode.Belgium-ER.Belgium_CC_...  ← the DOMESTIC leg alone
```

So a refinery's output was tied to its *domestic* anode input, and **imported anode was not an
input to the ER recipe at all**. Importing anode could not produce cathode, so the solver
deleted the imports; symmetrically, exporters' anode was pushed back onto their own domestic
`Anode→ER` leg. The result was a trade network that existed in the priors and not in the
posterior:

| cross-region CC retention | 2000 | 2015 | 2022 |
|---|---|---|---|
| Concentrate | 104.8 % | 82.1 % | 80.2 % |
| Matte | 91.5 % | 106.5 % | 111.4 % |
| **Anode** | **5.6 %** | **6.7 %** | **4.7 %** |
| Cathode | 112.1 % | 99.2 % | 95.9 % |

The deletion was **forced, not cheap**: anode legs carry a median cv of 0.0496 — as tight as
cathode's 0.0481 — yet a median |z| of 16.9 against cathode's 0.0, 76 legs paying 114,115 z².
The importer's whole chain went short by exactly the missing anode (Belgium 2015: `ER`, `CER`,
`CP` and `Cathode` all at ≈156 kt against anchors of ≈380, every row at z ≈ −34.5, against
214 kt of deleted anode imports).

This is the configuration [ADR 0005](0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md)
already rejected as its option C — *"Same-region-only pooling. Sum only inputs whose source
region equals the process region. Rejected: diverges from `_do_recipe`'s all-region `S`, so
prior and constraint would pool different sets."* ADR 0005 was scoped to Smelting and CFR
because those are where Scrap needed pooling; ER has no scrap input, so it was never revisited
and kept the state that ADR had ruled out.

## Decision

**`effer` carries `agg = true`.** ER's anode input is summed over all source regions, bucketed
on `(region_to, time, valuation)` exactly as `effsmelting` and `effcfr` are, so imported anode
feeds the refinery and the constraint pools the same set the imputer's `_do_recipe` always did.

The change is one cell of `data/input/params/eff/current/params.csv`:

```
-effer;efficiency;;Cu recovery in ER: …
+effer;efficiency;true;Cu recovery in ER: …
```

It also moves ER out of the `ParametricImputer`, which skips `agg` rows
(`src/pysuto/impute/parametric.py`), onto the `NetworkImputer`'s pooled recipe traversal — the
same path Smelting and CFR already take ([ADR 0004](0004-process-recipe-traversal-imputer.md)).

## Considered options

- **Leave it and treat the symptom with σ on the anode legs.** Rejected: the legs are already
  tight (cv 0.05) and are being deleted *against* a median z of 16.9, so σ is not what binds —
  the recipe simply cannot consume the imported anode. This is the "prices away the symptom"
  failure [ADR 0019](0019-contradicted-flow-nodes-lose-anchor-authority.md) measured and
  rejected.
- **A recipe-derived constraint builder** reading `acc.process_recipes` directly. Rejected as
  ADR 0005 rejected it (option B): pooling must be expressed in the params file, not in code.
- **Give `effsxew` / `effmfflotation` the flag too**, for symmetry. Deferred, not rejected:
  their input `Ore` has no cross-region edges, so the flag is inert there today. Doing it would
  be a no-op change to a file that cannot be version-controlled (below), so it waits for a
  reason.
- **Widen `a_sector_from` to `*`** as `effsmelting`/`effcfr` do. Rejected: ER's only admissible
  input is `Anode`, so the wildcard adds nothing and would silently pool a future second input
  that ought to be modelled explicitly.

## Consequences

- **Anode trade is reconciled instead of deleted**: cross-region CC retention 6.7 % → **95.7 %**
  (2015) and 4.7 % → **71.4 %** (2022).
- **Sum-row z² 3,834,168 → 2,821,249 (−26.4 %) from this change alone** — larger than every
  other change in the Mass–CC arc combined, and below anything the project has recorded. Total
  soft z² 4.010 M → 2.975 M; soft |z| p99 10.47 → 8.00; rows with |z|>10 4,835 → 3,727; max |z|
  71.1 → 63.5; tailings z² 8,840 → 6,221. 30/30 Solved cold.
- **Two long-standing sore points move for the first time.** `Cathode_China` max |z| 18.1 →
  **3.5** — the acknowledged cost of [ADR 0017](0017-size-damped-sigma-on-bilateral-trade-legs.md),
  open since 2026-09-03 — and the `Cathode_ROW` z ≈ 70 tail, recommended for a σ exception in
  four separate REPs and never adopted, falls to 61.6 without one. Both were downstream of this
  defect, not σ problems.
- World 2015 retention: CO Mass 0.649 → 0.815, Conc-CC 0.681 → 0.809, Smelt-CC 0.806 → 0.882;
  Chile conc exports 0.545 → 0.868. Cath-CC 0.994 → 0.964 is the one measure that falls.
- **It does not fix the contradicted Matte nodes** (still 36) — that is
  [ADR 0021](0021-icsg-refined-production-anchors-anode-into-er.md)'s job, and the two are
  complementary: applied together, contradicted nodes 36 → 9 and Conc-CC/Smelt-CC gain a further
  4.6/5.0 pp for no z² cost.
- ⚠ **Not version-controlled.** `data/` is gitignored, so this one-cell change lives only on
  disk and must be re-applied on a fresh checkout, or the model silently returns to deleting
  anode trade. The exact edit is quoted above. Same wart as ADR 0014 and ADR 0015, but now
  load-bearing for ~26 % of the model's tension.
