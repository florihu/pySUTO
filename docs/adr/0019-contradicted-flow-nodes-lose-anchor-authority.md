# Contradicted flow nodes: detector retained, σ remedy rejected

**Status:** **rejected as a remedy, detector kept** (the same shape as ADR 0016, and for a
related reason). σ-loosening the contradicted side reprices the contradiction without
resolving it: measured on `opt_20260908_101628_420a30dc` it returned total sum-row z² to
3.789M (from Round D's 3.944M, baseline 3.782M) while **0.46 % of the model's mass moved**
(441 kt of 96.1 Mt) and world Conc-CC/Smelt-CC retention stayed at Round D's degraded
0.728/0.682. The collapses it targeted are untouched — Iran Smelting 1996 posterior 4.2 → 4.6
against a 116 kt anchor; Russia Cathode 1995 unchanged at 34.0. **The detector
(`pysuto.flow_node_balance`) is retained and valuable** — it is a data-quality instrument that
localises where a source's own series contradict each other. Amends [ADR 0016](0016-provenance-aware-sigma-on-scale-totals.md)
(first live use of the `sigma_cv_resolver`, on a population disjoint from the one it rejected)
and [ADR 0018](0018-import-free-fans-distribute-domestically.md) (scopes the authority of what
it restores). Evidence: REP
[`…roundE-contradicted-nodes`](../rep/analysis_opt_20260908_101628_420a30dc_20260908_roundE-contradicted-nodes.md);
prior state in [`…roundD-fan-recovery`](../rep/analysis_opt_20260908_084500_724780b3_20260908_roundD-fan-recovery.md).

## Context

A Flow node carries a **hard** mass balance (`cone_type = "zero"`): inflow (`Process→Flow`)
must equal outflow (`Flow→Process` + `Flow→SB`). Two *scale totals*
([ADR 0006](0006-scale-total-supersedes-external-control-total.md)) sit on either side of it —
`supplytotout` on the inflow, `usetotout` on the outflow — each anchored at
`σ_c = ANCHOR_CV·|c|`.

When the two prior sums disagree the balance cannot honour both, and because `σ_c` scales with
`|c|` the **smaller** side is the cheaper anchor to satisfy. The solver therefore collapses the
node to the smaller prior *regardless of which side is better evidenced*:

| node | inflow prior (ICSG concentrate use) | outflow prior (ICSG refining feed) | cost to honour inflow | cost to collapse |
|---|---|---|---|---|
| Russia Matte 1995 | 545.0 kt CC (σ_c 9.3) | 50.8 (σ_c 0.86) | z ≈ 574 | z ≈ 53 |
| Iran Matte 1996 | 116.0 (σ_c 1.97) | 3.76 (σ_c 0.064) | z ≈ 1754 | z ≈ 57 |

ADR 0018 made this visible by restoring the ICSG refining series that the pipeline had been
discarding. In most regions the two series agree (Kazakhstan ratio 0.97, Chile 0.90) and the
restoration was a clean gain. In a minority the refining series is **under-reported** — Russia
1995 reports refining 50.8 kt CC against smelting 544 while exporting 501.8 kt Mass of
*cathode*, so the 493 kt gap has no outlet; Iran's 1995–96 values of 4.1/3.8 jump to 100.6 in
1997, a reporting ramp rather than real growth. There the collapse propagated (Russia
cathode-use posterior 582→34 kt) and cost +4.3 % total sum-row z².

Anode exports do **not** resolve the gap: anode is produced *by* CFR *from* matte (`effcfr`,
[ADR 0004](0004-process-recipe-traversal-imputer.md)/[ADR 0005](0005-pool-scrap-into-eff-constraint-and-stripping-ratio.md)),
so exporting anode *raises* the matte requirement. Bulgaria 1995 — 20.3 kt CC of matte into CFR
while exporting 42.7 kt Mass of anode — is contradicted, not explained.

## Decision

**Where a Flow node's outflow prior falls below `K_CLOSE` × its inflow prior, the outflow side
loses its authority — on both σ's that hold it there.** Detection is declared once in
`src/pysuto/flow_node_balance.py` (`contradicted_flow_nodes`) and consumed by both stages, so
Stage 2 and Stage 3 cannot drift apart ([ADR 0011](0011-one-edge-relation-shared-across-imputation-and-constraints.md)'s
two-enumeration hazard):

- **Stage 2** (`_screen_contradicted_nodes`): covered outflow edges get
  `σ = UNCLOSED_CV·|f_agg|`; the value is kept and reported
  (`diag/contradicted_node_screen.csv`), only its false authority goes — the same treatment the
  outlier-leg screen gives an implausible trade record ([ADR 0017](0017-size-damped-sigma-on-bilateral-trade-legs.md)).
- **Stage 3**: the ADR-0016 `sigma_cv_resolver` returns `UNCLOSED_ANCHOR_CV` for that node's
  `usetotout` rows. `supplytotout` rows key on the *process* sector and are untouched, so the
  reliable inflow anchor keeps its tight σ and the balance settles there.
- `K_CLOSE = 0.5`, `UNCLOSED_CV = UNCLOSED_ANCHOR_CV = 1.0`, acted on for `Matte` nodes only.
  The test is sector-generic; extending it to Anode/Cathode is a measured decision, not a
  default.

**This decision was implemented and measured, and the σ half of it failed** — see Status. What
follows records the design and the evidence; the constants ship disabled unless a later round
revives them.

**Why this does not re-run ADR 0016's rejected experiment.** ADR 0016 loosened *fabricated*
groups and the world chain collapsed, because those anchors are the counterweight to
[ADR 0013](0013-no-data-flat-prior-anchor.md)'s no-data zero anchors on the same subgraphs. The
population here is disjoint: contradicted nodes are **feed-backed and covered**, so the no-data
anchor never acts on them, and the loosened side is bounded by a hard balance against a *tight*
anchor on the other side — the value is pushed **up** to the reliable level, not down to zero.
`FABRICATED_ANCHOR_CV` stays inert.

## Considered options

- **Loosen the anchor only.** Rejected: it makes the *collapse* cheaper without letting the
  node recover. The edge prior blocks just as hard — Russia's restored `Matte→CFR` carries
  σ_a = 3.47, so moving it 50.8 → 545 costs z ≈ 142 on the edge alone. Both σ's must give.
- **`UNCLOSED_ANCHOR_CV = 0.15`.** Rejected: flipping the solver's preference requires
  `cv > ANCHOR_CV · c_inflow/c_outflow`, which is 0.18 at Russia 1995 but **0.52** at Iran
  1996. A fixed 0.15 leaves Iran collapsing.
- **Drop the restored fans that fail the test.** Rejected: discards real data on a consistency
  judgement, and re-introduces the selective admission ADR 0018 removed.
- **Keep the prior but exclude it from scale totals.** Rejected: invents a "prior but not
  anchor" category the accounting vocabulary does not have, for no gain over a wide σ.
- **Scale σ_c from the reliable side** (`σ_c = ANCHOR_CV·c_inflow`) instead of a flat cv.
  Considered and kept as the fallback if cv 1.0 over-loosens — it is self-scaling and needs no
  threshold, at the cost of a rule that reads across the node rather than within the group.
- **Credit anode/matte exports as an outlet** (the first draft of the test). Rejected on the
  recipe: anode exports consume matte through CFR, so crediting them would have passed
  Bulgaria 1995, whose anode exports exceed the anode its reported matte intake can produce.

## Measured outcome (why the remedy is rejected)

| | Round C | Round D | E anchor-only | E both-σ |
|---|---|---|---|---|
| sum-row z² | 3,781,532 | 3,943,708 | 3,842,019 | **3,788,641** |
| world 2015 Conc-CC / Smelt-CC | 0.758 / 0.710 | 0.729 / 0.683 | — | **0.728 / 0.682** |
| Iran Smelting 1996 posterior (anchor 116.0) | 103.6 | 4.2 | 4.6 | **4.6** |

The anchor-only variant behaved exactly as the arithmetic predicted — it silenced the outflow
anchor (Iran Matte usetotout z 6.5 → 0.2) without letting the node recover. Loosening **both**
σ's did not help either, because the binding limit is not σ at all: Iran 1995–96 has no
downstream capacity for the matte its smelting series implies (no refining, no anode exports,
no cathode exports), so there is nowhere for the chain to settle. Of the z² gain, −65,229 fell
on the 204 loosened rows themselves and −89,838 on rows whose posteriors barely moved.

**The contradiction is in the input data, and σ cannot resolve a data contradiction — it can
only stop charging for it.** Reducing the metric while leaving the model unchanged is the
"prices away the symptom" failure Round C rejected for hub σ exceptions; it is rejected here
for the same reason.

## Consequences

- **The detector ships; the σ policy does not.** `contradicted_flow_nodes` flags 48 Matte
  node-years, of which **35 are pre-existing** (Bulgaria 1997–2007, Brazil, Zambia) and were
  already collapsing in the shipped Round-C model — this ADR's lasting contribution is naming
  and locating them.
- The indicated remedy is **data-side**: of ADR 0018's 165 restored groups only **16
  (1,716 kt of 29,648)** fall in a contradicted node-year, so withholding those — the fan stays
  undistributed exactly as before Round D — would keep 94 % of the recovery while removing the
  collapse trigger. That is the candidate for the next round.
- The screen is a **diagnostic of the input data**, not of the reconciliation: a flagged node
  means ICSG's own two series disagree for that region-year. The report CSV is the list to take
  to the data source.
- Pre-existing contradictions (Bulgaria, Zambia — flagged in the shipped baseline too, not only
  in ADR 0018's restored set) are now treated as well; the REP reports the split.
- Glossary: see **contradicted flow node** in [GLOSSARY.md](../../GLOSSARY.md).
