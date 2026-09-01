# No-data (flat-prior) σ anchor, and the structurally-determined carve-out

**Status:** accepted (soft σ anchor); the hard flat-prior-zero *pin* is an
experimental sensitivity variant (branch `sensitivity/no-flat-prior-zeros`).

**Amended 2026-09-01:**
1. **The pin's `l`/`u` boxes only reach the feasibility pre-check, not the QP.**
   `ConstraintRun.l/.u` are read solely by `ClarabelSolver.feasibility_check`;
   neither solve path hands them to Clarabel (`FlowBoundsBuilder` is a
   documented no-op for `G`). The claim below that the pin is "mathematically
   identical to deleting the edge from every linear constraint" therefore holds
   only inside the feasibility probe. In the actual solve, pinned edges kept
   σ = 1e6 and acted as free slack — the QP-effective mechanism is the **soft σ
   anchor**, which must not be disabled in the pin's favour
   (`NODATA_ANCHOR_CV = 0.05` since the scrap-faucet fix; see REP
   `analysis_opt_20260901_080337_99aa122f_20260901_all-years-deviation-scan.md`).
2. **The anchor's exclusion list is widened to the pin's 7-sector hand-list**
   `{CO, Mining, Ore, WR, Tailings, Slag, PSC}`, closing the "latent hole"
   below. The structural-detection follow-up stays open with one added
   requirement: it must be **direction-aware**. Bare `entity_sub == "SEM"`
   matching would also exempt `Scrap` (a SEM secondary *input*, reached via a
   use edge) — exactly the sector the anchor exists to discipline. Only
   supply-side edges into SEM SB nodes (the residual outputs) qualify; see
   CONTEXT.md §"System-boundary category" for the residual-output vs
   secondary-input distinction.

## Context

~83 % of root edges have **no feed coverage** and are filled by
[`FlatPriorImputer`](../../src/pysuto/impute/flat_prior.py) with `a₀ = 0` and a σ
whose scale cascade falls back to the *global* flow median (≈1e6). At that σ the
prior pull `(F−0)²/σ² ≈ 0`, so the QP treats these edges as **free slack**: the
solver routes mass through entire data-free subgraphs (e.g. scrap on the
Taiwan/Panama CFR nodes), which inflates flows that no observation supports.

## Decision

Replace the global-median σ fallback on no-data edges with a sector-appropriate
scale: `σ = max(abs_floor, cv · median|a₀| of anchored siblings in the same
(from_sector, valuation))`, so the zero prior actually pulls and these edges stop
absorbing routed mass. The prior mean stays 0; a hard mass balance can still force
flow at a finite penalty (soft anchor, no hard infeasibility). Lives in
[`ConstraintEngine._anchor_nodata_sigma`](../../src/pysuto/constraints/engine.py),
gated by `nodata_anchor_cv` (`scripts/constraints.py:NODATA_ANCHOR_CV`).

### The carve-out, stated correctly

The anchor **excludes the extraction front-end** `{CO, Mining, Ore, WR, Tailings}`.
The reason is **not** that those edges "carry a real `a₀` magnitude prior" — a
no-data extraction edge has `a₀ = 0` like any other. The reason is that their
magnitude is **structurally determined by a ratio / mass-balance relation, not by
feed data**:

- the **stripping ratio** `WR_Mass = stripping_ratio · Ore_Mass` (a soft ratio,
  ADR 0005);
- the CC-conserved `CO → Ore` extraction chain (`eff = 1`, hard mass balance).

For such an edge `a₀ = 0` means *"value comes from the relation,"* not *"no flow."*
Anchoring (or pinning) it toward zero fights the relation.

## The flat-prior-zero pin (sensitivity variant)

To probe **"does the data-backed network balance without no-data slack?"** the
engine can *harden* the anchor: when `pin_flat_prior_zeros=True`, every targeted
no-data edge is pinned to `l = u = 0` in the flow bounds (mathematically identical
to deleting it from every linear constraint, but index-aligned with the baseline
run for diffing). The cold run then solves on the data-backed ~18 % of edges.

The pin's exclusion set is **broader than the anchor's**:
`{CO, Mining, Ore, WR, Tailings, Slag, PSC}`. All four SEM residual outputs
(`Slag / Tailings / WR / PSC`, [CONTEXT.md](../../CONTEXT.md) §"System-boundary
category") are the `(1−eff)·Σin` complement of a process — and the residual output
is the process's **only free output**, the mass relief valve that lets a hard
process balance close when the product edge is itself pinned. Pinning a no-data
`Slag`/`PSC` to zero would break the smelter/refinery mass balance, so the pin must
spare all four, not just the two the extraction list happens to name.

### Consequences

- **The anchor's 5-sector list has the same latent hole.** With `nodata_anchor_cv`
  active it soft-anchors no-data `Slag`/`PSC` toward zero while sparing `WR`/
  `Tailings`. It is harmless on the sensitivity branch (anchor disabled,
  `NODATA_ANCHOR_CV=0`) but should be widened to the SEM-residual set on `main` as
  a follow-up — ideally by detecting `entity_sub == "SEM"` residual outputs
  structurally rather than hand-listing sectors.
- Mass balance stays **hard** under the pin: an over-constrained node halts the run
  at — and localises — the first infeasible year (`_raise_infeasible`), which is the
  intended diagnostic signal, not a failure to paper over.

## Considered options

- **Pin vs. true column-drop.** Pinning `l = u = 0` is math-identical on every
  linear row and keeps `G`/`edge_index` aligned with the baseline; a column-drop
  re-indexes everything for no analytical gain. Pin chosen.
- **Mirror the anchor's literal 5-sector list for the pin.** Rejected: it pins
  4,612 structurally-determined `Slag`/`PSC` residuals, contradicting the very
  "structural ⇒ don't pin" principle and breaking data-poor smelter/refinery
  balances.
- **Soften mass balance so the run always completes.** Rejected: changes the
  experiment (it would measure a penalized imbalance under a different model rather
  than testing whether the hard-balanced data-backed network is self-consistent).
