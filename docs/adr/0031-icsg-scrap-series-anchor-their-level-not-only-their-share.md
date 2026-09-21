# The ICSG scrap series anchor their level, not only their share

**Status:** accepted at the constraint level, **measurement pending** — Round P.0, 2026-09-21. The
family is built and gated before any solve; the chain result is filled in *Measured*.
Amends [ADR 0006](0006-scale-total-supersedes-external-control-total.md) (a third `sum` family) ·
completes [ADR 0030](0030-secondary-smelter-copper-enters-the-chain-once.md) (the Round P it names) ·
reuses [ADR 0016](0016-provenance-aware-sigma-on-scale-totals.md) and
[ADR 0025](0025-scale-totals-carry-the-feeds-reporting-resolution-as-a-sigma-floor.md) unchanged ·
see [ADR 0012](0012-scrap-one-mixed-grade-commodity.md) for why Mass is not anchored.

## Context

ICSG's two scrap series enter Stage 2 as **direct, feed-backed observations on single edges**.
`Scrap` is an `SB` node, domestic-only, so the ADR 0008 leg filter leaves exactly one surviving leg
and the fan is degenerate — no trade-share weighting, no dilution:

| series | edge | covered CC edges | Σ f_agg |
|---|---|---|---|
| `Smelting/Scrap/Use` | `Scrap[SB,R] → Smelting[Process,R]` | 531 / 1410 | 76,873 kt |
| `Refining/Scrap/Use` | `Scrap[SB,R] → CFR[Process,R]` | 483 / 1410 | 24,733 kt |

**In Stage 3 no scale-total row has ever contained those edges.** `scripts/constraints.py` registers
two `from_mask_anchor` families: supply needs `entity_from == "Process"`, use needs
`entity_from == "Flow"`. `Scrap` is `SB`, so `Scrap→Process` matches neither — verified,
`row_meta[type=="sum"].sector` never contains `Scrap`. The driver already half-knew this: the
`SCRAP_SHARE_MODE` comment records that "no-data Scrap→Process edges (in neither scale-total sum
family) acted as free copper faucets".

Slovakia is the clean demonstration. `f_agg` holds **46.634 kt/yr, σ 3.188 (cv 6.8 %),
coverage=True on 29 of 30 years**; Stage 3 keeps it (`a0.npy[294359] = 46.634`); the posterior is
**0.012 kt**. Every row touching that edge — `massbal` (hard, zero cone), the `conc` CC/Mass ratio,
`effsmelting`, and the `scrapsharesmeltingslovakia` band `[0.95, 1.00]` — is **scale-invariant**, so
the whole Slovak secondary chain can shrink toward zero for the price of one edge's prior penalty.
Its anode followed: 43 → 6.5 kt against an unchanged 57 kt anchor, |z| 14 → 52.

The prior does **not** decay into this. Both the base and the Round O run used `--prior-mode cold`
(`run_solver.py`'s CLI default; `a_0_t = a_0_cold`), so `f_agg` is re-asserted every year and the
solver walks away every year regardless. The mechanism is therefore not ADR 0009's warm chain —
that would only aggravate it — but simply:

> **A per-edge prior penalty is cheaper to pay than to satisfy, independently in every year. Only a
> scale total prices the group rather than the edge.**

ADR 0030 moved copper from a route the model anchors **by level** (`Refining/Scrap`) to one it
anchors **only by share** (`Smelting/Scrap`), and a share band over a node with no throughput is
satisfied trivially. Where the smelter node had real throughput the substitution worked (Germany
+0 %, Japan −2 %, China −8 % anode); where it did not, the copper was lost (Belgium −21 %, Austria
−27 %, Slovakia −79 %), and the fit paid 13.3 % of sum-row z². Round O is mergeable only when this
closes.

## Decision

Two **soft sum** families (`type=sum`, `is_hard=False`, `cone_type="free"`), built by the existing
`SumConstraintBuilder.from_mask_anchor` with **no new machinery**, grouped on
`(sector_to, region_to, time, valuation)`:

```
smelterscrap     Σ(Scrap→Smelting)_{R,Y,CC} = the level ICSG reports in Smelting/Scrap/Use
directrefscrap   Σ(Scrap→CFR)_{R,Y,CC}      = the level ICSG reports in Refining/Scrap/Use
```

1. **Two registrations, not one.** Both routes share `sector_from == "Scrap"`; a single mask grouped
   on the scrap side would anchor only their **sum** and leave the solver free to move copper
   between the smelter and the refinery route — the exact substitution ADR 0030 constrained. The
   model now makes *two* claims per region-year, not one.
2. **CC only.** Zero Mass scrap edges are feed-covered; the Mass twins are `network_mass_twin`-derived
   and the implied CC/Mass is exactly **0.7000 on all 1,046 non-zero pairs** — the `conc` nominal
   restated. Anchoring Mass would pin the scrap grade to 0.70 and collapse ADR 0012's deliberately
   wide `[0.40, 0.98]` band to a point, and it would be *stiffer* than its CC twin (fabricated
   groups keep the tight cv with no ADR 0025 floor). The `concscrapsupply` ratio already ties them.
3. **Covered edges only.** The row asserts a level ICSG *reported*. This also makes every group
   100 % feed-backed, so ADR 0025's 0.3 kt floor applies unconditionally.
4. **σ via `_scale_total_cv`, unchanged** — cv 0.01, ADR 0025 floor, ADR 0016 split. No new constant.
5. **A region-year ICSG does not report gets NO row** and keeps exactly the freedom it had. This
   family can only ever *assert* a scrap inflow into Smelting or CFR, **never forbid one**.

Gated before any solve (`SCRAP_LEVEL_ANCHOR`, which `False` reverts to the Round O configuration):
row counts by variant, CC-only, `c` re-derived from `f_agg`, the σ floor, the
row-exists ⇔ feed-observed identity, and that no row spans both destinations.

### On ADR 0006, stated plainly

Because the fan is degenerate and the group is coverage-gated, `c = Σ f_agg` **numerically equals
ICSG's published figure** — Slovakia's `c` is 46.634 and so is ICSG's cell. The row also, plainly,
*adds information*: it is the reason Slovakia's smelter stops collapsing.

Mechanically it is still a scale total: same builder, RHS read from `f_agg`, free cone, no external
totals file, no reconciliation pass, and Stage 2 remains the only place a feed is read. The equality
is a **consequence of a one-leg fan**, not a reintroduction of the control-total mechanism ADR 0006
removed. The distinction is genuinely thinner here than for `supplytotout` / `usetotout`, and this
ADR records that rather than glossing it.

## Considered and rejected

- **Anchoring the Mass twin as well.** No feed information (0 covered edges), it would be 14× tighter
  than its CC twin through the fabricated branch, it voids ADR 0012's band, and it states one
  physical claim twice.
- **One registration grouped on `sector_from`.** Anchors only the routes' sum and licenses the
  smelter↔refinery substitution ADR 0030 exists to prevent. Locked out by a test.
- **Masking on `|f_agg| > 0` instead of coverage.** Admits the `time_linear` region-years as
  *fabricated*, un-floored pins — a 0.068 kt anchor priced at ±0.68 t, |z| ≈ 1,500 per kt.
- **Loosening the `scrap_share` band instead.** ADR 0028 measured what moving a wall does: it
  relocates the deletion rather than removing it.
- **Fixing the band derivation in the same round.** The known defects — `scrap_share_study.py`'s
  `.clip(upper=1.0)` forcing Argentina and Italy's CFR bands to a hard `[0.95, 1.00]`, and the
  absence of any band-width floor — are deferred to P.1 so P.0's effect stays attributable.
- **Anchoring the 107 "nil direct route" region-years at zero** — see Consequences.

## Measured

*Pending — filled after the `roundP_scrap_level` run; REP
`docs/rep/analysis_opt_<P>_2026MMDD_roundP-scrap-level.md`.*

## Consequences

- `variant` gains `smelterscrap` and `directrefscrap`. Any diagnostic filtering `type == "sum"`
  (notably `roundL_reexport_diag._sums`) now sees a **changed row population**, so a sum-row z²
  total is **not comparable** across the Round O / Round P boundary. Comparisons must use the
  constraint_ids common to both runs; the totals 1,719,806 (base) and 1,949,396 (Round O) are
  superseded as headline figures.
- **Known gap, deliberately left to P.1.** In **107 region-years** ADR 0030's nesting clips the
  direct route to nil — a *positive* claim that there is no direct route (Bulgaria 29, Iran 23,
  Japan 19, Korea 11). The coverage gate records that claim as merely *unstated*, so the solver may
  re-invent a direct route there. Pinning zero was rejected for this round because it would
  hard-code the Bulgaria clip ADR 0030 itself records as wrong, and because a `c = 0` group has no
  σ under the current resolver (`cv·|c| = 0`, and the engine's 1e-3 floor would make it a hard pin).
  The measurement reports the posterior in those 107 region-years by region.
- **A level anchor cannot lift a route its hard band ceiling blocks.** Measured on the Round O
  posterior, `(node input × band ceiling) ÷ reported level` — below 1 means the band binds first:

  | pair | ceiling | headroom | reading |
  |---|---|---|---|
  | **CFR / Belgium** | 0.195 | **0.222** | the only pair blocked by a band *ceiling*; honouring its 123 kt/yr needs the fire refinery to grow 4.5× |
  | Smelting / Austria, Belgium, UK; CFR / Argentina, Italy | 1.000 | 0.61–1.18 | no ceiling — the node must simply grow, which the anchor does |

  Belgium/CFR is therefore excluded from this round's acceptance and is P.1's pre-stated target.
- `prior_mode` is now persisted in every opt run's `stage_config` (it was not recoverable from the
  artifacts, so no existing run records whether it was warm or cold).
- `data/` is gitignored, but this round changes **code only** — nothing to re-apply on a fresh
  checkout.
