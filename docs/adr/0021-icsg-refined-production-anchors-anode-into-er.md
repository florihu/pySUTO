# ICSG refined production anchors the anode entering ER, not the matte entering CFR

**Status:** **proposed — implemented and measured on `fix/masscc-round-g`; merge decision
pending.** The placement error is confirmed and the contradiction is removed, but total sum-row
z² rises 4.8 % on a population this round does not touch (see Consequences). Amends
[ADR 0006](0006-scale-total-supersedes-external-control-total.md)
(removes a second anchor from the Matte node), resolves the population
[ADR 0019](0019-contradicted-flow-nodes-lose-anchor-authority.md) detected and
[ADR 0020](0020-withhold-restored-totals-that-contradict-their-own-chain.md) deferred, and
disables ADR 0020's gate. Evidence: REP [`…roundG-anode-anchor`](../rep/analysis_opt_20260909_084958_2742afa3_20260909_roundG-anode-anchor.md).

## Context

ICSG publishes two production series per country, and the feed places them at two points of
the chain `Concentrate → Smelting → Matte → CFR → Anode → ER → Cathode`:

- **smelter production** → the concentrate/scrap entering `Smelting`, ÷ `eff_Smelting`;
- **refined production** → the matte entering `CFR`, ÷ `eff_CFR · eff_ER`.

The second divides a series measured at the **end** of the chain back to its **middle**. That
back-propagation is only valid when the whole `CFR→Anode→ER→Cathode` chain is domestic; it
silently assumes a region refines at home every anode it makes. For a region that smelts and
ships half-finished anode or blister abroad — Zambia, Bulgaria and Congo D.R. have **no
domestic ER at all** — the refined series is small, so the matte was left with no anchored
outlet.

Meanwhile the Matte node's mass balance is **hard** and `σ_c = ANCHOR_CV·|c|` scales with the
total, so the smaller side is always the cheaper anchor to satisfy. Zambia 2022: the smelter
side implies 727 kt CC of matte, the refining side anchors 180 — and the solver collapsed
smelting to 32.

The damage was not only z². Across the 36 contradicted Matte node-years the shipped model kept

| edge family | prior | posterior | retained |
|---|---|---|---|
| Concentrate→Smelting | 13,388 | 3,248 | 24 % |
| Smelting→Matte | 15,817 | 3,546 | 22 % |
| **Anode exports (BACI-observed)** | **8,197** | **332** | **4 %** |

— it deleted 96 % of the observed anode export trade rather than violate the node. The
mis-placement is systemic, not confined to those 36: measured on the raw feed, **95,151 kt CC
of smelted matte across 841 region-years had no anchored outlet**, an 18.4 % shortfall
(Σ produced 477,367 vs Σ anchored 389,328 kt).

[ADR 0019](0019-contradicted-flow-nodes-lose-anchor-authority.md) read this population as data
that could not settle anywhere and rejected a σ remedy on measurement. That reading holds for
Russia and Iran, but not for the majority: splitting the 36 by whether the unplaced matte has
any observed outlet gives **25 node-years with 7,018 kt unplaced against 11,666 kt of observed
anode outflow** — there was somewhere to settle, and the anchor was in the wrong place.

## Decision

**ICSG refined production anchors `Anode→ER`, divided by `eff_ER` alone.** `Matte→CFR` carries
no ICSG anchor: the Matte node's hard balance already determines it from `Smelting→Matte` plus
matte trade. The surplus anode leaves through the BACI-observed export legs instead of being
deleted.

The anchored quantity is **total** refined production (primary + secondary), not primary alone.
Scrap-derived copper reaches ER as anode too; anchoring on primary would understate ER's feed
by 19 % world wide — Germany 49 %, Austria 100 %, Belgium 41 % — and would simply relocate the
contradiction from the Matte node to the Anode node. The Secondary series keeps its separate
role as the `Scrap→CFR` anchor: the same copper measured at a different point of its path, not
a double count.

Implementation is three coordinated edits (`icsg_clean.apply_input_divisors` /
`refining_anode_rows`, the `Refining_Anode` sector key in `icsg_feed` and its mirror in
`build_trade_shares`, and the `sector_map` row → `ER`). The magnitude of the anchor moves only
~1.3 % (÷0.997 rather than ÷0.9845); what changes is **where it attaches**. Every `Smelting`
series and the `Scrap→CFR` anchor are bit-for-bit unchanged.

Two consequences follow that are worth naming:

- **`Anode` becomes a fanned sector.** ADR 0010 recorded that "the dominant domestic `Anode→ER`
  leg has no measurement anywhere" and that the Anode trade-share fan was inert because "ICSG
  carries zero anode usage". Both cease to be true; the ADR-0002 option-D machinery already
  listed `Anode→ER` as a weighted route and needed no change.
- **[ADR 0020](0020-withhold-restored-totals-that-contradict-their-own-chain.md)'s gate is
  retained but disabled** (`WITHHOLD_CHAIN_INCONSISTENT = False`), following ADR 0019's
  "keep the detector, drop the remedy" pattern. Its test — refining feed below half the
  smelting-implied matte — now identifies an *anode exporter*, which the model can finally
  represent. Withholding there would discard exactly the data this re-placement accommodates.

## Considered options

- **Keep the status quo and treat the 36 nodes with σ.** Rejected: measured and rejected as
  ADR 0019 — 0.46 % of the model's mass moved and retention did not recover. σ cannot resolve
  a placement error; it can only stop charging for it.
- **Widen ADR 0020's withhold to all contradicted node-years.** Rejected: it withholds sound
  data. 25 of the 36 have a real, observed anode outlet; withholding would delete a correct
  smelting series to protect an anchor that is itself mis-placed.
- **Anchor `Anode→ER` on primary refined production only.** Rejected: understates ER's feed by
  19 % world wide and relocates the contradiction to the Anode node — Germany, itself one of
  the contradicted node-years, refines 49 % secondary.
- **Keep both anchors — refined production on `Matte→CFR` *and* on `Anode→ER`.** Rejected on
  [ADR 0006](0006-scale-total-supersedes-external-control-total.md)'s own objection to its
  option B: the same edges would be anchored twice across a hard balance, "an over-constraint".
  It is that double anchoring at the Matte node that produced the collapse.
- **Credit anode exports as an outlet in the ADR-0019 closure test** instead of moving the
  anchor. Rejected, as ADR 0019 rejected it: anode is made *by* CFR *from* matte, so anode
  exports raise the matte requirement rather than satisfying it. The fix has to be the anchor's
  placement, not the test's accounting.
- **Correct the ICSG values.** Rejected, as in ADR 0020: a data-source question. Nothing here
  edits a published number — only which edge each series is attached to.
- **Move the `Refining` Secondary series too** (out of `Scrap→CFR`). Considered and deferred:
  whether secondary scrap enters at the smelter or the refinery is a separate modelling
  question with its own evidence, and `Scrap→CFR` is left untouched so this round measures one
  change. Named as the follow-up if the Anode node develops tension.

## Consequences

- Contradicted Matte node-years fall **36 → 9** at the prior, and the nine that remain are all
  Argentina at 1–22 kt inflow — noise scale, not the collapse population. Every substantive
  case (Zambia, Bulgaria, Russia, Germany, Congo D.R., Sweden, Iran, Brazil, Mexico, ROW) is
  gone. The Anode node shows **0** contradicted node-years, so the contradiction is removed
  rather than relocated; the ADR-0019 detector now reports both sectors.
- Germany 1999's new `Anode→ER` anchor is **697.7 kt**, matching the model's own `CFR→Anode`
  prior to the decimal. Austria, a pure secondary refiner, gains an anchor it never had
  (0 → 103.2 kt).
- Stage-2 covered edges rise 66,483 → **68,573**; the ICSG `F_root` total rises to 2,148,797 kt.
  Taiwan's fans remain undistributed by design (ADR 0018) — 304 raw rows.
- The `data/` tree is gitignored, so the `sector_map` row (`Refining_Anode` → `ER`) must be
  re-applied on a fresh checkout or the feed silently maps the refined series nowhere — the
  same reproducibility wart ADR 0014 and ADR 0015 record.

### Measured on `opt_20260909_084958_2742afa3` (30/30 Solved cold)

- At the 36 previously-contradicted node-years the smelting chain roughly doubles its survival:
  `Concentrate→Smelting` 24 % → **47 %**, `Smelting→Matte` 22 % → **45 %**.
- World 2015 chain retention rises: CO Mass 0.649 → **0.666**, Conc-CC 0.681 → **0.735**,
  Smelt-CC 0.806 → **0.837**. Chile gains most (Smelting 0.478 → **0.742**, conc exports 2015
  0.545 → **0.566**); Japan is flat. Cath-CC falls 0.994 → 0.954.
- Fabricated class 200 → **189** group-years, hub∪fab z² 33.6 % → **30.0 %** of sum-row z².
- **Cost: sum-row z² 3,834,168 → 4,017,200 (+4.8 %).** Belgium (+222,696), Italy (+142,745) and
  Austria (+122,094) carry three quarters of it — anode-importing and secondary-dominated
  refiners, whose refining end is newly constrained because `Anode` also becomes a fanned
  sector for the first time. That is the deferred Secondary-series option above, and it is the
  reason this ADR is not yet accepted.
- **One prediction failed.** Anode-export retention at those nodes went 4 % → **2 %**: the fix
  did not save the deleted trade, because the BACI export priors exceed the regions' entire
  production (Zambia 2022: 838 kt CC of anode exports against 727 kt of smelter output). That
  is BACI over-statement / transit, which no anchor placement can address.
