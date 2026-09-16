# Round L.1 / L.2 — re-exports re-attributed at feed time (ADR 0027): the structure works, the exporters pay

- run `opt_20260915_111233_def72bc3` ← `const_20260915_110002_b000445c` ←
  `imp_20260915_105459_546b615b` ← `feed_baci_20260915_105105_187857d6` (BACI input **v7**,
  netted) + `feed_icsg_20260915_105214_dbd9ec05` (on `trade_shares` **v14**) +
  `feed_crude_ore_20260603_095651_558f7369`; cold, **30/30 Solved**
- predecessor: `opt_20260915_103911_38b3fffc` (Round L.0, ADR 0026, shipped) — sum-row z² 2,136,925
- baseline: `opt_20260914_090327_c21a5c48` (Round K.1) — sum-row z² 2,331,632
- reproducer: `python scripts/explo/roundL_reexport_diag.py opt_20260914_090327_c21a5c48 opt_20260915_103911_38b3fffc opt_20260915_111233_def72bc3`
- branch `feat/baci-reexport-netting`. ADR [0027](../adr/0027-re-exports-are-re-attributed-at-feed-time.md) — **measured, decision pending**.

## 1. What the round did

`scripts/build_baci_reexport.py` re-attributes `A→hub→Z` to `A→Z` on the registered BACI feed,
`T = min(X, I, max(R, E))` per (product, region, year) in contained copper, and emits a new feed
version with three ledgers beside it. ADR 0027 carries the rule, the three exclusions (ROW is never
a hub; screened origins are never re-attributed; ICSG gap-years are skipped), the crumb rule, and
the σ compounding.

Applied: **10,524 kt CC**, 2.14 % of gross trade, over 1,358 hub-years in 3 passes; 182 new legs;
467 kt of re-imports and ROW-folded self-pairs dropped. 60,600 kt of residual stays where it is
and is now **labelled** rather than silently clipped (`pass-through` 38,606, ROW aggregate 18,192,
`icsg-gap` 1,273, `inbound-only` 970).

Two defects were found and fixed before the shipped chain ran, both by measurement rather than
review: a hub could apply its volume once per pass (12,804 kt applied against a computed 10,553 —
now each hub's cumulative netting is capped at its own `T`), and legs netted to a rounding
remainder produced σ multipliers up to **1e48** (now zeroed below a floor, and the multiplier
capped at 5.0, past which ADR 0017's cv ceiling of 1.0 makes further inflation inert). BACI input
v6 is the pre-cap artifact and was superseded by **v7** within the session; v5 is byte-identical
to its recorded hash throughout.

## 2. It does exactly what it was built to do

| cell (sum-row z²) | base | L.0 | **L.1** |
|---|---|---|---|
| Ore / Germany | 53,606 | 53,607 | **0** |
| Concentrate / Germany | 21,043 | 20,826 | **0** |
| Concentrate / Korea, Rep. | 22,479 | 9,767 | **0** |
| Cathode / Portugal | 30,004 | 18,646 | **3,984** |
| Anode / Netherlands | 68,411 | 58,950 | **18,379** |
| Cathode / Netherlands | 908 | 871 | **28** |
| Cathode / Singapore | 1,107 | 1,091 | **227** |
| Concentrate / India (untouched by design, `T = 0`) | 23,214 | 20,475 | 19,277 |

Phantom production at the `P = 0` hubs, Σ posterior `Process→Flow` CC pooled over 30 years:

| | base | L.0 | L.1 |
|---|---|---|---|
| Cathode / Netherlands | 1,434 | 1,369 | **328** |
| Cathode / Singapore | 1,417 | 1,387 | **630** |
| Cathode / United Kingdom | 814 | 736 | **495** |
| Concentrate / Korea | 745 | 791 | **149** |
| Concentrate / Germany | 376 | 375 | **160** |
| Cathode / United Arab Emirates | 985 | 985 | 741 |
| Cathode / France | 1,866 | 1,848 | **1,818** |

France is the predicted exception and the ledger named it in advance: `ceiling-relabel`. Its fan
residual is unbounded because `BOUND_DOMESTIC_SECTORS = ["Concentrate"]`, so its phantom refinery
moves from the export side to the use side instead of disappearing. 459 hub-years / 1,827 kt CC
sit in that class (France, Singapore, UAE, Malaysia, Sweden, South Africa cathode; Netherlands
anode). **That is exactly what L.2 was scoped to close.**

Headline and classes:

| | base | L.0 | L.1 |
|---|---|---|---|
| sum-row z² | 2,331,632 | 2,136,925 | **1,769,941 (−24.1 % vs base)** |
| ratio-row z² | 133,403 | 128,525 | **114,239** |
| Round C **hub** class z² (classes fixed on the base run) | 199,324 | 163,480 | **47,757 (−76 %)** |
| hub ∪ fabricated | 343,788 (14.7 %) | 279,637 (13.1 %) | **157,444 (8.9 %)** |
| Σ\|Δposterior\| CC vs predecessor | — | 92,213 kt | **187,790 kt (2.73 %)** |
| PSC net-stock tail, \|z\| p95 / max | 12.23 / 41.7 | 12.34 / 41.7 | **9.76 / 35.7** |
| solver | 30/30 | 30/30 | 30/30 |

Every constraint family improved, so no tension was displaced into the ratio rows. The metric
moved with a large amount of mass. The Zambia phantom is still refused (retention into
Switzerland/Singapore/Luxembourg 0.019 → 0.026, its other destinations 1.064) — the finding stays
visible, as the no-code-change decision of 2026-09-10 requires.

## 3. And it fails the stop rule

| retention | base | L.0 | **L.1** | vs L.0 |
|---|---|---|---|---|
| Chile concentrate exports, 2010–19 | 0.618 | 0.636 | **0.561** | **−0.075** |
| Peru | 0.947 | 0.944 | **0.901** | **−0.043** |
| Indonesia | 0.841 | 0.850 | 0.847 | −0.003 |
| world cross-region concentrate | 0.815 | 0.823 | 0.809 | −0.014 |
| world CO Mass | 0.785 | 0.788 | 0.779 | −0.009 |
| world Concentrate CC | 0.881 | 0.882 | 0.872 | −0.010 |
| world `MF→Concentrate` CC | — | 0.893 | **0.871** | −0.022 |
| world `Ore→MF` Mass | — | 0.804 | **0.772** | −0.032 |
| world `CO→Ore` Mass | — | 0.787 | **0.761** | −0.026 |

The stop rule required exporter retention within ±0.01 or up. **Chile −0.075 and Peru −0.043 fail
it**, and the whole upstream mining chain loses about 3 pp. This is ADR 0016's collapse signature
in miniature, and it is the reason the rule exists.

**Where the loss is, traced.** Splitting the cross-region concentrate legs by whether the netting
touched them:

| | L.0 | L.1 |
|---|---|---|
| legs the netting **touched** (4,083) | prior 76,144 → post 61,970, ret **0.814** | prior 75,098 → post 59,708, ret **0.795** |
| legs it **did not touch** (17,537) | prior 5,140 → post 4,967, ret 0.966 | prior 5,143 → post 5,179, ret **1.007** |

Untouched legs *improve*; the loss is entirely on the legs the rule moved, and Chile's touched
legs carry it (0.631 → 0.557). Three candidate causes were tested and two are ruled out:

- **Not the fan withhold.** The 120 `imports_exhausted` groups removed only 3,511 kt of ICSG
  `F_root` in total and **124 kt** of concentrate demand — three orders of magnitude too small.
- **Not the σ compounding.** Chile's median leg cv is unchanged on every destination except Spain
  (0.035 → 0.106, where half the leg was netted away); China 0.988× and Japan 1.004× of their
  previous cv. Spain accounts for −314 of Chile's −1,267 kt posterior loss.
- **The mechanism that remains** is the concentrate balance itself. Chile's *Mass* export prior is
  exactly unchanged (75,549 kt, the invariant holds to the kilotonne), but its *content* prior
  rises (20,575 → 21,130 on touched legs) because the fan re-splits each consumer's fixed ICSG
  intake along the new geography — tonnage that used to reach Spain, where the clip discarded it,
  now reaches China and Japan, where it must fit inside an intake ICSG pins. China's concentrate
  node absorbs the collision: its import prior is unchanged (37,541) yet its posterior falls 846,
  and its domestic leg falls 1,834. **The re-attribution delivers copper to consumers whose intake
  cannot grow to receive it, and the QP resolves that by deleting upstream.**

That is a real finding about the model, not an implementation slip: closing the clip leak exposes
a disagreement between BACI's tonnage and ICSG's intake that the hub was previously absorbing.

## 4. What I did not do

No constant was tuned to recover the retention, and the feed was not reverted. Round E's rule and
the round's own plan both reserve this decision for the owner on the measured numbers, and the
trade-off is genuine in both directions:

- **Keep L.1.** −24.1 % on the headline, the Germany/Korea/Portugal/Netherlands phantom chains
  gone, the PSC tail down, the clip leak labelled — at the cost of 3 pp of mining-chain retention
  and 7.5 pp of Chile's export retention.
- **Revert to L.0.** Restore `data/input/feed/baci/v5/*` into `current/` (the registry then bumps
  to a v8 identical to v5) and rebuild the chain; L.0 already ships −8.4 % on its own merits.
- **Continue to L.2 first.** The `ceiling-relabel` class is 459 hub-years the bound does not yet
  cover; extending ADR 0024 to Cathode and Anode is the change L.1's own ledger points at, and it
  acts on the use side — the same side where China's node is now absorbing the collision. It may
  move the retention number, and it is the only one of the three that is a *measurement* rather
  than a choice.

## 5. Decision

**Pending — put to the owner on these numbers.** The current working tree has BACI at v7 (netted);
nothing is merged to `main`.

**State to preserve either way:** ADR 0026 / Round L.0 is shipped and independent of this decision.
The netting code, its 28 tests, the ledgers and this measurement stand regardless, and BACI v5
remains byte-identical to its registered hash, so the revert is exact.

---

# Addendum — two defects in this round's own code, and the corrected measurement

The first L.2 run (extending ADR 0024's bound to Cathode and Anode) appeared to **reverse** L.1's
main win: Concentrate/Germany 0 → 38,731 z², Korea's phantom mine 149 → 2,173 kt. That reading was
wrong, and chasing it found two defects in this round's own code. Both were caught by measurement,
neither by review.

1. **A non-producer must say "zero", not "nothing".** Extending the bound meant sourcing production
   from the raw ICSG rows rather than the ICSG `F_root`. `icsg_production` returns a row for every
   region-year (value 0 where the region produces none, because it groups over all domestic
   producing edges); a raw lookup returns **no row at all** for Germany or Korea, which mine no
   concentrate. ADR 0024's guard reads a missing row as *unknown* and declines to bound — so the
   helper silently un-bounded exactly the phantom-mine regions the bound exists for.
   `icsg_raw_production` now spans every group the fan covers and defaults to 0.0, which is what
   the `F_root` reading did implicitly. Regression test pins it with these numbers.
2. **The converter pools matte *and* scrap.** `P_Anode` was `matte · eff_CFR`, omitting the scrap
   that enters CFR directly (ADR 0005/0021). Belgium 2015 smelts 146 kt of scrap and feeds its
   converter a further **155 kt** directly, so its anode output read 139 kt against the 380 kt its
   own refinery consumes — a secondary refiner made to look like a pure importer, whose domestic
   residual the bound then capped far too low (CFR/Anode Belgium 44,349 → 75,656 z²). Corrected to
   `(matte + scrap) · eff_CFR`, Belgium reads 292 kt and the cell returns to 43,884. `U_Matte`
   takes the same correction (`anode use ÷ eff_CFR − scrap`).

The netting itself barely moves: applied **10,496** kt CC instead of 10,524, with Concentrate
unchanged at 5,252 — so §2–§4 above stand as the description of what the re-attribution does. BACI
input **v8** carries the corrected definitions and is the version both variants below were built on.

## Corrected, like-for-like (all on BACI v8, `feed_baci_20260915_122645_ef721a4b`)

| | L.0 `…103911_38b3fffc` | **L.1** netting, bound = Concentrate `…124806_d5ec9551` | **L.2** netting + bound on Cathode/Anode `…131115_b8bf05a9` |
|---|---|---|---|
| sum-row z² | 2,136,925 | 1,828,508 (−14.4 %) | **1,720,192 (−19.5 %)** |
| Round C hub-class z² | 151,887 | 67,300 | **35,136 (−77 %)** |
| hub ∪ fabricated share | 12.5 % | 9.9 % | **8.6 %** |
| Σ\|Δposterior\| CC vs predecessor | — | 154,873 kt (2.26 %) | +23,108 kt (0.34 %) |
| Cathode / France phantom production | 1,848 | 1,817 | **86** |
| Cathode / UAE · Malaysia · UK | 985 · 891 · 736 | 741 · 830 · 495 | **257 · 367 · 262** |
| Concentrate / Germany · Korea | 375 · 791 | 160 · 148 | 160 · 151 |
| CFR / Belgium | 43,493 | 43,955 | 43,884 |
| Cathode / Italy | 66,917 | 67,567 | **35,374** |
| **Chile · Peru · Indonesia retention** | **0.636 · 0.944 · 0.850** | **0.574 · 0.903 · 0.845** | **0.574 · 0.903 · 0.845** |
| world CO Mass / Conc CC / Cath CC | 0.788 / 0.882 / 1.012 | 0.779 / 0.872 / 1.008 | 0.779 / 0.872 / 1.003 |
| Zambia → CH/SG/LU | 0.019 | 0.025 | 0.025 |
| solver | 30/30 | 30/30 | 30/30 |

Three things this settles:

- **The extended bound is strictly better than not having it.** A further −5.9 % on the headline,
  the hub class halved again, France's phantom refinery reduced to 86 kt (−95 %), Italy's cathode
  cell halved — at no cost anywhere, including to Belgium once `P_Anode` is right. If the netting
  is kept, it is kept with the bound.
- **The exporter cost is inherent to the re-attribution, not to the unbounded cathode residual.**
  Chile (0.574), Peru (0.903) and Indonesia (0.845) are **identical to three decimals** in both
  variants. Extending the bound changes the use side and leaves the exporter side exactly where it
  was, which is what §3's trace predicted.
- **The stop rule still fails.** Chile −0.062 and Peru −0.041 against a ±0.01 band.

## Decision, restated on the corrected numbers

**Pending — the owner's call.** The trade is: **−19.5 % on the headline, the hub class down 77 %,
the Germany/Korea/France/Portugal/Netherlands phantom chains gone, the PSC tail down** against
**6.2 points of Chile's export retention, 4.1 of Peru's, and ~1 point of the world mining chain**.

The working tree carries BACI at **v8** and `BOUND_DOMESTIC_SECTORS = ["Concentrate", "Cathode",
"Anode"]` (the L.2 variant). Reverting is exact: BACI v5 is byte-identical to its registered hash,
so restoring it into `current/` and rebuilding returns the chain to L.0, which ships on its own.

---

# Addendum 2 — the exporter "failure" is a 4-point deepening of a 31-point defect

Asked to recommend keep-or-revert, I diagnosed the exporter loss before answering. The answer
changes what the stop-rule failure means.

**Chile's trade is physically supportable.** Over 2010–19, BACI has Chile exporting 20,776 kt CC of
concentrate against 24,572 available (ICSG mine + imports − its own smelters) — a ratio of
**0.845**. Chile is not exporting copper it did not mine, so deleting its export legs is not a
correction. (Peru sits at 1.098 and Indonesia at 1.248, mildly over-committed; Chile is the clean
case.)

**China's node is short, not long.** BACI imports plus its own mine fall **9,187 kt CC below**
ICSG's smelter intake, pooled. The fan's residual and ADR 0024's bound are already carrying that
gap; the re-attribution does not create it.

**And Chile's own chain is where the copper actually goes.** Prior → posterior, 2010–19:

| | L.0 | netting | netting + bound |
|---|---|---|---|
| CO → Ore (Mass) | 7,230,748 → 2,938,109 (**0.406**) | → 2,831,675 (0.392) | → 2,835,105 (0.392) |
| Ore → MF (Mass) | 7,230,747 → 2,893,317 (**0.400**) | → 2,790,102 (0.386) | → 2,792,817 (0.386) |
| MF → Concentrate (CC) | 38,000 → 26,043 (**0.685**) | → 24,857 (0.654) | → 24,877 (0.655) |
| Concentrate exports (CC) | 20,966 → 13,340 (0.636) | → 12,347 (0.574) | → 12,351 (0.574) |

The `MF→Concentrate` prior of 38,000 kt CC is ICSG's own Chilean mine output, and **the model
deletes a third of it in every run, including the shipped baseline**. Sixty per cent of Chile's ore
goes with it. That is the "Chile cut" named since ADR 0015 and re-flagged as the open lead after
K.1, and it is an order of magnitude larger than anything Round L buys or costs.

So the stop rule is flagging a **symptom**: the re-attribution deepens an existing 31 % deletion of
Chile's ICSG-reported mine output by about 4 points. It is not the re-attribution that is deleting
Chile's copper.

## Recommendation

**Keep the netting with the extended bound, and make Chile's mine chain the next round.**

- The gain is real, attributable and accompanied by mass movement — not a repricing. The
  Germany/Korea/France/Portugal/Netherlands phantom chains are fabrications the model no longer
  needs, and they do not come back.
- The cost is a marginal worsening of a defect that dominates the model and needs its own round
  regardless. Reverting protects a retention figure that is *already* an artefact of that unfixed
  defect (0.636 means 36 % of Chile's exports were being deleted before Round L touched anything).
- Reverting would not move Chile's ore chain at all: 0.406 at L.0 versus 0.392 with the netting.

**The case for reverting instead** is stop-rule discipline: the rule says do not ship a physical
regression, and deepening a known defect while it is unfixed makes the eventual fix harder to
attribute. That is a legitimate call, and it costs only the delay — the netting code, its tests and
this measurement keep.

**Either way the next round is the same**: why the model deletes a third of Chile's ICSG-reported
mine output, and 60 % of its ore.

---

# Decision (2026-09-16): kept, with the extended bound

The owner took the recommendation above. **ADR 0027 is accepted and ADR 0024 amended**; the
shipped run is `opt_20260915_131115_b8bf05a9` (L.2 on BACI v8), sum-row z² **1,720,192**, −19.5 %
against its L.0 predecessor and −26.2 % against the Round K baseline.

This is the first change in the arc shipped **over a failed stop rule**, so the reasoning is
recorded rather than assumed: the rule asks whether a change moves physics in the wrong direction,
and here it was measured to be reporting a pre-existing defect that the change deepens by about
4 points out of 31. Chile's exports are physically supportable (0.845), China's node is short
rather than long (−9,187 kt CC), and Chile's ore chain sits at 0.406 before Round L exists. The
counter-case — that deepening a known defect makes its eventual fix harder to attribute — is why
the two obligations below are obligations.

**Carried out of this round:**

1. **Release the `a_ref` pin** (`LEG_SIGMA_AREF = None`, ADR 0017's own definition) at the start
   of the next round, measured on its own against this shipped run. The pin did its job — it kept
   the L series measuring structure instead of σ — and it must not become permanent, or every
   later round inherits a reference size from the L.0 leg population.
2. **Chile's mine chain is the next round.** Why the model deletes ~31 % of Chile's
   ICSG-reported mine output and ~60 % of its ore, in every run to date, baseline included.

**A reading rule this round earned**: an exporter-retention move is not a verdict on a structural
change until the exporter's own upstream chain has been read. Here it took three read-only
measurements to tell "the change is deleting Chile's copper" from "the change is deepening a
deletion that was already there". Round E's stop rule (a change must move mass, not just the
metric) is unchanged — this round moved 187,790 kt CC — but its retention band now carries that
caveat.

---

# Addendum 3 — the `a_ref` sensitivity, and what it says about ADR 0017

The plan's last open acceptance item (g) was to report the **unpinned** `a_ref` variant as a
sensitivity line. Run `opt_20260916_072151_ac748919` (imp `imp_20260916_070419_9df3101b`), identical
to the shipped chain except `LEG_SIGMA_AREF = None`, so ADR 0017 re-derives the sector medians from
the netted leg population:

| | pinned to L.0 (**shipped**) | unpinned |
|---|---|---|
| `a_ref` Concentrate / Cathode / Anode / Matte (kt) | 10.26 · 0.268 · 0.213 · 0.223 | 12.0 · 0.3 · 0.3 · 0.3 *(as logged, 1 dp)* |
| sum-row z² | **1,720,192** | 1,748,645 (**+1.7 %**) |
| Chile · Peru · Indonesia retention | **0.574** · 0.903 · 0.845 | 0.550 · 0.900 · 0.845 |
| world CO Mass · Concentrate CC · Cathode CC | **0.779** · 0.872 · 1.003 | 0.767 · 0.864 · 1.001 |
| solved | 30/30 | 30/30 |

**The pin was not flattering the metric — it is the better configuration, on every axis measured.**
That was worth knowing, because a pin introduced to protect a measurement is exactly the kind of
control that can quietly become a thumb on the scale. It is not one here: releasing it costs
1.7 % of z², 2.4 points of Chile's retention and a point of world mining retention.

**Which turns a hygiene item into a finding about ADR 0017.** The mechanism runs the wrong way
round. Netting *removes the small hub-side legs*, so the sector median rises; a higher `a_ref`
raises the damped cv of every leg in the sector (more of them reach the 1.0 cap); looser legs are
cheaper for the QP to delete; and deleting them makes the scale-total rows — which anchor the node
sums independently — harder to satisfy, so the headline worsens while retention falls. ADR 0017's
median rule therefore **penalises a cleaner leg population**: improve the bilateral matrix and the
reference size drifts up, loosening σ on the legs that survived. `a_ref` is not a neutral
descriptive statistic; it is a free parameter the leg population moves.

**So obligation 1 changes shape.** The pin must be *decided*, not merely released: either ADR 0017
gets a reference size that does not move with the population it prices (a fixed per-sector constant,
or a quantile of the *gross* matrix, or a physically-motivated size), or the pin becomes that
constant with the reasoning recorded. Releasing it blindly at the next round's start would silently
give back 1.7 % and 2.4 points of Chile. Ranked against the Chile mine chain it is the smaller
question, but it now has a measured answer attached and should not be carried as an unexamined
"restore the default".
