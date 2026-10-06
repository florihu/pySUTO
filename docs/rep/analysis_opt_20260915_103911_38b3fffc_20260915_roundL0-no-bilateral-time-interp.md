# Round L.0 — the interpolator stops inventing bilateral trade (ADR 0026)

- run `opt_20260915_103911_38b3fffc` ← `const_20260915_100002_fe3ed012` ←
  `imp_20260915_095200_64dfaa88`, cold, 30/30 Solved
- feeds **pinned to the baseline's**: `feed_icsg_20260910_131054_1fffbce3`,
  `feed_baci_20260902_085153_de9e393f` (BACI input v5),
  `feed_crude_ore_20260603_095651_558f7369` — nothing upstream of Stage 2 was rebuilt, so the
  change is isolated to one constant
- change: `TIME_SCOPE_SAME_REGION = True` — `TimeImputer(scope_edges = region_from == region_to)`
- baseline: `opt_20260914_090327_c21a5c48` (shipped Round K.1), sum-row z² **2,331,632**
- reproducer: `python scripts/explo/roundL_reexport_diag.py opt_20260914_090327_c21a5c48 opt_20260915_103911_38b3fffc`
- branch `feat/baci-reexport-netting`; prerequisite for Round L.1. ADR [0026](../adr/0026-bilateral-trade-legs-are-not-time-interpolated.md).

## 1. Why

`TimeImputer` was registered with no scope, so it grouped every edge by
`(from_id, to_id, valuation)` and linearly interpolated any year missing between two observations
— bilateral BACI legs included. BACI is a **complete** matrix: a route carrying no row in a year
carried no trade. The pipeline already said so — ADR 0010 ("a bilateral import leg with no trade
observation means 'no trade on this route' — 0 is the honest value") and GLOSSARY's *Trade-seed*
entry — and ADR 0024 listed the question as an open follow-up. The unscoped imputer silently
overrode all three.

On the shipped baseline it invented **9,215 cross-region Mass legs carrying 20,252 kt** of trade,
of which the posterior kept 15,266 kt (retention 0.754). The largest blocks were the region-years
the model has least business inventing: Papua New Guinea concentrate 2,232 kt (a source defect the
Stage-2 outlier screen flags in 2019–2022, carried by the interpolator into the years between),
Zambia cathode 1,027, Peru 1,047, ROW 925, Zambia matte 724, Congo D.R. 702.

## 2. Measured

| | base | L.0 |
|---|---|---|
| solver | 30/30 Solved | 30/30 Solved |
| **sum-row z²** | **2,331,632** | **2,136,925 (−8.4 %)** |
| of which `usetotout` | 1,560,890 | 1,391,003 (−10.9 %) |
| of which `supplytotout` | 770,742 | 745,922 (−3.2 %) |
| `time_linear` fills, total | 18,838 | 408 |
| — cross-region | 18,430 | **0** |
| — cross-region **Mass** | 9,215 (20,252 kt) | **0** |
| total imputer fills | 785,927 | 785,927 |
| hub∪fabricated z² (Round C classes fixed on the base run) | 343,788 (14.7 %) | 279,637 (13.1 %) |
| Σ\|Δposterior\| on CC edges | — | **92,213 kt (1.34 %)** |

The edge set is unchanged and the total fill count is identical: the freed edges fall through to
the flat prior and ADR 0013's no-data anchor, which is the state ADR 0010 designed for them.

**The metric moved with mass.** 92,213 kt of contained copper relocated in the posterior for a
194,708 z² gain — the opposite of Round E's signature (0.46 % of mass for a 4 % metric gain), and
the reason this ships rather than reading as a repricing.

## 3. Against the stop rule

- **(a) cross-region `time_linear` fills = 0.** Met exactly, by construction.
- **(b) exporter retention (concentrate, cross-region CC, 2010–19).** Chile **0.618 → 0.636**,
  Peru 0.947 → 0.944, Indonesia 0.841 → 0.850, world 0.815 → 0.823. Chile's +0.018 exceeds the
  ±0.01 band **upward** and is reported as such: removing invented legs from the Chilean corridor
  lets more of the observed ones survive. No exporter regresses in volume.
- **(c) world retention.** CO Mass 0.785 → 0.788, Concentrate CC 0.881 → 0.882, Cathode CC
  1.014 → 1.012. All within ±0.003.
- **(d) the Zambia finding is not hidden.** Zambia→CH/SG/LU anode retention 0.021 → 0.019; its
  other destinations 1.047 → 1.063. The model still refuses the phantom by capacity, at the same
  node, for the same reason.
- **(e) 30/30 Solved; suite 909 passed / 1 skipped** (906 baseline + 3 new).

## 4. Where the gain came from — and where tension relocated

Every large fall is a region-year whose *interpolated* trade the model had been paying for:

| cell | base | L.0 | Δ |
|---|---|---|---|
| Concentrate / Japan | 36,385 | 21,794 | −14,591 |
| Concentrate / Austria | 14,818 | 323 | −14,495 |
| Matte / Italy | 33,869 | 20,835 | −13,034 |
| Concentrate / Korea, Rep. | 22,479 | 9,767 | −12,712 |
| Cathode / Portugal | 30,004 | 18,646 | −11,357 |
| Concentrate / Belgium | 13,645 | 3,274 | −10,372 |
| Anode / Netherlands | 68,411 | 58,950 | −9,461 |
| Matte / Austria | 8,355 | 1 | −8,354 |
| Anode / Papua New Guinea | 7,097 | 520 | −6,577 |

Portugal Cathode — ADR 0025's headline cell, 90,160 z² before Round K and 30,004 after — falls a
further third once its invented legs are gone. Austria concentrate and matte effectively
disappear: they were **entirely** interpolation.

Tension relocated rather than vanished in a few places, and is reported rather than smoothed:
CFR/Anode Belgium +5,049/+4,624, Cathode Zambia +2,316, Cathode Austria +2,249, CFR Zambia +2,177,
MF Botswana +1,710, ER/CER Portugal +1,543 each. These are nodes whose balance was being closed by
a fabricated import; with it gone, the disagreement between their remaining priors becomes visible.
That is the intended direction — a contradiction that is *seen* is worth more than one papered over
by invented trade — but it is a real residual and it is named here.

Two counters moved the wrong way and are small: phantom production at Concentrate/Korea
745 → 791 kt pooled over 30 years, and Anode/Netherlands 41 → 54. Both are hub nodes whose export
legs still require a domestic origin; that is precisely the defect **Round L.1** addresses, and
neither is affected by this change's mechanism.

## 5. Decision

**L.0 ships.** It restores an intent the codebase already documented in two places, it is one
constant with a test pinning the pre-ADR behaviour, and it is a prerequisite for L.1: the
re-export netting drives a hub's export legs to zero in some years, and the unscoped interpolator
would write them straight back from the adjacent partially-netted years — the mechanism that
carried Round J's phantom United States legs into 2000.

**Decided 2026-09-15: L.0 shipped.**
