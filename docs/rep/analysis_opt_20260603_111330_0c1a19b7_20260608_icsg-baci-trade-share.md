# ICSG-CC ↔ BACI-Mass trade-edge incoherence — fixed via BACI-trade-share fan weighting

Start path: [data/proc/opt/opt_20260603_111330_0c1a19b7/](../../data/proc/opt/opt_20260603_111330_0c1a19b7/)
(label *"icsg fan in fix row penal"*) — the newest run at investigation time. Builds on the prior REP
[`…_20260603_feed-scale-collapse.md`](analysis_opt_20260602_133908_46bc514e_20260603_feed-scale-collapse.md)
and [ADR 0002](../adr/0002-single-endpoint-rows-fan-out-not-replicate.md).

User question: identify the upstream cause of the extreme ICSG-CC ↔ BACI-Mass mismatch; (A) which edge
types are most reduced in scale and why; (B) the upstream reason for the cathode case; (C) check the
hypothesis "formula-impute cathode→FU Mass from CC via the concentration coefficient"; (D) additional /
substitutional recommendations.

Upstream chain: const `const_20260603_110619_8972e968` → imp `imp_20260603_100019_539b47cc` → feeds
`crude_ore` / `baci` / `icsg` (all 2026-06-03, post ADR-0002 fan-in fix).

---

## 0 — Executive summary

The fan-in fix (ADR 0002) already resolved the gross ICSG mass inflation; the **residual** mismatch is a
**both-observed feed conflict on bilateral trade edges**. On a trade edge X→Y, ICSG-CC is a *uniform
fan-out artifact* (ICSG knows the consumer Y, not the producer X, so it splits national usage evenly
across producers), while BACI-Mass is the *real* X→Y observation. The `conc_*_supply` ratio ties the two
valuations of the edge, so they conflict — up to ~3·10⁵× at the prior.

The cure is **ADR-0002 option D**: weight ICSG's producer fan by BACI trade shares (keep both feeds), so
ICSG-CC inherits BACI's bilateral geography. The domestic leg (X==Y, absent from BACI) carries the
residual `usage − imports`. Implemented and verified: `conc_*_supply` both-observed z>1e3 @2010 drops
**75.9% → 10.0%** (cathode), with the remainder being the legitimate transit-hub totals gap.

> **Verdict.** Not a new bug. ICSG's *uniform* producer fan (ADR-0002 option A) is incoherent with BACI's
> bilateral trade on the shared edge. Option D (BACI-share weighting) reconciles them. The user's
> formula-imputer hypothesis (C) was already implemented for the missing-side case and structurally cannot
> reach the both-observed case — so it is not the right lever.

---

## 1 — Data characterisation

### 1.1 The collapse is global, not cathode-specific (A)

In the start run **89.1%** of solved `a₀≠0` edges land `<1e-6` (F/a₀ p50 ≈ 3·10⁻¹²) — near-uniform across
every sector/valuation. This is the unguarded QP collapse (σ_c/σ_a weight imbalance ~10⁶ + `a=0` feasible
+ `prior_box_k=null`), **out of scope** here (fix = enable the box, REP §8.3 of the prior report). By
**absolute** mass lost it concentrates on high-tonnage low-grade Mass edges (Ore Σ|a₀|=7.2e7, CO 6.2e7,
Concentrate 2.1e6). The fan-in fix already cut median |a₀| 148 → 3.74.

### 1.2 The cathode residual is a both-observed conflict (B)

`conc_cathode` ratio_eq rows @2010, split by whether one side was conc-imputed vs both feed-observed:

| conc_cathode ratio_eq @2010 | n | % \|z\|>1e3 | median z |
|---|---:|---:|---:|
| one side imputed (parametric fills via conc) | 6293 | **0%** | 0 |
| both ICSG-CC ∧ BACI-Mass observed | 381 | **75.9%** | 3797 |

84% of cathode→FU edges already satisfy CC/Mass = 0.9975 exactly. The conflict is **entirely** the
both-observed case, and **364/366** of those are **trade edges** (region_from≠region_to). The conflict
exists on exactly four supplier sectors (Cathode, Concentrate, Anode, Matte), ~99% trade.

---

## 2 — Trace of one specific case

Worst both-observed row @2010: `conc_cathode_supply`, **Cathode/Flow/Switzerland → FU/SB/China**-class
edges. Concretely for consumer **China**, the uniform fan put `7393.1 / 47 = 157.3` CC on **every**
producer including domestic, while BACI shows China importing ~3014 kt Mass (and refining the rest
domestically). For consumer **Switzerland** (a transit hub) the gap is extreme: ICSG allocates CC ≈ 0.05
(tiny actual Swiss consumption) but BACI shows Mass = 277 (gross transit trade) → z ≈ 1.4·10⁵.

**Feed layer.** ICSG is 100% CC (172,399 edges), BACI 100% Mass (41,167). Neither edge was imputed — both
are direct feed observations at incompatible scales because ICSG's per-edge trade value is a fan artifact.

---

## 3 — Hypotheses & verdict

| H | mechanism | result |
|---|---|---|
| H1 | residual is an imputation artifact | **rejected** — both sides feed-observed, not imputed |
| H2 (user, C) | formula-impute Mass from CC via conc fixes it | **rejected as the lever** — already done for the *missing* side (0% violation, 42,406 fills); the engine only fills `np.isnan(f)` ([engine.py:209](../../src/pysuto/impute/engine.py#L209)), so it cannot override the both-observed BACI value — the exact broken edges |
| H3 | ICSG uniform producer fan ≠ BACI bilateral trade | **confirmed** — conflict is 99% trade edges; the fix is to reweight the fan (ADR-0002 option D) |

**Verdict.** H3. The uniform fan is the upstream cause. The conc-imputer (H2) is necessary for the
single-feed edges but structurally blind to the conflict; option D is the cure.

---

## 4 — Fix: BACI-trade-share fan weighting (option D)

Per (sector_from, sector_to, consumer Y, year t), in copper-content units:

```
import X≠Y : raw weight = conc_sector · BACI_Mass_Froot(X→Y, t)
domestic   : raw weight = max(0, ICSG_usage_CC(Y, t) − Σ_X conc_sector · BACI_Mass(X→Y, t))
```

Normalised per src fan ⇒ `F_root_CC(import) = conc·BACI_Mass` (conc ratio holds by construction) and
`F_root_CC(domestic) = usage − imports` (usage conserved). Scope = **Cathode, Concentrate, Anode, Matte**
(Scrap excluded — not traded). Time-variant; per-consumer-group complete fallback (uniform where BACI
absent). See [ADR 0002 §Implementation](../adr/0002-single-endpoint-rows-fan-out-not-replicate.md) and
`scripts/build_trade_shares.py`.

### 4.1 Verification (opt `opt_20260608_064938_44faa1f1`)

China cathode→FU @2010 fan: uniform 157.3×47 → **domestic 4389, imports 3004, total 7393** (= usage,
conserved); top importer Chile (1313). `conc_*_supply` both-observed z>1e3 @2010:

| family | old (uniform) | new (BACI-weighted) |
|---|---:|---:|
| conc_cathode | 4.33% | **0.57%** |
| conc_concentrate | 2.13% | **0.09%** |
| conc_anode | 1.20% | **0.09%** |
| conc_matte | 0.00% | 0.09% |

(cathode both-observed subset 75.9% → 10.0%.) ICSG mass conserved (ΣF_root = 2,069,585 kt).

---

## 5 — Residual & recommendations (D)

The remaining ~10% both-observed residual is two **separate** issues, not the uniform-fan bug:

1. **Transit-hub overstatement (dominant).** For Switzerland/Netherlands etc., BACI gross trade ≫ ICSG
   consumption; imports compress (domestic clips to 0) and CC < conc·Mass. This is the legitimate
   usage-vs-trade totals gap — the conc ratio assumes the traded mass is consumed there, which transit
   violates. *Recommendation:* route transit/re-export through a distinct path rather than final-use, or
   accept it as a soft-constraint residual.
2. **Stale BACI domestic self-trade artifact.** BACI has no self-trade in raw, yet a few domestic Mass
   edges carry a small BACI fan artifact (58 cathode→FU domestic edges, median 33) that now conflicts with
   the corrected domestic CC. *Recommendation:* suppress BACI's spurious domestic fan so the conc-imputer
   fills domestic Mass coherently.
3. **Collapse (unchanged, separate).** `Σ|F*|/Σ|a₀|` stays collapsed until `prior_box_k` is enabled — this
   change is data-coherence, not the collapse cure.

---

## Source artefacts

- Opt (new): [data/proc/opt/opt_20260608_064938_44faa1f1/](../../data/proc/opt/opt_20260608_064938_44faa1f1/)
- Const (new): [data/proc/const/const_20260608_064745_e014d751/](../../data/proc/const/const_20260608_064745_e014d751/)
- Imp (new): [data/proc/imp/imp_20260608_064502_f58f9470/](../../data/proc/imp/imp_20260608_064502_f58f9470/)
- ICSG feed (BACI-weighted): `data/proc/feed/registry/feed_icsg_20260608_064345_d619cf04/result.npz`
- Trade shares artifact: `data/input/params/trade_shares/current/params.csv` (153,361 rows, 3,263 groups)
- Old baseline const (uniform fan): [data/proc/const/const_20260603_110619_8972e968/](../../data/proc/const/const_20260603_110619_8972e968/)
- Code: `scripts/build_trade_shares.py`, `scripts/icsg_feed.py`, `src/pysuto/conc.py` (region+time `user_weights_df`)
