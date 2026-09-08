# Round C — transit/re-export diagnosis (measure before designing)

*Run analysed: `opt_20260903_124655_4b23df2f` (shipped γ=0.5 chain, main\@76dd0b3);
Stage-2 frame: `imp_20260903_122719_f13ea10b`. Script:
`scripts/explo/roundC_transit_diag.py`; tables under
`data/proc/opt/opt_20260903_124655_4b23df2f/diag/roundC_transit/`. Companion code
change (C1) validated in §5 against the re-run chain
`imp_20260907_100838_54df1d05 → const_20260907_101519_8367d1b1 →
opt_20260907_112840_a8677803`.*

Question from the Round-B wrap-up: the remaining sum-row tension (z² ≈ 3.78M) shows a
"fabricated chain / missing transit" signature — regions exporting what they only
imported, forced into invented domestic production chains because every BACI gross
export must originate from domestic production. **How much of the tension does that
mechanism actually explain, and what should Round D build?**

## 1 · Hub table: production basis, imports, exports (P / I / X)

Per (region, traded sector, year), from covered Stage-2 data only: implied CC exports
X and imports I (bilateral Mass legs × conc_nominal), and covered production basis P.
Since ICSG keys production data by *process input* (`Refining_Matte` → Matte→CFR
etc., `icsg_feed.py`), P = direct Process→Flow CC (Concentrate, CSXEW) **plus covered
feed of the producing chain** (Smelting feed → Matte; CFR feed → Anode; CFR+ER feed +
SXEW cathode → Cathode). Imported feed counts — Japan's smelter/refinery chain is
production, not transit. Classification (X > 1 kt CC):

| class | rule | group-years | X (kt CC) | I | P |
|---|---|---|---|---|---|
| **hub** | X > 3·P but X ≤ 1.5·(P+I) | 330 | 6 845 | 70 714 | 561 |
| **fabricated** | X > 3·(P+I) | 296 | **56 290** | 114 | 2 461 |
| ordinary | rest | 2 607 | 427 460 | 368 433 | 1 616 423 |

- **True hubs** (exports explained by imports): Germany Concentrate (30/30 years,
  I = 8.3 Mt), France Cathode (24), Singapore Cathode (21), Switzerland Concentrate
  (21), **Japan Concentrate (20)**, Korea Concentrate (17), UK Cathode (17),
  Netherlands Cathode (12), Malaysia (12+12), Slovakia/UAE Cathode (11) — the
  expected transit-hub roster, plus the smelter re-exporters the C1 screen fix
  unflags (§5).
- **Fabricated** (exports with *neither* production nor import basis): 56.3 Mt CC
  over 30 years. Top: PNG Concentrate (the screened BACI absurdity), Kazakhstan /
  Zambia / Poland / Russia / Philippines / Bulgaria Cathode, Zambia + Congo D.R. +
  Slovakia + Netherlands Anode. **Important split**: Kazakhstan, Zambia, Russia,
  Poland are *real producers with ICSG coverage gaps* — their exports are genuine and
  the missing piece is production data, not a re-export concept. The genuinely
  structural cases are the Anode hubs (Slovakia 97.7×, Netherlands 92.6× production)
  and small-refiner Cathode cases (Bulgaria, Portugal 58.8×).

## 2 · How much z² does it explain?

Sum-row (supplytotout/usetotout) z² pooled 1995–2024, both valuations, total
**3 782 125**:

| attribution | z² | share |
|---|---|---|
| usetotout rows of hub region-sectors | 787 159 | 20.8 % |
| usetotout rows of fabricated region-sectors | 925 582 | 24.5 % |
| union (no double count) | **1 387 003** | **36.7 %** |
| all sum rows in regions with any hub sector (upper bracket) | 1 421 738 | 37.6 % |

The Round-B guess ("~0.5–0.9M block") was an underestimate: the missing-transit +
missing-production-basis mechanism accounts for **≈ 1.4M of 3.78M (~37 %)** of the
remaining sum-row tension. The top-15 z² region-sectors are dominated by it:
Slovakia Anode 137k (fab), Portugal Cathode 93k (hub), Netherlands Anode 91k
(hub/fab), Japan Concentrate 75k (hub — C1 false positive, now fixed), Spain
Concentrate 71k (hub), Bulgaria Cathode/Anode 116k (fab). Not everything is transit:
India Concentrate 113k (transit index 0.5) and the Chile/Germany Ore rows (54k each)
are different mechanisms (usage-side and ore-data tension). Region-level
Spearman(z², transit excess) = 0.08 — transit is *a* driver, concentrated in specific
region-sectors, not a global gradient.

## 3 · The feed-time clip already sees the hubs (independent evidence + mass leak)

The trade-share fan clips the domestic residual at zero:
`max(0, usage − Σ imports)` (`scripts/build_trade_shares.py:105`). For a hub
(imports > usage) the excess import is **silently discarded at feed time**:
**55 248 kt CC over 285 group-years**. Top: ROW Concentrate 16.5 Mt, Philippines
Concentrate 13.0 Mt, Netherlands Cathode 7.6 Mt, Switzerland Cathode 7.2 Mt, UK
Cathode 3.3 Mt, then Spain/Japan Concentrate, Malaysia/Singapore Cathode — the same
roster as §1, derived from a completely independent code path. Round D must close
this leak: material that arrives at a hub and leaves again currently vanishes from
the priors *and* must be re-invented behind the export.

## 4 · FD-faucet ledger (signature 1, for completeness)

Posterior CC inflows into MESC/USC/FU (Flow→SB) total 608.5 Mt pooled; by the
imp-log method rule (final method per edge, **not** `f == 0`):

| prior method | Σ a_prior | Σ a_post | Σ\|Δ\| | edge-years |
|---|---|---|---|---|
| covered / untouched | 567 658 | 561 308 | 51 285 | 12 578 |
| parametric | 42 574 | 19 393 | 23 488 | 12 906 |
| **share_seed** (net-stock shares) | 0 | **21 956** | 21 957 | 13 780 |
| flat_prior_noisy_zero | 0 | 92 | 92 | 154 530 |
| network/time fills | 8 329 | 5 792 | 6 866 | 5 016 |

The data-free faucet is **share_seed**, not the flat fills (92 kt across 154k
edge-years — the noisy-zero design holds). 22.0 Mt CC of posterior flow sits on
share-seeded zero priors; top absorbers: **China MESC 10.3 Mt**, Belgium 2.1, Russia
1.7, ROW 1.0, Zambia 0.8. China's Cathode→MESC series grows from ~40 kt/yr
(1998–2003) to 400–1200 kt/yr (2010–2024) against ~0 priors — the *magnitude* is
coherent with known SHFE + bonded-warehouse inventory swings (hundreds of kt/yr;
sanity check, not calibration), so the faucet is partly real stock physics.
Pricing: `netstockbound` σ holds the bulk (|z| p50 = 0.10, p95 = 5.25) but the tail
saturates at |z| ≈ 35–42, and the worst rows are **PSC bounds in Singapore, Italy,
UAE, Netherlands, France** (lhs ≈ −110…−129 kt vs σ ≈ 3) — the transit signature
again: invented chains behind hub exports drain post-consumer stocks. The PSC z-tail
follow-up from the stock-bounds round and the transit problem are the same knot.

## 5 · C1 validation: import-adjusted screen reference

The outlier-leg screen's reference widened from covered production to **production +
covered imports** (`outlier_ref = "production+imports"` in `stage_config`); groups
without covered production stay unjudged — σ-nuking the ~1 100 P=0 exporter-years
(168 Mt CC of exports, §1's transit population) would have been a hub policy in
disguise, which belongs to Round D. Flag diff (`diag/outlier_leg_screen.csv`):

- before: 19 groups — Japan ×7, Spain ×6, Italy 2002, PNG ×4;
- after: **5 groups — PNG ×4 (ratios 62–136) + Italy 2002 (3.3, imports don't
  explain it)**. Japan 1996–2004 and Spain 1999/2011–18 unflag as intended.

Full-chain re-run vs the shipped baseline (same feeds, cold, 30 years;
`opt_20260907_112840_a8677803`):

| metric | baseline γ=0.5 | Round C screen |
|---|---|---|
| solver | 30/30 Solved | 30/30 Solved |
| total soft z² | 3.957 M | 3.957 M |
| soft z p99 | 10.38 | 10.39 |
| tailings z² | 8.57 k | 8.60 k |
| Cathode_China max \|z\| | 18.14 | 18.14 |
| deleted share d0→d9, bilateral legs 2015 | 0.17→0.20 | 0.17→0.20 |
| Chile 2015 conc exports / 2019 | 0.511 / 0.417 | **0.536** / 0.413 |
| world 2015 CO / Conc / Smelt / Cath retention | 0.647/0.755/0.706/0.993 | 0.652/0.758/0.710/0.996 |
| Japan conc exports, prev-flagged years | 0.113 | **0.129** |
| Spain conc exports, prev-flagged years | 0.227 | **0.257** |

Nothing moves materially; Chile even gains. Japan/Spain regain σ *authority* (cv
returns from 5 to the damped pedigree level) and their retention rises ~15 %
relative — full recovery is blocked by the same missing-transit structure this REP
diagnoses (their exports still lack a domestic production basis in the sum rows),
which is precisely Round D's target. (Decile-retention definition here: pooled
covered bilateral Mass legs, count-rank deciles, 2015 — internally consistent
across both columns.)

*Run note: the 30-probe feasibility pre-check left enabled by the sensitivity
branch costs ~350 s/probe on the full hard system (~3 h); baseline-era logs show
it completing in 0 ms, i.e. it was effectively not running then. Restored
`FEASIBILITY_CHECK_FIRST_ONLY = True` in `scripts/run_solver.py` — each year's QP
reports its own status regardless.*

## 6 · Round-D recommendation

The evidence splits the problem in three; only one part needs new structure.

1. **True transit hubs** (§1 hub class, §3 clip): the mechanism is *gross* trade
   recorded at re-export points. **Recommended: (a) net-trade preprocessing at the
   feed level, per region-sector-year total** (exports reduced by re-exported
   imports, import shares rescaled) — it simultaneously removes the fabricated-chain
   pressure *and* the §3 clip leak, is confined to Stage-1 feed semantics
   (CONTEXT §Feed-semantics + ADR 0014 amendment), and needs no accounting-structure
   change. Partner-pair netting is insufficient (A→hub→B is not symmetric).
   **(b) an explicit re-export edge class** (Flow→Flow cross-region; ADR 0011
   unit, new PSUT block) is the honest end-state *if* the manuscript needs true
   material provenance through hubs; defer until (a)'s residual says it is worth a
   structural block. **(c) hub-scoped σ/anchor exceptions: rejected** — it prices
   away the symptom, teaches the model nothing, and §1 shows the hub set is dynamic
   (France appears, Slovakia flips class over time).
2. **Producer coverage gaps** (Kazakhstan/Zambia/Russia/Poland Cathode, Congo
   Anode): genuine exports, missing production data. This is a *data* work item
   (ICSG refined/EW coverage for CIS + African producers, or an export-implied
   production prior), not a re-export concept — netting would wrongly delete these
   exports.

   > **Amended 2026-09-08 — the mechanism is not an ICSG coverage gap.** ICSG carries
   > full Refining/Smelting rows for every one of these regions (Kazakhstan 30/30 years,
   > 256–460 kt/yr; likewise Zambia, Russia, Uzbekistan). *The pipeline drops them*:
   > `build_trade_shares` emitted fan weights only where BACI recorded ≥1 import, and
   > ADR 0008's "no weight → no distribution" then discarded the whole fan — 469 fans,
   > ~45 Mt CC never reaching any edge, in region-years matching this class exactly.
   > So it is not a data-acquisition item but a one-predicate feed fix; see
   > [ADR 0018](../adr/0018-import-free-fans-distribute-domestically.md) and the
   > Round-D REP. The conclusion that netting must not fire on these exports stands —
   > they are real, and they now have a production basis behind them.
3. **The faucet knot** (§4): keep `share_seed` + `netstockbound` as designed, but
   Round D's netting should be validated against the PSC tail — the Singapore/UAE/NL
   PSC violations should collapse when hub exports no longer require invented
   domestic chains.

*No implementation in Round C (per plan); Round-D decision pending user review of
this REP.*
