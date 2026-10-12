# #40: negative stock changes reach the account, and expose two fan defects

**Run:** `opt_20261012_084619_3c11d751` (label `r40_signed_stock`; 29 Solved + 2016 AlmostSolved;
`prior_mode` cold)
**Against:** Round Z `opt_20261008_092950_625a6c7e` (28 Solved + 2018 and 2022 AlmostSolved), which
differs only in the trade-share weights of the MESC and USC fans
**Chain:** trade_shares **v24** → `feed_icsg_20261012_075625_4bf3d081` (+ Z's
`feed_baci_20261008_084113_ebca1805`, `feed_crude_ore_20260603_095651_558f7369`) →
`imp_20261012_080051_be290f12` → `const_20261012_080529_298b856a` → the run
**Feeds/params:** icsg v12, baci netted v14, trade_shares **v24** (Z: v23), acc copper v4, conc v10,
eff v4, ore_grade v7, stock_bounds v5, scrap_share v3, stripping_ratio v2
**Date:** 2026-10-12 · **Branch:** `fix/signed-stock-fans` · **Issue:** #40 · **ADR:** 0008, 0024,
0040, 0041 (none amended)

**Verdict: not accepted as measured.** The fix does what #40 asked: every negative MESC and USC change
outside Taiwan now reaches the prior, and the Round-E items hold. But pre-registered item 4 fails in
2003, and common-set sum-row z² rises **+9.22 % vs `opt_20261008_092950_625a6c7e`**. 97.2 % of that
rise sits on 48 cathode use totals of six exchange hubs, whose draw-downs the fan books on cathode
those hubs never produced. A second defect surfaces on the same legs: 1,250 stock leg-years carry a
negative CC prior and a positive Mass prior. Both are fan rules, not data, and need a decision
before the next round (§5).

## 1. Defect

`compute_trade_share_weights` (`scripts/build_trade_shares.py`) emitted a fan group only where the
reported usage is > 0. A MESC or USC fan therefore had trade-share weights if and only if its
reported change was positive, and the concordance refused every unweighted fan (ADR 0008).

In-memory concordance on the Round Z inputs (icsg v12, trade_shares v23):

| leg | raw rows | negative | distributed | negatives distributed |
|---|---|---|---|---|
| MESC | 225 | 102 | 76 | 0 |
| USC | 160 | 48 | 51 | 0 |

All 150 negatives and all 102 zeros were *undistributed*, none *unmatched*.

## 2. Change (commit `9e83a93`)

For the signed legs (MESC, USC) the fan is computed on the magnitude of the reported change. The
weights stay non-negative and the sign travels with the value. A zero still emits no fan: an
undistributed zero and the ADR 0040 net-stock seed carry the same prior and σ. Flows keep their
rule. trade_shares v24 differs from v23 only by 4,559 MESC and 2,256 USC weight rows.

With v24 the concordance distributes MESC 173 / 225 and USC 99 / 160. The five negatives left are
Taiwan, trade-blind for every sign (ADR 0018). No row is undistributed because of its sign.

## 3. Pre-registered items (posted on #40 before any number was read)

| # | prediction | result | |
|---|---|---|---|
| 1 | the ICSG feed's F_root carries negative Cathode→MESC / USC values | 1,573 MESC and 770 USC negatives (Z: 0) | pass |
| 2 | the imp prior carries negative MESC / USC edge-years | MESC 966 CC + 80 Mass, USC 390 CC + 26 Mass (Z: 0) | pass |
| 3 | the MESC / USC posterior carries both signs | yes, but so did Z's since Round Y | pass, uninformative |
| 4 | Σ\|a* − a_prior\| on MESC + USC (CC) grows < 100 kt in every year | max +121.2 kt in 2003; next 2018 +96.5, 2014 +72.7 | **fail (2003)** |
| 5 | Round-E: world retention, small-node median, exporter retention each within ±0.01 | all within (below) | pass |
| 6 | prior changes confined to edges leaving a Cathode node and the rows that contain them | 1,502 moved priors, all leaving a Cathode node; 643 scale totals moved, all cathode use totals | pass |

Item 1's other F_root values equal Z's within 1.4e-12, which is summation order in the larger
concordance.

**Round-E items (ADR 0025), Z → this run:**

- **(a) World retention:** CO Mass 0.860 → 0.861, Concentrate CC 0.990 → 0.990, Cathode CC
  0.987 → 0.987.
- **(b) Small nodes (|c| < 18 kt):** posterior/prior median 1.000 → 1.000, p90 1.140 → 1.124.
- **(c) Exporter retention:** Chile 0.964 → 0.966, Peru 1.009 → 1.010, Indonesia 0.911 → 0.910.
- **Mass moved, all CC edges:** 499,701 → 497,342 kt (−0.47 %).

## 4. Cost

**Common-set sum-row z²** (27,032 rows in both runs; built in memory, no per-year CSV):

| | 30 y | 1996 | 2000 | 2003 | 2014 | 2016 | 2018 |
|---|---|---|---|---|---|---|---|
| Z | 1,402,531 | 103,265 | 32,251 | 67,511 | 23,455 | 33,990 | 51,142 |
| **this run** | **1,531,809** | 118,946 | 43,739 | 89,707 | 30,193 | 47,576 | 70,184 |
| Δ vs Z | **+9.22 %** | +15.2 % | +35.6 % | +32.9 % | +28.7 % | +40.0 % | +37.2 % |

The move is far outside the ±0.1 % band of solver inexactness (ADR 0039), so no cross-evaluation
was run. By anchor band, Δz²: < 10 kt **+121,535**; 10–100 kt +8,620; 100–1,000 kt +528;
≥ 1,000 kt −1,404.

**Where it sits.** 48 rows, `sum_usetotout_Cathode_<hub>` with a negative target, carry 125,646 of
the 129,278 (97.2 %). Every one lands at z ≈ 58.6, because σ_c is the anchor's 1.7 % of |c| and the
posterior holds the row at about zero:

| hub | rows | Δz² | lowest c (kt) |
|---|---|---|---|
| Netherlands | 16 | 45,794 | −50.5 |
| Singapore | 16 | 38,749 | −41.6 |
| Malaysia | 8 | 20,539 | −66.5 |
| France | 4 | 9,300 | −17.5 |
| United Arab Emirates | 2 | 6,887 | −58.9 |
| United Kingdom | 2 | 4,376 | −11.1 |

All other sum rows move +3,632, which is +0.26 % of Z's total.

**Mechanism, United Arab Emirates 2003 (CC).** The use-total row sums every leg out of the UAE's own
cathode node, to all 47 destinations. ICSG reports a −59.9 kt LME draw-down at Jebel Ali. The UAE's
recorded cathode imports that year are too small to carry it, so ADR 0024's production bound does
not fire, and the fan puts the whole draw-down on the domestic leg
`Cathode(UAE) → MESC(UAE)` (prior −59.875, σ 4.1, bounds ±1e6). That books the draw-down as cathode
the UAE produced, but the UAE refines none. Its cathode node's hard balance therefore cannot release
59.9 kt. The posterior takes the leg to −1.84 kt and leaves the use total at +0.01 against its −58.9
target. A build at a hub has the same shape, but Z's builds at these hubs were small (c ≤ 11 kt), so
the defect stayed hidden until draw-downs entered.

**Second defect: the two valuations of a stock leg disagree in sign.** ICSG reports stock changes in
CC only. The Mass prior of the same legs comes from BACI's 98/1/1 cathode fan (#36 item 1), which is
always positive. On **1,250 leg-years** (886 MESC, 364 USC) the CC prior is now negative while the
Mass prior is positive: Σ CC −565.0 kt, Σ Mass +683.7 kt. The posterior keeps every one of them
(Σ CC −527.1, Σ Mass +653.1), so nothing ties a stock leg's two valuations together. The sum-row
metric does not see this.

## 5. Next round: two fan decisions

1. **Draw-downs and builds at hubs that produce no cathode.** Proposed: for the signed legs, bound
   the domestic leg by the region's own production unconditionally, and spread the remainder over
   the import legs in BACI proportion even where recorded imports are smaller than the change. A
   hub with neither imports nor production leaves the change undistributed, and the net-stock seed
   prices it. Alternative: leave such a change undistributed whenever imports cannot carry it.
2. **The Mass prior of a stock leg.** Proposed: #36 item 1, BACI's cathode fan becomes 100 / 0 / 0
   onto FU / USC / MESC, so a stock leg's prior comes from ICSG alone and its Mass follows from CC
   through the concentration. Alternative: keep BACI's 1 % and accept the sign disagreement.

Both decide how a signed stock change enters the prior, so they belong in one round, measured
against Z and against this run.

**State left behind.** trade_shares v24 is the registered latest version and `current/`. Main's
`build_trade_shares.py` would rebuild v23's content, and any feed rebuild runs it first (ADR 0024 /
0027 order), so main is not affected. The code fix stays on `fix/signed-stock-fans`, unmerged.

## 6. Reproduce

From the repo root on `fix/signed-stock-fans`, strictly in sequence (the registries have no
locking):

```
python scripts/build_trade_shares.py                           # → trade_shares v24
python scripts/icsg_feed.py                                    # sidecar: trade_shares v24
python scripts/collect_aggregate_impute.py --feed-id feed_icsg_20261012_075625_4bf3d081 \
       --feed-id feed_baci_20261008_084113_ebca1805 \
       --feed-id feed_crude_ore_20260603_095651_558f7369 --label r40_signed_stock
python scripts/constraints.py --imp-run-id imp_20261012_080051_be290f12 r40_signed_stock
python scripts/run_solver.py --const-run-id const_20261012_080529_298b856a \
       --label r40_signed_stock --prior-mode cold

python scripts/explo/issue40_signed_stock_diag.py opt_20261008_092950_625a6c7e \
       opt_20261012_084619_3c11d751
```

No per-year tables were written (#46). The diagnostic reproduces the Round Z report's figures
exactly when run on Z.base and Z (common-set z² 1,442,713 → 1,402,531, −2.79 %).
