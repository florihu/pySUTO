# Round Y — the producer stock change has no Mass prior, a hidden zero pin in CC, and the signed legs carry a one-signed hard band

**Run analysed:** `opt_20261005_134720_ee69cda9` (X.0, label `roundX0_stock_legs_at_prior`; 30 years
Solved; `prior_mode` cold) — the Round X reference run (ADR 0040)
**Runs measured:** Y.0 `opt_20261006_105904_61c5efe9` (label `roundY0_psc_mass_twin`; chain
`imp_20261006_101314_c6b6c588` → `const_20261006_101742_205d50d3`; code at commit `2e9240e`; 30 years
Solved; conc **v9**, feeds and trade_shares v21 as X.0) — #43; Y.1 `opt_20261006_120237_643c4b4a`
(label `roundY1_psc_stock_leg`; chain `imp_20261006_111616_6566f4d2` → `const_20261006_112045_f55da888`;
code at `48787a1`; 30 years Solved; conc **v10** — the comment cell of the PSC row only; same feeds) — #44
**Against:** X.0 (the run analysed)
**Chain (X.0):** `imp_20261005_125928_e9ffa8cb` → `const_20261005_130418_a5fb94a6` → the run
**Feeds/params (X.0):** icsg v12 / trade_shares v21, baci v11, crude_ore v3; acc copper v4; conc v8,
eff v4, stripping_ratio v1, ore_grade v7, scrap_share v3, stock_bounds v5
**Date:** 2026-10-06 · **Branch:** `fix/psc-prior-both-valuations` (off Round X) · **Spec:** #41
(measure tickets #42 → #43 → #44; feed follow-up #40) · **ADR:**
[0041](../adr/0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md) (proposed)

**Verdict (measurement, §7): Y.0 accepted, Y.1 rejected on measurement.** The cathode
concentration row plus the band rule close the Mass sink (gap 22–87 → 0–6 kt/yr) at z² −0.1 % with
Round-E held and the USC/MESC legs two-signed for the first time. The PSC net-stock seed at 1 % of
`CP→Cathode` is a sink: 870 seeded legs move 7,943 kt CC over 30 years (272 kt/yr against the 100 kt/yr
limit, 2.65σ on average, 6.2σ at p90 — China −1,546 kt, Zambia +857 kt at σ 3.8, Austria +356 kt at
σ 0.9), the PSC stock-bound rows go from max |z| 4.48 to 22.6, CP→Cathode retention moves 0.028 in
Chile, and the common-set z² "improves" −10.3 % because forced contradictions became discretionary
stock changes (ADR 0040's sink). The hidden eff-less pin (H5) stays open for a tighter price.

**Verdict (diagnosis):** the Mass PSC prior is flat at 0 on
every one of the 1,410 region-years because nothing can produce one — ICSG reports PSC in CC only
and no concentration row matches `*→PSC`, so the network twin, the concentration imputer and the
share transfer all have nothing to work with, and Stage 3 builds no Mass↔CC tie. Those legs sit at
σ 1e6, outside the no-data pin and anchor (ADR 0013), and are a free signed sink: ≈ 1,380 kt over
30 years, 22–87 kt in every year. The CC prior "varies" only where ICSG reports (187 region-years);
on 875 others it is a hard zero, because the Cathode Pool has no efficiency row and the recipe wrote
`PSC = (1 − 1.0)·Σin = 0` with σ 0, which Stage 3 floors to 1e-6. A third defect sits behind the
other two signed legs: `conccathodesupply` matches every edge leaving Cathode, so its hard band
already forbids a negative `Cathode→USC` / `Cathode→MESC` — not one of 265,080 edge-years is
negative on X.0, which is the one-directional +16 kt/yr residual ADR 0040 recorded without a cause.

---

## 1 — Observations as given

The user read SD-R-10 of X.0 (`diag/sd_r_10_boundary_flows.png`): in the SB-output panel the `PSC`
prior line is flat at 0 in the Mass valuation while the CC valuation shows a varying prior. Their
hypothesis: "the imputation did not use the Cathode conc to impute PSC".

## 2 — Pipeline trace setup

| observation | owning stage | first place to look |
|---|---|---|
| Mass PSC prior flat at 0 | Stage 2 (imputation) | `imputation_log.parquet` for the 1,410 Mass `CP→PSC` edges; the conc params |
| CC PSC prior varies | Stage 1/2 | `agg_result.npz` coverage (feed) vs the network's recipe writes |
| what the Mass leg does in the solve | Stage 3/4 | the rows of `G` touching a Mass PSC edge; σ_a from `sigma_a.npy` |

Sidecar knobs of note (X.0 const): `nodata_anchor_exclude` and the pin exclusion both list `PSC`;
`pin_flat_prior_zeros: true`; `sigma_a_abs_floor: 1e-6`; `stock_bounds_mode: soft`; imp:
`stock_sectors: [MESC, USC]`, `flat_mode: noisy_zero`, `flat_sigma_high: 1e6`.

## 3 — Data characterisation

### 3.1 Provenance of the 2,820 `CP→PSC` priors (1,410 per valuation)

| source (imputation log / coverage) | CC | Mass |
|---|---:|---:|
| feed-observed (`coverage > 0`, all from `icsg`) | 187 | 0 |
| `network_forward` (every one f = 0, σ = 0) | 872 | 0 |
| `time_linear` (f = 0, σ = 0, between network zeros) | 3 | 0 |
| `flat_prior_noisy_zero` (0, σ 1e6) | 348 | 1,410 |
| `network_mass_twin` / `parametric` / `share_transfer` | 0 | 0 |

Stage-3 σ_a on the same legs: CC 875 at 1e-6 (pinned), 348 at 1e6 (free); Mass 1,410 at 1e6.
Bounds `l/u` ±1e6 on all (the signed leg, ADR 0039). The observed CC legs: Chile 25 years, United
States 25, Japan 23, Canada 21, Brazil 16; 88 of the 187 values are negative (min −1,313 kt, the
garbled US row, #35).

### 3.2 What the free Mass sink absorbs: d = F_Mass − F_CC/0.9975 on PSC

| CC class of the region-year | n | Σ\|d\| kt (30 y) | Σ\|F_Mass\| kt |
|---|---:|---:|---:|
| pinned at 0 in CC | 875 | 845 | 845 |
| ICSG-observed | 187 | 532 | 2,954 |
| free in CC | 348 | 4 | 4 |

Per year Σ|d| runs 22–87 kt (2013: 79 kt with CC exactly 0 everywhere; 2016: 87 kt). By region over
30 years: Chile 290, Japan 123, Germany 89, China 89, United States 78 kt. The pinned CC legs never
move (max |F_opt| 1.3e-9 kt); the free CC legs move 0.1 kt in total (where CP has no inputs there is
nothing to discharge).

### 3.3 Worst region-years by |d|

| region | year | F_Mass | F_CC | d | CC class |
|---|---|---:|---:|---:|---|
| China | 2013 | −32.3 | 0.0 | −32.3 | pinned |
| China | 2016 | −31.5 | 0.0 | −31.5 | pinned |
| Chile | 2009 | 21.7 | 6.0 | +15.7 | observed |
| Chile | 2010 | −1.4 | −16.1 | +14.7 | observed |
| Chile | 2008 | −12.9 | −27.1 | +14.2 | observed |
| Chile | 2004 | −14.2 | 0.0 | −14.2 | pinned |

### 3.4 The other two signed legs

On X.0 no `Cathode→USC` or `Cathode→MESC` edge-year is negative, prior or posterior (0 of 265,080
CC+Mass), although ICSG reports 102 negative MESC and 48 negative USC rows. Two causes stack: the
ICSG feed keeps only 127 of its 385 MESC/USC source rows, all positive (the 258 dropped include every
negative one; PSC takes no trade-share fan and keeps its 88 negatives) — filed as #40; and the hard
concentration band from `conccathodesupply` sits on these legs (`row_meta`: type `ratio`, hard,
nonneg, kind conc, e.g. `ratio_conccathodesupply_Cathode.Russia-MESC.Japan_…_upper/lower_2015`) and
is satisfiable for a negative pair only at zero. The P10 baseline (§7): MESC drift per year mean
+13.0 kt, positive in 28 of 30 years; USC +16.3 kt, positive in 30 of 30.

## 4 — Pipeline trace of one case: Chile 2006

| quantity | Mass edge 452242 | CC edge 452243 |
|---|---:|---:|
| feed value / `f_agg` | none / 0.0 | ICSG 94.9 kt CC, σ 6.49 |
| imputation log | `flat_prior_noisy_zero` | (observed, no entry) |
| `a0` / `σ_a` | 0.0 / 1e6 | 94.9 / 6.51 |
| rows of `G` touching the column | `massbal_massbal_CP_Chile_Mass_2006` (hard) only | the hard CP CC balance + `ratio_netstockboundpscchile_Chile_CC_2006` (soft, σ_c 83.5) |
| `F_opt` | 129.9 | 116.9 |

The Mass leg has no tie to its CC twin and no prior pull: it is whatever the CP Mass balance needs.
The CC leg is held by its feed σ and the soft stock band.

## 5 — Hypotheses, ranked and tested

| # | hypothesis | predicted signature | verdict |
|---|---|---|---|
| H1 (user's) | no concentration relation covers `*→PSC/SB`, so no imputer can derive Mass from CC | no twin/parametric log rows on PSC Mass; no PSC row in conc v8 | **confirmed**: 15 rows, none with PSC; 0 twin fills (`_emit_twins` finds no twin) |
| H2 | ICSG maps PSC to CC only (`valuation_map kt→CC`) | coverage Mass = 0 | **confirmed** (0/1,410); not a defect |
| H3 | the share transfer cannot help: Mass has no observed share, CC is pre-empted by the recipe | 0 `share_transfer` rows | **confirmed** (0 rows on X.0) |
| H4 | Stage 3 leaves the Mass leg free: no conc tie, PSC excluded from pin and anchor | the G column touches only the CP Mass balance; σ_a 1e6 | **confirmed** (§4) |
| H5 (new) | CP has no eff row → eff 1.0 → the recipe writes PSC = 0, σ 0 → floored to 1e-6 | all `network_forward` PSC fills f = σ = 0; σ_a 1e-6; F_opt ≈ 0 | **confirmed** (872/872; 875 pinned; moved ≤ 1.3e-9 kt) |
| H6 | a Mass PSC was imputed and Stage 3 zeroed it | f_agg Mass ≠ 0 somewhere | **rejected** (all 1,410 are 0 before Stage 3) |

Why one conc row alone would not do: the network would emit a Mass twin of every CC value it knows,
including its 875 zeros with σ 0, so Mass PSC would be hard-pinned wherever CC is; and an ordinary
`min_max` row would add the hard band `CC − r_u·Mass ≤ 0`, `−CC + r_l·Mass ≤ 0`, which a negative
pair satisfies only at zero — 88 reported negative PSC years would be forced to 0 exactly as USC/MESC
are today.

## 6 — What was built (spec #41, ADR 0041)

- **Y.0**: `RatioBoundBuilder` emits no one-signed hard band (lower ratio bound ≥ 0, or an upper
  bound alone) on a pair in which either edge has a negative structural lower bound; zero-spanning
  bands (the stock bands) and agg rows are untouched; the count is logged (≈ 268k of 844k hard conc
  rows expected to go). Plus `concpscuse`: `*→PSC/SB` Mass ↔ CC at 0.9975, band 0.995–1.0 (conc
  v8 → v9 on the first Stage-2 run).
- **Y.1**: `PSC` joins the stock sectors (no recipe, twin or interpolator write); the net-stock seed
  prices it at 0 ± max(0.01·|CP→Cathode|, 0.3 kt) in both valuations; the Mass twin of a reported CC
  value comes from the conc ParametricImputer; the share transfer is retired; `stock_bounds` is a
  Stage-3 input only.
- Tests at three seams (a Stage-2 engine run, the bound builder on a signed toy edge, the driver's
  stock-sector set); ADR 0041 with pointers from ADR 0040 and 0039; glossary terms **stock-change
  leg**, **one-signed band**, the net-stock seed and the recipe exceptions.

## 7 — Measurement (pre-registered in #41; a miss is a fail, never re-tuned)

`scripts/explo/roundY_psc_diag.py <opt_base> [<opt_new> …]` prints every item; the per-year
tables of `scripts/solver_stats.py <opt> --years all` must exist for each run.

| item | pass rule (Y.1) | X.0 | Y.0 | Y.1 |
|---|---|---|---|---|
| P1 PSC prior classes CC: observed / twin / seed / pinned / free | pinned 0; free ≤ 350 | 187 / 0 / 0 / 875 / 348 | 187 / 0 / 0 / 875 / 348 (twin-only state: CC unchanged by design) | 187 / 0 / 875 / 0 / 348 ✓ |
| P1 Mass | pinned 0; free ≤ 350 | 0 / 0 / 0 / 0 / 1,410 | 0 / 1,059 / 0 / 875 / 348 | 0 / 187 / 875 / 0 / 348 ✓ |
| P2 Σ\|F_Mass − F_CC/0.9975\| on PSC per year | ≤ 10 kt every year | 22.2–86.5 kt; 30 years above | 0.0–5.6 kt; 0 years above | 1.7–10.6 kt; 1 year above (2006: 10.6) ✗ |
| P3 mass moved on PSC per year, US excluded (CC / Mass) | ≤ 100 kt/yr | max 50.7 / 286.9; mean 11.7 / 103.2 | max 40.2 / 40.1; mean 7.0 / 6.9 | max 610.1 / 607.3; mean **271.7 / 270.6** ✗ |
| P3 United States alone, Σ\|a* − a_prior\| 30 y (CC / Mass) | reported | 6,052 / 708 kt | 5,885 / 5,887 kt | 5,968 / 5,970 kt |
| P4 cumulative drift 1995–2024, US excluded (CC / Mass) | within ±1,000 kt; Mass ≈ CC | +113.8 / +820.7 kt | +70.8 / +67.4 kt | −590.1 / −648.4 kt; max \|drift\| 802 / 780 ✓ |
| P4 United States alone Σ(F − a0) (CC / Mass) | reported | +588 / −418 kt | +489 / +494 kt | +417 / +422 kt |
| P5 CP→Cathode CC retention per year, world | moves ≤ 0.01 vs X.0 | 0.968–1.006 | 0.967–1.005; max Δ 0.003 vs X.0 | 0.964–1.007; max Δ **0.010** (1995) ✗ at the limit |
| P5 Chile / China / United States | moves ≤ 0.01 vs X.0 | 0.886–1.029 / 0.962–1.037 / 0.936–1.060 | max Δ 0.003 / 0.002 / **0.021** (US 2008 1.035 → 1.056; #35 region) | max Δ **0.028 / 0.019** / 0.020 ✗ |
| P6 common-set sum-row z² vs X.0 | ≤ +1 % (sign reported) | base | **−0.1 %** (1,492,089 → 1,490,497 on 27,116 rows) | **−10.3 %** (1,338,564) ✓ — the sink's signature, see the reading |
| P7 Round-E: world retention CO Mass / Conc CC / Cathode CC | holds | 0.862 / 0.990 / 0.988 | 0.863 / 0.990 / 0.988 | 0.861 / 0.991 / 0.989 ✓ |
| P7 small-node (\|c\| < 18 kt) posterior/prior median / p90 | holds | 1.000 / 1.157 (n 4,772) | 1.000 / 1.149 | 1.000 / 1.145 ✓ |
| P7 exporter concentrate retention Chile / Peru / Indonesia / world | holds | 0.969 / 1.014 / 0.915 / 0.924 | 0.970 / 1.013 / 0.915 / 0.924 | 0.967 / 1.012 / 0.916 / 0.923 ✓ |
| P7 mass moved, all CC edges | reported | 502,941 kt | 501,359 kt | 497,430 kt |
| P8 PSC stock-bound rows: max \|z\|; rows \|z\| > 3; Σz² | not worse | 4.48; 5; 190.8 | **5.18; 9; 215.6** (worse) | **22.64; 77; 4,726.7** ✗ |
| P9 solver status | 30 Solved | 30/30 Solved | 30/30 Solved (29.6 min) | 30/30 Solved (30.1 min) ✓ |
| P10 USC: negative posterior edge-years; drift mean [min, max]; years > 0 | both signs; \|mean\| < 16 | 0; +16.3 [+8.7, +23.3]; 30/30 | **23,847**; −5.6 [−36.4, +7.0]; 11/30 | 23,343; −6.9 [−40.3, +3.4]; 6/30 ✓ |
| P10 MESC: same | both signs; \|mean\| < 16 | 0; +13.0 [−15.3, +23.4]; 28/30 | **23,822**; −8.2 [−40.3, +6.8]; 9/30 | 23,318; −9.4 [−42.6, +3.5]; 6/30 ✓ |

The USC drift of X.0 is one-signed in every year and already sits at the P10 limit; with the feed
still dropping the negative rows (#40), P10 is read as "both signs appear and the mean falls", not
as a target the hard-band change alone can reach.

**Y.0 (#43), read:** the one-row hypothesis closes the Mass sink on its own — with the cathode
concentration tying PSC's two valuations the Mass–CC gap falls from 22–87 kt to 0–6 kt in every
year, the Mass legs now move exactly as the CC legs do (P3 mean 6.9 vs 7.0 kt/yr, P4 +67 vs +71 kt)
instead of absorbing 103 kt/yr, and the common-set sum-row z² is −0.1 % vs X.0 with every Round-E
item within 0.008 of X.0. The 1,059 Mass twins are the network's: 187 carry the ICSG value (88 of them
negative, the sign survives the soft tie) and 872 are its eff-less zeros, so the Mass leg is now pinned
at σ 1e-6 wherever the CC leg is — the twin-only state the ticket was pre-registered to show, and the
pin Y.1 replaces by the seed. The band rule alone changes the other two signed legs: 133,950 one-signed
pairs skipped (hard conc rows 844,282 → 579,202, none left on USC/MESC/PSC), 23,800 USC and MESC
edge-years go negative where none could, and the per-year drift turns from +16.3 (USC, 30 of 30 years
up) and +13.0 kt (MESC) to −5.6 and −8.2 kt with 11 and 9 years up — two-signed, as ADR 0041 predicted,
although the feed still delivers no negative prior (#40). Two items moved the wrong way: the PSC
stock-bound rows are more stressed (max |z| 4.48 → 5.18, rows above 3 from 5 to 9) because the contradiction the Mass sink used to absorb now lands on the CC band in the reported years; and the United
States' CP→Cathode retention moves 0.021 in 2008 (the #35 region, 5,885 kt of PSC movement on its own).
The stop rule is judged on Y.1.

**Y.1 (#44), read:** every mechanical criterion holds — 875 `seed_psc` legs per valuation, 187
parametric Mass twins (88 negative), no recipe or interpolator write, 0 pinned legs, 348 free per
valuation where CP has no product prior, `stock_bounds` gone from the Stage-2 parents, 30 Solved —
and the stop rule fails on P3, P8 and P5 (P2 by 0.6 kt in one year). The seed is a sink. Of the 1,032
PSC CC legs that moved, the 162 reported ones move 209 kt in 30 years (0.69σ on average, the feed holds
them) while the 870 seeded ones move 7,943 kt (2.65σ on average, 6.2σ at p90): China −1,546 kt net at
σ 43 kt, Zambia +857 kt at σ 3.8 kt (its 150 kt/yr production-vs-uses contradiction of ADR 0040 now
discharges into a producer stock at z ≈ 7 instead of surfacing on exports at z 13), Kazakhstan 762,
Russia 547, Austria +356 kt at σ 0.9 kt (the secondary double count), Portugal 12.7 kt in 2020–22
against a band σ_c of 0.3–0.6 kt (z 21–23). The per-year PSC movement is 100–610 kt (mean 272) where
X.0 had 0–51; the cumulative drift reaches −590 kt CC. With p = 1.3 the prior penalty grows sub-
quadratically beyond 1σ, and 1 % of a region's whole cathode production is a wide gate next to the one
node where the data contradictions live; the common-set sum-row z² falls 10.3 % for that reason, not
because the fit improved — Round-E holds and CP→Cathode retention moves 0.028 in Chile and 0.019 in
China (limit 0.010). USC/MESC stay two-signed as on Y.0 (drift −6.9 / −9.4 kt/yr). SD-R-10 of Y.1:
the Mass PSC prior is no longer flat at 0 and sits on the CC prior; the two posteriors coincide
(P2 ≤ 10.6 kt).

**Verdict:** Y.0 (the band rule and the concentration row) is accepted; the Y.1 seed as priced is
rejected on measurement. What ships from Y.1 is a decision recorded in TODO: revert the seed commits
(PSC back to the recipe's hard zero, the H5 defect open and documented) or measure the no-trade floor
alone (σ = 0.3 kt, the price ADR 0040 gives a leg with no co-located flow) as Y.2.

## 8 — Follow-ups outside the round

- #40: the ICSG feed drops the 258 MESC/USC rows incl. all 150 negatives before concordance.
- #35: the garbled US producer-stock row (±2 Mt, 2006–2012) — its 30-year |a* − a_prior| of
  6,052 kt CC dwarfs the rest of the world (P3 US row) and is reported separately throughout.
- #36 items 1 and 3; #37–#39 from the Round X review; the manuscript SI row that still describes
  the PSC share transfer (re-baselined with the Results).

## 9 — Artefacts

- X.0: `data/proc/opt/opt_20261005_134720_ee69cda9/` (timesteps, `diag/constraints_*.csv`,
  `diag/edges_*.csv`, `diag/sd_r_10_boundary_flows.png`); const
  `data/proc/const/const_20261005_130418_a5fb94a6/` (`sigma_a.npy`, `row_meta.parquet`, `G.npz`);
  imp `data/proc/imp/imp_20261005_125928_e9ffa8cb/` (`agg_result.npz`, `imputation_log.parquet`)
- feed `data/proc/feed/registry/feed_icsg_20260922_115241_eb9cb9ff/` (`F_src.npy`, `feed.pkl`),
  source table `data/input/feed/icsg/current/values.csv`
- params `data/input/params/conc/current/params.csv` (v8 on X.0; v9 carries `concpscuse`),
  `data/input/params/stock_bounds/current/params.csv` (v5)
- Y.0: `data/proc/opt/opt_20261006_105904_61c5efe9/`, const
  `data/proc/const/const_20261006_101742_205d50d3/`, imp `data/proc/imp/imp_20261006_101314_c6b6c588/`
  (code at commit `2e9240e`; `scratchpad` chain log: Stage 2 2.8 min, Stage 3 8.6 min, Stage 4 38 min)
- Y.1: `data/proc/opt/opt_20261006_120237_643c4b4a/` (`diag/sd_r_10_boundary_flows.png`), const
  `data/proc/const/const_20261006_112045_f55da888/`, imp `data/proc/imp/imp_20261006_111616_6566f4d2/`
  (code at `48787a1`; Stage 2 2.8 min, Stage 3 8.6 min, Stage 4 38.5 min)
- diag `scripts/explo/roundY_psc_diag.py`; plan `~/.claude/plans/pls-diagnose-in-the-sleepy-nest.md`
