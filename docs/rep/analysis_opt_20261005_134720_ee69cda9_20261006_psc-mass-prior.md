# Round Y — the producer stock change has no Mass prior, a hidden zero pin in CC, and the signed legs carry a one-signed hard band

**Run analysed:** `opt_20261005_134720_ee69cda9` (X.0, label `roundX0_stock_legs_at_prior`; 30 years
Solved; `prior_mode` cold) — the Round X reference run (ADR 0040)
**Runs measured:** Y.0 — *pending* (#43); Y.1 — *pending* (#44)
**Against:** X.0 (the run analysed)
**Chain (X.0):** `imp_20261005_125928_e9ffa8cb` → `const_20261005_130418_a5fb94a6` → the run
**Feeds/params (X.0):** icsg v12 / trade_shares v21, baci v11, crude_ore v3; acc copper v4; conc v8,
eff v4, stripping_ratio v1, ore_grade v7, scrap_share v3, stock_bounds v5
**Date:** 2026-10-06 · **Branch:** `fix/psc-prior-both-valuations` (off Round X) · **Spec:** #41
(measure tickets #42 → #43 → #44; feed follow-up #40) · **ADR:**
[0041](../adr/0041-signed-edges-carry-no-one-signed-hard-band-and-are-seeded.md) (proposed)

**Verdict (diagnosis; the measurement verdict follows in §7):** the Mass PSC prior is flat at 0 on
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
| P1 PSC prior classes CC: observed / twin / seed / pinned / free | pinned 0; free ≤ 350 | 187 / 0 / 0 / 875 / 348 | — | — |
| P1 Mass | pinned 0; free ≤ 350 | 0 / 0 / 0 / 0 / 1,410 | — | — |
| P2 Σ\|F_Mass − F_CC/0.9975\| on PSC per year | ≤ 10 kt every year | 22.2–86.5 kt; 30 years above | — | — |
| P3 mass moved on PSC per year, US excluded (CC / Mass) | ≤ 100 kt/yr | max 50.7 / 286.9; mean 11.7 / 103.2 | — | — |
| P3 United States alone, Σ\|a* − a_prior\| 30 y (CC / Mass) | reported | 6,052 / 708 kt | — | — |
| P4 cumulative drift 1995–2024, US excluded (CC / Mass) | within ±1,000 kt; Mass ≈ CC | +113.8 / +820.7 kt | — | — |
| P4 United States alone Σ(F − a0) (CC / Mass) | reported | +588 / −418 kt | — | — |
| P5 CP→Cathode CC retention per year, world | moves ≤ 0.01 vs X.0 | 0.968–1.006 | — | — |
| P5 Chile / China / United States | moves ≤ 0.01 vs X.0 | 0.886–1.029 / 0.962–1.037 / 0.936–1.060 | — | — |
| P6 common-set sum-row z² vs X.0 | ≤ +1 % (sign reported) | base | — | — |
| P7 Round-E: world retention CO Mass / Conc CC / Cathode CC | holds | 0.862 / 0.990 / 0.988 | — | — |
| P7 small-node (\|c\| < 18 kt) posterior/prior median / p90 | holds | 1.000 / 1.157 (n 4,772) | — | — |
| P7 exporter concentrate retention Chile / Peru / Indonesia / world | holds | 0.969 / 1.014 / 0.915 / 0.924 | — | — |
| P7 mass moved, all CC edges | reported | 502,941 kt | — | — |
| P8 PSC stock-bound rows: max \|z\|; rows \|z\| > 3; Σz² | not worse | 4.48; 5; 190.8 | — | — |
| P9 solver status | 30 Solved | 30/30 Solved | — | — |
| P10 USC: negative posterior edge-years; drift mean [min, max]; years > 0 | both signs; \|mean\| < 16 | 0; +16.3 [+8.7, +23.3]; 30/30 | — | — |
| P10 MESC: same | both signs; \|mean\| < 16 | 0; +13.0 [−15.3, +23.4]; 28/30 | — | — |

The USC drift of X.0 is one-signed in every year and already sits at the P10 limit; with the feed
still dropping the negative rows (#40), P10 is read as "both signs appear and the mean falls", not
as a target the hard-band change alone can reach.

**Y.0 (#43):** pending. **Y.1 (#44):** pending — the verdict paragraph and the ADR 0041 status line
are written then.

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
- diag `scripts/explo/roundY_psc_diag.py`; plan `~/.claude/plans/pls-diagnose-in-the-sleepy-nest.md`
