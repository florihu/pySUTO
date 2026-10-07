# Round X — the stock-change legs are a free sink; they are held at their prior

**Run analysed:** `opt_20260930_092111_e3e0677d` (W.1 reference; label `roundW1_lower_bounds_tol`;
28 years Solved, 2015 and 2016 AlmostSolved; `prior_mode` cold)
**Run measured:** X.0 `opt_20261005_134720_ee69cda9` (label `roundX0_stock_legs_at_prior`; 30 years
Solved; `prior_mode` cold)
**Against:** W.1 (the run analysed)
**Chain (W.1):** `imp_20260929_113529_ab914090` → `const_20260929_122634_40ebe5c9` → the run
**Chain (X.0):** `imp_20261005_125928_e9ffa8cb` → `const_20261005_130418_a5fb94a6` → the run
**Feeds/params:** icsg v12 / trade_shares v21, baci v11, crude_ore v3; acc copper v4; eff v4,
stripping_ratio v1, conc v8, ore_grade v7, scrap_share v3; stock_bounds v4 (W.1) → **v5** (X.0)
**Date:** 2026-10-05 · **Branch:** `fix/mesc-usc-stock-sink` · **Issue:** #34 (follow-ups #35, #36) ·
**ADR:** [0040](../adr/0040-stock-change-legs-are-held-at-their-prior.md)

**Verdict: the sink is closed at no cost to the fit. With the stock legs held at their prior
(reported or zero, σ = 1 % of the co-located cathode flow or 0.3 kt; no imputer writes one; the
MESC/USC zero rows retired) the world MESC posterior falls from 15,657 to 4,259 kt CC over 30
years against a 3,869 kt prior (W.1: 6,009), the cumulative drift from +9.6 Mt to +0.39 Mt, the
mass moved on MESC legs per year from up to 766 kt to at most 60 kt, and Zambia's 2010 exchange
"exports" from 121 kt to 11 kt — while the common-set sum-row z² is −0.4 % vs W.1 (a figure that
also contains a −19.9 Mt domestic-FU prior change, §4.2 "Side effect"), every year
solves to Solved, and the Round-E items hold (world retention within 0.006, exporter
concentrate retention within 0.003). Zambia's 150 kt contradiction now shows where the data
disagree: production −62 kt (z −6.9) and exports +57 kt (z 13.2 on the use total). What remains
is a constant +16 kt/yr on each of MESC and USC, the sum of 60,000 zero-prior legs each moving
less than a σ (p99 |z| 0.89) and all in the same direction — a residual sink of the signed
structure itself, 1,000 kt over 30 years instead of 9,650, and the reason USC misses the
pre-registered retention band in 13 of 29 years. The first chain (§4.1) showed that closing the
sink relocates it unless the imputer is told whose edge production is.**

---

## 1. What was reported, and what it was

The user read SD-R-10 of the W.1 run and saw a system-boundary-output retention line swinging
between 1.2 and 5.4 and labelled it "Scrap". The line is **MESC**: `_fig_boundary_flows` built one
palette per direction (`_series_colors`, slot = sorted-name index) and merged both into one
legend, so Scrap (SB-input) and MESC (SB-output) shared the same colour. Fixed in
`90d091e` (one palette over the union of both directions; `tests/test_solver_diag.py::TestBoundaryPalette`).

| Category (SD-R-10, W.1) | Retention 1995–2024 | 30-y Σ prior → Σ posterior (kt CC) |
|---|---|---|
| Scrap→Smelting CC (SB-input) | 0.86 – 1.10 | 76,886 → 78,027 (1.015) |
| Scrap→CFR CC (SB-input) | 0.91 – 1.09 | 22,385 → 22,928 (1.024) |
| Scrap, Mass (both) | 0.79 – 1.15 | — |
| **MESC (SB-output)** | **1.19 – 10.3; undefined in 1996 (prior −99 kt)** | **6,009 → 15,657** |
| USC (SB-output) | 0.51 – 3.6, undefined in 5 years | 1,650 → 2,865 |
| PSC (SB-output) | undefined (sign flips) | −391 → −941 |

World MESC posterior is +325…+898 kt **every year** (prior 43…416 kt). Cumulative by destination
at 2024: United Kingdom 2,807 kt, Singapore 2,158, Netherlands 2,123, United States 1,513, China
882, Japan 785. By origin (posterior − prior over 30 years): Zambia +2,011, Russia +1,343, ROW
+1,000, Chile +839, Austria +675, Peru +493, India +474, Belgium +412, Kazakhstan +376. The
exchange stocks ICSG reports (LME + COMEX + SHFE) oscillate between 0.1 and 1 Mt in total.

### 1.1 Mechanism — the Zambia 2010 CC trace

| Item | Prior | Posterior | σ | z |
|---|---|---|---|---|
| CP→Cathode Zambia (ICSG refined production) | 527.1 | 512.7 | 31.7 | −0.46 |
| Σ uses of Cathode Zambia (FU legs + stock legs) | 377.0 | 512.7 | — | — |
| Cathode Zambia → **MESC Netherlands** (edge 827731) | 0.0 | **121.25** | **552.89** | **0.22** |
| `sum_usetotout_Cathode_Zambia_CC_2010` (c 379.7) | 379.7 | 512.7 | 6.47 | **20.5** |
| `sum_supplytotout_CP_Zambia_CC_2010` (c 527.1) | 527.1 | 512.7 | 8.98 | −1.6 |
| `ratio_netstockboundmescnetherlands_Netherlands_CC_2010` (c 0) | 2.4 | 165.0 | 552.89 | 0.30 |

Production is anchored at ≈ 1 % and exports come from BACI (338 kt), so the 150 kt gap must
leave the node. The cheapest outlet is a zero-prior MESC leg into an exchange location whose
σ is 553 kt: z 0.22 for 121 kt. The contradiction itself stays visible on the use scale total
(z 20.5); the sink only decides *where* the mass goes.

Three layers produce the sink (file:line on `main` at `6483346`):

- **Stage 2, the seed.** `ShareImputer` seed mode (`src/pysuto/impute/share.py:132-136`,
  `:198-221`; registered at `scripts/collect_aggregate_impute.py:642-648` with
  `scale_abs_floor = 100.0`): f = 0, σ = (p95/1.645)·max(|FU(i→j)|, 100). p95 comes from
  stock_bounds v4: the "hub" cluster {Netherlands, Singapore, United Kingdom} has p95 = 9.095 —
  the 95th percentile of a warehouse country's exchange-stock change over its own small semi
  use — so σ = 5.529 × 100 kt = **552.89 kt on 484 legs** (16 per year); the domestic cluster
  (p95 0.1013) gives 6.16 kt on 3,741 legs (USC: 1.44 kt). On W.1 the legs with σ > 100 kt carry
  6,865 kt of the 13,540 kt moved on MESC legs, the 2–10 kt class 3,045 kt (Kazakhstan→Kazakhstan
  1996: +65 kt at z 10.6).
- **Stage 2, the prior writers.** Before the seed, `NetworkImputer._do_flow_balance`
  (`src/pysuto/impute/network.py:454-474`) closes a Cathode Flow node whose only unknown
  same-region edge is the stock leg by writing the residual in − out onto it — 51 fills, Σ|f|
  2,780 kt, σ 60–147 kt (Germany USC 1999 −231 kt; Japan MESC 2014 +94 kt) — and `TimeImputer`
  interpolates stock changes linearly between reported years — 96 fills, Σ|f| 3,034 kt (US MESC
  2000 +118 kt at σ 7.8 kt). Together 5,814 kt of stock change in the prior that no source
  reported (5,815 kt in W, which since the code review also counts the conc twins of the two
  formula-written USC legs below, 0.8 kt). A third writer was dormant: the domestic residual formulas `cathode_usc_domestic` /
  `cathode_mesc_domestic` (`collect_aggregate_impute.py:299-331`, Mass, `clip_lower=0`) fired 2
  and 0 times on W.1 only because the balance had written the Mass twin first; with that writer
  barred they booked **7,290 kt** of one-signed stock change in the first X.0 imputation
  (`imp_20261005_115201_a8e40726`, Japan MESC 2016 +543 kt) and were retired (`4626edd`). W as
  first written missed them: it read only CC legs and excluded `parametric`, and these writers
  book Mass, so their CC twin is logged as `parametric`. The corrected W counts a twin whose
  Mass source was imputer-written, and reads **7,272 kt CC** on that run (53 legs).
- **Stage 3, the rows.** One zero-centred soft row per (destination, year), Σ MESC into r = 0 ±
  σ_c with σ_c = max(σ_r, 0.02)·max(Σ|FU into r|, `RATIO_SCALE_ABS_FLOOR` = 100)
  (`scripts/constraints.py:236-254`, `:547-553`; `src/pysuto/constraints/ratio.py:380-398`):
  552.9 kt for the three exchange locations, 461 kt for China, 65 kt for Japan. The floor was
  reasoned as "100 t" in the driver comment and in REP 0611 §8.2 ("raised 1→100 t") but applied
  in kt. On a region with a reported stock change the zero-centred row pulls the data point
  toward zero (United States: reported ≈ ±300 kt against σ_c ≈ 120 kt → z ≈ 2.4).

### 1.2 Initial data

ICSG reports exchange stocks by location — LME by warehouse country, COMEX as the United States,
SHFE as China: `values.csv` carries 225 MESC rows over 16 regions, world Δ between −357 and
+300 kt/yr — but only for **1996–2004, 2014 and 2016–2023**. The 2014-edition exchange block
parses to zero rows (`scripts/parse/icsg_parse/icsg_2014.py:331-337` relabels the columns from
`df.iloc[1]` although `main` reads the sheet with `header=2`, `:258`;
`data/proc/feedprep/icsg_clean/icsg_2014_stocks.csv` is PMC-only), `icsg_comb.py:262-266`
dedups stock *levels* across editions before `process_icsg.py:30-36` diffs *within* edition (one
net-addition year lost per boundary: 2004, 2013, 2015), and 1995 is the baseline. The United
States producer-stock rows of ±2 Mt/yr (2006–2012) trace to a garbled "U.S. (producers)" row
(1010 / 828 / 2828 / 2020 kt). Every BACI cathode leg is split 98/1/1 onto FU/USC/MESC
(`data/input/feed/baci/current/weights.csv`), a one-signed feed-backed MESC prior of +6.0 Mt over
30 years. These are issues #35 and #36; the round changes none of them — the user's principle is
that no better data exists than what the study already uses, so the fix is the σ design.

### 1.3 Scrap proper (side findings, no action)

The posterior scrap grade (CC/Mass) drifts from the 0.70 prior to 0.72–0.91 at CFR and 0.58–0.85
at Smelting; China 2024 sits on the 0.98 hard cap on both edges, Belgium at 0.50 and Slovakia at
0.47 near the 0.40 floor (the Mass twin's cv 0.244 is 3.5× the CC cv 0.069). Zero-prior scrap the
solver fabricates: 8–138 kt/yr, z up to 23 (South Africa, Philippines, Uzbekistan, Congo DR).
Large reported scrap deleted: Austria Smelting 2020 retention 0.148 (z −12.4), Russia 2003 0.013
(z −14.4).

## 2. Change (commits `8e8fc01`, `fdeff19`, `4626edd`)

A stock leg's prior is a reported value or a zero-centred seed, nothing else, and it is held
there (grilled with the user; ADR 0040):

| Layer | Before (W.1) | After (X.0) |
|---|---|---|
| Seed σ on `Cathode(i)→MESC/USC(j)` | (p95/1.645)·max(\|FU(i→j)\|, 100 kt): 553 kt (exchange locations), 6.16 / 1.44 kt (others) | max(0.01·\|FU(i→j)\|, 0.3 kt): no-trade pair 0.3 kt, China→China MESC 44 kt, Zambia→Netherlands 0.3 kt (`STOCK_SEED_CV`, `STOCK_SEED_ABS_FLOOR_KT`) |
| Imputers writing MESC/USC | network balance residual, time interpolation, (dormant) domestic residual formulas, seed, feed twins | seed and feed twins only (`STOCK_SECTORS`; `NetworkImputer(stock_sectors=…)`, time scope mask; the two residual formulas removed) |
| Stage-3 net-stock rows | 141 zero-centred rows (PSC, USC, MESC × 47), σ_c floor 100 kt | 47 PSC rows (stock_bounds v5, root labels, PSC p95 0.0504 → 0.0489), σ_c floor 0.1 kt (`STOCK_BOUNDS_SCALE_ABS_FLOOR`) |
| MESC/USC in balances, scale totals, bands | unchanged | unchanged |

Tests: `tests/test_impute_share.py` (seed σ, floor, no p95), `tests/test_network_imputer.py`
(a lone unknown stock leg is never written; a missing one counts as 0), `tests/test_constraints.py`
(stock floor in kt), `tests/test_solver_diag.py::TestBoundaryPalette`.

## 3. Pre-registered criterion

From `diag/edges_<year>.csv`, MESC and USC separately, CC (`scripts/explo/roundX_stock_sink_diag.py`):

| Metric | Definition | Limit | W.1 (MESC) |
|---|---|---|---|
| R | Σa*/Σa_prior per year, quoted where \|Σa_prior\| > 50 kt | within [0.8, 1.2] | 28 of 29 quoted years outside |
| M1 | Σ\|a* − a_prior\| over the legs, world, per year | ≤ 100 kt (USC ≤ 50) | max 766 kt |
| M2 | cumulative Σ_{t≤T}(Σa* − Σa_prior) | \|·\| ≤ 1 Mt | +9,648 kt at 2024 |
| M3 | Σ_j MESC*(i→j,t) / CP→Cathode*(i,t), origins ≥ 100 kt | ≤ 5 % | Zambia 2010 24.9 % |
| W | mass written onto MESC/USC legs by non-seed, non-feed imputers | 0 | 5,815 kt |

plus the Round-E stop rule (ADR 0025: world retention on CO Mass / MF→Concentrate CC /
CP→Cathode CC not down by > 0.01 — W.1: 0.862 / 0.992 / 0.994; small-node posterior/prior
median 1.000, p90 1.114; exporter concentrate retention Chile 0.972, Peru 1.014, Indonesia 0.917,
world 0.924), the common-set sum-row z², and the Zambia trace: the 150 kt must surface on an
identifiable row, never vanish.

## 4. Measurement

### 4.1 X.0a — the first chain: the sink moved into the prior's production edge

Chain `imp_20261005_115706_c6829d4a` → `const_20261005_120156_bc2c9765` →
`opt_20261005_124425_39ce7989` (29 Solved, 2016 AlmostSolved; superseded, kept for this record).
The stock legs behaved: W = 0; MESC R within [0.8, 1.2] in 26 of 29 quoted years (1996 1.31,
2000 1.21, 2008 1.20), M1 max 59 kt, M2 +288 kt at 2024, 30-y Σ 3,869 → 4,158; Zambia 2010 MESC
9.5 kt (share 2.1 %). But the common-set sum-row z² rose **+7.4 %** (27,114 rows: 1,497,777 →
1,608,732), all of it on `supplytotout` (+187,125) and the scrap anchors, with `usetotout`
down (−84,807); by sector Cathode −186,376 and Mining +101,275 / Ore +41,418 / CER +26,676 /
ER +26,676 / CFR +25,007 / CP +18,457; by region Germany +69,457, Zambia +38,554, Korea
+25,243, Austria +19,507. The top rows were `sum_supplytotout_Mining_Germany_*_2015/2016/2018/2021`
at z² 0 → 3,440 each: a **145 kt German mine chain in the prior** (CO→Mining→Ore→SXEW→CSXEW,
σ 85–133 kt) that the solver deleted to 0.06 kt at z −58.

The trace explained it. With missing stock legs counted as 0, the Cathode node's balance
(`_do_flow_balance`) has a lone writable unknown as soon as the FU legs are known — the supply
edge `CP→Cathode` — and fires before the CP recipe can set it from CER + CSXEW (ICSG refined
production): **836 `CP→Cathode` balance fills, Σ 544,515 kt** (W.1: 1 fill, 2,534 kt). Zambia 2010:
production prior 527.1 (`network_forward`) → 359.5 (`network_balance` over the 18 FU legs);
Germany 2015: 678.1 → 783.1, and the CP recipe then ran backwards (CSXEW→CP = 105 kt →
SXEW→CSXEW, 670 new fills → Ore→SXEW → Mining→Ore → CO→Mining = 144.8 kt). The contradiction
had not surfaced; it had moved from a stock leg into the prior's production edge and a
fabricated mine chain — the Round-M lesson (closing a sink relocates the deletion), this time
inside Stage 2. Fix (`03e9f64`): the balance never derives a node's **supply** from uses that
include a stock leg (the seed's 0 is an expectation, not a report); it may still close a lone
missing use. `tests/test_network_imputer.py::test_flow_balance_never_derives_supply_from_uses_that_include_a_stock_leg`.

### 4.2 X.0 vs W.1 — the pre-registered criterion

Chain `imp_20261005_125928_e9ffa8cb` → `const_20261005_130418_a5fb94a6` → `opt_20261005_134720_ee69cda9`
(30 years Solved, 41–75 s each). Sidecars: `stock_sectors [MESC, USC]`, `stock_seed_cv 0.01`,
`stock_seed_abs_floor 0.3`, `stock_bounds` v5, `stock_bounds_scale_abs_floor 0.1`. Rows
1,950,469 (W.1 1,952,021): 1,410 net-stock rows (PSC) instead of 4,230. σ_a on the 66,270 MESC
CC legs: median 0.015 kt (flat-prior legs under the no-data anchor), 0.3 kt and up on the 4,713
seeded legs, 5 legs above 100 kt (China's domestic pairs, max 124 kt).

| Metric | Limit | W.1 | X.0 | |
|---|---|---|---|---|
| W, mass written onto MESC/USC legs by non-seed, non-feed imputers | 0 | 5,815 kt | **0 kt** | pass |
| MESC R, years outside [0.8, 1.2] of 29 quoted | 0 | 28 | **6** (1996 1.36; 2000 1.23; 2005–2008 1.21–1.30) | fail, narrowly |
| MESC M1, max yearly mass moved | ≤ 100 kt | 766 kt | **60 kt** | pass |
| MESC M2, max cumulative drift | ≤ 1,000 kt | +9,648 kt | **+389 kt** | pass |
| MESC 30-y Σ prior → Σ posterior | — | 6,009 → 15,657 | **3,869 → 4,259** | |
| M3, MESC share of cathode output per origin-year (origins ≥ 100 kt) | ≤ 5 % | Zambia 2010 24.9 % | Zambia 2010 **2.3 %**; 10 origin-years above, all with *reported* stock changes (Germany 1999 17 %, Spain 1997 13 %, Korea 2016 12 %, US 2001 10 %) | fail as written; on the **moved** part max 5.8 % (Austria 2012), one origin-year above |
| USC R, years outside [0.8, 1.2] of 29 quoted | 0 | 10 (incl. negative) | **13**, all above (1.11–1.39) | fail |
| USC M1 / M2 | ≤ 50 kt / ≤ 1,000 kt | 402 kt / +1,242 kt | **27 kt / +489 kt** | pass |
| USC 30-y Σ prior → Σ posterior | — | 1,650 → 2,865 | **2,390 → 2,879** | |
| common-set sum-row z² (27,114 rows) | report | 1,497,777 | **1,492,089 (−0.4 %)** | |
| world retention CO Mass / MF→Concentrate CC / CP→Cathode CC (ADR 0025 a) | not down > 0.01 | 0.862 / 0.992 / 0.994 | 0.862 / 0.990 / 0.988 | pass |
| small-node (\|c\| < 18 kt) posterior/prior median / p90 (b) | hold | 1.000 / 1.114 | 1.000 / 1.157 | p90 +0.043 |
| exporter concentrate retention Chile / Peru / Indonesia / world (c) | not regress | 0.972 / 1.014 / 0.917 / 0.924 | 0.969 / 1.014 / 0.915 / 0.924 | pass (≤ 0.003) |
| mass moved, all CC edges (d) | report | 525,219 kt | 502,941 kt | −4.2 % |

**Where the Zambia 150 kt went.** `sum_supplytotout_CP_Zambia_CC_2010`: 527.1 → 464.8 (z −6.9,
W.1 −1.6); `sum_usetotout_Cathode_Zambia_CC_2010`: 379.7 → 464.8 (z 13.2, W.1 20.5); cathode
export legs 338.3 → 394.8; MESC legs at prior (largest →Switzerland 2.7 → 2.9 kt). The
disagreement between ICSG production and BACI exports is now priced on both rows instead of
being parked in a Dutch warehouse.

**The residual drift.** Σ(post − prior) over 30 years is +492 kt on MESC and +493 kt on USC
(the two signed sinks are structurally symmetric and the solver splits between them), all of it
on the 60,170 zero-prior legs — seeds and flat-prior legs under the no-data anchor — each moving
well under a σ (p99 |z| 0.89, max 6.1) and almost all upward; the 6,100 feed-backed legs stay at
their prior (median z 0.00, p90 0.11, Σdelta −102 kt MESC / −4 kt USC). It is the price of a
signed edge family with 60,000 members: 16 kt/yr world-wide per indicator against a prior of
70–130 kt/yr, enough to push USC's yearly retention to 1.1–1.4. Tightening the no-data anchor on
stock legs or removing the one-signed 1 % BACI fan (#36) are the levers left; neither was in scope.

**Side effect recorded (corrected 2026-10-06, code review).** With the stock legs no longer
writable, the domestic `Cathode→FU` leg is the Cathode node's lone same-region gap, and the
Flow-node balance now writes 227 of them (Σ 8,874 kt Mass over 30 years) as production − exports
− reported stock change. On W.1 the unknown stock legs blocked that balance, and 226 of these legs
were written instead by `cathode_fu_domestic` (`imp_20260929_113529_ab914090`: Σ 28,740 kt). Its
export term selects `entity_to == "Flow"`, while every Cathode export leg ends at an FU (SB) node,
so it never subtracted an export: 224 of the 226 fills carry the supply coefficient alone,
production/conc (Congo D.R. 2024: 2,578.4 kt on W.1, 568.0 kt on X.0). On X.0 the formula fires
0×, and the same 226 legs carry 8,845 kt. **The domestic-FU prior therefore falls by 19.9 Mt
between W.1 and X.0.** That change is not part of the stock-leg design, and the common-set
z² −0.4 % vs W.1 bundles the two; this REP does not measure how much of the −0.4 % each one
accounts for. The relation is now retired, together with the conc FormulaImputer that ran it.
Re-running Stage 2 on the X.0 inputs without it (`imp_20261006_081350_5f8edf28`) reproduces
`imp_20261005_125928_e9ffa8cb` bit for bit: f_agg, σ_agg, Ω, coverage, and every value, σ and
source of the 792,165 imputation-log rows. Only the log's step indices change, because one
cascade member is gone.

**Figure.** `data/proc/opt/opt_20261005_134720_ee69cda9/diag/sd_r_10_boundary_flows.png`: the
MESC and USC retention lines sit between 0.9 and 1.4 (MESC) / 1.1 and 1.4 (USC) instead of 1.2–5.4,
and the two directions now draw from one palette.

## 5. Verdict and what follows

The sink is closed and the fit did not pay for it: the mass that used to accumulate in
exchange-location legs now surfaces as z on the production and export rows of the regions whose
statistics disagree, which is where a reconciliation should show it. Against the pre-registered
criterion the round passes W, M1, M2, the Round-E items and the common-set z², passes M3 on the
moved part and fails it as written (the written form counted reported stock changes against
production, a definitional slip), passes MESC retention in 23 of 29 years and fails USC retention
in 13 — both failures the same +16 kt/yr residual on 60,000 zero-prior legs. ADR 0040 is accepted
on this measurement; the residual is recorded, not hidden.

Follow-ups, in order: (1) #36 — the BACI 98/1/1 fan and the no-data anchor on stock legs decide
the residual 16 kt/yr; the PSC exemption from the anchor stays the open signed sink (PSC 2006
−2,157 kt). (2) #35 — the ICSG stock parse (2014 exchange block, edition-boundary year, USC
types, the US producers row) restores 2005–2013 as reported values. (3) The manuscript Results
are re-baselined on the shipped run in a separate step (checklist below).

**Bookkeeping.** Superseded runs of this round, cited above and kept until the user prunes them
(`scripts/explo/prune_runs.py --dry-run` first, then the registries): `imp_20261005_115201_a8e40726`
(the formula-booked 7,290 kt, §1.1; no downstream runs) and the X.0a chain
`imp_20261005_115706_c6829d4a` → `const_20261005_120156_bc2c9765` → `opt_20261005_124425_39ce7989`
(§4.1; ≈ 1.6 GB). The manuscript's Results, SI S15–S17 and figures stay on W.1
(`opt_20260930_092111_e3e0677d`) until re-baselined in a separate step: rerun
`scripts/report/results_figs.py` on the shipped run, refresh the ledger rows (Table 3 / S15),
the retention-by-stage table, the A5 tolerance pair, the row counts quoted in Method §3.1.5
(1,952,021 rows; 433,421 soft ratio rows), the "stock_bounds v4" of the reference-chain
header (`method.md:16`, `si.md:631`), and the SI imputer table's "Formula (cascade)" row
(`si.md:177`): the cathode final-use split it describes is retired (§4.2 "Side effect"), and the
domestic Cathode→FU prior is the Flow-node balance's.

---

Artefacts: `data/proc/opt/opt_20260930_092111_e3e0677d/diag/edges_<year>.csv`,
`…/constraints_2010.csv`; `data/proc/const/const_20260929_122634_40ebe5c9/{sigma_a.npy,row_meta.parquet,blocks/g_2010.npz}`;
`data/proc/imp/imp_20260929_113529_ab914090/imputation_log.parquet`;
`data/input/feed/icsg/current/values.csv`; `data/input/params/stock_bounds/{v4,current}/params.csv`;
`figs`: `data/proc/opt/<OPT_X0>/diag/sd_r_10_boundary_flows.png`.
