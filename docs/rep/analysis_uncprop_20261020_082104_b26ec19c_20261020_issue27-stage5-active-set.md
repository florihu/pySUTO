# #27: Stage 5 on the final chain, propagated through the QP's active set (ADR 0044)

**Run:** `uncprop_20261020_082104_b26ec19c` (label `r27_final_chain`; 30/30 years; `bind_tol` 1e-6 kt, `rank_tol`
1e-10, `alpha` 0.05; 49–126 s per year, 1,980 s summed, 33.7 min wall clock, years solved one after another)
**Against:** the first copper posterior `uncprop_20261014_140949_fcf53674` (LU of `S`, chain
`opt_20261013_122432_24e811b7`; its numbers as ADR 0044 records them)
**Chain:** the final chain `opt_20261019_171746_3272682b` ← `const_20261019_170151_13460e94` ←
`imp_20261019_165706_dd1ae143`
**Feeds/params:** the final chain's (icsg v15, baci v27, trade_shares v36, crude_ore v3, acc copper v4, conc v10, eff v4,
ore_grade v8, stock_bounds v6, scrap_share v3, stripping_ratio v2)
**Date:** 2026-10-20 · **Branch:** `diag/uncprop-measure` (main 682c372 + `fix/uncprop-active-set`) · **Issue:** #27 ·
**ADR:** 0044 (proposed → accepted), 0039 (lower bounds)

**Verdict: ADR 0044's acceptance condition holds, so it is accepted.**
- **Every year completes,** and σ_a* ≤ σ_a on every one of the 854,460 edge-years: 0 violations, where the LU run had
  87.
- **The pinned set's dependence on `bind_tol` is reported (§2).**
- **The QR the code now uses keeps every κ in [0, 1].** An LU of `S` on the same chain does not: κ > 1 on 365 edges in
  1995 and 273 in 2016, κ < 0 on 6 in 2016 (down to −2.0e12). It also misplaces |Δκ| > 1e-3 on edges carrying
  6,111 kt in 2016.

The posterior is the local sensitivity of the estimator under the model's own σ. Whether its intervals are
calibrated is #87's question and is not tested here.

## 1. Change

Stage 5 linearises the QP where it stands (ADR 0044). Before #27, every hard row entered `S` as a zero-variance
equality, and on copper `S` was singular.
- **Bounds and bands:** a binding lower bound pins its edge (σ_a* = 0); a binding band is an equality; slack rows drop out.
- **Dependent hard rows** are found on the rows themselves (a pivoted Cholesky of the unit-diagonal Gram) and dropped.
- **The contraction** is read off a QR of the whitened system, never off an LU of `S`.
- **Intervals** clip at the bound row's own `l`.

## 2. The active set per year

| quantity | range over 1995–2024 | years at the ends |
|---|---|---|
| bound-active edges (lower bound binds, σ_a* = 0) | 2,958–9,794 | 2019, 2015 |
| binding bands | 9,395–14,089 | 2019, 2014 |
| dependent hard rows dropped | 80–663 | 2021, 2009 |
| rows kept in `S` | 15,753–22,414 | 2015, 2009 |
| max \|hard residual\| at `a*` | 5.8e-11–1.7e-7 kt | 2023, 2002 |
| edges with σ_a* > σ_a | **0 in every year** | |

- **Bound-active edge-years:** 245,943 of 854,460 (28.8 %).
- **Fewer pins mean more dependent rows.** The two counts move against each other (corr −0.866): in the years with
  fewest pins (2009, 2019, 2003, 2006, 1998), more of the kept band rows are linearly dependent, 504–634 ratio rows.
- **Trace contraction is 1.000 in every year.** It is uninformative: the flat priors' σ of 1e6 dominate the trace.

**`bind_tol` sensitivity** (edges pinned, and the kt they carry, Σ max(|a_prior|, |a*|), all years):

| `bind_tol` (kt) | 1e-8 | 1e-7 | **1e-6** | 1e-5 | 1e-4 |
|---|---|---|---|---|---|
| pinned edge-years | 160,087 | 208,710 | **245,943** | 274,828 | 306,201 |
| kt they carry | 7,584.6 | 9,446.9 | **10,416.7** | 11,420.6 | 11,662.9 |

- **The count is sensitive to the tolerance; the mass is not.**
  - Two decades tighter pins 35 % fewer edges; two decades looser pins 25 % more.
  - The kt they carry moves −27 % and +12 %.
  - At 1e-6 the pinned edges carry 21.8–1,082.4 kt a year, against 7,380,121 kt CC and 393,683,879 kt of gross mass
    in the weights of all edge-years.
- **The sensitive years** are the ten where tightening to 1e-8 pins less than half of the 1e-6 set: 1995, 1998, 1999,
  2003, 2006, 2009, 2010, 2018, 2019, 2023. In them, many lower-bound rows sit within 1e-6 of binding without
  touching it.

## 3. Posterior contraction κ by prior class (all years)

| valuation | prior class | edge-years | weight (kt) | median κ | mass-weighted κ | weight share with κ ≥ 0.5 |
|---|---|---|---|---|---|---|
| CC | from a feed | 31,434 | 2.131e6 | 0.045 | 0.928 | 0.953 |
| CC | imputed | 49,790 | 5.239e6 | 0.082 | 0.979 | 0.991 |
| CC | zero-pinned | 342,391 | 6,201 | 0.504 | 0.666 | 0.589 |
| CC | flat | 3,615 | 3,488 | 1.000 | 0.983 | 1.000 |
| Mass | from a feed | 28,793 | 5.917e7 | 0.034 | 0.977 | 0.984 |
| Mass | imputed | 49,713 | 2.061e8 | 0.076 | 0.982 | 1.000 |
| Mass | zero-pinned | 342,391 | 10,640 | 0.671 | 0.747 | 0.891 |
| Mass | flat | 6,333 | 1.284e8 | 0.998 | 1.000 | 1.000 |

- **The typical edge keeps most of its prior σ:** median κ 0.03–0.08 on observed and imputed edges.
- **The large flows contract strongly:** mass-weighted κ 0.93–0.98. Their posterior σ is set by the constraints, not
  by their own prior.
- **The flat class is uninformative.** A prior σ of 1e6 contracts fully by construction, so its κ and its quadrant
  shares (§5) carry no information about the data.

## 4. Largest posterior σ

The eight largest σ_a* are Chile's ore, MF-tailings, SX-EW ore and SX-EW tailings Mass in 2019 and 2009: σ_a* 25,830–28,180
kt on priors of 227,300–552,400 kt, κ 0.83–0.97. They are the Mass edges of the extraction chain that only the hard
recipes and the stripping relation tie down.

## 5. Quadrants (UncPropDiag)

Share of Σ max(|a_prior|, |a*|), all years:
- **CC (7,380,121 kt):** 94.9 % *confirmed*, 3.1 % *overruled*, 1.8 % *prior-carried*, 0.2 % *unexplained move*.
- **Mass (393,683,879 kt):** 95.4 % confirmed, 4.4 % overruled, 0.2 % prior-carried, 0.0 % unexplained.
- **Feed-observed share overruled:** 1.4 % in CC and 4.3 % in Mass.
- **The flat class** (32.6 % of Mass, all of it *confirmed*) is the uninformative case of §3.

## 6. LU of `S` against the QR (the change d54cc2f made)

| | 1995 | 2016 |
|---|---|---|
| live edges | 19,899 | 19,633 |
| κ range, QR (this run) | 2.5e-13–1 | 2.5e-13–1 |
| LU: κ < 0 / κ > 1 | 0 / 365 | 6 / 273 (min −2.0e12) |
| \|Δκ\| > 1e-3: edges, kt, largest edge | 49, 1.07 kt, 0.18 kt | 54, 6,111 kt, 536.9 kt |
| \|Δκ\| > 1e-6: edges, kt | 3,105, 123.4 kt | 1,564, 11,990 kt |

- **On the old chain** the LU run put κ outside [0, 1] on 331 edges of 1995 and 368 of 2016, and gave σ_a* > σ_a on
  87 edge-years (ADR 0044's status).
- **On the final chain** the LU is still wrong on the same scale, and in 2016 on edges that carry real copper.
- **The QR is the condition of the acceptance, not a refinement.**

## 7. The manuscript

The Method's Stage-5 paragraph and its limitations, the Results' A5 paragraph, SI S6.2 and the three headers said
Stage 5 had not run (#27). They now quote this run, with κ read as contraction *under the model* and calibration left
to #87.

## Reproduce

```
python scripts/uncertainty_propagation.py opt_20261019_171746_3272682b --label r27_final_chain
python scripts/explo/issue27_uncprop_active_set.py uncprop_20261020_082104_b26ec19c
```

The script (by the #27 build session) prints every number in this REP except:
- the timing, from the run log;
- the ranges, totals and percentages of §2, which are sums and extremes of its per-year tables;
- the old run's counts, which are ADR 0044's.
