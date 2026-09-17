"""
roundN_concentrate_grade_diag.py
================================
Round N measurement: **the concentrate grade is not the lever** (ADR 0029).

    python scripts/explo/roundN_concentrate_grade_diag.py [<opt_run>]

Round N proposed replacing the single global ``conc_Concentrate`` = 0.275 with a
region×year family derived from the concentrate node identity

    g_r·X_mass(r) − Σ_A g_A·mass(A→r) = P(r) − U(r)

This script builds that estimator, measures it, and measures the thing that
refutes it. Five sections:

  1  Chile's implied grade year by year — the pooled 0.325 the plan rested on
     is an artifact of averaging over years whose export MASS has holes.
  2  Is there a region signal at all? Between- vs within-region variance on the
     observations that are physically possible.
  3  Where the mass went: the raw BACI source, gross vs netted, by HS code.
  4  Discharge capacity — ``P > 0.35·X + U`` — the share of the mine-chain
     deletion that is forced before the solver starts.
  5  The coupled ridge estimator itself, with its three gates measured, showing
     that even fully gated it puts a fifth to a half of export mass on a bound.

Read-only. Feed and params versions come from the registries; the solver run is
read from its persisted timesteps (the Round C hygiene rule). Section 4 needs a
run; sections 1-3 and 5 do not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from pysuto.acc import AccountingStructure                      # noqa: E402
from pysuto.config import load_config                           # noqa: E402
from pysuto.input_registry import InputRegistries               # noqa: E402
from pysuto.param_relation import expand_edge_index             # noqa: E402
from build_baci_reexport import (                               # noqa: E402
    HUB_EXCLUDE, _conc_lookup, _source_dir, icsg_intake_production,
    load_root_trade, screened_origins,
)

ROOT       = Path("data/proc")
DEFAULT_RUN = "opt_20260915_131115_b8bf05a9"
# The global row this round proposed to replace (conc v8).
PRIOR, PRIOR_SD, BAND_LO, BAND_HI = 0.275, 0.0375, 0.20, 0.35
WINDOW = 2          # centred 5-year window of sums
MAJORS = ["Chile", "Peru", "Indonesia", "Australia", "Canada", "United States"]


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_inputs() -> tuple:
    """``(bilateral concentrate mass, P, U, screened origins)`` from the feeds."""
    inputs = InputRegistries.initialize(root="data/input")
    src = str(inputs.feed.resolve("baci"))
    print(f"   baci (netted) ← {src}")
    raw, legs, _, _ = load_root_trade(src, _conc_lookup())
    intake, production, _ = icsg_intake_production(["Concentrate"])
    screened = {(a, t) for (_s, a, t)
                in screened_origins(legs[legs.sector == "Concentrate"], production)}

    m = raw[(raw.sector == "Concentrate") & (raw.region_from != raw.region_to)]
    m = m.groupby(["region_from", "region_to", "time"])["Value"].sum().reset_index()
    P = {(r, t): v for (_s, r, t), v in production.items()}
    U = {(r, t): v for (_s, r, t), v in intake.items()}
    return m, P, U, screened


def implied_grades(m, P, U, screened) -> pd.DataFrame:
    """Per region-year: ``(P−U)/X``, export mass, import share, screened flag.

    The ratio is the node identity solved for one region in isolation, so it is
    the true grade only where imports are negligible; ``imp_share`` says where.
    """
    X = m.groupby(["region_from", "time"])["Value"].sum()
    I = m.groupby(["region_to", "time"])["Value"].sum()
    rows = []
    for (r, t) in sorted(set(P) & set(U)):
        x = float(X.get((r, t), 0.0))
        if x <= 0:
            continue
        rows.append(dict(region=r, time=t, pu=P[(r, t)] - U[(r, t)], x=x,
                         imp_share=float(I.get((r, t), 0.0)) / x,
                         g=(P[(r, t)] - U[(r, t)]) / x,
                         screened=(r, t) in screened))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# the estimator Round N proposed
# ---------------------------------------------------------------------------

def solve_window(m, P, U, screened, regions, years,
                 gate_reported=True, drop_screened=True, drop_hub_eq=True):
    """Ridge-regularised, box-bounded least squares over one window of sums.

    ``minimise ‖A g − b‖² + ‖(g − PRIOR)/PRIOR_SD‖²``, bounds ``[0.20, 0.35]``.
    Rows sum identically to zero, so ``A`` is singular by construction and the
    ridge is what makes the system identifiable, not merely anti-drift hygiene.

    The three gates are arguments so their effect can be measured rather than
    assumed:
      ``gate_reported``  a node equation needs BOTH P and U observed that year;
      ``drop_screened``  exclude origins whose exports no production explains;
      ``drop_hub_eq``    ROW is an aggregate whose internal trade ADR 0014 folds
                         away, so its own node equation cannot close — it stays
                         an unknown, it stops being an equation.
    """
    idx = {r: i for i, r in enumerate(regions)}
    n = len(regions)
    A = np.zeros((n, n))
    b = np.zeros(n)
    have = np.zeros(n, bool)

    w = m[m.time.isin(years)]
    if drop_screened:
        w = w[[(r.region_from, r.time) not in screened for r in w.itertuples(index=False)]]
    for r in w.itertuples(index=False):
        A[idx[r.region_from], idx[r.region_from]] += r.Value      # own exports
        A[idx[r.region_to],   idx[r.region_from]] -= r.Value      # supplier's grade

    for r in regions:
        for t in years:
            if gate_reported and not ((r, t) in P and (r, t) in U):
                continue
            b[idx[r]] += P.get((r, t), 0.0) - U.get((r, t), 0.0)
            have[idx[r]] = True

    keep = np.ones(n, bool)
    if gate_reported:
        keep &= have
    if drop_hub_eq:
        keep &= np.array([r not in HUB_EXCLUDE for r in regions])

    M = np.vstack([A[keep], np.eye(n) / PRIOR_SD])
    y = np.concatenate([b[keep], np.full(n, PRIOR) / PRIOR_SD])
    sol = lsq_linear(M, y, bounds=(BAND_LO, BAND_HI))
    g = pd.Series(sol.x, index=regions)
    return g, A[keep] @ sol.x - b[keep], int(keep.sum())


# ---------------------------------------------------------------------------
# the thing that refutes it
# ---------------------------------------------------------------------------

def mine_chain(opt: str) -> pd.DataFrame:
    """Per (region, year) the MF→Concentrate CC prior and posterior."""
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=ROOT / "acc")
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    rows = []
    for f in sorted((ROOT / "opt" / opt / "timesteps").glob("t_*.npz")):
        o = np.load(f, allow_pickle=True)
        rows.append(pd.DataFrame({"edge_id": np.asarray(o["cols_t"], dtype=int),
                                  "prior": o["a_prior"], "post": o["F_opt"]}))
    e = pd.concat(rows, ignore_index=True).merge(
        ei[["edge_id", "sector_from", "region_from", "sector_to", "valuation"]],
        on="edge_id")
    mf = e[(e.sector_from == "MF") & (e.sector_to == "Concentrate") & (e.valuation == "CC")]
    return (mf.groupby("region_from")[["prior", "post"]].sum()
              .rename_axis("region").reset_index())


def discharge_capacity(m, P, U, opt: str) -> pd.DataFrame:
    """Per region-year: mine prior vs what it can physically discharge.

    Copper leaves a mine node only as domestic smelter intake or as export mass
    times a grade the hard cone caps at ``BAND_HI``, so

        cap = BAND_HI · X_mass + U

    is the ceiling on discharge **at prior-level intake**. It is a diagnostic,
    not a theorem: the solver may also raise U, at its own cost — which is why a
    few regions show a forced shortfall larger than their realised deletion.
    """
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=ROOT / "acc")
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    rows = []
    for f in sorted((ROOT / "opt" / opt / "timesteps").glob("t_*.npz")):
        o = np.load(f, allow_pickle=True)
        rows.append(pd.DataFrame({"edge_id": np.asarray(o["cols_t"], dtype=int),
                                  "prior": o["a_prior"], "post": o["F_opt"]}))
    e = pd.concat(rows, ignore_index=True).merge(
        ei[["edge_id", "sector_from", "region_from", "sector_to", "valuation", "time"]],
        on="edge_id")
    mf = e[(e.sector_from == "MF") & (e.sector_to == "Concentrate") & (e.valuation == "CC")]
    d = (mf.groupby(["region_from", "time"])[["prior", "post"]].sum()
           .rename_axis(["region", "time"]))
    X = m.groupby(["region_from", "time"])["Value"].sum().rename("X_mass")
    d = d.join(X.rename_axis(["region", "time"]))
    d = d.join(pd.Series(U).rename("U").rename_axis(["region", "time"]))
    d = d.fillna({"X_mass": 0.0, "U": 0.0})
    d["cap"] = BAND_HI * d.X_mass + d.U
    d["headroom"] = d.cap - d.prior
    d["deleted"] = d.prior - d.post
    return d.reset_index()


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def main(opt: str = DEFAULT_RUN) -> None:
    print("=" * 78)
    print("Round N — is the concentrate grade the lever?   (ADR 0029)")
    m, P, U, screened = load_inputs()
    ig = implied_grades(m, P, U, screened)
    regions = sorted(set(m.region_from) | set(m.region_to) | {r for r, _ in P})
    years = sorted(set(m.time))

    # ── 1 ────────────────────────────────────────────────────────────────────
    print("\n== 1. Chile's implied grade (P−U)/X, year by year")
    c = ig[ig.region == "Chile"].sort_values("time")
    ok = c[(c.g >= BAND_LO) & (c.g <= BAND_HI)]
    for chunk in (c.iloc[:15], c.iloc[15:]):
        print("   " + "  ".join(f"{r.time[2:]}:{r.g:5.3f}{'*' if BAND_LO<=r.g<=BAND_HI else ' '}"
                                for r in chunk.itertuples(index=False)))
    print(f"   * = inside the physical band [{BAND_LO}, {BAND_HI}].  "
          f"{len(ok)}/{len(c)} years qualify.")
    print(f"   pooled over ALL years        g = {c.pu.sum()/c.x.sum():.4f}   "
          f"(the plan's evidence)")
    print(f"   pooled over the * years only g = {ok.pu.sum()/ok.x.sum():.4f}   "
          f"on {ok.x.sum():,.0f} kt = {ok.x.sum()/c.x.sum()*100:.0f} % of export mass")
    print(f"   the global prior it would replace: {PRIOR}  (σ {PRIOR_SD})")

    # ── 2 ────────────────────────────────────────────────────────────────────
    print("\n== 2. Is there a region signal? Near-pure exporters, in-band observations only")
    d = ig[(~ig.screened) & (ig.imp_share < 0.10)]
    print(f"   {'region':18s} {'n':>3s} {'Xtot':>9s} {'pooled':>7s} {'p10':>7s} "
          f"{'p90':>7s} {'in-band':>8s}")
    for r, s in sorted(d.groupby("region"), key=lambda kv: -kv[1].x.sum()):
        if s.x.sum() < 500:
            continue
        print(f"   {r:18s} {len(s):3d} {s.x.sum():9,.0f} {s.pu.sum()/s.x.sum():7.3f} "
              f"{s.g.quantile(.1):7.3f} {s.g.quantile(.9):7.3f} "
              f"{((s.g>=BAND_LO)&(s.g<=BAND_HI)).mean()*100:7.0f} %")
    ib = d[(d.g >= BAND_LO) & (d.g <= BAND_HI)]
    gm = np.average(ib.g, weights=ib.x)
    between = sum(s.x.sum() * (np.average(s.g, weights=s.x) - gm) ** 2
                  for _, s in ib.groupby("region"))
    within = sum((s.x * (s.g - np.average(s.g, weights=s.x)) ** 2).sum()
                 for _, s in ib.groupby("region"))
    print(f"\n   {len(ib)} region-years, {ib.x.sum():,.0f} kt:  world mean g = {gm:.4f}")
    print(f"   between-region variance {between/(between+within)*100:5.1f} %   "
          f"within-region (year) {within/(between+within)*100:5.1f} %")
    print(f"   mass-weighted sd of the region means = "
          f"{np.sqrt(between/ib.x.sum()):.4f}   vs prior σ {PRIOR_SD}")
    print("   → the between-region spread is a third of the uncertainty the single")
    print("     global row already declares. There is no signal to resolve.")

    # ── 3 ────────────────────────────────────────────────────────────────────
    print("\n== 3. Where the mass went — raw BACI, Chile, by HS code (kt)")
    raw_g, _, _, _ = load_root_trade(_source_dir(5), _conc_lookup())
    gross = raw_g[(raw_g.sector == "Concentrate") & (raw_g.region_from == "Chile")
                  & (raw_g.region_to != "Chile")].groupby("time")["Value"].sum()
    net = m[m.region_from == "Chile"].groupby("time")["Value"].sum()
    world_ex = m[(~m.region_from.isin(["Chile", "Papua New Guinea", "Italy"]))] \
        .groupby("time")["Value"].sum()
    print(f"   {'year':>5s} {'gross v5':>9s} {'netted':>9s} {'ICSG mine CC':>13s} "
          f"{'world ex-CL':>12s}")
    for t in [str(y) for y in range(2018, 2025)]:
        print(f"   {t:>5s} {gross.get(t,0):9,.0f} {net.get(t,0):9,.0f} "
              f"{P.get(('Chile',t),float('nan')):13,.0f} {world_ex.get(t,0):12,.0f}")
    print("   → gross ≡ netted (ADR 0027 never touched Chile); world ex-Chile is")
    print("     stable while Chile halves. The tonnage is absent from the source.")

    # ── 4 ────────────────────────────────────────────────────────────────────
    print(f"\n== 4. Discharge capacity: P > {BAND_HI}·X + U   (run {opt})")
    dc = discharge_capacity(m, P, U, opt)
    hard = dc[dc.headroom < 0]
    forced = (-hard.headroom).sum()
    print(f"   mine prior {dc.prior.sum():,.0f} kt CC   posterior {dc.post.sum():,.0f}   "
          f"deleted {dc.deleted.sum():,.0f} ({dc.deleted.sum()/dc.prior.sum()*100:.1f} %)")
    print(f"   HARD-CAPPED region-years {len(hard)} of {int((dc.prior>0).sum())} with production")
    print(f"   forced shortfall Σ max(0, P − cap) = {forced:,.0f} kt CC "
          f"= {forced/dc.deleted.sum()*100:.1f} % of ALL mine-chain deletion")
    top = (hard.assign(short=-hard.headroom).groupby("region")["short"].sum()
           .sort_values(ascending=False).head(6))
    tot = dc.groupby("region")["deleted"].sum()
    for r, v in top.items():
        print(f"     {r:18s} forced {v:8,.0f}   of deleted {tot[r]:8,.0f}   "
              f"({v/tot[r]*100:6.1f} %)")
    ch = dc[dc.region == "Chile"]
    free = ig[(ig.region == "Chile")
              & ig.time.isin(set(ch[ch.headroom >= 0].time))]
    print(f"   Chile: hard-capped in {int((ch.headroom<0).sum())}/{len(ch)} years; "
          f"in the {int((ch.headroom>=0).sum())} years WITH headroom the grade it needs is "
          f"{free.g.min():.3f}–{free.g.max():.3f} "
          f"(pooled {free.pu.sum()/free.x.sum():.4f})")
    print("   → raising a grade cannot help a year already evaluated at the ceiling,")
    print("     and the years with headroom need a grade at or BELOW the prior.")

    # ── 5 ────────────────────────────────────────────────────────────────────
    print("\n== 5. The proposed estimator, with each gate measured")
    print(f"   {'window':11s} {'gates':28s} {'eqs':>4s} {'at-bound':>9s} "
          f"{'mass@bound':>11s} {'Chile':>7s} {'‖res‖':>9s}")
    for centre in (2005, 2015, 2020):
        yrs = [str(centre + k) for k in range(-WINDOW, WINDOW + 1) if str(centre + k) in years]
        Xw = (m[m.time.isin(yrs)].groupby("region_from")["Value"].sum()
              .reindex(regions).fillna(0.0))
        for tag, gr, ds, dh in (("raw", False, False, False),
                                ("+require P and U", True, False, False),
                                ("+drop screened origins", True, True, False),
                                ("+drop ROW node equation", True, True, True)):
            g, res, neq = solve_window(m, P, U, screened, regions, yrs, gr, ds, dh)
            at = (g <= BAND_LO + 1e-9) | (g >= BAND_HI - 1e-9)
            print(f"   {yrs[0]}-{yrs[-1]:4s} {tag:28s} {neq:4d} "
                  f"{int(at.sum()):3d}/{len(regions):<5d} "
                  f"{Xw[at].sum()/Xw.sum()*100:10.1f} % {g['Chile']:7.3f} "
                  f"{np.linalg.norm(res):9,.0f}")
    print("   → even fully gated, a fifth to a half of export mass sits on a bound and")
    print("     Chile clips at the ceiling. For much of the record no grade inside the")
    print("     physical band closes these books — which is section 4 restated.")

    print("\n" + "=" * 78)
    print("VERDICT: the concentrate grade is not the lever (ADR 0029). The variance")
    print("is in the export MASS, not in the grade. See REP …roundN-concentrate-grade.")


if __name__ == "__main__":
    main(*sys.argv[1:2])
