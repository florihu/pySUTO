"""
roundZ_png_kg_diag.py
=====================
Round Z measurement (#48; ADR 0033, Round Z amendment): BACI concentrate legs whose
value buys far less than their recorded mass are rebuilt from the value. Reads
persisted run artefacts and the registered BACI input versions only, builds the
sum-row table in memory (no per-year CSVs, #46), and prints #48's pre-registered
items, every run against the first one named:

    Z-1  PNG concentrate export mass per year: gross and netted BACI (t → kt), and
         the Stage-2 Mass prior on its cross-region legs
    Z-2  SD-R-09 Concentrate (Flow → Process ∪ SB) prior, posterior and retention
         per valuation and year
    Z-3  PNG's share of the Philippines' and ROW's CC Concentrate→Smelting prior,
         PNG's CC export prior, and the z of PNG's 2020 legs
    Z-4  Congo D.R. → Zambia 2020: gross / netted mass, root-edge prior, posterior, z
    Z-h  ADR 0029 forced deletion (total, Chile) and the cross-region Concentrate
         legs of the exporter-years the low side does not flag
    C    common-set sum-row z² over 30 years, per year 2018–2023 and per |c| band;
         the Round-E items (ADR 0025); the ADR 0027 netting delta

    python scripts/explo/roundZ_png_kg_diag.py <opt>:<gross_v> [<opt>:<gross_v> ...]
    python scripts/explo/roundZ_png_kg_diag.py --sensitivity
    python scripts/explo/roundZ_png_kg_diag.py --crosseval <opt_base> <opt_new> <year> ...

``gross_v`` is the gross BACI input version the run's netted version was built from
(the netting does not record it). ``--sensitivity`` reads raw BACI read-only and
prints the low-side flags at leg gates 0.01 / 0.02 / 0.05 with the year gate fixed.
``--crosseval`` scores each run's posterior under each run's year-QP, in the
solver's own conditioning, so a ≤ 0.1 % move can be told from solver inexactness
(ADR 0039).

Run it only while no stage driver runs: the ICSG intake reader initialises the
input registry, which bumps any pending ``current/`` edit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
sys.path.insert(0, "scripts/parse/baci_parse")
from pysuto.acc import AccountingStructure                      # noqa: E402
from pysuto.config import load_config                           # noqa: E402
from pysuto.diag import SolverDiag, SolverStatsDiagnostic       # noqa: E402
from pysuto.param_relation import expand_edge_index             # noqa: E402
from build_baci_reexport import (                               # noqa: E402
    _conc_lookup, icsg_intake_production, load_root_trade)
import roundL_reexport_diag as L                                # noqa: E402
import roundN_concentrate_grade_diag as N                       # noqa: E402

ROOT       = Path("data/proc")
BACI_IN    = Path("data/input/feed/baci")
CONC_HS    = 260300
PNG, COD, ZMB = 598, 180, 894                       # BACI codes
R_PNG, R_COD, R_ZMB = "Papua New Guinea", "Congo, D.R.", "Zambia"
YEARS      = [str(y) for y in range(2015, 2025)]
FOCUS      = [str(y) for y in range(2018, 2024)]
SMALL_NODE = 18.0                                    # kt, ADR 0025 (b)
BANDS      = [0, 10, 100, 1000, np.inf]              # kt, ADR 0025's |c| bands
HUBS       = ["Philippines", "Germany", "India", "Singapore", "Korea, Rep.", "China"]


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def _versions(opt):
    """``(netted baci version, trade_shares version)`` through the run's sidecars."""
    _, _, feeds = L._chain(opt)
    b = json.load(open(ROOT / "feed" / "registry" / feeds["baci"] / "metadata.json"))
    i = json.load(open(ROOT / "feed" / "registry" / feeds["icsg"] / "metadata.json"))
    return (b["parents"]["inputs"]["feed"]["version"],
            i["parents"]["inputs"]["trade_shares"]["version"])


def _edges(opt, ei):
    const, _, _ = L._chain(opt)
    e = L._edges(opt, ei)
    e["sigma_a"] = np.load(ROOT / "const" / const / "sigma_a.npy")[e.edge_id.to_numpy()]
    e["delta"] = e.post - e.prior
    e["z"] = e.delta / e.sigma_a
    return e


def _sums(opt, acc):
    """Soft scale-total rows, all 30 years, built in memory (no CSV, #46)."""
    d = SolverDiag.from_opt_id(opt, acc=acc)
    cf = SolverStatsDiagnostic(d.results, d.constraint_run, acc).constraint_frame()
    s = cf[cf["type"].astype(str) == "sum"]
    out = pd.DataFrame({"constraint_id": s["constraint_id"].astype(str).to_numpy(),
                        "year": s["year"].astype(str).to_numpy(),
                        "c": s["c"].to_numpy(), "lhs_post": s["lhs_post"].to_numpy(),
                        "z": s["z"].to_numpy()})
    out["z2"] = out.z ** 2
    return out


def _png_mass(v):
    x = pd.read_csv(BACI_IN / f"v{v}" / "values.csv")
    x = x[(x.Sector == CONC_HS) & (x.Region_from == PNG) & (x.Region_to != PNG)]
    return x.groupby(x.Year.astype(str)).Value.sum()


def _leg_mass(v, i, j, year):
    x = pd.read_csv(BACI_IN / f"v{v}" / "values.csv")
    m = (x.Sector == CONC_HS) & (x.Region_from == i) & (x.Region_to == j) & (x.Year == year)
    return float(x.loc[m, "Value"].sum())


def _flagged_root(gross_v):
    """``{(root region, year)}`` of the exporter-years with a below-flagged leg."""
    rec = pd.read_csv(BACI_IN / f"v{gross_v}" / "quantity_recon.csv")
    if "side" not in rec.columns:
        return set()
    rmap = pd.read_excel(BACI_IN / f"v{gross_v}" / "conc.xlsx", sheet_name="region_map")
    r_of = dict(zip(rmap.src_region.astype(int), rmap.root_region))
    b = rec[rec.side == "below"]
    return set(zip(b.Region_from.map(r_of), b.Year.astype(str)))


# ---------------------------------------------------------------------------
# items
# ---------------------------------------------------------------------------

def _sd_r_09(e):
    m = ((e.sector_from == "Concentrate") & (e.entity_from == "Flow")
         & e.entity_to.isin(["Process", "SB"]) & e.time.isin(YEARS))
    g = e[m].groupby(["valuation", "time"])[["prior", "post"]].sum()
    g["ret"] = g.post / g.prior
    return g


def _png_cc(e):
    cc = e[(e.valuation == "CC") & (e.sector_from == "Concentrate")
           & (e.entity_from == "Flow") & e.time.isin(YEARS)]
    exp = cc[(cc.region_from == R_PNG) & (cc.region_to != R_PNG)].groupby("time").prior.sum()
    sm = cc[cc.sector_to == "Smelting"]
    shares = {}
    for imp in ("Philippines", "ROW"):
        s = sm[sm.region_to == imp]
        tot = s.groupby("time").prior.sum()
        png = s[s.region_from == R_PNG].groupby("time").prior.sum()
        shares[imp] = (png.reindex(tot.index).fillna(0.0) / tot)
    return exp, pd.DataFrame(shares)


def _png_mass_prior(e):
    m = e[(e.valuation == "Mass") & (e.sector_from == "Concentrate") & (e.entity_from == "Flow")
          & (e.region_from == R_PNG) & (e.region_to != R_PNG) & e.time.isin(YEARS)]
    return m.groupby("time").prior.sum()


def _legs(e, frm, to, year):
    m = ((e.sector_from == "Concentrate") & (e.sector_to == "Smelting")
         & (e.region_from == frm) & e.region_to.isin(to) & (e.time == year))
    return e[m][["region_to", "valuation", "prior", "post", "sigma_a", "z"]] \
        .sort_values(["region_to", "valuation"])


def _forced(opt, netted_v, P, U):
    raw, _, _, _ = load_root_trade(str(BACI_IN / f"v{netted_v}"), _conc_lookup())
    m = raw[(raw.sector == "Concentrate") & (raw.region_from != raw.region_to)]
    m = m.groupby(["region_from", "region_to", "time"])["Value"].sum().reset_index()
    dc = N.discharge_capacity(m, P, U, opt)
    hard = dc[dc.headroom < 0]
    ch = dc[dc.region == "Chile"]
    return (float((-hard.headroom).sum()), float(dc.deleted.sum()),
            float((-ch[ch.headroom < 0].headroom).sum()), int((ch.headroom < 0).sum()))


def _unflagged(e, flagged):
    m = ((e.sector_from == "Concentrate") & (e.entity_from == "Flow")
         & (e.region_from != e.region_to))
    g = e[m].groupby(["valuation", "region_from", "time"])[["prior", "post"]].sum()
    keep = [(r, t) not in flagged for _, r, t in g.index]
    return g[keep]


def _round_e(e, sums):
    small = sums[(sums.c.abs() < SMALL_NODE) & (sums.c.abs() > 0)]
    r = small.lhs_post / small.c
    exp = {}
    for reg in L.EXPORTERS + ["world"]:
        m = ((e.sector_from == "Concentrate") & (e.sector_to == "Smelting")
             & (e.valuation == "CC") & (e.region_from != e.region_to))
        if reg != "world":
            m &= e.region_from == reg
        exp[reg] = L._ret(e, m)
    return (L._world_retention(e), float(r.median()), float(r.quantile(0.9)), exp,
            float(e[e.valuation == "CC"].delta.abs().sum()))


def _ledger_T(netted_v):
    led = pd.read_csv(BACI_IN / f"v{netted_v}" / "reexport_ledger.csv")
    led = led[led.sector == "Concentrate"]
    return led.groupby(["region", led.time.astype(str)]).T_applied.sum().fillna(0.0)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _fmt(s, nd=0):
    return "  ".join(f"{k}:{v:,.{nd}f}" for k, v in s.items())


def main(specs):
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    runs = []
    for s in specs:
        opt, gross = s.split(":")
        nv, tv = _versions(opt)
        runs.append(dict(opt=opt, gross=int(gross), netted=nv, ts=tv))
    base = runs[0]
    E = {r["opt"]: _edges(r["opt"], ei) for r in runs}
    S = {r["opt"]: _sums(r["opt"], acc) for r in runs}
    U_, P_, _ = icsg_intake_production(["Concentrate"])
    P = {(r, t): v for (_s, r, t), v in P_.items()}
    U = {(r, t): v for (_s, r, t), v in U_.items()}

    print("=" * 78)
    for r in runs:
        const, imp, feeds = L._chain(r["opt"])
        print(f"{r['opt']}  {L._status(r['opt'])}\n   const {const}  imp {imp}\n"
              f"   baci gross v{r['gross']} → netted v{r['netted']}  trade_shares v{r['ts']}\n"
              f"   feeds {feeds['baci']}  {feeds['icsg']}")

    print("\n== Z-1 PNG concentrate exports, kt: gross / netted BACI / Stage-2 Mass prior")
    for r in runs:
        e = E[r["opt"]]
        print(f"   {r['opt']}\n     gross  {_fmt(_png_mass(r['gross']).reindex(YEARS), 1)}"
              f"\n     netted {_fmt(_png_mass(r['netted']).reindex(YEARS), 1)}"
              f"\n     prior  {_fmt(_png_mass_prior(e), 1)}")

    print("\n== Z-2 SD-R-09 Concentrate (Flow → Process ∪ SB): prior / posterior kt, retention")
    for r in runs:
        g = _sd_r_09(E[r["opt"]])
        print(f"   {r['opt']}")
        for val in ("Mass", "CC"):
            x = g.loc[val]
            print(f"     {val:4s} " + "  ".join(
                f"{t}: {row.prior:,.0f}/{row.post:,.0f} {row.ret:.3f}" for t, row in x.iterrows()))

    print("\n== Z-3 PNG's share of CC Concentrate→Smelting prior; PNG CC export prior (kt)")
    for r in runs:
        exp, sh = _png_cc(E[r["opt"]])
        print(f"   {r['opt']}\n     share PHL {_fmt(sh['Philippines'].reindex(YEARS), 3)}"
              f"\n     share ROW {_fmt(sh['ROW'].reindex(YEARS), 3)}"
              f"\n     PNG CC exports {_fmt(exp, 1)}")
        print("     2020 PNG legs:\n" + _legs(E[r["opt"]], R_PNG, ["Philippines", "ROW", "India"],
                                             "2020").round(2).to_string(index=False))

    print("\n== Z-4 Congo D.R. → Zambia 2020")
    for r in runs:
        print(f"   {r['opt']}  gross {_leg_mass(r['gross'], COD, ZMB, 2020):,.1f} kt  "
              f"netted {_leg_mass(r['netted'], COD, ZMB, 2020):,.1f} kt")
        print(_legs(E[r["opt"]], R_COD, [R_ZMB], "2020").round(2).to_string(index=False))

    print("\n== Z-h ADR 0029 forced deletion (kt CC): total / all mine deletion / Chile forced, years")
    for r in runs:
        f, d, cf, cy = _forced(r["opt"], r["netted"], P, U)
        print(f"   {r['opt']}  forced {f:,.0f} of {d:,.0f} deleted ({f / d * 100:.1f} %); "
              f"Chile forced {cf:,.0f}, hard-capped {cy}/30")
    flagged = set().union(*(_flagged_root(r["gross"]) for r in runs))
    print(f"   exporter-years with a below-flagged leg (root): {len(flagged)}")
    ub = _unflagged(E[base["opt"]], flagged)
    for r in runs[1:]:
        u = _unflagged(E[r["opt"]], flagged).reindex(ub.index).fillna(0.0)
        for val in ("Mass", "CC"):
            a, b = ub.loc[val], u.loc[val]
            dp, dq = (b.prior - a.prior).abs(), (b.post - a.post).abs()
            top = dq.sort_values(ascending=False).head(5)
            moved = dp > 0.01
            yr = dq.groupby(level="time").sum()
            print(f"   {r['opt']} vs base, unflagged {val}: Σ|Δprior| {dp.sum():,.1f}  "
                  f"Σ|Δpost| {dq.sum():,.1f} of Σpost {a.post.sum():,.0f} kt "
                  f"({dq.sum() / a.post.sum() * 100:.2f} %); on exporter-years whose prior "
                  f"moved {dq[moved].sum():,.1f}, prior unchanged {dq[~moved].sum():,.1f}"
                  f"\n     Σ|Δpost| per year: {_fmt(yr[yr > 1.0], 1)}"
                  "\n     largest |Δpost|: "
                  + "; ".join(f"{k[0]} {k[1]} {v:,.1f}" for k, v in top.items()))

    print("\n== C1 common-set sum-row z²")
    common = set.intersection(*(set(s.constraint_id) for s in S.values()))
    sb = S[base["opt"]][S[base["opt"]].constraint_id.isin(common)]
    zb = float(sb.z2.sum())
    print(f"   {len(common):,} rows present in every run")
    for r in runs:
        s = S[r["opt"]][S[r["opt"]].constraint_id.isin(common)]
        z = float(s.z2.sum())
        yrs = s[s.year.isin(FOCUS)].groupby("year").z2.sum()
        print(f"   {r['opt']}  {z:,.0f}  vs base {(z / zb - 1) * 100:+.2f} %   "
              f"per year {_fmt(yrs)}")
        band = pd.cut(sb.set_index("constraint_id").c.abs(), BANDS, right=False)
        zz = s.set_index("constraint_id").z2.reindex(band.index)
        print("     by |c| band: " + "  ".join(f"{k}: {v:,.0f}" for k, v in
                                               zz.groupby(band, observed=True).sum().items()))

    print("\n== C2 Round-E items (ADR 0025)")
    for r in runs:
        w, med, p90, exp, moved = _round_e(E[r["opt"]], S[r["opt"]])
        print(f"   {r['opt']}\n     world retention {_fmt(pd.Series(w), 3)}\n"
              f"     small-node (|c| < {SMALL_NODE:.0f} kt) post/prior median {med:.3f} p90 {p90:.3f}\n"
              f"     exporter conc retention {_fmt(pd.Series(exp), 3)}\n"
              f"     mass moved, all CC edges {moved:,.0f} kt")
        e = E[r["opt"]]
        for reg in L.EXPORTERS:
            m = ((e.sector_from == "Concentrate") & (e.sector_to == "Smelting")
                 & (e.valuation == "CC") & (e.region_from == reg)
                 & (e.region_to != reg) & e.time.isin(YEARS))
            g = e[m].groupby("time")[["prior", "post"]].sum()
            print(f"     {reg:9s} retention per year {_fmt(g.post / g.prior, 3)}")

    print("\n== C3 ADR 0027 netting: Concentrate T applied (kt CC)")
    T = {r["opt"]: _ledger_T(r["netted"]) for r in runs}
    for r in runs:
        t = T[r["opt"]]
        print(f"   {r['opt']} (v{r['netted']}) total {t.sum():,.0f}")
        by = t.groupby(level=0).sum()
        print("     hubs, 30 y: " + _fmt(by.reindex(HUBS).fillna(0.0)))
        yr = t[t.index.get_level_values(1).isin(["2019", "2020", "2021", "2022"])]
        print("     2019–22:    " + _fmt(yr.groupby(level=0).sum().reindex(HUBS).fillna(0.0)))


def _qp_year(cr, solver, t):
    """One year's QP of a const run, in the solver's own conditioning (ADR 0039)."""
    a0, sa = solver._condition_prior(cr.a_prior_global, cr.sigma_a)
    cols = np.asarray(cr.edge_cols_by_t[t])
    rows = np.where(cr.row_meta["time"].astype(str).to_numpy() == t)[0]
    ct = cr.row_meta["cone_type"].to_numpy()[rows]
    c = np.asarray(cr.timestep_data[t]["c"], dtype=float)[rows]
    sc = np.asarray(cr.timestep_data[t]["sigma_c"], dtype=float)[rows]
    sc = np.maximum(np.where(np.isnan(sc), solver.sigma_c_floor, sc), solver.sigma_c_floor)
    return dict(cols=cols, G=cr.G_blocks[t].tocsr(), ct=ct, c=c, sc=sc, a0=a0[cols], sa=sa[cols])


def _J(qp, F):
    """``Σ((F−a₀)/σ_a)² + Σ_free((GF−c)/σ_c)²`` and the hard-row violations of F."""
    r = qp["G"] @ F - qp["c"]
    free, zero, nn = qp["ct"] == "free", qp["ct"] == "zero", qp["ct"] == "nonneg"
    J = float((((F - qp["a0"]) / qp["sa"]) ** 2).sum() + ((r[free] / qp["sc"][free]) ** 2).sum())
    return J, float(np.abs(r[zero]).max(initial=0.0)), float(r[nn].max(initial=0.0))


def crosseval(base, new, years):
    """Each run's posterior under each run's year-QP (ADR 0039): a borrowed point that
    is feasible and scores lower than a run's own answer means that solve stopped short."""
    from pysuto.solver import ClarabelSolver
    solver = ClarabelSolver()
    crs = {o: solver.load_run(L._chain(o)[0]) for o in (base, new)}
    n_e = crs[base].G.shape[1]
    for y in years:
        t = str(y)
        F = {}
        for o in (base, new):
            z = np.load(ROOT / "opt" / o / "timesteps" / f"t_{t}.npz", allow_pickle=True)
            f = np.zeros(n_e)
            f[np.asarray(z["cols_t"], dtype=int)] = z["F_opt"]
            F[o] = f
        for qo in (base, new):
            qp = _qp_year(crs[qo], solver, t)
            const = float(((qp["a0"] / qp["sa"]) ** 2).sum())
            own, zv, nv = _J(qp, F[qo][qp["cols"]])
            other = new if qo == base else base
            bor, bzv, bnv = _J(qp, F[other][qp["cols"]])
            print(f"   {t} QP of {qo}: J(own) {own:,.1f} (obj {own - const:,.1f})  "
                  f"J(other's posterior) {bor:,.1f}  Δ {bor - own:+,.1f}  "
                  f"| other's hard violation: zero {bzv:.2e} nonneg {bnv:.2e} "
                  f"(own {zv:.2e} / {nv:.2e})")


def sensitivity():
    """Low-side flags at leg gates 0.01 / 0.02 / 0.05, year gate fixed (raw BACI)."""
    import os
    import baci_parse_feed as B
    hs = {B.CATHODE_HS} | set(B.RECON_HS)
    chunks = []
    for f in sorted(os.listdir(B.BACI_RAW_DIR)):
        if f.startswith("BACI") and f.endswith(".csv"):
            x = pd.read_csv(os.path.join(B.BACI_RAW_DIR, f),
                            usecols=["t", "i", "j", "k", "v", "q"], dtype={"k": int})
            chunks.append(x[x.k.isin(hs) & x.q.notna() & (x.q > 0)])
    raw = pd.concat(chunks, ignore_index=True)
    agg = (raw.groupby(["t", "k", "i", "j"])[["q", "v"]].sum().reset_index()
              .rename(columns={"t": "Year", "k": "Sector", "i": "Region_from", "j": "Region_to"}))
    p = B.copper_price(agg)
    rec = B.reconstruct_quantity(agg, B.reference_fractions(agg[agg.Sector.isin(B.RECON_HS)], p), p)
    base = (rec.Sector.isin(B.RECON_HS) & (rec.q_obs >= B.MIN_LEG_T)
            & (rec.year_frac < B.LO_YEAR_GATE))
    q_val = rec.v * 1000.0 / (rec.Year.map(p) * rec.g_ref)
    for gate in (0.01, B.LO_LEG_GATE, 0.05):
        s = base & (rec.cu_frac < gate)
        print(f"   leg gate {gate:.2f}: {int(s.sum())} legs in "
              f"{rec[s].groupby(['Region_from', 'Year']).ngroups} exporter-years; "
              f"{rec.q_obs[s].sum() / 1e3:,.0f} → "
              f"{np.minimum(q_val[s], rec.q_obs[s]).sum() / 1e3:,.0f} kt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--sensitivity":
        sensitivity()
    elif sys.argv[1] == "--crosseval":
        crosseval(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        main(sys.argv[1:])
