"""
roundX_stock_sink_diag.py
=========================
Round X measurement (issue #34; ADR 0040): the signed stock-change legs
(Cathode→MESC, Cathode→USC) held at their prior. Reads persisted run artifacts
only — the opt run's year-local timesteps, its const run's prior σ, its imp
run's imputation log — and prints the pre-registered metrics of the round:

    R   retention Σa*/Σa_prior per year (NaN where |Σa_prior| ≤ 50 kt)
    M1  mass moved Σ|a* − a_prior| over the indicator's legs, world, per year
    M2  cumulative drift Σ_{t≤T}(Σa* − Σa_prior)
    M3  origin plausibility Σ_j MESC*(i→j,t) / CP→Cathode*(i,t) over origins
        producing ≥ 100 kt, max per year
    W   mass written onto the legs by imputers other than the seed / the feeds

plus the σ-class split of the legs, the destination/origin tables, the Zambia
2010 trace rows, the Round-E stop-rule items (ADR 0025) and the common-set
sum-row z² against the base run.

    python scripts/explo/roundX_stock_sink_diag.py <opt_base> [<opt_new> ...]

Needs the per-year tables (``scripts/solver_stats.py <opt> --years all``) for
every run named.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/explo")
from pysuto.acc import AccountingStructure               # noqa: E402
from pysuto.config import load_config                    # noqa: E402
from pysuto.param_relation import expand_edge_index      # noqa: E402
import india_concentrate_diag as D                       # noqa: E402
import roundL_reexport_diag as L                         # noqa: E402

ROOT       = Path("data/proc")
STOCKS     = ("MESC", "USC")
R_MIN_PRIOR = 50.0          # kt: retention is quoted only above this prior sum
M1_LIMIT   = {"MESC": 100.0, "USC": 50.0}
M2_LIMIT   = 1000.0         # kt
M3_LIMIT   = 0.05
M3_MIN_PROD = 100.0         # kt: M3 is quoted for producing origins only (a merchant
                            # with 10 kt of pooled cathode and a 10 kt MESC leg is noise)
SMALL_NODE = 18.0           # kt, ADR 0025 (b)
ZAMBIA_ROWS = ["sum_usetotout_Cathode_Zambia_CC_2010",
               "sum_supplytotout_CP_Zambia_CC_2010"]
SEED_METHODS = {"share_seed"}
TWIN_PREFIX  = "parametric"         # conc twin (CC from Mass); counted only when its
                                    # Mass source was itself imputer-written, not a feed


def _edges_with_sigma(opt, ei):
    const, _, _ = L._chain(opt)
    e = L._edges(opt, ei)
    sig = np.load(ROOT / "const" / const / "sigma_a.npy")
    e["sigma_a"] = sig[e.edge_id.to_numpy()]
    e["delta"] = e.post - e.prior
    return e


def _writer_mass(opt, ei):
    _, imp, _ = L._chain(opt)
    return _writer_mass_imp(imp, ei)


def _writer_mass_imp(imp, ei):
    """Σ|f_after| written onto stock CC legs by non-seed, non-feed imputers.

    A Mass-booking writer (e.g. a FormulaRelation) shows up on the CC leg only
    through its conc twin, logged as ``parametric*`` — the same method as the
    twin of a feed-backed Mass leg. A twin is therefore counted iff its Mass
    source has an imputation-log entry of its own (a feed value has none)."""
    log = pd.read_parquet(ROOT / "imp" / imp / "imputation_log.parquet")
    written = set(log.edge_id[~log.method.isin(SEED_METHODS)].to_numpy().tolist())
    m = log.merge(ei[["edge_id", "sector_to", "valuation"]], on="edge_id")
    m = m[m.sector_to.isin(STOCKS) & (m.valuation == "CC")]
    twin = m.method.str.startswith(TWIN_PREFIX)
    derived = twin & m.source_edges.map(lambda s: any(int(x) in written for x in s))
    keep = ((~twin & ~m.method.isin(SEED_METHODS)) | derived) & (m.f_after.abs() > 0)
    other = m[keep]
    by = other.groupby(["sector_to", "method"]).agg(n=("edge_id", "size"),
                                                     mass=("f_after", lambda s: s.abs().sum()))
    return float(other.f_after.abs().sum()), by


def _metrics(e, ind):
    s = e[(e.sector_to == ind) & (e.valuation == "CC")]
    g = s.groupby("time").agg(prior=("prior", "sum"), post=("post", "sum"),
                              moved=("delta", lambda d: d.abs().sum()))
    g["R"] = np.where(g.prior.abs() > R_MIN_PRIOR, g.post / g.prior, np.nan)
    g["drift"] = (g.post - g.prior).cumsum()
    return g


def _origin_share(e):
    m = e[(e.sector_to == "MESC") & (e.valuation == "CC")]
    out = m.groupby(["region_from", "time"]).post.sum().rename("mesc")
    p = e[(e.sector_from == "CP") & (e.sector_to == "Cathode") & (e.valuation == "CC")]
    prod = p.groupby(["region_from", "time"]).post.sum().rename("prod")
    t = pd.concat([out, prod], axis=1).dropna()
    t = t[t["prod"] >= M3_MIN_PROD]
    t["share"] = t.mesc / t["prod"]
    return t


def _sigma_classes(e, ind):
    s = e[(e.sector_to == ind) & (e.valuation == "CC")]
    cls = pd.cut(s.sigma_a, [-1, 0.1, 2, 10, 100, 1e9],
                 labels=["<=0.1", "0.1-2", "2-10", "10-100", ">100"])
    return s.groupby(cls, observed=True).agg(n=("edge_id", "size"),
                                             moved=("delta", lambda d: d.abs().sum()))


def _small_nodes(opt):
    c = L._sums(opt)
    c = c[(c.c.abs() < SMALL_NODE) & (c.c.abs() > 0)]
    r = (c.lhs_post / c.c)
    return float(r.median()), float(r.quantile(0.9)), len(c)


def _exporter_retention(e):
    out = {}
    for reg in L.EXPORTERS + ["world"]:
        m = ((e.sector_from == "Concentrate") & (e.sector_to == "Smelting")
             & (e.valuation == "CC") & (e.region_from != e.region_to))
        if reg != "world":
            m &= (e.region_from == reg)
        out[reg] = L._ret(e, m)
    return out


def _zambia_trace(opt, e):
    const, _, _ = L._chain(opt)
    c = D._rows(opt, const)
    rows = c[c.constraint_id.isin(ZAMBIA_ROWS)][["constraint_id", "c", "lhs_prior", "lhs_post", "sigma_c", "z"]]
    z = e[(e.sector_from == "Cathode") & (e.region_from == "Zambia") & (e.valuation == "CC")
          & (e.time.astype(str) == "2010")]
    mesc = z[z.sector_to == "MESC"].sort_values("post", ascending=False).head(3)
    exp = z[(z.sector_to == "FU") & (z.region_to != "Zambia")]
    prod = e[(e.sector_from == "CP") & (e.region_from == "Zambia") & (e.valuation == "CC")
             & (e.time.astype(str) == "2010")]
    return rows, mesc, (float(exp.prior.sum()), float(exp.post.sum())), (float(prod.prior.sum()), float(prod.post.sum()))


def main(base, news):
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    runs = [("base", base)] + [(f"X.{i}", r) for i, r in enumerate(news)]

    print("=" * 78)
    E, S = {}, {}
    for lab, r in runs:
        const, imp, _ = L._chain(r)
        icfg = json.load(open(ROOT / "imp" / imp / "metadata.json")).get("stage_config", {})
        ccfg = json.load(open(ROOT / "const" / const / "metadata.json")).get("stage_config", {})
        print(f"{lab:5s} {r}  {L._status(r)}\n      const {const}  imp {imp}\n"
              f"      seed_cv={icfg.get('stock_seed_cv')} seed_floor={icfg.get('stock_seed_abs_floor')} "
              f"stock_sectors={icfg.get('stock_sectors')} "
              f"stock_bounds_floor={ccfg.get('stock_bounds_scale_abs_floor')}")
        E[lab] = _edges_with_sigma(r, ei)
        S[lab] = L._sums(r)

    if len(runs) > 1:
        common = set.intersection(*(set(s.constraint_id) for s in S.values()))
        print(f"\n== common-set sum-row z² ({len(common):,} rows present in every run)")
        zb = float(S["base"][S["base"].constraint_id.isin(common)].z2.sum())
        for lab, _ in runs:
            z = float(S[lab][S[lab].constraint_id.isin(common)].z2.sum())
            print(f"   {lab:5s} {z:,.0f}   vs base {(z / zb - 1) * 100:+.1f} %")

    for lab, r in runs:
        e = E[lab]
        print(f"\n{'#' * 78}\n# {lab}  {r}\n{'#' * 78}")
        w_total, w_by = _writer_mass(r, ei)
        print(f"\n== W  mass written onto MESC/USC CC legs by non-seed, non-feed imputers: {w_total:,.0f} kt")
        if len(w_by):
            print(w_by.round(0).to_string())
        for ind in STOCKS:
            g = _metrics(e, ind)
            print(f"\n== {ind} CC per year (kt): prior, posterior, R, M1 moved, M2 drift")
            print(g.round(2).to_string())
            n_r_out = int(((g.R < 0.8) | (g.R > 1.2)).sum())
            print(f"   R outside [0.8, 1.2]: {n_r_out} of {int(g.R.notna().sum())} quoted years"
                  f"   | M1 max {g.moved.max():,.0f} kt (limit {M1_LIMIT[ind]:.0f})"
                  f"   | M2 max |drift| {g.drift.abs().max():,.0f} kt (limit {M2_LIMIT:.0f})"
                  f"   | 30-y Σprior {g.prior.sum():,.0f} → Σpost {g.post.sum():,.0f}")
            print(f"   σ classes: " + "  ".join(f"{k}: n={v.n} moved={v.moved:,.0f}"
                                               for k, v in _sigma_classes(e, ind).iterrows()))
        t = _origin_share(e)
        worst = t.sort_values("share", ascending=False).head(6)
        print(f"\n== M3 MESC share of cathode output per origin-year: max {t.share.max():.3f} "
              f"(limit {M3_LIMIT}); origin-years above limit: {int((t.share > M3_LIMIT).sum())}")
        print(worst.round(3).to_string())
        zam = t.loc[("Zambia", "2010")] if ("Zambia", "2010") in t.index else None
        if zam is not None:
            print(f"   Zambia 2010: mesc {zam.mesc:.1f} / prod {zam['prod']:.1f} = {zam.share:.3f}")
        m = e[(e.sector_to == "MESC") & (e.valuation == "CC")]
        print("\n== MESC CC Σ(post − prior) 1995-2024 by destination (top 6) / by origin (top 8)")
        print("   dest:   " + "  ".join(f"{k} {v:+,.0f}" for k, v in m.groupby("region_to").delta.sum().sort_values(ascending=False).head(6).items()))
        print("   origin: " + "  ".join(f"{k} {v:+,.0f}" for k, v in m.groupby("region_from").delta.sum().sort_values(ascending=False).head(8).items()))

        rows, mesc, exp, prod = _zambia_trace(r, e)
        print("\n== Zambia 2010 CC trace")
        print(rows.round(2).to_string(index=False))
        print(f"   CP→Cathode Zambia prior {prod[0]:.1f} → post {prod[1]:.1f}; "
              f"cathode export legs (Cathode Zambia→FU abroad) prior {exp[0]:.1f} → post {exp[1]:.1f}")
        print("   largest MESC legs: " + "; ".join(
            f"→{row.region_to} prior {row.prior:.1f} post {row.post:.1f} σ {row.sigma_a:.1f}"
            for _, row in mesc.iterrows()))

        print("\n== Round-E items (ADR 0025)")
        print("   world retention: " + "  ".join(f"{k} {v:.3f}" for k, v in L._world_retention(e).items()))
        med, p90, n = _small_nodes(r)
        print(f"   small-node (|c| < {SMALL_NODE:.0f} kt) posterior/prior: median {med:.3f}  p90 {p90:.3f}  (n={n:,})")
        print("   exporter concentrate retention (cross-region Concentrate→Smelting CC): "
              + "  ".join(f"{k} {v:.3f}" for k, v in _exporter_retention(e).items()))
        cc = e[e.valuation == "CC"]
        print(f"   mass moved, all CC edges: {cc.delta.abs().sum():,.0f} kt")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
