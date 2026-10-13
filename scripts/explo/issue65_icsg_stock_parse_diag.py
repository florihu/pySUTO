"""
issue65_icsg_stock_parse_diag.py
================================
#65 measurement: the ICSG feed rebuilt from the repaired stock parse (#35).  Reads
persisted run artefacts only, builds the sum-row table in memory (no per-year CSVs, #46)
and prints #65's pre-registered items for <new> against <base>:

    C   common-set sum-row z² over 30 years, per year, per |c| band, and the rows whose
        z² moved most (by row_meta label)
    E   the Round-E items (ADR 0025): world retention, small-node posterior/prior,
        exporter retention, mass moved (CC)
    S   MESC / USC posterior edge-years by sign; per stock family (MESC, USC, PSC) and
        year the net Σ(a* − a_prior) and the gross Σ|a* − a_prior|, CC
    U   United States 2005–2012: PSC prior and posterior, and the summed z² of the
        United States sum rows
    G   Singapore's cathode import legs, prior and posterior per year (ADR 0027 netting)

    python scripts/explo/issue65_icsg_stock_parse_diag.py <opt_base> <opt_new>
    python scripts/explo/issue65_icsg_stock_parse_diag.py --crosseval <opt_base> <opt_new> <year> ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
from pysuto.acc import AccountingStructure              # noqa: E402
from pysuto.config import load_config                   # noqa: E402
from pysuto.diag import SolverDiag, SolverStatsDiagnostic  # noqa: E402
from pysuto.param_relation import expand_edge_index     # noqa: E402
import issue40_signed_stock_diag as D                   # noqa: E402
import roundL_reexport_diag as L                        # noqa: E402
import roundZ_png_kg_diag as Z                          # noqa: E402

ROOT   = Path("data/proc")
YEARS  = D.YEARS
US     = "United States"
US_YRS = [str(y) for y in range(2005, 2013)]
HUB    = "Singapore"


def _sums(opt, acc):
    """Soft scale-total rows, all 30 years, with their row_meta region and label."""
    d = SolverDiag.from_opt_id(opt, acc=acc)
    cf = SolverStatsDiagnostic(d.results, d.constraint_run, acc).constraint_frame()
    s = cf[cf["type"].astype(str) == "sum"]
    out = pd.DataFrame({"constraint_id": s["constraint_id"].astype(str).to_numpy(),
                        "year": s["year"].astype(str).to_numpy(),
                        "region": s["region"].astype(str).to_numpy(),
                        "label": s["label"].astype(str).to_numpy(),
                        "c": s["c"].to_numpy(), "lhs_post": s["lhs_post"].to_numpy(),
                        "z": s["z"].to_numpy()})
    out["z2"] = out.z ** 2
    return out


def _family(e):
    """Stock leg family of each edge-year: MESC, USC (Cathode→SB) or PSC (CP→SB)."""
    fam = np.where((e.sector_from == "Cathode") & e.sector_to.isin(D.STOCK), e.sector_to,
                   np.where((e.sector_from == "CP") & (e.sector_to == "PSC"), "PSC", ""))
    return e.assign(family=fam)[fam != ""]


def main(base, new):
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    E = {o: D._edges(o, ei) for o in (base, new)}
    S = {o: _sums(o, acc) for o in (base, new)}

    print("=" * 78)
    for o in (base, new):
        const, imp, feeds = L._chain(o)
        print(f"{o}  {L._status(o)}\n   const {const}  imp {imp}\n   feeds {feeds}")

    print("\n== C common-set sum-row z²")
    k = ["constraint_id", "year"]
    j = S[base][k + ["region", "label", "c", "z2"]].merge(S[new][k + ["c", "z2"]], on=k,
                                                         suffixes=("_base", "_new"))
    tb, tn = j.z2_base.sum(), j.z2_new.sum()
    print(f"   {len(j):,} rows in both runs: base {tb:,.0f}  new {tn:,.0f}  "
          f"Δ {100 * (tn / tb - 1):+.2f} % vs {base}")
    per = j.groupby("year")[["z2_base", "z2_new"]].sum()
    per["Δ %"] = 100 * (per.z2_new / per.z2_base - 1)
    print(per.round(2).T.to_string())
    band = pd.cut(j.c_base.abs(), Z.BANDS, right=False)
    print("   Δz² by |c| band (kt):",
          Z._fmt((j.z2_new - j.z2_base).groupby(band, observed=False).sum(), 0))
    j["dz2"] = j.z2_new - j.z2_base
    reg = j.groupby("region").dz2.sum().sort_values()
    print("   Δz² by region, largest falls:", Z._fmt(reg.head(6), 0))
    print("   Δz² by region, largest rises:", Z._fmt(reg.tail(6)[::-1], 0))
    top = j.reindex(j.dz2.abs().sort_values(ascending=False).index).head(15)
    print(top[["label", "year", "c_base", "c_new", "z2_base", "z2_new", "dz2"]]
          .round(1).to_string(index=False))
    cm = j[~np.isclose(j.c_base, j.c_new, rtol=1e-9, atol=1e-9)]
    print(f"   scale-total rows whose c moved: {len(cm):,}; by region:",
          Z._fmt(cm.groupby("region").size().sort_values(ascending=False).head(8), 0))

    print("\n== E Round-E items (ADR 0025)")
    for o in (base, new):
        wr, med, p90, exp, mass = Z._round_e(E[o], S[o])
        print(f"   {o}\n     world retention {Z._fmt(pd.Series(wr), 3)}"
              f"\n     small nodes posterior/prior median {med:.3f}  p90 {p90:.3f}"
              f"\n     exporter retention {Z._fmt(pd.Series(exp), 3)}"
              f"\n     mass moved, all CC edges {mass:,.0f} kt")

    print("\n== S stock legs")
    for o in (base, new):
        print(f"   posterior signs {o}\n{D._signs(D._stock(E[o]), 'post').to_string()}")
    rows = []
    for o in (base, new):
        f = _family(E[o]).query("valuation == 'CC'")
        g = f.assign(gross=f.delta.abs()).groupby(["family", "time"]).agg(
            prior=("prior", "sum"), post=("post", "sum"), net=("delta", "sum"),
            gross=("gross", "sum"))
        rows.append(g.reindex(pd.MultiIndex.from_product([["MESC", "USC", "PSC"], YEARS],
                                                         names=["family", "time"]))
                    .fillna(0.0).assign(run=o))
    st = pd.concat(rows).reset_index()
    for fam in ("MESC", "USC", "PSC"):
        x = st[st.family == fam].pivot_table(index="run", columns="time",
                                             values=["prior", "net", "gross"])
        print(f"   {fam}, kt CC per year")
        for col in ("prior", "net", "gross"):
            print(f"     {col:5s} " + "  ".join(
                f"{o[-8:]} Σ {x.loc[o, col].sum():9,.1f} max|·| {x.loc[o, col].abs().max():7,.1f}"
                f" ({x.loc[o, col].abs().idxmax()})" for o in (base, new)))
    tot = st.groupby(["run", "time"])[["prior", "net", "gross"]].sum()
    print("   MESC + USC + PSC, net and gross per year (kt CC)")
    print(pd.DataFrame({f"{c}_{o[-8:]}": tot.loc[o, c] for o in (base, new)
                        for c in ("net", "gross")}).round(1).T.to_string())

    print(f"\n== U {US} 2005–2012")
    for o in (base, new):
        p = _family(E[o]).query("family == 'PSC' and region_to == @US and time in @US_YRS")
        print(f"   {o} PSC CC prior/post: " + "  ".join(
            f"{r.time}:{r.prior:,.1f}/{r.post:,.1f}"
            for r in p[p.valuation == "CC"].sort_values("time").itertuples()))
    u = j[(j.region == US) & j.year.isin(US_YRS)]
    print(f"   {US} sum rows 2005–2012: {len(u)} rows, z² {u.z2_base.sum():,.0f} → "
          f"{u.z2_new.sum():,.0f}")
    print(u.reindex(u.dz2.abs().sort_values(ascending=False).index).head(8)
          [["label", "year", "c_base", "c_new", "z2_base", "z2_new"]].round(1).to_string(index=False))

    print(f"\n== G {HUB} cathode import legs, CC, kt")
    for o in (base, new):
        e = E[o]
        m = ((e.sector_from == "Cathode") & (e.region_to == HUB) & (e.region_from != HUB)
             & (e.valuation == "CC"))
        g = e[m].groupby("time")[["prior", "post"]].sum().reindex(YEARS).fillna(0.0)
        print(f"   {o} prior  {Z._fmt(g.prior.loc['2004':'2016'], 0)}")
        print(f"   {o} post   {Z._fmt(g.post.loc['2004':'2016'], 0)}")


if __name__ == "__main__":
    if sys.argv[1] == "--crosseval":
        Z.crosseval(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        main(sys.argv[1], sys.argv[2])
