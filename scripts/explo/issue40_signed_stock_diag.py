"""
issue40_signed_stock_diag.py
============================
#40 measurement: a stock draw-down fans like a build of the same size, so the negative
MESC / USC changes ICSG reports reach the account.  Reads persisted run artefacts only,
builds the sum-row table in memory (no per-year CSVs, #46) and prints #40's
pre-registered items for <new> against <base>:

    P2  MESC / USC prior edge-years by sign, per valuation
    P3  MESC / USC posterior edge-years by sign, per valuation
    P4  Σ|a* − a_prior| over the MESC + USC legs per year (CC), and its change
    P5  the Round-E items (ADR 0025): world retention, small-node posterior/prior,
        exporter retention, mass moved (CC)
    P6  priors that moved outside the edges leaving a Cathode node, and the scale-total
        rows whose c moved
    C   common-set sum-row z² over 30 years, per year and per |c| band

With a third run <opt_ref> (#40.1, against #40's own run):

    H   the cathode use totals that carried a negative target in <opt_ref>: their
        summed z² in <opt_ref> and in <opt_new>
    V   stock leg-years whose CC prior is negative while their Mass prior is positive

    python scripts/explo/issue40_signed_stock_diag.py <opt_base> <opt_new> [<opt_ref>]
    python scripts/explo/issue40_signed_stock_diag.py --crosseval <opt_base> <opt_new> <year> ...

``--crosseval`` scores each run's posterior under each run's year-QP (ADR 0039), so a
≤ 0.1 % move can be told from solver inexactness.
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
from pysuto.param_relation import expand_edge_index     # noqa: E402
import roundL_reexport_diag as L                        # noqa: E402
import roundZ_png_kg_diag as Z                          # noqa: E402

ROOT   = Path("data/proc")
YEARS  = [str(y) for y in range(1995, 2025)]
STOCK  = ("MESC", "USC")
BANDS  = Z.BANDS


def _edges(opt, ei):
    """Prior and posterior of every edge-year the solver saw, all 30 years."""
    rows = []
    for y in YEARS:
        o = np.load(ROOT / "opt" / opt / "timesteps" / f"t_{y}.npz", allow_pickle=True)
        rows.append(pd.DataFrame({"edge_id": np.asarray(o["cols_t"], dtype=int),
                                  "prior": o["a_prior"], "post": o["F_opt"]}))
    e = pd.concat(rows, ignore_index=True).merge(
        ei[["edge_id", "sector_from", "region_from", "sector_to", "region_to",
            "entity_from", "entity_to", "valuation", "time"]], on="edge_id")
    e["delta"] = e.post - e.prior
    return e


def _stock(e):
    return e[(e.sector_from == "Cathode") & e.sector_to.isin(STOCK)]


def _signs(s, col):
    v = s[col]
    out = pd.DataFrame({"neg": v.lt(0), "zero": v.eq(0), "pos": v.gt(0)})
    out[["sector_to", "valuation"]] = s[["sector_to", "valuation"]].to_numpy()
    return out.groupby(["sector_to", "valuation"])[["neg", "zero", "pos"]].sum()


def _mixed_signs(e):
    """Stock leg-years with a negative CC prior beside a positive Mass prior."""
    k = ["sector_from", "region_from", "sector_to", "region_to", "time"]
    w = _stock(e).pivot_table(index=k, columns="valuation", values="prior")
    return int(((w["CC"] < 0) & (w["Mass"] > 0)).sum())


def hubs_and_valuations(new, ref, acc, ei):
    S = {o: Z._sums(o, acc) for o in (ref, new)}
    r = S[ref]
    hub = r[r.constraint_id.str.startswith("sum_usetotout_Cathode_") & (r.c < 0)]
    k = ["constraint_id", "year"]
    j = hub[k + ["z2"]].merge(S[new][k + ["z2"]], on=k, how="left", suffixes=("_ref", "_new"))
    print(f"\n== H {len(j)} cathode use totals with c < 0 in {ref}: z² {j.z2_ref.sum():,.0f} → "
          f"{j.z2_new.sum():,.0f} in {new} ({100 * j.z2_new.sum() / j.z2_ref.sum():.2f} %)")
    print("\n== V stock leg-years with CC prior < 0 and Mass prior > 0")
    for o in (ref, new):
        print(f"   {o}: {_mixed_signs(_edges(o, ei)):,}")


def main(base, new):
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    E = {o: _edges(o, ei) for o in (base, new)}
    S = {o: Z._sums(o, acc) for o in (base, new)}

    print("=" * 78)
    for o in (base, new):
        const, imp, feeds = L._chain(o)
        print(f"{o}  {L._status(o)}\n   const {const}  imp {imp}\n   feeds {feeds}")

    for tag, col in (("P2 prior", "prior"), ("P3 posterior", "post")):
        print(f"\n== {tag}: Cathode→MESC / USC edge-years by sign")
        for o in (base, new):
            print(f"   {o}\n{_signs(_stock(E[o]), col).to_string()}")

    print("\n== P4 Σ|a* − a_prior| over the MESC + USC legs, CC, kt per year")
    moved = {o: (_stock(E[o]).query("valuation == 'CC'")
                 .assign(m=lambda d: d.delta.abs()).groupby("time").m.sum()
                 .reindex(YEARS).fillna(0.0)) for o in (base, new)}
    d = moved[new] - moved[base]
    print(pd.DataFrame({"base": moved[base], "new": moved[new], "Δ": d}).round(1).T.to_string())
    print(f"   max Δ {d.max():.1f} kt ({d.idxmax()}); limit < 100 kt in every year")

    print("\n== P5 Round-E items (ADR 0025)")
    for o in (base, new):
        wr, med, p90, exp, mass = Z._round_e(E[o], S[o])
        print(f"   {o}\n     world retention {Z._fmt(pd.Series(wr), 3)}"
              f"\n     small nodes posterior/prior median {med:.3f}  p90 {p90:.3f}"
              f"\n     exporter retention {Z._fmt(pd.Series(exp), 3)}"
              f"\n     mass moved, all CC edges {mass:,.0f} kt")

    print("\n== P6 priors that moved")
    k = ["edge_id", "time"]
    m = E[base][k + ["sector_from", "sector_to", "valuation", "prior"]].merge(
        E[new][k + ["prior"]], on=k, how="outer", suffixes=("_base", "_new"))
    m["moved"] = ~np.isclose(m.prior_base, m.prior_new, rtol=1e-9, atol=1e-9, equal_nan=True)
    mv = m[m.moved]
    cath = mv.sector_from == "Cathode"
    print(f"   edge-years with a moved prior: {len(mv):,} (leaving a Cathode node {int(cath.sum()):,})")
    if (~cath).any():
        o = mv[~cath].assign(d=lambda x: x.prior_new - x.prior_base)
        print(o.groupby(["sector_from", "sector_to", "valuation"])
               .agg(n=("d", "size"), sum_abs=("d", lambda x: x.abs().sum())).round(2).to_string())
    c = S[base][["constraint_id", "year", "c"]].merge(S[new][["constraint_id", "year", "c"]],
                                                      on=["constraint_id", "year"],
                                                      suffixes=("_base", "_new"))
    cm = c[~np.isclose(c.c_base, c.c_new, rtol=1e-9, atol=1e-9)]
    print(f"   scale-total rows whose c moved: {len(cm):,}")
    if len(cm):
        print(cm.assign(d=cm.c_new - cm.c_base).reindex(
            cm.assign(d=(cm.c_new - cm.c_base).abs()).d.sort_values(ascending=False).index)
              .head(12).round(3).to_string(index=False))

    print("\n== C common-set sum-row z²")
    j = S[base][["constraint_id", "year", "c", "z2"]].merge(
        S[new][["constraint_id", "year", "z2"]], on=["constraint_id", "year"],
        suffixes=("_base", "_new"))
    tb, tn = j.z2_base.sum(), j.z2_new.sum()
    print(f"   {len(j):,} rows in both runs: base {tb:,.0f}  new {tn:,.0f}  "
          f"Δ {100 * (tn / tb - 1):+.2f} % vs {base}")
    per = j.groupby("year")[["z2_base", "z2_new"]].sum()
    per["Δ %"] = 100 * (per.z2_new / per.z2_base - 1)
    print(per.round(2).T.to_string())
    band = pd.cut(j.c.abs(), BANDS, right=False)
    print("   Δz² by |c| band (kt):",
          Z._fmt((j.z2_new - j.z2_base).groupby(band, observed=False).sum(), 0))


if __name__ == "__main__":
    if sys.argv[1] == "--crosseval":
        Z.crosseval(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        main(sys.argv[1], sys.argv[2])
        if len(sys.argv) > 3:
            _acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                           storage_dir=str(ROOT / "acc"))
            hubs_and_valuations(sys.argv[2], sys.argv[3], _acc,
                                expand_edge_index(_acc.edge_index, _acc.node_index))
