"""
issue69_aggregate_gap_diag.py
=============================
#69 measurement of #67: Sweden's cathode is no longer netted in 1996–2003, because its
flows sit inside ICSG's "Scandinavia" (params ``icsg_aggregates``).  Reads persisted
run artefacts, the constraint runs' σ_a and the two netted BACI versions' ledgers only,
builds the sum-row table in memory (no per-year CSVs, #46) and prints #69's
pre-registered items for <new> against <base>:

    N   netting: per hub-year T and T_applied, baci v<base> ledger → v<new> ledger
    W   Sweden: Mining→Ore CC posterior/prior, where its ore goes (MF or SXEW) with σ,
        its cathode leg groups, the ROW → Sweden MESC leg's z, and its sum-row z²
    G   Singapore: sum-row z² over all years
    E   the Round-E items (ADR 0025): world retention, small-node posterior/prior,
        exporter retention, z² by anchor band and mass moved (CC)
    S   MESC, USC and PSC: net Σ(a* − a_prior) per year, CC, per family and per region
    C   common-set sum-row z² over 30 years, per year, and by region
    A   attribution: the rows behind the < 10 kt band's Δz² and behind the largest
        regional moves outside Sweden

    python scripts/explo/issue69_aggregate_gap_diag.py <opt_base> <opt_new> \\
                                                       <baci_v_base> <baci_v_new>
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
import issue40_signed_stock_diag as D                   # noqa: E402
import issue65_icsg_stock_parse_diag as P65             # noqa: E402
import roundL_reexport_diag as L                        # noqa: E402
import roundZ_png_kg_diag as Z                          # noqa: E402

ROOT    = Path("data/proc")
BACI_IN = Path("data/input/feed/baci")
YEARS   = D.YEARS
SWEDEN  = "Sweden"
SW_YRS  = [str(y) for y in range(1996, 2004)]
HUB     = "Singapore"
KEY     = ["sector", "region", "time"]


def _ledger(v: int) -> pd.DataFrame:
    """One netted BACI version's re-export ledger; years stay strings."""
    led = pd.read_csv(BACI_IN / f"v{v}" / "reexport_ledger.csv", dtype={"time": str})
    return led.fillna({"T_applied": 0.0})


def _top(x: pd.DataFrame, n: int, cols: list) -> str:
    """The ``n`` rows of ``x`` with the largest |dz2|, as a table."""
    return (x.reindex(x.dz2.abs().sort_values(ascending=False).index).head(n)[cols]
            .round(1).to_string(index=False))


def _with_sigma(e: pd.DataFrame, opt: str) -> pd.DataFrame:
    """``e`` with each edge's σ_a from the run's constraint run, and its z."""
    const, _, _ = L._chain(opt)
    sigma = np.load(ROOT / "const" / const / "sigma_a.npy")
    return e.assign(sigma=sigma[e.edge_id.to_numpy()],
                    z=lambda d: (d.post - d.prior) / d.sigma)


def netting(v_base: int, v_new: int) -> None:
    """N: which hub-years' T and T_applied change between the two ledgers."""
    print(f"\n== N netting, baci v{v_base} → v{v_new} ledger (kt CC)")
    m = _ledger(v_base).merge(_ledger(v_new), on=KEY, how="outer",
                              suffixes=("_base", "_new"), indicator=True)
    print(f"   hub-years in one ledger only: {int((m['_merge'] != 'both').sum())}")
    m["dT"], m["dTa"] = m.T_new - m.T_base, m.T_applied_new - m.T_applied_base
    print(f"   hub-years whose T changes: "
          f"{m[m.dT != 0.0].groupby(['sector', 'region']).size().to_dict()}")
    print(f"   Σ ΔT {m.dT.sum():,.1f}   Σ ΔT_applied {m.dTa.sum():,.1f}")
    other = m[(m.region != SWEDEN) & (m.dTa.abs() > 1e-9)]
    if len(other):
        top = other.loc[other.dTa.abs().idxmax()]
        print(f"   other hub-years whose T_applied moves: {len(other)}, largest "
              f"{top.sector} {top.region} {top.time} {top.dTa:+.3f}")
    else:
        print("   other hub-years whose T_applied moves: none")
    sw = m[(m.region == SWEDEN) & (m.sector == "Cathode") & m.time.isin(SW_YRS)]
    cols = ["time", "U_base", "T_base", "T_applied_base", "label_base", "T_new",
            "T_applied_new", "label_new", "icsg_aggregate"]     # the last: #67 ledgers only
    print(sw[[c for c in cols if c in sw.columns]].round(2).to_string(index=False))


def _sweden_cc(e: pd.DataFrame, sector_from: str, sector_to: str) -> pd.DataFrame:
    """Sweden's domestic ``sector_from → sector_to`` CC edges, one row per year."""
    m = ((e.sector_from == sector_from) & (e.sector_to == sector_to) & (e.valuation == "CC")
         & (e.region_from == SWEDEN) & (e.region_to == SWEDEN))
    return e[m]


def _leg_groups(e: pd.DataFrame) -> pd.DataFrame:
    """Sweden's cathode legs in CC by group and year, prior and posterior."""
    c = e[(e.sector_from == "Cathode") & (e.valuation == "CC")]
    into = (c.region_to == SWEDEN) & (c.region_from != SWEDEN)
    group = np.select([into & (c.sector_to == s) for s in ("FU", "MESC", "USC")]
                      + [(c.region_from == SWEDEN) & (c.region_to != SWEDEN)],
                      ["imports into FU", "imports into MESC", "imports into USC",
                       "exports"], default="")
    legs = c.assign(group=group)[group != ""]
    own = _sweden_cc(e, "CSXEW", "CP").assign(group="own CSXEW→CP")
    return pd.concat([legs, own]).groupby(["group", "time"])[["prior", "post"]].sum()


def sweden(E: dict, j: pd.DataFrame, base: str, new: str) -> None:
    """W: Sweden's mine output, where its ore goes, its cathode legs and its z²."""
    print(f"\n== W {SWEDEN}")
    for o in (base, new):
        m = _sweden_cc(E[o], "Mining", "Ore").set_index("time")
        r = (m.post / m.prior.where(m.prior > 0)).loc["1995":"2005"]
        print(f"   Mining→Ore CC posterior/prior {o[-8:]}: {Z._fmt(r, 2)}")
    for o in (base, new):
        for route in ("MF", "SXEW"):
            r = _with_sigma(_sweden_cc(E[o], "Ore", route), o).set_index("time")
            yrs, sw = r.loc["1995":"2005"], r.loc[SW_YRS]
            print(f"   Ore→{route:4s} CC {o[-8:]} prior {Z._fmt(yrs.prior, 1)}")
            print(f"   Ore→{route:4s} CC {o[-8:]} post  {Z._fmt(yrs.post, 1)}")
            print(f"   Ore→{route:4s} CC {o[-8:]} 1996–2003: Σ post {sw.post.sum():,.1f}, "
                  f"σ_a {sw.sigma.min():.3g}–{sw.sigma.max():.3g}; prior max over "
                  f"{len(r)} years {r.prior.max():.1f}")
    for o in (base, new):
        g = _leg_groups(E[o])
        print(f"   cathode legs {o[-8:]}, prior / post, 2001 and 2003:")
        for grp in g.index.get_level_values(0).unique():
            x = g.loc[grp].reindex(SW_YRS).fillna(0.0)
            print(f"     {grp:18s} " + "  ".join(f"{t}: {x.prior[t]:.1f} / {x.post[t]:.1f}"
                                                 for t in ("2001", "2003")))
    for o in (base, new):
        e = E[o]
        leg = _with_sigma(e[(e.sector_from == "Cathode") & (e.region_from == "ROW")
                            & (e.region_to == SWEDEN) & (e.sector_to == "MESC")
                            & (e.time == "2001")], o)
        print(f"   ROW → Sweden MESC 2001 {o[-8:]}: " + "  ".join(
            f"{r.valuation} prior {r.prior:.1f} post {r.post:.1f} σ {r.sigma:.3g} "
            f"z {r.z:.2f}"
            for r in leg.itertuples()))
    s = j[(j.region == SWEDEN) & j.year.isin(SW_YRS)]
    print(f"   sum rows 1996–2003: {len(s)} rows, z² {s.z2_base.sum():,.0f} → "
          f"{s.z2_new.sum():,.0f} ({s.dz2.sum():+,.0f})")
    print(s.groupby("year")[["z2_base", "z2_new", "dz2"]].sum().round(0).T.to_string())
    print(_top(s, 10, ["label", "year", "c_base", "c_new", "z2_base", "z2_new"]))


def singapore(j: pd.DataFrame, base: str) -> None:
    """G: Singapore's sum-row z² over all years."""
    s = j[j.region == HUB]
    print(f"\n== G {HUB}: {len(s)} sum rows, z² {s.z2_base.sum():,.0f} → "
          f"{s.z2_new.sum():,.0f} ({100 * (s.z2_new.sum() / s.z2_base.sum() - 1):+.3f} % "
          f"vs {base})")


def round_e(E: dict, S: dict, j: pd.DataFrame, base: str, new: str) -> None:
    """E: the Round-E items of ADR 0025 for both runs."""
    print(f"\n== E Round-E items (ADR 0025), vs {base}")
    out = {}
    for o in (base, new):
        out[o] = Z._round_e(E[o], S[o])
        wr, med, p90, exp, mass = out[o]
        print(f"   {o}\n     (a) world retention {Z._fmt(pd.Series(wr), 3)}"
              f"\n     (b) small nodes posterior/prior median {med:.3f}  p90 {p90:.3f}"
              f"\n     (c) exporter retention {Z._fmt(pd.Series(exp), 3)}"
              f"\n     (d) mass moved, all CC edges {mass:,.0f} kt")
    band = pd.cut(j.c_base.abs(), Z.BANDS, right=False)
    by = j.groupby(band, observed=False)[["z2_base", "z2_new", "dz2"]].sum()
    print("   (d) z² by anchor band |c| (kt), base / new / Δ:")
    print(by.round(0).to_string())
    mb, mn = out[base][4], out[new][4]
    print(f"   (d) mass moved {mb:,.0f} → {mn:,.0f} kt "
          f"({100 * (mn / mb - 1):+.2f} % vs {base})")


def stock(E: dict, base: str, new: str) -> None:
    """S: stock-leg net per family and year, and the region-years that move."""
    print("\n== S stock legs, net Σ(a* − a_prior) per year, kt CC")
    fam = {o: P65._family(E[o]).query("valuation == 'CC'") for o in (base, new)}
    idx = pd.MultiIndex.from_product([["MESC", "USC", "PSC"], YEARS],
                                     names=["family", "time"])
    st = pd.concat([fam[o].groupby(["family", "time"]).delta.sum().reindex(idx)
                    .fillna(0.0).rename(o) for o in (base, new)], axis=1)
    st["grows"] = st[new].abs() - st[base].abs()
    for f in ("MESC", "USC", "PSC"):
        x = st.loc[f]
        top = x.grows.idxmax()
        print(f"   {f}: Σ net {x[base].sum():,.1f} → {x[new].sum():,.1f}; largest growth "
              f"of |net| {x.grows.max():+.1f} in {top} ({x.loc[top, base]:,.1f} → "
              f"{x.loc[top, new]:,.1f})")
    print(st.loc[(slice(None), SW_YRS + ["2004"]), [base, new]].round(1).unstack(0)
          .to_string())
    r = pd.concat([fam[o].groupby(["family", "time", "region_to"]).delta.sum().rename(o)
                   for o in (base, new)], axis=1).fillna(0.0)
    r["d"] = r[new] - r[base]
    print("   region-years whose stock net moves by more than 1 kt CC:")
    print(r[r.d.abs() > 1.0].sort_values("d").round(1).to_string())


def common_set(j: pd.DataFrame, base: str) -> None:
    """C: the common-set sum-row z², per year and by region."""
    tb, tn = j.z2_base.sum(), j.z2_new.sum()
    print(f"\n== C common-set sum-row z²: {len(j):,} rows in both runs, "
          f"{tb:,.0f} → {tn:,.0f} ({100 * (tn / tb - 1):+.3f} % vs {base})")
    per = j.groupby("year")[["z2_base", "z2_new"]].sum()
    per = per.assign(dz2=per.z2_new - per.z2_base)
    print(per[per.dz2.abs() > 50].round(0).T.to_string())
    reg = j.groupby("region").dz2.sum().sort_values()
    print("   Δz² by region, largest falls:", Z._fmt(reg.head(6), 0))
    print("   Δz² by region, largest rises:", Z._fmt(reg.tail(6)[::-1], 0))


def attribution(j: pd.DataFrame) -> None:
    """A: the rows behind the < 10 kt band and the largest non-Swedish regional moves."""
    print("\n== A attribution")
    cols = ["label", "year", "c_base", "c_new", "z2_base", "z2_new", "dz2"]
    small = j[j.c_base.abs() < 10.0]
    print(f"   < 10 kt band: Σ Δz² {small.dz2.sum():+,.0f}; largest movers:")
    print(_top(small, 8, cols))
    reg = j[j.region != SWEDEN].groupby("region").dz2.sum()
    for region in reg.abs().sort_values(ascending=False).index[:3]:
        x = j[j.region == region]
        print(f"   {region}: Σ Δz² {x.dz2.sum():+,.0f}; largest movers:")
        print(_top(x, 6, cols))


def main(base: str, new: str, v_base: int, v_new: int) -> None:
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    E = {o: D._edges(o, ei) for o in (base, new)}
    S = {o: P65._sums(o, acc) for o in (base, new)}

    print("=" * 78)
    for o in (base, new):
        const, imp, feeds = L._chain(o)
        print(f"{o}  {L._status(o)}\n   const {const}  imp {imp}\n   feeds {feeds}")

    k = ["constraint_id", "year"]
    j = S[base][k + ["region", "label", "c", "z2"]].merge(S[new][k + ["c", "z2"]], on=k,
                                                         suffixes=("_base", "_new"))
    j["dz2"] = j.z2_new - j.z2_base

    netting(v_base, v_new)
    sweden(E, j, base, new)
    singapore(j, base)
    round_e(E, S, j, base, new)
    stock(E, base, new)
    common_set(j, base)
    attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
