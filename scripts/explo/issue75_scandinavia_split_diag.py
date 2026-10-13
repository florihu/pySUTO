"""
issue75_scandinavia_split_diag.py
=================================
#75 measurement of #73 and #74: ICSG's "Scandinavia" split to Sweden for 1995–2003 by
the 2004 handover year's shares, and its 2004 duplicate dropped (ADR 0032 amendment).
Reads persisted run artefacts, the constraint runs' σ_a and two netted BACI versions'
ledgers only, builds the sum-row table in memory (no per-year CSVs, #46) and prints
#75's pre-registered items for <new> against <base>.  The sections shared with #69 are
imported from ``issue69_aggregate_gap_diag``:

    N   netting: per hub-year T and T_applied, baci v<base> ledger → v<new> ledger
    W   Sweden: Mining→Ore CC posterior/prior, its ore routes with σ, its cathode leg
        groups, the ROW → Sweden MESC 2001 leg's z, and its common-set sum-row z²
    R   ROW 2004: every scale-total row whose c moves, and ROW's 2004 sum-row z²; ROW's
        refined cathode supply (CP→Cathode) and its domestic semis leg, 1995–2004
    E   the Round-E items (ADR 0025)
    S   stock legs: net per family and per region-year, each family's largest-growth
        year by region, and every stock leg whose |z| exceeds 5 in the new run only
    C   common-set sum-row z², and the sum rows only one run has, by region
    A   attribution of the largest moves

    python scripts/explo/issue75_scandinavia_split_diag.py <opt_base> <opt_new> \\
                                                           <baci_v_base> <baci_v_new>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
from pysuto.acc import AccountingStructure              # noqa: E402
from pysuto.config import load_config                   # noqa: E402
from pysuto.param_relation import expand_edge_index     # noqa: E402
import issue40_signed_stock_diag as D                   # noqa: E402
import issue65_icsg_stock_parse_diag as P65             # noqa: E402
import issue69_aggregate_gap_diag as P69                # noqa: E402
import roundL_reexport_diag as L                        # noqa: E402

ROOT  = Path("data/proc")
Z_BIG = 5.0


def row_2004(j: pd.DataFrame, base: str) -> None:
    """R: ROW's 2004 scale totals whose c moves, and ROW's 2004 sum-row z²."""
    r = j[(j.region == "ROW") & (j.year == "2004")].copy()
    r["dc"] = r.c_new - r.c_base
    print(f"\n== R ROW 2004: {len(r)} common sum rows, z² {r.z2_base.sum():,.0f} → "
          f"{r.z2_new.sum():,.0f} ({r.dz2.sum():+,.0f} vs {base})")
    moved = r[r.dc.abs() > 0.5].sort_values("dc")
    print("   scale totals whose c moves by more than 0.5:")
    print(moved[["label", "c_base", "c_new", "dc", "z2_base", "z2_new"]].round(1)
          .to_string(index=False))


def row_cathode(E: dict, base: str, new: str) -> None:
    """R (cathode): ROW's refined cathode supply and its domestic semis leg, CC."""
    yrs = [str(y) for y in range(1995, 2005)]
    for o in (base, new):
        e = E[o]
        row = (e.region_from == "ROW") & (e.region_to == "ROW") & (e.valuation == "CC")
        for name, frm, to in (("CP→Cathode", "CP", "Cathode"),
                              ("Cathode→FU", "Cathode", "FU")):
            m = row & (e.sector_from == frm) & (e.sector_to == to)
            g = e[m].groupby("time")[["prior", "post"]].sum().reindex(yrs)
            print(f"   ROW {name} {o[-8:]}, prior / post: " + "  ".join(
                f"{t}: {g.prior[t]:.0f}/{g.post[t]:.0f}" for t in yrs)
                + f"  | Σ(prior − post) {(g.prior - g.post).sum():,.0f}")


def stock_breakdown(E: dict, base: str, new: str) -> None:
    """S (by region): each family's year of largest |net| growth, split by region."""
    fam = {o: P65._family(E[o]).query("valuation == 'CC'") for o in (base, new)}
    for f in ("MESC", "USC", "PSC"):
        net = pd.concat([fam[o][fam[o].family == f].groupby("time").delta.sum().rename(o)
                         for o in (base, new)], axis=1).fillna(0.0)
        t = (net[new].abs() - net[base].abs()).idxmax()
        r = pd.concat([fam[o][(fam[o].family == f) & (fam[o].time == t)]
                       .groupby("region_to").delta.sum().rename(o) for o in (base, new)],
                      axis=1).fillna(0.0)
        r["d"] = r[new] - r[base]
        r = r.reindex(r.d.abs().sort_values(ascending=False).index)
        print(f"   {f} {t}: net {net.loc[t, base]:.1f} → {net.loc[t, new]:.1f}; by region "
              + "  ".join(f"{k} {v:+.1f}" for k, v in r.d.head(6).items())
              + f"  rest {r.d.iloc[6:].sum():+.1f} over {len(r) - 6} regions")


def stock_z(E: dict, base: str, new: str) -> None:
    """S (z): stock legs whose |z| exceeds Z_BIG in the new run but not in the base."""
    z = {}
    for o in (base, new):
        f = P69._with_sigma(P65._family(E[o]), o)
        z[o] = f.set_index(["edge_id"])[["family", "time", "region_from", "region_to",
                                         "valuation", "prior", "post", "z"]]
    m = z[new].join(z[base][["prior", "post", "z"]], rsuffix="_base", how="left")
    grew = m[(m.z.abs() > Z_BIG) & ~(m.z_base.abs() > Z_BIG)]
    n_base, n_new = (int((z[o].z.abs() > Z_BIG).sum()) for o in (base, new))
    print(f"\n== S (z) stock legs with |z| > {Z_BIG:g}: base {n_base}, new {n_new}; "
          f"new above it that were not: {len(grew)}")
    if len(grew):
        print(grew.sort_values("z", key=abs, ascending=False).head(12).round(2)
              .to_string())


def one_run_only(S: dict, base: str, new: str) -> None:
    """C (rows): sum rows only one run has, by region, with their z²."""
    k = ["constraint_id", "year"]
    m = S[base][k + ["region", "z2"]].merge(S[new][k + ["region", "z2"]], on=k,
                                             how="outer", indicator=True)
    for side, col, run in (("left_only", "z2_x", base), ("right_only", "z2_y", new)):
        x = m[m._merge == side]
        reg = "region_x" if side == "left_only" else "region_y"
        print(f"   rows only in {run}: {len(x)}, z² {x[col].sum():,.0f}; by region "
              f"{x.groupby(reg)[col].agg(['size', 'sum']).round(0).to_dict('index')}")


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

    P69.netting(v_base, v_new)
    P69.sweden(E, j, base, new)
    row_2004(j, base)
    row_cathode(E, base, new)
    P69.round_e(E, S, j, base, new)
    P69.stock(E, base, new)
    stock_breakdown(E, base, new)
    stock_z(E, base, new)
    P69.common_set(j, base)
    one_run_only(S, base, new)
    P69.attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
