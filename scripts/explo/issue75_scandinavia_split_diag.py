"""
issue75_scandinavia_split_diag.py
=================================
#75 measurement of #73 and #74: ICSG's "Scandinavia" split to Sweden for 1995–2003 by
the 2004 handover year's shares, and its 2004 duplicate dropped (ADR 0032 amendment).
Reads persisted run artefacts, the feed runs' F_root, the constraint runs' σ_a, the two
runs' ICSG input versions and two netted BACI versions' ledgers only, builds the sum-row
table in memory (no per-year CSVs, #46) and prints #75's pre-registered items for <new>
against <base>.  The sections shared with #69 are imported from
``issue69_aggregate_gap_diag``:

    I   inputs: the ICSG version diff by region, and every row the split dropped
    N   netting: per hub-year T and T_applied, baci v<base> ledger → v<new> ledger
    W   Sweden: Mining→Ore CC posterior/prior, its ore routes with σ, its cathode leg
        groups, the ROW → Sweden MESC 2001 leg's z, and its common-set sum-row z²
    R   ROW: its 2004 scale totals whose c moves and its 2004 sum-row z²; its refined
        cathode supply; its semis use as the ICSG feed fans it and as the model carries
        it, with the domestic leg's feed value; its Δz² by sector
    E   the Round-E items (ADR 0025)
    S   stock legs: net per family and per region-year; each family's largest-growth
        year by region, with its legs' Σz²; every leg whose |z| exceeds 5 in new only
    C   common-set sum-row z², and the sum rows only one run has, by region
    A   attribution of the largest moves

    python scripts/explo/issue75_scandinavia_split_diag.py <opt_base> <opt_new> \\
                                                           <baci_v_base> <baci_v_new>
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
import issue65_icsg_stock_parse_diag as P65             # noqa: E402
import issue69_aggregate_gap_diag as P69                # noqa: E402
import roundL_reexport_diag as L                        # noqa: E402

ROOT    = Path("data/proc")
ICSG_IN = Path("data/input/feed/icsg")
Z_BIG   = 5.0
YRS     = [str(y) for y in range(1995, 2005)]
KEY     = ["Region", "Type", "Process", "Supply_Use", "Year"]


def _icsg(opt: str) -> tuple:
    """The run's ICSG feed run id and the ICSG input version it was built from."""
    feed = L._chain(opt)[2]["icsg"]
    meta = json.loads((ROOT / "feed" / "registry" / feed / "metadata.json").read_text())
    return feed, meta["parents"]["inputs"]["feed"]["version"]


def inputs(base: str, new: str) -> None:
    """I: the ICSG input diff between the two runs, and the rows the split dropped."""
    vb, vn = _icsg(base)[1], _icsg(new)[1]
    a = pd.read_csv(ICSG_IN / f"v{vb}" / "values.csv")
    b = pd.read_csv(ICSG_IN / f"v{vn}" / "values.csv")
    m = a.merge(b, on=KEY, how="outer", suffixes=("_base", "_new"), indicator=True)
    d = m[(m._merge != "both") | ((m.Value_base - m.Value_new).abs() > 1e-9)]
    print(f"\n== I icsg v{vb} → v{vn}: {len(d)} rows differ; "
          f"{d.groupby(['Region', '_merge']).size().to_dict()}")
    gone = d[d._merge == "left_only"]
    print("   rows dropped:")
    print(gone[["Region", "Process", "Type", "Supply_Use", "Year", "Value_base"]]
          .to_string(index=False))


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


def _years(e: pd.DataFrame) -> pd.DataFrame:
    return e.groupby("time")[["prior", "post"]].sum().reindex(YRS)


def row_cathode(E: dict, base: str, new: str) -> None:
    """R: ROW's refined cathode supply, and its semis use as fed and as carried, CC."""
    for o in (base, new):
        e = E[o][E[o].valuation == "CC"]
        froot = np.load(ROOT / "feed" / "registry" / _icsg(o)[0] / "result.npz")["F_root"]
        e = e.assign(feed=froot[e.edge_id.to_numpy()])
        own = (e.region_from == "ROW") & (e.region_to == "ROW")
        semis = ((e.sector_from == "Cathode") & (e.sector_to == "FU")
                 & (e.region_to == "ROW"))
        g = _years(e[own & (e.sector_from == "CP") & (e.sector_to == "Cathode")])
        print(f"   ROW CP→Cathode {o[-8:]}, prior / post: " + "  ".join(
            f"{t}: {g.prior[t]:.0f}/{g.post[t]:.0f}" for t in YRS)
            + f"  | Σ(prior − post) {(g.prior - g.post).sum():,.0f}")
        fed = e[semis].groupby("time").feed.sum().reindex(YRS)
        dom = e[semis & own].set_index("time").reindex(YRS)
        tot = _years(e[semis])
        print(f"   ROW semis use {o[-8:]}, ICSG fan / model prior / model post; domestic "
              f"leg feed / prior / post; 1995–2004 model above the fan: prior "
              f"{(tot.prior - fed).sum():,.0f}, post {(tot.post - fed).sum():,.0f}")
        print("     " + "  ".join(
            f"{t}: {fed[t]:.0f}/{tot.prior[t]:.0f}/{tot.post[t]:.0f}; "
            f"{dom.feed[t]:.0f}/{dom.prior[t]:.0f}/{dom.post[t]:.0f}" for t in YRS))


def row_by_sector(j: pd.DataFrame, base: str) -> None:
    """R: ROW's common-set Δz² by the sector its sum row totals."""
    r = j[j.region == "ROW"]
    sector = r.label.str.split(" ").str[1].str.split("·").str[0]
    by = r.groupby(sector).dz2.sum().sort_values()
    print(f"   ROW Δz² by sector vs {base} (Σ {r.dz2.sum():+,.0f}): "
          + "  ".join(f"{k} {v:+,.0f}" for k, v in by.items()))


def stock_breakdown(E: dict, base: str, new: str) -> None:
    """S: each family's year of largest |net| growth, by region, with its legs' z²."""
    fam = {o: P69._with_sigma(P65._family(E[o]).query("valuation == 'CC'"), o)
           for o in (base, new)}
    for f in ("MESC", "USC", "PSC"):
        net = pd.concat([fam[o][fam[o].family == f].groupby("time").delta.sum().rename(o)
                         for o in (base, new)], axis=1).fillna(0.0)
        t = (net[new].abs() - net[base].abs()).idxmax()
        legs = {o: fam[o][(fam[o].family == f) & (fam[o].time == t)] for o in (base, new)}
        r = pd.concat([legs[o].groupby("region_to").delta.sum().rename(o)
                       for o in (base, new)], axis=1).fillna(0.0)
        r["d"] = r[new] - r[base]
        r = r.reindex(r.d.abs().sort_values(ascending=False).index)
        print(f"   {f} {t}: net {net.loc[t, base]:.1f} → {net.loc[t, new]:.1f}; by region "
              + "  ".join(f"{k} {v:+.1f}" for k, v in r.d.head(6).items())
              + f"  rest {r.d.iloc[6:].sum():+.1f} over {len(r) - 6} regions; legs' Σz² "
              + " → ".join(f"{(legs[o].z ** 2).sum():,.1f}" for o in (base, new))
              + ", legs above |z| 3 "
              + " → ".join(f"{int((legs[o].z.abs() > 3).sum())}" for o in (base, new)))


def stock_z(E: dict, base: str, new: str) -> None:
    """S: stock legs whose |z| exceeds Z_BIG in the new run but not in the base."""
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
    """C: sum rows only one run has, by region, with their z²."""
    k = ["constraint_id", "year"]
    m = S[base][k + ["region", "z2"]].merge(S[new][k + ["region", "z2"]], on=k,
                                             how="outer", indicator=True)
    for side, col, reg, run in (("left_only", "z2_x", "region_x", base),
                                ("right_only", "z2_y", "region_y", new)):
        x = m[m._merge == side]
        print(f"   rows only in {run}: {len(x)}, z² {x[col].sum():,.0f}; by region "
              f"{x.groupby(reg)[col].agg(['size', 'sum']).round(0).to_dict('index')}")


def main(base: str, new: str, v_base: int, v_new: int) -> None:
    E, S, j = P69.load(base, new)
    inputs(base, new)
    P69.netting(v_base, v_new)
    P69.sweden(E, j, base, new)
    row_2004(j, base)
    row_cathode(E, base, new)
    row_by_sector(j, base)
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
