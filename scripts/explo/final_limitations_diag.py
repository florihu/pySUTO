"""
final_limitations_diag.py
=========================
The numbers of the manuscript's limitations (method.md §3.2.8) on one run, read from its
persisted artefacts and its own constraint run (no per-year CSVs, #46):

    M   no-data mine chains, CC: CO→Mining and MF→Concentrate in region-years without
        ICSG mine production (#31), by region; the share of that concentrate exported
    S   CC stock legs with |z| > 5, by region and family (#95, #97); the 1995 stock legs
        by family (#37)
    K   world Σ CP→PSC (CC) against ICSG's reported PSC, 1996–2024 (#98)
    A   the scale-total Σz² of the named regions and their share of the family's

    python scripts/explo/final_limitations_diag.py <opt_run_id> [region ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import build_baci_reexport as R                           # noqa: E402
import issue37_stock_bounds_v6_diag as P37                # noqa: E402
import issue40_signed_stock_diag as D                     # noqa: E402
import issue65_icsg_stock_parse_diag as P65               # noqa: E402
import issue69_aggregate_gap_diag as P69                  # noqa: E402
import issue75_scandinavia_split_diag as P75              # noqa: E402
from pysuto.acc import AccountingStructure                # noqa: E402
from pysuto.config import load_config                     # noqa: E402
from pysuto.param_relation import expand_edge_index       # noqa: E402

Z_BIG   = 5.0
REGIONS = ["Austria", "Kazakhstan", "Russia"]


def mine_chains(e: pd.DataFrame) -> None:
    """M: CC flows of the mine chain where ICSG reports no mine production."""
    _, prod, _ = R.icsg_intake_production(["Concentrate"])
    mine = {(r, t) for (_, r, t), v in prod.items() if v > 0}
    cc = e[e.valuation == "CC"]
    no_mine = ~pd.Series([k in mine for k in zip(cc.region_from, cc.time.astype(str))],
                         index=cc.index)
    dom = cc.region_from == cc.region_to
    for sf, st in (("CO", "Mining"), ("MF", "Concentrate")):
        d = cc[dom & no_mine & (cc.sector_from == sf) & (cc.sector_to == st)]
        g = d.groupby("region_from")[["prior", "post"]].sum()
        g = g.loc[g.post.abs().sort_values(ascending=False).index]
        print(f"\n== M {sf}→{st}, no ICSG mine (kt CC): prior {d.prior.sum():,.1f}, post "
              f"{d.post.sum():,.1f}; {len(g[g.post > 1])} regions above 1 kt CC")
        print(g.head(12).round(1).to_string())
    out = cc[no_mine & (cc.sector_from == "Concentrate")]
    exp = out[out.region_from != out.region_to].post.sum()
    print(f"   Concentrate out of those region-years: {out.post.sum():,.1f} kt CC, exported "
          f"{exp:,.1f} ({exp / out.post.sum():.3f})")


def stock_legs(e: pd.DataFrame, opt: str) -> None:
    """S: CC stock legs above |z| 5 by region and family; the 1995 legs by family."""
    f = P69._with_sigma(P65._family(e), opt).query("valuation == 'CC'")
    big = f[f.z.abs() > Z_BIG]
    g = big.groupby(["region_from", "family"]).agg(n=("z", "size"), post=("post", "sum"),
                                                  prior=("prior", "sum"))
    g = g.loc[g.post.abs().sort_values(ascending=False).index]
    print(f"\n== S CC stock legs with |z| > {Z_BIG:g}: {len(big)}, Σ post {big.post.sum():+,.1f} kt CC")
    print(g.head(15).round(1).to_string())
    y = f[f.time.astype(str) == "1995"].groupby("family").post.sum()
    print(f"   1995 stock legs: Σ {y.sum():+,.1f} kt CC; "
          + "  ".join(f"{k} {v:+,.1f}" for k, v in y.items()))


def world_psc(e: pd.DataFrame, opt: str) -> None:
    """K: world CP→PSC posterior (CC) against ICSG's reported PSC."""
    v = P75._icsg(opt)[1]
    raw = pd.read_csv(P37.ICSG_IN / f"v{v}" / "values.csv")
    rep = raw[raw.Type == "PSC"]
    icsg = rep.groupby(rep.Year.astype(str)).Value.sum().reindex(P37.YEARS_K).fillna(0.0)
    p = e[(e.sector_from == "CP") & (e.sector_to == "PSC") & (e.valuation == "CC")]
    g = p.groupby(p.time.astype(str)).post.sum().reindex(P37.YEARS_K).fillna(0.0)
    print(f"\n== K world PSC (CC) vs ICSG (icsg v{v}), 1996–2024: Σ|post − ICSG| "
          f"{(g - icsg).abs().sum():,.1f} kt CC; Σ post {g.sum():+,.1f}, Σ ICSG {icsg.sum():+,.1f}")


def scale_totals(s: pd.DataFrame, regions: list) -> None:
    """A: scale-total Σz² of the named regions and their share of the family's."""
    tot = s.z2.sum()
    print(f"\n== A scale-total Σz² {tot:,.0f}")
    for r in regions:
        x = s[s.region == r].z2.sum()
        print(f"   {r}: {x:,.0f} ({100 * x / tot:.2f} %)")
    top = s.groupby("region").z2.sum().sort_values(ascending=False).head(10)
    print("   largest: " + "  ".join(f"{k} {v:,.0f}" for k, v in top.items()))


def main(opt: str, regions: list) -> None:
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(P37.ROOT / "acc"))
    e = D._edges(opt, expand_edge_index(acc.edge_index, acc.node_index))
    mine_chains(e)
    stock_legs(e, opt)
    world_psc(e, opt)
    scale_totals(P65._sums(opt, acc), regions)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:] or REGIONS)
