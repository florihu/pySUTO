"""
india_concentrate_diag.py
=========================
Why `usetotout_Concentrate_India` is the model's largest single cell (~113k z²
pooled), and how large the class it belongs to is. Reads persisted run
artifacts plus the raw ICSG / BACI feeds; nothing is modified.

    python scripts/explo/india_concentrate_diag.py [opt_run_id]

Mechanism (ADR 0002 option D): the domestic leg of an ICSG single-endpoint
usage fan is `max(0, usage − Σ conc·BACI_imports)` — a residual that "needs no
production data". Where BACI imports in copper content fall short of what the
smelters consumed, the shortfall is booked as domestic concentrate the region
never mined; the hard Concentrate-node balance then forces the posterior back
to mine production and the usage-side anchor pays for it every year.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from pysuto.acc import AccountingStructure              # noqa: E402
from pysuto.config import load_config                   # noqa: E402
from pysuto.param_relation import expand_edge_index     # noqa: E402

ROOT = Path("data/proc")
YEARS = list(range(1995, 2025))
CONC_CONCENTRATE = 0.275     # conc nominal the fan builder applies to BACI mass
BACI_INDIA, HS_CONC = 699, 260300
CHAINS = {"Concentrate": "Smelting", "Matte": "CFR", "Anode": "ER"}


def _lineage(opt):
    m = json.load(open(ROOT / "opt" / opt / "metadata.json"))
    const = m["parents"]["upstream"]["const_run_id"]
    imp = json.load(open(ROOT / "const" / const / "metadata.json"))["parents"]["upstream"]["imp_run_id"]
    return const, imp


def _rows(opt, const):
    rm = pd.read_parquet(ROOT / "const" / const / "row_meta.parquet")
    out = []
    for y in YEARS:
        c = pd.read_csv(ROOT / "opt" / opt / "diag" / f"constraints_{y}.csv")
        out.append(c.merge(rm[["constraint_id", "sector", "region", "time"]], on="constraint_id", how="left"))
    c = pd.concat(out)
    c["time"] = c["time"].astype(int)
    c["z2"] = c["z"] ** 2
    return c


def _domestic_vs_production(ei, f):
    """Per (flow, region, year): fanned domestic-leg prior vs own production prior (CC)."""
    cc = ei[ei.valuation == "CC"].assign(f=f[ei[ei.valuation == "CC"].edge_id.values])
    rows = []
    for flow, proc in CHAINS.items():
        dom = cc[(cc.sector_from == flow) & (cc.sector_to == proc) & (cc.region_from == cc.region_to)] \
            .groupby(["region_from", "time"]).f.sum().rename("dom")
        prod = cc[(cc.sector_to == flow) & (cc.entity_from == "Process") & (cc.region_from == cc.region_to)] \
            .groupby(["region_to", "time"]).f.sum().rename("prod")
        prod.index.names = ["region_from", "time"]
        imp = cc[(cc.sector_from == flow) & (cc.sector_to == proc) & (cc.region_from != cc.region_to)] \
            .groupby(["region_to", "time"]).f.sum().rename("imp")
        imp.index.names = ["region_from", "time"]
        d = pd.concat([dom, prod, imp], axis=1).fillna(0.0).reset_index().rename(columns={"region_from": "region"})
        d["excess"] = (d["dom"] - d["prod"]).clip(lower=0.0)
        d["flow"] = flow
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main(opt):
    const, imp = _lineage(opt)
    print(f"opt {opt}\nconst {const}\nimp {imp}\n")
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    f = np.nan_to_num(np.asarray(np.load(ROOT / "imp" / imp / "agg_result.npz", allow_pickle=True)["f_agg"]))
    c = _rows(opt, const)

    # 1. the cell
    ind = c[(c.region == "India") & (c.sector == "Concentrate")]
    print("== 1. India/Concentrate rows, z² pooled over 30 y")
    print(ind.groupby(["variant", "valuation"]).agg(z2=("z2", "sum"), n=("z2", "size")).round(0).to_string())
    print("India, all sectors:", c[c.region == "India"].groupby("sector").z2.sum().sort_values(ascending=False).head(3).round(0).to_dict())
    u = ind[(ind.variant == "usetotout") & (ind.valuation == "CC")].set_index("time")
    print("\nusetotout_Concentrate_India (CC): anchor c vs posterior sum, selected years")
    print(u.loc[[1998, 2003, 2008, 2012, 2015, 2017, 2019, 2020, 2022, 2024], ["c", "lhs_post", "z"]].round(1).to_string())

    # 2. the raw pair behind the anchor
    # v5 is the last GROSS BACI version; `current/` is netted of re-exports
    # from ADR 0027 onwards, and this block reports gross imports.
    b = pd.read_csv("data/input/feed/baci/v5/values.csv")
    b = b[b.Sector == HS_CONC]
    i = pd.read_csv("data/input/feed/icsg/current/values.csv")
    i = i[i.Region == "India"]
    d = pd.concat([
        b[b.Region_to == BACI_INDIA].groupby("Year").Value.sum().rename("baci_imp_gross"),
        i[(i.Process == "Smelting") & (i.Type == "Concentrate")].groupby("Year").Value.sum().rename("icsg_smelt_use_cc"),
        i[i.Process == "Mining"].groupby("Year").Value.sum().rename("icsg_mine_cc"),
    ], axis=1).fillna(0.0)
    d["imp_cc@nominal"] = d.baci_imp_gross * CONC_CONCENTRATE
    d["residual_domestic"] = (d.icsg_smelt_use_cc - d["imp_cc@nominal"]).clip(lower=0.0)
    d["grade_needed"] = (d.icsg_smelt_use_cc - d.icsg_mine_cc) / d.baci_imp_gross.replace(0, np.nan)
    print("\n== 2. India: ICSG smelter use vs BACI imports at the 0.275 nominal (kt)")
    print(d.loc[[2000, 2005, 2010, 2015, 2017, 2018, 2020, 2022, 2024]].round(2).to_string())
    print("grade the imports would need to carry, 2003–2018: %.2f–%.2f (concentrate is 0.20–0.35)"
          % (d.loc[2003:2018, "grade_needed"].min(), d.loc[2003:2018, "grade_needed"].max()))

    # 3. the class
    dv = _domestic_vs_production(ei, f)
    print("\n== 3. Class: fanned domestic leg prior > own production prior (CC kt, 30 y pooled)")
    print(dv.groupby("flow").agg(region_years=("excess", lambda s: int((s > 1).sum())), excess=("excess", "sum"),
                                 dom=("dom", "sum"), prod=("prod", "sum")).round(0).to_string())
    top = dv[dv.excess > 1].groupby(["flow", "region"]).agg(years=("excess", "size"), excess=("excess", "sum"),
                                                            dom=("dom", "sum"), prod=("prod", "sum"), imp=("imp", "sum"))
    print(top.sort_values("excess", ascending=False).head(10).round(0).to_string())

    # 4. what the class costs on the run, and which side is short per region
    uu = c[(c.variant == "usetotout") & (c.sector.isin(list(CHAINS)))].merge(
        dv[["flow", "region", "time", "excess"]].rename(columns={"flow": "sector"}).assign(time=lambda x: x.time.astype(int)),
        on=["sector", "region", "time"], how="left").fillna({"excess": 0.0})
    uu["has_excess"] = uu.excess > 1
    print("\n== 4. usetotout z² by sector × (domestic leg exceeds production)")
    print(uu.groupby(["sector", "has_excess"]).agg(rows=("z2", "size"), z2=("z2", "sum")).round(0).to_string())
    print("total sum-row z² on the run: %.0f" % c[c.type == "sum"].z2.sum())
    reg = pd.read_excel("data/input/feed/baci/v5/conc.xlsx", sheet_name="region_map")
    codes = reg.groupby("root_region").src_region.apply(list).to_dict()
    ii = pd.read_csv("data/input/feed/icsg/current/values.csv")
    print("\n== 5. Concentrate importers with excess: grade the BACI imports would need, (usage − mine)/gross,"
          " median over that region's excess years")
    ex_years = dv[(dv.flow == "Concentrate") & (dv.excess > 1)].groupby("region").time.apply(lambda s: sorted(int(t) for t in s))
    for r in ["India", "Japan", "Philippines", "China", "Spain", "Brazil", "United States", "Uzbekistan", "Poland"]:
        yrs = ex_years.get(r, [])
        if not yrs:
            continue
        jj = ii[ii.Region == r]
        use = jj[(jj.Process == "Smelting") & (jj.Type == "Concentrate")].groupby("Year").Value.sum()
        mine = jj[jj.Process == "Mining"].groupby("Year").Value.sum()
        g = b[b.Region_to.isin(codes.get(r, []))].groupby("Year").Value.sum()
        t = pd.concat([use.rename("u"), mine.rename("m"), g.rename("g")], axis=1).reindex(yrs).fillna(0)
        gn = ((t.u - t.m) / t.g.replace(0, np.nan)).median()
        print(f"   {r:14s} {len(yrs):2d} y ({yrs[0]}–{yrs[-1]})  needed {gn:5.2f}   BACI gross {t.g.median():7.0f} kt/y, "
              f"ICSG use {t.u.median():6.0f}, mine {t.m.median():6.0f}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "opt_20260910_114200_7f4040d9")
