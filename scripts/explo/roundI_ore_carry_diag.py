"""
roundI_ore_carry_diag.py
========================
Round I (ADR 0023) measurement: what the ore-mass chain does in 2023–24 once
the in-situ grade is carried past the crude_ore feed's last year, against the
shipped Round-H run. Reads only persisted run artifacts.

    python scripts/explo/roundI_ore_carry_diag.py <opt_base> <opt_new>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from pysuto.acc import AccountingStructure               # noqa: E402
from pysuto.config import load_config                    # noqa: E402
from pysuto.param_relation import expand_edge_index      # noqa: E402

ROOT = Path("data/proc")
YEARS = list(range(1995, 2025))


def _const_of(opt):
    return json.load(open(ROOT / "opt" / opt / "metadata.json"))["parents"]["upstream"]["const_run_id"]


def _rows(opt):
    rm = pd.read_parquet(ROOT / "const" / _const_of(opt) / "row_meta.parquet")
    out = []
    for y in YEARS:
        c = pd.read_csv(ROOT / "opt" / opt / "diag" / f"constraints_{y}.csv")
        c = c.merge(rm[["constraint_id", "sector", "region", "time"]], on="constraint_id", how="left")
        out.append(c)
    c = pd.concat(out)
    c["time"] = c["time"].astype(int)
    c["z2"] = c["z"] ** 2
    return c


def _ore_mass(opt, ei):
    co = ei[(ei.sector_from == "CO") & (ei.sector_to == "Mining") & (ei.valuation == "Mass")]
    rows = []
    for y in YEARS:
        o = np.load(ROOT / "opt" / opt / "timesteps" / f"t_{y}.npz", allow_pickle=True)
        pos = {int(c): i for i, c in enumerate(o["cols_t"])}
        sub = co[co.time.astype(str) == str(y)]
        idx = [pos[int(e)] for e in sub.edge_id if int(e) in pos]
        rows.append(dict(year=y, prior=o["a_prior"][idx].sum(), post=o["F_opt"][idx].sum(),
                         chile_post=o["F_opt"][[pos[int(e)] for e in sub[sub.region_from == "Chile"].edge_id]].sum()))
    return pd.DataFrame(rows).set_index("year")


def main(base, new):
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    print(f"base {base}\nnew  {new}\n")

    print("== world CO→Mining ore Mass (kt), prior / posterior")
    a, b = _ore_mass(base, ei), _ore_mass(new, ei)
    t = pd.concat({"base": a, "new": b}, axis=1).loc[2020:2024].round(0)
    print(t.to_string(), "\n")

    ra, rb = _rows(base), _rows(new)
    soft = lambda c: c[c.type != "massbal"]
    print("== soft z² by year (sum rows | all soft)")
    ya = soft(ra).groupby("time").z2.sum(); yb = soft(rb).groupby("time").z2.sum()
    sa = ra[ra.type == "sum"].groupby("time").z2.sum(); sb = rb[rb.type == "sum"].groupby("time").z2.sum()
    t = pd.DataFrame({"sum_base": sa, "sum_new": sb, "soft_base": ya, "soft_new": yb}).loc[2020:2024].round(0)
    print(t.to_string())
    print(f"TOTAL sum-row z²  base {sa.sum():,.0f}  new {sb.sum():,.0f}  ({(sb.sum()/sa.sum()-1)*100:+.1f} %)")
    print(f"TOTAL soft z²     base {ya.sum():,.0f}  new {yb.sum():,.0f}  ({(yb.sum()/ya.sum()-1)*100:+.1f} %)")
    print(f"1995–2022 sum-row z²  base {sa.loc[:2022].sum():,.0f}  new {sb.loc[:2022].sum():,.0f}\n")

    print("== Mining/Ore Mass anchor rows and ore-grade ratio rows per year")
    for lab, c in (("base", ra), ("new", rb)):
        m = c[(c.valuation == "Mass") & (c.sector.isin(["Mining", "Ore"])) & (c.type == "sum")]
        g = c[c.constraint_id.str.startswith("ratio_oregrade")]
        print(lab, "sum rows 2022/23/24:", [int((m.time == y).sum()) for y in (2022, 2023, 2024)],
              " oregrade ratio rows:", [int((g.time == y).sum()) for y in (2022, 2023, 2024)])

    print("\n== 2023–24 ore-chain Mass sum rows, largest |z| (new)")
    m = rb[(rb.valuation == "Mass") & (rb.sector.isin(["Mining", "Ore", "Concentrate"])) & (rb.type == "sum") & (rb.time >= 2023)]
    print(m.sort_values("z_abs", ascending=False)[["constraint_id", "c", "lhs_post", "z"]].head(8).round(1).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
