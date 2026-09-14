"""
roundJ_fan_bound_diag.py
========================
Round J (ADR 0024) measurement: the fan's domestic leg bounded by own
production, against the shipped Round-I run. Reads only persisted artifacts.

    python scripts/explo/roundJ_fan_bound_diag.py <opt_base> <opt_new>
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

ROOT = Path("data/proc")
YEARS = list(range(1995, 2025))
EXPORTERS = ["Chile", "Peru", "Indonesia", "Australia", "Brazil"]


def _status(opt):
    s = json.load(open(ROOT / "opt" / opt / "metadata.json"))["stage_summary"]["per_timestep"]
    return {k: v["solver_status"] for k, v in s.items() if v["solver_status"] != "Solved"} or "30/30 Solved"


def _post(opt, ei, mask):
    """Σ prior / Σ posterior per year over masked CC edges."""
    sub = ei[mask]
    rows = []
    for y in YEARS:
        o = np.load(ROOT / "opt" / opt / "timesteps" / f"t_{y}.npz", allow_pickle=True)
        pos = {int(e): i for i, e in enumerate(o["cols_t"])}
        idx = [pos[int(e)] for e in sub[sub.time.astype(str) == str(y)].edge_id if int(e) in pos]
        rows.append(dict(year=y, prior=o["a_prior"][idx].sum(), post=o["F_opt"][idx].sum()))
    return pd.DataFrame(rows).set_index("year")


def main(base, new):
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    print(f"base {base}  {_status(base)}\nnew  {new}  {_status(new)}\n")
    cb, ib = D._lineage(base); cn, i_n = D._lineage(new)
    rb, rn = D._rows(base, cb), D._rows(new, cn)

    print("== headline")
    for lab, c in (("base", rb), ("new", rn)):
        s = c[c.type == "sum"]
        print(f"  {lab}: sum-row z² {s.z2.sum():>12,.0f}   all soft {c[c.type != 'massbal'].z2.sum():>12,.0f}"
              f"   India/Concentrate {c[(c.region=='India')&(c.sector=='Concentrate')].z2.sum():>10,.0f}")

    # the class, fixed on the BASE run's excess population
    fb = np.nan_to_num(np.asarray(np.load(ROOT / "imp" / ib / "agg_result.npz", allow_pickle=True)["f_agg"]))
    dv = D._domestic_vs_production(ei, fb); dv["time"] = dv.time.astype(int)
    pop = dv[(dv.flow == "Concentrate") & (dv.excess > 1)][["region", "time"]]
    print("\n== usetotout_Concentrate z² on the base run's 210 excess region-years (fixed population)")
    for lab, c in (("base", rb), ("new", rn)):
        u = c[(c.variant == "usetotout") & (c.sector == "Concentrate")].merge(pop, on=["region", "time"])
        top = u.groupby("region").z2.sum().sort_values(ascending=False).head(6).round(0).to_dict()
        print(f"  {lab}: {u.z2.sum():,.0f}   top: {top}")
    print("\n== usetotout_Concentrate z² on the OTHER rows (exporters etc.)")
    for lab, c in (("base", rb), ("new", rn)):
        u = c[(c.variant == "usetotout") & (c.sector == "Concentrate")].merge(pop, on=["region", "time"], how="left", indicator=True)
        u = u[u._merge == "left_only"]
        top = u.groupby("region").z2.sum().sort_values(ascending=False).head(6).round(0).to_dict()
        print(f"  {lab}: {u.z2.sum():,.0f}   top: {top}")

    print("\n== sum-row z² by sector, base → new (top movers)")
    sb = rb[rb.type == "sum"].groupby("sector").z2.sum(); sn = rn[rn.type == "sum"].groupby("sector").z2.sum()
    d = pd.DataFrame({"base": sb, "new": sn}).fillna(0); d["delta"] = d.new - d.base
    print(d.reindex(d.delta.abs().sort_values(ascending=False).index).head(8).round(0).to_string())

    print("\n== concentrate trade retention (posterior/prior, CC, cross-region Concentrate→Smelting), pooled 2010–2019")
    cross = (ei.sector_from == "Concentrate") & (ei.sector_to == "Smelting") & (ei.region_from != ei.region_to) & (ei.valuation == "CC")
    for lab, opt in (("base", base), ("new", new)):
        w = _post(opt, ei, cross).loc[2010:2019]
        line = f"  {lab}: world {w.post.sum()/w.prior.sum():.3f} (prior {w.prior.sum():,.0f})"
        for x in EXPORTERS:
            t = _post(opt, ei, cross & (ei.region_from == x)).loc[2010:2019]
            line += f"  {x} {t.post.sum()/max(t.prior.sum(),1e-9):.3f}"
        print(line)
    print("\n== India smelter feed 2015, CC (prior → post): domestic | imports | ICSG use 821")
    for lab, opt in (("base", base), ("new", new)):
        dom = _post(opt, ei, (ei.sector_from == "Concentrate") & (ei.sector_to == "Smelting") & (ei.region_from == "India") & (ei.region_to == "India") & (ei.valuation == "CC")).loc[2015]
        imp = _post(opt, ei, cross & (ei.region_to == "India")).loc[2015]
        print(f"  {lab}: domestic {dom.prior:.0f} → {dom.post:.0f} | imports {imp.prior:.0f} → {imp.post:.0f} | total post {dom.post+imp.post:.0f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
