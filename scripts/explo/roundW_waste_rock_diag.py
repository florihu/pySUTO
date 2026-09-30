"""
roundW_waste_rock_diag.py
=========================
Round W measurement (issue #28; ADR 0038 and its successors): waste rock as a
satellite mass flow, the data-aware no-data anchor, and enforced lower bounds.
Reads persisted run artifacts only.

    python scripts/explo/roundW_waste_rock_diag.py <opt_base> <opt_pred> <opt_new>

``opt_base`` is the study's base run (common-set z² reference), ``opt_pred`` the
round's predecessor, ``opt_new`` the round's run. Needs the per-year tables
(``scripts/solver_stats.py <opt> --years all``) for every run.
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
from pysuto.input_registry import InputRegistries        # noqa: E402
from pysuto.param_relation import expand_edge_index      # noqa: E402
import india_concentrate_diag as D                       # noqa: E402
import roundL_reexport_diag as L                         # noqa: E402

ROOT = Path("data/proc")
MINE = ["Mining->WR", "Ore->MF", "Ore->SXEW", "MF->Tailings", "SXEW->Tailings"]
CHAIN = ["CO->Mining", "Mining->Ore", "Ore->MF", "Ore->SXEW", "MF->Concentrate",
         "SXEW->Cathode", "MF->Tailings", "SXEW->Tailings", "Mining->WR"]


def _common_set(runs):
    S = {lab: L._sums(r) for lab, r in runs}
    common = set.intersection(*(set(s.constraint_id) for s in S.values()))
    return {lab: float(S[lab][S[lab].constraint_id.isin(common)].z2.sum()) for lab, _ in runs}, len(common)


def _pair_totals(e, pair, val):
    m = (e.pair == pair) & (e.valuation == val)
    return float(e[m].prior.sum()), float(e[m].post.sum())


def main(base, pred, new):
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    l_struct = acc.bounds.l
    runs = [("base", base), ("pred", pred), ("new", new)]

    print("=" * 78)
    for lab, r in runs:
        const, imp, _ = L._chain(r)
        cfg = json.load(open(ROOT / "const" / const / "metadata.json")).get("stage_config", {})
        print(f"{lab:5s} {r}  {L._status(r)}\n      const {const}  imp {imp}  "
              f"massbal_mask={cfg.get('massbal_mask')}")

    z, n_common = _common_set(runs)
    print(f"\n== common-set sum-row z² ({n_common:,} rows present in all three runs)")
    for lab, _ in runs:
        print(f"   {lab:5s} {z[lab]:,.0f}   vs base {(z[lab] / z['base'] - 1) * 100:+.1f} %   "
              f"vs pred {(z[lab] / z['pred'] - 1) * 100:+.1f} %")

    E = {}
    for lab, r in runs[1:]:
        e = L._edges(r, ei)
        e["pair"] = e.sector_from + "->" + e.sector_to
        e["l"] = l_struct[e.edge_id.values]
        E[lab] = e

    print("\n== world retention (posterior/prior, 1995-2024)")
    for lab in ("pred", "new"):
        print(f"   {lab:5s} " + "  ".join(f"{k} {v:.3f}" for k, v in L._world_retention(E[lab]).items()))

    print("\n== waste rock (Mining->WR, Mass, world 1995-2024)")
    for lab in ("pred", "new"):
        e = E[lab]
        wr_p, wr = _pair_totals(e, "Mining->WR", "Mass")
        _, ore = _pair_totals(e, "Mining->Ore", "Mass")
        co_p, co = _pair_totals(e, "CO->Mining", "Mass")
        w = e[(e.pair == "Mining->WR") & (e.valuation == "Mass")]
        print(f"   {lab:5s} WR prior {wr_p:,.0f}  post {wr:,.0f} kt  (WR/Ore {wr / ore:.3f})  "
              f"edges < -1 kt {int((w.post < -1).sum())}  min {w.post.min():,.0f}")
        print(f"         CO post {co:,.0f}  Ore post {ore:,.0f}  CO - Ore {co - ore:,.0f} kt")
        mm = e[e.valuation == "Mass"]
        g = mm[mm.pair.isin(["CO->Mining", "Mining->Ore"])].pivot_table(
            index=["region_from", "time"], columns="pair", values="post", aggfunc="sum").fillna(0.0)
        gap = (g["CO->Mining"] - g["Mining->Ore"]).abs()
        print(f"         region-years with |CO - Ore| > 1 kt: {int((gap > 1).sum())}")
        mx = g.loc[("Mexico", "2019")] if ("Mexico", "2019") in g.index else None
        if mx is not None:
            print(f"         Mexico 2019: CO {mx['CO->Mining']:,.0f}  Ore {mx['Mining->Ore']:,.0f}")

    print("\n== lower-bound violations (post < structural l - 1 kt), by pair and valuation")
    for lab in ("pred", "new"):
        e = E[lab]
        v = e[e.post < e.l - 1.0]
        t = v.groupby(["pair", "valuation"]).post.agg(["size", "sum"]).round(0)
        print(f"   {lab}: {len(v):,} edges")
        if len(t):
            print(t.to_string())

    print("\n== extraction chain outside the ore-grade region-years (non-mining), world 1995-2024")
    og = pd.read_csv(InputRegistries.initialize(root="data/input").params.resolve("ore_grade")
                     / "params.csv", sep=None, engine="python", dtype=str)
    og = og[og.param_id.str.startswith("oregradecosupply")]
    ry = set(zip(og.a_region_from, og.a_time.astype(str)))
    print(f"   ore-grade region-years: {len(ry):,}")
    for lab in ("pred", "new"):
        e = E[lab]
        ch = e[e.pair.isin(CHAIN)]
        out = ~np.fromiter((k in ry for k in zip(ch.region_from.astype(str), ch.time.astype(str))),
                           dtype=bool, count=len(ch))
        t = ch[out].groupby(["pair", "valuation"]).post.agg(
            post_sum="sum", n_gt_1kt=lambda x: int((x.abs() > 1).sum()))
        print(f"   {lab}:")
        print(t[t.n_gt_1kt > 0].round(0).to_string())

    print("\n== zero-pin exceedances (pinned u = 0, |post| > 1 kt), by pair and valuation")
    for lab, r in runs[1:]:
        const, _, _ = L._chain(r)
        u = np.load(ROOT / "const" / const / "u.npy")
        e = E[lab]
        x = e[(u[e.edge_id.values] == 0.0) & (e.post.abs() > 1.0)]
        t = x.groupby(["pair", "valuation"]).post.agg(["size", "sum"]).sort_values("sum", ascending=False)
        print(f"   {lab}: {len(x):,} edges, {x.post.abs().sum():,.0f} kt |post| in total")
        print(t.head(8).round(0).to_string())

    print("\n== stripping-ratio rows and CO/Ore grade rows (soft), z² over 30 years")
    for lab, r in runs[1:]:
        const, _ = D._lineage(r)
        c = D._rows(r, const)
        s = c[c.variant.astype(str) == "strippingratio"]
        og = c[c.variant.astype(str).str.startswith("oregrade") & (c.cone_type == "free")]
        print(f"   {lab:5s} strippingratio rows {len(s):,}  z² {s.z2.sum():,.0f}  max|z| {s.z.abs().max():.2f}   "
              f"ore-grade rows z² {og.z2.sum():,.0f}")

    print("\n== copper content: largest world CC pair changes pred -> new (kt, 1995-2024)")
    cc = pd.concat([E[lab][E[lab].valuation == "CC"].groupby("pair").post.sum().rename(lab)
                    for lab in ("pred", "new")], axis=1)
    cc["delta"] = cc.new - cc.pred
    cc["delta_%"] = cc.delta / cc.pred.abs().replace(0, np.nan) * 100
    print(cc.reindex(cc.delta.abs().sort_values(ascending=False).index).head(8).round(1).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
