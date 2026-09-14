"""
roundK_sigma_floor_diag.py
==========================
Round K (ADR 0025) measurement: scale totals carry the feeds' reporting
resolution as an absolute σ floor. A repricing round, so the metric is
reported by anchor-size band and every physical check the stop rule names
is printed next to it. Reads persisted run artifacts only.

    python scripts/explo/roundK_sigma_floor_diag.py <opt_base> <opt_new>

Stop rule (plan 2026-09-14): ship only if (a) world CO Mass / Concentrate CC /
Cathode CC retention move < 0.01, (b) nodes with |c| < 18 kt do not deflate
(median posterior/prior within ±0.05 of base), (c) exporter retention
Chile/Peru/Indonesia unchanged, (d) the gain sits in the < 10 kt bands and the
> 100 kt band is unchanged within 1 %.
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
BANDS = [0, 1, 10, 100, 1000, 1e12]
LABELS = ["<1", "1–10", "10–100", "100–1000", ">1000"]
SMALL_KT = 18.0            # cv 0.017 · 18 kt ≈ 0.3 kt: rows the floor reaches
EXPORTERS = ["Chile", "Peru", "Indonesia"]


def _status(opt):
    s = json.load(open(ROOT / "opt" / opt / "metadata.json"))["stage_summary"]["per_timestep"]
    bad = {k: v["solver_status"] for k, v in s.items() if v["solver_status"] != "Solved"}
    return bad or "30/30 Solved"


def _sums(opt):
    const, _ = D._lineage(opt)
    c = D._rows(opt, const)
    s = c[c.type == "sum"].copy()
    s["ac"] = s.c.abs()
    s["band"] = pd.cut(s.ac, BANDS, labels=LABELS, include_lowest=True)
    return s


def _edges(opt, ei):
    """Per-edge prior and posterior over all years (CC and Mass)."""
    rows = []
    for y in YEARS:
        o = np.load(ROOT / "opt" / opt / "timesteps" / f"t_{y}.npz", allow_pickle=True)
        rows.append(pd.DataFrame({"edge_id": np.asarray(o["cols_t"], dtype=int),
                                  "prior": o["a_prior"], "post": o["F_opt"]}))
    e = pd.concat(rows, ignore_index=True)
    return e.merge(ei[["edge_id", "sector_from", "region_from", "sector_to", "region_to",
                       "entity_from", "entity_to", "valuation", "time"]], on="edge_id")


def _world_retention(e):
    def ret(mask):
        sub = e[mask]
        return sub.post.sum() / sub.prior.sum()
    return {
        "CO Mass": ret((e.sector_from == "CO") & (e.valuation == "Mass")),
        "Concentrate CC": ret((e.sector_from == "MF") & (e.sector_to == "Concentrate") & (e.valuation == "CC")),
        "Cathode CC": ret((e.sector_from == "CP") & (e.sector_to == "Cathode") & (e.valuation == "CC")),
    }


def main(base, new):
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    print(f"base {base}  {_status(base)}\nnew  {new}  {_status(new)}\n")

    sb, sn = _sums(base), _sums(new)
    print("== sum-row z² by anchor band (|c|, kt)")
    t = pd.DataFrame({"rows": sb.groupby("band", observed=False).size(),
                      "base": sb.groupby("band", observed=False).z2.sum(),
                      "new": sn.groupby("band", observed=False).z2.sum()})
    t["Δ%"] = (t.new / t.base - 1) * 100
    print(t.round(0).to_string())
    print(f"TOTAL  base {sb.z2.sum():,.0f}  new {sn.z2.sum():,.0f}  ({(sn.z2.sum()/sb.z2.sum()-1)*100:+.1f} %)")
    print(f"σ_c on new run: rows lifted ≥5 % vs base: "
          f"{int((sn.set_index('constraint_id').sigma_c.reindex(sb.constraint_id).values > 1.05 * sb.sigma_c.values).sum()):,}")

    print("\n== soft ratio rows (the floor applies to every free-cone row): z² by σ_c band of the base run")
    rows = []
    for lab, opt in (("base", base), ("new", new)):
        const, _ = D._lineage(opt)
        c = D._rows(opt, const)
        r = c[(c.type == "ratio") & (c.cone_type == "free")].copy()
        rows.append(r.set_index("constraint_id")[["sigma_c", "z2"]].rename(columns={"sigma_c": f"sc_{lab}", "z2": f"z2_{lab}"}))
    r = rows[0].join(rows[1], how="inner")
    r["band"] = pd.cut(r.sc_base, [0, 0.03, 0.3, 3, 1e12], labels=["σ<0.03", "0.03–0.3", "0.3–3", ">3 kt"], include_lowest=True)
    t = r.groupby("band", observed=False).agg(rows=("z2_base", "size"), base=("z2_base", "sum"), new=("z2_new", "sum"))
    print(t.round(0).to_string())
    print(f"   ratio-row TOTAL  base {r.z2_base.sum():,.0f}  new {r.z2_new.sum():,.0f}")

    eb, en = _edges(base, ei), _edges(new, ei)
    print("\n== stop rule (a): world retention (posterior/prior)")
    rb, rn = _world_retention(eb), _world_retention(en)
    for k in rb:
        print(f"   {k:16s} base {rb[k]:.3f}  new {rn[k]:.3f}  Δ {rn[k]-rb[k]:+.3f}")

    print("\n== mass moved: Σ|post_new − post_base| over CC edges, by node-total band of the edge's from-node")
    m = eb[["edge_id", "post"]].merge(en[["edge_id", "post"]], on="edge_id", suffixes=("_b", "_n"))
    m = m.merge(ei[["edge_id", "sector_from", "region_from", "time", "valuation"]], on="edge_id")
    m = m[m.valuation == "CC"]
    tot = sb[sb.valuation == "CC"].groupby(["sector", "region", "time"]).ac.max().rename("node_tot").reset_index()
    tot["time"] = tot.time.astype(str)
    m["time"] = m.time.astype(str)
    m = m.merge(tot, left_on=["sector_from", "region_from", "time"], right_on=["sector", "region", "time"], how="left")
    m["moved"] = (m.post_n - m.post_b).abs()
    m = m[m.node_tot.notna()]          # SB-origin edges (CO, Scrap) have no scale total — excluded
    m["band"] = pd.cut(m.node_tot, BANDS, labels=LABELS, include_lowest=True)
    g = m.groupby("band", observed=False).agg(moved=("moved", "sum"), post=("post_b", lambda s: s.abs().sum()))
    g["moved_%"] = g.moved / g.post.replace(0, np.nan) * 100
    print(g.round(1).to_string())
    print(f"   total CC moved {m.moved.sum():,.0f} kt of {m.post_b.abs().sum():,.0f} ({m.moved.sum()/m.post_b.abs().sum()*100:.2f} %)")

    print(f"\n== stop rule (b): nodes with |c| < {SMALL_KT:.0f} kt — posterior/prior of their scale totals")
    for lab, s in (("base", sb), ("new", sn)):
        small = s[(s.ac < SMALL_KT) & (s.ac > 0)]
        r = (small.lhs_post / small.c)
        print(f"   {lab}: n {len(small):,}  median {r.median():.3f}  p10 {r.quantile(.1):.3f}  p90 {r.quantile(.9):.3f}")

    print("\n== stop rule (c): exporter cross-region concentrate CC retention, 2010–2019")
    cross = (ei.sector_from == "Concentrate") & (ei.sector_to == "Smelting") & (ei.region_from != ei.region_to) & (ei.valuation == "CC")
    for lab, e in (("base", eb), ("new", en)):
        e = e.merge(ei[["edge_id"]].assign(cross=cross.values), on="edge_id")
        e = e[e.cross & e.time.astype(int).between(2010, 2019)]
        line = f"   {lab}: world {e.post.sum()/e.prior.sum():.3f}"
        for x in EXPORTERS:
            t = e[e.region_from == x]
            line += f"  {x} {t.post.sum()/max(t.prior.sum(), 1e-9):.3f}"
        print(line)

    print("\n== largest cells (sector, region) on new run, with base")
    gb = sb.groupby(["sector", "region"]).z2.sum(); gn = sn.groupby(["sector", "region"]).z2.sum()
    d = pd.DataFrame({"base": gb, "new": gn}).fillna(0)
    print(d.sort_values("new", ascending=False).head(12).round(0).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
