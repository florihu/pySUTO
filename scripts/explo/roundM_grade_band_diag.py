"""
roundM_grade_band_diag.py
=========================
Round M measurement: the grade bands are the deletion valve (ADR 0028).

    python scripts/explo/roundM_grade_band_diag.py <opt_base> <opt_M1> [<opt_M2>]

M.1 makes the ore-grade hard band year-local (it bracketed the region's 30-year
series envelope, so Chile's ceiling was 1.1x its 1995 grade and bound in 26 of
30 years).  M.2 narrows the global concentrate band, on whose floor 35 % of
Chile's export leg-years sat.  Both hold sigma fixed via `unc_type="sd"`, so the
ONLY thing that moves is the hard cone -- section 1 is the gate that proves it.

Reads persisted run artifacts only; every upstream id comes from the run's own
sidecar (the Round C hygiene rule).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/explo")
from pysuto.acc import AccountingStructure              # noqa: E402
from pysuto.config import load_config                   # noqa: E402
from pysuto.param_relation import expand_edge_index     # noqa: E402
import india_concentrate_diag as D                      # noqa: E402
import roundL_reexport_diag as L                        # noqa: E402

ROOT   = Path("data/proc")
YEARS  = (2010, 2019)
OG_DIR = Path("data/input/params/ore_grade")

# The split that defines the defect (REP …20260916_chile-mine-chain §1).
TRADE_DEPENDENT = ["China", "Japan", "Korea, Rep.", "Germany"]
SELF_SUFFICIENT = ["United States", "Russia", "Zambia", "Poland"]


def _ore_grade_version(imp: str) -> int:
    """The ore_grade input version a run consumed, from its own sidecar."""
    import json
    m = json.load(open(ROOT / "imp" / imp / "metadata.json"))
    for p in m["parents"]["inputs"].get("params", []):
        if p["id"] == "ore_grade":
            return int(p["version"])
    return -1


def _bands(version: int, prefix: str = "oregradeoresupply") -> pd.DataFrame:
    p = OG_DIR / f"v{version}" / "params.csv"
    if not p.exists():
        p = OG_DIR / "current" / "params.csv"
    d = pd.read_csv(p)
    d = d[d.param_id.astype(str).str.startswith(prefix)]
    return (d.assign(region=d.a_region_from, time=d.a_time.astype(int))
             .groupby(["region", "time"])[["value_nominal", "value_min", "value_max"]]
             .first().reset_index())


def _conc_band(imp: str) -> tuple:
    """The concentrate band the run actually used, from its own conc version."""
    import json
    m = json.load(open(ROOT / "imp" / imp / "metadata.json"))
    v = next((p["version"] for p in m["parents"]["inputs"].get("params", [])
              if p["id"] == "conc"), None)
    path = Path("data/input/params/conc") / (f"v{v}" if v else "current") / "params.csv"
    d = pd.read_csv(path)
    r = d[d.param_id == "concconcentratesupply"].iloc[0]
    return float(r["value_min"]), float(r["value_max"]), v


def _pair(e: pd.DataFrame, **sel) -> pd.DataFrame:
    """Mass/CC side by side for one edge family, indexed by region-year."""
    m = e
    for k, v in sel.items():
        m = m[m[k] == v]
    w = m.pivot_table(index=["region_from", "time"], columns="valuation",
                      values=["prior", "post"]).reset_index()
    w.columns = ["_".join(c).strip("_") for c in w.columns]
    return w


def gate(runs):
    print("=" * 78)
    print("1. ATTRIBUTION GATE — Stage 2 must be untouched")
    base_imp = L._chain(runs[0][1])[1]
    b = np.load(ROOT / "imp" / base_imp / "agg_result.npz", allow_pickle=True)
    for lab, r in runs:
        imp = L._chain(r)[1]
        a = np.load(ROOT / "imp" / imp / "agg_result.npz", allow_pickle=True)
        df = float(np.nanmax(np.abs(a["f_agg"] - b["f_agg"])))
        sb = np.abs(b["sigma_agg"])
        rel = np.abs(a["sigma_agg"] - b["sigma_agg"]) / np.where(sb > 0, sb, np.nan)
        ds = float(np.nanmax(rel)) if np.isfinite(rel).any() else 0.0
        cov = int((a["coverage"] != b["coverage"]).sum())
        # σ is written through a CSV, so the tolerance is float round-trip, not 0.
        ok = "PASS" if (df == 0.0 and ds < 1e-12 and cov == 0) else "*** FAIL ***"
        print(f"   {lab:5s} {imp}  ore_grade v{_ore_grade_version(imp)}"
              f"   max|Δf_agg| {df:.3e}  max rel Δσ_agg {ds:.3e}"
              f"  coverage Δ {cov}   {ok}")
    print("   (a FAIL means the run moved more than the cone — nothing below is attributable)")


def chile_chain(E, runs):
    print("\n" + "=" * 78)
    print("2. CHILE'S CHAIN — and the decomposition  ore Mass ret = CC ret x (g_prior / g_post)")
    for lab, _ in runs:
        e = E[lab]
        w = _pair(e, sector_from="Mining", sector_to="Ore")
        w = w[(w.region_from == "Chile") & w.time.between(*YEARS)]
        mp, ms = w.prior_Mass.sum(), w.post_Mass.sum()
        cp, cs = w.prior_CC.sum(), w.post_CC.sum()
        gp, gq = cp / mp, cs / ms
        mf = _pair(e, sector_from="MF", sector_to="Concentrate")
        mf = mf[(mf.region_from == "Chile") & mf.time.between(*YEARS)]
        print(f"   {lab:5s} ore Mass {ms/mp:.3f} = CC {cs/cp:.3f} x rock {(gp/gq):.3f}"
              f"   | in-situ grade {gp*100:.3f}% → {gq*100:.3f}%"
              f"   | MF→Concentrate CC {mf.post_CC.sum()/mf.prior_CC.sum():.3f}")


def ore_band(E, runs):
    print("\n" + "=" * 78)
    print("3. ORE-GRADE BAND OCCUPANCY (region-years, 2010-19)")
    for lab, r in runs:
        e = E[lab]
        b = _bands(_ore_grade_version(L._chain(r)[1]))
        w = _pair(e, sector_from="Mining", sector_to="Ore")
        w = w[(w.prior_Mass > 10) & (w.post_Mass > 1e-9)]
        w["g_post"] = w.post_CC / w.post_Mass
        w = w.merge(b, left_on=["region_from", "time"], right_on=["region", "time"],
                    how="left").dropna(subset=["value_max"])
        w = w[w.time.between(*YEARS)]
        w["at_cap"] = w.g_post >= w.value_max * 0.9995
        w["at_flr"] = w.g_post <= w.value_min * 1.0005
        ch = w[w.region_from == "Chile"]
        print(f"   {lab:5s} n={len(w)}  at cap {100*w.at_cap.mean():5.1f}%  at floor"
              f" {100*w.at_flr.mean():5.1f}%  median cap/nominal"
              f" {(w.value_max/w.value_nominal).median():.2f}"
              f"   | Chile at cap {int(ch.at_cap.sum())}/{len(ch)}")


def conc_band(E, runs):
    print("\n" + "=" * 78)
    print("4. CONCENTRATE-LEG GRADE (per edge, 2010-19) — share pinned on the band")
    for lab, r in runs:
        e = E[lab]
        lo, hi, v = _conc_band(L._chain(r)[1])   # the band THIS run used
        m = e[(e.sector_from == "Concentrate") & (e.sector_to == "Smelting")]
        key = ["sector_from", "region_from", "region_to", "time"]
        w = m.pivot_table(index=key, columns="valuation", values=["prior", "post"]).reset_index()
        w.columns = ["_".join(c).strip("_") for c in w.columns]
        w = w[(w.prior_Mass > 1) & (w.post_Mass > 1e-6) & w.time.between(*YEARS)]
        w["g"] = w.post_CC / w.post_Mass
        ch = w[(w.region_from == "Chile") & (w.region_to != "Chile")]
        jp = ch[ch.region_to == "Japan"]
        f = lambda d: 100 * (d.g <= lo * 1.0005).mean()
        print(f"   {lab:5s} conc v{v} band [{lo:.2f},{hi:.2f}]  world n={len(w)}"
              f" at floor {f(w):4.1f}%  at cap {100*(w.g >= hi*0.9995).mean():4.1f}%"
              f"  median {w.g.median():.3f}"
              f"   | Chile floor {f(ch):5.1f}% median {ch.g.median():.3f}"
              f"   | →Japan floor {f(jp):5.1f}%")


def mine_shortfall(runs):
    print("\n" + "=" * 78)
    print("5. MINE CONCENTRATE PRODUCTION (supplytotout MF CC) — anchor vs posterior, 2010-19")
    tab = {}
    for lab, r in runs:
        const, _ = D._lineage(r)
        c = D._rows(r, const); c["time"] = c.time.astype(int)
        w = c[(c.valuation == "CC") & (c.type == "sum") & (c.variant == "supplytotout")
              & (c.sector == "MF") & c.time.between(*YEARS)]
        g = w.groupby("region").agg(anchor=("c", "sum"), post=("lhs_post", "sum"))
        g["short"] = g.anchor - g.post
        tab[lab] = g
    base = tab[runs[0][0]]
    idx = base.sort_values("short", ascending=False).head(8).index
    hdr = "   " + f"{'region':16s}" + "".join(f"{lab:>22s}" for lab, _ in runs)
    print(hdr)
    for r_ in idx:
        line = f"   {r_:16s}"
        for lab, _ in runs:
            g = tab[lab]
            line += f"{g.loc[r_,'short']:>12,.0f} ({g.loc[r_,'post']/g.loc[r_,'anchor']:.3f})"
        print(line)
    line = f"   {'WORLD':16s}"
    for lab, _ in runs:
        g = tab[lab]
        line += f"{g.short.sum():>12,.0f} ({g.post.sum()/g.anchor.sum():.3f})"
    print(line)


def intake_split(E, runs):
    print("\n" + "=" * 78)
    print("6. SMELTER INTAKE RETENTION — trade-dependent vs self-sufficient")
    for lab, _ in runs:
        e = E[lab]
        w = e[(e.valuation == "CC") & (e.sector_from == "Concentrate")
              & (e.sector_to == "Smelting") & e.time.between(*YEARS)]
        g = w.groupby("region_to")[["prior", "post"]].sum()
        g["ret"] = g.post / g.prior
        td = "  ".join(f"{r.split(',')[0][:7]} {g.ret.get(r, np.nan):.3f}" for r in TRADE_DEPENDENT)
        ss = "  ".join(f"{r.split(' ')[0][:7]} {g.ret.get(r, np.nan):.3f}" for r in SELF_SUFFICIENT)
        print(f"   {lab:5s} trade-dep  {td}")
        print(f"   {'':5s} self-suff  {ss}")


def world(E, runs):
    print("\n" + "=" * 78)
    print("7. WORLD TOTALS (2010-19, prior → posterior)")
    for lab, _ in runs:
        e = E[lab]
        om = e[(e.sector_from == "CO") & (e.valuation == "Mass") & e.time.between(*YEARS)]
        cc = e[(e.sector_from == "MF") & (e.sector_to == "Concentrate")
               & (e.valuation == "CC") & e.time.between(*YEARS)]
        print(f"   {lab:5s} ore Mass {om.post.sum():>12,.0f} / {om.prior.sum():>12,.0f}"
              f" = {om.post.sum()/om.prior.sum():.3f}"
              f"   | concentrate CC {cc.post.sum():>9,.0f} / {cc.prior.sum():>9,.0f}"
              f" = {cc.post.sum()/cc.prior.sum():.3f}")


def chile_rows(runs):
    print("\n" + "=" * 78)
    print("8. CHILE'S ROWS — median z over 2010-19")
    want = [("supplytotout", "Mining", "Mass"), ("supplytotout", "Mining", "CC"),
            ("supplytotout", "MF", "CC"), ("usetotout", "Concentrate", "CC")]
    for lab, r in runs:
        const, _ = D._lineage(r)
        c = D._rows(r, const); c["time"] = c.time.astype(int)
        ch = c[(c.region == "Chile") & c.time.between(*YEARS)]
        bits = []
        for v, s, val in want:
            q = ch[(ch.variant == v) & (ch.sector == s) & (ch.valuation == val)]
            bits.append(f"{s}/{val} {q.z.median():>7.1f}")
        og = c[(c.variant.astype(str).str.startswith("oregradeoresupplychile"))
               & c.time.between(*YEARS)]
        bits.append(f"oregrade {og.z.median():>6.1f}" if len(og) else "oregrade    n/a")
        print(f"   {lab:5s} " + "  ".join(bits))


def healers(E, runs):
    print("\n" + "=" * 78)
    print("9. THE SILENT HEALERS — scrap into smelting, implied process efficiency")
    for lab, _ in runs:
        e = E[lab]
        w = e[(e.valuation == "CC") & e.time.between(*YEARS)]
        def _s(**sel):
            m = w
            for k, v in sel.items():
                m = m[m[k] == v]
            return m.prior.sum(), m.post.sum()
        cp, cq = _s(sector_from="Concentrate", sector_to="Smelting")
        sp, sq = _s(sector_from="Scrap", sector_to="Smelting")
        mp, mq = _s(sector_from="Smelting", sector_to="Matte")
        ap, aq = _s(sector_from="Matte", sector_to="CFR")
        rp, rq = _s(sector_from="Scrap", sector_to="CFR")
        np_, nq = _s(sector_from="CFR", sector_to="Anode")
        print(f"   {lab:5s} scrap→smelting {sp:>8,.0f} → {sq:>8,.0f}"
              f"   eff_smelt {mp/(cp+sp):.3f} → {mq/(cq+sq):.3f}"
              f"   eff_refine {np_/(ap+rp):.3f} → {nq/(aq+rq):.3f}")


def headline(runs):
    print("\n" + "=" * 78)
    print("10. HEADLINE")
    for lab, r in runs:
        const, _ = D._lineage(r)
        c = D._rows(r, const)
        s = c[c.type == "sum"]
        print(f"   {lab:5s} {r}  sum rows {len(s):,}  z² {float((s.z**2).sum()):>12,.0f}"
              f"   {L._status(r)}")


def main(argv):
    labels = ["base", "M.1", "M.2"]
    runs = [(labels[i], a) for i, a in enumerate(argv)]
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    E = {}
    for lab, r in runs:
        e = L._edges(r, ei)
        e["time"] = e.time.astype(int)
        E[lab] = e
    gate(runs)
    chile_chain(E, runs)
    ore_band(E, runs)
    conc_band(E, runs)
    mine_shortfall(runs)
    intake_split(E, runs)
    world(E, runs)
    chile_rows(runs)
    healers(E, runs)
    headline(runs)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
