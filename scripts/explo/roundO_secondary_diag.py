"""
roundO_secondary_diag.py
========================
Round O measurement: **secondary-smelter copper enters the chain once** (ADR 0030).

    python scripts/explo/roundO_secondary_diag.py --feed-only <opt_base>
    python scripts/explo/roundO_secondary_diag.py <opt_base> <opt_O> [<opt_P> ...]

ICSG's secondary *refined* production contains the copper that reached the
refinery from a secondary smelter, and the parse booked both series as inputs;
the same secondary refined copper was also added a second time to the cathode
production the netting (ADR 0027) and the fan bound (ADR 0024) use. Sections:

  0  FEED GATE (no run): the new icsg feed differs from the base run's only in
     the Refining/Scrap/Use rows; the excess list; the three pre-solver
     acceptance numbers (Austria closes, the world concentrate gap, P_Cathode).
  1  Stage-2 attribution: which edge classes moved in f_agg.
  2  Netting ledger by sector (the CFR fix vs the cathode term).
  3  Secondary share, China's smelters vs their band, band occupancy.
  4  Mine chain: retention, who pays, trade-dependent vs self-sufficient.
  5  Export-mass gap re-measured (ADR 0029's detector).
  6  Stock-change buffers.
  7  Sum-row z² on the COMMON row set (the population changes when a round
     adds a family), status, world retention.
  8  Round P: reported vs posterior scrap level on BOTH routes; invented
     direct scrap where none is reported; band-vs-level headroom.

Reads persisted run artifacts only; every upstream id comes from the run's own
sidecar (the Round C hygiene rule).
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
sys.path.insert(0, "scripts/report")

from pysuto.input_registry import InputRegistries               # noqa: E402
import roundL_reexport_diag as L                                # noqa: E402
import roundN_concentrate_grade_diag as N                       # noqa: E402
import copper_metabolism_insights as M                          # noqa: E402
from build_baci_reexport import _eff_nominals, production_recipes  # noqa: E402

ROOT = Path("data/proc")
POOL = ["2020", "2021", "2022", "2023", "2024"]
KEY = ["Region", "Type", "Year", "Process", "Supply_Use"]


# ---------------------------------------------------------------------------
# lineage
# ---------------------------------------------------------------------------

def _input_version(opt: str, kind: str) -> int:
    """The version of feed ``kind`` a run consumed, from its feed run's sidecar."""
    _, _, feeds = L._chain(opt)
    meta = json.load(open(ROOT / "feed" / "registry" / feeds[kind] / "metadata.json"))
    return int(meta["parents"]["inputs"]["feed"]["version"])


def _param_version(opt: str, pid: str) -> int:
    const, _ = L.D._lineage(opt)
    m = json.load(open(ROOT / "const" / const / "metadata.json"))
    return {p["id"]: p["version"] for p in m["parents"]["inputs"]["params"]}[pid]


def _icsg(version: int) -> pd.DataFrame:
    d = Path("data/input/feed/icsg") / f"v{version}"
    raw = pd.read_csv(d / "values.csv")
    rmap = pd.read_excel(d / "conc.xlsx", sheet_name="region_map")
    raw["reg"] = raw["Region"].map(dict(zip(rmap["src_region"], rmap["root_region"])))
    return raw


def _ser(raw, process, typ, su="Use"):
    x = raw[(raw.Process == process) & (raw.Type == typ) & (raw.Supply_Use == su)]
    return x.groupby(["reg", "Year"])["Value"].sum()


# ---------------------------------------------------------------------------
# 0. feed gate
# ---------------------------------------------------------------------------

def feed_gate(base: str) -> None:
    v_old = _input_version(base, "icsg")
    v_new = InputRegistries.initialize(root="data/input").feed.entry("icsg")["version"]
    print(f"\n== 0. FEED GATE — icsg v{v_old} (base run) → v{v_new} (current)")
    old, new = _icsg(v_old), _icsg(v_new)
    m = old.merge(new, on=KEY, how="outer", suffixes=("_old", "_new"))
    m = m[["Region", "Type", "Year", "Process", "Supply_Use", "Value_old", "Value_new"]]
    d = m[(m.Value_old.fillna(-1) - m.Value_new.fillna(-1)).abs() > 1e-6]
    classes = d.groupby(["Process", "Type", "Supply_Use"]).size()
    ok = set(classes.index) <= {("Refining", "Scrap", "Use")}
    print(f"   rows changed: {len(d):,} of {len(m):,}  classes: {classes.to_dict()}  "
          f"{'PASS' if ok else '*** FAIL: other series moved ***'}")
    print(f"   Refining/Scrap/Use world: {d.Value_old.sum():,.0f} → {d.Value_new.sum():,.0f} kt "
          f"({(d.Value_new.sum()/d.Value_old.sum()-1)*100:+.1f} %)")
    ex_path = Path("data/proc/feedprep/icsg_clean/icsg_secondary_excess.csv")
    if ex_path.exists():
        ex = pd.read_csv(ex_path)
        top = ex.groupby("Region")["excess_kt"].agg(["size", "sum"]).sort_values("sum", ascending=False)
        print(f"   excess (smelter route > secondary refined, clipped to 0): {len(ex)} region-years, "
              f"{ex.excess_kt.sum():,.0f} kt; top: "
              + ", ".join(f"{r} {int(s)}y {v:,.0f}" for r, (s, v) in top.head(5).iterrows()))

    eff = _eff_nominals()
    k = eff["CFR"] * eff["ER"]
    for lab, raw in (("v%d" % v_old, old), ("v%d" % v_new, new)):
        sm, sc, an = _ser(raw, "Smelting", "Scrap"), _ser(raw, "Refining", "Scrap"), _ser(raw, "Refining", "Anode")
        mine, sx = _ser(raw, "Mining", "Concentrate", "Supply"), _ser(raw, "SXEW", "Cathode", "Supply")
        # Austria: what enters in refined-copper terms over what leaves
        a = pd.DataFrame({"in": sm.xs("Austria") * eff["Smelting"] * k + sc.xs("Austria") * k,
                          "out": an.xs("Austria") * eff["ER"]}).dropna()
        r = a["in"] / a["out"]
        # world: concentrate the anode anchor can absorb vs mined
        w = pd.DataFrame({"sm": sm.groupby(level=1).sum(), "sc": sc.groupby(level=1).sum(),
                          "an": an.groupby(level=1).sum(), "mine": mine.groupby(level=1).sum()})
        w["conc_needed"] = ((w.an * eff["ER"] - w.sc * k) / k) / eff["Smelting"] - w.sm
        w["gap"] = w.mine - w.conc_needed
        prod, _ = production_recipes(mine, _ser(raw, "Smelting", "Concentrate"), sm, an, sc, sx,
                                     _ser(raw, "Semi", "Cathode"), eff)
        pc = prod["Cathode"].groupby(level=1).sum()
        ref = (an * eff["ER"]).groupby(level=1).sum() + sx.groupby(level=1).sum()
        print(f"   {lab}: Austria in/out 2024 {r.get(2024, np.nan):.3f} (30-y {r.min():.2f}–{r.max():.2f})   "
              f"world gap 2024 {w.gap[2024]:,.0f} kt   P_Cathode/refined+SXEW 2024 {pc[2024]/ref[2024]:.3f}")
    print("   acceptance: Austria → 1.00 ± 0.02, gap → ≈ −0.46 Mt (the nested arithmetic on the old feed), "
          "P_Cathode ratio → 1.000")


# ---------------------------------------------------------------------------
# 1-7. run comparisons
# ---------------------------------------------------------------------------

def stage2_attribution(runs: list, ei: pd.DataFrame) -> None:
    print("\n== 1. Stage-2 attribution — edge classes whose f_agg moved")
    imps = [L._chain(r)[1] for _, r in runs]
    a = np.load(ROOT / "imp" / imps[0] / "agg_result.npz", allow_pickle=True)
    b = np.load(ROOT / "imp" / imps[1] / "agg_result.npz", allow_pickle=True)
    d = pd.DataFrame({"edge_id": np.arange(len(a["f_agg"])), "d": b["f_agg"] - a["f_agg"]})
    d = d[d.d.abs() > 1e-9].merge(ei[["edge_id", "sector_from", "sector_to", "region_from",
                                       "region_to", "valuation"]], on="edge_id")
    d["cls"] = d.sector_from + "→" + d.sector_to + np.where(d.region_from != d.region_to, " (trade)", "")
    g = d.groupby(["valuation", "cls"]).d.agg(n="size", abs_sum=lambda s: s.abs().sum(), net="sum")
    print(g.sort_values("abs_sum", ascending=False).head(18).round(0).to_string())
    cov = int((a["coverage"] != b["coverage"]).sum())
    print(f"   coverage changed on {cov} edges")


def netting_by_sector(runs: list) -> None:
    print("\n== 2. ADR 0027 netting by sector — T_applied (kt CC, 30 y)")
    out = {}
    for lab, r in runs:
        led = L._ledger(r)
        if led is None:
            continue
        col = "T_applied" if "T_applied" in led else "T"
        out[lab] = led.groupby("sector")[col].sum()
    if out:
        t = pd.DataFrame(out)
        t["Δ"] = t.iloc[:, -1] - t.iloc[:, 0]
        print(t.round(0).to_string())


def secondary_and_bands(runs: list, E: dict) -> None:
    print("\n== 3. Secondary share (refinery-side input), prior → posterior, 2020-24 pooled")
    for lab, r in runs:
        cc = E[lab]
        t = M.t1e_secondary_share_by_group(cc)
        t = t[t.year.between(2020, 2024)].groupby("group")[["share_prior", "share_post"]].mean()
        world_sec = M.by_year(M.flow(cc, "Scrap", "Smelting")) + M.by_year(M.flow(cc, "Scrap", "CFR"))
        world_prim = M.by_year(M.flow(cc, "Concentrate", "Smelting")) + M.by_year(M.flow(cc, "SXEW", "CSXEW"))
        ws = (world_sec / (world_sec + world_prim)).loc[2020:2024].mean()
        print(f"   {lab:5s} world prior {ws.prior:.3f} post {ws.post:.3f} | " +
              "  ".join(f"{g} {t.loc[g,'share_prior']:.2f}→{t.loc[g,'share_post']:.2f}"
                        for g in ["China", "EU-14", "JP/KR/TW", "United States"]))
    print("   band occupancy (posterior pooled share vs the run's own scrap_share version):")
    for lab, r in runs:
        cc = E[lab]
        ver = _param_version(r, "scrap_share")
        p = pd.read_csv(Path("data/input/params/scrap_share") / f"v{ver}" / "params.csv", sep=";", dtype=str)
        rows = []
        for proc, prim in (("Smelting", "Concentrate"), ("CFR", "Matte")):
            s = M.flow(cc[cc.time.isin(POOL)], "Scrap", proc).groupby("region_to").post.sum()
            q = M.flow(cc[cc.time.isin(POOL)], prim, proc).groupby("region_to").post.sum()
            share = (s / (s + q.reindex(s.index).fillna(0.0))).dropna()
            b = p[p.a_sector_to == proc].set_index("a_region_to")
            for reg, v in share.items():
                if reg in b.index and (s[reg] + q.get(reg, 0.0)) > 50:
                    lo, hi = float(b.loc[reg, "value_min"]), float(b.loc[reg, "value_max"])
                    rows.append(dict(proc=proc, region=reg, share=v, lo=lo, hi=hi,
                                     on_floor=abs(v - lo) < 0.005, on_ceiling=abs(v - hi) < 0.005))
        d = pd.DataFrame(rows)
        g = d.groupby("proc")[["on_floor", "on_ceiling"]].sum()
        print(f"   {lab:5s} scrap_share v{ver}: " + "  ".join(
            f"{pr}: {int(g.loc[pr,'on_floor'])} on floor, {int(g.loc[pr,'on_ceiling'])} on ceiling of {int((d.proc==pr).sum())}"
            for pr in g.index))
        for reg in ("China", "Germany", "Austria", "Japan"):
            x = d[(d.region == reg)]
            print("         " + reg + ": " + "  ".join(f"{r.proc} {r.share:.3f} in [{r.lo:.3f}, {r.hi:.3f}]"
                                                      for r in x.itertuples()))


def mine_chain(runs: list, E: dict) -> None:
    print("\n== 4. Mine chain")
    for lab, r in runs:
        cc = E[lab]
        w = M.by_year(M.flow(cc, "MF", "Concentrate"))
        p = cc[cc.time.isin(POOL)]
        def ret(sf, st, region=None, col="region_from"):
            x = M.flow(p, sf, st)
            if region:
                x = x[x[col] == region]
            return x.post.sum() / x.prior.sum()
        dep = M.t2b_import_dependence(cc)
        big = dep[dep.intake_pool_prior > 1000]
        td = big[big.import_share_pool > 0.5]; ss = big[big.import_share_pool <= 0.5]
        wret = lambda d: d.intake_pool_post.sum() / d.intake_pool_prior.sum()
        print(f"   {lab:5s} world MF→Conc retention 30-y {w.post.sum()/w.prior.sum():.3f} (2020-24 {w.post.loc[2020:2024].sum()/w.prior.loc[2020:2024].sum():.3f})"
              f" | mine 2020-24: Chile {ret('MF','Concentrate','Chile'):.3f} Peru {ret('MF','Concentrate','Peru'):.3f}"
              f" | intake retained: trade-dependent {wret(td):.3f}  self-sufficient {wret(ss):.3f}"
              f" (China {ret('Concentrate','Smelting','China','region_to'):.3f}, Japan {ret('Concentrate','Smelting','Japan','region_to'):.3f})")
    print("   who pays 2020-24 (Δ post − prior, kt CC), base → O:")
    tabs = {}
    for lab, r in runs:
        p = E[lab][E[lab].time.isin(POOL)]
        def delta(sf, st, col):
            g = M.flow(p, sf, st).groupby(col)[["prior", "post"]].sum(); return g.post - g.prior
        tabs[lab] = pd.DataFrame({"scrap→Sm": delta("Scrap", "Smelting", "region_to"),
                                  "scrap→CFR": delta("Scrap", "CFR", "region_to"),
                                  "conc intake": delta("Concentrate", "Smelting", "region_to"),
                                  "mine": delta("MF", "Concentrate", "region_from")}).fillna(0.0)
    labs = [l for l, _ in runs]
    for reg in ["China", "Japan", "Germany", "Chile", "Peru", "Zambia", "Austria"]:
        print(f"      {reg:9s} " + " | ".join(
            f"{lab}: " + " ".join(f"{c} {tabs[lab].loc[reg, c]:+,.0f}" for c in tabs[lab].columns)
            for lab in labs if reg in tabs[lab].index))
    print("      world     " + " | ".join(
        f"{lab}: " + " ".join(f"{c} {tabs[lab][c].sum():+,.0f}" for c in tabs[lab].columns) for lab in labs))
    # fabrication check: intake retention above 1.05 that was not there in base
    base = M.t2b_import_dependence(E[labs[0]]).set_index("region")
    new = M.t2b_import_dependence(E[labs[-1]]).set_index("region")
    fab = [r for r in new.index if new.loc[r, "retention_pool"] > 1.05 and
           (r not in base.index or base.loc[r, "retention_pool"] <= 1.05) and new.loc[r, "intake_pool_prior"] > 500]
    print(f"   new intake retention > 1.05 (fabrication check): {fab or 'none'}")


def export_mass_gap(runs: list) -> None:
    print("\n== 5. Export-mass gap (ADR 0029 detector: P > 0.35·X + U)")
    m, P, U, _ = N.load_inputs()
    for lab, r in runs:
        dc = N.discharge_capacity(m, P, U, r)
        hard = dc[dc.headroom < 0]
        forced = (-hard.headroom).sum()
        ch = dc[dc.region == "Chile"]
        print(f"   {lab:5s} mine deleted {dc.deleted.sum():,.0f} kt; forced {forced:,.0f} kt = "
              f"{forced/dc.deleted.sum()*100:.1f} % of deletion; Chile hard-capped {int((ch.headroom<0).sum())}/30, "
              f"Chile deleted {ch.deleted.sum():,.0f}")


def stock_buffers(runs: list, E: dict) -> None:
    print("\n== 6. Stock-change buffers, cumulative 1995-2024 (Mt CC)")
    for lab, r in runs:
        w, hubs = M.t3c_stock_buffers(E[lab])
        print(f"   {lab:5s} MESC post {w.MESC_post[2024]/1e3:.1f} (prior {w.MESC_prior[2024]/1e3:.1f})  "
              f"USC {w.USC_post[2024]/1e3:.1f} ({w.USC_prior[2024]/1e3:.1f})  PSC {w.PSC_post[2024]/1e3:.1f} ({w.PSC_prior[2024]/1e3:.1f})  "
              f"hubs: " + ", ".join(f"{c} {hubs[c][2024]/1e3:.1f}" for c in hubs.columns if c != "other"))


def headline(runs: list, E: dict) -> None:
    """§7. Sum-row z² — on the COMMON row set, because the population changes.

    Round P (ADR 0031) adds ~1,014 scrap-level rows, so a sum-row z² *total* is
    not comparable across a round that adds a constraint family: the documented
    baselines (base 1,719,806 / Round O 1,949,396) are totals over a different
    row set. Three numbers, three different questions:
      common     — constraint_ids present in EVERY run: did the fit improve?
      only-here  — what the new rows themselves cost, by variant.
      total      — bookkeeping only, explicitly non-comparable.
    """
    print("\n== 7. Sum-row z², status, world retention")
    S = {lab: L._sums(r) for lab, r in runs}
    common = set.intersection(*(set(S[lab]["constraint_id"]) for lab, _ in runs))
    b0   = S[runs[0][0]]
    tot0 = b0.z2.sum()
    com0 = b0[b0.constraint_id.isin(common)].z2.sum()
    for lab, r in runs:
        d   = S[lab]
        inc = d.constraint_id.isin(common)
        z, zc, zn = d.z2.sum(), d[inc].z2.sum(), d[~inc].z2.sum()
        print(f"   {lab:5s} {r}  {L._status(r)}")
        print(f"         rows {len(d):,} (common {int(inc.sum()):,}, only-here "
              f"{int((~inc).sum()):,})   total z² {z:,.0f} ({(z / tot0 - 1) * 100:+.1f} %, "
              f"NOT comparable across a row-set change)")
        print(f"         **COMMON-SET z² {zc:,.0f} ({(zc / com0 - 1) * 100:+.1f} %)**   "
              f"only-here z² {zn:,.0f}")
        print("         " + "  ".join(f"{k} {v:.3f}"
                                      for k, v in L._world_retention(E[lab]).items()))
    if len(runs) > 1:
        a = S[runs[0][0]]; c = S[runs[-1][0]]
        ga = a[a.constraint_id.isin(common)].groupby("sector").z2.sum()
        gc = c[c.constraint_id.isin(common)].groupby("sector").z2.sum()
        d  = (gc - ga).sort_values()
        print("   z² by sector ON THE COMMON SET, largest moves: " + ", ".join(
            f"{k} {v:+,.0f}" for k, v in pd.concat([d.head(4), d.tail(3)]).items()))
        for lab, _ in runs[1:]:
            new_rows = S[lab][~S[lab].constraint_id.isin(common)]
            if len(new_rows):
                print(f"   {lab}: rows only here, z² by variant: " + ", ".join(
                    f"{v} {row['sum']:,.0f} ({int(row['size']):,} rows)"
                    for v, row in
                    new_rows.groupby("variant").z2.agg(["sum", "size"]).iterrows()))


def _reported_scrap(opt: str, E_cc: pd.DataFrame) -> pd.DataFrame:
    """The scrap level each feed actually observed, per (route, region, year).

    Coverage is the pre-imputation "did a feed see this edge?" flag, so this is
    exactly the population ADR 0031 anchors — imputed fills are excluded.
    """
    imp = L._chain(opt)[1]
    z   = np.load(ROOT / "imp" / imp / "agg_result.npz", allow_pickle=True)
    f   = np.nan_to_num(np.asarray(z["f_agg"], dtype=float))
    cov = np.asarray(z["coverage"], dtype=bool)
    e   = E_cc[["edge_id", "sector_from", "sector_to", "region_to", "time"]].drop_duplicates()
    e   = e[e.sector_from.eq("Scrap") & e.sector_to.isin(["Smelting", "CFR"])].copy()
    e["rep"] = f[e.edge_id.values]
    e["cov"] = cov[e.edge_id.values]
    return e.rename(columns={"sector_to": "route", "region_to": "reg"})


def scrap_level_acceptance(runs: list, E: dict) -> None:
    """§8 (Round P, ADR 0031) — "Smelting & Refining are still modelled with
    scrap inflow", as a falsifiable check on BOTH routes."""
    print("\n== 8. Scrap level: ICSG reported vs posterior, BOTH routes (kt CC, 30 y)")
    ref = _reported_scrap(runs[-1][1], E[runs[-1][0]])
    obs = ref[ref["cov"] & (ref.rep.abs() > 1e-9)][["edge_id", "route", "reg", "time", "rep"]]
    out = {}
    for lab, _ in runs:
        m = E[lab].merge(obs, on="edge_id")
        g = m.groupby(["route", "reg"]).agg(n=("rep", "size"), rep=("rep", "sum"),
                                            post=("post", "sum"))
        g["ratio"] = g.post / g.rep
        out[lab] = g
    print(pd.concat({lab: out[lab][["rep", "post", "ratio"]] for lab in out},
                    axis=1).round(2).to_string())
    for lab, _ in runs:
        g = out[lab]
        lo50 = sorted(g.index[g.ratio < 0.50].tolist())
        lo05 = sorted(g.index[g.ratio < 0.05].tolist())
        print(f"   {lab:5s} world post/reported {g.post.sum() / g.rep.sum():.3f} | "
              f"below 0.50: {len(lo50)} of {len(g)} {lo50} | "
              f"**below 0.05 (route closed): {len(lo05)} {lo05}**")
        m = E[lab].merge(obs, on="edge_id")
        dead = m[(m.rep > 1.0) & (m.post < 0.05 * m.rep)]
        print(f"         region-years reported > 1 kt but posterior < 5 % of it: {len(dead)}"
              + (f"  {dead.groupby(['route', 'reg']).size().to_dict()}" if len(dead) else ""))
        up   = float((g.post - g.rep).clip(lower=0).sum())
        down = float((g.rep - g.post).clip(lower=0).sum())
        print(f"         pushed UP {up:,.0f} kt / pushed DOWN {down:,.0f} kt "
              f"(direction of the secondary-share move)")

    # 8b. Where the direct route is NOT reported: anything there is invented.
    print("\n   8b. Direct route where ICSG reports a SMELTER route but no direct one")
    sm = set(zip(obs[obs.route.eq("Smelting")].reg, obs[obs.route.eq("Smelting")].time))
    unrep = ref[ref.route.eq("CFR") & ~(ref["cov"] & (ref.rep.abs() > 1e-9))]
    unrep = unrep[[(r.reg, r.time) in sm for r in unrep.itertuples()]]
    print(f"       {len(unrep)} region-years (ADR 0030's nil-route gap is inside this set)")
    for lab, _ in runs:
        m = E[lab].merge(unrep[["edge_id", "reg"]], on="edge_id")
        tot = m.post.sum()
        top = m.groupby("reg").post.sum().sort_values(ascending=False).head(5)
        print(f"       {lab:5s} invented direct scrap {tot:,.0f} kt | top: "
              + ", ".join(f"{k} {v:,.0f}" for k, v in top.items()))

    # 8c. Can the hard band even accommodate the anchored level?
    print("\n   8c. Band-vs-level headroom (posterior node input x ceiling) / reported")
    par = pd.read_csv("data/input/params/scrap_share/current/params.csv", sep=";")
    par["route"] = np.where(par.a_sector_to.eq("Smelting"), "Smelting", "CFR")
    bu = par.set_index(["route", "a_region_to"]).value_max
    lab, _ = runs[-1]
    tin = (E[lab][E[lab].sector_to.isin(["Smelting", "CFR"])]
           .groupby(["sector_to", "region_to", "time"]).post.sum()
           .rename_axis(["route", "reg", "time"]).rename("tin"))
    d = obs.merge(tin, on=["route", "reg", "time"], how="left")
    d["bu"] = [bu.get((r.route, r.reg), np.nan) for r in d.itertuples()]
    d["blocked"] = d.rep > d.tin * d.bu * 1.001
    g = d.groupby(["route", "reg"]).agg(rep=("rep", "sum"), tin=("tin", "sum"),
                                        bu=("bu", "first"), blocked=("blocked", "sum"))
    g["headroom"] = (g.tin * g.bu) / g.rep
    print(g[g.headroom < 1.5].sort_values("headroom").round(3).to_string())
    print("       headroom < 1 => the hard band binds before the level can be met; "
          "a ceiling of 1.000 means the node need only grow, which the anchor does.")


def main(argv: list) -> None:
    feed_only = "--feed-only" in argv
    ids = [a for a in argv if not a.startswith("--")]
    print("=" * 78)
    print("Round O — secondary-smelter copper enters the chain once   (ADR 0030)")
    print("Round P — the ICSG scrap series anchor their level           (ADR 0031)")
    feed_gate(ids[0])
    if feed_only or len(ids) < 2:
        return
    LABELS = ["base", "O", "P", "Q", "R"]
    runs = [(LABELS[i] if i < len(LABELS) else f"r{i}", rid) for i, rid in enumerate(ids)]
    E = {lab: M.load_run(r) for lab, r in runs}
    E = {lab: e[e.valuation == "CC"] for lab, e in E.items()}
    from pysuto.acc import AccountingStructure
    from pysuto.config import load_config
    from pysuto.param_relation import expand_edge_index
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=ROOT / "acc")
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    stage2_attribution(runs, ei)
    netting_by_sector(runs)
    secondary_and_bands(runs, E)
    mine_chain(runs, E)
    export_mass_gap(runs)
    stock_buffers(runs, E)
    headline(runs, E)
    scrap_level_acceptance(runs, E)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
