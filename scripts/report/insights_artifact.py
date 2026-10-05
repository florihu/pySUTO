"""
insights_artifact.py
====================
Data block of the "Copper Metabolism 1995–2024" insight artifact, read off one
opt run: absolute process outputs and system-boundary flows, a World Sankey with
the traded part of every commodity split out, scrap-versus-primary ratios,
concentration by chain step and the bilateral structure of the four traded
commodities. Every value is copper content in kt Cu and posterior unless the key
says ``_prior``.

    python scripts/report/insights_artifact.py <opt_run_id> [--years 1995 2010 2024]
                                               [--out <json>]

Reads ``data/proc/opt/<run>/diag/edges_<year>.csv`` (CC rows only) and the run's
``metadata.json``; writes one JSON file (default
``docs/rep/figs/insights_<run>/data.json``) and prints the headline numbers the
page's prose cites, so they can be re-checked per run. Nothing under ``data/`` is
written.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
ROOT       = Path("data/proc")
YEARS      = list(range(1995, 2025))
SNAPSHOTS  = [1995, 2010, 2024]     # years embedded for the Sankey and the trade figures
TOP_N      = 20                     # named regions in the trade figures; the rest is pooled
MIN_KT     = 300.0                  # scrap-share bars: regions whose feed exceeds this
LEG_MIN_KT = 0.1                    # bilateral legs below this are not written
EU14 = ["Germany", "Italy", "France", "Spain", "Poland", "Belgium", "Netherlands",
        "Austria", "Sweden", "Bulgaria", "Greece", "Portugal", "Slovakia", "Luxembourg"]
TRADED = ["Concentrate", "Matte", "Anode", "Cathode"]
# scrap-chapter groups (label → member region ids); World and Rest of world are added
SCRAP_GROUPS = {"China":         ["China"],
                "EU-14":         EU14,
                "United States": ["United States"],
                "Japan":         ["Japan"],
                "South Korea":   ["Korea, Rep."],
                "Taiwan":        ["Taiwan"]}
# ---------------------------------------------------------------------------
# END USER CONFIGURATION
# ---------------------------------------------------------------------------

# (label, from_sector, to_sector): the output Flow edge of each process
PROCESS_OUT = [("Mining (ore)",   "Mining",   "Ore"),
               ("Concentrate",    "MF",       "Concentrate"),
               ("SX-EW cathode",  "SXEW",     "CSXEW"),
               ("Smelter matte",  "Smelting", "Matte"),
               ("Anode",          "CFR",      "Anode"),
               ("Electrorefined", "ER",       "CER"),
               ("Cathode",        "CP",       "Cathode")]
SB_IN  = [("Ore in situ (nature)",  "CO",       "Mining"),
          ("Scrap to smelters",     "Scrap",    "Smelting"),
          ("Scrap to refineries",   "Scrap",    "CFR")]
SB_OUT = [("Final use",             "Cathode",  "FU"),
          ("Exchange stock change", "Cathode",  "MESC"),
          ("User stock change",     "Cathode",  "USC"),
          ("Producer stock change", "CP",       "PSC"),
          ("Tailings (mill)",       "MF",       "Tailings"),
          ("Tailings (SX-EW)",      "SXEW",     "Tailings"),
          ("Slag (smelting)",       "Smelting", "Slag"),
          ("Slag (fire ref.)",      "CFR",      "Slag"),
          ("Slag (electroref.)",    "ER",       "Slag"),
          ("Waste rock",            "Mining",   "WR")]
# step → (from_sector, to_sector, region column that owns the step)
CONC_STEPS = {"Mining":          ("Mining",   "Ore",         "from_region"),
              "Concentrate":     ("MF",       "Concentrate", "from_region"),
              "Smelting":        ("Smelting", "Matte",       "from_region"),
              "Refined cathode": ("CP",       "Cathode",     "from_region"),
              "Final use":       ("Cathode",  "FU",          "to_region")}

# Sankey nodes: id → (label, layer, kind). Layers are the columns of the drawing.
SANKEY_NODES = {
    "CO":       ("Ore in situ",                0, "ore"),
    "Mining":   ("Mining",                     1, "process"),
    "MF":       ("Milling & flotation",        2, "process"),
    "SXEW":     ("SX-EW leaching",             2, "process"),
    "Scrap":    ("Scrap",                      2, "scrap"),
    "Smelting": ("Smelting",                   3, "process"),
    "Tailings": ("Tailings",                   3, "loss"),
    "CFR":      ("Converting & fire refining", 4, "process"),
    "ER":       ("Electrorefining",            5, "process"),
    "CP":       ("Cathode pool",               6, "process"),
    "Slag":     ("Slag",                       6, "loss"),
    "FU":       ("Final use",                  7, "use"),
    "MESC":     ("Exchange stocks",            7, "stock"),
    "USC":      ("User stocks",                7, "stock"),
    "PSC":      ("Producer stocks",            7, "stock"),
}
# Sankey links: (source, target, label, kind, (from_sector, to_sector), split traded?)
# Commodity Flow nodes are contracted into the link between the producing and the
# using process; exact at World level because every Flow has one producing process.
SANKEY_LINKS = [
    ("CO",       "Mining",   "Ore in situ",           "ore",         ("CO", "Mining"),            False),
    ("Mining",   "MF",       "Ore",                   "ore",         ("Ore", "MF"),               False),
    ("Mining",   "SXEW",     "Ore",                   "ore",         ("Ore", "SXEW"),             False),
    ("MF",       "Smelting", "Concentrate",           "concentrate", ("Concentrate", "Smelting"), True),
    ("Smelting", "CFR",      "Matte",                 "matte",       ("Matte", "CFR"),            True),
    ("CFR",      "ER",       "Anode",                 "anode",       ("Anode", "ER"),             True),
    ("ER",       "CP",       "Refined copper (ER)",   "refined",     ("CER", "CP"),               False),
    ("SXEW",     "CP",       "SX-EW cathode",         "refined",     ("CSXEW", "CP"),             False),
    ("CP",       "FU",       "Cathode",               "cathode",     ("Cathode", "FU"),           True),
    ("CP",       "MESC",     "Cathode",               "stock",       ("Cathode", "MESC"),         False),
    ("CP",       "USC",      "Cathode",               "stock",       ("Cathode", "USC"),          False),
    ("CP",       "PSC",      "Cathode",               "stock",       ("CP", "PSC"),               False),
    ("Scrap",    "Smelting", "Scrap",                 "scrap",       ("Scrap", "Smelting"),       False),
    ("Scrap",    "CFR",      "Scrap",                 "scrap",       ("Scrap", "CFR"),            False),
    ("MF",       "Tailings", "Tailings",              "loss",        ("MF", "Tailings"),          False),
    ("SXEW",     "Tailings", "Tailings",              "loss",        ("SXEW", "Tailings"),        False),
    ("Smelting", "Slag",     "Slag",                  "loss",        ("Smelting", "Slag"),        False),
    ("CFR",      "Slag",     "Slag",                  "loss",        ("CFR", "Slag"),             False),
    ("ER",       "Slag",     "Slag",                  "loss",        ("ER", "Slag"),              False),
]


# ---------------------------------------------------------------------------
# loaders and small helpers
# ---------------------------------------------------------------------------

def load_cc(opt: str) -> pd.DataFrame:
    """Every CC edge of every year with its decoded address, prior and posterior."""
    frames = []
    for y in YEARS:
        e = pd.read_csv(ROOT / "opt" / opt / "diag" / f"edges_{y}.csv")
        e = e[e.valuation == "CC"].copy()
        e["year"] = y
        frames.append(e)
    cc = pd.concat(frames, ignore_index=True)
    return cc[["year", "from_sector", "from_region", "from_entity",
               "to_sector", "to_region", "to_entity", "a_prior", "a_post"]]


def meta_block(opt: str) -> dict:
    om    = json.load(open(ROOT / "opt" / opt / "metadata.json"))
    const = om["parents"]["upstream"]["const_run_id"]
    cm    = json.load(open(ROOT / "const" / const / "metadata.json"))
    up    = cm["parents"]["upstream"]
    imp   = next((v for k, v in up.items() if k.startswith("imp")), None)
    per   = om["stage_summary"]["per_timestep"]
    status = {}
    for v in per.values():
        status[v["solver_status"]] = status.get(v["solver_status"], 0) + 1
    return {"run": opt, "label": om.get("label", ""), "created_at": om.get("created_at"),
            "const": const, "imp": imp, "years": YEARS, "snapshots": SNAPSHOTS,
            "status": status}


def flow(cc: pd.DataFrame, sf: str, st: str) -> pd.DataFrame:
    return cc[(cc.from_sector == sf) & (cc.to_sector == st)]


def by_year(df: pd.DataFrame, col: str = "a_post") -> pd.Series:
    return df.groupby("year")[col].sum().reindex(YEARS).fillna(0.0)


def by_year_region(df: pd.DataFrame, col: str, val: str = "a_post") -> pd.DataFrame:
    """year × region table of one edge family."""
    return df.groupby(["year", col])[val].sum().unstack(col).reindex(YEARS).fillna(0.0)


def ser(s, nd: int = 1) -> list:
    """Rounded list for the JSON; a non-finite value (0/0 ratio) becomes null."""
    return [round(float(v), nd) if math.isfinite(float(v)) else None for v in s]


def group_conc(region: str) -> str:
    if region == "China":
        return "China"
    if region in EU14:
        return "EU-14"
    if region in ("United States", "Chile", "Peru"):
        return region
    return "Rest of world"


def group_scrap(region: str) -> str:
    g = group_conc(region)
    return g if g in ("China", "EU-14", "United States") else "Rest of world"


def scrap_members(group: str, regions) -> list:
    if group == "World":
        return list(regions)
    if group == "Rest of world":
        named = {r for m in SCRAP_GROUPS.values() for r in m}
        return [r for r in regions if r not in named]
    return [r for r in SCRAP_GROUPS[group] if r in regions]


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------

def series_blocks(cc: pd.DataFrame) -> dict:
    out = {"process_out": {}, "sb_in": {}, "sb_out": {}}
    for label, sf, st in PROCESS_OUT:
        out["process_out"][label] = ser(by_year(flow(cc, sf, st)))
    for label, sf, st in SB_IN:
        out["sb_in"][label] = ser(by_year(flow(cc, sf, st)))
    for label, sf, st in SB_OUT:
        out["sb_out"][label] = ser(by_year(flow(cc, sf, st)))
    return out


def scrap_block(cc: pd.DataFrame) -> dict:
    scrap = cc[cc.from_sector == "Scrap"]
    prim  = cc[((cc.from_sector == "Concentrate") & (cc.to_sector == "Smelting"))
               | ((cc.from_sector == "SXEW") & (cc.to_sector == "CSXEW"))]
    regions = sorted(cc.from_region.unique())
    tabs = {v: (by_year_region(scrap, "to_region", v).reindex(columns=regions).fillna(0.0),
                by_year_region(prim, "to_region", v).reindex(columns=regions).fillna(0.0))
            for v in ("a_post", "a_prior")}
    out = {}
    for g in ["World"] + list(SCRAP_GROUPS) + ["Rest of world"]:
        m = scrap_members(g, regions)
        s, p = tabs["a_post"][0][m].sum(1), tabs["a_post"][1][m].sum(1)
        sp, pp = tabs["a_prior"][0][m].sum(1), tabs["a_prior"][1][m].sum(1)
        out[g] = {"scrap": ser(s), "primary": ser(p), "ratio": ser(s / p, 4),
                  "share": ser(s / (s + p), 4), "ratio_prior": ser(sp / pp, 4)}
    s24, p24 = tabs["a_post"][0].loc[2024], tabs["a_post"][1].loc[2024]
    keep = (s24 + p24) > MIN_KT
    rows = pd.DataFrame({"scrap": s24[keep], "primary": p24[keep]})
    rows["share"] = rows.scrap / (rows.scrap + rows.primary)
    rows = rows.sort_values("share", ascending=False)
    out_regions = [{"region": r, "share": round(float(x.share), 4), "scrap": round(float(x.scrap), 1),
                    "primary": round(float(x.primary), 1)} for r, x in rows.iterrows()]
    return {"scrap": out, "scrap_regions_2024": out_regions}


def conc_block(cc: pd.DataFrame) -> dict:
    tables = {step: by_year_region(flow(cc, sf, st), col)
              for step, (sf, st, col) in CONC_STEPS.items()}
    tables["Scrap feed"] = by_year_region(cc[cc.from_sector == "Scrap"], "to_region")
    out = {}
    for step, t in tables.items():
        world = t.sum(1)
        sh = t.div(world, axis=0)
        groups = sh.T.groupby(group_conc).sum().T
        out[step] = {
            "world": ser(world), "hhi": ser((sh ** 2).sum(1), 4),
            "top3": ser(sh.apply(lambda r: r.nlargest(3).sum(), axis=1), 4),
            "leader": sh.idxmax(1).tolist(), "leader_share": ser(sh.max(1), 4),
            "groups": {g: ser(groups[g], 4) for g in
                       ["China", "EU-14", "United States", "Chile", "Peru", "Rest of world"]},
            "top_1995": [{"region": r, "share": round(float(v), 4)}
                         for r, v in sh.loc[1995].nlargest(8).items()],
            "top_2024": [{"region": r, "share": round(float(v), 4)}
                         for r, v in sh.loc[2024].nlargest(8).items()],
        }
    # imported share of smelter concentrate intake; "imported" = from outside the group
    cs = flow(cc, "Concentrate", "Smelting").copy()
    cs["g_to"], cs["g_from"] = cs.to_region.map(group_scrap), cs.from_region.map(group_scrap)
    imp = {}
    for g in ("China", "EU-14", "United States"):
        x = cs[cs.g_to == g]
        tot = by_year(x)
        ext = by_year(x[x.g_from != g])
        imp[g] = {"import_share": ser(ext / tot, 4), "intake": ser(tot), "imported": ser(ext)}
    return {"conc": out, "conc_import": imp}


def sankey_block(cc: pd.DataFrame, years: list) -> dict:
    out = {}
    for y in years:
        d = cc[cc.year == y]
        links = []
        for src, tgt, label, kind, (sf, st), split in SANKEY_LINKS:
            x = flow(d, sf, st)
            parts = [(False, x)] if not split else [
                (False, x[x.from_region == x.to_region]), (True, x[x.from_region != x.to_region])]
            for traded, part in parts:
                v = float(part.a_post.sum())
                if abs(v) < 1e-9:
                    continue
                flipped = v < 0
                links.append({"source": tgt if flipped else src, "target": src if flipped else tgt,
                              "label": label + (" (destocking)" if flipped else ""),
                              "kind": kind, "traded": traded, "value": round(abs(v), 1),
                              "flipped": flipped})
        nodes = [{"id": k, "label": lab, "layer": lay, "kind": kind}
                 for k, (lab, lay, kind) in SANKEY_NODES.items()]
        out[str(y)] = {"nodes": nodes, "links": links}
    return out


def trade_block(cc: pd.DataFrame, years: list) -> dict:
    x = cc[cc.from_sector.isin(TRADED) & (cc.from_region != cc.to_region)]
    y24 = x[x.year == 2024]
    gross = y24.groupby("from_region").a_post.sum().add(
        y24.groupby("to_region").a_post.sum(), fill_value=0.0)
    order = gross.sort_values(ascending=False).index[:TOP_N].tolist()
    pooled = sorted(set(cc.from_region.unique()) - set(order))
    cols = order + ["Other"]
    out = {"order": cols, "pooled": pooled, "commodities": TRADED, "years": {}}
    for y in years:
        d = x[x.year == y].copy()
        d["f"] = d.from_region.where(d.from_region.isin(order), "Other")
        d["t"] = d.to_region.where(d.to_region.isin(order), "Other")
        legs, other_other, exports, imports = {}, {}, {}, {}
        for c in TRADED:
            dc = d[d.from_sector == c]
            g = dc.groupby(["f", "t"]).a_post.sum()
            oo = float(g.get(("Other", "Other"), 0.0))
            other_other[c] = round(oo, 1)
            g = g[(g >= LEG_MIN_KT)]
            legs[c] = [{"f": f, "t": t, "v": round(float(v), 1)}
                       for (f, t), v in g.items() if not (f == "Other" and t == "Other")]
            exports[c] = dc.groupby("f").a_post.sum().reindex(cols).fillna(0.0)
            imports[c] = dc.groupby("t").a_post.sum().reindex(cols).fillna(0.0)
        regions = {}
        for r in cols:
            regions[r] = {"exports": {c: round(float(exports[c][r]), 1) for c in TRADED},
                          "imports": {c: round(float(imports[c][r]), 1) for c in TRADED}}
            regions[r]["net"] = round(sum(regions[r]["exports"].values())
                                      - sum(regions[r]["imports"].values()), 1)
        out["years"][str(y)] = {"legs": legs, "other_other": other_other, "regions": regions}
    ts = {}
    for c in TRADED:
        ts[c] = {"trade": ser(by_year(x[x.from_sector == c])),
                 "output": ser(by_year(cc[(cc.to_sector == c) & (cc.to_entity == "Flow")]))}
    out["ts"] = ts
    return out


# ---------------------------------------------------------------------------
# checks and headline numbers
# ---------------------------------------------------------------------------

def report(cc: pd.DataFrame, data: dict) -> None:
    N = len(YEARS)
    P, SI, SO = data["process_out"], data["sb_in"], data["sb_out"]
    print("== process outputs, kt Cu: 1995 → 2024 (×)")
    for k, v in P.items():
        print(f"  {k:16s} {v[0]:9.0f} → {v[N-1]:9.0f}  ×{v[N-1]/v[0]:.2f}")
    tot_in  = [sum(SI[k][i] for k in SI) for i in range(N)]
    tot_out = [sum(SO[k][i] for k in SO) for i in range(N)]
    gap = max(abs(a - b) for a, b in zip(tot_in, tot_out))
    print(f"== boundary: in 1995 {tot_in[0]:.0f} / 2024 {tot_in[N-1]:.0f} kt; "
          f"max |in − out| over years = {gap:.3g} kt")
    for k in SI:
        print(f"  in  {k:24s} {SI[k][0]:9.0f} → {SI[k][N-1]:9.0f}")
    for k in SO:
        print(f"  out {k:24s} {SO[k][0]:9.0f} → {SO[k][N-1]:9.0f}")
    print("== Sankey node balance (inflow − outflow, kt) per snapshot year")
    for y, g in data["sankey"].items():
        bal = {}
        for l in g["links"]:
            bal[l["target"]] = bal.get(l["target"], 0.0) + l["value"]
            bal[l["source"]] = bal.get(l["source"], 0.0) - l["value"]
        proc = {k: round(v, 1) for k, v in bal.items() if SANKEY_NODES[k][2] == "process"}
        print(f"  {y}: {proc}")
    S = data["scrap"]
    print("== scrap / primary 2024: " + ", ".join(
        f"{g} {S[g]['ratio'][N-1]:.2f} (prior {S[g]['ratio_prior'][N-1]:.2f})" for g in S))
    print(f"   world ratio 1995 {S['World']['ratio'][0]:.2f}, min "
          f"{min(S['World']['ratio']):.2f} in {YEARS[S['World']['ratio'].index(min(S['World']['ratio']))]}")
    C = data["conc"]
    print("== concentration 2024 (1995): HHI, leader, China share")
    for st, v in C.items():
        print(f"  {st:16s} HHI {v['hhi'][N-1]:.3f} ({v['hhi'][0]:.3f})  leader {v['leader'][N-1]} "
              f"{v['leader_share'][N-1]*100:.0f} %  China {v['groups']['China'][N-1]*100:.0f} % "
              f"({v['groups']['China'][0]*100:.0f} %)  EU-14 {v['groups']['EU-14'][N-1]*100:.0f} % "
              f"({v['groups']['EU-14'][0]*100:.0f} %)  US {v['groups']['United States'][N-1]*100:.0f} % "
              f"({v['groups']['United States'][0]*100:.0f} %)")
    I = data["conc_import"]
    print("== imported share of smelter concentrate intake 2024 (1995): " + ", ".join(
        f"{g} {I[g]['import_share'][N-1]*100:.0f} % ({I[g]['import_share'][0]*100:.0f} %)" for g in I))
    T = data["trade"]
    print("== trade: world cross-region trade per commodity, kt Cu and share of output")
    for c in TRADED:
        tr, op = T["ts"][c]["trade"], T["ts"][c]["output"]
        print(f"  {c:12s} " + "  ".join(f"{y}: {tr[YEARS.index(y)]:7.0f} ({tr[YEARS.index(y)]/op[YEARS.index(y)]*100:4.1f} %)"
                                        for y in SNAPSHOTS))
    print(f"   region order: {T['order']}")
    for y, g in T["years"].items():
        print(f"  -- {y}: legs between pooled regions (not drawn) {g['other_other']}")
        for c in TRADED:
            top = sorted(g["legs"][c], key=lambda l: -l["v"])[:6]
            print(f"     {c:12s} " + "; ".join(f"{l['f']}→{l['t']} {l['v']:.0f}" for l in top))
        ex = sorted(g["regions"].items(), key=lambda kv: -sum(kv[1]["exports"].values()))[:5]
        im = sorted(g["regions"].items(), key=lambda kv: -sum(kv[1]["imports"].values()))[:5]
        print("     top exporters " + ", ".join(f"{r} {sum(v['exports'].values()):.0f}" for r, v in ex))
        print("     top importers " + ", ".join(f"{r} {sum(v['imports'].values()):.0f}" for r, v in im))
    x = cc[(cc.from_sector == "Concentrate") & (cc.from_region == "Chile")
           & (cc.to_region != "Chile") & (cc.year == 2024)]
    print(f"== Chile concentrate exports 2024: {x.a_post.sum():.0f} kt, to China "
          f"{x[x.to_region == 'China'].a_post.sum() / x.a_post.sum() * 100:.0f} %; "
          f"retention Σpost/Σprior {x.a_post.sum() / x.a_prior.sum():.3f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("opt", help="opt run id")
    ap.add_argument("--years", nargs="+", type=int, default=SNAPSHOTS,
                    help="snapshot years for the Sankey and the trade figures")
    ap.add_argument("--out", default=None, help="JSON path (default docs/rep/figs/insights_<run>/data.json)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path("docs/rep/figs") / f"insights_{args.opt}" / "data.json"

    cc = load_cc(args.opt)
    data = {"meta": meta_block(args.opt)}
    data.update(series_blocks(cc))
    data.update(scrap_block(cc))
    data.update(conc_block(cc))
    data["sankey"] = sankey_block(cc, args.years)
    data["trade"]  = trade_block(cc, args.years)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB); run {args.opt} "
          f"label {data['meta']['label']!r}; status {data['meta']['status']}")
    report(cc, data)


if __name__ == "__main__":
    main()
