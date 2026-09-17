"""
copper_metabolism_insights.py
=============================
Three insights into the global copper supply chain, read off the latest run.

    python scripts/report/copper_metabolism_insights.py [--opt <opt_run_id>]

  1  Secondary copper is ~17 % of refined output and flat since 2010; the model
     showed 30 % because ICSG's secondary-smelter series is already inside its
     secondary-refined series and the parse books both as inputs — and that
     double count is, arithmetically, the world mine-chain deletion.
  2  The mine→smelter link has moved to China and is where the reconciliation
     discharges: trade-dependent smelters lose 20-35 %, self-sufficient hold.
  3  The in-use stock has moved east; scrap is untraded in the model and the
     stock-change nodes are used as buffers.

Writes tables (csv), figures (png) and data.json (the tables, for the
interactive page) to docs/rep/figs/metabolism_<opt>/. Reads persisted run
artifacts and the registered feeds only; every upstream id comes from the run's
own sidecar (the Round C hygiene rule). Nothing is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
sys.path.insert(0, "scripts/report")

from pysuto.acc import AccountingStructure                      # noqa: E402
from pysuto.config import load_config                           # noqa: E402
from pysuto.input_registry import InputRegistries               # noqa: E402
from pysuto.param_relation import expand_edge_index             # noqa: E402
import india_concentrate_diag as D                              # noqa: E402
import roundN_concentrate_grade_diag as N                       # noqa: E402
import supervisor_figs_20260907 as SF                           # noqa: E402

ROOT = Path("data/proc")
DEFAULT_RUN = "opt_20260916_124721_c4d1d944"
YEARS = list(range(1995, 2025))
YEARS_S = [str(y) for y in YEARS]
POOL = [str(y) for y in range(2020, 2025)]        # the "recent" pool

EU14 = ["Germany", "Italy", "France", "Spain", "Poland", "Belgium", "Netherlands",
        "Austria", "Sweden", "Bulgaria", "Greece", "Portugal", "Slovakia", "Luxembourg"]
GROUPS = ["China", "EU-14", "United States", "JP/KR/TW", "India", "Chile+Peru", "Rest"]


def group_of(region: str) -> str:
    if region == "China":
        return "China"
    if region in EU14:
        return "EU-14"
    if region == "United States":
        return "United States"
    if region in ("Japan", "Korea, Rep.", "Taiwan"):
        return "JP/KR/TW"
    if region == "India":
        return "India"
    if region in ("Chile", "Peru"):
        return "Chile+Peru"
    return "Rest"


# UN World Population Prospects 2024, mid-2024, millions, rounded. Used only for
# the per-capita indicator in T3b (cumulative cathode inflow to fabricators).
POP_2024 = {"China": 1419, "United States": 345, "Japan": 124, "Germany": 84,
            "Korea, Rep.": 52, "Italy": 59, "Russia": 145, "India": 1451,
            "Mexico": 131, "Brazil": 212, "Spain": 48}

# dataviz reference palette (light mode, fixed order); slots 1-7 are SF.PALETTE
PALETTE = SF.PALETTE + ["#e34948"]
GREY, INK, INK2, GRID = "#b3b2a9", SF.INK, SF.INK2, "#e1e0d9"
SURFACE = "#fcfcfb"
GROUP_COLOR = dict(zip(GROUPS, PALETTE[:6] + [GREY]))

plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_eff() -> dict:
    """Efficiency nominals from the registered eff params — never hard-coded."""
    inputs = InputRegistries.initialize(root="data/input")
    p = pd.read_csv(inputs.params.resolve("eff") / "params.csv", sep=None,
                    engine="python", comment="#", dtype=str)
    e = {r["param_id"]: float(r["value_nominal"]) for _, r in p.iterrows()
         if r.get("param_type") == "efficiency"}
    return {"sm": e["effsmelting"], "cfr": e["effcfr"], "er": e["effer"],
            "sxew": e["effsxew"], "mf": e["effmfflotation"]}


def load_run(opt: str) -> pd.DataFrame:
    """Every edge of every year: prior, posterior, decoded address."""
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=ROOT / "acc")
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    rows = []
    for y in YEARS:
        o = np.load(ROOT / "opt" / opt / "timesteps" / f"t_{y}.npz", allow_pickle=True)
        rows.append(pd.DataFrame({"edge_id": np.asarray(o["cols_t"], dtype=int),
                                  "prior": o["a_prior"], "post": o["F_opt"]}))
    e = pd.concat(rows, ignore_index=True).merge(
        ei[["edge_id", "sector_from", "region_from", "entity_from", "sector_to",
            "region_to", "entity_to", "valuation", "time"]], on="edge_id")
    e["year"] = e["time"].astype(int)
    return e


def load_icsg() -> pd.DataFrame:
    """The ICSG feed rows with the root region attached (input divisors applied)."""
    inputs = InputRegistries.initialize(root="data/input")
    fdir = inputs.feed.resolve("icsg")
    raw = pd.read_csv(fdir / "values.csv")
    rmap = pd.read_excel(fdir / "conc.xlsx", sheet_name="region_map")
    raw["reg"] = raw["Region"].map(dict(zip(rmap["src_region"], rmap["root_region"])))
    return raw


def lineage(opt: str) -> dict:
    const, imp = D._lineage(opt)
    m = json.load(open(ROOT / "const" / const / "metadata.json"))
    versions = {p["id"]: p["version"] for p in m["parents"]["inputs"].get("params", [])}
    feeds = json.load(open(ROOT / "imp" / imp / "metadata.json"))["parents"]["upstream"]["feed_run_ids"]
    om = json.load(open(ROOT / "opt" / opt / "metadata.json"))
    per = om["stage_summary"]["per_timestep"]
    return {"opt": opt, "label": om.get("label"), "const": const, "imp": imp,
            "params": versions, "feeds": feeds,
            "objective": float(sum(v["objective"] for v in per.values())),
            "solved": sum(v["solver_status"] == "Solved" for v in per.values())}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def flow(cc: pd.DataFrame, sf: str, st: str) -> pd.DataFrame:
    return cc[(cc.sector_from == sf) & (cc.sector_to == st)]


def by_year(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("year")[["prior", "post"]].sum().reindex(YEARS).fillna(0.0)


def icsg_world(icsg: pd.DataFrame, eff: dict) -> pd.DataFrame:
    """ICSG's published world series, kt Cu, recovered from the feed's input rows."""
    def ser(process, typ):
        x = icsg[(icsg.Process == process) & (icsg.Type == typ)]
        return x.groupby("Year")["Value"].sum().reindex(YEARS).fillna(0.0)
    w = pd.DataFrame({
        "scrap_in_smelting": ser("Smelting", "Scrap"),          # feed input rows
        "scrap_in_refining": ser("Refining", "Scrap"),
        "anode_in_er": ser("Refining", "Anode"),
        "sxew": ser("SXEW", "Cathode"),
        "mine_conc": ser("Mining", "Concentrate"),
    })
    w["S_smelter_prod"] = w.scrap_in_smelting * eff["sm"]                  # published
    w["S_refined_prod"] = w.scrap_in_refining * eff["cfr"] * eff["er"]     # published
    w["total_refined"] = w.anode_in_er * eff["er"]                         # published (P+S)
    w["P_primary_refined"] = w.total_refined - w.S_refined_prod
    return w


# ---------------------------------------------------------------------------
# tables — insight 1
# ---------------------------------------------------------------------------

def t0_world_accounts(cc: pd.DataFrame) -> pd.DataFrame:
    classes = {
        "mine_conc": ("MF", "Concentrate"), "sxew": ("SXEW", "CSXEW"),
        "scrap_to_smelting": ("Scrap", "Smelting"), "scrap_to_cfr": ("Scrap", "CFR"),
        "conc_to_smelting": ("Concentrate", "Smelting"), "matte_to_cfr": ("Matte", "CFR"),
        "anode_to_er": ("Anode", "ER"), "er_out": ("ER", "CER"),
        "cathode": ("CP", "Cathode"), "psc": ("CP", "PSC"),
        "final_use": ("Cathode", "FU"), "mesc": ("Cathode", "MESC"), "usc": ("Cathode", "USC"),
        "tailings_mf": ("MF", "Tailings"), "tailings_sxew": ("SXEW", "Tailings"),
        "slag_smelting": ("Smelting", "Slag"), "slag_cfr": ("CFR", "Slag"),
        "slag_er": ("ER", "Slag"), "ore_in_situ": ("CO", "Mining"),
    }
    out = []
    for k, (a, b) in classes.items():
        s = by_year(flow(cc, a, b))
        out.append(s.rename(columns={"prior": f"{k}_prior", "post": f"{k}_post"}))
    return pd.concat(out, axis=1)


def t1a_secondary_world(icsg: pd.DataFrame, eff: dict, cc: pd.DataFrame) -> pd.DataFrame:
    """ICSG as published vs both series booked as inputs, and the concentrate gap."""
    w = icsg_world(icsg, eff)
    k = eff["cfr"] * eff["er"]
    w["sec_via_smelters"] = w.S_smelter_prod * k             # refined-Cu equivalent
    w["sec_direct_to_refinery"] = w.S_refined_prod - w.sec_via_smelters
    w["secondary_both_booked"] = w.sec_via_smelters + w.S_refined_prod
    w["refined_plus_sxew"] = w.total_refined + w.sxew
    w["share_nested"] = w.S_refined_prod / w.refined_plus_sxew
    w["share_both_booked"] = w.secondary_both_booked / w.refined_plus_sxew
    # ADR 0021: Matte = P/(eff_CFR·eff_ER); conc = Matte/eff_Sm − Scrap→Smelting
    w["conc_needed_both"] = (w.P_primary_refined / k) / eff["sm"] - w.scrap_in_smelting
    w["conc_needed_nested"] = ((w.P_primary_refined + w.sec_via_smelters) / k) / eff["sm"] \
        - w.scrap_in_smelting
    w["gap_both"] = w.mine_conc - w.conc_needed_both
    w["gap_nested"] = w.mine_conc - w.conc_needed_nested
    # the model's own secondary share (refinery-side input basis), prior and post
    sec = by_year(flow(cc, "Scrap", "Smelting")) + by_year(flow(cc, "Scrap", "CFR"))
    prim = by_year(flow(cc, "Concentrate", "Smelting")) + by_year(flow(cc, "SXEW", "CSXEW"))
    w["model_share_prior"] = sec.prior / (sec.prior + prim.prior)
    w["model_share_post"] = sec.post / (sec.post + prim.post)
    return w


def t1b_nesting_test(icsg: pd.DataFrame, eff: dict, cc: pd.DataFrame) -> pd.DataFrame:
    """Per region: secondary smelter vs secondary refined, both-booked vs nested share."""
    k = eff["cfr"] * eff["er"]
    rows = []
    for r in ["Austria", "Belgium", "Germany", "China", "Japan"]:
        x = icsg[icsg.reg == r]
        sm = x[(x.Process == "Smelting") & (x.Type == "Scrap")].groupby("Year")["Value"].sum() * eff["sm"]
        rf = x[(x.Process == "Refining") & (x.Type == "Scrap")].groupby("Year")["Value"].sum() * k
        tot = x[(x.Process == "Refining") & (x.Type == "Anode")].groupby("Year")["Value"].sum() * eff["er"]
        for y in (2000, 2010, 2019, 2024):
            if y not in tot.index:
                continue
            rows.append(dict(region=r, year=y, sec_smelter_prod=sm.get(y, np.nan),
                             sec_refined_prod=rf.get(y, np.nan), total_refined=tot[y],
                             both_booked_share=(sm.get(y, 0.0) * k + rf.get(y, 0.0)) / tot[y],
                             nested_share=rf.get(y, np.nan) / tot[y]))
    t = pd.DataFrame(rows)
    # what the solver did with Austria's inputs (2024)
    a = cc[(cc.year == 2024) & (cc.region_to == "Austria")]
    for lab, (sf, st) in {"scrap_to_smelting": ("Scrap", "Smelting"),
                          "scrap_to_cfr": ("Scrap", "CFR"), "anode_to_er": ("Anode", "ER")}.items():
        s = flow(a, sf, st)
        t.loc[(t.region == "Austria") & (t.year == 2024), f"post_{lab}"] = s.post.sum()
        t.loc[(t.region == "Austria") & (t.year == 2024), f"prior_{lab}"] = s.prior.sum()
    return t


def t1c_deletion_vs_double_count(cc: pd.DataFrame, icsg: pd.DataFrame, eff: dict) -> pd.DataFrame:
    w = icsg_world(icsg, eff)
    m = by_year(flow(cc, "MF", "Concentrate"))
    t = pd.DataFrame({"mine_prior": m.prior, "mine_post": m.post,
                      "deleted": m.prior - m.post, "double_count": w.S_smelter_prod})
    t["ratio"] = t.deleted / t.double_count
    return t


def t1d_who_pays(cc: pd.DataFrame, icsg: pd.DataFrame) -> pd.DataFrame:
    p = cc[cc.time.isin(POOL)]

    def delta(sf, st, col):
        g = flow(p, sf, st).groupby(col)[["prior", "post"]].sum()
        return g.post - g.prior
    t = pd.DataFrame({
        "d_scrap_to_smelting": delta("Scrap", "Smelting", "region_to"),
        "d_scrap_to_cfr": delta("Scrap", "CFR", "region_to"),
        "d_conc_intake": delta("Concentrate", "Smelting", "region_to"),
        "d_mine": delta("MF", "Concentrate", "region_from"),
    }).fillna(0.0)
    x = icsg[(icsg.Type == "Scrap") & (icsg.Process == "Smelting") & icsg.Year.isin([int(y) for y in POOL])]
    t["icsg_sec_smelter_input"] = x.groupby("reg")["Value"].sum().reindex(t.index).fillna(0.0)
    return t.sort_values("d_conc_intake").rename_axis("region").reset_index()


def t1e_secondary_share_by_group(cc: pd.DataFrame) -> pd.DataFrame:
    """Refinery-side secondary share per group; plus China's smelter scrap share and band."""
    inp = cc[(cc.sector_from == "Scrap")
             | ((cc.sector_from == "Concentrate") & (cc.sector_to == "Smelting"))
             | ((cc.sector_from == "SXEW") & (cc.sector_to == "CSXEW"))].copy()
    inp["group"] = inp.region_to.map(group_of)
    inp["secondary"] = inp.sector_from == "Scrap"
    a = inp.groupby(["group", "year", "secondary"])[["prior", "post"]].sum().unstack("secondary")
    out = pd.DataFrame({
        "share_prior": a[("prior", True)] / (a[("prior", True)] + a[("prior", False)]),
        "share_post": a[("post", True)] / (a[("post", True)] + a[("post", False)]),
    }).reset_index()
    return out


def china_smelter_scrap_share(cc: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float]]:
    c = flow(cc[cc.region_to == "China"], "Scrap", "Smelting")
    k = flow(cc[cc.region_to == "China"], "Concentrate", "Smelting")
    s, q = by_year(c), by_year(k)
    t = pd.DataFrame({"share_prior": s.prior / (s.prior + q.prior),
                      "share_post": s.post / (s.post + q.post)})
    inputs = InputRegistries.initialize(root="data/input")
    p = pd.read_csv(inputs.params.resolve("scrap_share") / "params.csv", sep=None,
                    engine="python", comment="#", dtype=str)
    row = p[(p.a_sector_to == "Smelting") & (p.a_region_to == "China")].iloc[0]
    return t, (float(row["value_min"]), float(row["value_max"]))


# ---------------------------------------------------------------------------
# tables — insight 2
# ---------------------------------------------------------------------------

def t2a_intake_shares(cc: pd.DataFrame) -> pd.DataFrame:
    s = flow(cc, "Concentrate", "Smelting").copy()
    s["group"] = s.region_to.map(group_of)
    g = s.groupby(["year", "group"])[["prior", "post"]].sum()
    tot = g.groupby("year").transform("sum")
    return (g / tot).rename(columns={"prior": "share_prior", "post": "share_post"}).reset_index()


def t2b_import_dependence(cc: pd.DataFrame) -> pd.DataFrame:
    s = flow(cc, "Concentrate", "Smelting").copy()
    s["imported"] = s.region_from != s.region_to
    rows = []
    for r, x in s.groupby("region_to"):
        def stat(sub):
            tot = sub[["prior", "post"]].sum()
            imp = sub[sub.imported][["prior", "post"]].sum()
            return tot, imp
        tot24, imp24 = stat(x[x.year == 2024])
        tot00, imp00 = stat(x[x.year == 2000])
        totp, impp = stat(x[x.time.isin(POOL)])
        if totp.prior <= 0:
            continue
        rows.append(dict(region=r,
                         import_share_2000=imp00.post / tot00.post if tot00.post else np.nan,
                         import_share_2024=imp24.post / tot24.post if tot24.post else np.nan,
                         intake_2024_prior=tot24.prior, intake_2024_post=tot24.post,
                         retention_2024=tot24.post / tot24.prior if tot24.prior else np.nan,
                         import_share_pool=impp.prior / totp.prior,
                         intake_pool_prior=totp.prior, intake_pool_post=totp.post,
                         retention_pool=totp.post / totp.prior))
    return pd.DataFrame(rows).sort_values("intake_pool_prior", ascending=False)


def t2c_conc_legs(cc: pd.DataFrame) -> pd.DataFrame:
    s = flow(cc, "Concentrate", "Smelting")
    s = s[(s.region_from != s.region_to) & s.year.isin([2000, 2024])]
    g = s.groupby(["year", "region_from", "region_to"])[["prior", "post"]].sum().reset_index()
    g["retention"] = g.post / g.prior
    return g.sort_values(["year", "prior"], ascending=[True, False])


# ---------------------------------------------------------------------------
# tables — insight 3
# ---------------------------------------------------------------------------

def t3a_fu_shares(cc: pd.DataFrame) -> pd.DataFrame:
    s = flow(cc, "Cathode", "FU").copy()
    s["group"] = s.region_to.map(group_of)
    g = s.groupby(["year", "group"])[["prior", "post"]].sum()
    out = g.copy()
    tot = g.groupby("year").transform("sum")
    out["share_prior"], out["share_post"] = g.prior / tot.prior, g.post / tot.post
    return out.reset_index()


def t3b_fu_cumulative(cc: pd.DataFrame) -> pd.DataFrame:
    s = flow(cc, "Cathode", "FU")
    g = s.groupby("region_to")[["prior", "post"]].sum().sort_values("post", ascending=False)
    g["share_post"] = g.post / g.post.sum()
    g["kg_per_capita_2024pop"] = [g.post[r] / POP_2024[r] / 1e3 * 1e3 if r in POP_2024 else np.nan
                                  for r in g.index]   # kt / M people = kg per capita
    d = s.assign(decade=(s.year // 10) * 10).groupby(["region_to", "decade"])["post"].sum().unstack()
    return g.join(d.add_prefix("post_")).rename_axis("region").reset_index()


def t3c_stock_buffers(cc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = {}
    for st in ("MESC", "USC", "PSC"):
        b = by_year(cc[cc.sector_to == st])
        out[f"{st}_prior"], out[f"{st}_post"] = b.prior.cumsum(), b.post.cumsum()
    world = pd.DataFrame(out)
    m = cc[cc.sector_to == "MESC"]
    hubs = m.groupby("region_to")["post"].sum().sort_values(ascending=False)
    top = hubs.head(5).index.tolist()
    h = m.assign(hub=np.where(m.region_to.isin(top), m.region_to, "other")) \
         .groupby(["year", "hub"])["post"].sum().unstack().reindex(YEARS).fillna(0.0).cumsum()
    return world, h


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _style(ax, ylabel=None, title=None):
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.tick_params(labelsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10.5, loc="left", color=INK)


def fig0_sankey(cc: pd.DataFrame, out: Path) -> None:
    """World 2024, prior above / posterior below — reuse of the supervisor Sankey."""
    e = cc[cc.year == 2024]
    parts = []
    for col in ("prior", "post"):
        li = (e.groupby(["sector_from", "sector_to"])[col].sum().reset_index()
              .rename(columns={"sector_from": "src", "sector_to": "dst", col: "a_post"}))
        keep = set(SF.NODE_X) | set(SF.CONTRACT)      # commodity nodes become the bands
        li = li[(li.a_post > 1.0) & li.src.isin(keep) & li.dst.isin(keep)]
        parts.append(li)
    prior, post = parts
    fig = go.Figure([SF._sankey_trace(prior, (0.58, 0.97)),
                     SF._sankey_trace(post, (0.0, 0.42))])
    prior, post = SF._contract_flows(prior), SF._contract_flows(post)

    def line(li, s, d):
        return li[(li.src == s) & (li.dst == d)].a_post.sum() / 1e3
    fig.update_layout(
        width=1600, height=900, font=dict(color=INK, size=12), paper_bgcolor="white",
        margin=dict(l=30, r=140, t=90, b=55),
        title=dict(text="World copper 2024, contained copper — prior (top) and reconciled (bottom)",
                   font=dict(size=18), x=0.02, y=0.975),
        annotations=[
            dict(text=f"<b>Prior</b> — mine {line(prior, 'MF', 'Smelting') + line(prior, 'MF', 'Tailings'):.1f} Mt at MF, "
                      f"scrap {line(prior, 'Scrap', 'Smelting') + line(prior, 'Scrap', 'CFR'):.1f} Mt, "
                      f"cathode {line(prior, 'CP', 'FU') + line(prior, 'CP', 'MESC') + line(prior, 'CP', 'USC'):.1f} Mt",
                 x=0, y=1.005, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12, color=INK2), xanchor="left"),
            dict(text=f"<b>Reconciled</b> — mine {line(post, 'MF', 'Smelting') + line(post, 'MF', 'Tailings'):.1f} Mt at MF, "
                      f"scrap {line(post, 'Scrap', 'Smelting') + line(post, 'Scrap', 'CFR'):.1f} Mt, "
                      f"cathode {line(post, 'CP', 'FU') + line(post, 'CP', 'MESC') + line(post, 'CP', 'USC'):.1f} Mt"
                      " — grey losses, orange scrap",
                 x=0, y=0.47, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12, color=INK2), xanchor="left"),
        ])
    fig.write_image(out / "fig0_sankey_2024.png", scale=2)


def fig1a_supply_stacks(w: pd.DataFrame, cc: pd.DataFrame, eff: dict, out: Path) -> None:
    """Refinery-side supply of refined copper, three views, one scale."""
    k = eff["cfr"] * eff["er"]
    views = {}
    views["ICSG as published (secondary nested)"] = pd.DataFrame({
        "Primary via concentrate": w.P_primary_refined, "SX-EW": w.sxew,
        "Secondary via smelters": w.sec_via_smelters,
        "Secondary direct to refinery": w.sec_direct_to_refinery})
    for lab, col in (("Model prior (both series booked)", "prior"), ("Model posterior", "post")):
        views[lab] = pd.DataFrame({
            "Primary via concentrate": by_year(flow(cc, "Concentrate", "Smelting"))[col] * eff["sm"] * k,
            "SX-EW": by_year(flow(cc, "SXEW", "CSXEW"))[col],
            "Secondary via smelters": by_year(flow(cc, "Scrap", "Smelting"))[col] * eff["sm"] * k,
            "Secondary direct to refinery": by_year(flow(cc, "Scrap", "CFR"))[col] * k})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    colors = [PALETTE[0], PALETTE[2], PALETTE[1], PALETTE[3]]
    for ax, (lab, v) in zip(axes, views.items()):
        ax.stackplot(YEARS, [v[c] / 1e3 for c in v.columns], labels=v.columns,
                     colors=colors, alpha=0.9, lw=0)
        ax.plot(YEARS, w.refined_plus_sxew / 1e3, color=INK, lw=1.4,
                label="ICSG refined output (incl. SX-EW)")
        _style(ax, title=lab)
        ax.set_xlim(1995, 2024)
        ax.set_xticks(range(1995, 2025, 5))
        top = (v.sum(axis=1) / 1e3)
        ax.annotate(f"{top.loc[2024]:.1f} Mt", (2024, top.loc[2024]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8.5, color=INK, va="center")
        conc24 = v["Primary via concentrate"].loc[2024] / 1e3
        ax.annotate(f"concentrate {conc24:.1f}", (2019, conc24 / 2), fontsize=8.5,
                    color="white", ha="center", va="center")
    axes[0].set_ylabel("refined copper equivalent  [Mt Cu / a]", fontsize=9)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=1)
    fig.suptitle("Where refined copper comes from: ICSG as published, the model's prior, the reconciled result",
                 fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "fig1a_supply_stacks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1b_secondary_share(t: pd.DataFrame, china: pd.DataFrame, band: tuple, out: Path) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(15, 3.4), sharey=True)
    for ax, g in zip(axes, ["China", "EU-14", "JP/KR/TW", "United States", "Rest"]):
        x = t[t.group == g].set_index("year")
        ax.plot(x.index, x.share_prior, color=GREY, lw=1.6, label="prior")
        ax.plot(x.index, x.share_post, color=GROUP_COLOR[g], lw=2, label="posterior")
        _style(ax, title=g)
        ax.set_ylim(0, 0.7)
        ax.set_xticks([1995, 2010, 2024])
    ax = axes[-1]
    ax.axhspan(band[0], band[1], color=PALETTE[3], alpha=0.18, lw=0, label="hard band")
    ax.plot(china.index, china.share_prior, color=GREY, lw=1.6, label="prior")
    ax.plot(china.index, china.share_post, color=GROUP_COLOR["China"], lw=2, label="posterior")
    _style(ax, title="China smelters: scrap share of feed")
    ax.set_xticks([1995, 2010, 2024])
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    axes[0].set_ylabel("secondary share of refinery-side input", fontsize=9)
    fig.suptitle("Secondary share, prior vs reconciled — China's smelters sit on the ceiling of their hard scrap band",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out / "fig1b_secondary_share.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1c_deletion_vs_double_count(t: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t.index, t.deleted / 1e3, color=PALETTE[0], lw=2, label="mine output deleted (MF→Concentrate, prior − posterior)")
    ax.plot(t.index, t.double_count / 1e3, color=PALETTE[1], lw=2, label="secondary-smelter copper booked twice")
    _style(ax, ylabel="Mt Cu / a")
    ax.set_xticks(range(1995, 2025, 5))
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    r = np.corrcoef(t.deleted, t.double_count)[0, 1]
    ax.set_title(f"World mine-chain deletion tracks the double count — 30-y totals {t.deleted.sum()/1e3:.1f} vs "
                 f"{t.double_count.sum()/1e3:.1f} Mt, r = {r:.2f}", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig1c_deletion_vs_double_count.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2a_intake_by_origin(cc: pd.DataFrame, out: Path) -> None:
    s = flow(cc, "Concentrate", "Smelting")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharex=True)
    top_imp = (s[s.year == 2024].groupby("region_to").post.sum().sort_values(ascending=False)
               .head(6).index.tolist())
    top_exp = (s[(s.year == 2024) & (s.region_from != s.region_to)]
               .groupby("region_from").post.sum().sort_values(ascending=False).head(6).index.tolist())
    for ax, y in zip(axes, (2000, 2024)):
        x = s[(s.year == y) & s.region_to.isin(top_imp)].copy()
        x["origin"] = np.where(x.region_from == x.region_to, "domestic",
                               np.where(x.region_from.isin(top_exp), x.region_from, "other imports"))
        g = x.groupby(["region_to", "origin"]).post.sum().unstack().reindex(top_imp).fillna(0.0) / 1e3
        cols = ["domestic"] + top_exp + ["other imports"]
        g = g.reindex(columns=[c for c in cols if c in g.columns]).fillna(0.0)
        left = np.zeros(len(g))
        for c in g.columns:
            color = INK2 if c == "domestic" else GREY if c == "other imports" \
                else PALETTE[top_exp.index(c)]
            ax.barh(g.index, g[c], left=left, color=color, label=c, height=0.62)
            left += g[c].values
        for i, v in enumerate(left):
            ax.annotate(f"{v:.1f}", (v, i), xytext=(3, 0), textcoords="offset points",
                        fontsize=8, va="center", color=INK)
        _style(ax, title=f"{y}")
        ax.grid(axis="x", color=GRID, lw=0.7)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    axes[1].legend(fontsize=8, frameon=False, loc="lower right", title="origin", title_fontsize=8)
    axes[0].set_xlabel("smelter concentrate intake, reconciled  [Mt Cu]", fontsize=9)
    fig.suptitle("Who smelts the world's concentrate, and where it comes from", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig2a_intake_by_origin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2b_retention_vs_dependence(t: pd.DataFrame, out: Path) -> None:
    x = t[t.intake_pool_prior > 5 * 200].copy()          # > 200 kt/yr over the pool
    fig, ax = plt.subplots(figsize=(8, 5))
    dep = x.import_share_pool > 0.5
    ax.scatter(x.import_share_pool[dep], x.retention_pool[dep], s=x.intake_pool_prior[dep] / 40,
               color=PALETTE[1], alpha=0.75, lw=0, label="trade-dependent (> 50 % imported)")
    ax.scatter(x.import_share_pool[~dep], x.retention_pool[~dep], s=x.intake_pool_prior[~dep] / 40,
               color=PALETTE[0], alpha=0.75, lw=0, label="mostly own concentrate")
    placed = []      # (x, y, dy) of every label so far; a neighbour takes the opposite side
    for r in x.sort_values("retention_pool", ascending=False).itertuples():
        near = {dy for px, py, dy in placed
                if abs(r.import_share_pool - px) < 0.06 and abs(r.retention_pool - py) < 0.025}
        dy = 4 if not near else (-12 if 4 in near and -12 not in near else 4 if -12 in near and 4 not in near else -24)
        ax.annotate(r.region, (r.import_share_pool, r.retention_pool), xytext=(6, dy),
                    textcoords="offset points", fontsize=8, color=INK)
        placed.append((r.import_share_pool, r.retention_pool, dy))
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
    _style(ax, ylabel="smelter intake retained (posterior / prior), 2020-24")
    ax.set_xlabel("share of concentrate intake imported, 2020-24", fontsize=9)
    ax.set_xlim(-0.05, 1.08)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=PALETTE[1],
                              label="trade-dependent (> 50 % imported)"),
                       Line2D([], [], marker="o", ls="", ms=8, color=PALETTE[0],
                              label="mostly own concentrate")],
              fontsize=8.5, frameon=False, loc="lower left")
    ax.set_title("The reconciliation discharges through trade: import-dependent smelters lose, self-sufficient ones hold",
                 fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig2b_retention_vs_dependence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2c_chile_capacity(dc: pd.DataFrame, out: Path) -> None:
    c = dc[dc.region == "Chile"].copy()
    c["year"] = c.time.astype(int)
    c = c.set_index("year").sort_index()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    capped = c.headroom < 0
    for y in c.index[capped]:
        ax.axvspan(y - 0.5, y + 0.5, color=PALETTE[1], alpha=0.12, lw=0)
    ax.plot(c.index, c.prior / 1e3, color=INK, lw=2, label="mine production, ICSG (prior)")
    ax.plot(c.index, c.cap / 1e3, color=PALETTE[1], lw=2, label="max discharge = 0.35 · BACI export mass + smelter intake")
    ax.plot(c.index, c.post / 1e3, color=PALETTE[0], lw=1.6, ls="--", label="mine production, reconciled")
    ax.plot(c.index, c.U / 1e3, color=GREY, lw=1.4, label="domestic smelter intake, ICSG")
    _style(ax, ylabel="Mt Cu / a")
    ax.set_xticks(range(1995, 2025, 5))
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title(f"Chile: in {int(capped.sum())} of 30 years the mine output cannot leave at any grade in the band "
                 "(shaded)", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig2c_chile_capacity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3a_fu_stack(t: pd.DataFrame, out: Path) -> None:
    g = t.pivot(index="year", columns="group", values="post").reindex(columns=GROUPS).fillna(0.0) / 1e3
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.stackplot(g.index, [g[c] for c in GROUPS], labels=GROUPS, colors=[GROUP_COLOR[c] for c in GROUPS],
                 alpha=0.92, lw=0)
    _style(ax, ylabel="refined copper into final use  [Mt Cu / a]")
    ax.set_xlim(1995, 2024)
    ax.set_xticks(range(1995, 2025, 5))
    sh = t.pivot(index="year", columns="group", values="share_post")
    ax.annotate(f"China {sh.loc[2024, 'China']*100:.0f} %", (2019, g.loc[2019, 'China'] / 2),
                fontsize=9, color="white", ha="center", va="center", fontweight="bold")
    ax.annotate(f"{sh.loc[1995, 'China']*100:.0f} % in 1995", (1996, g.loc[1996, 'China'] / 2 + 0.3),
                fontsize=8, color=INK, ha="left", va="bottom")
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    ax.set_title("Where the refined copper went: final use by region group, reconciled", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig3a_final_use_stack.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3b_fu_cumulative(t: pd.DataFrame, out: Path) -> None:
    x = t.head(12).copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [GROUP_COLOR[group_of(r)] for r in x.region]
    ax.barh(x.region, x.post / 1e3, color=colors, height=0.62)
    for i, r in enumerate(x.itertuples()):
        lab = f"{r.post/1e3:.0f} Mt  ({r.share_post*100:.0f} %)"
        if np.isfinite(r.kg_per_capita_2024pop):
            lab += f"  ·  {r.kg_per_capita_2024pop:.0f} kg/cap"
        ax.annotate(lab, (r.post / 1e3, i), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, va="center", color=INK)
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.set_xlim(0, x.post.max() / 1e3 * 1.45)
    ax.set_xlabel("cumulative refined copper into final use 1995–2024, reconciled  [Mt Cu]", fontsize=9)
    ax.set_title("Cumulative absorption by fabricators — per-capita on 2024 population (gross inflow, not in-use stock)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig3b_final_use_cumulative.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3c_stock_buffers(world: pd.DataFrame, hubs: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    for i, st in enumerate(("MESC", "USC", "PSC")):
        ax.plot(world.index, world[f"{st}_prior"] / 1e3, color=PALETTE[i], lw=1.6, ls="--")
        ax.plot(world.index, world[f"{st}_post"] / 1e3, color=PALETTE[i], lw=2.2,
                label={"MESC": "exchange stocks (MESC)", "USC": "user stocks (USC)", "PSC": "producer stocks (PSC)"}[st])
    ax.plot([], [], color=INK2, lw=1.6, ls="--", label="prior (dashed)")
    _style(ax, ylabel="cumulative net stock change since 1995  [Mt Cu]",
           title="World stock-change sinks: reconciled (solid) vs prior (dashed)")
    ax.set_xticks(range(1995, 2025, 5))
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax = axes[1]
    cols = [c for c in hubs.columns if c != "other"] + ["other"]
    ax.stackplot(hubs.index, [hubs[c] / 1e3 for c in cols], labels=cols,
                 colors=PALETTE[:len(cols) - 1] + [GREY], alpha=0.92, lw=0)
    _style(ax, title="Exchange-stock additions by country, reconciled, cumulative")
    ax.set_xticks(range(1995, 2025, 5))
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("The stock-change nodes absorb what the trade matrix does not close", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "fig3c_stock_buffers.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _records(df: pd.DataFrame) -> list:
    d = df.reset_index() if df.index.name is not None or not isinstance(df.index, pd.RangeIndex) else df
    d = d.copy()
    d.columns = [str(c) if not isinstance(c, tuple) else "_".join(map(str, c)) for c in d.columns]
    return json.loads(d.to_json(orient="records", double_precision=6))


def main(opt: str = DEFAULT_RUN, figures: bool = True) -> None:
    out = Path("docs/rep/figs") / f"metabolism_{opt}"
    (out / "tables").mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    lin = lineage(opt)
    print(f"run {opt} ({lin['label']})  const {lin['const']}  imp {lin['imp']}")
    print(f"   params {lin['params']}   solved {lin['solved']}/30   Σ objective {lin['objective']:,.2f}")
    eff = load_eff()
    e = load_run(opt)
    cc = e[e.valuation == "CC"]
    icsg = load_icsg()

    tables = {
        "t0_world_accounts": t0_world_accounts(cc),
        "t1a_secondary_world": t1a_secondary_world(icsg, eff, cc),
        "t1b_nesting_test": t1b_nesting_test(icsg, eff, cc),
        "t1c_deletion_vs_double_count": t1c_deletion_vs_double_count(cc, icsg, eff),
        "t1d_who_pays": t1d_who_pays(cc, icsg),
        "t1e_secondary_share_by_group": t1e_secondary_share_by_group(cc),
        "t2a_intake_shares": t2a_intake_shares(cc),
        "t2b_import_dependence": t2b_import_dependence(cc),
        "t2c_conc_legs": t2c_conc_legs(cc),
        "t3a_fu_shares": t3a_fu_shares(cc),
        "t3b_fu_cumulative": t3b_fu_cumulative(cc),
    }
    china, band = china_smelter_scrap_share(cc)
    tables["t1e_china_smelter_scrap_share"] = china.assign(band_lo=band[0], band_hi=band[1])
    m, P, U, _ = N.load_inputs()
    dc = N.discharge_capacity(m, P, U, opt)
    tables["t2d_discharge_capacity"] = dc
    world_stock, hubs = t3c_stock_buffers(cc)
    tables["t3c_stock_buffers_world"] = world_stock
    tables["t3c_mesc_by_hub"] = hubs

    for k, t in tables.items():
        t.to_csv(out / "tables" / f"{k}.csv")
    with open(out / "data.json", "w") as f:
        json.dump({k: _records(t) for k, t in tables.items()}, f)

    # ── headline numbers (the REP quotes these) ───────────────────────────────
    w = tables["t1a_secondary_world"]
    d = tables["t1c_deletion_vs_double_count"]
    print("\nHEADLINE")
    print(f"  I1 ICSG secondary refined 2024 {w.S_refined_prod[2024]:,.0f} kt = {w.share_nested[2024]*100:.1f} % of refined+SXEW; "
          f"share 2010 {w.share_nested[2010]:.3f} → 2024 {w.share_nested[2024]:.3f}; refined+SXEW growth 2010→24 "
          f"{(w.refined_plus_sxew[2024]/w.refined_plus_sxew[2010]-1)*100:.0f} %")
    print(f"  I1 model secondary share 2020-24: prior {w.model_share_prior.loc[2020:2024].mean():.3f}  "
          f"post {w.model_share_post.loc[2020:2024].mean():.3f}; both-booked ICSG share 2024 {w.share_both_booked[2024]:.3f}")
    print(f"  I1 double count 2024 {w.sec_via_smelters[2024]/1e3:.2f} Mt (S_sm {w.S_smelter_prod[2024]:,.0f} kt); "
          f"conc absorbable both {w.conc_needed_both[2024]:,.0f} vs mine {w.mine_conc[2024]:,.0f} → gap {w.gap_both[2024]:,.0f} kt; "
          f"nested gap {w.gap_nested[2024]:,.0f} kt")
    print(f"  I1 30-y: deleted {d.deleted.sum():,.0f}  double count {d.double_count.sum():,.0f}  ratio {d.deleted.sum()/d.double_count.sum():.2f}  "
          f"corr {np.corrcoef(d.deleted, d.double_count)[0,1]:.2f}")
    a = tables["t1b_nesting_test"]; au = a[(a.region == "Austria") & (a.year == 2024)].iloc[0]
    print(f"  I1 Austria 2024: S_sm {au.sec_smelter_prod:.0f} S_ref {au.sec_refined_prod:.0f} total {au.total_refined:.0f} "
          f"both-booked share {au.both_booked_share:.2f}; posterior scrap→Sm {au.post_scrap_to_smelting:.1f} (prior {au.prior_scrap_to_smelting:.1f})")
    wp = tables["t1d_who_pays"].set_index("region")
    print(f"  I1 who pays 2020-24: China Δconc intake {wp.loc['China','d_conc_intake']:,.0f} Δscrap→Sm {wp.loc['China','d_scrap_to_smelting']:,.0f}; "
          f"Chile Δmine {wp.loc['Chile','d_mine']:,.0f}; world Δconc intake {wp.d_conc_intake.sum():,.0f} vs ICSG sec smelter input {wp.icsg_sec_smelter_input.sum():,.0f}")
    print(f"  I1 China smelter scrap share post 2020-24 {china.share_post.loc[2020:2024].min():.2f}–{china.share_post.loc[2020:2024].max():.2f}, band {band}")
    s = tables["t2a_intake_shares"].set_index(["year", "group"])
    dep = tables["t2b_import_dependence"].set_index("region")
    print(f"  I2 China intake share post 1995 {s.loc[(1995,'China'),'share_post']:.3f} → 2024 {s.loc[(2024,'China'),'share_post']:.3f} "
          f"(prior 2024 {s.loc[(2024,'China'),'share_prior']:.3f}); import share 2000 {dep.loc['China','import_share_2000']:.2f} → 2024 {dep.loc['China','import_share_2024']:.2f}")
    print("  I2 intake retention 2024: " + ", ".join(f"{r} {dep.loc[r,'retention_2024']:.2f}" for r in
          ["China", "Japan", "Korea, Rep.", "Germany", "India", "Chile", "Poland", "Russia", "United States"] if r in dep.index))
    ch = dc[dc.region == "Chile"]
    print(f"  I2 Chile hard-capped {int((ch.headroom<0).sum())}/30 years; world forced shortfall "
          f"{(-dc.headroom[dc.headroom<0]).sum():,.0f} kt = {(-dc.headroom[dc.headroom<0]).sum()/dc.deleted.sum()*100:.1f} % of mine deletion")
    fu = tables["t3a_fu_shares"].set_index(["year", "group"])
    west = sum(fu.loc[(y, g), "share_post"] for g in ("EU-14", "United States", "JP/KR/TW") for y in [1995])
    west24 = sum(fu.loc[(2024, g), "share_post"] for g in ("EU-14", "United States", "JP/KR/TW"))
    cum = tables["t3b_fu_cumulative"].set_index("region")
    print(f"  I3 FU share China 1995 {fu.loc[(1995,'China'),'share_post']:.3f} → 2024 {fu.loc[(2024,'China'),'share_post']:.3f}; "
          f"EU+US+JP/KR/TW {west:.3f} → {west24:.3f}; cumulative China {cum.loc['China','post']/1e3:.0f} of {cum.post.sum()/1e3:.0f} Mt "
          f"({cum.loc['China','share_post']*100:.0f} %), {cum.loc['China','kg_per_capita_2024pop']:.0f} kg/cap")
    print(f"  I3 stock buffers cumulative 2024: MESC post {world_stock.MESC_post[2024]/1e3:.1f} vs prior {world_stock.MESC_prior[2024]/1e3:.1f} Mt; "
          f"USC {world_stock.USC_post[2024]/1e3:.1f} vs {world_stock.USC_prior[2024]/1e3:.1f}; PSC {world_stock.PSC_post[2024]/1e3:.1f} vs {world_stock.PSC_prior[2024]/1e3:.1f}; "
          f"MESC hubs {', '.join(f'{c} {hubs[c][2024]/1e3:.1f}' for c in hubs.columns if c != 'other')}")
    acc0 = tables["t0_world_accounts"]
    print(f"  caveat ore grade: world in-situ grade prior {acc0.ore_in_situ_prior[1995]/e[(e.valuation=='Mass')&(e.sector_from=='CO')&(e.year==1995)].prior.sum()*100:.2f} "
          f"→ {acc0.ore_in_situ_prior[2024]/e[(e.valuation=='Mass')&(e.sector_from=='CO')&(e.year==2024)].prior.sum()*100:.2f} %, "
          f"post {acc0.ore_in_situ_post[1995]/e[(e.valuation=='Mass')&(e.sector_from=='CO')&(e.year==1995)].post.sum()*100:.2f} "
          f"→ {acc0.ore_in_situ_post[2024]/e[(e.valuation=='Mass')&(e.sector_from=='CO')&(e.year==2024)].post.sum()*100:.2f} %")

    if figures:
        fig0_sankey(cc, out)
        fig1a_supply_stacks(w, cc, eff, out)
        fig1b_secondary_share(tables["t1e_secondary_share_by_group"], china, band, out)
        fig1c_deletion_vs_double_count(d, out)
        fig2a_intake_by_origin(cc, out)
        fig2b_retention_vs_dependence(tables["t2b_import_dependence"], out)
        fig2c_chile_capacity(dc, out)
        fig3a_fu_stack(tables["t3a_fu_shares"], out)
        fig3b_fu_cumulative(tables["t3b_fu_cumulative"], out)
        fig3c_stock_buffers(world_stock, hubs, out)
    print(f"\nwrote {len(tables)} tables + data.json" + (" + 10 figures" if figures else "") + f" → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", default=DEFAULT_RUN)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    main(a.opt, figures=not a.no_figures)
