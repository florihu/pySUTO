"""
issue79_baci490_taiwan_diag.py
==============================
#79 measurement of #77: BACI code 490 "Other Asia, nes" mapped to Taiwan instead of ROW.
Reads persisted run artefacts, the feed runs' F_root, the constraint runs' σ_a, the runs'
imputation logs and two netted BACI versions' ledgers; rebuilds each run's ICSG concordance
in memory (nothing is registered) and builds the sum-row table in memory (no per-year CSVs,
#46). Prints #79's pre-registered items and predictions for <new> against <base>. The
sections shared with #69 and #75 are imported from ``issue69_aggregate_gap_diag`` and
``issue75_scandinavia_split_diag``:

    F   feeds: Taiwan observed and its undistributed ICSG rows (item 1); ROW's domestic
        semis leg as fed, imputed, aggregated and solved (item 2); ROW's semis prior
        against its ICSG fan (item 3)
    T   Taiwan: semis use against ICSG; partners by sector; concentrate inflow; scrap
    R   ROW: CP→Cathode posterior/prior; semis posterior against the fan; Δz² by sector
    N   netting: Taiwan's and ROW's ledger rows, baci v<base> → v<new>
    E   the Round-E items (ADR 0025)
    S   stock legs: net per family; largest-growth year; legs above |z| 5 in new only
    C   common-set sum-row z², and the sum rows only one run has
    A   attribution of the largest moves

    python scripts/explo/issue79_baci490_taiwan_diag.py <opt_base> <opt_new> \\
                                                        <baci_v_base> <baci_v_new>
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import icsg_feed as F                                    # noqa: E402
import issue69_aggregate_gap_diag as P69                 # noqa: E402
import issue75_scandinavia_split_diag as P75             # noqa: E402
import roundL_reexport_diag as L                         # noqa: E402
from build_trade_shares import baci_observed_regions     # noqa: E402
from pysuto.acc import AccountingStructure               # noqa: E402
from pysuto.config import load_config                    # noqa: E402

ROOT    = Path("data/proc")
ICSG_IN = Path("data/input/feed/icsg")
TS_IN   = Path("data/input/params/trade_shares")
TAIWAN  = "Taiwan"
YRS     = [str(y) for y in range(1995, 2012)]


def _fed(feed: str) -> np.ndarray:
    """The feed run's F_root where it covers an edge (σ_root > 0), NaN elsewhere."""
    r = np.load(ROOT / "feed" / "registry" / feed / "result.npz")
    return np.where(r["sigma_root"] > 0, r["F_root"], np.nan)


def _icsg_inputs(opt: str) -> tuple:
    """The run's ICSG feed run, ICSG input version and trade_shares version."""
    feed, v_icsg = P75._icsg(opt)
    meta = json.loads((ROOT / "feed" / "registry" / feed / "metadata.json").read_text())
    return feed, v_icsg, meta["parents"]["inputs"]["trade_shares"]["version"]


def _icsg_use(v_icsg: int, region: str, process: str) -> pd.Series:
    """ICSG's raw ``process`` Use rows for a root region, kt CC by year."""
    d = ICSG_IN / f"v{v_icsg}"
    raw = pd.read_csv(d / "values.csv")
    rmap = pd.read_excel(d / "conc.xlsx", sheet_name="region_map")
    root = raw["Region"].map(dict(zip(rmap["src_region"], rmap["root_region"])))
    x = raw[(root == region) & (raw.Process == process) & (raw.Supply_Use == "Use")]
    return x.groupby(x.Year.astype(str)).Value.sum()


# ---------------------------------------------------------------------------
# F — feeds
# ---------------------------------------------------------------------------

def feeds(E: dict, acc, base: str, new: str) -> None:
    """F: items 1–3, read from the feed runs and Stage 2 of each chain."""
    print("\n== F feeds (items 1–3)")
    for o in (base, new):
        feed, v_icsg, v_ts = _icsg_inputs(o)
        baci = L._chain(o)[2]["baci"]
        obs = baci_observed_regions(acc, np.load(ROOT / "feed" / "registry" / baci
                                                 / "result.npz")["F_root"])
        V = ICSG_IN / f"v{v_icsg}"
        F.RAW_DATA, F.CONC_FILE = str(V / "values.csv"), str(V / "conc.xlsx")
        with contextlib.redirect_stdout(io.StringIO()):
            raw, tables = F.load_raw_data(), F.load_concordance_tables()
            ct = F.build_concordance(raw, tables, acc, user_weights_df=pd.read_csv(
                TS_IN / f"v{v_ts}" / "params.csv"))
        root = raw["Region"].map(dict(zip(tables["region_map"]["src_region"],
                                          tables["region_map"]["root_region"])))
        und = raw.loc[ct.undistributed_raw_rows]
        tw = und[root.loc[und.index] == TAIWAN]
        print(f"   {o}: icsg v{v_icsg}, trade_shares v{v_ts}, {baci}\n"
              f"     BACI-observed regions {len(obs)}, Taiwan observed {TAIWAN in obs}; "
              f"undistributed ICSG rows {len(und)}, Taiwan's {len(tw)} "
              f"({tw.Value.sum():,.0f} kt CC)")
    for o in (base, new):
        feed = _icsg_inputs(o)[0]
        _, imp, _ = L._chain(o)
        fagg = np.load(ROOT / "imp" / imp / "agg_result.npz")["f_agg"]
        log = pd.read_parquet(ROOT / "imp" / imp / "imputation_log.parquet")
        e = E[o][(E[o].valuation == "CC") & (E[o].sector_from == "Cathode")
                 & (E[o].sector_to == "FU") & (E[o].region_to == "ROW")
                 & E[o].time.isin(YRS)]
        e = e.assign(fed=_fed(feed)[e.edge_id.to_numpy()],
                     fagg=fagg[e.edge_id.to_numpy()])
        dom = e[e.region_from == "ROW"].set_index("time").reindex(YRS)
        wrote = log[log.edge_id.isin(dom.edge_id)].method.value_counts().to_dict()
        print(f"   {o} ROW domestic semis leg 1995–2011: covered by the ICSG feed in "
              f"{int(dom.fed.notna().sum())} of {len(YRS)} years; imputer writes {wrote}")
        print("     feed / f_agg / post: " + "  ".join(
            f"{t[-2:]}:{np.nan_to_num(dom.fed[t]):.0f}/{dom.fagg[t]:.0f}/{dom.post[t]:.0f}"
            for t in YRS))
        g = e.groupby("time")[["fed", "fagg"]].sum().reindex(YRS)
        d = (g.fagg - g.fed).abs()
        print(f"     ROW semis f_agg − ICSG fan: max |Δ| {d.max():.3g} kt CC; years within "
              f"0.5 kt CC {int((d <= 0.5).sum())} of {len(YRS)}")


# ---------------------------------------------------------------------------
# T — Taiwan
# ---------------------------------------------------------------------------

def taiwan(E: dict, base: str, new: str) -> None:
    """T: Taiwan's semis use against ICSG, its partners, concentrate inflow and scrap."""
    print("\n== T Taiwan (kt CC)")
    v_icsg = _icsg_inputs(new)[1]
    icsg = _icsg_use(v_icsg, TAIWAN, "Semi")
    for o in (base, new):
        e = E[o][E[o].valuation == "CC"]
        fu = e[(e.sector_from == "Cathode") & (e.sector_to == "FU") & (e.region_to == TAIWAN)]
        g = fu.groupby("time")[["prior", "post"]].sum().reindex(icsg.index).fillna(0.0)
        print(f"   {o} semis use 1995–2024: ICSG {icsg.sum():,.0f}, prior "
              f"{g.prior.sum():,.0f}, post {g.post.sum():,.0f} "
              f"(post/ICSG {g.post.sum() / icsg.sum():.3f}); by year post/ICSG "
              f"{(g.post / icsg).min():.2f}–{(g.post / icsg).max():.2f}")
        into = e[(e.region_to == TAIWAN) & (e.region_from != TAIWAN)]
        for s in ("Cathode", "Concentrate", "Anode", "Matte", "Scrap"):
            x = into[into.sector_from == s]
            if x.post.abs().sum() + x.prior.abs().sum() < 0.5:
                continue
            top = x.groupby("region_from").post.sum().sort_values(ascending=False).head(5)
            print(f"     imports of {s}: prior {x.prior.sum():,.0f} post {x.post.sum():,.0f}"
                  f"; top origins (post) "
                  + ", ".join(f"{k} {v:,.0f}" for k, v in top.items()))
        sc = e[((e.region_to == TAIWAN) | (e.region_from == TAIWAN))
               & ((e.sector_from == "Scrap") | (e.sector_to == "Scrap"))]
        sg = sc.groupby(["sector_from", "sector_to"])[["prior", "post"]].sum()
        sg = sg[(sg.prior.abs() + sg.post.abs()) > 0.5]
        print("     scrap legs touching Taiwan, prior / post: "
              + ("  ".join(f"{a}→{b} {r.prior:,.0f}/{r.post:,.0f}"
                           for (a, b), r in sg.iterrows()) or "none"))


# ---------------------------------------------------------------------------
# R — ROW
# ---------------------------------------------------------------------------

def row(E: dict, j: pd.DataFrame, base: str, new: str) -> None:
    """R: ROW's refined cathode retention, its semis use against its ICSG fan, and its
    domestic semis leg against ROW's production less its exports, 1995–2011 and 2012–2024."""
    print("\n== R ROW (kt CC)")
    late = [str(y) for y in range(2012, 2025)]
    for o in (base, new):
        e = E[o][E[o].valuation == "CC"]
        cp = (e[(e.sector_from == "CP") & (e.sector_to == "Cathode") & (e.region_to == "ROW")]
              .groupby("time")[["prior", "post"]].sum())
        ret = (cp.post / cp.prior).reindex(YRS)
        fu = e[(e.sector_from == "Cathode") & (e.sector_to == "FU") & (e.region_to == "ROW")]
        fu = fu.assign(fed=_fed(_icsg_inputs(o)[0])[fu.edge_id.to_numpy()])
        g = fu.groupby("time")[["fed", "prior", "post"]].sum()
        dom = (fu[fu.region_from == "ROW"].set_index("time")[["fed", "post"]]
               .rename(columns={"fed": "dom_fed", "post": "dom_post"}))
        x = e[(e.sector_from == "Cathode") & (e.region_from == "ROW") & (e.region_to != "ROW")]
        exports = x.groupby("time").prior.sum().rename("exports")
        t = pd.concat([g, dom, cp.prior.rename("P"), exports], axis=1).fillna(0.0)
        print(f"   {o}\n     CP→Cathode post/prior 1995–2011 {ret.min():.2f}–{ret.max():.2f}: "
              + " ".join(f"{k[-2:]}:{v:.2f}" for k, v in ret.items()))
        for name, yrs in (("1995–2011", YRS), ("2012–2024", late)):
            w = t.reindex(yrs)
            r = w.post / w.fed
            print(f"     {name}: semis post/fan {r.min():.2f}–{r.max():.2f}, Σ post − fan "
                  f"{(w.post - w.fed).sum():+,.0f}; domestic leg Σ fed {w.dom_fed.sum():,.0f}, "
                  f"post {w.dom_post.sum():,.0f}; Σ (production prior − export prior) "
                  f"{(w.P - w.exports).sum():,.0f}")
        print("     by year, domestic fed / post / production − exports: " + "  ".join(
            f"{k[-2:]}:{v.dom_fed:.0f}/{v.dom_post:.0f}/{v.P - v.exports:.0f}"
            for k, v in t.reindex(YRS + late).iterrows()))
    P75.row_by_sector(j, base)
    r = j[j.region == "ROW"]
    late_z = r[r.year.isin(late)].dz2.sum()
    print(f"   ROW Δz² 1995–2011 {r[r.year.isin(YRS)].dz2.sum():+,.0f}, 2012–2024 {late_z:+,.0f}")


# ---------------------------------------------------------------------------
# N — netting
# ---------------------------------------------------------------------------

def netting(v_base: int, v_new: int) -> None:
    """N: Taiwan's and ROW's ledger rows and the total T applied, base → new ledger."""
    print(f"\n== N netting, baci v{v_base} → v{v_new} ledger (kt CC)")
    for v in (v_base, v_new):
        led = P69._ledger(v)
        tw = led[led.region == TAIWAN]
        print(f"   v{v}: Σ T_applied {led.T_applied.sum():,.0f}; ROW "
              f"{led[led.region == 'ROW'].T_applied.sum():,.0f}; Taiwan rows {len(tw)}, "
              f"T_applied {tw.T_applied.sum():,.0f}")
        if len(tw):
            print(tw.groupby(["sector", "label"]).agg(
                n=("time", "size"), I=("I", "sum"), X=("X", "sum"),
                T_applied=("T_applied", "sum"), resid=("resid", "sum")).round(0).to_string())
        r = led[(led.region == "ROW") & (led.sector == "Cathode")].set_index("time").sort_index()
        gap = r.U - (r.P + r.I - r.X)
        print(f"     ROW cathode, ICSG use U − (ICSG production P + imports I − exports X): "
              f"1995–2011 Σ {gap.reindex(YRS).sum():+,.0f}, 2012–2024 Σ "
              f"{gap.drop(YRS, errors='ignore').sum():+,.0f}; by year "
              + " ".join(f"{t[-2:]}:{v:+.0f}" for t, v in gap.items()))


def main(base: str, new: str, v_base: int, v_new: int) -> None:
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    E, S, j = P69.load(base, new)
    feeds(E, acc, base, new)
    taiwan(E, base, new)
    row(E, j, base, new)
    netting(v_base, v_new)
    P69.round_e(E, S, j, base, new)
    P69.stock(E, base, new)
    P75.stock_breakdown(E, base, new)
    P75.stock_z(E, base, new)
    P69.common_set(j, base)
    P75.one_run_only(S, base, new)
    P69.attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
