"""
issue79_baci490_taiwan_diag.py
==============================
#79 measurement of #77: BACI code 490 "Other Asia, nes" mapped to Taiwan instead of ROW.
Reads persisted run artefacts, the feed runs' F_root, the constraint runs' σ_a, the runs'
imputation logs, two netted BACI versions' ledgers and the base BACI version's raw rows;
rebuilds each run's ICSG concordance in memory (nothing is registered) and builds the
sum-row table in memory (no per-year CSVs, #46). Prints #79's pass-rule items and the
numbers its predictions are scored on, for <new> against <base>. The sections shared with
#69 and #75 are imported from ``issue69_aggregate_gap_diag`` and
``issue75_scandinavia_split_diag``:

    F   feeds: Taiwan observed and its undistributed ICSG rows (item 1); ROW's domestic
        semis leg as fed, imputed, aggregated and solved (item 2); ROW's semis prior
        against its ICSG fan (item 3)
    T   Taiwan: its ICSG rows; semis use against ICSG; partners by sector, both ways;
        concentrate; scrap
    R   ROW: CP→Cathode posterior/prior; semis use against its fan; the domestic semis
        leg against ROW's production less exports; Δz² by sector and anchor band
    N   netting: Taiwan's ledger rows and T by year; ROW's cathode balance in each ledger;
        BACI 490's cathode imports in the base version
    E   the Round-E items (ADR 0025)
    S   stock legs: net per family; largest-growth year; every leg newly above |z| 5
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
import icsg_feed                                          # noqa: E402
import issue65_icsg_stock_parse_diag as P65               # noqa: E402
import issue69_aggregate_gap_diag as P69                  # noqa: E402
import issue75_scandinavia_split_diag as P75              # noqa: E402
import roundL_reexport_diag as L                          # noqa: E402
import roundZ_png_kg_diag as Z                            # noqa: E402
from build_trade_shares import baci_observed_regions      # noqa: E402
from pysuto.acc import AccountingStructure                # noqa: E402
from pysuto.config import load_config                     # noqa: E402

ROOT    = Path("data/proc")
ICSG_IN = Path("data/input/feed/icsg")
BACI_IN = Path("data/input/feed/baci")
TS_IN   = Path("data/input/params/trade_shares")
TAIWAN  = "Taiwan"
EARLY   = [str(y) for y in range(1995, 2012)]
LATE    = [str(y) for y in range(2012, 2025)]
Z_BIG   = 5.0


def _fed(feed: str) -> np.ndarray:
    """The feed run's F_root where it covers an edge (σ_root > 0), NaN elsewhere."""
    r = np.load(ROOT / "feed" / "registry" / feed / "result.npz")
    return np.where(r["sigma_root"] > 0, r["F_root"], np.nan)


def _icsg_inputs(opt: str) -> tuple:
    """The run's ICSG feed run, ICSG input version and trade_shares version."""
    feed, v_icsg = P75._icsg(opt)
    meta = json.loads((ROOT / "feed" / "registry" / feed / "metadata.json").read_text())
    return feed, v_icsg, meta["parents"]["inputs"]["trade_shares"]["version"]


def _icsg_raw(v_icsg: int) -> pd.DataFrame:
    """The ICSG input version's raw rows, with each row's root region."""
    d = ICSG_IN / f"v{v_icsg}"
    raw = pd.read_csv(d / "values.csv")
    rmap = pd.read_excel(d / "conc.xlsx", sheet_name="region_map")
    return raw.assign(root=raw["Region"].map(dict(zip(rmap["src_region"],
                                                      rmap["root_region"]))))


# ---------------------------------------------------------------------------
# F — feeds
# ---------------------------------------------------------------------------

def feeds(E: dict, acc, base: str, new: str) -> None:
    """F: items 1–3, read from the feed runs and Stage 2 of each chain."""
    print("\n== F feeds (items 1–3)")
    for o in (base, new):
        _, v_icsg, v_ts = _icsg_inputs(o)
        baci = L._chain(o)[2]["baci"]
        froot = np.load(ROOT / "feed" / "registry" / baci / "result.npz")["F_root"]
        observed = baci_observed_regions(acc, froot)
        V = ICSG_IN / f"v{v_icsg}"
        # icsg_feed's loaders read module paths that its main() normally sets
        icsg_feed.RAW_DATA = str(V / "values.csv")
        icsg_feed.CONC_FILE = str(V / "conc.xlsx")
        weights = pd.read_csv(TS_IN / f"v{v_ts}" / "params.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            raw, tables = icsg_feed.load_raw_data(), icsg_feed.load_concordance_tables()
            ct = icsg_feed.build_concordance(raw, tables, acc, user_weights_df=weights)
        root = _icsg_raw(v_icsg).root
        und = raw.loc[ct.undistributed_raw_rows]
        tw = und[root.loc[und.index] == TAIWAN]
        print(f"   {o}: icsg v{v_icsg}, trade_shares v{v_ts}, {baci}\n"
              f"     BACI-observed regions {len(observed)}, Taiwan observed "
              f"{TAIWAN in observed}; undistributed ICSG rows {len(und)}, Taiwan's "
              f"{len(tw)} ({tw.Value.sum():,.0f} kt CC)")
    for o in (base, new):
        feed = _icsg_inputs(o)[0]
        _, imp, _ = L._chain(o)
        fagg = np.load(ROOT / "imp" / imp / "agg_result.npz")["f_agg"]
        log = pd.read_parquet(ROOT / "imp" / imp / "imputation_log.parquet")
        e = E[o][(E[o].valuation == "CC") & (E[o].sector_from == "Cathode")
                 & (E[o].sector_to == "FU") & (E[o].region_to == "ROW")
                 & E[o].time.isin(EARLY)]
        e = e.assign(fed=_fed(feed)[e.edge_id.to_numpy()],
                     fagg=fagg[e.edge_id.to_numpy()])
        dom = e[e.region_from == "ROW"].set_index("time").reindex(EARLY)
        wrote = log[log.edge_id.isin(dom.edge_id)].method.value_counts().to_dict()
        print(f"   {o} ROW domestic semis leg 1995–2011: covered by the ICSG feed in "
              f"{int(dom.fed.notna().sum())} of {len(EARLY)} years, fed "
              f"{dom.fed.min():.0f}–{dom.fed.max():.0f}; imputer writes {wrote}")
        print("     feed / f_agg / post: " + "  ".join(
            f"{t[-2:]}:{np.nan_to_num(dom.fed[t]):.0f}/{dom.fagg[t]:.0f}/"
            f"{dom.post[t]:.0f}" for t in EARLY))
        g = e.groupby("time")[["fed", "fagg"]].sum().reindex(EARLY)
        d = (g.fagg - g.fed).abs()
        print(f"     ROW semis f_agg − ICSG fan: max |Δ| {d.max():.3g} kt CC; years "
              f"within 0.5 kt CC {int((d <= 0.5).sum())} of {len(EARLY)}")


# ---------------------------------------------------------------------------
# T — Taiwan
# ---------------------------------------------------------------------------

def _partners(legs: pd.DataFrame, side: str) -> str:
    top = legs.groupby(side).post.sum().sort_values(ascending=False).head(5)
    return ", ".join(f"{k} {v:,.0f}" for k, v in top.items())


def taiwan(E: dict, base: str, new: str) -> None:
    """T: Taiwan's ICSG rows, its semis use against ICSG, its partners by sector both
    ways, its concentrate, and its scrap."""
    print("\n== T Taiwan (kt CC)")
    raw = _icsg_raw(_icsg_inputs(new)[1])
    tw = raw[raw.root == TAIWAN]
    print("   ICSG rows for Taiwan by process / type / side: "
          + str(tw.groupby(["Process", "Type", "Supply_Use"]).size().to_dict()))
    semi = tw[(tw.Process == "Semi") & (tw.Supply_Use == "Use")]
    icsg = semi.groupby(semi.Year.astype(str)).Value.sum()
    for o in (base, new):
        e = E[o][E[o].valuation == "CC"]
        fu = e[(e.sector_from == "Cathode") & (e.sector_to == "FU")
               & (e.region_to == TAIWAN)]
        g = fu.groupby("time")[["prior", "post"]].sum().reindex(icsg.index).fillna(0.0)
        print(f"   {o} semis use 1995–2024: ICSG {icsg.sum():,.0f}, prior "
              f"{g.prior.sum():,.0f}, post {g.post.sum():,.0f} "
              f"(post/ICSG {g.post.sum() / icsg.sum():.3f}); by year post/ICSG "
              f"{(g.post / icsg).min():.2f}–{(g.post / icsg).max():.2f}")
        into = e[(e.region_to == TAIWAN) & (e.region_from != TAIWAN)]
        out = e[(e.region_from == TAIWAN) & (e.region_to != TAIWAN)]
        for s in ("Cathode", "Concentrate", "Anode", "Matte", "Scrap"):
            i, x = into[into.sector_from == s], out[out.sector_from == s]
            if (i.post.abs().sum() + i.prior.abs().sum() + x.post.abs().sum()
                    + x.prior.abs().sum()) < 0.5:
                continue
            print(f"     {s}: imports prior {i.prior.sum():,.0f} post "
                  f"{i.post.sum():,.0f}, top origins (post) {_partners(i, 'region_from')}"
                  f"; exports prior {x.prior.sum():,.0f} post {x.post.sum():,.0f}, top destinations "
                  f"(post) {_partners(x, 'region_to') or 'none'}")
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

def rest_of_world(E: dict, j: pd.DataFrame, base: str, new: str) -> None:
    """R: ROW's refined cathode retention, its semis use against its ICSG fan, and its
    domestic semis leg against ROW's production less its exports; its Δz² by sector and
    by anchor band."""
    print("\n== R ROW (kt CC)")
    for o in (base, new):
        e = E[o][E[o].valuation == "CC"]
        cat = e[e.sector_from == "Cathode"]
        cp = (e[(e.sector_from == "CP") & (e.sector_to == "Cathode")
                & (e.region_to == "ROW")].groupby("time")[["prior", "post"]].sum())
        fu = cat[(cat.sector_to == "FU") & (cat.region_to == "ROW")]
        fu = fu.assign(fed=_fed(_icsg_inputs(o)[0])[fu.edge_id.to_numpy()])
        g = fu.groupby("time")[["fed", "prior", "post"]].sum()
        dom = (fu[fu.region_from == "ROW"].set_index("time")[["fed", "post"]]
               .rename(columns={"fed": "dom_fed", "post": "dom_post"}))
        x = cat[(cat.region_from == "ROW") & (cat.region_to != "ROW")]
        exports = x.groupby("time").prior.sum().rename("exports")
        t = pd.concat([g, dom, cp.prior.rename("P"), exports], axis=1).fillna(0.0)
        print(f"   {o}")
        for name, yrs in (("1995–2011", EARLY), ("2012–2024", LATE)):
            w, ret = t.reindex(yrs), (cp.post / cp.prior).reindex(yrs)
            r = w.post / w.fed
            print(f"     {name}: CP→Cathode post/prior {ret.min():.2f}–{ret.max():.2f}; "
                  f"semis post/fan {r.min():.2f}–{r.max():.2f}, years below 0.85: "
                  f"{int((r < 0.85).sum())}, Σ post − fan "
                  f"{(w.post - w.fed).sum():+,.0f};\n"
                  f"       domestic leg Σ fed {w.dom_fed.sum():,.0f}, Σ post "
                  f"{w.dom_post.sum():,.0f}; Σ (production prior − export prior) "
                  f"{(w.P - w.exports).sum():,.0f}")
        print("     by year, CP→Cathode post/prior; domestic fed / post / production − "
              "exports:\n       " + "  ".join(
                  f"{k[-2:]}:{v.post_cp:.2f};{v.dom_fed:.0f}/{v.dom_post:.0f}/"
                  f"{v.P - v.exports:.0f}"
                  for k, v in t.assign(post_cp=cp.post / cp.prior)
                               .reindex(EARLY + LATE).iterrows()))
    P75.row_by_sector(j, base)
    r = j[j.region == "ROW"]
    print(f"   ROW Δz² 1995–2011 {r[r.year.isin(EARLY)].dz2.sum():+,.0f}, 2012–2024 "
          f"{r[r.year.isin(LATE)].dz2.sum():+,.0f}")
    cathode = r[r.label.str.split(" ").str[1].str.split("·").str[0] == "Cathode"]
    band = pd.cut(cathode.c_base.abs(), Z.BANDS, right=False)
    print("   ROW Cathode rows' Δz² by anchor band |c| (kt): "
          + str(cathode.groupby(band, observed=False).dz2.sum().round(0).to_dict()))


# ---------------------------------------------------------------------------
# N — netting
# ---------------------------------------------------------------------------

def netting(v_base: int, v_new: int) -> None:
    """N: Taiwan's ledger rows and T by year, ROW's cathode balance in each ledger, and
    BACI 490's cathode imports in the base version."""
    print(f"\n== N netting (Taiwan, ROW), baci v{v_base} → v{v_new} ledger (kt CC)")
    for v in (v_base, v_new):
        led = P69._ledger(v)
        tw = led[led.region == TAIWAN]
        print(f"   v{v}: Σ T_applied {led.T_applied.sum():,.0f}; ROW "
              f"{led[led.region == 'ROW'].T_applied.sum():,.0f}; Taiwan rows {len(tw)}, "
              f"T_applied {tw.T_applied.sum():,.0f}")
        if len(tw):
            print(tw.groupby(["sector", "label"]).agg(
                n=("time", "size"), I=("I", "sum"), X=("X", "sum"),
                T_applied=("T_applied", "sum"),
                resid=("resid", "sum")).round(0).to_string())
            by_year = tw.pivot_table(index="sector", columns="time", values="T_applied",
                                     aggfunc="sum").fillna(0.0)
            print("     Taiwan T_applied by year:")
            for s, row in by_year.iterrows():
                print(f"       {s}: "
                      + " ".join(f"{t[-2:]}:{v:.0f}" for t, v in row.items()))
        r = led[(led.region == "ROW") & (led.sector == "Cathode")].set_index("time")
        gap = (r.U - (r.P + r.I - r.X)).sort_index()
        print(f"     ROW cathode, ICSG use U − (ICSG production P + imports I − "
              f"exports X): 1995–2011 Σ {gap.reindex(EARLY).sum():+,.0f} "
              f"({gap.reindex(EARLY).min():+.0f} … {gap.reindex(EARLY).max():+.0f}), "
              f"2012–2024 Σ {gap.reindex(LATE).sum():+,.0f} "
              f"({gap.reindex(LATE).min():+.0f} … {gap.reindex(LATE).max():+.0f}); "
              f"by year "
              + " ".join(f"{t[-2:]}:{v:+.0f}" for t, v in gap.items()))
    v = pd.read_csv(BACI_IN / f"v{v_base}" / "values.csv")
    imp490 = v[(v.Sector == 740311) & (v.Region_to == 490)].groupby("Year").Value.sum()
    imp490 = imp490[(imp490.index >= 1995) & (imp490.index <= 2011)]
    print(f"   baci v{v_base}: BACI 490's cathode (HS 740311) imports 1995–2011 "
          f"{imp490.min():,.0f}–{imp490.max():,.0f} kt a year")


# ---------------------------------------------------------------------------
# S — stock legs newly above |z| 5
# ---------------------------------------------------------------------------

def stock_z(E: dict, base: str, new: str) -> None:
    """S: every stock leg whose |z| exceeds Z_BIG in the new run but not in the base."""
    z = {}
    for o in (base, new):
        f = P69._with_sigma(P65._family(E[o]), o)
        z[o] = f.set_index(["edge_id"])[["family", "time", "region_from", "region_to",
                                         "valuation", "prior", "post", "z"]]
    m = z[new].join(z[base][["prior", "post", "z"]], rsuffix="_base", how="left")
    grew = m[(m.z.abs() > Z_BIG) & ~(m.z_base.abs() > Z_BIG)]
    n_base, n_new = (int((z[o].z.abs() > Z_BIG).sum()) for o in (base, new))
    print(f"\n== S (z) stock legs with |z| > {Z_BIG:g}: base {n_base}, new {n_new}; "
          f"new above it that were not: {len(grew)}; into ROW "
          f"{int((grew.region_to == 'ROW').sum())}")
    print(grew.sort_values("z", key=abs, ascending=False).round(2).to_string())


def main(base: str, new: str, v_base: int, v_new: int) -> None:
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    E, S, j = P69.load(base, new)
    feeds(E, acc, base, new)
    taiwan(E, base, new)
    rest_of_world(E, j, base, new)
    netting(v_base, v_new)
    P69.round_e(E, S, j, base, new)
    P69.stock(E, base, new)
    P75.stock_breakdown(E, base, new)
    stock_z(E, base, new)
    P69.common_set(j, base)
    P75.one_run_only(S, base, new)
    P69.attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
