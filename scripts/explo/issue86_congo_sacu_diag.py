"""
issue86_congo_sacu_diag.py
==========================
#86 measurement of #84 and #85: BACI 178 "Congo" mapped to Congo, D.R. and 711 "Southern
African Customs Union" to South Africa instead of ROW (#84), and the 2005 ICSG yearbook's
"Congo" relabelled to Congo, D.R. for 1995–2003 with its 2004 rows dropped (#85).
Reads persisted run artefacts, the feed runs' sidecars, the constraint runs' σ_a, two
netted BACI versions' ledgers, the gross BACI version's raw rows and the ICSG input
versions; builds the sum-row table in memory (no per-year CSVs, #46). Prints #86's
pass-rule items and the numbers its predictions are scored on, for <new> against <base>.
The sections shared with #69, #75 and #79 are imported from their diagnostics:

    F   inputs: the region-map rows of 178 and 711 in the gross and each chain's netted
        BACI version; the ICSG diff between the chains; the Congo rows by vintage (item 1)
    N   netting: the cathode U − (P + I − X) of ROW (item 2), DR Congo and South Africa in
        each ledger, by year; DR Congo's and South Africa's T and labels (hubs)
    R   ROW: its cathode-use sum rows' Σz² in 2012–2024 (item 3); #79's ROW section
    D   DR Congo: CP→Cathode posterior/prior by year (item 6); its cathode export legs by
        partner and year, 2019–2023; its gross cathode to Tanzania and Singapore; its
        concentrate exports; its 1995–2003 nodes; its Δz² by year and sector
    Z   South Africa 1995–1999: refined cathode, trade by sector, partners, its semis-use
        fan; 711's anode against Namibia's own from 2000
    G   ROW's remaining cathode gap by country, ICSG labels against BACI codes (#82)
    E   the Round-E items (ADR 0025) (item 4)
    S   stock legs: net per family; every leg newly above |z| 5; ROW's 2022–23 MESC/USC
        legs and the Kazakhstan/Russia → ROW legs of #82 (item 5)
    C   common-set sum-row z², every year's Δ, and the sum rows only one run has
    A   attribution of the largest moves

    python scripts/explo/issue86_congo_sacu_diag.py <opt_base> <opt_new> \\
                                        <baci_v_base> <baci_v_new> <baci_v_new_gross>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import build_baci_reexport as B                           # noqa: E402
import issue65_icsg_stock_parse_diag as P65               # noqa: E402
import issue69_aggregate_gap_diag as P69                  # noqa: E402
import issue75_scandinavia_split_diag as P75              # noqa: E402
import issue79_baci490_taiwan_diag as P79                 # noqa: E402
import roundL_reexport_diag as L                          # noqa: E402

ROOT          = Path("data/proc")
ICSG_IN       = Path("data/input/feed/icsg")
BACI_IN       = Path("data/input/feed/baci")
COUNTRY_CODES = Path("data/input/raw/BACI_HS92_V202601/country_codes_V202601.csv")
DRC           = "Congo, D.R."
ZA            = "South Africa"
CODES         = {178: "Congo", 711: "Southern African Customs Union"}
EARLY         = P79.EARLY
LATE          = P79.LATE
SACU_YEARS    = [str(y) for y in range(1995, 2000)]    # 711 carries South Africa's trade
DRC_OLD       = [str(y) for y in range(1995, 2004)]    # the 2005 yearbook's years (#85)
EXCESS_YEARS  = [str(y) for y in range(2019, 2024)]    # DR Congo's cathode excess
PREDICTED     = [str(y) for y in range(2020, 2024)]    # the window #86's predictions name
HS_CATHODE    = 740311
HS_ANODE      = 740200
DRC_CODES     = (178, 180)
TANZANIA      = 834
SINGAPORE     = 702
NAMIBIA       = 516
# ICSG source labels whose BACI country name differs beyond case and punctuation;
# "congorep" matches only on BACI versions that still fold 178 into ROW
ICSG_TO_BACI = {"bolivia": "boliviaplurinationalstateof", "congorep": "congo",
                "czechrep": "czechia", "laos": "laopeoplesdemrep",
                "northkorea": "dempeoplesrepofkorea", "turkey": "trkiye"}


def _baci_version(opt: str) -> int:
    """The netted BACI input version the run's BACI feed was built from."""
    feed = L._chain(opt)[2]["baci"]
    meta = json.loads((ROOT / "feed" / "registry" / feed / "metadata.json").read_text())
    return meta["parents"]["inputs"]["feed"]["version"]


def _key(label: str) -> str:
    """A country label reduced to its lower-case letters, to match ICSG to BACI names."""
    return re.sub(r"[^a-z]", "", str(label).lower())


def _region_map(v: int) -> pd.Series:
    """BACI version ``v``'s region map, source code → root region."""
    rm = pd.read_excel(BACI_IN / f"v{v}" / "conc.xlsx", sheet_name="region_map")
    return rm.set_index("src_region").root_region


def _gross_by_year(v_gross: int, sector: int, src, dst=None) -> pd.Series:
    """Gross BACI rows of one HS sector from codes ``src`` (to ``dst``), summed per year."""
    v = pd.read_csv(BACI_IN / f"v{v_gross}" / "values.csv")
    m = (v.Sector == sector) & v.Region_from.isin(src)
    if dst is not None:
        m &= v.Region_to.isin(dst)
    return v[m].groupby("Year").Value.sum()


# ---------------------------------------------------------------------------
# F — inputs (item 1)
# ---------------------------------------------------------------------------

def inputs(base: str, new: str, v_gross: int) -> None:
    """F: the region-map rows of 178 and 711, the ICSG diff, and the Congo rows by vintage."""
    print("\n== F inputs (item 1)")
    root = _region_map(v_gross)
    print(f"   gross baci v{v_gross} maps "
          + ", ".join(f"{k} {n!r} → {root[k]}" for k, n in CODES.items()))
    v = {}
    for o in (base, new):
        v_baci, v_icsg = _baci_version(o), P75._icsg(o)[1]
        v[o] = v_icsg
        root = _region_map(v_baci)
        raw = pd.read_csv(ICSG_IN / f"v{v_icsg}" / "values.csv")
        c = raw[raw.Region.isin(["Congo Rep.", DRC])]
        g = c.groupby(["Region", "Year_loaded_from_file"]).Year.agg(["min", "max", "size"])
        y04 = c[c.Year == 2004].groupby("Region").size().to_dict()
        print(f"   {o}: baci v{v_baci} maps "
              + ", ".join(f"{k} {n!r} → {root[k]}" for k, n in CODES.items())
              + f"; icsg v{v_icsg} Congo rows by vintage (first, last year, rows):\n"
              + "\n".join(f"     {r} {v}: {a['min']}–{a['max']}, {a['size']}"
                          for (r, v), a in g.iterrows())
              + f"\n     2004 rows by label: {y04}")
    a, b = (pd.read_csv(ICSG_IN / f"v{v[o]}" / "values.csv") for o in (base, new))
    m = a.merge(b, on=P75.KEY, how="outer", suffixes=("_base", "_new"), indicator=True)
    d = m[(m._merge != "both") | ((m.Value_base - m.Value_new).abs() > 1e-9)]
    print(f"   icsg v{v[base]} → v{v[new]}: {len(d)} rows differ, in regions "
          f"{sorted(d.Region.unique())}; only in v{v[base]} "
          f"{int((d._merge == 'left_only').sum())}, only in v{v[new]} "
          f"{int((d._merge == 'right_only').sum())}")


# ---------------------------------------------------------------------------
# N — netting (item 2)
# ---------------------------------------------------------------------------

def netting(v_base: int, v_new: int) -> None:
    """N: the cathode balance U − (P + I − X) of ROW, DR Congo and South Africa in each
    ledger; a missing U (DR Congo reports no semis use) counts as 0."""
    print(f"\n== N netting, baci v{v_base} → v{v_new} ledger (kt CC)")
    for v in (v_base, v_new):
        led = P69._ledger(v)
        c = led[led.sector == "Cathode"]
        print(f"   v{v}")
        for reg in ("ROW", DRC, ZA):
            d = c[c.region == reg].set_index("time").sort_index()
            gap = d.U.fillna(0.0) - (d.P + d.I - d.X)
            print(f"     {reg} cathode U − (P + I − X): 1995–2011 Σ "
                  f"{gap.reindex(EARLY).sum():+,.0f}, 2012–2024 Σ "
                  f"{gap.reindex(LATE).sum():+,.0f} ({gap.reindex(LATE).min():+.0f} … "
                  f"{gap.reindex(LATE).max():+.0f}); by year "
                  + " ".join(f"{k[-2:]}:{x:+.0f}" for k, x in gap.items()))
        for reg in (DRC, ZA):
            r = led[led.region == reg]
            t = r[r.T_applied > 0.5].pivot_table(index="sector", columns="time",
                                                 values="T_applied", aggfunc="sum")
            print(f"     {reg}: labels {r.label.value_counts().to_dict()}; T_applied > 0.5 by "
                  "sector and year: " + ("; ".join(
                      f"{s} " + " ".join(f"{k[-2:]}:{x:.0f}" for k, x in row.dropna().items())
                      for s, row in t.iterrows()) or "none"))


# ---------------------------------------------------------------------------
# R — ROW's cathode-use sum rows (item 3)
# ---------------------------------------------------------------------------

def row_cathode(j: pd.DataFrame, base: str) -> None:
    """R: Σz² of ROW's Cathode sum rows in 2012–2024, base against new."""
    r = j[(j.region == "ROW")
          & (j.label.str.split(" ").str[1].str.split("·").str[0] == "Cathode")]
    late = r[r.year.isin(LATE)]
    zb, zn = late.z2_base.sum(), late.z2_new.sum()
    print(f"\n== R ROW cathode-use sum rows, 2012–2024 (item 3): {len(late)} rows, Σz² "
          f"{zb:,.0f} → {zn:,.0f} (new/base {zn / zb:.4f}, vs {base}); 1995–2011 "
          f"{r[r.year.isin(EARLY)].z2_base.sum():,.0f} → "
          f"{r[r.year.isin(EARLY)].z2_new.sum():,.0f}")
    by = late.groupby("year")[["z2_base", "z2_new"]].sum().round(0)
    print("   by year, base/new: " + "  ".join(f"{t[-2:]}:{a:,.0f}/{b:,.0f}"
                                                for t, (a, b) in by.iterrows()))


# ---------------------------------------------------------------------------
# D — DR Congo (item 6)
# ---------------------------------------------------------------------------

def dr_congo(E: dict, j: pd.DataFrame, base: str, new: str, v_gross: int) -> None:
    """D: refined cathode retention, cathode export legs by partner and year, gross cathode
    to Tanzania and Singapore, concentrate exports, the 1995–2003 nodes, and DR Congo's
    Δz² by year and by sector."""
    print(f"\n== D {DRC} (kt CC)")
    for o in (base, new):
        e = E[o][E[o].valuation == "CC"]
        cp = (e[(e.sector_from == "CP") & (e.sector_to == "Cathode") & (e.region_to == DRC)]
              .groupby("time")[["prior", "post"]].sum())
        ret = cp.post / cp.prior
        print(f"   {o}")
        for name, yrs in (("2019–2023", EXCESS_YEARS), ("2020–2023", PREDICTED)):
            w = cp.reindex(yrs)
            print(f"     CP→Cathode {name} Σ prior {w.prior.sum():,.0f}, Σ post "
                  f"{w.post.sum():,.0f} ({w.post.sum() - w.prior.sum():+,.0f})")
        zero = cp[cp.prior == 0.0]
        print(f"     CP→Cathode post/prior (item 6): years above 1.05 with a prior: "
              f"{int(((ret > 1.05) & (cp.prior > 0)).sum())}; zero-prior years' posterior "
              + (" ".join(f"{t[-2:]}:{x:.2f}" for t, x in zero.post.items()) or "none")
              + "; by year " + " ".join(f"{t[-2:]}:{x:.3f}" for t, x in ret.items()))
        x = e[(e.sector_from == "Cathode") & (e.region_from == DRC) & (e.region_to != DRC)]
        g = x.groupby("time")[["prior", "post"]].sum()
        print("     cathode exports prior/post by year: " + " ".join(
            f"{t[-2:]}:{a:.0f}/{b:.0f}" for t, (a, b) in g.round(0).iterrows()))
        for name, yrs in (("2019–2023", EXCESS_YEARS), ("2020–2023", PREDICTED)):
            w = x[x.time.isin(yrs)]
            hi = w.groupby("region_to")[["prior", "post"]].sum()
            hi = hi.assign(cut=hi.post - hi.prior).sort_values("cut").head(8)
            print(f"     {name} by partner (Σ prior {w.prior.sum():,.0f}, post "
                  f"{w.post.sum():,.0f}), largest cuts: "
                  + ", ".join(f"{k} {r.prior:,.0f}→{r.post:,.0f}" for k, r in hi.iterrows()))
        w = x[x.time.isin(EXCESS_YEARS)]
        top = w.groupby("region_to").prior.sum().sort_values(ascending=False).head(8).index
        py = w[w.region_to.isin(top)].pivot_table(index="region_to", columns="time",
                                                  values=["prior", "post"], aggfunc="sum")
        print("     2019–2023 by partner and year, prior/post:")
        for k in top:
            print(f"       {k}: " + " ".join(
                f"{t[-2:]}:{py.loc[k, ('prior', t)]:.0f}/{py.loc[k, ('post', t)]:.0f}"
                for t in EXCESS_YEARS if ('prior', t) in py.columns))
        cx = e[(e.sector_from == "Concentrate") & (e.region_from == DRC) & (e.region_to != DRC)]
        g = cx.groupby("time")[["prior", "post"]].sum().reindex(EARLY).dropna()
        print("     concentrate exports prior/post, 1995–2011: " + " ".join(
            f"{t[-2:]}:{a:.0f}/{b:.0f}" for t, (a, b) in g.round(0).iterrows()))
        old = e[e.time.isin(DRC_OLD) & ((e.region_from == DRC) | (e.region_to == DRC))]
        sp = old.groupby(["sector_from", "sector_to"])[["prior", "post"]].sum()
        sp = sp[(sp.prior.abs() + sp.post.abs()) > 1.0]
        print("     1995–2003, by sector pair, prior/post: " + ("  ".join(
            f"{a}→{b} {r.prior:,.0f}/{r.post:,.0f}" for (a, b), r in sp.iterrows())
            or "none"))
    for dst, name in ((TANZANIA, "Tanzania"), (SINGAPORE, "Singapore")):
        g = _gross_by_year(v_gross, HS_CATHODE, DRC_CODES, [dst]).reindex(range(2018, 2024))
        print(f"   gross baci v{v_gross}: cathode from codes {DRC_CODES} to {name}, 2018–2023 Σ "
              f"{g.sum():,.0f} kt; by year "
              + " ".join(f"{t % 100:02d}:{x:.0f}" for t, x in g.fillna(0.0).items()))
    d = j[j.region == DRC]
    by_year = d.groupby("year").dz2.sum()
    print(f"   {DRC} common-set Δz² {d.dz2.sum():+,.0f}; by year "
          + " ".join(f"{t[-2:]}:{x:+,.0f}" for t, x in by_year.items()))
    sector = d.label.str.split(" ").str[1].str.split("·").str[0]
    print("     by sector, all years: " + "  ".join(
        f"{k} {v:+,.0f}" for k, v in d.groupby(sector).dz2.sum().sort_values().items()))


# ---------------------------------------------------------------------------
# Z — South Africa 1995–1999
# ---------------------------------------------------------------------------

def south_africa(E: dict, base: str, new: str, v_gross: int) -> None:
    """Z: South Africa's 1995–1999 refined cathode, trade by sector, partners and semis-use
    fan; 711's anode exports against Namibia's own from 2000."""
    print(f"\n== Z {ZA}, 1995–1999 (kt CC)")
    for o in (base, new):
        e = E[o][(E[o].valuation == "CC") & E[o].time.isin(SACU_YEARS)]
        out = e[(e.region_from == ZA) & (e.region_to != ZA)]
        inn = e[(e.region_to == ZA) & (e.region_from != ZA)]
        cp = e[(e.sector_from == "CP") & (e.sector_to == "Cathode") & (e.region_to == ZA)]
        print(f"   {o}: CP→Cathode prior {cp.prior.sum():,.0f} post {cp.post.sum():,.0f}")
        for side, legs, who in (("exports", out, "region_to"), ("imports", inn, "region_from")):
            g = legs.groupby("sector_from")[["prior", "post"]].sum()
            g = g[(g.prior.abs() + g.post.abs()) > 0.5]
            print(f"     {side} by sector, prior/post: " + ("  ".join(
                f"{s} {r.prior:,.0f}/{r.post:,.0f}" for s, r in g.iterrows()) or "none"))
            for s in ("Cathode", "Anode"):
                p = P79._partners(legs[legs.sector_from == s], who)
                if p:
                    print(f"       {s} {side}, top partners (post): {p}")
        fu = e[(e.sector_from == "Cathode") & (e.sector_to == "FU") & (e.region_to == ZA)]
        dom = fu.region_from == ZA
        print(f"     semis-use fan (Cathode→FU into {ZA}), prior/post: domestic "
              f"{fu[dom].prior.sum():,.0f}/{fu[dom].post.sum():,.0f}, imported "
              f"{fu[~dom].prior.sum():,.0f}/{fu[~dom].post.sum():,.0f}")
    sacu = _gross_by_year(v_gross, HS_ANODE, [711]).reindex(range(1995, 2000))
    nam = _gross_by_year(v_gross, HS_ANODE, [NAMIBIA]).reindex(range(2000, 2005))
    za = _gross_by_year(v_gross, HS_ANODE, [710]).reindex(range(2000, 2005))
    print(f"   gross baci v{v_gross} anode (HS {HS_ANODE}) exports, kt: 711 1995–1999 "
          + " ".join(f"{x:.0f}" for x in sacu.fillna(0.0))
          + "; from 2000, South Africa (710) " + " ".join(f"{x:.0f}" for x in za.fillna(0.0))
          + ", Namibia (516) " + " ".join(f"{x:.0f}" for x in nam.fillna(0.0)))


# ---------------------------------------------------------------------------
# G — ROW's remaining gap by country (#82)
# ---------------------------------------------------------------------------

def row_by_country(v_icsg: int, v_gross: int) -> None:
    """G: per ICSG label folded into ROW, cathode U − (P + I − X) on gross BACI values."""
    print(f"\n== G ROW's cathode gap by country, icsg v{v_icsg} + gross baci v{v_gross} "
          "(kt CC), largest |gap|")
    eff = B._eff_nominals()
    raw = P79._icsg_raw(v_icsg)
    r = raw[raw.root == "ROW"].assign(yr=lambda d: d.Year.astype(int))

    def ser(p, t, su):
        x = r[(r.Process == p) & (r.Type == t) & (r.Supply_Use == su)]
        return x.groupby(["Region", "yr"]).Value.sum()
    U = ser("Semi", "Cathode", "Use")
    P = ser("Refining", "Anode", "Use").mul(eff["ER"]).add(
        ser("SXEW", "Cathode", "Supply"), fill_value=0.0)
    icsg = pd.concat([U.rename("U"), P.rename("P")], axis=1).fillna(0.0).reset_index()
    rm = pd.read_excel(BACI_IN / f"v{v_gross}" / "conc.xlsx", sheet_name="region_map")
    row = set(rm.loc[rm.root_region == "ROW", "src_region"].astype(int))
    cc = pd.read_csv(COUNTRY_CODES)
    name = dict(zip(cc.country_code, cc.country_name))
    v = pd.read_csv(BACI_IN / f"v{v_gross}" / "values.csv")
    c = v[(v.Sector == HS_CATHODE) & v.Year.between(1995, 2024)]
    I = c[c.Region_to.isin(row)].groupby(["Region_to", "Year"]).Value.sum()
    X = c[c.Region_from.isin(row)].groupby(["Region_from", "Year"]).Value.sum()
    baci = pd.concat([I.rename("I"), X.rename("X")], axis=1).fillna(0.0)
    baci.index = baci.index.set_names(["code", "yr"])
    baci = baci.reset_index().assign(country=lambda d: d.code.map(name))
    icsg["key"] = icsg.Region.map(_key).replace(ICSG_TO_BACI)
    baci["key"] = baci.country.map(_key)
    for d in (icsg, baci):
        d["period"] = np.where(d.yr <= 2011, "1995-2011", "2012-2024")
    t = (icsg.groupby(["key", "period"])[["U", "P"]].sum()
         .join(baci.groupby(["key", "period"])[["I", "X"]].sum(), how="outer").fillna(0.0))
    lab = icsg.groupby("key").Region.first().combine_first(baci.groupby("key").country.first())
    t["gap"] = t.U - (t.P + t.I - t.X)
    unmatched = sorted(set(icsg.Region[~icsg.key.isin(set(baci.key))]))
    print(f"   ICSG ROW labels with no BACI country: {unmatched}")
    for per in ("1995-2011", "2012-2024"):
        x = t.xs(per, level="period")
        x = x.assign(name=lab.reindex(x.index).to_numpy())
        print(f"   {per}: Σ U {x.U.sum():,.0f}  P {x.P.sum():,.0f}  I {x.I.sum():,.0f}  "
              f"X {x.X.sum():,.0f}  gap {x.gap.sum():+,.0f}")
        top = x.reindex(x.gap.abs().sort_values(ascending=False).index).head(14)
        print("     " + "  ".join(f"{r['name']} {r.gap:+,.0f}" for _, r in top.iterrows()))


# ---------------------------------------------------------------------------
# S — the stock legs #82 names (item 5)
# ---------------------------------------------------------------------------

def stock_row(E: dict, base: str, new: str) -> None:
    """S: ROW's 2022–23 MESC/USC legs and the Kazakhstan/Russia → ROW legs of #82."""
    print("\n== S (#82) stock legs into ROW, z base → new")
    f = {o: P69._with_sigma(P65._family(E[o]), o).set_index("edge_id") for o in (base, new)}
    m = f[new][["family", "time", "region_from", "region_to", "valuation", "z"]].join(
        f[base][["z"]], rsuffix="_base")
    pick = m[(m.region_to == "ROW") & (
        ((m.region_from == "ROW") & m.time.isin(["2022", "2023"]))
        | (m.region_from.isin(["Kazakhstan", "Russia"])
           & m.time.isin(["1999", "2000", "2002"])))]
    pick = pick[(pick.z.abs() > 1.0) | (pick.z_base.abs() > 1.0)]
    print(pick.sort_values(["region_from", "time"]).round(2).to_string())


def by_year(j: pd.DataFrame) -> None:
    """C: the common-set Δz² of every year, unfiltered."""
    d = j.groupby("year").dz2.sum()
    print("   common-set Δz² by year: " + " ".join(f"{t[-2:]}:{x:+,.0f}" for t, x in d.items()))


def main(base: str, new: str, v_base: int, v_new: int, v_gross: int) -> None:
    E, S, j = P69.load(base, new)
    inputs(base, new, v_gross)
    netting(v_base, v_new)
    row_cathode(j, base)
    P79.rest_of_world(E, j, base, new)
    dr_congo(E, j, base, new, v_gross)
    south_africa(E, base, new, v_gross)
    row_by_country(P75._icsg(new)[1], v_gross)
    P69.round_e(E, S, j, base, new)
    P69.stock(E, base, new)
    P75.stock_breakdown(E, base, new)
    P79.stock_z(E, base, new)
    stock_row(E, base, new)
    P69.common_set(j, base)
    by_year(j)
    P75.one_run_only(S, base, new)
    P69.attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
