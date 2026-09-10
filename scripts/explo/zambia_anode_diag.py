"""Zambia's phantom anode exports — raw-data forensics (no model state).

Reproduces every figure in
``docs/rep/analysis_20260910_zambia-anode-phantom-trade.md`` and the data block
of ``docs/rep/figs/zambia_anode_phantom_trade.html``.

Deliberately reads **inputs only** — the claim is about bias in the source, so
touching priors or posteriors would be circular:

  BACI  data/input/feed/baci/current/values.csv        (the feed as loaded)
  ICSG  data/proc/feedprep/icsg_clean/icsg_flows_comb.csv  (pre-``icsg_clean.cl``)

BACI reports gross mass of the traded commodity; ICSG already reports contained
copper. Everything below is put on the **contained-copper** basis using the
model's nominal copper contents, so the two are comparable — mixing the bases
overstates the anode ratio by about 6 %.

Usage:  python scripts/explo/zambia_anode_diag.py [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)

BACI = "data/input/feed/baci/current/values.csv"
ICSG = "data/proc/feedprep/icsg_clean/icsg_flows_comb.csv"
CONC = {260300: 0.275, 740200: 0.94, 740311: 0.9975}   # params/conc nominals
HS_ANODE = 740200
# UN M49 codes (data/input/feed/baci/current/conc.xlsx :: region_map)
ZAM, CD, IND = 894, 180, 356
CHINA = {156, 344, 446}                     # China + Hong Kong + Macao
HUBS = {"Switzerland": 757, "Singapore": 702, "Luxembourg": 442}
YEARS = list(range(1995, 2025))


def load():
    b = pd.read_csv(BACI)
    i = pd.read_csv(ICSG)
    return b, i


def _icsg(i, region, process, typ):
    s = i[(i.Region == region) & (i.Process == process) & (i.Supply_Use == "Supply")
          & (i.Type == typ)]
    return s.groupby("Year")["Value"].sum()


def series(b, i) -> dict:
    """Per-year series, all in kt contained copper."""
    a = b[b.Sector == HS_ANODE]
    k = CONC[HS_ANODE]
    sm = _icsg(i, "Zambia", "Smelting", "Primary")
    rf = _icsg(i, "Zambia", "Refining", "Primary")
    out = {"years": YEARS, "smelter": [], "refined": [],
           "zam": {n: [] for n in list(HUBS) + ["China", "India", "Other"]},
           "hub_in": {n: [] for n in HUBS}, "hub_out": {n: [] for n in HUBS},
           "chn_from_zam": [], "chn_total": []}
    for y in YEARS:
        z = a[(a.Region_from == ZAM) & (a.Year == y)]
        tot, named = z.Value.sum() * k, 0.0
        for n, c in HUBS.items():
            v = z[z.Region_to == c].Value.sum() * k
            out["zam"][n].append(round(v, 2)); named += v
        for n, sel in (("China", z.Region_to.isin(CHINA)), ("India", z.Region_to == IND)):
            v = z[sel].Value.sum() * k
            out["zam"][n].append(round(v, 2)); named += v
        out["zam"]["Other"].append(round(tot - named, 2))
        out["smelter"].append(round(float(sm.get(y, 0)), 1))
        out["refined"].append(round(float(rf.get(y, 0)), 1))
        for n, c in HUBS.items():
            out["hub_in"][n].append(round(a[(a.Region_to == c) & (a.Year == y)].Value.sum() * k, 2))
            out["hub_out"][n].append(round(a[(a.Region_from == c) & (a.Year == y)].Value.sum() * k, 2))
        ci = a[a.Region_to.isin(CHINA) & (a.Year == y)]
        out["chn_from_zam"].append(round(ci[ci.Region_from == ZAM].Value.sum() * k, 2))
        out["chn_total"].append(round(ci.Value.sum() * k, 2))
    return out


def report(b, i, d) -> None:
    Y = d["years"]; H = list(HUBS)
    hub = [sum(d["zam"][n][j] for n in H) for j in range(len(Y))]
    tot = [sum(d["zam"][n][j] for n in d["zam"]) for j in range(len(Y))]

    print("\nA · Zambia anode exports vs the physical ceiling (kt contained Cu)")
    print(f"{'yr':>5}{'smelter':>9}{'exports':>9}{'hubs':>8}{'China':>8}"
          f"{'other':>8}{'exp/sm':>8}{'exHub/sm':>9}")
    for j, y in enumerate(Y):
        if y < 2010:
            continue
        sm = d["smelter"][j]
        print(f"{y:>5}{sm:>9.1f}{tot[j]:>9.1f}{hub[j]:>8.1f}{d['zam']['China'][j]:>8.1f}"
              f"{d['zam']['Other'][j]:>8.1f}{tot[j]/sm:>8.2f}{(tot[j]-hub[j])/sm:>9.2f}")

    i16 = Y.index(2016)
    hin = [sum(d["hub_in"][n][j] for n in H) for j in range(len(Y))]
    hout = [sum(d["hub_out"][n][j] for n in H) for j in range(len(Y))]
    print(f"\nB · The hubs absorb and do not release, 2016-2024")
    print(f"    in {sum(hin[i16:]):,.0f} kt   out {sum(hout[i16:]):,.1f} kt "
          f"({sum(hout[i16:])/sum(hin[i16:]):.1%})   max out in any year "
          f"{max(max(d['hub_out'][n]) for n in H):.1f} kt")

    print("\nC · China's imports from Zambia are not diverted (kt Cu)")
    for y in (2014, 2015, 2016, 2017, 2019, 2022, 2024):
        j = Y.index(y)
        print(f"    {y}  from Zambia {d['chn_from_zam'][j]:7.1f}   total {d['chn_total'][j]:7.1f}")

    print("\nD · Congolese transit ruled out (kt gross as reported)")
    for c, n in ((260300, "Concentrate"), (740200, "Anode"), (740311, "Cathode")):
        s = b[(b.Sector == c) & (b.Region_from == CD) & (b.Region_to == ZAM)] \
            .groupby("Year")["Value"].sum()
        print(f"    Congo->Zambia {n:<12}"
              + "  ".join(f"{y} {s.get(y, 0):7.1f}" for y in (2013, 2017, 2022)))

    print("\nE · Zambia's full copper balance, contained Cu (kt)")
    for y in (2013, 2017, 2022):
        zi = lambda p, t: float(_icsg(i, "Zambia", p, t).get(y, 0))
        mine, sm, sx = zi("Mining", "Concentrates"), zi("Smelting", "Primary"), zi("SXEW", "Cathode")
        impc = b[(b.Sector == 260300) & (b.Region_to == ZAM) & (b.Year == y)].Value.sum() * CONC[260300]
        exp = {c: b[(b.Sector == c) & (b.Region_from == ZAM) & (b.Year == y)].Value.sum() * CONC[c]
               for c in CONC}
        te, avail = sum(exp.values()), sm + sx
        print(f"    {y}  mine {mine:6.1f} + Congo conc {impc:6.1f} -> plant output {avail:6.1f}"
              f"   recorded exports {te:7.1f}   = {te/avail:.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", help="write the per-year series to this path")
    args = ap.parse_args()
    b, i = load()
    d = series(b, i)
    report(b, i, d)
    if args.json:
        Path(args.json).write_text(json.dumps(d, separators=(",", ":")))
        print(f"\nseries -> {args.json}")


if __name__ == "__main__":
    main()
