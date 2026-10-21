"""
issue110_russia_refinery_diag.py
================================
#110 measurement: the 2005 yearbook's merged refinery cells pair their values with primary
and secondary where the SX-EW table reports no output (icsg v16 → v17; Russia's 1995–2003
anode intake 50–296 → 552–897 kt). Prints #110's decision items for <new> against <base>:

    R   Russia: scale-total Σz² over 1995–2003 and in all (item 1), its CC stock legs above
        |z| 5 (item 2), and its CC chain prior / posterior by step, 1995–2003 (predictions)
    E   the Round-E items (ADR 0025) (item 4)
    C   common-set sum-row z² by region, sector and year (item 3); the regions whose
        scale-total Σz² rises most (item 5)

    python scripts/explo/issue110_russia_refinery_diag.py <opt_base> <opt_new>
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import issue65_icsg_stock_parse_diag as P65               # noqa: E402
import issue69_aggregate_gap_diag as P69                  # noqa: E402

REGION = "Russia"
YEARS  = [str(y) for y in range(1995, 2004)]
Z_BIG  = 5.0
STEPS  = [("CO", "Mining"), ("MF", "Concentrate"), ("Smelting", "Matte"), ("CFR", "Anode"),
          ("CP", "Cathode"), ("Scrap", "Smelting")]


def russia(E: dict, S: dict, base: str, new: str) -> None:
    """R: Russia's scale totals, stock legs and chain, base → new."""
    print(f"\n== R {REGION}")
    for o in (base, new):
        s = S[o][S[o].region == REGION]
        f = P69._with_sigma(P65._family(E[o]), o)
        f = f[(f.valuation == "CC") & (f.region_from == REGION)]
        big = f[f.z.abs() > Z_BIG]
        print(f"   {o}: scale-total Σz² {s.z2.sum():,.0f}, 1995–2003 "
              f"{s[s.year.isin(YEARS)].z2.sum():,.0f}; CC stock legs |z| > {Z_BIG:g}: {len(big)}, "
              f"Σ post {big.post.sum():+,.1f} kt CC")
    rows = []
    for o in (base, new):
        e = E[o]
        e = e[(e.valuation == "CC") & (e.region_from == REGION) & (e.region_to == REGION)
              & e.time.astype(str).isin(YEARS)]
        for sf, st in STEPS:
            x = e[(e.sector_from == sf) & (e.sector_to == st)]
            rows.append({"run": o[-8:], "step": f"{sf}→{st}", "prior": x.prior.sum(),
                         "post": x.post.sum()})
    t = pd.DataFrame(rows).pivot(index="step", columns="run", values=["prior", "post"])
    print("   domestic CC chain, Σ 1995–2003 (kt CC):")
    print(t.round(0).to_string())


def main(base: str, new: str) -> None:
    E, S, j = P69.load(base, new)
    russia(E, S, base, new)
    P69.round_e(E, S, j, base, new)
    P69.common_set(j, base)
    by = j.groupby(j.label.str.split("·").str[0].str.split(" ").str[-1]).dz2.sum().sort_values()
    print("   common-set Δz² by sector: " + "  ".join(f"{k} {v:+,.0f}" for k, v in by.items()))
    reg = pd.concat({o: S[o].groupby("region").z2.sum() for o in (base, new)}, axis=1).fillna(0.0)
    reg["d"] = reg[new] - reg[base]
    print("   scale-total Σz² by region, largest rises (item 5):")
    print(reg.sort_values("d", ascending=False).head(6).round(0).to_string())
    P69.attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
