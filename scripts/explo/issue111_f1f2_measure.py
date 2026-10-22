"""
issue111_f1f2_measure.py
========================
#111 measurement of F1 (ADR 0027's ``imports_exhausted`` withhold needs gross imports at the
feeds' 0.3 kt resolution; rejected on measurement, opt_20261022_105118_b7a659a8) and F2 (a
newer yearbook's exact zero supersedes the older vintage's value; measured alone after).
Prints #111's pre-registered items for <new> against <base>:

    1   South Africa's 2017–2019 scale-total Σz²
    2   DR Congo's 2019 ``supplytotout ER`` and ``usetotout CER`` rows, CC and Mass
    3   common-set scale-total Σz²
    4   the Round-E items (ADR 0025)
    5   the regions whose scale-total Σz² rises most, DR Congo included
    P   predictions: South Africa's 2017–2019 domestic smelter chain against ICSG's intake,
        and the C1–C4 contradiction blocks of the diagnosis REP

    python scripts/explo/issue111_f1f2_measure.py <opt_base> <opt_new>
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import issue69_aggregate_gap_diag as P69                  # noqa: E402
import issue111_four_region_diag as R                     # noqa: E402

ZA_YRS   = ["2017", "2018", "2019"]
ICSG_SM  = {"2017": 72.536, "2018": 51.813, "2019": 15.544}   # icsg v18 Smelting Use Concentrate
C_BLOCKS = {"C1 DRC 2004–2013 + 2019": (R.DRC, lambda s: s.year.astype(int).between(2004, 2013)
                                                    | (s.year == "2019")),
            "C2 ZA 2020–2024":         (R.ZA, lambda s: s.year.astype(int) >= 2020),
            "C3 Zambia cathode/anode": (R.ZM, lambda s: s.label.str.contains("Cathode|Anode|CFR")),
            "C4 Uzbekistan 1995–1998": (R.UZ, lambda s: s.year.astype(int) <= 1998)}


def items(E: dict, S: dict, base: str, new: str) -> None:
    """Items 1, 2 and the predictions."""
    print("\n== 1 South Africa 2017–2019 scale-total Σz²")
    for o in (base, new):
        s = S[o]
        print(f"   {o}: {s[(s.region == R.ZA) & s.year.isin(ZA_YRS)].z2.sum():,.0f}")
    print("\n== 2 DR Congo 2019 supplytotout ER + usetotout CER (CC, Mass)")
    for o in (base, new):
        s = S[o]
        x = s[(s.region == R.DRC) & (s.year == "2019")
              & s.label.str.contains(r"supplytotout ER·|usetotout CER·")]
        print(f"   {o}: {len(x)} rows, Σz² {x.z2.sum():,.1f}")
    print("\n== P South Africa domestic Concentrate→Smelting, CC: prior | posterior vs ICSG intake")
    for o in (base, new):
        e = E[o]
        e = e[(e.valuation == "CC") & (e.sector_from == "Concentrate") & (e.sector_to == "Smelting")
              & (e.region_from == R.ZA) & (e.region_to == R.ZA)]
        g = e.groupby(e.time.astype(str))[["prior", "post"]].sum()
        print(f"   {o}: " + "  ".join(f"{y} {g.prior.get(y, 0.0):.1f} | {g.post.get(y, 0.0):.1f}"
                                      f" (ICSG {ICSG_SM[y]:.1f})" for y in ZA_YRS))
    print("\n== P contradiction blocks C1–C4, Σz² base → new")
    for name, (r, m) in C_BLOCKS.items():
        v = [S[o][(S[o].region == r) & m(S[o])].z2.sum() for o in (base, new)]
        print(f"   {name:26s} {v[0]:9,.0f} → {v[1]:9,.0f}  ({100 * (v[1] / v[0] - 1):+.1f} %)")


def main(base: str, new: str) -> None:
    E, S, j = P69.load(base, new)
    items(E, S, base, new)
    P69.common_set(j, base)
    P69.round_e(E, S, j, base, new)
    reg = pd.concat({o: S[o].groupby("region").z2.sum() for o in (base, new)}, axis=1).fillna(0.0)
    reg["d"] = reg[new] - reg[base]
    print("\n== 5 scale-total Σz² by region, largest rises:")
    print(reg.sort_values("d", ascending=False).head(6).round(0).to_string())
    print("   the four #111 regions:")
    print(reg.loc[[R.DRC, R.ZA, R.ZM, R.UZ]].round(0).to_string())


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
