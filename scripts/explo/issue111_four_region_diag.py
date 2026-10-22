"""
issue111_four_region_diag.py
============================
#111 diagnosis: why D.R. Congo, South Africa, Zambia and Uzbekistan carry 34.7 % of the
scale-total Σz² on <opt>.  Reads persisted artefacts and the registered inputs only (no
per-year CSVs, #46) and prints:

    A   each region's scale-total Σz² split into the year blocks / row families the REP
        attributes to one mechanism each
    C   DR Congo's and South Africa's smelter chain, domestic and traded, prior and
        posterior per year (CC)
    W   the fan groups the ADR 0027 netting flags ``imports_exhausted``, which
        build_trade_shares withholds: by netting label, by gross imports against the
        feeds' 0.3 kt resolution, and by region
    V   ICSG cells an older yearbook keeps although a newer yearbook covering that year
        lists the region but not the cell (a revision to an exact zero, dropped at parse)

    python scripts/explo/issue111_four_region_diag.py <opt>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
from pysuto.acc import AccountingStructure              # noqa: E402
from pysuto.config import load_config                   # noqa: E402
from pysuto.param_relation import expand_edge_index     # noqa: E402
import issue40_signed_stock_diag as D                   # noqa: E402
import issue65_icsg_stock_parse_diag as P65             # noqa: E402

ROOT     = Path("data/proc")
LEDGER   = Path("data/input/feed/baci/current/reexport_ledger.csv")
ICSG_DIR = ROOT / "feedprep" / "icsg_clean"
VINTAGES = (2005, 2014, 2023, 2025)
RES_KT   = 0.3                  # the feeds' reporting resolution (ADR 0025)
DRC, ZA, ZM, UZ = "Congo, D.R.", "South Africa", "Zambia", "Uzbekistan"


def _y(s: pd.DataFrame) -> pd.Series:
    return s.year.astype(int)


BLOCKS = [
    (DRC, "1995–2003",                                      lambda s: _y(s) <= 2003),
    (DRC, "2004–2013: no ICSG smelter row / 0",             lambda s: _y(s).between(2004, 2013)),
    (DRC, "2014–2018: ICSG smelter 0, dropped at parse",    lambda s: _y(s).between(2014, 2018)),
    (DRC, "2019: ICSG smelter 1.5",                         lambda s: _y(s) == 2019),
    (DRC, "2020–2024",                                      lambda s: _y(s) >= 2020),
    (ZA,  "1995–2016",                                      lambda s: _y(s) <= 2016),
    (ZA,  "2017–2019: ICSG usage withheld",                 lambda s: _y(s).between(2017, 2019)),
    (ZA,  "2020–2024: ICSG smelter + refinery 0",           lambda s: _y(s) >= 2020),
    (ZM,  "cathode rows",                                   lambda s: s.label.str.contains("Cathode")),
    (ZM,  "anode / CFR rows",                               lambda s: s.label.str.contains("Anode|CFR")),
    (ZM,  "other rows",                                     lambda s: ~s.label.str.contains("Cathode|Anode|CFR")),
    (UZ,  "1995–1998",                                      lambda s: _y(s) <= 1998),
    (UZ,  "1999–2024",                                      lambda s: _y(s) >= 1999),
]

STEPS = [
    ("conc→smelt dom",  "Concentrate", "Smelting", "dom"),
    ("conc exp",        "Concentrate", "Smelting", "exp"),
    ("smelt→matte",     "Smelting",    None,       "own"),
    ("matte→CFR dom",   "Matte",       "CFR",      "dom"),
    ("CFR→anode",       "CFR",         "Anode",    "own"),
    ("anode→ER dom",    "Anode",       "ER",       "dom"),
    ("anode exp",       "Anode",       "ER",       "exp"),
]


def attribution(S: pd.DataFrame) -> None:
    """A: the four regions' scale-total Σz² by block."""
    tot = S.z2.sum()
    print(f"\n== A scale-total Σz² {tot:,.0f} ({len(S):,} rows)")
    for r, name, m in BLOCKS:
        s = S[S.region == r]
        s = s[m(s)]
        print(f"   {r:13s} {name:46s} {s.z2.sum():9,.0f}  {100 * s.z2.sum() / tot:5.2f} %"
              f"  ({len(s)} rows)")
    four = S[S.region.isin([DRC, ZA, ZM, UZ])].z2.sum()
    print(f"   four regions together {four:,.0f} = {100 * four / tot:.1f} %")


def chain(E: pd.DataFrame, region: str) -> None:
    """C: one region's smelter chain, CC, Σ per year."""
    e = E[E.valuation == "CC"]
    frm, to = e.region_from == region, e.region_to == region
    side = {"dom": frm & to, "exp": frm & ~to, "own": frm}
    cols = {}
    for name, sf, st, k in STEPS:
        m = (e.sector_from == sf) & side[k]
        if st is not None:
            m &= e.sector_to == st
        g = e[m].groupby(e.time.astype(str))[["prior", "post"]].sum()
        cols[(name, "prior")], cols[(name, "post")] = g.prior, g.post
    print(f"\n== C {region}, CC, Σ per year (prior | posterior)")
    print(pd.DataFrame(cols).fillna(0.0).round(1).to_string())


def withheld() -> None:
    """W: the groups build_trade_shares withholds as imports_exhausted."""
    led = pd.read_csv(LEDGER)
    ex = led[led.imports_exhausted.astype(bool)]
    print(f"\n== W imports_exhausted: {len(ex)} groups, ICSG usage U {ex.U.sum():,.0f} kt CC")
    print("   by netting label:")
    print(ex.groupby("label").agg(n=("U", "size"), U=("U", "sum")).round(0).to_string())
    small = ex.I < RES_KT
    print(f"   gross imports below {RES_KT} kt: {int(small.sum())} groups, U {ex[small].U.sum():,.0f}"
          f" kt CC; T_applied NaN (not netted): {int(ex.T_applied.isna().sum())} groups,"
          f" all of them below {RES_KT} kt: {bool((ex[ex.T_applied.isna()].I < RES_KT).all())}")
    by = ex.groupby(["sector", "region"]).agg(n=("U", "size"), U=("U", "sum"), I=("I", "sum"),
                                              X=("X", "sum"), P_missing=("P_missing", "sum"))
    print(by.sort_values("U", ascending=False).round(1).to_string())


def stale_vintages() -> None:
    """V: cells a newer yearbook revised away (the region is listed, the cell is not)."""
    key = ["Region", "Type", "Process", "Supply_Use", "Year"]
    V = {v: pd.read_csv(ICSG_DIR / f"icsg_{v}_flows.csv") for v in VINTAGES}
    span = {v: (int(x.Year.min()), int(x.Year.max())) for v, x in V.items()}
    tabs = {v: set(map(tuple, x[["Process", "Supply_Use"]].drop_duplicates().to_numpy()))
            for v, x in V.items()}
    regs = {v: set(x.Region) for v, x in V.items()}
    comb = pd.read_csv(ICSG_DIR / "icsg_flows_comb.csv")
    rows = []
    for r in comb.itertuples(index=False):
        for v in VINTAGES:
            if (v > r.Year_loaded_from_file and span[v][0] <= r.Year <= span[v][1]
                    and (r.Process, r.Supply_Use) in tabs[v] and r.Region in regs[v]):
                rows.append([getattr(r, k) for k in key] + [r.Value, r.Year_loaded_from_file, v])
    o = pd.DataFrame(rows, columns=key + ["Value", "from", "newer"]).drop_duplicates(key)
    print(f"\n== V cells kept from an older yearbook that a newer one revised away: {len(o)},"
          f" {o.Value.sum():,.0f} kt")
    print(o.sort_values(["Region", "Process", "Year"]).to_string(index=False))


def main(opt: str) -> None:
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id, storage_dir=str(ROOT / "acc"))
    ei = expand_edge_index(acc.edge_index, acc.node_index)
    S = P65._sums(opt, acc)
    E = D._edges(opt, ei)
    attribution(S)
    for r in (DRC, ZA):
        chain(E, r)
    withheld()
    stale_vintages()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
