"""
issue92_refined_scope_diag.py
=============================
#92 measurement of #91 and #88: BACI's Cathode sector reads all refined copper (HS
740311, 740312, 740313, 740319), and BACI 178 is dropped in 2015–2023 as a mirror
duplicate of DR Congo's own declarations. Reads persisted run artefacts, the feed runs'
sidecars, the constraint runs' σ_a, two netted BACI versions' ledgers and two gross BACI
versions' raw rows; builds the sum-row table in memory (no per-year CSVs, #46). Prints
#92's pass-rule items and the numbers its predictions are scored on, for <new> against
<base>. Shared sections come from the #69, #75, #79 and #86 diagnostics:

    I   inputs: the gross diff by kind — new refined codes, 178's dropped rows, and the
        concentrate legs ADR 0033's copper price moves (item 1); 178 in the netted
        ledger's DR Congo rows, 2015–2023 (item 5)
    N   netting: the cathode U − (P + I − X) of ROW (item 6), DR Congo and South Africa
    U   Austria: cathode-use sum rows' Σz², 1995–2014 (item 2); its cathode node prior
        and posterior; its scrap rows (reported)
    D   DR Congo: CP→Cathode retention; cathode exports by year (reported)
    T   cathode trade of Chile, China and Italy, prior/post, by period (reported)
    E   the Round-E items (ADR 0025) (item 3)
    S   stock legs newly above |z| 5, their size by exporter, the net stock posterior (item 4)
    C   common-set sum-row z² by region, sector and year; rows only one run has
    A   attribution of the largest moves

    python scripts/explo/issue92_refined_scope_diag.py <opt_base> <opt_new> \\
        <baci_v_base> <baci_v_new> <baci_v_gross_base> <baci_v_gross_new>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import issue65_icsg_stock_parse_diag as P65               # noqa: E402
import issue69_aggregate_gap_diag as P69                  # noqa: E402
import issue75_scandinavia_split_diag as P75              # noqa: E402
import issue79_baci490_taiwan_diag as P79                 # noqa: E402
import issue86_congo_sacu_diag as P86                     # noqa: E402

BACI_IN    = Path("data/input/feed/baci")
NEW_CODES  = (740312, 740313, 740319)
CONGO      = 178
WINDOW     = (2015, 2023)
AUSTRIA    = "Austria"
TRADERS    = ("Chile", "China", "Italy", "Austria")
EARLY_AT   = [str(y) for y in range(1995, 2015)]         # Austria's cathode cluster
LATE_AT    = [str(y) for y in range(2015, 2025)]
KEY        = ["Year", "Sector", "Region_from", "Region_to"]


def _sector(label: pd.Series) -> pd.Series:
    """The sector a sum-row label totals ("usetotout Cathode·Austria CC" → Cathode)."""
    return label.str.split(" ").str[1].str.split("·").str[0]


# ---------------------------------------------------------------------------
# I — inputs (items 1 and 5)
# ---------------------------------------------------------------------------

def inputs(v_gross_base: int, v_gross_new: int) -> None:
    """I: the gross diff by kind, and code 178 in DR Congo's 2015–2023 ledger rows."""
    print(f"\n== I inputs, gross baci v{v_gross_base} → v{v_gross_new} (items 1, 5)")
    a = pd.read_csv(BACI_IN / f"v{v_gross_base}" / "values.csv")
    b = pd.read_csv(BACI_IN / f"v{v_gross_new}" / "values.csv")
    m = a.merge(b, on=KEY, how="outer", suffixes=("_a", "_b"), indicator=True)
    m["d"] = m.Value_b.fillna(0.0) - m.Value_a.fillna(0.0)
    d = m[(m._merge != "both") | (m.d.abs() > 1e-9)]
    new_code = d.Sector.isin(NEW_CODES) & (d._merge == "right_only")
    congo = ((d.Region_from == CONGO) | (d.Region_to == CONGO)) \
        & d.Year.between(*WINDOW) & (d._merge == "left_only")
    moved = (d._merge == "both") & (d.Sector == 260300)
    other = d[~(new_code | congo | moved)]
    print(f"   rows differing {len(d)}: new refined-code rows {int(new_code.sum())} "
          f"({d[new_code].Value_b.sum():,.0f} kt), code {CONGO} rows removed in "
          f"{WINDOW[0]}–{WINDOW[1]} {int(congo.sum())} "
          f"({d[congo].groupby('Sector').Value_a.sum().round(1).to_dict()}), "
          f"concentrate legs moved {int(moved.sum())}, anything else {len(other)}")
    mv = d[moved]
    print(f"   concentrate legs moved by ADR 0033's copper price: Σ|Δ| {mv.d.abs().sum():,.1f} "
          f"of {mv.Value_a.sum():,.0f} kt ({100 * mv.d.abs().sum() / mv.Value_a.sum():.3f} %), "
          f"net {mv.d.sum():+,.1f}")
    if len(other):
        print(other[KEY + ["Value_a", "Value_b", "_merge"]].head(10).to_string(index=False))
    nc = b[b.Sector.isin(NEW_CODES)]
    print("   new refined codes by sector, kt: "
          + str(nc.groupby("Sector").Value.sum().round(0).to_dict()))
    left = b[((b.Region_from == CONGO) | (b.Region_to == CONGO)) & b.Year.between(*WINDOW)]
    print(f"   code {CONGO} rows left in gross v{v_gross_new}, {WINDOW[0]}–{WINDOW[1]} "
          f"(item 5): {len(left)}")


# ---------------------------------------------------------------------------
# U — Austria (item 2)
# ---------------------------------------------------------------------------

def austria(E: dict, j: pd.DataFrame, base: str, new: str) -> None:
    """U: Austria's cathode-use rows' Σz², its cathode node, and its scrap rows."""
    a = j[j.region == AUSTRIA]
    s = _sector(a.label)
    cat = a[s == "Cathode"]
    for name, yrs in (("1995–2014", EARLY_AT), ("2015–2024", LATE_AT)):
        w = cat[cat.year.isin(yrs)]
        zb, zn = w.z2_base.sum(), w.z2_new.sum()
        print(f"{chr(10) if name == '1995–2014' else ''}== U {AUSTRIA} cathode-use sum rows, "
              f"{name}{' (item 2)' if name == '1995–2014' else ''}: {len(w)} rows, Σz² "
              f"{zb:,.0f} → {zn:,.0f} (new/base {zn / zb:.4f}, vs {base})")
    print(f"   {AUSTRIA} common-set Δz² {a.dz2.sum():+,.0f}; by sector "
          + "  ".join(f"{k} {v:+,.0f}" for k, v in a.groupby(s).dz2.sum().sort_values().items()))
    for o in (base, new):
        e = E[o][E[o].valuation == "CC"]
        x = e[(e.sector_from == "Cathode") & (e.region_from == AUSTRIA)
              & (e.region_to != AUSTRIA)]
        cp = e[(e.sector_from == "CP") & (e.sector_to == "Cathode") & (e.region_to == AUSTRIA)]
        for name, yrs in (("1995–2014", EARLY_AT), ("2015–2024", LATE_AT)):
            xs, cs = x[x.time.isin(yrs)], cp[cp.time.isin(yrs)]
            print(f"   {o} {name}: CP→Cathode prior {cs.prior.sum():,.0f} post "
                  f"{cs.post.sum():,.0f}; cathode exports prior {xs.prior.sum():,.0f} post "
                  f"{xs.post.sum():,.0f}")


# ---------------------------------------------------------------------------
# T — cathode trade of the largest movers (reported)
# ---------------------------------------------------------------------------

def traders(E: dict, base: str, new: str) -> None:
    """T: cathode exports and imports of Chile, China, Italy and Austria, prior/post."""
    print("\n== T cathode trade, Σ prior / Σ post (kt CC)")
    for reg in TRADERS:
        for o in (base, new):
            e = E[o][(E[o].valuation == "CC") & (E[o].sector_from == "Cathode")]
            x = e[(e.region_from == reg) & (e.region_to != reg)]
            i = e[(e.region_to == reg) & (e.region_from != reg)]
            print(f"   {reg:8s} {o}: " + "  ".join(
                f"{n} exp {x[x.time.isin(y)].prior.sum():,.0f}/{x[x.time.isin(y)].post.sum():,.0f}"
                f" imp {i[i.time.isin(y)].prior.sum():,.0f}/{i[i.time.isin(y)].post.sum():,.0f}"
                for n, y in (("1995–2011", P79.EARLY), ("2012–2024", P79.LATE))))


def stock_sizes(E: dict, base: str, new: str) -> None:
    """S: the CC stock legs newly above |z| 5 by exporter, and the net stock posterior."""
    z = {o: P69._with_sigma(P65._family(E[o]), o).set_index("edge_id") for o in (base, new)}
    m = z[new][["region_from", "valuation", "post", "z"]].join(
        z[base][["post", "z"]], rsuffix="_base")
    cc = m[m.valuation == "CC"]
    g = cc[(cc.z.abs() > P79.Z_BIG) & ~(cc.z_base.abs() > P79.Z_BIG)]
    print(f"   CC legs newly above |z| {P79.Z_BIG:g}: {len(g)}, Σ|post| {g.post.abs().sum():,.1f} "
          f"kt CC (base {g.post_base.abs().sum():,.1f}); by exporter "
          + "  ".join(f"{k} {int(n)} legs {v:+,.1f}" for k, (n, v) in
                      g.groupby("region_from").post.agg(["size", "sum"]).sort_values("sum")
                      .iterrows()))
    print(f"   all CC stock legs, 1995–2024: Σ post {cc.post_base.sum():+,.0f} → "
          f"{cc.post.sum():+,.0f} kt CC; Σ|post| {cc.post_base.abs().sum():,.0f} → "
          f"{cc.post.abs().sum():,.0f}")


def main(base, new, v_base, v_new, v_gross_base, v_gross_new) -> None:
    E, S, j = P69.load(base, new)
    inputs(v_gross_base, v_gross_new)
    P86.netting(v_base, v_new)
    austria(E, j, base, new)
    P86.dr_congo(E, j, base, new, v_gross_new)
    traders(E, base, new)
    P69.round_e(E, S, j, base, new)
    P79.stock_z(E, base, new)
    stock_sizes(E, base, new)
    P69.common_set(j, base)
    P86.by_year(j)
    by = j.groupby(_sector(j.label)).dz2.sum().sort_values()
    print("   common-set Δz² by sector: " + "  ".join(f"{k} {v:+,.0f}" for k, v in by.items()))
    P75.one_run_only(S, base, new)
    P69.attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:]))
