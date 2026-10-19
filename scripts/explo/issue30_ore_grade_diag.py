"""
issue30_ore_grade_diag.py
=========================
#30 measurement: ``ore_grade`` rebuilt (v7 → the new version) on the current ICSG input
(icsg v15, after ADR 0035 removed the phantom Mining/Total rows). Reads the two params
versions, the runs' sidecars and persisted artefacts; builds the sum-row and ore-grade
ratio-row tables in memory (no per-year CSVs, #46). Prints #30's pass-rule items and the
numbers its predictions are scored on, for <new> against <base>:

    I   inputs: the ore_grade version each run's imp cites (item 1); ROW's in-situ grade by
        year in both versions and the phantom-window test (item 2); the largest grade
        changes by region; rows with relative σ = (max − min)/4/nominal above 100 %
    R   CO Mass and Concentrate CC retention by mining region; the ore-grade ratio rows' Σz²
    E   the Round-E items (ADR 0025) (item 3)
    S   stock legs newly above |z| 5 and the net stock posterior (item 4)
    C   common-set sum-row z² by region, sector and year; rows only one run has
    A   attribution of the largest moves

    python scripts/explo/issue30_ore_grade_diag.py <opt_base> <opt_new> <og_v_base> <og_v_new>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scripts/explo")
import issue69_aggregate_gap_diag as P69                  # noqa: E402
import issue75_scandinavia_split_diag as P75              # noqa: E402
import issue79_baci490_taiwan_diag as P79                 # noqa: E402
import issue86_congo_sacu_diag as P86                     # noqa: E402
import issue92_refined_scope_diag as P92                  # noqa: E402
import roundL_reexport_diag as L                          # noqa: E402
from pysuto.acc import AccountingStructure                # noqa: E402
from pysuto.config import load_config                     # noqa: E402
from pysuto.diag import SolverDiag, SolverStatsDiagnostic  # noqa: E402

ROOT     = Path("data/proc")
OG_IN    = Path("data/input/params/ore_grade")
GRADE    = "oregradecosupply"            # the CO/Mining in-situ grade rows
MINERS   = ("Chile", "Peru", "China", "Congo, D.R.", "United States", "Australia",
            "Zambia", "Indonesia", "Russia", "ROW")
WINDOW   = [str(y) for y in range(2004, 2014)]
EARLY    = [str(y) for y in range(2000, 2004)]


def _grades(v: int) -> pd.DataFrame:
    """ore_grade v's CO/Mining grade rows: region, year, nominal, relative σ."""
    p = pd.read_csv(OG_IN / f"v{v}" / "params.csv", sep=None, engine="python", dtype=str)
    g = p[p.param_id.str.startswith(GRADE, na=False)].copy()
    g["region"] = g.param_id.str[len(GRADE):-4]          # the id carries region and year
    g["year"] = g.param_id.str[-4:]
    for c in ("value_nominal", "value_min", "value_max"):
        g[c] = g[c].astype(float)
    g["rel_sigma"] = (g.value_max - g.value_min) / 4.0 / g.value_nominal.abs()
    return g[["region", "year", "value_nominal", "rel_sigma"]]


def _imp_ore_grade(opt: str) -> int:
    """The ore_grade version the run's imp sidecar cites."""
    imp = L._chain(opt)[1]
    m = json.loads((ROOT / "imp" / imp / "metadata.json").read_text())
    return {p["id"]: p["version"] for p in m["parents"]["inputs"]["params"]}["ore_grade"]


def inputs(base: str, new: str, v_base: int, v_new: int) -> None:
    """I: the cited versions, ROW's grade, the largest changes, the >100 % relative σ."""
    print(f"\n== I inputs: imp cites ore_grade v{_imp_ore_grade(base)} (base), "
          f"v{_imp_ore_grade(new)} (new) (item 1)")
    a, b = _grades(v_base), _grades(v_new)
    for v, g in ((v_base, a), (v_new, b)):
        r = g[g.region == "row"].set_index("year").value_nominal
        cap = 1.5 * r.reindex(EARLY).max()
        w = r.reindex(WINDOW)
        print(f"   v{v} ROW grade by year: " + " ".join(
            f"{t[-2:]}:{100 * x:.2f}" for t, x in r.loc["1998":"2016"].items())
              + f" (%); 2004–2013 max {100 * w.max():.2f} %, cap 1.5 × 2000–03 max "
                f"{100 * cap:.2f} % (item 2: {'holds' if w.max() <= cap else 'fails'})")
    m = a.merge(b, on=["region", "year"], how="outer", suffixes=("_base", "_new"))
    m["d"] = m.value_nominal_new - m.value_nominal_base
    by = m.groupby("region").d.apply(lambda s: s.abs().max()).sort_values(ascending=False)
    print("   largest |Δ grade| by region (percentage points): "
          + "  ".join(f"{k} {100 * v:.2f}" for k, v in by.head(8).items()))
    for v, g in ((v_base, a), (v_new, b)):
        hi = g[g.rel_sigma > 1.0]
        print(f"   v{v}: {len(g)} grade rows, {len(hi)} with relative σ > 100 % (max "
              f"{100 * g.rel_sigma.max():.0f} %), by region "
              f"{hi.groupby('region').size().sort_values(ascending=False).head(6).to_dict()}")


def _ratio_rows(opt: str, acc) -> pd.DataFrame:
    """The ore-grade ratio rows of a run, with their z."""
    d = SolverDiag.from_opt_id(opt, acc=acc)
    cf = SolverStatsDiagnostic(d.results, d.constraint_run, acc).constraint_frame()
    r = cf[(cf["type"].astype(str) == "ratio")
           & cf["constraint_id"].astype(str).str.contains("_oregrade|^ratio_oregrade",
                                                          regex=True)]
    return pd.DataFrame({"constraint_id": r["constraint_id"].astype(str).to_numpy(),
                         "region": r["region"].astype(str).to_numpy(),
                         "z": r["z"].to_numpy()})


def retention(E: dict, acc, base: str, new: str) -> None:
    """R: CO Mass and Concentrate CC retention by mining region; ore-grade rows' Σz²."""
    print("\n== R retention by mining region, Σ post / Σ prior over 1995–2024")
    for reg in MINERS:
        out = []
        for o in (base, new):
            e = E[o]
            co = e[(e.sector_from == "CO") & (e.valuation == "Mass") & (e.region_to == reg)]
            mf = e[(e.sector_from == "MF") & (e.sector_to == "Concentrate")
                   & (e.valuation == "CC") & (e.region_to == reg)]
            out.append(f"CO Mass {co.post.sum() / co.prior.sum():.3f}, Concentrate CC "
                       f"{mf.post.sum() / mf.prior.sum():.3f}")
        print(f"   {reg:14s} base {out[0]}; new {out[1]}")
    z = {o: _ratio_rows(o, acc) for o in (base, new)}
    for o in (base, new):
        print(f"   {o}: ore-grade ratio rows {len(z[o])}, Σz² {(z[o].z ** 2).sum():,.0f}, "
              f"|z| > 3: {int((z[o].z.abs() > 3).sum())}")


def main(base: str, new: str, v_base: int, v_new: int) -> None:
    acc = AccountingStructure.load(id=load_config().pipeline.acc_id,
                                   storage_dir=str(ROOT / "acc"))
    E, S, j = P69.load(base, new)
    inputs(base, new, v_base, v_new)
    retention(E, acc, base, new)
    P69.round_e(E, S, j, base, new)
    P79.stock_z(E, base, new)
    P92.stock_sizes(E, base, new)
    P69.common_set(j, base)
    P86.by_year(j)
    by = j.groupby(P92._sector(j.label)).dz2.sum().sort_values()
    print("   common-set Δz² by sector: " + "  ".join(f"{k} {v:+,.0f}" for k, v in by.items()))
    P75.one_run_only(S, base, new)
    P69.attribution(j)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
