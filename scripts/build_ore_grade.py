"""
build_ore_grade.py
==================
Build region×year **in-situ** ore-grade concentration params for the CO/Ore
edges and emit them as the ``ore_grade`` params artifact (ADR 0015). These
rows replace the conc family's global CO/Ore placeholders (nominal 1 %, band
[0.3, 1.7] %); the hand-authored conc table carries no CO/Ore row.

Basis (ADR 0015)
----------------
The crude_ore feed carries S&P *price-allocated* ore mass, and the model's
CO/Ore CC edges are *in-situ* mill-feed copper (production ÷ recovery, filled
backward by the NetworkImputer). The in-situ grade that makes the triangle
{OreMass feed} × {grade rows} ≡ {backward CC priors} hold identically is

    g_insitu(r,t) = [ ConcSupply_CC(r,t)/eff_MF + SXEWSupply_CC(r,t)/eff_SXEW ]
                    / OreMass(r,t)

with productions and ore mass read from the registered **icsg** and
**crude_ore** feed runs' F_root (root-aligned, the build_trade_shares
pattern) and the eff nominals read from the eff params file (never
hard-coded). Fallback where ICSG production is absent: ``(1/factor)/eff_MF``
from the S&P factor file via ore_grade_map.xlsx (ROW = arithmetic mean of its
member countries).

Output rows mirror the conc family's three roles, region/time-resolved:
    oregradecosupply{region}{year}   a: CO/SB from      (CO→Mining)
    oregradeoreuse{region}{year}     a: to Ore/Flow     (Mining→Ore)
    oregradeoresupply{region}{year}  a: Ore/Flow from   (Ore→MF, Ore→SXEW)
value_min/value_max = the region's in-situ series min/max ∓/± MARGIN_REL —
consumed as the hard band by RatioBoundBuilder and as σ (min_max) by the soft
RatioConstraintBuilder and the imputer twin map.

Tailings extension (ADR 0015 amendment)
---------------------------------------
The same artifact also owns the region×year **tailings** concentration, per
route (replacing the conc family's global ``conctailingsuse``, nominal 0.55 %):
    oregradetailingsusemf{region}{year}    a: MF→Tailings/SB
    oregradetailingsusesxew{region}{year}  a: SXEW→Tailings/SB
built from the imputer's own prior construction so the tailings Mass twin
closes the hard process Mass balance at prior (the WR≈0 trick):

    TailCC_route   = ProdCC · (1 − eff_route)/eff_route
    TailMass_route = (ProdCC/eff_route)/g_insitu − ProdCC/conc_product
    g_tail(route)  = TailCC_route / TailMass_route

with the product conc nominals (Concentrate 0.275, CSXEW 0.9975) read from the
conc params file. Regions without an ICSG production source get no tailings
row (honest flat-prior gap, as for CO/Ore).

Temporal carry-forward (ADR 0023)
--------------------------------
The crude_ore feed ends in 2022; ICSG mine production continues. A region-year
with production but no ore-mass source gets its **last observed in-situ grade
carried forward** (src="carry"), on a band widened to MARGIN_REL_CARRY, so the
NetworkImputer derives the whole Mass chain (CO→Mining, Mining→Ore, Ore→MF/
SXEW, Tailings) from that year's ICSG production through the grade — ore mass
follows production, the grade is what is assumed constant. Tailings rows for
carried years are computed from the carried grade and the year's own
production, not carried themselves. Forward only, from the last observed year.

Output: data/input/params/ore_grade/current/params.csv (registry-versioned;
the raw S&P factor file and ore_grade_map.xlsx stay alongside) and parents.json beside
it, which names the ICSG and crude-ore feed runs it was built from and the eff and conc
versions (the ICSG feed reads these params, and the ore mass reads the runs: the order
loop of README *Workflow*). NOTE data/ is
gitignored — regenerate on a fresh checkout.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysuto import acc_store
from copper_feed import ORE_GRADE_LABELS, latest_feed_run
from pysuto.config import load_config
from pysuto.feedprep.builder_lock import builder_lock
from pysuto.feedprep.ore_grade import (
    MARGIN_REL_CARRY, build_param_rows, build_production_frame, build_tailings_rows,
    carry_grades_forward, compute_insitu_grades, compute_tailings_grades, region_bands,
    tailings_bands)
from pysuto.input_registry import PARAMS_PARENTS, InputRegistries
from pysuto.relation.params import read_params

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------
# The margins (MARGIN_REL, MARGIN_REL_CARRY) are the library's, ``pysuto.feedprep.
# ore_grade``; the sector labels the case's, ``copper_feed``.

_INPUT_ROOT = os.path.join("data", "input")
ACC_DIR     = os.path.join("data", "proc", "acc")
FEED_REG    = os.path.join("data", "proc", "feed", "registry")
OG_DIR      = os.path.join(_INPUT_ROOT, "params", "ore_grade", "current")
OUT_CSV     = os.path.join(OG_DIR, "params.csv")


def set_data_dir(data_dir: str) -> None:
    """Point the script at ``data_dir`` instead of ``data`` (as the other builders do)."""
    global _INPUT_ROOT, ACC_DIR, FEED_REG, OG_DIR, OUT_CSV
    _INPUT_ROOT = os.path.join(data_dir, "input")
    ACC_DIR     = os.path.join(data_dir, "proc", "acc")
    FEED_REG    = os.path.join(data_dir, "proc", "feed", "registry")
    OG_DIR      = os.path.join(_INPUT_ROOT, "params", "ore_grade", "current")
    OUT_CSV     = os.path.join(OG_DIR, "params.csv")


# ---------------------------------------------------------------------------
# I/O helpers (build_trade_shares pattern)
# ---------------------------------------------------------------------------

def _latest_feed_froot(label: str) -> tuple[str, np.ndarray]:
    """``(run id, F_root)`` of the newest ``feed_<label>_*`` run."""
    run = latest_feed_run(FEED_REG, label)
    print(f"   {label:10s} ← {os.path.basename(run)}")
    return os.path.basename(run), np.load(os.path.join(run, "result.npz"))["F_root"]


def write_parents(out_dir: str, feed_run_ids: list) -> dict:
    """Record the params' parents beside their params.csv, where the params registry hashes
    it: the ICSG and crude-ore feed runs they were built from and the eff and conc versions
    in force, in a sidecar's vocabulary (the ``trade_shares`` record's form, #53)."""
    reg = InputRegistries.load(root=_INPUT_ROOT).params
    built_from = {"upstream": {"feed_run_ids": list(feed_run_ids)},
                  "inputs": {"params": {i: reg.entry(i)["version"] for i in ("conc", "eff")}}}
    with open(os.path.join(out_dir, PARAMS_PARENTS), "w", encoding="utf-8") as f:
        json.dump(built_from, f, indent=2, sort_keys=True)
    return built_from


def _eff_nominals() -> tuple[float, float]:
    inputs = InputRegistries.load(root=_INPUT_ROOT)       # read-only (#93)
    csv = inputs.params.resolve("eff") / "params.csv"
    eff = {row.param_id: float(row.value_nominal) for row in read_params(csv, ("efficiency",))}
    return eff["effmfflotation"], eff["effsxew"]


def _conc_nominals() -> tuple[float, float]:
    """Product conc nominals (Concentrate, CSXEW) from the conc params file."""
    inputs = InputRegistries.load(root=_INPUT_ROOT)       # read-only (#93)
    csv = inputs.params.resolve("conc") / "params.csv"
    conc = {row.param_id: float(row.value_nominal) for row in read_params(csv, ("concentration",))}
    return conc["concconcentratesupply"], conc["conccsxewsupply"]


def load_factor_fallback() -> pd.DataFrame:
    """S&P factor per (root region, year); ROW = arithmetic mean of members."""
    fac_files = sorted(glob.glob(os.path.join(OG_DIR, "all_integrated_est_fac_*.csv")))
    raw = pd.read_csv(fac_files[-1], sep=None, engine="python")
    raw = raw[raw["material_id"] == "Me.Cu"]
    amap = pd.ExcelFile(os.path.join(OG_DIR, "ore_grade_map.xlsx")).parse("alphanumiso_map")
    m = raw.merge(amap[["alphanumiso", "root_region"]], on="alphanumiso", how="inner")
    out = (m.groupby(["root_region", "year"])["factor"].mean()
             .reset_index()
             .rename(columns={"root_region": "region", "year": "time"}))
    out["time"] = out["time"].astype(str)
    return out


def main() -> None:
    print("=" * 60)
    print("Building in-situ ore-grade params (ADR 0015)")
    InputRegistries.initialize(root=_INPUT_ROOT)   # the step registers; lookups read (#93)

    acc_id = load_config().pipeline.acc_id
    acc = acc_store.load(id=acc_id, storage_dir=ACC_DIR)
    eff_mf, eff_sxew = _eff_nominals()
    print(f"   eff_MF={eff_mf}  eff_SXEW={eff_sxew}  (from eff params)")
    icsg_run,  icsg  = _latest_feed_froot("icsg")
    crude_run, crude = _latest_feed_froot("crude_ore")

    conc_concentrate, conc_csxew = _conc_nominals()
    print(f"   conc_Concentrate={conc_concentrate}  conc_CSXEW={conc_csxew}  (from conc params)")

    prod = build_production_frame(acc, icsg, crude, ORE_GRADE_LABELS)
    observed = compute_insitu_grades(prod, load_factor_fallback(), eff_mf, eff_sxew)
    carried = carry_grades_forward(observed, prod)
    grades = pd.concat([observed, carried], ignore_index=True)
    tails = compute_tailings_grades(prod, grades, eff_mf, eff_sxew,
                                    conc_concentrate, conc_csxew)
    is_carry_t = tails["src"] == "carry"
    labels, gross, content = ORE_GRADE_LABELS, acc.valuation_roles.gross, acc.valuation_roles.content
    params = pd.concat([
        build_param_rows(observed, region_bands(grades), labels, gross, content),
        build_param_rows(carried, region_bands(grades, MARGIN_REL_CARRY), labels, gross,
                         content),
        build_tailings_rows(tails[~is_carry_t], tailings_bands(tails), labels, gross,
                            content),
        build_tailings_rows(tails[is_carry_t], tailings_bands(tails, MARGIN_REL_CARRY),
                            labels, gross, content),
    ], ignore_index=True)

    n_icsg = (observed["src"] == "icsg").sum()
    n_fb   = (observed["src"] == "factor").sum()
    n_gap  = int((prod["ore_mass"] > 0).sum()) - len(observed)
    print(f"   region-years: {n_icsg} icsg-derived + {n_fb} factor-fallback "
          f"({n_gap} with ore mass but no grade source)")
    if len(carried):
        span = carried["time"].astype(int) - carried["from_time"].astype(int)
        print(f"   carried forward: {len(carried)} region-years into "
              f"{sorted(carried['time'].unique())} (from {sorted(carried['from_time'].unique())}; "
              f"max span {int(span.max())} y; {int((tails['src'] == 'carry').sum())} tailings rows)")
    print(f"   tailings rows: {len(tails)} region-year-routes "
          f"({(tails['route'] == 'mf').sum()} mf + {(tails['route'] == 'sxew').sum()} sxew)")
    for r in ("Chile", "Australia"):
        s = grades[grades["region"] == r]
        if len(s):
            print(f"   {r:10s} g range {s.g.min()*100:.3f}–{s.g.max()*100:.3f} %  "
                  f"(2015: {float(s[s.time=='2015'].g.iloc[0])*100:.3f} %)" if (s.time == '2015').any() else "")

    os.makedirs(OG_DIR, exist_ok=True)
    params.to_csv(OUT_CSV, index=False)
    print(f"   wrote {len(params):,} param rows → {OUT_CSV}")
    built_from = write_parents(OG_DIR, [icsg_run, crude_run])
    print(f"   built from : {built_from['upstream']['feed_run_ids']}, {built_from['inputs']}")

    inputs = InputRegistries.initialize(root=_INPUT_ROOT)
    entry = inputs.params.entry("ore_grade")
    print(f"   registered ore_grade v{entry['version']} ({entry['code']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default="data",
                    help="data root holding input/ and proc/ (default: %(default)s)")
    args = ap.parse_args()
    set_data_dir(args.data_dir)
    with builder_lock("build_ore_grade", args.data_dir):
        main()
