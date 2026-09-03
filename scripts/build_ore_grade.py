"""
build_ore_grade.py
==================
Build region×year **in-situ** ore-grade concentration params for the CO/Ore
edges and emit them as the ``ore_grade`` params artifact (ADR 0015). These
rows replace the conc family's global CO/Ore placeholders (nominal 1 %, band
[0.3, 1.7] %) — `copper_params_parse.convert_conc` skips CO/Ore accordingly.

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
member countries, as in the superseded scripts/parse/ore_grade_parse.py).

Output rows mirror the conc family's three roles, region/time-resolved:
    oregradecosupply{region}{year}   a: CO/SB from      (CO→Mining)
    oregradeoreuse{region}{year}     a: to Ore/Flow     (Mining→Ore)
    oregradeoresupply{region}{year}  a: Ore/Flow from   (Ore→MF, Ore→SXEW)
value_min/value_max = the region's in-situ series min/max ∓/± MARGIN_REL —
consumed as the hard band by RatioBoundBuilder and as σ (min_max) by the soft
RatioConstraintBuilder and the imputer twin map.

Output: data/input/params/ore_grade/current/params.csv (registry-versioned;
the raw S&P factor file and ore_grade_map.xlsx stay alongside). NOTE data/ is
gitignored — regenerate on a fresh checkout.
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from pysuto.acc import AccountingStructure
from pysuto.config import load_config

# ---------------------------------------------------------------------------
# USER CONFIGURATION
# ---------------------------------------------------------------------------

_INPUT_ROOT = os.path.join("data", "input")
ACC_DIR     = os.path.join("data", "proc", "acc")
FEED_REG    = os.path.join("data", "proc", "feed", "registry")
OG_DIR      = os.path.join(_INPUT_ROOT, "params", "ore_grade", "current")
OUT_CSV     = os.path.join(OG_DIR, "params.csv")

# Relative margin applied to a region's in-situ series min/max for the hard band.
MARGIN_REL = 0.10

TEMPLATE_COLS = [
    "param_id", "param_type", "description",
    "a_sector_from", "a_entity_from", "a_region_from",
    "a_sector_to", "a_entity_to", "a_region_to", "a_time", "a_valuation",
    "b_sector_from", "b_entity_from", "b_region_from",
    "b_sector_to", "b_entity_to", "b_region_to", "b_time", "b_valuation",
    "value_nominal", "unc_type", "value_sd", "value_min", "value_max",
    "pedigree_reliability", "pedigree_completeness",
    "pedigree_temporal_correlation", "pedigree_geographical_correlation",
    "pedigree_other_correlation", "epsilon", "bound_lower", "bound_upper",
    "source", "comment",
]

# The three conc-family roles, region/time-resolved. Each dict fills the
# a-side selectors; the b-side is "same" except b_valuation=CC (a=Mass edge,
# b=its CC twin — the concentration row convention).
ROLES = [
    ("cosupply", {"a_sector_from": "CO", "a_entity_from": "SB",
                  "region_col": "a_region_from"},
     "Cu in-situ grade of CO (SB) — supply edges (from-node=CO/SB)"),
    ("oreuse", {"a_sector_to": "Ore", "a_entity_to": "Flow",
                "region_col": "a_region_to"},
     "Cu in-situ grade of Ore (Flow) — use edges (to-node=Ore/Flow)"),
    ("oresupply", {"a_sector_from": "Ore", "a_entity_from": "Flow",
                   "region_col": "a_region_from"},
     "Cu in-situ grade of Ore (Flow) — supply edges (from-node=Ore/Flow)"),
]


# ---------------------------------------------------------------------------
# Pure computation (unit-tested)
# ---------------------------------------------------------------------------

def compute_insitu_grades(
    prod: pd.DataFrame,
    fallback: pd.DataFrame,
    eff_mf: float,
    eff_sxew: float,
) -> pd.DataFrame:
    """Compute g_insitu per (region, time).

    ``prod`` columns: region, time, conc_cc (MF→Concentrate CC),
    sxew_cc (SXEW supply CC), ore_mass (CO→Mining Mass). ``fallback``
    columns: region, time, factor (S&P t-ore per t-metal). Returns rows
    (region, time, g, src) for region-years with ore_mass > 0 and a grade
    source; src ∈ {"icsg", "factor"}.
    """
    df = prod.copy()
    df["insitu_cc"] = df["conc_cc"].fillna(0.0) / eff_mf \
        + df["sxew_cc"].fillna(0.0) / eff_sxew
    df = df[(df["ore_mass"] > 0)]
    icsg = df[df["insitu_cc"] > 0].copy()
    icsg["g"] = icsg["insitu_cc"] / icsg["ore_mass"]
    icsg["src"] = "icsg"

    # Fallback: allocated recovered grade (1/factor) converted to in-situ via
    # the flotation recovery (SXEW split unknown at fallback level).
    fb = df[df["insitu_cc"] <= 0].merge(fallback, on=["region", "time"], how="inner")
    fb = fb[fb["factor"] > 0].copy()
    fb["g"] = (1.0 / fb["factor"]) / eff_mf
    fb["src"] = "factor"

    out = pd.concat([icsg[["region", "time", "g", "src"]],
                     fb[["region", "time", "g", "src"]]], ignore_index=True)
    return out[out["g"] > 0].reset_index(drop=True)


def region_bands(grades: pd.DataFrame, margin_rel: float = MARGIN_REL) -> pd.DataFrame:
    """Per-region hard band = series min/max ∓/± relative margin."""
    g = grades.groupby("region")["g"]
    return pd.DataFrame({
        "g_min": g.min() * (1.0 - margin_rel),
        "g_max": g.max() * (1.0 + margin_rel),
    }).reset_index()


def build_param_rows(grades: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """Expand (region, time, g) into the three conc-family roles."""
    merged = grades.merge(bands, on="region")
    rows = []
    for _, r in merged.iterrows():
        slug = "".join(ch for ch in str(r["region"]).lower() if ch.isalnum())
        for role, sel, desc in ROLES:
            row = {c: "" for c in TEMPLATE_COLS}
            row.update({
                "param_id": f"oregrade{role}{slug}{r['time']}",
                "param_type": "concentration",
                "description": desc,
                "a_sector_from": "*", "a_entity_from": "*", "a_region_from": "*",
                "a_sector_to": "*", "a_entity_to": "*", "a_region_to": "*",
                "a_time": str(r["time"]), "a_valuation": "Mass",
                "b_sector_from": "same", "b_entity_from": "same",
                "b_region_from": "same", "b_sector_to": "same",
                "b_entity_to": "same", "b_region_to": "same",
                "b_time": "same", "b_valuation": "CC",
                "value_nominal": round(float(r["g"]), 8),
                "unc_type": "min_max",
                "value_min": round(float(r["g_min"]), 8),
                "value_max": round(float(r["g_max"]), 8),
                "bound_lower": 0, "bound_upper": 1,
                "source": "ICSG productions / crude_ore allocated mass (ADR 0015)"
                          if r["src"] == "icsg" else
                          "S&P factor (1/factor)/eff_MF fallback (ADR 0015)",
                "comment": f"in-situ basis; src={r['src']}",
            })
            for k, v in sel.items():
                if k != "region_col":
                    row[k] = v
            row[sel["region_col"]] = r["region"]
            rows.append(row)
    return pd.DataFrame(rows, columns=TEMPLATE_COLS)


# ---------------------------------------------------------------------------
# I/O helpers (build_trade_shares pattern)
# ---------------------------------------------------------------------------

def _latest_feed_froot(label: str) -> np.ndarray:
    dirs = sorted(glob.glob(os.path.join(FEED_REG, f"feed_{label}_*")))
    if not dirs:
        raise FileNotFoundError(f"no feed_{label}_* run under {FEED_REG}")
    run = dirs[-1]
    print(f"   {label:10s} ← {os.path.basename(run)}")
    return np.load(os.path.join(run, "result.npz"))["F_root"]


def _eff_nominals() -> tuple[float, float]:
    from pysuto.input_registry import InputRegistries
    inputs = InputRegistries.initialize(root=_INPUT_ROOT)
    csv = inputs.params.resolve("eff") / "params.csv"
    p = pd.read_csv(csv, sep=None, engine="python", comment="#", dtype=str)
    eff = {row["param_id"]: float(row["value_nominal"]) for _, row in p.iterrows()
           if row.get("param_type") == "efficiency"}
    return eff["effmfflotation"], eff["effsxew"]


def _decoded_edges(acc: AccountingStructure) -> pd.DataFrame:
    ei = acc.edge_index.copy()
    nidx = acc.node_index.reset_index(drop=True)
    ei["sector_from"] = nidx["sector"].values[ei["from_id"].values]
    ei["sector_to"]   = nidx["sector"].values[ei["to_id"].values]
    ei["region_from"] = nidx["region"].values[ei["from_id"].values]
    return ei


def build_production_frame(acc, icsg_froot, crude_froot) -> pd.DataFrame:
    """Per (region, time): conc_cc, sxew_cc, ore_mass from feed-run F_roots."""
    ei = _decoded_edges(acc)

    def _sum(sector_from, sector_to, valuation, froot):
        sub = ei[(ei["sector_from"] == sector_from) & (ei["sector_to"] == sector_to)
                 & (ei["valuation"] == valuation)]
        vals = froot[sub["edge_id"].values]
        out = sub.assign(v=np.nan_to_num(vals, nan=0.0)) \
                 .groupby(["region_from", "time"])["v"].sum()
        return out

    conc = _sum("MF", "Concentrate", "CC", icsg_froot).rename("conc_cc")
    sxew = _sum("SXEW", "CSXEW", "CC", icsg_froot).rename("sxew_cc")
    ore  = _sum("CO", "Mining", "Mass", crude_froot).rename("ore_mass")
    df = pd.concat([conc, sxew, ore], axis=1).reset_index()
    df = df.rename(columns={"region_from": "region"})
    df["time"] = df["time"].astype(str)
    return df


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

    acc_id = load_config().pipeline.acc_id
    acc = AccountingStructure.load(id=acc_id, storage_dir=ACC_DIR)
    eff_mf, eff_sxew = _eff_nominals()
    print(f"   eff_MF={eff_mf}  eff_SXEW={eff_sxew}  (from eff params)")
    icsg  = _latest_feed_froot("icsg")
    crude = _latest_feed_froot("crude_ore")

    prod = build_production_frame(acc, icsg, crude)
    grades = compute_insitu_grades(prod, load_factor_fallback(), eff_mf, eff_sxew)
    bands = region_bands(grades)
    params = build_param_rows(grades, bands)

    n_icsg = (grades["src"] == "icsg").sum()
    n_fb   = (grades["src"] == "factor").sum()
    n_gap  = int((prod["ore_mass"] > 0).sum()) - len(grades)
    print(f"   region-years: {n_icsg} icsg-derived + {n_fb} factor-fallback "
          f"({n_gap} with ore mass but no grade source)")
    for r in ("Chile", "Australia"):
        s = grades[grades["region"] == r]
        if len(s):
            print(f"   {r:10s} g range {s.g.min()*100:.3f}–{s.g.max()*100:.3f} %  "
                  f"(2015: {float(s[s.time=='2015'].g.iloc[0])*100:.3f} %)" if (s.time == '2015').any() else "")

    os.makedirs(OG_DIR, exist_ok=True)
    params.to_csv(OUT_CSV, index=False)
    print(f"   wrote {len(params):,} param rows → {OUT_CSV}")

    from pysuto.input_registry import InputRegistries
    inputs = InputRegistries.initialize(root=_INPUT_ROOT)
    entry = inputs.params.entry("ore_grade")
    print(f"   registered ore_grade v{entry['version']} ({entry['code']})")


if __name__ == "__main__":
    main()
