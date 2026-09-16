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

# Relative margin. ADR 0028: applied to *that year's own* grade for the hard
# band (`value_min`/`value_max`), and to the region's in-situ series min/max for
# the σ the band used to imply (`value_sd` = envelope width / 4).
MARGIN_REL = 0.10
# Wider margin for carried-forward grades (ADR 0023): the grade is assumed, not
# measured, in those years.
MARGIN_REL_CARRY = 0.20

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
    """Per-region series envelope = series min/max ∓/± relative margin.

    ADR 0028: this is the **σ source**, not the hard band. `(g_max − g_min)/4`
    is written to `value_sd`, which reproduces exactly the σ the `min_max`
    unc_type used to derive from these two numbers — so the soft ratio row and
    the Stage-2 twin map are unchanged. The hard band is year-local and is
    computed per row in `build_param_rows`.
    """
    g = grades.groupby("region")["g"]
    return pd.DataFrame({
        "g_min": g.min() * (1.0 - margin_rel),
        "g_max": g.max() * (1.0 + margin_rel),
    }).reset_index()


def carry_grades_forward(grades: pd.DataFrame, prod: pd.DataFrame) -> pd.DataFrame:
    """Carry each region's last observed grade into later production years.

    A (region, time) qualifies when ``prod`` shows ICSG production
    (conc_cc + sxew_cc > 0) and ``time`` lies after the region's last grade in
    ``grades``. Forward only; regions without any observed grade, and years
    without production, are left uncovered. Returns rows
    (region, time, g, src="carry", from_time).
    """
    cols = ["region", "time", "g", "src", "from_time"]
    has_prod = prod[(prod["conc_cc"].fillna(0.0) + prod["sxew_cc"].fillna(0.0)) > 0]
    rows = []
    for region, s in grades.groupby("region"):
        last = s.sort_values("time").iloc[-1]
        later = has_prod[(has_prod["region"] == region) & (has_prod["time"] > last["time"])]
        for t in sorted(later["time"]):
            rows.append((region, t, float(last["g"]), "carry", last["time"]))
    return pd.DataFrame(rows, columns=cols)


def compute_tailings_grades(
    prod: pd.DataFrame,
    grades: pd.DataFrame,
    eff_mf: float,
    eff_sxew: float,
    conc_concentrate: float,
    conc_csxew: float,
) -> pd.DataFrame:
    """g_tail per (region, time, route) from the prior's own construction.

    Per route the tailings CC is the recipe residual ``ProdCC·(1−eff)/eff``
    and the tailings Mass is the process Mass-balance residual under the
    prior twins, ``(ProdCC/eff)/g_insitu − ProdCC/conc_product`` — so the
    resulting g_tail makes the tailings Mass twin close the hard MF/SXEW
    Mass balance at prior level. ICSG-derived and carried-forward grades
    qualify (the factor fallback has no route split). Returns
    (region, time, route ∈ {"mf", "sxew"}, g, src).
    """
    g_ok = grades[grades["src"].isin(("icsg", "carry"))][["region", "time", "g", "src"]]
    df = prod.merge(g_ok, on=["region", "time"], how="inner")
    routes = []
    for route, cc_col, eff, conc_prod in (
        ("mf", "conc_cc", eff_mf, conc_concentrate),
        ("sxew", "sxew_cc", eff_sxew, conc_csxew),
    ):
        r = df[df[cc_col].fillna(0.0) > 0].copy()
        tail_cc = r[cc_col] * (1.0 - eff) / eff
        tail_mass = (r[cc_col] / eff) / r["g"] - r[cc_col] / conc_prod
        r["g"] = tail_cc / tail_mass
        r["route"] = route
        routes.append(r[tail_mass > 0][["region", "time", "route", "g", "src"]])
    out = pd.concat(routes, ignore_index=True)
    return out[out["g"] > 0].reset_index(drop=True)


def tailings_bands(tails: pd.DataFrame, margin_rel: float = MARGIN_REL) -> pd.DataFrame:
    """Per (region, route) hard band = series min/max ∓/± relative margin."""
    g = tails.groupby(["region", "route"])["g"]
    return pd.DataFrame({
        "g_min": g.min() * (1.0 - margin_rel),
        "g_max": g.max() * (1.0 + margin_rel),
    }).reset_index()


def build_tailings_rows(tails: pd.DataFrame, bands: pd.DataFrame) -> pd.DataFrame:
    """Expand (region, time, route, g) into conctailingsuse-shaped rows."""
    route_sector = {"mf": "MF", "sxew": "SXEW"}
    merged = tails.merge(bands, on=["region", "route"])
    rows = []
    for _, r in merged.iterrows():
        slug = "".join(ch for ch in str(r["region"]).lower() if ch.isalnum())
        row = {c: "" for c in TEMPLATE_COLS}
        row.update({
            "param_id": f"oregradetailingsuse{r['route']}{slug}{r['time']}",
            "param_type": "concentration",
            "description": ("Cu concentration in Tailings (SB) — use edges "
                            f"(from-node={route_sector[r['route']]}, to-node=Tailings/SB)"),
            "a_sector_from": route_sector[r["route"]], "a_entity_from": "Process",
            "a_region_from": r["region"],
            "a_sector_to": "Tailings", "a_entity_to": "SB", "a_region_to": "*",
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
            "source": "recipe residual / prior Mass balance (ADR 0015 amendment)",
            "comment": f"allocated basis; route={r['route']}; src={r.get('src', 'icsg')}",
        })
        rows.append(row)
    return pd.DataFrame(rows, columns=TEMPLATE_COLS)


_SOURCE = {
    "icsg":   "ICSG productions / crude_ore allocated mass (ADR 0015)",
    "factor": "S&P factor (1/factor)/eff_MF fallback (ADR 0015)",
    "carry":  "last observed in-situ grade carried forward (ADR 0023)",
}


def build_param_rows(grades: pd.DataFrame, bands: pd.DataFrame,
                     margin_rel: float = MARGIN_REL) -> pd.DataFrame:
    """Expand (region, time, g) into the three conc-family roles.

    ADR 0028 splits the two jobs `value_min`/`value_max` used to do at once:

    * the **hard band** (`RatioBoundBuilder`) brackets *that year's own* grade,
      `g·(1 ∓ margin_rel)`. Applying the margin to the region's 30-year series
      envelope instead let a declining-grade region price this year's ore at
      the grade of decades earlier — Chile's ceiling was 1.1× its 1995 grade
      and bound in 26 of 30 years;
    * the **σ** (`RatioConstraintBuilder`, `ParametricImputer`, the
      `NetworkImputer` twin map) stays the series-envelope quarter-width via an
      explicit `value_sd`, so this change moves the cone and nothing else.
    """
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
                "unc_type": "sd",
                # Rounded ENDS, unrounded quotient: this is bit-for-bit the σ
                # the retired `min_max` branch derived from the written band,
                # so Stage 2 cannot move (ADR 0028's attribution gate).
                "value_sd": (round(float(r["g_max"]), 8)
                             - round(float(r["g_min"]), 8)) / 4.0,
                "value_min": round(float(r["g"]) * (1.0 - margin_rel), 8),
                "value_max": round(float(r["g"]) * (1.0 + margin_rel), 8),
                "bound_lower": 0, "bound_upper": 1,
                "source": _SOURCE[r["src"]],
                "comment": (f"in-situ basis; src=carry; from={r['from_time']}"
                            if r["src"] == "carry" else f"in-situ basis; src={r['src']}"),
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


def _conc_nominals() -> tuple[float, float]:
    """Product conc nominals (Concentrate, CSXEW) from the conc params file."""
    from pysuto.input_registry import InputRegistries
    inputs = InputRegistries.initialize(root=_INPUT_ROOT)
    csv = inputs.params.resolve("conc") / "params.csv"
    p = pd.read_csv(csv, sep=None, engine="python", comment="#", dtype=str)
    conc = {row["param_id"]: float(row["value_nominal"]) for _, row in p.iterrows()
            if row.get("param_type") == "concentration"}
    return conc["concconcentratesupply"], conc["conccsxewsupply"]


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

    conc_concentrate, conc_csxew = _conc_nominals()
    print(f"   conc_Concentrate={conc_concentrate}  conc_CSXEW={conc_csxew}  (from conc params)")

    prod = build_production_frame(acc, icsg, crude)
    observed = compute_insitu_grades(prod, load_factor_fallback(), eff_mf, eff_sxew)
    carried = carry_grades_forward(observed, prod)
    grades = pd.concat([observed, carried], ignore_index=True)
    tails = compute_tailings_grades(prod, grades, eff_mf, eff_sxew,
                                    conc_concentrate, conc_csxew)
    is_carry_t = tails["src"] == "carry"
    params = pd.concat([
        build_param_rows(observed, region_bands(grades), MARGIN_REL),
        build_param_rows(carried, region_bands(grades, MARGIN_REL_CARRY),
                         MARGIN_REL_CARRY),
        build_tailings_rows(tails[~is_carry_t], tailings_bands(tails)),
        build_tailings_rows(tails[is_carry_t], tailings_bands(tails, MARGIN_REL_CARRY)),
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

    from pysuto.input_registry import InputRegistries
    inputs = InputRegistries.initialize(root=_INPUT_ROOT)
    entry = inputs.params.entry("ore_grade")
    print(f"   registered ore_grade v{entry['version']} ({entry['code']})")


if __name__ == "__main__":
    main()
