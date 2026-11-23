"""ore_grade.py
=============
The in-situ ore-grade triangle of the CO/Ore edges (ADR 0015, 0023, 0028), as library
functions with the case's labels as parameters (#147).

The crude-ore feed carries S&P *price-allocated* ore mass; the model's CO/Ore CC edges
are *in-situ* mill-feed copper (production ÷ recovery). The in-situ grade that makes the
triangle {ore mass} × {grade rows} ≡ {backward CC priors} hold identically is

    g_insitu(r,t) = [ ConcSupply_CC(r,t)/eff_MF + SXEWSupply_CC(r,t)/eff_SXEW ] / OreMass(r,t)

:func:`compute_insitu_grades` computes it, :func:`carry_grades_forward` carries the last
observed grade into later production years (ADR 0023), :func:`compute_tailings_grades`
derives the tailings grades from the prior's own construction (ADR 0015 amendment), and
:func:`build_param_rows` / :func:`build_tailings_rows` expand them into params rows.
The case's sector labels travel in :class:`OreGradeLabels`; the registered inputs
(eff, conc nominals, the S&P factor file) are read by ``scripts/copper_feed.py`` and
``scripts/build_ore_grade.py``, which is argument parsing and calls.

Public API
----------
OreGradeLabels                       The case's sector labels
compute_insitu_grades, region_bands  The observed grades and their bands
carry_grades_forward                 ADR 0023
compute_tailings_grades, tailings_bands
build_param_rows, build_tailings_rows
build_production_frame               Per (region, time) production and ore mass
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Knobs: a change is a measured change (ADR 0015, 0023)
# ---------------------------------------------------------------------------

# Relative margin applied to a region's in-situ series min/max, which sets both
# the hard band and (as envelope width / 4) the σ now written to `value_sd`.
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


@dataclass(frozen=True)
class OreGradeLabels:
    """The case's sector labels of the ore-grade triangle.

    roles           ``(role, selectors, description)`` of the conc-family rows, one per
                    role. ``selectors`` fills the a-side sector/entity cells and names the
                    cell the row's region goes in under ``"region_col"``; the b-side is
                    ``same`` except its valuation (the conc row convention).
    route_sector    ``{"mf": sector, "sxew": sector}``: the process each tailings route
                    leaves.
    process_entity, tail_sector, tail_entity
                    The route's entity kind and the tailings boundary node.
    conc_edge, sxew_edge, ore_edge
                    ``(sector_from, sector_to)`` of the concentrate and SX-EW production
                    edges (content valuation) and of the ore-mass edge (gross valuation).
    """
    roles:          tuple
    route_sector:   dict
    process_entity: str
    tail_sector:    str
    tail_entity:    str
    conc_edge:      tuple
    sxew_edge:      tuple
    ore_edge:       tuple


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
    """Per-region hard band = series min/max ∓/± relative margin.

    Also the σ source: `(g_max − g_min)/4` is written to `value_sd` (ADR 0028).
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


def build_tailings_rows(tails: pd.DataFrame, bands: pd.DataFrame, labels: OreGradeLabels,
                        gross: str, content: str) -> pd.DataFrame:
    """Expand (region, time, route, g) into conctailingsuse-shaped rows."""
    route_sector, tail_sector = labels.route_sector, labels.tail_sector
    merged = tails.merge(bands, on=["region", "route"])
    rows = []
    for _, r in merged.iterrows():
        slug = "".join(ch for ch in str(r["region"]).lower() if ch.isalnum())
        row = {c: "" for c in TEMPLATE_COLS}
        row.update({
            "param_id": f"oregradetailingsuse{r['route']}{slug}{r['time']}",
            "param_type": "concentration",
            "description": (f"Cu concentration in {tail_sector} ({labels.tail_entity}) — "
                            f"use edges (from-node={route_sector[r['route']]}, "
                            f"to-node={tail_sector}/{labels.tail_entity})"),
            "a_sector_from": route_sector[r["route"]], "a_entity_from": labels.process_entity,
            "a_region_from": r["region"],
            "a_sector_to": tail_sector, "a_entity_to": labels.tail_entity,
            "a_region_to": "*",
            "a_time": str(r["time"]), "a_valuation": gross,
            "b_sector_from": "same", "b_entity_from": "same",
            "b_region_from": "same", "b_sector_to": "same",
            "b_entity_to": "same", "b_region_to": "same",
            "b_time": "same", "b_valuation": content,
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


def build_param_rows(grades: pd.DataFrame, bands: pd.DataFrame, labels: OreGradeLabels,
                     gross: str, content: str) -> pd.DataFrame:
    """Expand (region, time, g) into the three conc-family roles.

    ADR 0028 states σ explicitly (`unc_type="sd"`) instead of leaving it to be
    derived from the band, so that the band's two jobs — a hard plausibility
    limit and a measurement uncertainty — can be changed independently. The
    values are unchanged: `value_sd` is the quarter-width of the same series
    envelope the retired `min_max` mode read, so the soft ratio row, the
    parametric fill's prior and the `NetworkImputer` twin map are identical.

    The band itself stays the region's series envelope (ADR 0015). A year-local
    band was built and measured in Round M and **rejected** — it removed Chile's
    impossible 1.66 % posterior ore grade but merely relocated the deletion
    (US ore 0.990 → 1.356); see ADR 0028 and its REP.
    """
    merged = grades.merge(bands, on="region")
    rows = []
    for _, r in merged.iterrows():
        slug = "".join(ch for ch in str(r["region"]).lower() if ch.isalnum())
        for role, sel, desc in labels.roles:
            row = {c: "" for c in TEMPLATE_COLS}
            row.update({
                "param_id": f"oregrade{role}{slug}{r['time']}",
                "param_type": "concentration",
                "description": desc,
                "a_sector_from": "*", "a_entity_from": "*", "a_region_from": "*",
                "a_sector_to": "*", "a_entity_to": "*", "a_region_to": "*",
                "a_time": str(r["time"]), "a_valuation": gross,
                "b_sector_from": "same", "b_entity_from": "same",
                "b_region_from": "same", "b_sector_to": "same",
                "b_entity_to": "same", "b_region_to": "same",
                "b_time": "same", "b_valuation": content,
                "value_nominal": round(float(r["g"]), 8),
                "unc_type": "sd",
                # Rounded ENDS, unrounded quotient: bit-for-bit the σ the
                # retired `min_max` branch derived from the band (ADR 0028).
                "value_sd": (round(float(r["g_max"]), 8)
                             - round(float(r["g_min"]), 8)) / 4.0,
                "value_min": round(float(r["g_min"]), 8),
                "value_max": round(float(r["g_max"]), 8),
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
# Production frame
# ---------------------------------------------------------------------------

def _decoded_edges(acc) -> pd.DataFrame:
    ei = acc.edge_index.copy()
    nidx = acc.node_index.reset_index(drop=True)
    ei["sector_from"] = nidx["sector"].values[ei["from_id"].values]
    ei["sector_to"]   = nidx["sector"].values[ei["to_id"].values]
    ei["region_from"] = nidx["region"].values[ei["from_id"].values]
    return ei


def build_production_frame(acc, icsg_froot, crude_froot,
                           labels: OreGradeLabels) -> pd.DataFrame:
    """Per (region, time): conc_cc, sxew_cc, ore_mass from feed-run F_roots."""
    ei = _decoded_edges(acc)

    def _sum(sector_from, sector_to, valuation, froot):
        # NaN (an edge the feed does not cover) reads 0 here on purpose: this frame asks
        # "how much production / ore mass", and a region-year without either has none to
        # grade. The crude-ore feed carries no zeros, so 0 is never an observation (U1b-S5-2).
        sub = ei[(ei["sector_from"] == sector_from) & (ei["sector_to"] == sector_to)
                 & (ei["valuation"] == valuation)]
        vals = froot[sub["edge_id"].values]
        out = sub.assign(v=np.nan_to_num(vals, nan=0.0)) \
                 .groupby(["region_from", "time"])["v"].sum()
        return out

    conc = _sum(*labels.conc_edge, acc.valuation_roles.content, icsg_froot).rename("conc_cc")
    sxew = _sum(*labels.sxew_edge, acc.valuation_roles.content, icsg_froot).rename("sxew_cc")
    ore  = _sum(*labels.ore_edge, acc.valuation_roles.gross, crude_froot).rename("ore_mass")
    df = pd.concat([conc, sxew, ore], axis=1).reset_index()
    df = df.rename(columns={"region_from": "region"})
    df["time"] = df["time"].astype(str)
    return df

