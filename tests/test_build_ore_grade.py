"""Unit tests for the in-situ ore-grade params generator (ADR 0015)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from build_ore_grade import (  # noqa: E402
    build_param_rows,
    build_tailings_rows,
    compute_insitu_grades,
    compute_tailings_grades,
    region_bands,
    tailings_bands,
)

EFF_MF, EFF_SXEW = 0.8, 0.725


def _prod(rows):
    return pd.DataFrame(rows, columns=["region", "time", "conc_cc", "sxew_cc", "ore_mass"])


def test_insitu_identity():
    # g * ore_mass == conc/eff_MF + sxew/eff_SXEW  (the ADR 0015 identity)
    prod = _prod([("Chile", "2015", 3993.7, 1778.4, 732723.3)])
    g = compute_insitu_grades(prod, pd.DataFrame(columns=["region", "time", "factor"]),
                              EFF_MF, EFF_SXEW)
    assert len(g) == 1 and g.iloc[0]["src"] == "icsg"
    insitu = 3993.7 / EFF_MF + 1778.4 / EFF_SXEW
    assert g.iloc[0]["g"] * 732723.3 == pytest.approx(insitu)
    assert g.iloc[0]["g"] == pytest.approx(0.01016, abs=1e-4)


def test_factor_fallback_and_gaps():
    prod = _prod([
        ("A", "2000", 80.0, 0.0, 10000.0),   # icsg-derived
        ("B", "2000", 0.0, 0.0, 5000.0),     # no production → factor fallback
        ("C", "2000", 0.0, 0.0, 300.0),      # no production, no factor → gap
        ("D", "2000", 50.0, 0.0, 0.0),       # no ore mass → dropped
    ])
    fb = pd.DataFrame([("B", "2000", 125.0)], columns=["region", "time", "factor"])
    g = compute_insitu_grades(prod, fb, EFF_MF, EFF_SXEW)
    assert set(g["region"]) == {"A", "B"}
    b = g[g.region == "B"].iloc[0]
    assert b["src"] == "factor"
    assert b["g"] == pytest.approx((1.0 / 125.0) / EFF_MF)


def test_region_bands_min_max_margin():
    g = pd.DataFrame({"region": ["A", "A", "A"], "time": ["1", "2", "3"],
                      "g": [0.010, 0.008, 0.012], "src": ["icsg"] * 3})
    b = region_bands(g, margin_rel=0.10).set_index("region")
    assert b.loc["A", "g_min"] == pytest.approx(0.008 * 0.9)
    assert b.loc["A", "g_max"] == pytest.approx(0.012 * 1.1)


def test_param_rows_shape_and_grammar():
    g = pd.DataFrame({"region": ["Korea, Rep."], "time": ["2015"],
                      "g": [0.01], "src": ["icsg"]})
    rows = build_param_rows(g, region_bands(g))
    # three conc-family roles per region-year
    assert len(rows) == 3
    assert set(rows["param_type"]) == {"concentration"}
    assert set(rows["unc_type"]) == {"min_max"}
    # underscore-free, punctuation-free param ids (naming._safe grammar)
    assert rows["param_id"].str.fullmatch(r"[a-z0-9]+").all()
    assert (rows["param_id"] == "oregradecosupplykorearep2015").any()
    # a = Mass edge, b = same edge in CC; region/time-resolved
    assert set(rows["a_valuation"]) == {"Mass"}
    assert set(rows["b_valuation"]) == {"CC"}
    assert set(rows["a_time"]) == {"2015"}
    co = rows[rows.param_id.str.contains("cosupply")].iloc[0]
    assert (co["a_sector_from"], co["a_entity_from"], co["a_region_from"]) == \
        ("CO", "SB", "Korea, Rep.")
    use = rows[rows.param_id.str.contains("oreuse")].iloc[0]
    assert (use["a_sector_to"], use["a_entity_to"], use["a_region_to"]) == \
        ("Ore", "Flow", "Korea, Rep.")
    sup = rows[rows.param_id.str.contains("oresupply")].iloc[0]
    assert (sup["a_sector_from"], sup["a_region_from"]) == ("Ore", "Korea, Rep.")
    # value_min/max feed both the hard band and the min_max σ
    assert (rows["value_min"] < rows["value_nominal"]).all()
    assert (rows["value_max"] > rows["value_nominal"]).all()


CONC_CONCENTRATE, CONC_CSXEW = 0.275, 0.9975


def test_tailings_grade_closes_prior_mass_balance():
    # The point of the construction (ADR 0015 amendment): the tailings Mass
    # twin TailCC/g_tail must equal the process Mass-balance residual
    # OreMass_route − ProdMass under the prior twins, per route.
    prod = _prod([("Chile", "2015", 3993.7, 1778.4, 732723.3)])
    grades = compute_insitu_grades(
        prod, pd.DataFrame(columns=["region", "time", "factor"]), EFF_MF, EFF_SXEW)
    tails = compute_tailings_grades(prod, grades, EFF_MF, EFF_SXEW,
                                    CONC_CONCENTRATE, CONC_CSXEW)
    assert set(tails["route"]) == {"mf", "sxew"}
    g = grades.iloc[0]["g"]
    mf = tails[tails.route == "mf"].iloc[0]
    tail_cc = 3993.7 * (1 - EFF_MF) / EFF_MF
    ore_mass_mf = (3993.7 / EFF_MF) / g
    tail_mass = ore_mass_mf - 3993.7 / CONC_CONCENTRATE
    assert tail_cc / mf["g"] == pytest.approx(tail_mass)
    # regression: matches the validated run's Chile 2015 prior construction
    assert tail_cc == pytest.approx(998.4, abs=0.1)
    assert mf["g"] == pytest.approx(0.002094, abs=1e-5)
    sx = tails[tails.route == "sxew"].iloc[0]
    assert 1778.4 * (1 - EFF_SXEW) / EFF_SXEW / sx["g"] == pytest.approx(
        (1778.4 / EFF_SXEW) / g - 1778.4 / CONC_CSXEW)


def test_tailings_grades_skip_fallback_and_missing_routes():
    prod = _prod([
        ("A", "2000", 80.0, 0.0, 10000.0),   # icsg, MF only → 1 mf row, no sxew
        ("B", "2000", 0.0, 0.0, 5000.0),     # factor fallback → no tailings row
    ])
    fb = pd.DataFrame([("B", "2000", 125.0)], columns=["region", "time", "factor"])
    grades = compute_insitu_grades(prod, fb, EFF_MF, EFF_SXEW)
    tails = compute_tailings_grades(prod, grades, EFF_MF, EFF_SXEW,
                                    CONC_CONCENTRATE, CONC_CSXEW)
    assert len(tails) == 1
    assert tuple(tails.iloc[0][["region", "route"]]) == ("A", "mf")


def test_tailings_rows_shape_and_grammar():
    tails = pd.DataFrame({"region": ["Korea, Rep."] * 2, "time": ["2015"] * 2,
                          "route": ["mf", "sxew"], "g": [0.002, 0.003]})
    rows = build_tailings_rows(tails, tailings_bands(tails))
    assert len(rows) == 2
    assert rows["param_id"].str.fullmatch(r"[a-z0-9]+").all()
    assert (rows["param_id"] == "oregradetailingsusemfkorearep2015").any()
    assert (rows["param_id"] == "oregradetailingsusesxewkorearep2015").any()
    mf = rows[rows.param_id.str.contains("usemf")].iloc[0]
    assert (mf["a_sector_from"], mf["a_entity_from"], mf["a_region_from"]) == \
        ("MF", "Process", "Korea, Rep.")
    assert (mf["a_sector_to"], mf["a_entity_to"]) == ("Tailings", "SB")
    assert set(rows["a_valuation"]) == {"Mass"}
    assert set(rows["b_valuation"]) == {"CC"}
    assert (rows["value_min"] < rows["value_nominal"]).all()
    assert (rows["value_max"] > rows["value_nominal"]).all()
