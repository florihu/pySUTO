"""BACI's quantity field contradicts its own value field (ADR 0033).

HS 740311 is refined cathode (~99.99 % Cu), so its unit value *is* the copper
price in the same release. The **value-implied Cu fraction**
``unit_value(product) / unit_value(cathode)`` must then lie inside ADR 0028's
physical band [0.20, 0.35]. Where it does not, the *quantity* is the broken
field — the value agrees with ICSG production in every year — so the quantity
is reconstructed from it and the leg carries the grade's declared σ.

_Avoid_: reading the fraction as a grade. GLOSSARY.md's **Concentrate grade**
entry already warns that values of 0.6–0.9 "are not grades at all".
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "scripts", "parse", "baci_parse"))
from baci_parse_feed import (  # noqa: E402
    BAND_HI, CV_GRADE_DECL, CV_PED, MAX_CV_SCALE, MIN_LEG_T,
    reconstruct_quantity, reference_fractions, value_implied_cu_fraction)

CONC = 260300


def _legs(rows):
    """rows: (year, exporter, importer, v_kusd, q_t) for HS 260300."""
    return pd.DataFrame(
        [dict(Year=y, Sector=CONC, Region_from=i, Region_to=j, v=v, q=q)
         for y, i, j, v, q in rows])


def _price(years, usd_t=8000.0):
    return pd.Series({y: usd_t for y in years})


# ---------------------------------------------------------------------------
# the fraction itself
# ---------------------------------------------------------------------------

def test_fraction_is_unit_value_over_the_copper_price():
    """A 27 % concentrate at an 8000 USD/t copper price is 2160 USD/t."""
    legs = _legs([(2020, 152, 156, 2160.0, 1000.0)])       # 2.16 MUSD / 1 kt
    f = value_implied_cu_fraction(legs, _price([2020]))
    assert f.iloc[0] == pytest.approx(0.27)


def test_fraction_flags_only_outside_the_physical_band():
    legs = _legs([(2020, 152, 156, 2160.0, 1000.0),        # 0.270  in band
                  (2020, 152, 392, 5200.0, 1000.0)])       # 0.650  impossible
    f = value_implied_cu_fraction(legs, _price([2020]))
    assert list(f > BAND_HI) == [False, True]


# ---------------------------------------------------------------------------
# g_ref — measured from the region's own in-band years, never assumed
# ---------------------------------------------------------------------------

def test_g_ref_is_the_median_of_the_regions_own_in_band_YEARS():
    legs = _legs([(2018, 152, 156, 2000.0, 1000.0),        # 0.250 in band
                  (2019, 152, 156, 2400.0, 1000.0),        # 0.300 in band
                  (2020, 152, 156, 2200.0, 1000.0),        # 0.275 in band
                  (2021, 152, 156, 5200.0, 1000.0)])       # 0.650 EXCLUDED
    r = reference_fractions(legs, _price([2018, 2019, 2020, 2021]))
    assert r.loc[152, "g_ref"] == pytest.approx(0.275)     # median of the three
    assert r.loc[152, "n_in_band"] == 3                    # years, not legs


def test_a_tight_in_band_history_earns_the_declared_grade_sigma():
    """Chile-shaped: the model must never claim to know a grade better than
    the conc row it already declares."""
    legs = _legs([(y, 152, 156, 2200.0, 1000.0) for y in range(2010, 2020)])
    r = reference_fractions(legs, _price(range(2010, 2020)))
    assert r.loc[152, "cv_gref"] == pytest.approx(CV_GRADE_DECL)
    assert r.loc[152, "cv_scale"] == pytest.approx(
        np.sqrt(1.0 + (CV_GRADE_DECL / CV_PED) ** 2))


def test_a_scattered_history_earns_a_looser_sigma():
    """A re-exporter whose fraction wanders gets genuinely more doubt."""
    vals = [1600.0, 2400.0, 1700.0, 2700.0, 2000.0, 2800.0]     # 0.20-0.35
    legs = _legs([(2010 + n, 40, 156, v, 1000.0) for n, v in enumerate(vals)])
    r = reference_fractions(legs, _price(range(2010, 2016)))
    assert r.loc[40, "cv_gref"] > CV_GRADE_DECL
    assert r.loc[40, "cv_scale"] > np.sqrt(1.0 + (CV_GRADE_DECL / CV_PED) ** 2)


def test_too_few_in_band_years_means_more_doubt_not_less():
    """One in-band year gives no measurable spread; that is ignorance, and it
    must widen σ rather than fall back to the tight declared floor."""
    legs = _legs([(2020, 76, 156, 2200.0, 1000.0),         # the only in-band one
                  (2021, 76, 156, 5200.0, 1000.0)])
    r = reference_fractions(legs, _price([2020, 2021]))
    assert r.loc[76, "cv_scale"] == pytest.approx(MAX_CV_SCALE)


def test_cv_scale_is_capped():
    vals = [1600.0, 2800.0] * 6                             # maximal spread
    legs = _legs([(2005 + n, 40, 156, v, 1000.0) for n, v in enumerate(vals)])
    r = reference_fractions(legs, _price(range(2005, 2017)))
    assert r.loc[40, "cv_scale"] <= MAX_CV_SCALE


# ---------------------------------------------------------------------------
# the reconstruction
# ---------------------------------------------------------------------------

def test_reconstruction_restores_the_regions_own_fraction():
    """q_new is exactly what the leg's value buys at the region's g_ref."""
    legs = _legs([(2018, 152, 156, 2200.0, 1000.0),        # 0.275 in band
                  (2021, 152, 156, 5200.0, 1000.0)])       # 0.650 impossible
    px = _price([2018, 2021])
    rec = reconstruct_quantity(legs, reference_fractions(legs, px), px)
    bad = rec[rec.Year == 2021].iloc[0]
    assert bad.q_new == pytest.approx(5200.0 * 1000 / (8000.0 * 0.275))
    # and the reconstructed leg now sits exactly at g_ref
    assert (bad.v * 1000 / bad.q_new) / 8000.0 == pytest.approx(0.275)


def test_in_band_legs_are_left_alone():
    legs = _legs([(2018, 152, 156, 2200.0, 1000.0),
                  (2021, 152, 156, 5200.0, 1000.0)])
    px = _price([2018, 2021])
    rec = reconstruct_quantity(legs, reference_fractions(legs, px), px)
    ok = rec[rec.Year == 2018].iloc[0]
    assert ok.q_new == pytest.approx(ok.q_obs)
    assert not ok.flagged


def test_reconstruction_only_ever_raises_the_quantity():
    """The defect is under-recording; a fraction above the band always means
    too little mass for the value, never too much."""
    legs = _legs([(2021, 152, j, 5200.0, 1000.0) for j in (156, 392, 76)])
    px = _price([2021])
    legs = pd.concat([legs, _legs([(2018, 152, 156, 2200.0, 1000.0)])])
    px = _price([2018, 2021])
    rec = reconstruct_quantity(legs, reference_fractions(legs, px), px)
    assert (rec.q_new >= rec.q_obs - 1e-9).all()


def test_tiny_legs_are_not_flagged():
    """Sub-floor legs have noisy unit values — Chile 2021 carries partners at
    an implied 8.6 on rounding dust — and are left exactly as reported."""
    legs = _legs([(2018, 152, 156, 2200.0, 1000.0),
                  (2021, 152, 826, 80.0, MIN_LEG_T / 10.0)])
    px = _price([2018, 2021])
    rec = reconstruct_quantity(legs, reference_fractions(legs, px), px)
    dust = rec[rec.Region_to == 826].iloc[0]
    assert not dust.flagged and dust.q_new == pytest.approx(dust.q_obs)


def test_sidecar_carries_every_column_the_gate_needs():
    legs = _legs([(2018, 152, 156, 2200.0, 1000.0),
                  (2021, 152, 156, 5200.0, 1000.0)])
    px = _price([2018, 2021])
    rec = reconstruct_quantity(legs, reference_fractions(legs, px), px)
    for c in ("Year", "Sector", "Region_from", "Region_to",
              "q_obs", "q_new", "cu_frac", "g_ref", "cv_scale", "flagged"):
        assert c in rec.columns, c


# ---------------------------------------------------------------------------
# the cv_root_scale seam: two files must COMPOSE, never overwrite (ADR 0033)
# ---------------------------------------------------------------------------
# 3,593 of 6,706 Concentrate legs already carry an ADR 0027 re-export scale, so
# a second file that assigned would silently undo the netting's σ on every
# overlapping leg.

import types  # noqa: E402


def _stub_feed(tmp_path, monkeypatch):
    """`baci_feed` with its module paths pointed at a temp feed dir."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import baci_feed
    monkeypatch.setattr(baci_feed, "RAW_DATA", str(tmp_path / "values.csv"))
    return baci_feed


class _Acc:
    """Two cross-region Flow-origin Concentrate edges, Chile→China 2021."""
    edge_index = list(range(4))
    node_index = None


def _ei_frame():
    return pd.DataFrame({
        "edge_id": [0, 1, 2, 3],
        "sector_from": ["Concentrate"] * 4,
        "region_from": ["Chile"] * 4,
        "region_to": ["China", "China", "Chile", "China"],
        "entity_from": ["Flow", "Flow", "Flow", "Process"],
        "time": ["2021", "2021", "2021", "2021"],
        "valuation": ["Mass", "CC", "Mass", "Mass"],
    })


def _scale_csv(path, cv):
    pd.DataFrame([dict(sector="Concentrate", region_from="Chile",
                       region_to="China", time=2021, cv_scale=cv)]).to_csv(path, index=False)


@pytest.fixture
def wired(tmp_path, monkeypatch):
    bf = _stub_feed(tmp_path, monkeypatch)
    import pysuto.param_relation as pr
    monkeypatch.setattr(pr, "expand_edge_index", lambda *a, **k: _ei_frame())
    return bf, tmp_path


def test_two_scale_files_multiply(wired):
    bf, d = wired
    _scale_csv(d / "reexport_cv_scale.csv", 2.0)
    _scale_csv(d / "quantity_cv_scale.csv", 1.5)
    s = bf.load_cv_scales(_Acc)
    assert s[0] == pytest.approx(3.0) and s[1] == pytest.approx(3.0)


def test_one_file_alone_reproduces_todays_array(wired):
    bf, d = wired
    _scale_csv(d / "reexport_cv_scale.csv", 2.0)
    s = bf.load_cv_scales(_Acc)
    assert s[0] == pytest.approx(2.0)
    assert s[2] == pytest.approx(1.0)        # domestic leg untouched
    assert s[3] == pytest.approx(1.0)        # Process-origin edge untouched


def test_neither_file_returns_none(wired):
    bf, _ = wired
    assert bf.load_cv_scales(_Acc) is None


def test_the_product_is_capped(wired):
    bf, d = wired
    _scale_csv(d / "reexport_cv_scale.csv", 4.0)
    _scale_csv(d / "quantity_cv_scale.csv", 4.0)
    s = bf.load_cv_scales(_Acc)
    assert s[0] == pytest.approx(bf.MAX_CV_SCALE)


def test_a_leg_in_a_clean_year_is_left_alone():
    """The ICSG cross-check licenses the aggregate, not the leg: where an
    exporter-year's own fraction is in band its recorded mass already matches
    its value, and lifting one leg there would fabricate mass."""
    legs = _legs([(2019, 152, 156, 2000.0, 1000.0),        # 0.250, the bulk
                  (2019, 152, 392, 500.0, 100.0)])         # 0.625 but tiny
    px = _price([2019])
    # aggregate: 2500 kUSD / 1100 t = 2273 USD/t -> 0.284, in band
    rec = reconstruct_quantity(legs, reference_fractions(legs, px), px)
    assert not rec.flagged.any()
    assert (rec.q_new == rec.q_obs).all()


def test_a_leg_in_a_flagged_year_is_reconstructed():
    legs = _legs([(2021, 152, 156, 5200.0, 1000.0),        # 0.650
                  (2021, 152, 76, 2000.0, 1000.0),         # 0.250 in band
                  (2018, 152, 156, 2200.0, 1000.0)])       # clean year
    px = _price([2018, 2021])
    rec = reconstruct_quantity(legs, reference_fractions(legs, px), px)
    f = rec.set_index(["Year", "Region_to"]).flagged
    assert f[(2021, 156)] and not f[(2021, 76)] and not f[(2018, 156)]
