"""
test_objective_stats.py
=======================
Numeric tests for the objective-share builders in :mod:`pysuto.diag.stats.objective`
and the frame aggregators in :mod:`pysuto.diag.stats._core` (``sector_totals``,
``boundary_edges``, ``filter_regions``).

Hand-built frames only (no solver, no registry): four edge identities over three
years and a small constraint frame with sum rows, a pair-scoped soft ratio, two
set-scoped soft-ratio rows that share every identity column (→ unidentified) and
one hard mass-balance row per year. Every expected number is checkable by hand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.diag.stats._core import boundary_edges, filter_regions, sector_totals
from pysuto.diag.stats.objective import (
    FAMILIES, family_frame, objective_decomposition, profiles, rank_identities,
    share_by_key,
)

YEARS = ["1995", "1996", "1997"]


def _edges() -> pd.DataFrame:
    """4 identities × 3 years (+ one SB-output edge for the boundary test)."""
    rows = []
    spec = {
        # identity: (from_sector, from_region, from_entity, to_sector, to_region, to_entity, valuation, z per year, prior per year)
        0: ("Mining", "A", "Process", "Ore", "A", "Flow", "Mass", [1.0, 2.0, 3.0], [10.0, 10.0, 10.0]),
        1: ("Ore", "A", "Flow", "Smelting", "B", "Process", "Mass", [0.5, 0.5, 0.5], [4.0, 4.0, 4.0]),
        2: ("Concentrate", "B", "Flow", "Smelting", "B", "Process", "CC", [4.0, np.nan, 0.0], [2.0, 2.0, 2.0]),
        3: ("CO", "A", "SB", "Mining", "A", "Process", "Mass", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        4: ("Mining", "A", "Process", "Tailings", "A", "SB", "Mass", [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
    }
    for ident, (fs, fr, fe, ts, tr, te, val, zs, priors) in spec.items():
        for y, z, prior in zip(YEARS, zs, priors):
            sigma = 1.0
            defect = z * sigma if np.isfinite(z) else 0.0
            rows.append({
                "identity": ident, "label": f"{fs}|{fr}→{ts}|{tr}|{val}",
                "from_sector": fs, "from_region": fr, "from_entity": fe,
                "to_sector": ts, "to_region": tr, "to_entity": te, "valuation": val,
                "from_entity_sub": "Nature" if fs == "CO" else "",
                "to_entity_sub": "SEM" if ts == "Tailings" else "",
                "year": y, "z": z, "defect": defect,
                "a_prior": prior, "a_post": prior + defect, "sigma_a": sigma,
            })
    return pd.DataFrame(rows)


def _rows() -> pd.DataFrame:
    """Constraint frame: per year 2 sum rows, 1 pair-scoped soft ratio, 2 set-scoped
    soft ratios sharing every identity column, 1 hard mass-balance row."""
    rows = []
    for i, y in enumerate(["1995", "1996"]):
        base = dict(year=y, num_sector_from=None, num_region_from=None, num_sector_to=None,
                    num_region_to=None, num_valuation=None, denom_valuation=None,
                    sector=None, region=None, resid_solver=np.nan)
        rows += [
            {**base, "constraint_id": f"sum_supplytotout_Mining_A_Mass_{y}", "type": "sum", "variant": "supplytotout",
             "cone_type": "free", "valuation": "Mass", "sector": "Mining", "region": "A",
             "identity": 10, "label": "supplytotout Mining·A Mass", "z": 2.0, "residual": 20.0},
            {**base, "constraint_id": f"sum_usetotout_Ore_B_Mass_{y}", "type": "sum", "variant": "usetotout",
             "cone_type": "free", "valuation": "Mass", "sector": "Ore", "region": "B",
             "identity": 11, "label": "usetotout Ore·B Mass", "z": [1.0, 0.0][i], "residual": [5.0, 0.0][i]},
            {**base, "constraint_id": f"ratio_concore_x_{y}", "type": "ratio", "variant": "concore",
             "cone_type": "free", "valuation": "CC", "num_sector_from": "Mining", "num_region_from": "A",
             "num_sector_to": "Ore", "num_region_to": "A", "num_valuation": "CC", "denom_valuation": "Mass",
             "identity": 12, "label": "concore Mining.A→Ore.A CC/Mass", "z": 0.5, "residual": 0.1},
            {**base, "constraint_id": f"ratio_effsmelting_A_CC_{y}", "type": "ratio", "variant": "effsmelting",
             "cone_type": "free", "valuation": "CC", "identity": -1,
             "label": f"(unidentified) ratio_effsmelting_A_CC_{y}", "z": 10.0, "residual": 1.0},
            {**base, "constraint_id": f"ratio_effsmelting_B_CC_{y}", "type": "ratio", "variant": "effsmelting",
             "cone_type": "free", "valuation": "CC", "identity": -1,
             "label": f"(unidentified) ratio_effsmelting_B_CC_{y}", "z": 10.0, "residual": 1.0},
            {**base, "constraint_id": f"massbal_massbal_Mining_A_Mass_{y}", "type": "massbal", "variant": "massbal",
             "cone_type": "zero", "valuation": "Mass", "sector": "Mining", "region": "A",
             "identity": 13, "label": "massbal Mining·A Mass", "z": np.nan, "residual": 0.0,
             "resid_solver": [1e-9, 5e-9][i]},
        ]
    return pd.DataFrame(rows)


@pytest.fixture
def edges():
    return _edges()


@pytest.fixture
def rows():
    return _rows()


# ---------------------------------------------------------------------------
# family_frame
# ---------------------------------------------------------------------------

def test_family_frame_edges_harmonised_columns_and_sector_key(edges):
    f = family_frame(edges, None, "edges")
    assert list(f.columns[:7]) == ["family", "identity", "label", "sector_key", "year", "z", "delta"]
    assert len(f) == 15
    assert set(f["sector_key"]) == {"Mining→Ore", "Ore→Smelting", "Concentrate→Smelting",
                                    "CO→Mining", "Mining→Tailings"}
    assert (f["delta"] == edges["defect"]).all()
    assert "a_post" in f.columns                       # source columns retained


def test_family_frame_returns_none_without_source(edges, rows):
    assert family_frame(None, rows, "edges") is None
    assert family_frame(edges, None, "sum_rows") is None
    assert family_frame(edges, None, "soft_ratios") is None


def test_family_frame_sum_rows_and_soft_ratios(rows):
    s = family_frame(None, rows, "sum_rows")
    assert len(s) == 4 and set(s["sector_key"]) == {"Mining", "Ore"}
    assert (s["delta"] == rows.loc[rows.type == "sum", "residual"].to_numpy()).all()
    r = family_frame(None, rows, "soft_ratios")
    assert len(r) == 6                                   # 2 concore + 4 eff
    assert set(r["sector_key"]) == {"Mining→Ore", "effsmelting"}   # variant fallback on set-scoped rows
    assert (r["family"] == "soft_ratios").all()


def test_family_frame_region_touch(edges, rows):
    e = family_frame(edges, None, "edges", regions=["B"])
    assert set(e["identity"]) == {1, 2}                  # either endpoint in B
    s = family_frame(None, rows, "sum_rows", regions=["A"])
    assert set(s["identity"]) == {10}
    r = family_frame(None, rows, "soft_ratios", regions=["A"])
    assert set(r["identity"]) == {12}                    # eff rows carry no region → dropped


def test_family_frame_unknown_family(edges):
    with pytest.raises(ValueError, match="unknown family"):
        family_frame(edges, None, "hard_rows")


# ---------------------------------------------------------------------------
# share_by_key
# ---------------------------------------------------------------------------

def test_share_by_key_totals_and_other_fold(edges):
    f = family_frame(edges, None, "edges")
    s = share_by_key(f)
    assert list(s.columns) == ["sector_key", "zsq", "share", "n_rows", "n_identities"]
    total = 14 + 0.75 + 16 + 0 + 0
    assert s["zsq"].sum() == pytest.approx(total)
    assert s["share"].sum() == pytest.approx(1.0)
    assert s.iloc[0]["sector_key"] == "Concentrate→Smelting" and s.iloc[0]["zsq"] == pytest.approx(16.0)
    assert s.loc[s.sector_key == "Concentrate→Smelting", "n_rows"].item() == 2   # NaN z dropped
    s2 = share_by_key(f, top=2)
    assert list(s2["sector_key"]) == ["Concentrate→Smelting", "Mining→Ore", "other"]
    assert s2.iloc[2]["zsq"] == pytest.approx(0.75)
    assert s2["share"].sum() == pytest.approx(1.0)


def test_share_by_key_none_passthrough():
    assert share_by_key(None) is None


# ---------------------------------------------------------------------------
# rank_identities
# ---------------------------------------------------------------------------

def test_rank_identities_edges(edges):
    r = rank_identities(family_frame(edges, None, "edges"))
    assert list(r.columns) == ["rank", "identity", "label", "sector_key", "zsq", "share",
                               "sum_abs_delta", "n_years", "z_absmax", "year_absmax", "n_years_gt3"]
    assert list(r["identity"][:3]) == [2, 0, 1]
    top = r.set_index("identity")
    assert top.loc[0, "zsq"] == pytest.approx(14.0)
    assert top.loc[0, "sum_abs_delta"] == pytest.approx(6.0)
    assert top.loc[0, "n_years"] == 3 and top.loc[0, "z_absmax"] == 3.0 and top.loc[0, "year_absmax"] == "1997"
    assert top.loc[0, "n_years_gt3"] == 0                # 3 is not > 3
    assert top.loc[2, "zsq"] == pytest.approx(16.0) and top.loc[2, "n_years"] == 2   # NaN year dropped
    assert top.loc[2, "n_years_gt3"] == 1 and top.loc[2, "year_absmax"] == "1995"
    assert r["share"].sum() == pytest.approx(1.0)
    assert r.attrs["n_unidentified"] == 0
    assert r.attrs["zsq_total"] == pytest.approx(30.75)
    assert list(rank_identities(family_frame(edges, None, "edges"), top=2)["rank"]) == [1, 2]


def test_rank_identities_excludes_unidentified_but_counts_them(rows):
    r = rank_identities(family_frame(None, rows, "soft_ratios"))
    assert list(r["identity"]) == [12]
    assert r.attrs["n_unidentified"] == 4
    assert r.attrs["zsq_total"] == pytest.approx(400.5)
    assert r.iloc[0]["share"] == pytest.approx(0.5 / 400.5)   # share vs the family total incl. unidentified


def test_rank_identities_empty_family_keeps_schema(rows):
    empty = family_frame(None, rows[rows.type == "massbal"].assign(type="sum").iloc[0:0], "sum_rows")
    r = rank_identities(empty)
    assert r.empty and "zsq" in r.columns and r.attrs["n_unidentified"] == 0


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

def test_profiles_wide_with_gaps(edges):
    f = family_frame(edges, None, "edges")
    p = profiles(f, [2, 0], years=YEARS)
    assert list(p.index) == [2, 0] and list(p.columns) == YEARS
    assert p.loc[0].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(p.loc[2, "1996"]) and p.loc[2, "1995"] == 4.0


# ---------------------------------------------------------------------------
# objective_decomposition
# ---------------------------------------------------------------------------

def test_objective_decomposition_sums(edges, rows):
    d = objective_decomposition(edges, rows).set_index("year")
    assert list(d.index) == YEARS                          # union of years, sorted
    assert d.loc["1995", "zsq_edges"] == pytest.approx(1 + 0.25 + 16 + 0 + 0)
    assert d.loc["1996", "zsq_edges"] == pytest.approx(4 + 0.25)          # NaN z ignored
    assert d.loc["1995", "zsq_sum_rows"] == pytest.approx(4 + 1)
    assert d.loc["1996", "zsq_sum_rows"] == pytest.approx(4 + 0)
    assert d.loc["1995", "zsq_soft_ratios"] == pytest.approx(0.25 + 200)  # unidentified rows DO count
    assert np.isnan(d.loc["1997", "zsq_sum_rows"])                        # no rows that year
    assert d.loc["1995", "zsq_total"] == pytest.approx(17.25 + 5 + 200.25)
    assert d.loc["1995", "max_abs_resid_massbal"] == pytest.approx(1e-9)  # from resid_solver, not residual
    assert d.loc["1996", "max_abs_resid_massbal"] == pytest.approx(5e-9)
    assert d.loc["1995", "share_gt3_edges"] == pytest.approx(1 / 5)
    assert d.loc["1996", "share_gt3_edges"] == pytest.approx(0.0)
    assert d.loc["1995", "share_gt3_soft_ratios"] == pytest.approx(2 / 3)
    assert d.loc["1995", "n_sum_rows"] == 2


def test_objective_decomposition_missing_family_is_nan(rows):
    d = objective_decomposition(None, rows, years=["1995"]).set_index("year")
    assert np.isnan(d.loc["1995", "zsq_edges"]) and np.isnan(d.loc["1995", "n_edges"])
    assert d.loc["1995", "zsq_total"] == pytest.approx(5 + 200.25)        # min_count=1: NaN family skipped


def test_objective_decomposition_region_filter(edges, rows):
    d = objective_decomposition(edges, rows, years=["1995"], regions=["B"]).set_index("year")
    assert d.loc["1995", "zsq_edges"] == pytest.approx(0.25 + 16)         # ids 1, 2 touch B
    assert d.loc["1995", "zsq_sum_rows"] == pytest.approx(1.0)            # usetotout Ore·B
    assert np.isnan(d.loc["1995", "max_abs_resid_massbal"])               # massbal row is in A


# ---------------------------------------------------------------------------
# _core aggregators
# ---------------------------------------------------------------------------

def test_sector_totals_retention(edges):
    t = sector_totals(edges, "from_sector", years=YEARS, valuations=["Mass", "CC"])
    assert list(t.columns) == ["group", "valuation", "year", "sum_post", "sum_prior", "retention"]
    mining = t[(t.group == "Mining") & (t.valuation == "Mass")].set_index("year")
    # Mining→Ore (prior 10, post 10+z) + Mining→Tailings (prior 1, post 1): 1995 → (11+1)/(10+1)
    assert mining.loc["1995", "sum_prior"] == pytest.approx(11.0)
    assert mining.loc["1995", "sum_post"] == pytest.approx(12.0)
    assert mining.loc["1995", "retention"] == pytest.approx(12 / 11)
    co = t[(t.group == "CO") & (t.valuation == "Mass")].set_index("year")
    assert np.isnan(co.loc["1995", "retention"])                          # zero prior → NaN
    assert np.isnan(t[(t.group == "Mining") & (t.valuation == "CC")]["sum_post"]).all()   # grid completed


def test_boundary_edges_directions_and_entity_sub(edges):
    b = boundary_edges(edges)
    assert set(b["identity"]) == {3, 4}
    assert (b.loc[b.identity == 3, "direction"] == "sb_input").all()
    assert (b.loc[b.identity == 3, "sb_sector"] == "CO").all()
    assert (b.loc[b.identity == 3, "sb_entity_sub"] == "Nature").all()
    assert (b.loc[b.identity == 4, "direction"] == "sb_output").all()
    assert (b.loc[b.identity == 4, "sb_sector"] == "Tailings").all()
    assert (b.loc[b.identity == 4, "sb_entity_sub"] == "SEM").all()
    no_sub = boundary_edges(edges.drop(columns=["from_entity_sub", "to_entity_sub"]))
    assert (no_sub["sb_entity_sub"] == "").all()


def test_filter_regions_or_semantics(edges):
    assert filter_regions(edges, ("from_region", "to_region"), None) is edges
    b = filter_regions(edges, ("from_region", "to_region"), ["B"])
    assert set(b["identity"]) == {1, 2}
    only_to = filter_regions(edges, ("to_region",), ["B"])
    assert set(only_to["identity"]) == {1, 2}
    missing_col = filter_regions(edges, ("nope",), ["B"])
    assert missing_col.empty
