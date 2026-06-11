"""
test_solver_stats_diag.py
=========================
Tests for :class:`pysuto.diag.SolverStatsDiagnostic` — the post-solve row-level
CSVs: per-edge table (``a_prior``/``a_post``/``defect``/``z``) and per-constraint
table (``lhs_prior``/``lhs_post``/``residual``/``z`` over all rows).

Synthetic fixtures only (no registry): a stub ``acc`` exposing ``edge_meta`` (the
``CovAcc`` topology of the coverage/stats tests), a stub ``ConstraintRun`` (with a
small dense ``G`` so ``lhs_prior = G·a0`` is checkable by eye) and a stub
``results`` dict. All values are hand-chosen so the stats are checkable by eye.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.diag import SolverStatsDiagnostic


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class AccStub:
    """Minimal acc exposing ``edge_meta`` (9 edges; e7 is 1996, e6 is CC)."""

    @property
    def edge_meta(self) -> pd.DataFrame:
        return pd.DataFrame({
            "edge_id":     list(range(9)),
            "time":        ["1995"] * 6 + ["1995", "1996", "1995"],
            "valuation":   ["Mass", "Mass", "Mass", "Mass", "Mass", "Mass", "CC", "Mass", "Mass"],
            "from_sector": ["Mining", "Cathode", "Cathode", "CO", "Scrap", "Mining", "Mining", "Mining", "Mining"],
            "from_region": ["A"] * 9,
            "to_sector":   ["Cathode", "Mining", "FU", "Mining", "Mining", "Slag", "Cathode", "Cathode", "Anode"],
            "to_region":   ["A"] * 9,
        })


class CRStub:
    """Stub ConstraintRun: prior a0/σ_a + G + 4 constraint rows + per-t c/σ_c."""
    acc_id = "fake-acc"
    a0      = np.array([10, 0, 5, 3, 7, 0, 8, 9, 0], dtype=float)
    sigma_a = np.array([1, 1, 1, 1, 1, 0, 2, 1, 1], dtype=float)   # e5: σ_a = 0
    # G·a0 = [25, 0, -4, 9]  (sumA: Σ e0..e5; ratioA: e0 − 2·e2; mbA: e3 − e4; sumB: e7)
    G = np.array([
        [1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, -2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
    ], dtype=float)
    row_meta = pd.DataFrame([
        {"constraint_id": "sumA",   "type": "sum",     "variant": "anchor",  "cone_type": "free", "time": "1995", "valuation": "Mass"},
        {"constraint_id": "ratioA", "type": "ratio",   "variant": "ratioeq", "cone_type": "free", "time": "1995"},                       # no valuation
        {"constraint_id": "mbA",    "type": "massbal", "variant": "massbal", "cone_type": "zero", "time": "1995", "valuation": "Mass"},   # hard → z NaN
        {"constraint_id": "sumB",   "type": "sum",     "variant": "anchor",  "cone_type": "free", "time": "1996", "valuation": "Mass"},   # other year
    ])
    timestep_data = {
        "1995": {"c": np.array([100, 0, 0, 50], dtype=float),
                 "sigma_c": np.array([10, 2, 0, 5], dtype=float)},
        "1996": {"c": np.array([0, 0, 0, 50], dtype=float),
                 "sigma_c": np.array([1, 1, 1, 5], dtype=float)},
    }


class ResStub:
    F_opt  = np.array([12, 1, 5, 6, 7, 0, 9, 50, 2], dtype=float)  # defect = F_opt - a0
    c_star = np.array([104, 1, 0, 50], dtype=float)                 # residual = c_star - c


class Res96Stub:
    F_opt  = np.array([10, 0, 5, 3, 7, 0, 8, 50, 0], dtype=float)  # only e7 moved (+41)
    c_star = np.array([100, 0, 0, 52], dtype=float)


@pytest.fixture
def diag():
    return SolverStatsDiagnostic(
        {"1995": ResStub(), "1996": Res96Stub()}, CRStub(), AccStub(),
        years=("1995",),
    )


_EDGE_COLS = ["edge_id", "from_sector", "from_region", "to_sector", "to_region",
              "valuation", "a_prior", "a_post", "sigma_a", "defect", "defect_abs", "z"]

_CONST_COLS = ["constraint_id", "type", "variant", "cone_type", "valuation",
               "c", "lhs_prior", "lhs_post", "residual", "residual_abs",
               "sigma_c", "z", "z_abs"]


# ---------------------------------------------------------------------------
# Year resolution
# ---------------------------------------------------------------------------

def test_years_all_resolves_to_result_keys():
    d = SolverStatsDiagnostic(
        {"1995": ResStub(), "1996": Res96Stub()}, CRStub(), AccStub(), years="all",
    )
    assert d.years == ["1995", "1996"]


def test_years_accept_ints():
    d = SolverStatsDiagnostic(
        {"1995": ResStub()}, CRStub(), AccStub(), years=(1995,),
    )
    assert d.years == ["1995"]


def test_unknown_year_raises():
    with pytest.raises(KeyError, match="2024"):
        SolverStatsDiagnostic({"1995": ResStub()}, CRStub(), AccStub(), years=("2024",))


# ---------------------------------------------------------------------------
# Edge table
# ---------------------------------------------------------------------------

def test_edge_table_columns_and_year_filter(diag):
    t = diag.edge_table("1995")
    assert list(t.columns) == _EDGE_COLS
    assert list(t["edge_id"]) == [0, 1, 2, 3, 4, 5, 6, 8]   # e7 is 1996


def test_edge_table_values(diag):
    t = diag.edge_table("1995").set_index("edge_id")
    e0 = t.loc[0]
    assert e0["a_prior"] == 10.0 and e0["a_post"] == 12.0
    assert e0["defect"] == pytest.approx(2.0)
    assert e0["defect_abs"] == pytest.approx(2.0)
    assert e0["z"] == pytest.approx(2.0)                     # σ_a = 1
    assert e0["from_sector"] == "Mining" and e0["valuation"] == "Mass"


def test_edge_table_z_uses_sigma_a(diag):
    t = diag.edge_table("1995").set_index("edge_id")
    assert t.loc[6, "z"] == pytest.approx(0.5)               # defect 1 / σ_a 2


def test_edge_table_sigma_zero_gives_nan_z(diag):
    t = diag.edge_table("1995").set_index("edge_id")
    assert t.loc[5, "defect"] == 0.0
    assert pd.isna(t.loc[5, "z"])                            # σ_a = 0 → NaN, not inf


def test_edge_table_other_year(diag):
    t = diag.edge_table("1996")
    assert list(t["edge_id"]) == [7]
    assert t.iloc[0]["defect"] == pytest.approx(41.0)


# ---------------------------------------------------------------------------
# Constraint table
# ---------------------------------------------------------------------------

def test_constraint_table_all_rows_in_year(diag):
    t = diag.constraint_table("1995")
    assert list(t.columns) == _CONST_COLS
    assert list(t["constraint_id"]) == ["sumA", "ratioA", "mbA"]   # hard included, 1996 excluded


def test_constraint_table_prior_post_residual_z(diag):
    t = diag.constraint_table("1995").set_index("constraint_id")
    s = t.loc["sumA"]
    assert s["c"] == 100.0
    assert s["lhs_prior"] == pytest.approx(25.0)     # G·a0
    assert s["lhs_post"] == pytest.approx(104.0)     # c_star
    assert s["residual"] == pytest.approx(4.0)       # lhs_post − c
    assert s["residual_abs"] == pytest.approx(4.0)
    assert s["sigma_c"] == 10.0
    assert s["z"] == pytest.approx(0.4)
    assert s["z_abs"] == pytest.approx(0.4)
    assert s["type"] == "sum" and s["cone_type"] == "free" and s["valuation"] == "Mass"


def test_constraint_table_hard_row_z_nan(diag):
    t = diag.constraint_table("1995").set_index("constraint_id")
    mb = t.loc["mbA"]
    assert mb["cone_type"] == "zero"
    assert mb["lhs_prior"] == pytest.approx(-4.0)
    assert mb["residual"] == pytest.approx(0.0)      # c_star = c = 0 on the hard row
    assert pd.isna(mb["z"]) and pd.isna(mb["z_abs"]) # σ_c = 0


def test_constraint_table_missing_valuation_nan(diag):
    t = diag.constraint_table("1995").set_index("constraint_id")
    r = t.loc["ratioA"]
    assert r["residual"] == pytest.approx(1.0)
    assert r["z"] == pytest.approx(0.5)              # 1 / 2
    assert pd.isna(r["valuation"])                   # ratio_eq omits valuation


# ---------------------------------------------------------------------------
# write()
# ---------------------------------------------------------------------------

def test_write_emits_pair_per_year(tmp_path):
    d = SolverStatsDiagnostic(
        {"1995": ResStub(), "1996": Res96Stub()}, CRStub(), AccStub(), years="all",
    )
    paths = d.write(str(tmp_path))
    names = [p.split("/")[-1] for p in paths]
    assert names == ["edges_1995.csv", "constraints_1995.csv",
                     "edges_1996.csv", "constraints_1996.csv"]
    for p in paths:
        assert pd.read_csv(p).shape[0] > 0
    edges = pd.read_csv(paths[0])
    assert "edge_id" in edges.columns                # index=False: ids are columns
