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


# ---------------------------------------------------------------------------
# All-years frames (the figure data layer) — same numbers as the tables
# ---------------------------------------------------------------------------

class ResWithResid(ResStub):
    residuals = np.array([4.0, 1.0, 1e-9, 0.0])          # true G·a*−c, also on the hard row


class Res96WithResid(Res96Stub):
    residuals = np.array([0.0, 0.0, 0.0, 2.0])


class WarmRes(ResStub):
    """Legacy full-length per-period prior: e0 moved to 11 (a0 was 10)."""
    a_prior = np.array([11, 0, 5, 3, 7, 0, 8, 9, 0], dtype=float)
    cols_t = None


@pytest.fixture
def frames_diag():
    return SolverStatsDiagnostic(
        {"1995": ResWithResid(), "1996": Res96WithResid()}, CRStub(), AccStub(), years="all",
    )


def test_constraint_frame_extends_the_tables(frames_diag):
    cf = frames_diag.constraint_frame()
    assert len(cf) == 4
    for extra in ("year", "identity", "label", "lhs_prior_t", "resid_solver"):
        assert extra in cf.columns
    for y in ("1995", "1996"):
        sl = cf[cf.year == y].reset_index(drop=True)
        tbl = frames_diag.constraint_table(y)
        assert list(tbl.columns) == _CONST_COLS
        pd.testing.assert_frame_equal(sl[_CONST_COLS].astype(tbl.dtypes.to_dict()), tbl, check_dtype=False)


def test_constraint_frame_identity_is_cross_year_and_from_columns(frames_diag):
    cf = frames_diag.constraint_frame().set_index("constraint_id")
    # sumA (1995) and sumB (1996) share every identity column → one identity
    assert cf.loc["sumA", "identity"] == cf.loc["sumB", "identity"]
    assert cf.loc["sumA", "identity"] != cf.loc["ratioA", "identity"] != cf.loc["mbA", "identity"]
    assert (cf["identity"] >= 0).all()
    assert cf.loc["sumA", "label"] == "sum anchor Mass"      # no sector/region columns in the stub


def test_constraint_frame_resid_solver_comes_from_results(frames_diag):
    cf = frames_diag.constraint_frame().set_index("constraint_id")
    assert cf.loc["mbA", "residual"] == 0.0                  # c_star = c on the hard row
    assert cf.loc["mbA", "resid_solver"] == pytest.approx(1e-9)
    assert cf.loc["sumB", "resid_solver"] == pytest.approx(2.0)


def test_constraint_frame_lhs_prior_t_cold_equals_lhs_prior(frames_diag):
    cf = frames_diag.constraint_frame()
    np.testing.assert_allclose(cf["lhs_prior_t"], cf["lhs_prior"])


def test_constraint_frame_lhs_prior_t_warm_differs():
    d = SolverStatsDiagnostic({"1995": WarmRes(), "1996": Res96Stub()}, CRStub(), AccStub(), years="all")
    cf = d.constraint_frame().set_index("constraint_id")
    assert cf.loc["sumA", "lhs_prior"] == pytest.approx(25.0)      # G·a0 (CSV column, frozen)
    assert cf.loc["sumA", "lhs_prior_t"] == pytest.approx(26.0)    # G·a_prior(1995)
    e = d.edge_table("1995").set_index("edge_id")
    assert e.loc[0, "a_prior"] == 11.0 and e.loc[0, "defect"] == pytest.approx(1.0)


def test_constraint_frame_unidentified_rows_get_minus_one():
    class CRDup(CRStub):
        row_meta = pd.DataFrame([
            {"constraint_id": "sumA",  "type": "sum",   "variant": "anchor", "cone_type": "free", "time": "1995", "valuation": "Mass"},
            {"constraint_id": "effA",  "type": "ratio", "variant": "effx",   "cone_type": "free", "time": "1995", "valuation": "CC"},
            {"constraint_id": "mbA",   "type": "massbal", "variant": "massbal", "cone_type": "zero", "time": "1995", "valuation": "Mass"},
            {"constraint_id": "effB",  "type": "ratio", "variant": "effx",   "cone_type": "free", "time": "1995", "valuation": "CC"},
        ])
    d = SolverStatsDiagnostic({"1995": ResWithResid()}, CRDup(), AccStub(), years="all")
    cf = d.constraint_frame().set_index("constraint_id")
    assert cf.loc["effA", "identity"] == -1 and cf.loc["effB", "identity"] == -1
    assert cf.loc["sumA", "identity"] >= 0 and cf.loc["mbA", "identity"] >= 0
    assert cf.loc["effA", "label"].startswith("(unidentified)")


def test_edge_frame_identity_and_label(frames_diag):
    ef = frames_diag.edge_frame().set_index("edge_id")
    assert set(ef["year"]) == {"1995", "1996"}
    # e0 (1995) and e7 (1996) are the same Mining/A→Cathode/A Mass edge across years
    assert ef.loc[0, "identity"] == ef.loc[7, "identity"]
    assert ef.loc[0, "identity"] != ef.loc[6, "identity"]         # CC twin is another identity
    assert ef.loc[0, "label"] == "Mining|A→Cathode|A|Mass"
    for y in ("1995", "1996"):
        sl = ef[ef.year == y].reset_index()
        tbl = frames_diag.edge_table(y)
        pd.testing.assert_frame_equal(sl[_EDGE_COLS], tbl)


def test_acc_none_disables_only_the_edge_side():
    d = SolverStatsDiagnostic({"1995": ResWithResid()}, CRStub(), None, years="all")
    assert d.edge_frame() is None
    with pytest.raises(RuntimeError, match="edge_meta"):
        d.edge_table("1995")
    assert len(d.constraint_frame()) == 3


def test_write_rankings_emits_one_csv_per_family(frames_diag, tmp_path):
    paths = frames_diag.write_rankings(str(tmp_path))
    names = sorted(p.split("/")[-1] for p in paths)
    assert names == ["ranking_edges.csv", "ranking_soft_ratios.csv", "ranking_sum_rows.csv"]
    r = pd.read_csv(tmp_path / "ranking_sum_rows.csv")
    assert list(r.columns)[:6] == ["rank", "identity", "label", "sector_key", "zsq", "share"]
    assert len(r) == 1                                            # sumA/sumB = one identity
    assert r.iloc[0]["n_years"] == 2
    suffixed = frames_diag.write_rankings(str(tmp_path), regions=["A"], suffix="__a")
    assert all(p.endswith("__a.csv") for p in suffixed)
