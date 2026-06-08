"""
test_solver_stats_diag.py
=========================
Tests for :class:`pysuto.diag.SolverStatsDiagnostic` — the post-solve per-bucket
CSVs: edge **defect** (``F_opt − a0``) distribution by from/to endpoint, plus the
soft-constraint **violation** table.

Synthetic fixtures only (no registry): a stub ``acc`` exposing ``edge_meta`` (the
``CovAcc`` topology of the coverage/stats tests), a stub ``ConstraintRun`` and a stub
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
    """Stub ConstraintRun: prior a0/σ_a + 4 constraint rows + per-t c/σ_c."""
    acc_id = "fake-acc"
    a0      = np.array([10, 0, 5, 3, 7, 0, 8, 9, 0], dtype=float)
    sigma_a = np.array([1, 1, 1, 1, 1, 1, 2, 1, 1], dtype=float)
    row_meta = pd.DataFrame([
        {"constraint_id": "sum_A",   "type": "sum_eq",       "time": "1995", "valuation": "Mass"},
        {"constraint_id": "ratio_A", "type": "ratio_eq",     "time": "1995"},                       # no valuation
        {"constraint_id": "mb_A",    "type": "mass_balance", "time": "1995", "valuation": "Mass"},   # hard → excluded
        {"constraint_id": "sum_B",   "type": "sum_eq",       "time": "1996", "valuation": "Mass"},   # year → excluded
    ])
    timestep_data = {
        "1995": {"c": np.array([100, 0, 0, 50], dtype=float),
                 "sigma_c": np.array([10, 2, 0, 5], dtype=float)},
    }


class ResStub:
    F_opt  = np.array([12, 1, 5, 6, 7, 0, 9, 50, 2], dtype=float)  # defect = F_opt - a0
    c_star = np.array([104, 1, 0, 50], dtype=float)                 # eps = c_star - c


@pytest.fixture
def diag():
    return SolverStatsDiagnostic({"1995": ResStub()}, CRStub(), AccStub(), year="1995")


_COLS = ["min", "p25", "p50", "p75", "p95", "mean", "max", "sum",
         "n_val", "n_nan", "val_pct", "median_abs_z"]


# ---------------------------------------------------------------------------
# Defect tables — value, grouping, columns
# ---------------------------------------------------------------------------

def test_defect_index_and_columns(diag):
    t = diag.defect_from_table()
    assert list(t.index.names) == ["sector", "region", "valuation"]
    assert list(t.columns) == _COLS          # median_abs_z present and last


def test_defect_from_mining_mass(diag):
    """from (Mining,A,Mass): e0=2, e5=0, e8=2 (defects). e6 is CC, e7 is 1996."""
    t = diag.defect_from_table()
    row = t.loc[("Mining", "A", "Mass")]
    assert row["n_val"] == 3
    assert row["min"] == 0.0 and row["max"] == 2.0
    assert row["sum"] == pytest.approx(4.0)
    assert row["mean"] == pytest.approx(4.0 / 3.0)
    assert row["median_abs_z"] == pytest.approx(2.0)   # |z| = [2, 0, 2]


def test_defect_no_mask(diag):
    """No has_val mask: every solved edge counts → n_nan 0, val_pct 100."""
    t = diag.defect_from_table()
    assert (t["n_nan"] == 0).all()
    assert (t["val_pct"] == 100.0).all()


def test_defect_cc_bucket_uses_sigma_a(diag):
    """from (Mining,A,CC): only e6 → defect 1, z = 1/σ_a(=2) = 0.5."""
    t = diag.defect_from_table()
    row = t.loc[("Mining", "A", "CC")]
    assert row["n_val"] == 1
    assert row["sum"] == pytest.approx(1.0)
    assert row["median_abs_z"] == pytest.approx(0.5)


def test_defect_from_vs_to_differ(diag):
    """Supply edge e0 Mining→Cathode (defect 2): from-Mining vs to-Cathode."""
    tf = diag.defect_from_table()
    tt = diag.defect_to_table()
    assert ("Cathode", "A", "Mass") not in tf.index or tf.loc[("Cathode", "A", "Mass"), "max"] != 2.0
    assert tt.loc[("Cathode", "A", "Mass"), "max"] == 2.0   # e0 lands in to-Cathode
    # to-Mining bucket gathers e1(1), e3(3), e4(0)
    assert tt.loc[("Mining", "A", "Mass"), "sum"] == pytest.approx(4.0)
    assert tt.loc[("Mining", "A", "Mass"), "max"] == 3.0


def test_defect_year_filter_drops_1996(diag):
    """e7 (1996, Mining→Cathode, defect 41) must not leak into any 1995 bucket."""
    tt = diag.defect_to_table()
    row = tt.loc[("Cathode", "A", "Mass")]
    assert row["n_val"] == 1          # only e0
    assert row["max"] == 2.0          # not 41


# ---------------------------------------------------------------------------
# Constraint-violation table
# ---------------------------------------------------------------------------

def test_violations_only_soft_in_year(diag):
    v = diag.violations_table()
    assert list(v.index) == ["sum_A", "ratio_A"]
    assert "mb_A" not in v.index      # hard excluded
    assert "sum_B" not in v.index     # 1996 excluded
    assert v.index.name == "constraint_id"


def test_violations_residual_and_z(diag):
    v = diag.violations_table()
    assert list(v.columns) == ["type", "valuation", "c", "lhs", "residual", "sigma_c", "z"]
    s = v.loc["sum_A"]
    assert s["c"] == 100.0 and s["lhs"] == 104.0  # lhs = c_star
    assert s["residual"] == pytest.approx(4.0)    # lhs - c
    assert s["sigma_c"] == 10.0
    assert s["z"] == pytest.approx(0.4)           # residual / sigma_c
    assert s["type"] == "sum_eq" and s["valuation"] == "Mass"


def test_violations_ratio_valuation_nan(diag):
    v = diag.violations_table()
    r = v.loc["ratio_A"]
    assert r["residual"] == pytest.approx(1.0)
    assert r["z"] == pytest.approx(0.5)          # 1 / 2
    assert pd.isna(r["valuation"])               # ratio_eq omits valuation


# ---------------------------------------------------------------------------
# write()
# ---------------------------------------------------------------------------

def test_write_emits_three_csvs(diag, tmp_path):
    paths = diag.write(str(tmp_path))
    assert len(paths) == 3
    names = {p.split("/")[-1] for p in paths}
    assert names == {"defect_from_1995.csv", "defect_to_1995.csv",
                     "constraint_violations_1995.csv"}
    for p in paths:
        assert pd.read_csv(p).shape[0] > 0
