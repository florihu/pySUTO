"""
test_const_stats_diag.py
========================
Tests for :class:`pysuto.diag.ConstStatsDiagnostic` — the Stage-3 prior
row-level constraint table (``lhs_prior = G·a0``, ``residual = lhs_prior − c``),
the exact twin of the solver-stage table minus ``lhs_post``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pysuto.diag import ConstStatsDiagnostic


class CRStub:
    """Stub ConstraintRun: prior a0 + G + 4 constraint rows + per-t c/σ_c."""
    a0 = np.array([10, 0, 5, 3], dtype=float)
    # rows: sumA=a0[0]+a0[2]=15 ; ratioA=a0[0]-a0[3]=7 ; mbA=a0[0]+a0[1]=10 ; sumB=5
    G = sp.csr_matrix(np.array([
        [1, 0, 1, 0],   # sumA   → 15
        [1, 0, 0, -1],  # ratioA → 7
        [1, 1, 0, 0],   # mbA    (hard, σ_c = 0)
        [0, 0, 1, 0],   # sumB   (1996)
    ], dtype=float))
    row_meta = pd.DataFrame([
        {"constraint_id": "sumA",   "type": "sum",     "variant": "anchor",   "cone_type": "free",   "time": "1995", "valuation": "Mass"},
        {"constraint_id": "ratioA", "type": "ratio",   "variant": "ratioeq",  "cone_type": "free",   "time": "1995"},                       # no valuation
        {"constraint_id": "mbA",    "type": "massbal", "variant": "massbal",  "cone_type": "zero",   "time": "1995", "valuation": "Mass"},   # hard
        {"constraint_id": "sumB",   "type": "sum",     "variant": "anchor",   "cone_type": "free",   "time": "1996", "valuation": "Mass"},   # year
    ])
    timestep_data = {
        "1995": {"c": np.array([12, 0, 10, 99], dtype=float),
                 "sigma_c": np.array([6, 2, 0, 5], dtype=float)},
        "1996": {"c": np.array([12, 0, 10, 4], dtype=float),
                 "sigma_c": np.array([6, 2, 0, 5], dtype=float)},
    }


@pytest.fixture
def diag():
    return ConstStatsDiagnostic(CRStub(), years=("1995",))


_COLS = ["constraint_id", "type", "variant", "cone_type", "valuation",
         "c", "lhs_prior", "residual", "residual_abs", "sigma_c", "z", "z_abs"]


def test_schema_all_rows_no_lhs_post(diag):
    t = diag.constraint_table("1995")
    assert list(t.columns) == _COLS                          # no lhs_post pre-solve
    assert list(t["constraint_id"]) == ["sumA", "ratioA", "mbA"]   # hard included
    assert "sumB" not in t["constraint_id"].values           # 1996 excluded


def test_prior_residual_is_g_a0_minus_c(diag):
    """lhs_prior = G·a0, residual = lhs_prior − c, z = residual/σ_c."""
    t = diag.constraint_table("1995").set_index("constraint_id")
    s = t.loc["sumA"]
    assert s["lhs_prior"] == pytest.approx(15.0)  # a0[0] + a0[2]
    assert s["c"] == 12.0
    assert s["residual"] == pytest.approx(3.0)    # 15 - 12
    assert s["residual_abs"] == pytest.approx(3.0)
    assert s["z"] == pytest.approx(0.5)           # 3 / 6
    assert s["valuation"] == "Mass" and s["cone_type"] == "free"


def test_hard_row_z_nan(diag):
    t = diag.constraint_table("1995").set_index("constraint_id")
    mb = t.loc["mbA"]
    assert mb["lhs_prior"] == pytest.approx(10.0)
    assert mb["residual"] == pytest.approx(0.0)   # prior already balances
    assert pd.isna(mb["z"]) and pd.isna(mb["z_abs"])  # σ_c = 0


def test_ratio_row_valuation_nan(diag):
    t = diag.constraint_table("1995").set_index("constraint_id")
    r = t.loc["ratioA"]
    assert r["lhs_prior"] == pytest.approx(7.0)   # a0[0] - a0[3]
    assert r["residual"] == pytest.approx(7.0)    # 7 - 0
    assert r["z"] == pytest.approx(3.5)           # 7 / 2
    assert pd.isna(r["valuation"])


def test_years_all_resolves_to_timestep_keys():
    d = ConstStatsDiagnostic(CRStub(), years="all")
    assert d.years == ["1995", "1996"]


def test_write_emits_prior_csv_per_year(tmp_path):
    d = ConstStatsDiagnostic(CRStub(), years="all")
    paths = d.write(str(tmp_path))
    names = [p.split("/")[-1] for p in paths]
    assert names == ["constraints_prior_1995.csv", "constraints_prior_1996.csv"]
    assert pd.read_csv(paths[0]).shape[0] == 3
    assert "constraint_id" in pd.read_csv(paths[0]).columns   # index=False
