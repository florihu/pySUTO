"""
test_const_stats_diag.py
========================
Tests for :class:`pysuto.diag.ConstStatsDiagnostic` — the Stage-3 prior
soft-constraint residual/violation table (``lhs = G·a0``, ``residual = G·a0 − c``),
sharing the schema of the solver-stage table.
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
    # rows: sumA=a0[0]+a0[2]=15 ; ratioA=a0[0]-a0[3]=7 ; mb/sumB arbitrary
    G = sp.csr_matrix(np.array([
        [1, 0, 1, 0],   # sumA   → 15
        [1, 0, 0, -1],  # ratioA → 7
        [1, 1, 0, 0],   # mbA    (hard, excluded)
        [0, 0, 1, 0],   # sumB   (1996, excluded)
    ], dtype=float))
    row_meta = pd.DataFrame([
        {"constraint_id": "sumA",   "type": "sum",     "variant": "anchor",   "cone_type": "free",   "time": "1995", "valuation": "Mass"},
        {"constraint_id": "ratioA", "type": "ratio",   "variant": "ratioeq",  "cone_type": "free",   "time": "1995"},                       # no valuation
        {"constraint_id": "mbA",    "type": "massbal", "variant": "massbal",  "cone_type": "zero",   "time": "1995", "valuation": "Mass"},   # hard
        {"constraint_id": "sumB",   "type": "sum",     "variant": "anchor",   "cone_type": "free",   "time": "1996", "valuation": "Mass"},   # year
    ])
    timestep_data = {
        "1995": {"c": np.array([12, 0, 0, 99], dtype=float),
                 "sigma_c": np.array([6, 2, 0, 5], dtype=float)},
    }


@pytest.fixture
def diag():
    return ConstStatsDiagnostic(CRStub(), year="1995")


def test_schema_and_soft_only(diag):
    v = diag.violations_table()
    assert list(v.columns) == ["type", "variant", "valuation", "c", "lhs", "residual", "sigma_c", "z"]
    assert list(v.index) == ["sumA", "ratioA"]
    assert "mbA" not in v.index      # hard excluded
    assert "sumB" not in v.index     # 1996 excluded
    assert v.index.name == "constraint_id"


def test_prior_residual_is_g_a0_minus_c(diag):
    """lhs = G·a0, residual = lhs − c, z = residual/σ_c."""
    v = diag.violations_table()
    s = v.loc["sumA"]
    assert s["lhs"] == pytest.approx(15.0)       # a0[0] + a0[2]
    assert s["c"] == 12.0
    assert s["residual"] == pytest.approx(3.0)   # 15 - 12
    assert s["z"] == pytest.approx(0.5)          # 3 / 6
    assert s["valuation"] == "Mass"


def test_ratio_row_valuation_nan(diag):
    v = diag.violations_table()
    r = v.loc["ratioA"]
    assert r["lhs"] == pytest.approx(7.0)        # a0[0] - a0[3]
    assert r["residual"] == pytest.approx(7.0)   # 7 - 0
    assert r["z"] == pytest.approx(3.5)          # 7 / 2
    assert pd.isna(r["valuation"])


def test_write_emits_prior_csv(diag, tmp_path):
    paths = diag.write(str(tmp_path))
    assert len(paths) == 1
    assert paths[0].split("/")[-1] == "constraint_violations_prior_1995.csv"
    assert pd.read_csv(paths[0]).shape[0] == 2
