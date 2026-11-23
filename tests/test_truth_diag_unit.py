"""test_truth_diag_unit.py — ``TruthDiag``'s pure tables on a six-edge frame (#279).

The toy chain reaches these through the whole pipeline (``test_toy_case.py``); here each
table and each validation raise is checked on numbers that can be read off the frame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysuto.diag.truth import TruthDiag

N = 6


def _edge_meta() -> pd.DataFrame:
    return pd.DataFrame({
        "edge_id":     np.arange(N),
        "from_sector": ["Mine", "Mine", "Mine", "Conc", "Conc", "Conc"],
        "to_sector":   ["Conc"] * 3 + ["Smelt"] * 3,
        "from_region": ["A"] * N, "to_region": ["A"] * N,
        "time":        ["2000", "2001", "2002"] * 2,
        "valuation":   ["Gross"] * N,
    })


TRUTH = np.array([10.0, 0.0, 20.0, 5.0, -3.0, 8.0])


def _diag(**kw) -> TruthDiag:
    base = dict(
        estimates={"prior": np.array([12.0, 1.0, 10.0, 5.0, 2.0, 4.0]),
                   "reconciled": np.array([11.0, 0.5, 18.0, 5.0, -3.0, np.nan])},
        truth=TRUTH, sigma_post=np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        edge_meta=_edge_meta(), sigma_prior=np.full(N, 2.0), planted=[2])
    base.update(kw)
    return TruthDiag(**base)


class TestTables:

    def test_relative_error_is_nan_over_a_zero_truth_or_a_missing_estimate(self):
        rel = _diag().relative_error("reconciled")
        assert rel[0] == pytest.approx(0.1, rel=1e-12)
        assert np.isnan(rel[1]) and np.isnan(rel[5])           # zero truth, NaN estimate
        assert rel[2] == pytest.approx(0.1, rel=1e-12)
        assert rel[4] == pytest.approx(0.0, abs=1e-12)

    def test_error_by_stage_counts_only_the_edges_with_a_truth_and_an_estimate(self):
        table = _diag().error_by_stage()
        assert table.loc["reconciled", "n_edges"] == 4
        assert table.loc["prior", "n_edges"] == 5
        assert table.loc["reconciled", "max"] == pytest.approx(0.1, rel=1e-12)

    def test_an_edge_without_a_posterior_sigma_is_covered_only_where_exact(self):
        covered = _diag().covered()
        assert covered.tolist() == [True, False, False, True, True, False]
        # edge 0: |1| <= 1.96; edge 1: sigma 0 and |0.5| > 1e-9; edge 2: |2| > 1.96;
        # edge 4: sigma 0 and exact; edge 5: NaN estimate never holds the truth.
        assert _diag().coverage() == pytest.approx(3 / 6, rel=1e-12)

    def test_z_post_is_nan_where_sigma_is_zero(self):
        z = _diag().z_post()
        assert z[0] == pytest.approx(1.0, rel=1e-12) and np.isnan(z[1]) and np.isnan(z[4])

    def test_the_residual_rank_orders_by_the_largest_prior_z_and_flags_the_planted(self):
        rank = _diag().residual_rank()
        assert rank["edge_id"].iloc[0] == 2                    # |18 - 10| / 2 is the largest
        assert bool(rank["planted"].iloc[0]) and rank["planted"].sum() == 1

    def test_retention_leaves_out_a_category_with_a_negative_truth(self):
        table = _diag().retention("Gross")
        categories = set(table.index.get_level_values("category"))
        assert categories == {"Mine → Conc"}                   # Conc → Smelt holds -3


class TestValidation:

    @pytest.mark.parametrize("kw, err", [
        ({"truth": np.ones((N, 1))}, "1-D"),
        ({"truth": np.ones(N - 1)}, "edge_meta must hold"),
        ({"estimates": {"prior": np.ones(N - 1)}}, "expected"),
        ({"sigma_post": np.ones(N - 1)}, "sigma_post"),
        ({"planted": [N]}, "outside"),
        ({"alpha": 1.5}, "alpha"),
    ])
    def test_a_malformed_input_is_refused(self, kw, err):
        with pytest.raises(ValueError, match=err):
            _diag(**kw)

    def test_calibration_without_a_reconciled_estimate_or_sigma_is_refused(self):
        with pytest.raises(ValueError, match="calibration needs"):
            _diag(sigma_post=None).z_post()
        with pytest.raises(ValueError, match="residual rank needs"):
            _diag(sigma_prior=None).residual_rank()
